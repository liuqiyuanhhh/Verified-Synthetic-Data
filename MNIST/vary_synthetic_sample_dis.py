# ===== Mixed discriminator (use your prepare_discriminator_dataset for each generator) =====
import os
import sys
from datetime import datetime
import json
import math

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd

# ------------------ Paths  ------------------
VAE_PATH = "/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/conv_cvae"
FULL_GEN_CKPT_DIR = "/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/conv_cvae/model_saved_full_dataset"
FULL_GEN_CKPT = os.path.join(FULL_GEN_CKPT_DIR, "full_dataset_model.pth")

ROOT = "/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/conv_cvae/one_strong_vae_appendix"
MODEL_SAVED_PATH = os.path.join(ROOT, "model_saved_more")
DISC_SAVED_PATH  = os.path.join(MODEL_SAVED_PATH, "discriminators")
SUBSET_MODEL_DIR = os.path.join(MODEL_SAVED_PATH, "subset_generators")
RESULTS_PATH     = os.path.join(ROOT, "results_saved_more")
MANIFEST_CSV     = os.path.join(RESULTS_PATH, "discriminator_manifest.csv")

for d in [MODEL_SAVED_PATH, DISC_SAVED_PATH, SUBSET_MODEL_DIR, RESULTS_PATH]:
    os.makedirs(d, exist_ok=True)

# ------------------ Import your project modules ------------------
sys.path.append(VAE_PATH)
import models
import train_helper
import utils
import data_helper  

# ------------------ Config ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
torch.manual_seed(seed); np.random.seed(seed)
GEN_REAL_SIZES = [0, 500, 1000, 3000, 4000, 5000, 6000, 10000]
AUTO_TRAIN_MISSING = True    
DISC_EPOCHS = 80
DISC_LR = 1e-3
DISC_PATIENCE = 5
DISC_BATCH_SIZE = 128

print(f"[DEVICE] {device}")

# ------------------ Load MNIST train set ------------------
transform = transforms.ToTensor()
full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

full_digit_indices = utils.create_balanced_subset_indices(full_dataset, seed=seed)
# ------------------ Helpers ------------------
def load_or_train_generator_for_size(real_subset_size: int):
 
    model = models.CVAE(input_dim=784, label_dim=10, latent_dim=20, arch="conv").to(device)

    if real_subset_size == 0:
        tag = "full"
        ckpt = FULL_GEN_CKPT
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.name = f"cvae_conv_{tag}"
        print(f"[GEN] Loaded: {tag}")
        return tag, model

    tag = f"real_{real_subset_size}"
    ckpt = os.path.join(SUBSET_MODEL_DIR, f"{tag}.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.name = f"cvae_conv_{tag}"
        print(f"[GEN] Loaded: {tag}")
        return tag, model

    if not AUTO_TRAIN_MISSING:
        raise FileNotFoundError(f"Missing generator ckpt: {ckpt}")

    print(f"[GEN] Training generator on {real_subset_size} real samples ...")
    subset_idx = utils.get_balanced_subset(full_digit_indices, real_subset_size)
    subset_ds = Subset(full_dataset, subset_idx)
    subset_loader = DataLoader(subset_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    model.name = f"cvae_conv_{tag}"
    train_helper.train_model(model, subset_loader, device, epochs=200, lr=1e-3, patience=5, verbose=False)
    torch.save(model.state_dict(), ckpt)
    print(f"[GEN] Saved -> {ckpt}")
    return tag, model

def split_real_dataset_into_shards_per_class(full_digit_indices, n_shards: int):
   
    shards_indices = [[] for _ in range(n_shards)]
    for c in range(10):
        idx_c = np.array(full_digit_indices[c])
        n_c = len(idx_c)
        base = n_c // n_shards
        rem  = n_c % n_shards
        start = 0
        for s in range(n_shards):
            take = base + (1 if s < rem else 0)
            part = idx_c[start:start+take]
            start += take
            shards_indices[s].extend(part.tolist())

    shards = [Subset(full_dataset, inds) for inds in shards_indices]
    sizes = [len(s) for s in shards]
    print(f"[SHARD] #shards={n_shards}, sizes={sizes}, total={sum(sizes)} (full={len(full_dataset)})")
    return shards

# ------------------ Build per-generator datasets with your function ------------------

real_shards = split_real_dataset_into_shards_per_class(full_digit_indices, n_shards=len(GEN_REAL_SIZES))

per_gen_datasets = []
gen_tags = []

for i, sz in enumerate(GEN_REAL_SIZES):
    tag, gen_model = load_or_train_generator_for_size(sz)

    real_shard = real_shards[i]
    print(f"[DATASET] Building dataset with your prepare_discriminator_dataset for generator: {tag}")
    ds = data_helper.prepare_discriminator_dataset(real_shard, gen_model, device=device)
    per_gen_datasets.append(ds)
    gen_tags.append(tag)

mixed_disc_dataset = ConcatDataset(per_gen_datasets)
print(f"[MIX] Combined mixed dataset size = {len(mixed_disc_dataset)}")


# ------------------ Train ONE mixed discriminator ------------------
disc_model = models.SyntheticDiscriminator(input_dim=784).to(device)
disc_loader = DataLoader(mixed_disc_dataset, batch_size=DISC_BATCH_SIZE,
                         shuffle=True, num_workers=0, pin_memory=False)

print(f"[DISC] Training mixed discriminator on {len(mixed_disc_dataset)} samples ...")
train_helper.train_model(
    model=disc_model,
    train_loader=disc_loader,
    device=device,
    epochs=DISC_EPOCHS,
    lr=DISC_LR,
    patience=DISC_PATIENCE,
    verbose=False
)


mix_tag = "mix_" + "_".join(gen_tags)
ckpt_path = os.path.join(DISC_SAVED_PATH, f"disc_{mix_tag}.pth")
torch.save(disc_model.state_dict(), ckpt_path)
print(f"[SAVE] Mixed discriminator -> {ckpt_path}")

row = {
    "tag": mix_tag,
    "disc_ckpt": ckpt_path,
    "gen_sizes": json.dumps(GEN_REAL_SIZES),
    "shard_sizes": json.dumps([len(s) for s in real_shards]),
    "timestamp": datetime.now().isoformat()
}
df = pd.DataFrame([row])
header_needed = not os.path.exists(MANIFEST_CSV)
df.to_csv(MANIFEST_CSV, mode="a", header=header_needed, index=False)
print(f"[LOG] Manifest appended -> {MANIFEST_CSV}")

print("Done: trained & saved ONE mixed discriminator using your prepare_discriminator_dataset().")
