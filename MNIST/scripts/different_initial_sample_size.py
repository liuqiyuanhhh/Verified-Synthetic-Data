import sys
import os
import re
import argparse
import random
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------- paths & imports ---------------------
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.append(str(SRC_DIR))

import models as models
import train_helper as train_helper
import utils as utils
import data_helper as data_helper
from FID import calculate_fid_score

# --------------------- argparse ---------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fixed-size", type=int, default=int(os.getenv("FIXED_SIZE", "5000")),
                    help="(unused in this script’s loop) left for compatibility")
parser.add_argument("--init-size", type=int, default=int(os.getenv("INIT_SIZE", "500")),
                    help="initial real subset size")
parser.add_argument("--rounds", type=int, default=40, help="number of retraining rounds (k)")
parser.add_argument("--synthetic-per-round", type=int, default=20000,
                    help="filtered synthetic size per round")
parser.add_argument("--threshold", type=float, default=0.1,
                    help="discriminator selection threshold (quantile)")
parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
args = parser.parse_args()

FIXED = int(args.fixed_size)
init_size = int(args.init_size)
k = int(args.rounds)
synthetic_per_round = int(args.synthetic_per_round)
threshold = float(args.threshold)
base_seed = int(args.seed)

# --------------------- device & seeds ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(base_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(base_seed)
np.random.seed(base_seed)
random.seed(base_seed)

# --------------------- output layout (per-init folders) ---------------------
BASE_ROOT = THIS_DIR.parent / "conv_cvae" / "larger_initial_sample_size"
ROOT = BASE_ROOT / f"init_{init_size}"
model_saved_path   = ROOT / "model_saved_more"
data_saved_path    = ROOT / "data_saved_more"
results_saved_path = ROOT / "results_saved_more"
picture_saved_path = ROOT / "picture_saved_more"
os.makedirs(model_saved_path, exist_ok=True)
os.makedirs(data_saved_path, exist_ok=True)
os.makedirs(results_saved_path, exist_ok=True)
os.makedirs(picture_saved_path, exist_ok=True)

# --------------------- schedule ---------------------
size_schedule = [synthetic_per_round] * k  # constant size each round

# --------------------- data ---------------------
transform = transforms.ToTensor()
full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

full_digit_indices = utils.create_balanced_subset_indices(full_dataset, seed=base_seed)

# --------------------- small utils ---------------------
def append_result(csv_path, model_name, val_loss, val_recon, val_kl, fid_score):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row = pd.DataFrame([{
        "model_name": model_name,
        "val_loss": float(val_loss),
        "val_recon": float(val_recon),
        "val_kl": float(val_kl),
        "fid": float(fid_score),
    }])
    header_needed = not os.path.exists(csv_path)
    row.to_csv(csv_path, mode="a", header=header_needed, index=False)

@torch.no_grad()
def generate_images_in_batches(model, total_samples, latent_dim, num_classes, batch_size=10000, device='cuda'):
    model.eval()
    generated_images = []
    all_labels = []

    labels_full = torch.arange(total_samples) % num_classes
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        n = end - start
        z = torch.randn(n, latent_dim, device=device)
        y = labels_full[start:end].to(device)
        y_onehot = F.one_hot(y, num_classes=num_classes).float().to(device)
        logits_flat = model.decoder.decode(z, y_onehot)  # (n, 784) logits
        imgs = torch.sigmoid(logits_flat).view(-1, 1, 28, 28).cpu()
        generated_images.append(imgs)
        all_labels.append(y.cpu())
    images = torch.cat(generated_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return images, labels

@torch.no_grad()
def plot_model_samples(model, save_path=None, latent_dim=20, num_classes=10, per_class=8, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    z = torch.randn(num_classes * per_class, latent_dim, device=device)
    y = torch.arange(num_classes, device=device).repeat_interleave(per_class)
    y_onehot = F.one_hot(y, num_classes=num_classes).float().to(device)
    logits_flat = model.decoder.decode(z, y_onehot)
    imgs = torch.sigmoid(logits_flat).view(-1, 1, 28, 28)
    imgs_np = imgs.detach().cpu().numpy()

    fig, axes = plt.subplots(num_classes, per_class, figsize=(2*per_class, 2*num_classes))
    for c in range(num_classes):
        for j in range(per_class):
            idx = c * per_class + j
            axes[c, j].imshow(imgs_np[idx].squeeze(), cmap='gray')
            axes[c, j].axis('off')
            if j == 0:
                axes[c, j].set_ylabel(f"Class {c}", fontsize=10)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[SAVE] Sample grid saved -> {save_path}")
        plt.close(fig)
    return fig, axes

def fid(model):
    model = model.to(device)
    synthetic_gen_size = 6000
    gen_imgs_before_filter, y_before_filter = generate_images_in_batches(
        model=model,
        total_samples=synthetic_gen_size,
        latent_dim=20,
        num_classes=10,
        batch_size=10000,
        device=device
    )
    real_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    synthetic_ds = TensorDataset(gen_imgs_before_filter, y_before_filter)
    fid_value = calculate_fid_score(real_ds, synthetic_ds)
    return fid_value

# --------------------- training: initial real subset ---------------------
init_subset = utils.get_balanced_subset(full_digit_indices, init_size)
init_dataset = Subset(full_dataset, init_subset)
init_loader = DataLoader(init_dataset, batch_size=128, shuffle=True)

this_model = models.CVAE(
    input_dim=784, label_dim=10, latent_dim=20,
    name=f"cvae_conv_real_{init_size}", arch="conv"
).to(device)

train_helper.train_model(this_model, init_loader, device, epochs=200, lr=1e-3, patience=5, verbose=False)
plot_model_samples(this_model, save_path=os.path.join(picture_saved_path, f"initial_model_samples_{init_size}.png"), device=device)
val_loss, val_recon, val_kl = train_helper.calculate_validation_loss(this_model, test_loader, device)
fid_score = fid(this_model)

csv_path = os.path.join(results_saved_path, f"results_table_init{init_size}_{k}rounds.csv")
append_result(csv_path, getattr(this_model, "get_name", lambda: this_model.__dict__.get("name", "init"))(),
              val_loss, val_recon, val_kl, fid_score)
print(f"Init loss: {init_size} - Val Loss: {val_loss:.4f} - Val KL: {val_kl:.4f} - Val Recon: {val_recon:.4f} - FID: {fid_score:.4f}")

utils.save_model(this_model, getattr(this_model, "get_name", lambda: this_model.__dict__.get("name", "init"))(), model_saved_path)

# --------------------- k rounds of synthetic retraining ---------------------
all_models = [this_model]
test_results = {"val_loss": [val_loss], "val_recon": [val_recon], "val_kl": [val_kl],
                "fid": [fid_score], "model_name": [getattr(this_model, "get_name", lambda: this_model.__dict__.get("name", "init"))()]}

curr_model = this_model
for round_id in range(1, k + 1):
    synthetic_size = int(size_schedule[round_id - 1])

    # (A) train discriminator for current generator
    print(f"\n[Round {round_id}] Training discriminator for current model...")
    discriminator_dataset = data_helper.prepare_discriminator_dataset(full_dataset, curr_model, device)
    disc_loader = DataLoader(discriminator_dataset, batch_size=128, shuffle=True)
    disc_model = models.SyntheticDiscriminator(input_dim=784).to(device)
    train_helper.train_model(model=disc_model, train_loader=disc_loader, device=device,
                             epochs=80, lr=1e-3, patience=5, verbose=False)
    del disc_loader, discriminator_dataset

    # (B) generate filtered synthetic data to a temp dir
    model_name = f'cvae_conv_init{init_size}_q{threshold}_s{synthetic_size}_r{round_id}'
    synthetic_data_load_path = os.path.join(data_saved_path, model_name)
    print(f"[Round {round_id}] Generating filtered synthetic data -> {synthetic_data_load_path}")

    data_helper.generate_balanced_images_with_filtering(
        model=curr_model,
        save_directory=synthetic_data_load_path,
        total_samples=synthetic_size,
        discriminator=disc_model,
        selection_threshold=threshold,
        verbose=False,
        use_quantile_filtering=True
    )

    # (C) train new model on filtered synthetic
    synthetic_loader = data_helper.create_directory_based_dataloader(synthetic_data_load_path, batch_size=128)
    synthetic_model = models.CVAE(input_dim=784, label_dim=10, latent_dim=20,
                                  name=model_name, arch="conv").to(device)
    train_helper.train_model(synthetic_model, synthetic_loader, device,
                             epochs=200, lr=1e-3, patience=5, verbose=False)
    plot_model_samples(synthetic_model, save_path=os.path.join(picture_saved_path, f"round{round_id}_model_samples.png"), device=device)
    val_loss, val_recon, val_kl = train_helper.calculate_validation_loss(synthetic_model, test_loader, device)
    fid_score = fid(synthetic_model)

    # log and save
    test_results["val_loss"].append(val_loss)
    test_results["val_recon"].append(val_recon)
    test_results["val_kl"].append(val_kl)
    test_results["fid"].append(fid_score)
    test_results["model_name"].append(getattr(synthetic_model, "get_name", lambda: synthetic_model.__dict__.get("name", model_name))())

    print(f"[Round {round_id}] Model: {model_name} | Val Loss: {val_loss:.4f} | KL: {val_kl:.4f} | Recon: {val_recon:.4f} | FID: {fid_score:.4f}")
    utils.save_model(synthetic_model, getattr(synthetic_model, "get_name", lambda: synthetic_model.__dict__.get("name", model_name))(), model_saved_path)
    append_result(csv_path, getattr(synthetic_model, "get_name", lambda: synthetic_model.__dict__.get("name", model_name))(),
                  val_loss, val_recon, val_kl, fid_score)

    # (D) advance chain
    curr_model = synthetic_model
    all_models.append(synthetic_model)

    # (E) cleanup temp dir
    try:
        if os.path.exists(synthetic_data_load_path):
            shutil.rmtree(synthetic_data_load_path)
            print(f"[CLEAN] Removed temp dir: {synthetic_data_load_path}")
    except Exception as e:
        print(f"[WARN] Failed to remove {synthetic_data_load_path}: {e}")

    # (F) free GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("[DONE] All rounds completed.")
