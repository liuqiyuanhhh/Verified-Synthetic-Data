import sys
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
from torchvision.utils import make_grid
import torch.nn.functional as F
import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# test of commit
vae_path = "/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/conv_cvae"
sys.path.append(vae_path)

import models as models
import train_helper as train_helper
import utils as utils
import data_helper as data_helper

parser = argparse.ArgumentParser()
parser.add_argument("--fixed-size", type=int, default=int(os.getenv("FIXED_SIZE", "5000")))
args = parser.parse_args()
FIXED = int(args.fixed_size) 

# Set up device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_seed = 0
torch.manual_seed(base_seed)
torch.cuda.manual_seed_all(base_seed)
np.random.seed(base_seed)
random.seed(base_seed)

ROOT = "/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/conv_cvae/one_strong_vae_appendix"
model_saved_path = os.path.join(ROOT,"model_saved_more")
data_saved_path = os.path.join(ROOT,"data_saved_more")
results_saved_path = os.path.join(ROOT,"results_saved_more")
picture_saved_path = os.path.join(ROOT,"picture_saved_more")
os.makedirs(results_saved_path, exist_ok=True)
#################  size schedule #################

#start = 10_000
#end = 256_000
#k = 40 
#/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/conv_cvae/model_saved_full_dataset/full_dataset_model.pth
#size_schedule = ((np.linspace(start, end, k) / 10).round().astype(int) * 10).tolist()

start = 30000
end = 1000000
k = 20
size_schedule = ((np.linspace(start, end, k) / 10).round().astype(int) * 10).tolist()
#size_schedule = (np.arange(k) * step + start).tolist()

#k = 40
#size_schedule = [20_000] * k
################### real ############################
full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

test_dataset = datasets.MNIST(root="./data", train=False, download=True,transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
full_digit_indices = utils.create_balanced_subset_indices(full_dataset,seed=base_seed)

#train_dataset_5000 = Subset(full_dataset, range(5000))
import torch
import torch.nn.functional as F

def append_result(csv_path, model_name, val_loss, val_recon, val_kl, fid_score):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row = pd.DataFrame([{
        "model_name": model_name,
        "val_loss": float(val_loss),
        "val_recon": float(val_recon),
        "val_kl": float(val_kl),
        "fid": float(fid_score),
    }])
    header_needed = not os.path.exists(csv_path)  # only first write has header
    row.to_csv(csv_path, mode="a", header=header_needed, index=False)


@torch.no_grad()
def generate_images_in_batches(model, total_samples, latent_dim, num_classes, batch_size=10000, device='cuda'):
    model.eval()
    generated_images = []
    all_labels = []

    # Balanced label assignment (always length = total_samples)
    labels_full = torch.arange(total_samples) % num_classes

    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        n = end - start

        # Sample latent z
        z = torch.randn(n, latent_dim, device=device)

        # Labels
        y = labels_full[start:end]
        y_onehot = F.one_hot(y, num_classes=num_classes).float().to(device)

        # Decode logits → sigmoid → [0,1]
        logits_flat = model.decoder.decode(z, y_onehot)     # (n, 784), logits
        imgs = torch.sigmoid(logits_flat).view(-1, 1, 28, 28).cpu()

        generated_images.append(imgs)
        all_labels.append(y)

    images = torch.cat(generated_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return images, labels

from FID import calculate_fid_score

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_model_samples(model, save_path=None, latent_dim=20, num_classes=10, per_class=8, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # latent + labels
    z = torch.randn(num_classes * per_class, latent_dim, device=device)
    y = torch.arange(num_classes, device=device).repeat_interleave(per_class)
    y_onehot = F.one_hot(y, num_classes=num_classes).float().to(device)

    # decode -> logits, then map to [0,1]
    logits_flat = model.decoder.decode(z, y_onehot)            # (n, 784), logits
    imgs = torch.sigmoid(logits_flat).view(-1, 1, 28, 28)      # tensor on device, no grad (due to @no_grad)

    # move for plotting
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
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        print(f"[SAVE] Sample grid saved -> {save_path}")
    return fig, axes


def fid(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    synthetic_gen_size = 6000
    gen_imgs_before_filter,y_before_filter = generate_images_in_batches(
        model=model,
        total_samples=synthetic_gen_size,
        latent_dim=20,
        num_classes=10,
        batch_size=10000,
        device=device
    )
    # Load synthetic data
    #synthetic = torch.load(f"data_saved/synthetic_mnist_cvae_{sample_size}_2.pt")
    images = gen_imgs_before_filter # [N, 1, 28, 28]
    labels = y_before_filter  # [N]

    transform = transforms.ToTensor()

    real_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    synthetic_ds = TensorDataset(images, labels)
    
    synthetic_ds = TensorDataset(images, labels)
    fid = calculate_fid_score(real_ds, synthetic_ds)
    
    return fid

import shutil

init_size = 500

all_models = []
test_results = {"val_loss":[], "val_recon":[], "val_kl":[], "fid":[],"model_name":[]}

# Seed real subset & train initial model
init_subset = utils.get_balanced_subset(full_digit_indices, init_size)
init_dataset = Subset(full_dataset, init_subset)
init_loader = DataLoader(init_dataset, batch_size=128, shuffle=True)

this_model = models.CVAE(input_dim=784, label_dim=10, latent_dim=20,
                         name=f"cvae_conv_real_{init_size}", arch="conv").to(device)
train_helper.train_model(this_model, init_loader, device, epochs=200, lr=1e-3, patience=5, verbose=False)
plot_model_samples(this_model, save_path=os.path.join(picture_saved_path,f"initial_model_samples_{init_size}.png"), device=device)
val_loss, val_recon, val_kl = train_helper.calculate_validation_loss(this_model, test_loader, device)
fid_score = fid(this_model)
test_results["val_loss"].append(val_loss)
test_results["val_recon"].append(val_recon)
test_results["val_kl"].append(val_kl)
test_results["fid"].append(fid_score)
test_results["model_name"].append(this_model.name)
print(f"Init loss: {init_size} - Val Loss: {val_loss:.4f} - Val KL: {val_kl:.4f} - Val Recon: {val_recon:.4f}")
all_models.append(this_model)

utils.save_model(this_model, this_model.get_name(), model_saved_path)

csv_path = os.path.join(results_saved_path, f"results_table_{init_size}_{k}rounds.csv")
append_result(csv_path, this_model.get_name() if hasattr(this_model, "get_name") else this_model.name,
              val_loss, val_recon, val_kl, fid_score)

# ----- k-round synthetic retraining -----                   # <-- number of retraining rounds
threshold = 0.1           # discriminator selection threshold
#synthetic_size = 200_000         # list of synthetic attempt sizes to run per round
#SIZE_START = 50_000
#SIZE_END   = 400_000

#def round_to_multiple(n, m=10):
#    return int(n // m * m)

# Linear schedule: 50k -> 200k over k rounds
#size_schedule = np.linspace(SIZE_START, SIZE_END, k).astype(int)
#size_schedule = np.array([round_to_multiple(s, 10) for s in size_schedule])


################### full dataset trained discriminator ############################
full_real_model = models.CVAE(input_dim=784, label_dim=10, latent_dim=20, arch="conv").to(device)
model_saved_path1 = "/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/conv_cvae/model_saved_full_dataset/"
ckpt_path = os.path.join(model_saved_path1, "full_dataset_model.pth")
full_real_model.load_state_dict(torch.load(ckpt_path, map_location=device))
discriminator_dataset = data_helper.prepare_discriminator_dataset(full_dataset, full_real_model, device)
disc_loader = DataLoader(discriminator_dataset, batch_size=128, shuffle=True)
disc_model = models.SyntheticDiscriminator(input_dim=784).to(device)
train_helper.train_model(model=disc_model, train_loader=disc_loader, device=device,
                            epochs=80, lr=1e-3, patience=5, verbose=False)
del disc_loader, discriminator_dataset
#####################################################################################

curr_model = this_model
for round_id in range(1, k + 1):
    synthetic_size = int(size_schedule[round_id - 1])
    # (A) Train a fresh discriminator for the CURRENT generator
    print(f"\n[Round {round_id}] Training discriminator for current model...")
    #discriminator_dataset = data_helper.prepare_discriminator_dataset(full_dataset, curr_model, device)
    #disc_loader = DataLoader(discriminator_dataset, batch_size=128, shuffle=True)
    #disc_model = models.SyntheticDiscriminator(input_dim=784).to(device)
    #train_helper.train_model(model=disc_model, train_loader=disc_loader, device=device,
    #                         epochs=80, lr=1e-3, patience=5, verbose=False)
    #del disc_loader, discriminator_dataset
    # (B) Generate filtered synthetic dataset to a TEMP dir (unique per round)
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

    synthetic_loader = data_helper.create_directory_based_dataloader(synthetic_data_load_path, batch_size=128)
    synthetic_model = models.CVAE(input_dim=784, label_dim=10, latent_dim=20,
                                    name=model_name, arch="conv").to(device)
    train_helper.train_model(synthetic_model, synthetic_loader, device,
                                epochs=200, lr=1e-3, patience=5, verbose=False)
    plot_model_samples(synthetic_model, save_path=os.path.join(picture_saved_path,f"round{round_id}_model_samples.png"), device=device)
    val_loss, val_recon, val_kl = train_helper.calculate_validation_loss(synthetic_model, test_loader, device)
    fid_score = fid(synthetic_model)
    test_results["val_loss"].append(val_loss)
    test_results["val_recon"].append(val_recon)
    test_results["val_kl"].append(val_kl)
    test_results["model_name"].append(synthetic_model.get_name())
    test_results["fid"].append(fid_score)
    print(f"[Round {round_id}] Model: {model_name} | Val Loss: {val_loss:.4f} | KL: {val_kl:.4f} | Recon: {val_recon:.4f} | FID: {fid_score:.4f}")

    utils.save_model(synthetic_model, synthetic_model.get_name(), model_saved_path)
    all_models.append(synthetic_model)
    append_result(csv_path, synthetic_model.get_name(),
              val_loss, val_recon, val_kl, fid_score)

    # (D) Advance the chain
    curr_model = synthetic_model

    # (E) Cleanup temp dir
    del synthetic_loader
    try:
        if os.path.exists(synthetic_data_load_path):
            shutil.rmtree(synthetic_data_load_path)
            print(f"[CLEAN] Removed temp dir: {synthetic_data_load_path}")
    except Exception as e:
        print(f"[WARN] Failed to remove {synthetic_data_load_path}: {e}")

    # Optional: free GPU cache between rounds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
