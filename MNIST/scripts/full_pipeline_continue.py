import sys
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
from torchvision.utils import make_grid
import torch.nn.functional as F


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

vae_path = "/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/conv_cvae"
sys.path.append(vae_path)

import models as models
import train_helper as train_helper
import utils as utils
import data_helper as data_helper

# Set up device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_seed = 0
torch.manual_seed(base_seed)
torch.cuda.manual_seed_all(base_seed)
np.random.seed(base_seed)
random.seed(base_seed)

ROOT = "/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/conv_cvae/continued_256_4"

model_saved_path = os.path.join(ROOT,"model_saved_more")
data_saved_path = os.path.join(ROOT,"data_saved_more")
results_saved_path = os.path.join(ROOT,"results_saved_more")
picture_saved_path = os.path.join(ROOT,"picture_saved_more")
os.makedirs(results_saved_path, exist_ok=True)

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

all_models = []
init_size=500
test_results = {"val_loss":[], "val_recon":[], "val_kl":[], "fid":[],"model_name":[]}
model_name = f"cvae_conv_init{init_size}_q0.1_256000_r78"
this_model = models.CVAE(input_dim=784, label_dim=10, latent_dim=20, name=model_name, arch="conv").to(device)
model_saved_path = "/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/conv_cvae/fixed_size_256_000/model_saved_more"
ckpt_path = os.path.join(model_saved_path, "cvae_conv_init500_q0.1_256000_r112.pth")
this_model.load_state_dict(torch.load(ckpt_path, map_location=device))

full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./data", train=False, download=True,transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ----- k-round synthetic retraining -----
k = 40                    
threshold = 0.1          
size_schedule = [256000]*k        

csv_path = os.path.join(results_saved_path, f"results_table_{init_size}_{k}rounds.csv")

curr_model = this_model
for round_id in range(113, 113+k):#range(k, 2*k + 1):
    # (A) Train a fresh discriminator for the CURRENT generator
    synthetic_size = 256000
    print(f"\n[Round {round_id}] Training discriminator for current model...")
    discriminator_dataset = data_helper.prepare_discriminator_dataset(full_dataset, curr_model, device)
    disc_loader = DataLoader(discriminator_dataset, batch_size=128, shuffle=True)
    disc_model = models.SyntheticDiscriminator(input_dim=784).to(device)
    train_helper.train_model(model=disc_model, train_loader=disc_loader, device=device,
                             epochs=80, lr=1e-3, patience=5, verbose=False)
    del disc_loader, discriminator_dataset
    # (B) Generate filtered synthetic dataset to a TEMP dir (unique per round)
    model_name = f'cvae_conv_init{init_size}_q{threshold}_{synthetic_size}_r{round_id}'
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
