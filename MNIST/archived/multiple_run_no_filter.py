# -*- coding: utf-8 -*-
"""
End-to-end CVAE pipeline that:
1) Trains on REAL MNIST (sample size is a parameter) -> saves model_00_real.pth
2) Iteratively refines K times WITHOUT saving any data:
   - Use current model to generate attempts
   - (NO FILTERING) Train the next CVAE directly on generated stream
   - Save only model weights per round (no data saved)
Outputs (consistent with filter pipeline):
- work_dir/metrics/metrics.csv    (tag,fid,precision,recall; appended each step)
- work_dir/samples/{tag}_grid.png (sample grids per step)
All comments in English.
"""

import os, sys, random, math
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

# ---------- project root ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# ---------- local modules ----------
from cvae_model import CVAE, cvae_loss
from discriminator import Discriminator
from FID import calculate_fid_score   # keep same FID impl as filter pipeline

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes).float()

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def safe_load(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)  # torch >= 2.4
    except TypeError:
        return torch.load(path, map_location=device)

# -----------------------------
# Evaluation (matches filter pipeline)
# -----------------------------
import csv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from torchvision.models.inception import inception_v3
from sklearn.neighbors import NearestNeighbors

def _ensure_dirs_for_eval(work_dir: str):
    metrics_dir = os.path.join(work_dir, "metrics")
    samples_dir = os.path.join(work_dir, "samples")
    ensure_dir(metrics_dir); ensure_dir(samples_dir)
    return metrics_dir, samples_dir

@torch.no_grad()
def plot_model_samples(model, save_path=None, latent_dim=20, num_classes=10, per_class=8, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    z = torch.randn(num_classes * per_class, latent_dim, device=device)
    y = torch.arange(num_classes).repeat_interleave(per_class).to(device)
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    imgs = model.decode(z, y_onehot).view(-1, 1, 28, 28).cpu()

    fig, axes = plt.subplots(num_classes, per_class, figsize=(2*per_class, 2*num_classes))
    for c in range(num_classes):
        for j in range(per_class):
            idx = c * per_class + j
            axes[c, j].imshow(imgs[idx].squeeze(), cmap='gray')
            axes[c, j].axis('off')
            if j == 0:
                axes[c, j].set_ylabel(f"Class {c}", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[SAVE] Sample grid saved -> {save_path}")
    plt.close(fig)

@torch.no_grad()
def extract_inception_features(dataset, batch_size=256, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()
    model.eval()

    feats = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if imgs.shape[1] == 1:  # grayscale -> RGB
            imgs = imgs.repeat(1, 3, 1, 1)
        imgs = resize(imgs, [299, 299], antialias=True).to(device, dtype=torch.float32)
        if imgs.max() <= 1:
            imgs = imgs * 255.0
        feats.append(model(imgs).cpu())
    feats = torch.cat(feats, dim=0).numpy().astype(np.float32)
    return feats

def _knn_radius(features, k):
    N = features.shape[0]
    k_eff = min(k, max(1, N - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm='auto').fit(features)
    dists, _ = nbrs.kneighbors(features)
    dists_wo_self = dists[:, 1:]
    radii = dists_wo_self[:, -1]
    return radii, nbrs

def _compute_precision_recall(real_feats, gen_feats, k=5):
    real_feats = np.asarray(real_feats, dtype=np.float32)
    gen_feats  = np.asarray(gen_feats,  dtype=np.float32)
    radii_real, nbrs_real = _knn_radius(real_feats, k)
    radii_gen,  nbrs_gen  = _knn_radius(gen_feats,  k)

    distances_rg, indices_rg = nbrs_real.kneighbors(gen_feats, n_neighbors=1)
    precision = float(np.mean(distances_rg[:, 0] <= radii_real[indices_rg[:, 0]]))

    distances_gr, indices_gr = nbrs_gen.kneighbors(real_feats, n_neighbors=1)
    recall = float(np.mean(distances_gr[:, 0] <= radii_gen[indices_gr[:, 0]]))
    return precision, recall

@torch.no_grad()
def calculate_prd_scores(real_ds, synth_ds, batch_size=256, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_feats = extract_inception_features(real_ds, batch_size=batch_size, device=device)
    gen_feats  = extract_inception_features(synth_ds, batch_size=batch_size, device=device)
    return _compute_precision_recall(real_feats, gen_feats, k=k)

@torch.no_grad()
def fid(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    synthetic_gen_size = 6000
    gen_imgs, y = generate_images_in_batches(
        model=model,
        total_samples=synthetic_gen_size,
        latent_dim=20,
        num_classes=10,
        batch_size=10000,
        device=device
    )

    transform = transforms.ToTensor()
    real_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    synthetic_ds = TensorDataset(gen_imgs, y)

    fid_score = calculate_fid_score(real_ds, synthetic_ds)
    precision, recall = calculate_prd_scores(real_ds, synthetic_ds, batch_size=256, k=5)
    return fid_score, precision, recall

def evaluate_and_log(model, work_dir: str, tag: str, latent_dim=20, num_classes=10, device=None):
    metrics_dir, samples_dir = _ensure_dirs_for_eval(work_dir)
    csv_path = os.path.join(metrics_dir, "metrics.csv")
    sample_path = os.path.join(samples_dir, f"{tag}_grid.png")

    plot_model_samples(model, sample_path, latent_dim=latent_dim, num_classes=num_classes, per_class=8, device=device)
    fid_score, precision, recall = fid(model)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "fid", "precision", "recall"])
        if write_header:
            writer.writeheader()
        writer.writerow({"tag": tag, "fid": fid_score, "precision": precision, "recall": recall})

    print(f"[EVAL] {tag}: FID={fid_score:.3f} | Precision={precision:.4f} | Recall={recall:.4f}")
    print(f"[EVAL] Metrics appended to: {csv_path}")
    print(f"[EVAL] Sample grid saved to: {sample_path}")

# -----------------------------
# REAL data training
# -----------------------------
def train_cvae_on_real(
    *, sample_size: int, latent_dim: int = 20, label_dim: int = 10,
    batch_size: int = 128, epochs: int = 200, lr: float = 1e-3, patience: int = 5,
    save_dir: str = "model_saved", save_name: str = "model_00_real.pth",
    device: Optional[str] = None, seed: int = 0,
) -> Tuple[torch.nn.Module, str]:
    set_seed(seed); ensure_dir(save_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    full_dataset = datasets.MNIST(root=os.path.join(ROOT, "data"), train=True, download=True, transform=transform)
    dataset = Subset(full_dataset, range(sample_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = CVAE(latent_dim=latent_dim, label_dim=label_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    save_path = os.path.join(save_dir, save_name)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x = x.view(-1, 784).to(device)
            y = one_hot(y).to(device)
            opt.zero_grad()
            recon_x, mu, logvar = model(x, y)
            loss = cvae_loss(recon_x, x, mu, logvar)
            loss.backward(); opt.step()
            total_loss += loss.item()
        avg = total_loss / len(loader.dataset)
        print(f"[REAL] Epoch {ep}/{epochs} | Train Loss: {avg:.6f}")

    torch.save(model.state_dict(), save_path)
    state = safe_load(save_path, device); model.load_state_dict(state); model.eval()
    print(f"[REAL] Saved model -> {save_path}")
    return model, save_path

# -----------------------------
# Generation (no filter pipeline reuses this)
# -----------------------------
def generate_images_in_batches(model, total_samples, latent_dim, num_classes, batch_size=10000, device='cuda'):
    model.eval()
    generated_images, all_labels = [], []
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        bs = end - start
        z = torch.randn(bs, latent_dim).to(device)
        # balanced labels
        y = torch.arange(num_classes).repeat_interleave(total_samples // num_classes)[start:end].to(device)
        y_onehot = F.one_hot(y, num_classes=num_classes).float()
        with torch.no_grad():
            imgs = model.decode(z, y_onehot).view(-1, 1, 28, 28).cpu()
        generated_images.append(imgs); all_labels.append(y.cpu())
    images = torch.cat(generated_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return images, labels

# -----------------------------
# Train next model on unfiltered synthetic stream
# -----------------------------
def train_cvae_on_unfiltered_stream(
    data_model: torch.nn.Module,
    *, total_sample_size: int = 6_000_000, latent_dim: int = 20, label_dim: int = 10,
    batch_size: int = 128, epochs: int = 200, lr: float = 1e-3, patience: int = 5,
    save_path: Optional[str] = None, device: Optional[str] = None,
) -> torch.nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = generate_images_in_batches(
        model=data_model, total_samples=total_sample_size,
        latent_dim=latent_dim, num_classes=label_dim, batch_size=10000, device=device
    )
    images = images.view(-1, 784)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CVAE(latent_dim=latent_dim, label_dim=label_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if save_path: ensure_dir(os.path.dirname(save_path))
    for ep in range(1, epochs + 1):
        model.train(); total_loss = 0.0
        for x, y in loader:
            x = x.view(-1, 784).to(device)
            y = one_hot(y).to(device)
            opt.zero_grad()
            recon_x, mu, logvar = model(x, y)
            loss = cvae_loss(recon_x, x, mu, logvar)
            loss.backward(); opt.step()
            total_loss += loss.item()
        avg = total_loss / len(loader.dataset)
        print(f"[UNFILT] Epoch {ep}/{epochs} | Train Loss: {avg:.6f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        state = safe_load(save_path, device); model.load_state_dict(state)
    model.eval()
    return model

# -----------------------------
# One refinement round (NO FILTER)
# -----------------------------
def refinement_round(
    *, round_idx: int, curr_model: torch.nn.Module, work_dir: str,
    total_sample_size: int = 6_000_000, latent_dim: int = 20, num_classes: int = 10,
    batch_size: int = 128, epochs: int = 200, lr: float = 1e-3, patience: int = 5,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, str]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(work_dir)
    save_path = os.path.join(work_dir, f"model_{round_idx+1:02d}.pth")
    next_model = train_cvae_on_unfiltered_stream(
        data_model=curr_model, total_sample_size=total_sample_size,
        latent_dim=latent_dim, label_dim=num_classes, batch_size=batch_size,
        epochs=epochs, lr=lr, patience=patience, save_path=save_path, device=device,
    )
    print(f"[ROUND {round_idx}] Saved model -> {save_path}")
    return next_model, save_path

# -----------------------------
# Orchestrator
# -----------------------------
def run_pipeline(
    *, sample_size: int, discriminator_path: str,  # kept for signature parity
    k: int = 2, work_dir: str = "iter_models_only",
    total_sample_size: int = 6_000_000,
    latent_dim: int = 20, num_classes: int = 10,
    batch_size: int = 128, epochs: int = 200, lr: float = 1e-3,
    patience: int = 5, seed: int = 0,
):
    """
    1) Train on REAL MNIST and save model_00_real.pth
    2) For i in 0..k-1: generate (unfiltered) -> train next -> save model_{i+1:02d}.pth
    Also evaluate+log after each checkpoint, matching the filter pipeline outputs.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(work_dir)

    # 1) Real data training
    real_model, real_path = train_cvae_on_real(
        sample_size=sample_size, latent_dim=latent_dim, label_dim=num_classes,
        batch_size=batch_size, epochs=epochs, lr=lr, patience=patience,
        save_dir=work_dir, save_name="model_00_real.pth", device=device, seed=seed,
    )
    # --- Evaluate & log: model_00_real ---
    evaluate_and_log(real_model, work_dir, tag="model_00_real",
                     latent_dim=latent_dim, num_classes=num_classes, device=device)

    # (discriminator is unused here but kept for API parity)
    _ = discriminator_path

    # 2) K refinement rounds
    curr_model = real_model
    paths = [real_path]
    for i in range(k):
        print(f"\n========== Refinement Round {i} (NO FILTER) ==========")
        curr_model, p = refinement_round(
            round_idx=i, curr_model=curr_model, work_dir=work_dir,
            total_sample_size=total_sample_size, latent_dim=latent_dim,
            num_classes=num_classes, batch_size=batch_size, epochs=epochs,
            lr=lr, patience=patience, device=device,
        )
        tag = f"model_{i+1:02d}"
        evaluate_and_log(curr_model, work_dir, tag=tag,
                         latent_dim=latent_dim, num_classes=num_classes, device=device)
        paths.append(p)

    print("\nCheckpoints saved (models only):")
    for p in paths:
        print(p)
    return paths

# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":
    DISC_PATH = os.path.join(ROOT, "model_saved", "discriminator_mnist_cvae_2.pth")
    _ = run_pipeline(
        sample_size=5000,
        discriminator_path=DISC_PATH,   # unused in no-filter path
        k=10,
        work_dir=os.path.join(ROOT, "iter_models_only_no_filter_2"),
        total_sample_size=10500,
        latent_dim=20,
        num_classes=10,
        batch_size=128,
        epochs=150,
        lr=1e-3,
        patience=100,
        seed=0,
    )
