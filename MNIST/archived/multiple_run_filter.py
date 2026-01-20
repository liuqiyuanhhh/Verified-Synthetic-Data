# -*- coding: utf-8 -*-
"""
End-to-end CVAE pipeline that:
1) Trains on REAL MNIST (sample size is a parameter) -> saves model_00_real.pth
2) Iteratively refines K times WITHOUT saving any data:
   - Use current model to generate attempts
   - Filter on-the-fly with a discriminator (p > threshold)
   - Train the next CVAE ONLY on filtered stream
   - Save only model weights per round (no data saved)
"""

import os
import sys
import torch
import random
import numpy as np
from typing import Tuple, Optional
from torch.utils.data import IterableDataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F


# Resolve project root relative to this file and ensure imports work from here
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# Your local modules must be importable from ROOT
from cvae_model import CVAE, cvae_loss
from discriminator import Discriminator


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
    """Load state dict safely; silence FutureWarning when torch supports weights_only."""
    try:
        return torch.load(path, map_location=device, weights_only=True)  # torch >= 2.4
    except TypeError:
        return torch.load(path, map_location=device)


# -----------------------------
# REAL data training
# -----------------------------
def train_cvae_on_real(
    *,
    sample_size: int,
    latent_dim: int = 20,
    label_dim: int = 10,
    batch_size: int = 128,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 5,
    save_dir: str = "model_saved",
    save_name: str = "model_00_real.pth",
    device: Optional[str] = None,
    seed: int = 0,
) -> Tuple[torch.nn.Module, str]:
    """
    Train CVAE on real MNIST (first `sample_size` samples) and save weights.
    Only model weights are saved; no data saved anywhere.
    """
    set_seed(seed)
    ensure_dir(save_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MNIST train set
    transform = transforms.ToTensor()
    full_dataset = datasets.MNIST(root=os.path.join(ROOT, "data"), train=True, download=True, transform=transform)
    dataset = Subset(full_dataset, range(sample_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Init model/optimizer
    model = CVAE(latent_dim=latent_dim, label_dim=label_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    save_path = os.path.join(save_dir, save_name)

    for ep in range(1, epochs + 1):
        torch.set_grad_enabled(True)
        model.train()
        total_loss = 0.0
        n_samples = 0

        for x, y in loader:
            # Flatten to [B, 784]; one-hot labels
            x = x.view(-1, 784).to(device)
            y = one_hot(y).to(device)

            opt.zero_grad()
            recon_x, mu, logvar = model(x, y)
            loss = cvae_loss(recon_x, x, mu, logvar)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader.dataset)
        print(f"[REAL] Epoch {ep}/{epochs} | Train Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    # Load best and return both model and path
    state = safe_load(save_path, device)
    model.load_state_dict(state)
    model.eval()
    print(f"[REAL] Saved model -> {save_path}")
    return model, save_path

def generate_images_in_batches(model, total_samples, latent_dim, num_classes, batch_size=10000, device='cuda'):
    model.eval()
    generated_images = []
    all_labels = []

    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        batch_size_actual = end - start

        # Generate z and y
        z = torch.randn(batch_size_actual, latent_dim).to(device)
        y = torch.arange(num_classes).repeat_interleave(total_samples // num_classes)[start:end]
        y_onehot = F.one_hot(y, num_classes=num_classes).float().to(device)

        with torch.no_grad():
            imgs = model.decode(z, y_onehot).view(-1, 1, 28, 28).cpu()
            generated_images.append(imgs)
            all_labels.append(y)

    images = torch.cat(generated_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return images, labels

def train_cvae_on_filtered_synthetic(
    data_model: torch.nn.Module,
    *,
    D: torch.nn.Module,
    filter_threshold: float=0.5,
    total_sample_size: int = 6_000_000,
    latent_dim: int = 20,
    label_dim: int = 10,
    batch_size: int = 128,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 5,
    save_path: Optional[str] = None,
    device: Optional[str] = None,
) -> torch.nn.Module:
    """
    Train a CVAE using a streaming filtered dataset.
    Only model weights are saved. No data is written to disk.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_imgs_before_filter,y_before_filter = generate_images_in_batches(
        model=data_model,
        total_samples=total_sample_size,
        latent_dim=latent_dim,
        num_classes=10,
        batch_size=10000,
        device=device
    )
    synthetic_loader = DataLoader(gen_imgs_before_filter, batch_size=512)

    all_probs = []

    with torch.no_grad():
        for batch in synthetic_loader:
            batch = batch.to(device)
            probs = D(batch)  # [batch_size, 1], already sigmoid activated
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    # Flatten probs to shape [N]
    probs = all_probs.squeeze(1)

    # Load images and labels
    images = gen_imgs_before_filter#data['images']      # [N, 1, 28, 28]
    labels = y_before_filter #data['labels']      # [N]
    # Create mask for p > filter_threshold
    #mask = probs > filter_threshole
    # create mask for top 1% largest probs synthetic data
    mask = probs >= torch.quantile(probs, 0.95)

    # Apply mask
    filtered_images = images[mask]
    filtered_labels = labels[mask]

    print(f"Filtered {filtered_images.shape[0]} samples out of {images.shape[0]} total generated samples using threshold {filter_threshold}")

    ######## filtered synthetic data training ########
    images = filtered_images  # shape: [N, 1, 28, 28]
    labels = filtered_labels  # shape: [N]

    # Preprocess: flatten images and convert labels to one-hot
    images = images.view(-1, 784)  # flatten to [N, 784]

    # Create dataset and dataloader
    dataset = TensorDataset(images, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = CVAE(latent_dim=latent_dim, label_dim=label_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x = x.view(-1, 784).to(device)
            y = one_hot(y).to(device)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x, y)
            loss = cvae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{ep+1}/{epochs}], Train Loss: {avg_loss:.4f}")
        
    # Ensure we have something saved
    if save_path is not None:
                torch.save(model.state_dict(), save_path)

    # Load best checkpoint to return
    state = safe_load(save_path, device) if save_path is not None else model.state_dict()
    if save_path is not None:
        model.load_state_dict(state)
    model.eval()
    return model


# -----------------------------
# One refinement round: model_i -> model_{i+1}
# -----------------------------
def refinement_round(
    *,
    round_idx: int,
    curr_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    work_dir: str,
    # generation/filtering
    total_sample_size: int = 6_000_000,
    filter_threshold: float = 0.5,
    # training
    latent_dim: int = 20,
    num_classes: int = 10,
    batch_size: int = 128,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 5,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, str]:
    """
    Use `curr_model` to stream filtered synthetic data and train the next CVAE.
    Saves ONLY the next model weights to disk. Returns (next_model, save_path).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ensure_dir(work_dir)
    save_path = os.path.join(work_dir, f"model_{round_idx+1:02d}.pth")

    # Train next model on the stream
    next_model = train_cvae_on_filtered_synthetic(
        data_model=curr_model,
        D=disc_model,
        total_sample_size=total_sample_size,
        filter_threshold=filter_threshold,
        latent_dim=latent_dim,
        label_dim=num_classes,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience,
        save_path=save_path,
        device=device,
    )
    print(f"[ROUND {round_idx}] Saved model -> {save_path}")
    return next_model, save_path

def train_D(fake_imgs,epochs):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

    # real MNIST (60,000）
    transform = transforms.ToTensor()
    mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
    real_imgs = torch.stack([mnist[i][0] for i in range(60000)])  # shape: [60000, 1, 28, 28]
    real_labels = torch.ones(60000, 1)  # label = 1

    # synthetic data（60,000）
    fake_labels = torch.zeros(60000, 1)  # label = 0

    X_all = torch.cat([real_imgs, fake_imgs], dim=0)
    y_all = torch.cat([real_labels, fake_labels], dim=0)

    perm = torch.randperm(len(X_all))
    X_all = X_all[perm]
    y_all = y_all[perm]

    from torch.utils.data import TensorDataset
    disc_dataset = TensorDataset(X_all, y_all)
    disc_loader = DataLoader(disc_dataset, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from discriminator import Discriminator
    from torch import nn
    D = Discriminator().to(device)
    optimizer = torch.optim.Adam(D.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):

        D.train()
        total_loss = 0
        correct = 0
        total = 0

        for x_batch, y_batch in disc_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            preds = D(x_batch)
            loss = loss_fn(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (preds > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    D.eval()
    return D

# -----------------------------
# Orchestrator: real training then K refinements (NO DATA SAVED)
# -----------------------------
def run_pipeline(
    *,
    # real data stage
    sample_size: int,                 # <<-- parameterized real sample size
    # discriminator
    discriminator_path: str,
    # refinement
    k: int = 2,
    work_dir: str = "iter_models_only",
    # generation/filtering
    total_sample_size: int = 6_000_000,
    filter_threshold: float = 0.5,
    # training hyperparams (both stages)
    latent_dim: int = 20,
    num_classes: int = 10,
    batch_size: int = 128,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 0,
):
    """
    1) Train on REAL MNIST with `sample_size` and save model_00_real.pth
    2) For i in 0..k-1:
         curr_model -> (generate+filter stream) -> train -> save model_{i+1:02d}.pth
    Only model weights are saved. No data is saved.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(work_dir)

    # 1) Real data training -> model_00_real.pth
    real_model, real_path = train_cvae_on_real(
        sample_size=sample_size,
        latent_dim=latent_dim,
        label_dim=num_classes,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience,
        save_dir=work_dir,
        save_name="model_00_real.pth",
        device=device,
        seed=seed,
    )

    # 2) Load discriminator
    D = Discriminator().to(device)
    D.load_state_dict(safe_load(discriminator_path, device))
    D.eval()
    
    # 3) K refinement rounds (no data saved)
    curr_model = real_model
    paths = [real_path]
    for i in range(k):
        print(f"\n========== Refinement Round {i} ==========")

        print("[Disc] Sampling new synthetic for discriminator training...")
        synth_imgs, _ = generate_images_in_batches(
            model=curr_model,
            total_samples=60000,
            latent_dim=latent_dim,
            num_classes=10,
            batch_size=10000,
            device=device
        )

        D = train_D(synth_imgs, 10)

        curr_model, p = refinement_round(
            round_idx=i,
            curr_model=curr_model,
            disc_model=D,
            work_dir=work_dir,
            total_sample_size=total_sample_size,
            filter_threshold=filter_threshold,
            latent_dim=latent_dim,
            num_classes=num_classes,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            patience=patience,
            device=device,
        )
        paths.append(p)

    print("\nCheckpoints saved (models only):")
    for p in paths:
        print(p)
    return paths


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Discriminator path relative to ROOT (adjust if needed)
    DISC_PATH = os.path.join(ROOT, "model_saved", "discriminator_mnist_cvae_2.pth")

    _ = run_pipeline(
        sample_size=5000,                     # real data count
        discriminator_path=DISC_PATH,
        k=5,                                  # number of refinement rounds
        work_dir=os.path.join(ROOT, "iter_models_only_new_D"),
        total_sample_size=6_000_000,          # generate attempts per round
        filter_threshold=0.5,
        latent_dim=20,
        num_classes=10,
        batch_size=128,
        epochs=200,
        lr=1e-3,
        patience=5,
        seed=0,
    )
