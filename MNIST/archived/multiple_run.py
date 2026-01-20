# -*- coding: utf-8 -*-
"""
End-to-end CVAE pipeline that:
1) Trains on REAL MNIST (sample size is a parameter) -> saves model_00_real.pth
2) Iteratively refines K times WITHOUT saving any data:
   - Use current model to generate attempts
   - Filter on-the-fly with a discriminator (p > threshold)
   - Train the next CVAE ONLY on filtered stream
   - Save only model weights per round (no data saved)
All comments in English.
"""

import os
import sys
import torch
import random
import numpy as np
from typing import Tuple, Optional
from torch.utils.data import IterableDataset, DataLoader, Subset
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

    best_loss = float("inf")
    trigger = 0
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

        # Early stopping on training loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger = 0
            torch.save(model.state_dict(), save_path)  # save best-so-far model
        else:
            trigger += 1
            print(f"[REAL] EarlyStopping counter: {trigger}/{patience}")
            if trigger >= patience:
                print("[REAL] Early stopping triggered.")
                break

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
# -----------------------------
# Streaming filtered synthetic data (NO SAVING)
# -----------------------------


class FilteredSyntheticStream(IterableDataset):
    """
    Streams filtered (x_flat, y) pairs from a generator CVAE and a Discriminator.
    Uses `generate_images_in_batches` to generate candidate images in chunks.
    No data is saved to disk.
    """
    def __init__(
        self,
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
        *,
        total_attempts: int,
        target_kept_per_epoch: int,
        latent_dim: int,
        num_classes: int,
        gen_batch_size: int,
        filter_threshold: float,
        device: str = "cuda",
        seed: int = 0,
        verbose: bool = True,
    ):
        super().__init__()
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.total_attempts = int(total_attempts)
        self.target_kept = int(target_kept_per_epoch)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.gen_batch_size = gen_batch_size
        self.filter_threshold = float(filter_threshold)
        self.device = device
        self.seed = seed
        self.verbose = verbose

    def __iter__(self):
        # Determinism: seed the global RNGs for this epoch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.gen_model.eval()
        self.disc_model.eval()

        kept = 0
        attempts_done = 0

        # Generate candidates in chunks via your helper
        for start in range(0, self.total_attempts, self.gen_batch_size):
            remaining = self.total_attempts - start
            bs = min(self.gen_batch_size, remaining)
            attempts_done += bs

            # --- Generate a chunk (on CPU; your function returns CPU tensors) ---
            # Balanced labels are already handled inside your function
            imgs_cpu, y_cpu = generate_images_in_batches(
                model=self.gen_model,
                total_samples=bs,
                latent_dim=self.latent_dim,
                num_classes=self.num_classes,
                batch_size=bs,            # single-shot for this chunk
                device=self.device        # use same device for decode; function .cpu()'s outputs
            )  # imgs_cpu: [bs, 1, 28, 28] (CPU), y_cpu: [bs] (CPU)

            # --- Score with discriminator on device ---
            with torch.no_grad():
                probs = self.disc_model(imgs_cpu.to(self.device))  # [bs, 1] or [bs]
                if probs.dim() == 2:
                    probs = probs.squeeze(1)
                mask = (probs > self.filter_threshold).to("cpu")   # boolean mask on CPU

            if mask.any():
                # Keep filtered items (yield CPU tensors)
                x_kept = imgs_cpu[mask].flatten(start_dim=1).detach().cpu()  # [k, 784]
                y_kept = y_cpu[mask].detach().cpu()
                for i in range(x_kept.size(0)):
                    yield x_kept[i], y_kept[i]
                    kept += 1
                    if kept >= self.target_kept:
                        if self.verbose:
                            pr = kept / max(1, attempts_done)
                            print(f"[STREAM] attempts={attempts_done:,} kept={kept:,} pass_rate={pr:.4f}", flush=True)
                        return

            # periodic progress log
            if self.verbose and (attempts_done % (10 * self.gen_batch_size) == 0):
                pr = kept / max(1, attempts_done)
                print(f"[STREAM] attempts={attempts_done:,} kept={kept:,} pass_rate={pr:.4f}", flush=True)

            if attempts_done >= self.total_attempts:
                if self.verbose:
                    pr = kept / max(1, attempts_done)
                    print(f"[STREAM] attempts={attempts_done:,} kept={kept:,} pass_rate={pr:.4f}", flush=True)
                break 
# -----------------------------
# Train CVAE on a streaming dataset (NO SAVING DATA)
# -----------------------------
def train_cvae_on_stream(
    stream_ds: FilteredSyntheticStream,
    *,
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

    model = CVAE(latent_dim=latent_dim, label_dim=label_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    trigger = 0
    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))

    for ep in range(1, epochs + 1):
        # Change the stream seed each epoch to avoid identical samples
        stream_ds.seed = ep

        # Re-wrap dataloader each epoch so the iterable restarts
        loader = DataLoader(stream_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        torch.set_grad_enabled(True)
        model.train()
        total_loss = 0.0
        n_samples = 0

        for x, y in loader:
            # Flatten to [B, 784]; one-hot labels
            x = x.view(-1, 784).to(device)
            y = one_hot(y, num_classes=label_dim).to(device)

            opt.zero_grad()
            recon_x, mu, logvar = model(x, y)
            loss = cvae_loss(recon_x, x, mu, logvar)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_samples += x.size(0)
            

        if n_samples == 0:
            print(f"[STREAM] Epoch {ep}: 0 samples kept; lower threshold or increase attempts.")
            break

        avg = total_loss / n_samples
        print(f"[STREAM] Epoch {ep}/{epochs} | Train Loss: {avg:.6f} | Samples: {n_samples}")

        # Early stopping on training loss
        if avg < best_loss:
            best_loss = avg
            trigger = 0
            if save_path is not None:
                torch.save(model.state_dict(), save_path)  # save best-so-far
        else:
            trigger += 1
            print(f"[STREAM] EarlyStopping counter: {trigger}/{patience}")
            if trigger >= patience:
                print("[STREAM] Early stopping triggered.")
                break

    # Ensure we have something saved
    if save_path is not None and not os.path.exists(save_path):
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
    total_attempts: int = 6_000_000,
    kept_per_epoch: int = 200_000,
    gen_batch_size: int = 10_000,
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

    # Build streaming dataset for this round
    stream = FilteredSyntheticStream(
        gen_model=curr_model,
        disc_model=disc_model,
        total_attempts=total_attempts,
        target_kept_per_epoch=kept_per_epoch,
        latent_dim=latent_dim,
        num_classes=num_classes,
        gen_batch_size=gen_batch_size,      
        filter_threshold=filter_threshold,
        device=device,
        seed=round_idx,          # different seed per round for variety
        verbose=True
    )


    # Train next model on the stream
    next_model = train_cvae_on_stream(
        stream_ds=stream,
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
    total_attempts_per_round: int = 6_000_000,
    kept_per_epoch: int = 200_000,
    gen_batch_size: int = 10_000,
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
        curr_model, p = refinement_round(
            round_idx=i,
            curr_model=curr_model,
            disc_model=D,
            work_dir=work_dir,
            total_attempts=total_attempts_per_round,
            kept_per_epoch=kept_per_epoch,
            gen_batch_size=gen_batch_size,
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
        work_dir=os.path.join(ROOT, "iter_models_only"),
        total_attempts_per_round=6_000_000,   # generate attempts per round
        kept_per_epoch=200_000,               # number of kept samples per epoch
        gen_batch_size=10_000,
        filter_threshold=0.5,
        latent_dim=20,
        num_classes=10,
        batch_size=128,
        epochs=200,
        lr=1e-3,
        patience=5,
        seed=0,
    )
