# -*- coding: utf-8 -*-
"""
End-to-end CVAE pipeline that:
1) Trains on REAL MNIST (sample size is a parameter) -> saves model_00_real.pth
2) Iteratively refines K times WITHOUT saving any data:
   - Use current model to generate attempts
   - Filter on-the-fly with a discriminator
   - Train the next CVAE ONLY on filtered stream
   - Save only model weights per round (no data saved)
"""
import sys
from torchvision import datasets, transforms
import torch
from torchvision.utils import make_grid

import random
from torch.utils.data import TensorDataset
sys.path.append("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")
import torch.nn.functional as F
from cvae_model import CVAE, cvae_loss
from torch.utils.data import Subset
import os
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torch.utils.data import IterableDataset, DataLoader, Subset, TensorDataset
import torch.nn.functional as F


# Resolve project root relative to this file and ensure imports work from here
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# Your local modules must be importable from ROOT
from cvae_model import CVAE, cvae_loss
from discriminator import Discriminator
from FID import calculate_fid_score

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
# model evaluation 
# -----------------------------

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from torchvision.models.inception import inception_v3
from sklearn.neighbors import NearestNeighbors

@torch.no_grad()
def extract_inception_features(dataset, batch_size=256, device=None):
    """
    Extract 2048-D features from the Inception-v3 pool3 layer.
    Returns a numpy array of shape (N, 2048) with float32 dtype.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()  # remove classification head
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
    """
    For each point in a set, compute the distance to its k-th nearest neighbor
    within the same set (excluding itself). This distance is called the "radius"
    and defines the local neighborhood size for that point.
    Returns:
        radii: (N,) array of distances to the k-th NN
        nbrs: fitted NearestNeighbors object
    """
    N = features.shape[0]
    k_eff = min(k, max(1, N - 1))
    # Use k_eff + 1 because the closest neighbor is the point itself (distance 0)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm='auto').fit(features)
    dists, idxs = nbrs.kneighbors(features)
    # Remove the self-distance (first column)
    dists_wo_self = dists[:, 1:]
    radii = dists_wo_self[:, -1]  # distance to the k-th NN (after removing self)
    return radii, nbrs


def compute_precision_recall(real_feats, gen_feats, k=5):
    """
    Compute precision, recall, and coverage as in:
    Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing 
    Generative Models" (ICLR 2019).
    
    Precision: fraction of generated samples that lie within the "real data manifold"
    Recall:    fraction of real samples that lie within the "generated data manifold"
    Coverage:  here, same definition as recall (can be defined differently if needed)
    """
    real_feats = np.asarray(real_feats, dtype=np.float32)
    gen_feats  = np.asarray(gen_feats,  dtype=np.float32)

    # Radii within each set
    radii_real, nbrs_real = _knn_radius(real_feats, k)
    radii_gen,  nbrs_gen  = _knn_radius(gen_feats,  k)

    # Precision: For each generated sample, find the nearest real sample.
    # If the distance is <= that real sample's radius, count as "in real manifold".
    distances_rg, indices_rg = nbrs_real.kneighbors(gen_feats, n_neighbors=1)
    precision = float(np.mean(distances_rg[:, 0] <= radii_real[indices_rg[:, 0]]))

    # Recall: For each real sample, find the nearest generated sample.
    # If the distance is <= that generated sample's radius, count as "in generated manifold".
    distances_gr, indices_gr = nbrs_gen.kneighbors(real_feats, n_neighbors=1)
    recall = float(np.mean(distances_gr[:, 0] <= radii_gen[indices_gr[:, 0]]))

    return precision, recall

@torch.no_grad()
def calculate_prd_scores(real_ds, synth_ds, batch_size=256, k=5):
    """
    Full pipeline: extract features for real and synthetic datasets,
    then compute precision, recall, and coverage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_feats = extract_inception_features(real_ds, batch_size=batch_size, device=device)
    gen_feats  = extract_inception_features(synth_ds, batch_size=batch_size, device=device)
    return compute_precision_recall(real_feats, gen_feats, k=k)

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

    precision, recall = calculate_prd_scores(real_ds, synthetic_ds, batch_size=256, k=5)
    
    return fid, precision, recall



import csv
from torchvision.utils import save_image

def _ensure_dirs_for_eval(work_dir: str):
    metrics_dir = os.path.join(work_dir, "metrics")
    samples_dir = os.path.join(work_dir, "samples")
    ensure_dir(metrics_dir); ensure_dir(samples_dir)
    return metrics_dir, samples_dir

import matplotlib.pyplot as plt

@torch.no_grad()
def plot_model_samples(model, save_path=None, latent_dim=20, num_classes=10, per_class=8, device=None):
    """
    从单个模型生成样本，并用 matplotlib 保存为灰度网格图。

    Args:
        model: 生成模型 (CVAE/decoder)
        save_path: str or None，保存路径；None 时只显示不保存
        latent_dim: 潜变量维度
        num_classes: 类别数 (MNIST=10)
        per_class: 每个类别展示多少张
        device: 'cuda' 或 'cpu'
    """
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


def evaluate_and_log(model, work_dir: str, tag: str,
                     latent_dim=20, num_classes=10, device=None):
    """
    计算 FID / Precision / Recall，保存到 metrics.csv，并存样本网格图。
    """
    metrics_dir, samples_dir = _ensure_dirs_for_eval(work_dir)
    csv_path = os.path.join(metrics_dir, "metrics.csv")
    sample_path = os.path.join(samples_dir, f"{tag}_grid.png")

    plot_model_samples(model, sample_path, latent_dim=latent_dim, num_classes=num_classes, per_class=8, device=device)

    fid_score, precision, recall = fid(model)   
    row = {
        "tag": tag,
        "fid": fid_score,
        "precision": precision,
        "recall": recall,
    }

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "fid", "precision", "recall"])
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[EVAL] {tag}: FID={fid_score:.3f} | Precision={precision:.4f} | Recall={recall:.4f}")
    print(f"[EVAL] Metrics appended to: {csv_path}")
    print(f"[EVAL] Sample grid saved to: {sample_path}")


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
import numpy as np
from sklearn.metrics import pairwise_distances
import torch, math

def get_embedder(discriminator: nn.Module) -> nn.Module:
    """
    Returns a module that outputs the penultimate 128-D embedding:
    Conv -> LeakyReLU -> Conv -> LeakyReLU -> Flatten -> Linear(64*7*7->128) -> LeakyReLU
    """
    layers = list(discriminator.net.children())[:-2]  # drop final Linear(128->1) and Sigmoid
    embedder = nn.Sequential(*layers)
    # share weights; keep device consistent
    embedder.to(next(discriminator.parameters()).device)
    embedder.eval()
    return embedder

def kcenter_greedy(emb, k, seed=0):
    M = emb.shape[0]
    if k <= 0: return np.empty(0, dtype=int)
    if M <= k: return np.arange(M, dtype=int)
    rng = np.random.default_rng(seed)
    centers = [int(rng.integers(M))]
    min_dist = pairwise_distances(emb[centers], emb).ravel()
    for _ in range(1, k):
        nxt = int(np.argmax(min_dist))
        centers.append(nxt)
        min_dist = np.minimum(min_dist, pairwise_distances(emb[[nxt]], emb).ravel())
    return np.array(centers, dtype=int)

def select_fixed_size_per_class_half_band(
    probs, labels_np, E_syn,
    top_pct=0.005,      
    next_pct=0.005,    
    keep_ratio=0.5,   
    num_classes=10,
    seed=0
):
   
    # to numpy
    if torch.is_tensor(probs): probs = probs.detach().cpu().numpy()
    probs = probs.reshape(-1)
    labels_np = np.asarray(labels_np).reshape(-1)
    E_syn = np.asarray(E_syn)

    all_high, all_div = [], []

    for c in range(num_classes):
        idx_c = np.where(labels_np == c)[0]
        Nc = len(idx_c)
        if Nc == 0:
            continue

        K_high = max(1, math.ceil(Nc * top_pct))
        K_band = max(0, min(Nc - K_high, math.ceil(Nc * next_pct)))  # 避免越界
        K_keep = max(0, min(K_band, int(round(K_band * keep_ratio)))) # 只保留 band 的一半

        order_c = idx_c[np.argsort(-probs[idx_c])]
        high_c  = order_c[:K_high]
        band_c  = order_c[K_high:K_high + K_band] 

        if K_keep > 0 and len(band_c) > 0:
            emb_div = E_syn[band_c]
            sel_local = kcenter_greedy(emb_div, k=min(K_keep, len(band_c)), seed=seed)
            div_c = band_c[sel_local]
        else:
            div_c = np.empty(0, dtype=int)

        all_high.append(high_c)
        all_div.append(div_c)

    high_idx = np.unique(np.concatenate(all_high)) if all_high else np.empty(0, dtype=int)
    div_idx  = np.unique(np.concatenate(all_div))  if all_div  else np.empty(0, dtype=int)
    final_idx = np.unique(np.concatenate([high_idx, div_idx]))
    return high_idx, div_idx, final_idx

def train_cvae_on_filtered_synthetic(
    data_model: torch.nn.Module,
    *,
    D: torch.nn.Module,
    upper_percent: float = 0.005,
    lower_percent: float = 0.005,
    lower_keep: float = 0.5,
    total_sample_size: int = 2_000_000,
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
    probs_list, emb_list = [], []
    N = gen_imgs_before_filter.shape[0]
    batch_size_scoring = 1024
    embedder = get_embedder(D).eval().to(device)
    D.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size_scoring):
            xb = gen_imgs_before_filter[i:i+batch_size_scoring].to(device)
            pb = D(xb).squeeze(1).detach().cpu().numpy()       # D has Sigmoid -> probs
            hb = embedder(xb)                                   # (B,128)
            hb = F.normalize(hb, p=2, dim=1).cpu().numpy()      # L2 normalize for Euclidean
            probs_list.append(pb); emb_list.append(hb)
    probs = np.concatenate(probs_list, 0)                       # (N,)
    E_syn = np.concatenate(emb_list, 0)                         # (N,128)
    # Load images and labels
    images = gen_imgs_before_filter#data['images']      # [N, 1, 28, 28]
    labels = y_before_filter #data['labels']      # [N]
    
    labels_np = y_before_filter.detach().cpu().numpy() if torch.is_tensor(y_before_filter) else np.asarray(y_before_filter)

    _, _, mask = select_fixed_size_per_class_half_band(
        probs, labels_np, E_syn,
        top_pct=upper_percent,
        next_pct=lower_percent,
        keep_ratio=lower_keep,
        num_classes=10,
        seed=0
    )

    # Apply mask
    filtered_images = images[mask]
    filtered_labels = labels[mask]

    print(f"Filtered {filtered_images.shape[0]} / {images.shape[0]} "f"(top={upper_percent:.3%}, band={lower_percent:.3%}, keep={lower_keep:.0%})")
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
    total_sample_size: int = 2_000_000,
    upper_percent: float = 0.005,
    lower_percent: float = 0.005,
    lower_keep: float = 0.5,
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
        upper_percent=upper_percent,
        lower_percent=lower_percent,
        lower_keep=lower_keep,
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
# Orchestrator: real training then K refinements 
# -----------------------------
def run_pipeline(
    *,
    # real data stage
    sample_size: int,               
    # discriminator
    discriminator_path: str,
    # refinement
    k: int = 2,
    work_dir: str = "iter_models_only",
    # generation/filtering
    total_sample_size: int = 2_000_000,
    upper_percent: float = 0.005,
    lower_percent: float = 0.005,
    lower_keep: float = 0.5,
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
    evaluate_and_log(real_model, work_dir, tag="model_00_real", 
                     latent_dim=latent_dim, num_classes=num_classes, device=device)
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
            upper_percent=upper_percent,
            lower_percent=lower_percent,
            lower_keep=lower_keep,
            latent_dim=latent_dim,
            num_classes=num_classes,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            patience=patience,
            device=device,
        )
        tag = f"model_{i+1:02d}"
        evaluate_and_log(curr_model, work_dir, tag=tag,
                         latent_dim=latent_dim, num_classes=num_classes, device=device)
        paths.append(p)
    print("\nCheckpoints saved (models only):")
    for p in paths:
        print(p)
    return paths

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MNIST CVAE iterative filtering pipeline")
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--discriminator_path", type=str, default=os.path.join(ROOT, "model_saved", "discriminator_mnist_cvae_2.pth"))
    parser.add_argument("--k", type=int, default=5, help="number of refinement rounds")
    parser.add_argument("--work_dir", type=str, default="iter_models_only_new_D_smallestsample_diverse")
    parser.add_argument("--total_sample_size", type=int, default=2_000_000, help="generate attempts per round")
    parser.add_argument("--upper_percent", type=float, default=0.005, help="top x% to always keep")
    parser.add_argument("--lower_percent", type=float, default=0.005, help="diverse percent")
    parser.add_argument("--lower_keep", type=float, default=0.5, help="keep half of the diverse percent")
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    work_dir = os.path.join(ROOT, args.work_dir)

    _ = run_pipeline(
        sample_size=args.sample_size,
        discriminator_path=args.discriminator_path,
        k=args.k,
        work_dir=work_dir,
        total_sample_size=args.total_sample_size,
        upper_percent=args.upper_percent,
        lower_percent=args.lower_percent,
        lower_keep=args.lower_keep,
        latent_dim=args.latent_dim,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )

