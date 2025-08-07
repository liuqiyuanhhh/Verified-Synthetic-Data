import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import resize
from torchmetrics.image.fid import FrechetInceptionDistance

class SimpleMNISTFeatureExtractor(nn.Module):
    """A small CNN to extract features from MNIST-style grayscale images."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # -> [batch, 64, 4, 4]
        )

    def forward(self, x):
        return self.encoder(x).view(x.size(0), -1)  # Flatten to [batch, 1024]


@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    for x, _ in tqdm(dataloader, desc="Extracting features"):
        x = x.to(device)
        f = model(x).cpu()
        features.append(f)
    return torch.cat(features, dim=0).numpy()


def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Compute Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)


def calculate_fid_score(dataset1, dataset2, batch_size=256, device=None):
    """
    Calculate FID between two datasets of shape [N, 1, 28, 28].
    
    Args:
        dataset1: torch.utils.data.Dataset (e.g., real MNIST)
        dataset2: torch.utils.data.Dataset (e.g., generated MNIST)
        batch_size: int
        device: 'cuda' or 'cpu'

    Returns:
        FID score (float)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)

    model = SimpleMNISTFeatureExtractor().to(device)

    feats1 = extract_features(model, loader1, device)
    feats2 = extract_features(model, loader2, device)

    mu1, sigma1 = feats1.mean(0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(0), np.cov(feats2, rowvar=False)

    fid = calculate_fid(mu1, sigma1, mu2, sigma2)
    return fid


@torch.no_grad()
def calculate_fid_score_2(real_ds, synth_ds, batch_size: int = 256) -> float:
    """
    FID(real_ds, synth_ds) – works for 1- or 3-channel tensors in [0,1] or uint8.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def preprocess(batch: torch.Tensor) -> torch.Tensor:
        """
        Make (N,3,299,299) uint8 as required by torch-fidelity's Inception-v3.
        """
        if batch.dim() == 3:                               # (C,H,W) or (H,W,C)
            batch = batch.unsqueeze(0)                     # -> (1,C,H,W) / (1,H,W,C)

        if batch.dim() == 4 and batch.shape[1] not in {1, 3}:
            batch = batch.permute(0, 3, 1, 2)              # channel-last -> channel-first

        if batch.shape[1] == 1:                            # grayscale -> RGB
            batch = batch.repeat(1, 3, 1, 1)

        batch = resize(batch.float(), [299, 299], antialias=True)  # 299×299, float

        if batch.max() <= 1.0:                             # [0,1] -> [0,255] uint8
            batch = (batch * 255).clamp(0, 255)
        return batch.to(torch.uint8)

    for loader, is_real in [
        (DataLoader(real_ds,   batch_size=batch_size, shuffle=False), True),
        (DataLoader(synth_ds,  batch_size=batch_size, shuffle=False), False),
    ]:
        for batch in loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            fid.update(preprocess(imgs).to(device), real=is_real)

    score = float(fid.compute())
    fid.reset()
    return score
