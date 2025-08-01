import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm


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
