import sys
import os
import argparse
import random
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.append(str(SRC_DIR))

import models as models
import train_helper as train_helper
import utils as utils
import data_helper as data_helper
from FID import calculate_fid_score

# --------------------- utils: device & seed ---------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# --------------------- helper: result append ---------------------
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

def get_model_name(model):
    # robust to either .get_name() or .name
    if hasattr(model, "get_name"):
        return model.get_name()
    elif hasattr(model, "name"):
        return model.name
    else:
        return "unnamed_model"

# --------------------- helper: sampling & plotting ---------------------
@torch.no_grad()
def generate_images_in_batches(
    model,
    total_samples: int,
    latent_dim: int,
    num_classes: int,
    batch_size: int = 10000,
    device: str = "cuda",
):
    """
    Generate `total_samples` balanced across num_classes using the CVAE decoder.
    Returns:
        images: (N, 1, 28, 28) tensor on CPU
        labels: (N,) long tensor on CPU
    """
    model.eval()
    model.to(device)

    generated_images = []
    all_labels = []

    # Balanced labels: 0,1,...,9,0,1,... repeated
    labels_full = torch.arange(total_samples) % num_classes

    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        n = end - start

        # Sample latent z
        z = torch.randn(n, latent_dim, device=device)

        # Labels for this batch
        y = labels_full[start:end].to(device)
        y_onehot = F.one_hot(y, num_classes=num_classes).float().to(device)

        # Decode logits → sigmoid → [0,1]
        logits_flat = model.decoder.decode(z, y_onehot)     # (n, 784)
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

    # latent + labels
    z = torch.randn(num_classes * per_class, latent_dim, device=device)
    y = torch.arange(num_classes, device=device).repeat_interleave(per_class)
    y_onehot = F.one_hot(y, num_classes=num_classes).float().to(device)

    # decode -> logits, then map to [0,1]
    logits_flat = model.decoder.decode(z, y_onehot)            # (n, 784)
    imgs = torch.sigmoid(logits_flat).view(-1, 1, 28, 28)

    # move for plotting
    imgs_np = imgs.detach().cpu().numpy()

    fig, axes = plt.subplots(num_classes, per_class, figsize=(2 * per_class, 2 * num_classes))
    for c in range(num_classes):
        for j in range(per_class):
            idx = c * per_class + j
            axes[c, j].imshow(imgs_np[idx].squeeze(), cmap="gray")
            axes[c, j].axis("off")
            if j == 0:
                axes[c, j].set_ylabel(f"Class {c}", fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[SAVE] Sample grid saved -> {save_path}")
    return fig, axes

# --------------------- helper: FID ---------------------
def compute_fid(model, device, real_dataset, latent_dim=20, num_classes=10,
                n_samples=6000, batch_size=1000):
    """
    Compute FID between real_dataset (MNIST test) and synthetic samples from `model`.
    """
    model = model.to(device).eval()
    images, labels = generate_images_in_batches(
        model=model,
        total_samples=n_samples,
        latent_dim=latent_dim,
        num_classes=num_classes,
        batch_size=batch_size,
        device=device,
    )
    synth_ds = TensorDataset(images, labels)
    fid_value = calculate_fid_score(real_dataset, synth_ds)
    return float(fid_value)

# --------------------- main pipeline ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-size", type=int, default=int(os.getenv("INIT_SIZE", "500")),
                        help="Initial real subset size")
    parser.add_argument("--rounds", type=int, default=40,
                        help="Number of retraining rounds")
    parser.add_argument("--synthetic-per-round", type=int, default=20000,
                        help="Synthetic samples per round")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Quantile / selection threshold for discriminator filtering")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    init_size = int(args.init_size)
    k = int(args.rounds)
    synthetic_per_round = int(args.synthetic_per_round)
    threshold = float(args.threshold)
    seed = int(args.seed)

    # device & seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    set_seed(seed)

    # --------------------- paths ---------------------
    BASE_ROOT = "/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/conv_cvae/schedule_size_linear_no_filter"
    ROOT = os.path.join(BASE_ROOT, f"init_{init_size}")
    model_saved_path = os.path.join(ROOT, "model_saved_no_filter_more")
    data_saved_path = os.path.join(ROOT, "data_saved_no_filter_more")
    results_saved_path = os.path.join(ROOT, "results_saved_no_filter_more")
    picture_saved_path = os.path.join(ROOT, "picture_saved_no_filter_more")

    os.makedirs(model_saved_path, exist_ok=True)
    os.makedirs(data_saved_path, exist_ok=True)
    os.makedirs(results_saved_path, exist_ok=True)
    os.makedirs(picture_saved_path, exist_ok=True)

    # --------------------- data ---------------------
    transform = transforms.ToTensor()
    full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # real dataset for FID (use test set)
    fid_real_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # balanced indices for initial subset
    base_seed = seed
    full_digit_indices = utils.create_balanced_subset_indices(full_dataset, seed=base_seed)

    # --------------------- size schedule ---------------------
    size_schedule = [synthetic_per_round] * k

    # --------------------- bookkeeping ---------------------
    test_results = {
        "model_name": [],
        "val_loss": [],
        "val_recon": [],
        "val_kl": [],
        "fid": [],
    }
    csv_path = os.path.join(results_saved_path, f"results_table_{init_size}_{k}rounds.csv")

    # --------------------- initial model ---------------------
    print(f"[Init] Training initial model with {init_size} real samples...")
    init_subset = utils.get_balanced_subset(full_digit_indices, init_size)
    init_dataset = Subset(full_dataset, init_subset)
    init_loader = DataLoader(init_dataset, batch_size=128, shuffle=True)

    this_model = models.CVAE(
        input_dim=784, label_dim=10, latent_dim=20,
        name=f"cvae_conv_real_{init_size}", arch="conv"
    ).to(device)

    train_helper.train_model(this_model, init_loader, device,
                             epochs=200, lr=1e-3, patience=5, verbose=False)

    # sample grid
    plot_model_samples(
        this_model,
        save_path=os.path.join(picture_saved_path, f"initial_model_samples_{init_size}.png"),
        device=device,
    )

    # validation + FID
    val_loss, val_recon, val_kl = train_helper.calculate_validation_loss(this_model, test_loader, device)
    fid_score = compute_fid(this_model, device, fid_real_ds)

    name0 = get_model_name(this_model)
    print(f"[Init] {init_size} samples | Model: {name0} | "
          f"Val Loss: {val_loss:.4f} | KL: {val_kl:.4f} | Recon: {val_recon:.4f} | FID: {fid_score:.4f}")

    # log + save
    test_results["model_name"].append(name0)
    test_results["val_loss"].append(val_loss)
    test_results["val_recon"].append(val_recon)
    test_results["val_kl"].append(val_kl)
    test_results["fid"].append(fid_score)

    utils.save_model(this_model, name0, model_saved_path)
    append_result(csv_path, name0, val_loss, val_recon, val_kl, fid_score)

    curr_model = this_model

    # --------------------- retraining rounds ---------------------
    for round_id in range(1, k + 1):
        synthetic_size = int(size_schedule[round_id - 1])
        print(f"\n[Round {round_id}] starting; synthetic_size = {synthetic_size}")

        # (A) Train discriminator for current generator
        print(f"[Round {round_id}] Preparing discriminator dataset...")
        discriminator_dataset = data_helper.prepare_discriminator_dataset(full_dataset, curr_model, device)
        disc_loader = DataLoader(discriminator_dataset, batch_size=128, shuffle=True)

        print(f"[Round {round_id}] Training discriminator...")
        disc_model = models.SyntheticDiscriminator(input_dim=784).to(device)
        train_helper.train_model(
            model=disc_model,
            train_loader=disc_loader,
            device=device,
            epochs=80,
            lr=1e-3,
            patience=5,
            verbose=False,
        )
        del disc_loader, discriminator_dataset

        # (B) Generate filtered synthetic dataset on disk
        model_name = f"cvae_conv_init{init_size}_q{threshold}_{synthetic_size}_r{round_id}"
        synthetic_data_dir = os.path.join(data_saved_path, model_name)
        os.makedirs(synthetic_data_dir, exist_ok=True)

        print(f"[Round {round_id}] Generating filtered synthetic data -> {synthetic_data_dir}")
        data_helper.generate_balanced_images_with_filtering(
            model=curr_model,
            save_directory=synthetic_data_dir,
            total_samples=synthetic_size,
            discriminator=disc_model,
            selection_threshold=threshold,
            verbose=False,
            use_quantile_filtering=True,   # important: quantile mode
        )

        # (C) Train new CVAE on filtered synthetic data
        print(f"[Round {round_id}] Creating dataloader for synthetic data...")
        synthetic_loader = data_helper.create_directory_based_dataloader(
            synthetic_data_dir,
            batch_size=128,
        )

        synthetic_model = models.CVAE(
            input_dim=784,
            label_dim=10,
            latent_dim=20,
            name=model_name,
            arch="conv",
        ).to(device)

        print(f"[Round {round_id}] Training synthetic model {model_name} ...")
        train_helper.train_model(
            synthetic_model,
            synthetic_loader,
            device=device,
            epochs=200,
            lr=1e-3,
            patience=5,
            verbose=False,
        )

        # sample grid
        plot_model_samples(
            synthetic_model,
            save_path=os.path.join(picture_saved_path, f"round{round_id}_model_samples.png"),
            device=device,
        )

        # validation + FID
        val_loss, val_recon, val_kl = train_helper.calculate_validation_loss(synthetic_model, test_loader, device)
        fid_score = compute_fid(synthetic_model, device, fid_real_ds)

        name_r = get_model_name(synthetic_model)
        print(f"[Round {round_id}] Model: {name_r} | "
              f"Val Loss: {val_loss:.4f} | KL: {val_kl:.4f} | Recon: {val_recon:.4f} | FID: {fid_score:.4f}")

        test_results["model_name"].append(name_r)
        test_results["val_loss"].append(val_loss)
        test_results["val_recon"].append(val_recon)
        test_results["val_kl"].append(val_kl)
        test_results["fid"].append(fid_score)

        utils.save_model(synthetic_model, name_r, model_saved_path)
        append_result(csv_path, name_r, val_loss, val_recon, val_kl, fid_score)

        # (D) advance chain
        curr_model = synthetic_model

        # (E) cleanup temp dir
        del synthetic_loader
        try:
            if os.path.exists(synthetic_data_dir):
                shutil.rmtree(synthetic_data_dir)
                print(f"[CLEAN] Removed temp dir: {synthetic_data_dir}")
        except Exception as e:
            print(f"[WARN] Failed to remove {synthetic_data_dir}: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------- final table ---------------------
    res_table = pd.DataFrame.from_dict(test_results, orient="columns")
    res_table.to_csv(csv_path, index=False)
    print(f"\n[DONE] Saved results table to {csv_path}")


if __name__ == "__main__":
    main()
