"""
Iterative retraining with label-smoothed conditional discriminator.
"""
import sys
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
import torch.nn.functional as F

import pandas as pd
import numpy as np
import os
import random
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.append(str(SRC_DIR))

import models as models
import train_helper as train_helper
import utils as utils
import data_helper as data_helper
import fid as fid_helper

# ---------------------------------------------------------------------------
# Device, seed, paths
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_seed = 0
torch.manual_seed(base_seed)
torch.cuda.manual_seed_all(base_seed)
np.random.seed(base_seed)
random.seed(base_seed)

ROOT = THIS_DIR.parent
model_saved_path = os.path.join(ROOT, "model_saved")
data_saved_path = os.path.join(ROOT, "data_saved")
results_saved_path = os.path.join(ROOT, "results_saved")
os.makedirs(results_saved_path, exist_ok=True)
os.makedirs(model_saved_path, exist_ok=True)
os.makedirs(data_saved_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
full_digit_indices = utils.create_balanced_subset_indices(full_dataset, seed=base_seed)

# ---------------------------------------------------------------------------
# Train init model on 500 real samples
# ---------------------------------------------------------------------------
init_size = 500
init_subset = utils.get_balanced_subset(full_digit_indices, init_size)
init_dataset = Subset(full_dataset, init_subset)
init_train_loader = DataLoader(init_dataset, batch_size=128, shuffle=True)

init_model = models.CVAE(input_dim=784, label_dim=10, latent_dim=20, name="cvae_real_500", arch="conv").to(device)
train_helper.train_model(model=init_model, train_loader=init_train_loader, device=device, epochs=200, lr=1e-3, patience=5, verbose=False)
val_loss, val_recon, val_kl = train_helper.calculate_validation_loss(init_model, test_loader, device)
fid = fid_helper.calculate_fid_from_model(real_ds=test_dataset, model=init_model, device=device)
print("init model fid", fid, "val_NELBO", val_loss, "val_recon", val_recon, "val_kl", val_kl)

# ---------------------------------------------------------------------------
# Initialize results dict and size schedule
# ---------------------------------------------------------------------------
delta_size = 5_000
total_iterations = 50
test_results = {
    "model_name": [], "fid": [], "val_loss": [], "val_recon": [], "val_kl": [],
    "disc_train_loss": [], "disc_val_loss": [], "disc_test_accuracy": [],
}
size_schedule = [delta_size * i for i in range(1, total_iterations + 1)]
all_models = []

# ---------------------------------------------------------------------------
# Iterative retraining loop
# ---------------------------------------------------------------------------
this_model = init_model

for i, synthetic_size in enumerate(size_schedule):
    filter_thres = 0.1
    i = i + 1
    synthetic_size = int(synthetic_size)

    discriminator_dataset = data_helper.prepare_discriminator_dataset_with_labels(full_dataset, this_model, device)
    disc_loader = DataLoader(discriminator_dataset, batch_size=128, shuffle=True)

    disc_test_dataset = data_helper.prepare_discriminator_dataset_with_labels(test_dataset, this_model, device)
    disc_test_loader = DataLoader(disc_test_dataset, batch_size=128, shuffle=True)

    # Train Discriminator with Label Smoothing and dropout
    disc_model = models.ConditionalDiscriminator(input_dim=784, name="disc_mlp_" + str(synthetic_size), arch="mlp", dropout=0.1, label_smoothing=0.05)
    disc_history = train_helper.train_model_with_validation(
        model=disc_model, train_loader=disc_loader, val_loader=disc_test_loader,
        device=device, epochs=200, lr=1e-3, wd=0, patience=5, verbose=False,
    )

    print(f"Iteration {i}, disc_epochs_trained: {disc_history['epochs_trained']}, disc_best_train_loss: {disc_history['best_train_loss']}, disc_best_val_loss: {disc_history['best_val_loss']}")
    print("disc_train_last_summary:", disc_history['train_last_summary'])
    print("disc_val_last_summary:", disc_history['val_last_summary'])
    print(f"filter_thres: {filter_thres}")

    # Generate Synthetic Data
    synthetic_data_load_path = os.path.join(data_saved_path, this_model.get_name() + f'_q{filter_thres}_gen{synthetic_size}')
    data_helper.generate_balanced_images_with_filtering(
        model=this_model, save_directory=synthetic_data_load_path,
        total_samples=synthetic_size, discriminator=disc_model,
        selection_threshold=filter_thres, verbose=False, use_quantile_filtering=True,
    )

    # Train Synthetic Model
    synthetic_loader = data_helper.create_directory_based_dataloader(synthetic_data_load_path, batch_size=128, keep_data=False)

    synthetic_model = models.CVAE(
        input_dim=784, label_dim=10, latent_dim=20,
        name=f"cvae_q{filter_thres}_iter{i}_{synthetic_size}", arch="conv",
    ).to(device)
    train_helper.train_model(synthetic_model, synthetic_loader, device, epochs=200, lr=1e-3, patience=5, verbose=False)

    this_model = synthetic_model
    all_models.append(this_model)

    fid = fid_helper.calculate_fid_from_model(real_ds=test_dataset, model=this_model, device=device)
    val_loss, val_recon, val_kl = train_helper.calculate_validation_loss(this_model, test_loader, device)

    test_results["model_name"].append(this_model.get_name())
    test_results["fid"].append(fid)
    test_results["val_loss"].append(val_loss)
    test_results["val_recon"].append(val_recon)
    test_results["val_kl"].append(val_kl)
    test_results["disc_train_loss"].append(disc_history['best_train_loss'])
    test_results["disc_val_loss"].append(disc_history['best_val_loss'])
    test_results["disc_test_accuracy"].append(disc_history['val_last_summary']['accuracy'])

    print(f"Iteration {i} - Ending model: {this_model.get_name()}, FID: {test_results['fid'][-1]:.2f},  Test NELBO: {test_results['val_loss'][-1]:.2f}")

    del synthetic_loader
    del disc_model
    del discriminator_dataset
    del disc_loader
    del disc_test_dataset
    del disc_test_loader

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
res_table = pd.DataFrame.from_dict(test_results, orient="columns")
csv_path = os.path.join(results_saved_path, f"label_smoothing_D{delta_size}_results.csv")
res_table.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")
print(res_table)
