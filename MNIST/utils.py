from re import sub
import torch
import matplotlib.pyplot as plt
import os
import models as models
import numpy as np
import pandas as pd
import data_helper as data_helper
import random


def save_model(model, model_name, path):
    model_path = os.path.join(path, model_name + ".pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def load_model(model_name, path, input_device=None, model_args=None):
    name_tokens = model_name.split("_")
    if name_tokens[0] == "cvae":
        if model_args is None:
            model = models.CVAE()
        else:
            model = models.CVAE(*model_args)
    elif name_tokens[0] == "disc":
        if model_args is None:
            model = models.SyntheticDiscriminator()
        else:
            model = models.SyntheticDiscriminator(*model_args)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model.load_state_dict(torch.load(os.path.join(path, model_name + ".pth")))
    model.eval()
    if input_device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = input_device
    model = model.to(device)
    return model


def compute_discriminator_score_distribution(model, disc_model, num_samples=50000, device=None):
    synthetic_data, synthetic_labels = data_helper.generate_balanced_synthetic_data(
        model, num_samples, device=device)

    # Dictionary to store all scores and statistics
    digit_stats = {}

    for digit in range(10):
        this_digit_data = synthetic_data[synthetic_labels == digit]
        this_digit_scores = disc_model.score(
            this_digit_data).cpu().detach().numpy().flatten()

    # Calculate comprehensive statistics
    digit_stats[digit] = {
        'count': len(this_digit_scores),
        'mean': this_digit_scores.mean(),
        'std': this_digit_scores.std(),
        'min': this_digit_scores.min(),

        'q50': np.percentile(this_digit_scores, 50),  # median
        'q90': np.percentile(this_digit_scores, 90),
        'q95': np.percentile(this_digit_scores, 95),
        'q99': np.percentile(this_digit_scores, 99),
        'q99.5': np.percentile(this_digit_scores, 99.5),
        'max': this_digit_scores.max()
    }

    # Create DataFrame
    df_detailed = pd.DataFrame.from_dict(digit_stats, orient='index')

    # Rename columns for clarity
    df_detailed.columns = ['Count', 'Mean', 'Std', 'Min',
                           'Q50 (Median)', 'Q90', 'Q95', 'Q99', 'Q99.5', 'Max']

    # Round numeric columns to 4 decimal places
    numeric_columns = ['Mean', 'Std', 'Min',
                       'Q50 (Median)', 'Q90', 'Q95', 'Q99', 'Q99.5', 'Max']
    df_detailed[numeric_columns] = df_detailed[numeric_columns].round(4)

    return df_detailed


# Plot n random samples for each digit
def plot_samples_per_digit(num_samples, model, binary_format=False):
    fig, axes = plt.subplots(10, num_samples, figsize=(num_samples, 12))
    fig.suptitle("10 Random Samples for Each Digit (0-9)", fontsize=16)

    for digit in range(10):
        # Find indices for this digit
        samples = model.sample_x_given_y(
            digit, num_samples, binary_format=binary_format)

        for i in range(num_samples):
            img = samples[i].view(28, 28).cpu().detach().numpy()
            axes[digit, i].imshow(img, cmap='gray')
            axes[digit, i].set_title(f'Digit {digit}')
            axes[digit, i].axis('off')

    plt.tight_layout()
    plt.show()


def display_samples_from_pt_file(num_samples_per_digit, pt_file_path):
    """
    Display random samples from a saved .pt data file.

    Args:
        num_samples_per_digit: Number of random samples to show for each digit
        pt_file_path: Path to the .pt file containing images and labels
    """
    # Load the data
    data = torch.load(pt_file_path, map_location='cpu')

    if 'images' not in data or 'labels' not in data:
        raise ValueError("PT file must contain 'images' and 'labels' keys")

    images = data['images']
    labels = data['labels']

    print(f"Loaded {len(images)} samples from {pt_file_path}")
    print(f"Image shape: {images.shape}")
    print(f"Label shape: {labels.shape}")

    # Create figure
    fig, axes = plt.subplots(10, num_samples_per_digit,
                             figsize=(num_samples_per_digit, 12))
    fig.suptitle(
        f"Random Samples from {os.path.basename(pt_file_path)}", fontsize=16)

    # For each digit (0-9)
    for digit in range(10):
        # Find indices for this digit
        digit_indices = (labels == digit).nonzero(as_tuple=True)[0]

        if len(digit_indices) == 0:
            print(f"Warning: No samples found for digit {digit}")
            continue

        # Select random samples for this digit
        if len(digit_indices) >= num_samples_per_digit:
            selected_indices = digit_indices[torch.randperm(
                len(digit_indices))[:num_samples_per_digit]]
        else:
            # If not enough samples, use all available
            selected_indices = digit_indices
            print(
                f"Warning: Only {len(selected_indices)} samples available for digit {digit}")

        # Display samples
        for i, idx in enumerate(selected_indices):
            img = images[idx]

            # Handle different image formats
            if img.dim() == 3 and img.shape[0] == 1:  # [1, 28, 28]
                img = img.squeeze(0)  # [28, 28]
            elif img.dim() == 2:  # [28, 28]
                pass
            else:
                img = img.view(28, 28)  # Flatten and reshape

            img_np = img.cpu().detach().numpy()

            if num_samples_per_digit == 1:
                axes[digit].imshow(img_np, cmap='gray')
                axes[digit].set_title(f'Digit {digit}')
                axes[digit].axis('off')
            else:
                axes[digit, i].imshow(img_np, cmap='gray')
                axes[digit, i].set_title(f'Digit {digit}')
                axes[digit, i].axis('off')

    plt.tight_layout()
    plt.show()


def create_balanced_subset_indices(dataset, seed=0):
    """
    Create balanced subset indices by grouping indices by digit and shuffling each separately.

    Args:
        dataset: Full MNIST dataset
        seed: Random seed for reproducibility

    Returns:
        Dictionary with digit -> shuffled indices mapping
    """
    # Set seed for reproducibility
    random.seed(seed)

    # Group indices by digit
    digit_indices = {i: [] for i in range(10)}

    for i in range(len(dataset)):
        _, label = dataset[i]
        digit = label
        digit_indices[digit].append(i)

    # Shuffle each digit's indices separately
    for digit in range(10):
        random.shuffle(digit_indices[digit])

    return digit_indices


def get_balanced_subset(digit_indices, subset_size):
    """
    Get a balanced subset by taking equal samples from each digit's shuffled indices.

    Args:        
        digit_indices: Dictionary with digit -> shuffled indices mapping
        subset_size: Size of subset to return

    Returns:
        Subset with balanced digit distribution
    """
    # Calculate samples per digit
    if not isinstance(subset_size, list) and not isinstance(subset_size, int):
        raise ValueError(
            f"Invalid subset size: {subset_size}, must be a list of two integers or an integer")

    if isinstance(subset_size, list) and len(subset_size) != 2:
        raise ValueError(
            f"Invalid subset size: {subset_size}, must be a list of two integers")

    # Take samples from each digit
    subset_indices = []
    for digit in range(10):
        # Take samples_per_digit samples, plus one extra for first 'remainder' digits
        if isinstance(subset_size, list):
            if len(digit_indices[digit]) < subset_size[0]:
                raise ValueError(
                    f"Not enough samples for digit {digit}, required at least {subset_size[0]} samples, but only {len(digit_indices[digit])} samples available")

            end_index = subset_size[1]
            if len(digit_indices[digit]) < end_index:
                print(
                    f"Warning: Only {len(digit_indices[digit])} samples for digit {digit}, taking all available")
                end_index = len(digit_indices[digit])

            subset_indices.extend(
                digit_indices[digit][subset_size[0]:end_index])

        elif isinstance(subset_size, int):
            num_samples = subset_size // 10

            if num_samples > len(digit_indices[digit]):
                print(
                    f"Warning: Only {len(digit_indices[digit])} samples for digit {digit}, taking all available")
                num_samples = len(digit_indices[digit])

            # Take first num_samples from this digit's shuffled indices
            subset_indices.extend(digit_indices[digit][:num_samples])

    return subset_indices


def verify_balance(dataset):
    """
    Verify the balance of a dataset by counting samples per digit.

    Args:
        dataset: Dataset to verify        
    """
    digit_counts = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        digit = label
        digit_counts[digit] = digit_counts.get(digit, 0) + 1

    print(f"\n digit distribution:")
    for digit in sorted(digit_counts.keys()):
        print(f"Digit {digit}: {digit_counts[digit]} samples")

    return digit_counts
