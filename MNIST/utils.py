import torch
import matplotlib.pyplot as plt
import os
import models as models
# Plot n random samples for each digit


def plot_samples_per_digit(num_samples, model):
    fig, axes = plt.subplots(10, num_samples, figsize=(2*num_samples, 20))
    fig.suptitle("10 Random Samples for Each Digit (0-9)", fontsize=16)

    for digit in range(10):
        # Find indices for this digit
        samples = model.sample_x_given_y(digit, num_samples)

        for i in range(num_samples):
            img = samples[i].view(28, 28).cpu().detach().numpy()
            axes[digit, i].imshow(img, cmap='gray')
            axes[digit, i].set_title(f'Digit {digit}')
            axes[digit, i].axis('off')

    plt.tight_layout()
    plt.show()


def save_model(model, model_name, path):
    model_path = os.path.join(path, model_name + ".pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


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
                             figsize=(2*num_samples_per_digit, 20))
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


def load_model(model_name, path, input_device=None, model_dimension=None):
    name_tokens = model_name.split("_")
    if name_tokens[0] == "cvae":
        if model_dimension is None:
            model = models.CVAE()
        else:
            model = models.CVAE(*model_dimension)
    elif name_tokens[0] == "disc":
        if model_dimension is None:
            model = models.SyntheticDiscriminator()
        else:
            model = models.SyntheticDiscriminator(*model_dimension)
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
