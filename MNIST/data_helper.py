import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from typing import List, Tuple, Optional
import logging
import torch.nn.functional as F
import numpy as np


class DirectoryBasedSyntheticDataset(Dataset):
    """
    Dataset that reads synthetic data from multiple .pt files in a directory.
    Each .pt file contains one batch of data with potentially different batch sizes.
    """

    def __init__(self, directory_path: str, file_pattern: str = "*.pt"):
        """
        Initialize dataset from directory containing .pt files.

        Args:
            directory_path: Path to directory containing .pt files
            file_pattern: Pattern to match .pt files (default: "*.pt")
        """
        self.directory_path = directory_path
        self.file_pattern = file_pattern

        # Find all .pt files in directory
        self.pt_files = self._find_pt_files()

        if not self.pt_files:
            raise ValueError(
                f"No .pt files found in {directory_path} matching pattern {file_pattern}")

        # Build index mapping from sample index to file and local index
        self.sample_to_file_index = self._build_sample_index()

        # Cache for loaded files (only one file in memory at a time)
        self.current_file_data = None
        self.current_file_path = None

        logging.info(
            f"Found {len(self.pt_files)} .pt files with {len(self.sample_to_file_index)} total samples")

    def _find_pt_files(self) -> List[str]:
        """Find all .pt files in the directory, sorted by name."""
        pattern = os.path.join(self.directory_path, self.file_pattern)
        pt_files = glob.glob(pattern)

        # Sort files to ensure consistent ordering
        pt_files.sort()

        return pt_files

    def _build_sample_index(self) -> List[Tuple[str, int]]:
        """
        Build mapping from global sample index to (file_path, local_sample_index).
        This allows __getitem__ to quickly find which file contains a given sample.
        """
        sample_to_file_index = []

        for file_path in self.pt_files:
            try:
                # Load file to get its size
                file_data = torch.load(file_path, map_location='cpu')

                if 'images' in file_data and 'labels' in file_data:
                    images = file_data['images']
                    labels = file_data['labels']

                    # Ensure images and labels have same first dimension
                    if images.shape[0] == labels.shape[0]:
                        batch_size = images.shape[0]

                        # Add mapping for each sample in this file
                        for local_idx in range(batch_size):
                            sample_to_file_index.append((file_path, local_idx))

                        logging.info(
                            f"File {os.path.basename(file_path)}: {batch_size} samples")
                    else:
                        logging.warning(
                            f"Skipping {file_path}: images and labels have different sizes")
                        continue
                else:
                    logging.warning(
                        f"Skipping {file_path}: missing 'images' or 'labels' keys")
                    continue

            except Exception as e:
                logging.warning(f"Skipping corrupted file {file_path}: {e}")
                continue

        return sample_to_file_index

    def _load_file_if_needed(self, file_path: str):
        """Load file data into memory if it's not already loaded."""
        if self.current_file_path != file_path:
            try:
                self.current_file_data = torch.load(
                    file_path, map_location='cpu')
                self.current_file_path = file_path
                logging.debug(f"Loaded file: {os.path.basename(file_path)}")
            except Exception as e:
                logging.error(f"Failed to load file {file_path}: {e}")
                raise RuntimeError(f"Cannot load file {file_path}")

    def __len__(self) -> int:
        """Return total number of samples across all valid .pt files."""
        return len(self.sample_to_file_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get sample at given index.

        Args:
            idx: Global sample index

        Returns:
            Tuple of (image, label)
        """
        if idx < 0 or idx >= len(self.sample_to_file_index):
            raise IndexError(
                f"Index {idx} out of range [0, {len(self.sample_to_file_index)})")

        file_path, local_idx = self.sample_to_file_index[idx]

        # Load file if needed
        self._load_file_if_needed(file_path)

        # Extract image and label
        images = self.current_file_data['images']
        labels = self.current_file_data['labels']

        # Get specific sample
        image = images[local_idx]
        label = labels[local_idx]

        return image, label

    def get_batch_info(self) -> List[Tuple[str, int]]:
        """
        Get information about each batch file.

        Returns:
            List of (file_path, batch_size) tuples
        """
        batch_info = []

        for file_path in self.pt_files:
            try:
                file_data = torch.load(file_path, map_location='cpu')
                if 'images' in file_data:
                    batch_size = file_data['images'].shape[0]
                    batch_info.append((file_path, batch_size))
            except Exception as e:
                logging.warning(
                    f"Could not read batch info from {file_path}: {e}")

        return batch_info


def generate_images_with_filtering(
    model: torch.nn.Module,
    save_directory: str,
    model_name: str,
    total_samples: int,
    batch_size: int = 60000,
    discriminator: Optional[torch.nn.Module] = None,
    selection_percentile: float = 80.0,
    per_digit_filtering: bool = True,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Generate synthetic images and save them to disk with optional discriminator filtering.

    Args:
        model: Trained CVAE model for generating images
        save_directory: Directory to save .pt files
        samples_per_digit: Number of samples to generate for each digit (0-9)
        discriminator: Optional discriminator for filtering (if None, no filtering)
        selection_percentile: Percentile for filtering (e.g., 80 means keep top 80%)
        batch_size: Batch size for generation (for memory efficiency)
        file_prefix: Prefix for saved .pt files
        verbose: Whether to print progress information

    Returns:
        Tuple of (total_samples_generated, total_samples_saved)
    """
    model.eval()
    if discriminator:
        discriminator.eval()

    # Create save directory
    save_directory = os.path.join(save_directory, model_name)
    os.makedirs(save_directory, exist_ok=True)

    # Remove any existing .pt files in the directory
    existing_pt_files = glob.glob(os.path.join(save_directory, "*.pt"))
    if existing_pt_files:
        if verbose:
            print(
                f"Removing {len(existing_pt_files)} existing .pt files from {save_directory}")
        for pt_file in existing_pt_files:
            os.remove(pt_file)

    # Calculate total samples
    num_classes = 10

    if verbose:
        print(f"Generating {total_samples} samples")
        if discriminator:
            print(
                f"Using discriminator filtering with {selection_percentile}th percentile")
        print(f"Using batch size: {batch_size}")
        print(f"Saving to: {save_directory}")

    samples_to_generate = total_samples

    batch_idx = 0
    total_generated = 0
    total_saved = 0

    # Initialize containers for accumulating samples
    to_write_images = None
    to_write_labels = None

    while samples_to_generate > 0:
        # Calculate samples per digit for this batch
        samples_per_digit = (batch_size + num_classes - 1) // num_classes
        this_batch_images = []
        this_batch_labels = []

        # Generate samples for each digit
        for digit in range(num_classes):
            if verbose:
                print(
                    f"Generating digit {digit}: {samples_per_digit} samples")

            # Generate samples for this digit
            samples = model.sample_x_given_y(digit, samples_per_digit)
            total_generated += len(samples)

            # Apply discriminator filtering if provided
            if discriminator is not None and per_digit_filtering:
                with torch.no_grad():
                    scores = discriminator.score(samples).squeeze(1)
                    threshold = torch.quantile(
                        scores, 1.0 - selection_percentile / 100.0)
                    mask = scores >= threshold

                    if verbose:
                        print(
                            f"Digit {digit}: {len(samples)} -> {torch.sum(mask).item()} samples after filtering")

                    samples = samples[mask]

                # If we generated more samples than needed for this digit, slice to exact amount
                samples_needed_per_digit = (
                    samples_to_generate + num_classes - 1) // num_classes
                if len(samples) > samples_needed_per_digit:
                    if verbose:
                        print(
                            f"Digit {digit}: {len(samples)} -> {samples_needed_per_digit} samples after slicing")

                    samples = samples[:samples_needed_per_digit]

            # Store samples and labels
            this_batch_images.append(samples)
            this_batch_labels.append(torch.full(
                (len(samples),), digit, dtype=torch.long, device=samples.device))

        # Concatenate all digits for this batch
        if this_batch_images:
            this_batch_images = torch.cat(this_batch_images, dim=0)
            this_batch_labels = torch.cat(this_batch_labels, dim=0)

            if discriminator is not None and not per_digit_filtering:
                with torch.no_grad():
                    scores = discriminator.score(this_batch_images).squeeze(1)
                    threshold = torch.quantile(
                        scores, 1.0 - selection_percentile / 100.0)
                    mask = scores >= threshold

                    if verbose:
                        print(
                            f"All Digits: {len(this_batch_images)} -> {torch.sum(mask).item()} samples after filtering")

                    this_batch_images = this_batch_images[mask]
                    this_batch_labels = this_batch_labels[mask]

                    if len(this_batch_images) > samples_to_generate:
                        # Shuffle first to ensure even distribution of digits when slicing
                        indices = torch.randperm(len(this_batch_images))
                        this_batch_images = this_batch_images[indices]
                        this_batch_labels = this_batch_labels[indices]

                        # Now slice to exact amount needed
                        this_batch_images = this_batch_images[:samples_to_generate]
                        this_batch_labels = this_batch_labels[:samples_to_generate]
                        if verbose:
                            print(
                                f"All Digits: {len(this_batch_images)} -> {samples_to_generate} samples after slicing")

            # Check if we need to save (either batch is full or we're done)
            current_total = len(this_batch_images)
            if to_write_images is not None:
                current_total += len(to_write_images)

            if current_total >= batch_size or samples_to_generate <= len(this_batch_images):
                # Time to save
                if to_write_images is not None:
                    # Concatenate with existing samples
                    all_images = torch.cat(
                        [to_write_images, this_batch_images], dim=0)
                    all_labels = torch.cat(
                        [to_write_labels, this_batch_labels], dim=0)
                else:
                    # First batch, no existing samples
                    all_images = this_batch_images
                    all_labels = this_batch_labels

                # Save to disk
                if discriminator:
                    batch_filename = f"{model_name}_{len(all_images)}_q{selection_percentile}_b{batch_idx}.pt"
                else:
                    batch_filename = f"{model_name}_{len(all_images)}_b{batch_idx}.pt"

                batch_path = os.path.join(save_directory, batch_filename)

                torch.save({
                    'images': all_images.cpu(),
                    'labels': all_labels.cpu()
                }, batch_path)

                total_saved += len(all_images)

                if verbose:
                    print(
                        f"Saved batch {batch_idx}: {len(all_images)} samples to {batch_filename}")

                batch_idx += 1
                # Clear containers and free memory
                to_write_images = None
                to_write_labels = None
                del all_images, all_labels
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            else:
                # Accumulate samples for next batch
                if to_write_images is None:
                    # First time, just assign
                    to_write_images = this_batch_images
                    to_write_labels = this_batch_labels
                else:
                    # Concatenate with existing
                    to_write_images = torch.cat(
                        [to_write_images, this_batch_images], dim=0)
                    to_write_labels = torch.cat(
                        [to_write_labels, this_batch_labels], dim=0)

                if verbose:
                    print(
                        f"Accumulated {len(to_write_images)} samples, waiting for more...")

            # Update counters even when accumulating
            samples_to_generate -= len(this_batch_images)

        # Free memory for this batch
        del this_batch_images, this_batch_labels
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save any remaining samples
    if to_write_images is not None and len(to_write_images) > 0:
        if discriminator:
            batch_filename = f"{model_name}_{len(to_write_images)}_q{selection_percentile}_b{batch_idx}.pt"
        else:
            batch_filename = f"{model_name}_{len(to_write_images)}_b{batch_idx}.pt"

        batch_path = os.path.join(save_directory, batch_filename)

        torch.save({
            'images': to_write_images.cpu(),
            'labels': to_write_labels.cpu()
        }, batch_path)

        total_saved += len(to_write_images)

        if verbose:
            print(
                f"Saved final batch {batch_idx}: {len(to_write_images)} samples to {batch_filename}")

    if verbose:
        print(
            f"Generation completed. Total generated: {total_generated}, Total saved: {total_saved}")

    return total_generated, total_saved


def create_directory_based_dataloader(
    directory_path: str,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    file_pattern: str = "*.pt"
) -> DataLoader:
    """
    Create a DataLoader that reads from multiple .pt files in a directory.

    Args:
        directory_path: Path to directory containing .pt files
        batch_size: Batch size for training
        shuffle: Whether to shuffle samples within batches
        num_workers: Number of worker processes (0 for single process)
        file_pattern: Pattern to match .pt files

    Returns:
        DataLoader instance
    """
    dataset = DirectoryBasedSyntheticDataset(directory_path, file_pattern)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )


def generate_balanced_synthetic_data(synthetic_model, target_size, device=None):
    """
    Generate balanced synthetic data with equal samples per digit.

    Args:
        synthetic_model: Trained CVAE model for generating synthetic data
        target_size: Target total size for synthetic dataset
        device: Device to use for model (default: auto-detect)

    Returns:
        synthetic_images: Tensor of synthetic images [target_size, 1, 28, 28]
        synthetic_labels: Tensor of synthetic digit labels [target_size] (0-9)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Calculate samples per digit for balanced generation
    num_classes = 10
    samples_per_digit = (target_size + num_classes -
                         1) // num_classes  # Ceiling division

    print(
        f"Generating {samples_per_digit} samples per digit for synthetic data")

    # Generate synthetic data
    synthetic_model.eval()
    synthetic_images = []
    synthetic_labels = []

    with torch.no_grad():
        for digit in range(num_classes):
            # Generate samples for this digit
            samples = synthetic_model.sample_x_given_y(
                digit, samples_per_digit)

            # Convert from [batch_size, 784] to [batch_size, 1, 28, 28]
            if samples.dim() == 2 and samples.shape[1] == 784:
                samples = samples.view(-1, 1, 28, 28)

            synthetic_images.append(samples)
            synthetic_labels.append(torch.full(
                (len(samples),), digit, dtype=torch.long))

    # Concatenate all synthetic images and labels
    synthetic_images = torch.cat(synthetic_images, dim=0)
    synthetic_labels = torch.cat(synthetic_labels, dim=0)

    # Trim to exact target size if needed
    if len(synthetic_images) > target_size:
        synthetic_images = synthetic_images[:target_size]
        synthetic_labels = synthetic_labels[:target_size]

    print(f"Generated synthetic dataset size: {len(synthetic_images)}")

    return synthetic_images, synthetic_labels


def prepare_discriminator_dataset(real_dataset, synthetic_model, device=None):
    """
    Prepare a balanced dataset of real and synthetic data for discriminator training.

    Args:
        real_dataset: PyTorch dataset (e.g., MNIST) or subset
        synthetic_model: Trained CVAE model for generating synthetic data
        device: Device to use for model and tensors (default: auto-detect)

    Returns:
        TensorDataset: Dataset object ready for DataLoader (DataLoader will handle shuffling)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get real dataset size
    real_size = len(real_dataset)
    print(f"Real dataset size: {real_size}")

    # Generate synthetic data to match real dataset size
    synthetic_images, synthetic_labels = generate_balanced_synthetic_data(
        synthetic_model, real_size, device
    )

    # Prepare real data
    real_images = []
    for i in range(real_size):
        img, _ = real_dataset[i]  # Ignore original label
        real_images.append(img)

    real_images = torch.stack(real_images)

    # Ensure all tensors are on the same device
    real_images = real_images.to(device)
    synthetic_images = synthetic_images.to(device)

    # Create discriminator labels: 1 for real, 0 for synthetic
    real_disc_labels = torch.ones(
        real_size, dtype=torch.long, device=device)
    synthetic_disc_labels = torch.zeros(
        len(synthetic_images), dtype=torch.long, device=device)

    # Concatenate real and synthetic data (now all on same device)
    X_all = torch.cat([real_images, synthetic_images], dim=0)
    y_all = torch.cat([real_disc_labels, synthetic_disc_labels], dim=0)

    print(f"Final combined dataset size: {len(X_all)}")
    print(f"Real samples: {torch.sum(y_all).item()}")
    print(f"Synthetic samples: {len(y_all) - torch.sum(y_all).item()}")

    # Return simple TensorDataset - DataLoader will handle shuffling
    return torch.utils.data.TensorDataset(X_all, y_all)
