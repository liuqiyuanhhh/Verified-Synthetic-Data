import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from typing import List, Tuple, Optional
import logging
import torch.nn.functional as F
import numpy as np
import gc


class DirectoryBasedSyntheticDataset(Dataset):
    """
    Dataset that reads synthetic data from multiple .pt files in a directory.
    Each .pt file contains one batch of data with potentially different batch sizes.
    """

    def __init__(self, directory_path: str, file_pattern: str = "*.pt", keep_data: bool = False):
        """
        Initialize dataset from directory containing .pt files.

        Args:
            directory_path: Path to directory containing .pt files
            file_pattern: Pattern to match .pt files (default: "*.pt")
            keep_data: If True, directory will not be deleted when object goes out of scope (default: False)
        """
        self.directory_path = directory_path
        self.file_pattern = file_pattern
        self.keep_data = keep_data

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
        if keep_data:
            logging.info(f"Data will be preserved (keep_data=True)")
        else:
            logging.info(
                f"Data will be automatically deleted when dataset goes out of scope (keep_data=False)")

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
                # Explicitly free previous file data to save memory
                if hasattr(self, 'current_file_data') and self.current_file_data is not None:
                    del self.current_file_data
                    gc.collect()  # Force garbage collection

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

        image = image.view(1, 28, 28)
        label = int(label.item()) if hasattr(label, 'item') else int(label)
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

    def debug_dataset(self):
        """
        Debug method to check dataset integrity and print useful information.
        """
        print(f"=== Dataset Debug Info ===")
        print(f"Total files: {len(self.pt_files)}")
        print(f"Total samples: {len(self.sample_to_file_index)}")
        print(f"Files found:")

        for i, file_path in enumerate(self.pt_files):
            try:
                file_data = torch.load(file_path, map_location='cpu')
                if 'images' in file_data and 'labels' in file_data:
                    img_shape = file_data['images'].shape
                    label_shape = file_data['labels'].shape
                    print(f"  File {i}: {os.path.basename(file_path)}")
                    print(f"    Images: {img_shape}, Labels: {label_shape}")
                    print(
                        f"    Sample range: {self._get_file_sample_range(file_path)}")
                else:
                    print(
                        f"  File {i}: {os.path.basename(file_path)} - INVALID (missing keys)")
            except Exception as e:
                print(
                    f"  File {i}: {os.path.basename(file_path)} - ERROR: {e}")

        # Test a few samples
        print(f"\nTesting sample access:")
        for i in [0, len(self.sample_to_file_index)//2, len(self.sample_to_file_index)-1]:
            try:
                sample, label = self[i]
                print(f"  Sample {i}: shape={sample.shape}, label={label}")
            except Exception as e:
                print(f"  Sample {i}: ERROR - {e}")

    def _get_file_sample_range(self, file_path: str) -> Tuple[int, int]:
        """Get the global sample index range for a specific file."""
        start_idx = None
        end_idx = None

        for i, (fpath, _) in enumerate(self.sample_to_file_index):
            if fpath == file_path:
                if start_idx is None:
                    start_idx = i
                end_idx = i

        return (start_idx, end_idx) if start_idx is not None else (None, None)

    def __del__(self):
        """
        Destructor that deletes the directory containing the synthetic data.
        This is called when the dataset object goes out of scope.

        Warning: This will permanently delete the directory and all its contents!
        """
        if not self.keep_data:
            try:
                # Check if directory exists and is not empty
                if os.path.exists(self.directory_path):
                    # Remove all files in the directory first
                    for file_path in self.pt_files:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logging.info(f"Deleted file: {file_path}")

                    # Remove the directory itself
                    os.rmdir(self.directory_path)
                    logging.info(f"Deleted directory: {self.directory_path}")

            except Exception as e:
                # Log error but don't raise exception in destructor
                logging.warning(
                    f"Error during cleanup of {self.directory_path}: {e}")

        # Clean up any loaded data
        if hasattr(self, 'current_file_data') and self.current_file_data is not None:
            del self.current_file_data
            self.current_file_data = None

    def cleanup_directory(self):
        """
        Manually trigger cleanup of the directory and its contents.
        This method can be called explicitly to clean up the data.

        Note: This method respects the keep_data flag. If keep_data=True, 
        this method will not delete the directory.

        Returns:
            bool: True if cleanup was successful or skipped, False if cleanup failed
        """
        # Early return if keep_data is True
        if self.keep_data:
            logging.info(
                f"Skipping cleanup for {self.directory_path} (keep_data=True)")
            return True

        try:
            if os.path.exists(self.directory_path):
                # Remove all files in the directory first
                for file_path in self.pt_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logging.info(f"Deleted file: {file_path}")

                # Remove the directory itself
                os.rmdir(self.directory_path)
                logging.info(
                    f"Successfully cleaned up directory: {self.directory_path}")
                return True
            else:
                logging.info(f"Directory {self.directory_path} does not exist")
                return True

        except Exception as e:
            logging.error(
                f"Error during manual cleanup of {self.directory_path}: {e}")
            return False

        finally:
            # Clean up any loaded data
            if hasattr(self, 'current_file_data') and self.current_file_data is not None:
                del self.current_file_data
                self.current_file_data = None

    def __enter__(self):
        """
        Context manager entry point.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point. Automatically cleans up the directory.
        """
        self.cleanup_directory()


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


def generate_balanced_images_with_filtering(
    model: torch.nn.Module,
    save_directory: str,
    total_samples: int,
    batch_size: int = 60000,
    discriminator: Optional[torch.nn.Module] = None,
    selection_threshold: Optional[float] = None,
    use_quantile_filtering: bool = False,
    max_iterations: int = 20000,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Generate balanced synthetic images with equal samples per digit after filtering.

    This function ensures each digit has exactly the same number of samples after filtering,
    regardless of how difficult it is to generate high-quality samples for certain digits.

    Args:
        model: Trained CVAE model for generating images
        save_directory: Directory to save .pt files        
        total_samples: Total number of samples to generate (must be divisible by 10)
        batch_size: Batch size for saving files (must be divisible by 10)
        discriminator: Optional discriminator for filtering (if None, no filtering)
        selection_percentile: Percentile for filtering when use_quantile_filtering=True (e.g., 80 means keep top 80%)
        selection_threshold: Score threshold for filtering when use_quantile_filtering=False
        use_quantile_filtering: If True, use percentile-based filtering; if False, use threshold-based filtering
        verbose: Whether to print progress information

    Returns:
        Tuple of (total_samples_generated, total_samples_saved)
    """
    model.eval()
    if discriminator:
        discriminator.eval()

    # Validate inputs
    if total_samples % 10 != 0:
        raise ValueError(
            "total_samples must be divisible by 10 for balanced generation")
    if batch_size % 10 != 0:
        raise ValueError(
            "batch_size must be divisible by 10 for balanced generation")

    samples_per_digit = total_samples // 10
    batch_samples_per_digit = batch_size // 10

    if use_quantile_filtering and (selection_threshold < 0 or selection_threshold > 1):
        raise ValueError(
            "selection_threshold must be between 0 and 1 when use_quantile_filtering=True")

    # Create save directory
    os.makedirs(save_directory, exist_ok=True)

    # Remove any existing .pt files in the directory
    existing_pt_files = glob.glob(os.path.join(save_directory, "*.pt"))
    if existing_pt_files:
        if verbose:
            print(
                f"Removing {len(existing_pt_files)} existing .pt files from {save_directory}")
        for pt_file in existing_pt_files:
            os.remove(pt_file)

    if verbose:
        print(
            f"Generating {total_samples} samples ({samples_per_digit} per digit)")
        if discriminator:
            if use_quantile_filtering:
                print(
                    f"Using quantile-based filtering with keeping {selection_threshold} percentile of samples")
            else:
                print(
                    f"Using threshold-based filtering with threshold {selection_threshold}")
        print(
            f"Batch size: {batch_size} ({batch_samples_per_digit} per digit)")
        print(f"Saving to: {save_directory}")

    total_generated = 0
    total_saved = 0
    batch_idx = 0

    # Process in batches
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        current_batch_size = batch_end - batch_start
        current_samples_per_digit = current_batch_size // 10

        if verbose:
            print(
                f"\n--- Processing batch {batch_idx + 1}: {current_batch_size} samples ({current_samples_per_digit} per digit) ---")

        # Generate balanced samples for this batch
        batch_images = []
        batch_labels = []

        for digit in range(10):
            if verbose:
                print(f"Generating digit {digit}...")

            digit_samples = []
            digit_generated = 0
            digits_count = 0
            safe_count = 0

            # Keep generating until we have enough samples for this digit
            while digits_count < current_samples_per_digit and safe_count < max_iterations:
                safe_count += 1
                # Generate a batch of samples for this digit
                # Generate extra to account for filtering
                samples = model.sample_x_given_y(
                    digit, batch_samples_per_digit)
                digit_generated += len(samples)

                if discriminator is not None:
                    # Apply filtering
                    with torch.no_grad():
                        scores = discriminator.score(samples).squeeze(1)

                        if use_quantile_filtering:
                            # Use percentile-based filtering (selection_threshold is the percentile)
                            threshold = torch.quantile(
                                scores, 1.0 - selection_threshold)
                        else:
                            # Use threshold-based filtering
                            threshold = selection_threshold

                        mask = scores >= threshold
                        filtered_samples = samples[mask]

                        if verbose and len(samples) > 0:
                            filter_rate = len(
                                filtered_samples) / len(samples) * 100
                            print(
                                f"  Digit {digit}: {len(samples)} -> {len(filtered_samples)} samples (filter rate: {filter_rate:.1f}%)")

                        samples = filtered_samples

                # Add samples to digit collection
                if len(samples) > current_samples_per_digit - digits_count:
                    samples = samples[:(
                        current_samples_per_digit - digits_count)]

                digit_samples.append(samples)
                digits_count += len(samples)

            # Concatenate all samples for this digit and take exactly what we need
            digit_samples = torch.cat(digit_samples, dim=0)

            # Create labels for this digit
            digit_labels = torch.full(
                (len(digit_samples),), digit, dtype=torch.long, device=digit_samples.device)

            batch_images.append(digit_samples)
            batch_labels.append(digit_labels)
            total_generated += digit_generated

            if verbose:
                print(
                    f"  Digit {digit}: Generated {digit_generated} samples, kept {len(digit_samples)}")

        # Concatenate all digits for this batch
        batch_images = torch.cat(batch_images, dim=0)
        batch_labels = torch.cat(batch_labels, dim=0)

        # Create shuffled indices
        indices = torch.randperm(len(batch_images))
        batch_images = batch_images[indices]
        batch_labels = batch_labels[indices]

        # Save batch to disk
        if discriminator:
            if use_quantile_filtering:
                batch_filename = f"{len(batch_images)}_q{selection_threshold}_b{batch_idx}.pt"
            else:
                batch_filename = f"{len(batch_images)}_t{selection_threshold}_b{batch_idx}.pt"
        else:
            batch_filename = f"{len(batch_images)}_b{batch_idx}.pt"

        batch_path = os.path.join(save_directory, batch_filename)

        torch.save({
            'images': batch_images.cpu(),
            'labels': batch_labels.cpu()
        }, batch_path)

        total_saved += len(batch_images)

        if verbose:
            print(
                f"Saved batch {batch_idx}: {len(batch_images)} samples to {batch_filename}")

        # Clean up memory
        del batch_images, batch_labels
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        batch_idx += 1

    if verbose:
        print(f"\nGeneration completed.")
        print(f"Total generated: {total_generated}")
        print(f"Total saved: {total_saved}")
        print(f"Filter efficiency: {total_saved/total_generated*100:.1f}%")

    return total_generated, total_saved


def create_directory_based_dataloader(
    directory_path: str,
    batch_size: int = 128,
    file_pattern: str = "*.pt",
    keep_data: bool = False
) -> DataLoader:
    """
    Create a DataLoader for the DirectoryBasedSyntheticDataset.

    Note: num_workers is hardcoded to 0 because multiple workers cause memory issues
    with disk-based datasets that switch between files.

    Args:
        directory_path: Path to directory containing .pt files
        batch_size: Batch size for the DataLoader
        file_pattern: Pattern to match .pt files (default: "*.pt")
        keep_data: If True, directory will not be deleted when dataset goes out of scope (default: False)

    Returns:
        DataLoader configured for disk-based synthetic data
    """
    dataset = DirectoryBasedSyntheticDataset(
        directory_path, file_pattern, keep_data)

    # Do not shuffle the data for DirectoryBasedSyntheticDataset, it will cause memory issue
    # num_workers hardcoded to 0 to prevent memory issues with file switching
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Hardcoded to prevent memory issues
        pin_memory=False  # No need for pin_memory with single worker
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

    # print(
    #     f"Generating {samples_per_digit} samples per digit for synthetic data")

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

    # print(f"Generated synthetic dataset size: {len(synthetic_images)}")

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
    # print(f"Real dataset size: {real_size}")

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

    # print(f"Final combined dataset size: {len(X_all)}")
    # print(f"Real samples: {torch.sum(y_all).item()}")
    # print(f"Synthetic samples: {len(y_all) - torch.sum(y_all).item()}")

    # Return simple TensorDataset - DataLoader will handle shuffling
    return torch.utils.data.TensorDataset(X_all, y_all)


# ---------- small in-RAM k-center for pruning/finalization ----------
def kcenter_in_memory(embeddings: torch.Tensor, k: int) -> torch.Tensor:
    """
    embeddings: (N, d) CPU or GPU tensor of embeddings
    k: number to select
    returns: (k,) LongTensor of selected indices (w.r.t. embeddings)
    """
    input_device = embeddings.device
    N = embeddings.size(0)

    if k >= N:
        return torch.arange(N, device=input_device)

    # Move to GPU for computation if not already there
    if embeddings.device.type == 'cpu' and torch.cuda.is_available():
        embeddings_gpu = embeddings.to('cuda')
        device = 'cuda'
    else:
        embeddings_gpu = embeddings
        device = embeddings.device

    # start with a random point
    idx = [torch.randint(N, (), device=device).item()]
    dist = torch.cdist(embeddings_gpu[idx], embeddings_gpu).squeeze(0)  # (N,)

    for _ in range(1, k):
        far = torch.argmax(dist).item()
        idx.append(far)
        dist = torch.minimum(dist, torch.cdist(
            embeddings_gpu[[far]], embeddings_gpu).squeeze(0))

    # Convert final indices to input device
    return torch.tensor(idx, device=input_device, dtype=torch.long)

# ---------- incremental streaming coreset across shard files ----------


@torch.no_grad()
def incremental_coreset_across_files(
    directory_path: str,
    output_path: str,              # path to save selected images and labels
    embed_fn,                      # callable (x_flat, y_int) -> (B,d)
    K_final: int,                  # final number to select
    pct_per_file: float = 1.0,     # percentage of samples to take per file before pruning
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True            # whether to print progress information
):
    """
    Streaming k-center algorithm that processes files sequentially.

    Args:
        directory_path: Path to directory containing .pt files
        output_path: Path to save the selected images and labels
        embed_fn: Function that takes (x_flat, y_int) and returns embeddings (B,d)
        K_final: Final number of samples to select
        pct_per_file: Percentage of samples to select from each file
        device: Device to use for computations
        verbose: Whether to print progress information

    Returns:
        None (saves results to output_path)
    """

    pattern = os.path.join(directory_path, "*.pt")
    pt_files = glob.glob(pattern)
    pt_files.sort()

    # Global coreset: stores embeddings and metadata per digit
    # Dictionary with digit as key, each value contains:
    # - 'embeddings': list of (d,) CPU tensors
    # - 'images': list of CPU tensors
    # - 'labels': list of CPU tensors
    coreset_by_digit = {digit: {'embeddings': [],
                                'images': [], 'labels': []} for digit in range(10)}

    if verbose:
        print(
            f"Processing {len(pt_files)} files for incremental k-center selection...")

    # Process each file
    for file_idx, file_path in enumerate(pt_files):
        if verbose:
            print(
                f"Processing file {file_idx + 1}/{len(pt_files)}: {os.path.basename(file_path)}")

        # Load file data
        data = torch.load(file_path, map_location="cpu")
        images = data["images"]  # (N, 784) or (N, 1, 28, 28)
        labels = data["labels"]  # (N,)

        # Process each digit separately
        for digit in range(10):
            # Get mask for current digit
            mask = (labels == digit)
            digit_images = images[mask]
            digit_labels = labels[mask]

            if len(digit_images) == 0:
                continue  # Skip if no samples for this digit

            # Calculate how many samples to select for this digit in this file
            n_samples_digit = len(digit_images)
            m = max(1, int(n_samples_digit * pct_per_file))

            # Move to device for embedding computation
            Xd = digit_images.to(device).float()
            Yd = digit_labels.to(device)

            # Get embeddings for current digit batch
            # (M, d) where M = n_samples_digit
            embeddings_this_batch = embed_fn(Xd, Yd).detach()

            # --- initialize per-point min distance to current coreset for this digit ---
            dmin = torch.full((n_samples_digit,), float(
                "inf"), device=device)  # (M,)

            # --- greedy farthest-first within this file (exact k-center logic) ---
            selected_local = []

            if len(coreset_by_digit[digit]['embeddings']) > 0:
                # Coreset exists: compute distances to existing centers
                coreset_dev = torch.stack(coreset_by_digit[digit]['embeddings'], 0).to(
                    device)  # (|coreset|, d)
                # distance to nearest existing center in coreset for this digit
                dmin = torch.cdist(embeddings_this_batch, coreset_dev).min(
                    dim=1).values  # (M,)
            else:
                # Coreset is empty: seed with a random point from this file
                seed = torch.randint(n_samples_digit, (), device=device).item()
                selected_local.append(seed)
                coreset_by_digit[digit]['embeddings'].append(
                    embeddings_this_batch[seed].detach().cpu())
                coreset_by_digit[digit]['images'].append(digit_images[seed])
                coreset_by_digit[digit]['labels'].append(digit_labels[seed])

                d_seed = torch.cdist(
                    # (M,)
                    embeddings_this_batch, embeddings_this_batch[seed:seed+1]).squeeze(1)
                dmin = torch.minimum(dmin, d_seed)
                dmin[seed] = -1.0  # mark selected so it won't be chosen again

            while len(selected_local) < min(m, n_samples_digit):
                far = int(torch.argmax(dmin).item())  # farthest point
                selected_local.append(far)

                # --- append chosen embedding and data to global coreset for this digit ---
                coreset_by_digit[digit]['embeddings'].append(
                    embeddings_this_batch[far].detach().cpu())
                coreset_by_digit[digit]['images'].append(digit_images[far])
                coreset_by_digit[digit]['labels'].append(digit_labels[far])

                # update dmin using ONLY the new center (just like the canonical code)
                d_new = torch.cdist(
                    # (M,)
                    embeddings_this_batch, embeddings_this_batch[far:far+1]).squeeze(1)
                dmin = torch.minimum(dmin, d_new)
                dmin[far] = -1.0  # prevent re-selection

            if verbose:
                print(
                    f"  Digit {digit}: selected {len(selected_local)}/{n_samples_digit} samples, coreset size: {len(coreset_by_digit[digit]['embeddings'])}")

    # Final balanced selection: select K_final/10 samples from each digit
    K_final_per_digit = K_final // 10
    if verbose:
        print(
            f"\nRunning final balanced k-center selection: {K_final_per_digit} samples per digit...")

    final_images = []
    final_labels = []

    for digit in range(10):
        if len(coreset_by_digit[digit]['embeddings']) == 0:
            if verbose:
                print(f"  Digit {digit}: No samples available, skipping")
            continue

        digit_coreset_size = len(coreset_by_digit[digit]['embeddings'])

        if K_final_per_digit >= digit_coreset_size:
            # Take all samples for this digit
            if verbose:
                print(
                    f"  Digit {digit}: Taking all {digit_coreset_size} samples")
            digit_final_indices = torch.arange(digit_coreset_size)
        else:
            # Run k-center selection for this digit
            if verbose:
                print(
                    f"  Digit {digit}: Selecting {K_final_per_digit}/{digit_coreset_size} samples")
            # Stack embeddings list into a tensor
            digit_embeddings = torch.stack(
                coreset_by_digit[digit]['embeddings'], 0)
            digit_final_indices = kcenter_in_memory(
                digit_embeddings, K_final_per_digit)

        # Get selected samples for this digit
        digit_final_images = [coreset_by_digit[digit]
                              ['images'][i] for i in digit_final_indices]
        digit_final_labels = [coreset_by_digit[digit]
                              ['labels'][i] for i in digit_final_indices]

        final_images.extend(digit_final_images)
        final_labels.extend(digit_final_labels)

    # Convert to tensors
    final_images = torch.stack(final_images)
    final_labels = torch.stack(final_labels)

    # Shuffle the final selection
    shuffle_indices = torch.randperm(len(final_images))
    final_images = final_images[shuffle_indices]
    final_labels = final_labels[shuffle_indices]

    # Save to output path
    output_path = os.path.join(output_path, f"coreset_{K_final}.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "images": final_images,
        "labels": final_labels
    }, output_path)

    print(f"Saved selected samples after k-center to: {output_path}")

# ---------- helper: embed with your CVAE μ(x|y) ----------


def make_embed_mu_from_cvae(cvae):
    def _embed(x_flat, y_int):
        y_oh = F.one_hot(y_int.long(), num_classes=cvae.label_dim).float().to(
            x_flat.device)
        mu, _ = cvae.encoder.encode(x_flat, y_oh)
        return mu
    return _embed
