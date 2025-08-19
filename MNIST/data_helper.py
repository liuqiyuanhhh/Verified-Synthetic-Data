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
            raise ValueError(f"No .pt files found in {directory_path} matching pattern {file_pattern}")
        
        # Build index mapping from sample index to file and local index
        self.sample_to_file_index = self._build_sample_index()
        
        # Cache for loaded files (only one file in memory at a time)
        self.current_file_data = None
        self.current_file_path = None
        
        logging.info(f"Found {len(self.pt_files)} .pt files with {len(self.sample_to_file_index)} total samples")
    
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
                        
                        logging.info(f"File {os.path.basename(file_path)}: {batch_size} samples")
                    else:
                        logging.warning(f"Skipping {file_path}: images and labels have different sizes")
                        continue
                else:
                    logging.warning(f"Skipping {file_path}: missing 'images' or 'labels' keys")
                    continue
                    
            except Exception as e:
                logging.warning(f"Skipping corrupted file {file_path}: {e}")
                continue
        
        return sample_to_file_index
    
    def _load_file_if_needed(self, file_path: str):
        """Load file data into memory if it's not already loaded."""
        if self.current_file_path != file_path:
            try:
                self.current_file_data = torch.load(file_path, map_location='cpu')
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
            raise IndexError(f"Index {idx} out of range [0, {len(self.sample_to_file_index)})")
        
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
                logging.warning(f"Could not read batch info from {file_path}: {e}")
        
        return batch_info

def generate_images_with_filtering(
    model: torch.nn.Module,
    save_directory: str,
    model_name: str,
    samples_per_digit: int,
    discriminator: Optional[torch.nn.Module] = None,
    selection_percentile: float = 20.0,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Generate synthetic images and save them to disk with optional discriminator filtering.
    
    Args:
        model: Trained CVAE model for generating images
        save_directory: Directory to save .pt files
        samples_per_digit: Number of samples to generate for each digit (0-9)
        discriminator: Optional discriminator for filtering (if None, no filtering)
        selection_percentile: Percentile for filtering (e.g., 80 means keep top 20%)
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
    
    # Calculate total samples
    num_classes = 10
    total_samples = samples_per_digit * num_classes
    
    if verbose:
        logging.info(f"Generating {total_samples} samples ({samples_per_digit} per digit)")
        if discriminator:
            logging.info(f"Using discriminator filtering with {selection_percentile}th percentile")
        logging.info(f"Saving to: {save_directory}")
    
    generated_images = []
    all_labels = []    
    
    # Generate samples for each digit
    for digit in range(num_classes):
        if verbose:
            logging.info(f"Generating digit {digit}: {samples_per_digit} samples")
        
        # Generate samples for this digit
        samples = model.sample_x_given_y(digit, samples_per_digit)
        
        # Apply discriminator filtering if provided
        if discriminator is not None:
            with torch.no_grad():
                # Reshape samples to image format for discriminator
                samples_reshaped = samples.view(-1, 1, 28, 28)
                # Get discriminator scores
                scores = discriminator(samples_reshaped).squeeze(1)
                
                # Calculate threshold based on percentile
                threshold = torch.quantile(scores, selection_percentile / 100.0)
                
                # Filter samples above threshold
                mask = scores >= threshold
                filtered_samples = samples[mask]
                filtered_scores = scores[mask]
                
                if verbose:
                    logging.info(f"Digit {digit}: {len(samples)} -> {len(filtered_samples)} samples after filtering at threshold {threshold}")
                
                # Use filtered samples
                samples = filtered_samples
                scores = filtered_scores
        
        # Store samples and labels
        generated_images.append(samples)
        all_labels.append(torch.full((len(samples),), digit, dtype=torch.long))
    
    # Concatenate all digits
    if generated_images:
        all_images = torch.cat(generated_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        total_generated = samples_per_digit * num_classes
        total_saved = len(all_images)
        
        if verbose:
            logging.info(f"Total generated: {total_generated}, Total saved: {total_saved}")
        
        # Save all images at once
        if discriminator:
            batch_filename = f"{model_name}_{total_saved}_q{selection_percentile}.pt"
        else:
            batch_filename = f"{model_name}_{total_saved}.pt"
        batch_path = os.path.join(save_directory, batch_filename)
        
        torch.save({
            'images': all_images.cpu(),
            'labels': all_labels.cpu()
        }, batch_path)
        
        if verbose:
            logging.info(f"Saved all {total_saved} samples to {batch_filename}")
        
        return total_generated, total_saved
    else:
        logging.warning("No samples were generated!")
        return 0, 0

def generate_images_sequential(
    model: torch.nn.Module,
    save_directory: str,
    total_samples: int,
    discriminator: Optional[torch.nn.Module] = None,
    selection_percentile: float = 80.0,
    batch_size: int = 1000,
    file_prefix: str = "synthetic_sequential",
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Generate synthetic images sequentially (matching original generate_images_in_batches logic).
    
    Args:
        model: Trained CVAE model for generating images
        save_directory: Directory to save .pt files
        total_samples: Total number of samples to generate
        discriminator: Optional discriminator for filtering
        selection_percentile: Percentile for filtering
        batch_size: Batch size for generation
        file_prefix: Prefix for saved .pt files
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (total_samples_generated, total_samples_saved)
    """
    model.eval()
    if discriminator:
        discriminator.eval()
    
    # Create save directory
    os.makedirs(save_directory, exist_ok=True)
    
    device = next(model.parameters()).device
    num_classes = 10
    
    if verbose:
        logging.info(f"Generating {total_samples} samples sequentially")
        if discriminator:
            logging.info(f"Using discriminator filtering with {selection_percentile}th percentile")
        logging.info(f"Saving to: {save_directory}")
    
    generated_images = []
    all_labels = []
    batch_idx = 0
    
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        batch_size_actual = end - start
        
        if verbose:
            logging.info(f"Generating batch {batch_idx}: samples {start}-{end-1}")
        
        # Generate z and y (matching original logic)
        z = torch.randn(batch_size_actual, model.latent_dim, device=device)
        y = torch.arange(num_classes, device=device).repeat_interleave(total_samples // num_classes)[start:end]
        y_onehot = F.one_hot(y, num_classes=num_classes).float().to(device)
        
        # Generate images
        with torch.no_grad():
            imgs = model.decode(z, y_onehot).view(-1, 1, 28, 28)
            
            # Apply discriminator filtering if provided
            if discriminator is not None:
                scores = discriminator(imgs).squeeze(1)
                threshold = torch.quantile(scores, selection_percentile / 100.0)
                mask = scores >= threshold
                
                imgs = imgs[mask]
                y = y[mask]
                
                if verbose:
                    logging.info(f"Batch {batch_idx}: {batch_size_actual} -> {len(imgs)} samples after filtering")
            
            if len(imgs) > 0:
                generated_images.append(imgs.cpu())
                all_labels.append(y.cpu())
        
        batch_idx += 1
    
    # Concatenate and save all batches at once
    if generated_images:
        all_images = torch.cat(generated_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        total_saved = len(all_images)
        
        batch_filename = f"{file_prefix}_all.pt"
        batch_path = os.path.join(save_directory, batch_filename)
        
        torch.save({
            'images': all_images,
            'labels': all_labels
        }, batch_path)
        
        if verbose:
            logging.info(f"Saved all {total_saved} samples to {batch_filename}")
        
        return total_samples, total_saved
    else:
        logging.warning("No samples were generated!")
        return total_samples, 0

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

