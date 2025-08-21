import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Optional, Callable, Tuple
import logging


def one_hot(labels, num_classes=10):
    """Convert labels to one-hot encoding."""
    return F.one_hot(labels, num_classes).float()


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 5,
    verbose: bool = True
) -> dict:
    """
    Train a CVAE model with early stopping.

    Args:
        model: The CVAE model to train
        train_loader: DataLoader for training data
        device: Device to train on
        epochs: Maximum number of training epochs
        lr: Learning rate
        patience: Number of epochs to wait before early stopping
        subset_range: Optional tuple (start, end) to train on subset of data
        one_hot_func: Function to convert labels to one-hot encoding
        verbose: Whether to print training progress

    Returns:
        Tuple of (trained_model, training_history)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training history
    history = {
        'train_losses': [],
        'best_loss': float('inf'),
        'epochs_trained': 0,
        'early_stopped': False
    }

    # Early stopping variables
    best_loss = float('inf')
    trigger_times = 0
    total_samples = len(train_loader.dataset)
    print(f"DEBUG - total_samples: {total_samples}")

    if verbose:
        logging.info(f"Starting training for {epochs} epochs")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_summary = {}  # Accumulate summary statistics across batches

        for x, y in train_loader:
            x = x.view(-1, 784).to(device)

            # Get number of classes from model
            if hasattr(model, 'label_dim'):
                # CVAE model
                num_classes = model.label_dim
            else:
                # Fallback: assume 10 classes (MNIST)
                num_classes = 10

            # Convert labels to one-hot encoding
            y = one_hot(y, num_classes).to(device)

            # Forward pass and loss computation
            optimizer.zero_grad()
            loss, summary = model.loss(x, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accumulate summary statistics
            if not total_summary:
                # First batch: initialize total_summary
                total_summary = {k: v for k, v in summary.items()}
            else:
                # Subsequent batches: accumulate values
                for key in summary:
                    if key in total_summary:
                        total_summary[key] += summary[key]

        # Calculate average loss and average summary statistics
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            # Calculate per-sample averages for summary statistics
            for key, value in total_summary.items():
                total_summary[key] = value / total_samples
        else:
            avg_loss = float('inf')

        # Record history
        history['train_losses'].append(avg_loss)
        history['epochs_trained'] = epoch + 1

        if verbose:
            # Print epoch summary with accumulated statistics
            if total_summary:
                summary_str = ", ".join(
                    [f"{k}: {v:.4f}" for k, v in total_summary.items()])
                print(
                    f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}, {summary_str}")
            else:
                print(
                    f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            history['best_loss'] = best_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if verbose:
                print(
                    f"EarlyStopping counter: {trigger_times} out of {patience}")
            if trigger_times >= patience:
                if verbose:
                    print("Early stopping triggered.")
                history['early_stopped'] = True
                break

    if verbose:
        logging.info(
            f"Training completed. Best loss: {history['best_loss']:.4f}")
        if history['early_stopped']:
            logging.info(f"Early stopped at epoch {history['epochs_trained']}")
        else:
            logging.info(f"Completed all {epochs} epochs")

    return history


def train_model_with_validation(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 5,
    verbose: bool = True
) -> dict:
    """
    Train a CVAE model with validation and early stopping.

    Args:
        model: The CVAE model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        epochs: Maximum number of training epochs
        lr: Learning rate
        patience: Number of epochs to wait before early stopping
        one_hot_func: Function to convert labels to one-hot encoding
        verbose: Whether to print training progress

    Returns:
        Tuple of (trained_model, training_history)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)

    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'best_train_loss': float('inf'),
        'best_val_loss': float('inf'),
        'epochs_trained': 0,
        'early_stopped': False
    }

    # Early stopping variables
    best_train_loss = float('inf')
    trigger_times = 0

    if verbose:
        logging.info(f"Starting training with validation for {epochs} epochs")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_total_summary = {}  # Accumulate training summary statistics across batches

        for x, y in train_loader:
            x = x.view(-1, 784).to(device)

            # Get number of classes from model
            if hasattr(model, 'label_dim'):
                # CVAE model
                num_classes = model.label_dim
            else:
                # Fallback: assume 10 classes (MNIST)
                num_classes = 10

            # Convert labels to one-hot encoding
            y = one_hot(y, num_classes).to(device)

            optimizer.zero_grad()
            loss, summary = model.loss(x, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Accumulate training summary statistics
            if not train_total_summary:
                # First batch: initialize train_total_summary
                train_total_summary = {k: v for k, v in summary.items()}
            else:
                # Subsequent batches: accumulate values
                for key in summary:
                    if key in train_total_summary:
                        train_total_summary[key] += summary[key]

        # Calculate average training loss and summary statistics
        if train_samples > 0:
            avg_train_loss = train_loss / train_samples
            for key, value in train_total_summary.items():
                train_total_summary[key] = value / train_samples
        else:
            avg_train_loss = float('inf')

        # Validation phase
        model.eval()
        val_loss = 0
        val_total_summary = {}  # Accumulate validation summary statistics across batches

        with torch.no_grad():
            for x, y in val_loader:
                x = x.view(-1, 784).to(device)

                # Get number of classes from model
                if hasattr(model, 'label_dim'):
                    # CVAE model
                    num_classes = model.label_dim
                else:
                    # Fallback: assume 10 classes (MNIST)
                    num_classes = 10

                # Convert labels to one-hot encoding
                y = one_hot(y, num_classes).to(device)

                loss, summary = model.loss(x, y)
                val_loss += loss.item()

                # Accumulate validation summary statistics
                if not val_total_summary:
                    # First batch: initialize val_total_summary
                    val_total_summary = {k: v for k, v in summary.items()}
                else:
                    # Subsequent batches: accumulate values
                    for key in summary:
                        if key in val_total_summary:
                            val_total_summary[key] += summary[key]

        # Calculate average validation loss and summary statistics
        if val_samples > 0:
            avg_val_loss = val_loss / val_samples
            # Calculate per-sample averages for validation summary statistics
            for key, value in val_total_summary.items():
                val_total_summary[key] = value / val_samples
        else:
            avg_val_loss = float('inf')

        # Record history
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['epochs_trained'] = epoch + 1

        if verbose:
            print(
                f"Epoch [{epoch+1}/{epochs}] completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            # Print epoch summaries with accumulated statistics
            if train_total_summary and val_total_summary:
                train_summary_str = ", ".join(
                    [f"{k}: {v:.4f}" for k, v in train_total_summary.items()])
                val_summary_str = ", ".join(
                    [f"{k}: {v:.4f}" for k, v in val_total_summary.items()])
                print(
                    f"Train Summary: {train_summary_str}, Val Summary: {val_summary_str}")
            elif train_total_summary:
                train_summary_str = ", ".join(
                    [f"{k}: {v:.4f}" for k, v in train_total_summary.items()])
                print(f"Train Summary: {train_summary_str}")
            elif val_total_summary:
                val_summary_str = ", ".join(
                    [f"{k}: {v:.4f}" for k, v in val_total_summary.items()])
                print(f"Val Summary: {val_summary_str}")

        # Early stopping based on training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            history['best_train_loss'] = best_train_loss
            history['best_val_loss'] = avg_val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if verbose:
                print(
                    f"EarlyStopping counter: {trigger_times} out of {patience}")
            if trigger_times >= patience:
                if verbose:
                    print("Early stopping triggered.")
                history['early_stopped'] = True
                break

    if verbose:
        logging.info(
            f"Training completed. Best training loss: {history['best_train_loss']:.4f}. Best validation loss: {history['best_val_loss']:.4f}")
        if history['early_stopped']:
            logging.info(f"Early stopped at epoch {history['epochs_trained']}")
        else:
            logging.info(f"Completed all {epochs} epochs")

    return history


def calculate_validation_loss(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    val_loss = 0
    val_recon = 0
    val_kld = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.view(-1, 784).to(device)
            y = one_hot(y).to(device)
            loss, summary = model.loss(x, y)
            val_loss += loss.item()
            if 'recon' in summary.keys():
                val_recon += summary['recon']
            if 'kld' in summary.keys():
                val_kld += summary['kld']
    return val_loss / len(val_loader.dataset), val_recon / len(val_loader.dataset), val_kld / len(val_loader.dataset)
