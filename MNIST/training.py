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
    
    if verbose:
        logging.info(f"Starting training for {epochs} epochs")        
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        
        for x, y in train_loader:
            x = x.view(-1, 784).to(device)
            y = one_hot(y).to(device)            
            # Forward pass and loss computation
            optimizer.zero_grad()
            loss = model.loss(x, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()            
        
        # Calculate average loss
        if total_samples > 0:
            avg_loss = total_loss / total_samples
        else:
            avg_loss = float('inf')
        
        # Record history
        history['train_losses'].append(avg_loss)
        history['epochs_trained'] = epoch + 1
        
        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            history['best_loss'] = best_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if verbose:
                print(f"EarlyStopping counter: {trigger_times} out of {patience}")
            if trigger_times >= patience:
                if verbose:
                    print("Early stopping triggered.")
                history['early_stopped'] = True
                break
    
    if verbose:
        logging.info(f"Training completed. Best loss: {history['best_loss']:.4f}")
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
        
        for x, y in train_loader:
            x = x.view(-1, 784).to(device)
            y = one_hot(y).to(device)
            
            optimizer.zero_grad()
            loss = model.loss(x, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        
        avg_train_loss = train_loss / train_samples if train_samples > 0 else float('inf')
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.view(-1, 784).to(device)
                y = one_hot(y).to(device)
                
                loss = model.loss(x, y)
                val_loss += loss.item()
                
        
        avg_val_loss = val_loss / val_samples if val_samples > 0 else float('inf')
        
        # Record history
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['epochs_trained'] = epoch + 1
        
        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping based on training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            history['best_train_loss'] = best_train_loss
            history['best_val_loss'] = avg_val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if verbose:
                print(f"EarlyStopping counter: {trigger_times} out of {patience}")
            if trigger_times >= patience:
                if verbose:
                    print("Early stopping triggered.")
                history['early_stopped'] = True
                break
    
    if verbose:
        logging.info(f"Training completed. Best training loss: {history['best_train_loss']:.4f}. Best validation loss: {history['best_val_loss']:.4f}")
        if history['early_stopped']:
            logging.info(f"Early stopped at epoch {history['epochs_trained']}")
        else:
            logging.info(f"Completed all {epochs} epochs")
    
    return history

# Example usage functions
def create_train_val_loaders(
    full_dataset,
    train_ratio: float = 0.8,
    batch_size: int = 128,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders from a full dataset.
    
    Args:
        full_dataset: Full dataset to split
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for DataLoaders
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
