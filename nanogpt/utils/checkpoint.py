"""
Model checkpointing utilities
"""

import os
import torch

def save_checkpoint(model, optimizer, step, val_loss, config, filename=None):
    """
    Save a model checkpoint
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        step (int): Current training step
        val_loss (float): Validation loss
        config: Model configuration
        filename (str, optional): Output filename. Defaults to None, which uses a default format.
    
    Returns:
        str: Path to the saved checkpoint
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f"ckpt_{step:07d}.pt"
    
    checkpoint_path = os.path.join(config.checkpoint_dir, filename)
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'step': step,
        'val_loss': val_loss
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load a model checkpoint
    
    Args:
        checkpoint_path (str): Path to checkpoint
        model: The model to load
        optimizer (optional): The optimizer to load. Defaults to None.
    
    Returns:
        tuple: (model, optimizer, config, step, val_loss)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    config = checkpoint.get('config', None)
    step = checkpoint.get('step', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    
    return model, optimizer, config, step, val_loss

def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in a directory
    
    Args:
        checkpoint_dir (str): Directory to search
    
    Returns:
        str: Path to the latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('ckpt_') and f.endswith('.pt')]
    
    if not checkpoint_files:
        return None
    
    # Sort by step number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_checkpoint = checkpoint_files[-1]
    
    return os.path.join(checkpoint_dir, latest_checkpoint) 