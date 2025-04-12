"""
Evaluation metrics for NanoGPT
"""

import numpy as np
import torch
import torch.nn.functional as F

def calculate_perplexity(loss):
    """
    Calculate perplexity from loss
    
    Args:
        loss (float): Cross-entropy loss
    
    Returns:
        float: Perplexity
    """
    return torch.exp(torch.tensor(loss)).item()

def calculate_accuracy(logits, targets):
    """
    Calculate top-1 accuracy
    
    Args:
        logits (torch.Tensor): Logits from model
        targets (torch.Tensor): True targets
    
    Returns:
        float: Accuracy
    """
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).float()
    return correct.mean().item()

def calculate_loss(logits, targets, ignore_index=-100):
    """
    Calculate cross-entropy loss
    
    Args:
        logits (torch.Tensor): Logits from model
        targets (torch.Tensor): True targets
        ignore_index (int, optional): Index to ignore in loss calculation. Defaults to -100.
    
    Returns:
        torch.Tensor: Loss
    """
    # Shift logits and targets for autoregressive loss
    logits = logits[:, :-1, :].contiguous()
    targets = targets[:, 1:].contiguous()
    
    # Calculate loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index
    )
    return loss 