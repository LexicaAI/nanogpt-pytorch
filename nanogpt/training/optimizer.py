"""
Optimization utilities for NanoGPT
"""

import math
import torch

def get_lr(step, config):
    """
    Get learning rate at a given step
    
    Args:
        step (int): Current training step
        config (TrainingConfig): Training configuration
    
    Returns:
        float: Learning rate
    """
    # 1) Linear warmup for warmup_steps steps
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    
    # 2) If step > max_steps, return min learning rate
    if step > config.max_steps:
        return config.min_lr
    
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def create_optimizer(model, config, device_type):
    """
    Create an AdamW optimizer for the model
    
    Args:
        model: The model to optimize
        config (TrainingConfig): Training configuration
        device_type (str): Device type ('cuda' or 'cpu')
    
    Returns:
        torch.optim.AdamW: AdamW optimizer
    """
    # Start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    # Create optimizer groups. Any parameter that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    # Report parameter statistics
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    use_fused = fused_available and device_type == "cuda"
    print(f"using fused AdamW: {use_fused}")
    
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=1e-8,
        fused=use_fused
    )
    
    return optimizer 