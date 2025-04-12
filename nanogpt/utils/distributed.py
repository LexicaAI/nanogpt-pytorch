"""
Distributed training utilities
"""

import os
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """
    Set up distributed training environment
    
    Returns:
        tuple: (is_distributed, world_size, process_rank, local_rank, master_process, device)
    """
    # Check if we should use distributed training
    is_distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
    
    if is_distributed:
        # Initialize process group
        init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        process_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        master_process = process_rank == 0
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        # Set random seed
        seed = 0
        torch.manual_seed(seed + process_rank)
        torch.backends.cudnn.deterministic = True
    else:
        world_size = 1
        process_rank = 0
        local_rank = 0
        master_process = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return is_distributed, world_size, process_rank, local_rank, master_process, device

def wrap_model_for_distributed(model, local_rank):
    """
    Wrap a model for distributed training
    
    Args:
        model: PyTorch model
        local_rank (int): Local rank of this process
    
    Returns:
        DDP: Wrapped model
    """
    return DDP(model, device_ids=[local_rank])

def cleanup_distributed():
    """
    Clean up distributed training environment
    """
    if dist.is_initialized():
        destroy_process_group()

def all_reduce_mean(tensor):
    """
    Average a tensor across all processes
    
    Args:
        tensor (torch.Tensor): Tensor to reduce
    """
    if not dist.is_initialized():
        return tensor
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor 