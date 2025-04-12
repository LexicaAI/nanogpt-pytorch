#!/usr/bin/env python

"""
Train a GPT model.
"""

import os
import argparse
import torch

from nanogpt.model.gpt import GPT, GPTConfig
from nanogpt.training.config import TrainingConfig
from nanogpt.training.trainer import Trainer
from nanogpt.utils.distributed import setup_distributed
from nanogpt.utils.logging import TrainingLogger

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, default=None, help="Model type (gpt2, gpt2-medium, gpt2-large, gpt2-xl, or None for from scratch)")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--block_size", type=int, default=1024, help="Block size (context length)")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=5, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--steps", type=int, default=19073, help="Number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=715, help="Warmup steps")
    parser.add_argument("--eval_interval", type=int, default=250, help="Evaluation interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    
    # System configuration
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for training (float32, float16, bfloat16)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup distributed training
    is_distributed, world_size, process_rank, local_rank, master_process, device = setup_distributed()
    
    # Create training config
    config = TrainingConfig()
    config.learning_rate = args.lr
    config.min_lr = args.lr * 0.1
    config.batch_size = args.batch_size
    config.grad_accumulation_steps = args.grad_accum
    config.total_batch_size = config.batch_size * config.seq_length * config.grad_accumulation_steps * world_size
    config.weight_decay = args.weight_decay
    config.warmup_steps = args.warmup_steps
    config.max_steps = args.steps
    config.eval_interval = args.eval_interval
    config.checkpoint_interval = args.save_interval
    config.checkpoint_dir = args.checkpoint_dir
    config.device = args.device
    config.dtype = args.dtype
    config.distributed = is_distributed
    config.world_size = world_size
    config.local_rank = local_rank
    
    # Create logger
    logger = TrainingLogger(args.log_dir, master_process=master_process)
    if master_process:
        logger.log_config(config)
    
    # Create model
    if args.model_type is not None:
        if master_process:
            logger.info(f"Loading pretrained model: {args.model_type}")
        model = GPT.from_pretrained(args.model_type)
    else:
        if master_process:
            logger.info("Creating model from scratch")
        model_config = GPTConfig(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            vocab_size=args.vocab_size,
            block_size=args.block_size
        )
        model = GPT(model_config)
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        if master_process:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        # Checkpoint loading is handled by the trainer
    
    # Train the model
    if master_process:
        logger.info(f"Starting training for {args.steps} steps")
    trainer.train(args.steps)
    
    if master_process:
        logger.info("Training completed")

if __name__ == "__main__":
    main() 