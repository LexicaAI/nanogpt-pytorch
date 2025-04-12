#!/usr/bin/env python

"""
Evaluate a trained GPT model.
"""

import argparse
import torch
from nanogpt.model.gpt import GPT
from nanogpt.training.config import TrainingConfig
from nanogpt.evaluation.evaluator import Evaluator
from nanogpt.data.dataloader import DataLoaderLite
from nanogpt.utils.checkpoint import load_checkpoint, find_latest_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained GPT model")
    
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (use latest if not specified)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--eval_steps", type=int, default=10, help="Number of evaluation steps")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split to evaluate on")
    parser.add_argument("--task", type=str, default="perplexity", choices=["perplexity", "hellaswag", "generation"], help="Evaluation task")
    parser.add_argument("--prompt", type=str, default="Hello, I'm a language model,", help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path is None:
            print("No checkpoint found. Please specify one with --checkpoint.")
            return
    
    print(f"Loading model from {checkpoint_path}")
    
    # Create device
    device = torch.device(args.device)
    
    # Create model and load checkpoint
    model, _, checkpoint_config, step, val_loss = load_checkpoint(checkpoint_path, GPT(None))
    
    # Create config
    config = TrainingConfig()
    config.device = args.device
    config.batch_size = args.batch_size
    config.eval_steps = args.eval_steps
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Create evaluator
    evaluator = Evaluator(model, config)
    
    if args.task == "perplexity":
        # Create data loader
        data_loader = DataLoaderLite(
            B=args.batch_size,
            T=model.config.block_size,
            process_rank=0,
            num_processes=1,
            split=args.split
        )
        
        # Evaluate perplexity
        perplexity = evaluator.evaluate_perplexity(data_loader)
        print(f"Perplexity on {args.split} split: {perplexity:.2f}")
    
    elif args.task == "hellaswag":
        # Evaluate on HellaSwag
        num_correct, num_total, accuracy = evaluator.evaluate_hellaswag()
        print(f"HellaSwag accuracy: {num_correct}/{num_total}={accuracy:.4f}")
    
    elif args.task == "generation":
        # Generate text
        output = evaluator.generate_text(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print("\nGenerated text:")
        print(f"\nPrompt: {args.prompt}")
        print(f"\nOutput: {output}")

if __name__ == "__main__":
    main() 