#!/usr/bin/env python

"""
Generate text using a trained GPT model.
"""

import argparse
import torch
from nanogpt.model.gpt import GPT
from nanogpt.training.config import TrainingConfig
from nanogpt.evaluation.evaluator import Evaluator
from nanogpt.utils.checkpoint import load_checkpoint, find_latest_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with a trained GPT model")
    
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (use latest if not specified)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu)")
    parser.add_argument("--prompt", type=str, default="Hello, I'm a language model,", help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
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
    model, _, _, _, _ = load_checkpoint(checkpoint_path, GPT(None))
    
    # Create config
    config = TrainingConfig()
    config.device = args.device
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Create evaluator
    evaluator = Evaluator(model, config)
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples with prompt: \"{args.prompt}\"")
    print(f"Parameters: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, max_tokens={args.max_tokens}")
    print("=" * 80)
    
    for i in range(args.num_samples):
        # Generate text
        output = evaluator.generate_text(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        print(f"\nSample {i+1}:")
        print("-" * 40)
        print(output)
        print("-" * 40)
    
    print("\nGeneration complete.")

if __name__ == "__main__":
    main() 