#!/usr/bin/env python

"""
Download and process the FineWeb-Edu dataset for NanoGPT training.
"""

import os
import argparse
from nanogpt.data.datasets.fineweb import FinewebProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Download and process the FineWeb-Edu dataset")
    
    parser.add_argument("--local_dir", type=str, default="edu_fineweb10B", 
                        help="Local directory to save processed dataset")
    parser.add_argument("--remote_name", type=str, default="sample-10BT", 
                        help="Remote dataset name on HuggingFace")
    parser.add_argument("--shard_size", type=int, default=int(1e8), 
                        help="Number of tokens per shard")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Downloading and processing FineWeb-Edu dataset ({args.remote_name})...")
    print(f"Dataset will be saved to: {args.local_dir}")
    
    # Create the processor
    processor = FinewebProcessor(
        local_dir=args.local_dir,
        remote_name=args.remote_name,
        shard_size=args.shard_size
    )
    
    # Process the dataset
    num_shards = processor.process()
    
    print(f"Successfully processed dataset into {num_shards} shards")
    print(f"Dataset is ready for training!")

if __name__ == "__main__":
    main()
