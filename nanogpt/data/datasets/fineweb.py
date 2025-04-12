"""
FineWeb-Edu dataset processor for srs pretraining
"""

import os
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import tiktoken

class FinewebProcessor:
    """
    Processor for the FineWeb-Edu dataset
    """
    def __init__(self, local_dir="edu_fineweb10B", remote_name="sample-10BT", shard_size=int(1e8)):
        """
        Initialize the FineWeb-Edu dataset processor
        
        Args:
            local_dir (str): Local directory to save shards
            remote_name (str): Remote dataset name
            shard_size (int): Number of tokens per shard
        """
        self.local_dir = local_dir
        self.remote_name = remote_name
        self.shard_size = shard_size
        
        # Create the cache in the local directory if it doesn't exist yet
        self.data_cache_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, local_dir)
        os.makedirs(self.data_cache_dir, exist_ok=True)
        
        # Init the tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']  # end of text token
    
    def tokenize(self, doc):
        """
        Tokenizes a single document and returns a numpy array of uint16 tokens
        """
        tokens = [self.eot]  # the special <|endoftext|> token delimits all documents
        tokens.extend(self.enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16
    
    def write_datafile(self, filename, tokens_np):
        """
        Write the tokenized data to a numpy file
        """
        np.save(filename, tokens_np)
    
    def process(self):
        """
        Process the FineWeb-Edu dataset and save shards
        """
        # Download the dataset
        #fw = load_dataset("HuggingFaceFW/fineweb-edu", name=self.remote_name, split="train")
        fw = load_dataset("AIGym/ajibawa-2023-wikihow", split="train")
        
        # Tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
        nprocs = max(1, os.cpu_count()//2)
        with mp.Pool(nprocs) as pool:
            shard_index = 0
            # Preallocate buffer to hold current shard
            all_tokens_np = np.empty((self.shard_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = None
            for tokens in pool.imap(self.tokenize, fw, chunksize=16):

                # Is there enough space in the current shard for the new tokens?
                if token_count + len(tokens) < self.shard_size:
                    # Simply append tokens to current shard
                    all_tokens_np[token_count:token_count+len(tokens)] = tokens
                    token_count += len(tokens)
                    # Update progress bar
                    if progress_bar is None:
                        progress_bar = tqdm(total=self.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(len(tokens))
                else:
                    # Write the current shard and start a new one
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(self.data_cache_dir, f"edufineweb_{split}_{shard_index:06d}")
                    # Split the document into whatever fits in this shard; the remainder goes to next one
                    remainder = self.shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                    self.write_datafile(filename, all_tokens_np)
                    shard_index += 1
                    progress_bar = None
                    # Populate the next shard with the leftovers of the current doc
                    all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                    token_count = len(tokens)-remainder

            # Write any remaining tokens as the last shard
            if token_count != 0:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(self.data_cache_dir, f"edufineweb_{split}_{shard_index:06d}")
                self.write_datafile(filename, all_tokens_np[:token_count])
                
        print(f"Processed FineWeb-Edu dataset into {shard_index+1} shards in {self.data_cache_dir}")
        return shard_index + 1
