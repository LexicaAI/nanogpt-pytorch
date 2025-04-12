import os
import numpy as np
import torch

def load_tokens(filename):
    """
    Load tokenized data from a numpy file
    """
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    """
    Simple data loader that streams batches from disk
    """
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B  # batch size
        self.T = T  # sequence length
        self.process_rank = process_rank  # which process this is
        self.num_processes = num_processes  # how many processes in total
        self.split = split  # 'train' or 'val'
        self.reset()

    def reset(self):
        """
        Reset the data loader state, starting from shard zero
        """
        self.shard_ix = 0
        self.start_ix = 0
        # load the first shard
        self.load_next_shard()

    def load_next_shard(self):
        """
        Load the next shard of tokenized data
        """
        dirname = os.path.dirname(__file__)
        dataset_dir = os.path.join(dirname, os.pardir, os.pardir, "edu_fineweb10B")
        fname = os.path.join(dataset_dir, f"edufineweb_{self.split}_{self.shard_ix:06d}.npy")
        if not os.path.exists(fname):
            # we've exausted all shards, reset back to shard 0
            self.shard_ix = 0
            fname = os.path.join(dataset_dir, f"edufineweb_{self.split}_{self.shard_ix:06d}.npy")
        self.data = load_tokens(fname)
        self.start_ix = 0
        self.shard_ix += 1

    def next_batch(self):
        """
        Return the next batch of data
        """
        B, T = self.B, self.T
        if self.data.size(0) - self.start_ix < B * T * 2:
            # if we're about to exhaust the current shard, load the next one
            self.load_next_shard()
        
        # get a contiguous chunk of data
        ix = self.start_ix
        data_size = self.data.size(0)
        # ensure we always have enough data
        assert data_size >= B * T, f"Not enough data in shard {data_size} < {B * T}"
        # get a batch
        x = torch.stack([self.data[ix + i*T:ix + (i+1)*T] for i in range(B)])
        y = torch.stack([self.data[ix + i*T+1:ix + (i+1)*T+1] for i in range(B)])
        self.start_ix += B * T
        return x, y 