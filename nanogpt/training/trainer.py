"""
Trainer implementation for NanoGPT
"""

import os
import time
import torch
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from nanogpt.data.dataloader import DataLoaderLite
from nanogpt.training.optimizer import get_lr, create_optimizer
from nanogpt.data.datasets.hellaswag import iterate_examples, render_example, get_most_likely_row
from nanogpt.tokenizer.tokenizer import Tokenizer

class Trainer:
    """
    Trainer class for NanoGPT
    """
    def __init__(self, model, config):
        """
        Initialize the trainer
        
        Args:
            model: The model to train
            config (TrainingConfig): Training configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.raw_model = model  # store reference to unwrapped model
        
        # Setup distributed training if needed
        self.setup_distributed()
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup torch compile if needed
        self.use_compile = False  # torch.compile interferes with HellaSwag eval and Generation
        if self.use_compile:
            self.model = torch.compile(self.model)
        
        # If distributed, wrap model
        if self.config.distributed:
            self.model = DDP(self.model, device_ids=[self.config.local_rank])
            self.raw_model = self.model.module  # update reference to raw model
        
        # Create optimizer
        self.optimizer = create_optimizer(self.raw_model, config, config.device)
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer()
        
        # Create data loaders
        self.train_loader = DataLoaderLite(
            B=config.batch_size, 
            T=config.seq_length, 
            process_rank=self.process_rank, 
            num_processes=self.world_size, 
            split="train"
        )
        self.val_loader = DataLoaderLite(
            B=config.batch_size, 
            T=config.seq_length, 
            process_rank=self.process_rank, 
            num_processes=self.world_size, 
            split="val"
        )
        
        # Create checkpoint directory
        if self.master_process:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            # Create log file
            self.log_file = os.path.join(config.checkpoint_dir, "train_log.txt")
            with open(self.log_file, "w") as f:
                f.write("step,type,value\n")
    
    def setup_distributed(self):
        """
        Setup distributed training
        """
        if self.config.distributed:
            init_process_group(backend="nccl")
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.master_process = self.process_rank == 0
            
            # Set device
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            
            # Set random seed
            seed = 0
            torch.manual_seed(seed + self.process_rank)
            torch.backends.cudnn.deterministic = True
        else:
            self.world_size = 1
            self.process_rank = 0
            self.local_rank = 0
            self.master_process = True
    
    def save_checkpoint(self, step, val_loss):
        """
        Save a model checkpoint
        """
        if not self.master_process:
            return
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"ckpt_{step:07d}.pt")
        checkpoint = {
            'model': self.raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.raw_model.config,
            'step': step,
            'val_loss': val_loss
        }
        # You might also want to add optimizer.state_dict() and
        # rng seeds etc., if you wanted to more exactly resume training
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def train_step(self, step):
        """
        Execute one training step
        """
        self.model.train()
        t0 = time.time()
        
        # Initialize loss accumulator
        loss_accum = 0.0
        
        # Perform gradient accumulation
        for micro_step in range(self.config.grad_accumulation_steps):
            # Get batch
            x, y = self.train_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            if self.config.distributed:
                # Only synchronize gradients on last micro-step
                self.model.require_backward_grad_sync = (
                    micro_step == self.config.grad_accumulation_steps - 1
                )
            
            # Use autocast for mixed precision
            with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                logits, loss = self.model(x, y)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.grad_accumulation_steps
            loss_accum += loss.detach()
            
            # Backward pass
            loss.backward()
        
        # All-reduce loss across processes if distributed
        if self.config.distributed:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        # Clip gradients
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        # Update learning rate
        lr = get_lr(step, self.config)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Update parameters
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        if self.config.device == "cuda":
            torch.cuda.synchronize()  # wait for GPU to finish work
        
        # Log progress
        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        tokens_processed = (
            self.config.batch_size 
            * self.config.seq_length 
            * self.config.grad_accumulation_steps 
            * self.world_size
        )
        tokens_per_sec = tokens_processed / dt
        
        if self.master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(self.log_file, "a") as f:
                f.write(f"{step},train,{loss_accum.item():.6f}\n")
        
        return loss_accum.item()
    
    def evaluate(self, step):
        """
        Evaluate the model on the validation set
        """
        self.model.eval()
        
        # Validation loss
        val_loss_accum = 0.0
        for i in range(self.config.eval_steps):
            x, y = self.val_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
            val_loss_accum += loss.detach()
        
        val_loss_accum = val_loss_accum / self.config.eval_steps
        
        # All-reduce validation loss if distributed
        if self.config.distributed:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        
        if self.master_process:
            print(f"step {step:5d} | val loss: {val_loss_accum.item():.6f}")
            with open(self.log_file, "a") as f:
                f.write(f"{step},val,{val_loss_accum.item():.6f}\n")
        
        return val_loss_accum.item()
    
    def evaluate_hellaswag(self, step):
        """
        Evaluate the model on the HellaSwag dataset
        """
        if (not self.master_process) or self.use_compile:
            return
        
        # Skip evaluation if not at evaluation interval
        if step % self.config.eval_interval != 0 and step != 0:
            return
        
        self.model.eval()
        
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # Only process examples where i % world_size == process_rank
            if i % self.world_size != self.process_rank:
                continue
            
            # Render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(self.device)
            mask = mask.to(self.device)
            
            # Get the logits
            with torch.no_grad():
                with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                    logits, loss = self.model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        
        # Reduce the stats across all processes if distributed
        if self.config.distributed:
            num_total_tensor = torch.tensor(num_total, dtype=torch.long, device=self.device)
            num_correct_norm_tensor = torch.tensor(num_correct_norm, dtype=torch.long, device=self.device)
            dist.all_reduce(num_total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm_tensor, op=dist.ReduceOp.SUM)
            num_total = num_total_tensor.item()
            num_correct_norm = num_correct_norm_tensor.item()
        
        if num_total > 0:
            acc_norm = num_correct_norm / num_total
        else:
            acc_norm = 0.0
        
        if self.master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(self.log_file, "a") as f:
                f.write(f"{step},hella,{acc_norm:.4f}\n")
    
    def generate_samples(self, step):
        """
        Generate text samples from the model
        """
        # Skip generation if not at generation interval
        if ((step == 0) or (step % self.config.generate_interval != 0)) or self.use_compile:
            return
        
        self.model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = self.tokenizer.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(self.device)
        
        # Set RNG for reproducibility
        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(42 + self.process_rank)
        
        # Generate tokens
        while xgen.size(1) < max_length:
            # Forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                    logits, loss = self.model(xgen)  # (B, T, vocab_size)
                
                # Take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                
                # Get the probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Do top-k sampling of 50 (huggingface pipeline default)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                
                # Select a token from the top-k probabilities
                # Note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                
                # Gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                
                # Append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        
        # Print the generated text
        if self.master_process:
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = self.tokenizer.decode(tokens)
                print(f"Sample {i}: {decoded}")
    
    def train(self, num_steps):
        """
        Train the model for a specified number of steps
        
        Args:
            num_steps (int): Number of training steps
        """
        # Set precision
        torch.set_float32_matmul_precision('high')
        
        # Main training loop
        for step in range(num_steps):
            # Training step
            train_loss = self.train_step(step)
            
            # Evaluate
            if step % self.config.eval_interval == 0:
                val_loss = self.evaluate(step)
                self.evaluate_hellaswag(step)
                self.generate_samples(step)
                # Save checkpoint
                if step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(step, val_loss)
            
            # Last step
            if step == num_steps - 1:
                val_loss = self.evaluate(step)
                self.evaluate_hellaswag(step)
                self.generate_samples(step)
                self.save_checkpoint(step, val_loss)
        
        # Clean up
        if self.config.distributed:
            destroy_process_group()
