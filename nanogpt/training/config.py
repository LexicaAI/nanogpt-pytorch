"""
Training configurations for NanoGPT
"""

class TrainingConfig:
    """
    Configuration for training a GPT model
    """
    def __init__(self):
        # Training parameters
        self.learning_rate = 6e-4
        self.min_lr = self.learning_rate * 0.1
        self.batch_size = 32
        self.seq_length = 1024
        self.grad_accumulation_steps = 5
        self.total_batch_size = self.batch_size * self.seq_length * self.grad_accumulation_steps
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        
        # Schedule parameters
        self.warmup_steps = 715
        self.max_steps = 19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
        
        # Checkpoint parameters
        self.checkpoint_interval = 1000
        self.checkpoint_dir = "checkpoints"
        
        # Evaluation parameters
        self.eval_interval = 250
        self.eval_steps = 10
        self.generate_interval = 250
        
        # Distributed training parameters
        self.distributed = False
        self.world_size = 1
        self.local_rank = 0
        
        # Device parameters
        self.device = "cuda"
        self.dtype = "bfloat16"

# Model type configurations
model_configs = {
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
}

# Default config
default_training_config = TrainingConfig() 