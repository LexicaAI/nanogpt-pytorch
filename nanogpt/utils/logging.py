"""
Logging utilities for NanoGPT
"""

import os
import time
import json
import logging
from datetime import datetime

class Logger:
    """
    Simple logger for NanoGPT
    """
    def __init__(self, log_dir, name="train", should_write_to_file=True):
        """
        Initialize the logger
        
        Args:
            log_dir (str): Directory to save logs
            name (str, optional): Logger name. Defaults to "train".
            should_write_to_file (bool, optional): Whether to write logs to file. Defaults to True.
        """
        self.log_dir = log_dir
        self.name = name
        self.should_write_to_file = should_write_to_file
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # Create file handler if needed
        if should_write_to_file:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
            self.metrics_file = os.path.join(log_dir, f"{name}_{timestamp}_metrics.jsonl")
            
            fh = logging.FileHandler(self.log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
    
    def info(self, message):
        """Log an info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log an error message"""
        self.logger.error(message)
    
    def log_metrics(self, step, metrics):
        """
        Log metrics to file
        
        Args:
            step (int): Current step
            metrics (dict): Dictionary of metrics
        """
        if not self.should_write_to_file:
            return
        
        log_entry = {
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def log_config(self, config):
        """
        Log configuration
        
        Args:
            config: Configuration object
        """
        self.info("Configuration:")
        for key, value in vars(config).items():
            self.info(f"  {key}: {value}")

class TrainingLogger(Logger):
    """
    Training logger for NanoGPT
    """
    def __init__(self, log_dir, master_process=True):
        """
        Initialize the training logger
        
        Args:
            log_dir (str): Directory to save logs
            master_process (bool, optional): Whether this is the master process. Defaults to True.
        """
        super().__init__(log_dir, "train", master_process)
        self.start_time = time.time()
    
    def log_training_step(self, step, loss, learning_rate, grad_norm, batch_time, tokens_per_sec):
        """
        Log a training step
        
        Args:
            step (int): Current step
            loss (float): Training loss
            learning_rate (float): Current learning rate
            grad_norm (float): Gradient norm
            batch_time (float): Batch processing time in seconds
            tokens_per_sec (float): Tokens processed per second
        """
        self.info(
            f"step {step:5d} | "
            f"loss: {loss:.6f} | "
            f"lr: {learning_rate:.4e} | "
            f"norm: {grad_norm:.4f} | "
            f"dt: {batch_time*1000:.2f}ms | "
            f"tokens/sec: {tokens_per_sec:.2f}"
        )
        
        self.log_metrics(step, {
            "type": "train",
            "loss": loss,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "batch_time": batch_time,
            "tokens_per_sec": tokens_per_sec
        })
    
    def log_validation(self, step, val_loss, perplexity=None):
        """
        Log validation results
        
        Args:
            step (int): Current step
            val_loss (float): Validation loss
            perplexity (float, optional): Perplexity. Defaults to None.
        """
        if perplexity is None:
            perplexity = float('inf')
        
        self.info(f"step {step:5d} | val_loss: {val_loss:.6f} | perplexity: {perplexity:.2f}")
        
        self.log_metrics(step, {
            "type": "val",
            "val_loss": val_loss,
            "perplexity": perplexity
        })
    
    def log_hellaswag(self, step, num_correct, num_total, accuracy):
        """
        Log HellaSwag evaluation results
        
        Args:
            step (int): Current step
            num_correct (int): Number of correct predictions
            num_total (int): Total number of examples
            accuracy (float): Accuracy
        """
        self.info(f"HellaSwag accuracy: {num_correct}/{num_total}={accuracy:.4f}")
        
        self.log_metrics(step, {
            "type": "hellaswag",
            "num_correct": num_correct,
            "num_total": num_total,
            "accuracy": accuracy
        })
    
    def log_elapsed_time(self, step, num_steps):
        """
        Log elapsed time
        
        Args:
            step (int): Current step
            num_steps (int): Total number of steps
        """
        elapsed_time = time.time() - self.start_time
        steps_per_sec = step / elapsed_time if step > 0 else 0
        remaining_steps = num_steps - step
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
        
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        eta_hours, eta_remainder = divmod(eta_seconds, 3600)
        eta_minutes, eta_seconds = divmod(eta_remainder, 60)
        
        self.info(
            f"Elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} | "
            f"ETA: {int(eta_hours):02d}:{int(eta_minutes):02d}:{int(eta_seconds):02d} | "
            f"Progress: {step}/{num_steps} ({100*step/num_steps:.2f}%)"
        ) 