"""
Evaluation pipeline for NanoGPT
"""

import torch
from nanogpt.data.datasets.hellaswag import evaluate as evaluate_hellaswag
from nanogpt.evaluation.metrics import calculate_perplexity

class Evaluator:
    """
    Evaluator class for NanoGPT
    """
    def __init__(self, model, config):
        """
        Initialize the evaluator
        
        Args:
            model: The model to evaluate
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
    
    def evaluate_perplexity(self, data_loader):
        """
        Evaluate model perplexity on a data loader
        
        Args:
            data_loader: DataLoader instance
        
        Returns:
            float: Average perplexity
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for i in range(self.config.eval_steps):
                x, y = data_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
                
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
        
        avg_loss = total_loss / total_samples
        perplexity = calculate_perplexity(avg_loss)
        
        return perplexity
    
    def evaluate_hellaswag(self):
        """
        Evaluate model on the HellaSwag benchmark
        
        Returns:
            tuple: (num_correct, num_total, accuracy)
        """
        num_correct, num_total = evaluate_hellaswag(self.model, self.device)
        accuracy = num_correct / num_total if num_total > 0 else 0.0
        
        return num_correct, num_total, accuracy
    
    def generate_text(self, prompt, max_tokens=100, temperature=1.0, top_k=50, top_p=0.9):
        """
        Generate text from the model
        
        Args:
            prompt (str): Input prompt
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_k (int, optional): Top-k sampling parameter. Defaults to 50.
            top_p (float, optional): Top-p sampling parameter. Defaults to 0.9.
        
        Returns:
            str: Generated text
        """
        from nanogpt.tokenizer.tokenizer import Tokenizer
        
        tokenizer = Tokenizer()
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                # Forward pass
                with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                    logits, _ = self.model(tokens)
                
                # Focus on the last token
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(1, top_k_indices, top_k_values)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for batch_idx in range(logits.size(0)):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        logits[batch_idx][indices_to_remove] = float("-inf")
                
                # Sample from the filtered distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append the sampled token
                tokens = torch.cat((tokens, next_token), dim=1)
                
                # If we generated an EOS token, stop
                if next_token.item() == tokenizer.eot:
                    break
        
        output = tokenizer.decode(tokens[0].tolist())
        return output 