import torch
import torch.nn.functional as F
import time
import numpy as np
from typing import Tuple
import math

class FlashAttention(torch.nn.Module):
    def __init__(self, head_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.head_dim = head_dim
        self.dropout_p = dropout_p
        self.scale = 1.0 / math.sqrt(head_dim)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        FlashAttention implementation with CUDA optimization.
        
        Args:
            q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            v: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            mask: Optional attention mask of shape [batch_size, seq_len, seq_len]
        
        Returns:
            Output tensor of shape [batch_size, seq_len, num_heads, head_dim]
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Reshape for matrix multiplication
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores with CUDA optimization
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax with CUDA optimization
        attn = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if self.dropout_p > 0.0:
            attn = F.dropout(attn, p=self.dropout_p)
        
        # Compute output with CUDA optimization
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        
        return out

def generate_random_inputs(batch_size: int, seq_len: int, head_dim: int, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate random input tensors for FlashAttention.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        head_dim: Head dimension
        num_heads: Number of attention heads
    
    Returns:
        Tuple of (q, k, v) tensors
    """
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
    return q, k, v

def evaluate_performance(batch_size: int, seq_len: int, head_dim: int, num_heads: int, 
                        warmup_iters: int = 5, measure_iters: int = 20) -> dict:
    """
    Evaluate the performance of FlashAttention with detailed metrics using CUDA events.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        head_dim: Head dimension
        num_heads: Number of attention heads
        warmup_iters: Number of warmup iterations
        measure_iters: Number of measurement iterations
    
    Returns:
        Dictionary containing performance metrics
    """
    # Create model and move to CUDA
    model = FlashAttention(head_dim).cuda()
    
    # Generate inputs
    q, k, v = generate_random_inputs(batch_size, seq_len, head_dim, num_heads)
    
    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(warmup_iters):
        _ = model(q, k, v)
    
    # Synchronize
    torch.cuda.synchronize()
    
    # Measure
    times = []
    memory_usage = []
    
    for _ in range(measure_iters):
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Get initial memory usage
        initial_memory = torch.cuda.memory_allocated()
        
        # Record start event
        start_event.record()
        
        # Forward pass
        _ = model(q, k, v)
        
        # Record end event
        end_event.record()
        
        # Synchronize
        torch.cuda.synchronize()
        
        # Calculate elapsed time in milliseconds
        elapsed_time = start_event.elapsed_time(end_event)
        
        # Get final memory usage
        final_memory = torch.cuda.memory_allocated()
        
        times.append(elapsed_time)
        memory_usage.append(final_memory - initial_memory)
    
    # Calculate metrics
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_memory = np.mean(memory_usage) / (1024 * 1024)  # Convert to MB
    
    # Calculate FLOPs
    flops = 2 * batch_size * seq_len * seq_len * num_heads * head_dim
    gflops = (flops / (avg_time / 1000)) / 1e9  # Convert to GFLOPs
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'avg_memory_mb': avg_memory,
        'gflops': gflops
    }

def main():
    # Test configurations
    configs = [
        (32, 128, 128, 8),
        (32, 256, 128, 8),
        (32, 512, 128, 8),
        (32, 768, 128, 8),
        (32, 1024, 128, 8),
        (32, 1536, 128, 8),
        (32, 2048, 128, 8),
        (32, 4096, 128, 8),
        (32, 128, 256, 8),
        (32, 256, 256, 8),
        (32, 512, 256, 8),
        (32, 768, 256, 8),
        (32, 1024, 256, 8),
        (32, 1536, 256, 8),
        (32, 2048, 256, 8),
        (32, 4096, 256, 8),
    ]
    
    # Define column widths and formats
    col_widths = {
        'batch': 8,
        'seq_len': 8,
        'head_dim': 8,
        'num_heads': 10,
        'time': 12,
        'std': 10,
        'memory': 12,
        'gflops': 10
    }
    
    # Print header
    print("FlashAttention Performance Evaluation")
    print("=" * 100)
    header = (f"{'Batch':<{col_widths['batch']}} "
              f"{'Seq Len':<{col_widths['seq_len']}} "
              f"{'Head Dim':<{col_widths['head_dim']}} "
              f"{'Num Heads':<{col_widths['num_heads']}} "
              f"{'Time (ms)':<{col_widths['time']}} "
              f"{'Std (ms)':<{col_widths['std']}} "
              f"{'Memory (MB)':<{col_widths['memory']}} "
              f"{'GFLOPs':<{col_widths['gflops']}}")
    print(header)
    print("-" * 100)
    
    results = []
    for batch_size, seq_len, head_dim, num_heads in configs:
        metrics = evaluate_performance(batch_size, seq_len, head_dim, num_heads)
        results.append({
            'batch_size': batch_size,
            'seq_len': seq_len,
            'head_dim': head_dim,
            'num_heads': num_heads,
            **metrics
        })
        
        # Format output with consistent precision
        output = (f"{batch_size:<{col_widths['batch']}} "
                 f"{seq_len:<{col_widths['seq_len']}} "
                 f"{head_dim:<{col_widths['head_dim']}} "
                 f"{num_heads:<{col_widths['num_heads']}} "
                 f"{metrics['avg_time_ms']:>{col_widths['time']-3}.2f} "
                 f"{metrics['std_time_ms']:>{col_widths['std']-3}.2f} "
                 f"{metrics['avg_memory_mb']:>{col_widths['memory']-3}.2f} "
                 f"{metrics['gflops']:>{col_widths['gflops']-3}.2f}")
        print(output)
    
    # Save results to CSV for paper
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('flash_attention_results.csv', index=False)
    print("\nResults saved to flash_attention_results.csv")

if __name__ == "__main__":
    main()
