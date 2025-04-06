import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

def fused_gemm_pytorch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Fused matrix multiplication D = (A @ B) @ C using PyTorch
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
        c: Input matrix C of shape (N, L)
    Returns:
        Output matrix D of shape (M, L)
    """
    # Check constraints
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions for A and B"
    # assert b.shape[1] == c.shape[0], "Incompatible dimensions for B and C"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    # assert c.is_contiguous(), "Matrix C must be contiguous"
    
    # Perform fused GEMM
    return torch.matmul(torch.matmul(a, b), c)

def evaluate_performance(M: int, N: int, K: int, L: int, 
                        warmup_iters: int = 5, measure_iters: int = 20) -> dict:
    """
    Evaluate the performance of Fused GEMM with detailed metrics using CUDA events.
    
    Args:
        M: Rows of matrix A
        N: Columns of matrix B
        K: Columns of matrix A / Rows of matrix B
        L: Columns of matrix C
        warmup_iters: Number of warmup iterations
        measure_iters: Number of measurement iterations
    
    Returns:
        Dictionary containing performance metrics
    """
    # Create random input matrices
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.randn((N, L), device='cuda', dtype=torch.float16)
    
    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(warmup_iters):
        _ = fused_gemm_pytorch(a, b, c)
    
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
        _ = fused_gemm_pytorch(a, b, c)
        
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
    flops = 2 * M * N * K + 2 * M * N * L  # (A @ B) + (AB @ C)
    gflops = (flops / (avg_time / 1000)) / 1e9  # Convert to GFLOPs
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'avg_memory_mb': avg_memory,
        'gflops': gflops
    }

def main():
    # Benchmark configurations
    configs = [
        (1024, 1024, 128, 128),
        (2048, 2048, 128, 128),
        (4096, 4096, 128, 128),
        (8192, 8192, 128, 128)
    ]
    
    # Define column widths and formats
    col_widths = {
        'M': 8,
        'N': 8,
        'K': 8,
        'L': 8,
        'time': 12,
        'std': 10,
        'memory': 12,
        'gflops': 10
    }
    
    # Print header
    print("Fused GEMM Performance Evaluation")
    print("=" * 100)
    header = (f"{'M':<{col_widths['M']}} "
              f"{'N':<{col_widths['N']}} "
              f"{'K':<{col_widths['K']}} "
              f"{'L':<{col_widths['L']}} "
              f"{'Time (ms)':<{col_widths['time']}} "
              f"{'Std (ms)':<{col_widths['std']}} "
              f"{'Memory (MB)':<{col_widths['memory']}} "
              f"{'GFLOPs':<{col_widths['gflops']}}")
    print(header)
    print("-" * 100)
    
    results = []
    for M, N, K, L in configs:
        metrics = evaluate_performance(M, N, K, L)
        results.append({
            'M': M,
            'N': N,
            'K': K,
            'L': L,
            **metrics
        })
        
        # Format output with consistent precision
        output = (f"{M:<{col_widths['M']}} "
                 f"{N:<{col_widths['N']}} "
                 f"{K:<{col_widths['K']}} "
                 f"{L:<{col_widths['L']}} "
                 f"{metrics['avg_time_ms']:>{col_widths['time']-3}.4f} "
                 f"{metrics['std_time_ms']:>{col_widths['std']-3}.4f} "
                 f"{metrics['avg_memory_mb']:>{col_widths['memory']-3}.2f} "
                 f"{metrics['gflops']:>{col_widths['gflops']-3}.2f}")
        print(output)
    
    # Save results to CSV for paper
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('fused_gemm_results.csv', index=False)
    print("\nResults saved to fused_gemm_results.csv")

if __name__ == '__main__':
    main() 