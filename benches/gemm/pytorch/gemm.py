import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

@torch.jit.script
def jit_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled PyTorch implementation of matrix multiplication
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
    Returns:
        Output matrix C of shape (M, N)
    """
    return torch.matmul(a, b)

def pytorch_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Regular PyTorch implementation of matrix multiplication
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
    Returns:
        Output matrix C of shape (M, N)
    """
    return torch.matmul(a, b)

def measure_time(func, *args, num_warmup: int = 3, num_repeats: int = 100) -> Tuple[float, float, float]:
    """
    Measure execution time using PyTorch CUDA events
    Args:
        func: Function to measure
        *args: Arguments to pass to the function
        num_warmup: Number of warmup runs
        num_repeats: Number of measurement runs
    Returns:
        Tuple of (median_time_ms, min_time_ms, max_time_ms)
    """
    # Warmup runs
    for _ in range(num_warmup):
        func(*args)
    
    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Measure execution time
    times = []
    for _ in range(num_repeats):
        start_event.record()
        func(*args)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    return np.median(times), np.min(times), np.max(times)

def benchmark(M: int, N: int, K: int, provider: str) -> Tuple[float, float, float]:
    """
    Benchmark matrix multiplication implementations
    Args:
        M: Number of rows in matrix A
        N: Number of columns in matrix B
        K: Number of columns in matrix A and rows in matrix B
        provider: Implementation to benchmark ('pytorch', 'jit', or 'cublas')
    Returns:
        Tuple of (median_time_ms, min_time_ms, max_time_ms)
    """
    # Create random input matrices
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    # Define the function to benchmark
    if provider == 'jit':
        func = jit_gemm
        args = (a, b)
    elif provider == 'pytorch':
        func = pytorch_gemm
        args = (a, b)
    else:  # cublas
        func = torch.matmul
        args = (a, b)
    
    # Measure execution time
    return measure_time(func, *args)

def plot_results(results: List[Tuple[float, float, float]], sizes: List[int], providers: List[str]):
    """
    Plot benchmark results
    """
    plt.figure(figsize=(10, 6))
    
    for i, provider in enumerate(providers):
        times = [r[i][0] for r in results]  # Use median values
        plt.plot(sizes, times, label=provider, marker='o')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (ms)')
    plt.title('GEMM Performance Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig('gemm_performance.png')
    plt.close()

def main():
    # Benchmark configurations
    sizes = [512, 1024, 2048, 4096, 8192]
    providers = ['cublas', 'pytorch', 'jit']
    
    # Run benchmarks
    results = []
    for size in sizes:
        print(f"\nTesting matrix size: {size}x{size}")
        size_results = []
        for provider in providers:
            median_ms, min_ms, max_ms = benchmark(size, size, size, provider)
            print(f"{provider}:")
            print(f"  Median time: {median_ms:.3f} ms")
            print(f"  Min time:    {min_ms:.3f} ms")
            print(f"  Max time:    {max_ms:.3f} ms")
            size_results.append((median_ms, min_ms, max_ms))
        results.append(size_results)
    
    # Plot results
    plot_results(results, sizes, providers)

if __name__ == '__main__':
    main() 