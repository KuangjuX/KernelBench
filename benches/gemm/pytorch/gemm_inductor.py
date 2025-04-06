import torch
import torch._dynamo as dynamo
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

# Enable Inductor backend
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False

def gemm_inductor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication C = A @ B using PyTorch Inductor
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
    Returns:
        Output matrix C of shape (M, N)
    """
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for A and B"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    # Perform GEMM
    return torch.matmul(a, b)

# Compile the function with Inductor backend
@dynamo.optimize("inductor")
def gemm_inductor_compiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return gemm_inductor(a, b)

def measure_time(func, *args, num_warmup: int = 3, num_repeats: int = 100) -> Tuple[float, float, float]:
    """
    Measure execution time using PyTorch CUDA events
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

def benchmark(M: int, N: int, K: int) -> Tuple[float, float, float]:
    """
    Benchmark PyTorch Inductor GEMM implementation
    """
    # Create random input matrices
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    def inductor_wrapper():
        gemm_inductor_compiled(a, b)
    
    return measure_time(inductor_wrapper)

def plot_results(results: List[Tuple[float, float, float]], sizes: List[str]):
    """
    Plot benchmark results
    """
    plt.figure(figsize=(10, 6))
    
    times = [r[0] for r in results]  # Use median values
    plt.plot(sizes, times, label='PyTorch Inductor', marker='o')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (ms)')
    plt.title('PyTorch Inductor GEMM Performance')
    plt.grid(True)
    plt.legend()
    plt.savefig('pytorch_inductor_gemm_performance.png')
    plt.close()

def verify_correctness(M: int, N: int, K: int):
    """
    Verify the correctness of the implementation
    """
    # Create random input matrices
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    # Compute using Inductor
    result = gemm_inductor_compiled(a, b)
    
    # Compute using standard PyTorch for verification
    expected = torch.matmul(a, b)
    
    # Check if results match
    max_diff = torch.max(torch.abs(result - expected))
    print(f"Maximum difference between results: {max_diff.item()}")
    
    if torch.allclose(result, expected, rtol=1e-2, atol=1e-2):
        print("✅ Results match within tolerance")
    else:
        print("❌ Results do not match")

def main():
    # Benchmark configurations
    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192)
    ]
    
    # Verify correctness first
    print("Verifying correctness...")
    verify_correctness(*shapes[0])
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    results = []
    for M, N, K in shapes:
        print(f"\nTesting matrix shapes:")
        print(f"  A: {M}x{K}")
        print(f"  B: {K}x{N}")
        print(f"  C: {M}x{N}")
        
        median_ms, min_ms, max_ms = benchmark(M, N, K)
        print(f"PyTorch Inductor:")
        print(f"  Median time: {median_ms:.3f} ms")
        print(f"  Min time:    {min_ms:.3f} ms")
        print(f"  Max time:    {max_ms:.3f} ms")
        results.append((median_ms, min_ms, max_ms))
    
    # Plot results
    plot_results(results, [f"{M}x{K}" for M, N, K in shapes])

if __name__ == '__main__':
    main() 