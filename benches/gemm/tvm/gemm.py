import numpy as np
import tvm
from tvm import te
import tvm.testing
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple

def create_tvm_gemm(M: int, N: int, K: int, dtype: str = "float16"):
    """
    Create TVM GEMM implementation
    Args:
        M: Number of rows in matrix A
        N: Number of columns in matrix B
        K: Number of columns in matrix A and rows in matrix B
        dtype: Data type for computation
    Returns:
        Compiled TVM function
    """
    try:
        logger.info(f"Creating TVM GEMM for matrix size {M}x{K} x {K}x{N}")
        
        # Define the computation
        A = te.placeholder((M, K), name='A', dtype=dtype)
        B = te.placeholder((K, N), name='B', dtype=dtype)
        
        # Define the computation
        k = te.reduce_axis((0, K), name='k')
        C = te.compute((M, N),
                      lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                      name='C')
        
        # Create schedule
        s = te.create_schedule(C.op)
        
        # Get the default schedule
        i, j = s[C].op.axis
        k = C.op.reduce_axis[0]
        
        # Split the computation into blocks
        BLOCK_SIZE = 32
        i_outer, i_inner = s[C].split(i, factor=BLOCK_SIZE)
        j_outer, j_inner = s[C].split(j, factor=BLOCK_SIZE)
        k_outer, k_inner = s[C].split(k, factor=BLOCK_SIZE)
        
        # Reorder the computation
        s[C].reorder(i_outer, j_outer, k_outer, k_inner, i_inner, j_inner)
        
        # Create target
        target = tvm.target.Target("cuda")
        logger.info(f"Using target: {target}")
        
        # Create function
        func = tvm.build(s, [A, B, C], target=target, name="gemm")
        logger.info("TVM function built successfully")
        
        return func
    except Exception as e:
        logger.error(f"Error creating TVM GEMM: {str(e)}")
        raise

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
    try:
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
    except Exception as e:
        logger.error(f"Error in measure_time: {str(e)}")
        raise

def benchmark(M: int, N: int, K: int, provider: str) -> Tuple[float, float, float]:
    """
    Benchmark matrix multiplication implementations
    Args:
        M: Number of rows in matrix A
        N: Number of columns in matrix B
        K: Number of columns in matrix A and rows in matrix B
        provider: Implementation to benchmark ('tvm', 'pytorch', or 'cublas')
    Returns:
        Tuple of (median_time_ms, min_time_ms, max_time_ms)
    """
    try:
        logger.info(f"Starting benchmark for {provider} with matrix size {M}x{K} x {K}x{N}")
        
        # Create random input matrices
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        
        if provider == 'tvm':
            # Create TVM function
            tvm_func = create_tvm_gemm(M, N, K)
            
            # Convert PyTorch tensors to TVM tensors
            ctx = tvm.cuda()
            logger.info("Converting PyTorch tensors to TVM tensors")
            a_tvm = tvm.nd.array(a.cpu().numpy(), ctx)
            b_tvm = tvm.nd.array(b.cpu().numpy(), ctx)
            c_tvm = tvm.nd.array(np.zeros((M, N), dtype=np.float16), ctx)
            
            # Create wrapper function for timing
            def tvm_wrapper():
                tvm_func(a_tvm, b_tvm, c_tvm)
            
            return measure_time(tvm_wrapper)
        elif provider == 'pytorch':
            def pytorch_wrapper():
                torch.matmul(a, b)
            return measure_time(pytorch_wrapper)
        else:  # cublas
            def cublas_wrapper():
                torch.matmul(a, b)
            return measure_time(cublas_wrapper)
    except Exception as e:
        logger.error(f"Error in benchmark for {provider}: {str(e)}")
        raise

def plot_results(results: List[Tuple[float, float, float]], sizes: List[int], providers: List[str]):
    """
    Plot benchmark results
    """
    try:
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
        logger.info("Performance plot saved as gemm_performance.png")
    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")
        raise

def main():
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Benchmark configurations
        sizes = [512, 1024, 2048, 4096, 8192]
        providers = ['cublas', 'pytorch', 'tvm']
        
        # Run benchmarks
        results = []
        for size in sizes:
            print(f"\nTesting matrix size: {size}x{size}")
            size_results = []
            for provider in providers:
                try:
                    median_ms, min_ms, max_ms = benchmark(size, size, size, provider)
                    print(f"{provider}:")
                    print(f"  Median time: {median_ms:.3f} ms")
                    print(f"  Min time:    {min_ms:.3f} ms")
                    print(f"  Max time:    {max_ms:.3f} ms")
                    size_results.append((median_ms, min_ms, max_ms))
                except Exception as e:
                    logger.error(f"Failed to benchmark {provider} for size {size}: {str(e)}")
                    continue
            results.append(size_results)
        
        # Plot results
        plot_results(results, sizes, providers)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main() 