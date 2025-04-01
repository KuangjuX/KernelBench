import torch
import triton
import triton.language as tl
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

@triton.jit
def fused_gemm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, d_ptr,
    # Matrix dimensions
    M, N, K, L,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,  #
    stride_bk, stride_bn,  #
    stride_cn, stride_cl,  #
    stride_dm, stride_dl,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
    BLOCK_SIZE_L: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the fused matmul D = (A @ B) @ C.
    A has shape (M, K), B has shape (K, N), C has shape (N, L) and D has shape (M, L)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of D it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_l = tl.cdiv(L, BLOCK_SIZE_L)
    num_pid_in_group = GROUP_SIZE_M * num_pid_l
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_l = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A, B, and C.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bl = (pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)) % L
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c_ptrs = c_ptr + (offs_n[:, None] * stride_cn + offs_bl[None, :] * stride_cl)

    # -----------------------------------------------------------
    # Iterate to compute a block of the D matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_L]` block
    # of fp32 values for higher accuracy.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # First GEMM: A @ B
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Second GEMM: (A @ B) @ C
    # Create accumulator for second GEMM
    accumulator2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_L), dtype=tl.float32)
    
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # Load the next block of C
        c = tl.load(c_ptrs, mask=offs_n[:, None] < N - n * BLOCK_SIZE_N, other=0.0)
        # Convert c to float32 for consistent types
        c = c.to(tl.float32)
        # We accumulate along the N dimension
        accumulator2 = tl.dot(accumulator, c, accumulator2)
        # Advance the ptrs to the next N block
        c_ptrs += BLOCK_SIZE_N * stride_cn
    
    # -----------------------------------------------------------
    # Write back the block of the output matrix D with masks.
    offs_dm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_dl = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    d_ptrs = d_ptr + stride_dm * offs_dm[:, None] + stride_dl * offs_dl[None, :]
    d_mask = (offs_dm[:, None] < M) & (offs_dl[None, :] < L)
    tl.store(d_ptrs, accumulator2.to(tl.float16), mask=d_mask)

def fused_gemm(a, b, c):
    """
    Fused matrix multiplication D = (A @ B) @ C
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
        c: Input matrix C of shape (N, L)
    Returns:
        Output matrix D of shape (M, L)
    """
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for A and B"
    assert b.shape[1] == c.shape[0], "Incompatible dimensions for B and C"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert c.is_contiguous(), "Matrix C must be contiguous"
    
    M, K = a.shape
    K, N = b.shape
    N, L = c.shape
    
    # Allocates output
    d = torch.empty((M, L), device=a.device, dtype=torch.float16)
    
    # 1D launch kernel where each block gets its own program
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(L, META['BLOCK_SIZE_L']),)
    
    fused_gemm_kernel[grid](
        a, b, c, d,  #
        M, N, K, L,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        d.stride(0), d.stride(1),  #
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, BLOCK_SIZE_L=64,  #
        GROUP_SIZE_M=8,  #
        num_warps=4,    # 设置 warp 数量为 4
        num_stages=1,   # 设置流水线阶段数为 1
        num_ctas=1,     # 设置 CTA 数量为 1
    )
    
    return d

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

def benchmark(M: int, N: int, K: int, L: int, provider: str) -> Tuple[float, float, float]:
    """
    Benchmark matrix multiplication implementations
    """
    # Create random input matrices
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.randn((N, L), device='cuda', dtype=torch.float16)
    
    if provider == 'triton':
        def triton_wrapper():
            fused_gemm(a, b, c)
        return measure_time(triton_wrapper)
    else:  # pytorch
        def pytorch_wrapper():
            torch.matmul(torch.matmul(a, b), c)
        return measure_time(pytorch_wrapper)

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
    plt.title('Fused GEMM Performance Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig('fused_gemm_performance.png')
    plt.close()

def main():
    # Benchmark configurations
    shapes = [
        (1024, 1024, 128, 128),
        (2048, 2048, 128, 128),
        (4096, 4096, 128, 128),
        (8192, 8192, 128, 128)
    ]
    providers = ['pytorch', 'triton']
    
    # Run benchmarks
    results = []
    for M, N, K, L in shapes:
        print(f"\nTesting matrix shapes:")
        print(f"  A: {M}x{K}")
        print(f"  B: {K}x{N}")
        print(f"  C: {N}x{L}")
        print(f"  D: {M}x{L}")
        
        size_results = []
        for provider in providers:
            median_ms, min_ms, max_ms = benchmark(M, N, K, L, provider)
            print(f"{provider}:")
            print(f"  Median time: {median_ms:.3f} ms")
            print(f"  Min time:    {min_ms:.3f} ms")
            print(f"  Max time:    {max_ms:.3f} ms")
            size_results.append((median_ms, min_ms, max_ms))
        results.append(size_results)
    
    # Plot results
    plot_results(results, [f"{M}x{K}" for M, N, K, L in shapes], providers)

if __name__ == '__main__':
    main() 