"""Kernel Validation Harness for AMDGPU .co Code Objects.

Loads original vs modified .co, runs both with same inputs, checks correctness
via torch.allclose, and measures speedup using HIP event timing.

Supports bf16gemm kernels from aiter with the full KernelArgs structure.
Uses ctypes to call the HIP runtime directly (no hip-python dependency).
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

# Kernel arg layout matching aiter's asm_gemm_a16w16.cu KernelArgs struct
# Padding types from aiter_hip_common.h:
#   p2 = struct { uint32 _p0, _p1; }  = 8 bytes
#   p3 = struct { uint32 _p0, _p1, _p2; } = 12 bytes
# The struct is __attribute__((packed))
class BF16GemmKernelArgs(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("ptr_D", ctypes.c_void_p),         # 8 bytes (void*)
        ("_p0", ctypes.c_char * 8),          # p2 = 8 bytes
        ("ptr_C", ctypes.c_void_p),          # 8 bytes
        ("_p1", ctypes.c_char * 8),          # p2
        ("ptr_A", ctypes.c_void_p),          # 8 bytes
        ("_p2", ctypes.c_char * 8),          # p2
        ("ptr_B", ctypes.c_void_p),          # 8 bytes
        ("_p3", ctypes.c_char * 8),          # p2
        ("alpha", ctypes.c_float),           # 4 bytes
        ("_p4", ctypes.c_char * 12),         # p3 = 12 bytes
        ("beta", ctypes.c_float),            # 4 bytes
        ("_p5", ctypes.c_char * 12),         # p3
        ("stride_D0", ctypes.c_uint32),      # 4 bytes
        ("_p6", ctypes.c_char * 12),         # p3
        ("stride_D1", ctypes.c_uint32),
        ("_p7", ctypes.c_char * 12),
        ("stride_C0", ctypes.c_uint32),
        ("_p8", ctypes.c_char * 12),
        ("stride_C1", ctypes.c_uint32),
        ("_p9", ctypes.c_char * 12),
        ("stride_A0", ctypes.c_uint32),
        ("_p10", ctypes.c_char * 12),
        ("stride_A1", ctypes.c_uint32),
        ("_p11", ctypes.c_char * 12),
        ("stride_B0", ctypes.c_uint32),
        ("_p12", ctypes.c_char * 12),
        ("stride_B1", ctypes.c_uint32),
        ("_p13", ctypes.c_char * 12),
        ("M", ctypes.c_uint32),
        ("_p14", ctypes.c_char * 12),
        ("N", ctypes.c_uint32),
        ("_p15", ctypes.c_char * 12),
        ("K", ctypes.c_uint32),
        ("_p16", ctypes.c_char * 12),
        ("splitk", ctypes.c_uint32),
        ("_p17", ctypes.c_char * 12),
        ("is_out_b16", ctypes.c_uint32),
        ("_p18", ctypes.c_char * 12),
        ("ptr_Bias", ctypes.c_void_p),       # 8 bytes
        ("_p19", ctypes.c_char * 8),         # p2
        ("add_bias", ctypes.c_uint32),
        ("_p20", ctypes.c_char * 12),        # p3
        ("ptr_semaphore", ctypes.c_void_p),  # 8 bytes
        ("_p21", ctypes.c_char * 8),         # p2
    ]


# HIP constants
HIP_SUCCESS = 0

# Lazy-loaded HIP runtime handle
_libhip = None

def _get_hip():
    """Load the HIP runtime library via ctypes."""
    global _libhip
    if _libhip is None:
        _libhip = ctypes.cdll.LoadLibrary("libamdhip64.so")
    return _libhip


@dataclass
class ValidationResult:
    """Result of kernel validation."""
    correctness_pass: bool
    max_abs_error: float
    max_rel_error: float
    original_time_us: float
    modified_time_us: float
    reference_time_us: float
    speedup: float
    kernel_name: str = ""
    dims: tuple = (0, 0, 0)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "correctness_pass": self.correctness_pass,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "original_time_us": round(self.original_time_us, 2),
            "modified_time_us": round(self.modified_time_us, 2),
            "reference_time_us": round(self.reference_time_us, 2),
            "speedup": round(self.speedup, 4),
            "kernel_name": self.kernel_name,
            "dims": list(self.dims),
            "details": self.details,
        }

    def summary(self) -> str:
        status = "PASS" if self.correctness_pass else "FAIL"
        lines = [
            f"=== Validation Result: {status} ===",
            f"Kernel:    {self.kernel_name}",
            f"Dims:      M={self.dims[0]}, N={self.dims[1]}, K={self.dims[2]}",
            f"",
            f"Correctness:     {status}",
            f"  Max abs error: {self.max_abs_error:.6e}",
            f"  Max rel error: {self.max_rel_error:.6e}",
            f"",
            f"Performance:",
            f"  Reference (torch): {self.reference_time_us:.2f} us",
            f"  Original .co:      {self.original_time_us:.2f} us",
            f"  Modified .co:      {self.modified_time_us:.2f} us",
            f"  Speedup:           {self.speedup:.4f}x",
        ]
        return "\n".join(lines)


class KernelValidator:
    """Validate correctness and measure speedup of modified .co kernels."""

    def __init__(self, warmup_iters: int = 10, bench_iters: int = 100):
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self._hip = None
        self._check_gpu()

    def _check_gpu(self) -> None:
        """Verify GPU is available."""
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available (torch.cuda.is_available() returned False)")

    def validate_bf16gemm(
        self,
        original_co: str,
        modified_co: str,
        kernel_name: str,
        M: int = 4096,
        N: int = 4096,
        K: int = 4096,
        tile_m: int = 64,
        tile_n: int = 64,
        split_k: int = 1,
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ) -> ValidationResult:
        """Validate a modified bf16gemm .co against the original.

        Strategy: use aiter's own API for the reference run with the original .co,
        then temporarily swap the .co file to run the modified version through
        the same correct dispatch path. Falls back to ctypes direct launch if
        aiter is not available.
        """
        device = torch.device("cuda:0")

        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        B = torch.randn(N, K, dtype=torch.bfloat16, device=device)

        # Reference: torch computation
        out_ref = torch.zeros(M, N, dtype=torch.float32, device=device)
        ref_time = self._bench_torch_mm(A, B, out_ref)
        ref_output = out_ref.clone()

        # Try file-swap approach: run original, swap file, run modified
        import shutil
        import os

        original_backup = original_co + ".bak"

        # Run original
        out_orig = torch.zeros(M, N, dtype=torch.float32, device=device)
        orig_time = self._run_via_aiter_or_ctypes(
            original_co, kernel_name, A, B, out_orig, tile_m, tile_n, split_k,
        )
        orig_output = out_orig.clone()

        # Swap .co file and run modified (aiter caches the module, so we need
        # to use ctypes for the modified version)
        out_mod = torch.zeros(M, N, dtype=torch.float32, device=device)
        mod_time = self._run_via_aiter_or_ctypes(
            modified_co, kernel_name, A, B, out_mod, tile_m, tile_n, split_k,
        )
        mod_output = out_mod.clone()

        # Check correctness of modified vs original
        abs_diff = (mod_output - orig_output).abs()
        max_abs_error = abs_diff.max().item()
        denom = orig_output.abs().clamp(min=1e-8)
        max_rel_error = (abs_diff / denom).max().item()

        correctness = torch.allclose(mod_output, orig_output, rtol=rtol, atol=atol)

        speedup = orig_time / max(mod_time, 1e-9) if mod_time > 0 else 0.0

        return ValidationResult(
            correctness_pass=correctness,
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
            original_time_us=orig_time,
            modified_time_us=mod_time,
            reference_time_us=ref_time,
            speedup=speedup,
            kernel_name=kernel_name,
            dims=(M, N, K),
            details={
                "tile_m": tile_m,
                "tile_n": tile_n,
                "split_k": split_k,
                "orig_vs_ref_max_error": (orig_output - ref_output).abs().max().item(),
            },
        )

    def _run_via_aiter_or_ctypes(
        self,
        co_path: str,
        kernel_name: str,
        A: torch.Tensor,
        B: torch.Tensor,
        out: torch.Tensor,
        tile_m: int,
        tile_n: int,
        split_k: int,
    ) -> float:
        """Run a kernel via ctypes, benchmarking with torch events."""
        M, K = A.shape
        N = B.shape[0]
        device = A.device

        semaphore = torch.zeros((16, 64), dtype=torch.uint32, device=device)

        gdx = (N + tile_n - 1) // tile_n
        gdy = (M + tile_m - 1) // tile_m
        gdz = split_k

        args = self._build_bf16gemm_args(A, B, out, semaphore, M, N, K, split_k)

        return self._run_co_kernel(
            co_path, kernel_name, args,
            grid=(gdx, gdy, gdz), block=(256, 1, 1),
        )

    def _build_bf16gemm_args(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: torch.Tensor,
        semaphore: torch.Tensor,
        M: int, N: int, K: int,
        split_k: int,
    ) -> BF16GemmKernelArgs:
        """Build the packed KernelArgs struct for bf16gemm.

        Matches the C++ layout in asm_gemm_a16w16.cu exactly:
        stride_*1, stride_C* fields left as 0 (not set in C++ code).
        """
        args = BF16GemmKernelArgs()
        args.ptr_D = out.data_ptr()
        args.ptr_C = 0  # nullptr
        args.ptr_A = A.data_ptr()
        args.ptr_B = B.data_ptr()
        args.alpha = 1.0
        args.beta = 0.0
        args.stride_A0 = A.stride(0) * A.element_size()
        args.stride_A1 = 0
        args.stride_B0 = B.stride(0) * B.element_size()
        args.stride_B1 = 0
        args.stride_D0 = N * out.element_size()
        args.stride_D1 = 0
        args.stride_C0 = args.stride_D0
        args.stride_C1 = 0
        args.M = M
        args.N = N
        args.K = K
        args.splitk = split_k
        args.is_out_b16 = 0  # FP32 output
        args.ptr_Bias = 0    # nullptr
        args.add_bias = 0
        if split_k > 1 and semaphore.numel() > 0:
            args.ptr_semaphore = semaphore.data_ptr()
        else:
            args.ptr_semaphore = 0
        return args

    def _run_co_kernel(
        self,
        co_path: str,
        kernel_name: str,
        args: BF16GemmKernelArgs,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
    ) -> float:
        """Load a .co file, run kernel, and return average time in microseconds.

        Uses ctypes to call HIP runtime directly.
        """
        libhip = _get_hip()

        with open(co_path, "rb") as f:
            co_data = f.read()

        # hipModuleLoadData(hipModule_t *module, const void *image)
        module = ctypes.c_void_p()
        co_buf = ctypes.create_string_buffer(co_data)
        ret = libhip.hipModuleLoadData(ctypes.byref(module), co_buf)
        if ret != HIP_SUCCESS:
            raise RuntimeError(f"hipModuleLoadData failed for {co_path}: error {ret}")

        # hipModuleGetFunction(hipFunction_t *function, hipModule_t module, const char *kname)
        func = ctypes.c_void_p()
        ret = libhip.hipModuleGetFunction(
            ctypes.byref(func), module, kernel_name.encode("utf-8"),
        )
        if ret != HIP_SUCCESS:
            libhip.hipModuleUnload(module)
            raise RuntimeError(f"hipModuleGetFunction failed for '{kernel_name}': error {ret}")

        # Build HIP_LAUNCH_PARAM extra args
        # HIP_LAUNCH_PARAM_BUFFER_POINTER = (void*)0x01
        # HIP_LAUNCH_PARAM_BUFFER_SIZE    = (void*)0x02
        # HIP_LAUNCH_PARAM_END            = (void*)0x03
        HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
        HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
        HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)

        arg_size = ctypes.c_size_t(ctypes.sizeof(args))
        arg_ptr = ctypes.cast(ctypes.pointer(args), ctypes.c_void_p)

        extra = (ctypes.c_void_p * 5)(
            HIP_LAUNCH_PARAM_BUFFER_POINTER,
            arg_ptr,
            HIP_LAUNCH_PARAM_BUFFER_SIZE,
            ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
            HIP_LAUNCH_PARAM_END,
        )

        # hipEventCreate
        start_event = ctypes.c_void_p()
        stop_event = ctypes.c_void_p()
        libhip.hipEventCreate(ctypes.byref(start_event))
        libhip.hipEventCreate(ctypes.byref(stop_event))

        stream = ctypes.c_void_p(0)  # default stream

        # Warmup
        for _ in range(self.warmup_iters):
            libhip.hipModuleLaunchKernel(
                func,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                0, stream, None, extra,
            )
        libhip.hipDeviceSynchronize()

        # Benchmark
        libhip.hipEventRecord(start_event, stream)
        for _ in range(self.bench_iters):
            libhip.hipModuleLaunchKernel(
                func,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                0, stream, None, extra,
            )
        libhip.hipEventRecord(stop_event, stream)
        libhip.hipEventSynchronize(stop_event)

        elapsed_ms = ctypes.c_float()
        libhip.hipEventElapsedTime(ctypes.byref(elapsed_ms), start_event, stop_event)
        avg_us = (elapsed_ms.value * 1000.0) / self.bench_iters

        libhip.hipEventDestroy(start_event)
        libhip.hipEventDestroy(stop_event)
        libhip.hipModuleUnload(module)

        return avg_us

    def _bench_torch_mm(self, A: torch.Tensor, B: torch.Tensor,
                        out: torch.Tensor) -> float:
        """Benchmark torch.mm for reference timing."""
        # Warmup
        for _ in range(self.warmup_iters):
            torch.mm(A.float(), B.float().t(), out=out)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(self.bench_iters):
            torch.mm(A.float(), B.float().t(), out=out)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        return (elapsed_ms * 1000.0) / self.bench_iters


def quick_validate(original_co: str, modified_co: str,
                   kernel_name: str,
                   M: int = 128, N: int = 128, K: int = 128,
                   tile_m: int = 64, tile_n: int = 64) -> ValidationResult:
    """Quick validation helper with small dimensions."""
    validator = KernelValidator(warmup_iters=5, bench_iters=20)
    return validator.validate_bf16gemm(
        original_co, modified_co, kernel_name,
        M=M, N=N, K=K, tile_m=tile_m, tile_n=tile_n,
    )
