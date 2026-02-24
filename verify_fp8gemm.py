#!/usr/bin/env python3
"""FP8 BlockScale GEMM A/B/C Three-Version Verification Tool.

Compiles and compares three versions of FP8 BlockScale GEMM:
  Version A: fp8gemm_blockscale_naive.hip compiled directly (no AFTT)
  Version B: fp8gemm_blockscale_optimized.hip (AFTT CppTemplateEngine swap) compiled
  Version C: Version B .co + AFTT AsmOptimizer + PatternReplacer binary patch

Outputs: static ASM analysis comparison + GPU execution benchmark + correctness.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import struct
import sys
import tempfile
from pathlib import Path

AFTT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(AFTT_ROOT))

from src.compiler import Compiler, LLVM_OBJDUMP
from src.asm_editor import AsmEditor
from src.asm_optimizer import AsmOptimizer
from src.pattern_replacer import PatternReplacer
from src.cycle_estimator import CycleEstimator
from src.knowledge_base import KnowledgeBase
from src.instruction import EditOperation


SCALE_BLOCK = 128


# ── FP8 E4M3 FNUZ utilities (host-side, pure Python) ─────────────

def fp8e4m3fnuz_to_f32(val: int) -> float:
    """Convert a single FP8 E4M3 FNUZ byte to float32 (gfx942 convention)."""
    if val == 0 or val == 0x80:
        return 0.0
    sign = (val >> 7) & 1
    exp = (val >> 3) & 0xF
    man = val & 0x7
    result = math.ldexp(1.0 + man * 0.125, exp - 8)
    return -result if sign else result


def f32_to_fp8e4m3fnuz(val: float) -> int:
    """Convert float32 to FP8 E4M3 FNUZ (round-to-nearest-even, clamp to max)."""
    if val == 0.0 or math.isnan(val):
        return 0
    sign = 0
    if val < 0:
        sign = 1
        val = -val
    max_val = math.ldexp(1.0 + 7 * 0.125, 7)  # exp=15 -> 15-8=7
    if val > max_val:
        val = max_val
    exp_unbiased = math.floor(math.log2(val)) if val > 0 else -8
    exp_biased = exp_unbiased + 8
    if exp_biased < 1:
        return 0
    if exp_biased > 15:
        exp_biased = 15
    mantissa_f = val / math.ldexp(1.0, exp_biased - 8) - 1.0
    man = int(round(mantissa_f * 8.0))
    if man > 7:
        man = 7
    if man < 0:
        man = 0
    return (sign << 7) | (exp_biased << 3) | man


# ── Helpers ────────────────────────────────────────────────────────

def find_kernel_names(co_path: str) -> list[str]:
    """Extract kernel function names from a .co ELF using llvm-objdump --syms."""
    result = subprocess.run(
        [LLVM_OBJDUMP, "--syms", co_path],
        capture_output=True, text=True, timeout=10)
    names = []
    for line in result.stdout.splitlines():
        if ".text" in line and ("g" in line.split()[1:2] or "F" in line):
            parts = line.split()
            if parts:
                names.append(parts[-1])
    return names


def count_instruction_categories(instructions) -> dict:
    """Tally instruction categories for the comparison table."""
    cats = {
        "total": len(instructions),
        "VALU": 0, "SALU": 0, "VMEM_load": 0, "VMEM_store": 0,
        "LDS": 0, "NOP": 0, "Waitcnt": 0, "Barrier": 0,
        "Branch": 0, "DPP": 0, "MFMA": 0,
    }
    for instr in instructions:
        mn = instr.mnemonic
        full = instr.full_text or ""
        if "mfma" in mn:
            cats["MFMA"] += 1
        elif mn.startswith(("global_load", "buffer_load", "flat_load")):
            cats["VMEM_load"] += 1
        elif mn.startswith(("global_store", "buffer_store", "flat_store")):
            cats["VMEM_store"] += 1
        elif mn.startswith("ds_"):
            cats["LDS"] += 1
        elif mn == "s_nop":
            cats["NOP"] += 1
        elif mn == "s_waitcnt":
            cats["Waitcnt"] += 1
        elif mn == "s_barrier":
            cats["Barrier"] += 1
        elif mn.startswith("s_cbranch") or mn == "s_branch":
            cats["Branch"] += 1
        elif "dpp" in full:
            cats["DPP"] += 1
        elif mn.startswith("v_"):
            cats["VALU"] += 1
        elif mn.startswith("s_"):
            cats["SALU"] += 1
    return cats


def format_table(versions: dict[str, dict]) -> str:
    """Format a comparison table."""
    labels = list(versions.keys())
    all_keys: list[str] = []
    for v in versions.values():
        for k in v:
            if k not in all_keys:
                all_keys.append(k)

    col_w = max(14, *(len(l) + 2 for l in labels))
    key_w = max(16, *(len(k) for k in all_keys))

    sep = "+" + "-" * (key_w + 2)
    for _ in labels:
        sep += "+" + "-" * (col_w + 2)
    sep += "+"

    header = "| " + " " * key_w
    for l in labels:
        header += " | " + l.center(col_w)
    header += " |"

    lines = [sep, header, sep]
    for key in all_keys:
        row = "| " + key.ljust(key_w)
        for l in labels:
            val = versions[l].get(key, "-")
            if isinstance(val, float):
                cell = f"{val:,.1f}" if abs(val) >= 1 else f"{val:.4f}"
            elif isinstance(val, int):
                cell = f"{val:,}"
            else:
                cell = str(val)
            row += " | " + cell.rjust(col_w)
            row += " |" if l == labels[-1] else ""
        lines.append(row)
    lines.append(sep)
    return "\n".join(lines)


# ── Quantization (numpy-based, for host-side data prep) ───────────

def _vectorized_f32_to_fp8(arr_f32_flat):
    """Vectorized FP8 E4M3 FNUZ conversion using numpy (approximate)."""
    import numpy as np
    result = np.zeros_like(arr_f32_flat, dtype=np.uint8)
    nonzero = arr_f32_flat != 0
    vals = arr_f32_flat[nonzero]
    signs = (vals < 0).astype(np.uint8)
    abs_vals = np.abs(vals)
    max_fp8 = float(math.ldexp(1.0 + 7 * 0.125, 7))
    abs_vals = np.clip(abs_vals, 0, max_fp8)
    log2_v = np.floor(np.log2(np.maximum(abs_vals, 1e-30))).astype(np.int32)
    exp_biased = np.clip(log2_v + 8, 1, 15).astype(np.uint8)
    mantissa_f = abs_vals / np.ldexp(np.ones_like(abs_vals), (exp_biased.astype(np.int32) - 8)) - 1.0
    man = np.clip(np.round(mantissa_f * 8.0), 0, 7).astype(np.uint8)
    result[nonzero] = (signs << 7) | (exp_biased << 3) | man
    return result


def quantize_tensor_fp8(arr_f32, block_size=SCALE_BLOCK):
    """Quantize a 2D float32 array to FP8 E4M3 FNUZ with per-block scales.

    Args:
        arr_f32: numpy array of shape [rows, cols], dtype float32
        block_size: number of elements per scale block along last dim

    Returns:
        fp8_data: numpy uint8 array [rows, cols]
        scales: numpy float32 array [rows, cols // block_size]
    """
    import numpy as np

    rows, cols = arr_f32.shape
    num_blocks = (cols + block_size - 1) // block_size
    max_fp8 = float(math.ldexp(1.0 + 7 * 0.125, 7))

    # Reshape into blocks for vectorized amax
    padded_cols = num_blocks * block_size
    padded = np.zeros((rows, padded_cols), dtype=np.float32)
    padded[:, :cols] = arr_f32
    blocked = padded.reshape(rows, num_blocks, block_size)

    amax = np.max(np.abs(blocked), axis=2)
    scales = np.where(amax == 0, 1.0, amax / max_fp8).astype(np.float32)

    # Scale and convert
    scale_expanded = np.repeat(scales, block_size, axis=1)[:, :cols]
    scaled = arr_f32 / scale_expanded
    fp8_data = _vectorized_f32_to_fp8(scaled.ravel()).reshape(rows, cols)

    return fp8_data, scales


def quantize_tensor_fp8_2d_block(arr_f32, block_k=SCALE_BLOCK, block_n=SCALE_BLOCK):
    """Quantize B matrix with 2D block scales: [K/block_k, N/block_n].

    Returns:
        fp8_data: uint8 [K, N]
        scales: float32 [K//block_k, N//block_n]
    """
    import numpy as np

    K, N = arr_f32.shape
    nk = (K + block_k - 1) // block_k
    nn = (N + block_n - 1) // block_n
    max_fp8 = float(math.ldexp(1.0 + 7 * 0.125, 7))

    # Pad to full blocks
    K_pad = nk * block_k
    N_pad = nn * block_n
    padded = np.zeros((K_pad, N_pad), dtype=np.float32)
    padded[:K, :N] = arr_f32

    # Reshape to [nk, block_k, nn, block_n], compute amax per 2D block
    blocked = padded.reshape(nk, block_k, nn, block_n)
    amax = np.max(np.abs(blocked), axis=(1, 3))
    scales = np.where(amax == 0, 1.0, amax / max_fp8).astype(np.float32)

    # Scale each element by its block's scale
    scale_expanded = np.repeat(np.repeat(scales, block_k, axis=0), block_n, axis=1)[:K, :N]
    scaled = arr_f32 / scale_expanded
    fp8_data = _vectorized_f32_to_fp8(scaled.ravel()).reshape(K, N)

    return fp8_data, scales


def _vectorized_fp8_to_f32(fp8_flat):
    """Vectorized FP8 E4M3 FNUZ to float32 conversion using numpy."""
    import numpy as np
    result = np.zeros(len(fp8_flat), dtype=np.float32)
    vals = fp8_flat.astype(np.uint32)
    nonzero = (vals != 0) & (vals != 0x80)
    v = vals[nonzero]
    signs = ((v >> 7) & 1).astype(np.float32)
    exps = ((v >> 3) & 0xF).astype(np.int32)
    mans = (v & 0x7).astype(np.float32)
    fvals = np.ldexp(1.0 + mans * 0.125, exps - 8)
    fvals = np.where(signs > 0, -fvals, fvals)
    result[nonzero] = fvals
    return result


def dequantize_A(fp8_data, scales, block_size=SCALE_BLOCK):
    """Dequantize A: fp8[M,K] * scales[M, K/block_size] -> float32[M,K]."""
    import numpy as np
    rows, cols = fp8_data.shape
    f32_vals = _vectorized_fp8_to_f32(fp8_data.ravel()).reshape(rows, cols)
    scale_expanded = np.repeat(scales, block_size, axis=1)[:, :cols]
    return f32_vals * scale_expanded


def dequantize_B(fp8_data, scales, block_k=SCALE_BLOCK, block_n=SCALE_BLOCK):
    """Dequantize B: fp8[K,N] * scales[K/block_k, N/block_n] -> float32[K,N]."""
    import numpy as np
    K, N = fp8_data.shape
    f32_vals = _vectorized_fp8_to_f32(fp8_data.ravel()).reshape(K, N)
    scale_expanded = np.repeat(np.repeat(scales, block_k, axis=0), block_n, axis=1)[:K, :N]
    return f32_vals * scale_expanded


# ── Step 1: Compile three versions ─────────────────────────────────

def compile_versions(work_dir: Path, arch: str) -> dict[str, Path]:
    """Compile Version A, B, C .co files."""
    compiler = Compiler(default_arch=arch)
    editor = AsmEditor(arch=arch)
    kb = KnowledgeBase()
    kb.load()
    optimizer = AsmOptimizer(arch=arch, kb=kb)
    replacer = PatternReplacer(kb=kb)

    naive_src = AFTT_ROOT / "templates" / "fp8gemm_blockscale_naive.hip"
    opt_src = AFTT_ROOT / "templates" / "fp8gemm_blockscale_optimized.hip"

    co_a = work_dir / "version_a_naive.co"
    co_b = work_dir / "version_b_optimized.co"
    co_c = work_dir / "version_c_asm_tuned.co"

    # Version A
    print("[1/6] Compiling Version A: fp8gemm_blockscale_naive.hip → naive.co")
    res_a = compiler.compile_to_co(str(naive_src), output_path=str(co_a))
    if not res_a.success:
        print(f"  ERROR: {res_a.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"  → {co_a} ({co_a.stat().st_size:,} bytes)")

    # Version B
    print("[2/6] Compiling Version B: fp8gemm_blockscale_optimized.hip → optimized.co")
    res_b = compiler.compile_to_co(str(opt_src), output_path=str(co_b))
    if not res_b.success:
        print(f"  ERROR: {res_b.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"  → {co_b} ({co_b.stat().st_size:,} bytes)")

    # Version C: disassemble B → optimize → patch
    print("[3/6] Generating Version C: optimized.co → ASM optimize → binary patch")
    kernel_info, instructions = editor.disassemble(str(co_b))
    print(f"  Disassembled {len(instructions)} instructions from Version B")
    print(f"  Kernel: {kernel_info.name}")

    opt_result = optimizer.optimize(instructions, aggressive=False)
    print(f"  AsmOptimizer: {len(opt_result.edits)} edits, "
          f"{len(opt_result.recommendations)} recommendations")

    repl_result = replacer.find_replacements_standalone(instructions, max_level=1)
    print(f"  PatternReplacer: {len(repl_result.applied_edits)} edits (level ≤ 1, safe only)")

    all_edits: dict[int, EditOperation] = {}
    skipped_waitcnt = 0
    for e in opt_result.edits:
        # Skip s_waitcnt relaxation — dangerous for double-buffered shared memory
        if e.new_mnemonic == "s_waitcnt":
            skipped_waitcnt += 1
            continue
        all_edits[e.target_index] = e
    for e in repl_result.applied_edits:
        if e.new_mnemonic == "s_waitcnt":
            skipped_waitcnt += 1
            continue
        if e.target_index not in all_edits:
            all_edits[e.target_index] = e

    edit_list = sorted(all_edits.values(), key=lambda e: e.target_index)
    print(f"  Merged: {len(edit_list)} unique edits to apply "
          f"(skipped {skipped_waitcnt} s_waitcnt edits for safety)")

    if edit_list:
        patch_result = editor.binary_patch(str(co_b), str(co_c), edit_list, instructions)
        print(f"  Binary patch: {patch_result['applied_count']} applied, "
              f"{patch_result['skipped_count']} skipped")
    else:
        shutil.copy2(str(co_b), str(co_c))
        print("  No edits to apply — Version C = Version B (copy)")

    print(f"  → {co_c} ({co_c.stat().st_size:,} bytes)")

    return {"A": co_a, "B": co_b, "C": co_c}


# ── Step 2: Static ASM analysis ───────────────────────────────────

def static_analysis(co_paths: dict[str, Path], arch: str) -> dict:
    """Disassemble and analyse all three versions."""
    editor = AsmEditor(arch=arch)
    estimator = CycleEstimator(arch=arch)

    results = {}
    for label, co in co_paths.items():
        print(f"  Analyzing Version {label}: {co.name}")
        info, instrs = editor.disassemble(str(co))
        cats = count_instruction_categories(instrs)
        asm_lines = editor.get_instruction_lines(instrs)
        cycles = estimator.estimate(asm_lines)
        results[label] = {
            "kernel_name": info.name,
            "instructions": instrs,
            "categories": cats,
            "cycles": cycles,
        }
    return results


# ── Step 3: GPU execution ─────────────────────────────────────────

def gpu_benchmark(co_paths: dict[str, Path],
                  analysis: dict,
                  M: int = 2048,
                  N: int = 2048,
                  K: int = 2048,
                  warmup: int = 10,
                  bench: int = 100) -> dict:
    """Run all three versions on GPU and collect outputs + timing."""
    import torch
    import numpy as np
    from src.kernel_validator import (
        FP8GemmBlockscaleArgs, _get_hip, HIP_SUCCESS)
    import ctypes

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    scale_k = (K + SCALE_BLOCK - 1) // SCALE_BLOCK
    scale_n = (N + SCALE_BLOCK - 1) // SCALE_BLOCK

    # Generate random FP32 data and quantize to FP8
    print("  Preparing FP8 quantized data...")
    A_f32_np = np.random.randn(M, K).astype(np.float32) * 0.1
    B_f32_np = np.random.randn(K, N).astype(np.float32) * 0.1

    A_fp8, scale_A_np = quantize_tensor_fp8(A_f32_np, SCALE_BLOCK)
    B_fp8, scale_B_np = quantize_tensor_fp8_2d_block(B_f32_np, SCALE_BLOCK, SCALE_BLOCK)

    # Compute reference: dequantize → matmul
    A_deq = dequantize_A(A_fp8, scale_A_np, SCALE_BLOCK)
    B_deq = dequantize_B(B_fp8, scale_B_np, SCALE_BLOCK, SCALE_BLOCK)
    ref_output_np = A_deq @ B_deq
    ref_output = torch.from_numpy(ref_output_np)  # keep on CPU
    print(f"  Reference computed: max={ref_output.abs().max().item():.4f}")

    # Move data to GPU
    A_gpu = torch.from_numpy(A_fp8.astype(np.uint8)).to(device)
    B_gpu = torch.from_numpy(B_fp8.astype(np.uint8)).to(device)
    scale_A_gpu = torch.from_numpy(scale_A_np).to(device)
    scale_B_gpu = torch.from_numpy(scale_B_np).to(device)

    # A & B must be contiguous uint8 on device
    assert A_gpu.is_contiguous() and A_gpu.dtype == torch.uint8
    assert B_gpu.is_contiguous() and B_gpu.dtype == torch.uint8

    tile_naive = 16
    tile_opt = 64

    gpu_results = {}
    outputs = {}

    libhip = _get_hip()

    for label in ("A", "B", "C"):
        co_path = str(co_paths[label])
        kernel_name = analysis[label]["kernel_name"]

        if label == "A":
            grid_x = (N + tile_naive - 1) // tile_naive
            grid_y = (M + tile_naive - 1) // tile_naive
            block_x, block_y = tile_naive, tile_naive
        else:
            grid_x = (N + tile_opt - 1) // tile_opt
            grid_y = (M + tile_opt - 1) // tile_opt
            block_x, block_y = 16, 16

        out = torch.zeros(M, N, dtype=torch.float32, device=device)

        args = FP8GemmBlockscaleArgs()
        args.output = out.data_ptr()
        args.A = A_gpu.data_ptr()
        args.B = B_gpu.data_ptr()
        args.scale_A = scale_A_gpu.data_ptr()
        args.scale_B = scale_B_gpu.data_ptr()
        args.M = M
        args.N = N
        args.K = K
        args.scale_k = scale_k
        args.scale_n = scale_n

        # Load module
        module = ctypes.c_void_p()
        with open(co_path, "rb") as f:
            co_data = f.read()
        co_buf = ctypes.create_string_buffer(co_data)
        ret = libhip.hipModuleLoadData(ctypes.byref(module), co_buf)
        if ret != HIP_SUCCESS:
            print(f"  ERROR: hipModuleLoadData failed for Version {label}: {ret}")
            gpu_results[label] = {"error": f"hipModuleLoadData failed: {ret}"}
            continue

        func = ctypes.c_void_p()
        ret = libhip.hipModuleGetFunction(
            ctypes.byref(func), module, kernel_name.encode("utf-8"))
        if ret != HIP_SUCCESS:
            libhip.hipModuleUnload(module)
            print(f"  ERROR: hipModuleGetFunction('{kernel_name}') failed for "
                  f"Version {label}: {ret}")
            gpu_results[label] = {"error": f"hipModuleGetFunction failed: {ret}"}
            continue

        HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
        HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
        HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)

        arg_size = ctypes.c_size_t(ctypes.sizeof(args))
        arg_ptr = ctypes.cast(ctypes.pointer(args), ctypes.c_void_p)

        extra = (ctypes.c_void_p * 5)(
            HIP_LAUNCH_PARAM_BUFFER_POINTER, arg_ptr,
            HIP_LAUNCH_PARAM_BUFFER_SIZE,
            ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
            HIP_LAUNCH_PARAM_END)

        stream = ctypes.c_void_p(0)

        def _sync_check() -> int:
            """Synchronize and return HIP error code."""
            return libhip.hipDeviceSynchronize()

        try:
            # Warmup
            for _ in range(warmup):
                libhip.hipModuleLaunchKernel(
                    func, grid_x, grid_y, 1, block_x, block_y, 1,
                    0, stream, None, extra)
            sync_ret = _sync_check()
            if sync_ret != HIP_SUCCESS:
                print(f"  ERROR: Version {label} warmup sync failed: {sync_ret}")
                gpu_results[label] = {"error": f"kernel crash (hipErr={sync_ret})"}
                libhip.hipModuleUnload(module)
                continue

            # Capture output
            out.zero_()
            args.output = out.data_ptr()
            arg_ptr = ctypes.cast(ctypes.pointer(args), ctypes.c_void_p)
            extra = (ctypes.c_void_p * 5)(
                HIP_LAUNCH_PARAM_BUFFER_POINTER, arg_ptr,
                HIP_LAUNCH_PARAM_BUFFER_SIZE,
                ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
                HIP_LAUNCH_PARAM_END)

            libhip.hipModuleLaunchKernel(
                func, grid_x, grid_y, 1, block_x, block_y, 1,
                0, stream, None, extra)
            sync_ret = _sync_check()
            if sync_ret != HIP_SUCCESS:
                print(f"  ERROR: Version {label} capture sync failed: {sync_ret}")
                gpu_results[label] = {"error": f"kernel crash (hipErr={sync_ret})"}
                libhip.hipModuleUnload(module)
                continue

            outputs[label] = out.cpu().clone()

            # Benchmark
            start_ev = ctypes.c_void_p()
            stop_ev = ctypes.c_void_p()
            libhip.hipEventCreate(ctypes.byref(start_ev))
            libhip.hipEventCreate(ctypes.byref(stop_ev))

            libhip.hipEventRecord(start_ev, stream)
            for _ in range(bench):
                libhip.hipModuleLaunchKernel(
                    func, grid_x, grid_y, 1, block_x, block_y, 1,
                    0, stream, None, extra)
            libhip.hipEventRecord(stop_ev, stream)
            libhip.hipEventSynchronize(stop_ev)

            elapsed_ms = ctypes.c_float()
            libhip.hipEventElapsedTime(
                ctypes.byref(elapsed_ms), start_ev, stop_ev)
            avg_us = (elapsed_ms.value * 1000.0) / bench

            libhip.hipEventDestroy(start_ev)
            libhip.hipEventDestroy(stop_ev)
            libhip.hipModuleUnload(module)

            flops = 2.0 * M * N * K
            tflops = flops / (avg_us * 1e-6) / 1e12

            gpu_results[label] = {
                "time_us": avg_us,
                "tflops": tflops,
            }
            print(f"  Version {label}: {avg_us:.2f} us, {tflops:.3f} TFLOPS")

        except Exception as exc:
            print(f"  ERROR: Version {label} exception: {exc}")
            gpu_results[label] = {"error": str(exc)}

    # Correctness (all on CPU to avoid device-state issues)
    ref_cpu = ref_output
    for label in ("A", "B", "C"):
        if label not in outputs:
            continue
        try:
            out_cpu = outputs[label]  # already on CPU
            diff = (out_cpu - ref_cpu).abs()
            max_abs = diff.max().item()
            denom = ref_cpu.abs().clamp(min=1e-8)
            max_rel = (diff / denom).max().item()
            passed = torch.allclose(out_cpu, ref_cpu, rtol=1e-2, atol=1e-2)
            gpu_results[label]["vs_ref_max_abs"] = max_abs
            gpu_results[label]["vs_ref_max_rel"] = max_rel
            gpu_results[label]["correctness"] = "PASS" if passed else "FAIL"
        except Exception as exc:
            print(f"  Correctness check failed for {label}: {exc}")

    ref_out = outputs.get("A")
    for label in ("B", "C"):
        if label not in outputs or ref_out is None:
            continue
        try:
            diff = (outputs[label] - ref_out).abs()
            max_abs_vs_a = diff.max().item()
            gpu_results[label]["vs_A_max_abs"] = max_abs_vs_a
        except Exception:
            pass

    return gpu_results


# ── Step 4: Report ─────────────────────────────────────────────────

def print_report(analysis: dict, gpu_results: dict | None, M: int, N: int, K: int):
    """Print the final comparison report."""
    print("\n" + "=" * 72)
    print("  FP8 BlockScale GEMM A/B/C Verification Report")
    print(f"  M={M}, N={N}, K={K} (scale_block={SCALE_BLOCK})")
    print("=" * 72)

    # Static analysis table
    static_table = {}
    for label in ("A", "B", "C"):
        entry = analysis[label]
        cats = entry["categories"]
        cyc = entry["cycles"]
        desc = {"A": "(naive)", "B": "(C++ opt)", "C": "(+ASM)"}[label]
        col_name = f"Version {label} {desc}"
        static_table[col_name] = {
            "Kernel": entry["kernel_name"][:30],
            "Total instr": cats["total"],
            "VALU": cats["VALU"],
            "SALU": cats["SALU"],
            "VMEM load": cats["VMEM_load"],
            "VMEM store": cats["VMEM_store"],
            "LDS": cats["LDS"],
            "DPP": cats["DPP"],
            "MFMA": cats["MFMA"],
            "NOP": cats["NOP"],
            "Waitcnt": cats["Waitcnt"],
            "Barrier": cats["Barrier"],
            "Branch": cats["Branch"],
            "Est. cycles": cyc.total_cycles,
            "Bottleneck": cyc.bottleneck[:20],
        }

    print("\n── Static ASM Analysis ──")
    print(format_table(static_table))

    # Cycle breakdown
    print("\n── Cycle Breakdown ──")
    cyc_table = {}
    for label in ("A", "B", "C"):
        cyc = analysis[label]["cycles"]
        desc = {"A": "(naive)", "B": "(C++ opt)", "C": "(+ASM)"}[label]
        cyc_table[f"V{label} {desc}"] = {
            "Total": cyc.total_cycles,
            "MFMA": cyc.mfma_cycles,
            "VALU": cyc.valu_cycles,
            "VMEM": cyc.vmem_cycles,
            "LDS": cyc.lds_cycles,
            "SALU": cyc.salu_cycles,
            "Wait stall": cyc.wait_stall_cycles,
            "Barrier stall": cyc.barrier_stall_cycles,
            "NOP": cyc.nop_cycles,
        }
    print(format_table(cyc_table))

    cycles_a = analysis["A"]["cycles"].total_cycles
    cycles_b = analysis["B"]["cycles"].total_cycles
    cycles_c = analysis["C"]["cycles"].total_cycles
    print(f"\nStatic speedup estimate:")
    print(f"  B vs A: {cycles_a / max(cycles_b, 1):.2f}x")
    print(f"  C vs A: {cycles_a / max(cycles_c, 1):.2f}x")
    print(f"  C vs B: {cycles_b / max(cycles_c, 1):.2f}x")

    if gpu_results is None:
        print("\n(GPU benchmark skipped — use without --no-gpu for full results)")
        return

    # GPU results table
    print("\n── GPU Execution Results ──")
    gpu_table = {}
    time_a = gpu_results.get("A", {}).get("time_us", 0)
    for label in ("A", "B", "C"):
        r = gpu_results.get(label, {})
        if "error" in r:
            desc = {"A": "(naive)", "B": "(C++ opt)", "C": "(+ASM)"}[label]
            gpu_table[f"V{label} {desc}"] = {"Status": f"ERROR: {r['error']}"}
            continue
        desc = {"A": "(naive)", "B": "(C++ opt)", "C": "(+ASM)"}[label]
        col = {
            "GPU time (us)": r.get("time_us", 0),
            "TFLOPS": r.get("tflops", 0),
        }
        if label == "A":
            col["vs A speedup"] = "1.00x (ref)"
        else:
            speedup = time_a / max(r.get("time_us", 1e-9), 1e-9) if time_a else 0
            col["vs A speedup"] = f"{speedup:.3f}x"
            col["vs A max_abs"] = r.get("vs_A_max_abs", "-")

        col["vs ref max_abs"] = r.get("vs_ref_max_abs", "-")
        col["vs ref max_rel"] = r.get("vs_ref_max_rel", "-")
        col["Correctness"] = r.get("correctness", "N/A")
        gpu_table[f"V{label} {desc}"] = col

    print(format_table(gpu_table))

    # Final verdict
    print("\n── Verdict ──")
    for label in ("A", "B", "C"):
        r = gpu_results.get(label, {})
        status = r.get("correctness", "N/A")
        if status == "PASS":
            if label == "A":
                print(f"  Version A: PASS (correctness OK vs dequant reference)")
            else:
                speedup = time_a / max(r.get("time_us", 1e-9), 1e-9) if time_a else 0
                print(f"  Version {label}: PASS (correctness OK, {speedup:.3f}x vs naive)")
        elif status == "FAIL":
            print(f"  Version {label}: FAIL (output differs from reference)")
        else:
            print(f"  Version {label}: {status}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FP8 BlockScale GEMM A/B/C three-version verification tool")
    parser.add_argument("--arch", default="gfx942",
                        help="Target GPU architecture (default: gfx942)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Skip GPU execution (static analysis only)")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Working directory for .co files (default: temp dir)")
    parser.add_argument("-M", type=int, default=2048,
                        help="M dimension (default: 2048)")
    parser.add_argument("-N", type=int, default=2048,
                        help="N dimension (default: 2048)")
    parser.add_argument("-K", type=int, default=2048,
                        help="K dimension (must be multiple of 128, default: 2048)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--bench", type=int, default=100,
                        help="Benchmark iterations (default: 100)")
    parser.add_argument("--json", type=str, default=None,
                        help="Write results to JSON file")
    args = parser.parse_args()

    if args.K % SCALE_BLOCK != 0:
        print(f"ERROR: K ({args.K}) must be a multiple of {SCALE_BLOCK}", file=sys.stderr)
        sys.exit(1)

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="aftt_fp8gemm_"))
        cleanup = True

    print(f"Work directory: {work_dir}")
    print(f"Architecture:   {args.arch}")
    print(f"Dimensions:     M={args.M}, N={args.N}, K={args.K}")
    print(f"Scale block:    {SCALE_BLOCK}")
    print()

    try:
        # Step 1: Compile
        co_paths = compile_versions(work_dir, args.arch)
        print()

        # Step 2: Static analysis
        print("[4/6] Static ASM analysis")
        analysis = static_analysis(co_paths, args.arch)
        print()

        # Step 3: GPU execution
        gpu_results = None
        if not args.no_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[5/6] GPU not available — skipping GPU benchmark")
                else:
                    print(f"[5/6] GPU benchmark (warmup={args.warmup}, "
                          f"bench={args.bench}, M={args.M}, N={args.N}, K={args.K})")
                    gpu_results = gpu_benchmark(
                        co_paths, analysis,
                        M=args.M, N=args.N, K=args.K,
                        warmup=args.warmup,
                        bench=args.bench)
            except ImportError:
                print("[5/6] torch not available — skipping GPU benchmark")
        else:
            print("[5/6] GPU benchmark skipped (--no-gpu)")

        # Step 4: Report
        print("\n[6/6] Generating comparison report")
        print_report(analysis, gpu_results, args.M, args.N, args.K)

        # Optional JSON output
        if args.json:
            json_data = {
                "arch": args.arch,
                "M": args.M, "N": args.N, "K": args.K,
                "scale_block": SCALE_BLOCK,
                "versions": {},
            }
            for label in ("A", "B", "C"):
                entry = {
                    "kernel_name": analysis[label]["kernel_name"],
                    "instruction_categories": analysis[label]["categories"],
                    "cycle_estimate": analysis[label]["cycles"].to_dict(),
                }
                if gpu_results and label in gpu_results:
                    entry["gpu"] = gpu_results[label]
                json_data["versions"][label] = entry

            with open(args.json, "w") as f:
                json.dump(json_data, f, indent=2, default=str)
            print(f"\nResults written to {args.json}")

    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
