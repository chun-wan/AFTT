#!/usr/bin/env python3
"""RMSNorm A/B/C Three-Version Verification Tool.

Compiles and compares three versions of RMSNorm:
  Version A: rmsnorm_naive.hip compiled directly (no AFTT)
  Version B: rmsnorm_optimized.hip (AFTT CppTemplateEngine swap) compiled
  Version C: Version B .co + AFTT AsmOptimizer + PatternReplacer binary patch

Outputs: static ASM analysis comparison + GPU execution benchmark + correctness.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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
    """Format the comparison table."""
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


# ── Step 1: Compile three versions ─────────────────────────────────

def compile_versions(work_dir: Path, arch: str) -> dict[str, Path]:
    """Compile Version A, B, C .co files."""
    compiler = Compiler(default_arch=arch)
    editor = AsmEditor(arch=arch)
    kb = KnowledgeBase()
    kb.load()
    optimizer = AsmOptimizer(arch=arch, kb=kb)
    replacer = PatternReplacer(kb=kb)

    naive_src = AFTT_ROOT / "templates" / "rmsnorm_naive.hip"
    opt_src = AFTT_ROOT / "templates" / "rmsnorm_optimized.hip"

    co_a = work_dir / "version_a_naive.co"
    co_b = work_dir / "version_b_optimized.co"
    co_c = work_dir / "version_c_asm_tuned.co"

    # Version A
    print("[1/6] Compiling Version A: rmsnorm_naive.hip → naive.co")
    res_a = compiler.compile_to_co(str(naive_src), output_path=str(co_a))
    if not res_a.success:
        print(f"  ERROR: {res_a.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"  → {co_a} ({co_a.stat().st_size:,} bytes)")

    # Version B
    print("[2/6] Compiling Version B: rmsnorm_optimized.hip → optimized.co")
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

    repl_result = replacer.find_replacements_standalone(instructions, max_level=2)
    print(f"  PatternReplacer: {len(repl_result.applied_edits)} edits (level ≤ 2)")

    # Merge edits, deduplicate by target_index
    all_edits: dict[int, EditOperation] = {}
    for e in opt_result.edits:
        all_edits[e.target_index] = e
    for e in repl_result.applied_edits:
        if e.target_index not in all_edits:
            all_edits[e.target_index] = e

    edit_list = sorted(all_edits.values(), key=lambda e: e.target_index)
    print(f"  Merged: {len(edit_list)} unique edits to apply")

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
                  num_tokens: int = 128,
                  hidden_size: int = 8192,
                  epsilon: float = 1e-6,
                  warmup: int = 10,
                  bench: int = 100) -> dict:
    """Run all three versions on GPU and collect outputs + timing."""
    import torch
    from src.kernel_validator import (
        KernelValidator, RMSNormKernelArgs, _get_hip, HIP_SUCCESS)
    import ctypes

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    inp = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float32, device=device)

    # Torch reference
    rms = torch.sqrt(inp.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    ref_output = inp / rms * weight

    block_sizes = {"A": 256, "B": 1024, "C": 1024}

    gpu_results = {}
    outputs = {}

    libhip = _get_hip()

    for label in ("A", "B", "C"):
        co_path = str(co_paths[label])
        kernel_name = analysis[label]["kernel_name"]
        block_size = block_sizes[label]

        out = torch.zeros_like(inp)

        # Build args
        args = RMSNormKernelArgs()
        args.output = out.data_ptr()
        args.input = inp.data_ptr()
        args.weight = weight.data_ptr()
        args.hidden_size = hidden_size
        args.epsilon = epsilon

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

        # Launch params
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

        # Warmup
        for _ in range(warmup):
            libhip.hipModuleLaunchKernel(
                func, num_tokens, 1, 1, block_size, 1, 1, 0, stream, None, extra)
        libhip.hipDeviceSynchronize()

        # Capture output (re-zero and run once)
        out.zero_()
        args.output = out.data_ptr()
        arg_ptr = ctypes.cast(ctypes.pointer(args), ctypes.c_void_p)
        extra = (ctypes.c_void_p * 5)(
            HIP_LAUNCH_PARAM_BUFFER_POINTER, arg_ptr,
            HIP_LAUNCH_PARAM_BUFFER_SIZE,
            ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
            HIP_LAUNCH_PARAM_END)

        libhip.hipModuleLaunchKernel(
            func, num_tokens, 1, 1, block_size, 1, 1, 0, stream, None, extra)
        libhip.hipDeviceSynchronize()
        outputs[label] = out.clone()

        # Benchmark
        start_ev = ctypes.c_void_p()
        stop_ev = ctypes.c_void_p()
        libhip.hipEventCreate(ctypes.byref(start_ev))
        libhip.hipEventCreate(ctypes.byref(stop_ev))

        libhip.hipEventRecord(start_ev, stream)
        for _ in range(bench):
            libhip.hipModuleLaunchKernel(
                func, num_tokens, 1, 1, block_size, 1, 1, 0, stream, None, extra)
        libhip.hipEventRecord(stop_ev, stream)
        libhip.hipEventSynchronize(stop_ev)

        elapsed_ms = ctypes.c_float()
        libhip.hipEventElapsedTime(ctypes.byref(elapsed_ms), start_ev, stop_ev)
        avg_us = (elapsed_ms.value * 1000.0) / bench

        libhip.hipEventDestroy(start_ev)
        libhip.hipEventDestroy(stop_ev)
        libhip.hipModuleUnload(module)

        # Bandwidth calculation
        total_bytes = num_tokens * hidden_size * 4 * 3 + hidden_size * 4
        bw_gbps = total_bytes / (avg_us * 1e-6) / 1e9

        gpu_results[label] = {
            "time_us": avg_us,
            "bandwidth_gbps": bw_gbps,
        }
        print(f"  Version {label}: {avg_us:.2f} us, {bw_gbps:.1f} GB/s")

    # Correctness comparison
    ref_out = outputs.get("A")
    for label in ("B", "C"):
        if label not in outputs or ref_out is None:
            continue
        diff = (outputs[label] - ref_out).abs()
        max_abs = diff.max().item()
        denom = ref_out.abs().clamp(min=1e-8)
        max_rel = (diff / denom).max().item()
        passed = torch.allclose(outputs[label], ref_out, rtol=1e-3, atol=1e-4)
        gpu_results[label]["max_abs_error"] = max_abs
        gpu_results[label]["max_rel_error"] = max_rel
        gpu_results[label]["correctness"] = "PASS" if passed else "FAIL"

    # Version A vs torch reference
    if ref_out is not None:
        a_vs_torch = (ref_out - ref_output).abs().max().item()
        gpu_results["A"]["vs_torch_max_error"] = a_vs_torch

    return gpu_results


# ── Step 4: Report ─────────────────────────────────────────────────

def print_report(analysis: dict, gpu_results: dict | None):
    """Print the final comparison report."""
    print("\n" + "=" * 72)
    print("  RMSNorm A/B/C Verification Report")
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

    # Speedup estimates from static analysis
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
            "Bandwidth (GB/s)": r.get("bandwidth_gbps", 0),
        }
        if label == "A":
            col["vs A speedup"] = "1.00x (ref)"
            col["Correctness"] = "REF"
        else:
            speedup = time_a / max(r.get("time_us", 1e-9), 1e-9) if time_a else 0
            col["vs A speedup"] = f"{speedup:.3f}x"
            col["Max abs error"] = r.get("max_abs_error", "-")
            col["Max rel error"] = r.get("max_rel_error", "-")
            col["Correctness"] = r.get("correctness", "N/A")
        gpu_table[f"V{label} {desc}"] = col

    print(format_table(gpu_table))

    # Final verdict
    print("\n── Verdict ──")
    for label in ("B", "C"):
        r = gpu_results.get(label, {})
        status = r.get("correctness", "N/A")
        if status == "PASS":
            speedup = time_a / max(r.get("time_us", 1e-9), 1e-9) if time_a else 0
            print(f"  Version {label}: PASS (correctness OK, {speedup:.3f}x vs naive)")
        elif status == "FAIL":
            print(f"  Version {label}: FAIL (output differs from Version A)")
        else:
            print(f"  Version {label}: {status}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RMSNorm A/B/C three-version verification tool")
    parser.add_argument("--arch", default="gfx942",
                        help="Target GPU architecture (default: gfx942)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Skip GPU execution (static analysis only)")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Working directory for .co files (default: temp dir)")
    parser.add_argument("--num-tokens", type=int, default=128,
                        help="Number of tokens for benchmark (default: 128)")
    parser.add_argument("--hidden-size", type=int, default=8192,
                        help="Hidden size for benchmark (default: 8192)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--bench", type=int, default=100,
                        help="Benchmark iterations (default: 100)")
    parser.add_argument("--json", type=str, default=None,
                        help="Write results to JSON file")
    args = parser.parse_args()

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="aftt_rmsnorm_"))
        cleanup = True

    print(f"Work directory: {work_dir}")
    print(f"Architecture:   {args.arch}")
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
                          f"bench={args.bench}, tokens={args.num_tokens}, "
                          f"hidden={args.hidden_size})")
                    gpu_results = gpu_benchmark(
                        co_paths, analysis,
                        num_tokens=args.num_tokens,
                        hidden_size=args.hidden_size,
                        warmup=args.warmup,
                        bench=args.bench)
            except ImportError:
                print("[5/6] torch not available — skipping GPU benchmark")
        else:
            print("[5/6] GPU benchmark skipped (--no-gpu)")

        # Step 4: Report
        print("\n[6/6] Generating comparison report")
        print_report(analysis, gpu_results)

        # Optional JSON output
        if args.json:
            json_data = {
                "arch": args.arch,
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
