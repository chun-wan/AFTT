#!/usr/bin/env python3
"""Phase 1B: GPU Benchmark Harness.

Swaps patched .co files into aiter's hsa directory, runs the appropriate
aiter benchmark script, collects timing and correctness data, then restores
the original .co.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

AFTT_ROOT = Path(__file__).resolve().parent.parent
AITER_ROOT = Path("/home/root123/aiter")

# Mapping: aiter category → (benchmark script relative to aiter root, extra_args)
BENCH_SCRIPTS = {
    "bf16gemm":          ("op_tests/op_benchmarks/triton/bench_gemm_a16w16.py", ["-M", "256"]),
    "fp8gemm_blockscale":("op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py", []),
    "i8gemm":            ("op_tests/op_benchmarks/triton/bench_gemm_a8w8.py", []),
    "f4gemm":            ("op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py", []),
    "fmha_v3_fwd":       ("op_tests/op_benchmarks/triton/bench_mha.py", []),
    "fmha_v3_bwd":       ("op_tests/op_benchmarks/triton/bench_mha.py", []),
    "fmoe":              ("op_tests/op_benchmarks/triton/bench_moe.py", []),
    "fmoe_2stages":      ("op_tests/op_benchmarks/triton/bench_moe.py", []),
    "pa":                ("op_tests/op_benchmarks/triton/bench_pa_decode.py", []),
    "mla":               ("op_tests/op_benchmarks/triton/bench_mla_decode.py", []),
    "topksoftmax":       ("op_tests/op_benchmarks/triton/bench_topk.py", []),
}


@dataclass
class BenchResult:
    co_name: str
    category: str
    strategy: str
    original_co_path: str
    patched_co_path: str
    bench_script: str = ""
    original_time_us: float = 0.0
    patched_time_us: float = 0.0
    speedup: float = 1.0
    correctness: str = "not_checked"
    error: str = ""
    raw_output_original: str = ""
    raw_output_patched: str = ""


def find_bench_script(category: str) -> Optional[tuple[str, list[str]]]:
    """Find the aiter benchmark script and extra args for a category."""
    entry = BENCH_SCRIPTS.get(category)
    if entry:
        script, args = entry
        full = AITER_ROOT / script
        if full.exists():
            return str(full), args
    return None


def parse_timing_from_output(output: str) -> Optional[float]:
    """Extract kernel timing from aiter benchmark output.

    Aiter bench scripts output pandas DataFrames with columns like:
      'Time_(ms)' or 'Time (ms)' for latency
      'Bandwidth_(GB/s)' or 'TFLOPS' for throughput
    We extract the first numeric time value and convert to us.
    """
    # Try Time_(ms) column values (aiter GEMM/MoE/PA bench format)
    # Lines look like: "0  256.0  1280.0  8192.0               0.120332"
    # After header with Time_(ms)
    lines = output.split("\n")
    in_time_table = False
    time_col_idx = -1

    for line in lines:
        stripped = line.strip()
        # Detect header with time column
        if "Time_(ms)" in stripped or "Time (ms)" in stripped:
            parts = re.split(r'\s{2,}', stripped)
            for ci, col in enumerate(parts):
                if "Time" in col and "ms" in col.lower():
                    time_col_idx = ci
                    break
            in_time_table = True
            continue

        if in_time_table and stripped and stripped[0].isdigit():
            parts = stripped.split()
            # The last numeric value is often the time
            for val in reversed(parts):
                try:
                    t_ms = float(val)
                    if 0 < t_ms < 100000:
                        return t_ms * 1000  # ms → us
                except ValueError:
                    continue

    # Try Bandwidth/TFLOPS format (no direct time)
    bw_patterns = [
        r"Bandwidth_\(GB/s\).*?(\d+\.\d+)",
        r"TFLOPS.*?(\d+\.\d+)",
    ]
    for pat in bw_patterns:
        m = re.search(pat, output, re.DOTALL)
        if m:
            return None  # Bandwidth-only output, no time

    # Fallback: any "time: X.X us/ms" pattern
    fallback_patterns = [
        r"(?:kernel|avg|mean|median)\s*(?:time)?[:\s]*([0-9.]+)\s*(?:us|μs)",
        r"time\s*\(?us\)?[:\s]*([0-9.]+)",
        r"([0-9.]+)\s*us\s*(?:per|/)\s*(?:kernel|iter)",
        r"latency[:\s]*([0-9.]+)\s*(?:us|μs)",
    ]
    for pat in fallback_patterns:
        m = re.search(pat, output, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def run_benchmark_script(script_path: str, timeout: int = 120,
                         extra_args: Optional[list[str]] = None) -> tuple[str, float]:
    """Run an aiter benchmark script and return (raw_output, time_us)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(AITER_ROOT) + ":" + env.get("PYTHONPATH", "")

    cmd = [sys.executable, script_path, "--metric", "time"]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(AITER_ROOT), env=env)
        output = result.stdout + "\n" + result.stderr

        # If --metric time is not supported, retry without it
        if result.returncode != 0 and "--metric" in (result.stderr or ""):
            cmd = [sys.executable, script_path]
            if extra_args:
                cmd.extend(extra_args)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd=str(AITER_ROOT), env=env)
            output = result.stdout + "\n" + result.stderr

        timing = parse_timing_from_output(output)
        return output, timing or 0.0
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 0.0
    except Exception as exc:
        return str(exc), 0.0


def swap_and_bench(original_co: str, patched_co: str, category: str,
                   strategy_name: str, bench_timeout: int = 120) -> BenchResult:
    """Swap a patched .co into place, benchmark, then restore original."""
    result = BenchResult(
        co_name=Path(original_co).name,
        category=category,
        strategy=strategy_name,
        original_co_path=original_co,
        patched_co_path=patched_co,
    )

    bench_info = find_bench_script(category)
    if not bench_info:
        result.error = f"no benchmark script for category '{category}'"
        return result
    bench_script, bench_extra_args = bench_info
    result.bench_script = bench_script

    if not os.path.exists(patched_co):
        result.error = f"patched .co not found: {patched_co}"
        return result

    backup_path = original_co + ".aftt_backup"

    try:
        # Benchmark original
        raw_orig, time_orig = run_benchmark_script(bench_script, bench_timeout, bench_extra_args)
        result.raw_output_original = raw_orig[:2000]
        result.original_time_us = time_orig

        # Swap in patched
        shutil.copy2(original_co, backup_path)
        shutil.copy2(patched_co, original_co)

        # Benchmark patched
        raw_patched, time_patched = run_benchmark_script(bench_script, bench_timeout, bench_extra_args)
        result.raw_output_patched = raw_patched[:2000]
        result.patched_time_us = time_patched

        if time_orig > 0 and time_patched > 0:
            result.speedup = time_orig / time_patched

    except Exception as exc:
        result.error = str(exc)
    finally:
        # Restore original
        if os.path.exists(backup_path):
            shutil.move(backup_path, original_co)

    return result


def run_gpu_harness(static_results_dir: Path, output_dir: Path,
                    arch: str = "gfx942",
                    strategy: str = "nop_waitcnt",
                    categories: Optional[list[str]] = None,
                    max_kernels: Optional[int] = None,
                    bench_timeout: int = 120):
    """Run GPU benchmarks for all kernels that have patched .co files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load static evaluation results
    result_files = sorted(static_results_dir.glob("*.json"))
    if not result_files:
        print(f"No static results found in {static_results_dir}")
        return

    kernels_to_bench = []
    for rf in result_files:
        if rf.name == "summary.json":
            continue
        try:
            with open(rf) as f:
                data = json.load(f)
        except Exception:
            continue

        if not data.get("disasm_ok"):
            continue
        if categories and data.get("category") not in categories:
            continue

        strat_data = data.get("strategies", {}).get(strategy, {})
        if not strat_data.get("patch_ok"):
            continue
        if strat_data.get("num_applied", 0) == 0:
            continue

        kernels_to_bench.append({
            "co_path": data["co_path"],
            "co_name": data["co_name"],
            "category": data["category"],
            "patched_co_path": strat_data["patched_co_path"],
            "static_improvement_pct": strat_data.get("improvement_pct", 0),
        })

    if max_kernels:
        kernels_to_bench = kernels_to_bench[:max_kernels]

    print(f"GPU benchmark: {len(kernels_to_bench)} kernels with strategy '{strategy}'")

    all_results = []
    for i, entry in enumerate(kernels_to_bench):
        print(f"  [{i+1}/{len(kernels_to_bench)}] {entry['co_name']} ({entry['category']})")

        br = swap_and_bench(
            entry["co_path"], entry["patched_co_path"],
            entry["category"], strategy, bench_timeout)
        all_results.append(asdict(br))

        # Save per-kernel
        result_path = output_dir / f"{Path(entry['co_name']).stem}_{strategy}.json"
        with open(result_path, "w") as f:
            json.dump(asdict(br), f, indent=2)

    # Summary
    summary = {
        "strategy": strategy,
        "total_benchmarked": len(all_results),
        "improved": sum(1 for r in all_results if r.get("speedup", 1) > 1.005),
        "regressed": sum(1 for r in all_results if r.get("speedup", 1) < 0.995),
        "neutral": sum(1 for r in all_results if 0.995 <= r.get("speedup", 1) <= 1.005),
        "errors": sum(1 for r in all_results if r.get("error")),
    }
    if all_results:
        speedups = [r["speedup"] for r in all_results if r.get("speedup", 0) > 0]
        if speedups:
            summary["avg_speedup"] = sum(speedups) / len(speedups)
            summary["max_speedup"] = max(speedups)
            summary["min_speedup"] = min(speedups)

    summary_path = output_dir / f"gpu_summary_{strategy}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nGPU Benchmark Summary ({strategy}):")
    print(f"  Benchmarked: {summary['total_benchmarked']}")
    print(f"  Improved:    {summary['improved']}")
    print(f"  Regressed:   {summary['regressed']}")
    print(f"  Neutral:     {summary['neutral']}")
    print(f"  Errors:      {summary['errors']}")
    if "avg_speedup" in summary:
        print(f"  Avg speedup: {summary['avg_speedup']:.4f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1B: GPU benchmark harness for AFTT-patched kernels")
    parser.add_argument("--static-results",
                        default=str(AFTT_ROOT / "results" / "phase1_static" / "gfx942"),
                        help="Directory containing Phase 1A static results")
    parser.add_argument("--output",
                        default=str(AFTT_ROOT / "results" / "phase1_gpu"),
                        help="Output directory for GPU benchmark results")
    parser.add_argument("--arch", default="gfx942")
    parser.add_argument("--strategy", default="nop_waitcnt",
                        choices=["nop_only", "nop_waitcnt", "full_level4"],
                        help="Which edit strategy to benchmark")
    parser.add_argument("--categories", nargs="*",
                        help="Filter categories")
    parser.add_argument("--max-kernels", type=int, default=None)
    parser.add_argument("--bench-timeout", type=int, default=120)
    args = parser.parse_args()

    run_gpu_harness(
        Path(args.static_results), Path(args.output),
        arch=args.arch, strategy=args.strategy,
        categories=args.categories,
        max_kernels=args.max_kernels,
        bench_timeout=args.bench_timeout)


if __name__ == "__main__":
    main()
