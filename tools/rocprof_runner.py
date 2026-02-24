#!/usr/bin/env python3
"""Phase 1C: rocprof Hardware Counter Profiling.

Collects hardware performance counters via rocprofv3 for original
and AFTT-patched .co kernel pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

AFTT_ROOT = Path(__file__).resolve().parent.parent
AITER_ROOT = Path("/home/root123/aiter")
ROCPROF = os.environ.get("ROCPROF", "/opt/rocm/bin/rocprofv3")

# Hardware counters to collect, grouped to avoid per-pass limits
COUNTER_GROUPS = [
    # Group 1: instruction mix
    ["SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_LDS", "SQ_INSTS_VMEM_RD",
     "SQ_INSTS_VMEM_WR"],
    # Group 2: stalls and waves
    ["SQ_WAVES", "SQ_WAIT_INST_ANY", "SQ_ACTIVE_INST_ANY"],
    # Group 3: cache
    ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_READ_sum", "TCC_WRITE_sum"],
]


@dataclass
class ProfileResult:
    co_name: str
    category: str
    strategy: str
    counters_original: dict
    counters_patched: dict
    counter_deltas: dict
    error: str = ""


def write_rocprof_input(counters: list[str], output_path: str):
    """Write a rocprofv3-compatible counter input file."""
    with open(output_path, "w") as f:
        f.write("pmc: " + " ".join(counters) + "\n")


def run_rocprof(bench_script: str, counter_input: str,
                output_csv: str, timeout: int = 180) -> Optional[str]:
    """Run rocprofv3 with a benchmark script and collect counters."""
    cmd = [
        ROCPROF,
        "-i", counter_input,
        "-o", output_csv,
        sys.executable, bench_script,
        "--warmup", "2", "--iters", "10",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(AITER_ROOT) + ":" + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(AITER_ROOT), env=env)
        if result.returncode != 0:
            return f"rocprof failed: {result.stderr[:500]}"
        return None
    except subprocess.TimeoutExpired:
        return "rocprof timeout"
    except FileNotFoundError:
        return f"rocprof not found at {ROCPROF}"


def parse_rocprof_csv(csv_path: str) -> dict:
    """Parse rocprofv3 output CSV into a counter dict."""
    counters = {}
    if not os.path.exists(csv_path):
        return counters

    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, val in row.items():
                    key = key.strip()
                    if key in ("Index", "KernelName", "gpu-id", "queue-id",
                               "process-id", "thread-id"):
                        continue
                    try:
                        fval = float(val)
                        if key in counters:
                            counters[key] = max(counters[key], fval)
                        else:
                            counters[key] = fval
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass

    return counters


def profile_kernel_pair(original_co: str, patched_co: str,
                        category: str, strategy_name: str,
                        bench_script: str,
                        timeout: int = 180) -> ProfileResult:
    """Profile original and patched .co with rocprof."""
    result = ProfileResult(
        co_name=Path(original_co).name,
        category=category,
        strategy=strategy_name,
        counters_original={},
        counters_patched={},
        counter_deltas={},
    )

    if not os.path.exists(ROCPROF):
        result.error = f"rocprof not found at {ROCPROF}"
        return result

    backup_path = original_co + ".aftt_prof_backup"

    try:
        with tempfile.TemporaryDirectory(prefix="aftt_prof_") as tmpdir:
            all_counters_orig = {}
            all_counters_patched = {}

            for gi, group in enumerate(COUNTER_GROUPS):
                input_file = os.path.join(tmpdir, f"input_{gi}.txt")
                write_rocprof_input(group, input_file)

                # Profile original
                out_orig = os.path.join(tmpdir, f"orig_{gi}.csv")
                err = run_rocprof(bench_script, input_file, out_orig, timeout)
                if err:
                    result.error = f"original group {gi}: {err}"
                    continue
                all_counters_orig.update(parse_rocprof_csv(out_orig))

                # Swap in patched
                shutil.copy2(original_co, backup_path)
                shutil.copy2(patched_co, original_co)

                # Profile patched
                out_patched = os.path.join(tmpdir, f"patched_{gi}.csv")
                err = run_rocprof(bench_script, input_file, out_patched, timeout)
                if err:
                    result.error = f"patched group {gi}: {err}"

                # Restore original immediately
                if os.path.exists(backup_path):
                    shutil.move(backup_path, original_co)

                all_counters_patched.update(parse_rocprof_csv(out_patched))

            result.counters_original = all_counters_orig
            result.counters_patched = all_counters_patched

            # Compute deltas
            all_keys = set(all_counters_orig.keys()) | set(all_counters_patched.keys())
            for k in sorted(all_keys):
                orig_v = all_counters_orig.get(k, 0)
                patch_v = all_counters_patched.get(k, 0)
                delta = patch_v - orig_v
                pct = (delta / orig_v * 100) if orig_v != 0 else 0
                result.counter_deltas[k] = {
                    "original": orig_v,
                    "patched": patch_v,
                    "delta": delta,
                    "pct_change": round(pct, 2),
                }

    except Exception as exc:
        result.error = str(exc)
    finally:
        if os.path.exists(backup_path):
            shutil.move(backup_path, original_co)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1C: rocprof hardware counter profiling")
    parser.add_argument("--static-results",
                        default=str(AFTT_ROOT / "results" / "phase1_static" / "gfx942"),
                        help="Phase 1A results directory")
    parser.add_argument("--output",
                        default=str(AFTT_ROOT / "results" / "phase1_profile"),
                        help="Output directory")
    parser.add_argument("--strategy", default="nop_waitcnt")
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--max-kernels", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    sys.path.insert(0, str(AFTT_ROOT))
    from tools.gpu_bench_harness import find_bench_script

    static_dir = Path(args.static_results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(static_dir.glob("*.json"))
    kernels = []
    for rf in result_files:
        if rf.name == "summary.json":
            continue
        try:
            data = json.load(open(rf))
        except Exception:
            continue
        if not data.get("disasm_ok"):
            continue
        if args.categories and data.get("category") not in args.categories:
            continue
        strat = data.get("strategies", {}).get(args.strategy, {})
        if not strat.get("patch_ok") or strat.get("num_applied", 0) == 0:
            continue
        bench_info = find_bench_script(data["category"])
        if not bench_info:
            continue
        bench_script, _ = bench_info
        kernels.append({
            "co_path": data["co_path"],
            "category": data["category"],
            "patched_co_path": strat["patched_co_path"],
            "bench_script": bench_script,
        })

    if args.max_kernels:
        kernels = kernels[:args.max_kernels]

    print(f"Profiling {len(kernels)} kernel pairs with rocprof...")

    for i, entry in enumerate(kernels):
        print(f"  [{i+1}/{len(kernels)}] {Path(entry['co_path']).name}")
        pr = profile_kernel_pair(
            entry["co_path"], entry["patched_co_path"],
            entry["category"], args.strategy,
            entry["bench_script"], args.timeout)

        out_path = output_dir / f"{Path(entry['co_path']).stem}_{args.strategy}_profile.json"
        with open(out_path, "w") as f:
            json.dump(asdict(pr), f, indent=2)

    print("Profiling complete.")


if __name__ == "__main__":
    main()
