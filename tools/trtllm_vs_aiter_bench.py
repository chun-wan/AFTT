#!/usr/bin/env python3
"""Phase 2B: TRT-LLM vs aiter Performance Comparison Matrix.

Benchmarks 4 versions of each kernel where both TRT-LLM HIPified
and aiter production .co exist:
  - TRT-HIP-naive: HIPified TRT-LLM, compiled with hipcc -O3
  - TRT-HIP-AFTT-B: AFTT C++ template swap + compile
  - TRT-HIP-AFTT-C: Version B + ASM optimization
  - aiter-production: aiter's hand-tuned .co
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

AFTT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AFTT_ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.asm_editor import AsmEditor
from src.cycle_estimator import CycleEstimator


@dataclass
class VersionBenchmark:
    """Benchmark result for a single version of a kernel."""
    version: str  # trt_naive, aftt_b, aftt_c, aiter_prod
    co_path: str = ""
    exists: bool = False
    instruction_count: int = 0
    estimated_cycles: int = 0
    bottleneck: str = ""
    gpu_time_us: float = 0.0
    error: str = ""


@dataclass
class ComparisonEntry:
    """4-way comparison for one kernel."""
    kernel_name: str
    category: str
    trt_cu_file: str = ""
    versions: dict = field(default_factory=dict)
    best_version: str = ""
    aftt_vs_aiter_ratio: float = 0.0


def static_analyze_co(co_path: str, editor: AsmEditor,
                      estimator: CycleEstimator) -> dict:
    """Perform static analysis on a .co file."""
    result = {"exists": False, "instruction_count": 0, "estimated_cycles": 0, "bottleneck": ""}
    if not co_path or not os.path.exists(co_path):
        return result

    try:
        _, instructions = editor.disassemble(co_path)
        result["exists"] = True
        result["instruction_count"] = len(instructions)
        lines = [i.full_text for i in instructions]
        est = estimator.estimate(lines)
        result["estimated_cycles"] = est.total_cycles
        result["bottleneck"] = est.bottleneck
    except Exception:
        pass
    return result


def build_comparison_matrix(hipify_dir: Path, arch: str = "gfx942") -> list[ComparisonEntry]:
    """Build the 4-version comparison matrix."""
    editor = AsmEditor(arch=arch)
    estimator = CycleEstimator(arch=arch)

    # Load all hipify results
    result_files = sorted(hipify_dir.glob("*_result.json"))
    if not result_files:
        print(f"No hipify results found in {hipify_dir}")
        return []

    entries = []
    for rf in result_files:
        try:
            data = json.load(open(rf))
        except Exception:
            continue

        entry = ComparisonEntry(
            kernel_name=data.get("cu_name", ""),
            category=data.get("category_guess", ""),
            trt_cu_file=data.get("cu_path", ""),
        )

        # Version A: TRT-HIP-naive
        co_a = data.get("co_a_path", "")
        vb_a = VersionBenchmark(version="trt_naive", co_path=co_a)
        sa = static_analyze_co(co_a, editor, estimator)
        vb_a.exists = sa["exists"]
        vb_a.instruction_count = sa["instruction_count"]
        vb_a.estimated_cycles = sa["estimated_cycles"]
        vb_a.bottleneck = sa["bottleneck"]
        entry.versions["trt_naive"] = asdict(vb_a)

        # Version B: AFTT-B (template swap)
        co_b = data.get("co_b_path", "")
        vb_b = VersionBenchmark(version="aftt_b", co_path=co_b)
        sb = static_analyze_co(co_b, editor, estimator)
        vb_b.exists = sb["exists"]
        vb_b.instruction_count = sb["instruction_count"]
        vb_b.estimated_cycles = sb["estimated_cycles"]
        vb_b.bottleneck = sb["bottleneck"]
        entry.versions["aftt_b"] = asdict(vb_b)

        # Version C: AFTT-C (ASM optimized)
        co_c = data.get("co_c_path", "")
        vb_c = VersionBenchmark(version="aftt_c", co_path=co_c)
        sc = static_analyze_co(co_c, editor, estimator)
        vb_c.exists = sc["exists"]
        vb_c.instruction_count = sc["instruction_count"]
        vb_c.estimated_cycles = sc["estimated_cycles"]
        vb_c.bottleneck = sc["bottleneck"]
        entry.versions["aftt_c"] = asdict(vb_c)

        # aiter-production
        aiter_co = data.get("aiter_co_path", "")
        vb_aiter = VersionBenchmark(version="aiter_prod", co_path=aiter_co)
        sp = static_analyze_co(aiter_co, editor, estimator)
        vb_aiter.exists = sp["exists"]
        vb_aiter.instruction_count = sp["instruction_count"]
        vb_aiter.estimated_cycles = sp["estimated_cycles"]
        vb_aiter.bottleneck = sp["bottleneck"]
        entry.versions["aiter_prod"] = asdict(vb_aiter)

        # Determine best version (lowest cycle count among existing)
        best_name = ""
        best_cycles = float("inf")
        for vname, vdata in entry.versions.items():
            if vdata["exists"] and vdata["estimated_cycles"] > 0:
                if vdata["estimated_cycles"] < best_cycles:
                    best_cycles = vdata["estimated_cycles"]
                    best_name = vname
        entry.best_version = best_name

        # AFTT vs aiter ratio (using best AFTT version vs aiter)
        aftt_best = float("inf")
        for vname in ["aftt_c", "aftt_b", "trt_naive"]:
            vd = entry.versions.get(vname, {})
            if vd.get("exists") and vd.get("estimated_cycles", 0) > 0:
                aftt_best = min(aftt_best, vd["estimated_cycles"])
                break

        aiter_cycles = entry.versions.get("aiter_prod", {}).get("estimated_cycles", 0)
        if aiter_cycles > 0 and aftt_best < float("inf"):
            entry.aftt_vs_aiter_ratio = aiter_cycles / aftt_best

        entries.append(entry)

    return entries


def write_bench_summary(entries: list[ComparisonEntry], output_dir: Path):
    """Write benchmark comparison summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full results
    results_path = output_dir / "bench_comparison.json"
    with open(results_path, "w") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)

    # Summary for report
    total = len(entries)
    has_aiter = sum(1 for e in entries if e.versions.get("aiter_prod", {}).get("exists", False))
    has_aftt_c = sum(1 for e in entries if e.versions.get("aftt_c", {}).get("exists", False))
    aftt_wins = sum(1 for e in entries if e.best_version in ("aftt_b", "aftt_c"))
    aiter_wins = sum(1 for e in entries if e.best_version == "aiter_prod")

    ratios = [e.aftt_vs_aiter_ratio for e in entries if e.aftt_vs_aiter_ratio > 0]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0

    summary = {
        "total_kernels": total,
        "has_aiter_equivalent": has_aiter,
        "has_aftt_version_c": has_aftt_c,
        "aftt_wins_static": aftt_wins,
        "aiter_wins_static": aiter_wins,
        "avg_aftt_vs_aiter_ratio": round(avg_ratio, 4),
        "results": [],
    }

    for e in entries:
        trt_naive_cycles = e.versions.get("trt_naive", {}).get("estimated_cycles", 0)
        aftt_b_cycles = e.versions.get("aftt_b", {}).get("estimated_cycles", 0)
        aftt_c_cycles = e.versions.get("aftt_c", {}).get("estimated_cycles", 0)
        aiter_cycles = e.versions.get("aiter_prod", {}).get("estimated_cycles", 0)

        summary["results"].append({
            "name": e.kernel_name,
            "category": e.category,
            "trt_naive_cycles": trt_naive_cycles,
            "aftt_b_cycles": aftt_b_cycles,
            "aftt_c_cycles": aftt_c_cycles,
            "aiter_cycles": aiter_cycles,
            "best_version": e.best_version,
            "aftt_vs_aiter": round(e.aftt_vs_aiter_ratio, 4),
        })

    summary_path = output_dir / "bench_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nComparison Matrix Summary:")
    print(f"  Total kernels: {total}")
    print(f"  With aiter equivalent: {has_aiter}")
    print(f"  With AFTT Version C: {has_aftt_c}")
    print(f"  AFTT wins (static): {aftt_wins}")
    print(f"  aiter wins (static): {aiter_wins}")
    if ratios:
        print(f"  Avg AFTT vs aiter ratio: {avg_ratio:.4f}")
        print(f"    (>1.0 means AFTT is better, <1.0 means aiter is better)")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2B: TRT-LLM vs aiter comparison matrix")
    parser.add_argument("--hipify-results",
                        default=str(AFTT_ROOT / "results" / "phase2" / "hipify"),
                        help="Phase 2A hipify results directory")
    parser.add_argument("--output",
                        default=str(AFTT_ROOT / "results" / "phase2" / "comparison"),
                        help="Output directory")
    parser.add_argument("--arch", default="gfx942")
    args = parser.parse_args()

    entries = build_comparison_matrix(Path(args.hipify_results), args.arch)
    write_bench_summary(entries, Path(args.output))


if __name__ == "__main__":
    main()
