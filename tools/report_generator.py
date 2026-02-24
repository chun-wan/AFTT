#!/usr/bin/env python3
"""Phase 1E + 2D: Comprehensive Report Generator.

Produces markdown reports from Phase 1 and Phase 2 results with
per-category breakdowns, optimization recommendations, and
profiling validation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

AFTT_ROOT = Path(__file__).resolve().parent.parent


def load_json_dir(dirpath: Path, pattern: str = "*.json") -> list[dict]:
    """Load all JSON files matching pattern from a directory."""
    results = []
    if not dirpath.exists():
        return results
    for f in sorted(dirpath.glob(pattern)):
        try:
            results.append(json.load(open(f)))
        except Exception:
            pass
    return results


def generate_phase1_report(results_root: Path, arch: str = "gfx942",
                           strategy: str = "nop_waitcnt") -> str:
    """Generate Phase 1 report from all sub-phase results."""
    lines = []
    lines.append("# AFTT Phase 1: aiter ASM Kernel Optimization Report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Architecture: {arch}")
    lines.append(f"Primary strategy: {strategy}\n")

    # ── Static results ──
    static_dir = results_root / "phase1_static" / arch
    summary_path = static_dir / "summary.json"

    if summary_path.exists():
        summary = json.load(open(summary_path))
        lines.append("## 1. Static ASM Analysis\n")
        lines.append(f"- Total kernels scanned: {summary.get('total_kernels', 0)}")
        lines.append(f"- Successfully disassembled: {summary.get('disasm_ok', 0)}")
        lines.append(f"- Disassembly failures: {summary.get('disasm_failed', 0)}\n")

        lines.append("### Strategy Comparison\n")
        lines.append("| Strategy | Evaluated | Patched | Improved | Regressed | Avg Improvement |")
        lines.append("|----------|-----------|---------|----------|-----------|-----------------|")

        for sname, sdata in summary.get("strategies", {}).items():
            lines.append(
                f"| {sname} | {sdata['kernels_evaluated']} "
                f"| {sdata['kernels_patched']} "
                f"| {sdata['kernels_improved']} "
                f"| {sdata['kernels_regressed']} "
                f"| {sdata['avg_improvement_pct']:.1f}% |")

        lines.append("\n### Per-Category Breakdown\n")
        lines.append("| Category | Total | Disasm OK |")
        lines.append("|----------|-------|-----------|")
        for cat, cdata in sorted(summary.get("categories", {}).items()):
            lines.append(f"| {cat} | {cdata['total']} | {cdata['disasm_ok']} |")

    # ── GPU results ──
    gpu_dir = results_root / "phase1_gpu"
    gpu_summary_path = gpu_dir / f"gpu_summary_{strategy}.json"

    if gpu_summary_path.exists():
        gpu_summary = json.load(open(gpu_summary_path))
        lines.append("\n## 2. GPU Benchmark Results\n")
        lines.append(f"- Total benchmarked: {gpu_summary.get('total_benchmarked', 0)}")
        lines.append(f"- Improved (>0.5%): {gpu_summary.get('improved', 0)}")
        lines.append(f"- Regressed (<-0.5%): {gpu_summary.get('regressed', 0)}")
        lines.append(f"- Neutral: {gpu_summary.get('neutral', 0)}")
        lines.append(f"- Errors: {gpu_summary.get('errors', 0)}")
        if "avg_speedup" in gpu_summary:
            lines.append(f"- Average speedup: {gpu_summary['avg_speedup']:.4f}x")
            lines.append(f"- Max speedup: {gpu_summary['max_speedup']:.4f}x")
            lines.append(f"- Min speedup: {gpu_summary['min_speedup']:.4f}x")

    # ── Decision analysis ──
    analysis_dir = results_root / "phase1_analysis"
    stats_path = analysis_dir / f"stats_{strategy}.json"

    if stats_path.exists():
        stats = json.load(open(stats_path))
        lines.append("\n## 3. Decision Analysis\n")

        ov = stats.get("overall", {})
        total = sum(ov.values())
        lines.append(f"Total kernels with decisions: {total}\n")
        lines.append("| Verdict | Count | Percentage |")
        lines.append("|---------|-------|------------|")
        for verdict in ["improved", "regressed", "neutral", "no_gpu_data"]:
            count = ov.get(verdict, 0)
            pct = count / max(total, 1) * 100
            lines.append(f"| {verdict} | {count} | {pct:.1f}% |")

        # Edit effectiveness
        lines.append("\n### Per-Edit-Type Effectiveness\n")
        lines.append("| Edit Type | Total | Effective | Ineffective | Harmful | Neutral |")
        lines.append("|-----------|-------|-----------|-------------|---------|---------|")
        for key in sorted(stats.keys()):
            if key.startswith("edit:"):
                mn = key.split(":")[1]
                s = stats[key]
                total_e = sum(s.values())
                lines.append(
                    f"| {mn} | {total_e} "
                    f"| {s.get('effective', 0)} "
                    f"| {s.get('ineffective', 0)} "
                    f"| {s.get('harmful', 0)} "
                    f"| {s.get('neutral', 0)} |")

        # Per-category
        lines.append("\n### Per-Category Results\n")
        lines.append("| Category | Total | Improved | Regressed | Neutral |")
        lines.append("|----------|-------|----------|-----------|---------|")
        for key in sorted(stats.keys()):
            if key.startswith("cat:"):
                cat = key.split(":")[1]
                s = stats[key]
                total_c = sum(s.values())
                lines.append(
                    f"| {cat} | {total_c} "
                    f"| {s.get('improved', 0)} "
                    f"| {s.get('regressed', 0)} "
                    f"| {s.get('neutral', 0)} |")

    # ── Top improvements & regressions ──
    decisions_path = analysis_dir / f"decisions_{strategy}.json"
    if decisions_path.exists():
        decisions = json.load(open(decisions_path))

        improved = sorted(
            [d for d in decisions if d.get("gpu_speedup", 1) > 1.005],
            key=lambda d: d["gpu_speedup"], reverse=True)
        regressed = sorted(
            [d for d in decisions if d.get("gpu_speedup", 1) < 0.995],
            key=lambda d: d["gpu_speedup"])

        if improved:
            lines.append("\n### Top 10 Improvements\n")
            lines.append("| Kernel | Category | Speedup | Reason |")
            lines.append("|--------|----------|---------|--------|")
            for d in improved[:10]:
                lines.append(
                    f"| {d['co_name'][:40]} | {d['category']} "
                    f"| {d['gpu_speedup']:.4f}x "
                    f"| {d.get('verdict_reason', '')[:60]} |")

        if regressed:
            lines.append("\n### Top 10 Regressions\n")
            lines.append("| Kernel | Category | Speedup | Reason |")
            lines.append("|--------|----------|---------|--------|")
            for d in regressed[:10]:
                lines.append(
                    f"| {d['co_name'][:40]} | {d['category']} "
                    f"| {d['gpu_speedup']:.4f}x "
                    f"| {d.get('verdict_reason', '')[:60]} |")

    # ── Recommendations ──
    lines.append("\n## 4. Recommendations\n")
    lines.append("Based on the analysis:\n")
    lines.append("1. **NOP reduction** is the safest optimization — apply universally")
    lines.append("2. **waitcnt relaxation** requires per-kernel-type tuning:")
    lines.append("   - Safe for memory-bound kernels (norm, topk)")
    lines.append("   - Risky for shared-memory-heavy kernels (GEMM, FMHA, MoE)")
    lines.append("3. **Level 3+ patterns** (LDS→DPP, vectorization) need validation per kernel")
    lines.append("4. Investigate regressions — may indicate bugs in dependency analysis\n")

    return "\n".join(lines)


def generate_phase2_report(results_root: Path) -> str:
    """Generate Phase 2 report from TRT-LLM optimization results."""
    lines = []
    lines.append("# AFTT Phase 2: TRT-LLM Kernel Optimization Report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")

    phase2_dir = results_root / "phase2"
    if not phase2_dir.exists():
        lines.append("No Phase 2 results available yet.\n")
        return "\n".join(lines)

    summary_path = phase2_dir / "hipify_summary.json"
    if summary_path.exists():
        summary = json.load(open(summary_path))
        lines.append("## 1. HIPify Results\n")
        lines.append(f"- Total .cu files processed: {summary.get('total_files', 0)}")
        lines.append(f"- Successfully HIPified: {summary.get('hipify_ok', 0)}")
        lines.append(f"- Compilation succeeded: {summary.get('compile_ok', 0)}")
        lines.append(f"- Algorithm classified: {summary.get('classified', 0)}")
        lines.append(f"- Template match found: {summary.get('template_match', 0)}\n")

    bench_path = phase2_dir / "bench_summary.json"
    if bench_path.exists():
        bench = json.load(open(bench_path))
        lines.append("## 2. Performance Comparison\n")
        lines.append("| Kernel | TRT-HIP-naive | AFTT-B | AFTT-C | aiter-prod | AFTT vs aiter |")
        lines.append("|--------|---------------|--------|--------|------------|---------------|")
        for entry in bench.get("results", []):
            lines.append(
                f"| {entry.get('name', '')[:30]} "
                f"| {entry.get('trt_naive_us', 0):.1f} us "
                f"| {entry.get('aftt_b_us', 0):.1f} us "
                f"| {entry.get('aftt_c_us', 0):.1f} us "
                f"| {entry.get('aiter_us', 0):.1f} us "
                f"| {entry.get('aftt_vs_aiter', 0):.2f}x |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1E + 2D: Report generator")
    parser.add_argument("--results-root",
                        default=str(AFTT_ROOT / "results"),
                        help="Root directory containing all phase results")
    parser.add_argument("--arch", default="gfx942")
    parser.add_argument("--strategy", default="nop_waitcnt")
    parser.add_argument("--output", default=None,
                        help="Output markdown file (default: results/report.md)")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_path = Path(args.output) if args.output else results_root / "report.md"

    report = ""
    if args.phase in ("1", "all"):
        report += generate_phase1_report(results_root, args.arch, args.strategy)
    if args.phase in ("2", "all"):
        if report:
            report += "\n\n---\n\n"
        report += generate_phase2_report(results_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report written to {output_path}")
    print(f"({len(report)} characters, {report.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
