#!/usr/bin/env python3
"""Phase 1D: Decision Analysis Engine.

Correlates per-edit rationale with static cycle estimates, actual GPU
speedup, and hardware counter deltas to classify each AFTT edit as
effective, ineffective, harmful, or neutral.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

AFTT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EditClassification:
    """Classification of a single edit's effectiveness."""
    edit_index: int
    edit_mnemonic: str
    original_mnemonic: str
    edit_comment: str
    classification: str  # effective, ineffective, harmful, neutral
    reason: str = ""


@dataclass
class KernelDecision:
    """Decision analysis for a single kernel."""
    co_name: str
    category: str
    strategy: str

    # Static analysis
    static_improvement_pct: float = 0.0
    static_cycle_reduction: int = 0

    # GPU benchmark
    gpu_speedup: float = 1.0
    gpu_time_original_us: float = 0.0
    gpu_time_patched_us: float = 0.0

    # Hardware counters
    key_counter_changes: dict = field(default_factory=dict)

    # Edit classifications
    edit_classifications: list = field(default_factory=list)
    num_edits: int = 0

    # Overall verdict
    verdict: str = ""  # improved, regressed, neutral, error
    verdict_reason: str = ""


def classify_kernel(static_data: dict, gpu_data: Optional[dict],
                    profile_data: Optional[dict],
                    strategy: str) -> KernelDecision:
    """Classify a kernel's optimization result."""
    strat = static_data.get("strategies", {}).get(strategy, {})

    kd = KernelDecision(
        co_name=static_data.get("co_name", ""),
        category=static_data.get("category", ""),
        strategy=strategy,
        static_improvement_pct=strat.get("improvement_pct", 0),
        static_cycle_reduction=strat.get("cycle_reduction", 0),
        num_edits=strat.get("num_applied", 0),
    )

    # GPU data
    if gpu_data and not gpu_data.get("error"):
        kd.gpu_speedup = gpu_data.get("speedup", 1.0)
        kd.gpu_time_original_us = gpu_data.get("original_time_us", 0)
        kd.gpu_time_patched_us = gpu_data.get("patched_time_us", 0)

    # Profile data
    if profile_data and not profile_data.get("error"):
        deltas = profile_data.get("counter_deltas", {})
        # Extract key counters
        for counter in ["SQ_WAIT_INST_ANY", "SQ_INSTS_VALU", "SQ_INSTS_SALU",
                        "SQ_INSTS_LDS", "SQ_INSTS_VMEM_RD", "TCC_HIT_sum",
                        "TCC_MISS_sum", "SQ_WAVES"]:
            if counter in deltas:
                kd.key_counter_changes[counter] = deltas[counter]

    # Classify each edit
    edit_details = strat.get("edit_details", [])
    for ed in edit_details:
        ec = classify_single_edit(ed, kd)
        kd.edit_classifications.append(asdict(ec))

    # Overall verdict
    if kd.gpu_speedup > 1.005:
        kd.verdict = "improved"
        kd.verdict_reason = _explain_improvement(kd)
    elif kd.gpu_speedup < 0.995:
        kd.verdict = "regressed"
        kd.verdict_reason = _explain_regression(kd)
    elif kd.gpu_speedup == 1.0 and kd.gpu_time_original_us == 0:
        kd.verdict = "no_gpu_data"
        kd.verdict_reason = "GPU benchmark not available"
    else:
        kd.verdict = "neutral"
        kd.verdict_reason = "No significant performance change"

    return kd


def classify_single_edit(edit: dict, kd: KernelDecision) -> EditClassification:
    """Classify a single edit based on overall kernel results."""
    ec = EditClassification(
        edit_index=edit.get("index", -1),
        edit_mnemonic=edit.get("mnemonic", ""),
        original_mnemonic=edit.get("original_mnemonic", ""),
        edit_comment=edit.get("comment", ""),
        classification="neutral",
    )

    mnemonic = ec.edit_mnemonic

    if mnemonic == "s_nop" and ec.original_mnemonic == "s_nop":
        if kd.gpu_speedup > 1.002:
            ec.classification = "effective"
            ec.reason = "NOP reduction contributed to measured speedup"
        elif kd.gpu_speedup < 0.998:
            ec.classification = "harmful"
            ec.reason = "NOP reduction may have removed needed hazard spacing"
        else:
            ec.classification = "neutral"
            ec.reason = "NOP reduction had no measurable effect (latency hidden by pipeline)"

    elif mnemonic == "s_waitcnt":
        wait_change = kd.key_counter_changes.get("SQ_WAIT_INST_ANY", {})
        wait_delta = wait_change.get("delta", 0) if isinstance(wait_change, dict) else 0

        if kd.gpu_speedup > 1.002 and wait_delta < 0:
            ec.classification = "effective"
            ec.reason = f"waitcnt relaxation reduced stalls (SQ_WAIT delta={wait_delta:.0f})"
        elif kd.gpu_speedup < 0.998:
            ec.classification = "harmful"
            ec.reason = "waitcnt relaxation caused data hazard or increased stalls"
        elif kd.static_improvement_pct > 0 and kd.gpu_speedup <= 1.002:
            ec.classification = "ineffective"
            ec.reason = "Static predicted improvement but no GPU effect (pipeline not stalled here)"
        else:
            ec.classification = "neutral"
            ec.reason = "waitcnt change had no measurable impact"
    else:
        ec.classification = "neutral"
        ec.reason = f"Unknown edit type: {mnemonic}"

    return ec


def _explain_improvement(kd: KernelDecision) -> str:
    """Explain why a kernel improved."""
    reasons = []
    if kd.static_improvement_pct > 0:
        reasons.append(f"static estimate predicted {kd.static_improvement_pct:.1f}% improvement")

    wait_change = kd.key_counter_changes.get("SQ_WAIT_INST_ANY", {})
    if isinstance(wait_change, dict) and wait_change.get("delta", 0) < 0:
        reasons.append(f"reduced stall cycles ({wait_change['pct_change']:.1f}%)")

    nop_edits = sum(1 for e in kd.edit_classifications
                    if e.get("edit_mnemonic") == "s_nop")
    waitcnt_edits = sum(1 for e in kd.edit_classifications
                        if e.get("edit_mnemonic") == "s_waitcnt")
    if nop_edits:
        reasons.append(f"{nop_edits} NOP reductions")
    if waitcnt_edits:
        reasons.append(f"{waitcnt_edits} waitcnt relaxations")

    return "; ".join(reasons) if reasons else "improvement from combined edits"


def _explain_regression(kd: KernelDecision) -> str:
    """Explain why a kernel regressed."""
    reasons = []
    cache_miss = kd.key_counter_changes.get("TCC_MISS_sum", {})
    if isinstance(cache_miss, dict) and cache_miss.get("delta", 0) > 0:
        reasons.append(f"increased L2 cache misses ({cache_miss['pct_change']:.1f}%)")

    wait_change = kd.key_counter_changes.get("SQ_WAIT_INST_ANY", {})
    if isinstance(wait_change, dict) and wait_change.get("delta", 0) > 0:
        reasons.append(f"increased stall cycles ({wait_change['pct_change']:.1f}%)")

    waitcnt_edits = sum(1 for e in kd.edit_classifications
                        if e.get("edit_mnemonic") == "s_waitcnt")
    if waitcnt_edits:
        reasons.append(f"unsafe waitcnt relaxation ({waitcnt_edits} edits)")

    return "; ".join(reasons) if reasons else "regression from combined edits"


def run_analysis(static_dir: Path, gpu_dir: Optional[Path],
                 profile_dir: Optional[Path], output_dir: Path,
                 strategy: str = "nop_waitcnt"):
    """Run decision analysis across all kernels."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load static results
    static_files = {f.stem: f for f in static_dir.glob("*.json") if f.name != "summary.json"}

    # Load GPU results
    gpu_map = {}
    if gpu_dir and gpu_dir.exists():
        for gf in gpu_dir.glob(f"*_{strategy}.json"):
            stem = gf.stem.replace(f"_{strategy}", "")
            try:
                gpu_map[stem] = json.load(open(gf))
            except Exception:
                pass

    # Load profile results
    prof_map = {}
    if profile_dir and profile_dir.exists():
        for pf in profile_dir.glob(f"*_{strategy}_profile.json"):
            stem = pf.stem.replace(f"_{strategy}_profile", "")
            try:
                prof_map[stem] = json.load(open(pf))
            except Exception:
                pass

    all_decisions = []
    stats = defaultdict(lambda: defaultdict(int))

    for stem, sf in sorted(static_files.items()):
        try:
            static_data = json.load(open(sf))
        except Exception:
            continue

        gpu_data = gpu_map.get(stem)
        prof_data = prof_map.get(stem)

        kd = classify_kernel(static_data, gpu_data, prof_data, strategy)
        all_decisions.append(asdict(kd))

        # Aggregate stats
        cat = kd.category
        stats["overall"][kd.verdict] += 1
        stats[f"cat:{cat}"][kd.verdict] += 1

        for ec in kd.edit_classifications:
            cl = ec.get("classification", "neutral")
            mn = ec.get("edit_mnemonic", "unknown")
            stats[f"edit:{mn}"][cl] += 1

    # Write all decisions
    decisions_path = output_dir / f"decisions_{strategy}.json"
    with open(decisions_path, "w") as f:
        json.dump(all_decisions, f, indent=2)

    # Write stats
    stats_path = output_dir / f"stats_{strategy}.json"
    with open(stats_path, "w") as f:
        json.dump(dict(stats), f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Decision Analysis Summary (strategy: {strategy})")
    print(f"{'='*60}")
    print(f"  Total kernels analyzed: {len(all_decisions)}")

    ov = stats["overall"]
    for verdict in ["improved", "regressed", "neutral", "no_gpu_data", "error"]:
        count = ov.get(verdict, 0)
        if count > 0:
            print(f"  {verdict}: {count}")

    print(f"\n  Per edit-type effectiveness:")
    for key in sorted(stats.keys()):
        if key.startswith("edit:"):
            mn = key.split(":")[1]
            s = stats[key]
            total = sum(s.values())
            eff = s.get("effective", 0)
            harm = s.get("harmful", 0)
            print(f"    {mn}: {eff}/{total} effective, {harm}/{total} harmful")

    print(f"\n  Per category:")
    for key in sorted(stats.keys()):
        if key.startswith("cat:"):
            cat = key.split(":")[1]
            s = stats[key]
            total = sum(s.values())
            imp = s.get("improved", 0)
            reg = s.get("regressed", 0)
            print(f"    {cat}: {imp}/{total} improved, {reg}/{total} regressed")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1D: Decision analysis engine")
    parser.add_argument("--static-results",
                        default=str(AFTT_ROOT / "results" / "phase1_static" / "gfx942"))
    parser.add_argument("--gpu-results",
                        default=str(AFTT_ROOT / "results" / "phase1_gpu"))
    parser.add_argument("--profile-results",
                        default=str(AFTT_ROOT / "results" / "phase1_profile"))
    parser.add_argument("--output",
                        default=str(AFTT_ROOT / "results" / "phase1_analysis"))
    parser.add_argument("--strategy", default="nop_waitcnt")
    args = parser.parse_args()

    run_analysis(
        Path(args.static_results),
        Path(args.gpu_results) if args.gpu_results else None,
        Path(args.profile_results) if args.profile_results else None,
        Path(args.output),
        strategy=args.strategy)


if __name__ == "__main__":
    main()
