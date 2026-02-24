#!/usr/bin/env python3
"""Phase 1A: Batch ASM Evaluation Engine.

Scans all aiter production .co kernels, applies AFTT ASM modifications
with multiple edit strategies, generates patched .co files and static
cycle comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

AFTT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AFTT_ROOT))

from src.asm_editor import AsmEditor
from src.asm_optimizer import AsmOptimizer
from src.pattern_replacer import PatternReplacer
from src.cycle_estimator import CycleEstimator
from src.knowledge_base import KnowledgeBase
from src.instruction import EditOperation


@dataclass
class EditStrategy:
    name: str
    description: str
    use_optimizer: bool = True
    use_replacer: bool = True
    max_replace_level: int = 2
    skip_waitcnt: bool = False
    aggressive: bool = False


STRATEGIES = [
    EditStrategy(
        name="nop_only",
        description="NOP reduction only (safest)",
        use_optimizer=True,
        use_replacer=False,
        skip_waitcnt=True,
    ),
    EditStrategy(
        name="nop_waitcnt",
        description="NOP + waitcnt + barrier + waitcnt-split + MFMA-interleave",
        use_optimizer=True,
        use_replacer=False,
        skip_waitcnt=False,
    ),
    EditStrategy(
        name="full_level4",
        description="All optimizer passes + LDS→DPP + load vectorization",
        use_optimizer=True,
        use_replacer=True,
        max_replace_level=4,
        skip_waitcnt=False,
    ),
]


@dataclass
class KernelResult:
    co_path: str
    co_name: str
    kernel_name: str
    category: str
    arch: str
    num_instructions: int = 0
    disasm_ok: bool = False
    error: str = ""
    strategies: dict = field(default_factory=dict)


@dataclass
class StrategyResult:
    strategy_name: str
    num_edits_optimizer: int = 0
    num_edits_replacer: int = 0
    num_edits_merged: int = 0
    num_edits_skipped_waitcnt: int = 0
    num_applied: int = 0
    num_skipped: int = 0
    patch_ok: bool = False
    patched_co_path: str = ""
    original_cycles: int = 0
    patched_cycles: int = 0
    cycle_reduction: int = 0
    improvement_pct: float = 0.0
    original_bottleneck: str = ""
    patched_bottleneck: str = ""
    edit_details: list = field(default_factory=list)
    error: str = ""


# ── CSV manifest scanning ──────────────────────────────────────────

def scan_aiter_co_files(aiter_hsa_dir: str, arch: str) -> list[dict]:
    """Scan all CSV manifests under aiter/hsa/{arch}/ and resolve .co paths."""
    base = Path(aiter_hsa_dir) / arch
    if not base.exists():
        print(f"  WARNING: {base} does not exist")
        return []

    results = []
    for csv_path in sorted(base.rglob("*.csv")):
        rel_dir = csv_path.parent.relative_to(base)
        category = str(rel_dir).split("/")[0]

        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    co_name = row.get("co_name", "").strip()
                    knl_name = row.get("knl_name", "").strip()
                    if not co_name:
                        continue
                    co_full = csv_path.parent / co_name
                    results.append({
                        "co_path": str(co_full),
                        "co_name": co_name,
                        "kernel_name": knl_name,
                        "category": category,
                        "csv_path": str(csv_path),
                        "exists": co_full.exists(),
                    })
        except Exception as exc:
            print(f"  WARNING: failed to parse {csv_path}: {exc}")

    # Also scan for .co files not in any CSV (loose files in arch root)
    for co_file in sorted(base.glob("*.co")):
        co_name = co_file.name
        if not any(r["co_name"] == co_name for r in results):
            results.append({
                "co_path": str(co_file),
                "co_name": co_name,
                "kernel_name": "",
                "category": "other",
                "csv_path": "",
                "exists": True,
            })

    return results


def scan_all_co_files(aiter_hsa_dir: str, arch: str) -> list[dict]:
    """Scan all .co files on disk (not just CSV-listed ones)."""
    base = Path(aiter_hsa_dir) / arch
    if not base.exists():
        return []

    csv_entries = scan_aiter_co_files(aiter_hsa_dir, arch)
    csv_paths = {e["co_path"] for e in csv_entries}

    # Add any .co files found on disk but not in CSVs
    for co_file in sorted(base.rglob("*.co")):
        if str(co_file) not in csv_paths:
            rel_dir = co_file.parent.relative_to(base)
            category = str(rel_dir).split("/")[0] if str(rel_dir) != "." else "other"
            csv_entries.append({
                "co_path": str(co_file),
                "co_name": co_file.name,
                "kernel_name": "",
                "category": category,
                "csv_path": "",
                "exists": True,
            })

    return [e for e in csv_entries if e["exists"]]


# ── Core evaluation logic ──────────────────────────────────────────

def evaluate_kernel(co_entry: dict, editor: AsmEditor, optimizer: AsmOptimizer,
                    replacer: PatternReplacer, estimator: CycleEstimator,
                    output_dir: Path, arch: str) -> KernelResult:
    """Evaluate a single .co kernel with all edit strategies."""
    result = KernelResult(
        co_path=co_entry["co_path"],
        co_name=co_entry["co_name"],
        kernel_name=co_entry["kernel_name"],
        category=co_entry["category"],
        arch=arch,
    )

    # Disassemble
    try:
        kernel_info, instructions = editor.disassemble(co_entry["co_path"])
        result.num_instructions = len(instructions)
        result.disasm_ok = True
        if not result.kernel_name:
            result.kernel_name = kernel_info.name
    except Exception as exc:
        result.error = f"disassemble failed: {exc}"
        return result

    if not instructions:
        result.error = "empty instruction list"
        return result

    # Original cycle estimate
    orig_lines = editor.get_instruction_lines(instructions)
    try:
        orig_cycles = estimator.estimate(orig_lines)
    except Exception:
        orig_cycles = None

    # Apply each strategy
    for strategy in STRATEGIES:
        sr = apply_strategy(
            strategy, co_entry["co_path"], instructions,
            editor, optimizer, replacer, estimator,
            orig_lines, orig_cycles, output_dir)
        result.strategies[strategy.name] = asdict(sr)

    return result


def apply_strategy(strategy: EditStrategy, co_path: str,
                   instructions, editor: AsmEditor,
                   optimizer: AsmOptimizer, replacer: PatternReplacer,
                   estimator: CycleEstimator, orig_lines, orig_cycles,
                   output_dir: Path) -> StrategyResult:
    """Apply a single edit strategy to a kernel."""
    sr = StrategyResult(strategy_name=strategy.name)
    if orig_cycles:
        sr.original_cycles = orig_cycles.total_cycles
        sr.original_bottleneck = orig_cycles.bottleneck

    try:
        all_edits: dict[int, EditOperation] = {}
        skipped_waitcnt = 0

        # AsmOptimizer edits
        if strategy.use_optimizer:
            opt_result = optimizer.optimize(instructions, aggressive=strategy.aggressive)
            sr.num_edits_optimizer = len(opt_result.edits)
            for e in opt_result.edits:
                if strategy.skip_waitcnt and e.new_mnemonic == "s_waitcnt":
                    skipped_waitcnt += 1
                    continue
                all_edits[e.target_index] = e

        # PatternReplacer edits (levels 3+ add LDS→DPP and vectorization)
        if strategy.use_replacer:
            repl_result = replacer.find_replacements_standalone(
                instructions, max_level=strategy.max_replace_level)
            safe_edits = replacer.get_safe_edits(
                repl_result.replacements, max_level=strategy.max_replace_level)
            sr.num_edits_replacer = len(safe_edits)
            for e in safe_edits:
                if strategy.skip_waitcnt and e.new_mnemonic == "s_waitcnt":
                    skipped_waitcnt += 1
                    continue
                if e.target_index not in all_edits:
                    all_edits[e.target_index] = e

        sr.num_edits_skipped_waitcnt = skipped_waitcnt
        edit_list = sorted(all_edits.values(), key=lambda e: e.target_index)
        sr.num_edits_merged = len(edit_list)

        # Record edit details
        for e in edit_list:
            sr.edit_details.append({
                "index": e.target_index,
                "mnemonic": e.new_mnemonic,
                "operands": e.new_operands,
                "comment": e.comment,
                "original_mnemonic": instructions[e.target_index].mnemonic
                    if e.target_index < len(instructions) else "",
            })

        if not edit_list:
            sr.patched_cycles = sr.original_cycles
            sr.patch_ok = True
            return sr

        # Binary patch
        co_name = Path(co_path).stem
        patched_path = output_dir / f"{co_name}_{strategy.name}.co"
        patch_result = editor.binary_patch(
            co_path, str(patched_path), edit_list, instructions)
        sr.num_applied = patch_result["applied_count"]
        sr.num_skipped = patch_result["skipped_count"]
        sr.patched_co_path = str(patched_path)
        sr.patch_ok = sr.num_applied > 0

        # Patched cycle estimate
        if sr.num_applied > 0:
            mod_lines = editor.apply_and_get_modified_lines(instructions, edit_list)
            try:
                mod_cycles = estimator.estimate(mod_lines)
                sr.patched_cycles = mod_cycles.total_cycles
                sr.patched_bottleneck = mod_cycles.bottleneck
                sr.cycle_reduction = sr.original_cycles - sr.patched_cycles
                if sr.original_cycles > 0:
                    sr.improvement_pct = (sr.cycle_reduction / sr.original_cycles) * 100
            except Exception:
                sr.patched_cycles = sr.original_cycles
        else:
            sr.patched_cycles = sr.original_cycles

    except Exception as exc:
        sr.error = str(exc)

    return sr


# ── Main batch runner ──────────────────────────────────────────────

def run_batch(aiter_hsa_dir: str, arch: str, output_dir: Path,
              categories: Optional[list[str]] = None,
              max_kernels: Optional[int] = None,
              verbose: bool = False) -> list[KernelResult]:
    """Run batch evaluation on all aiter .co kernels."""
    print(f"Scanning .co files in {aiter_hsa_dir}/{arch}/ ...")
    co_entries = scan_all_co_files(aiter_hsa_dir, arch)
    print(f"  Found {len(co_entries)} .co files on disk")

    if categories:
        co_entries = [e for e in co_entries if e["category"] in categories]
        print(f"  Filtered to {len(co_entries)} (categories: {categories})")

    if max_kernels and len(co_entries) > max_kernels:
        co_entries = co_entries[:max_kernels]
        print(f"  Limited to {max_kernels} kernels")

    # Initialize AFTT components
    kb = KnowledgeBase()
    kb.load()
    editor = AsmEditor(arch=arch)
    optimizer = AsmOptimizer(arch=arch, kb=kb)
    replacer = PatternReplacer(kb=kb)
    estimator = CycleEstimator(arch=arch)

    output_dir.mkdir(parents=True, exist_ok=True)
    patched_dir = output_dir / "patched"
    patched_dir.mkdir(exist_ok=True)

    results = []
    total = len(co_entries)
    t0 = time.time()

    for i, entry in enumerate(co_entries):
        if verbose or (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.1)
            eta = (total - i - 1) / max(rate, 0.01)
            print(f"  [{i+1}/{total}] {entry['co_name']} "
                  f"({entry['category']}) "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

        kr = evaluate_kernel(entry, editor, optimizer, replacer, estimator,
                             patched_dir, arch)
        results.append(kr)

        # Write per-kernel result
        result_path = output_dir / f"{Path(entry['co_name']).stem}.json"
        try:
            with open(result_path, "w") as f:
                json.dump(asdict(kr), f, indent=2, default=str)
        except Exception:
            pass

    elapsed_total = time.time() - t0
    print(f"\nBatch complete: {total} kernels in {elapsed_total:.1f}s "
          f"({elapsed_total/max(total,1):.2f}s/kernel)")

    return results


def write_summary(results: list[KernelResult], output_dir: Path):
    """Write aggregate summary JSON."""
    summary = {
        "total_kernels": len(results),
        "disasm_ok": sum(1 for r in results if r.disasm_ok),
        "disasm_failed": sum(1 for r in results if not r.disasm_ok),
        "categories": {},
        "strategies": {},
    }

    # Per-category stats
    cats = {}
    for r in results:
        if r.category not in cats:
            cats[r.category] = {"total": 0, "disasm_ok": 0}
        cats[r.category]["total"] += 1
        if r.disasm_ok:
            cats[r.category]["disasm_ok"] += 1
    summary["categories"] = cats

    # Per-strategy stats
    for strat in STRATEGIES:
        sname = strat.name
        s_results = []
        for r in results:
            if sname in r.strategies:
                s_results.append(r.strategies[sname])

        improved = [s for s in s_results if s.get("cycle_reduction", 0) > 0]
        regressed = [s for s in s_results if s.get("cycle_reduction", 0) < 0]
        patched = [s for s in s_results if s.get("patch_ok", False) and s.get("num_applied", 0) > 0]

        total_edits = sum(s.get("num_edits_merged", 0) for s in s_results)
        total_applied = sum(s.get("num_applied", 0) for s in s_results)

        avg_improvement = 0.0
        if improved:
            avg_improvement = sum(s["improvement_pct"] for s in improved) / len(improved)

        summary["strategies"][sname] = {
            "description": strat.description,
            "kernels_evaluated": len(s_results),
            "kernels_patched": len(patched),
            "kernels_improved": len(improved),
            "kernels_regressed": len(regressed),
            "total_edits_proposed": total_edits,
            "total_edits_applied": total_applied,
            "avg_improvement_pct": round(avg_improvement, 2),
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  AFTT Batch ASM Evaluation Summary")
    print("=" * 70)
    print(f"  Total kernels: {summary['total_kernels']}")
    print(f"  Disassembled OK: {summary['disasm_ok']}")
    print(f"  Disassembly failed: {summary['disasm_failed']}")

    print("\n  Per-strategy results:")
    for sname, sdata in summary["strategies"].items():
        print(f"\n  [{sname}] {sdata['description']}")
        print(f"    Evaluated: {sdata['kernels_evaluated']}")
        print(f"    Patched:   {sdata['kernels_patched']}")
        print(f"    Improved:  {sdata['kernels_improved']} "
              f"(avg {sdata['avg_improvement_pct']:.1f}%)")
        print(f"    Regressed: {sdata['kernels_regressed']}")
        print(f"    Edits proposed/applied: "
              f"{sdata['total_edits_proposed']}/{sdata['total_edits_applied']}")

    print("\n  Per-category:")
    for cat, cdata in sorted(cats.items()):
        print(f"    {cat}: {cdata['disasm_ok']}/{cdata['total']} OK")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1A: Batch ASM evaluation of aiter .co kernels")
    parser.add_argument("--aiter-hsa", default="/home/root123/aiter/hsa",
                        help="Path to aiter/hsa directory")
    parser.add_argument("--arch", default="gfx942",
                        help="Target architecture (default: gfx942)")
    parser.add_argument("--output", default=str(AFTT_ROOT / "results" / "phase1_static"),
                        help="Output directory for results")
    parser.add_argument("--categories", nargs="*",
                        help="Filter to specific categories (e.g. gemm fmha moe)")
    parser.add_argument("--max-kernels", type=int, default=None,
                        help="Limit number of kernels to process")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress for every kernel")
    args = parser.parse_args()

    output_dir = Path(args.output) / args.arch
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_batch(
        args.aiter_hsa, args.arch, output_dir,
        categories=args.categories,
        max_kernels=args.max_kernels,
        verbose=args.verbose)

    write_summary(results, output_dir)


if __name__ == "__main__":
    main()
