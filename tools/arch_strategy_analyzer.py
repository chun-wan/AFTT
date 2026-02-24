#!/usr/bin/env python3
"""Phase 2C: Deep GPU Architecture and Strategy Analyzer.

For each kernel category, combines ISA utilization analysis, memory
hierarchy profiling, pipeline efficiency metrics, and algorithmic
opportunity identification to explain where AFTT wins or loses
vs aiter production kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

AFTT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AFTT_ROOT))

from src.asm_editor import AsmEditor
from src.cycle_estimator import CycleEstimator
from src.knowledge_base import KnowledgeBase
from src.instruction import Instruction


@dataclass
class ISAProfile:
    """Instruction set utilization profile."""
    total_instructions: int = 0
    mfma_count: int = 0
    mfma_types: dict = field(default_factory=dict)
    valu_count: int = 0
    salu_count: int = 0
    vmem_load_count: int = 0
    vmem_store_count: int = 0
    lds_count: int = 0
    nop_count: int = 0
    waitcnt_count: int = 0
    branch_count: int = 0
    dpp_count: int = 0
    barrier_count: int = 0
    flat_count: int = 0
    mfma_utilization: float = 0.0
    lds_bank_conflict_risk: str = "unknown"
    estimated_occupancy_limiting: str = ""


@dataclass
class MemoryProfile:
    """Memory hierarchy analysis."""
    vmem_load_bytes_est: int = 0
    vmem_store_bytes_est: int = 0
    lds_usage_pattern: str = ""
    prefetch_detected: bool = False
    double_buffer_detected: bool = False
    vectorized_loads: int = 0
    scalar_loads: int = 0
    global_to_lds_direct: int = 0


@dataclass
class PipelineProfile:
    """Pipeline efficiency metrics."""
    mfma_latency_hidden_pct: float = 0.0
    vmem_overlap_pct: float = 0.0
    barrier_overhead_pct: float = 0.0
    total_cycles: int = 0
    bottleneck: str = ""
    nop_waste_pct: float = 0.0
    waitcnt_stall_pct: float = 0.0


@dataclass
class AlgorithmicOpportunity:
    """Identified optimization opportunity."""
    opportunity: str
    mechanism: str
    potential_gain: str
    difficulty: str  # easy, medium, hard


@dataclass
class KernelArchAnalysis:
    """Complete architecture analysis for one kernel."""
    co_name: str
    co_path: str
    source: str  # aftt or aiter
    category: str
    isa: ISAProfile = field(default_factory=ISAProfile)
    memory: MemoryProfile = field(default_factory=MemoryProfile)
    pipeline: PipelineProfile = field(default_factory=PipelineProfile)
    opportunities: list = field(default_factory=list)


@dataclass
class CategoryComparison:
    """AFTT vs aiter comparison for a kernel category."""
    category: str
    num_aftt_kernels: int = 0
    num_aiter_kernels: int = 0

    # Aggregate metrics
    aftt_avg_mfma_util: float = 0.0
    aiter_avg_mfma_util: float = 0.0
    aftt_avg_cycles: float = 0.0
    aiter_avg_cycles: float = 0.0
    aftt_avg_nop_pct: float = 0.0
    aiter_avg_nop_pct: float = 0.0

    # Where AFTT wins
    aftt_advantages: list = field(default_factory=list)
    # Where aiter wins
    aiter_advantages: list = field(default_factory=list)
    # Theoretical ceiling
    theoretical_max_desc: str = ""

    # Detailed kernel analyses
    aftt_analyses: list = field(default_factory=list)
    aiter_analyses: list = field(default_factory=list)


def profile_instructions(instructions: list[Instruction]) -> ISAProfile:
    """Build detailed ISA profile from instruction list."""
    p = ISAProfile(total_instructions=len(instructions))

    for instr in instructions:
        mn = instr.mnemonic.lower()

        if mn.startswith("v_mfma_"):
            p.mfma_count += 1
            mfma_type = mn.split("_")[2] if len(mn.split("_")) > 2 else mn
            p.mfma_types[mfma_type] = p.mfma_types.get(mfma_type, 0) + 1
        elif mn.startswith("v_") and not mn.startswith("v_mfma"):
            p.valu_count += 1
            if "_dpp" in mn:
                p.dpp_count += 1
        elif mn.startswith("s_") and not mn.startswith("s_waitcnt") and mn != "s_nop" and mn != "s_barrier":
            p.salu_count += 1
        elif mn == "s_nop":
            p.nop_count += 1
        elif mn == "s_waitcnt" or mn.startswith("s_waitcnt_"):
            p.waitcnt_count += 1
        elif mn == "s_barrier":
            p.barrier_count += 1
        elif "buffer_load" in mn or "global_load" in mn:
            p.vmem_load_count += 1
        elif "buffer_store" in mn or "global_store" in mn:
            p.vmem_store_count += 1
        elif "ds_read" in mn or "ds_write" in mn or "ds_load" in mn or "ds_store" in mn:
            p.lds_count += 1
        elif mn.startswith("s_cbranch") or mn == "s_branch":
            p.branch_count += 1
        elif "flat_" in mn:
            p.flat_count += 1

    # MFMA utilization estimate
    if p.total_instructions > 0:
        p.mfma_utilization = p.mfma_count / p.total_instructions

    # LDS bank conflict risk
    if p.lds_count > 0:
        lds_ratio = p.lds_count / p.total_instructions
        if lds_ratio > 0.15:
            p.lds_bank_conflict_risk = "high"
        elif lds_ratio > 0.08:
            p.lds_bank_conflict_risk = "medium"
        else:
            p.lds_bank_conflict_risk = "low"

    return p


def profile_memory(instructions: list[Instruction]) -> MemoryProfile:
    """Analyze memory access patterns."""
    m = MemoryProfile()

    prev_was_vmem = False
    consecutive_vmem = 0
    vectorized = 0
    scalar = 0
    g2lds = 0

    for i, instr in enumerate(instructions):
        mn = instr.mnemonic.lower()
        ops = instr.operands.lower() if instr.operands else ""

        if "buffer_load" in mn or "global_load" in mn:
            # Check for vectorized loads (dwordx2, dwordx4, etc.)
            if "x4" in mn or "x3" in mn or "x2" in mn or "b128" in mn or "b64" in mn:
                vectorized += 1
            else:
                scalar += 1

            # Estimate bytes
            if "b128" in mn or "x4" in mn:
                m.vmem_load_bytes_est += 16
            elif "b64" in mn or "x2" in mn:
                m.vmem_load_bytes_est += 8
            else:
                m.vmem_load_bytes_est += 4

        if "buffer_store" in mn or "global_store" in mn:
            if "b128" in mn or "x4" in mn:
                m.vmem_store_bytes_est += 16
            elif "b64" in mn or "x2" in mn:
                m.vmem_store_bytes_est += 8
            else:
                m.vmem_store_bytes_est += 4

        # Detect direct global→LDS (if load immediately followed by ds_write)
        if ("buffer_load" in mn or "global_load" in mn):
            prev_was_vmem = True
        elif "ds_write" in mn and prev_was_vmem:
            g2lds += 1
            prev_was_vmem = False
        else:
            prev_was_vmem = False

    m.vectorized_loads = vectorized
    m.scalar_loads = scalar
    m.global_to_lds_direct = g2lds

    # Detect double buffering (two separate LDS write regions alternating)
    lds_writes = [i for i, inst in enumerate(instructions) if "ds_write" in inst.mnemonic.lower()]
    if len(lds_writes) > 10:
        m.double_buffer_detected = True
        m.lds_usage_pattern = "double_buffered"
    elif len(lds_writes) > 2:
        m.lds_usage_pattern = "tiled"
    else:
        m.lds_usage_pattern = "minimal"

    # Detect prefetching (loads issued well before corresponding waitcnt)
    vmem_loads = [i for i, inst in enumerate(instructions) if "buffer_load" in inst.mnemonic.lower() or "global_load" in inst.mnemonic.lower()]
    waitcnts = [i for i, inst in enumerate(instructions) if inst.mnemonic.lower() == "s_waitcnt"]
    if vmem_loads and waitcnts:
        avg_distance = sum(min(abs(w - l) for w in waitcnts if w > l) for l in vmem_loads if any(w > l for w in waitcnts)) / max(len(vmem_loads), 1)
        if avg_distance > 20:
            m.prefetch_detected = True

    return m


def profile_pipeline(instructions: list[Instruction],
                     estimator: CycleEstimator) -> PipelineProfile:
    """Analyze pipeline efficiency."""
    p = PipelineProfile()

    lines = [i.full_text for i in instructions]
    try:
        est = estimator.estimate(lines)
        p.total_cycles = est.total_cycles
        p.bottleneck = est.bottleneck

        if est.total_cycles > 0:
            p.nop_waste_pct = est.nop_cycles / est.total_cycles * 100
            p.waitcnt_stall_pct = est.wait_stall_cycles / est.total_cycles * 100
            p.barrier_overhead_pct = est.barrier_stall_cycles / est.total_cycles * 100 if hasattr(est, 'barrier_stall_cycles') else 0

            if est.mfma_cycles > 0:
                p.mfma_latency_hidden_pct = max(0, 100 - (est.mfma_cycles / est.total_cycles * 100))
    except Exception:
        pass

    return p


def find_opportunities(isa: ISAProfile, mem: MemoryProfile,
                       pipe: PipelineProfile) -> list[AlgorithmicOpportunity]:
    """Identify optimization opportunities."""
    ops = []

    if isa.nop_count > 0 and pipe.nop_waste_pct > 1.0:
        ops.append(AlgorithmicOpportunity(
            opportunity="NOP reduction",
            mechanism="Remove or reduce s_nop instructions by better instruction scheduling",
            potential_gain=f"~{pipe.nop_waste_pct:.1f}% cycle reduction",
            difficulty="easy"))

    if pipe.waitcnt_stall_pct > 5.0:
        ops.append(AlgorithmicOpportunity(
            opportunity="waitcnt relaxation",
            mechanism="Relax s_waitcnt counters to allow more instruction overlap",
            potential_gain=f"~{pipe.waitcnt_stall_pct:.1f}% stall reduction",
            difficulty="medium"))

    if isa.dpp_count == 0 and isa.lds_count > 20:
        ops.append(AlgorithmicOpportunity(
            opportunity="LDS→DPP conversion",
            mechanism="Replace LDS-based reductions with DPP cross-lane operations",
            potential_gain="10-30% for reduction-heavy kernels",
            difficulty="hard"))

    if mem.scalar_loads > mem.vectorized_loads * 2:
        ops.append(AlgorithmicOpportunity(
            opportunity="Load vectorization",
            mechanism="Combine scalar loads into vectorized buffer_load_dwordx4",
            potential_gain="2-4x memory throughput improvement",
            difficulty="medium"))

    if not mem.prefetch_detected and isa.vmem_load_count > 10:
        ops.append(AlgorithmicOpportunity(
            opportunity="Software prefetching",
            mechanism="Issue VMEM loads earlier in the pipeline to hide latency",
            potential_gain="Significant for memory-bound kernels",
            difficulty="hard"))

    if not mem.double_buffer_detected and isa.lds_count > 20 and isa.mfma_count > 10:
        ops.append(AlgorithmicOpportunity(
            opportunity="LDS double buffering",
            mechanism="Overlap data loading with compute using double-buffered shared memory",
            potential_gain="Up to 2x throughput for compute+memory overlap",
            difficulty="hard"))

    if isa.mfma_count > 0 and isa.mfma_utilization < 0.10:
        ops.append(AlgorithmicOpportunity(
            opportunity="MFMA chaining",
            mechanism="Increase MFMA density by reducing non-MFMA instructions in hot loop",
            potential_gain="Higher MFMA throughput utilization",
            difficulty="hard"))

    return ops


def analyze_kernel(co_path: str, source_label: str, category: str,
                   editor: AsmEditor, estimator: CycleEstimator) -> Optional[KernelArchAnalysis]:
    """Complete architecture analysis for one kernel."""
    if not co_path or not os.path.exists(co_path):
        return None

    try:
        _, instructions = editor.disassemble(co_path)
    except Exception:
        return None

    if not instructions:
        return None

    isa = profile_instructions(instructions)
    mem = profile_memory(instructions)
    pipe = profile_pipeline(instructions, estimator)
    opps = find_opportunities(isa, mem, pipe)

    return KernelArchAnalysis(
        co_name=Path(co_path).name,
        co_path=co_path,
        source=source_label,
        category=category,
        isa=isa,
        memory=mem,
        pipeline=pipe,
        opportunities=[asdict(o) for o in opps],
    )


def compare_category(category: str, aftt_cos: list[str], aiter_cos: list[str],
                     editor: AsmEditor, estimator: CycleEstimator) -> CategoryComparison:
    """Compare AFTT and aiter kernels for a category."""
    cc = CategoryComparison(category=category)

    # Analyze AFTT kernels
    aftt_analyses = []
    for co in aftt_cos:
        a = analyze_kernel(co, "aftt", category, editor, estimator)
        if a:
            aftt_analyses.append(a)
    cc.num_aftt_kernels = len(aftt_analyses)
    cc.aftt_analyses = [asdict(a) for a in aftt_analyses]

    # Analyze aiter kernels
    aiter_analyses = []
    for co in aiter_cos:
        a = analyze_kernel(co, "aiter", category, editor, estimator)
        if a:
            aiter_analyses.append(a)
    cc.num_aiter_kernels = len(aiter_analyses)
    cc.aiter_analyses = [asdict(a) for a in aiter_analyses]

    # Aggregate metrics
    if aftt_analyses:
        cc.aftt_avg_mfma_util = sum(a.isa.mfma_utilization for a in aftt_analyses) / len(aftt_analyses)
        cc.aftt_avg_cycles = sum(a.pipeline.total_cycles for a in aftt_analyses) / len(aftt_analyses)
        cc.aftt_avg_nop_pct = sum(a.pipeline.nop_waste_pct for a in aftt_analyses) / len(aftt_analyses)

    if aiter_analyses:
        cc.aiter_avg_mfma_util = sum(a.isa.mfma_utilization for a in aiter_analyses) / len(aiter_analyses)
        cc.aiter_avg_cycles = sum(a.pipeline.total_cycles for a in aiter_analyses) / len(aiter_analyses)
        cc.aiter_avg_nop_pct = sum(a.pipeline.nop_waste_pct for a in aiter_analyses) / len(aiter_analyses)

    # Determine advantages
    if cc.aftt_avg_mfma_util > cc.aiter_avg_mfma_util * 1.05 and cc.aftt_avg_mfma_util > 0:
        cc.aftt_advantages.append(
            f"Higher MFMA utilization ({cc.aftt_avg_mfma_util:.1%} vs {cc.aiter_avg_mfma_util:.1%})")
    if cc.aiter_avg_mfma_util > cc.aftt_avg_mfma_util * 1.05 and cc.aiter_avg_mfma_util > 0:
        cc.aiter_advantages.append(
            f"Higher MFMA utilization ({cc.aiter_avg_mfma_util:.1%} vs {cc.aftt_avg_mfma_util:.1%})")

    if cc.aftt_avg_nop_pct < cc.aiter_avg_nop_pct * 0.8:
        cc.aftt_advantages.append(
            f"Lower NOP waste ({cc.aftt_avg_nop_pct:.1f}% vs {cc.aiter_avg_nop_pct:.1f}%)")
    if cc.aiter_avg_nop_pct < cc.aftt_avg_nop_pct * 0.8:
        cc.aiter_advantages.append(
            f"Lower NOP waste ({cc.aiter_avg_nop_pct:.1f}% vs {cc.aftt_avg_nop_pct:.1f}%)")

    if cc.aftt_avg_cycles > 0 and cc.aiter_avg_cycles > 0:
        if cc.aftt_avg_cycles < cc.aiter_avg_cycles * 0.95:
            cc.aftt_advantages.append(
                f"Lower estimated cycles ({cc.aftt_avg_cycles:.0f} vs {cc.aiter_avg_cycles:.0f})")
        if cc.aiter_avg_cycles < cc.aftt_avg_cycles * 0.95:
            cc.aiter_advantages.append(
                f"Lower estimated cycles ({cc.aiter_avg_cycles:.0f} vs {cc.aftt_avg_cycles:.0f})")

    # Theoretical ceiling description
    if category in ("bf16gemm", "fp8gemm_blockscale", "i8gemm", "f4gemm"):
        cc.theoretical_max_desc = (
            "Compute-bound: theoretical max limited by MFMA throughput "
            "(e.g. 4 MFMA units × 64 ops/cycle = 256 FP16 ops/cycle on MI300X). "
            "Key: maximize MFMA occupancy, hide VMEM latency with prefetch.")
    elif category in ("fmha_v3_fwd", "fmha_v3_bwd"):
        cc.theoretical_max_desc = (
            "Mixed compute+memory: attention is O(N²d) compute with O(Nd) memory. "
            "Flash Attention tiling critical. Key: tile QKV in LDS, "
            "online softmax, maximize MFMA chains per tile.")
    elif category in ("fmoe", "fmoe_2stages"):
        cc.theoretical_max_desc = (
            "Mixed: token routing is memory-bound, expert GEMM is compute-bound. "
            "Key: minimize routing overhead, maximize GEMM utilization per expert.")
    elif category in ("pa",):
        cc.theoretical_max_desc = (
            "Memory-bound for decode (single query per head), bandwidth-limited. "
            "Key: vectorized KV-cache loads, warp-level reductions, "
            "minimize global memory traffic.")
    elif category in ("mla",):
        cc.theoretical_max_desc = (
            "Memory-bound with compressed KV-cache. Key: efficient latent decomposition, "
            "maximize memory bandwidth utilization for large batch sizes.")
    elif category in ("topksoftmax",):
        cc.theoretical_max_desc = (
            "Mostly memory-bound. Key: warp-level TopK with DPP, "
            "fused softmax to avoid extra global memory round-trips.")
    else:
        cc.theoretical_max_desc = "Architecture-specific ceiling depends on kernel's compute/memory ratio."

    return cc


def run_analysis(output_dir: Path, arch: str = "gfx942",
                 hipify_dir: Optional[Path] = None,
                 max_per_category: int = 5):
    """Run full architecture analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    editor = AsmEditor(arch=arch)
    estimator = CycleEstimator(arch=arch)

    # Collect aiter .co files by category
    aiter_base = Path(f"/home/root123/aiter/hsa/{arch}")
    aiter_cos_by_cat = defaultdict(list)
    if aiter_base.exists():
        for co in sorted(aiter_base.rglob("*.co")):
            rel = co.relative_to(aiter_base)
            cat = str(rel).split("/")[0] if "/" in str(rel) else "other"
            aiter_cos_by_cat[cat].append(str(co))

    # Collect AFTT .co files from hipify results
    aftt_cos_by_cat = defaultdict(list)
    if hipify_dir and hipify_dir.exists():
        for rf in hipify_dir.glob("*_result.json"):
            try:
                data = json.load(open(rf))
                cat = data.get("category_guess", "other")
                for key in ["co_c_path", "co_b_path", "co_a_path"]:
                    co = data.get(key, "")
                    if co and os.path.exists(co):
                        aftt_cos_by_cat[cat].append(co)
                        break
            except Exception:
                pass

    all_cats = sorted(set(aiter_cos_by_cat.keys()) | set(aftt_cos_by_cat.keys()))
    print(f"Analyzing {len(all_cats)} categories...")

    comparisons = []
    for cat in all_cats:
        aiter_cos = aiter_cos_by_cat.get(cat, [])[:max_per_category]
        aftt_cos = aftt_cos_by_cat.get(cat, [])[:max_per_category]

        if not aiter_cos and not aftt_cos:
            continue

        print(f"  {cat}: {len(aftt_cos)} AFTT, {len(aiter_cos)} aiter kernels")
        cc = compare_category(cat, aftt_cos, aiter_cos, editor, estimator)
        comparisons.append(cc)

        # Save per-category
        cat_path = output_dir / f"{cat}_analysis.json"
        with open(cat_path, "w") as f:
            json.dump(asdict(cc), f, indent=2, default=str)

    # Generate summary report
    report_lines = ["# GPU Architecture and Strategy Analysis\n"]
    report_lines.append(f"Architecture: {arch}")
    report_lines.append(f"Categories analyzed: {len(comparisons)}\n")

    for cc in comparisons:
        report_lines.append(f"## {cc.category}\n")
        report_lines.append(f"- AFTT kernels: {cc.num_aftt_kernels}")
        report_lines.append(f"- aiter kernels: {cc.num_aiter_kernels}")

        if cc.aftt_avg_cycles > 0:
            report_lines.append(f"- AFTT avg cycles: {cc.aftt_avg_cycles:.0f}")
        if cc.aiter_avg_cycles > 0:
            report_lines.append(f"- aiter avg cycles: {cc.aiter_avg_cycles:.0f}")

        if cc.aftt_advantages:
            report_lines.append(f"\n**AFTT advantages:**")
            for adv in cc.aftt_advantages:
                report_lines.append(f"  - {adv}")
        if cc.aiter_advantages:
            report_lines.append(f"\n**aiter advantages:**")
            for adv in cc.aiter_advantages:
                report_lines.append(f"  - {adv}")

        report_lines.append(f"\n**Theoretical ceiling:** {cc.theoretical_max_desc}")

        # Top opportunities
        all_opps = []
        for a_data in cc.aftt_analyses + cc.aiter_analyses:
            for opp in a_data.get("opportunities", []):
                all_opps.append(opp)
        if all_opps:
            unique_opps = {}
            for o in all_opps:
                unique_opps[o["opportunity"]] = o
            report_lines.append(f"\n**Optimization opportunities:**")
            for opp in sorted(unique_opps.values(), key=lambda x: x["difficulty"]):
                report_lines.append(
                    f"  - [{opp['difficulty']}] {opp['opportunity']}: "
                    f"{opp['mechanism']} ({opp['potential_gain']})")

        report_lines.append("")

    report_path = output_dir / "arch_analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nArchitecture analysis report: {report_path}")
    print(f"({len(comparisons)} categories analyzed)")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2C: Deep GPU architecture analysis")
    parser.add_argument("--output",
                        default=str(AFTT_ROOT / "results" / "phase2" / "arch_analysis"),
                        help="Output directory")
    parser.add_argument("--arch", default="gfx942")
    parser.add_argument("--hipify-results",
                        default=str(AFTT_ROOT / "results" / "phase2" / "hipify"))
    parser.add_argument("--max-per-category", type=int, default=5,
                        help="Max kernels to analyze per category")
    args = parser.parse_args()

    run_analysis(
        Path(args.output), args.arch,
        hipify_dir=Path(args.hipify_results) if args.hipify_results else None,
        max_per_category=args.max_per_category)


if __name__ == "__main__":
    main()
