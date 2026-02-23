#!/usr/bin/env python3
"""Deep CK Pipeline Pattern Analyzer.

Deeply analyzes Composable Kernel pipeline headers to extract:
- sched_group_barrier mask patterns and their meanings
- Ping-pong / double-buffer strategies
- PrefetchStages and their effects
- HotLoop scheduler instruction ratios
- Tail handling strategies
"""

import json
import re
from pathlib import Path

AITER_ROOT = Path("/home/root123/aiter")
CK_BASE = AITER_ROOT / "3rdparty" / "composable_kernel" / "include" / "ck_tile" / "ops"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "db" / "patterns"

SCHED_BARRIER_MASKS = {
    0x008: "MFMA (matrix fused multiply-add)",
    0x020: "VMEM_READ (global/buffer memory load)",
    0x040: "VMEM_WRITE (global/buffer memory store)",
    0x100: "DS_READ (LDS load)",
    0x200: "DS_WRITE (LDS store)",
}

RE_SCHED_GROUP_BARRIER = re.compile(
    r'__builtin_amdgcn_sched_group_barrier\(\s*(0x[0-9a-fA-F]+)\s*,\s*([^,]+?)\s*,\s*(\d+)\s*\)'
)
RE_SCHED_BARRIER = re.compile(r'__builtin_amdgcn_sched_barrier\(\s*(\d+)\s*\)')
RE_PREFETCH_STAGES = re.compile(r'PrefetchStages\s*=\s*(\d+)')
RE_PREFILL_STAGES = re.compile(r'PrefillStages\s*=\s*(\d+)')
RE_GLOBAL_BUFFER_NUM = re.compile(r'GlobalBufferNum\s*=\s*(\d+)')
RE_HOTLOOP_UNROLL = re.compile(r'HotloopUnroll\s*=\s*(\d+)')
RE_BLOCK_SYNC = re.compile(r'block_sync_lds\(\)')
RE_DOUBLE_SMEM = re.compile(r'DoubleSmemBuffer')
RE_PING_PONG = re.compile(r'(ping|pong|block0|block1)', re.IGNORECASE)
RE_TAIL_NUMBER = re.compile(r'TailNumber::(One|Two|Three|Odd|Even|Empty)')


def analyze_pipeline_file(filepath: Path) -> dict:
    """Analyze a single CK pipeline header file."""
    text = filepath.read_text()
    lines = text.split("\n")

    result = {
        "file": str(filepath.relative_to(AITER_ROOT)),
        "pipeline_name": filepath.stem,
        "parameters": {},
        "sched_group_barriers": [],
        "sched_barrier_calls": 0,
        "buffer_strategy": "unknown",
        "ping_pong_detected": False,
        "tail_variants": [],
        "block_sync_count": 0,
        "scheduler_structure": {},
    }

    # Extract pipeline parameters
    for m in RE_PREFETCH_STAGES.finditer(text):
        result["parameters"]["PrefetchStages"] = int(m.group(1))
    for m in RE_PREFILL_STAGES.finditer(text):
        result["parameters"]["PrefillStages"] = int(m.group(1))
    for m in RE_GLOBAL_BUFFER_NUM.finditer(text):
        result["parameters"]["GlobalBufferNum"] = int(m.group(1))
    for m in RE_HOTLOOP_UNROLL.finditer(text):
        result["parameters"]["HotloopUnroll"] = int(m.group(1))

    # Extract sched_group_barrier calls
    barrier_by_mask = {}
    for m in RE_SCHED_GROUP_BARRIER.finditer(text):
        mask_val = int(m.group(1), 16)
        count_expr = m.group(2).strip()
        mask_name = SCHED_BARRIER_MASKS.get(mask_val, f"UNKNOWN(0x{mask_val:03x})")

        entry = {
            "mask_hex": f"0x{mask_val:03x}",
            "mask_name": mask_name,
            "count_expr": count_expr,
        }
        result["sched_group_barriers"].append(entry)
        barrier_by_mask.setdefault(mask_name, 0)
        barrier_by_mask[mask_name] += 1

    result["sched_barrier_summary"] = barrier_by_mask

    # Count plain sched_barrier calls
    result["sched_barrier_calls"] = len(RE_SCHED_BARRIER.findall(text))

    # Detect buffer strategy
    global_buf = result["parameters"].get("GlobalBufferNum", 1)
    has_double_smem = bool(RE_DOUBLE_SMEM.search(text))
    has_ping_pong = bool(RE_PING_PONG.search(text))

    if has_ping_pong:
        result["ping_pong_detected"] = True
        if global_buf >= 2:
            result["buffer_strategy"] = "double_buffer_ping_pong"
        else:
            result["buffer_strategy"] = "single_buffer_ping_pong_lds"
    elif global_buf >= 2:
        result["buffer_strategy"] = "double_global_buffer"
    elif has_double_smem:
        result["buffer_strategy"] = "double_smem_buffer"
    else:
        result["buffer_strategy"] = "single_buffer"

    # Count sync points
    result["block_sync_count"] = len(RE_BLOCK_SYNC.findall(text))

    # Extract tail variants
    tail_variants = set()
    for m in RE_TAIL_NUMBER.finditer(text):
        tail_variants.add(m.group(1))
    result["tail_variants"] = sorted(tail_variants)

    # Detect warp specialization
    result["warp_specialization"] = "NumWaveGroups" in text or "warp_id" in text and "MemoryOpsStep" in text

    # Detect key features
    result["features"] = {
        "uses_sched_group_barrier": len(result["sched_group_barriers"]) > 0,
        "uses_sched_barrier": result["sched_barrier_calls"] > 0,
        "has_ping_pong": has_ping_pong,
        "has_double_smem": has_double_smem,
        "has_block_sync": result["block_sync_count"] > 0,
        "has_tail_handling": len(tail_variants) > 0,
        "has_warp_specialization": result["warp_specialization"],
        "prefetch_stages": result["parameters"].get("PrefetchStages", 0),
    }

    return result


def build_pipeline_comparison() -> list[dict]:
    """Build a structured comparison of all CK pipeline variants."""
    return [
        {
            "pipeline": "gemm_pipeline_ag_bg_cr_comp_v3",
            "strategy": "single_buffer_compute_optimized",
            "prefetch_stages": 2,
            "prefill_stages": 1,
            "global_buffers": 1,
            "lds_buffers": 1,
            "sched_group_barrier_masks": ["0x008 (MFMA)", "0x100 (DS_READ)", "0x200 (DS_WRITE)", "0x020 (VMEM_READ)"],
            "scheduler_structure": "Two-stage: (1) DS_WRITE+VMEM_READ interleaved with MFMA, (2) DS_READ interleaved with MFMA",
            "hotloop": "do-while with num_loop - 1 iterations. Each iteration: sync -> prefill LDS -> global_load next -> block_gemm -> sync -> local_prefetch -> HotLoopScheduler",
            "tail_handling": ["Odd", "Even"],
            "key_insight": "Single LDS buffer requires block_sync_lds between write and read phases. Scheduling barriers ensure DS writes complete before VMEM reads are issued.",
            "performance_tradeoff": "Half the LDS usage of v4 but requires more synchronization barriers.",
        },
        {
            "pipeline": "gemm_pipeline_ag_bg_cr_comp_v4",
            "strategy": "double_buffer_ping_pong",
            "prefetch_stages": 2,
            "prefill_stages": 1,
            "global_buffers": 1,
            "lds_buffers": 2,
            "sched_group_barrier_masks": ["0x008 (MFMA)", "0x100 (DS_READ)", "0x200 (DS_WRITE)", "0x020 (VMEM_READ)"],
            "scheduler_structure": "Single-stage per VMEM issue: MFMA(1) -> DS_READ(N) -> MFMA(1) -> DS_WRITE(N) -> MFMA(1) -> VMEM(1) -> MFMA(remaining)",
            "hotloop": "do-while with iCounter = num_loop - 2. Each iteration has ping phase (read block1, fill block0, compute block0) then pong phase (read block0, fill block1, compute block1). iCounter -= 2 per iteration.",
            "tail_handling": ["One", "Two", "Three"],
            "key_insight": "Two LDS buffers (a_lds_block0/1, b_lds_block0/1) at offset smem_size. While computing on one buffer, the other is being filled. Eliminates read-after-write hazard without sync.",
            "performance_tradeoff": "2x LDS usage but significantly reduced sync overhead. Best for large tile sizes where LDS is not the bottleneck.",
        },
        {
            "pipeline": "gemm_pipeline_ag_bg_cr_comp_v5",
            "strategy": "warp_specialization",
            "prefetch_stages": 1,
            "prefill_stages": 1,
            "global_buffers": 1,
            "lds_buffers": 1,
            "sched_group_barrier_masks": [],
            "scheduler_structure": "No instruction-level scheduling. Uses warp-level ping-pong: warp 0 does memory ops while warp 1 computes, then they swap.",
            "hotloop": "while(num_compute_steps > 1): sync -> warp 0: MemoryOpsStep / warp 1: ComputeStep -> operation_id alternates",
            "tail_handling": ["Empty"],
            "key_insight": "Warp specialization: different warps assigned different roles (memory vs compute). No __builtin_amdgcn_sched_group_barrier needed. Two register tile sets for alternating computation.",
            "performance_tradeoff": "Simpler scheduling but requires careful warp role assignment. Each warp only does half the work types.",
        },
        {
            "pipeline": "gemm_pipeline_ag_bg_cr_comp_v6",
            "strategy": "triple_prefetch_double_buffer",
            "prefetch_stages": 3,
            "prefill_stages": 1,
            "global_buffers": 2,
            "lds_buffers": 1,
            "sched_group_barrier_masks": ["0x008 (MFMA)", "0x100 (DS_READ)", "0x200 (DS_WRITE)", "0x020 (VMEM_READ)"],
            "scheduler_structure": "Three-stage: (1) DS_READ_A + MFMA, DS_READ_B + MFMA, (2) DS_WRITE + VMEM_READ + MFMA for A and B, (3) DS_READ_A + MFMA, DS_READ_B + MFMA",
            "hotloop": "do-while with HotloopUnroll=2. Each iteration: LoopFunc(I0) then LoopFunc(I1). Inner K-loop with KRepeat steps. Prefill/prefetch at k0 == KRepeat-1.",
            "tail_handling": ["Odd", "Even"],
            "key_insight": "Three prefetch stages allow deeper memory latency hiding. Double global buffers (I0/I1) with HotloopUnroll=2 processes two K-tiles per iteration. Three-stage scheduler separates read and write phases more explicitly.",
            "performance_tradeoff": "Highest latency hiding but requires 3 pipeline stages before hot loop starts. Best for large GEMM shapes where the overhead of pipeline fill is amortized.",
        },
        {
            "pipeline": "flatmm_pipeline_agmem_bgmem_creg_v1",
            "strategy": "mixed_ping_pong",
            "prefetch_stages": 2,
            "prefill_stages": 1,
            "global_buffers": 1,
            "lds_buffers": 2,
            "sched_group_barrier_masks": ["0x008 (MFMA)", "0x100 (DS_READ)", "0x200 (DS_WRITE)", "0x020 (VMEM_READ)"],
            "scheduler_structure": "Per-M scheduling via SchedulerPerM: for each MFMA, interleave DS_WRITE(1), VMEM_READ(1), DS_READ(1) based on instruction order. K x M nested loop.",
            "hotloop": "while(iCounter > 0) with 2 K-steps per iteration. A: LDS ping/pong (p_smem_ping, p_smem_pong). B: register ping/pong (b_warp_tensor_ping/pong). GEMM alternates between ping and pong data.",
            "tail_handling": ["Odd", "Even"],
            "key_insight": "A matrix goes through DRAM->reg->LDS(ping/pong)->reg->GEMM. B matrix stays in registers with ping/pong register tiles. Asymmetric strategy: LDS double-buffer for A, register double-buffer for B.",
            "performance_tradeoff": "Optimized for flat matrix multiply where B is reused across M tiles. Register double-buffering for B avoids LDS bandwidth contention.",
        },
        {
            "pipeline": "gemm_pipeline_ag_bg_cr_comp_async",
            "strategy": "async_copy",
            "prefetch_stages": 2,
            "prefill_stages": 1,
            "global_buffers": 1,
            "lds_buffers": 1,
            "sched_group_barrier_masks": [],
            "scheduler_structure": "Uses async copy intrinsics for global-to-LDS transfers, overlapping with compute.",
            "hotloop": "Similar to v3 but with async copy for memory transfers.",
            "tail_handling": ["Odd", "Even"],
            "key_insight": "Async copy allows global memory loads to be issued and completed in the background without explicit waitcnt management.",
            "performance_tradeoff": "Requires hardware support for async copy. Simplifies scheduling but may have limitations on copy granularity.",
        },
    ]


def build_sched_barrier_reference() -> dict:
    """Build a reference for __builtin_amdgcn_sched_group_barrier usage."""
    return {
        "description": "__builtin_amdgcn_sched_group_barrier(mask, count, flags) constrains the instruction scheduler to issue exactly 'count' instructions matching 'mask' before proceeding.",
        "masks": {
            "0x008": {
                "name": "MFMA",
                "instructions": "v_mfma_*, v_smfma_*",
                "purpose": "Control matrix multiply-accumulate instruction placement",
            },
            "0x020": {
                "name": "VMEM_READ",
                "instructions": "global_load_*, buffer_load_*",
                "purpose": "Control global/buffer memory load placement",
            },
            "0x040": {
                "name": "VMEM_WRITE",
                "instructions": "global_store_*, buffer_store_*",
                "purpose": "Control global/buffer memory store placement",
            },
            "0x100": {
                "name": "DS_READ",
                "instructions": "ds_read_*, ds_load_*",
                "purpose": "Control LDS read placement",
            },
            "0x200": {
                "name": "DS_WRITE",
                "instructions": "ds_write_*, ds_store_*",
                "purpose": "Control LDS write placement",
            },
        },
        "common_patterns": [
            {
                "name": "MFMA-DS_READ interleave",
                "sequence": "sched_group_barrier(0x008, 1) -> sched_group_barrier(0x100, N)",
                "purpose": "Issue 1 MFMA then N DS reads, hiding MFMA latency with LDS access",
            },
            {
                "name": "MFMA-DS_WRITE-VMEM pipeline",
                "sequence": "sched_group_barrier(0x200, 1) -> sched_group_barrier(0x008, 1) -> sched_group_barrier(0x020, 1)",
                "purpose": "Interleave LDS store, MFMA compute, and global memory prefetch",
            },
            {
                "name": "Full barrier",
                "sequence": "__builtin_amdgcn_sched_barrier(0)",
                "purpose": "Reset scheduler state, ensuring all prior scheduled groups are issued",
            },
        ],
        "usage_guidelines": [
            "Use sched_group_barrier to explicitly control instruction interleaving in hot loops",
            "The 'count' parameter determines how many instructions of that type must be issued",
            "Pair with sched_barrier(0) at loop boundaries to reset scheduler state",
            "Ratios should match the actual instruction mix: e.g., if you have 8 MFMA and 2 DS_READ per iteration, schedule ratio 4:1",
            "Too many barriers can hurt performance by over-constraining the scheduler",
        ],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Analyze all pipeline files
    pipeline_dirs = [
        CK_BASE / "gemm" / "pipeline",
        CK_BASE / "flatmm" / "pipeline",
    ]

    all_analyses = []
    for pdir in pipeline_dirs:
        if not pdir.exists():
            continue
        for hpp in sorted(pdir.glob("*.hpp")):
            if "default_policy" in hpp.name or "problem" in hpp.name or "shape" in hpp.name or "traits" in hpp.name or "pipelines.hpp" == hpp.name:
                continue
            analysis = analyze_pipeline_file(hpp)
            all_analyses.append(analysis)
            print(f"  Analyzed {analysis['pipeline_name']}: {len(analysis['sched_group_barriers'])} sched_group_barrier calls, buffer={analysis['buffer_strategy']}")

    # Build output
    output = {
        "total_pipelines_analyzed": len(all_analyses),
        "pipeline_analyses": all_analyses,
        "pipeline_comparison": build_pipeline_comparison(),
        "sched_barrier_reference": build_sched_barrier_reference(),
        "key_optimization_patterns": [
            {
                "name": "Ping-Pong Double Buffer (LDS)",
                "description": "Allocate 2x LDS space. While computing from buffer A, fill buffer B from global memory. Swap on next iteration.",
                "used_in": ["v4", "flatmm_v1"],
                "lds_cost": "2x base",
                "benefit": "Eliminates read-after-write sync between LDS fill and GEMM compute",
                "code_pattern": "a_lds_block0 = GetABLdsTensorViews(p_smem); a_lds_block1 = GetABLdsTensorViews(p_smem + smem_size);",
            },
            {
                "name": "Register Double Buffer",
                "description": "Keep two sets of register tiles (ping/pong). While computing with one set, prefetch into the other.",
                "used_in": ["flatmm_v1 (for B matrix)", "v5 (for both)"],
                "lds_cost": "1x base",
                "benefit": "Avoids LDS bandwidth contention for frequently reused data",
                "code_pattern": "b_warp_tensor_ping / b_warp_tensor_pong",
            },
            {
                "name": "Warp Specialization",
                "description": "Assign different warps to different roles: memory warps load data, compute warps run MFMA.",
                "used_in": ["v5"],
                "lds_cost": "1x base",
                "benefit": "No instruction-level scheduling needed. Warps can be independently optimized.",
                "code_pattern": "if(operation_id == 0) MemoryOpsStep(warp_id); else ComputeStep(warp_id);",
            },
            {
                "name": "Deep Prefetch Pipeline",
                "description": "Issue 3+ prefetch stages before the hot loop starts to fully hide memory latency.",
                "used_in": ["v6 (PrefetchStages=3)"],
                "lds_cost": "1-2x base",
                "benefit": "Deeper pipeline hides longer memory latencies. Best for large GEMM.",
                "code_pattern": "GlobalPrefetch1 -> GlobalPrefetch2 -> LocalPrefill -> GlobalPrefetch3 -> HotLoop",
            },
            {
                "name": "Instruction Scheduling Barriers",
                "description": "Use __builtin_amdgcn_sched_group_barrier to explicitly control which instruction types are issued in what order.",
                "used_in": ["v3", "v4", "v6", "flatmm_v1"],
                "lds_cost": "none",
                "benefit": "Prevents compiler from reordering instructions in ways that cause pipeline stalls",
                "code_pattern": "sched_group_barrier(0x008, 1, 0); sched_group_barrier(0x100, N, 0);",
            },
        ],
    }

    out_file = OUTPUT_DIR / "ck_deep_patterns.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CK Deep Pattern Analysis Complete")
    print(f"  Pipelines analyzed: {len(all_analyses)}")
    print(f"  Comparison table entries: {len(output['pipeline_comparison'])}")
    print(f"  Optimization patterns cataloged: {len(output['key_optimization_patterns'])}")
    print(f"  Output: {out_file}")


if __name__ == "__main__":
    main()
