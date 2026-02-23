#!/usr/bin/env python3
"""Deep ASM Pattern Analyzer for disassembled .co kernels.

Processes all disassembled kernels to extract deep patterns:
- MFMA chaining and interleaving
- LDS double-buffer detection
- Prefetch depth analysis
- Scheduling patterns (s_nop, s_setprio, waitcnt)
- Architecture differences (gfx942 vs gfx950)
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict

DISASM_DIR = Path(__file__).resolve().parent.parent / "db" / "disassembly"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "db" / "patterns"

RE_INSTRUCTION = re.compile(r'^\s+([a-z_]\S+)\s*(.*?)(?:\s*//.*)?$')
RE_MFMA = re.compile(r'v_mfma|v_smfma')
RE_DS_READ = re.compile(r'ds_read|ds_load')
RE_DS_WRITE = re.compile(r'ds_write|ds_store')
RE_GLOBAL_LOAD = re.compile(r'global_load')
RE_BUFFER_LOAD = re.compile(r'buffer_load')
RE_WAITCNT = re.compile(r's_waitcnt')
RE_NOP = re.compile(r's_nop')
RE_SETPRIO = re.compile(r's_setprio')
RE_BARRIER = re.compile(r's_barrier')
RE_VMCNT = re.compile(r'vmcnt\((\d+)\)')
RE_LGKMCNT = re.compile(r'lgkmcnt\((\d+)\)')
RE_DS_OFFSET = re.compile(r'offset:(\d+|0x[0-9a-fA-F]+)')


def parse_instructions(asm_text: str) -> list[dict]:
    """Parse ASM into instruction list."""
    instrs = []
    for i, line in enumerate(asm_text.split("\n")):
        m = RE_INSTRUCTION.match(line)
        if not m:
            continue
        mnemonic = m.group(1)
        operands = m.group(2).strip()
        instrs.append({"line": i, "mnemonic": mnemonic, "operands": operands})
    return instrs


def analyze_mfma_chains(instrs: list[dict]) -> dict:
    """Analyze MFMA chaining patterns."""
    chains = []
    current_chain = 0
    chain_types = defaultdict(int)
    mfma_interleave_counts = {"ds_read": 0, "ds_write": 0, "vmem": 0, "valu": 0, "other": 0}
    between_mfma = []
    in_mfma_region = False

    for instr in instrs:
        mn = instr["mnemonic"]
        if RE_MFMA.match(mn):
            if in_mfma_region and between_mfma:
                for bi in between_mfma:
                    bmn = bi["mnemonic"]
                    if RE_DS_READ.match(bmn):
                        mfma_interleave_counts["ds_read"] += 1
                    elif RE_DS_WRITE.match(bmn):
                        mfma_interleave_counts["ds_write"] += 1
                    elif RE_GLOBAL_LOAD.match(bmn) or RE_BUFFER_LOAD.match(bmn):
                        mfma_interleave_counts["vmem"] += 1
                    elif bmn.startswith("v_"):
                        mfma_interleave_counts["valu"] += 1
                    else:
                        mfma_interleave_counts["other"] += 1
            current_chain += 1
            chain_types[mn] += 1
            between_mfma = []
            in_mfma_region = True
        else:
            if in_mfma_region:
                between_mfma.append(instr)
                if RE_BARRIER.match(mn) or RE_WAITCNT.match(mn):
                    if current_chain > 0:
                        chains.append(current_chain)
                    current_chain = 0
                    in_mfma_region = False
                    between_mfma = []

    if current_chain > 0:
        chains.append(current_chain)

    return {
        "chain_lengths": chains,
        "max_chain": max(chains) if chains else 0,
        "avg_chain": sum(chains) / len(chains) if chains else 0,
        "total_chains": len(chains),
        "mfma_types": dict(chain_types),
        "interleave_between_mfma": dict(mfma_interleave_counts),
    }


def analyze_lds_patterns(instrs: list[dict]) -> dict:
    """Analyze LDS access patterns for double-buffering detection."""
    ds_read_offsets = []
    ds_write_offsets = []

    for instr in instrs:
        mn = instr["mnemonic"]
        m = RE_DS_OFFSET.search(instr["operands"])
        offset_val = 0
        if m:
            offset_str = m.group(1)
            offset_val = int(offset_str, 16) if offset_str.startswith("0x") else int(offset_str)

        if RE_DS_READ.match(mn):
            ds_read_offsets.append(offset_val)
        elif RE_DS_WRITE.match(mn):
            ds_write_offsets.append(offset_val)

    # Detect double-buffering: check for two distinct offset clusters
    all_offsets = sorted(set(ds_read_offsets + ds_write_offsets))
    has_double_buffer = False
    buffer_boundary = 0

    if len(all_offsets) >= 4:
        max_off = max(all_offsets) if all_offsets else 0
        if max_off > 0:
            midpoint = max_off // 2
            low_count = sum(1 for o in all_offsets if o < midpoint)
            high_count = sum(1 for o in all_offsets if o >= midpoint)
            if low_count > 0 and high_count > 0 and min(low_count, high_count) / max(low_count, high_count) > 0.3:
                has_double_buffer = True
                buffer_boundary = midpoint

    return {
        "ds_read_count": len(ds_read_offsets),
        "ds_write_count": len(ds_write_offsets),
        "unique_offsets": len(all_offsets),
        "max_offset": max(all_offsets) if all_offsets else 0,
        "has_double_buffer": has_double_buffer,
        "buffer_boundary": buffer_boundary,
        "estimated_lds_bytes": (max(all_offsets) + 128) if all_offsets else 0,
    }


def analyze_prefetch_depth(instrs: list[dict]) -> dict:
    """Analyze prefetch depth: how many loads before first wait."""
    loads_before_first_wait = 0
    found_load = False
    found_wait = False
    load_wait_gaps = []
    current_gap = 0

    for instr in instrs:
        mn = instr["mnemonic"]
        if RE_GLOBAL_LOAD.match(mn) or RE_BUFFER_LOAD.match(mn):
            if not found_wait:
                loads_before_first_wait += 1
            found_load = True
            current_gap += 1
        elif RE_WAITCNT.match(mn) and found_load:
            found_wait = True
            if current_gap > 0:
                load_wait_gaps.append(current_gap)
            current_gap = 0

    return {
        "loads_before_first_wait": loads_before_first_wait,
        "avg_loads_between_waits": sum(load_wait_gaps) / len(load_wait_gaps) if load_wait_gaps else 0,
        "max_loads_between_waits": max(load_wait_gaps) if load_wait_gaps else 0,
        "total_load_wait_transitions": len(load_wait_gaps),
    }


def analyze_scheduling(instrs: list[dict]) -> dict:
    """Analyze scheduling patterns: NOPs, priorities, partial waitcnts."""
    nop_positions = []
    setprio_values = []
    partial_waitcnts = 0
    full_waitcnts = 0
    nop_near_mfma = 0

    for i, instr in enumerate(instrs):
        mn = instr["mnemonic"]

        if RE_NOP.match(mn):
            nop_positions.append(i)
            # Check if near MFMA
            for j in range(max(0, i-3), min(len(instrs), i+4)):
                if RE_MFMA.match(instrs[j]["mnemonic"]):
                    nop_near_mfma += 1
                    break

        if RE_SETPRIO.match(mn):
            try:
                val = int(instr["operands"].strip())
                setprio_values.append(val)
            except ValueError:
                pass

        if RE_WAITCNT.match(mn):
            ops = instr["operands"]
            vm = RE_VMCNT.search(ops)
            lgkm = RE_LGKMCNT.search(ops)
            is_partial = False
            if vm and int(vm.group(1)) > 0:
                is_partial = True
            if lgkm and int(lgkm.group(1)) > 0:
                is_partial = True
            if is_partial:
                partial_waitcnts += 1
            else:
                full_waitcnts += 1

    return {
        "nop_count": len(nop_positions),
        "nop_near_mfma": nop_near_mfma,
        "setprio_values": setprio_values,
        "uses_priority_scheduling": len(setprio_values) > 0,
        "partial_waitcnts": partial_waitcnts,
        "full_waitcnts": full_waitcnts,
        "partial_waitcnt_ratio": partial_waitcnts / max(partial_waitcnts + full_waitcnts, 1),
    }


def analyze_vectorization(instrs: list[dict]) -> dict:
    """Analyze memory access vectorization level."""
    loads = {"dword": 0, "dwordx2": 0, "dwordx3": 0, "dwordx4": 0, "short": 0, "byte": 0}

    for instr in instrs:
        mn = instr["mnemonic"]
        if "load" not in mn:
            continue
        if "dwordx4" in mn:
            loads["dwordx4"] += 1
        elif "dwordx3" in mn:
            loads["dwordx3"] += 1
        elif "dwordx2" in mn:
            loads["dwordx2"] += 1
        elif "dword" in mn:
            loads["dword"] += 1
        elif "short" in mn or "u16" in mn or "b16" in mn:
            loads["short"] += 1
        elif "byte" in mn or "u8" in mn or "b8" in mn:
            loads["byte"] += 1

    total = sum(loads.values())
    wide = loads["dwordx4"] + loads["dwordx3"]
    return {
        "load_widths": loads,
        "total_loads": total,
        "vectorization_pct": 100 * wide / max(total, 1),
    }


def analyze_arch_differences(gfx942_stats: list[dict], gfx950_stats: list[dict]) -> dict:
    """Compare patterns between gfx942 and gfx950."""
    def avg_stat(stats_list, key):
        vals = [s.get(key, 0) for s in stats_list if s.get(key, 0) > 0]
        return sum(vals) / len(vals) if vals else 0

    gfx950_mfma_types = set()
    gfx942_mfma_types = set()
    for s in gfx942_stats:
        gfx942_mfma_types.update(s.get("mfma_chains", {}).get("mfma_types", {}).keys())
    for s in gfx950_stats:
        gfx950_mfma_types.update(s.get("mfma_chains", {}).get("mfma_types", {}).keys())

    new_in_950 = gfx950_mfma_types - gfx942_mfma_types

    return {
        "gfx942_kernel_count": len(gfx942_stats),
        "gfx950_kernel_count": len(gfx950_stats),
        "gfx942_mfma_types": sorted(gfx942_mfma_types),
        "gfx950_mfma_types": sorted(gfx950_mfma_types),
        "new_mfma_in_gfx950": sorted(new_in_950),
        "gfx942_avg_mfma_chain": avg_stat(
            [s.get("mfma_chains", {}) for s in gfx942_stats], "avg_chain"
        ),
        "gfx950_avg_mfma_chain": avg_stat(
            [s.get("mfma_chains", {}) for s in gfx950_stats], "avg_chain"
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_kernel_analyses = []
    by_arch = defaultdict(list)
    by_category = defaultdict(list)

    # Process all .asm files
    asm_files = sorted(DISASM_DIR.rglob("*.asm"))
    print(f"Found {len(asm_files)} disassembled kernel files")

    for i, asm_file in enumerate(asm_files):
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(asm_files)}] Analyzing {asm_file.name}...")

        asm_text = asm_file.read_text()
        if len(asm_text) < 50:
            continue

        instrs = parse_instructions(asm_text)
        if not instrs:
            continue

        # Determine arch and category from path
        rel = asm_file.relative_to(DISASM_DIR)
        parts = list(rel.parts)
        arch = parts[0] if parts else "unknown"
        category = parts[1] if len(parts) > 1 else "unknown"

        analysis = {
            "kernel": asm_file.stem,
            "arch": arch,
            "category": category,
            "total_instructions": len(instrs),
            "mfma_chains": analyze_mfma_chains(instrs),
            "lds_patterns": analyze_lds_patterns(instrs),
            "prefetch": analyze_prefetch_depth(instrs),
            "scheduling": analyze_scheduling(instrs),
            "vectorization": analyze_vectorization(instrs),
        }

        all_kernel_analyses.append(analysis)
        by_arch[arch].append(analysis)
        by_category[category].append(analysis)

    print(f"\nAnalyzed {len(all_kernel_analyses)} kernels")

    # Build aggregated pattern database
    patterns_db = {
        "total_kernels_analyzed": len(all_kernel_analyses),
        "mfma_pattern_summary": build_mfma_summary(all_kernel_analyses),
        "lds_pattern_summary": build_lds_summary(all_kernel_analyses),
        "prefetch_summary": build_prefetch_summary(all_kernel_analyses),
        "scheduling_summary": build_scheduling_summary(all_kernel_analyses),
        "vectorization_summary": build_vectorization_summary(all_kernel_analyses),
        "arch_differences": analyze_arch_differences(by_arch.get("gfx942", []), by_arch.get("gfx950", [])),
        "category_summaries": {cat: build_category_summary(analyses) for cat, analyses in sorted(by_category.items())},
    }

    out_file = OUTPUT_DIR / "deep_asm_patterns.json"
    with open(out_file, "w") as f:
        json.dump(patterns_db, f, indent=2)
    print(f"\nWrote deep ASM patterns to {out_file}")

    # Print key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    ms = patterns_db["mfma_pattern_summary"]
    print(f"\nMFMA Chains:")
    print(f"  Max chain length: {ms['max_chain_across_all']}")
    print(f"  Avg chain length: {ms['avg_chain_length']:.1f}")
    print(f"  Top MFMA types: {list(ms['top_mfma_types'].items())[:5]}")

    ls = patterns_db["lds_pattern_summary"]
    print(f"\nLDS Double-Buffering:")
    print(f"  Kernels with double-buffer: {ls['double_buffer_count']} / {ls['total_with_lds']}")

    ps = patterns_db["prefetch_summary"]
    print(f"\nPrefetch Depth:")
    print(f"  Avg loads before first wait: {ps['avg_loads_before_first_wait']:.1f}")
    print(f"  Max loads before first wait: {ps['max_loads_before_first_wait']}")

    ss = patterns_db["scheduling_summary"]
    print(f"\nScheduling:")
    print(f"  Kernels using priority: {ss['kernels_using_priority']}")
    print(f"  Avg partial waitcnt ratio: {ss['avg_partial_waitcnt_ratio']:.2%}")

    ad = patterns_db["arch_differences"]
    print(f"\nArchitecture Differences:")
    print(f"  New MFMA in gfx950: {ad['new_mfma_in_gfx950']}")


def build_mfma_summary(analyses: list[dict]) -> dict:
    all_chains = []
    mfma_type_counts = defaultdict(int)
    interleave_totals = defaultdict(int)

    for a in analyses:
        mc = a.get("mfma_chains", {})
        all_chains.extend(mc.get("chain_lengths", []))
        for t, c in mc.get("mfma_types", {}).items():
            mfma_type_counts[t] += c
        for k, v in mc.get("interleave_between_mfma", {}).items():
            interleave_totals[k] += v

    sorted_types = sorted(mfma_type_counts.items(), key=lambda x: -x[1])
    return {
        "max_chain_across_all": max(all_chains) if all_chains else 0,
        "avg_chain_length": sum(all_chains) / len(all_chains) if all_chains else 0,
        "total_chains": len(all_chains),
        "top_mfma_types": dict(sorted_types[:15]),
        "interleave_totals": dict(interleave_totals),
    }


def build_lds_summary(analyses: list[dict]) -> dict:
    total_with_lds = 0
    double_buffer_count = 0
    lds_sizes = []

    for a in analyses:
        lp = a.get("lds_patterns", {})
        if lp.get("ds_read_count", 0) > 0 or lp.get("ds_write_count", 0) > 0:
            total_with_lds += 1
            if lp.get("has_double_buffer"):
                double_buffer_count += 1
            if lp.get("estimated_lds_bytes", 0) > 0:
                lds_sizes.append(lp["estimated_lds_bytes"])

    return {
        "total_with_lds": total_with_lds,
        "double_buffer_count": double_buffer_count,
        "double_buffer_pct": 100 * double_buffer_count / max(total_with_lds, 1),
        "avg_lds_bytes": sum(lds_sizes) / len(lds_sizes) if lds_sizes else 0,
        "max_lds_bytes": max(lds_sizes) if lds_sizes else 0,
    }


def build_prefetch_summary(analyses: list[dict]) -> dict:
    depths = [a["prefetch"]["loads_before_first_wait"] for a in analyses if a.get("prefetch", {}).get("loads_before_first_wait", 0) > 0]
    avg_between = [a["prefetch"]["avg_loads_between_waits"] for a in analyses if a.get("prefetch", {}).get("avg_loads_between_waits", 0) > 0]

    return {
        "avg_loads_before_first_wait": sum(depths) / len(depths) if depths else 0,
        "max_loads_before_first_wait": max(depths) if depths else 0,
        "avg_loads_between_waits": sum(avg_between) / len(avg_between) if avg_between else 0,
    }


def build_scheduling_summary(analyses: list[dict]) -> dict:
    priority_count = sum(1 for a in analyses if a.get("scheduling", {}).get("uses_priority_scheduling"))
    partial_ratios = [a["scheduling"]["partial_waitcnt_ratio"] for a in analyses if "scheduling" in a]

    return {
        "kernels_using_priority": priority_count,
        "avg_partial_waitcnt_ratio": sum(partial_ratios) / len(partial_ratios) if partial_ratios else 0,
    }


def build_vectorization_summary(analyses: list[dict]) -> dict:
    pcts = [a["vectorization"]["vectorization_pct"] for a in analyses if "vectorization" in a]
    return {
        "avg_vectorization_pct": sum(pcts) / len(pcts) if pcts else 0,
        "kernels_above_50pct": sum(1 for p in pcts if p > 50),
        "kernels_below_10pct": sum(1 for p in pcts if p < 10),
    }


def build_category_summary(analyses: list[dict]) -> dict:
    mfma_counts = [sum(a.get("mfma_chains", {}).get("mfma_types", {}).values()) for a in analyses]
    instr_counts = [a.get("total_instructions", 0) for a in analyses]

    return {
        "count": len(analyses),
        "avg_instructions": sum(instr_counts) / len(instr_counts) if instr_counts else 0,
        "avg_mfma": sum(mfma_counts) / len(mfma_counts) if mfma_counts else 0,
        "max_mfma": max(mfma_counts) if mfma_counts else 0,
    }


if __name__ == "__main__":
    main()
