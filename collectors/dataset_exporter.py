#!/usr/bin/env python3
"""Dataset Exporter for GLM-4 Fine-tuning.

Converts the knowledge base into training-ready formats:
1. Instruction-following format (prompt -> analysis)
2. Code-pair format (suboptimal -> optimized)
3. QA format (question about ISA -> answer)
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.compiler import Compiler
from src.parser import parse_asm
from src.analyzer import Analyzer
from src.knowledge_base import KnowledgeBase


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "db" / "training_data"


def export_isa_qa_pairs(kb: KnowledgeBase) -> list[dict]:
    """Generate QA pairs about ISA instructions."""
    pairs = []

    for instr in kb.isa.all_instructions():
        # Basic instruction lookup
        pairs.append({
            "type": "isa_qa",
            "prompt": f"What does the AMDGPU instruction {instr.mnemonic} do?",
            "response": (
                f"{instr.mnemonic} is a {instr.category} instruction: {instr.description}. "
                f"Operands: {instr.operands}. "
                f"Latency: {instr.latency_cycles} cycles, throughput: {instr.throughput_ops_per_cycle} ops/cycle. "
                f"Supported on: {', '.join(instr.supported_archs)}."
                + (f" Note: {instr.notes}" if instr.notes else "")
            ),
        })

        # Architecture support query
        for arch in instr.supported_archs:
            pairs.append({
                "type": "isa_arch_support",
                "prompt": f"Is {instr.mnemonic} available on {arch}?",
                "response": f"Yes, {instr.mnemonic} is supported on {arch}. It is a {instr.category} instruction "
                            f"with {instr.latency_cycles} cycle latency."
                            + (f" It was introduced in {instr.new_in}." if instr.new_in else ""),
            })

        # Latency/throughput query
        if instr.category == "MFMA":
            pairs.append({
                "type": "isa_performance",
                "prompt": f"What is the latency and throughput of {instr.mnemonic}?",
                "response": (
                    f"{instr.mnemonic} has a latency of {instr.latency_cycles} cycles and "
                    f"throughput of {instr.throughput_ops_per_cycle} ops/cycle. "
                    f"{instr.notes}"
                ),
            })

    return pairs


def export_anti_pattern_pairs(kb: KnowledgeBase) -> list[dict]:
    """Generate training pairs from anti-patterns."""
    pairs = []

    for ap in kb.anti_patterns:
        # Problem identification
        pairs.append({
            "type": "anti_pattern_identify",
            "prompt": f"What is the '{ap['name']}' anti-pattern in GPU kernel programming?",
            "response": (
                f"{ap['description']} "
                f"Severity: {ap['severity']}. "
                f"Suggestion: {ap['suggestion']}"
            ),
        })

        # Code review format
        if "example_bad" in ap:
            pairs.append({
                "type": "anti_pattern_review",
                "prompt": f"Review this GPU kernel code for potential issues:\n```\n{ap['example_bad']}\n```",
                "response": (
                    f"Issue found: {ap['name']} ({ap['severity']})\n"
                    f"{ap['description']}\n\n"
                    f"Suggested fix:\n```\n{ap.get('example_good', ap['suggestion'])}\n```\n\n"
                    f"Explanation: {ap['suggestion']}"
                ),
            })

    return pairs


def export_best_practice_pairs(kb: KnowledgeBase) -> list[dict]:
    """Generate training pairs from best practices."""
    pairs = []

    for bp in kb.best_practices:
        pairs.append({
            "type": "best_practice",
            "prompt": f"Explain the '{bp['name']}' optimization technique for GPU kernels.",
            "response": (
                f"{bp['description']} "
                f"This pattern is applicable to: {', '.join(bp.get('applicable_to', []))}. "
                f"Performance impact: {bp.get('performance_impact', 'Varies')}. "
                f"Implementation: {bp.get('implementation_notes', '')}"
            ),
        })

        # Architecture-specific queries
        pairs.append({
            "type": "best_practice_how",
            "prompt": f"How do I implement {bp['name']} in an AMDGPU kernel?",
            "response": (
                f"{bp.get('implementation_notes', bp['description'])} "
                f"Look for these ASM indicators: {', '.join(bp.get('asm_indicators', [])[:5])}. "
                f"Reference: {bp.get('reference', 'AMD optimization guides')}"
            ),
        })

    return pairs


def export_analysis_pairs(kb: KnowledgeBase) -> list[dict]:
    """Generate training pairs from compiled kernel analysis."""
    pairs = []
    compiler = Compiler()
    analyzer = Analyzer(kb)

    templates_dir = Path(__file__).resolve().parent.parent / "templates"
    templates = sorted(templates_dir.glob("*.hip"))

    for template in templates:
        source = template.read_text()
        template_name = template.stem

        for arch in ["gfx942"]:
            for opt in ["-O0", "-O3"]:
                result = compiler.compile_to_asm(template, arch=arch, opt_level=opt)
                if not result.success:
                    continue

                kernel = parse_asm(result.asm_output)
                analysis = analyzer.analyze(kernel, arch=arch)

                # Build analysis response
                findings_text = []
                for f in analysis.findings:
                    findings_text.append(
                        f"[{f.severity.upper()}] {f.title}: {f.description} "
                        f"Suggestion: {f.suggestion}"
                    )

                if findings_text:
                    response = (
                        f"Analysis of {template_name} compiled for {arch} with {opt}:\n\n"
                        f"Summary: {analysis.summary.get('total_instructions', 0)} instructions, "
                        f"{analysis.summary.get('vgpr_count', 0)} VGPRs, "
                        f"{analysis.summary.get('estimated_occupancy_waves', 0)} waves occupancy.\n\n"
                        f"Findings:\n" + "\n".join(f"- {ft}" for ft in findings_text)
                    )
                else:
                    response = (
                        f"Analysis of {template_name} compiled for {arch} with {opt}:\n\n"
                        f"Summary: {analysis.summary.get('total_instructions', 0)} instructions, "
                        f"{analysis.summary.get('vgpr_count', 0)} VGPRs, "
                        f"{analysis.summary.get('estimated_occupancy_waves', 0)} waves occupancy.\n\n"
                        f"No significant issues detected. The kernel appears well-optimized for the compilation level."
                    )

                pairs.append({
                    "type": "kernel_analysis",
                    "prompt": f"Analyze this HIP kernel for {arch} optimization issues:\n```cpp\n{source}\n```",
                    "response": response,
                    "metadata": {
                        "template": template_name,
                        "arch": arch,
                        "opt_level": opt,
                        "findings_count": len(analysis.findings),
                    },
                })

    return pairs


def export_profiling_rule_pairs(kb: KnowledgeBase) -> list[dict]:
    """Generate training pairs from profiling rules."""
    pairs = []

    for rule in kb.profiling_rules:
        pairs.append({
            "type": "profiling_qa",
            "prompt": f"What does the '{rule['name']}' profiling metric indicate for GPU kernels?",
            "response": (
                f"{rule['description']} "
                f"Diagnosis: {rule['diagnosis']} "
                f"Suggestion: {rule['suggestion']} "
                f"Relevant counter: {rule.get('rocprof_counter', 'N/A')}"
            ),
        })

    return pairs


def export_disassembly_analysis_pairs(kb: KnowledgeBase) -> list[dict]:
    """Generate training pairs from disassembled production .co kernels."""
    pairs = []
    disasm_dir = Path(__file__).resolve().parent.parent / "db" / "disassembly"

    summary_file = disasm_dir / "disassembly_summary.json"
    if not summary_file.exists():
        return pairs

    with open(summary_file) as f:
        summary = json.load(f)

    # Per-kernel analysis pairs from JSON sidecar files
    json_files = sorted(disasm_dir.rglob("*.json"))
    processed = 0
    for jf in json_files:
        if jf.name == "disassembly_summary.json":
            continue
        try:
            with open(jf) as f:
                kstats = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        kernel_name = jf.stem
        arch = kstats.get("arch", "unknown")
        category = kstats.get("category", "unknown")
        mfma_count = kstats.get("mfma_count", 0)
        total_instr = kstats.get("total_instructions", 0)

        if total_instr < 10:
            continue

        # Kernel overview pair
        mfma_types_str = ", ".join(f"{k}: {v}" for k, v in sorted(kstats.get("mfma_types", {}).items(), key=lambda x: -x[1])[:3])
        pairs.append({
            "type": "disasm_analysis",
            "prompt": f"Analyze the production AMD GPU kernel '{kernel_name}' ({arch}, category: {category}). "
                     f"It has {total_instr} instructions and {mfma_count} MFMA instructions.",
            "response": (
                f"Production kernel '{kernel_name}' for {arch} ({category} category):\n\n"
                f"Instruction breakdown: {total_instr} total, {mfma_count} MFMA, "
                f"{kstats.get('ds_read_count', 0)} DS reads, {kstats.get('ds_write_count', 0)} DS writes, "
                f"{kstats.get('global_load_count', 0)} global loads, {kstats.get('buffer_load_count', 0)} buffer loads.\n\n"
                f"MFMA types: {mfma_types_str or 'none'}.\n"
                f"Registers: VGPR up to v{kstats.get('max_vgpr', 0)}, SGPR up to s{kstats.get('max_sgpr', 0)}, "
                f"AccVGPR up to a{kstats.get('max_agpr', 0)}.\n"
                f"Scheduling: {kstats.get('waitcnt_count', 0)} waitcnts "
                f"({kstats.get('waitcnt_partial', 0)} partial), "
                f"{kstats.get('nop_count', 0)} NOPs, {kstats.get('barrier_count', 0)} barriers.\n"
                f"Vectorization: {kstats.get('dwordx4_loads', 0)} dwordx4 loads vs "
                f"{kstats.get('dword_single_loads', 0)} single-dword loads."
            ),
        })

        # MFMA-specific analysis for kernels with significant MFMA usage
        if mfma_count > 32:
            mfma_types = kstats.get("mfma_types", {})
            dominant_type = max(mfma_types.items(), key=lambda x: x[1])[0] if mfma_types else "none"
            pairs.append({
                "type": "disasm_mfma_pattern",
                "prompt": f"What MFMA patterns does the {category} kernel '{kernel_name}' use on {arch}?",
                "response": (
                    f"The kernel uses {mfma_count} MFMA instructions. "
                    f"Dominant type: {dominant_type} ({mfma_types.get(dominant_type, 0)} occurrences). "
                    f"All types: {mfma_types_str}. "
                    f"This is a production-optimized kernel from aiter for {arch}."
                ),
            })

        # Architecture-specific optimization pairs
        if arch == "gfx950" and kstats.get("mfma_types"):
            new_types = [t for t in kstats["mfma_types"] if "scale" in t or "f8f6f4" in t or "smfma" in t]
            if new_types:
                pairs.append({
                    "type": "disasm_gfx950_feature",
                    "prompt": f"What gfx950-specific features does kernel '{kernel_name}' use?",
                    "response": (
                        f"Kernel '{kernel_name}' uses gfx950-specific MFMA: {', '.join(new_types)}. "
                        f"These are new instructions not available on gfx942."
                    ),
                })

        # Vectorization analysis pair
        dwordx4 = kstats.get("dwordx4_loads", 0)
        dword_single = kstats.get("dword_single_loads", 0)
        total_ld = dwordx4 + dword_single
        if total_ld > 4:
            vec_pct = 100 * dwordx4 / total_ld
            pairs.append({
                "type": "disasm_vectorization",
                "prompt": f"How well vectorized are the memory accesses in kernel '{kernel_name}'?",
                "response": (
                    f"Kernel '{kernel_name}' ({arch}): {vec_pct:.0f}% vectorized loads "
                    f"({dwordx4} dwordx4 loads, {dword_single} single-dword loads). "
                    + ("Good vectorization." if vec_pct > 60 else "Could benefit from wider loads.")
                ),
            })

        # Scheduling analysis pair
        waitcnt = kstats.get("waitcnt_count", 0)
        partial = kstats.get("waitcnt_partial", 0)
        if waitcnt > 2:
            partial_ratio = partial / waitcnt
            pairs.append({
                "type": "disasm_scheduling",
                "prompt": f"How does kernel '{kernel_name}' manage memory latency?",
                "response": (
                    f"Kernel '{kernel_name}' ({arch}): {waitcnt} waitcnts total, "
                    f"{partial} partial ({partial_ratio:.0%}). "
                    f"{kstats.get('nop_count', 0)} NOPs, {kstats.get('barrier_count', 0)} barriers, "
                    f"{kstats.get('setprio_count', 0)} s_setprio instructions. "
                    + ("Uses partial waits effectively." if partial_ratio > 0.5 else "Many full stalls present.")
                ),
            })

        processed += 1

    # Architecture comparison pairs
    for arch_info in summary.get("by_arch", {}).items():
        arch_name, arch_data = arch_info
        pairs.append({
            "type": "disasm_arch_summary",
            "prompt": f"Summarize the production kernel characteristics for {arch_name}.",
            "response": (
                f"On {arch_name}: {arch_data['count']} production kernels analyzed, "
                f"containing {arch_data['total_mfma']} total MFMA instructions and "
                f"{arch_data['total_instructions']} total instructions. "
                f"Average MFMA density: {arch_data['total_mfma'] / max(arch_data['total_instructions'], 1):.1%}."
            ),
        })

    # Category comparison pairs
    for cat_info in summary.get("by_category", {}).items():
        cat_name, cat_data = cat_info
        pairs.append({
            "type": "disasm_category_summary",
            "prompt": f"What are the characteristics of {cat_name} kernels on AMD GPUs?",
            "response": (
                f"{cat_name} kernel category: {cat_data['count']} production kernels, "
                f"{cat_data['total_mfma']} total MFMA instructions. "
                f"Average MFMA per kernel: {cat_data['total_mfma'] / max(cat_data['count'], 1):.0f}."
            ),
        })

    return pairs


def export_deep_asm_pattern_pairs(kb: KnowledgeBase) -> list[dict]:
    """Generate training pairs from deep ASM pattern analysis."""
    pairs = []
    patterns = kb.deep_asm_patterns
    if not patterns:
        return pairs

    # MFMA pattern QA
    ms = patterns.get("mfma_pattern_summary", {})
    if ms:
        pairs.append({
            "type": "deep_pattern_mfma",
            "prompt": "What are the typical MFMA chaining patterns in production AMD GPU kernels?",
            "response": (
                f"Based on analysis of {patterns.get('total_kernels_analyzed', 0)} production kernels:\n\n"
                f"- Average MFMA chain length: {ms.get('avg_chain_length', 0):.1f} instructions\n"
                f"- Maximum chain length: {ms.get('max_chain_across_all', 0)} instructions\n"
                f"- Total MFMA chains: {ms.get('total_chains', 0)}\n\n"
                f"Top MFMA instruction types:\n" +
                "\n".join(f"  - {k}: {v:,} occurrences" for k, v in list(ms.get("top_mfma_types", {}).items())[:5]) +
                f"\n\nInterleaving between MFMA groups: {ms.get('interleave_totals', {})}"
            ),
        })

        for mfma_type, count in list(ms.get("top_mfma_types", {}).items())[:10]:
            pairs.append({
                "type": "deep_pattern_mfma_type",
                "prompt": f"How is {mfma_type} used in production AMD GPU kernels?",
                "response": (
                    f"{mfma_type} appears {count:,} times across {patterns.get('total_kernels_analyzed', 0)} "
                    f"production kernels, making it "
                    f"{'the most common' if count > 500000 else 'a commonly used'} MFMA variant. "
                    f"It is part of chains averaging {ms.get('avg_chain_length', 0):.1f} instructions."
                ),
            })

    # LDS pattern QA
    ls = patterns.get("lds_pattern_summary", {})
    if ls:
        pairs.append({
            "type": "deep_pattern_lds",
            "prompt": "How common is LDS double-buffering in production AMD GPU kernels?",
            "response": (
                f"LDS double-buffering analysis across {patterns.get('total_kernels_analyzed', 0)} kernels:\n\n"
                f"- Kernels using LDS: {ls.get('total_with_lds', 0)}\n"
                f"- Kernels with detected double-buffer: {ls.get('double_buffer_count', 0)} "
                f"({ls.get('double_buffer_pct', 0):.0f}%)\n"
                f"- Average LDS allocation: {ls.get('avg_lds_bytes', 0):,.0f} bytes\n"
                f"- Maximum LDS allocation: {ls.get('max_lds_bytes', 0):,} bytes\n\n"
                "Double-buffering (ping-pong) is the dominant strategy, eliminating "
                "read-after-write synchronization between LDS fill and GEMM compute."
            ),
        })

    # Prefetch depth QA
    ps = patterns.get("prefetch_summary", {})
    if ps:
        pairs.append({
            "type": "deep_pattern_prefetch",
            "prompt": "What is the typical prefetch depth in production AMD GPU kernels?",
            "response": (
                f"Prefetch depth analysis:\n\n"
                f"- Average loads before first wait: {ps.get('avg_loads_before_first_wait', 0):.1f}\n"
                f"- Maximum loads before first wait: {ps.get('max_loads_before_first_wait', 0)}\n"
                f"- Average loads between waits: {ps.get('avg_loads_between_waits', 0):.1f}\n\n"
                "Deep prefetching (7-8 loads before wait) hides global memory latency "
                "(~300 cycles). CK pipelines use PrefetchStages=2-3 for this purpose."
            ),
        })

    # Architecture difference QA
    ad = patterns.get("arch_differences", {})
    if ad:
        new_mfma = ad.get("new_mfma_in_gfx950", [])
        if new_mfma:
            pairs.append({
                "type": "deep_pattern_arch_diff",
                "prompt": "What new MFMA instructions does gfx950 introduce compared to gfx942?",
                "response": (
                    f"gfx950 introduces {len(new_mfma)} new MFMA instruction variants:\n\n" +
                    "\n".join(f"  - {m}" for m in new_mfma) +
                    f"\n\ngfx942 has {len(ad.get('gfx942_mfma_types', []))} MFMA types, "
                    f"gfx950 has {len(ad.get('gfx950_mfma_types', []))} types. "
                    "Notable additions include v_mfma_scale_f32_16x16x128_f8f6f4 for "
                    "hardware-accelerated FP8 scaling and v_mfma_f32_32x32x64_f8f6f4 for "
                    "wider FP4/FP6/FP8 operations."
                ),
            })

    # Category-specific patterns
    for cat, cat_data in patterns.get("category_summaries", {}).items():
        pairs.append({
            "type": "deep_pattern_category",
            "prompt": f"What are the typical instruction characteristics of {cat} kernels on AMD GPUs?",
            "response": (
                f"{cat} kernel category ({cat_data.get('count', 0)} kernels analyzed):\n\n"
                f"- Average instructions per kernel: {cat_data.get('avg_instructions', 0):,.0f}\n"
                f"- Average MFMA per kernel: {cat_data.get('avg_mfma', 0):,.0f}\n"
                f"- Maximum MFMA in a single kernel: {cat_data.get('max_mfma', 0):,}"
            ),
        })

    # Scheduling pattern QA
    ss = patterns.get("scheduling_summary", {})
    if ss:
        pairs.append({
            "type": "deep_pattern_scheduling",
            "prompt": "How do production AMD GPU kernels use waitcnt and scheduling?",
            "response": (
                f"Scheduling patterns across production kernels:\n\n"
                f"- Kernels using s_setprio: {ss.get('kernels_using_priority', 0)}\n"
                f"- Average partial waitcnt ratio: {ss.get('avg_partial_waitcnt_ratio', 0):.0%}\n\n"
                "Partial waitcnts (vmcnt(N) where N>0) are preferred over full stalls "
                "(vmcnt(0)) because they allow some memory operations to remain in-flight. "
                "The 57% partial ratio indicates production kernels carefully manage "
                "memory latency hiding through partial waits."
            ),
        })

    return pairs


def export_ck_pipeline_pairs(kb: KnowledgeBase) -> list[dict]:
    """Generate training pairs from CK pipeline pattern analysis."""
    pairs = []
    ck_data = kb.ck_deep_patterns
    if not ck_data:
        return pairs

    # Pipeline comparison pairs
    for pipeline in ck_data.get("pipeline_comparison", []):
        pairs.append({
            "type": "ck_pipeline_overview",
            "prompt": f"Explain the CK {pipeline['pipeline']} optimization strategy.",
            "response": (
                f"Pipeline: {pipeline['pipeline']}\n"
                f"Strategy: {pipeline['strategy']}\n\n"
                f"Configuration:\n"
                f"- PrefetchStages: {pipeline.get('prefetch_stages', 'N/A')}\n"
                f"- Global buffers: {pipeline.get('global_buffers', 'N/A')}\n"
                f"- LDS buffers: {pipeline.get('lds_buffers', 'N/A')}\n\n"
                f"Scheduler: {pipeline.get('scheduler_structure', 'N/A')}\n\n"
                f"Hot loop: {pipeline.get('hotloop', 'N/A')}\n\n"
                f"Key insight: {pipeline.get('key_insight', 'N/A')}\n\n"
                f"Performance tradeoff: {pipeline.get('performance_tradeoff', 'N/A')}"
            ),
        })

        # Comparison queries
        pairs.append({
            "type": "ck_pipeline_comparison",
            "prompt": f"What are the tradeoffs of using CK {pipeline['pipeline']}?",
            "response": (
                f"{pipeline['pipeline']} ({pipeline['strategy']}):\n\n"
                f"Advantages: {pipeline.get('key_insight', '')}\n"
                f"Tradeoffs: {pipeline.get('performance_tradeoff', '')}\n\n"
                f"Uses scheduling barriers: {', '.join(pipeline.get('sched_group_barrier_masks', ['none']))}\n"
                f"Tail handling: {', '.join(pipeline.get('tail_handling', ['none']))}"
            ),
        })

    # sched_group_barrier reference
    sched_ref = ck_data.get("sched_barrier_reference", {})
    if sched_ref:
        pairs.append({
            "type": "ck_sched_barrier_ref",
            "prompt": "How do I use __builtin_amdgcn_sched_group_barrier in AMD GPU kernels?",
            "response": (
                f"{sched_ref.get('description', '')}\n\n"
                f"Available masks:\n" +
                "\n".join(f"  - {k}: {v['name']} ({v['instructions']})" for k, v in sched_ref.get("masks", {}).items()) +
                "\n\nCommon patterns:\n" +
                "\n".join(f"  - {p['name']}: {p['sequence']}\n    Purpose: {p['purpose']}" for p in sched_ref.get("common_patterns", []))
            ),
        })

        for mask_hex, mask_info in sched_ref.get("masks", {}).items():
            pairs.append({
                "type": "ck_sched_barrier_mask",
                "prompt": f"What does sched_group_barrier mask {mask_hex} control?",
                "response": (
                    f"Mask {mask_hex} ({mask_info['name']}) controls scheduling of "
                    f"{mask_info['instructions']} instructions. Purpose: {mask_info['purpose']}."
                ),
            })

    # Optimization pattern pairs
    for pattern in ck_data.get("key_optimization_patterns", []):
        pairs.append({
            "type": "ck_opt_pattern",
            "prompt": f"Explain the '{pattern['name']}' optimization pattern for AMD GPU kernels.",
            "response": (
                f"{pattern['description']}\n\n"
                f"Used in CK pipelines: {', '.join(pattern.get('used_in', []))}\n"
                f"LDS cost: {pattern.get('lds_cost', 'N/A')}\n"
                f"Benefit: {pattern.get('benefit', 'N/A')}\n"
                f"Code pattern: {pattern.get('code_pattern', 'N/A')}"
            ),
        })

    # Individual pipeline analysis pairs
    for analysis in ck_data.get("pipeline_analyses", []):
        if analysis.get("sched_group_barriers"):
            barrier_summary = analysis.get("sched_barrier_summary", {})
            pairs.append({
                "type": "ck_pipeline_analysis",
                "prompt": f"What scheduling barriers does {analysis['pipeline_name']} use?",
                "response": (
                    f"{analysis['pipeline_name']} uses {len(analysis['sched_group_barriers'])} "
                    f"sched_group_barrier calls.\n\n"
                    f"Barrier types: {barrier_summary}\n"
                    f"Buffer strategy: {analysis.get('buffer_strategy', 'unknown')}\n"
                    f"Features: {analysis.get('features', {})}"
                ),
            })

    return pairs


def export_trtllm_algorithm_pairs(kb: KnowledgeBase) -> list[dict]:
    """Generate training pairs from TensorRT-LLM algorithm catalog."""
    pairs = []
    trt_data = kb.trtllm_algorithms
    if not trt_data:
        return pairs

    for algo in trt_data.get("algorithms", []):
        # Algorithm overview
        pairs.append({
            "type": "trtllm_algorithm",
            "prompt": f"Describe the {algo['algorithm']} kernel from TensorRT-LLM.",
            "response": (
                f"Algorithm: {algo['algorithm']} (Category: {algo['category']})\n"
                f"Location: {algo['trtllm_path']}\n\n"
                f"{algo['description']}\n\n"
                f"Key techniques:\n" +
                "\n".join(f"  - {t}" for t in algo.get("key_techniques", [])) +
                f"\n\nComplexity: {algo.get('complexity', 'N/A')}\n"
                f"Variants: {', '.join(algo.get('variants', []))}"
            ),
        })

        # How-to-implement pair
        pairs.append({
            "type": "trtllm_technique",
            "prompt": f"What techniques does TensorRT-LLM use for {algo['algorithm']}?",
            "response": (
                f"{algo['algorithm']} uses: " +
                "; ".join(algo.get("key_techniques", [])) +
                f". CUDA features: {', '.join(algo.get('cuda_features', []))}. "
                f"Complexity: {algo.get('complexity', 'N/A')}."
            ),
        })

    return pairs


def export_cross_platform_mapping_pairs(kb: KnowledgeBase) -> list[dict]:
    """Generate training pairs from NVIDIA<->AMD kernel mapping."""
    pairs = []
    mapping_data = kb.trtllm_amd_mapping
    if not mapping_data:
        return pairs

    for m in mapping_data.get("mappings", []):
        # Cross-platform comparison
        pairs.append({
            "type": "cross_platform_mapping",
            "prompt": f"Compare the NVIDIA and AMD implementations of {m['algorithm']}.",
            "response": (
                f"Algorithm: {m['algorithm']} (Category: {m['category']}, Sub: {m.get('subcategory', 'N/A')})\n\n"
                f"NVIDIA (TensorRT-LLM):\n"
                f"  Implementation: {m['trtllm_impl']}\n"
                f"  Key technique: {m['trtllm_key_technique']}\n\n"
                f"AMD (aiter/CK):\n"
                f"  aiter: {m['aiter_impl']}\n"
                f"  CK: {m.get('ck_impl', 'N/A')}\n"
                f"  .co kernels: {m.get('co_kernels', 'N/A')}\n\n"
                f"Key differences:\n" +
                "\n".join(f"  - {d}" for d in m.get("key_differences", [])) +
                f"\n\nOptimization notes: {m.get('optimization_notes', 'N/A')}"
            ),
        })

        # AMD equivalent query
        pairs.append({
            "type": "nvidia_to_amd",
            "prompt": f"What is the AMD equivalent of TensorRT-LLM's {m['trtllm_impl']}?",
            "response": (
                f"The AMD equivalent of TensorRT-LLM's {m['trtllm_impl']} ({m['algorithm']}) is:\n\n"
                f"- aiter implementation: {m['aiter_impl']}\n"
                f"- CK implementation: {m.get('ck_impl', 'N/A')}\n"
                f"- Pre-compiled kernels: {m.get('co_kernels', 'N/A')}\n\n"
                f"Key differences: " + "; ".join(m.get("key_differences", [])[:2])
            ),
        })

        # Optimization guidance
        pairs.append({
            "type": "cross_platform_optimization",
            "prompt": f"How should I optimize {m['algorithm']} for AMD MI300 GPUs?",
            "response": (
                f"Optimizing {m['algorithm']} for AMD MI300:\n\n"
                f"{m.get('optimization_notes', '')}\n\n"
                f"Recommended approach:\n"
                f"1. Use aiter's optimized implementation: {m['aiter_impl']}\n"
                f"2. CK pipeline: {m.get('ck_impl', 'Build custom CK kernel')}\n"
                f"3. Key AMD-specific considerations:\n" +
                "\n".join(f"   - {d}" for d in m.get("key_differences", []))
            ),
        })

    return pairs


def export_compiler_flag_qa_pairs(kb: KnowledgeBase) -> list[dict]:
    """Generate QA pairs about compiler flag effects."""
    pairs = []
    flags = kb.compiler_flags
    if not flags:
        return pairs

    for comp in flags.get("comparisons", []):
        flag_a = comp.get("flag_a", "")
        flag_b = comp.get("flag_b", "")
        arch = comp.get("arch", "gfx942")
        template = comp.get("template", "unknown")

        pairs.append({
            "type": "compiler_flag_qa",
            "prompt": f"How does compiling with {flag_a} vs {flag_b} affect the assembly output for a {template} kernel on {arch}?",
            "response": (
                f"Comparing {flag_a} vs {flag_b} for {template} on {arch}:\n\n"
                f"Instructions: {comp.get('instructions_a', 'N/A')} vs {comp.get('instructions_b', 'N/A')}\n"
                f"VGPRs: {comp.get('vgprs_a', 'N/A')} vs {comp.get('vgprs_b', 'N/A')}\n"
                f"Differences: {comp.get('notable_differences', 'See detailed analysis')}"
            ),
        })

    return pairs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kb = KnowledgeBase()
    kb.load()

    all_pairs = []
    breakdown = {}

    generators = [
        ("ISA QA", export_isa_qa_pairs),
        ("Anti-patterns", export_anti_pattern_pairs),
        ("Best practices", export_best_practice_pairs),
        ("Kernel analysis", export_analysis_pairs),
        ("Profiling rules", export_profiling_rule_pairs),
        ("Disassembly analysis", export_disassembly_analysis_pairs),
        ("Deep ASM patterns", export_deep_asm_pattern_pairs),
        ("CK pipeline patterns", export_ck_pipeline_pairs),
        ("TRT-LLM algorithms", export_trtllm_algorithm_pairs),
        ("Cross-platform mapping", export_cross_platform_mapping_pairs),
        ("Compiler flags", export_compiler_flag_qa_pairs),
    ]

    for name, gen_func in generators:
        print(f"Generating {name} pairs...")
        pairs = gen_func(kb)
        all_pairs.extend(pairs)
        breakdown[name] = len(pairs)
        print(f"  Generated {len(pairs)} pairs")

    # Write combined dataset
    dataset = {
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "total_pairs": len(all_pairs),
        "breakdown": breakdown,
        "pairs": all_pairs,
    }

    out_file = OUTPUT_DIR / "glm4_training_data.json"
    with open(out_file, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(all_pairs)} total training pairs to {out_file}")

    # JSONL format
    jsonl_file = OUTPUT_DIR / "glm4_training_data.jsonl"
    with open(jsonl_file, "w") as f:
        for pair in all_pairs:
            entry = {
                "instruction": pair["prompt"],
                "output": pair["response"],
                "type": pair["type"],
            }
            if "metadata" in pair:
                entry["metadata"] = pair["metadata"]
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote JSONL format to {jsonl_file}")

    # ChatML format for GLM-4 fine-tuning
    chatml_file = OUTPUT_DIR / "glm4_chatml.jsonl"
    with open(chatml_file, "w") as f:
        for pair in all_pairs:
            entry = {
                "messages": [
                    {"role": "system", "content": "You are an expert AMDGPU kernel optimization advisor. You analyze GPU assembly code, identify performance issues, and suggest optimizations for AMD Instinct GPUs (MI300, MI325X, MI350). You have deep knowledge of MFMA instructions, CK pipelines, and cross-platform NVIDIA/AMD kernel algorithm mapping."},
                    {"role": "user", "content": pair["prompt"]},
                    {"role": "assistant", "content": pair["response"]},
                ],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote ChatML format to {chatml_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset Generation Summary (v2.0)")
    print(f"{'='*60}")
    print(f"Total training pairs: {len(all_pairs):,}")
    print(f"\nBreakdown:")
    for name, count in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {name:30s}: {count:>6,} pairs")
    print(f"\nFiles written:")
    print(f"  {out_file}")
    print(f"  {jsonl_file}")
    print(f"  {chatml_file}")


if __name__ == "__main__":
    main()
