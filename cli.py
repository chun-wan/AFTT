#!/usr/bin/env python3
"""AFTT - ASM Fine-Tuning Tool - CLI Entry Point.

Usage:
    aftt analyze <source.hip> --arch gfx942
    aftt suggest <kernel.s> --arch gfx942
    aftt compile-compare <source.hip> --arch gfx942 --flags "-O2,-O3,-Ofast"
    aftt isa <mnemonic> [--arch gfx942]
    aftt stats
"""

import sys
import os
import json
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import click

from src.compiler import Compiler
from src.parser import parse_asm
from src.analyzer import Analyzer
from src.reporter import (
    format_report_plain,
    format_report_rich,
    format_instruction_info,
)
from src.knowledge_base import KnowledgeBase


@click.group()
@click.version_option(version="0.1.0")
def main():
    """AFTT - ASM Fine-Tuning Tool for AMD GPUs.

    Analyzes C++/HIP kernels and AMDGPU assembly to provide
    optimization suggestions for AMD Instinct GPUs.
    """
    pass


@main.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--arch", default="gfx942", help="Target GPU architecture")
@click.option("--opt", default="-O3", help="Optimization level")
@click.option("--flags", default="", help="Extra compiler flags (comma-separated)")
@click.option("--rich/--plain", default=True, help="Use rich formatting")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.option("--save-asm", type=click.Path(), help="Save generated ASM to file")
def analyze(source, arch, opt, flags, rich, json_output, save_asm):
    """Analyze a C++/HIP kernel file.

    Compiles the source to AMDGPU assembly and runs pattern matching
    against the knowledge base to find optimization opportunities.
    """
    compiler = Compiler()
    extra_flags = [f.strip() for f in flags.split(",") if f.strip()]

    log = click.echo if not json_output else (lambda msg, **kw: click.echo(msg, err=True, **kw))
    log(f"Compiling {source} for {arch} with {opt} {' '.join(extra_flags)}...")
    result = compiler.compile_to_asm(source, arch=arch, opt_level=opt, extra_flags=extra_flags)

    if not result.success:
        click.echo(f"Compilation failed:\n{result.stderr}", err=True)
        sys.exit(1)

    if save_asm:
        Path(save_asm).write_text(result.asm_output)
        log(f"ASM saved to {save_asm}")

    log("Parsing assembly...")
    kernel = parse_asm(result.asm_output)

    log("Analyzing patterns...")
    kb = KnowledgeBase()
    analyzer = Analyzer(kb)
    analysis = analyzer.analyze(kernel, arch=arch)

    if json_output:
        output = {
            "kernel_name": analysis.kernel_name,
            "arch": analysis.arch,
            "summary": analysis.summary,
            "findings": [
                {
                    "id": f.finding_id,
                    "severity": f.severity,
                    "category": f.category,
                    "title": f.title,
                    "description": f.description,
                    "suggestion": f.suggestion,
                    "line_numbers": f.line_numbers,
                    "pattern_id": f.pattern_id,
                    "metrics": f.metrics,
                    "reference": f.reference,
                }
                for f in analysis.findings
            ],
        }
        click.echo(json.dumps(output, indent=2))
    else:
        if rich:
            click.echo(format_report_rich(analysis))
        else:
            click.echo(format_report_plain(analysis))


@main.command()
@click.argument("asm_file", type=click.Path(exists=True))
@click.option("--arch", default="gfx942", help="Target GPU architecture")
@click.option("--rich/--plain", default=True, help="Use rich formatting")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def suggest(asm_file, arch, rich, json_output):
    """Get optimization suggestions for an existing ASM file.

    Parses the assembly and runs pattern matching to find issues
    and suggest improvements.
    """
    asm_text = Path(asm_file).read_text()

    log = click.echo if not json_output else (lambda msg, **kw: click.echo(msg, err=True, **kw))
    log(f"Parsing {asm_file}...")
    kernel = parse_asm(asm_text)

    log("Analyzing patterns...")
    kb = KnowledgeBase()
    analyzer = Analyzer(kb)
    analysis = analyzer.analyze(kernel, arch=arch)

    if json_output:
        output = {
            "kernel_name": analysis.kernel_name,
            "arch": analysis.arch,
            "summary": analysis.summary,
            "findings": [
                {
                    "id": f.finding_id,
                    "severity": f.severity,
                    "category": f.category,
                    "title": f.title,
                    "description": f.description,
                    "suggestion": f.suggestion,
                    "line_numbers": f.line_numbers,
                    "pattern_id": f.pattern_id,
                    "metrics": f.metrics,
                    "reference": f.reference,
                }
                for f in analysis.findings
            ],
        }
        click.echo(json.dumps(output, indent=2))
    else:
        if rich:
            click.echo(format_report_rich(analysis))
        else:
            click.echo(format_report_plain(analysis))


@main.command("compile-compare")
@click.argument("source", type=click.Path(exists=True))
@click.option("--arch", default="gfx942", help="Target GPU architecture")
@click.option("--flags", default="-O0,-O2,-O3,-Ofast", help="Comma-separated flag sets to compare")
def compile_compare(source, arch, flags):
    """Compare compiler output across different flag sets.

    Compiles the same source with multiple flag combinations and
    shows the instruction count differences.
    """
    compiler = Compiler()
    flag_sets = [[f.strip()] for f in flags.split(",") if f.strip()]

    click.echo(f"Compiling {source} for {arch} with {len(flag_sets)} flag sets...\n")

    results = []
    for flag_set in flag_sets:
        result = compiler.compile_to_asm(source, arch=arch, opt_level=flag_set[0])
        if result.success:
            kernel = parse_asm(result.asm_output)
            results.append((flag_set, kernel))
            click.echo(f"  {flag_set[0]:>8}: {kernel.total_instructions:>5} instructions "
                        f"(VALU={kernel.valu_count}, MFMA={kernel.mfma_count}, "
                        f"VMEM={kernel.vmem_count}, LDS={kernel.lds_count}, "
                        f"VGPRs={kernel.metadata.vgpr_count})")
        else:
            click.echo(f"  {flag_set[0]:>8}: FAILED - {result.stderr[:100]}")

    if len(results) >= 2:
        click.echo(f"\n  Comparison ({results[0][0][0]} -> {results[-1][0][0]}):")
        first = results[0][1]
        last = results[-1][1]
        delta = last.total_instructions - first.total_instructions
        pct = 100 * delta / max(first.total_instructions, 1)
        click.echo(f"    Instructions: {delta:+d} ({pct:+.1f}%)")
        click.echo(f"    VALU: {last.valu_count - first.valu_count:+d}")
        click.echo(f"    VMEM: {last.vmem_count - first.vmem_count:+d}")
        click.echo(f"    VGPRs: {last.metadata.vgpr_count - first.metadata.vgpr_count:+d}")


@main.command()
@click.argument("mnemonic")
@click.option("--arch", default=None, help="Filter by architecture")
def isa(mnemonic, arch):
    """Look up an ISA instruction.

    Shows details about an AMDGPU instruction including latency,
    throughput, operands, and architecture support.
    """
    kb = KnowledgeBase()
    instr = kb.lookup_instruction(mnemonic)

    if instr:
        click.echo(format_instruction_info(instr))
        if arch and not instr.supports_arch(arch):
            click.echo(f"\n  NOTE: This instruction is NOT supported on {arch}")

        # Also show related instructions
        results = kb.search_instructions(mnemonic)
        related = [r for r in results if r.mnemonic != instr.mnemonic]
        if related:
            click.echo(f"\n  Related instructions ({len(related)}):")
            for r in related[:15]:
                archs = ", ".join(r.supported_archs[:3])
                if len(r.supported_archs) > 3:
                    archs += "..."
                click.echo(f"    {r.mnemonic:45s} {r.category:8s} [{archs}]")
    else:
        results = kb.search_instructions(mnemonic)
        if results:
            if arch:
                results = [r for r in results if r.supports_arch(arch)]
            click.echo(f"Found {len(results)} instruction(s) matching '{mnemonic}':")
            for r in results[:20]:
                archs = ", ".join(r.supported_archs[:3])
                if len(r.supported_archs) > 3:
                    archs += "..."
                click.echo(f"  {r.mnemonic:45s} {r.category:8s} lat={r.latency_cycles:>3d}c  [{archs}]")
        else:
            click.echo(f"No instruction found matching '{mnemonic}'")


@main.command()
def stats():
    """Show knowledge base statistics."""
    kb = KnowledgeBase()
    s = kb.get_stats()

    click.echo("ASM Kernel Advisor - Knowledge Base Statistics")
    click.echo("=" * 50)
    click.echo(f"  ISA Instructions:        {s['isa_instructions']:>6}")
    click.echo(f"  ISA Architectures:       {s['isa_architectures']:>6}")
    click.echo(f"  Anti-patterns:           {s['anti_patterns']:>6}")
    click.echo(f"  Best Practices:          {s['best_practices']:>6}")
    click.echo(f"  Profiling Rules:         {s['profiling_rules']:>6}")
    click.echo(f"  Pipeline Patterns:       {s['pipeline_patterns']:>6}")
    click.echo(f"  ASM Kernel Configs:      {s['asm_kernel_configs']:>6}")
    click.echo(f"  Compiler Flag Tests:     {s['compiler_flag_comparisons']:>6}")


@main.command()
@click.argument("co_path", type=click.Path(exists=True))
@click.option("--arch", default="gfx942", help="Target architecture (default: gfx942)")
@click.option("--kernel-name", default=None, help="Kernel symbol name (auto-detect if not given)")
@click.option("--aggressive", is_flag=True, help="Apply more aggressive optimizations")
@click.option("--output", "-o", default=None, help="Output path for optimized .co")
@click.option("--json-output", is_flag=True, help="Output results as JSON")
def optimize(co_path, arch, kernel_name, aggressive, output, json_output):
    """Optimize a .co code object via ASM-level binary patching."""
    from src.asm_editor import AsmEditor
    from src.asm_optimizer import AsmOptimizer
    from src.cycle_estimator import CycleEstimator
    from src.instruction import EditOperation

    editor = AsmEditor(arch)
    info, instrs = editor.disassemble(co_path)

    if not kernel_name:
        kernel_name = info.name

    optimizer = AsmOptimizer(arch)
    opt_result = optimizer.optimize(instrs, aggressive=aggressive)

    if not opt_result.edits:
        if json_output:
            click.echo(json.dumps({"edits": 0, "recommendations": opt_result.recommendations}))
        else:
            click.echo("No applicable optimizations found.")
            for r in opt_result.recommendations:
                click.echo(f"  [rec] {r['type']}: {r['description']}")
        return

    estimator = CycleEstimator(arch)
    orig_lines = editor.get_instruction_lines(instrs)
    mod_lines = editor.apply_and_get_modified_lines(instrs, opt_result.edits)
    est_orig = estimator.estimate(orig_lines)
    est_mod = estimator.estimate(mod_lines)
    comparison = estimator.compare(est_orig, est_mod)

    if not output:
        output = co_path.replace(".co", "_optimized.co")

    patch_result = editor.binary_patch(co_path, output, opt_result.edits, instrs)

    if json_output:
        click.echo(json.dumps({
            "patches": patch_result,
            "cycle_estimate": comparison,
            "output": output,
        }, indent=2))
    else:
        click.echo(f"Optimized {co_path} -> {output}")
        click.echo(f"  Applied: {patch_result['applied_count']} patches")
        for p in patch_result["applied"]:
            click.echo(f"    {p['address']}: {p['original']} -> {p['replacement']}")
        click.echo(f"  Cycle reduction: {comparison['cycle_reduction']:,} ({comparison['improvement_pct']:.2f}%)")


@main.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--arch", default="gfx942", help="Target GPU architecture")
@click.option("--no-cpp-swap", is_flag=True, help="Disable C++ template swap")
@click.option("--no-asm-replace", is_flag=True, help="Disable ASM pattern replacement")
@click.option("--no-asm-optimize", is_flag=True, help="Disable ASM optimization passes")
@click.option("--max-level", default=3, type=int, help="Max replacement safety level (1-6)")
@click.option("--aggressive", is_flag=True, help="Aggressive optimization")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.option("--report", type=click.Path(), help="Save detailed report to file")
def transform(source, arch, no_cpp_swap, no_asm_replace, no_asm_optimize,
              max_level, aggressive, json_output, report):
    """Transform a HIP C++ kernel through the full optimization pipeline.

    Analyzes the input, matches against production templates, applies
    C++ and ASM-level optimizations, and reports estimated improvements.
    """
    from src.pipeline import OptimizationPipeline

    source_code = Path(source).read_text()

    log = click.echo if not json_output else (lambda msg, **kw: click.echo(msg, err=True, **kw))
    log(f"Running AFTT v2 pipeline on {source} for {arch}...")

    pipeline = OptimizationPipeline(arch=arch)
    result = pipeline.run(
        source_code,
        source_path=source,
        enable_cpp_swap=not no_cpp_swap,
        enable_asm_replace=not no_asm_replace,
        enable_asm_optimize=not no_asm_optimize,
        max_replacement_level=max_level,
        aggressive=aggressive,
    )

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        click.echo(result.summary())

    if report:
        report_data = result.to_dict()
        Path(report).write_text(json.dumps(report_data, indent=2))
        log(f"Report saved to {report}")


if __name__ == "__main__":
    main()
