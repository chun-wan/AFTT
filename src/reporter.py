"""Report Generator for ASM Kernel Analysis.

Produces human-readable reports from analysis results with severity levels,
line references, and actionable suggestions.
"""

from __future__ import annotations

from .analyzer import AnalysisResult, Finding
from .parser import ParsedKernel
from .isa_db import Instruction


SEVERITY_SYMBOLS = {
    "critical": "[CRITICAL]",
    "warning": "[WARNING]",
    "info": "[INFO]",
}

SEVERITY_COLORS = {
    "critical": "red",
    "warning": "yellow",
    "info": "cyan",
}


def format_finding_plain(finding: Finding, index: int) -> str:
    """Format a single finding as plain text."""
    lines = []
    severity = SEVERITY_SYMBOLS.get(finding.severity, "[?]")
    lines.append(f"  {severity} #{index}: {finding.title}")
    lines.append(f"    {finding.description}")

    if finding.line_numbers:
        line_refs = ", ".join(str(ln) for ln in finding.line_numbers[:8])
        if len(finding.line_numbers) > 8:
            line_refs += f" ... (+{len(finding.line_numbers) - 8} more)"
        lines.append(f"    Lines: {line_refs}")

    lines.append(f"    Suggestion: {finding.suggestion}")

    if finding.reference:
        lines.append(f"    Reference: {finding.reference}")

    if finding.metrics:
        metrics_str = ", ".join(f"{k}={v}" for k, v in finding.metrics.items())
        lines.append(f"    Metrics: {metrics_str}")

    return "\n".join(lines)


def format_summary_plain(result: AnalysisResult) -> str:
    """Format the analysis summary as plain text."""
    s = result.summary
    lines = []

    lines.append("=" * 70)
    lines.append(f"  ASM Kernel Analysis Report")
    lines.append(f"  Kernel: {result.kernel_name}")
    lines.append(f"  Target: {result.arch}")
    lines.append("=" * 70)

    # Instruction counts
    lines.append("\n  Instruction Summary:")
    lines.append(f"    Total Instructions:  {s.get('total_instructions', 0):>6}")
    lines.append(f"    VALU (Vector ALU):   {s.get('valu_instructions', 0):>6}")
    lines.append(f"    SALU (Scalar ALU):   {s.get('salu_instructions', 0):>6}")
    lines.append(f"    VMEM (Vector Mem):   {s.get('vmem_instructions', 0):>6}")
    lines.append(f"    SMEM (Scalar Mem):   {s.get('smem_instructions', 0):>6}")
    lines.append(f"    LDS:                 {s.get('lds_instructions', 0):>6}")
    lines.append(f"    MFMA:                {s.get('mfma_instructions', 0):>6}")
    lines.append(f"    Branches:            {s.get('branch_instructions', 0):>6}")
    lines.append(f"    Waitcnts:            {s.get('waitcnt_instructions', 0):>6}")
    lines.append(f"    Barriers:            {s.get('barrier_instructions', 0):>6}")
    lines.append(f"    NOPs:                {s.get('nop_instructions', 0):>6}")

    # Resource usage
    lines.append("\n  Resource Usage:")
    lines.append(f"    VGPRs:               {s.get('vgpr_count', 0):>6}")
    lines.append(f"    SGPRs:               {s.get('sgpr_count', 0):>6}")
    lines.append(f"    LDS (bytes):         {s.get('lds_size_bytes', 0):>6}")
    lines.append(f"    Scratch (bytes):     {s.get('scratch_size_bytes', 0):>6}")
    lines.append(f"    Est. Occupancy:      {s.get('estimated_occupancy_waves', 0):>4} waves/SIMD")

    # Finding counts
    crit = s.get("findings_critical", 0)
    warn = s.get("findings_warning", 0)
    info = s.get("findings_info", 0)
    total = crit + warn + info
    lines.append(f"\n  Findings: {total} total ({crit} critical, {warn} warning, {info} info)")

    return "\n".join(lines)


def format_report_plain(result: AnalysisResult) -> str:
    """Generate a complete plain-text analysis report."""
    parts = [format_summary_plain(result)]

    if result.findings:
        parts.append("\n" + "-" * 70)
        parts.append("  FINDINGS")
        parts.append("-" * 70)

        # Sort: critical first, then warning, then info
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        sorted_findings = sorted(
            result.findings, key=lambda f: severity_order.get(f.severity, 3)
        )

        for i, finding in enumerate(sorted_findings, 1):
            parts.append("")
            parts.append(format_finding_plain(finding, i))

    parts.append("\n" + "=" * 70)

    return "\n".join(parts)


def format_report_rich(result: AnalysisResult) -> str:
    """Generate a report using Rich markup for terminal output."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from io import StringIO

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=100)

        # Header
        console.print(Panel(
            f"[bold]ASM Kernel Analysis Report[/bold]\n"
            f"Kernel: [cyan]{result.kernel_name}[/cyan]  |  "
            f"Target: [cyan]{result.arch}[/cyan]",
            title="asm-advisor",
        ))

        # Instruction summary table
        s = result.summary
        table = Table(title="Instruction Summary", show_lines=False)
        table.add_column("Category", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Pct", justify="right")

        total = s.get("total_instructions", 1)
        for label, key in [
            ("VALU", "valu_instructions"),
            ("SALU", "salu_instructions"),
            ("VMEM", "vmem_instructions"),
            ("SMEM", "smem_instructions"),
            ("LDS", "lds_instructions"),
            ("MFMA", "mfma_instructions"),
            ("Branch", "branch_instructions"),
            ("Waitcnt", "waitcnt_instructions"),
            ("Barrier", "barrier_instructions"),
            ("NOP", "nop_instructions"),
        ]:
            count = s.get(key, 0)
            pct = f"{100 * count / total:.1f}%" if total > 0 else "0%"
            table.add_row(label, str(count), pct)

        table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]", "100%")
        console.print(table)

        # Resource usage
        res_table = Table(title="Resource Usage", show_lines=False)
        res_table.add_column("Resource", style="bold")
        res_table.add_column("Value", justify="right")
        res_table.add_column("Impact")
        res_table.add_row("VGPRs", str(s.get("vgpr_count", 0)), "")
        res_table.add_row("SGPRs", str(s.get("sgpr_count", 0)), "")
        res_table.add_row("LDS", f"{s.get('lds_size_bytes', 0)} B", "")
        res_table.add_row("Scratch", f"{s.get('scratch_size_bytes', 0)} B",
                          "[red]SPILLING[/red]" if s.get("scratch_size_bytes", 0) > 0 else "[green]None[/green]")
        res_table.add_row("Occupancy", f"{s.get('estimated_occupancy_waves', 0)} waves/SIMD", "")
        console.print(res_table)

        # Findings
        if result.findings:
            severity_order = {"critical": 0, "warning": 1, "info": 2}
            sorted_findings = sorted(
                result.findings, key=lambda f: severity_order.get(f.severity, 3)
            )

            for i, f in enumerate(sorted_findings, 1):
                color = SEVERITY_COLORS.get(f.severity, "white")
                sev = SEVERITY_SYMBOLS.get(f.severity, "[?]")

                text = Text()
                text.append(f"\n  {sev} ", style=f"bold {color}")
                text.append(f"#{i}: {f.title}\n", style="bold")
                text.append(f"    {f.description}\n")

                if f.line_numbers:
                    line_refs = ", ".join(str(ln) for ln in f.line_numbers[:8])
                    text.append(f"    Lines: {line_refs}\n", style="dim")

                text.append(f"    Suggestion: ", style="bold green")
                text.append(f"{f.suggestion}\n")

                if f.reference:
                    text.append(f"    Ref: {f.reference}\n", style="dim")

                console.print(text)
        else:
            console.print("[green]No issues found![/green]")

        return buf.getvalue()

    except ImportError:
        return format_report_plain(result)


def format_instruction_info(instr: Instruction) -> str:
    """Format ISA instruction information for display."""
    lines = []
    lines.append(f"Instruction: {instr.mnemonic}")
    lines.append(f"  Category:    {instr.category}")
    lines.append(f"  Description: {instr.description}")
    lines.append(f"  Operands:    {instr.operands}")
    lines.append(f"  Encoding:    {instr.encoding}")
    lines.append(f"  Latency:     {instr.latency_cycles} cycles")
    lines.append(f"  Throughput:  {instr.throughput_ops_per_cycle} ops/cycle")
    lines.append(f"  Architectures: {', '.join(instr.supported_archs)}")
    if instr.new_in:
        lines.append(f"  New in:      {instr.new_in}")
    if instr.deprecated_in:
        lines.append(f"  Deprecated:  {instr.deprecated_in}")
    if instr.notes:
        lines.append(f"  Notes:       {instr.notes}")
    return "\n".join(lines)
