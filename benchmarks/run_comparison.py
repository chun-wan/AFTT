#!/usr/bin/env python3
"""HIP Kernel Benchmark Comparison.

Compiles, validates, benchmarks, and analyzes one or more HIP kernel sources.
Produces a JSON comparison report with timing, bandwidth, correctness, and
ASM-level metrics.

Usage:
  python run_comparison.py --sources kernel_a.hip kernel_b.hip ...
  python run_comparison.py --sources aiter_rmsnorm_standalone.hip
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.compiler import Compiler

HIPCC = os.environ.get("HIPCC", "/opt/rocm/bin/hipcc")
ARCH = "gfx942"
NUM_TOKENS = 128
HIDDEN_SIZE = 8192
CORRECTNESS_TOL = 1e-3


@dataclass
class VariantResult:
    name: str
    source_path: str = ""
    compile_ok: bool = False
    compile_error: str = ""
    correctness_ok: bool = False
    max_error: float = -1.0
    time_us: float = -1.0
    bandwidth_gbs: float = -1.0
    asm_instruction_count: int = 0
    asm_analysis: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.compile_ok and self.correctness_ok


def compile_hip(source_path: str, output_path: str) -> tuple[bool, str]:
    cmd = [HIPCC, "-O3", f"--offload-arch={ARCH}", str(source_path), "-o", str(output_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return r.returncode == 0, r.stderr
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out"


def run_benchmark(exe_path: str) -> tuple[bool, float, float, float]:
    try:
        r = subprocess.run(
            [str(exe_path), str(NUM_TOKENS), str(HIDDEN_SIZE)],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False, -1, -1, -1

    output = r.stdout + r.stderr
    time_us = _parse_float(output, r"Time:\s*([\d.]+)\s*us")
    bw = _parse_float(output, r"Bandwidth:\s*([\d.]+)\s*GB/s")
    max_err = _parse_float(output, r"Max error:\s*([\d.eE+-]+)")
    ok = "PASS" in output and r.returncode == 0
    return ok, time_us, bw, max_err


def _parse_float(text: str, pattern: str) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else -1.0


def disassemble_and_analyze(exe_path: str) -> tuple[int, dict]:
    compiler = Compiler()
    try:
        r = subprocess.run(
            ["llvm-objdump", "--offloading", str(exe_path)],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0, {}

    co_files = re.findall(r"file '([^']+gfx942[^']*)'", r.stdout)
    if not co_files:
        return 0, {}

    result = compiler.disassemble_binary(co_files[0], ARCH)
    if not result.success:
        return 0, {}

    instr_count = len(result.instruction_lines)
    analysis = {}
    try:
        asm = result.asm_output
        analysis = {
            "total_instructions": instr_count,
            "valu_ops": len(re.findall(r'\bv_\w+', asm)),
            "vmem_ops": len(re.findall(r'\b(?:global|buffer)_\w+', asm)),
            "salu_ops": len(re.findall(r'\bs_\w+', asm)),
            "lds_ops": len(re.findall(r'\bds_\w+', asm)),
            "barriers": len(re.findall(r'\bs_barrier\b', asm)),
            "waitcnts": len(re.findall(r'\bs_waitcnt\b', asm)),
            "vec4_loads": len(re.findall(r'dwordx4', asm)),
            "shuffle_ops": len(re.findall(r'dpp|permute|swizzle|readlane', asm)),
        }
    except Exception:
        pass
    return instr_count, analysis


def evaluate_variant(name: str, source_path: str, workdir: Path) -> VariantResult:
    result = VariantResult(name=name, source_path=source_path)

    exe_path = workdir / name
    ok, err = compile_hip(source_path, str(exe_path))
    result.compile_ok = ok
    if not ok:
        result.compile_error = err
        print(f"  [{name}] COMPILE FAILED: {err[:200]}")
        return result

    print(f"  [{name}] Compiled successfully")

    ok, time_us, bw, max_err = run_benchmark(str(exe_path))
    result.correctness_ok = ok and max_err >= 0 and max_err < CORRECTNESS_TOL
    result.max_error = max_err
    result.time_us = time_us
    result.bandwidth_gbs = bw

    if not result.correctness_ok:
        print(f"  [{name}] CORRECTNESS FAILED: max_error={max_err:.2e}")
    else:
        print(f"  [{name}] PASS: {time_us:.2f} us, {bw:.1f} GB/s, err={max_err:.2e}")

    instr_count, analysis = disassemble_and_analyze(str(exe_path))
    result.asm_instruction_count = instr_count
    result.asm_analysis = analysis

    return result


def main():
    parser = argparse.ArgumentParser(description="HIP Kernel Benchmark Comparison")
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="HIP source files to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "output" / "comparison"),
    )
    args = parser.parse_args()

    workdir = Path(args.output_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HIP Kernel Benchmark Comparison")
    print(f"Config: {NUM_TOKENS} tokens, hidden_size={HIDDEN_SIZE}, arch={ARCH}")
    print("=" * 60)

    results = []
    for src in args.sources:
        name = Path(src).stem
        print(f"\n--- Evaluating: {name} ---")
        results.append(evaluate_variant(name, src, workdir))

    baseline = results[0] if results and results[0].passed else None

    report = {
        "config": {
            "num_tokens": NUM_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "arch": ARCH,
        },
        "variants": {},
    }

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    for res in results:
        entry = {
            "compile_ok": res.compile_ok,
            "correctness_ok": res.correctness_ok,
            "max_error": res.max_error,
            "time_us": res.time_us,
            "bandwidth_gbs": res.bandwidth_gbs,
            "asm_instruction_count": res.asm_instruction_count,
            "asm_analysis": res.asm_analysis,
        }
        report["variants"][res.name] = entry

        if res.passed:
            speedup = ""
            if baseline and baseline is not res and baseline.time_us > 0 and res.time_us > 0:
                ratio = baseline.time_us / res.time_us
                speedup = f"  ({ratio:.2f}x vs {baseline.name})"
            print(f"  {res.name:30s}  {res.time_us:8.2f} us  {res.bandwidth_gbs:8.1f} GB/s{speedup}")
        else:
            status = "compile_fail" if not res.compile_ok else "incorrect"
            print(f"  {res.name:30s}  {status}")

    report_path = workdir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
