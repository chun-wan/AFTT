#!/usr/bin/env python3
"""Compiler Flag Effect Tester.

Compiles the same kernels with different compiler flag combinations and
catalogs the differences in generated ASM output.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from difflib import unified_diff

HIPCC = os.environ.get("HIPCC", "/opt/rocm-7.1.1/bin/hipcc")
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "db" / "compiler_flags"

TARGET_ARCH = "gfx942"

FLAG_COMPARISONS = [
    {
        "name": "O0_vs_O3",
        "description": "Effect of optimization level: no optimization vs aggressive optimization",
        "baseline": ["-O0"],
        "variant": ["-O3"],
    },
    {
        "name": "O2_vs_O3",
        "description": "Marginal improvement from O2 to O3",
        "baseline": ["-O2"],
        "variant": ["-O3"],
    },
    {
        "name": "O3_vs_Ofast",
        "description": "Effect of -Ofast (enables unsafe math optimizations)",
        "baseline": ["-O3"],
        "variant": ["-Ofast"],
    },
    {
        "name": "O3_vs_fast_math",
        "description": "Effect of -ffast-math flag",
        "baseline": ["-O3"],
        "variant": ["-O3", "-ffast-math"],
    },
    {
        "name": "O3_vs_unroll",
        "description": "Effect of aggressive loop unrolling",
        "baseline": ["-O3"],
        "variant": ["-O3", "-funroll-loops"],
    },
]


def compile_to_asm(source_path: Path, arch: str, flags: list[str]) -> str | None:
    with tempfile.NamedTemporaryFile(suffix=".s", delete=False) as tmp:
        tmp_path = tmp.name
    cmd = [HIPCC, "-S", f"--offload-arch={arch}", "-nogpulib", *flags, str(source_path), "-o", tmp_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.exists(tmp_path):
            with open(tmp_path) as f:
                asm = f.read()
            os.unlink(tmp_path)
            return asm
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None


def count_instructions(asm_text: str) -> dict:
    """Count instruction types in ASM output."""
    counts = {
        "total_lines": 0,
        "instructions": 0,
        "v_instructions": 0,
        "s_instructions": 0,
        "ds_instructions": 0,
        "global_instructions": 0,
        "buffer_instructions": 0,
        "mfma_instructions": 0,
        "waitcnt": 0,
        "barriers": 0,
        "branches": 0,
    }

    for line in asm_text.split("\n"):
        line = line.strip()
        if not line or line.startswith(("//", ";", ".", "@")):
            continue
        counts["total_lines"] += 1

        parts = line.split()
        if not parts:
            continue
        mnemonic = parts[0].rstrip(":")

        if mnemonic.startswith("v_"):
            counts["v_instructions"] += 1
            counts["instructions"] += 1
            if "mfma" in mnemonic:
                counts["mfma_instructions"] += 1
        elif mnemonic.startswith("s_"):
            counts["s_instructions"] += 1
            counts["instructions"] += 1
            if "waitcnt" in mnemonic:
                counts["waitcnt"] += 1
            elif "barrier" in mnemonic:
                counts["barriers"] += 1
            elif "branch" in mnemonic:
                counts["branches"] += 1
        elif mnemonic.startswith("ds_"):
            counts["ds_instructions"] += 1
            counts["instructions"] += 1
        elif mnemonic.startswith("global_"):
            counts["global_instructions"] += 1
            counts["instructions"] += 1
        elif mnemonic.startswith("buffer_"):
            counts["buffer_instructions"] += 1
            counts["instructions"] += 1

    return counts


def analyze_diff(baseline_asm: str, variant_asm: str) -> dict:
    """Analyze differences between two ASM outputs."""
    baseline_counts = count_instructions(baseline_asm)
    variant_counts = count_instructions(variant_asm)

    diff_lines = list(unified_diff(
        baseline_asm.split("\n"),
        variant_asm.split("\n"),
        lineterm="",
        n=0,
    ))

    changes = {
        "added_lines": sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++")),
        "removed_lines": sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---")),
        "baseline_counts": baseline_counts,
        "variant_counts": variant_counts,
        "count_diffs": {},
    }

    for key in baseline_counts:
        diff = variant_counts[key] - baseline_counts[key]
        if diff != 0:
            changes["count_diffs"][key] = {
                "baseline": baseline_counts[key],
                "variant": variant_counts[key],
                "delta": diff,
                "pct_change": round(100 * diff / max(baseline_counts[key], 1), 1),
            }

    return changes


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    templates = sorted(TEMPLATES_DIR.glob("*.hip"))

    if not templates:
        print("No templates found")
        return

    all_results = []

    for comparison in FLAG_COMPARISONS:
        print(f"\n=== {comparison['name']}: {comparison['description']} ===")

        for template in templates:
            template_name = template.stem
            print(f"  {template_name}...", end=" ")

            baseline_asm = compile_to_asm(template, TARGET_ARCH, comparison["baseline"])
            variant_asm = compile_to_asm(template, TARGET_ARCH, comparison["variant"])

            if baseline_asm and variant_asm:
                analysis = analyze_diff(baseline_asm, variant_asm)
                result = {
                    "comparison": comparison["name"],
                    "description": comparison["description"],
                    "template": template_name,
                    "arch": TARGET_ARCH,
                    "baseline_flags": comparison["baseline"],
                    "variant_flags": comparison["variant"],
                    "analysis": analysis,
                }
                all_results.append(result)

                diffs = analysis["count_diffs"]
                if diffs:
                    summary_parts = []
                    for k, v in diffs.items():
                        summary_parts.append(f"{k}: {v['delta']:+d} ({v['pct_change']:+.1f}%)")
                    print("; ".join(summary_parts[:3]))
                else:
                    print("no difference")
            else:
                print("compilation failed")

    out_file = OUTPUT_DIR / "flag_effects.json"
    with open(out_file, "w") as f:
        json.dump({"comparisons": all_results, "arch": TARGET_ARCH}, f, indent=2)
    print(f"\nWrote {len(all_results)} comparisons to {out_file}")


if __name__ == "__main__":
    main()
