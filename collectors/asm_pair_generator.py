#!/usr/bin/env python3
"""C++ to ASM Pair Generator.

Compiles HIP kernel templates through amdclang++ for multiple architectures
and compiler flag combinations, producing JSON training pairs.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime

HIPCC = os.environ.get("HIPCC", "/opt/rocm/bin/hipcc")
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "db" / "cpp_asm_pairs"

TARGET_ARCHS = ["gfx908", "gfx90a", "gfx942"]

FLAG_SETS = [
    {"name": "O0", "flags": ["-O0"]},
    {"name": "O1", "flags": ["-O1"]},
    {"name": "O2", "flags": ["-O2"]},
    {"name": "O3", "flags": ["-O3"]},
    {"name": "Ofast", "flags": ["-Ofast"]},
    {"name": "O3_unroll", "flags": ["-O3", "-funroll-loops"]},
    {"name": "O3_fast_math", "flags": ["-O3", "-ffast-math"]},
]


def compile_to_asm(source_path: Path, arch: str, extra_flags: list[str]) -> tuple[str | None, str]:
    """Compile a HIP source file to AMDGPU assembly.

    Returns (asm_output, stderr).
    """
    with tempfile.NamedTemporaryFile(suffix=".s", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        HIPCC,
        "-S",  # output assembly
        f"--offload-arch={arch}",
        "-nogpulib",
        *extra_flags,
        str(source_path),
        "-o", tmp_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            with open(tmp_path, "r") as f:
                asm = f.read()
            os.unlink(tmp_path)
            return asm, result.stderr
        else:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return None, result.stderr
    except subprocess.TimeoutExpired:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None, "Compilation timed out"
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None, str(e)


def extract_kernel_asm(full_asm: str, kernel_name: str = None) -> str:
    """Extract only the GPU kernel assembly from full compiler output."""
    lines = full_asm.split("\n")
    in_kernel = False
    kernel_lines = []

    for line in lines:
        if ".amdgcn_target" in line or ".amdgpu_metadata" in line:
            in_kernel = True
        if in_kernel:
            kernel_lines.append(line)
        if kernel_name and line.strip().startswith(f"{kernel_name}:"):
            in_kernel = True
            kernel_lines.append(line)
        if in_kernel and line.strip() == "s_endpgm":
            kernel_lines.append(line)

    if not kernel_lines:
        return full_asm
    return "\n".join(kernel_lines)


def generate_pairs():
    """Generate all C++ -> ASM pairs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    templates = sorted(TEMPLATES_DIR.glob("*.hip"))

    if not templates:
        print("No .hip templates found in", TEMPLATES_DIR)
        return

    all_pairs = []
    total = 0
    errors = 0

    for template in templates:
        source_code = template.read_text()
        template_name = template.stem

        for arch in TARGET_ARCHS:
            for flag_set in FLAG_SETS:
                pair_id = f"{template_name}_{arch}_{flag_set['name']}"
                print(f"  Compiling {pair_id}...", end=" ")

                asm_output, stderr = compile_to_asm(
                    template, arch, flag_set["flags"]
                )

                if asm_output:
                    pair = {
                        "id": pair_id,
                        "template_name": template_name,
                        "cpp_source": source_code,
                        "asm_output": asm_output,
                        "gfx_arch": arch,
                        "compiler_flags": flag_set["flags"],
                        "flag_set_name": flag_set["name"],
                        "compiler": HIPCC,
                        "timestamp": datetime.now().isoformat(),
                    }
                    all_pairs.append(pair)
                    total += 1
                    print("OK")
                else:
                    errors += 1
                    print(f"FAILED: {stderr[:200]}")

    # Write combined output
    out_file = OUTPUT_DIR / "cpp_asm_pairs.json"
    with open(out_file, "w") as f:
        json.dump({"pairs": all_pairs, "total": total}, f, indent=2)
    print(f"\nGenerated {total} pairs ({errors} errors) -> {out_file}")

    # Write per-arch summaries
    for arch in TARGET_ARCHS:
        arch_pairs = [p for p in all_pairs if p["gfx_arch"] == arch]
        arch_file = OUTPUT_DIR / f"pairs_{arch}.json"
        with open(arch_file, "w") as f:
            json.dump({"arch": arch, "pairs": arch_pairs, "total": len(arch_pairs)}, f, indent=2)
        print(f"  {arch}: {len(arch_pairs)} pairs")


if __name__ == "__main__":
    generate_pairs()
