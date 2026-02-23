#!/usr/bin/env python3
"""Batch disassembler for AMDGPU .co (code object) kernel binaries.

Disassembles all .co files from aiter/hsa/ using llvm-objdump and stores
the results as .asm text files plus a summary JSON for each kernel.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

LLVM_OBJDUMP = os.environ.get("LLVM_OBJDUMP", "/opt/rocm-7.1.1/llvm/bin/llvm-objdump")
AITER_HSA = Path(os.environ.get("AITER_HSA", "/home/root123/aiter/hsa"))
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "db" / "disassembly"

RE_FUNC_LABEL = re.compile(r'^([0-9a-f]+)\s+<(.+)>:')
RE_INSTRUCTION = re.compile(r'^\s+([a-z_]\S+)\s*(.*?)(?:\s*//.*)?$')
RE_MFMA = re.compile(r'v_mfma|v_smfma')
RE_WAITCNT = re.compile(r's_waitcnt')
RE_VMCNT = re.compile(r'vmcnt\((\d+)\)')
RE_LGKMCNT = re.compile(r'lgkmcnt\((\d+)\)')
RE_DS_READ = re.compile(r'ds_read|ds_load')
RE_DS_WRITE = re.compile(r'ds_write|ds_store')
RE_GLOBAL_LOAD = re.compile(r'global_load')
RE_BUFFER_LOAD = re.compile(r'buffer_load')
RE_GLOBAL_STORE = re.compile(r'global_store')
RE_BUFFER_STORE = re.compile(r'buffer_store')
RE_NOP = re.compile(r's_nop')
RE_BARRIER = re.compile(r's_barrier')
RE_SETPRIO = re.compile(r's_setprio')
RE_VGPR = re.compile(r'v(\d+)|v\[(\d+):(\d+)\]')
RE_SGPR = re.compile(r's(\d+)|s\[(\d+):(\d+)\]')
RE_AGPR = re.compile(r'a(\d+)|a\[(\d+):(\d+)\]')


def categorize_co(path: Path) -> str:
    """Categorize a .co file by its directory and name."""
    name = path.stem.lower()
    parent = path.parent.name.lower()
    if "gemm" in name or "gemm" in parent:
        return "gemm"
    if "pa_" in name or parent == "pa":
        return "attention"
    if "mla" in name or parent == "mla":
        return "mla"
    if "moe" in name or "fmoe" in name or "moe" in parent or "fmoe" in parent:
        return "moe"
    if "fmha" in name or "fmha" in parent:
        return "fmha"
    if "topk" in name or "topk" in parent:
        return "topk"
    if "norm" in name:
        return "norm"
    if "reduce" in name or "allreduce" in name:
        return "comm"
    return "other"


def disassemble_co(co_path: Path) -> tuple[str, str]:
    """Disassemble a single .co file. Returns (asm_text, error)."""
    try:
        result = subprocess.run(
            [LLVM_OBJDUMP, "-d", "--no-leading-addr", str(co_path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return result.stdout, ""
        return "", result.stderr[:500]
    except subprocess.TimeoutExpired:
        return "", "timeout"
    except Exception as e:
        return "", str(e)[:500]


def quick_analyze(asm_text: str) -> dict:
    """Quick instruction-level analysis of disassembled ASM."""
    stats = {
        "total_instructions": 0,
        "mfma_count": 0,
        "mfma_types": {},
        "ds_read_count": 0,
        "ds_write_count": 0,
        "global_load_count": 0,
        "buffer_load_count": 0,
        "global_store_count": 0,
        "buffer_store_count": 0,
        "waitcnt_count": 0,
        "waitcnt_vmcnt0": 0,
        "waitcnt_lgkmcnt0": 0,
        "waitcnt_partial": 0,
        "barrier_count": 0,
        "nop_count": 0,
        "setprio_count": 0,
        "valu_count": 0,
        "salu_count": 0,
        "max_vgpr": 0,
        "max_sgpr": 0,
        "max_agpr": 0,
        "functions": [],
        "dwordx4_loads": 0,
        "dword_single_loads": 0,
    }

    for line in asm_text.split("\n"):
        func_m = RE_FUNC_LABEL.match(line)
        if func_m:
            stats["functions"].append(func_m.group(2))
            continue

        instr_m = RE_INSTRUCTION.match(line)
        if not instr_m:
            continue

        mnemonic = instr_m.group(1)
        operands = instr_m.group(2)
        full_line = line.strip()
        stats["total_instructions"] += 1

        if RE_MFMA.match(mnemonic):
            stats["mfma_count"] += 1
            stats["mfma_types"][mnemonic] = stats["mfma_types"].get(mnemonic, 0) + 1

        if RE_DS_READ.match(mnemonic):
            stats["ds_read_count"] += 1
        elif RE_DS_WRITE.match(mnemonic):
            stats["ds_write_count"] += 1
        elif RE_GLOBAL_LOAD.match(mnemonic):
            stats["global_load_count"] += 1
            if "dwordx4" in mnemonic:
                stats["dwordx4_loads"] += 1
            elif "dwordx2" not in mnemonic and "dwordx3" not in mnemonic:
                stats["dword_single_loads"] += 1
        elif RE_BUFFER_LOAD.match(mnemonic):
            stats["buffer_load_count"] += 1
            if "dwordx4" in mnemonic:
                stats["dwordx4_loads"] += 1
            elif "dwordx2" not in mnemonic and "dwordx3" not in mnemonic:
                stats["dword_single_loads"] += 1
        elif RE_GLOBAL_STORE.match(mnemonic):
            stats["global_store_count"] += 1
        elif RE_BUFFER_STORE.match(mnemonic):
            stats["buffer_store_count"] += 1

        if RE_WAITCNT.match(mnemonic):
            stats["waitcnt_count"] += 1
            vm = RE_VMCNT.search(full_line)
            lgkm = RE_LGKMCNT.search(full_line)
            is_zero = False
            if vm and int(vm.group(1)) == 0:
                stats["waitcnt_vmcnt0"] += 1
                is_zero = True
            if lgkm and int(lgkm.group(1)) == 0:
                stats["waitcnt_lgkmcnt0"] += 1
                is_zero = True
            if (vm and int(vm.group(1)) > 0) or (lgkm and int(lgkm.group(1)) > 0):
                stats["waitcnt_partial"] += 1

        if RE_BARRIER.match(mnemonic):
            stats["barrier_count"] += 1
        if RE_NOP.match(mnemonic):
            stats["nop_count"] += 1
        if RE_SETPRIO.match(mnemonic):
            stats["setprio_count"] += 1

        if mnemonic.startswith("v_") and not RE_MFMA.match(mnemonic):
            stats["valu_count"] += 1
        elif mnemonic.startswith("s_"):
            stats["salu_count"] += 1

        # Track register usage
        for m in RE_VGPR.finditer(operands):
            if m.group(1):
                stats["max_vgpr"] = max(stats["max_vgpr"], int(m.group(1)))
            elif m.group(3):
                stats["max_vgpr"] = max(stats["max_vgpr"], int(m.group(3)))
        for m in RE_SGPR.finditer(operands):
            if m.group(1):
                stats["max_sgpr"] = max(stats["max_sgpr"], int(m.group(1)))
            elif m.group(3):
                stats["max_sgpr"] = max(stats["max_sgpr"], int(m.group(3)))
        for m in RE_AGPR.finditer(operands):
            if m.group(1):
                stats["max_agpr"] = max(stats["max_agpr"], int(m.group(1)))
            elif m.group(3):
                stats["max_agpr"] = max(stats["max_agpr"], int(m.group(3)))

    return stats


def process_one(co_path: Path, arch: str) -> dict:
    """Process a single .co file: disassemble + analyze."""
    category = categorize_co(co_path)
    rel_path = co_path.relative_to(AITER_HSA / arch) if arch in str(co_path) else co_path.name

    out_dir = OUTPUT_DIR / arch / category
    out_dir.mkdir(parents=True, exist_ok=True)
    asm_file = out_dir / f"{co_path.stem}.asm"

    asm_text, error = disassemble_co(co_path)
    if error:
        return {"file": str(co_path), "error": error}

    asm_file.write_text(asm_text)
    stats = quick_analyze(asm_text)
    stats["source_co"] = str(rel_path)
    stats["arch"] = arch
    stats["category"] = category
    stats["asm_file"] = str(asm_file.relative_to(OUTPUT_DIR))

    json_file = out_dir / f"{co_path.stem}.json"
    with open(json_file, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    errors = 0

    for arch in ["gfx942", "gfx950"]:
        arch_dir = AITER_HSA / arch
        if not arch_dir.exists():
            print(f"Skipping {arch}: directory not found")
            continue

        co_files = sorted(arch_dir.rglob("*.co"))
        print(f"\n{arch}: found {len(co_files)} .co files")

        for i, co_path in enumerate(co_files):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i+1}/{len(co_files)}] Processing {co_path.name}...")

            result = process_one(co_path, arch)
            if "error" in result:
                errors += 1
            else:
                all_results.append(result)

    # Write combined summary
    summary = {
        "total_kernels": len(all_results),
        "errors": errors,
        "by_arch": {},
        "by_category": {},
        "kernels": all_results,
    }

    for r in all_results:
        arch = r.get("arch", "unknown")
        cat = r.get("category", "unknown")
        summary["by_arch"].setdefault(arch, {"count": 0, "total_mfma": 0, "total_instructions": 0})
        summary["by_arch"][arch]["count"] += 1
        summary["by_arch"][arch]["total_mfma"] += r.get("mfma_count", 0)
        summary["by_arch"][arch]["total_instructions"] += r.get("total_instructions", 0)

        summary["by_category"].setdefault(cat, {"count": 0, "total_mfma": 0})
        summary["by_category"][cat]["count"] += 1
        summary["by_category"][cat]["total_mfma"] += r.get("mfma_count", 0)

    summary_file = OUTPUT_DIR / "disassembly_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Disassembled {len(all_results)} kernels ({errors} errors)")
    print(f"\nBy architecture:")
    for arch, info in summary["by_arch"].items():
        print(f"  {arch}: {info['count']} kernels, {info['total_mfma']} MFMA, {info['total_instructions']} total instructions")
    print(f"\nBy category:")
    for cat, info in sorted(summary["by_category"].items()):
        print(f"  {cat:12s}: {info['count']:>5} kernels, {info['total_mfma']:>7} MFMA")
    print(f"\nSummary written to {summary_file}")


if __name__ == "__main__":
    main()
