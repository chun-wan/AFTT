#!/usr/bin/env python3
"""Pattern Extractor.

Extracts optimization patterns from the aiter/CK codebase for the pattern database.
Analyzes pipeline headers, ASM kernel configs, and tutorial code.
"""

import json
import re
import os
from pathlib import Path

AITER_DIR = Path("/home/root123/aiter")
CK_DIR = AITER_DIR / "3rdparty" / "composable_kernel"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "db" / "patterns"


def extract_pipeline_patterns():
    """Extract pipeline patterns from CK GEMM pipeline headers."""
    pipeline_dir = CK_DIR / "include" / "ck_tile" / "ops" / "gemm" / "pipeline"
    patterns = []

    if not pipeline_dir.exists():
        print(f"Warning: Pipeline dir not found: {pipeline_dir}")
        return patterns

    for hpp in sorted(pipeline_dir.glob("*.hpp")):
        content = hpp.read_text(errors="ignore")
        name = hpp.stem

        pattern_info = {
            "source_file": str(hpp),
            "pipeline_name": name,
            "features": [],
        }

        if "DoubleSmemBuffer" in content:
            pattern_info["features"].append("double_smem_buffer")
        if "ping" in content.lower() or "pong" in content.lower():
            pattern_info["features"].append("ping_pong_lds")
        if "async" in content.lower() or "Async" in content:
            pattern_info["features"].append("async_copy")
        if "prefetch" in content.lower() or "Prefetch" in content:
            pattern_info["features"].append("prefetch")
        if "s_waitcnt" in content:
            pattern_info["features"].append("explicit_waitcnt")
        if "mfma" in content.lower():
            pattern_info["features"].append("mfma_usage")

        # Extract template parameters
        template_match = re.findall(r'template\s*<([^>]+)>', content[:2000])
        if template_match:
            pattern_info["template_params"] = template_match[0].strip()[:200]

        if pattern_info["features"]:
            patterns.append(pattern_info)

    return patterns


def extract_asm_kernel_configs():
    """Extract ASM kernel configurations from aiter/hsa/ directory."""
    hsa_dir = AITER_DIR / "hsa"
    configs = []

    if not hsa_dir.exists():
        return configs

    for arch_dir in sorted(hsa_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name
        for csv_file in sorted(arch_dir.rglob("*.csv")):
            try:
                content = csv_file.read_text(errors="ignore")
                lines = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("#")]
                configs.append({
                    "arch": arch,
                    "config_file": str(csv_file.relative_to(hsa_dir)),
                    "kernel_count": len(lines),
                    "sample_entries": lines[:5],
                })
            except Exception:
                pass

    return configs


def extract_tutorial_patterns():
    """Extract learning patterns from CK tutorials."""
    tutorial_dir = CK_DIR / "tutorial" / "ck_tile"
    patterns = []

    if not tutorial_dir.exists():
        return patterns

    for tutorial in sorted(tutorial_dir.iterdir()):
        if not tutorial.is_dir():
            continue

        info = {
            "tutorial_name": tutorial.name,
            "files": [],
            "concepts": [],
        }

        for hpp in sorted(tutorial.rglob("*.hpp")):
            content = hpp.read_text(errors="ignore")
            info["files"].append(str(hpp.relative_to(tutorial_dir)))

            if "gmem" in hpp.name.lower():
                info["concepts"].append("global_memory_pipeline")
            if "smem" in hpp.name.lower():
                info["concepts"].append("shared_memory_pipeline")
            if "warp" in hpp.name.lower():
                info["concepts"].append("warp_level_pipeline")
            if "block" in hpp.name.lower():
                info["concepts"].append("block_level_pipeline")
            if "host" in hpp.name.lower():
                info["concepts"].append("host_level_pipeline")

        info["concepts"] = list(set(info["concepts"]))
        if info["files"]:
            patterns.append(info)

    return patterns


def extract_aiter_kernel_types():
    """Catalog kernel types available in aiter."""
    kernel_types = []
    py_itfs_dir = AITER_DIR / "csrc" / "py_itfs_cu"

    if py_itfs_dir.exists():
        for cu_file in sorted(py_itfs_dir.glob("asm_*.cu")):
            name = cu_file.stem
            content = cu_file.read_text(errors="ignore")[:3000]

            kernel_info = {
                "name": name,
                "file": str(cu_file.relative_to(AITER_DIR)),
                "type": "asm_kernel",
            }

            if "gemm" in name.lower():
                kernel_info["category"] = "GEMM"
            elif "pa" in name.lower() or "attention" in name.lower():
                kernel_info["category"] = "Attention"
            elif "mla" in name.lower():
                kernel_info["category"] = "MLA"
            elif "moe" in name.lower() or "fmoe" in name.lower():
                kernel_info["category"] = "MoE"
            elif "topk" in name.lower():
                kernel_info["category"] = "TopK"
            else:
                kernel_info["category"] = "Other"

            kernel_types.append(kernel_info)

    return kernel_types


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Extracting pipeline patterns from CK...")
    pipelines = extract_pipeline_patterns()
    print(f"  Found {len(pipelines)} pipeline patterns")

    print("Extracting ASM kernel configs from hsa/...")
    asm_configs = extract_asm_kernel_configs()
    print(f"  Found {len(asm_configs)} ASM kernel configs")

    print("Extracting tutorial patterns...")
    tutorials = extract_tutorial_patterns()
    print(f"  Found {len(tutorials)} tutorial patterns")

    print("Extracting aiter kernel types...")
    kernel_types = extract_aiter_kernel_types()
    print(f"  Found {len(kernel_types)} ASM kernel types")

    extracted = {
        "pipeline_patterns": pipelines,
        "asm_kernel_configs": asm_configs,
        "tutorial_patterns": tutorials,
        "aiter_kernel_types": kernel_types,
    }

    out_file = OUTPUT_DIR / "extracted_patterns.json"
    with open(out_file, "w") as f:
        json.dump(extracted, f, indent=2)
    print(f"\nWrote extracted patterns to {out_file}")


if __name__ == "__main__":
    main()
