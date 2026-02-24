#!/usr/bin/env python3
"""Phase 2A: TRT-LLM Batch HIPify Pipeline.

Batch-converts TensorRT-LLM CUDA kernels to HIP using hipify-perl,
classifies algorithms with AFTT's AlgorithmClassifier, applies AFTT's
full optimization pipeline, and produces Version A (naive HIP),
Version B (template-swapped), and Version C (ASM-optimized) .co files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

AFTT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AFTT_ROOT))

TRTLLM_KERNELS = Path("/home/root123/TensorRT-LLM/cpp/tensorrt_llm/kernels")
HIPIFY = "/opt/rocm/bin/hipify-perl"
HIPCC = "/opt/rocm/bin/hipcc"

from src.knowledge_base import KnowledgeBase
from src.algorithm_classifier import AlgorithmClassifier
from src.pipeline import OptimizationPipeline


@dataclass
class HipifyResult:
    """Result for a single .cu file's HIPification and optimization."""
    cu_path: str
    cu_name: str
    relative_dir: str
    category_guess: str = ""

    # HIPify stage
    hipify_ok: bool = False
    hip_path: str = ""
    hipify_error: str = ""

    # Classification
    algo_type: str = ""
    algo_confidence: float = 0.0
    algo_sub_type: str = ""

    # Compilation (Version A: naive HIP)
    compile_a_ok: bool = False
    co_a_path: str = ""
    compile_a_error: str = ""

    # AFTT Pipeline (Version B: template swap + compile)
    has_template: bool = False
    compile_b_ok: bool = False
    co_b_path: str = ""
    compile_b_error: str = ""

    # ASM Optimization (Version C: B + ASM, or A + ASM if no template)
    compile_c_ok: bool = False
    co_c_path: str = ""
    num_edits: int = 0
    cycle_improvement_pct: float = 0.0

    # Matching aiter production .co
    aiter_co_path: str = ""
    aiter_category: str = ""

    error: str = ""


# TRT-LLM directory → aiter category mapping
TRTLLM_TO_AITER = {
    "contextFusedMultiHeadAttention": "fmha_v3_fwd",
    "decoderMaskedMultiheadAttention": "pa",
    "flashMLA": "mla",
    "cutlass_kernels": "bf16gemm",
    "weightOnlyBatchedGemv": "i8gemm",
    "customMoeRoutingKernels": "fmoe",
    "moeLoadBalance": "fmoe",
    "rmsnormKernels": "norm",
    "layernormKernels": "norm",
    "fusedLayernormKernels": "norm",
    "unfusedAttentionKernels": "fmha_v3_fwd",
    "selectiveScan": "other",
    "samplingTopKKernels": "topksoftmax",
    "samplingTopPKernels": "topksoftmax",
}


def guess_aiter_category(cu_path: str) -> str:
    """Guess the aiter category from the TRT-LLM file path."""
    rel = Path(cu_path).relative_to(TRTLLM_KERNELS) if TRTLLM_KERNELS in Path(cu_path).parents else Path(cu_path)
    parts = str(rel).split(os.sep)
    for part in parts:
        if part in TRTLLM_TO_AITER:
            return TRTLLM_TO_AITER[part]
    name_lower = Path(cu_path).stem.lower()
    if "gemm" in name_lower or "matmul" in name_lower:
        return "bf16gemm"
    if "attention" in name_lower or "mha" in name_lower:
        return "fmha_v3_fwd"
    if "moe" in name_lower:
        return "fmoe"
    if "norm" in name_lower:
        return "norm"
    if "topk" in name_lower:
        return "topksoftmax"
    if "mla" in name_lower:
        return "mla"
    return "other"


def find_aiter_production_co(category: str, arch: str = "gfx942") -> Optional[str]:
    """Find the first aiter production .co for comparison."""
    base = Path("/home/root123/aiter/hsa") / arch / category
    if not base.exists():
        return None
    cos = sorted(base.rglob("*.co"))
    return str(cos[0]) if cos else None


def hipify_file(cu_path: str, output_dir: Path) -> tuple[bool, str, str]:
    """Convert a CUDA .cu file to HIP using hipify-perl."""
    cu_name = Path(cu_path).stem
    hip_path = output_dir / f"{cu_name}.hip"

    try:
        result = subprocess.run(
            [HIPIFY, cu_path],
            capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, "", f"hipify-perl failed: {result.stderr[:200]}"

        hip_source = result.stdout
        if not hip_source.strip():
            with open(cu_path) as f:
                cuda_source = f.read()
            hip_source = cuda_to_hip_basic(cuda_source)

        with open(hip_path, "w") as f:
            f.write(hip_source)

        return True, str(hip_path), ""
    except Exception as exc:
        return False, "", str(exc)


def cuda_to_hip_basic(source: str) -> str:
    """Basic CUDA→HIP text replacement fallback."""
    replacements = [
        ("cuda", "hip"), ("CUDA", "HIP"),
        ("cudaMalloc", "hipMalloc"), ("cudaFree", "hipFree"),
        ("cudaMemcpy", "hipMemcpy"), ("cudaDeviceSynchronize", "hipDeviceSynchronize"),
        ("cudaStream_t", "hipStream_t"), ("cudaError_t", "hipError_t"),
        ("cudaSuccess", "hipSuccess"),
        ("__syncthreads", "__syncthreads"),  # same in HIP
        ("#include <cuda_runtime.h>", "#include <hip/hip_runtime.h>"),
        ("#include <cuda_fp16.h>", "#include <hip/hip_fp16.h>"),
        ("cub::", "hipcub::"),
        ("thrust::", "thrust::"),
    ]
    for old, new in replacements:
        source = source.replace(old, new)
    return source


def compile_hip_to_co(hip_path: str, output_co: str,
                      arch: str = "gfx942") -> tuple[bool, str]:
    """Compile a .hip file to .co using hipcc.

    TRT-LLM kernels have deep CUDA dependencies, so compilation may fail
    for many files. We try multiple strategies with increasingly relaxed
    settings.
    """
    base_flags = [
        HIPCC,
        "--genco",
        f"--offload-arch={arch}",
        "-O3",
        "-std=c++17",
        "-I/home/root123/TensorRT-LLM/cpp/include",
        "-I/home/root123/TensorRT-LLM/cpp/tensorrt_llm/kernels",
        "-I/opt/rocm/include",
        "-I/opt/rocm/include/hip",
        "-DUSE_ROCM",
        "-D__HIP_PLATFORM_AMD__",
        "-w",
    ]

    # Strategy 1: Normal compile
    cmd = base_flags + ["-o", output_co, hip_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return True, ""
    except Exception:
        pass

    # Strategy 2: Try with -fsyntax-only disabled and permissive
    cmd2 = base_flags + ["-Wno-error", "-ferror-limit=0", "-o", output_co, hip_path]
    try:
        result = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return True, ""
        return False, result.stderr[:500]
    except Exception as exc:
        return False, str(exc)


def process_single_file(cu_path: str, output_dir: Path,
                        classifier: AlgorithmClassifier,
                        pipeline: OptimizationPipeline,
                        arch: str = "gfx942") -> HipifyResult:
    """Process a single TRT-LLM .cu file through the full pipeline."""
    cu_name = Path(cu_path).stem
    rel_dir = ""
    try:
        rel_dir = str(Path(cu_path).parent.relative_to(TRTLLM_KERNELS))
    except ValueError:
        rel_dir = Path(cu_path).parent.name

    hr = HipifyResult(
        cu_path=cu_path,
        cu_name=cu_name,
        relative_dir=rel_dir,
        category_guess=guess_aiter_category(cu_path),
    )

    file_output = output_dir / cu_name
    file_output.mkdir(parents=True, exist_ok=True)

    # Step 1: HIPify
    ok, hip_path, err = hipify_file(cu_path, file_output)
    hr.hipify_ok = ok
    hr.hip_path = hip_path
    hr.hipify_error = err
    if not ok:
        return hr

    # Step 2: Classify
    try:
        with open(hip_path) as f:
            hip_source = f.read()
        algo_info = classifier.classify_from_hip(hip_source)
        hr.algo_type = algo_info.algo_type
        hr.algo_confidence = algo_info.confidence
        hr.algo_sub_type = algo_info.sub_type
    except Exception as exc:
        hr.algo_type = "unknown"
        hr.algo_confidence = 0.0

    # Step 3: Compile Version A (naive HIPified)
    co_a = str(file_output / f"{cu_name}_version_a.co")
    ok, err = compile_hip_to_co(hip_path, co_a, arch)
    hr.compile_a_ok = ok
    hr.co_a_path = co_a if ok else ""
    hr.compile_a_error = err

    # Step 4: AFTT Pipeline (Version B + C)
    base_co = co_a if ok else ""

    if ok:
        try:
            pipe_result = pipeline.run(
                hip_source,
                source_path=hip_path,
                enable_cpp_swap=True,
                enable_asm_replace=True,
                enable_asm_optimize=True,
                max_replacement_level=2,
                aggressive=False)

            hr.has_template = pipe_result.cpp_swap is not None
            hr.num_edits = len(pipe_result.applied_edits)

            if pipe_result.cycle_comparison:
                hr.cycle_improvement_pct = pipe_result.cycle_comparison.get("improvement_pct", 0)

            # If template swap succeeded, Version B is the template-compiled .co
            if pipe_result.cpp_swap:
                co_b = str(file_output / f"{cu_name}_version_b.co")
                # The pipeline internally compiles the swapped template
                # We need the .co file: re-compile the swapped source
                swap_hip = str(file_output / f"{cu_name}_swapped.hip")
                with open(swap_hip, "w") as f:
                    f.write(pipe_result.cpp_swap.source_code)
                ok_b, err_b = compile_hip_to_co(swap_hip, co_b, arch)
                hr.compile_b_ok = ok_b
                hr.co_b_path = co_b if ok_b else ""
                hr.compile_b_error = err_b
                base_co = co_b if ok_b else co_a

        except Exception as exc:
            hr.error = f"pipeline error: {str(exc)[:200]}"

    # Step 5: ASM optimization to produce Version C
    if base_co and os.path.exists(base_co):
        co_c = str(file_output / f"{cu_name}_version_c.co")
        try:
            co_result = pipeline.run_co_to_co(
                base_co, co_c,
                enable_asm_replace=True,
                enable_asm_optimize=True,
                max_replacement_level=2,
                skip_waitcnt=True)

            hr.compile_c_ok = any(s.success for s in co_result.stages if s.stage_name == "Binary Patch")
            hr.co_c_path = co_c if hr.compile_c_ok else ""
            if co_result.cycle_comparison:
                hr.cycle_improvement_pct = co_result.cycle_comparison.get("improvement_pct", 0)
        except Exception as exc:
            hr.compile_c_ok = False

    # Step 6: Find matching aiter .co
    aiter_co = find_aiter_production_co(hr.category_guess, arch)
    if aiter_co:
        hr.aiter_co_path = aiter_co
        hr.aiter_category = hr.category_guess

    return hr


def run_batch_hipify(output_dir: Path, arch: str = "gfx942",
                     max_files: Optional[int] = None,
                     categories: Optional[list[str]] = None) -> list[HipifyResult]:
    """Run batch HIPification of all TRT-LLM kernel files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scan all .cu files
    cu_files = sorted(TRTLLM_KERNELS.rglob("*.cu"))
    print(f"Found {len(cu_files)} .cu files in TRT-LLM kernels")

    if categories:
        cu_files = [f for f in cu_files if guess_aiter_category(str(f)) in categories]
        print(f"Filtered to {len(cu_files)} files (categories: {categories})")

    if max_files:
        cu_files = cu_files[:max_files]
        print(f"Limited to {max_files} files")

    # Initialize AFTT components
    kb = KnowledgeBase()
    kb.load()
    classifier = AlgorithmClassifier()
    pipeline = OptimizationPipeline(arch=arch)

    results = []
    t0 = time.time()

    for i, cu_file in enumerate(cu_files):
        elapsed = time.time() - t0
        if i == 0 or (i + 1) % 10 == 0:
            rate = (i + 1) / max(elapsed, 0.1)
            eta = (len(cu_files) - i - 1) / max(rate, 0.01)
            print(f"  [{i+1}/{len(cu_files)}] {cu_file.name} "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

        hr = process_single_file(str(cu_file), output_dir, classifier, pipeline, arch)
        results.append(hr)

        # Save per-file result
        result_file = output_dir / f"{cu_file.stem}_result.json"
        with open(result_file, "w") as f:
            json.dump(asdict(hr), f, indent=2, default=str)

    elapsed_total = time.time() - t0
    print(f"\nBatch HIPify complete: {len(cu_files)} files in {elapsed_total:.1f}s")

    return results


def write_hipify_summary(results: list[HipifyResult], output_dir: Path):
    """Write summary of HIPification results."""
    summary = {
        "total_files": len(results),
        "hipify_ok": sum(1 for r in results if r.hipify_ok),
        "classified": sum(1 for r in results if r.algo_confidence > 0.3),
        "compile_ok": sum(1 for r in results if r.compile_a_ok),
        "template_match": sum(1 for r in results if r.has_template),
        "version_b_ok": sum(1 for r in results if r.compile_b_ok),
        "version_c_ok": sum(1 for r in results if r.compile_c_ok),
        "has_aiter_equiv": sum(1 for r in results if r.aiter_co_path),
        "categories": {},
    }

    for r in results:
        cat = r.category_guess
        if cat not in summary["categories"]:
            summary["categories"][cat] = {
                "total": 0, "hipify_ok": 0, "compile_ok": 0,
                "template": 0, "version_c": 0
            }
        summary["categories"][cat]["total"] += 1
        if r.hipify_ok:
            summary["categories"][cat]["hipify_ok"] += 1
        if r.compile_a_ok:
            summary["categories"][cat]["compile_ok"] += 1
        if r.has_template:
            summary["categories"][cat]["template"] += 1
        if r.compile_c_ok:
            summary["categories"][cat]["version_c"] += 1

    summary_path = output_dir / "hipify_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nHIPify Summary:")
    print(f"  Total files: {summary['total_files']}")
    print(f"  HIPified OK: {summary['hipify_ok']}")
    print(f"  Classified:  {summary['classified']}")
    print(f"  Compiled OK: {summary['compile_ok']}")
    print(f"  Template:    {summary['template_match']}")
    print(f"  Version C:   {summary['version_c_ok']}")
    print(f"  Has aiter:   {summary['has_aiter_equiv']}")

    print(f"\n  Per category:")
    for cat, cdata in sorted(summary["categories"].items()):
        print(f"    {cat}: {cdata['total']} total, "
              f"{cdata['hipify_ok']} hipified, "
              f"{cdata['compile_ok']} compiled, "
              f"{cdata['version_c']} ASM-optimized")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2A: Batch HIPify TRT-LLM kernels")
    parser.add_argument("--output",
                        default=str(AFTT_ROOT / "results" / "phase2" / "hipify"),
                        help="Output directory")
    parser.add_argument("--arch", default="gfx942")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--categories", nargs="*")
    args = parser.parse_args()

    results = run_batch_hipify(
        Path(args.output), args.arch,
        max_files=args.max_files,
        categories=args.categories)

    write_hipify_summary(results, Path(args.output))


if __name__ == "__main__":
    main()
