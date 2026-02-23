#!/usr/bin/env python3
"""TensorRT-LLM Classic Kernel Algorithm Analyzer.

Catalogs all classic kernel algorithms from TensorRT-LLM by scanning
the kernel source tree and extracting key algorithm characteristics.
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

TRTLLM_ROOT = Path(os.environ.get("TRTLLM_ROOT", "/home/root123/TensorRT-LLM"))
KERNELS_DIR = TRTLLM_ROOT / "cpp" / "tensorrt_llm" / "kernels"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "db" / "patterns"


def count_files(directory: Path, extensions: set) -> int:
    """Count files with given extensions in directory tree."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*") if f.suffix in extensions)


def scan_file_for_patterns(filepath: Path) -> dict:
    """Scan a CUDA/C++ file for key patterns."""
    text = filepath.read_text(errors="ignore")
    return {
        "has_shared_memory": "__shared__" in text or "extern __shared__" in text,
        "has_warp_primitives": "__shfl" in text or "__ballot" in text or "__syncwarp" in text,
        "has_tensor_core": "wmma" in text.lower() or "mma" in text.lower(),
        "has_fp8": "fp8" in text.lower() or "__nv_fp8" in text,
        "has_quantization": "quant" in text.lower() or "dequant" in text.lower(),
        "has_softmax": "softmax" in text.lower(),
        "has_flash_attention": "flash" in text.lower() and "attention" in text.lower(),
        "has_online_softmax": "online" in text.lower() and "softmax" in text.lower(),
        "has_paged_kv": "paged" in text.lower() or "page_table" in text.lower(),
        "has_causal_mask": "causal" in text.lower(),
        "has_cutlass": "cutlass" in text.lower(),
        "has_cublas": "cublas" in text.lower(),
        "line_count": text.count("\n"),
    }


def analyze_kernel_directory(dirpath: Path) -> dict:
    """Analyze a kernel directory for its contents."""
    cu_files = list(dirpath.rglob("*.cu"))
    cuh_files = list(dirpath.rglob("*.cuh"))
    h_files = list(dirpath.rglob("*.h"))
    cpp_files = list(dirpath.rglob("*.cpp"))

    all_patterns = defaultdict(int)
    total_lines = 0

    for f in cu_files + cuh_files + h_files + cpp_files:
        patterns = scan_file_for_patterns(f)
        for k, v in patterns.items():
            if k == "line_count":
                total_lines += v
            elif v:
                all_patterns[k] += 1

    return {
        "cu_files": len(cu_files),
        "cuh_files": len(cuh_files),
        "h_files": len(h_files),
        "cpp_files": len(cpp_files),
        "total_lines": total_lines,
        "patterns": dict(all_patterns),
    }


def build_algorithm_catalog() -> list[dict]:
    """Build a structured catalog of all classic TRT-LLM kernel algorithms."""
    catalog = []

    # ATTENTION KERNELS
    catalog.extend([
        {
            "category": "Attention",
            "algorithm": "Context Fused Multi-Head Attention (FMHA)",
            "trtllm_path": "contextFusedMultiHeadAttention/",
            "description": "Flash Attention implementation for prefill/context phase. Tiles QKV in shared memory, computes attention with online softmax to avoid materializing full attention matrix.",
            "key_techniques": [
                "Tiled QKV computation in shared memory",
                "Online softmax (numerically stable, single-pass)",
                "Causal masking support",
                "Multi-head / Multi-query / Grouped-query attention",
                "FP16/BF16/FP8 support",
                "Paged KV-cache support",
            ],
            "cuda_features": ["shared_memory", "tensor_cores", "warp_primitives"],
            "complexity": "O(N*d) per head with tiling",
            "variants": ["MHA", "MQA", "GQA", "with_ALiBi", "with_RoPE"],
        },
        {
            "category": "Attention",
            "algorithm": "Decoder Masked Multi-Head Attention (DMHA)",
            "trtllm_path": "decoderMaskedMultiheadAttention/",
            "description": "Optimized attention for autoregressive decode phase. Single query token attends to all cached KV pairs.",
            "key_techniques": [
                "Single-query optimization",
                "Paged KV-cache traversal",
                "Cross-attention support",
                "Beam search integration",
            ],
            "cuda_features": ["shared_memory", "warp_primitives"],
            "complexity": "O(N*d) per query token",
            "variants": ["MHA", "MQA", "GQA", "cross_attention"],
        },
        {
            "category": "Attention",
            "algorithm": "XQA (Cross-Query Attention) Dispatcher",
            "trtllm_path": "xqaDispatcher.cpp",
            "description": "Dispatcher for optimized attention variants that fuses Q projection with attention computation.",
            "key_techniques": [
                "Fused Q projection + attention",
                "Dynamic dispatch based on sequence length",
                "Multi-GPU sharding support",
            ],
            "cuda_features": ["tensor_cores"],
            "complexity": "Variable",
            "variants": ["multi_gpu", "single_gpu"],
        },
        {
            "category": "Attention",
            "algorithm": "Flash MLA (Multi-Level Attention)",
            "trtllm_path": "flashMLA/",
            "description": "Multi-level attention implementation for models like DeepSeek that use compressed KV representations with latent attention.",
            "key_techniques": [
                "Compressed KV-cache",
                "Latent attention decomposition",
                "Flash attention style tiling",
            ],
            "cuda_features": ["shared_memory", "tensor_cores"],
            "complexity": "O(N*d_compressed)",
            "variants": ["chunked_prefill", "decode"],
        },
        {
            "category": "Attention",
            "algorithm": "Unfused Attention Kernels",
            "trtllm_path": "unfusedAttentionKernels/",
            "description": "Separate QK matmul, softmax, and PV matmul kernels for cases where fusion is not beneficial.",
            "key_techniques": [
                "Separate Q*K^T computation",
                "Standalone softmax",
                "Separate softmax(QK^T)*V computation",
                "Useful for debugging and small batch sizes",
            ],
            "cuda_features": ["shared_memory"],
            "complexity": "O(N^2*d)",
            "variants": ["standard", "with_mask"],
        },
        {
            "category": "Attention",
            "algorithm": "Sage Attention",
            "trtllm_path": "sageAttentionKernels.cu",
            "description": "Approximate attention using locality-sensitive hashing or sparse patterns for ultra-long sequences.",
            "key_techniques": [
                "Sparse attention patterns",
                "Approximate attention for long sequences",
            ],
            "cuda_features": ["shared_memory", "warp_primitives"],
            "complexity": "O(N*sqrt(N)*d) approximate",
            "variants": ["standard"],
        },
        {
            "category": "Attention",
            "algorithm": "Sparse Attention",
            "trtllm_path": "sparseAttentionKernels.cu",
            "description": "Structured sparse attention patterns (local + global) for efficient long-sequence processing.",
            "key_techniques": [
                "Local windowed attention",
                "Global attention tokens",
                "Structured sparsity patterns",
            ],
            "cuda_features": ["shared_memory"],
            "complexity": "O(N*w*d) where w is window size",
            "variants": ["local_global", "strided"],
        },
    ])

    # GEMM KERNELS
    catalog.extend([
        {
            "category": "GEMM",
            "algorithm": "CUTLASS GEMM Kernels",
            "trtllm_path": "cutlass_kernels/",
            "description": "High-performance GEMM implementations based on NVIDIA CUTLASS library. Covers standard GEMM, grouped GEMM, split-K, and stream-K variants.",
            "key_techniques": [
                "CUTLASS tile-based GEMM",
                "Split-K parallelism for small M",
                "Stream-K for load balancing",
                "Grouped GEMM for batched operations",
                "FP16/BF16/INT8/FP8 support",
                "Epilogue fusion (bias, activation, quantization)",
            ],
            "cuda_features": ["tensor_cores", "shared_memory", "cutlass"],
            "complexity": "O(M*N*K)",
            "variants": ["standard", "split_k", "stream_k", "grouped", "batched"],
        },
        {
            "category": "GEMM",
            "algorithm": "Weight-Only Batched GEMV",
            "trtllm_path": "weightOnlyBatchedGemv/",
            "description": "Optimized GEMV for weight-only quantized models. Dequantizes on-the-fly during matrix-vector multiplication.",
            "key_techniques": [
                "On-the-fly weight dequantization",
                "W4A16, W8A16 support",
                "Vectorized memory access",
                "Per-channel and per-group scales",
            ],
            "cuda_features": ["shared_memory", "warp_primitives"],
            "complexity": "O(M*K)",
            "variants": ["W4A16", "W8A16", "per_channel", "per_group"],
        },
        {
            "category": "GEMM",
            "algorithm": "QServe GEMM",
            "trtllm_path": "qserveGemm*.cu",
            "description": "Quantized serving GEMM with per-channel and per-group quantization support.",
            "key_techniques": [
                "Per-channel quantization",
                "Per-group quantization",
                "Fused dequantization",
            ],
            "cuda_features": ["tensor_cores"],
            "complexity": "O(M*N*K)",
            "variants": ["per_channel", "per_group"],
        },
        {
            "category": "GEMM",
            "algorithm": "Group GEMM",
            "trtllm_path": "groupGemm.cu",
            "description": "Grouped GEMM for executing multiple small GEMMs concurrently, commonly used in MoE.",
            "key_techniques": [
                "Multiple concurrent GEMMs",
                "Variable batch sizes per group",
                "CUDA graph compatible",
            ],
            "cuda_features": ["tensor_cores", "cutlass"],
            "complexity": "O(sum(M_i*N_i*K_i))",
            "variants": ["standard", "cuda_graph"],
        },
        {
            "category": "GEMM",
            "algorithm": "Split-K Group GEMM",
            "trtllm_path": "splitkGroupGemm.cu",
            "description": "Split-K parallelized group GEMM for better utilization with small problem sizes.",
            "key_techniques": [
                "Split-K across multiple CTAs",
                "Atomic reduction",
                "Better GPU utilization for small M",
            ],
            "cuda_features": ["tensor_cores", "atomics"],
            "complexity": "O(M*N*K/split_k)",
            "variants": ["standard"],
        },
        {
            "category": "GEMM",
            "algorithm": "TinyGEMM",
            "trtllm_path": "tinygemm2/",
            "description": "Highly optimized small-matrix GEMM kernels for decode-phase where batch sizes are very small.",
            "key_techniques": [
                "Optimized for very small M (1-8)",
                "Register-level tiling",
                "Minimal shared memory usage",
            ],
            "cuda_features": ["tensor_cores", "warp_primitives"],
            "complexity": "O(M*N*K) with small M",
            "variants": ["fp16", "bf16", "fp8"],
        },
    ])

    # MoE KERNELS
    catalog.extend([
        {
            "category": "MoE",
            "algorithm": "MoE Routing Kernels",
            "trtllm_path": "customMoeRoutingKernels.cu",
            "description": "Expert routing for Mixture of Experts: TopK gating, expert assignment, and token dispatch.",
            "key_techniques": [
                "TopK expert selection",
                "Load-balanced routing",
                "Token-to-expert assignment",
                "Capacity factor handling",
            ],
            "cuda_features": ["shared_memory", "warp_primitives"],
            "complexity": "O(N*E) where E is num experts",
            "variants": ["topk", "expert_choice"],
        },
        {
            "category": "MoE",
            "algorithm": "MoE Alignment Kernels",
            "trtllm_path": "moeAlignKernels.cu",
            "description": "Token alignment and permutation for efficient expert GEMM execution.",
            "key_techniques": [
                "Token permutation for grouped GEMM",
                "Expert-parallel layout transformation",
                "Padding and alignment",
            ],
            "cuda_features": ["shared_memory"],
            "complexity": "O(N*K)",
            "variants": ["standard"],
        },
        {
            "category": "MoE",
            "algorithm": "MoE Prepare Kernels",
            "trtllm_path": "moePrepareKernels.cu",
            "description": "Prepares token layout for MoE GEMM execution including sorting and grouping.",
            "key_techniques": [
                "Token sorting by expert ID",
                "Group size computation",
                "Expert metadata preparation",
            ],
            "cuda_features": ["shared_memory"],
            "complexity": "O(N*log(N))",
            "variants": ["standard"],
        },
        {
            "category": "MoE",
            "algorithm": "Fused MoE Communication",
            "trtllm_path": "fusedMoeCommKernels.cu",
            "description": "Fused MoE computation with inter-GPU communication for expert parallelism.",
            "key_techniques": [
                "All-to-all communication",
                "Expert-parallel distribution",
                "Fused compute + communication overlap",
            ],
            "cuda_features": ["nccl", "shared_memory"],
            "complexity": "O(N*K + comm)",
            "variants": ["all_to_all"],
        },
        {
            "category": "MoE",
            "algorithm": "MoE Load Balance",
            "trtllm_path": "moeLoadBalance/",
            "description": "Load balancing mechanisms for MoE including auxiliary loss computation and expert utilization tracking.",
            "key_techniques": [
                "Auxiliary load balancing loss",
                "Expert utilization statistics",
                "Dynamic capacity adjustment",
            ],
            "cuda_features": ["atomics"],
            "complexity": "O(N*E)",
            "variants": ["standard"],
        },
    ])

    # NORMALIZATION KERNELS
    catalog.extend([
        {
            "category": "Normalization",
            "algorithm": "LayerNorm",
            "trtllm_path": "layernormKernels.cu",
            "description": "Layer normalization with optional bias and residual connection.",
            "key_techniques": [
                "Two-pass (mean, variance) or Welford's online algorithm",
                "Vectorized memory access",
                "Fused residual add",
                "Per-channel scaling",
            ],
            "cuda_features": ["shared_memory", "warp_primitives"],
            "complexity": "O(N*d)",
            "variants": ["standard", "fused_residual", "with_bias"],
        },
        {
            "category": "Normalization",
            "algorithm": "RMSNorm",
            "trtllm_path": "rmsnormKernels.cu",
            "description": "Root Mean Square normalization, commonly used in LLaMA-family models.",
            "key_techniques": [
                "Single-pass RMS computation",
                "No mean subtraction (unlike LayerNorm)",
                "Vectorized loads (float4)",
                "Fused quantization output",
            ],
            "cuda_features": ["shared_memory", "warp_primitives"],
            "complexity": "O(N*d)",
            "variants": ["standard", "fused_quant", "group_rmsnorm"],
        },
        {
            "category": "Normalization",
            "algorithm": "Group RMSNorm",
            "trtllm_path": "groupRmsNormKernels/",
            "description": "Group-wise RMS normalization for grouped architectures.",
            "key_techniques": [
                "Per-group normalization",
                "Shared memory reduction within groups",
            ],
            "cuda_features": ["shared_memory", "warp_primitives"],
            "complexity": "O(N*d)",
            "variants": ["standard"],
        },
        {
            "category": "Normalization",
            "algorithm": "Fused LayerNorm Kernels",
            "trtllm_path": "fusedLayernormKernels/",
            "description": "Fused LayerNorm with other operations like quantization, residual add, and activation.",
            "key_techniques": [
                "Fused norm + quantization",
                "Fused norm + residual",
                "Fused norm + activation",
            ],
            "cuda_features": ["shared_memory", "warp_primitives"],
            "complexity": "O(N*d)",
            "variants": ["fused_quant", "fused_residual"],
        },
    ])

    # QUANTIZATION KERNELS
    catalog.extend([
        {
            "category": "Quantization",
            "algorithm": "Dynamic Quantization",
            "trtllm_path": "quantization.cu",
            "description": "Dynamic per-tensor and per-token quantization for INT8/FP8 inference.",
            "key_techniques": [
                "Per-tensor scale computation",
                "Per-token quantization",
                "Symmetric quantization",
                "FP8 E4M3 format support",
            ],
            "cuda_features": ["shared_memory", "warp_primitives"],
            "complexity": "O(N*d)",
            "variants": ["per_tensor", "per_token", "fp8", "int8"],
        },
        {
            "category": "Quantization",
            "algorithm": "Fused Activation + Quantization",
            "trtllm_path": "fusedActivationQuant.cu",
            "description": "Fused activation function (SiLU, GELU) with quantization in a single kernel pass.",
            "key_techniques": [
                "Single-pass activation + quantization",
                "SwiGLU / GeGLU support",
                "Per-token output scale",
            ],
            "cuda_features": ["shared_memory"],
            "complexity": "O(N*d)",
            "variants": ["silu_quant", "gelu_quant"],
        },
        {
            "category": "Quantization",
            "algorithm": "Pre-Quantization Scaling",
            "trtllm_path": "preQuantScaleKernel.cu",
            "description": "SmoothQuant-style per-channel scaling applied before quantization to reduce outlier impact.",
            "key_techniques": [
                "Per-channel scale factors",
                "Outlier migration from activations to weights",
                "SmoothQuant algorithm",
            ],
            "cuda_features": ["shared_memory"],
            "complexity": "O(N*d)",
            "variants": ["smoothquant"],
        },
    ])

    # KV-CACHE KERNELS
    catalog.extend([
        {
            "category": "KV-Cache",
            "algorithm": "KV-Cache Utils",
            "trtllm_path": "kvCacheUtils.h",
            "description": "Paged KV-cache management with block-level page tables for efficient memory usage.",
            "key_techniques": [
                "Block-level page table",
                "Dynamic page allocation",
                "FP8 KV-cache quantization",
                "Copy-on-write support",
            ],
            "cuda_features": ["shared_memory"],
            "complexity": "O(num_pages)",
            "variants": ["paged", "continuous", "fp8_quantized"],
        },
        {
            "category": "KV-Cache",
            "algorithm": "KV-Cache Partial Copy",
            "trtllm_path": "kvCachePartialCopy.cu",
            "description": "Partial copy kernels for updating KV-cache during decode with speculative decoding.",
            "key_techniques": [
                "Partial page copy",
                "Scatter-based update",
                "Speculative decoding support",
            ],
            "cuda_features": ["shared_memory"],
            "complexity": "O(new_tokens*d)",
            "variants": ["standard", "speculative"],
        },
    ])

    # SAMPLING KERNELS
    catalog.extend([
        {
            "category": "Sampling",
            "algorithm": "Top-K Sampling",
            "trtllm_path": "samplingTopKKernels.cu",
            "description": "Efficient top-K sampling from logit distributions using partial sorting.",
            "key_techniques": [
                "Radix-based top-K selection",
                "Per-request K values",
                "Temperature scaling",
            ],
            "cuda_features": ["shared_memory", "warp_primitives"],
            "complexity": "O(V) where V is vocab size",
            "variants": ["standard", "batched"],
        },
        {
            "category": "Sampling",
            "algorithm": "Top-P (Nucleus) Sampling",
            "trtllm_path": "samplingTopPKernels.cu",
            "description": "Nucleus sampling with cumulative probability threshold.",
            "key_techniques": [
                "Sort-based cumulative probability",
                "Dynamic vocabulary truncation",
                "Air Top-P variant for efficiency",
            ],
            "cuda_features": ["shared_memory", "warp_primitives"],
            "complexity": "O(V*log(V))",
            "variants": ["standard", "air_top_p"],
        },
    ])

    # COMMUNICATION KERNELS
    catalog.extend([
        {
            "category": "Communication",
            "algorithm": "Custom AllReduce",
            "trtllm_path": "customAllReduceKernels.cu",
            "description": "Custom all-reduce implementation using NVLink for lower latency than NCCL for small messages.",
            "key_techniques": [
                "NVLink peer-to-peer access",
                "Two-phase reduce-scatter + all-gather",
                "Fused with computation",
            ],
            "cuda_features": ["nvlink", "shared_memory"],
            "complexity": "O(N*d/P + latency)",
            "variants": ["one_shot", "two_shot"],
        },
    ])

    # SPECIALIZED KERNELS
    catalog.extend([
        {
            "category": "Specialized",
            "algorithm": "Selective Scan (Mamba)",
            "trtllm_path": "selectiveScan/",
            "description": "Efficient selective scan for Mamba/SSM models. Hardware-aware parallel scan.",
            "key_techniques": [
                "Parallel prefix scan",
                "Selective gating",
                "Hardware-aware chunked computation",
            ],
            "cuda_features": ["shared_memory", "warp_primitives"],
            "complexity": "O(N*d*state_dim)",
            "variants": ["standard", "chunked"],
        },
        {
            "category": "Specialized",
            "algorithm": "Causal Conv1D",
            "trtllm_path": "causalConv1d/",
            "description": "1D causal convolution for Mamba and similar architectures.",
            "key_techniques": [
                "Causal padding",
                "Depthwise separable",
                "Fused with activation",
            ],
            "cuda_features": ["shared_memory"],
            "complexity": "O(N*d*k) where k is kernel size",
            "variants": ["standard", "fused_activation"],
        },
        {
            "category": "Specialized",
            "algorithm": "Beam Search",
            "trtllm_path": "beamSearchKernels/",
            "description": "Beam search decoding with KV-cache management.",
            "key_techniques": [
                "Top-K beam selection",
                "KV-cache reordering",
                "Length penalty",
            ],
            "cuda_features": ["shared_memory", "warp_primitives"],
            "complexity": "O(B*K*V) where B is beam width",
            "variants": ["standard", "diverse"],
        },
        {
            "category": "Specialized",
            "algorithm": "Speculative Decoding",
            "trtllm_path": "speculativeDecoding/",
            "description": "Speculative decoding verification and acceptance kernels.",
            "key_techniques": [
                "Draft-verify pattern",
                "Token acceptance/rejection",
                "Tree-based speculation",
            ],
            "cuda_features": ["shared_memory"],
            "complexity": "O(N*draft_length)",
            "variants": ["standard", "medusa", "eagle"],
        },
        {
            "category": "Specialized",
            "algorithm": "Fused QK-Norm + RoPE",
            "trtllm_path": "fusedQKNormRopeKernel.cu",
            "description": "Fused query/key normalization with rotary position embedding in a single kernel.",
            "key_techniques": [
                "Fused norm + RoPE",
                "Reduced memory bandwidth",
                "Per-head processing",
            ],
            "cuda_features": ["shared_memory"],
            "complexity": "O(N*d)",
            "variants": ["standard"],
        },
        {
            "category": "Specialized",
            "algorithm": "LoRA Kernels",
            "trtllm_path": "lora/",
            "description": "Low-Rank Adaptation kernels for efficient fine-tuned model serving.",
            "key_techniques": [
                "Low-rank decomposition (A*B)",
                "Multi-LoRA batching",
                "Fused with base GEMM",
            ],
            "cuda_features": ["tensor_cores"],
            "complexity": "O(N*d*r) where r is rank",
            "variants": ["standard", "batched_multi_lora"],
        },
        {
            "category": "Specialized",
            "algorithm": "DoRA Scaling",
            "trtllm_path": "doraScaling.cu",
            "description": "Weight-Decomposed Low-Rank Adaptation scaling kernel.",
            "key_techniques": [
                "Direction + magnitude decomposition",
                "Per-column scaling",
            ],
            "cuda_features": ["shared_memory"],
            "complexity": "O(N*d)",
            "variants": ["standard"],
        },
    ])

    return catalog


def scan_actual_kernels() -> dict:
    """Scan the actual TRT-LLM kernel tree for statistics."""
    stats = {
        "total_cu_files": 0,
        "total_h_files": 0,
        "total_lines": 0,
        "directories": {},
    }

    if not KERNELS_DIR.exists():
        return stats

    for item in sorted(KERNELS_DIR.iterdir()):
        if item.is_dir():
            analysis = analyze_kernel_directory(item)
            stats["directories"][item.name] = analysis
            stats["total_cu_files"] += analysis["cu_files"]
            stats["total_h_files"] += analysis["h_files"]
            stats["total_lines"] += analysis["total_lines"]
        elif item.suffix in {".cu", ".cuh"}:
            patterns = scan_file_for_patterns(item)
            stats["total_cu_files"] += 1
            stats["total_lines"] += patterns["line_count"]

    return stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning TensorRT-LLM kernel directory...")
    scan_stats = scan_actual_kernels()

    print("Building algorithm catalog...")
    catalog = build_algorithm_catalog()

    # Categorize
    by_category = defaultdict(list)
    for algo in catalog:
        by_category[algo["category"]].append(algo["algorithm"])

    output = {
        "source": str(TRTLLM_ROOT),
        "kernel_scan_stats": scan_stats,
        "total_algorithms_cataloged": len(catalog),
        "categories": {cat: len(algos) for cat, algos in sorted(by_category.items())},
        "algorithms": catalog,
    }

    out_file = OUTPUT_DIR / "trtllm_algorithms.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TensorRT-LLM Algorithm Catalog")
    print(f"{'='*60}")
    print(f"Total algorithms cataloged: {len(catalog)}")
    print(f"Kernel source stats: {scan_stats['total_cu_files']} .cu files, {scan_stats['total_lines']} total lines")
    print(f"\nBy category:")
    for cat, algos in sorted(by_category.items()):
        print(f"  {cat:20s}: {len(algos)} algorithms")
        for a in algos:
            print(f"    - {a}")
    print(f"\nOutput: {out_file}")


if __name__ == "__main__":
    main()
