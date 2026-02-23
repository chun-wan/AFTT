#!/usr/bin/env python3
"""NVIDIA-to-AMD Kernel Algorithm Mapping Table Builder.

Creates a structured mapping between TensorRT-LLM CUDA kernels and
aiter/CK AMD equivalents, covering all major kernel categories.
"""

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "db" / "patterns"


def build_mapping() -> list[dict]:
    """Build the complete NVIDIA <-> AMD mapping table."""
    mappings = []

    # ============================================================
    # ATTENTION
    # ============================================================
    mappings.extend([
        {
            "category": "Attention",
            "subcategory": "Prefill / Context FMHA",
            "algorithm": "Flash Attention (Prefill)",
            "trtllm_impl": "contextFusedMultiHeadAttention/",
            "trtllm_key_technique": "Tiled QKV in shared memory, online softmax, causal masking, FP16/BF16/FP8 Tensor Core WMMA",
            "aiter_impl": "aiter/ops/triton/fused_moe_gelu.py, csrc/kernels/ + hsa/gfx942/fmha_v3_fwd/*.co",
            "ck_impl": "fmha_fwd_pipeline (ck_tile/ops/fmha/pipeline/), fmha_fwd_appendkv pipeline",
            "co_kernels": "fmha_v3_fwd/*.co (multiple head dimensions: hd32, hd64, hd128, hd192, hd256)",
            "key_differences": [
                "AMD uses AccVGPR (accumulation registers) instead of shared memory for partial softmax",
                "AMD uses MFMA instructions (v_mfma_f32_16x16x16_f16) vs NVIDIA HMMA/WMMA",
                "CK uses __builtin_amdgcn_sched_group_barrier for explicit instruction scheduling",
                "CK pipeline uses ping-pong LDS double buffering for QKV tiles",
            ],
            "optimization_notes": "CK fmha_v3 pipelines use deep prefetch (2-3 stages) with explicit scheduling barriers. Production .co kernels show MFMA chains of 16-32 instructions interleaved with DS reads.",
        },
        {
            "category": "Attention",
            "subcategory": "Decode / Generation DMHA",
            "algorithm": "Masked Multi-Head Attention (Decode)",
            "trtllm_impl": "decoderMaskedMultiheadAttention/",
            "trtllm_key_technique": "Single query KV-cache traversal, warp-level reduction, paged KV support",
            "aiter_impl": "hsa/gfx942/pa/*.co (paged attention ASM kernels), aiter/ops/triton/paged_attn.py",
            "ck_impl": "Not directly in CK; aiter provides custom ASM paged attention",
            "co_kernels": "pa/*.co (pa_fwd variants for different head dims and data types)",
            "key_differences": [
                "aiter uses hand-written ASM for paged attention decode (pa/*.co)",
                "NVIDIA uses Tensor Core for small batch; AMD uses MFMA even for single-token",
                "aiter PA kernels highly optimized with AccVGPR accumulation and explicit scheduling",
            ],
            "optimization_notes": "Paged attention .co kernels show heavy use of buffer_load for KV-cache page traversal and partial waitcnt (vmcnt > 0) for overlapping loads with compute.",
        },
        {
            "category": "Attention",
            "subcategory": "Multi-Level Attention",
            "algorithm": "Flash MLA (DeepSeek-style)",
            "trtllm_impl": "flashMLA/, mlaKernels.cu, mlaChunkedPrefill.cu",
            "trtllm_key_technique": "Compressed KV-cache with latent decomposition, chunked prefill",
            "aiter_impl": "hsa/gfx942/mla/*.co, aiter/ops/triton/mla.py",
            "ck_impl": "CK MLA pipeline extensions",
            "co_kernels": "mla/*.co (mla_a8w8 variants with various GQA ratios)",
            "key_differences": [
                "AMD MLA kernels support INT8 quantized attention (mla_a8w8)",
                "aiter provides both Triton and ASM implementations",
                "gfx950 adds SMFMA support for potentially faster MLA",
            ],
            "optimization_notes": "MLA .co kernels show quantized MFMA (v_mfma_i32_16x16x32_i8) for INT8 attention, reducing memory bandwidth requirements.",
        },
        {
            "category": "Attention",
            "subcategory": "Attention Variants",
            "algorithm": "GQA / MQA Support",
            "trtllm_impl": "contextFusedMultiHeadAttention/ (GQA path), decoderMaskedMultiheadAttention/ (GQA decode)",
            "trtllm_key_technique": "Shared KV heads across query groups, efficient KV broadcast",
            "aiter_impl": "fmha_v3_fwd/*.co (GQA variants), pa/*.co (paged attention with GQA)",
            "ck_impl": "fmha pipeline with GQA ratio parameter",
            "co_kernels": "Kernel names contain gqaratio (e.g., mla_a8w8_qh16_qseqlen4_gqaratio16)",
            "key_differences": [
                "CK handles GQA by adjusting the Q/KV head mapping in the pipeline template",
                "aiter ASM kernels have GQA baked into the kernel binary for each ratio",
            ],
            "optimization_notes": "GQA reduces KV memory bandwidth proportionally to the ratio, making decode attention more compute-bound.",
        },
    ])

    # ============================================================
    # GEMM
    # ============================================================
    mappings.extend([
        {
            "category": "GEMM",
            "subcategory": "Standard GEMM",
            "algorithm": "High-Performance GEMM",
            "trtllm_impl": "cutlass_kernels/ (CUTLASS-based)",
            "trtllm_key_technique": "CUTLASS tile-based GEMM with Tensor Core, epilogue fusion, split-K",
            "aiter_impl": "hsa/gfx942/bf16gemm/*.co, csrc/ck_gemm_*, ck_tile_gemm_*",
            "ck_impl": "gemm_pipeline_ag_bg_cr_comp_v3/v4/v5/v6 (multiple pipeline strategies)",
            "co_kernels": "bf16gemm/*.co (bf16gemm_fp32bf16_tn variants with split-K)",
            "key_differences": [
                "NVIDIA uses CUTLASS with WMMA/HMMA; AMD uses CK with MFMA",
                "CK offers 6 pipeline variants (v3-v6, warp-spec, async) vs CUTLASS 2.x/3.x",
                "CK v4 ping-pong double buffer ≈ CUTLASS multistage pipeline",
                "CK v5 warp specialization ≈ CUTLASS warp-specialized Hopper kernels",
                "AMD uses AccVGPR for accumulation, NVIDIA uses register file",
            ],
            "optimization_notes": "Production bf16gemm .co kernels use split-K with various tile sizes (32x64, 48x64, 64x64, 80x64, 128x64, 160x64). CK v6 pipeline with PrefetchStages=3 provides deepest latency hiding.",
        },
        {
            "category": "GEMM",
            "subcategory": "Quantized GEMM",
            "algorithm": "INT8 GEMM (W8A8)",
            "trtllm_impl": "cutlass_kernels/ (INT8 CUTLASS), qserveGemm*.cu",
            "trtllm_key_technique": "INT8 Tensor Core GEMM with per-channel/per-group dequantization",
            "aiter_impl": "csrc/ck_gemm_a8w8*, hsa/gfx942/fmoe/*int8*.co",
            "ck_impl": "CK INT8 GEMM pipelines with i32 accumulation",
            "co_kernels": "fmoe/*int8*.co (extensive INT8 MoE GEMM variants)",
            "key_differences": [
                "AMD uses v_mfma_i32_16x16x32_i8 for INT8 MFMA",
                "CK INT8 GEMM supports blockscale and per-token quantization",
                "Production shows INT8 MFMA is the second most common instruction after FP8 MFMA",
            ],
            "optimization_notes": "INT8 GEMM kernels show 765,920 total INT8 MFMA instructions across all disassembled kernels, making it the workhorse for quantized inference.",
        },
        {
            "category": "GEMM",
            "subcategory": "FP8 GEMM",
            "algorithm": "FP8 GEMM (E4M3/E5M2)",
            "trtllm_impl": "cutlass_kernels/ (FP8 path), internal_cutlass_kernels/",
            "trtllm_key_technique": "FP8 Tensor Core GEMM with block-scale quantization",
            "aiter_impl": "csrc/ck_gemm_a8w8_blockscale*, hsa/gfx942/fmoe/*fp8*.co",
            "ck_impl": "CK FP8 pipelines using v_mfma_f32_16x16x32_fp8_fp8",
            "co_kernels": "fmoe/*fp8*.co, fmoe_2stages/*fp8*.co",
            "key_differences": [
                "AMD v_mfma_f32_16x16x32_fp8_fp8 is the dominant FP8 instruction (1,002,900 occurrences)",
                "gfx950 adds v_mfma_scale_f32_16x16x128_f8f6f4 for scaled FP8 with larger K dimension",
                "aiter supports block-scale FP8 with explicit scale factor handling",
            ],
            "optimization_notes": "FP8 MFMA is the single most common instruction across all production kernels. gfx950 introduces new scaled MFMA variants that combine quantization scaling with matrix multiply.",
        },
        {
            "category": "GEMM",
            "subcategory": "Weight-Only Quantization",
            "algorithm": "Weight-Only GEMV (W4A16, W8A16)",
            "trtllm_impl": "weightOnlyBatchedGemv/",
            "trtllm_key_technique": "On-the-fly dequantization during GEMV, per-channel/per-group scales",
            "aiter_impl": "csrc/ck_gemm_a8w8_bpreshuffle* (pre-shuffled weights)",
            "ck_impl": "CK weight-only GEMM with dequantization epilogue",
            "co_kernels": "Limited direct .co; mostly CK-compiled",
            "key_differences": [
                "aiter uses pre-shuffled weight layouts (bpreshuffle) for efficient dequant",
                "NVIDIA has dedicated weight-only GEMV; AMD typically uses GEMM pipeline",
                "CK supports blockscale variants for weight-only",
            ],
            "optimization_notes": "Pre-shuffled weight layouts avoid runtime transposition overhead during dequantization.",
        },
        {
            "category": "GEMM",
            "subcategory": "Grouped GEMM",
            "algorithm": "Grouped/Batched GEMM",
            "trtllm_impl": "groupGemm.cu, splitkGroupGemm.cu, cuda_graph_grouped_gemm.cu",
            "trtllm_key_technique": "Multiple concurrent GEMMs, variable sizes per group, CUDA graph compatible",
            "aiter_impl": "csrc/ck_batched_gemm_*, ck_tile_gemm_moe_2stages/",
            "ck_impl": "CK batched GEMM with variable group sizes",
            "co_kernels": "fmoe_2stages/*.co (two-stage grouped GEMM for MoE)",
            "key_differences": [
                "CK batched GEMM supports bf16 and INT8 variants",
                "aiter 2-stage MoE GEMM splits routing and expert compute",
                "NVIDIA uses CUDA graphs for grouped GEMM; AMD uses persistent kernel approach",
            ],
            "optimization_notes": "Two-stage MoE GEMM in .co kernels shows separate routing stage (stage1) followed by expert GEMM, allowing better load balancing.",
        },
        {
            "category": "GEMM",
            "subcategory": "FP4/MX Formats",
            "algorithm": "FP4 / MX Format GEMM",
            "trtllm_impl": "tinygemm2/ (low-precision GEMM)",
            "trtllm_key_technique": "Sub-byte quantized GEMM for extreme compression",
            "aiter_impl": "hsa/gfx950/fmoe/*MXfp4*.co, csrc/ck_gemm_a4w4_blockscale/",
            "ck_impl": "CK A4W4 GEMM with block-scale, MX format pipelines",
            "co_kernels": "gfx950 only: fmoe_*MXfp4*.co (MX FP4 MoE kernels)",
            "key_differences": [
                "gfx950 introduces v_mfma_f32_16x16x128_f8f6f4 supporting FP4/FP6 in hardware",
                "v_mfma_scale_f32_16x16x128_f8f6f4 combines scaling with FP4 multiply",
                "Only available on gfx950; gfx942 requires software dequantization",
            ],
            "optimization_notes": "MX FP4 kernels are gfx950-exclusive and leverage new scaled MFMA instructions for 2x throughput over FP8.",
        },
    ])

    # ============================================================
    # MoE
    # ============================================================
    mappings.extend([
        {
            "category": "MoE",
            "subcategory": "MoE Full Pipeline",
            "algorithm": "Fused MoE (Routing + Expert GEMM)",
            "trtllm_impl": "customMoeRoutingKernels.cu + cutlass_kernels/ (expert GEMM)",
            "trtllm_key_technique": "TopK routing, token permutation, grouped GEMM across experts",
            "aiter_impl": "hsa/gfx942/fmoe/*.co (1000+ fused MoE kernel variants)",
            "ck_impl": "moe_flatmm_pipeline_agmem_bgmem_creg, ck_tile_gemm_moe_2stages",
            "co_kernels": "fmoe/*.co covering: bf16/fp16 × fp8/int8/noquant × g1u0/g1u1 × gelu/silu × various tile sizes",
            "key_differences": [
                "aiter provides 1000+ pre-compiled MoE kernel variants covering all combinations",
                "Each variant is optimized for specific data type, activation, tile size, and expert config",
                "AMD uses flatmm pipeline for MoE (A in LDS, B in registers) vs CUTLASS grouped GEMM",
                "aiter supports both single-stage (fmoe/) and two-stage (fmoe_2stages/) pipelines",
            ],
            "optimization_notes": "MoE kernels are the largest category (2,071 kernels, 1,754,016 MFMA total). Naming convention encodes: datatype_quanttype_gateconfig_activation_threadgroups_tilesize.",
        },
        {
            "category": "MoE",
            "subcategory": "Two-Stage MoE",
            "algorithm": "Two-Stage MoE Pipeline",
            "trtllm_impl": "moePrepareKernels.cu + groupGemm.cu (separate routing + GEMM)",
            "trtllm_key_technique": "Stage 1: token routing and expert assignment; Stage 2: expert GEMM",
            "aiter_impl": "hsa/gfx942/fmoe_2stages/*.co",
            "ck_impl": "ck_gemm_moe_2stages_codegen, ck_tile_gemm_moe_2stages",
            "co_kernels": "fmoe_2stages/fmoe_stage1_*.co (stage1 routing + small GEMM)",
            "key_differences": [
                "Two-stage approach separates routing from expert compute for better load balancing",
                "Stage 1 kernels handle token sorting and expert assignment",
                "Stage 2 uses grouped GEMM for actual expert computation",
                "Supports dynamic expert count and variable token assignment",
            ],
            "optimization_notes": "Stage 1 .co kernels show various tile sizes (48x128, 64x64, 64x128, 80x128, 96x64, 96x128) and prefetch depths (pf2, pf3) for different problem sizes.",
        },
        {
            "category": "MoE",
            "subcategory": "MoE Communication",
            "algorithm": "Expert Parallel Communication",
            "trtllm_impl": "fusedMoeCommKernels.cu, communicationKernels/",
            "trtllm_key_technique": "All-to-all token dispatch/combine for expert parallelism",
            "aiter_impl": "hsa/gfx942/all_reduce.co, aiter/ops/allreduce.py",
            "ck_impl": "Not CK; aiter provides custom communication kernels",
            "co_kernels": "all_reduce.co, allreduce_layernorm_*.co, allreduce_rmsnorm_*.co",
            "key_differences": [
                "aiter fuses allreduce with normalization (allreduce_rmsnorm, allreduce_layernorm)",
                "Fused allreduce+norm reduces memory traffic by avoiding intermediate buffer",
                "AMD uses custom RDMA/RoCE communication vs NVIDIA NVLink-optimized",
            ],
            "optimization_notes": "Fused allreduce+normalization kernels eliminate one memory round-trip, critical for multi-GPU inference latency.",
        },
    ])

    # ============================================================
    # NORMALIZATION
    # ============================================================
    mappings.extend([
        {
            "category": "Normalization",
            "subcategory": "LayerNorm / RMSNorm",
            "algorithm": "LayerNorm and RMSNorm",
            "trtllm_impl": "layernormKernels.cu, rmsnormKernels.cu, fusedLayernormKernels/, groupRmsNormKernels/",
            "trtllm_key_technique": "Welford's algorithm, vectorized loads, warp-level reduction, fused residual",
            "aiter_impl": "aiter/ops/triton/layernorm.py, hsa/gfx942/allreduce_rmsnorm_*.co",
            "ck_impl": "CK normalization primitives",
            "co_kernels": "allreduce_rmsnorm_N8192.co, allreduce_layernorm_N8192.co (fused with allreduce)",
            "key_differences": [
                "aiter fuses normalization with allreduce for multi-GPU",
                "aiter primarily uses Triton for standalone normalization",
                "Fused variants available as .co for specific hidden dimensions (N=8192)",
                "AMD uses wave-level reductions (DPP) vs NVIDIA warp shuffles",
            ],
            "optimization_notes": "Normalization kernels optimized for specific hidden dimensions. Fused allreduce+norm eliminates intermediate write-back.",
        },
        {
            "category": "Normalization",
            "subcategory": "Fused Norm + Quantization",
            "algorithm": "Fused RMSNorm + Quantization",
            "trtllm_impl": "fusedLayernormKernels/ (norm + quant epilogue), fusedActivationQuant.cu",
            "trtllm_key_technique": "Single-pass normalization with INT8/FP8 output quantization",
            "aiter_impl": "hsa/gfx942/allreduce_rmsnorm_qnt_*.co",
            "ck_impl": "CK normalization with quantization epilogue",
            "co_kernels": "allreduce_rmsnorm_qnt_N8192.co",
            "key_differences": [
                "AMD fuses allreduce + RMSNorm + quantization in a single kernel",
                "Three-way fusion (allreduce + norm + quant) not common in NVIDIA stack",
            ],
            "optimization_notes": "Triple-fused kernel (allreduce+norm+quant) maximizes compute density and minimizes memory traffic for multi-GPU quantized inference.",
        },
    ])

    # ============================================================
    # QUANTIZATION
    # ============================================================
    mappings.extend([
        {
            "category": "Quantization",
            "subcategory": "FP8 Block-Scale",
            "algorithm": "FP8 Block-Scale Quantization",
            "trtllm_impl": "quantization.cu (FP8 path), internal_cutlass_kernels/ (FP8 GEMM)",
            "trtllm_key_technique": "Per-block FP8 scaling with E4M3 format, block size typically 128",
            "aiter_impl": "csrc/ck_gemm_a8w8_blockscale*, hsa/gfx942/fmoe/*blockscale*.co",
            "ck_impl": "CK FP8 GEMM with block-scale quantization support",
            "co_kernels": "fmoe_*blockscale*.co (block-scale quantized MoE)",
            "key_differences": [
                "AMD supports block-scale at both per-token and per-block granularity",
                "gfx950 v_mfma_scale_f32_16x16x128_f8f6f4 handles scale in hardware",
                "CK embeds block-scale handling in the GEMM pipeline epilogue",
            ],
            "optimization_notes": "Block-scale FP8 with gfx950 scaled MFMA eliminates separate dequantization step, improving both throughput and efficiency.",
        },
        {
            "category": "Quantization",
            "subcategory": "Per-Token Quantization",
            "algorithm": "Per-Token INT8/FP8 Quantization",
            "trtllm_impl": "quantization.cu (per_token path), preQuantScaleKernel.cu (SmoothQuant)",
            "trtllm_key_technique": "Per-token dynamic scale, SmoothQuant migration",
            "aiter_impl": "hsa/gfx942/fmoe/*pertokenFp8*.co, fmoe/*pertokenInt8*.co",
            "ck_impl": "CK per-token quantization in MoE pipeline",
            "co_kernels": "fmoe_*pertokenFp8*.co, fmoe_*pertokenInt8*.co (extensive per-token variants)",
            "key_differences": [
                "aiter integrates per-token quantization directly into MoE GEMM kernels",
                "No separate quantization step needed; dequant is fused in GEMM pipeline",
                "Both FP8 and INT8 per-token variants available with various tile sizes",
            ],
            "optimization_notes": "Per-token quantized MoE kernels are the most numerous kernel variant, indicating this is the primary quantization strategy for production deployment.",
        },
    ])

    # ============================================================
    # KV-CACHE
    # ============================================================
    mappings.extend([
        {
            "category": "KV-Cache",
            "subcategory": "Paged KV-Cache",
            "algorithm": "Paged KV-Cache Management",
            "trtllm_impl": "kvCacheUtils.h, kvCachePartialCopy.cu",
            "trtllm_key_technique": "Block-level page tables, dynamic allocation, FP8 quantized KV",
            "aiter_impl": "aiter/ops/triton/paged_attn.py, hsa/gfx942/pa/*.co",
            "ck_impl": "CK KV-cache integration in attention pipelines",
            "co_kernels": "pa/*.co (paged attention kernels with KV-cache access)",
            "key_differences": [
                "AMD paged attention uses buffer_load for page table traversal",
                "aiter PA kernels have KV-cache access patterns baked into ASM",
                "FP8 KV-cache quantization supported in both platforms",
            ],
            "optimization_notes": "Paged attention .co kernels use careful buffer_load scheduling to overlap page table lookups with KV-cache reads.",
        },
    ])

    # ============================================================
    # SOFTMAX / TOPK
    # ============================================================
    mappings.extend([
        {
            "category": "Sampling",
            "subcategory": "TopK / Softmax",
            "algorithm": "TopK Softmax",
            "trtllm_impl": "samplingTopKKernels.cu, samplingTopPKernels.cu",
            "trtllm_key_technique": "Radix-based TopK, cumulative probability, temperature scaling",
            "aiter_impl": "hsa/gfx942/topksoftmax/*.co, aiter/ops/triton/topk_softmax.py",
            "ck_impl": "Not CK; custom kernels",
            "co_kernels": "topksoftmax/*.co (topksoftmax_4x256x6, topksoftmax_4x128x6, etc.)",
            "key_differences": [
                "aiter provides pre-compiled TopK+Softmax fused kernels",
                "Naming encodes dimensions: batch x vocab_chunk x top_k",
                "No MFMA in TopK kernels (compute-bound on VALU/SALU)",
            ],
            "optimization_notes": "TopK+Softmax kernels are VALU-heavy (no MFMA) and optimized for specific vocabulary chunk sizes.",
        },
    ])

    # ============================================================
    # SPECIALIZED
    # ============================================================
    mappings.extend([
        {
            "category": "Specialized",
            "subcategory": "SSM / Mamba",
            "algorithm": "Selective Scan (Mamba)",
            "trtllm_impl": "selectiveScan/, causalConv1d/",
            "trtllm_key_technique": "Parallel prefix scan, selective gating, causal 1D convolution",
            "aiter_impl": "aiter/ops/triton/ (Triton-based SSM kernels)",
            "ck_impl": "Not in CK; Triton implementations",
            "co_kernels": "No dedicated .co; Triton-compiled",
            "key_differences": [
                "Both platforms use Triton for SSM kernels",
                "AMD Triton compiles to AMDGPU ISA via LLVM backend",
                "No dedicated ASM optimization for SSM yet in aiter",
            ],
            "optimization_notes": "SSM/Mamba kernels are memory-bandwidth-bound, making hardware memory bandwidth (MI300X: 5.3 TB/s) the key differentiator.",
        },
        {
            "category": "Specialized",
            "subcategory": "Speculative Decoding",
            "algorithm": "Speculative Decoding",
            "trtllm_impl": "speculativeDecoding/",
            "trtllm_key_technique": "Draft-verify pattern, token acceptance/rejection, tree-based speculation",
            "aiter_impl": "aiter/ops/ (speculative decoding support at framework level)",
            "ck_impl": "Not in CK; framework-level implementation",
            "co_kernels": "No dedicated .co",
            "key_differences": [
                "Speculative decoding is primarily a framework-level optimization",
                "Kernel-level differences minimal; relies on fast attention decode",
                "AMD advantage: high memory bandwidth for draft model parallel execution",
            ],
            "optimization_notes": "Speculative decoding benefits from fast single-token decode attention, where AMD's paged attention .co kernels provide the critical path optimization.",
        },
    ])

    return mappings


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mappings = build_mapping()

    # Build category summary
    from collections import defaultdict
    by_category = defaultdict(list)
    for m in mappings:
        by_category[m["category"]].append(m["algorithm"])

    output = {
        "description": "Structured mapping between TensorRT-LLM (NVIDIA) and aiter/CK (AMD) kernel algorithms",
        "total_mappings": len(mappings),
        "categories": {cat: {"count": len(algos), "algorithms": algos} for cat, algos in sorted(by_category.items())},
        "mappings": mappings,
    }

    out_file = OUTPUT_DIR / "trtllm_amd_mapping.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"{'='*60}")
    print(f"NVIDIA <-> AMD Kernel Algorithm Mapping")
    print(f"{'='*60}")
    print(f"Total mappings: {len(mappings)}")
    print(f"\nBy category:")
    for cat, algos in sorted(by_category.items()):
        print(f"  {cat:20s}: {len(algos)} mappings")
        for a in algos:
            print(f"    - {a}")
    print(f"\nOutput: {out_file}")


if __name__ == "__main__":
    main()
