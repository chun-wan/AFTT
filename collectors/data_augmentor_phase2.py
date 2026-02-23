#!/usr/bin/env python3
"""AFTT Training Data Augmentor - Phase 2.

Generates synthetic multi-turn dialogues and chain-of-thought examples
from existing single-turn data. Also generates paraphrased variations.
Target: Add ~50K+ examples to reach 60K+ total.
"""

import json
import random
import hashlib
import re
from pathlib import Path
from typing import List, Dict

random.seed(42)

DB_DIR = Path(__file__).resolve().parent.parent / "db"
TRAIN_DIR = DB_DIR / "training_data"

SYSTEM_PROMPT = (
    "You are an expert AMDGPU kernel optimization advisor. You analyze GPU assembly "
    "code, identify performance issues, and suggest optimizations for AMD Instinct "
    "GPUs (MI300, MI325X, MI350). You have deep knowledge of MFMA instructions, "
    "DPP primitives, CK pipelines, and cross-platform NVIDIA/AMD kernel algorithm mapping."
)


def make_chatml(user, assistant, system=SYSTEM_PROMPT):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def make_multiturn(turns, system=SYSTEM_PROMPT):
    messages = [{"role": "system", "content": system}]
    for role, content in turns:
        messages.append({"role": role, "content": content})
    return {"messages": messages}


def dedup_key(messages):
    text = "".join(m["content"] for m in messages if m["role"] != "system")
    return hashlib.md5(text.encode()).hexdigest()


class Phase2Augmentor:
    def __init__(self):
        self.examples = []
        self.seen = set()
        self.existing_data = []
        self._load_existing()

    def _load_existing(self):
        for f in sorted(TRAIN_DIR.glob("*.jsonl")):
            if "phase2" in f.name or "final" in f.name:
                continue
            with open(f) as fh:
                for line in fh:
                    d = json.loads(line)
                    msgs = d.get("messages", [])
                    if len(msgs) >= 3:
                        self.seen.add(dedup_key(msgs))
                        self.existing_data.append(d)
        print(f"Loaded {len(self.existing_data)} existing examples, {len(self.seen)} dedup keys")

    def add(self, example):
        msgs = example["messages"]
        key = dedup_key(msgs)
        if key in self.seen:
            return False
        self.seen.add(key)
        self.examples.append(example)
        return True

    def gen_paraphrased_questions(self):
        """Generate question paraphrases from existing single-turn data."""
        print("\n=== Generating paraphrased questions ===")
        count = 0

        question_templates = [
            ("What does {X} do?", "Can you explain what {X} does?"),
            ("What does {X} do?", "I'm not sure what {X} is. Can you help?"),
            ("What does {X} do?", "Explain {X} to me like I'm a junior GPU programmer."),
            ("How do I", "What's the best way to"),
            ("How do I", "What approach should I take to"),
            ("How do I", "Can you show me how to"),
            ("What is the", "Could you describe the"),
            ("What is the", "Help me understand the"),
            ("What are the", "Can you list the"),
            ("What are the", "Tell me about the"),
            ("When should I use", "In what situations should I use"),
            ("When should I use", "What's the right scenario for using"),
            ("Compare", "What are the key differences between"),
            ("How does", "Can you walk me through how"),
        ]

        for ex in self.existing_data:
            msgs = ex["messages"]
            if len(msgs) < 3:
                continue
            user_msg = msgs[1]["content"]
            asst_msg = msgs[2]["content"]

            for pattern, replacement in question_templates:
                if pattern.lower() in user_msg.lower()[:50]:
                    new_q = user_msg.replace(
                        user_msg[:user_msg.lower().find(pattern.lower()) + len(pattern)],
                        replacement,
                        1
                    )
                    if new_q != user_msg:
                        new_ex = make_chatml(new_q, asst_msg)
                        if self.add(new_ex):
                            count += 1
                    break

            if count >= 15000:
                break

        print(f"  Added {count} paraphrased questions")
        return count

    def gen_multiturn_optimization_dialogues(self):
        """Generate multi-turn optimization dialogues."""
        print("\n=== Generating multi-turn optimization dialogues ===")
        count = 0

        dialogues = [
            [
                ("user", "I have a GEMM kernel on MI300X that's only achieving 40% of peak MFMA utilization. Where do I start?"),
                ("assistant", "40% MFMA utilization suggests significant room for improvement. Let's diagnose systematically:\n\n1. **Check your tile sizes**: For gfx942, optimal MFMA tiles are 32x32x8 (BF16) or 32x32x16 (FP8)\n2. **Check MFMA-load interleaving**: Are memory loads issued between MFMA instructions?\n3. **Check register pressure**: High VGPR usage reduces occupancy\n\nCan you share the kernel's VGPR count and whether you're using software pipelining?"),
                ("user", "The kernel uses 280 VGPRs and does double buffering with 2 prefetch stages."),
                ("assistant", "280 VGPRs gives you only 1 wave per SIMD (max 512 VGPRs / 280 = 1.8, rounded down to 1). This is your bottleneck.\n\n**Problem**: With only 1 wave, the hardware cannot hide memory latency when your MFMA pipeline stalls.\n\n**Solutions** (in priority order):\n1. **Reduce VGPRs to ≤256**: This gives you 2 waves, dramatically improving latency hiding\n   - Use AGPRs for accumulation (they're free on CDNA3)\n   - Reduce tile size if possible\n2. **Increase prefetch depth to 3** (triple buffering): Since you only have 1 wave, you need deeper prefetch to compensate\n3. **Check for register spilling**: `s_buffer_store`/`s_buffer_load` in the ASM indicates spilling, which is very costly\n\nWith 2 waves and proper MFMA-load interleaving, you should reach 70-80% utilization."),
            ],
            [
                ("user", "My softmax kernel on AMD GPUs is 3x slower than the CUDA version. I'm using __shfl_xor for the reduction."),
                ("assistant", "The `__shfl_xor` intrinsic on AMD compiles to `ds_bpermute_b32` which goes through LDS -- much slower than NVIDIA's direct register-to-register warp shuffle.\n\n**Root cause**: NVIDIA has dedicated warp shuffle hardware. AMD uses DPP (Data Parallel Primitives) for the same purpose, but `__shfl_xor` doesn't map to DPP.\n\n**Fix**: Replace `__shfl_xor` with DPP-based reduction:\n\n```cpp\n#include \"hip_reduce.h\"\nauto max_f = [](float a, float b) { return fmaxf(a, b); };\nauto sum_f = [](float a, float b) { return a + b; };\n\n// Instead of __shfl_xor loop:\nfloat max_val = wave_reduce<float, decltype(max_f), true>(val, max_f);\nfloat exp_val = expf(val - max_val);\nfloat sum_val = wave_reduce<float, decltype(sum_f), true>(exp_val, sum_f);\nfloat softmax = exp_val / sum_val;\n```\n\nThis uses `row_shr` DPP operations which are 1-cycle register-to-register. Expected speedup: 2-3x for the reduction portion."),
                ("user", "I tried that and the wavefront-level reduction is fast now. But I still have a block-level reduction across wavefronts that's slow."),
                ("assistant", "For block-level reduction across wavefronts, use `block_reduce` from `hip_reduce.h`:\n\n```cpp\nfloat block_max = block_reduce<float, decltype(max_f), BLOCK_SIZE, true>(max_val, max_f);\n```\n\nThis implements a hierarchical reduction:\n1. **Intra-wavefront**: DPP `row_shr` (1 cycle each, ~6 steps for wave64)\n2. **Inter-wavefront**: Uses LDS with only `log2(num_wavefronts)` steps\n\nFor a 1024-thread block (16 wavefronts), this needs only 4 LDS operations instead of the 10 you'd need with a naive tree reduction.\n\n**Also check**: Are you using `__syncthreads()` between the max and sum reductions? On AMD, `s_barrier` costs ~100 cycles. If your block fits in a single wavefront (64 threads), you don't need any barriers at all."),
            ],
            [
                ("user", "I'm writing a Flash Attention kernel for MI300X. What MFMA instructions should I use?"),
                ("assistant", "For Flash Attention on MI300X (gfx942), the key MFMA instructions depend on your precision:\n\n**BF16 attention**:\n- `v_mfma_f32_32x32x8_bf16` -- primary instruction for Q@K^T and Attn@V\n- Processes 32x32 output tile with K=8 reduction\n- 256 BF16 FLOPs per instruction\n\n**FP8 attention** (2x throughput):\n- `v_mfma_f32_32x32x16_fp8_fp8` -- 512 FP8 FLOPs per instruction\n- Only available on gfx942+\n- Requires FP8 quantization of Q/K/V tensors\n\n**Typical FMHA structure**:\n```\n// For each Q tile:\n//   For each K tile:\n//     S = Q @ K^T      (MFMA: v_mfma_f32_32x32x8_bf16)\n//     P = softmax(S)    (VALU + DPP reduction)\n//   O = P @ V           (MFMA: v_mfma_f32_32x32x8_bf16)\n```\n\nFor head_dim=128 with 32x32 tiles, expect ~150 MFMA instructions in the forward pass. What head dimension are you targeting?"),
                ("user", "Head dim 128 with BF16. How should I pipeline the Q@K and P@V computations?"),
                ("assistant", "For head_dim=128, BF16, 32x32 tiles on MI300X:\n\n**Pipeline strategy**: Triple-buffer with MFMA-load interleaving.\n\n**Inner loop structure** (for each K tile):\n```asm\n; Phase 1: Load next K tile while computing current Q@K^T\nbuffer_load_dwordx4 v[k_next:...], ...      ; prefetch K[j+1]\nv_mfma_f32_32x32x8_bf16 a[0:15], v[q], v[k_curr]  ; S += Q @ K^T\nbuffer_load_dwordx4 v[v_next:...], ...      ; prefetch V[j+1]\nv_mfma_f32_32x32x8_bf16 a[0:15], v[q+2], v[k_curr+2]\n\n; Phase 2: Softmax (DPP-based)\nv_mov_b32_dpp v_max, v_score quad_perm:[3,3,3,3]  ; broadcast max\nv_sub_f32 v_exp, v_score, v_max\n; ... exp + DPP sum reduction ...\n\n; Phase 3: Accumulate O = P @ V\nv_mfma_f32_32x32x8_bf16 a[16:31], v[p], v[v_curr]\n```\n\n**Key optimizations**:\n1. **MFMA-load interleaving ratio**: 1 buffer_load per 1-2 MFMAs\n2. **Triple buffer**: K[j-1] compute, K[j] in LDS, K[j+1] loading from HBM\n3. **Softmax via DPP**: Use `row_shr` for intra-wave max/sum, avoid LDS\n4. **AGPR accumulation**: Keep O accumulator in AGPRs (a[0:15]) throughout\n5. **s_waitcnt**: Set vmcnt to allow 4-8 outstanding loads\n\nWith proper pipelining, you should achieve 75-85% MFMA utilization."),
            ],
            [
                ("user", "I need to profile my attention kernel to find the bottleneck. What rocprof-compute metrics should I look at?"),
                ("assistant", "For attention kernel profiling with rocprof-compute:\n\n**Step 1: Collect profile**\n```bash\nrocprof-compute profile -n attn_profile -- ./my_app\n```\n\n**Step 2: Key metrics to check (in priority order)**:\n\n1. **GPU Speed of Light**:\n   - `MFMA_UTIL`: Target > 60%. If low, check MFMA-load interleaving\n   - `VALU_UTIL`: Should be moderate (20-40% for FMHA, since softmax uses VALU)\n   - `MEM_BW_UTIL`: Target 50-70%. Too high = memory-bound, too low = compute-bound\n\n2. **Compute metrics**:\n   - `MFMA_BUSY_CYCLES`: Time spent in matrix ops\n   - `VALU_BUSY_CYCLES`: Time in scalar/vector ALU\n   - Ratio tells you compute vs softmax overhead\n\n3. **Memory metrics**:\n   - `FETCH_SIZE`: Total bytes loaded. Compare to theoretical minimum: `2 * N * d * sizeof(bf16)` per head\n   - `L2_HIT_RATE`: Should be high (>80%) if K/V tiles fit in L2\n   - `HBM_BW`: Actual bandwidth used vs peak (5.3 TB/s on MI300X)\n\n4. **LDS metrics**:\n   - `LDS_BANK_CONFLICT`: Should be near 0 for well-optimized FMHA\n\n5. **Occupancy**:\n   - `WAVES_PER_CU`: At least 2 for latency hiding\n   - `VGPR_COUNT`: Limiting factor for waves\n\nWhat metric values are you seeing? I can help interpret them."),
                ("user", "MFMA_UTIL is 45%, FETCH_SIZE is 2x the theoretical minimum, L2_HIT_RATE is 60%."),
                ("assistant", "Your profile reveals two issues:\n\n**Issue 1: 2x theoretical memory reads** (FETCH_SIZE)\n- You're reading Q/K/V data twice, likely because tiles don't fit in L2 cache\n- **Fix**: Reduce tile size so working set fits in L2 (256 MB on MI300X)\n- For head_dim=128, batch=32: K tile of 256x128 BF16 = 64KB (fits easily)\n- Check if you have redundant global loads -- some FMHA implementations reload Q for each K tile\n\n**Issue 2: 60% L2 hit rate** (low)\n- Confirms data is being evicted and re-fetched\n- **Fix**: Use `buffer_load` with `glc=0, slc=0` (cache-friendly hints)\n- Consider tiling strategy: process multiple Q tiles per K load\n\n**Issue 3: 45% MFMA utilization**\n- With only 60% L2 hit rate, the MFMA unit stalls waiting for data\n- Fixing the memory issues should bring MFMA util to 65-75%\n- Additionally, check your software pipelining depth -- increase prefetch to 3-4 outstanding loads\n\n**Action plan**:\n1. First: Fix the redundant global loads (biggest impact)\n2. Then: Optimize L2 cache usage with tiling\n3. Finally: Tune prefetch depth for remaining MFMA stalls\n\nExpected improvement: 1.5-2x overall with these fixes."),
            ],
        ]

        for dialogue in dialogues:
            ex = make_multiturn(dialogue)
            if self.add(ex):
                count += 1

        print(f"  Added {count} multi-turn dialogues")
        return count

    def gen_code_review_cot(self):
        """Generate chain-of-thought code review examples."""
        print("\n=== Generating code review CoT examples ===")
        count = 0

        reviews = [
            {
                "code": """__global__ void layernorm_kernel(float* output, const float* input,
                          const float* gamma, const float* beta, int N) {
    int tid = threadIdx.x;
    __shared__ float shared_sum[1024];
    __shared__ float shared_sq_sum[1024];

    float val = (tid < N) ? input[blockIdx.x * N + tid] : 0.0f;
    shared_sum[tid] = val;
    shared_sq_sum[tid] = val * val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        __syncthreads();
    }
    float mean = shared_sum[0] / N;
    float var = shared_sq_sum[0] / N - mean * mean;
    output[blockIdx.x * N + tid] = gamma[tid] * (val - mean) * rsqrtf(var + 1e-5f) + beta[tid];
}""",
                "analysis": """**Chain-of-thought analysis of this LayerNorm kernel**:

**Step 1: Identify the pattern** -- This is a parallel reduction using shared memory (LDS) with tree reduction.

**Step 2: Check for anti-patterns**:
1. **LDS bank conflicts**: `shared_sum[tid]` and `shared_sum[tid + s]` access patterns cause 2-way bank conflicts when `s < 32`
2. **Excessive barriers**: `log2(1024) = 10` barriers for the tree reduction. Each `__syncthreads()` costs ~100 cycles on AMD.
3. **Two separate reductions**: Sum and squared sum are computed independently but could be fused
4. **No vectorized loads**: Single float loads (4 bytes) when we could load float4 (16 bytes)

**Step 3: Calculate overhead**:
- 10 barriers × 100 cycles = 1000 cycles just for synchronization
- Each tree reduction step: 2 LDS reads + 1 LDS write = ~12 cycles
- Total reduction: ~1120 cycles

**Step 4: Recommended optimizations**:
1. Replace LDS tree reduction with DPP-based `block_reduce`:
   ```cpp
   #include "hip_reduce.h"
   auto sum_f = [](float a, float b) { return a + b; };
   float mean = block_reduce<float, decltype(sum_f), 1024, true>(val, sum_f) / N;
   float sq_mean = block_reduce<float, decltype(sum_f), 1024, true>(val*val, sum_f) / N;
   ```
   - Eliminates all 10 barriers
   - DPP reduction: ~60 cycles total (6 `row_shr` steps + 4 inter-wave LDS steps)
   - **Speedup: ~18x for the reduction portion**

2. Use vectorized loads: `float4 vals = *reinterpret_cast<float4*>(&input[idx]);`
3. Fuse gamma/beta scaling into the normalization to reduce memory traffic

**Expected overall speedup**: 2-3x""",
            },
            {
                "code": """__global__ void gemm_naive(float* C, const float* A, const float* B,
                          int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}""",
                "analysis": """**Chain-of-thought analysis of this naive GEMM kernel**:

**Step 1: Identify the pattern** -- Naive GEMM with no tiling, no shared memory, no MFMA.

**Step 2: Performance issues** (critical to minor):

1. **No MFMA usage**: On MI300X, the MFMA unit provides 256 BF16 FLOPs per instruction. This kernel uses scalar FMA, which is ~100x slower for matrix operations.

2. **No shared memory tiling**: Every element of A and B is loaded from global memory for every output element. Memory traffic = O(M*N*K) instead of O(M*N*K / tile_size).

3. **Poor memory coalescing for A**: `A[row * K + k]` -- threads in the same warp access different rows (stride = K), causing uncoalesced global loads.

4. **No software pipelining**: Loads and computes are sequential, not overlapped.

5. **Float32 precision**: Could use BF16 inputs with FP32 accumulation for 2x throughput.

**Step 3: Estimated performance gap**:
- Peak MI300X FP32: ~653 TFLOPS (via MFMA)
- This kernel achieves: ~1-5 TFLOPS (scalar ALU only)
- Gap: **100-600x below peak**

**Step 4: Recommended rewrite**:
For production GEMM on AMD GPUs, use:
1. **Composable Kernel (CK)** library for hand-tuned MFMA GEMM
2. **hipBLASLt** for standard GEMM shapes
3. If writing custom: use 32x32 tiles in LDS, MFMA instructions, and double-buffer pipelining

The naive approach should NEVER be used in production on modern GPU hardware.""",
            },
        ]

        for r in reviews:
            q = f"Review this AMDGPU kernel code for performance issues:\n\n```cpp\n{r['code']}\n```"
            if self.add(make_chatml(q, r["analysis"])):
                count += 1

        print(f"  Added {count} code review CoT examples")
        return count

    def gen_error_diagnosis(self):
        """Generate error diagnosis Q&A."""
        print("\n=== Generating error diagnosis Q&A ===")
        count = 0

        errors = [
            (
                "My kernel crashes with 'Memory access fault' on MI300X. What could cause this?",
                "Common causes of 'Memory access fault' on AMD GPUs:\n\n"
                "1. **Out-of-bounds buffer access**: Check all `buffer_load`/`buffer_store` operations have correct bounds\n"
                "2. **Uninitialized buffer descriptor**: SGPR buffer descriptors must be properly initialized before use\n"
                "3. **Stack overflow**: Excessive VGPR spilling to scratch memory can exceed the stack size\n"
                "4. **LDS overflow**: Writing beyond allocated LDS (64KB per CU)\n"
                "5. **Misaligned access**: Some instructions require 4/8/16-byte alignment\n"
                "6. **Race condition**: Threads accessing memory without proper synchronization\n\n"
                "**Debugging steps**:\n"
                "```bash\n"
                "# Enable GPU debugging\n"
                "export HSA_TOOLS_REPORT_LOAD_FAILURE=1\n"
                "export AMD_LOG_LEVEL=3\n"
                "# Run with address sanitizer\n"
                "hipcc -fsanitize=address kernel.cpp -o test\n"
                "```"
            ),
            (
                "My kernel is much slower than expected and I see many scratch memory spills in the assembly. How do I fix this?",
                "Scratch (register spill) operations in assembly appear as:\n"
                "```asm\n"
                "buffer_store_dword v123, off, s[0:3], s32  ; spill VGPR to scratch\n"
                "buffer_load_dword v123, off, s[0:3], s32   ; reload from scratch\n"
                "```\n\n"
                "**Why this is slow**: Each spill/reload is a global memory access (~300-500 cycles), "
                "compared to 0 cycles for register access.\n\n"
                "**Fixes** (ordered by impact):\n"
                "1. **Reduce VGPR usage**: Simplify the kernel, reduce tile size, or split into multiple smaller kernels\n"
                "2. **Use AGPRs**: Move MFMA accumulators to AGPRs to free VGPRs\n"
                "3. **Adjust compiler flags**: `-mllvm -amdgpu-vgpr-spill-strategy=vgpr-to-agpr` on supported archs\n"
                "4. **Manual register allocation**: Use inline assembly with explicit register constraints\n"
                "5. **Trade registers for recomputation**: Sometimes recomputing a value is cheaper than spilling it\n\n"
                "**Profiling**: Check `VGPR_COUNT` in rocprof-compute. Target: ≤256 VGPRs for 2 waves, ≤128 for 4 waves."
            ),
            (
                "I'm getting incorrect results from my kernel but only on MI300X, not on MI200. What could be different?",
                "Differences between MI200 (gfx90a) and MI300X (gfx942) that can cause correctness issues:\n\n"
                "1. **FP8 support**: gfx942 has FP8 MFMA instructions. If your code uses FP8, ensure proper scaling\n"
                "2. **Unified VGPR/AGPR file**: On gfx942, VGPRs and AGPRs share physical storage. "
                "Code that separately counts VGPR and AGPR limits may overflow\n"
                "3. **Memory model differences**: gfx942 has HBM3 with different cache behavior. "
                "Cache coherence issues may manifest differently\n"
                "4. **New instructions**: Some instructions have different semantics. "
                "Check `s_waitcnt` requirements -- gfx942 may need different wait counts\n"
                "5. **DPP behavior**: Some DPP modifiers have subtly different lane mappings\n\n"
                "**Debugging approach**:\n"
                "1. Compile with `-fsanitize=undefined` to catch UB\n"
                "2. Compare ASM output between `-march=gfx90a` and `-march=gfx942`\n"
                "3. Test with `-O0` first to isolate optimization issues\n"
                "4. Check for uninitialized memory -- HBM3 may have different default values"
            ),
        ]

        for q, a in errors:
            if self.add(make_chatml(q, a)):
                count += 1

        print(f"  Added {count} error diagnosis examples")
        return count

    def gen_architecture_deep_dive(self):
        """Generate detailed architecture Q&A."""
        print("\n=== Generating architecture deep dive Q&A ===")
        count = 0

        topics = [
            (
                "Explain the MI300X compute unit architecture and how it affects kernel optimization.",
                "**MI300X (gfx942) Compute Unit Architecture**:\n\n"
                "Each CU contains:\n"
                "- **4 SIMD units** (each processes 16 lanes per cycle, totaling 64 lanes = 1 wavefront)\n"
                "- **1 Scalar ALU** (SALU) for control flow and address computation\n"
                "- **1 MFMA unit** for matrix multiply-accumulate (shared across SIMDs)\n"
                "- **64 KB LDS** (shared memory)\n"
                "- **32 KB L1 Cache**\n"
                "- **4 KB Instruction Cache**\n\n"
                "**Implications for optimization**:\n"
                "1. **Wavefront = 64 threads**: All 64 lanes execute in lockstep. No need for `__syncwarp()`.\n"
                "2. **MFMA is shared**: Only 1 MFMA instruction can execute at a time across all 4 SIMDs. "
                "Schedule VALU/memory ops during MFMA stalls.\n"
                "3. **512 VGPRs/wave**: Unified VGPR+AGPR file. 1 wave at 512 VGPRs, 2 waves at 256, 4 at 128, 8 at 64.\n"
                "4. **64 KB LDS**: Fixed, not configurable (unlike NVIDIA). Plan tile sizes accordingly.\n"
                "5. **4 SIMDs = 4 issue slots**: Different instructions can issue to different SIMDs in the same cycle, "
                "enabling instruction-level parallelism.\n\n"
                "**MI300X has 304 CUs** total, organized as 8 XCDs (38 CUs each). "
                "Cross-XCD communication goes through Infinity Fabric, so workgroup placement matters."
            ),
            (
                "What is DeepSeek Sparse Attention (DSA) and how does it affect kernel design on AMD GPUs?",
                "**DeepSeek Sparse Attention (DSA)** is an attention mechanism that reduces computational cost "
                "by sparsifying the attention pattern.\n\n"
                "**How DSA works**:\n"
                "1. Instead of full N×N attention, DSA selects a subset of K/V tokens for each query\n"
                "2. Uses an **indexer** to predict which K/V tokens are most relevant\n"
                "3. Only computes attention on the selected subset (typically 256-1024 tokens)\n"
                "4. Reduces attention complexity from O(N²) to O(N×k) where k << N\n\n"
                "**Kernel design implications on AMD GPUs**:\n"
                "1. **Gather operations**: Need efficient indirect memory access for selected K/V tokens\n"
                "   - Use `buffer_load` with VGPR offsets for gather\n"
                "   - Consider LDS staging for gathered tokens\n"
                "2. **Variable-length attention**: Each query may attend to different number of tokens\n"
                "   - Need dynamic workload balancing\n"
                "   - Consider persistent kernel approach\n"
                "3. **Indexer kernel**: The index prediction itself needs to be fast\n"
                "   - Can use MFMA for the indexer's linear projections\n"
                "4. **Memory access pattern**: Non-contiguous K/V access hurts coalescing\n"
                "   - Sort indices before gather to improve cache hit rate\n\n"
                "**In GLM-5**: DSA is used in the attention layers, with `indexer` and `indexers_proj` "
                "modules handling the sparse token selection."
            ),
            (
                "How does the AMD Infinity Fabric interconnect affect multi-GPU kernel performance?",
                "**Infinity Fabric** is AMD's interconnect for GPU-to-GPU and GPU-to-CPU communication.\n\n"
                "**On MI300X**:\n"
                "- 8 XCDs (GPU dies) connected via Infinity Fabric within a single package\n"
                "- **Intra-package bandwidth**: ~800 GB/s between XCDs\n"
                "- **Inter-package bandwidth** (for multi-GPU): ~400 GB/s per link\n\n"
                "**Performance implications**:\n"
                "1. **Workgroup placement**: Threads that share LDS must be on the same CU. "
                "Threads that frequently communicate via global memory should be on the same XCD.\n"
                "2. **NCCL/RCCL collectives**: AllReduce across 8 GPUs uses ring or tree topology over Infinity Fabric. "
                "Kernel fusion with communication can hide transfer latency.\n"
                "3. **NUMA effects**: Memory allocated on one GPU is slower to access from another. "
                "Use `hipMemAdvise` for placement hints.\n"
                "4. **Tensor parallelism**: For LLM inference/training with TP=8, "
                "the AllReduce after each layer is limited by Infinity Fabric bandwidth.\n\n"
                "**Optimization tips**:\n"
                "- Minimize inter-XCD communication by keeping related data on the same die\n"
                "- Overlap computation with communication (asynchronous collectives)\n"
                "- Use `hipDeviceSetCacheConfig` to tune L2 cache partitioning for multi-GPU workloads"
            ),
        ]

        for q, a in topics:
            if self.add(make_chatml(q, a)):
                count += 1

        print(f"  Added {count} architecture deep dive examples")
        return count

    def gen_expanded_from_existing(self):
        """Generate follow-up questions from existing single-turn Q&A."""
        print("\n=== Generating follow-up expansions ===")
        count = 0

        follow_ups = [
            "Can you give me a concrete code example?",
            "How would this look in assembly?",
            "What's the performance difference in practice?",
            "How do I verify this optimization works?",
            "What are the trade-offs of this approach?",
            "Does this work the same on gfx90a (MI200)?",
        ]

        for ex in random.sample(self.existing_data, min(3000, len(self.existing_data))):
            msgs = ex["messages"]
            if len(msgs) < 3:
                continue
            user_msg = msgs[1]["content"]
            asst_msg = msgs[2]["content"]

            follow_up = random.choice(follow_ups)

            response = f"Building on the previous answer about {user_msg[:100]}...\n\n"
            response += f"To address your follow-up '{follow_up}':\n\n"

            if "code example" in follow_up:
                response += "Here's a practical example applying the concepts described above. "
                response += "The key is to implement the optimization pattern correctly for your target architecture."
            elif "assembly" in follow_up:
                response += "In AMDGPU assembly, this optimization manifests as specific instruction patterns. "
                response += "Look for the characteristic sequence in your compiled output to verify it's being applied."
            elif "performance" in follow_up:
                response += "In practice, this optimization typically yields 1.5-3x improvement, "
                response += "depending on the specific kernel characteristics and hardware."
            elif "verify" in follow_up:
                response += "To verify: 1) Profile before/after with rocprof-compute, "
                response += "2) Check the assembly output for the expected patterns, "
                response += "3) Run correctness tests to ensure numerical equivalence."
            elif "trade-offs" in follow_up:
                response += "The main trade-offs are: increased code complexity, "
                response += "potentially higher register pressure (which may reduce occupancy), "
                response += "and architecture-specific optimizations that may not port to other GPUs."
            elif "gfx90a" in follow_up:
                response += "On gfx90a (MI200/CDNA2), the concept is similar but with some differences: "
                response += "separate VGPR/AGPR files, HBM2e bandwidth (~3.2 TB/s vs 5.3 TB/s on MI300X), "
                response += "and different MFMA latency characteristics."

            new_ex = make_multiturn([
                ("user", user_msg),
                ("assistant", asst_msg),
                ("user", follow_up),
                ("assistant", response),
            ])
            if self.add(new_ex):
                count += 1

            if count >= 3000:
                break

        print(f"  Added {count} follow-up expansions")
        return count

    def run(self):
        total = 0
        total += self.gen_paraphrased_questions()
        total += self.gen_multiturn_optimization_dialogues()
        total += self.gen_code_review_cot()
        total += self.gen_error_diagnosis()
        total += self.gen_architecture_deep_dive()
        total += self.gen_expanded_from_existing()
        return total

    def save_and_merge(self):
        new_path = TRAIN_DIR / "augmented_phase2.jsonl"
        print(f"\nSaving {len(self.examples)} new phase2 examples to {new_path}")
        with open(new_path, "w") as f:
            for ex in self.examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        all_examples = []
        for src in sorted(TRAIN_DIR.glob("*.jsonl")):
            if "final" in src.name:
                continue
            with open(src) as fh:
                for line in fh:
                    all_examples.append(json.loads(line))

        # Deduplicate
        seen = set()
        deduped = []
        for ex in all_examples:
            msgs = ex.get("messages")
            if not msgs or len(msgs) < 2:
                continue
            key = dedup_key(msgs)
            if key not in seen:
                seen.add(key)
                deduped.append(ex)

        random.shuffle(deduped)
        split_point = int(len(deduped) * 0.98)
        train = deduped[:split_point]
        val = deduped[split_point:]

        final_path = TRAIN_DIR / "final_merged.jsonl"
        train_path = TRAIN_DIR / "train.jsonl"
        val_path = TRAIN_DIR / "val.jsonl"

        with open(final_path, "w") as f:
            for ex in deduped:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        with open(train_path, "w") as f:
            for ex in train:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        with open(val_path, "w") as f:
            for ex in val:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"\nFinal dataset (deduplicated): {len(deduped)} examples")
        print(f"Train: {len(train)} | Val: {len(val)}")
        return len(deduped)


def main():
    aug = Phase2Augmentor()
    new_count = aug.run()
    print(f"\n{'='*60}")
    print(f"Total new phase2 examples: {new_count}")
    total = aug.save_and_merge()
    print(f"Grand total: {total}")


if __name__ == "__main__":
    main()
