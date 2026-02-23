#!/usr/bin/env python3
"""AFTT Training Data Augmentor - Phase 1b.

Aggressive augmentation to reach 40K+ examples:
1. ISA instruction pairing / "which is better for X" questions
2. Architecture migration guides
3. Kernel optimization workflow Q&A
4. Register pressure / occupancy Q&A
5. Multi-turn reasoning chains from existing disassembly data
6. MFMA instruction-specific deep Q&A
7. LDS/memory hierarchy Q&A
"""

import json
import random
import hashlib
from pathlib import Path
from itertools import combinations

random.seed(42)

DB_DIR = Path(__file__).resolve().parent.parent / "db"
TRAIN_DIR = DB_DIR / "training_data"

SYSTEM_PROMPT = (
    "You are an expert AMDGPU kernel optimization advisor. You analyze GPU assembly "
    "code, identify performance issues, and suggest optimizations for AMD Instinct "
    "GPUs (MI300, MI325X, MI350). You have deep knowledge of MFMA instructions, "
    "DPP primitives, CK pipelines, and cross-platform NVIDIA/AMD kernel algorithm mapping."
)

ARCH_GPU_MAP = {
    "gfx900": ("MI25", "Vega 10", "GCN5"),
    "gfx906": ("MI50/MI60", "Vega 20", "GCN5"),
    "gfx908": ("MI100", "Arcturus", "CDNA"),
    "gfx90a": ("MI200", "Aldebaran", "CDNA2"),
    "gfx940": ("MI300A", "Aqua Vanjaram", "CDNA3"),
    "gfx942": ("MI300X/MI325X", "Aqua Vanjaram", "CDNA3"),
    "gfx950": ("MI350", "Blackwell-class", "CDNA4"),
}

ARCH_FEATURES = {
    "gfx908": {"mfma": True, "agpr": True, "dpp": True, "hbm2": True, "lds_kb": 64, "vgpr_max": 256, "wave64": True},
    "gfx90a": {"mfma": True, "agpr": True, "dpp": True, "hbm2e": True, "lds_kb": 64, "vgpr_max": 512, "wave64": True, "unified_mem": True},
    "gfx942": {"mfma": True, "agpr": True, "dpp": True, "hbm3": True, "lds_kb": 64, "vgpr_max": 512, "wave64": True, "fp8": True},
    "gfx950": {"mfma": True, "agpr": True, "dpp": True, "hbm3e": True, "lds_kb": 64, "vgpr_max": 512, "wave64": True, "fp8": True, "fp4": True},
}


def make_chatml(user, assistant):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def dedup_key(user, assistant):
    return hashlib.md5((user + assistant).encode()).hexdigest()


class Phase1bAugmentor:
    def __init__(self):
        self.examples = []
        self.seen = set()
        self._load_existing_keys()

    def _load_existing_keys(self):
        for f in TRAIN_DIR.glob("*.jsonl"):
            try:
                with open(f) as fh:
                    for line in fh:
                        d = json.loads(line)
                        msgs = d.get("messages", [])
                        if len(msgs) >= 3:
                            self.seen.add(dedup_key(msgs[1]["content"], msgs[2]["content"]))
            except Exception:
                pass
        print(f"Loaded {len(self.seen)} existing dedup keys")

    def add(self, user, assistant):
        key = dedup_key(user, assistant)
        if key in self.seen:
            return False
        self.seen.add(key)
        self.examples.append(make_chatml(user, assistant))
        return True

    def gen_arch_migration(self):
        """Generate architecture migration guides."""
        print("\n=== Architecture migration Q&A ===")
        count = 0
        archs = list(ARCH_FEATURES.keys())

        for src, dst in [(a, b) for a in archs for b in archs if a != b]:
            src_info = ARCH_GPU_MAP.get(src, (src, "", ""))
            dst_info = ARCH_GPU_MAP.get(dst, (dst, "", ""))
            src_feat = ARCH_FEATURES.get(src, {})
            dst_feat = ARCH_FEATURES.get(dst, {})

            q = f"What should I change when migrating a kernel from {src} ({src_info[0]}) to {dst} ({dst_info[0]})?"
            a = f"Migration guide from {src} ({src_info[2]}) to {dst} ({dst_info[2]}):\n\n"

            changes = []
            if src_feat.get("vgpr_max") != dst_feat.get("vgpr_max"):
                changes.append(f"- VGPR limit changes: {src_feat.get('vgpr_max', 'N/A')} -> {dst_feat.get('vgpr_max', 'N/A')}. Adjust register usage for optimal occupancy.")

            if dst_feat.get("fp8") and not src_feat.get("fp8"):
                changes.append("- FP8 support available on target. Consider FP8 MFMA for 2x throughput over FP16.")

            if dst_feat.get("fp4") and not src_feat.get("fp4"):
                changes.append("- FP4 support available on target. Consider FP4 for inference-only paths.")

            if dst_feat.get("unified_mem") and not src_feat.get("unified_mem"):
                changes.append("- Unified memory architecture on target. VGPR and AGPR share the same file.")

            if dst_feat.get("hbm3") and src_feat.get("hbm2"):
                changes.append("- HBM3 on target provides higher bandwidth. May benefit from larger tile sizes.")

            if not changes:
                changes.append(f"- Both use {src_info[2]}/{dst_info[2]} architecture. Most code portable directly.")
                changes.append("- Recompile with `-march={} ` to enable architecture-specific optimizations.".format(dst))

            changes.append(f"- Always recompile with `amdclang++ -march={dst}` for optimal instruction selection.")
            a += "\n".join(changes)

            if self.add(q, a):
                count += 1

        print(f"  Added {count} migration examples")
        return count

    def gen_register_occupancy(self):
        """Generate register pressure / occupancy Q&A."""
        print("\n=== Register pressure Q&A ===")
        count = 0

        vgpr_scenarios = [
            (64, 8, "Excellent occupancy with 8 waves per SIMD. Each wave uses 64 VGPRs = 512 total out of 512 available."),
            (128, 4, "Good occupancy with 4 waves. 128 VGPRs per wave = 512 total. Still allows latency hiding."),
            (256, 2, "Reduced occupancy with only 2 waves. Memory latency harder to hide. Consider reducing register usage."),
            (512, 1, "Minimum occupancy with 1 wave. Cannot hide any memory latency. MUST reduce VGPRs or accept performance loss."),
        ]

        for vgpr, waves, desc in vgpr_scenarios:
            q = f"My gfx942 kernel uses {vgpr} VGPRs. What is the expected occupancy and is it a problem?"
            a = f"With {vgpr} VGPRs per wavefront on gfx942 (MI300X/MI325X):\n\n"
            a += f"**Occupancy**: {waves} waves per SIMD unit\n"
            a += f"**Assessment**: {desc}\n\n"
            a += "**VGPR occupancy table for gfx942**:\n"
            a += "| VGPRs | Max Waves | Occupancy |\n|-------|-----------|----------|\n"
            a += "| <= 64 | 8 | 100% |\n| <= 128 | 4 | 50% |\n| <= 256 | 2 | 25% |\n| <= 512 | 1 | 12.5% |\n"
            if self.add(q, a):
                count += 1

        # AGPR usage
        q = "What are AGPRs and when should I use them vs VGPRs on CDNA GPUs?"
        a = ("**AGPRs (Accumulator GPRs)** are special registers on CDNA architectures (gfx908+) "
             "designed for MFMA (Matrix Fused Multiply-Add) accumulation.\n\n"
             "**Key differences**:\n"
             "- VGPRs: General purpose vector registers for ALU, memory addresses, data\n"
             "- AGPRs: Accumulator registers, only writable by MFMA instructions and `v_accvgpr_write`\n\n"
             "**When to use AGPRs**:\n"
             "- MFMA output always goes to AGPRs\n"
             "- Use `v_accvgpr_read` to move results from AGPR to VGPR for further processing\n"
             "- On gfx90a+ (CDNA2+), VGPR and AGPR share the same physical register file (unified architecture)\n"
             "- On gfx908 (CDNA1), VGPR and AGPR are separate physical files\n\n"
             "**Impact on occupancy**:\n"
             "- gfx908: AGPRs don't count against VGPR occupancy limit (free extra registers!)\n"
             "- gfx90a/gfx942: AGPRs and VGPRs share the same file, so total = max(VGPR, AGPR)")
        if self.add(q, a):
            count += 1

        print(f"  Added {count} register/occupancy examples")
        return count

    def gen_mfma_deep_qa(self):
        """Generate detailed MFMA instruction Q&A."""
        print("\n=== MFMA deep Q&A ===")
        count = 0

        mfma_variants = [
            ("v_mfma_f32_32x32x8_bf16", "32x32", "8", "bf16", "f32", 256, 16, 64,
             "Primary BF16 GEMM instruction. Processes 32x32 output tile with K=8 reduction. "
             "Requires 16 accumulator registers (a[0:15]). Ideal for large matrix multiplications."),
            ("v_mfma_f32_16x16x16_f16", "16x16", "16", "f16", "f32", 256, 4, 32,
             "Smaller tile FP16 MFMA. Higher K dimension (16) means fewer outer loop iterations. "
             "Uses only 4 accumulator registers. Good for attention score computation."),
            ("v_mfma_f32_32x32x16_fp8", "32x32", "16", "fp8", "f32", 512, 16, 64,
             "FP8 MFMA with 2x compute throughput vs FP16. Available on gfx942+. "
             "Processes 2x more K elements per instruction. Key for FP8 GEMM kernels."),
            ("v_mfma_f32_4x4x4_f16", "4x4", "4", "f16", "f32", 64, 1, 4,
             "Smallest MFMA tile. Low register usage (1 AGPR). "
             "Used for small reductions or when VGPR pressure is extremely high."),
        ]

        for mnemonic, tile, k, in_dtype, out_dtype, flops, agprs, vgprs, desc in mfma_variants:
            q = f"Explain the `{mnemonic}` instruction and when to use it."
            a = (f"**`{mnemonic}`**:\n\n"
                 f"- **Tile size**: {tile} output elements\n"
                 f"- **K dimension**: {k} (reduction depth per instruction)\n"
                 f"- **Input precision**: {in_dtype}\n"
                 f"- **Output precision**: {out_dtype}\n"
                 f"- **FLOPs**: {flops} per instruction\n"
                 f"- **Accumulator registers**: {agprs} AGPRs\n"
                 f"- **Input registers**: ~{vgprs} VGPRs for operands\n\n"
                 f"{desc}\n\n"
                 f"**Typical usage in ASM**:\n"
                 f"```asm\n{mnemonic} a[0:{agprs-1}], v[0:1], v[2:3], a[0:{agprs-1}]\n"
                 f"s_waitcnt vmcnt(0)  ; ensure loads complete\n"
                 f"{mnemonic} a[{agprs}:{2*agprs-1}], v[4:5], v[6:7], a[{agprs}:{2*agprs-1}]\n```")
            if self.add(q, a):
                count += 1

        # MFMA scheduling strategies
        q = "How should I schedule MFMA instructions for maximum throughput on gfx942?"
        a = ("MFMA scheduling on gfx942 (MI300X/MI325X):\n\n"
             "**Key principle**: Interleave MFMA with memory operations to hide latency.\n\n"
             "**Good pattern** (MFMA-load interleaving):\n"
             "```asm\n"
             "v_mfma_f32_32x32x8_bf16 a[0:15], v[0:1], v[2:3], a[0:15]\n"
             "buffer_load_dwordx4 v[100:103], ...    ; issue load during MFMA\n"
             "v_mfma_f32_32x32x8_bf16 a[16:31], v[4:5], v[6:7], a[16:31]\n"
             "buffer_load_dwordx4 v[104:107], ...    ; another load\n"
             "v_mfma_f32_32x32x8_bf16 a[32:47], v[8:9], v[10:11], a[32:47]\n"
             "ds_read_b128 v[108:111], ...           ; LDS read\n"
             "```\n\n"
             "**Bad pattern** (stalled pipeline):\n"
             "```asm\n"
             "v_mfma_f32_32x32x8_bf16 a[0:15], ...  ; MFMA\n"
             "v_mfma_f32_32x32x8_bf16 a[16:31], ... ; MFMA (waiting for operands)\n"
             "v_mfma_f32_32x32x8_bf16 a[32:47], ... ; MFMA (still waiting)\n"
             "buffer_load_dwordx4 ...                 ; loads too late!\n"
             "```\n\n"
             "**Rules**:\n"
             "1. Issue 1-2 memory operations per MFMA instruction\n"
             "2. Place `s_waitcnt vmcnt(N)` to allow N outstanding loads\n"
             "3. Use software pipelining: load for iteration N+1 while computing iteration N\n"
             "4. On gfx942, MFMA latency is ~8 cycles for 32x32x8_bf16")
        if self.add(q, a):
            count += 1

        print(f"  Added {count} MFMA deep Q&A examples")
        return count

    def gen_memory_hierarchy(self):
        """Generate memory hierarchy / LDS / global memory Q&A."""
        print("\n=== Memory hierarchy Q&A ===")
        count = 0

        topics = [
            (
                "What is the memory hierarchy on AMD MI300X (gfx942)?",
                "**MI300X Memory Hierarchy**:\n\n"
                "1. **Registers** (fastest):\n"
                "   - VGPRs: 512 per wavefront, ~0 cycle access\n"
                "   - AGPRs: 512 per wavefront (unified with VGPRs on CDNA3)\n"
                "   - SGPRs: 106 per wavefront, scalar operations\n\n"
                "2. **LDS (Local Data Share)** - Shared memory:\n"
                "   - 64 KB per Compute Unit\n"
                "   - ~2-4 cycle latency\n"
                "   - 32 banks, 4 bytes per bank\n"
                "   - Bandwidth: ~12.5 TB/s aggregate\n\n"
                "3. **L1 Cache**:\n"
                "   - 32 KB per CU (texture/data cache)\n"
                "   - ~50-80 cycle latency\n\n"
                "4. **L2 Cache**:\n"
                "   - 256 MB total (shared across all CUs)\n"
                "   - ~150-200 cycle latency\n\n"
                "5. **HBM3 (Global Memory)**:\n"
                "   - 256 GB capacity\n"
                "   - ~5.3 TB/s bandwidth\n"
                "   - ~300-500 cycle latency\n\n"
                "**Optimization strategy**: Keep hot data in registers > LDS > L1 > L2 > HBM"
            ),
            (
                "How do I avoid LDS bank conflicts on AMD GPUs?",
                "**LDS Bank Conflict Avoidance on AMD GPUs**:\n\n"
                "AMD LDS has 32 banks, each 4 bytes wide. A bank conflict occurs when "
                "multiple threads in the same wavefront access different addresses in the same bank.\n\n"
                "**Bank mapping**: `bank = (address / 4) % 32`\n\n"
                "**Common conflict pattern** (BAD):\n"
                "```cpp\n"
                "__shared__ float data[256];\n"
                "float val = data[threadIdx.x * 32];  // stride-32 = all threads hit same bank!\n"
                "```\n\n"
                "**Fix 1: Padding**:\n"
                "```cpp\n"
                "__shared__ float data[256 + 8];  // pad by 8 to shift banks\n"
                "float val = data[threadIdx.x * 32];  // now distributed across banks\n"
                "```\n\n"
                "**Fix 2: Swizzle access pattern**:\n"
                "```cpp\n"
                "int idx = threadIdx.x * 32;\n"
                "int swizzled = idx ^ (threadIdx.x & 31);  // XOR-based swizzle\n"
                "float val = data[swizzled];\n"
                "```\n\n"
                "**Fix 3: Use DPP instead of LDS** for cross-lane communication:\n"
                "DPP operations are register-to-register and bypass LDS entirely.\n\n"
                "**Profiling**: Use `rocprof-compute` and check `LDS_BANK_CONFLICT` metric."
            ),
            (
                "What is the difference between `buffer_load` and `global_load` on AMDGPU?",
                "Both load from global memory but use different addressing modes:\n\n"
                "**`buffer_load_dwordx4`** (Buffer instructions):\n"
                "- Uses buffer resource descriptors (SGPRs)\n"
                "- Supports structured addressing: base + offset + stride\n"
                "- Hardware bounds checking\n"
                "- Better for regular, strided access patterns\n"
                "- Preferred by compiler for most cases\n\n"
                "**`global_load_dwordx4`** (Flat/Global instructions):\n"
                "- Uses 64-bit virtual address in VGPRs\n"
                "- Simpler addressing model\n"
                "- No bounds checking overhead\n"
                "- Better for irregular/pointer-chased access\n\n"
                "**Performance on gfx942**:\n"
                "- Both have similar throughput (~512 bytes/cycle per CU)\n"
                "- `buffer_load` can be slightly faster due to hardware address generation\n"
                "- Use `buffer_load_dwordx4` (128-bit) for vectorized loads where possible\n"
                "- Avoid `buffer_load_dword` (32-bit) -- 4x lower throughput\n\n"
                "**In practice**: The compiler chooses automatically. Use `__builtin_amdgcn_raw_buffer_load` "
                "for explicit buffer loads in performance-critical code."
            ),
            (
                "How do I optimize global memory coalescing on AMD GPUs?",
                "**Memory coalescing** groups adjacent thread memory accesses into fewer transactions.\n\n"
                "**Coalesced access** (GOOD):\n"
                "```cpp\n"
                "// Threads 0-63 load addresses 0-63 (contiguous)\n"
                "float val = input[threadIdx.x];  // 1 transaction\n"
                "```\n\n"
                "**Uncoalesced access** (BAD):\n"
                "```cpp\n"
                "// Threads 0-63 load addresses 0, 64, 128, ... (strided)\n"
                "float val = input[threadIdx.x * 64];  // 64 transactions!\n"
                "```\n\n"
                "**Rules for AMD GPUs**:\n"
                "1. **Consecutive lanes access consecutive addresses**: Thread N loads address base+N*sizeof(element)\n"
                "2. **Vectorize loads**: Use `float4` or `__hip_ds_loadx4` to load 128 bits per thread\n"
                "3. **Align to 128 bytes**: Start addresses should be 128-byte aligned for full cache line usage\n"
                "4. **Avoid strided access**: If needed, load contiguous block into LDS, then read with stride from LDS\n"
                "5. **On gfx942**: The cache line is 128 bytes. A wavefront of 64 threads loading 4 bytes each = 256 bytes = 2 cache lines (best case)\n\n"
                "**Profiling**: Check `FETCH_SIZE` and `WRITE_SIZE` metrics in rocprof-compute. "
                "If much larger than expected, you have coalescing issues."
            ),
        ]

        for q, a in topics:
            if self.add(q, a):
                count += 1

        print(f"  Added {count} memory hierarchy examples")
        return count

    def gen_kernel_workflow(self):
        """Generate kernel optimization workflow Q&A."""
        print("\n=== Kernel optimization workflow Q&A ===")
        count = 0

        workflows = [
            (
                "What is a systematic approach to optimize an AMDGPU kernel?",
                "**Systematic kernel optimization workflow**:\n\n"
                "**Step 1: Profile** (identify bottleneck)\n"
                "```bash\nrocprof-compute analyze -k <kernel_name> ./my_app\n```\n"
                "Check: ALU utilization, memory bandwidth, occupancy, LDS conflicts\n\n"
                "**Step 2: Classify bottleneck**\n"
                "- Compute-bound: MFMA utilization < 60%, ALU busy > 80%\n"
                "- Memory-bound: Bandwidth utilization > 70%, ALU busy < 50%\n"
                "- Latency-bound: Low occupancy, many stalls\n\n"
                "**Step 3: Apply targeted optimizations**\n"
                "- Compute-bound: Better instruction scheduling, MFMA interleaving\n"
                "- Memory-bound: Coalescing, vectorized loads, LDS caching, prefetch\n"
                "- Latency-bound: Increase occupancy (reduce VGPRs), software pipelining\n\n"
                "**Step 4: Examine assembly**\n"
                "```bash\namdclang++ -S -march=gfx942 -O3 kernel.cpp -o kernel.s\n```\n"
                "Check for anti-patterns: excessive s_waitcnt, unvectorized loads, LDS bank conflicts\n\n"
                "**Step 5: Iterate**\n"
                "Apply one optimization at a time. Profile after each change."
            ),
            (
                "How do I use rocprof-compute to diagnose kernel performance issues?",
                "**rocprof-compute** (formerly omniperf) is AMD's kernel profiling tool.\n\n"
                "**Basic usage**:\n"
                "```bash\n"
                "# Collect profile\n"
                "rocprof-compute profile -n my_profile -- ./my_app\n\n"
                "# Analyze results\n"
                "rocprof-compute analyze -p my_profile/\n"
                "```\n\n"
                "**Key metrics to check**:\n\n"
                "1. **GPU Speed of Light** (top-level efficiency):\n"
                "   - VALU utilization: % of peak vector ALU throughput\n"
                "   - MFMA utilization: % of peak matrix compute\n"
                "   - Memory bandwidth: % of peak HBM bandwidth\n\n"
                "2. **Compute**:\n"
                "   - `VALU_ACTIVE`: Vector ALU active cycles\n"
                "   - `MFMA_ACTIVE`: Matrix unit active cycles\n"
                "   - `SALU_ACTIVE`: Scalar ALU active cycles\n\n"
                "3. **Memory**:\n"
                "   - `FETCH_SIZE`: Bytes loaded from global memory\n"
                "   - `WRITE_SIZE`: Bytes stored to global memory\n"
                "   - `L2_HIT_RATE`: L2 cache hit rate\n\n"
                "4. **LDS**:\n"
                "   - `LDS_BANK_CONFLICT`: Bank conflict count\n"
                "   - `LDS_ADDR_CONFLICT`: Address collision count\n\n"
                "5. **Occupancy**:\n"
                "   - Achieved waves per CU\n"
                "   - Limiting factor (VGPRs, SGPRs, LDS, workgroup size)\n\n"
                "**Rule of thumb**: Fix the highest severity bottleneck first."
            ),
            (
                "How do I port a CUDA kernel to HIP for AMD GPUs?",
                "**CUDA to HIP Porting Guide**:\n\n"
                "**Automatic conversion**:\n"
                "```bash\nhipify-clang my_kernel.cu -o my_kernel.hip\n```\n\n"
                "**Key API mappings**:\n"
                "| CUDA | HIP | Notes |\n"
                "|------|-----|-------|\n"
                "| `__syncthreads()` | `__syncthreads()` | Same API |\n"
                "| `__syncwarp()` | No-op | AMD wavefronts are lockstep |\n"
                "| `__shfl_xor_sync()` | `__shfl_xor()` | Drop sync mask; consider DPP |\n"
                "| `__shfl_down_sync()` | `__shfl_down()` | Drop sync mask |\n"
                "| `atomicAdd(float)` | `atomicAdd(float)` | Same API |\n"
                "| `cub::BlockReduce` | `hipcub::BlockReduce` | Or use DPP-based `block_reduce` |\n"
                "| `__ldg()` | Direct load | AMD has no texture cache hint |\n"
                "| `__launch_bounds__` | `__launch_bounds__` | Same API |\n\n"
                "**Critical differences**:\n"
                "1. **Warp size**: CUDA=32, AMD=64 (wavefront). Affects shuffle masks and reduction depth.\n"
                "2. **Shared memory**: CUDA configurable (48-164KB), AMD fixed 64KB LDS per CU\n"
                "3. **Register file**: AMD has separate AGPR for MFMA accumulation\n"
                "4. **Barriers**: `__syncwarp()` is unnecessary on AMD (remove it)\n"
                "5. **Math intrinsics**: `__fmaf_rn` -> `__builtin_fmaf`, or just use `fmaf()`\n\n"
                "**Performance optimization** (post-port):\n"
                "- Replace `__shfl_xor` with DPP intrinsics for 3-5x faster lane shuffles\n"
                "- Replace `cub::BlockReduce` with `hip_reduce.h::block_reduce` for DPP-based reduction\n"
                "- Use MFMA intrinsics for any matrix multiply paths"
            ),
        ]

        for q, a in workflows:
            if self.add(q, a):
                count += 1

        print(f"  Added {count} workflow examples")
        return count

    def gen_isa_instruction_pairs(self):
        """Generate ISA instruction comparison and selection Q&A from DB."""
        print("\n=== ISA instruction pair Q&A ===")
        count = 0

        isa_path = DB_DIR / "isa" / "amdgpu_isa.json"
        if not isa_path.exists():
            print("  No ISA DB found")
            return 0

        with open(isa_path) as f:
            data = json.load(f)
        instructions = data if isinstance(data, list) else data.get("instructions", [])

        categories = {}
        for inst in instructions:
            cat = inst.get("category", "other")
            categories.setdefault(cat, []).append(inst)

        for cat, insts in categories.items():
            if len(insts) < 2:
                continue

            q = f"What are the available {cat} instructions on AMDGPU and their key differences?"
            a = f"**{cat} instructions on AMDGPU**:\n\n"
            for inst in insts[:20]:
                m = inst.get("mnemonic", "")
                d = inst.get("description", "")
                lat = inst.get("latency", "N/A")
                tp = inst.get("throughput", "N/A")
                a += f"- `{m}`: {d} (latency: {lat}, throughput: {tp})\n"
            if len(insts) > 20:
                a += f"\n... and {len(insts) - 20} more {cat} instructions."
            if self.add(q, a):
                count += 1

            # Pairwise comparisons within category
            for i1, i2 in list(combinations(insts[:10], 2))[:5]:
                m1, m2 = i1["mnemonic"], i2["mnemonic"]
                q = f"When should I use `{m1}` vs `{m2}`?"
                a = (f"**`{m1}`**: {i1.get('description', 'N/A')}. "
                     f"Latency: {i1.get('latency', 'N/A')}, throughput: {i1.get('throughput', 'N/A')}.\n\n"
                     f"**`{m2}`**: {i2.get('description', 'N/A')}. "
                     f"Latency: {i2.get('latency', 'N/A')}, throughput: {i2.get('throughput', 'N/A')}.\n\n")
                archs1 = set(i1.get("supported_archs", []))
                archs2 = set(i2.get("supported_archs", []))
                common = archs1 & archs2
                if common:
                    a += f"Both available on: {', '.join(sorted(common))}. "
                only1 = archs1 - archs2
                only2 = archs2 - archs1
                if only1:
                    a += f"`{m1}` also on: {', '.join(sorted(only1))}. "
                if only2:
                    a += f"`{m2}` also on: {', '.join(sorted(only2))}. "
                a += f"\nChoose based on your operation: {i1.get('description', '')} vs {i2.get('description', '')}."
                if self.add(q, a):
                    count += 1

        print(f"  Added {count} instruction pair examples")
        return count

    def gen_detailed_isa_per_arch(self):
        """Generate per-arch detailed ISA Q&A from detailed JSON files."""
        print("\n=== Detailed ISA per-arch Q&A ===")
        count = 0

        for detailed_file in sorted((DB_DIR / "isa").glob("*_detailed.json")):
            arch = detailed_file.stem.replace("_detailed", "")
            with open(detailed_file) as f:
                data = json.load(f)
            instructions = data if isinstance(data, list) else data.get("instructions", [])

            for inst in instructions:
                m = inst.get("mnemonic", "")
                desc = inst.get("description", "")
                operands = inst.get("operands", "")
                encoding = inst.get("encoding", "")
                latency = inst.get("latency", "N/A")
                throughput = inst.get("throughput", "N/A")

                if not m or not desc:
                    continue

                q = f"What is the detailed specification of `{m}` on {arch}?"
                a = f"**`{m}`** on {arch} ({ARCH_GPU_MAP.get(arch, ('', '', ''))[0]}):\n\n"
                a += f"- **Description**: {desc}\n"
                if operands:
                    a += f"- **Operands**: {operands}\n"
                if encoding:
                    a += f"- **Encoding**: {encoding}\n"
                a += f"- **Latency**: {latency} cycles\n"
                a += f"- **Throughput**: {throughput} ops/cycle\n"
                if self.add(q, a):
                    count += 1

        print(f"  Added {count} detailed ISA examples")
        return count

    def gen_common_patterns_qa(self):
        """Generate Q&A from deep ASM patterns."""
        print("\n=== Deep ASM patterns Q&A ===")
        count = 0

        deep_path = DB_DIR / "patterns" / "deep_asm_patterns.json"
        if not deep_path.exists():
            return 0

        with open(deep_path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            for section, content in data.items():
                if isinstance(content, dict):
                    q = f"What are the key statistics and patterns for `{section}` in production AMDGPU kernels?"
                    a = f"**{section}** patterns from production kernel analysis:\n\n"
                    for k, v in content.items():
                        if isinstance(v, (int, float, str)):
                            a += f"- **{k}**: {v}\n"
                        elif isinstance(v, list) and len(v) <= 10:
                            a += f"- **{k}**: {', '.join(str(x) for x in v)}\n"
                        elif isinstance(v, dict):
                            a += f"- **{k}**:\n"
                            for kk, vv in list(v.items())[:10]:
                                a += f"  - {kk}: {vv}\n"
                    if self.add(q, a):
                        count += 1

        print(f"  Added {count} deep pattern examples")
        return count

    def run(self):
        total = 0
        total += self.gen_arch_migration()
        total += self.gen_register_occupancy()
        total += self.gen_mfma_deep_qa()
        total += self.gen_memory_hierarchy()
        total += self.gen_kernel_workflow()
        total += self.gen_isa_instruction_pairs()
        total += self.gen_detailed_isa_per_arch()
        total += self.gen_common_patterns_qa()
        return total

    def save_and_merge(self):
        """Save new examples and merge with all existing data."""
        new_path = TRAIN_DIR / "augmented_phase1b.jsonl"
        print(f"\nSaving {len(self.examples)} new phase1b examples to {new_path}")
        with open(new_path, "w") as f:
            for ex in self.examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        all_examples = []
        for src in [
            TRAIN_DIR / "glm4_chatml.jsonl",
            TRAIN_DIR / "augmented_phase1.jsonl",
            new_path,
        ]:
            if src.exists():
                with open(src) as fh:
                    for line in fh:
                        all_examples.append(json.loads(line))

        random.shuffle(all_examples)
        split_point = int(len(all_examples) * 0.98)
        train = all_examples[:split_point]
        val = all_examples[split_point:]

        merged_path = TRAIN_DIR / "merged_chatml.jsonl"
        train_path = TRAIN_DIR / "train.jsonl"
        val_path = TRAIN_DIR / "val.jsonl"

        with open(merged_path, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        with open(train_path, "w") as f:
            for ex in train:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        with open(val_path, "w") as f:
            for ex in val:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"\nFinal merged dataset: {len(all_examples)} examples")
        print(f"Train: {len(train)} | Val: {len(val)}")
        return len(all_examples)


def main():
    aug = Phase1bAugmentor()
    new_count = aug.run()
    print(f"\n{'='*60}")
    print(f"Total new phase1b examples: {new_count}")
    total = aug.save_and_merge()
    print(f"Grand total: {total}")


if __name__ == "__main__":
    main()
