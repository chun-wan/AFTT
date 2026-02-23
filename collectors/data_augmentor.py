#!/usr/bin/env python3
"""AFTT Training Data Augmentor - Phase 1.

Expands training data from ~12K to ~40K+ examples by:
1. Converting C++/ASM pairs to ChatML training format
2. Generating ISA Q&A variations (cross-arch comparisons, edge cases)
3. Expanding pattern files into diverse Q&A
4. Fixing broken compiler flag examples
5. Generating cross-architecture comparison Q&A
"""

import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any
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
    "gfx900": "MI25 (Vega 10)",
    "gfx906": "MI50/MI60 (Vega 20)",
    "gfx908": "MI100 (CDNA)",
    "gfx90a": "MI200 (CDNA2)",
    "gfx940": "MI300A (CDNA3)",
    "gfx942": "MI300X/MI325X (CDNA3)",
    "gfx950": "MI350 (CDNA4)",
}


def make_chatml(user: str, assistant: str, system: str = SYSTEM_PROMPT) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def dedup_key(user: str, assistant: str) -> str:
    return hashlib.md5((user + assistant).encode()).hexdigest()


class DataAugmentor:
    def __init__(self):
        self.examples: List[dict] = []
        self.seen: set = set()
        self._load_existing()

    def _load_existing(self):
        chatml_path = TRAIN_DIR / "glm4_chatml.jsonl"
        if chatml_path.exists():
            with open(chatml_path) as f:
                for line in f:
                    d = json.loads(line)
                    msgs = d["messages"]
                    if len(msgs) >= 3:
                        key = dedup_key(msgs[1]["content"], msgs[2]["content"])
                        self.seen.add(key)
        print(f"Loaded {len(self.seen)} existing dedup keys")

    def add(self, user: str, assistant: str):
        key = dedup_key(user, assistant)
        if key in self.seen:
            return False
        self.seen.add(key)
        self.examples.append(make_chatml(user, assistant))
        return True

    # -------------------------------------------------------------------------
    # 1. C++/ASM Pairs -> Training Examples
    # -------------------------------------------------------------------------
    def augment_cpp_asm_pairs(self):
        print("\n=== Augmenting C++/ASM pairs ===")
        count = 0
        for pairs_file in sorted(DB_DIR.glob("cpp_asm_pairs/*.json")):
            with open(pairs_file) as f:
                data = json.load(f)
            pairs = data if isinstance(data, list) else data.get("pairs", [])

            for p in pairs:
                cpp = p.get("cpp_source", "")
                asm = p.get("asm_output", "")
                arch = p.get("gfx_arch", "unknown")
                template = p.get("template_name", "unknown")
                flags = p.get("compiler_flags", "")

                if not cpp or not asm:
                    continue

                asm_truncated = asm[:3000] + "\n... (truncated)" if len(asm) > 3000 else asm

                # Q1: What ASM does this C++ produce?
                q1 = f"What AMDGPU assembly does the following HIP kernel compile to on {arch}?\n\n```cpp\n{cpp}\n```"
                a1 = f"When compiled for {arch} with flags `{flags}`, the kernel produces the following assembly:\n\n```asm\n{asm_truncated}\n```\n\nKey observations:\n- Architecture: {arch} ({ARCH_GPU_MAP.get(arch, '')})\n- Template: {template}"
                if self.add(q1, a1):
                    count += 1

                # Q2: Analyze this ASM for optimization
                q2 = f"Analyze the following {arch} assembly for the `{template}` kernel and suggest optimizations:\n\n```asm\n{asm_truncated}\n```"
                patterns = []
                asm_lower = asm.lower()
                if "v_mfma" in asm_lower:
                    patterns.append("MFMA matrix multiply instructions detected")
                if "buffer_load" in asm_lower:
                    patterns.append("Global memory loads via buffer instructions")
                if "ds_read" in asm_lower or "ds_write" in asm_lower:
                    patterns.append("LDS (shared memory) operations present")
                if "s_waitcnt" in asm_lower:
                    patterns.append("Pipeline synchronization via s_waitcnt")
                if "_dpp" in asm_lower:
                    patterns.append("DPP (Data Parallel Primitives) cross-lane operations")
                if "s_barrier" in asm_lower:
                    patterns.append("Workgroup barriers for thread synchronization")

                a2 = f"Analysis of `{template}` kernel on {arch}:\n\n"
                a2 += "Detected patterns:\n" + "\n".join(f"- {p}" for p in patterns) if patterns else "No notable patterns detected in this snippet."
                a2 += f"\n\nThis is a {template} kernel compiled for {arch}. "
                a2 += "Consider checking for MFMA-load interleaving opportunities and DPP-based reductions where applicable."
                if self.add(q2, a2):
                    count += 1

                # Q3: What C++ design produces this ASM pattern?
                q3 = f"Given this {arch} assembly output, what would the original HIP/C++ kernel design look like?\n\n```asm\n{asm_truncated}\n```"
                a3 = f"Based on the assembly patterns, the original C++ kernel is likely a `{template}` implementation:\n\n```cpp\n{cpp}\n```\n\nThe assembly was generated for {arch} with compiler flags `{flags}`. Key indicators that reveal the C++ design:\n"
                if "v_mfma" in asm_lower:
                    a3 += "- MFMA instructions suggest matrix multiplication or GEMM-like operations\n"
                if "ds_read" in asm_lower:
                    a3 += "- LDS operations indicate shared memory tiling strategy\n"
                if "buffer_load" in asm_lower:
                    a3 += "- Buffer loads indicate global memory access patterns\n"
                if self.add(q3, a3):
                    count += 1

        print(f"  Added {count} C++/ASM pair examples")
        return count

    # -------------------------------------------------------------------------
    # 2. ISA Q&A Variations
    # -------------------------------------------------------------------------
    def augment_isa_qa(self):
        print("\n=== Augmenting ISA Q&A ===")
        count = 0
        isa_dir = DB_DIR / "isa"

        master_isa = {}
        master_path = isa_dir / "amdgpu_isa.json"
        if master_path.exists():
            with open(master_path) as f:
                data = json.load(f)
            instructions = data if isinstance(data, list) else data.get("instructions", [])
            for inst in instructions:
                mnemonic = inst.get("mnemonic", "")
                if mnemonic:
                    master_isa[mnemonic] = inst

        arch_instructions = {}
        for arch_file in sorted(isa_dir.glob("gfx*.json")):
            arch = arch_file.stem
            with open(arch_file) as f:
                data = json.load(f)
            insts = data if isinstance(data, list) else data.get("instructions", [])
            arch_instructions[arch] = {
                i.get("mnemonic", ""): i for i in insts if i.get("mnemonic")
            }

        for mnemonic, inst in master_isa.items():
            cat = inst.get("category", "unknown")
            desc = inst.get("description", "No description")
            operands = inst.get("operands", "N/A")
            latency = inst.get("latency", "N/A")
            throughput = inst.get("throughput", "N/A")
            archs = inst.get("supported_archs", [])

            # Cross-arch comparison
            arch_pairs = list(combinations(archs[:5], 2))
            for a1, a2 in arch_pairs[:2]:
                q = f"Compare the behavior of `{mnemonic}` on {a1} vs {a2}."
                gpu1 = ARCH_GPU_MAP.get(a1, a1)
                gpu2 = ARCH_GPU_MAP.get(a2, a2)
                a = (
                    f"`{mnemonic}` is a {cat} instruction available on both {a1} ({gpu1}) and {a2} ({gpu2}). "
                    f"Description: {desc}. "
                    f"On both architectures: latency ~{latency} cycles, throughput ~{throughput} ops/cycle. "
                )
                i1 = arch_instructions.get(a1, {}).get(mnemonic, {})
                i2 = arch_instructions.get(a2, {}).get(mnemonic, {})
                lat1 = i1.get("latency", latency)
                lat2 = i2.get("latency", latency)
                if lat1 != lat2:
                    a += f"Note: latency differs -- {a1}: {lat1} cycles vs {a2}: {lat2} cycles. "
                else:
                    a += f"The instruction has the same latency ({lat1} cycles) on both architectures. "
                a += f"Operands: {operands}."
                if self.add(q, a):
                    count += 1

            # "When should I use X" questions
            use_q = f"When should I use `{mnemonic}` in an AMDGPU kernel?"
            use_a = (
                f"`{mnemonic}` is a {cat} instruction: {desc}. "
                f"Operands: {operands}. Latency: {latency} cycles, throughput: {throughput} ops/cycle. "
                f"Supported on: {', '.join(archs)}. "
            )
            if "mfma" in mnemonic.lower():
                use_a += "Use this for matrix multiply-accumulate in GEMM/attention kernels. "
            elif "dpp" in mnemonic.lower() or "permute" in mnemonic.lower():
                use_a += "Use this for cross-lane data movement in reductions and shuffles. "
            elif mnemonic.startswith("s_"):
                use_a += "This is a scalar instruction -- use for control flow, address computation, or loop counters. "
            elif mnemonic.startswith("v_"):
                use_a += "This is a vector instruction -- operates across all lanes in a wavefront. "
            elif mnemonic.startswith("buffer_") or mnemonic.startswith("global_"):
                use_a += "Use for global memory access. Ensure coalesced access patterns for best performance. "
            elif mnemonic.startswith("ds_"):
                use_a += "This is an LDS (shared memory) instruction. Minimize bank conflicts for peak bandwidth. "
            if self.add(use_q, use_a):
                count += 1

            # "What is the latency of X" questions
            lat_q = f"What is the latency and throughput of `{mnemonic}` on {archs[0] if archs else 'gfx942'}?"
            lat_a = (
                f"`{mnemonic}` ({cat}): latency = {latency} cycles, throughput = {throughput} ops/cycle "
                f"on {archs[0] if archs else 'gfx942'}. {desc}"
            )
            if self.add(lat_q, lat_a):
                count += 1

        # Architecture-exclusive instructions
        all_archs = sorted(arch_instructions.keys())
        for i, arch1 in enumerate(all_archs):
            for arch2 in all_archs[i + 1:]:
                mnemonics1 = set(arch_instructions[arch1].keys())
                mnemonics2 = set(arch_instructions[arch2].keys())
                only1 = mnemonics1 - mnemonics2
                only2 = mnemonics2 - mnemonics1

                if only1 or only2:
                    q = f"What ISA instructions are available on {arch1} but not {arch2}, and vice versa?"
                    a = f"Instructions exclusive to {arch1} ({ARCH_GPU_MAP.get(arch1, '')}):\n"
                    for m in sorted(list(only1))[:15]:
                        inst = arch_instructions[arch1][m]
                        a += f"- `{m}`: {inst.get('description', 'N/A')}\n"
                    if len(only1) > 15:
                        a += f"  ... and {len(only1) - 15} more\n"
                    a += f"\nInstructions exclusive to {arch2} ({ARCH_GPU_MAP.get(arch2, '')}):\n"
                    for m in sorted(list(only2))[:15]:
                        inst = arch_instructions[arch2][m]
                        a += f"- `{m}`: {inst.get('description', 'N/A')}\n"
                    if len(only2) > 15:
                        a += f"  ... and {len(only2) - 15} more\n"
                    if self.add(q, a):
                        count += 1

        print(f"  Added {count} ISA Q&A examples")
        return count

    # -------------------------------------------------------------------------
    # 3. Pattern-based Q&A
    # -------------------------------------------------------------------------
    def augment_patterns(self):
        print("\n=== Augmenting pattern Q&A ===")
        count = 0

        # Anti-patterns
        anti_path = DB_DIR / "patterns" / "anti_patterns.json"
        if anti_path.exists():
            with open(anti_path) as f:
                data = json.load(f)
            patterns = data if isinstance(data, list) else data.get("patterns", data.get("anti_patterns", []))
            for p in patterns:
                pid = p.get("pattern_id", p.get("id", "unknown"))
                desc = p.get("description", "")
                suggestion = p.get("suggestion", p.get("fix", ""))
                bad = p.get("example_bad", p.get("bad_example", ""))
                good = p.get("example_good", p.get("good_example", ""))

                if not desc:
                    continue

                # Diagnosis question
                q = f"I see this pattern in my AMDGPU kernel:\n```\n{bad}\n```\nIs this a performance problem?"
                a = f"Yes, this is the **{pid}** anti-pattern: {desc}\n\n"
                a += f"**Problem**: {desc}\n"
                a += f"**Fix**: {suggestion}\n"
                if good:
                    a += f"\n**Optimized version**:\n```\n{good}\n```"
                if self.add(q, a):
                    count += 1

                # How to fix question
                q2 = f"How do I fix the `{pid}` performance issue in an AMD GPU kernel?"
                a2 = f"The `{pid}` issue: {desc}\n\n**Solution**: {suggestion}\n"
                if bad and good:
                    a2 += f"\n**Before** (slow):\n```\n{bad}\n```\n\n**After** (fast):\n```\n{good}\n```"
                if self.add(q2, a2):
                    count += 1

        # Best practices
        bp_path = DB_DIR / "patterns" / "best_practices.json"
        if bp_path.exists():
            with open(bp_path) as f:
                data = json.load(f)
            practices = data if isinstance(data, list) else data.get("practices", data.get("best_practices", []))
            for p in practices:
                name = p.get("name", p.get("pattern_id", "unknown"))
                desc = p.get("description", "")
                example = p.get("example", p.get("code_example", ""))

                if not desc:
                    continue

                q = f"What is the `{name}` optimization technique for AMDGPU kernels?"
                a = f"**{name}**: {desc}\n"
                if example:
                    a += f"\n**Example**:\n```\n{example}\n```"
                if self.add(q, a):
                    count += 1

                q2 = f"How do I implement {name} in my HIP/AMDGPU kernel?"
                a2 = f"To implement {name}:\n\n{desc}\n"
                if example:
                    a2 += f"\n**Implementation example**:\n```\n{example}\n```"
                if self.add(q2, a2):
                    count += 1

        # DPP/crosslane patterns
        dpp_path = DB_DIR / "patterns" / "dpp_crosslane_patterns.json"
        if dpp_path.exists():
            with open(dpp_path) as f:
                data = json.load(f)
            patterns = data if isinstance(data, list) else data.get("patterns", [])
            for p in patterns:
                modifier = p.get("modifier", "")
                desc = p.get("description", "")
                use = p.get("primary_use", "")
                why = p.get("why_fast", "")
                pat = p.get("pattern", {})

                if not modifier or not desc:
                    continue

                q = f"How does DPP `{modifier}` work and when should I use it?"
                a = f"**DPP `{modifier}`**: {desc}\n\n**Primary use**: {use}\n**Why it's fast**: {why}\n"
                if pat.get("explanation"):
                    a += f"\n**Details**: {pat['explanation']}\n"
                if pat.get("typical_sequence"):
                    a += f"\n**Typical ASM sequence**:\n```asm\n{pat['typical_sequence']}\n```"
                if self.add(q, a):
                    count += 1

        # CK deep patterns
        ck_path = DB_DIR / "patterns" / "ck_deep_patterns.json"
        if ck_path.exists():
            with open(ck_path) as f:
                data = json.load(f)
            patterns = data if isinstance(data, list) else data.get("patterns", data.get("pipeline_patterns", []))
            for p in patterns:
                name = p.get("name", p.get("pattern_name", "unknown"))
                desc = p.get("description", "")

                if not desc:
                    continue

                q = f"Explain the CK (Composable Kernel) optimization pattern: {name}"
                a = f"**{name}** in Composable Kernel:\n\n{desc}"
                if self.add(q, a):
                    count += 1

        # FMHA ASM patterns
        fmha_path = DB_DIR / "patterns" / "fmha_asm_patterns.json"
        if fmha_path.exists():
            with open(fmha_path) as f:
                data = json.load(f)
            for key, val in data.items():
                if isinstance(val, dict):
                    desc = val.get("description", json.dumps(val)[:500])
                    q = f"What is the `{key}` optimization pattern in FMHA (Flash Attention) ASM kernels?"
                    a = f"**{key}** in FMHA ASM kernels:\n\n{desc}"
                    if self.add(q, a):
                        count += 1

        # TRT-LLM algorithm mapping
        trtllm_path = DB_DIR / "patterns" / "trtllm_algorithms.json"
        if trtllm_path.exists():
            with open(trtllm_path) as f:
                data = json.load(f)
            algos = data if isinstance(data, list) else data.get("algorithms", [])
            for algo in algos:
                name = algo.get("name", "unknown")
                desc = algo.get("description", "")
                cuda = algo.get("cuda_implementation", algo.get("nvidia_approach", ""))
                amd = algo.get("amd_equivalent", algo.get("amd_implementation", ""))

                if not desc:
                    continue

                q = f"How is the `{name}` algorithm implemented differently on NVIDIA vs AMD GPUs?"
                a = f"**{name}**: {desc}\n\n"
                if cuda:
                    a += f"**NVIDIA/CUDA approach**: {cuda}\n"
                if amd:
                    a += f"**AMD/HIP equivalent**: {amd}\n"
                a += "\nWhen porting from NVIDIA to AMD, ensure you adapt warp size (32->64), use DPP instead of __shfl, and leverage MFMA for matrix operations."
                if self.add(q, a):
                    count += 1

        trtllm_map_path = DB_DIR / "patterns" / "trtllm_amd_mapping.json"
        if trtllm_map_path.exists():
            with open(trtllm_map_path) as f:
                data = json.load(f)
            mappings = data if isinstance(data, list) else data.get("mappings", [])
            for m in mappings:
                nvidia = m.get("nvidia_kernel", m.get("cuda_kernel", "unknown"))
                amd = m.get("amd_kernel", m.get("hip_kernel", "unknown"))
                notes = m.get("notes", m.get("description", ""))

                q = f"What is the AMD equivalent of the NVIDIA `{nvidia}` kernel?"
                a = f"The AMD equivalent of NVIDIA's `{nvidia}` is `{amd}`. {notes}"
                if self.add(q, a):
                    count += 1

        print(f"  Added {count} pattern Q&A examples")
        return count

    # -------------------------------------------------------------------------
    # 4. Profiling Rules Q&A
    # -------------------------------------------------------------------------
    def augment_profiling(self):
        print("\n=== Augmenting profiling Q&A ===")
        count = 0

        rules_path = DB_DIR / "profiling_rules" / "profiling_rules.json"
        if rules_path.exists():
            with open(rules_path) as f:
                data = json.load(f)
            rules = data if isinstance(data, list) else data.get("rules", [])
            for r in rules:
                name = r.get("name", r.get("rule_id", "unknown"))
                desc = r.get("description", "")
                metric = r.get("metric", "")
                threshold = r.get("threshold", "")
                suggestion = r.get("suggestion", r.get("fix", ""))

                if not desc:
                    continue

                q = f"What does `{name}` mean in rocprof-compute profiling output?"
                a = f"**{name}**: {desc}\n"
                if metric:
                    a += f"**Metric**: {metric}\n"
                if threshold:
                    a += f"**Threshold**: {threshold}\n"
                if suggestion:
                    a += f"**Recommended action**: {suggestion}"
                if self.add(q, a):
                    count += 1

                q2 = f"How do I diagnose and fix `{name}` performance issues on AMD GPUs?"
                a2 = f"When profiling shows {name} issues ({desc}):\n\n"
                if metric:
                    a2 += f"1. Check the `{metric}` metric in rocprof-compute output\n"
                if threshold:
                    a2 += f"2. Values beyond `{threshold}` indicate a problem\n"
                if suggestion:
                    a2 += f"3. Fix: {suggestion}\n"
                if self.add(q2, a2):
                    count += 1

        # Arch specs Q&A
        specs_path = DB_DIR / "profiling_rules" / "arch_specs.json"
        if specs_path.exists():
            with open(specs_path) as f:
                data = json.load(f)
            specs = data if isinstance(data, dict) else {}
            for arch, spec in specs.items():
                if not isinstance(spec, dict):
                    continue

                q = f"What are the key hardware specifications of {arch} for kernel optimization?"
                a = f"**{arch}** ({ARCH_GPU_MAP.get(arch, '')}) specifications:\n\n"
                for k, v in spec.items():
                    a += f"- **{k}**: {v}\n"
                if self.add(q, a):
                    count += 1

        print(f"  Added {count} profiling Q&A examples")
        return count

    # -------------------------------------------------------------------------
    # 5. Compiler Flag Q&A (fix broken examples)
    # -------------------------------------------------------------------------
    def augment_compiler_flags(self):
        print("\n=== Augmenting compiler flag Q&A ===")
        count = 0

        flags_path = DB_DIR / "compiler_flags" / "flag_effects.json"
        if flags_path.exists():
            with open(flags_path) as f:
                data = json.load(f)
            effects = data if isinstance(data, list) else data.get("effects", data.get("flag_effects", []))
            if isinstance(data, dict) and not effects:
                effects = []
                for flag, info in data.items():
                    if isinstance(info, dict):
                        info["flag"] = flag
                        effects.append(info)

            for e in effects:
                flag = e.get("flag", e.get("name", "unknown"))
                desc = e.get("description", e.get("effect", ""))
                impact = e.get("impact", e.get("performance_impact", ""))

                if not desc and not impact:
                    continue

                q = f"What does the compiler flag `{flag}` do when compiling AMDGPU kernels with amdclang++?"
                a = f"**`{flag}`**: {desc}\n"
                if impact:
                    a += f"\n**Performance impact**: {impact}"
                if self.add(q, a):
                    count += 1

        common_flags = [
            ("-O0", "No optimization", "Fastest compile, largest/slowest code. Good for debugging."),
            ("-O1", "Basic optimization", "Reduces code size and execution time with minimal compile overhead."),
            ("-O2", "Standard optimization", "Good balance of compile time and runtime performance. Default for production."),
            ("-O3", "Aggressive optimization", "Enables vectorization, loop unrolling, and function inlining. May increase code size."),
            ("-Ofast", "Maximum speed", "Like -O3 but allows non-IEEE-compliant math optimizations. Use with caution for precision-sensitive kernels."),
            ("-ffast-math", "Fast math", "Allows algebraic simplifications that may change results. Enables FMA contraction, reciprocal approximations."),
            ("-funroll-loops", "Loop unrolling", "Unrolls loops to reduce branch overhead. Increases register pressure."),
            ("-march=gfx942", "Target architecture", "Generates code specifically optimized for gfx942 (MI300X/MI325X). Enables MFMA, DPP, and architecture-specific instructions."),
            ("-mcumode", "CU mode", "Generates code for Compute Unit mode instead of WGP mode. Affects barrier and LDS behavior."),
            ("-mno-wavefrontsize64", "Wavefront size 32", "Forces wavefront size 32 (NVIDIA-like). Default on CDNA is wavefront 64."),
        ]
        for flag, name, desc in common_flags:
            q = f"What does `{flag}` do for amdclang++ and how does it affect AMDGPU assembly output?"
            a = f"**`{flag}`** ({name}): {desc}"
            if self.add(q, a):
                count += 1

        # Flag comparison Q&A
        comparisons = [
            ("-O2", "-O3", "O3 enables more aggressive loop unrolling, vectorization, and function inlining. May increase register pressure and code size but typically 5-15% faster for compute-bound kernels."),
            ("-O3", "-Ofast", "Ofast adds non-IEEE math optimizations on top of O3. Can change numerical results but allows more FMA contractions and reciprocal approximations."),
            ("-O0", "-O3", "O3 applies extensive optimizations: dead code elimination, register allocation, instruction scheduling, vectorization. Typical speedup: 10-100x over O0."),
        ]
        for f1, f2, diff in comparisons:
            for arch in ["gfx942", "gfx90a", "gfx908"]:
                q = f"How does `{f1}` vs `{f2}` affect assembly output for AMDGPU kernels on {arch}?"
                a = f"Comparing `{f1}` vs `{f2}` on {arch} ({ARCH_GPU_MAP.get(arch, '')}):\n\n{diff}\n\nOn {arch} specifically, the instruction scheduling and register allocation will be architecture-aware, leveraging {arch}-specific MFMA instructions and memory hierarchy."
                if self.add(q, a):
                    count += 1

        print(f"  Added {count} compiler flag Q&A examples")
        return count

    # -------------------------------------------------------------------------
    # 6. Multi-turn Chain-of-Thought examples
    # -------------------------------------------------------------------------
    def augment_reasoning_chains(self):
        print("\n=== Generating reasoning chain examples ===")
        count = 0

        scenarios = [
            {
                "user": "My RMSNorm kernel on MI300X is slower than expected. The profiler shows high LDS bank conflict rate. Here's my reduction code:\n```cpp\n__shared__ float shared[1024];\nshared[threadIdx.x] = val;\n__syncthreads();\nfor (int s = blockDim.x/2; s > 0; s >>= 1) {\n    if (threadIdx.x < s)\n        shared[threadIdx.x] += shared[threadIdx.x + s];\n    __syncthreads();\n}\n```",
                "assistant": "The issue is your tree reduction using LDS (shared memory) with `__syncthreads()`. This is a classic anti-pattern on AMD GPUs.\n\n**Problem Analysis**:\n1. **LDS bank conflicts**: Adjacent threads accessing `shared[threadIdx.x]` and `shared[threadIdx.x + s]` creates 2-way bank conflicts\n2. **Excessive barriers**: Each reduction step requires `__syncthreads()`, which is expensive (costs ~100+ cycles per barrier)\n3. **Wasted parallelism**: After each step, half the threads become idle\n\n**Solution**: Replace LDS-based reduction with DPP-based wavefront reduction:\n\n```cpp\n#include \"hip_reduce.h\"\nauto sum_f = [](float a, float b) { return a + b; };\nfloat result = block_reduce<float, decltype(sum_f), 1024, true>(val, sum_f);\n```\n\n**Why this is faster**:\n- DPP uses register-to-register cross-lane data movement (no LDS needed)\n- `row_shr` operations complete in 1 cycle vs 2-4 cycles for LDS read+write\n- Eliminates all `__syncthreads()` within a wavefront\n- Reduces total cycles from ~800+ to ~60 for a 1024-thread block\n\nExpected speedup: 2-3x for the reduction portion of your RMSNorm kernel."
            },
            {
                "user": "I'm porting a CUDA GEMM kernel from A100 to MI300X. The CUDA version uses `__shfl_xor_sync` for warp-level reductions with warpSize=32. How should I adapt this for AMD?",
                "assistant": "Key differences when porting CUDA warp operations to AMD:\n\n**1. Wavefront size**: AMD CDNA uses wavefront 64 (not 32). Your `__shfl_xor_sync` with width=32 maps to subwave operations.\n\n**2. Replace `__shfl_xor_sync`**: The CUDA shuffle intrinsic compiles to `ds_bpermute_b32` on AMD, which goes through LDS and is slower.\n\n**Better approach** - use DPP intrinsics:\n```cpp\n// CUDA (slow on AMD):\nfor (int mask = 16; mask > 0; mask >>= 1)\n    val += __shfl_xor_sync(0xffffffff, val, mask);\n\n// AMD optimized (DPP-based):\n#include \"hip_reduce.h\"\nauto sum_f = [](float a, float b) { return a + b; };\nfloat result = wave_reduce<float, decltype(sum_f), true>(val, sum_f);\n```\n\n**3. For GEMM specifically**: On MI300X, use `v_mfma_f32_32x32x8_bf16` (or fp16 variant) instead of manual warp-level GEMM tiles. The MFMA unit provides 256 FP16 FLOPs per instruction.\n\n**4. Synchronization**: `__syncwarp()` is a no-op on AMD (wavefronts are lock-step). Remove it. Use `s_barrier` only for workgroup-level sync.\n\n**5. Shared memory**: AMD LDS is 64KB per CU (vs 48-164KB configurable on A100). Adjust tile sizes accordingly."
            },
            {
                "user": "How do I check if my FMHA kernel on gfx942 has good MFMA utilization? What should I look for in the assembly?",
                "assistant": "To evaluate MFMA utilization in your FMHA kernel assembly:\n\n**1. Count MFMA instructions**:\n```bash\ngrep -c 'v_mfma' kernel.asm\n```\nA well-optimized FMHA forward (head_dim=128) should have ~150-200 MFMA instructions. Backward: ~400-500.\n\n**2. Check MFMA-load interleaving**:\nLook for this pattern (good):\n```asm\nv_mfma_f32_32x32x8_bf16 a[0:15], ...\nbuffer_load_dwordx4 v[100:103], ...    ; load interleaved with MFMA\nv_mfma_f32_32x32x8_bf16 a[16:31], ...\nds_read_b128 v[104:107], ...           ; LDS read interleaved\nv_mfma_f32_32x32x8_bf16 a[32:47], ...\n```\nBad pattern: multiple consecutive MFMAs without loads (memory pipeline starved).\n\n**3. Check AGPR usage** (accumulator registers):\nFMHA should use AGPRs (`a[0:N]`) for accumulation. Look for `a[0:255]` range.\n\n**4. Barrier count**:\n```bash\ngrep -c 's_barrier' kernel.asm\n```\nForward: ~30 barriers is typical. Backward: ~50. Excessive barriers (>80) suggest suboptimal pipelining.\n\n**5. DPP for softmax**:\nBackward FMHA should use `quad_perm` for softmax gradient computation:\n```asm\nv_mov_b32_dpp v127, v124 quad_perm:[3,3,3,3]\n```\n\n**6. Profiling metrics** (rocprof-compute):\n- MFMA utilization > 60% is good, > 80% is excellent\n- Memory bandwidth utilization > 50% of peak\n- Occupancy limited by VGPR/AGPR count"
            },
            {
                "user": "What's the difference between software pipelining with double buffering vs triple buffering in CK kernels?",
                "assistant": "**Software pipelining** overlaps memory operations with compute to hide latency. The buffer depth determines how many stages run concurrently.\n\n**Double Buffering (2 stages)**:\n```\nLoad[i+1]  |  Compute[i]  |  Store[i-1]\n```\n- 2 sets of buffers (registers/LDS)\n- Simpler, lower register pressure\n- Hides latency if load_time <= compute_time\n- In CK ASM, you'll see 2 `buffer_load` groups interleaved with MFMA\n\n**Triple Buffering (3 stages)**:\n```\nLoad[i+2]  |  Compute[i+1]  |  Store[i]\n```\n- 3 sets of buffers\n- Higher register pressure (~50% more VGPRs)\n- Hides latency even when load_time > compute_time\n- In CK, used for kernels where global memory latency is the bottleneck\n\n**CK implementation pattern (ASM)**:\n```asm\n; === Triple buffer: Prefetch depth 3 ===\n; Buffer A: v[0:63]\n; Buffer B: v[64:127]\n; Buffer C: v[128:191]\n\n; Compute on A while loading into C\nv_mfma_f32_32x32x8_bf16 a[0:15], v[0:3], v[4:7]   ; compute A\nbuffer_load_dwordx4 v[128:131], ...                  ; load into C\nv_mfma_f32_32x32x8_bf16 a[16:31], v[8:11], v[12:15]\nbuffer_load_dwordx4 v[132:135], ...\n; ... rotate: next iteration computes B, loads into A\n```\n\n**Trade-offs**:\n| Aspect | Double Buffer | Triple Buffer |\n|--------|--------------|---------------|\n| VGPRs | Lower | ~50% higher |\n| Occupancy | Higher | Lower |\n| Latency hiding | Good | Better |\n| Complexity | Simpler | More complex |\n\n**Rule of thumb**: Use double buffering when VGPR pressure is high (occupancy-bound). Use triple buffering when memory latency is the bottleneck (latency-bound)."
            },
        ]

        for s in scenarios:
            if self.add(s["user"], s["assistant"]):
                count += 1

        print(f"  Added {count} reasoning chain examples")
        return count

    # -------------------------------------------------------------------------
    # Run all augmentations
    # -------------------------------------------------------------------------
    def run(self) -> int:
        total = 0
        total += self.augment_cpp_asm_pairs()
        total += self.augment_isa_qa()
        total += self.augment_patterns()
        total += self.augment_profiling()
        total += self.augment_compiler_flags()
        total += self.augment_reasoning_chains()
        return total

    def save(self, output_path: Path):
        print(f"\nSaving {len(self.examples)} new examples to {output_path}")
        with open(output_path, "w") as f:
            for ex in self.examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def merge_and_save(self, output_path: Path):
        """Merge new examples with existing training data."""
        existing = []
        chatml_path = TRAIN_DIR / "glm4_chatml.jsonl"
        if chatml_path.exists():
            with open(chatml_path) as f:
                for line in f:
                    existing.append(json.loads(line))

        all_examples = existing + self.examples
        random.shuffle(all_examples)

        print(f"\nMerging: {len(existing)} existing + {len(self.examples)} new = {len(all_examples)} total")
        print(f"Saving merged dataset to {output_path}")
        with open(output_path, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        # Also save the split
        split_point = int(len(all_examples) * 0.98)
        train = all_examples[:split_point]
        val = all_examples[split_point:]

        train_path = output_path.parent / "train.jsonl"
        val_path = output_path.parent / "val.jsonl"

        with open(train_path, "w") as f:
            for ex in train:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        with open(val_path, "w") as f:
            for ex in val:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"Train split: {len(train)} examples -> {train_path}")
        print(f"Val split: {len(val)} examples -> {val_path}")

        return len(all_examples)


def main():
    aug = DataAugmentor()
    new_count = aug.run()
    print(f"\n{'='*60}")
    print(f"Total new examples generated: {new_count}")

    aug.save(TRAIN_DIR / "augmented_phase1.jsonl")
    total = aug.merge_and_save(TRAIN_DIR / "merged_chatml.jsonl")
    print(f"Final merged dataset: {total} examples")


if __name__ == "__main__":
    main()
