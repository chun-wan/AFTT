"""Algorithm recognition from HIP C++ source and AMDGPU ASM instruction features.

Provides classification of GPU kernels into algorithm types (GEMM, FMHA, norms,
softmax, etc.) using both static analysis of HIP source and pattern matching
on disassembled instruction sequences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from .instruction import Instruction


# ── Algorithm Types ────────────────────────────────────────────────────────


class AlgorithmType(str, Enum):
    """Recognized GPU kernel algorithm types."""
    GEMM = "GEMM"
    FMHA = "FMHA"
    LAYERNORM = "LAYERNORM"
    RMSNORM = "RMSNORM"
    SOFTMAX = "SOFTMAX"
    TOPK = "TOPK"
    MOE = "MOE"
    REDUCTION = "REDUCTION"
    TRANSPOSE = "TRANSPOSE"
    ELEMENTWISE = "ELEMENTWISE"
    CUSTOM = "CUSTOM"


# ── Result Dataclasses ──────────────────────────────────────────────────────


@dataclass
class AlgorithmInfo:
    """Result of algorithm classification."""
    algo_type: str
    confidence: float  # 0.0-1.0
    parameters: dict
    features: dict
    sub_type: str = ""


@dataclass
class ASMFeatureFingerprint:
    """Feature fingerprint extracted from AMDGPU instruction sequence."""
    mfma_count: int = 0
    mfma_types: dict[str, int] = field(default_factory=dict)
    dpp_count: int = 0
    dpp_modifiers: dict[str, int] = field(default_factory=dict)
    lds_reads: int = 0
    lds_writes: int = 0
    vmem_loads: int = 0
    vmem_stores: int = 0
    barrier_count: int = 0
    waitcnt_count: int = 0
    nop_count: int = 0
    valu_count: int = 0
    salu_count: int = 0
    max_vgpr: int = 0
    max_sgpr: int = 0
    max_agpr: int = 0
    total_instructions: int = 0
    vectorization_ratio: float = 0.0
    branch_count: int = 0
    has_direct_lds_loads: bool = False
    bpermute_count: int = 0


# ── Algorithm Classifier ────────────────────────────────────────────────────


class AlgorithmClassifier:
    """Classifies GPU kernels from HIP source or ASM instructions."""

    def __init__(self) -> None:
        pass

    def classify_from_hip(self, source: str) -> AlgorithmInfo:
        """Parse HIP C++ source to identify algorithm type via regex patterns."""
        source_lower = source.lower()
        source_upper = source
        features: dict = {}
        sub_type = ""
        algo_type = AlgorithmType.CUSTOM.value
        confidence = 0.3

        # ── Kernel function signatures ──
        func_names = []
        for m in re.finditer(
            r"__global__\s+(?:void|extern\s+\w+\s+)?(\w+)\s*\(",
            source,
        ):
            func_names.append(m.group(1))
        features["kernel_functions"] = func_names

        # ── Parameter types ──
        has_float = "float*" in source or "const float*" in source
        has_half = "half*" in source or "__half*" in source or "const half*" in source
        has_bf16 = "__nv_bfloat16" in source or "hip_bfloat16" in source or "bfloat16" in source_lower
        has_fp8 = "fp8" in source_lower or "__fp8" in source
        dtype = "unknown"
        if has_fp8:
            dtype = "fp8"
        elif has_bf16:
            dtype = "bf16"
        elif has_half:
            dtype = "fp16"
        elif has_float:
            dtype = "float"
        features["dtype_hints"] = {
            "float": has_float,
            "half": has_half,
            "bf16": has_bf16,
            "fp8": has_fp8,
        }
        features["inferred_dtype"] = dtype

        # ── Shared memory ──
        shared_matches = re.findall(r"__shared__\s+(?:\w+\s+)+(\w+)\s*\[([^\]]+)\]", source)
        features["shared_mem"] = [(n, s) for n, s in shared_matches] if shared_matches else []

        # ── Function name hints ──
        all_text = " ".join(func_names) + " " + source_lower
        name_hints = {
            "gemm": ("gemm" in all_text or "matmul" in all_text or "matrix_mul" in all_text),
            "layernorm": ("layernorm" in all_text or "layer_norm" in all_text),
            "rmsnorm": ("rmsnorm" in all_text or "rms_norm" in all_text),
            "softmax": ("softmax" in all_text or "soft_max" in all_text),
            "attention": ("attention" in all_text or "fmha" in all_text or "flash_attn" in all_text),
            "topk": ("topk" in all_text or "top_k" in all_text),
            "moe": ("moe" in all_text or "expert" in all_text or "gate" in all_text),
            "reduce": ("reduce" in all_text or "reduction" in all_text),
            "transpose": ("transpose" in all_text or "matrix_transpose" in all_text),
        }
        features["name_hints"] = name_hints

        # ── Computation patterns ──
        has_k_loop = bool(re.search(r"for\s*\([^)]*[kK]\s*[<>=]", source))
        has_macc = bool(re.search(r"\*\s*\w+\s*\+\s*\w+|\+\s*\w+\s*\*", source))
        has_syncthreads = "__syncthreads" in source or "sync" in source_lower
        has_reduce_sum = "sum" in source_lower and ("+=" in source or "reduce" in source_lower)
        has_reduce_max = "max" in source_lower and ("fmaxf" in source or "max(" in source)
        has_qkv = ("q_ptr" in source_lower or "qkv" in source_lower or
                  "query" in source_lower and "key" in source_lower)
        has_expert_gate = ("expert" in source_lower and "gate" in source_lower) or "routing" in source_lower
        has_tiled_access = "tile" in source_lower or "TILE" in source or "blockIdx" in source
        features["computation"] = {
            "k_loop": has_k_loop,
            "macc": has_macc,
            "syncthreads": has_syncthreads,
            "reduce_sum": has_reduce_sum,
            "reduce_max": has_reduce_max,
            "qkv": has_qkv,
            "expert_gate": has_expert_gate,
            "tiled": has_tiled_access,
        }

        # ── Classification logic ──
        if name_hints["gemm"] or (has_k_loop and has_macc and not has_qkv):
            algo_type = AlgorithmType.GEMM.value
            confidence = 0.85 if name_hints["gemm"] else 0.65
            sub_type = f"gemm_{dtype}" if dtype != "unknown" else "gemm"

        elif name_hints["attention"] or (has_qkv and has_syncthreads):
            algo_type = AlgorithmType.FMHA.value
            confidence = 0.9 if name_hints["attention"] else 0.7
            sub_type = "fmha_forward" if "backward" not in all_text else "fmha_backward"

        elif name_hints["rmsnorm"] or ("rms" in all_text and has_reduce_sum):
            algo_type = AlgorithmType.RMSNORM.value
            confidence = 0.9 if name_hints["rmsnorm"] else 0.6
            sub_type = "rmsnorm"

        elif name_hints["layernorm"]:
            algo_type = AlgorithmType.LAYERNORM.value
            confidence = 0.9
            sub_type = "layernorm"

        elif name_hints["softmax"] or (has_reduce_max and has_reduce_sum):
            algo_type = AlgorithmType.SOFTMAX.value
            confidence = 0.85 if name_hints["softmax"] else 0.6
            sub_type = "softmax"

        elif name_hints["topk"]:
            algo_type = AlgorithmType.TOPK.value
            confidence = 0.8
            sub_type = "topk"

        elif name_hints["moe"] or has_expert_gate:
            algo_type = AlgorithmType.MOE.value
            confidence = 0.85 if name_hints["moe"] else 0.6
            sub_type = "moe"

        elif name_hints["reduce"] or (has_syncthreads and has_reduce_sum and not has_macc):
            algo_type = AlgorithmType.REDUCTION.value
            confidence = 0.8 if name_hints["reduce"] else 0.55
            sub_type = "reduction"

        elif name_hints["transpose"] or (has_tiled_access and "transpose" in all_text):
            algo_type = AlgorithmType.TRANSPOSE.value
            confidence = 0.85 if name_hints["transpose"] else 0.6
            sub_type = "transpose"

        elif not has_macc and not has_syncthreads and "add" in source_lower:
            algo_type = AlgorithmType.ELEMENTWISE.value
            confidence = 0.6
            sub_type = "elementwise"

        parameters = self.extract_parameters(source, algo_type)
        features["classification_input"] = {
            "name_hints": name_hints,
            "computation": features["computation"],
            "dtype": dtype,
        }

        return AlgorithmInfo(
            algo_type=algo_type,
            confidence=min(1.0, confidence),
            parameters=parameters,
            features=features,
            sub_type=sub_type,
        )

    def classify_from_asm(self, instructions: list[Instruction]) -> AlgorithmInfo:
        """Classify from instruction list by building fingerprint and pattern matching."""
        fp = self.build_fingerprint(instructions)
        features: dict = {
            "fingerprint": {
                "mfma_count": fp.mfma_count,
                "mfma_types": fp.mfma_types,
                "dpp_count": fp.dpp_count,
                "dpp_modifiers": fp.dpp_modifiers,
                "lds_reads": fp.lds_reads,
                "lds_writes": fp.lds_writes,
                "barrier_count": fp.barrier_count,
                "bpermute_count": fp.bpermute_count,
                "has_direct_lds_loads": fp.has_direct_lds_loads,
            },
        }

        algo_type = AlgorithmType.CUSTOM.value
        sub_type = ""
        confidence = 0.3

        has_fp8_mfma = any("fp8" in t for t in fp.mfma_types)
        has_bf16_mfma = any("bf16" in t for t in fp.mfma_types)
        has_f16_mfma = any(
            "f16" in t and "bf16" not in t
            for t in fp.mfma_types
        )
        has_i8_mfma = any("i8" in t or "i32" in t for t in fp.mfma_types)

        quad_perm = fp.dpp_modifiers.get("quad_perm", 0)
        row_newbcast = fp.dpp_modifiers.get("row_newbcast", 0)

        # MFMA type distribution
        if fp.mfma_count > 0:
            if fp.dpp_count > 0 and has_fp8_mfma:
                algo_type = AlgorithmType.GEMM.value
                sub_type = "gemm_fp8"
                confidence = 0.85
            elif fp.dpp_count > 0 and has_i8_mfma:
                algo_type = AlgorithmType.GEMM.value
                sub_type = "gemm_int8"
                confidence = 0.82
            elif has_fp8_mfma:
                algo_type = AlgorithmType.GEMM.value
                sub_type = "gemm_fp8"
                confidence = 0.8
            elif has_bf16_mfma and fp.has_direct_lds_loads and fp.lds_reads > 50:
                algo_type = AlgorithmType.GEMM.value
                sub_type = "gemm_bf16"
                confidence = 0.78
            elif has_bf16_mfma:
                algo_type = AlgorithmType.GEMM.value
                sub_type = "gemm_bf16"
                confidence = 0.75
            elif has_f16_mfma and fp.dpp_count > 10 and quad_perm > 5:
                algo_type = AlgorithmType.FMHA.value
                sub_type = "fmha_forward"
                confidence = 0.82
            elif fp.mfma_count > 100 and fp.dpp_count > 20 and quad_perm > 10:
                algo_type = AlgorithmType.FMHA.value
                sub_type = "fmha_backward"
                confidence = 0.8
            elif fp.mfma_count > 50 and fp.barrier_count > 15:
                algo_type = AlgorithmType.FMHA.value
                sub_type = "fmha"
                confidence = 0.7
            else:
                algo_type = AlgorithmType.GEMM.value
                sub_type = "gemm_generic"
                confidence = 0.65

        # No MFMA: reduction / norm / softmax
        else:
            if fp.bpermute_count > 0:
                algo_type = AlgorithmType.FMHA.value
                sub_type = "attention_non_mfma"
                confidence = 0.65
            elif fp.barrier_count > 2 and fp.lds_reads + fp.lds_writes > 20:
                if fp.valu_count > fp.salu_count * 2:
                    algo_type = AlgorithmType.SOFTMAX.value
                    sub_type = "softmax"
                    confidence = 0.6
                else:
                    algo_type = AlgorithmType.REDUCTION.value
                    sub_type = "reduction"
                    confidence = 0.6
            elif fp.lds_reads + fp.lds_writes > 10 and fp.barrier_count >= 1:
                algo_type = AlgorithmType.RMSNORM.value
                sub_type = "rmsnorm"
                confidence = 0.55
            else:
                algo_type = AlgorithmType.ELEMENTWISE.value
                sub_type = "elementwise"
                confidence = 0.5

        parameters = self._extract_asm_params(fp)
        return AlgorithmInfo(
            algo_type=algo_type,
            confidence=min(1.0, confidence),
            parameters=parameters,
            features=features,
            sub_type=sub_type,
        )

    def extract_parameters(self, source: str, algo_type: str) -> dict:
        """Extract dimensions and parameters from C++ source."""
        params: dict = {}

        # GEMM: M, N, K
        if algo_type == AlgorithmType.GEMM.value or "GEMM" in algo_type:
            for name, pattern in [
                ("M", r"\bM\s*=\s*(\d+)"),
                ("N", r"\bN\s*=\s*(\d+)"),
                ("K", r"\bK\s*=\s*(\d+)"),
            ]:
                m = re.search(pattern, source, re.IGNORECASE)
                if m:
                    g = m.group(1)
                    params[name] = int(g) if g.isdigit() else g
            m = re.search(r"template\s*<\s*\w+\s+(\d+)\s*,\s*\w+\s+(\d+)\s*,\s*\w+\s+(\d+)", source)
            if m:
                params.setdefault("M", int(m.group(1)))
                params.setdefault("N", int(m.group(2)))
                params.setdefault("K", int(m.group(3)))
            if "int M" in source or "int m" in source:
                params.setdefault("M", "arg")
            if "int N" in source or "int n" in source:
                params.setdefault("N", "arg")
            if "int K" in source or "int k" in source:
                params.setdefault("K", "arg")

        # Norm: hidden_size, epsilon
        if algo_type in (AlgorithmType.LAYERNORM.value, AlgorithmType.RMSNORM.value):
            m = re.search(r"hidden_size\s*[=,]\s*(\d+)", source, re.IGNORECASE)
            if m:
                params["hidden_size"] = int(m.group(1))
            m = re.search(r"epsilon\s*[=,]\s*([\deE.+-]+)f?", source, re.IGNORECASE)
            if m:
                try:
                    params["epsilon"] = float(m.group(1))
                except ValueError:
                    params["epsilon"] = m.group(1)

        # Softmax: batch, seq_len
        if algo_type == AlgorithmType.SOFTMAX.value:
            for name, pat in [
                ("batch", r"batch\s*[=,]\s*(\d+)"),
                ("seq_len", r"(?:seq_len|seq_length|N)\s*[=,]\s*(\d+)"),
            ]:
                m = re.search(pat, source, re.IGNORECASE)
                if m and m.group(1).isdigit():
                    params[name] = int(m.group(1))

        # Template dims
        tmpl = re.search(r"template\s*<\s*[^>]+>", source)
        if tmpl:
            nums = re.findall(r"\b(\d+)\b", tmpl.group(0))
            if nums:
                params["template_dims"] = [int(x) for x in nums]

        return params

    def _extract_asm_params(self, fp: ASMFeatureFingerprint) -> dict:
        """Extract parameters from ASM fingerprint."""
        return {
            "mfma_count": fp.mfma_count,
            "max_vgpr": fp.max_vgpr,
            "max_sgpr": fp.max_sgpr,
            "max_agpr": fp.max_agpr,
            "total_instructions": fp.total_instructions,
            "barrier_count": fp.barrier_count,
            "lds_reads": fp.lds_reads,
            "lds_writes": fp.lds_writes,
        }

    def build_fingerprint(self, instructions: list[Instruction]) -> ASMFeatureFingerprint:
        """Build feature fingerprint from instruction list."""
        mfma_count = 0
        mfma_types: dict[str, int] = {}
        dpp_count = 0
        dpp_modifiers: dict[str, int] = {}
        lds_reads = 0
        lds_writes = 0
        vmem_loads = 0
        vmem_stores = 0
        barrier_count = 0
        waitcnt_count = 0
        nop_count = 0
        valu_count = 0
        salu_count = 0
        branch_count = 0
        bpermute_count = 0
        max_vgpr = 0
        max_sgpr = 0
        max_agpr = 0
        has_direct_lds_loads = False

        dpp_mod_list = (
            "row_newbcast", "quad_perm", "row_shr", "row_shl",
            "row_ror", "row_bcast", "wave_shl", "row_mirror",
        )

        for instr in instructions:
            mn = instr.mnemonic
            ops = instr.operands or ""
            full = instr.full_text or ""

            if "mfma" in mn:
                mfma_count += 1
                mfma_types[mn] = mfma_types.get(mn, 0) + 1
            elif mn.startswith(("global_load", "buffer_load", "flat_load")):
                vmem_loads += 1
                if "lds" in full.lower():
                    has_direct_lds_loads = True
            elif mn.startswith(("global_store", "buffer_store", "flat_store")):
                vmem_stores += 1
            elif mn.startswith("ds_read") or mn == "ds_load":
                lds_reads += 1
            elif mn.startswith("ds_write") or mn == "ds_store":
                lds_writes += 1
            elif mn == "ds_bpermute_b32":
                bpermute_count += 1
            elif "dpp" in full.lower():
                dpp_count += 1
                for mod in dpp_mod_list:
                    if mod in full:
                        dpp_modifiers[mod] = dpp_modifiers.get(mod, 0) + 1
            elif instr.is_barrier or "barrier" in mn:
                barrier_count += 1
            elif mn == "s_nop":
                nop_count += 1
            elif mn == "s_waitcnt" or "waitcnt" in mn:
                waitcnt_count += 1
            elif instr.is_branch or mn.startswith("s_cbranch") or mn == "s_branch":
                branch_count += 1
            elif mn.startswith("v_"):
                valu_count += 1
            elif mn.startswith("s_"):
                salu_count += 1

            for m in re.finditer(r"\bv(\d+)\b", ops):
                max_vgpr = max(max_vgpr, int(m.group(1)))
            for m in re.finditer(r"\bv\[(\d+):(\d+)\]", ops):
                max_vgpr = max(max_vgpr, int(m.group(2)))
            for m in re.finditer(r"\bs(\d+)\b", ops):
                max_sgpr = max(max_sgpr, int(m.group(1)))
            for m in re.finditer(r"\ba\[(\d+):(\d+)\]", ops):
                max_agpr = max(max_agpr, int(m.group(2)))
            for m in re.finditer(r"\ba(\d+)\b", ops):
                max_agpr = max(max_agpr, int(m.group(1)))

        total = len(instructions)
        vectorization_ratio = valu_count / total if total > 0 else 0.0

        return ASMFeatureFingerprint(
            mfma_count=mfma_count,
            mfma_types=mfma_types,
            dpp_count=dpp_count,
            dpp_modifiers=dpp_modifiers,
            lds_reads=lds_reads,
            lds_writes=lds_writes,
            vmem_loads=vmem_loads,
            vmem_stores=vmem_stores,
            barrier_count=barrier_count,
            waitcnt_count=waitcnt_count,
            nop_count=nop_count,
            valu_count=valu_count,
            salu_count=salu_count,
            max_vgpr=max_vgpr,
            max_sgpr=max_sgpr,
            max_agpr=max_agpr,
            total_instructions=total,
            vectorization_ratio=round(vectorization_ratio, 4),
            branch_count=branch_count,
            has_direct_lds_loads=has_direct_lds_loads,
            bpermute_count=bpermute_count,
        )
