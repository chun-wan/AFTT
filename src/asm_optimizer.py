"""Pattern-Based ASM Optimizer for AMDGPU kernels.

Implements optimization analysis and transforms based on patterns learned from
2,660 production kernels. Covers DPP cross-lane operations, software pipelining,
MFMA scheduling, wavefront-level synchronization, and register pressure.

Transform categories:
  1. Same-length binary patches (can be applied directly)
  2. Structural recommendations (require instruction reordering/insertion)
  3. Algorithm-level recommendations (require kernel redesign)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .instruction import Instruction, EditOperation
from .knowledge_base import KnowledgeBase


@dataclass
class OptimizationResult:
    """Result from running the optimizer on an instruction sequence."""
    edits: list[EditOperation] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Optimization Results ===",
            f"Applicable edits (same-length patches): {len(self.edits)}",
            f"Recommendations (require structural changes): {len(self.recommendations)}",
        ]
        if self.edits:
            lines.append("\nEdits to apply:")
            for e in self.edits:
                lines.append(f"  [{e.target_index}] {e.comment}")
        if self.recommendations:
            lines.append("\nRecommendations:")
            for r in self.recommendations:
                sev = r.get("severity", "info")
                cycles = r.get("estimated_cycle_savings", "?")
                lines.append(f"  [{sev}] {r['type']}: {r['description']} "
                             f"(~{cycles} cycles saved)")
        if self.stats:
            lines.append(f"\nKernel Profile:")
            for k, v in self.stats.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def _parse_waitcnt_operands(operands: str) -> dict[str, int]:
    counters = {}
    for name in ("vmcnt", "lgkmcnt", "expcnt", "vscnt"):
        m = re.search(rf'{name}\((\d+)\)', operands)
        if m:
            counters[name] = int(m.group(1))
    return counters


def _build_waitcnt_operands(counters: dict[str, int]) -> str:
    parts = []
    for name in ("vmcnt", "expcnt", "lgkmcnt"):
        if name in counters:
            parts.append(f"{name}({counters[name]})")
    return " ".join(parts)


def _classify_pipe(mnemonic: str) -> str:
    if "mfma" in mnemonic:
        return "MFMA"
    if mnemonic.startswith(("global_load", "global_store", "buffer_load",
                            "buffer_store", "flat_load", "flat_store")):
        return "VMEM"
    if mnemonic.startswith("ds_"):
        return "LDS"
    if mnemonic.startswith(("v_", "v_mfma")):
        return "VALU"
    if mnemonic.startswith("s_"):
        return "SALU"
    return "OTHER"


def _extract_vgpr_indices(operands: str) -> set[int]:
    """Extract VGPR register indices from operand string."""
    indices = set()
    for m in re.finditer(r'\bv(\d+)\b', operands):
        indices.add(int(m.group(1)))
    for m in re.finditer(r'\bv\[(\d+):(\d+)\]', operands):
        lo, hi = int(m.group(1)), int(m.group(2))
        indices.update(range(lo, hi + 1))
    return indices


def _extract_agpr_indices(operands: str) -> set[int]:
    """Extract AGPR register indices from operand string."""
    indices = set()
    for m in re.finditer(r'\ba\[(\d+):(\d+)\]', operands):
        lo, hi = int(m.group(1)), int(m.group(2))
        indices.update(range(lo, hi + 1))
    for m in re.finditer(r'\ba(\d+)\b', operands):
        indices.add(int(m.group(1)))
    return indices


class AsmOptimizer:
    """Pattern-based ASM optimizer using learned patterns from production kernels.

    Analyzes kernels across multiple dimensions:
    - DPP / cross-lane patterns
    - Software pipelining depth and interleaving quality
    - Wavefront synchronization efficiency
    - Register pressure and occupancy impact
    - s_waitcnt precision
    """

    def __init__(self, arch: str = "gfx942", kb: Optional[KnowledgeBase] = None):
        self.arch = arch
        self.kb = kb
        if self.kb is not None:
            self.kb.load()
        self.max_vgpr_per_simd = 512
        self.max_waves_per_simd = 8
        self.lds_per_cu_kb = 64
        self.wavefront_size = 64

    def optimize(self, instructions: list[Instruction],
                 aggressive: bool = False) -> OptimizationResult:
        result = OptimizationResult()

        profile = self._build_kernel_profile(instructions)

        self._opt_waitcnt_relaxation(instructions, result, aggressive)
        self._opt_nop_elimination(instructions, result)
        self._opt_full_wait_splitting(instructions, result)
        self._opt_redundant_barrier(instructions, result)
        self._opt_mfma_vmem_interleave(instructions, result)

        self._analyze_dpp_opportunities(instructions, result, profile)
        self._analyze_software_pipelining(instructions, result, profile)
        self._analyze_mfma_scheduling(instructions, result, profile)
        self._analyze_register_pressure(instructions, result, profile)
        self._analyze_barrier_efficiency(instructions, result, profile)
        self._analyze_load_vectorization(instructions, result, profile)
        self._analyze_lds_to_dpp_replacement(instructions, result, profile)
        self._analyze_non_dpp_reduction(instructions, result, profile)
        self._analyze_fmha_patterns(instructions, result, profile)

        if self.kb is not None:
            self._analyze_kb_dpp_patterns(instructions, result, profile)
            self._analyze_kb_fmha_techniques(instructions, result, profile)

        result.stats = profile
        return result

    def optimize_with_report(self, instructions: list[Instruction],
                              aggressive: bool = False) -> OptimizationResult:
        """Like optimize(), but each edit gets a detailed rationale in its comment."""
        result = self.optimize(instructions, aggressive)

        for edit in result.edits:
            orig = instructions[edit.target_index] if edit.target_index < len(instructions) else None
            rationale_parts = []
            if orig and orig.mnemonic == "s_nop":
                rationale_parts.append(f"NOP_REDUCTION: s_nop {orig.operands} → {edit.new_mnemonic} {edit.new_operands}")
                rationale_parts.append("reduces pipeline bubbles")
            elif edit.new_mnemonic == "s_waitcnt":
                rationale_parts.append(f"WAITCNT_RELAX: relaxed from {orig.operands if orig else '?'} to {edit.new_operands}")
                rationale_parts.append("allows more instruction overlap")
            elif edit.new_mnemonic == "s_nop" and orig and orig.mnemonic == "s_nop":
                rationale_parts.append(f"NOP_REDUCE: fewer idle cycles")
            else:
                rationale_parts.append(f"PATTERN_EDIT: {orig.mnemonic if orig else '?'} → {edit.new_mnemonic}")
            if edit.comment:
                rationale_parts.append(edit.comment)
            edit.comment = " | ".join(rationale_parts)

        return result

    # ── Kernel Profiling ─────────────────────────────────────────────

    def _build_kernel_profile(self, instructions: list[Instruction]) -> dict:
        """Build a comprehensive profile of the kernel's instruction mix."""
        mfma_count = 0
        mfma_types: dict[str, int] = {}
        vmem_loads = 0
        vmem_stores = 0
        lds_reads = 0
        lds_writes = 0
        lds_direct_loads = 0
        dpp_count = 0
        dpp_modifiers: dict[str, int] = {}
        barrier_count = 0
        nop_count = 0
        waitcnt_count = 0
        waitcnt_full = 0
        waitcnt_partial = 0
        branch_count = 0
        readlane_count = 0
        readfirstlane_count = 0
        bpermute_count = 0
        perm_b32_count = 0
        valu_count = 0
        salu_count = 0
        cvt_fp8_count = 0
        max_vgpr = 0
        max_agpr = 0
        max_sgpr = 0

        for instr in instructions:
            mn = instr.mnemonic
            ops = instr.operands or ""
            full = instr.full_text or ""

            if "mfma" in mn:
                mfma_count += 1
                mfma_types[mn] = mfma_types.get(mn, 0) + 1
            elif mn.startswith(("global_load", "buffer_load", "flat_load")):
                vmem_loads += 1
                if "lds" in full:
                    lds_direct_loads += 1
            elif mn.startswith(("global_store", "buffer_store", "flat_store")):
                vmem_stores += 1
            elif mn.startswith("ds_read") or mn == "ds_load":
                lds_reads += 1
            elif mn.startswith("ds_write") or mn == "ds_store":
                lds_writes += 1
            elif mn == "ds_bpermute_b32":
                bpermute_count += 1
            elif "dpp" in full:
                dpp_count += 1
                for mod in ("row_newbcast", "quad_perm", "row_shr", "row_shl",
                            "row_ror", "row_bcast", "wave_shl", "row_mirror"):
                    if mod in full:
                        dpp_modifiers[mod] = dpp_modifiers.get(mod, 0) + 1
            elif mn == "s_barrier":
                barrier_count += 1
            elif mn == "s_nop":
                nop_count += 1
            elif mn == "s_waitcnt":
                waitcnt_count += 1
                counters = _parse_waitcnt_operands(ops)
                if counters.get("vmcnt", -1) == 0 and counters.get("lgkmcnt", -1) == 0:
                    waitcnt_full += 1
                elif any(v > 0 for v in counters.values()):
                    waitcnt_partial += 1
            elif mn.startswith("s_cbranch") or mn == "s_branch":
                branch_count += 1
            elif mn == "v_readlane_b32":
                readlane_count += 1
            elif mn == "v_readfirstlane_b32":
                readfirstlane_count += 1
            elif mn == "v_perm_b32":
                perm_b32_count += 1
            elif mn.startswith("v_cvt_pk_fp8") or mn.startswith("v_cvt_pk_bf8"):
                cvt_fp8_count += 1
            elif mn.startswith("v_"):
                valu_count += 1
            elif mn.startswith("s_"):
                salu_count += 1

            for idx in _extract_vgpr_indices(ops):
                max_vgpr = max(max_vgpr, idx)
            for idx in _extract_agpr_indices(ops):
                max_agpr = max(max_agpr, idx)
            for m in re.finditer(r'\bs(\d+)\b', ops):
                max_sgpr = max(max_sgpr, int(m.group(1)))

        vgpr_used = max_vgpr + 1 if max_vgpr > 0 else 0
        agpr_used = max_agpr + 1 if max_agpr > 0 else 0
        total_gprs = vgpr_used + agpr_used
        occupancy_waves = min(self.max_waves_per_simd,
                              self.max_vgpr_per_simd // max(total_gprs, 1))

        partial_ratio = (waitcnt_partial / max(waitcnt_count, 1)) * 100

        kernel_type = self._classify_kernel_type(
            mfma_count, dpp_count, lds_direct_loads, cvt_fp8_count,
            bpermute_count, barrier_count, mfma_types)

        return {
            "total_instructions": len(instructions),
            "kernel_type": kernel_type,
            "mfma_count": mfma_count,
            "mfma_types": mfma_types,
            "vmem_loads": vmem_loads,
            "vmem_stores": vmem_stores,
            "lds_reads": lds_reads,
            "lds_writes": lds_writes,
            "lds_direct_loads": lds_direct_loads,
            "dpp_count": dpp_count,
            "dpp_modifiers": dpp_modifiers,
            "barrier_count": barrier_count,
            "nop_count": nop_count,
            "waitcnt_total": waitcnt_count,
            "waitcnt_full_zero": waitcnt_full,
            "waitcnt_partial": waitcnt_partial,
            "partial_wait_ratio_pct": round(partial_ratio, 1),
            "branch_count": branch_count,
            "readlane_count": readlane_count,
            "readfirstlane_count": readfirstlane_count,
            "bpermute_count": bpermute_count,
            "perm_b32_count": perm_b32_count,
            "cvt_fp8_count": cvt_fp8_count,
            "valu_count": valu_count,
            "salu_count": salu_count,
            "vgpr_used": vgpr_used,
            "agpr_used": agpr_used,
            "sgpr_used": max_sgpr + 1 if max_sgpr > 0 else 0,
            "total_gprs": total_gprs,
            "estimated_occupancy_waves": occupancy_waves,
            "production_avg_partial_wait_ratio": 57.2,
        }

    def _classify_kernel_type(self, mfma_count, dpp_count, lds_direct_loads,
                               cvt_fp8_count, bpermute_count, barrier_count,
                               mfma_types) -> str:
        internal_type: str
        if mfma_count == 0:
            if bpermute_count > 0:
                internal_type = "attention_non_mfma"
            else:
                internal_type = "non_mfma"
        else:
            has_fp8_mfma = any("fp8" in t for t in mfma_types)
            has_bf16_mfma = any("bf16" in t for t in mfma_types)
            has_i8_mfma = any("i8" in t or "i32" in t for t in mfma_types)

            if dpp_count > 0 and has_fp8_mfma:
                internal_type = "fp8_gemm_with_dequant"
            elif dpp_count > 0 and has_i8_mfma:
                internal_type = "int8_gemm_with_dequant"
            elif has_fp8_mfma:
                internal_type = "fp8_gemm"
            elif has_bf16_mfma and lds_direct_loads > 100:
                internal_type = "bf16_gemm_direct_lds"
            elif has_bf16_mfma:
                internal_type = "bf16_gemm"
            elif bpermute_count > 0:
                internal_type = "attention_with_permute"
            elif barrier_count > 20:
                internal_type = "multi_barrier_kernel"
            else:
                internal_type = "mfma_generic"

        if self.kb is not None:
            kb_type = self._map_to_kb_kernel_type(
                internal_type, dpp_count, bpermute_count, mfma_types
            )
            if kb_type is not None:
                return kb_type
        return internal_type

    def _map_to_kb_kernel_type(
        self, internal_type: str, dpp_count: int, bpermute_count: int,
        mfma_types: dict
    ) -> Optional[str]:
        """Map internal kernel type to KB optimization_techniques_by_kernel_type key."""
        dpp = self.kb.dpp_crosslane_patterns
        techniques = dpp.get("optimization_techniques_by_kernel_type", {})
        if not techniques:
            return None

        has_bf16 = any("bf16" in t for t in mfma_types)

        if internal_type in ("fp8_gemm_with_dequant", "int8_gemm_with_dequant", "fp8_gemm"):
            return "gemm_fp8_blockscale" if "gemm_fp8_blockscale" in techniques else None
        if internal_type in ("bf16_gemm_direct_lds", "bf16_gemm"):
            return "gemm_bf16" if "gemm_bf16" in techniques else None
        if internal_type == "attention_non_mfma":
            return "paged_attention" if "paged_attention" in techniques else None
        if internal_type == "attention_with_permute":
            if dpp_count > 20 and has_bf16 and "fmha_backward" in techniques:
                return "fmha_backward"
            if "paged_attention" in techniques:
                return "paged_attention"
            return None
        if internal_type == "multi_barrier_kernel" and dpp_count > 20 and has_bf16:
            return "fmha_backward" if "fmha_backward" in techniques else None
        return None

    # ── Same-Length Patches ──────────────────────────────────────────

    def _opt_waitcnt_relaxation(self, instructions: list[Instruction],
                                 result: OptimizationResult,
                                 aggressive: bool) -> None:
        """Relax s_waitcnt vmcnt(0) with register-level dependency tracking.

        Key safety rules:
        1. Track which VGPRs each VMEM load writes to
        2. Scan until next barrier/branch/waitcnt for actual consumers
        3. Block relaxation if barrier follows before all loads are consumed
        4. Compute safe vmcnt based on load-to-consumer distance
        """
        # Per-load tracking: list of (load_index, dest_vgprs)
        pending_loads: list[tuple[int, set[int]]] = []

        for i, instr in enumerate(instructions):
            mn = instr.mnemonic
            full = instr.full_text or ""
            ops = instr.operands or ""

            if mn.startswith(("global_load", "buffer_load", "flat_load")):
                dest_vgprs = set()
                if "lds" not in full:
                    dest_vgprs = _extract_vgpr_indices(ops.split(",")[0] if "," in ops else ops)
                pending_loads.append((i, dest_vgprs))

            elif mn.startswith("ds_read"):
                pass  # lgkm tracked separately

            elif mn == "s_waitcnt":
                counters = _parse_waitcnt_operands(ops)

                if "vmcnt" in counters and counters["vmcnt"] == 0 and len(pending_loads) > 2:
                    # Find distance to next barrier
                    dist_to_barrier = len(instructions) - i
                    for j in range(i + 1, len(instructions)):
                        if instructions[j].mnemonic in ("s_barrier", "s_endpgm", "s_branch"):
                            dist_to_barrier = j - i
                            break
                        if instructions[j].mnemonic == "s_waitcnt":
                            dist_to_barrier = j - i
                            break

                    # If barrier is very close (<=4 instructions), don't relax:
                    # the waitcnt is likely a pre-barrier synchronization point
                    if dist_to_barrier <= 4:
                        self._reset_pending(pending_loads, counters)
                        continue

                    # Collect consumer VGPRs until next sync point
                    consumer_vgprs: set[int] = set()
                    for j in range(i + 1, min(i + dist_to_barrier, len(instructions))):
                        next_instr = instructions[j]
                        next_mn = next_instr.mnemonic
                        if next_mn in ("s_waitcnt", "s_barrier", "s_endpgm"):
                            break
                        next_ops = next_instr.operands or ""
                        # For MFMA/VALU, source operands are consumers
                        if "mfma" in next_mn or next_mn.startswith("v_"):
                            parts = next_ops.split(",")
                            for part in parts[1:]:  # skip dest (first operand)
                                consumer_vgprs.update(_extract_vgpr_indices(part))
                        elif next_mn.startswith("ds_write"):
                            consumer_vgprs.update(_extract_vgpr_indices(next_ops))

                    # Count how many pending loads are NOT consumed before the barrier
                    # Those loads are safe to still be in-flight
                    unconsumed = 0
                    for _, dest_regs in reversed(pending_loads):
                        if dest_regs and not dest_regs.intersection(consumer_vgprs):
                            unconsumed += 1
                        else:
                            break  # once we hit a consumed load, older ones are needed too

                    safe_vmcnt = min(unconsumed, len(pending_loads) - 1)
                    if aggressive:
                        safe_vmcnt = min(safe_vmcnt + 1, len(pending_loads) - 1)

                    # Clamp: never relax more than half the in-flight loads
                    safe_vmcnt = min(safe_vmcnt, len(pending_loads) // 2)

                    if safe_vmcnt > 0:
                        new_counters = dict(counters)
                        new_counters["vmcnt"] = safe_vmcnt
                        new_operands = _build_waitcnt_operands(new_counters)

                        result.edits.append(EditOperation(
                            target_index=i,
                            new_mnemonic="s_waitcnt",
                            new_operands=new_operands,
                            comment=f"Relax vmcnt(0)->vmcnt({safe_vmcnt}): "
                                    f"{len(pending_loads)} loads in-flight, "
                                    f"{unconsumed} unconsumed before next sync, "
                                    f"barrier dist={dist_to_barrier}",
                        ))

                self._reset_pending(pending_loads, counters)

            elif mn == "s_barrier":
                pending_loads.clear()

    @staticmethod
    def _reset_pending(pending_loads: list, counters: dict) -> None:
        """Reset pending load tracking after a waitcnt."""
        if "vmcnt" in counters:
            keep = counters["vmcnt"]
            if keep > 0 and len(pending_loads) > keep:
                pending_loads[:] = pending_loads[-keep:]
            elif keep == 0:
                pending_loads.clear()

    def _opt_nop_elimination(self, instructions: list[Instruction],
                              result: OptimizationResult) -> None:
        """Remove unnecessary s_nop not required for MFMA hazard avoidance.

        In production kernels, s_nop is primarily used for:
        1. MFMA read-after-write hazards (1-2 cycles between dependent MFMAs)
        2. DPP instruction hazards (1 cycle after DPP result is read)
        3. v_readlane/v_readfirstlane hazards
        """
        for i, instr in enumerate(instructions):
            if instr.mnemonic != "s_nop":
                continue

            nop_val = 0
            m = re.search(r'(\d+)', instr.operands or "0")
            if m:
                nop_val = int(m.group(1))
            if nop_val == 0:
                continue

            near_hazard = False
            for delta in range(-3, 4):
                j = i + delta
                if 0 <= j < len(instructions) and j != i:
                    jmn = instructions[j].mnemonic
                    jfull = instructions[j].full_text or ""
                    if ("mfma" in jmn or "dpp" in jfull or
                            jmn in ("v_readlane_b32", "v_readfirstlane_b32")):
                        near_hazard = True
                        break

            if not near_hazard:
                result.edits.append(EditOperation(
                    target_index=i,
                    new_mnemonic="s_nop",
                    new_operands="0",
                    comment=f"Reduce s_nop {nop_val}->0: not adjacent to MFMA/DPP/readlane hazard",
                ))
            elif nop_val > 3:
                result.edits.append(EditOperation(
                    target_index=i,
                    new_mnemonic="s_nop",
                    new_operands="1",
                    comment=f"Reduce s_nop {nop_val}->1: oversized hazard NOP "
                            f"(MFMA hazard needs 1-2 cycles max)",
                ))

    def _opt_full_wait_splitting(self, instructions: list[Instruction],
                                  result: OptimizationResult) -> None:
        """Split full waits when only one counter matters.

        s_waitcnt vmcnt(0) lgkmcnt(0) -> s_waitcnt vmcnt(0) when only VMEM needed
        s_waitcnt vmcnt(0) lgkmcnt(0) -> s_waitcnt lgkmcnt(0) when only LDS needed
        This is a same-length binary patch (operand-only change).
        """
        for i, instr in enumerate(instructions):
            if instr.mnemonic != "s_waitcnt":
                continue

            counters = _parse_waitcnt_operands(instr.operands)
            if not ("vmcnt" in counters and counters["vmcnt"] == 0 and
                    "lgkmcnt" in counters and counters["lgkmcnt"] == 0):
                continue

            needs_vmem = False
            needs_lds = False
            for j in range(i + 1, min(i + 15, len(instructions))):
                next_instr = instructions[j]
                next_mn = next_instr.mnemonic
                if next_mn.startswith("ds_"):
                    needs_lds = True
                if next_mn.startswith(("global_", "buffer_", "flat_")):
                    needs_vmem = True
                next_ops = next_instr.operands or ""
                if "mfma" in next_mn:
                    src_regs = _extract_vgpr_indices(next_ops)
                    if src_regs:
                        needs_vmem = True
                if next_mn in ("s_waitcnt", "s_barrier", "s_endpgm"):
                    break

            if needs_vmem and not needs_lds:
                new_counters = {"vmcnt": 0}
                if "expcnt" in counters:
                    new_counters["expcnt"] = counters["expcnt"]
                new_operands = _build_waitcnt_operands(new_counters)
                result.edits.append(EditOperation(
                    target_index=i,
                    new_mnemonic="s_waitcnt",
                    new_operands=new_operands,
                    comment=f"Split full wait: only VMEM needed, removing lgkmcnt(0)",
                ))
            elif needs_lds and not needs_vmem:
                new_counters = {"lgkmcnt": 0}
                if "expcnt" in counters:
                    new_counters["expcnt"] = counters["expcnt"]
                new_operands = _build_waitcnt_operands(new_counters)
                result.edits.append(EditOperation(
                    target_index=i,
                    new_mnemonic="s_waitcnt",
                    new_operands=new_operands,
                    comment=f"Split full wait: only LDS needed, removing vmcnt(0)",
                ))

    def _opt_redundant_barrier(self, instructions: list[Instruction],
                                result: OptimizationResult) -> None:
        """Remove redundant s_barrier instructions.

        A barrier is redundant if:
        1. Two barriers appear with no LDS read/write between them
        2. A barrier immediately follows s_waitcnt vmcnt(0) lgkmcnt(0) with
           no LDS activity between the waitcnt and barrier
        """
        prev_barrier_idx = -1
        lds_between = False

        for i, instr in enumerate(instructions):
            mn = instr.mnemonic

            if mn.startswith("ds_"):
                lds_between = True

            elif mn == "s_barrier":
                if prev_barrier_idx >= 0 and not lds_between:
                    result.edits.append(EditOperation(
                        target_index=i,
                        new_mnemonic="s_nop",
                        new_operands="0",
                        comment=f"Remove redundant barrier: no LDS ops since barrier at [{prev_barrier_idx}]",
                    ))
                else:
                    prev_barrier_idx = i
                lds_between = False

            elif mn in ("s_endpgm", "s_branch") or mn.startswith("s_cbranch"):
                prev_barrier_idx = -1
                lds_between = False

    def _opt_mfma_vmem_interleave(self, instructions: list[Instruction],
                                    result: OptimizationResult) -> None:
        """Interleave VMEM loads into MFMA chains to hide memory latency.

        When 3+ consecutive MFMAs appear without memory ops, and a VMEM load
        exists nearby (before or after the chain), swap it into the middle of
        the chain. MFMA has 64-cycle latency, which can hide VMEM latency.
        Only swaps if no register dependency exists between the two instructions.
        """
        i = 0
        while i < len(instructions) - 3:
            # Detect MFMA chain start
            if "mfma" not in instructions[i].mnemonic:
                i += 1
                continue

            chain_start = i
            chain_end = i
            while chain_end + 1 < len(instructions) and "mfma" in instructions[chain_end + 1].mnemonic:
                chain_end += 1

            chain_len = chain_end - chain_start + 1
            if chain_len < 3:
                i = chain_end + 1
                continue

            # Look for a VMEM load right before or after the chain
            vmem_idx = -1
            if chain_start > 0:
                prev_mn = instructions[chain_start - 1].mnemonic
                if prev_mn.startswith(("global_load", "buffer_load")):
                    vmem_idx = chain_start - 1
            if vmem_idx < 0 and chain_end + 1 < len(instructions):
                next_mn = instructions[chain_end + 1].mnemonic
                if next_mn.startswith(("global_load", "buffer_load")):
                    vmem_idx = chain_end + 1

            if vmem_idx < 0:
                i = chain_end + 1
                continue

            # Target: move the load to the middle of the chain
            target_slot = chain_start + chain_len // 2
            vmem_instr = instructions[vmem_idx]
            target_instr = instructions[target_slot]

            # Check register dependency: the load's dest must not be used by the target MFMA
            load_dest = _extract_vgpr_indices(
                vmem_instr.operands.split(",")[0] if "," in (vmem_instr.operands or "") else (vmem_instr.operands or ""))
            mfma_srcs = set()
            mfma_ops = target_instr.operands or ""
            parts = mfma_ops.split(",")
            for p in parts[1:]:
                mfma_srcs.update(_extract_vgpr_indices(p))

            if load_dest.intersection(mfma_srcs):
                i = chain_end + 1
                continue

            # Swap: replace the two instructions with each other's content
            result.edits.append(EditOperation(
                target_index=vmem_idx,
                new_mnemonic=target_instr.mnemonic,
                new_operands=target_instr.operands or "",
                comment=f"MFMA_INTERLEAVE: swap VMEM load with MFMA at [{target_slot}] "
                        f"to hide memory latency in {chain_len}-deep chain",
            ))
            result.edits.append(EditOperation(
                target_index=target_slot,
                new_mnemonic=vmem_instr.mnemonic,
                new_operands=vmem_instr.operands or "",
                comment=f"MFMA_INTERLEAVE: VMEM load moved from [{vmem_idx}] into MFMA chain",
            ))

            i = chain_end + 1

    # ── DPP / Cross-Lane Analysis ────────────────────────────────────

    def _analyze_dpp_opportunities(self, instructions: list[Instruction],
                                    result: OptimizationResult,
                                    profile: dict) -> None:
        """Detect patterns where DPP could replace LDS-based communication.

        Key patterns from production kernels:
        - row_newbcast: 309k uses for FP8/INT8 scale broadcast (avoids LDS)
        - quad_perm: 38k uses for softmax derivative broadcast
        - row_shr/ror: 8.4k uses for butterfly wavefront reduction
        """
        lds_write_read_pairs = self._find_lds_write_barrier_read_patterns(instructions)

        for pattern in lds_write_read_pairs:
            ws_idx, barrier_idx, read_indices = pattern
            write_instr = instructions[ws_idx]
            reads = [instructions[r] for r in read_indices]

            is_broadcast = len(set(r.operands for r in reads)) == len(reads)
            read_count = len(read_indices)

            if read_count <= 16 and is_broadcast:
                result.recommendations.append({
                    "type": "lds_to_dpp_broadcast",
                    "severity": "high",
                    "description": (
                        f"LDS write [{ws_idx}] + barrier [{barrier_idx}] + "
                        f"{read_count} reads can likely be replaced with DPP "
                        f"row_newbcast (1 cycle per lane vs 20-40 cycle LDS round-trip). "
                        f"Production FP8 GEMMs use row_newbcast for scale distribution."
                    ),
                    "estimated_cycle_savings": 30 * read_count // 16,
                    "write_index": ws_idx,
                    "barrier_index": barrier_idx,
                    "read_indices": read_indices,
                    "suggested_dpp": "v_*_dpp ... row_newbcast:N",
                })

        self._detect_reduction_via_lds(instructions, result)

    def _find_lds_write_barrier_read_patterns(
            self, instructions: list[Instruction]) -> list[tuple]:
        """Find ds_write -> s_barrier -> ds_read sequences."""
        patterns = []
        i = 0
        while i < len(instructions):
            if instructions[i].mnemonic.startswith("ds_write"):
                write_idx = i
                barrier_idx = None
                for j in range(i + 1, min(i + 20, len(instructions))):
                    if instructions[j].mnemonic == "s_barrier":
                        barrier_idx = j
                        break
                if barrier_idx is not None:
                    read_indices = []
                    for j in range(barrier_idx + 1, min(barrier_idx + 30, len(instructions))):
                        if instructions[j].mnemonic.startswith("ds_read"):
                            read_indices.append(j)
                        elif instructions[j].mnemonic in ("s_barrier", "s_endpgm"):
                            break
                    if read_indices:
                        patterns.append((write_idx, barrier_idx, read_indices))
            i += 1
        return patterns

    def _detect_reduction_via_lds(self, instructions: list[Instruction],
                                   result: OptimizationResult) -> None:
        """Detect wavefront-local reductions done via LDS that could use DPP.

        Pattern: ds_write + barrier + multiple ds_read + reduce (add/max)
        Better: DPP row_shr:1,2,4,8 butterfly reduction (no LDS, no barrier)
        """
        for i, instr in enumerate(instructions):
            if not instr.mnemonic.startswith("ds_write"):
                continue

            barrier_idx = None
            for j in range(i + 1, min(i + 10, len(instructions))):
                if instructions[j].mnemonic == "s_barrier":
                    barrier_idx = j
                    break

            if barrier_idx is None:
                continue

            reads = []
            reduces = []
            for j in range(barrier_idx + 1, min(barrier_idx + 40, len(instructions))):
                mn = instructions[j].mnemonic
                if mn.startswith("ds_read"):
                    reads.append(j)
                elif mn.startswith(("v_add_f32", "v_max_f32", "v_max3_f32",
                                    "v_add_u32", "v_or_b32")):
                    reduces.append(j)
                elif mn in ("s_barrier", "s_endpgm"):
                    break

            if len(reads) >= 4 and len(reduces) >= 2:
                result.recommendations.append({
                    "type": "reduction_lds_to_dpp",
                    "severity": "high",
                    "description": (
                        f"Wavefront-local reduction at [{i}]: {len(reads)} LDS reads + "
                        f"{len(reduces)} reduce ops. Replace with DPP butterfly: "
                        f"row_shr:1,2,4,8 + row_bcast:15,31 "
                        f"(production layernorm uses this, saving ~50-100 cycles per reduction)"
                    ),
                    "estimated_cycle_savings": 60,
                    "lds_write": i,
                    "barrier": barrier_idx,
                    "reads": reads,
                    "reduces": reduces,
                })

    def _analyze_lds_to_dpp_replacement(self, instructions: list[Instruction],
                                         result: OptimizationResult,
                                         profile: dict) -> None:
        """Check if kernel does intra-wavefront communication via LDS
        when DPP would be more efficient."""
        if profile["dpp_count"] > 0:
            return

        if profile["mfma_count"] > 0 and profile["lds_writes"] > 0:
            mfma_with_scale = False
            for i, instr in enumerate(instructions):
                if "mfma" in instr.mnemonic:
                    for j in range(i + 1, min(i + 20, len(instructions))):
                        mn = instructions[j].mnemonic
                        if mn.startswith(("v_mul_f32", "v_fma_f32", "v_pk_fma")):
                            mfma_with_scale = True
                            break
                    if mfma_with_scale:
                        break

            if mfma_with_scale and profile.get("cvt_fp8_count", 0) > 0:
                result.recommendations.append({
                    "type": "missing_dpp_dequant",
                    "severity": "high",
                    "description": (
                        "Kernel has MFMA + scale multiply + FP8 conversion "
                        "but no DPP instructions. Production FP8 GEMMs use "
                        "row_newbcast to broadcast scale factors within wavefront "
                        "instead of LDS. This could save 30-60 cycles per tile."
                    ),
                    "estimated_cycle_savings": 45,
                })

    # ── Software Pipelining Analysis ─────────────────────────────────

    def _analyze_software_pipelining(self, instructions: list[Instruction],
                                      result: OptimizationResult,
                                      profile: dict) -> None:
        """Analyze software pipelining depth and quality.

        Production patterns:
        - bf16gemm pf3: prefetch depth 3 (triple buffering), vmcnt(20) with 40+ loads
        - FP8 blockscale: lgkmcnt(14) with 16 LDS reads pending
        - Average: 7.8 loads before first wait
        """
        loads_before_first_wait = 0
        for instr in instructions:
            mn = instr.mnemonic
            if mn.startswith(("global_load", "buffer_load")):
                loads_before_first_wait += 1
            elif mn == "s_waitcnt":
                break

        max_vmcnt_seen = 0
        max_lgkmcnt_seen = 0
        for instr in instructions:
            if instr.mnemonic == "s_waitcnt":
                counters = _parse_waitcnt_operands(instr.operands)
                if "vmcnt" in counters and counters["vmcnt"] > 0:
                    max_vmcnt_seen = max(max_vmcnt_seen, counters["vmcnt"])
                if "lgkmcnt" in counters and counters["lgkmcnt"] > 0:
                    max_lgkmcnt_seen = max(max_lgkmcnt_seen, counters["lgkmcnt"])

        has_direct_lds = profile["lds_direct_loads"] > 0
        has_triple_buffer = max_vmcnt_seen >= 15

        if profile["mfma_count"] > 50:
            if loads_before_first_wait < 4:
                result.recommendations.append({
                    "type": "shallow_prefetch",
                    "severity": "high",
                    "description": (
                        f"Only {loads_before_first_wait} loads before first wait. "
                        f"Production average is 7.8. bf16gemm pf3 issues 40+ loads "
                        f"before waiting. Deeper prefetch hides memory latency."
                    ),
                    "estimated_cycle_savings": 100,
                    "current_depth": loads_before_first_wait,
                    "target_depth": 8,
                })

            if not has_direct_lds and profile["lds_reads"] > 20:
                result.recommendations.append({
                    "type": "missing_direct_lds_load",
                    "severity": "medium",
                    "description": (
                        "Kernel uses LDS heavily but doesn't use direct-to-LDS "
                        "loading (buffer_load_dword ... lds). Production bf16gemm "
                        "uses 300 direct-to-LDS loads to bypass VGPRs and reduce "
                        "register pressure."
                    ),
                    "estimated_cycle_savings": 50,
                })

            if not has_triple_buffer and profile["barrier_count"] > 5:
                buffer_depth = 1
                if max_vmcnt_seen > 0:
                    if max_vmcnt_seen >= 15:
                        buffer_depth = 3
                    elif max_vmcnt_seen >= 5:
                        buffer_depth = 2

                if buffer_depth < 3:
                    result.recommendations.append({
                        "type": "increase_buffer_depth",
                        "severity": "medium",
                        "description": (
                            f"Estimated buffer depth: {buffer_depth}. "
                            f"Production bf16gemm uses triple buffering (pf3) with "
                            f"vmcnt(20)+. Moving from double to triple buffering "
                            f"can reduce stall cycles by 20-30%."
                        ),
                        "estimated_cycle_savings": 200,
                        "current_depth": buffer_depth,
                    })

    # ── MFMA Scheduling Analysis ─────────────────────────────────────

    def _analyze_mfma_scheduling(self, instructions: list[Instruction],
                                  result: OptimizationResult,
                                  profile: dict) -> None:
        """Analyze MFMA chain interleaving with memory operations.

        Best practice: MFMA → buffer_load_lds + ds_read → MFMA
        Bad: long MFMA chains with no loads between them
        """
        if profile["mfma_count"] == 0:
            return

        chains = []
        current = []
        max_gap_without_mem = 0
        gap_count = 0

        for i, instr in enumerate(instructions):
            is_mfma = "mfma" in instr.mnemonic
            is_mem = (instr.mnemonic.startswith(("global_load", "buffer_load",
                                                  "ds_read", "ds_write")))

            if is_mfma:
                current.append(i)
            elif is_mem:
                if len(current) > 1:
                    chains.append(current[:])
                current = []
            else:
                pass

        if len(current) > 1:
            chains.append(current)

        pure_mfma_chains = []
        for chain in chains:
            mem_between = 0
            for j in range(chain[0], chain[-1] + 1):
                mn = instructions[j].mnemonic
                if mn.startswith(("global_load", "buffer_load", "ds_read", "ds_write")):
                    mem_between += 1
            if mem_between == 0 and len(chain) > 4:
                pure_mfma_chains.append(chain)

        for chain in pure_mfma_chains:
            if len(chain) > 8:
                result.recommendations.append({
                    "type": "mfma_chain_no_interleave",
                    "severity": "high",
                    "description": (
                        f"MFMA chain [{chain[0]}-{chain[-1]}] ({len(chain)} MFMAs) "
                        f"has no memory operations interleaved. Production kernels "
                        f"interleave 1-2 buffer_load_lds + ds_read between every "
                        f"2-4 MFMAs to hide memory latency. This chain wastes "
                        f"{len(chain) * 4}+ cycles of memory bandwidth."
                    ),
                    "estimated_cycle_savings": len(chain) * 8,
                    "chain_start": chain[0],
                    "chain_length": len(chain),
                })

        if profile["mfma_count"] > 0:
            total_mfma_in_pure_chains = sum(len(c) for c in pure_mfma_chains)
            interleave_ratio = 1.0 - (total_mfma_in_pure_chains / profile["mfma_count"])
            if interleave_ratio < 0.5 and profile["mfma_count"] > 20:
                result.recommendations.append({
                    "type": "low_interleave_ratio",
                    "severity": "medium",
                    "description": (
                        f"Only {interleave_ratio*100:.0f}% of MFMAs are interleaved "
                        f"with memory ops. Production kernels achieve >90% interleaving. "
                        f"Restructure main loop to alternate MFMA-load-MFMA-load."
                    ),
                    "estimated_cycle_savings": 150,
                    "current_ratio": round(interleave_ratio, 2),
                })

    # ── Register Pressure Analysis ───────────────────────────────────

    def _analyze_register_pressure(self, instructions: list[Instruction],
                                    result: OptimizationResult,
                                    profile: dict) -> None:
        """Analyze register pressure and its impact on occupancy.

        gfx942: 512 VGPRs/SIMD, max 8 waves. VGPRs include AGPRs.
        Occupancy = min(8, 512 / (vgprs + agprs_used))
        """
        total_gprs = profile["total_gprs"]
        occupancy = profile["estimated_occupancy_waves"]

        if occupancy <= 1 and total_gprs > 256:
            result.recommendations.append({
                "type": "register_pressure_critical",
                "severity": "critical",
                "description": (
                    f"Extremely high register pressure: {profile['vgpr_used']} VGPRs + "
                    f"{profile['agpr_used']} AGPRs = {total_gprs} total. "
                    f"Occupancy: {occupancy} wave(s)/SIMD. Consider reducing "
                    f"accumulator registers or using register spilling to LDS."
                ),
                "estimated_cycle_savings": 500,
                "vgpr_used": profile["vgpr_used"],
                "agpr_used": profile["agpr_used"],
                "occupancy": occupancy,
            })
        elif occupancy <= 2 and total_gprs > 128:
            result.recommendations.append({
                "type": "register_pressure_high",
                "severity": "medium",
                "description": (
                    f"High register pressure: {total_gprs} GPRs, {occupancy} waves. "
                    f"Each VGPR saved can increase occupancy. Consider: "
                    f"(1) Smaller tile sizes (2) Register reuse "
                    f"(3) Direct-to-LDS loads to avoid VGPR temporaries."
                ),
                "estimated_cycle_savings": 100,
                "total_gprs": total_gprs,
                "occupancy": occupancy,
            })

    # ── Barrier Efficiency Analysis ──────────────────────────────────

    def _analyze_barrier_efficiency(self, instructions: list[Instruction],
                                     result: OptimizationResult,
                                     profile: dict) -> None:
        """Analyze s_barrier placement for double/triple buffering efficiency.

        Barriers are expensive (~40 cycles on gfx942). Production kernels minimize
        barriers and maximize work between them.
        """
        if profile["barrier_count"] == 0:
            return

        barrier_positions = [i for i, ins in enumerate(instructions)
                             if ins.mnemonic == "s_barrier"]

        instructions_between = []
        for idx in range(len(barrier_positions) - 1):
            gap = barrier_positions[idx + 1] - barrier_positions[idx]
            instructions_between.append(gap)

        if instructions_between:
            avg_gap = sum(instructions_between) / len(instructions_between)
            min_gap = min(instructions_between)

            if min_gap < 5:
                close_pairs = [(barrier_positions[j], barrier_positions[j + 1])
                               for j in range(len(barrier_positions) - 1)
                               if barrier_positions[j + 1] - barrier_positions[j] < 5]
                result.recommendations.append({
                    "type": "redundant_barriers",
                    "severity": "medium",
                    "description": (
                        f"Found {len(close_pairs)} barrier pair(s) with <5 instructions "
                        f"between them (avg gap: {avg_gap:.0f}). Close barriers indicate "
                        f"possible redundant synchronization. Production GEMM kernels "
                        f"have 100+ instructions between barriers."
                    ),
                    "estimated_cycle_savings": len(close_pairs) * 40,
                    "close_pairs": close_pairs[:5],
                })

    # ── Non-DPP Reduction Detection ─────────────────────────────────

    def _analyze_non_dpp_reduction(self, instructions: list[Instruction],
                                    result: OptimizationResult,
                                    profile: dict) -> None:
        """Detect reductions using __shfl_xor (ds_bpermute) instead of DPP.

        Pattern from fused_qk_norm_rope_cache_quant.cu optimization:
        - __shfl_xor(val, mask, 32) compiles to ds_bpermute on AMD
        - ds_bpermute goes through LDS (~20 cycles per round-trip)
        - DPP butterfly (quad_perm + row_ror) is register-level (~1 cycle)

        Also detects hipcub::BlockReduce patterns which compile to a tree
        reduction via shared memory instead of DPP butterfly.
        """
        bpermute_clusters = []
        current_cluster = []

        for i, instr in enumerate(instructions):
            if instr.mnemonic == "ds_bpermute_b32":
                current_cluster.append(i)
            else:
                if len(current_cluster) >= 3:
                    bpermute_clusters.append(current_cluster[:])
                current_cluster = []

        if len(current_cluster) >= 3:
            bpermute_clusters.append(current_cluster)

        for cluster in bpermute_clusters:
            has_reduce_op = False
            for j in range(cluster[0], min(cluster[-1] + 10, len(instructions))):
                mn = instructions[j].mnemonic
                if mn.startswith(("v_add_f32", "v_max_f32", "v_min_f32")):
                    has_reduce_op = True
                    break

            if has_reduce_op:
                result.recommendations.append({
                    "type": "bpermute_reduction_to_dpp",
                    "severity": "high",
                    "description": (
                        f"Wavefront reduction at [{cluster[0]}-{cluster[-1]}] uses "
                        f"{len(cluster)} ds_bpermute_b32 (from __shfl_xor). Each "
                        f"ds_bpermute costs ~20 cycles via LDS. Replace with DPP "
                        f"butterfly: quad_perm:[1,0,3,2] + quad_perm:[2,3,0,1] + "
                        f"row_ror:4 + row_ror:8 + row_bcast:15 + row_bcast:31 "
                        f"(~6 cycles total vs ~120 cycles for 6x ds_bpermute). "
                        f"Use wave_reduce/multithread_reduce from hip_reduce.h."
                    ),
                    "estimated_cycle_savings": len(cluster) * 18,
                    "cluster_start": cluster[0],
                    "cluster_length": len(cluster),
                })

        if profile["dpp_count"] == 0 and profile["barrier_count"] > 0:
            lds_reduction_candidates = 0
            for i, instr in enumerate(instructions):
                if not instr.mnemonic.startswith("ds_write"):
                    continue
                for j in range(i + 1, min(i + 15, len(instructions))):
                    if instructions[j].mnemonic == "s_barrier":
                        for k in range(j + 1, min(j + 20, len(instructions))):
                            mn_k = instructions[k].mnemonic
                            if mn_k.startswith("ds_read"):
                                lds_reduction_candidates += 1
                                break
                        break

            if lds_reduction_candidates > 0:
                result.recommendations.append({
                    "type": "generic_reduction_no_dpp",
                    "severity": "medium",
                    "description": (
                        f"Kernel has {lds_reduction_candidates} LDS-based reduction "
                        f"pattern(s) (ds_write + barrier + ds_read) but zero DPP "
                        f"instructions. This is a common anti-pattern in HIP kernels "
                        f"ported from CUDA that use hipcub::BlockReduce or __shfl_xor. "
                        f"Replace with block_reduce from hip_reduce.h which uses DPP "
                        f"butterfly (as done in rmsnorm_quant_kernels.cu)."
                    ),
                    "estimated_cycle_savings": lds_reduction_candidates * 50,
                })

    # ── FMHA-Pattern Awareness ────────────────────────────────────────

    def _analyze_fmha_patterns(self, instructions: list[Instruction],
                                result: OptimizationResult,
                                profile: dict) -> None:
        """Check for FMHA-like kernels and compare against known optimal patterns.

        Learned from 180 production FMHA kernels:
        - Forward: 176 MFMAs, 0 DPP, 30 barriers, 2:1 MFMA-to-load ratio
        - Backward: 480 MFMAs, 92 DPP (quad_perm), 51 barriers, full AGPR
        """
        if profile["mfma_count"] < 50:
            return

        has_bf16_mfma = any("bf16" in t for t in profile.get("mfma_types", {}))
        has_f16_mfma = any("f16" in t and "bf16" not in t
                          for t in profile.get("mfma_types", {}))
        if not (has_bf16_mfma or has_f16_mfma):
            return

        agpr_used = profile.get("agpr_used", 0)
        dpp_count = profile.get("dpp_count", 0)
        mfma_count = profile["mfma_count"]
        barrier_count = profile.get("barrier_count", 0)
        quad_perm = profile.get("dpp_modifiers", {}).get("quad_perm", 0)

        if mfma_count > 100 and dpp_count > 20 and quad_perm > 10:
            if agpr_used < 128 and mfma_count > 200:
                result.recommendations.append({
                    "type": "fmha_bwd_low_agpr",
                    "severity": "medium",
                    "description": (
                        f"FMHA-backward-like kernel ({mfma_count} MFMAs, "
                        f"{dpp_count} DPP with {quad_perm} quad_perm) uses only "
                        f"{agpr_used} AGPRs. Production FMHA backward uses "
                        f"a[0:255] (full 256 AGPRs) for MFMA operand storage. "
                        f"Moving data to AGPRs frees VGPRs for indexing/control."
                    ),
                    "estimated_cycle_savings": 100,
                })

        if mfma_count > 100 and barrier_count > 20:
            mfma_per_barrier = mfma_count / max(barrier_count, 1)
            if mfma_per_barrier < 3:
                result.recommendations.append({
                    "type": "fmha_excessive_barriers",
                    "severity": "medium",
                    "description": (
                        f"Only {mfma_per_barrier:.1f} MFMAs per barrier "
                        f"({mfma_count} MFMAs / {barrier_count} barriers). "
                        f"Production FMHA forward achieves 5.9 MFMAs/barrier. "
                        f"Restructure to do more compute between synchronization points."
                    ),
                    "estimated_cycle_savings": barrier_count * 20,
                })

    # ── KB-Driven Analysis ──────────────────────────────────────────

    def _analyze_kb_dpp_patterns(self, instructions: list[Instruction],
                                  result: OptimizationResult,
                                  profile: dict) -> None:
        """Use KB dpp_crosslane_patterns to enhance DPP analysis with technique
        recommendations for the kernel type."""
        if self.kb is None:
            return
        dpp = self.kb.dpp_crosslane_patterns
        techniques = dpp.get("optimization_techniques_by_kernel_type", {})
        if not techniques:
            return

        kernel_type = profile.get("kernel_type", "")
        if kernel_type not in techniques:
            return

        entry = techniques[kernel_type]
        key_techniques = entry.get("key_techniques", [])
        dpp_type = entry.get("dpp_type", "")
        scheduling = entry.get("scheduling_pattern", "")

        for tech in key_techniques:
            result.recommendations.append({
                "type": "kb_dpp_technique",
                "severity": "info",
                "description": f"[{kernel_type}] {tech}",
                "estimated_cycle_savings": 0,
                "technique_name": tech,
                "kernel_type": kernel_type,
                "source": "dpp_crosslane_patterns",
            })
        if dpp_type:
            result.recommendations.append({
                "type": "kb_dpp_type",
                "severity": "info",
                "description": f"[{kernel_type}] DPP pattern: {dpp_type}",
                "estimated_cycle_savings": 0,
                "dpp_type": dpp_type,
                "kernel_type": kernel_type,
                "source": "dpp_crosslane_patterns",
            })
        if scheduling:
            result.recommendations.append({
                "type": "kb_scheduling",
                "severity": "info",
                "description": f"[{kernel_type}] Scheduling: {scheduling}",
                "estimated_cycle_savings": 0,
                "scheduling_pattern": scheduling,
                "kernel_type": kernel_type,
                "source": "dpp_crosslane_patterns",
            })

    def _analyze_kb_fmha_techniques(self, instructions: list[Instruction],
                                    result: OptimizationResult,
                                    profile: dict) -> None:
        """Use KB fmha_asm_patterns to check which FMHA techniques are
        applied and recommend missing ones."""
        if self.kb is None:
            return
        fmha = self.kb.fmha_asm_patterns
        techniques = fmha.get("key_optimization_techniques", {})
        if not techniques:
            return

        if profile["mfma_count"] < 50:
            return
        has_bf16 = any("bf16" in t for t in profile.get("mfma_types", {}))
        has_f16 = any(
            "f16" in t and "bf16" not in t
            for t in profile.get("mfma_types", {})
        )
        if not (has_bf16 or has_f16):
            return

        dpp_count = profile.get("dpp_count", 0)
        quad_perm = profile.get("dpp_modifiers", {}).get("quad_perm", 0)
        agpr_used = profile.get("agpr_used", 0)
        lds_direct = profile.get("lds_direct_loads", 0)

        max_vmcnt = 0
        for instr in instructions:
            if instr.mnemonic == "s_waitcnt":
                m = re.search(r'vmcnt\((\d+)\)', instr.operands or "")
                if m:
                    max_vmcnt = max(max_vmcnt, int(m.group(1)))

        perm_b32_count = profile.get("perm_b32_count", 0)

        for key, tech_entry in techniques.items():
            desc = tech_entry.get("description", "")
            applicable = True
            applied = False
            reason = ""

            if "dpp" in key.lower() or "quad_perm" in desc.lower():
                if dpp_count > 20 and quad_perm > 10:
                    applied = True
                elif dpp_count == 0:
                    reason = "No DPP instructions; FMHA backward uses quad_perm for softmax gradient."
            elif "agpr" in key.lower() or "agpr" in desc.lower():
                if agpr_used >= 200:
                    applied = True
                elif agpr_used < 128:
                    reason = f"Only {agpr_used} AGPRs used; production FMHA backward uses full a[0:255]."
            elif "lds" in key.lower() or "direct" in desc.lower():
                if lds_direct > 0:
                    applied = True
                else:
                    reason = "No direct-to-LDS loads; production FMHA uses buffer_load_dword ... lds for K tiles."
            elif "waitcnt" in key.lower() or "vmcnt" in desc.lower():
                if max_vmcnt > 10:
                    applied = True
                else:
                    reason = f"vmcnt max {max_vmcnt}; production uses vmcnt(32) for deeper pipelining."
            elif "perm" in key.lower() or "v_perm" in desc.lower():
                if perm_b32_count > 0:
                    applied = True
                else:
                    reason = "No v_perm_b32 for BF16 packing between MFMA stages."
            elif "interleav" in key.lower() or "mfma" in desc.lower():
                applied = True

            if applicable and not applied and reason:
                result.recommendations.append({
                    "type": "kb_fmha_missing_technique",
                    "severity": "medium",
                    "description": f"FMHA technique '{key}': {desc}. {reason}",
                    "estimated_cycle_savings": 50,
                    "technique_key": key,
                    "technique_details": tech_entry,
                    "source": "fmha_asm_patterns",
                })

    # ── Load Vectorization Analysis ──────────────────────────────────

    def _analyze_load_vectorization(self, instructions: list[Instruction],
                                     result: OptimizationResult,
                                     profile: dict) -> None:
        """Check for scalar load opportunities.

        Production bshuffle variant uses buffer_load_dwordx4 for 4x bandwidth.
        """
        consecutive_scalar = []
        for i, instr in enumerate(instructions):
            is_scalar = (instr.mnemonic in ("global_load_dword", "buffer_load_dword")
                         and "lds" not in (instr.full_text or ""))
            if is_scalar:
                consecutive_scalar.append(i)
            else:
                if len(consecutive_scalar) >= 4:
                    result.recommendations.append({
                        "type": "load_vectorization",
                        "severity": "medium",
                        "description": (
                            f"{len(consecutive_scalar)} consecutive scalar loads at "
                            f"[{consecutive_scalar[0]}-{consecutive_scalar[-1]}]. "
                            f"Merge into dwordx4 for 4x bandwidth. "
                            f"Production bshuffle GEMM uses buffer_load_dwordx4."
                        ),
                        "estimated_cycle_savings": len(consecutive_scalar) * 2,
                        "indices": consecutive_scalar[:8],
                    })
                consecutive_scalar = []

        if len(consecutive_scalar) >= 4:
            result.recommendations.append({
                "type": "load_vectorization",
                "severity": "medium",
                "description": (
                    f"{len(consecutive_scalar)} consecutive scalar loads at "
                    f"[{consecutive_scalar[0]}-{consecutive_scalar[-1]}]. "
                    f"Merge into dwordx4."
                ),
                "estimated_cycle_savings": len(consecutive_scalar) * 2,
                "indices": consecutive_scalar[:8],
            })
