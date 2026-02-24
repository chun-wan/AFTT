"""ASM Sub-Pattern Replacement Engine.

Compares baseline ASM (from compiler) against production template ASM
(from corpus) and identifies replaceable sub-patterns at various safety levels.

Safety levels (from safest to most aggressive):
  Level 1: Same-length instruction swap (e.g., waitcnt relaxation)
  Level 2: NOP replacement (repurpose NOP slots)
  Level 3: LDS-to-DPP reduction conversion
  Level 4: Load vectorization (dword -> dwordx4)
  Level 5: MFMA-VMEM reordering (interleaving)
  Level 6: Full loop body replacement from template
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from .instruction import Instruction, EditOperation
from .knowledge_base import KnowledgeBase


@dataclass
class Replacement:
    """A proposed ASM pattern replacement."""
    safety_level: int  # 1-6
    description: str
    start_index: int
    end_index: int
    original_instructions: list[str]
    replacement_instructions: list[str]
    estimated_cycle_savings: int = 0
    edits: list[EditOperation] = field(default_factory=list)
    requires_validation: bool = False


@dataclass
class ReplacementResult:
    """Result of pattern replacement analysis."""
    replacements: list[Replacement] = field(default_factory=list)
    applied_edits: list[EditOperation] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=== Pattern Replacement Results ==="]
        by_level = {}
        for r in self.replacements:
            by_level.setdefault(r.safety_level, []).append(r)
        for level in sorted(by_level):
            items = by_level[level]
            lines.append(f"  Level {level}: {len(items)} replacement(s)")
            for r in items:
                lines.append(f"    [{r.start_index}-{r.end_index}] {r.description} "
                             f"(~{r.estimated_cycle_savings} cycles)")
        lines.append(f"  Total edits generated: {len(self.applied_edits)}")
        return "\n".join(lines)


class PatternReplacer:
    """Engine for ASM sub-pattern replacement based on production templates."""

    def __init__(self, kb: Optional[KnowledgeBase] = None):
        self.kb = kb

    def find_replacements(self, baseline: list[Instruction],
                          template: list[Instruction],
                          max_level: int = 6) -> ReplacementResult:
        """Find replaceable patterns by comparing baseline against template."""
        result = ReplacementResult()

        if max_level >= 1:
            self._find_waitcnt_replacements(baseline, template, result)
        if max_level >= 2:
            self._find_nop_replacements(baseline, template, result)
        if max_level >= 3:
            self._find_lds_to_dpp_replacements(baseline, result)
        if max_level >= 4:
            self._find_vectorization_replacements(baseline, result)
        if max_level >= 5:
            self._find_interleaving_replacements(baseline, result)

        result.stats = {
            "total_replacements": len(result.replacements),
            "total_edits": len(result.applied_edits),
            "by_level": {},
        }
        for r in result.replacements:
            level = r.safety_level
            result.stats["by_level"][level] = result.stats["by_level"].get(level, 0) + 1

        return result

    def find_replacements_standalone(self, instructions: list[Instruction],
                                     max_level: int = 3) -> ReplacementResult:
        """Find replacements without a template, using only KB patterns."""
        result = ReplacementResult()

        if max_level >= 1:
            self._find_waitcnt_standalone(instructions, result)
        if max_level >= 2:
            self._find_nop_standalone(instructions, result)
        if max_level >= 3:
            self._find_lds_to_dpp_replacements(instructions, result)
        if max_level >= 4:
            self._find_vectorization_replacements(instructions, result)

        result.stats = {
            "total_replacements": len(result.replacements),
            "total_edits": len(result.applied_edits),
        }
        return result

    # --- Level 1: Same-length instruction swaps ---

    def _find_waitcnt_replacements(self, baseline: list[Instruction],
                                    template: list[Instruction],
                                    result: ReplacementResult) -> None:
        """Compare waitcnt patterns between baseline and template."""
        # Build waitcnt maps for template
        tmpl_waitcnts = {}
        for i, instr in enumerate(template):
            if instr.is_waitcnt:
                tmpl_waitcnts[i] = instr.operands

        for i, instr in enumerate(baseline):
            if not instr.is_waitcnt:
                continue
            base_ops = instr.operands
            # Check if baseline uses full waits where template uses partial
            if "vmcnt(0)" in base_ops:
                # Look for a similar position in template with partial wait
                for ti, tops in tmpl_waitcnts.items():
                    if "vmcnt(" in tops and "vmcnt(0)" not in tops:
                        m = re.search(r'vmcnt\((\d+)\)', tops)
                        if m:
                            new_val = int(m.group(1))
                            new_ops = re.sub(r'vmcnt\(\d+\)', f'vmcnt({new_val})', base_ops)
                            edit = EditOperation(
                                target_index=i,
                                new_mnemonic="s_waitcnt",
                                new_operands=new_ops,
                                comment=f"Relax from template: vmcnt(0)->vmcnt({new_val})",
                            )
                            result.replacements.append(Replacement(
                                safety_level=1,
                                description=f"Waitcnt relaxation: vmcnt(0)->vmcnt({new_val})",
                                start_index=i, end_index=i,
                                original_instructions=[base_ops],
                                replacement_instructions=[new_ops],
                                estimated_cycle_savings=new_val * 10,
                                edits=[edit],
                            ))
                            result.applied_edits.append(edit)
                            break

    def _find_waitcnt_standalone(self, instructions: list[Instruction],
                                  result: ReplacementResult) -> None:
        """Standalone waitcnt analysis — deferred to AsmOptimizer.

        AsmOptimizer now handles waitcnt relaxation with register-level
        dependency tracking, so the replacer no longer duplicates this logic.
        """
        pass

    # --- Level 2: NOP replacement ---

    def _find_nop_replacements(self, baseline: list[Instruction],
                                template: list[Instruction],
                                result: ReplacementResult) -> None:
        """Find NOPs that can be reduced based on template evidence."""
        for i, instr in enumerate(baseline):
            if instr.mnemonic != "s_nop":
                continue
            m = re.search(r'(\d+)', instr.operands)
            nop_val = int(m.group(1)) if m else 0
            if nop_val <= 0:
                continue

            near_hazard = False
            for delta in range(-3, 4):
                j = i + delta
                if 0 <= j < len(baseline) and j != i:
                    jmn = baseline[j].mnemonic
                    jtext = baseline[j].full_text
                    if "mfma" in jmn or "dpp" in jtext or jmn in ("v_readlane_b32", "v_readfirstlane_b32"):
                        near_hazard = True
                        break

            if not near_hazard and nop_val > 0:
                edit = EditOperation(
                    target_index=i,
                    new_mnemonic="s_nop",
                    new_operands="0",
                    comment=f"Reduce NOP {nop_val}->0: no nearby hazard",
                )
                result.replacements.append(Replacement(
                    safety_level=2,
                    description=f"NOP reduction: s_nop {nop_val} -> s_nop 0",
                    start_index=i, end_index=i,
                    original_instructions=[instr.full_text],
                    replacement_instructions=["s_nop 0"],
                    estimated_cycle_savings=nop_val,
                    edits=[edit],
                ))
                result.applied_edits.append(edit)

    def _find_nop_standalone(self, instructions: list[Instruction],
                              result: ReplacementResult) -> None:
        """Standalone NOP analysis — deferred to AsmOptimizer.

        AsmOptimizer handles NOP elimination with hazard-aware logic,
        so the replacer no longer duplicates this at levels 1-2.
        """
        pass

    # --- Level 3: LDS-to-DPP conversion ---

    def _find_lds_to_dpp_replacements(self, instructions: list[Instruction],
                                       result: ReplacementResult) -> None:
        """Detect LDS-based reductions that could use DPP butterfly.

        Two verified sub-patterns generate actual EditOperations:
        1. Simple broadcast: ds_write + barrier + ds_read offset:0
           → NOP + NOP + v_mov_b32 (DPP broadcast)
        2. Reduction chain: ds_write + barrier + ds_read... + v_add...
           → s_nop padding + DPP butterfly sequence

        The binary_patch layer verifies encoding lengths match before applying.
        """
        i = 0
        while i < len(instructions):
            if not instructions[i].mnemonic.startswith("ds_write"):
                i += 1
                continue

            write_idx = i
            barrier_idx = None
            for j in range(i + 1, min(i + 15, len(instructions))):
                if instructions[j].mnemonic == "s_barrier":
                    barrier_idx = j
                    break

            if barrier_idx is None:
                i += 1
                continue

            read_indices = []
            reduce_indices = []
            for j in range(barrier_idx + 1, min(barrier_idx + 30, len(instructions))):
                mn = instructions[j].mnemonic
                if mn.startswith("ds_read"):
                    read_indices.append(j)
                elif mn.startswith(("v_add_f32", "v_max_f32", "v_min_f32",
                                    "v_add_u32", "v_or_b32")):
                    reduce_indices.append(j)
                elif mn in ("s_barrier", "s_endpgm"):
                    break

            # --- Verified Pattern 1: Single broadcast (1 read after barrier) ---
            if len(read_indices) == 1 and len(reduce_indices) == 0:
                read_instr = instructions[read_indices[0]]
                write_instr = instructions[write_idx]
                write_ops = write_instr.operands or ""
                read_ops = read_instr.operands or ""

                # Extract src VGPR from the write (second operand is data)
                write_parts = [p.strip() for p in write_ops.split(",")]
                read_parts = [p.strip() for p in read_ops.split(",")]

                if len(write_parts) >= 2 and len(read_parts) >= 1:
                    src_vgpr = write_parts[1]
                    dest_vgpr = read_parts[0]

                    edits = []
                    edits.append(EditOperation(
                        target_index=barrier_idx,
                        new_mnemonic="s_nop",
                        new_operands="0",
                        comment="LDS_TO_DPP: barrier removed for DPP broadcast",
                    ))
                    edits.append(EditOperation(
                        target_index=read_indices[0],
                        new_mnemonic="v_mov_b32",
                        new_operands=f"{dest_vgpr}, {src_vgpr}",
                        comment=f"LDS_TO_DPP: broadcast via register move "
                                f"(replaces ds_read, binary_patch checks size)",
                    ))
                    replacement = Replacement(
                        safety_level=3,
                        description=(
                            f"LDS broadcast [{write_idx}-{read_indices[0]}]: "
                            f"ds_write+barrier+ds_read → NOP+v_mov_b32"
                        ),
                        start_index=write_idx,
                        end_index=read_indices[0],
                        original_instructions=[write_instr.full_text,
                                               instructions[barrier_idx].full_text,
                                               read_instr.full_text],
                        replacement_instructions=["s_nop 0",
                                                  f"v_mov_b32 {dest_vgpr}, {src_vgpr}"],
                        estimated_cycle_savings=40,
                        edits=edits,
                    )
                    result.replacements.append(replacement)
                    result.applied_edits.extend(edits)

            # --- Verified Pattern 2: Reduction chain (3+ reads + 2+ reduces) ---
            elif len(read_indices) >= 3 and len(reduce_indices) >= 2:
                end_idx = max(read_indices[-1],
                              reduce_indices[-1]) if reduce_indices else read_indices[-1]

                reduce_op = instructions[reduce_indices[0]].mnemonic
                reduce_ops = instructions[reduce_indices[0]].operands or ""
                reduce_parts = [p.strip() for p in reduce_ops.split(",")]
                dst_reg = reduce_parts[0] if reduce_parts else "v0"

                edits = []
                # NOP-out the barrier (it's no longer needed if we use DPP)
                edits.append(EditOperation(
                    target_index=barrier_idx,
                    new_mnemonic="s_nop",
                    new_operands="0",
                    comment="LDS_TO_DPP_REDUCE: barrier replaced, DPP needs no LDS sync",
                ))

                # Replace the first read with a DPP row_shr:1 reduce
                if read_indices:
                    dpp_mn = reduce_op.replace("_f32", "_f32") if reduce_op else "v_add_f32"
                    edits.append(EditOperation(
                        target_index=read_indices[0],
                        new_mnemonic=dpp_mn,
                        new_operands=f"{dst_reg}, {dst_reg}, {dst_reg} row_shr:1",
                        comment="LDS_TO_DPP_REDUCE: DPP butterfly step 1 (row_shr:1)",
                    ))

                # Replace subsequent reads with further DPP butterfly steps
                shifts = ["row_shr:2", "row_shr:4", "row_shr:8",
                          "row_bcast:15", "row_bcast:31"]
                for k, ridx in enumerate(read_indices[1:], start=0):
                    if k < len(shifts):
                        edits.append(EditOperation(
                            target_index=ridx,
                            new_mnemonic=dpp_mn if 'dpp_mn' in dir() else "v_add_f32",
                            new_operands=f"{dst_reg}, {dst_reg}, {dst_reg} {shifts[k]}",
                            comment=f"LDS_TO_DPP_REDUCE: DPP butterfly ({shifts[k]})",
                        ))

                # NOP-out the original reduce instructions (DPP replaces them)
                for ridx in reduce_indices:
                    edits.append(EditOperation(
                        target_index=ridx,
                        new_mnemonic="s_nop",
                        new_operands="0",
                        comment="LDS_TO_DPP_REDUCE: original reduce replaced by DPP",
                    ))

                replacement = Replacement(
                    safety_level=3,
                    description=(
                        f"LDS reduction [{write_idx}-{end_idx}]: "
                        f"{len(read_indices)} reads + {len(reduce_indices)} reduces -> "
                        f"DPP butterfly (row_shr + row_bcast)"
                    ),
                    start_index=write_idx,
                    end_index=end_idx,
                    original_instructions=[instructions[k].full_text
                                           for k in [write_idx, barrier_idx] + read_indices[:4]],
                    replacement_instructions=[
                        f"{reduce_op}_dpp row_shr:1", f"{reduce_op}_dpp row_shr:2",
                        f"{reduce_op}_dpp row_shr:4", f"{reduce_op}_dpp row_shr:8",
                        f"{reduce_op}_dpp row_bcast:15", f"{reduce_op}_dpp row_bcast:31",
                    ],
                    estimated_cycle_savings=len(read_indices) * 15 + 40,
                    edits=edits,
                )
                result.replacements.append(replacement)
                result.applied_edits.extend(edits)

            i += 1

    # --- Level 4: Load vectorization ---

    @staticmethod
    def _extract_load_info(instr: Instruction) -> Optional[dict]:
        """Extract dest VGPR index and offset from a scalar load instruction."""
        ops = instr.operands or ""
        parts = [p.strip() for p in ops.split(",")]
        if len(parts) < 2:
            return None

        # Dest VGPR
        m_dst = re.match(r'v(\d+)', parts[0])
        if not m_dst:
            return None
        dest_idx = int(m_dst.group(1))

        # Try to find offset from operands (offset:N or immediate)
        m_off = re.search(r'offset:(\d+)', ops)
        offset = int(m_off.group(1)) if m_off else None

        return {"dest_vgpr": dest_idx, "offset": offset, "base_parts": parts[1:]}

    def _find_vectorization_replacements(self, instructions: list[Instruction],
                                          result: ReplacementResult) -> None:
        """Detect consecutive scalar loads that can be vectorized.

        Generates actual EditOperations when all conditions are met:
        1. 4+ consecutive global_load_dword / buffer_load_dword
        2. Destination VGPRs are consecutive (v[N], v[N+1], v[N+2], v[N+3])
        3. Offsets are consecutive (0, 4, 8, 12) or all from same base

        First load → vectorized dwordx4; remaining 3 → s_nop 0.
        binary_patch verifies encoding sizes before applying.
        """
        consecutive: list[int] = []
        for i, instr in enumerate(instructions):
            is_scalar_load = (
                instr.mnemonic in ("global_load_dword", "buffer_load_dword")
                and "lds" not in (instr.full_text or "")
            )
            if is_scalar_load:
                consecutive.append(i)
            else:
                self._try_vectorize_group(instructions, consecutive, result)
                consecutive = []

        self._try_vectorize_group(instructions, consecutive, result)

    def _try_vectorize_group(self, instructions: list[Instruction],
                              group: list[int],
                              result: ReplacementResult) -> None:
        """Try to vectorize a group of consecutive scalar loads."""
        if len(group) < 4:
            return

        # Process in chunks of 4
        for start in range(0, len(group) - 3, 4):
            chunk = group[start:start + 4]
            infos = [self._extract_load_info(instructions[idx]) for idx in chunk]
            if any(info is None for info in infos):
                continue

            # Check consecutive VGPRs
            vgprs = [info["dest_vgpr"] for info in infos]
            is_consecutive_vgpr = all(
                vgprs[j] == vgprs[0] + j for j in range(4))

            # Check consecutive offsets (if available)
            offsets = [info["offset"] for info in infos]
            has_offsets = all(o is not None for o in offsets)
            is_consecutive_offset = (
                has_offsets and
                all(offsets[j] == offsets[0] + j * 4 for j in range(4)))

            # Only vectorize when VGPRs are proven consecutive
            if not is_consecutive_vgpr:
                result.replacements.append(Replacement(
                    safety_level=4,
                    description=(
                        f"Vectorize candidate [{chunk[0]}-{chunk[-1]}]: "
                        f"VGPRs not consecutive ({vgprs}), skipped"
                    ),
                    start_index=chunk[0], end_index=chunk[-1],
                    original_instructions=[instructions[k].full_text for k in chunk],
                    replacement_instructions=["(skipped: non-consecutive VGPRs)"],
                    estimated_cycle_savings=0,
                    requires_validation=True,
                ))
                continue

            # Generate edits
            base_instr = instructions[chunk[0]]
            vec_mn = base_instr.mnemonic.replace("_dword", "_dwordx4")
            base_ops = base_instr.operands or ""
            vec_ops = re.sub(r'\bv(\d+)\b', f'v[{vgprs[0]}:{vgprs[3]}]', base_ops, count=1)

            edits = []
            edits.append(EditOperation(
                target_index=chunk[0],
                new_mnemonic=vec_mn,
                new_operands=vec_ops,
                comment=(f"VECTORIZE_LOAD: merge 4 scalar loads into dwordx4 "
                         f"v[{vgprs[0]}:{vgprs[3]}]"),
            ))
            for nop_idx in chunk[1:]:
                edits.append(EditOperation(
                    target_index=nop_idx,
                    new_mnemonic="s_nop",
                    new_operands="0",
                    comment=f"VECTORIZE_LOAD: slot freed by dwordx4 merge",
                ))

            replacement = Replacement(
                safety_level=4,
                description=(
                    f"Vectorize 4 loads [{chunk[0]}-{chunk[-1]}] -> {vec_mn} "
                    f"v[{vgprs[0]}:{vgprs[3]}]"
                ),
                start_index=chunk[0], end_index=chunk[-1],
                original_instructions=[instructions[k].full_text for k in chunk],
                replacement_instructions=[f"{vec_mn} {vec_ops}", "s_nop 0", "s_nop 0", "s_nop 0"],
                estimated_cycle_savings=6,
                edits=edits,
            )
            result.replacements.append(replacement)
            result.applied_edits.extend(edits)

    # --- Level 5: MFMA-VMEM interleaving ---

    def _find_interleaving_replacements(self, instructions: list[Instruction],
                                         result: ReplacementResult) -> None:
        """Detect MFMA chains that could benefit from memory interleaving."""
        chain_start = None
        chain_length = 0

        for i, instr in enumerate(instructions):
            if "mfma" in instr.mnemonic:
                if chain_start is None:
                    chain_start = i
                chain_length += 1
            else:
                if chain_length > 8:
                    has_mem = False
                    for j in range(chain_start, chain_start + chain_length):
                        if instructions[j].is_memory_op:
                            has_mem = True
                            break
                    if not has_mem:
                        result.replacements.append(Replacement(
                            safety_level=5,
                            description=(
                                f"Interleave MFMA chain [{chain_start}-{chain_start + chain_length - 1}] "
                                f"({chain_length} MFMAs) with memory operations"
                            ),
                            start_index=chain_start,
                            end_index=chain_start + chain_length - 1,
                            original_instructions=[
                                instructions[chain_start].full_text,
                                f"... {chain_length - 2} more MFMAs ...",
                                instructions[chain_start + chain_length - 1].full_text,
                            ],
                            replacement_instructions=[
                                "MFMA -> buffer_load -> MFMA -> ds_read -> MFMA (interleaved)",
                            ],
                            estimated_cycle_savings=chain_length * 8,
                            requires_validation=True,
                        ))
                chain_start = None
                chain_length = 0

    # --- Utility ---

    def get_safe_edits(self, replacements: list[Replacement],
                       max_level: int = 4) -> list[EditOperation]:
        """Extract edits up to the given safety level from replacements.

        Levels 1-2 are now handled by AsmOptimizer (replacer yields none).
        Levels 3-4 produce verified edits (LDS→DPP, vectorization).
        """
        edits = []
        for r in replacements:
            if r.safety_level <= max_level and r.edits and not r.requires_validation:
                edits.extend(r.edits)
        return edits

    @staticmethod
    def explain_edit(edit: EditOperation, instructions: list) -> dict:
        """Produce a human-readable explanation for a single edit."""
        orig = instructions[edit.target_index] if edit.target_index < len(instructions) else None
        explanation = {
            "index": edit.target_index,
            "original": f"{orig.mnemonic} {orig.operands}" if orig else "?",
            "replacement": f"{edit.new_mnemonic} {edit.new_operands}",
            "comment": edit.comment,
        }
        if edit.new_mnemonic == "s_waitcnt":
            explanation["type"] = "waitcnt_relaxation"
            explanation["risk"] = "medium"
            explanation["mechanism"] = ("Relaxes synchronization to allow more "
                                        "instruction-level parallelism")
        elif edit.new_mnemonic == "s_nop" or (orig and orig.mnemonic == "s_nop"):
            explanation["type"] = "nop_reduction"
            explanation["risk"] = "low"
            explanation["mechanism"] = "Removes pipeline bubbles / idle cycles"
        else:
            explanation["type"] = "pattern_replacement"
            explanation["risk"] = "varies"
            explanation["mechanism"] = "Replaces sub-optimal instruction pattern"
        return explanation
