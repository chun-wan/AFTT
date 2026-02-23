"""Static Pipeline-Aware Cycle Estimator for AMDGPU ASM.

Models the execution pipeline of CDNA3/CDNA4 GPUs to estimate total cycles
for an instruction sequence. Tracks MFMA/VALU/VMEM/LDS/SALU issue widths,
dual-issue capabilities, and s_waitcnt dependencies.

Pipeline model:
- SALU can dual-issue with any other pipe (VALU, MFMA, VMEM, LDS)
- MFMA has 64-cycle latency, 4 MFMA units (can overlap)
- VMEM has ~100-300 cycle latency (tracked by vmcnt)
- LDS has ~20 cycle latency (tracked by lgkmcnt)
- VALU has 4-8 cycle latency, 1 issue per cycle
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .isa_db import ISADatabase


@dataclass
class PipelineState:
    """Tracks in-flight operations across all pipeline units."""
    cycle: int = 0
    # Outstanding operation counters (like hardware counters)
    vmcnt: int = 0      # outstanding VMEM loads
    lgkmcnt: int = 0    # outstanding LDS/SMEM ops
    vscnt: int = 0      # outstanding VMEM stores

    # In-flight MFMA tracking (up to 4 overlapping)
    mfma_finish_cycles: list[int] = field(default_factory=list)

    # Pipe busy-until tracking
    valu_busy_until: int = 0
    salu_busy_until: int = 0
    vmem_busy_until: int = 0
    lds_busy_until: int = 0
    mfma_busy_until: int = 0


@dataclass
class CycleEstimate:
    """Result of cycle estimation for an instruction sequence."""
    total_cycles: int
    mfma_cycles: int
    valu_cycles: int
    vmem_cycles: int
    lds_cycles: int
    salu_cycles: int
    wait_stall_cycles: int
    barrier_stall_cycles: int
    nop_cycles: int
    instruction_count: int
    mfma_count: int
    vmem_count: int
    lds_count: int

    @property
    def compute_intensity(self) -> float:
        """Ratio of MFMA compute cycles to total."""
        return self.mfma_cycles / max(self.total_cycles, 1)

    @property
    def bottleneck(self) -> str:
        """Identify the primary bottleneck."""
        pipes = {
            "MFMA (compute-bound)": self.mfma_cycles,
            "VMEM (memory-bound)": self.vmem_cycles + self.wait_stall_cycles,
            "LDS (LDS-bound)": self.lds_cycles,
            "VALU (ALU-bound)": self.valu_cycles,
        }
        return max(pipes, key=pipes.get)

    def to_dict(self) -> dict:
        return {
            "total_cycles": self.total_cycles,
            "mfma_cycles": self.mfma_cycles,
            "valu_cycles": self.valu_cycles,
            "vmem_cycles": self.vmem_cycles,
            "lds_cycles": self.lds_cycles,
            "salu_cycles": self.salu_cycles,
            "wait_stall_cycles": self.wait_stall_cycles,
            "barrier_stall_cycles": self.barrier_stall_cycles,
            "nop_cycles": self.nop_cycles,
            "instruction_count": self.instruction_count,
            "mfma_count": self.mfma_count,
            "vmem_count": self.vmem_count,
            "lds_count": self.lds_count,
            "compute_intensity": round(self.compute_intensity, 3),
            "bottleneck": self.bottleneck,
        }

    def summary(self) -> str:
        lines = [
            f"=== Cycle Estimate ({self.instruction_count} instructions) ===",
            f"Total cycles:       {self.total_cycles:,}",
            f"  MFMA cycles:      {self.mfma_cycles:,} ({self.mfma_count} ops)",
            f"  VALU cycles:      {self.valu_cycles:,}",
            f"  VMEM cycles:      {self.vmem_cycles:,} ({self.vmem_count} ops)",
            f"  LDS cycles:       {self.lds_cycles:,} ({self.lds_count} ops)",
            f"  SALU cycles:      {self.salu_cycles:,}",
            f"  Wait stalls:      {self.wait_stall_cycles:,}",
            f"  Barrier stalls:   {self.barrier_stall_cycles:,}",
            f"  NOP cycles:       {self.nop_cycles:,}",
            f"Compute intensity:  {self.compute_intensity:.1%}",
            f"Bottleneck:         {self.bottleneck}",
        ]
        return "\n".join(lines)


def _classify_instruction(mnemonic: str) -> str:
    """Classify an instruction mnemonic into a pipeline category."""
    mn = mnemonic.lower().strip()
    if mn.startswith("v_mfma") or mn.startswith("v_smfma"):
        return "MFMA"
    if mn.startswith("v_pk_") or mn.startswith("v_dot"):
        return "VALU"
    if mn.startswith("v_"):
        return "VALU"
    if mn.startswith(("global_load", "global_store", "buffer_load", "buffer_store",
                      "global_atomic", "buffer_atomic")):
        return "VMEM"
    if mn.startswith(("flat_load", "flat_store", "flat_atomic")):
        return "VMEM"
    if mn.startswith("ds_"):
        return "LDS"
    if mn.startswith("s_load") or mn.startswith("s_store") or mn.startswith("s_buffer"):
        return "SMEM"
    if mn.startswith("s_"):
        return "SALU"
    return "VALU"


def _parse_waitcnt(operands: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Parse s_waitcnt operands to extract vmcnt, lgkmcnt, vscnt values."""
    vmcnt = lgkmcnt = vscnt = None
    m = re.search(r'vmcnt\((\d+)\)', operands)
    if m:
        vmcnt = int(m.group(1))
    m = re.search(r'lgkmcnt\((\d+)\)', operands)
    if m:
        lgkmcnt = int(m.group(1))
    m = re.search(r'vscnt\((\d+)\)', operands)
    if m:
        vscnt = int(m.group(1))
    return vmcnt, lgkmcnt, vscnt


def _parse_nop(operands: str) -> int:
    """Parse s_nop operand to get number of NOP cycles (N+1)."""
    m = re.search(r'(\d+)', operands)
    return int(m.group(1)) + 1 if m else 1


class CycleEstimator:
    """Pipeline-aware static cycle estimator for AMDGPU instruction sequences."""

    def __init__(self, arch: str = "gfx942", isa_db: Optional[ISADatabase] = None):
        self.arch = arch
        self.isa_db = isa_db or ISADatabase()
        self.isa_db.load()

        model = self.isa_db.get_pipeline_model(arch) or {}
        self.mfma_latency = model.get("mfma_latency", 64)
        self.mfma_pipeline_depth = model.get("mfma_pipeline_depth", 4)
        self.vmem_latency = model.get("vmem_latency_l2_hit", 100)
        self.lds_latency = model.get("lds_latency", 20)
        self.barrier_est_cycles = 50

    def estimate(self, asm_lines: list[str]) -> CycleEstimate:
        """Estimate execution cycles for an ASM instruction sequence.

        Each line should be a single instruction, e.g.:
            "v_mfma_f32_16x16x16_bf16 a[0:3], v[0:1], v[2:3], a[0:3]"
            "s_waitcnt vmcnt(0) lgkmcnt(0)"
        """
        state = PipelineState()
        stats = {
            "mfma_cycles": 0, "valu_cycles": 0, "vmem_cycles": 0,
            "lds_cycles": 0, "salu_cycles": 0, "wait_stall": 0,
            "barrier_stall": 0, "nop_cycles": 0,
            "instr_count": 0, "mfma_count": 0, "vmem_count": 0, "lds_count": 0,
        }

        for line in asm_lines:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith(";") or line.endswith(":"):
                continue

            parts = line.split(None, 1)
            if not parts:
                continue
            mnemonic = parts[0].lower().rstrip(":")
            operands = parts[1] if len(parts) > 1 else ""

            if mnemonic.startswith(".") or mnemonic.startswith("//"):
                continue

            pipe = _classify_instruction(mnemonic)
            stats["instr_count"] += 1

            if mnemonic == "s_waitcnt":
                vmcnt, lgkmcnt, vscnt = _parse_waitcnt(operands)
                stall = self._handle_waitcnt(state, vmcnt, lgkmcnt, vscnt)
                stats["wait_stall"] += stall
                stats["salu_cycles"] += 1

            elif mnemonic == "s_waitcnt_vscnt":
                _, _, vscnt = _parse_waitcnt(operands)
                stall = self._handle_waitcnt(state, None, None, vscnt)
                stats["wait_stall"] += stall
                stats["salu_cycles"] += 1

            elif mnemonic == "s_barrier":
                stall = self.barrier_est_cycles
                state.cycle += stall
                stats["barrier_stall"] += stall

            elif mnemonic == "s_nop":
                nop_cy = _parse_nop(operands)
                state.cycle += nop_cy
                stats["nop_cycles"] += nop_cy

            elif pipe == "MFMA":
                self._issue_mfma(state)
                stats["mfma_cycles"] += self.mfma_latency
                stats["mfma_count"] += 1

            elif pipe == "VMEM":
                self._issue_vmem(state, mnemonic)
                is_store = "store" in mnemonic
                if is_store:
                    state.vscnt += 1
                else:
                    state.vmcnt += 1
                stats["vmem_cycles"] += 1
                stats["vmem_count"] += 1

            elif pipe == "LDS":
                self._issue_lds(state)
                state.lgkmcnt += 1
                stats["lds_cycles"] += 1
                stats["lds_count"] += 1

            elif pipe == "SMEM":
                state.lgkmcnt += 1
                state.cycle = max(state.cycle, state.salu_busy_until)
                state.salu_busy_until = state.cycle + 1
                state.cycle += 1
                stats["salu_cycles"] += 1

            elif pipe == "SALU":
                # SALU can dual-issue, so it uses the salu pipe independently
                state.cycle = max(state.cycle, state.salu_busy_until)
                state.salu_busy_until = state.cycle + 1
                # Don't advance main cycle — dual-issue
                stats["salu_cycles"] += 1

            elif pipe == "VALU":
                state.cycle = max(state.cycle, state.valu_busy_until)
                lat = self.isa_db.get_latency(mnemonic, self.arch)
                state.valu_busy_until = state.cycle + 1  # issue rate = 1/cy
                state.cycle += 1
                stats["valu_cycles"] += 1

            else:
                state.cycle += 1

        # Account for any still in-flight MFMA at end
        if state.mfma_finish_cycles:
            remaining = max(state.mfma_finish_cycles) - state.cycle
            if remaining > 0:
                state.cycle += remaining

        return CycleEstimate(
            total_cycles=state.cycle,
            mfma_cycles=stats["mfma_cycles"],
            valu_cycles=stats["valu_cycles"],
            vmem_cycles=stats["vmem_cycles"],
            lds_cycles=stats["lds_cycles"],
            salu_cycles=stats["salu_cycles"],
            wait_stall_cycles=stats["wait_stall"],
            barrier_stall_cycles=stats["barrier_stall"],
            nop_cycles=stats["nop_cycles"],
            instruction_count=stats["instr_count"],
            mfma_count=stats["mfma_count"],
            vmem_count=stats["vmem_count"],
            lds_count=stats["lds_count"],
        )

    def _issue_mfma(self, state: PipelineState) -> None:
        """Issue an MFMA instruction into the pipeline."""
        # Clean up completed MFMAs
        state.mfma_finish_cycles = [
            c for c in state.mfma_finish_cycles if c > state.cycle
        ]
        # If all MFMA slots full, stall until oldest completes
        if len(state.mfma_finish_cycles) >= self.mfma_pipeline_depth:
            earliest = min(state.mfma_finish_cycles)
            if earliest > state.cycle:
                state.cycle = earliest
            state.mfma_finish_cycles.remove(earliest)

        state.cycle = max(state.cycle, state.mfma_busy_until)
        state.mfma_busy_until = state.cycle + 1  # issue rate 1/cy
        state.mfma_finish_cycles.append(state.cycle + self.mfma_latency)
        state.cycle += 1

    def _issue_vmem(self, state: PipelineState, mnemonic: str) -> None:
        """Issue a VMEM instruction."""
        state.cycle = max(state.cycle, state.vmem_busy_until)
        state.vmem_busy_until = state.cycle + 1
        state.cycle += 1

    def _issue_lds(self, state: PipelineState) -> None:
        """Issue an LDS instruction."""
        state.cycle = max(state.cycle, state.lds_busy_until)
        state.lds_busy_until = state.cycle + 1
        state.cycle += 1

    def _handle_waitcnt(self, state: PipelineState,
                        vmcnt: Optional[int], lgkmcnt: Optional[int],
                        vscnt: Optional[int]) -> int:
        """Handle s_waitcnt by stalling until counters reach target values."""
        stall = 0

        if vmcnt is not None and state.vmcnt > vmcnt:
            # Need to wait for (vmcnt_current - vmcnt_target) VMEM ops
            ops_to_wait = state.vmcnt - vmcnt
            wait_cycles = ops_to_wait * (self.vmem_latency // max(state.vmcnt, 1))
            wait_cycles = min(wait_cycles, self.vmem_latency)
            stall += wait_cycles
            state.vmcnt = vmcnt

        if lgkmcnt is not None and state.lgkmcnt > lgkmcnt:
            ops_to_wait = state.lgkmcnt - lgkmcnt
            wait_cycles = ops_to_wait * (self.lds_latency // max(state.lgkmcnt, 1))
            wait_cycles = min(wait_cycles, self.lds_latency)
            stall += wait_cycles
            state.lgkmcnt = lgkmcnt

        if vscnt is not None and state.vscnt > vscnt:
            ops_to_wait = state.vscnt - vscnt
            stall += ops_to_wait * 2
            state.vscnt = vscnt

        state.cycle += stall
        return stall

    def estimate_from_file(self, asm_path: str) -> CycleEstimate:
        """Estimate cycles from a disassembled ASM file."""
        with open(asm_path) as f:
            lines = f.readlines()
        # Filter to instruction lines (skip labels, directives, comments, blanks)
        instr_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(("//", ";", ".", "#")):
                continue
            if stripped.endswith(":"):
                continue
            # Skip hex dump lines from objdump output
            if re.match(r'^[0-9a-f]+:', stripped):
                parts = stripped.split("\t")
                if len(parts) >= 3:
                    instr_lines.append(parts[2].strip())
                continue
            instr_lines.append(stripped)
        return self.estimate(instr_lines)

    def compare(self, original: CycleEstimate, modified: CycleEstimate) -> dict:
        """Compare two cycle estimates and produce improvement metrics."""
        diff = original.total_cycles - modified.total_cycles
        pct = diff / max(original.total_cycles, 1) * 100
        return {
            "original_cycles": original.total_cycles,
            "modified_cycles": modified.total_cycles,
            "cycle_reduction": diff,
            "improvement_pct": round(pct, 2),
            "original_bottleneck": original.bottleneck,
            "modified_bottleneck": modified.bottleneck,
            "details": {
                "mfma_diff": original.mfma_cycles - modified.mfma_cycles,
                "wait_stall_diff": original.wait_stall_cycles - modified.wait_stall_cycles,
                "nop_diff": original.nop_cycles - modified.nop_cycles,
                "instr_count_diff": original.instruction_count - modified.instruction_count,
            },
        }
