"""Unified Instruction Model for AMDGPU Assembly.

Provides a single Instruction dataclass that replaces the two incompatible
AsmInstruction definitions from parser.py (text-mode) and asm_editor.py
(binary-mode). Supports both analysis workflows and binary patching.

Also contains shared dataclasses: KernelInfo, EditOperation, BasicBlock,
KernelMetadata, RegisterUsage, and ParsedKernel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Instruction:
    """Unified AMDGPU assembly instruction representation.

    Supports two modes:
    - Text mode: populated by parser from ASM text (raw_text, operands_list, registers)
    - Binary mode: populated by disassembler from .co (address, raw_bytes, file_offset)

    Both modes share: line_number, mnemonic, category, and boolean flags.
    """
    line_number: int = 0
    mnemonic: str = ""
    category: str = ""

    # Text mode fields (from parser)
    raw_text: str = ""
    operands_list: list[str] = field(default_factory=list)
    encoding: str = ""
    label: str = ""
    comment: str = ""
    dst_registers: list[str] = field(default_factory=list)
    src_registers: list[str] = field(default_factory=list)

    # Binary mode fields (from disassembly)
    address: int = 0
    raw_bytes: bytes = b""
    file_offset: int = 0
    operands_str: str = ""

    # Boolean flags
    is_memory_op: bool = False
    is_lds_op: bool = False
    is_barrier: bool = False
    is_waitcnt: bool = False
    is_branch: bool = False
    is_mfma: bool = False

    @property
    def is_vector(self) -> bool:
        return self.mnemonic.startswith("v_")

    @property
    def is_scalar(self) -> bool:
        return self.mnemonic.startswith("s_")

    @property
    def size(self) -> int:
        return len(self.raw_bytes)

    @property
    def operands(self) -> str:
        """Return operand string, preferring operands_str for binary mode."""
        if self.operands_str:
            return self.operands_str
        return ", ".join(self.operands_list)

    @property
    def full_text(self) -> str:
        """Full instruction text for display and cycle estimation."""
        if self.raw_text:
            return self.raw_text
        ops = self.operands
        return f"{self.mnemonic} {ops}".strip()

    @classmethod
    def from_parser_line(
        cls,
        line_number: int,
        raw_text: str,
        mnemonic: str,
        operands: list[str],
        category: str,
        comment: str = "",
        is_memory_op: bool = False,
        is_lds_op: bool = False,
        is_barrier: bool = False,
        is_waitcnt: bool = False,
        is_branch: bool = False,
        is_mfma: bool = False,
    ) -> Instruction:
        return cls(
            line_number=line_number,
            raw_text=raw_text,
            mnemonic=mnemonic,
            operands_list=operands,
            category=category,
            comment=comment,
            is_memory_op=is_memory_op,
            is_lds_op=is_lds_op,
            is_barrier=is_barrier,
            is_waitcnt=is_waitcnt,
            is_branch=is_branch,
            is_mfma=is_mfma,
        )

    @classmethod
    def from_disassembly(
        cls,
        address: int,
        raw_bytes: bytes,
        mnemonic: str,
        operands: str,
        line_number: int = 0,
        file_offset: int = 0,
    ) -> Instruction:
        mn = mnemonic.strip()
        cat = _classify_mnemonic(mn)
        return cls(
            line_number=line_number,
            mnemonic=mn,
            category=cat,
            address=address,
            raw_bytes=raw_bytes,
            file_offset=file_offset,
            operands_str=operands.strip().rstrip(","),
            is_memory_op=cat in ("VMEM", "FLAT", "SMEM"),
            is_lds_op=cat == "LDS",
            is_barrier="barrier" in mn,
            is_waitcnt="waitcnt" in mn,
            is_branch="branch" in mn or mn == "s_branch",
            is_mfma="mfma" in mn,
        )

    def __repr__(self) -> str:
        if self.raw_bytes:
            hex_bytes = self.raw_bytes.hex().upper()
            return f"0x{self.address:08X}: {hex_bytes:16s} {self.full_text}"
        return f"L{self.line_number}: {self.full_text}"


@dataclass
class KernelInfo:
    """Metadata about a kernel in a .co file."""
    name: str
    text_vma: int
    text_offset: int
    text_size: int
    arch: str = "gfx942"


@dataclass
class EditOperation:
    """A single edit to apply to an instruction."""
    target_index: int
    new_mnemonic: str
    new_operands: str
    comment: str = ""


@dataclass
class BasicBlock:
    """A basic block of sequential instructions."""
    label: str
    instructions: list[Instruction] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)

    @property
    def instruction_count(self) -> int:
        return len(self.instructions)


@dataclass
class KernelMetadata:
    """Kernel metadata extracted from AMDGPU assembly directives."""
    name: str = ""
    arch: str = ""
    vgpr_count: int = 0
    sgpr_count: int = 0
    agpr_count: int = 0
    lds_size: int = 0
    scratch_size: int = 0
    wavefront_size: int = 64
    max_flat_workgroup_size: int = 0
    kernarg_size: int = 0


@dataclass
class RegisterUsage:
    """Register usage analysis."""
    vgpr_used: set = field(default_factory=set)
    sgpr_used: set = field(default_factory=set)
    agpr_used: set = field(default_factory=set)
    max_vgpr: int = 0
    max_sgpr: int = 0
    max_agpr: int = 0


@dataclass
class ParsedKernel:
    """Fully parsed kernel representation."""
    metadata: KernelMetadata = field(default_factory=KernelMetadata)
    instructions: list[Instruction] = field(default_factory=list)
    basic_blocks: list[BasicBlock] = field(default_factory=list)
    register_usage: RegisterUsage = field(default_factory=RegisterUsage)

    total_instructions: int = 0
    valu_count: int = 0
    salu_count: int = 0
    vmem_count: int = 0
    smem_count: int = 0
    lds_count: int = 0
    mfma_count: int = 0
    branch_count: int = 0
    waitcnt_count: int = 0
    barrier_count: int = 0
    nop_count: int = 0


def _classify_mnemonic(mnemonic: str) -> str:
    """Classify an instruction mnemonic into a pipeline category."""
    if mnemonic.startswith("v_mfma") or mnemonic.startswith("v_smfma"):
        return "MFMA"
    if mnemonic.startswith("v_pk_"):
        return "VOP3P"
    if mnemonic.startswith("v_"):
        return "VALU"
    if mnemonic.startswith("s_load") or mnemonic.startswith("s_buffer_load"):
        return "SMEM"
    if mnemonic.startswith("s_"):
        return "SALU"
    if mnemonic.startswith("ds_"):
        return "LDS"
    if mnemonic.startswith("global_") or mnemonic.startswith("buffer_"):
        return "VMEM"
    if mnemonic.startswith("flat_"):
        return "FLAT"
    return "MISC"
