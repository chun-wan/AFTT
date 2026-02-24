"""AMDGPU Assembly Parser.

Parses AMDGPU assembly output into a structured representation with
instructions, basic blocks, register usage, and memory access patterns.

Uses the unified Instruction model from instruction.py.
"""

from __future__ import annotations

import re
from typing import Optional

from .instruction import (
    Instruction,
    BasicBlock,
    KernelMetadata,
    RegisterUsage,
    ParsedKernel,
    _classify_mnemonic as classify_instruction,
)


# Regex patterns for parsing
RE_LABEL = re.compile(r'^(\S+):')
RE_INSTRUCTION = re.compile(r'^\s+(\S+)\s*(.*?)(?:\s*//\s*(.*))?$')
RE_DIRECTIVE = re.compile(r'^\s*\.([\w_]+)\s*(.*)')
RE_COMMENT = re.compile(r'^\s*[;/]')
RE_REGISTER_V = re.compile(r'v\[(\d+)(?::(\d+))?\]|v(\d+)')
RE_REGISTER_S = re.compile(r's\[(\d+)(?::(\d+))?\]|s(\d+)')
RE_REGISTER_A = re.compile(r'a\[(\d+)(?::(\d+))?\]|a(\d+)')
RE_WAITCNT_VMCNT = re.compile(r'vmcnt\((\d+)\)')
RE_WAITCNT_LGKMCNT = re.compile(r'lgkmcnt\((\d+)\)')
RE_VGPR_COUNT = re.compile(r'\.vgpr_count:\s*(\d+)')
RE_SGPR_COUNT = re.compile(r'\.sgpr_count:\s*(\d+)')
RE_AGPR_COUNT = re.compile(r'\.agpr_count:\s*(\d+)')
RE_LDS_SIZE = re.compile(r'\.group_segment_fixed_size:\s*(\d+)')
RE_SCRATCH_SIZE = re.compile(r'\.private_segment_fixed_size:\s*(\d+)')
RE_KERNEL_NAME = re.compile(r'\.name:\s*(\S+)')
RE_WAVEFRONT_SIZE = re.compile(r'\.wavefront_size:\s*(\d+)')
RE_WORKGROUP_SIZE = re.compile(r'\.max_flat_workgroup_size:\s*(\d+)')


def extract_registers(operand_text: str) -> tuple[list[str], list[str], list[str]]:
    """Extract VGPR, SGPR, and AGPR references from operand text."""
    vgprs = []
    sgprs = []
    agprs = []

    for m in RE_REGISTER_V.finditer(operand_text):
        if m.group(3):
            vgprs.append(f"v{m.group(3)}")
        elif m.group(1):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            for i in range(start, end + 1):
                vgprs.append(f"v{i}")

    for m in RE_REGISTER_S.finditer(operand_text):
        if m.group(3):
            sgprs.append(f"s{m.group(3)}")
        elif m.group(1):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            for i in range(start, end + 1):
                sgprs.append(f"s{i}")

    for m in RE_REGISTER_A.finditer(operand_text):
        if m.group(3):
            agprs.append(f"a{m.group(3)}")
        elif m.group(1):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            for i in range(start, end + 1):
                agprs.append(f"a{i}")

    return vgprs, sgprs, agprs


def parse_instruction_line(line: str, line_number: int) -> Optional[Instruction]:
    """Parse a single instruction line into a unified Instruction."""
    m = RE_INSTRUCTION.match(line)
    if not m:
        return None

    mnemonic = m.group(1)
    operands_raw = m.group(2).strip() if m.group(2) else ""
    comment = m.group(3) or ""

    if mnemonic.startswith("."):
        return None

    operands = [op.strip() for op in operands_raw.split(",") if op.strip()]
    category = classify_instruction(mnemonic)

    vgprs, sgprs, agprs = extract_registers(operands_raw)

    instr = Instruction.from_parser_line(
        line_number=line_number,
        raw_text=line.strip(),
        mnemonic=mnemonic,
        operands=operands,
        category=category,
        comment=comment,
        is_memory_op=category in ("VMEM", "FLAT", "SMEM"),
        is_lds_op=category == "LDS",
        is_barrier="barrier" in mnemonic,
        is_waitcnt="waitcnt" in mnemonic,
        is_branch="branch" in mnemonic or mnemonic == "s_branch",
        is_mfma="mfma" in mnemonic,
    )

    # Assign src/dst registers
    all_regs = vgprs + sgprs + agprs
    if operands and all_regs:
        first_comma = operands_raw.find(",")
        if first_comma > 0:
            dst_text = operands_raw[:first_comma]
            src_text = operands_raw[first_comma + 1:]
        else:
            dst_text = operands_raw
            src_text = ""

        dst_v, dst_s, dst_a = extract_registers(dst_text)
        src_v, src_s, src_a = extract_registers(src_text)
        instr.dst_registers = dst_v + dst_s + dst_a
        instr.src_registers = src_v + src_s + src_a

    return instr


def parse_metadata(asm_text: str) -> KernelMetadata:
    """Extract kernel metadata from AMDGPU assembly."""
    meta = KernelMetadata()

    m = RE_KERNEL_NAME.search(asm_text)
    if m:
        meta.name = m.group(1)

    m = RE_VGPR_COUNT.search(asm_text)
    if m:
        meta.vgpr_count = int(m.group(1))

    m = RE_SGPR_COUNT.search(asm_text)
    if m:
        meta.sgpr_count = int(m.group(1))

    m = RE_AGPR_COUNT.search(asm_text)
    if m:
        meta.agpr_count = int(m.group(1))

    m = RE_LDS_SIZE.search(asm_text)
    if m:
        meta.lds_size = int(m.group(1))

    m = RE_SCRATCH_SIZE.search(asm_text)
    if m:
        meta.scratch_size = int(m.group(1))

    m = RE_WAVEFRONT_SIZE.search(asm_text)
    if m:
        meta.wavefront_size = int(m.group(1))

    m = RE_WORKGROUP_SIZE.search(asm_text)
    if m:
        meta.max_flat_workgroup_size = int(m.group(1))

    return meta


def parse_asm(asm_text: str) -> ParsedKernel:
    """Parse AMDGPU assembly text into a structured ParsedKernel."""
    kernel = ParsedKernel()
    kernel.metadata = parse_metadata(asm_text)

    current_block = BasicBlock(label="entry")
    lines = asm_text.split("\n")

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        if not stripped or RE_COMMENT.match(stripped):
            continue

        if RE_DIRECTIVE.match(stripped):
            continue

        label_m = RE_LABEL.match(stripped)
        if label_m:
            if current_block.instructions:
                kernel.basic_blocks.append(current_block)
            current_block = BasicBlock(label=label_m.group(1))
            continue

        instr = parse_instruction_line(line, line_num)
        if instr:
            kernel.instructions.append(instr)
            current_block.instructions.append(instr)

            for r in instr.dst_registers + instr.src_registers:
                if r.startswith("v"):
                    idx = int(r[1:])
                    kernel.register_usage.vgpr_used.add(idx)
                    kernel.register_usage.max_vgpr = max(
                        kernel.register_usage.max_vgpr, idx
                    )
                elif r.startswith("s"):
                    idx = int(r[1:])
                    kernel.register_usage.sgpr_used.add(idx)
                    kernel.register_usage.max_sgpr = max(
                        kernel.register_usage.max_sgpr, idx
                    )
                elif r.startswith("a"):
                    idx = int(r[1:])
                    kernel.register_usage.agpr_used.add(idx)
                    kernel.register_usage.max_agpr = max(
                        kernel.register_usage.max_agpr, idx
                    )

            if instr.is_branch and instr.operands_list:
                target = instr.operands_list[-1]
                current_block.successors.append(target)

    if current_block.instructions:
        kernel.basic_blocks.append(current_block)

    # Compute summary statistics
    kernel.total_instructions = len(kernel.instructions)
    for instr in kernel.instructions:
        cat = instr.category
        if cat == "VALU":
            kernel.valu_count += 1
        elif cat == "VOP3P":
            kernel.valu_count += 1
        elif cat == "SALU":
            kernel.salu_count += 1
        elif cat == "SMEM":
            kernel.smem_count += 1
        elif cat in ("VMEM", "FLAT"):
            kernel.vmem_count += 1
        elif cat == "LDS":
            kernel.lds_count += 1
        elif cat == "MFMA":
            kernel.mfma_count += 1

        if instr.is_branch:
            kernel.branch_count += 1
        if instr.is_waitcnt:
            kernel.waitcnt_count += 1
        if instr.is_barrier:
            kernel.barrier_count += 1
        if instr.mnemonic == "s_nop":
            kernel.nop_count += 1

    return kernel
