import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.instruction import (
    Instruction, KernelInfo, EditOperation, BasicBlock,
    KernelMetadata, RegisterUsage, ParsedKernel, _classify_mnemonic,
)


def test_from_parser_line():
    instr = Instruction.from_parser_line(
        line_number=10,
        raw_text="v_mfma_f32_16x16x16_bf16 a[0:3], v[0:1], v[2:3], a[0:3]",
        mnemonic="v_mfma_f32_16x16x16_bf16",
        operands=["a[0:3]", "v[0:1]", "v[2:3]", "a[0:3]"],
        category="MFMA",
        is_mfma=True,
    )
    assert instr.mnemonic == "v_mfma_f32_16x16x16_bf16"
    assert instr.category == "MFMA"
    assert instr.is_mfma
    assert instr.line_number == 10
    assert instr.is_vector


def test_from_disassembly():
    instr = Instruction.from_disassembly(
        address=0x1000,
        raw_bytes=b"\x00\x00\x80\xbf",
        mnemonic="s_nop",
        operands="0",
        line_number=5,
        file_offset=0x200,
    )
    assert instr.address == 0x1000
    assert instr.size == 4
    assert instr.mnemonic == "s_nop"
    assert instr.operands == "0"
    assert instr.category == "SALU"
    assert instr.is_scalar


def test_classify_mnemonic():
    assert _classify_mnemonic("v_mfma_f32_16x16x16_bf16") == "MFMA"
    assert _classify_mnemonic("v_add_f32") == "VALU"
    assert _classify_mnemonic("s_waitcnt") == "SALU"
    assert _classify_mnemonic("ds_read_b32") == "LDS"
    assert _classify_mnemonic("global_load_dword") == "VMEM"
    assert _classify_mnemonic("buffer_store_dword") == "VMEM"
    assert _classify_mnemonic("s_load_dword") == "SMEM"
    assert _classify_mnemonic("flat_load_dword") == "FLAT"
    assert _classify_mnemonic("v_pk_fma_f16") == "VOP3P"


def test_instruction_properties():
    instr = Instruction(mnemonic="v_add_f32", raw_text="v_add_f32 v0, v1, v2")
    assert instr.is_vector
    assert not instr.is_scalar
    assert instr.full_text == "v_add_f32 v0, v1, v2"


def test_edit_operation():
    edit = EditOperation(target_index=5, new_mnemonic="s_waitcnt", new_operands="vmcnt(2)")
    assert edit.target_index == 5
    assert edit.new_mnemonic == "s_waitcnt"


def test_kernel_info():
    ki = KernelInfo(name="test_kernel", text_vma=0x1000, text_offset=0x200, text_size=512)
    assert ki.arch == "gfx942"


def test_parsed_kernel():
    pk = ParsedKernel()
    assert pk.total_instructions == 0
    assert pk.mfma_count == 0


def test_basic_block():
    instr = Instruction(mnemonic="s_nop", raw_text="s_nop 0")
    bb = BasicBlock(label="LBB0_0", instructions=[instr], successors=["LBB0_1"])
    assert bb.instruction_count == 1
    assert bb.label == "LBB0_0"


def test_kernel_metadata():
    km = KernelMetadata(name="k", arch="gfx942", vgpr_count=64)
    assert km.name == "k"
    assert km.arch == "gfx942"
