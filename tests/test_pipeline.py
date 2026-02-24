import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.instruction import Instruction, ParsedKernel
from src.pattern_replacer import PatternReplacer, ReplacementResult, Replacement, EditOperation
from src.cpp_template_engine import CppTemplateEngine


def test_pattern_replacer_standalone():
    instrs = [
        Instruction.from_disassembly(0, b"\x00" * 4, "global_load_dword", "v0, v1, s[0:1]"),
        Instruction.from_disassembly(4, b"\x00" * 4, "global_load_dword", "v2, v3, s[0:1]"),
        Instruction.from_disassembly(8, b"\x00" * 4, "global_load_dword", "v4, v5, s[0:1]"),
        Instruction.from_disassembly(12, b"\x00" * 4, "s_waitcnt", "vmcnt(0) lgkmcnt(0)"),
        Instruction.from_disassembly(16, b"\x00" * 4, "v_add_f32", "v0, v0, v2"),
    ]
    replacer = PatternReplacer()
    result = replacer.find_replacements_standalone(instrs, max_level=1)
    assert isinstance(result, ReplacementResult)


def test_pattern_replacer_nop():
    instrs = [
        Instruction.from_disassembly(0, b"\x00" * 4, "v_add_f32", "v0, v1, v2"),
        Instruction.from_disassembly(4, b"\x00" * 4, "s_nop", "3"),
        Instruction.from_disassembly(8, b"\x00" * 4, "v_add_f32", "v3, v4, v5"),
    ]
    replacer = PatternReplacer()
    result = replacer.find_replacements_standalone(instrs, max_level=2)
    assert isinstance(result, ReplacementResult)
    nop_replacements = [r for r in result.replacements if r.safety_level == 2]
    assert len(nop_replacements) > 0


def test_cpp_template_engine():
    engine = CppTemplateEngine()
    engine.load()
    types = engine.get_available_types()
    assert isinstance(types, list)
    assert len(types) > 0


def test_cpp_template_get_best():
    engine = CppTemplateEngine()
    tmpl = engine.get_best_template("gemm")
    if tmpl:
        assert tmpl.algorithm_type == "gemm"
        assert "bad" not in tmpl.variant


def test_cpp_template_instantiate():
    engine = CppTemplateEngine()
    tmpl = engine.get_best_template("gemm")
    if tmpl:
        result = engine.instantiate(tmpl, {"M": 4096, "N": 4096, "K": 4096})
        assert result.source_code
        assert result.template_name == tmpl.name


def test_safe_edits():
    r1 = Replacement(
        safety_level=1,
        description="test",
        start_index=0,
        end_index=0,
        original_instructions=[],
        replacement_instructions=[],
        edits=[EditOperation(0, "s_nop", "0")],
    )
    r2 = Replacement(
        safety_level=3,
        description="test2",
        start_index=1,
        end_index=5,
        original_instructions=[],
        replacement_instructions=[],
    )
    replacer = PatternReplacer()
    edits = replacer.get_safe_edits([r1, r2], max_level=2)
    assert len(edits) == 1
    assert edits[0].target_index == 0
