"""AFTT v2 Optimization Pipeline Orchestrator.

Provides a unified pipeline that takes HIP C++ source code and produces
optimized ASM with validation, combining all AFTT v2 components:
  Stage 1: Compile and Parse
  Stage 2: Algorithm Recognition
  Stage 3: Template Matching
  Stage 4: C++ Template Swap (optional)
  Stage 5: ASM Pattern Replacement
  Stage 6: ASM Optimization Passes
  Stage 7: Assembly and Validation
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .instruction import Instruction, EditOperation, ParsedKernel
from .compiler import Compiler
from .parser import parse_asm
from .knowledge_base import KnowledgeBase
from .analyzer import Analyzer
from .algorithm_classifier import AlgorithmClassifier, AlgorithmInfo
from .template_matcher import TemplateMatcher, TemplateMatch
from .cpp_template_engine import CppTemplateEngine, InstantiatedTemplate
from .pattern_replacer import PatternReplacer, ReplacementResult
from .asm_optimizer import AsmOptimizer
from .cycle_estimator import CycleEstimator


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage_name: str
    success: bool
    duration_ms: float = 0.0
    data: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    stages: list[StageResult] = field(default_factory=list)

    # Key outputs
    algo_info: Optional[AlgorithmInfo] = None
    template_matches: list[TemplateMatch] = field(default_factory=list)
    cpp_swap: Optional[InstantiatedTemplate] = None
    replacements: Optional[ReplacementResult] = None

    original_kernel: Optional[ParsedKernel] = None
    optimized_kernel: Optional[ParsedKernel] = None

    original_cycle_estimate: Optional[dict] = None
    optimized_cycle_estimate: Optional[dict] = None
    cycle_comparison: Optional[dict] = None

    applied_edits: list[EditOperation] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=" * 60, "AFTT v2 Pipeline Summary", "=" * 60]

        for stage in self.stages:
            status = "OK" if stage.success else "FAIL"
            lines.append(f"  [{status}] {stage.stage_name} ({stage.duration_ms:.0f}ms)")
            if stage.error:
                lines.append(f"       Error: {stage.error}")

        if self.algo_info:
            lines.append(f"\n  Algorithm: {self.algo_info.algo_type} "
                        f"({self.algo_info.sub_type}, confidence={self.algo_info.confidence:.2f})")

        if self.template_matches:
            top = self.template_matches[0]
            lines.append(f"  Best template: {top.kernel_name} "
                        f"(score={top.similarity_score:.2f})")

        if self.cycle_comparison:
            lines.append(f"\n  Original cycles:  {self.cycle_comparison.get('original_cycles', '?'):,}")
            lines.append(f"  Optimized cycles: {self.cycle_comparison.get('modified_cycles', '?'):,}")
            lines.append(f"  Improvement:      {self.cycle_comparison.get('improvement_pct', 0):.2f}%")

        lines.append(f"\n  Edits applied: {len(self.applied_edits)}")
        lines.append(f"  Recommendations: {len(self.recommendations)}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        result = {
            "stages": [
                {"name": s.stage_name, "success": s.success,
                 "duration_ms": s.duration_ms, "error": s.error}
                for s in self.stages
            ],
            "edits_applied": len(self.applied_edits),
            "recommendations_count": len(self.recommendations),
        }
        if self.algo_info:
            result["algorithm"] = {
                "type": self.algo_info.algo_type,
                "sub_type": self.algo_info.sub_type,
                "confidence": self.algo_info.confidence,
                "parameters": self.algo_info.parameters,
            }
        if self.template_matches:
            result["top_matches"] = [
                {"name": m.kernel_name, "category": m.category,
                 "score": round(m.similarity_score, 3)}
                for m in self.template_matches[:5]
            ]
        if self.cycle_comparison:
            result["cycle_comparison"] = self.cycle_comparison
        if self.recommendations:
            result["recommendations"] = self.recommendations
        return result


class OptimizationPipeline:
    """End-to-end HIP C++ -> optimized ASM pipeline."""

    def __init__(self, arch: str = "gfx942"):
        self.arch = arch
        self.kb = KnowledgeBase()
        self.kb.load()
        self.compiler = Compiler()
        self.classifier = AlgorithmClassifier()
        self.matcher = TemplateMatcher()
        self.template_engine = CppTemplateEngine()
        self.replacer = PatternReplacer(self.kb)
        self.optimizer = AsmOptimizer(arch, kb=self.kb)
        self.analyzer = Analyzer(self.kb)

    def run(self, source: str, *,
            source_path: str = "<input>",
            enable_cpp_swap: bool = True,
            enable_asm_replace: bool = True,
            enable_asm_optimize: bool = True,
            max_replacement_level: int = 3,
            aggressive: bool = False) -> PipelineResult:
        """Run the full optimization pipeline on HIP C++ source code."""
        result = PipelineResult()

        # Stage 1: Compile and Parse
        t0 = time.time()
        try:
            # Compiler.compile_to_asm accepts either a file path or source string
            # (string is used when path doesn't exist; it writes to temp .hip file internally)
            compile_input = source_path if source_path != "<input>" else source
            compile_result = self.compiler.compile_to_asm(
                compile_input,
                arch=self.arch,
                opt_level="-O3",
            )

            if not compile_result.success:
                result.stages.append(StageResult(
                    "Compile & Parse", False,
                    error=compile_result.stderr[:200]))
                return result

            kernel = parse_asm(compile_result.asm_output)
            result.original_kernel = kernel
            result.stages.append(StageResult(
                "Compile & Parse", True,
                duration_ms=(time.time() - t0) * 1000,
                data={"instructions": kernel.total_instructions}))
        except Exception as e:
            result.stages.append(StageResult("Compile & Parse", False, error=str(e)))
            return result

        # Stage 2: Algorithm Recognition
        t0 = time.time()
        try:
            algo_hip = self.classifier.classify_from_hip(source)
            algo_asm = self.classifier.classify_from_asm(kernel.instructions)

            # Use HIP classification if confident, else fall back to ASM
            if algo_hip.confidence >= algo_asm.confidence:
                algo_info = algo_hip
            else:
                algo_info = algo_asm

            result.algo_info = algo_info
            result.stages.append(StageResult(
                "Algorithm Recognition", True,
                duration_ms=(time.time() - t0) * 1000,
                data={"type": algo_info.algo_type, "sub_type": algo_info.sub_type,
                      "confidence": algo_info.confidence}))
        except Exception as e:
            result.stages.append(StageResult("Algorithm Recognition", False, error=str(e)))
            algo_info = AlgorithmInfo(algo_type="custom", confidence=0.0,
                                      parameters={}, features={}, sub_type="custom")
            result.algo_info = algo_info

        # Stage 3: Template Matching
        t0 = time.time()
        try:
            matches = self.matcher.search(algo_info, arch=self.arch, top_k=5)
            result.template_matches = matches
            result.stages.append(StageResult(
                "Template Matching", True,
                duration_ms=(time.time() - t0) * 1000,
                data={"matches_found": len(matches),
                      "top_score": matches[0].similarity_score if matches else 0}))
        except Exception as e:
            result.stages.append(StageResult("Template Matching", False, error=str(e)))
            matches = []

        # Stage 4: C++ Template Swap (optional)
        if enable_cpp_swap and algo_info.confidence > 0.5:
            t0 = time.time()
            try:
                swap = self.template_engine.get_optimized_replacement(
                    algo_info.algo_type, algo_info.parameters)
                if swap:
                    swap_compile = self.compiler.compile_to_asm(
                        swap.source_code,
                        arch=self.arch,
                        opt_level="-O3",
                    )
                    if swap_compile.success:
                        swapped_kernel = parse_asm(swap_compile.asm_output)
                        if swapped_kernel.total_instructions > 0:
                            kernel = swapped_kernel
                            result.cpp_swap = swap
                            result.stages.append(StageResult(
                                "C++ Template Swap", True,
                                duration_ms=(time.time() - t0) * 1000,
                                data={"template": swap.template_name,
                                      "variant": swap.variant,
                                      "new_instructions": kernel.total_instructions}))
                        else:
                            result.stages.append(StageResult(
                                "C++ Template Swap", False,
                                duration_ms=(time.time() - t0) * 1000,
                                error="Swapped template produced empty kernel"))
                    else:
                        result.stages.append(StageResult(
                            "C++ Template Swap", False,
                            duration_ms=(time.time() - t0) * 1000,
                            error=f"Compilation failed: {swap_compile.stderr[:100]}"))
                else:
                    result.stages.append(StageResult(
                        "C++ Template Swap", True,
                        duration_ms=(time.time() - t0) * 1000,
                        data={"skipped": "No optimized template available"}))
            except Exception as e:
                result.stages.append(StageResult("C++ Template Swap", False, error=str(e)))

        # Stage 5: ASM Optimization Passes (run FIRST — produces core edits)
        all_edits = []
        if enable_asm_optimize:
            t0 = time.time()
            try:
                opt_result = self.optimizer.optimize(kernel.instructions, aggressive=aggressive)
                all_edits.extend(opt_result.edits)
                result.recommendations.extend(opt_result.recommendations)
                result.stages.append(StageResult(
                    "ASM Optimization", True,
                    duration_ms=(time.time() - t0) * 1000,
                    data={"edits": len(opt_result.edits),
                          "recommendations": len(opt_result.recommendations)}))
            except Exception as e:
                result.stages.append(StageResult("ASM Optimization", False, error=str(e)))

        # Stage 6: ASM Pattern Replacement (adds levels 3+ edits)
        if enable_asm_replace:
            t0 = time.time()
            try:
                replacements = self.replacer.find_replacements_standalone(
                    kernel.instructions, max_level=max_replacement_level)
                result.replacements = replacements
                safe_edits = self.replacer.get_safe_edits(
                    replacements.replacements, max_level=max_replacement_level)
                all_edits.extend(safe_edits)
                result.stages.append(StageResult(
                    "ASM Pattern Replacement", True,
                    duration_ms=(time.time() - t0) * 1000,
                    data={"replacements_found": len(replacements.replacements),
                          "safe_edits": len(safe_edits)}))
            except Exception as e:
                result.stages.append(StageResult("ASM Pattern Replacement", False, error=str(e)))

        # Deduplicate edits by target_index (first writer wins)
        seen_indices = set()
        unique_edits = []
        for edit in all_edits:
            if edit.target_index not in seen_indices:
                seen_indices.add(edit.target_index)
                unique_edits.append(edit)
        result.applied_edits = unique_edits

        # Stage 7: Cycle Estimation
        t0 = time.time()
        try:
            estimator = CycleEstimator(self.arch)
            orig_lines = [i.full_text for i in kernel.instructions]

            # Apply edits to get modified lines
            edit_map = {e.target_index: e for e in unique_edits}
            mod_lines = []
            for idx, instr in enumerate(kernel.instructions):
                if idx in edit_map:
                    e = edit_map[idx]
                    mod_lines.append(f"{e.new_mnemonic} {e.new_operands}")
                else:
                    mod_lines.append(instr.full_text)

            est_orig = estimator.estimate(orig_lines)
            est_mod = estimator.estimate(mod_lines)
            comparison = estimator.compare(est_orig, est_mod)

            result.original_cycle_estimate = est_orig.to_dict()
            result.optimized_cycle_estimate = est_mod.to_dict()
            result.cycle_comparison = comparison

            result.stages.append(StageResult(
                "Cycle Estimation", True,
                duration_ms=(time.time() - t0) * 1000,
                data=comparison))
        except Exception as e:
            result.stages.append(StageResult("Cycle Estimation", False, error=str(e)))

        # Run analysis for additional findings
        try:
            analysis = self.analyzer.analyze(kernel, arch=self.arch)
            for f in analysis.findings:
                if f.severity in ("critical", "warning"):
                    result.recommendations.append({
                        "type": f"analyzer_{f.category}",
                        "severity": f.severity,
                        "description": f"{f.title}: {f.description}",
                        "suggestion": f.suggestion,
                    })
        except Exception:
            pass

        return result

    def run_co_to_co(self, co_path: str, output_co_path: str, *,
                     enable_asm_replace: bool = True,
                     enable_asm_optimize: bool = True,
                     max_replacement_level: int = 2,
                     aggressive: bool = False,
                     skip_waitcnt: bool = False) -> PipelineResult:
        """Optimize a .co binary directly (skip C++ stages).

        Stages executed:
          1. Disassemble .co → instructions
          2. ASM Pattern Replacement
          3. ASM Optimization Passes
          4. Filter edits (optional waitcnt skip)
          5. Binary patch → output .co
          6. Cycle Estimation (before/after)
        """
        from .asm_editor import AsmEditor

        result = PipelineResult()
        editor = AsmEditor(arch=self.arch)

        # Stage 1: Disassemble
        t0 = time.time()
        try:
            kernel_info, instructions = editor.disassemble(co_path)
            result.stages.append(StageResult(
                "Disassemble .co", True,
                duration_ms=(time.time() - t0) * 1000,
                data={"instructions": len(instructions),
                      "kernel_name": kernel_info.name}))
        except Exception as e:
            result.stages.append(StageResult("Disassemble .co", False, error=str(e)))
            return result

        if not instructions:
            result.stages.append(StageResult("Disassemble .co", False,
                                             error="empty instruction list"))
            return result

        # Stage 2: ASM Optimization (core passes run first)
        all_edits = []

        if enable_asm_optimize:
            t0 = time.time()
            try:
                opt_result = self.optimizer.optimize(instructions, aggressive=aggressive)
                all_edits.extend(opt_result.edits)
                result.recommendations.extend(opt_result.recommendations)
                result.stages.append(StageResult(
                    "ASM Optimization", True,
                    duration_ms=(time.time() - t0) * 1000,
                    data={"edits": len(opt_result.edits),
                          "recommendations": len(opt_result.recommendations)}))
            except Exception as e:
                result.stages.append(StageResult(
                    "ASM Optimization", False, error=str(e)))

        # Stage 3: Pattern Replacement (adds levels 3+ edits)
        if enable_asm_replace:
            t0 = time.time()
            try:
                replacements = self.replacer.find_replacements_standalone(
                    instructions, max_level=max_replacement_level)
                result.replacements = replacements
                safe_edits = self.replacer.get_safe_edits(
                    replacements.replacements, max_level=max_replacement_level)
                all_edits.extend(safe_edits)
                result.stages.append(StageResult(
                    "ASM Pattern Replacement", True,
                    duration_ms=(time.time() - t0) * 1000,
                    data={"safe_edits": len(safe_edits)}))
            except Exception as e:
                result.stages.append(StageResult(
                    "ASM Pattern Replacement", False, error=str(e)))

        # Deduplicate + filter
        seen = set()
        unique_edits = []
        for edit in all_edits:
            if edit.target_index in seen:
                continue
            if skip_waitcnt and edit.new_mnemonic == "s_waitcnt":
                continue
            seen.add(edit.target_index)
            unique_edits.append(edit)
        result.applied_edits = unique_edits

        # Stage 5: Binary patch
        t0 = time.time()
        if unique_edits:
            try:
                import shutil
                shutil.copy2(co_path, output_co_path)
                patch_result = editor.binary_patch(
                    co_path, output_co_path, unique_edits, instructions)
                result.stages.append(StageResult(
                    "Binary Patch", True,
                    duration_ms=(time.time() - t0) * 1000,
                    data={"applied": patch_result["applied_count"],
                          "skipped": patch_result["skipped_count"]}))
            except Exception as e:
                result.stages.append(StageResult("Binary Patch", False, error=str(e)))
        else:
            import shutil
            shutil.copy2(co_path, output_co_path)
            result.stages.append(StageResult(
                "Binary Patch", True,
                duration_ms=(time.time() - t0) * 1000,
                data={"applied": 0, "skipped": 0, "note": "no edits to apply"}))

        # Stage 6: Cycle estimation
        t0 = time.time()
        try:
            estimator = CycleEstimator(self.arch)
            orig_lines = editor.get_instruction_lines(instructions)
            mod_lines = editor.apply_and_get_modified_lines(instructions, unique_edits)

            est_orig = estimator.estimate(orig_lines)
            est_mod = estimator.estimate(mod_lines)
            comparison = estimator.compare(est_orig, est_mod)

            result.original_cycle_estimate = est_orig.to_dict()
            result.optimized_cycle_estimate = est_mod.to_dict()
            result.cycle_comparison = comparison
            result.stages.append(StageResult(
                "Cycle Estimation", True,
                duration_ms=(time.time() - t0) * 1000,
                data=comparison))
        except Exception as e:
            result.stages.append(StageResult("Cycle Estimation", False, error=str(e)))

        return result
