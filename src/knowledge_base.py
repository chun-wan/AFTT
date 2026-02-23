"""Unified Knowledge Base Access Layer.

Provides a single interface to access all databases: ISA instructions,
optimization patterns, profiling rules, compiler flags, and C++/ASM pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .isa_db import ISADatabase, Instruction


DB_ROOT = Path(__file__).resolve().parent.parent / "db"


class KnowledgeBase:
    """Unified access to all knowledge base components."""

    def __init__(self, db_root: Optional[Path] = None):
        self.db_root = db_root or DB_ROOT
        self._isa_db: Optional[ISADatabase] = None
        self._anti_patterns: list[dict] = []
        self._best_practices: list[dict] = []
        self._profiling_rules: list[dict] = []
        self._extracted_patterns: dict = {}
        self._compiler_flags: dict = {}
        self._deep_asm_patterns: dict = {}
        self._ck_deep_patterns: dict = {}
        self._trtllm_algorithms: dict = {}
        self._trtllm_amd_mapping: dict = {}
        self._dpp_crosslane_patterns: dict = {}
        self._fmha_asm_patterns: dict = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        # ISA database
        self._isa_db = ISADatabase(self.db_root / "isa")
        self._isa_db.load()

        # Anti-patterns
        ap_file = self.db_root / "patterns" / "anti_patterns.json"
        if ap_file.exists():
            with open(ap_file) as f:
                data = json.load(f)
            self._anti_patterns = data.get("patterns", [])

        # Best practices
        bp_file = self.db_root / "patterns" / "best_practices.json"
        if bp_file.exists():
            with open(bp_file) as f:
                data = json.load(f)
            self._best_practices = data.get("patterns", [])

        # Extracted patterns
        ep_file = self.db_root / "patterns" / "extracted_patterns.json"
        if ep_file.exists():
            with open(ep_file) as f:
                self._extracted_patterns = json.load(f)

        # Profiling rules
        pr_file = self.db_root / "profiling_rules" / "profiling_rules.json"
        if pr_file.exists():
            with open(pr_file) as f:
                data = json.load(f)
            self._profiling_rules = data.get("rules", [])

        # Compiler flags
        cf_file = self.db_root / "compiler_flags" / "flag_effects.json"
        if cf_file.exists():
            with open(cf_file) as f:
                self._compiler_flags = json.load(f)

        # Phase 1.5: Deep pattern databases
        dap_file = self.db_root / "patterns" / "deep_asm_patterns.json"
        if dap_file.exists():
            with open(dap_file) as f:
                self._deep_asm_patterns = json.load(f)

        ckdp_file = self.db_root / "patterns" / "ck_deep_patterns.json"
        if ckdp_file.exists():
            with open(ckdp_file) as f:
                self._ck_deep_patterns = json.load(f)

        ta_file = self.db_root / "patterns" / "trtllm_algorithms.json"
        if ta_file.exists():
            with open(ta_file) as f:
                self._trtllm_algorithms = json.load(f)

        tam_file = self.db_root / "patterns" / "trtllm_amd_mapping.json"
        if tam_file.exists():
            with open(tam_file) as f:
                self._trtllm_amd_mapping = json.load(f)

        dpp_file = self.db_root / "patterns" / "dpp_crosslane_patterns.json"
        if dpp_file.exists():
            with open(dpp_file) as f:
                self._dpp_crosslane_patterns = json.load(f)

        fmha_file = self.db_root / "patterns" / "fmha_asm_patterns.json"
        if fmha_file.exists():
            with open(fmha_file) as f:
                self._fmha_asm_patterns = json.load(f)

        self._loaded = True

    @property
    def isa(self) -> ISADatabase:
        self.load()
        return self._isa_db

    @property
    def anti_patterns(self) -> list[dict]:
        self.load()
        return self._anti_patterns

    @property
    def best_practices(self) -> list[dict]:
        self.load()
        return self._best_practices

    @property
    def profiling_rules(self) -> list[dict]:
        self.load()
        return self._profiling_rules

    @property
    def extracted_patterns(self) -> dict:
        self.load()
        return self._extracted_patterns

    @property
    def compiler_flags(self) -> dict:
        self.load()
        return self._compiler_flags

    @property
    def deep_asm_patterns(self) -> dict:
        self.load()
        return self._deep_asm_patterns

    @property
    def ck_deep_patterns(self) -> dict:
        self.load()
        return self._ck_deep_patterns

    @property
    def trtllm_algorithms(self) -> dict:
        self.load()
        return self._trtllm_algorithms

    @property
    def trtllm_amd_mapping(self) -> dict:
        self.load()
        return self._trtllm_amd_mapping

    @property
    def dpp_crosslane_patterns(self) -> dict:
        self.load()
        return self._dpp_crosslane_patterns

    @property
    def fmha_asm_patterns(self) -> dict:
        self.load()
        return self._fmha_asm_patterns

    def get_anti_pattern(self, pattern_id: str) -> Optional[dict]:
        self.load()
        for p in self._anti_patterns:
            if p["pattern_id"] == pattern_id:
                return p
        return None

    def get_best_practice(self, pattern_id: str) -> Optional[dict]:
        self.load()
        for p in self._best_practices:
            if p["pattern_id"] == pattern_id:
                return p
        return None

    def get_profiling_rule(self, rule_id: str) -> Optional[dict]:
        self.load()
        for r in self._profiling_rules:
            if r["rule_id"] == rule_id:
                return r
        return None

    def lookup_instruction(self, mnemonic: str) -> Optional[Instruction]:
        return self.isa.lookup(mnemonic)

    def search_instructions(self, pattern: str) -> list[Instruction]:
        return self.isa.search(pattern)

    def get_cycle_estimator(self, arch: str = "gfx942"):
        """Get a cycle estimator for the given architecture."""
        from .cycle_estimator import CycleEstimator
        return CycleEstimator(arch, self._isa_db)

    def get_asm_editor(self, arch: str = "gfx942"):
        """Get an ASM editor for the given architecture."""
        from .asm_editor import AsmEditor
        return AsmEditor(arch)

    def get_optimizer(self, arch: str = "gfx942"):
        """Get an ASM optimizer for the given architecture."""
        from .asm_optimizer import AsmOptimizer
        return AsmOptimizer(arch)

    def get_stats(self) -> dict:
        self.load()
        return {
            "isa_instructions": self.isa.instruction_count,
            "isa_architectures": self.isa.arch_count,
            "anti_patterns": len(self._anti_patterns),
            "best_practices": len(self._best_practices),
            "profiling_rules": len(self._profiling_rules),
            "pipeline_patterns": len(self._extracted_patterns.get("pipeline_patterns", [])),
            "asm_kernel_configs": len(self._extracted_patterns.get("asm_kernel_configs", [])),
            "compiler_flag_comparisons": len(self._compiler_flags.get("comparisons", [])),
            "deep_asm_kernels_analyzed": self._deep_asm_patterns.get("total_kernels_analyzed", 0),
            "ck_pipelines_analyzed": self._ck_deep_patterns.get("total_pipelines_analyzed", 0),
            "trtllm_algorithms": self._trtllm_algorithms.get("total_algorithms_cataloged", 0),
            "trtllm_amd_mappings": self._trtllm_amd_mapping.get("total_mappings", 0),
            "per_arch_isa_tables": len(self._isa_db._detailed) if self._isa_db else 0,
            "dpp_patterns_cataloged": self._dpp_crosslane_patterns.get("total_dpp_instructions", 0),
            "fmha_kernels_analyzed": (
                self._fmha_asm_patterns.get("summary", {}).get("forward_kernels", 0) +
                self._fmha_asm_patterns.get("summary", {}).get("backward_kernels", 0)
            ),
        }
