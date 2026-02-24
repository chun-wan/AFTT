"""Similarity-based search for matching production kernels in the ASM corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .instruction import Instruction
from .algorithm_classifier import AlgorithmInfo, ASMFeatureFingerprint


@dataclass
class TemplateMatch:
    kernel_name: str
    category: str
    arch: str
    similarity_score: float
    asm_path: Path
    metadata: dict
    optimization_gap: dict = field(default_factory=dict)


class TemplateMatcher:
    def __init__(self, db_root: Optional[Path] = None):
        self.db_root = db_root or Path(__file__).resolve().parent.parent / "db"
        self._corpus: list[dict] = []
        self._signatures: dict = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        # Load corpus from disassembly_summary.json
        summary_path = self.db_root / "disassembly" / "disassembly_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            self._corpus = data.get("kernels", [])

        # Load algorithm signatures
        sig_path = self.db_root / "algorithm_signatures.json"
        if sig_path.exists():
            with open(sig_path) as f:
                sig_data = json.load(f)
            self._signatures = sig_data.get("algorithms", {})

        self._loaded = True

    def search(self, algo_info: AlgorithmInfo, arch: str = "gfx942",
               top_k: int = 5) -> list[TemplateMatch]:
        """Search corpus for similar production kernels."""
        self.load()

        # Get target categories from signatures
        sig = self._signatures.get(algo_info.sub_type, {})
        target_categories = sig.get("corpus_categories", [])

        # Filter corpus by arch and category
        candidates = []
        for kernel in self._corpus:
            if kernel.get("arch") != arch:
                continue
            cat = kernel.get("category", "")
            if target_categories and cat not in target_categories:
                continue
            candidates.append(kernel)

        if not candidates and target_categories:
            # Fallback: search all categories for this arch
            candidates = [k for k in self._corpus if k.get("arch") == arch]

        # Score each candidate
        scored = []
        for kernel in candidates:
            score = self._compute_similarity(algo_info, kernel)
            if score > 0.1:
                asm_file = kernel.get("asm_file", "")
                asm_path = self.db_root / "disassembly" / asm_file if asm_file else Path("")
                gap = self._compute_optimization_gap(algo_info, kernel)
                scored.append(TemplateMatch(
                    kernel_name=kernel.get("kernel_name", kernel.get("asm_file", "unknown")),
                    category=kernel.get("category", "unknown"),
                    arch=arch,
                    similarity_score=score,
                    asm_path=asm_path,
                    metadata=kernel,
                    optimization_gap=gap,
                ))

        scored.sort(key=lambda x: x.similarity_score, reverse=True)
        return scored[:top_k]

    def _compute_similarity(self, algo_info: AlgorithmInfo, template: dict) -> float:
        """Weighted similarity score between input and template."""
        score = 0.0
        features = algo_info.features

        # Algorithm type match (0.4 weight)
        cat = template.get("category", "")
        sig = self._signatures.get(algo_info.sub_type, {})
        if cat in sig.get("corpus_categories", []):
            score += 0.4
        elif algo_info.algo_type.lower() in cat.lower():
            score += 0.2

        # MFMA type overlap (0.2 weight)
        input_mfma = set(features.get("mfma_types", {}).keys())
        tmpl_mfma = set(template.get("mfma_types", {}).keys()) if isinstance(template.get("mfma_types"), dict) else set()
        if input_mfma and tmpl_mfma:
            overlap = len(input_mfma & tmpl_mfma) / max(len(input_mfma | tmpl_mfma), 1)
            score += 0.2 * overlap
        elif not input_mfma and not tmpl_mfma:
            score += 0.1  # both non-MFMA kernels

        # Instruction count similarity (0.1 weight)
        input_count = features.get("total_instructions", 0)
        tmpl_count = template.get("total_instructions", 0)
        if input_count > 0 and tmpl_count > 0:
            ratio = min(input_count, tmpl_count) / max(input_count, tmpl_count)
            score += 0.1 * ratio

        # Register pressure similarity (0.1 weight)
        input_vgpr = features.get("max_vgpr", 0)
        tmpl_vgpr = template.get("max_vgpr", 0)
        if input_vgpr > 0 and tmpl_vgpr > 0:
            ratio = min(input_vgpr, tmpl_vgpr) / max(input_vgpr, tmpl_vgpr)
            score += 0.1 * ratio

        # Vectorization ratio (0.1 weight)
        input_vec = features.get("vectorization_ratio", 0)
        tmpl_dwordx4 = template.get("dwordx4_loads", 0)
        tmpl_single = template.get("dword_single_loads", 0)
        tmpl_vec = tmpl_dwordx4 / max(tmpl_dwordx4 + tmpl_single, 1)
        if input_vec > 0 or tmpl_vec > 0:
            diff = abs(input_vec - tmpl_vec)
            score += 0.1 * max(0, 1.0 - diff)

        # DPP usage pattern (0.1 weight)
        input_dpp = features.get("dpp_count", 0) > 0
        tmpl_dpp = template.get("dpp_count", 0) if "dpp_count" in template else False
        if input_dpp == bool(tmpl_dpp):
            score += 0.1

        return min(score, 1.0)

    def _compute_optimization_gap(self, algo_info: AlgorithmInfo, template: dict) -> dict:
        """Identify what the template does better than the input."""
        gap = {}
        features = algo_info.features

        # Check vectorization gap
        input_vec = features.get("vectorization_ratio", 0)
        tmpl_dwordx4 = template.get("dwordx4_loads", 0)
        tmpl_single = template.get("dword_single_loads", 0)
        tmpl_vec = tmpl_dwordx4 / max(tmpl_dwordx4 + tmpl_single, 1)
        if tmpl_vec > input_vec + 0.2:
            gap["vectorization"] = {
                "input": round(input_vec, 2),
                "template": round(tmpl_vec, 2),
                "description": f"Template uses {tmpl_vec:.0%} vectorized loads vs input {input_vec:.0%}"
            }

        # Check DPP gap
        if not features.get("dpp_count", 0) and template.get("dpp_count", 0):
            gap["dpp"] = {
                "description": "Template uses DPP instructions but input does not"
            }

        # Check MFMA gap
        input_mfma = features.get("mfma_count", 0)
        tmpl_mfma = template.get("mfma_count", 0)
        if tmpl_mfma > 0 and input_mfma == 0:
            gap["mfma"] = {
                "description": f"Template uses {tmpl_mfma} MFMA instructions"
            }

        return gap

    def get_template_asm(self, match: TemplateMatch) -> Optional[str]:
        """Read the ASM text for a matched template."""
        if match.asm_path and match.asm_path.exists():
            return match.asm_path.read_text()
        return None

    def get_corpus_categories(self) -> dict[str, int]:
        """Return category distribution of the corpus."""
        self.load()
        categories: dict[str, int] = {}
        for k in self._corpus:
            cat = k.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        return categories
