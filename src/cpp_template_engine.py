"""C++ Template Engine for AMDGPU kernel optimization.

Manages a library of parameterizable HIP C++ templates for known algorithm
types. Supports template selection based on algorithm classification and
parameter instantiation for specific dimensions/data types.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TemplateInfo:
    """Metadata about a C++ kernel template."""
    name: str
    path: Path
    algorithm_type: str
    variant: str  # "naive", "optimized", "tiled", etc.
    parameters: list[str]  # what can be parameterized
    description: str = ""


@dataclass
class InstantiatedTemplate:
    """A template after parameter substitution."""
    source_code: str
    template_name: str
    variant: str
    parameters: dict
    original_path: Path


class CppTemplateEngine:
    """Engine for selecting and instantiating C++ kernel templates."""

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or Path(__file__).resolve().parent.parent / "templates"
        self._templates: dict[str, list[TemplateInfo]] = {}
        self._loaded = False

    def load(self):
        """Scan templates directory and index available templates."""
        if self._loaded:
            return
        
        # Map template files to algorithm types
        template_map = {
            "vector_add.hip": ("elementwise", "basic", ["size"]),
            "reduction_sum.hip": ("reduction", "tree", ["size", "block_size"]),
            "matrix_transpose.hip": ("transpose", "tiled", ["rows", "cols", "tile_dim"]),
            "gemm_naive.hip": ("gemm", "naive", ["M", "N", "K"]),
            "gemm_tiled.hip": ("gemm", "tiled", ["M", "N", "K", "TILE_SIZE"]),
            "softmax.hip": ("softmax", "row_wise", ["batch", "seq_len"]),
            "layernorm.hip": ("layernorm", "basic", ["hidden_size", "epsilon"]),
            "rmsnorm_naive.hip": ("rmsnorm", "naive", ["hidden_size", "epsilon"]),
            "rmsnorm_optimized.hip": ("rmsnorm", "optimized", ["hidden_size", "epsilon"]),
            "fp8gemm_blockscale_naive.hip": ("gemm_fp8_blockscale", "naive", ["M", "N", "K"]),
            "fp8gemm_blockscale_optimized.hip": ("gemm_fp8_blockscale", "optimized", ["M", "N", "K"]),
            "uncoalesced_bad.hip": ("elementwise", "uncoalesced_bad", ["size"]),
            "bank_conflict_bad.hip": ("reduction", "bank_conflict_bad", ["size"]),
        }
        
        for filename, (algo_type, variant, params) in template_map.items():
            path = self.templates_dir / filename
            if path.exists():
                info = TemplateInfo(
                    name=filename,
                    path=path,
                    algorithm_type=algo_type,
                    variant=variant,
                    parameters=params,
                    description=f"{algo_type} kernel ({variant} variant)",
                )
                if algo_type not in self._templates:
                    self._templates[algo_type] = []
                self._templates[algo_type].append(info)
        
        self._loaded = True

    def get_templates_for_type(self, algorithm_type: str) -> list[TemplateInfo]:
        """Get available templates for an algorithm type."""
        self.load()
        return self._templates.get(algorithm_type.lower(), [])

    def get_best_template(self, algorithm_type: str, 
                          prefer_optimized: bool = True) -> Optional[TemplateInfo]:
        """Select the best template for a given algorithm type."""
        self.load()
        templates = self._templates.get(algorithm_type.lower(), [])
        if not templates:
            return None
        
        if prefer_optimized:
            # Prefer "optimized" > "tiled" > everything else > "naive" > "*_bad"
            priority = {"optimized": 0, "tiled": 1, "row_wise": 2, "tree": 3,
                        "basic": 4, "naive": 5}
            templates.sort(key=lambda t: priority.get(t.variant, 3))
            # Filter out known-bad templates
            good = [t for t in templates if "bad" not in t.variant]
            return good[0] if good else templates[0]
        
        return templates[0]

    def instantiate(self, template: TemplateInfo, 
                    parameters: dict) -> InstantiatedTemplate:
        """Instantiate a template with specific parameters.
        
        Performs string substitution for known parameter patterns in the 
        template source code (e.g., TILE_SIZE, BLOCK_SIZE, dimension constants).
        """
        self.load()
        source = template.path.read_text()
        
        # Apply parameter substitutions
        substitutions = {
            "M": parameters.get("M"),
            "N": parameters.get("N"),
            "K": parameters.get("K"),
            "TILE_SIZE": parameters.get("TILE_SIZE", parameters.get("tile_size")),
            "TILE_DIM": parameters.get("TILE_DIM", parameters.get("tile_dim")),
            "BLOCK_SIZE": parameters.get("BLOCK_SIZE", parameters.get("block_size")),
            "HIDDEN_SIZE": parameters.get("hidden_size", parameters.get("HIDDEN_SIZE")),
        }
        
        for param_name, value in substitutions.items():
            if value is not None:
                # Replace #define PARAM_NAME <old_val> with #define PARAM_NAME <new_val>
                source = re.sub(
                    rf'(#define\s+{param_name}\s+)\d+',
                    rf'\g<1>{value}',
                    source
                )
                # Also replace constexpr int param_name = <old>; patterns
                source = re.sub(
                    rf'(constexpr\s+\w+\s+{param_name}\s*=\s*)\d+',
                    rf'\g<1>{value}',
                    source
                )
        
        return InstantiatedTemplate(
            source_code=source,
            template_name=template.name,
            variant=template.variant,
            parameters=parameters,
            original_path=template.path,
        )

    def get_optimized_replacement(self, algorithm_type: str,
                                   parameters: dict) -> Optional[InstantiatedTemplate]:
        """One-shot: find best template and instantiate with parameters."""
        template = self.get_best_template(algorithm_type, prefer_optimized=True)
        if not template:
            return None
        return self.instantiate(template, parameters)

    def list_all_templates(self) -> list[TemplateInfo]:
        """List all available templates."""
        self.load()
        result = []
        for templates in self._templates.values():
            result.extend(templates)
        return result

    def get_available_types(self) -> list[str]:
        """Return list of algorithm types with templates."""
        self.load()
        return sorted(self._templates.keys())
