"""ISA Instruction Database for AMDGPU architectures.

Provides lookup for instruction mnemonics, categories, latency, throughput,
and architecture support across multiple gfx generations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DB_DIR = Path(__file__).resolve().parent.parent / "db" / "isa"

SUPPORTED_ARCHS = [
    "gfx900", "gfx906", "gfx908", "gfx90a",
    "gfx940", "gfx941", "gfx942", "gfx950",
]

ARCH_FAMILIES = {
    "gfx9":  ["gfx900", "gfx906", "gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"],
    "cdna":  ["gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"],
    "cdna2": ["gfx90a"],
    "cdna3": ["gfx940", "gfx941", "gfx942"],
    "cdna4": ["gfx950"],
}

INSTRUCTION_CATEGORIES = [
    "SALU",   # Scalar ALU
    "SMEM",   # Scalar memory
    "VALU",   # Vector ALU
    "VOP3P",  # Packed vector ops
    "VMEM",   # Vector memory (buffer/global/scratch)
    "FLAT",   # Flat memory
    "LDS",    # Local data share
    "GDS",    # Global data share
    "MFMA",   # Matrix fused multiply-add
    "WMMA",   # Wave matrix multiply-accumulate
    "EXPORT", # Export
    "BRANCH", # Branch / flow control
    "MSG",    # Message / barrier / sync
    "MISC",   # Misc instructions
]


@dataclass
class Instruction:
    mnemonic: str
    category: str
    description: str
    operands: str
    latency_cycles: int
    throughput_ops_per_cycle: float
    supported_archs: list[str] = field(default_factory=list)
    encoding: str = ""
    notes: str = ""
    new_in: str = ""
    deprecated_in: str = ""
    issue_rate: int = 1
    pipe: str = ""
    can_dual_issue: bool = False

    def supports_arch(self, arch: str) -> bool:
        return arch in self.supported_archs

    def to_dict(self) -> dict:
        return {
            "mnemonic": self.mnemonic,
            "category": self.category,
            "description": self.description,
            "operands": self.operands,
            "latency_cycles": self.latency_cycles,
            "throughput_ops_per_cycle": self.throughput_ops_per_cycle,
            "supported_archs": self.supported_archs,
            "encoding": self.encoding,
            "notes": self.notes,
            "new_in": self.new_in,
            "deprecated_in": self.deprecated_in,
            "issue_rate": self.issue_rate,
            "pipe": self.pipe,
            "can_dual_issue": self.can_dual_issue,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Instruction":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)


class ISADatabase:
    """In-memory ISA instruction database with per-arch lookup."""

    def __init__(self, db_dir: Optional[Path] = None):
        self.db_dir = db_dir or DB_DIR
        self._instructions: dict[str, Instruction] = {}
        self._by_category: dict[str, list[Instruction]] = {}
        self._by_arch: dict[str, list[Instruction]] = {}
        self._detailed: dict[str, dict] = {}  # per-arch detailed data
        self._pipeline_models: dict[str, dict] = {}
        self._loaded = False

    def load(self) -> None:
        """Load all ISA JSON files from the database directory."""
        if self._loaded:
            return
        # Load main DB (amdgpu_isa.json)
        main_file = self.db_dir / "amdgpu_isa.json"
        if main_file.exists():
            with open(main_file) as f:
                data = json.load(f)
            for entry in data.get("instructions", []):
                instr = Instruction.from_dict(entry)
                self._instructions[instr.mnemonic] = instr
                self._by_category.setdefault(instr.category, []).append(instr)
                for arch in instr.supported_archs:
                    self._by_arch.setdefault(arch, []).append(instr)

        # Load per-arch detailed files (gfx942_detailed.json, etc.)
        for arch_file in sorted(self.db_dir.glob("*_detailed.json")):
            with open(arch_file) as f:
                detail = json.load(f)
            arch = detail.get("arch", arch_file.stem.split("_")[0])
            self._detailed[arch] = detail
            if "pipeline_model" in detail:
                self._pipeline_models[arch] = detail["pipeline_model"]
        self._loaded = True

    def get_pipeline_model(self, arch: str) -> Optional[dict]:
        """Get the hardware pipeline model for a specific architecture."""
        self.load()
        return self._pipeline_models.get(arch)

    def get_latency(self, mnemonic: str, arch: str = "gfx942") -> int:
        """Get instruction latency for a specific architecture."""
        self.load()
        if arch in self._detailed:
            for instr in self._detailed[arch].get("instructions", []):
                if instr["mnemonic"] == mnemonic:
                    key = f"latency_{arch}"
                    if key in instr:
                        return instr[key]
                    return instr.get("latency_cycles", 4)
        instr = self.lookup(mnemonic)
        return instr.latency_cycles if instr else 4

    def get_throughput(self, mnemonic: str, arch: str = "gfx942") -> float:
        """Get instruction throughput for a specific architecture."""
        self.load()
        if arch in self._detailed:
            for instr in self._detailed[arch].get("instructions", []):
                if instr["mnemonic"] == mnemonic:
                    key = f"throughput_{arch}"
                    if key in instr:
                        return instr[key]
                    return instr.get("throughput_ops_per_cycle", 1.0)
        instr = self.lookup(mnemonic)
        return instr.throughput_ops_per_cycle if instr else 1.0

    def get_pipe(self, mnemonic: str) -> str:
        """Get the pipeline unit an instruction uses."""
        self.load()
        for arch_data in self._detailed.values():
            for instr in arch_data.get("instructions", []):
                if instr["mnemonic"] == mnemonic:
                    return instr.get("pipe", "VALU")
        instr = self.lookup(mnemonic)
        if instr:
            cat = instr.category
            pipe_map = {
                "SALU": "SALU", "SMEM": "SMEM", "VALU": "VALU",
                "VOP3P": "VALU", "VMEM": "VMEM", "FLAT": "VMEM",
                "LDS": "LDS", "MFMA": "MFMA", "BRANCH": "SALU",
                "MSG": "SALU",
            }
            return pipe_map.get(cat, "VALU")
        return "VALU"

    def lookup(self, mnemonic: str) -> Optional[Instruction]:
        """Look up a single instruction by mnemonic."""
        self.load()
        clean = mnemonic.strip().lower()
        if clean in self._instructions:
            return self._instructions[clean]
        for key, instr in self._instructions.items():
            if clean in key or key in clean:
                return instr
        return None

    def search(self, pattern: str) -> list[Instruction]:
        """Search instructions matching a substring pattern."""
        self.load()
        pattern = pattern.strip().lower()
        return [
            instr for key, instr in self._instructions.items()
            if pattern in key or pattern in instr.description.lower()
        ]

    def get_by_category(self, category: str) -> list[Instruction]:
        self.load()
        return self._by_category.get(category, [])

    def get_by_arch(self, arch: str) -> list[Instruction]:
        self.load()
        return self._by_arch.get(arch, [])

    def get_mfma_instructions(self, arch: str = "gfx942") -> list[Instruction]:
        self.load()
        return [
            i for i in self._by_category.get("MFMA", [])
            if i.supports_arch(arch)
        ]

    def all_instructions(self) -> list[Instruction]:
        self.load()
        return list(self._instructions.values())

    @property
    def instruction_count(self) -> int:
        self.load()
        return len(self._instructions)

    @property
    def arch_count(self) -> int:
        self.load()
        return len(self._by_arch)
