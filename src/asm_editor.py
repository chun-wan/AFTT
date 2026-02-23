"""ASM Editor: Disassemble .co -> Parse -> Apply edits -> Reassemble.

Provides two edit strategies:
1. Binary patching: For same-length instruction replacements (fast, reliable)
2. Full reassembly: For insertions/deletions via llvm-mc + ld.lld

The binary patching approach is preferred when possible because it preserves
all ELF metadata, kernel descriptors, and the .note section exactly.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


LLVM_BIN = Path("/opt/rocm-7.1.1/lib/llvm/bin")
LLVM_OBJDUMP = LLVM_BIN / "llvm-objdump"
LLVM_OBJCOPY = LLVM_BIN / "llvm-objcopy"
LLVM_MC = LLVM_BIN / "llvm-mc"
LLD = LLVM_BIN / "ld.lld"


@dataclass
class AsmInstruction:
    """A single parsed ASM instruction from disassembly."""
    address: int            # Virtual memory address (VMA)
    raw_bytes: bytes        # Raw instruction encoding
    mnemonic: str           # Instruction mnemonic
    operands: str           # Operand string
    line_number: int = 0    # Line number in disassembly output
    file_offset: int = 0    # Offset within .co file

    @property
    def size(self) -> int:
        return len(self.raw_bytes)

    @property
    def full_text(self) -> str:
        return f"{self.mnemonic} {self.operands}".strip()

    def __repr__(self) -> str:
        hex_bytes = self.raw_bytes.hex().upper()
        return f"0x{self.address:08X}: {hex_bytes:16s} {self.full_text}"


@dataclass
class KernelInfo:
    """Metadata about a kernel in a .co file."""
    name: str
    text_vma: int           # VMA of .text section start
    text_offset: int        # File offset of .text section
    text_size: int          # Size of .text section
    arch: str = "gfx942"


@dataclass
class EditOperation:
    """A single edit to apply to an instruction."""
    target_index: int       # Index in instruction list
    new_mnemonic: str       # New instruction mnemonic
    new_operands: str       # New operand string
    comment: str = ""       # Why this edit was made


class AsmEditor:
    """Disassemble, edit, and reassemble AMDGPU .co code objects."""

    def __init__(self, arch: str = "gfx942"):
        self.arch = arch
        self._verify_tools()

    def _verify_tools(self) -> None:
        """Check that required LLVM tools exist."""
        for tool in [LLVM_OBJDUMP, LLVM_MC]:
            if not tool.exists():
                raise FileNotFoundError(f"Required tool not found: {tool}")

    def disassemble(self, co_path: str) -> tuple[KernelInfo, list[AsmInstruction]]:
        """Disassemble a .co file into structured instructions.

        Returns kernel metadata and a list of parsed instructions.
        """
        co_path = str(co_path)
        if not os.path.exists(co_path):
            raise FileNotFoundError(f"Code object not found: {co_path}")

        kernel_info = self._get_kernel_info(co_path)
        instructions = self._parse_disassembly(co_path, kernel_info)
        return kernel_info, instructions

    def _get_kernel_info(self, co_path: str) -> KernelInfo:
        """Extract kernel metadata from ELF headers."""
        result = subprocess.run(
            [str(LLVM_OBJDUMP), "--section-headers", co_path],
            capture_output=True, text=True, timeout=30,
        )
        text_vma = 0
        text_size = 0
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[1] == ".text":
                text_size = int(parts[2], 16)
                text_vma = int(parts[3], 16)
                break

        # Get file offset: parse ELF to find .text section file offset
        text_offset = self._find_text_file_offset(co_path, text_vma)

        # Get kernel name from symbols
        sym_result = subprocess.run(
            [str(LLVM_OBJDUMP), "--syms", co_path],
            capture_output=True, text=True, timeout=30,
        )
        kernel_name = "unknown"
        for line in sym_result.stdout.splitlines():
            if ".text" in line and ("g" in line.split()[1:2] or "F" in line):
                parts = line.split()
                if parts:
                    kernel_name = parts[-1]
                    break
        if kernel_name == "unknown":
            kernel_name = Path(co_path).stem

        return KernelInfo(
            name=kernel_name,
            text_vma=text_vma,
            text_offset=text_offset,
            text_size=text_size,
            arch=self.arch,
        )

    def _find_text_file_offset(self, co_path: str, text_vma: int) -> int:
        """Find the file offset of the .text section by reading ELF headers."""
        with open(co_path, "rb") as f:
            # ELF64 header
            magic = f.read(4)
            if magic != b'\x7fELF':
                raise ValueError(f"Not an ELF file: {co_path}")
            f.seek(0)
            header = f.read(64)

            ei_class = header[4]
            if ei_class != 2:
                raise ValueError("Expected 64-bit ELF")

            # e_shoff: section header table offset
            e_shoff = struct.unpack_from('<Q', header, 40)[0]
            e_shentsize = struct.unpack_from('<H', header, 58)[0]
            e_shnum = struct.unpack_from('<H', header, 60)[0]
            e_shstrndx = struct.unpack_from('<H', header, 62)[0]

            # Read section header string table
            f.seek(e_shoff + e_shstrndx * e_shentsize)
            shstrtab_hdr = f.read(e_shentsize)
            shstrtab_offset = struct.unpack_from('<Q', shstrtab_hdr, 24)[0]
            shstrtab_size = struct.unpack_from('<Q', shstrtab_hdr, 32)[0]
            f.seek(shstrtab_offset)
            shstrtab = f.read(shstrtab_size)

            # Find .text section
            for i in range(e_shnum):
                f.seek(e_shoff + i * e_shentsize)
                sh = f.read(e_shentsize)
                sh_name_idx = struct.unpack_from('<I', sh, 0)[0]
                name_end = shstrtab.index(b'\0', sh_name_idx)
                name = shstrtab[sh_name_idx:name_end].decode('ascii')
                if name == ".text":
                    sh_offset = struct.unpack_from('<Q', sh, 24)[0]
                    return sh_offset

        return text_vma  # fallback

    def _parse_disassembly(self, co_path: str, info: KernelInfo) -> list[AsmInstruction]:
        """Parse llvm-objdump output into structured instructions.

        llvm-objdump format for AMDGPU:
            \\tmnemonic operands   // ADDR: HEXBYTES
        Example:
            \\ts_mov_b32 s49, s4   // 000000002900: BEB10004
        """
        result = subprocess.run(
            [str(LLVM_OBJDUMP), "-d", f"--mcpu={self.arch}", co_path],
            capture_output=True, text=True, timeout=60,
        )

        instructions = []
        # Match: leading whitespace, mnemonic+operands, // address: hex_bytes
        pattern = re.compile(
            r'^\s+'                            # leading whitespace
            r'(\S+)'                           # mnemonic
            r'(.*?)'                           # operands (may be empty)
            r'\s*//\s*'                        # comment separator
            r'([0-9a-fA-F]+):\s+'              # address
            r'([0-9a-fA-F ]+?)\s*$'            # hex bytes
        )

        # Pre-read the entire .text section for raw byte extraction
        with open(co_path, "rb") as f:
            f.seek(info.text_offset)
            text_bytes = f.read(info.text_size)

        for line_no, line in enumerate(result.stdout.splitlines()):
            m = pattern.match(line)
            if not m:
                continue

            mnemonic, operands, addr_str, hex_str = m.groups()
            addr = int(addr_str, 16)

            # Determine instruction size from the hex string in objdump
            # (displayed as big-endian 32-bit words, but we read actual
            # bytes from file to get correct little-endian encoding)
            hex_clean = hex_str.strip().replace(" ", "")
            instr_size = len(hex_clean) // 2

            file_offset = info.text_offset + (addr - info.text_vma)
            text_local = addr - info.text_vma
            raw_bytes = text_bytes[text_local:text_local + instr_size]

            instr = AsmInstruction(
                address=addr,
                raw_bytes=raw_bytes,
                mnemonic=mnemonic.strip(),
                operands=operands.strip().rstrip(","),
                line_number=line_no,
                file_offset=file_offset,
            )
            instructions.append(instr)

        return instructions

    def encode_instruction(self, mnemonic: str, operands: str) -> bytes:
        """Encode a single instruction using llvm-mc."""
        asm_text = f"\t{mnemonic} {operands}\n"
        result = subprocess.run(
            [str(LLVM_MC), f"--arch=amdgcn", f"--mcpu={self.arch}", "--show-encoding"],
            input=asm_text, capture_output=True, text=True, timeout=10,
        )

        if result.returncode != 0:
            raise ValueError(f"llvm-mc encoding failed for '{mnemonic} {operands}': {result.stderr}")

        # Parse encoding from output: "; encoding: [0x00,0x00,0x80,0xbf]"
        enc_match = re.search(r'encoding:\s*\[([^\]]+)\]', result.stdout)
        if not enc_match:
            raise ValueError(f"Could not parse encoding from: {result.stdout}")

        hex_vals = enc_match.group(1).split(",")
        return bytes(int(h.strip(), 16) for h in hex_vals)

    def binary_patch(self, co_path: str, output_path: str,
                     edits: list[EditOperation],
                     instructions: list[AsmInstruction]) -> dict:
        """Apply edits via binary patching (same-length replacements only).

        This is the preferred method as it preserves all ELF metadata exactly.
        Returns a summary of applied patches.
        """
        shutil.copy2(co_path, output_path)

        applied = []
        skipped = []

        with open(output_path, "r+b") as f:
            for edit in edits:
                orig = instructions[edit.target_index]
                try:
                    new_bytes = self.encode_instruction(edit.new_mnemonic, edit.new_operands)
                except ValueError as e:
                    skipped.append({"index": edit.target_index, "reason": str(e)})
                    continue

                if len(new_bytes) != orig.size:
                    skipped.append({
                        "index": edit.target_index,
                        "reason": f"Size mismatch: original {orig.size}B, new {len(new_bytes)}B",
                    })
                    continue

                f.seek(orig.file_offset)
                verify = f.read(orig.size)
                if verify != orig.raw_bytes:
                    skipped.append({
                        "index": edit.target_index,
                        "reason": f"Verification failed: expected {orig.raw_bytes.hex()}, got {verify.hex()}",
                    })
                    continue

                f.seek(orig.file_offset)
                f.write(new_bytes)
                applied.append({
                    "index": edit.target_index,
                    "address": f"0x{orig.address:08X}",
                    "original": orig.full_text,
                    "replacement": f"{edit.new_mnemonic} {edit.new_operands}",
                    "comment": edit.comment,
                    "old_bytes": orig.raw_bytes.hex(),
                    "new_bytes": new_bytes.hex(),
                })

        return {
            "method": "binary_patch",
            "applied": applied,
            "skipped": skipped,
            "total_edits": len(edits),
            "applied_count": len(applied),
            "skipped_count": len(skipped),
        }

    def verify_patch(self, original_co: str, modified_co: str) -> dict:
        """Verify a patched .co by re-disassembling and comparing."""
        _, orig_instrs = self.disassemble(original_co)
        _, mod_instrs = self.disassemble(modified_co)

        differences = []
        for i, (o, m) in enumerate(zip(orig_instrs, mod_instrs)):
            if o.raw_bytes != m.raw_bytes:
                differences.append({
                    "index": i,
                    "address": f"0x{o.address:08X}",
                    "original": o.full_text,
                    "modified": m.full_text,
                })

        return {
            "original_instructions": len(orig_instrs),
            "modified_instructions": len(mod_instrs),
            "count_match": len(orig_instrs) == len(mod_instrs),
            "differences": differences,
            "difference_count": len(differences),
        }

    def get_instruction_lines(self, instructions: list[AsmInstruction]) -> list[str]:
        """Convert instructions back to ASM text lines (for cycle estimation)."""
        return [instr.full_text for instr in instructions]

    def apply_and_get_modified_lines(self, instructions: list[AsmInstruction],
                                      edits: list[EditOperation]) -> list[str]:
        """Apply edits to instruction list and return modified ASM lines."""
        edit_map = {e.target_index: e for e in edits}
        lines = []
        for i, instr in enumerate(instructions):
            if i in edit_map:
                e = edit_map[i]
                lines.append(f"{e.new_mnemonic} {e.new_operands}")
            else:
                lines.append(instr.full_text)
        return lines
