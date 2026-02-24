"""Compiler wrapper for AMDGPU assembly generation.

Wraps amdclang++/hipcc to compile C++/HIP source files to AMDGPU assembly
with configurable architecture targets and compiler flags.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


HIPCC = os.environ.get("HIPCC", "/opt/rocm/bin/hipcc")
AMDCLANG = os.environ.get("AMDCLANG", "/opt/rocm/llvm/bin/amdclang++")
LLVM_OBJDUMP = os.environ.get("LLVM_OBJDUMP", "/opt/rocm/llvm/bin/llvm-objdump")
CLANG_OFFLOAD_BUNDLER = os.environ.get(
    "CLANG_OFFLOAD_BUNDLER", "/opt/rocm/llvm/bin/clang-offload-bundler")

DEFAULT_ARCH = "gfx942"
DEFAULT_OPT_LEVEL = "-O3"


@dataclass
class CompilationResult:
    success: bool
    asm_output: str = ""
    stderr: str = ""
    return_code: int = 0
    source_path: str = ""
    arch: str = ""
    flags: list[str] = field(default_factory=list)
    compiler: str = ""

    @property
    def instruction_lines(self) -> list[str]:
        """Return only instruction lines (excluding directives, labels, comments)."""
        result = []
        for line in self.asm_output.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("//", ";", ".", "@")):
                continue
            if stripped.endswith(":"):
                continue
            result.append(stripped)
        return result


class Compiler:
    """Wrapper around amdclang++/hipcc for compiling HIP/C++ to AMDGPU assembly."""

    def __init__(
        self,
        compiler_path: str = HIPCC,
        default_arch: str = DEFAULT_ARCH,
        default_opt: str = DEFAULT_OPT_LEVEL,
        timeout: int = 120,
    ):
        self.compiler_path = compiler_path
        self.default_arch = default_arch
        self.default_opt = default_opt
        self.timeout = timeout

    def compile_to_asm(
        self,
        source: str | Path,
        arch: Optional[str] = None,
        extra_flags: Optional[list[str]] = None,
        opt_level: Optional[str] = None,
    ) -> CompilationResult:
        """Compile a HIP/C++ source file to AMDGPU assembly.

        Args:
            source: Path to source file, or source code string
            arch: Target GPU architecture (e.g., "gfx942")
            extra_flags: Additional compiler flags
            opt_level: Optimization level (e.g., "-O3")

        Returns:
            CompilationResult with assembly output or error
        """
        arch = arch or self.default_arch
        opt_level = opt_level or self.default_opt
        extra_flags = extra_flags or []

        # Handle source as string (write to temp file)
        if isinstance(source, str) and not os.path.exists(source):
            return self._compile_from_string(source, arch, opt_level, extra_flags)

        source_path = Path(source)
        if not source_path.exists():
            return CompilationResult(
                success=False, stderr=f"Source file not found: {source_path}"
            )

        return self._compile_file(source_path, arch, opt_level, extra_flags)

    def _compile_file(
        self, source_path: Path, arch: str, opt_level: str, extra_flags: list[str]
    ) -> CompilationResult:
        with tempfile.NamedTemporaryFile(suffix=".s", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            self.compiler_path,
            "-S",
            f"--offload-arch={arch}",
            "-nogpulib",
            opt_level,
            *extra_flags,
            str(source_path),
            "-o", tmp_path,
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout
            )
            asm_output = ""
            if result.returncode == 0 and os.path.exists(tmp_path):
                with open(tmp_path) as f:
                    asm_output = f.read()

            return CompilationResult(
                success=result.returncode == 0,
                asm_output=asm_output,
                stderr=result.stderr,
                return_code=result.returncode,
                source_path=str(source_path),
                arch=arch,
                flags=[opt_level] + extra_flags,
                compiler=self.compiler_path,
            )
        except subprocess.TimeoutExpired:
            return CompilationResult(
                success=False, stderr="Compilation timed out",
                source_path=str(source_path), arch=arch,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _compile_from_string(
        self, source_code: str, arch: str, opt_level: str, extra_flags: list[str]
    ) -> CompilationResult:
        with tempfile.NamedTemporaryFile(
            suffix=".hip", delete=False, mode="w"
        ) as tmp:
            tmp.write(source_code)
            src_path = tmp.name

        try:
            result = self._compile_file(Path(src_path), arch, opt_level, extra_flags)
            result.source_path = "<string>"
            return result
        finally:
            os.unlink(src_path)

    def compile_to_co(
        self,
        source: str | Path,
        arch: Optional[str] = None,
        opt_level: Optional[str] = None,
        extra_flags: Optional[list[str]] = None,
        output_path: Optional[str] = None,
    ) -> CompilationResult:
        """Compile a HIP/C++ source file to a loadable .co code object.

        Uses hipcc --genco which produces a fat binary that can be loaded
        via hipModuleLoadData. Host code in the source is ignored.

        Args:
            source: Path to source file, or source code string
            arch: Target GPU architecture (e.g., "gfx942")
            opt_level: Optimization level (e.g., "-O3")
            extra_flags: Additional compiler flags
            output_path: Where to write the .co file (auto-generated if None)

        Returns:
            CompilationResult with asm_output set to the .co file path
        """
        arch = arch or self.default_arch
        opt_level = opt_level or self.default_opt
        extra_flags = extra_flags or []

        # Handle source as string
        tmp_src = None
        if isinstance(source, str) and not os.path.exists(source):
            tmp_src = tempfile.NamedTemporaryFile(
                suffix=".hip", delete=False, mode="w")
            tmp_src.write(source)
            tmp_src.close()
            src_path = Path(tmp_src.name)
        else:
            src_path = Path(source)
            if not src_path.exists():
                return CompilationResult(
                    success=False, stderr=f"Source file not found: {src_path}")

        if output_path is None:
            tmp_out = tempfile.NamedTemporaryFile(suffix=".co", delete=False)
            output_path = tmp_out.name
            tmp_out.close()

        bundle_path = output_path + ".bundle"

        cmd = [
            self.compiler_path,
            "--genco",
            f"--offload-arch={arch}",
            opt_level,
            *extra_flags,
            str(src_path),
            "-o", bundle_path,
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout
            )
            if result.returncode != 0 or not os.path.exists(bundle_path):
                return CompilationResult(
                    success=False,
                    stderr=result.stderr,
                    return_code=result.returncode,
                    source_path=str(src_path),
                    arch=arch,
                    flags=["--genco", opt_level] + extra_flags,
                    compiler=self.compiler_path,
                )

            # hipcc --genco produces a Clang offload bundle; extract the
            # raw device ELF so it works with both hipModuleLoadData and
            # llvm-objdump / AsmEditor.
            unbundle_ok = self._unbundle_co(bundle_path, output_path, arch)
            if not unbundle_ok:
                # Fallback: keep the bundle as-is (hipModuleLoadData still
                # handles it, but llvm-objdump won't work directly).
                os.rename(bundle_path, output_path)

            if os.path.exists(bundle_path):
                os.unlink(bundle_path)

            return CompilationResult(
                success=True,
                asm_output=output_path,
                stderr=result.stderr,
                return_code=result.returncode,
                source_path=str(src_path),
                arch=arch,
                flags=["--genco", opt_level] + extra_flags,
                compiler=self.compiler_path,
            )
        except subprocess.TimeoutExpired:
            return CompilationResult(
                success=False, stderr="Compilation timed out",
                source_path=str(src_path), arch=arch,
            )
        finally:
            if tmp_src is not None:
                os.unlink(tmp_src.name)

    @staticmethod
    def _unbundle_co(bundle_path: str, output_path: str, arch: str) -> bool:
        """Extract device ELF from a Clang offload bundle."""
        target = f"hipv4-amdgcn-amd-amdhsa--{arch}"
        cmd = [
            CLANG_OFFLOAD_BUNDLER,
            "--unbundle",
            f"--input={bundle_path}",
            f"--output={output_path}",
            "--type=o",
            f"--targets={target}",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0 and os.path.exists(output_path)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def compile_multiple_flags(
        self,
        source: str | Path,
        arch: Optional[str] = None,
        flag_sets: Optional[list[list[str]]] = None,
    ) -> list[CompilationResult]:
        """Compile the same source with multiple flag combinations."""
        if flag_sets is None:
            flag_sets = [["-O0"], ["-O1"], ["-O2"], ["-O3"], ["-Ofast"]]

        arch = arch or self.default_arch
        results = []
        for flags in flag_sets:
            opt = flags[0] if flags and flags[0].startswith("-O") else self.default_opt
            extra = [f for f in flags if not f.startswith("-O")]
            result = self.compile_to_asm(source, arch=arch, opt_level=opt, extra_flags=extra)
            results.append(result)
        return results

    def disassemble_binary(self, binary_path: str | Path, arch: Optional[str] = None) -> CompilationResult:
        """Disassemble a .co (code object) binary file using llvm-objdump."""
        arch = arch or self.default_arch
        cmd = [
            LLVM_OBJDUMP,
            "-d",
            f"--mcpu={arch}",
            str(binary_path),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout
            )
            return CompilationResult(
                success=result.returncode == 0,
                asm_output=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                source_path=str(binary_path),
                arch=arch,
                compiler=LLVM_OBJDUMP,
            )
        except subprocess.TimeoutExpired:
            return CompilationResult(
                success=False, stderr="Disassembly timed out",
                source_path=str(binary_path), arch=arch,
            )
