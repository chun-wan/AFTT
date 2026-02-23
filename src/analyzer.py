"""Pattern Matching Analyzer for AMDGPU Assembly.

Cross-references parsed ASM against the knowledge base to find issues,
anti-patterns, and suggest optimizations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .parser import ParsedKernel, AsmInstruction
from .knowledge_base import KnowledgeBase


@dataclass
class Finding:
    """A single analysis finding (issue or suggestion)."""
    finding_id: str
    severity: str  # "critical", "warning", "info"
    category: str
    title: str
    description: str
    suggestion: str
    line_numbers: list[int] = field(default_factory=list)
    related_instructions: list[str] = field(default_factory=list)
    reference: str = ""
    pattern_id: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete analysis result for a kernel."""
    kernel_name: str = ""
    arch: str = ""
    findings: list[Finding] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "info")


class Analyzer:
    """Pattern matching engine for AMDGPU assembly analysis."""

    def __init__(self, kb: Optional[KnowledgeBase] = None):
        self.kb = kb or KnowledgeBase()
        self.kb.load()
        self._finding_counter = 0

    def _next_id(self) -> str:
        self._finding_counter += 1
        return f"F{self._finding_counter:04d}"

    def analyze(self, kernel: ParsedKernel, arch: str = "gfx942") -> AnalysisResult:
        """Run all analysis passes on a parsed kernel."""
        self._finding_counter = 0
        result = AnalysisResult(
            kernel_name=kernel.metadata.name or "unknown",
            arch=arch,
        )

        # Original 12 checks
        self._check_register_pressure(kernel, arch, result)
        self._check_waitcnt_patterns(kernel, result)
        self._check_memory_patterns(kernel, result)
        self._check_lds_patterns(kernel, result)
        self._check_mfma_usage(kernel, arch, result)
        self._check_barrier_usage(kernel, result)
        self._check_vectorization(kernel, result)
        self._check_instruction_mix(kernel, result)
        self._check_fp8_opportunity(kernel, arch, result)
        self._check_prefetch_patterns(kernel, result)
        self._check_nop_overhead(kernel, result)
        self._check_scratch_usage(kernel, result)

        # 20 new checks based on production kernel patterns
        self._check_mfma_chaining(kernel, result)
        self._check_sched_barrier_usage(kernel, result)
        self._check_double_buffer_lds(kernel, result)
        self._check_prefetch_depth(kernel, result)
        self._check_mfma_vmem_interleaving(kernel, result)
        self._check_accvgpr_usage(kernel, result)
        self._check_setprio_scheduling(kernel, arch, result)
        self._check_gfx950_opportunities(kernel, arch, result)
        self._check_ds_read_write_balance(kernel, result)
        self._check_partial_waitcnt_effectiveness(kernel, result)
        self._check_nop_near_mfma(kernel, result)
        self._check_global_vs_buffer_load(kernel, result)
        self._check_instruction_density(kernel, result)
        self._check_agpr_partitioning(kernel, arch, result)
        self._check_mfma_type_optimization(kernel, arch, result)
        self._check_bf16_vs_fp16_mfma(kernel, arch, result)
        self._check_load_store_symmetry(kernel, result)
        self._check_loop_structure(kernel, result)
        self._check_wavefront_size_alignment(kernel, arch, result)
        self._check_flat_vs_global_addressing(kernel, result)
        self._check_mfma_utilization_ratio(kernel, result)

        self._compute_summary(kernel, arch, result)

        return result

    def _check_register_pressure(self, kernel: ParsedKernel, arch: str, result: AnalysisResult):
        """Check for high register pressure that limits occupancy."""
        vgpr_count = kernel.metadata.vgpr_count or (kernel.register_usage.max_vgpr + 1)
        sgpr_count = kernel.metadata.sgpr_count or (kernel.register_usage.max_sgpr + 1)

        if vgpr_count == 0:
            return

        # CDNA3 occupancy thresholds
        if arch in ("gfx940", "gfx941", "gfx942"):
            max_waves = 8
            vgprs_per_simd = 512
            waves_possible = min(max_waves, vgprs_per_simd // max(vgpr_count, 1))

            if waves_possible <= 1:
                result.findings.append(Finding(
                    finding_id=self._next_id(),
                    severity="critical",
                    category="resource",
                    title="Very High Register Pressure (Occupancy = 1 wave)",
                    description=f"Kernel uses {vgpr_count} VGPRs, limiting occupancy to {waves_possible} wave(s) per SIMD. "
                                f"On {arch}, each SIMD has {vgprs_per_simd} VGPRs with max {max_waves} waves.",
                    suggestion="Reduce VGPR usage below 256 to allow at least 2 waves. Consider using __launch_bounds__, "
                               "reducing tile sizes, or offloading some values to LDS.",
                    pattern_id="AP004",
                    metrics={"vgpr_count": vgpr_count, "waves_possible": waves_possible},
                    reference="AMD Occupancy Calculator",
                ))
            elif waves_possible <= 2:
                result.findings.append(Finding(
                    finding_id=self._next_id(),
                    severity="warning",
                    category="resource",
                    title=f"High Register Pressure (Occupancy = {waves_possible} waves)",
                    description=f"Kernel uses {vgpr_count} VGPRs ({sgpr_count} SGPRs), "
                                f"allowing only {waves_possible} wave(s) per SIMD out of max {max_waves}.",
                    suggestion="Consider if the register usage can be reduced. For memory-bound kernels, "
                               "higher occupancy helps hide latency.",
                    pattern_id="AP004",
                    metrics={"vgpr_count": vgpr_count, "sgpr_count": sgpr_count, "waves_possible": waves_possible},
                ))

    def _check_waitcnt_patterns(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check for excessive or suboptimal s_waitcnt usage."""
        waitcnt_zero_count = 0
        total_waitcnt = 0
        waitcnt_lines = []

        for instr in kernel.instructions:
            if instr.is_waitcnt:
                total_waitcnt += 1
                raw = instr.raw_text
                if "vmcnt(0)" in raw or "lgkmcnt(0)" in raw:
                    waitcnt_zero_count += 1
                    waitcnt_lines.append(instr.line_number)

        if total_waitcnt == 0:
            return

        zero_ratio = waitcnt_zero_count / total_waitcnt if total_waitcnt else 0

        if zero_ratio > 0.7 and waitcnt_zero_count > 3:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="warning",
                category="pipeline",
                title="Excessive Full-Stall Waitcnts",
                description=f"{waitcnt_zero_count} out of {total_waitcnt} s_waitcnt instructions use vmcnt(0) or lgkmcnt(0), "
                            f"which stalls until ALL outstanding memory ops complete. Ratio: {zero_ratio:.0%}.",
                suggestion="Use partial waitcnts (vmcnt(N) where N>0) to allow some memory ops to remain in-flight. "
                           "This enables overlapping memory latency with computation.",
                line_numbers=waitcnt_lines[:10],
                pattern_id="AP003",
                metrics={"waitcnt_zero": waitcnt_zero_count, "total_waitcnt": total_waitcnt, "ratio": zero_ratio},
                reference="AMD s_waitcnt best practices",
            ))

    def _check_memory_patterns(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check global memory access patterns for potential issues."""
        # Look for sequences of single-dword loads that could be vectorized
        consecutive_single_loads = 0
        max_consecutive = 0
        load_lines = []

        for instr in kernel.instructions:
            if instr.mnemonic in ("global_load_dword", "buffer_load_dword"):
                consecutive_single_loads += 1
                load_lines.append(instr.line_number)
            else:
                max_consecutive = max(max_consecutive, consecutive_single_loads)
                consecutive_single_loads = 0

        max_consecutive = max(max_consecutive, consecutive_single_loads)

        if max_consecutive >= 4:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="warning",
                category="memory",
                title="Consecutive Single-Dword Global Loads",
                description=f"Found {max_consecutive} consecutive single-dword (32-bit) global loads. "
                            "These could potentially be combined into wider dwordx4 (128-bit) loads.",
                suggestion="Use vectorized types (float4, int4, half8) to enable 128-bit loads. "
                           "Ensure alignment to 16 bytes. Check with -O3 if compiler auto-vectorizes.",
                line_numbers=load_lines[:8],
                pattern_id="AP009",
                metrics={"consecutive_single_loads": max_consecutive},
            ))

    def _check_lds_patterns(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check LDS access patterns for bank conflicts."""
        lds_ops = [i for i in kernel.instructions if i.is_lds_op]
        if not lds_ops:
            return

        lds_size = kernel.metadata.lds_size
        if lds_size > 0:
            # LDS size occupancy check
            max_lds_per_cu = 65536  # 64KB for CDNA3
            max_workgroups = max_lds_per_cu // max(lds_size, 1)
            if max_workgroups <= 1:
                result.findings.append(Finding(
                    finding_id=self._next_id(),
                    severity="warning",
                    category="resource",
                    title="Large LDS Allocation Limits Workgroup Concurrency",
                    description=f"Kernel allocates {lds_size} bytes of LDS per workgroup. "
                                f"With {max_lds_per_cu} bytes per CU, only {max_workgroups} workgroup(s) "
                                "can run concurrently per CU.",
                    suggestion="Reduce LDS usage to allow more concurrent workgroups. Consider multi-pass "
                               "algorithms that use less LDS per pass.",
                    metrics={"lds_bytes": lds_size, "max_workgroups_per_cu": max_workgroups},
                ))

    def _check_mfma_usage(self, kernel: ParsedKernel, arch: str, result: AnalysisResult):
        """Check MFMA instruction usage patterns."""
        mfma_instrs = [i for i in kernel.instructions if i.is_mfma]

        # Check for small MFMA tiles
        small_mfma = [i for i in mfma_instrs if "4x4x4" in i.mnemonic]
        large_mfma = [i for i in mfma_instrs if "32x32" in i.mnemonic or "16x16" in i.mnemonic]

        if small_mfma and not large_mfma:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="compute",
                title="Only Small MFMA Tiles Used",
                description=f"Found {len(small_mfma)} small (4x4x4) MFMA instructions but no larger tiles. "
                            "For large matrix operations, larger tiles provide better throughput.",
                suggestion="Consider using v_mfma_f32_32x32x8_f16 (512 FLOPs/inst) or "
                           "v_mfma_f32_16x16x16_f16 (256 FLOPs/inst) for better arithmetic intensity.",
                line_numbers=[i.line_number for i in small_mfma[:5]],
                pattern_id="AP006",
            ))

        # Check for FMA chains that could be MFMA
        if kernel.mfma_count == 0:
            fma_instrs = [
                i for i in kernel.instructions
                if i.mnemonic in ("v_fma_f32", "v_fmac_f32_e32", "v_fma_f16")
            ]
            if len(fma_instrs) >= 8 and arch in ("gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"):
                result.findings.append(Finding(
                    finding_id=self._next_id(),
                    severity="critical",
                    category="compute",
                    title="No MFMA Instructions in Compute-Heavy Kernel",
                    description=f"Found {len(fma_instrs)} scalar FMA instructions but zero MFMA. "
                                f"On {arch}, MFMA provides 10-100x higher matrix multiply throughput.",
                    suggestion="Restructure the computation to use MFMA tiles. Use CK or aiter GEMM "
                               "primitives for optimized MFMA-based implementations.",
                    line_numbers=[i.line_number for i in fma_instrs[:5]],
                    pattern_id="AP007",
                    metrics={"fma_count": len(fma_instrs)},
                ))

    def _check_barrier_usage(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check for redundant barriers."""
        prev_was_barrier = False
        redundant_count = 0
        redundant_lines = []

        for instr in kernel.instructions:
            if instr.is_barrier:
                if prev_was_barrier:
                    redundant_count += 1
                    redundant_lines.append(instr.line_number)
                prev_was_barrier = True
            elif instr.is_lds_op or instr.is_memory_op:
                prev_was_barrier = False
            # Non-memory, non-barrier instructions don't reset the flag

        if redundant_count > 0:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="sync",
                title=f"Potentially Redundant Barriers ({redundant_count})",
                description=f"Found {redundant_count} barrier(s) that appear back-to-back or without "
                            "intervening LDS/memory operations.",
                suggestion="Remove redundant barriers. Only use s_barrier when there is a "
                           "producer-consumer LDS dependency between threads.",
                line_numbers=redundant_lines,
                pattern_id="AP008",
            ))

    def _check_vectorization(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check memory access vectorization level."""
        single_loads = sum(
            1 for i in kernel.instructions
            if i.mnemonic in ("global_load_dword", "buffer_load_dword", "flat_load_dword")
        )
        wide_loads = sum(
            1 for i in kernel.instructions
            if any(w in i.mnemonic for w in ("dwordx4", "dwordx2"))
            and "load" in i.mnemonic
        )

        total_loads = single_loads + wide_loads
        if total_loads > 4 and single_loads > wide_loads * 3:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="warning",
                category="memory",
                title="Low Memory Access Vectorization",
                description=f"{single_loads} single-dword loads vs {wide_loads} wide loads. "
                            f"Only {100 * wide_loads / max(total_loads, 1):.0f}% of loads are vectorized.",
                suggestion="Use float4/half8 types for vectorized 128-bit loads. Ensure 16-byte alignment. "
                           "Consider reinterpret_cast to wider types for contiguous data.",
                pattern_id="AP009",
                metrics={"single_loads": single_loads, "wide_loads": wide_loads},
            ))

    def _check_instruction_mix(self, kernel: ParsedKernel, result: AnalysisResult):
        """Analyze the instruction mix balance."""
        if kernel.total_instructions < 10:
            return

        compute = kernel.valu_count + kernel.mfma_count
        memory = kernel.vmem_count + kernel.smem_count + kernel.lds_count

        if memory == 0:
            return

        ratio = compute / max(memory, 1)

        if ratio < 1.5:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="compute",
                title="Memory-Dominated Instruction Mix",
                description=f"Compute-to-memory instruction ratio is {ratio:.1f}:1 "
                            f"({compute} compute, {memory} memory). "
                            "This kernel appears to be memory-bandwidth-bound.",
                suggestion="Increase data reuse via tiling and LDS caching. Use wider loads. "
                           "Consider algorithmic changes to increase arithmetic intensity.",
                metrics={"compute_ratio": ratio, "compute": compute, "memory": memory},
            ))
        elif ratio > 50:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="compute",
                title="Compute-Dominated Instruction Mix",
                description=f"Compute-to-memory instruction ratio is {ratio:.1f}:1. "
                            "This kernel is compute-bound.",
                suggestion="Ensure MFMA utilization is high. Consider lower precision (FP8 vs FP16) "
                           "to improve compute throughput.",
                metrics={"compute_ratio": ratio},
            ))

    def _check_fp8_opportunity(self, kernel: ParsedKernel, arch: str, result: AnalysisResult):
        """Check if FP8 MFMA could be used on CDNA3+."""
        if arch not in ("gfx940", "gfx941", "gfx942", "gfx950"):
            return

        fp16_mfma = [i for i in kernel.instructions if "mfma" in i.mnemonic and "f16" in i.mnemonic]
        fp8_mfma = [i for i in kernel.instructions if "mfma" in i.mnemonic and "fp8" in i.mnemonic]

        if fp16_mfma and not fp8_mfma:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="precision",
                title=f"FP8 MFMA Available on {arch} but Not Used",
                description=f"Found {len(fp16_mfma)} FP16 MFMA instructions. On {arch}, FP8 MFMA provides "
                            "2x throughput (e.g., v_mfma_f32_32x32x16_fp8_fp8 vs v_mfma_f32_32x32x8_f16).",
                suggestion="For inference workloads with acceptable accuracy, consider FP8 quantization "
                           "with block-scaling to achieve 2x compute throughput.",
                line_numbers=[i.line_number for i in fp16_mfma[:5]],
                pattern_id="AP010",
                metrics={"fp16_mfma_count": len(fp16_mfma)},
            ))

    def _check_prefetch_patterns(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check for missing prefetch/software pipelining."""
        if kernel.total_instructions < 20:
            return

        # Look for load-wait-compute without interleaving
        for i in range(len(kernel.instructions) - 2):
            instr = kernel.instructions[i]
            if not instr.is_memory_op:
                continue

            # Check if immediate next instructions are waitcnt then compute
            next_instrs = kernel.instructions[i + 1:i + 4]
            has_immediate_wait = any(n.is_waitcnt for n in next_instrs[:2])
            has_compute_after = any(
                n.category in ("VALU", "VOP3P", "MFMA") for n in next_instrs
            )

            if has_immediate_wait and has_compute_after:
                # Check if there's another load before the waitcnt (prefetch)
                has_prefetch = False
                for j in range(max(0, i - 5), i):
                    if kernel.instructions[j].is_memory_op:
                        has_prefetch = True
                        break

                if not has_prefetch:
                    result.findings.append(Finding(
                        finding_id=self._next_id(),
                        severity="warning",
                        category="pipeline",
                        title="No Prefetch Before Load-Stall Pattern",
                        description="A load instruction is immediately followed by a wait and compute, "
                                    "with no evidence of prefetching from a previous iteration.",
                        suggestion="Implement software pipelining: issue loads for the next iteration "
                                   "before waiting for the current one. Use double buffering for LDS data.",
                        line_numbers=[instr.line_number],
                        pattern_id="AP005",
                    ))
                    break  # Only report once

    def _check_nop_overhead(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check for excessive NOPs."""
        if kernel.nop_count > 0 and kernel.total_instructions > 0:
            nop_pct = 100 * kernel.nop_count / kernel.total_instructions
            if nop_pct > 10:
                result.findings.append(Finding(
                    finding_id=self._next_id(),
                    severity="info",
                    category="pipeline",
                    title=f"High NOP Overhead ({nop_pct:.1f}%)",
                    description=f"{kernel.nop_count} s_nop instructions out of {kernel.total_instructions} total "
                                f"({nop_pct:.1f}%). NOPs are used for hazard avoidance but reduce effective throughput.",
                    suggestion="Some NOPs are required by hardware hazards and cannot be removed. "
                               "Check if instruction reordering could reduce the number needed.",
                    metrics={"nop_count": kernel.nop_count, "nop_pct": nop_pct},
                ))

    def _check_scratch_usage(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check for scratch memory usage (register spills)."""
        scratch_size = kernel.metadata.scratch_size
        if scratch_size > 0:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="critical",
                category="resource",
                title=f"Register Spilling to Scratch Memory ({scratch_size} bytes)",
                description=f"Kernel uses {scratch_size} bytes of scratch (off-chip) memory per thread. "
                            "Scratch access has global memory latency (~300 cycles), severely impacting performance.",
                suggestion="Reduce register usage to eliminate spilling. Use __launch_bounds__ to hint "
                           "the compiler. Break the kernel into smaller kernels. Offload some data to LDS.",
                metrics={"scratch_bytes": scratch_size},
                reference="AMD scratch memory best practices",
            ))

    # ================================================================
    # NEW CHECKS based on real production kernel pattern analysis
    # ================================================================

    def _check_mfma_chaining(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check MFMA chaining quality against production kernel baselines.

        Production kernels average 17.8-instruction MFMA chains with max 96.
        Short chains indicate poor compute utilization.
        """
        mfma_instrs = [i for i in kernel.instructions if i.is_mfma]
        if len(mfma_instrs) < 2:
            return

        chains = []
        current_chain = 0
        for instr in kernel.instructions:
            if instr.is_mfma:
                current_chain += 1
            elif instr.is_barrier or instr.is_waitcnt:
                if current_chain > 0:
                    chains.append(current_chain)
                current_chain = 0
        if current_chain > 0:
            chains.append(current_chain)

        if not chains:
            return

        avg_chain = sum(chains) / len(chains)
        max_chain = max(chains)

        # Production baseline: avg ~18, max ~96
        if avg_chain < 4:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="warning",
                category="compute",
                title="Short MFMA Chains (Poor Compute Pipelining)",
                description=f"Average MFMA chain length is {avg_chain:.1f} (max {max_chain}). "
                            f"Production kernels average 18 instructions per chain. "
                            f"Short chains indicate frequent pipeline stalls between MFMA groups.",
                suggestion="Increase MFMA chain length by batching more matrix operations before "
                           "synchronization. Use software pipelining to overlap data loading with "
                           "MFMA compute. See CK v4 ping-pong pipeline for reference.",
                metrics={"avg_chain": avg_chain, "max_chain": max_chain,
                         "total_chains": len(chains), "production_avg": 17.8},
            ))
        elif avg_chain >= 16:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="compute",
                title=f"Good MFMA Chain Length ({avg_chain:.0f} avg)",
                description=f"Average MFMA chain length is {avg_chain:.1f} (max {max_chain}), "
                            "which is comparable to production kernel baselines.",
                suggestion="No action needed. Chain length is at production quality.",
                metrics={"avg_chain": avg_chain, "max_chain": max_chain},
            ))

    def _check_sched_barrier_usage(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check for __builtin_amdgcn_sched_group_barrier patterns in compiled output.

        CK production kernels use sched_group_barrier masks:
        0x008=MFMA, 0x100=DS_READ, 0x200=DS_WRITE, 0x020=VMEM_READ
        """
        has_mfma = kernel.mfma_count > 0
        has_lds = kernel.lds_count > 0

        # Look for s_sched_group_barrier or equivalent patterns in output
        sched_instrs = [i for i in kernel.instructions
                       if "sched" in i.mnemonic or "s_setprio" in i.mnemonic]

        if has_mfma and has_lds and kernel.mfma_count > 8 and not sched_instrs:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="scheduling",
                title="No Scheduling Barriers in MFMA+LDS Kernel",
                description=f"Kernel has {kernel.mfma_count} MFMA and {kernel.lds_count} LDS instructions "
                            "but no scheduling barriers. Production CK kernels use "
                            "__builtin_amdgcn_sched_group_barrier to explicitly interleave MFMA "
                            "with memory operations.",
                suggestion="Add scheduling barriers to control instruction interleaving. "
                           "Use masks: 0x008 (MFMA), 0x100 (DS_READ), 0x200 (DS_WRITE), 0x020 (VMEM_READ). "
                           "Example: sched_group_barrier(0x008, 1) -> sched_group_barrier(0x100, N).",
                metrics={"mfma_count": kernel.mfma_count, "lds_count": kernel.lds_count},
                reference="CK pipeline sched_group_barrier patterns",
            ))

    def _check_double_buffer_lds(self, kernel: ParsedKernel, result: AnalysisResult):
        """Detect and verify LDS double-buffering patterns.

        Production kernels show 94% use double-buffering (2460/2604 with LDS).
        """
        lds_ops = [i for i in kernel.instructions if i.is_lds_op]
        if len(lds_ops) < 4:
            return

        # Check LDS allocation size
        lds_size = kernel.metadata.lds_size
        if lds_size == 0:
            return

        has_mfma = kernel.mfma_count > 0

        # Heuristic: if LDS size is relatively small and kernel has lots of MFMA,
        # it might benefit from double-buffering
        if has_mfma and kernel.mfma_count > 16:
            # Count barrier points between LDS write and LDS read sections
            barrier_count = kernel.barrier_count
            lds_per_barrier = len(lds_ops) / max(barrier_count, 1)

            if barrier_count > 4 and lds_per_barrier < 3:
                result.findings.append(Finding(
                    finding_id=self._next_id(),
                    severity="warning",
                    category="pipeline",
                    title="High Barrier-to-LDS Ratio (Consider Double Buffering)",
                    description=f"Found {barrier_count} barriers for {len(lds_ops)} LDS operations "
                                f"({lds_per_barrier:.1f} LDS ops per barrier). Production kernels "
                                "use double-buffered LDS (ping-pong) to eliminate read-after-write barriers.",
                    suggestion="Allocate 2x LDS space and alternate between buffers: fill buffer A while "
                               "computing from buffer B. This eliminates sync between LDS write and read. "
                               "See CK v4 pipeline for implementation reference.",
                    metrics={"barrier_count": barrier_count, "lds_ops": len(lds_ops),
                             "lds_size": lds_size, "production_double_buffer_rate": "94%"},
                    reference="CK gemm_pipeline_ag_bg_cr_comp_v4 ping-pong pattern",
                ))

    def _check_prefetch_depth(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check prefetch depth against production kernel baselines.

        Production kernels average 7.8 loads before first wait, max 57.
        """
        loads = [i for i in kernel.instructions
                if i.is_memory_op and "load" in i.mnemonic]
        if not loads:
            return

        # Count loads before first waitcnt
        loads_before_wait = 0
        found_load = False
        for instr in kernel.instructions:
            if instr.is_memory_op and "load" in instr.mnemonic:
                found_load = True
                loads_before_wait += 1
            elif instr.is_waitcnt and found_load:
                break

        if loads_before_wait < 2 and len(loads) > 8:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="warning",
                category="pipeline",
                title="Shallow Prefetch Depth",
                description=f"Only {loads_before_wait} load(s) issued before first wait. "
                            f"Production kernels average 7.8 loads before first wait. "
                            "Deeper prefetch hides memory latency better.",
                suggestion="Issue multiple loads before waiting for any of them. "
                           "Use CK's PrefetchStages=2 or 3 pattern: issue loads for 2-3 "
                           "iterations ahead before consuming data from the first iteration.",
                metrics={"loads_before_wait": loads_before_wait, "total_loads": len(loads),
                         "production_avg": 7.8},
                reference="CK pipeline PrefetchStages parameter",
            ))

    def _check_mfma_vmem_interleaving(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check MFMA-to-VMEM interleaving ratio for optimal latency hiding."""
        mfma_count = kernel.mfma_count
        vmem_count = kernel.vmem_count

        if mfma_count < 4 or vmem_count < 1:
            return

        ratio = mfma_count / vmem_count

        # Check actual interleaving: look for VMEM between MFMA groups
        vmem_between_mfma = 0
        in_mfma_region = False
        for instr in kernel.instructions:
            if instr.is_mfma:
                in_mfma_region = True
            elif in_mfma_region and instr.is_memory_op and "load" in instr.mnemonic:
                vmem_between_mfma += 1
                in_mfma_region = False

        interleave_ratio = vmem_between_mfma / max(vmem_count, 1)

        if interleave_ratio < 0.3 and vmem_count > 4:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="warning",
                category="scheduling",
                title="Poor MFMA-VMEM Interleaving",
                description=f"MFMA:VMEM ratio is {ratio:.1f}:1 but only {interleave_ratio:.0%} of VMEM loads "
                            "are interleaved between MFMA groups. Production kernels interleave loads "
                            "with MFMA to hide global memory latency (~300 cycles).",
                suggestion="Schedule VMEM loads between MFMA chains. CK pipeline scheduler places "
                           "1 VMEM read per MFMA group: sched_group_barrier(0x008, 1) -> "
                           "sched_group_barrier(0x020, 1) to ensure interleaving.",
                metrics={"mfma_vmem_ratio": ratio, "interleave_ratio": interleave_ratio},
            ))

    def _check_accvgpr_usage(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check AccVGPR usage patterns for MFMA accumulation.

        AMD CDNA uses AccVGPRs (a-registers) for MFMA accumulation.
        Production kernels with MFMA should show AccVGPR usage.
        """
        agpr_refs = sum(1 for i in kernel.instructions
                       if re.search(r'\ba\d+\b|\ba\[\d+:\d+\]', i.raw_text))
        accvgpr_moves = sum(1 for i in kernel.instructions
                          if "accvgpr" in i.mnemonic.lower() or "v_accvgpr" in i.mnemonic)

        if kernel.mfma_count > 0 and kernel.register_usage.max_agpr == 0:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="compute",
                title="MFMA Without Detected AccVGPR Usage",
                description=f"Kernel has {kernel.mfma_count} MFMA instructions but no AccVGPR (a-register) "
                            "usage was detected. MFMA results accumulate in AccVGPRs.",
                suggestion="Ensure MFMA results are being properly accumulated in AccVGPRs. "
                           "Use v_accvgpr_read/write to transfer between VGPR and AccVGPR files.",
                metrics={"mfma_count": kernel.mfma_count},
            ))
        elif kernel.register_usage.max_agpr > 0 and kernel.mfma_count == 0:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="warning",
                category="resource",
                title="AccVGPR Allocated But No MFMA",
                description=f"AccVGPRs up to a{kernel.register_usage.max_agpr} are referenced "
                            "but no MFMA instructions found. AccVGPRs are primarily used for MFMA "
                            "accumulation.",
                suggestion="If MFMA is not needed, avoid AccVGPR usage to reduce register pressure. "
                           "AccVGPRs share the register file with VGPRs on CDNA.",
                metrics={"max_agpr": kernel.register_usage.max_agpr},
            ))

    def _check_setprio_scheduling(self, kernel: ParsedKernel, arch: str, result: AnalysisResult):
        """Check s_setprio usage for priority-based instruction scheduling.

        Production data shows only 18/2660 kernels use s_setprio,
        indicating it's an advanced optimization.
        """
        setprio_instrs = [i for i in kernel.instructions if "s_setprio" in i.mnemonic]

        if setprio_instrs:
            values = []
            for instr in setprio_instrs:
                try:
                    val = int(instr.raw_text.strip().split()[-1])
                    values.append(val)
                except (ValueError, IndexError):
                    pass

            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="scheduling",
                title=f"Advanced Priority Scheduling ({len(setprio_instrs)} s_setprio)",
                description=f"Kernel uses {len(setprio_instrs)} s_setprio instructions "
                            f"with values {sorted(set(values))}. Only 0.7% of production kernels "
                            "use this advanced scheduling technique.",
                suggestion="s_setprio is an advanced optimization. Values: 0=normal, 1=high, 2=higher, 3=highest. "
                           "Use higher priority around critical MFMA chains to reduce stalls.",
                line_numbers=[i.line_number for i in setprio_instrs[:5]],
                metrics={"setprio_count": len(setprio_instrs), "values": sorted(set(values))},
            ))

    def _check_gfx950_opportunities(self, kernel: ParsedKernel, arch: str, result: AnalysisResult):
        """Check for gfx950-specific optimization opportunities.

        gfx950 introduces SMFMA (sparse MFMA) and scaled MFMA (f8f6f4).
        """
        if arch != "gfx950":
            return

        # Check for scaled MFMA opportunity
        fp8_mfma = [i for i in kernel.instructions
                   if "mfma" in i.mnemonic and "fp8" in i.mnemonic
                   and "scale" not in i.mnemonic]
        scaled_mfma = [i for i in kernel.instructions
                      if "mfma_scale" in i.mnemonic or "f8f6f4" in i.mnemonic]

        if fp8_mfma and not scaled_mfma:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="compute",
                title="gfx950 Scaled MFMA Opportunity",
                description=f"Found {len(fp8_mfma)} FP8 MFMA but no scaled MFMA (v_mfma_scale_f32_16x16x128_f8f6f4). "
                            "gfx950 scaled MFMA combines scaling with FP8 multiply, potentially "
                            "eliminating separate dequantization steps.",
                suggestion="Use v_mfma_scale_f32_16x16x128_f8f6f4 for FP8 GEMM with block-scale "
                           "quantization. This can replace FP8 MFMA + separate scale multiply.",
                line_numbers=[i.line_number for i in fp8_mfma[:5]],
                metrics={"fp8_mfma_count": len(fp8_mfma)},
                reference="gfx950 SMFMA and scaled MFMA ISA extensions",
            ))

        # Check for SMFMA (sparse MFMA) opportunity
        regular_mfma = [i for i in kernel.instructions
                       if "v_mfma" in i.mnemonic and "smfma" not in i.mnemonic]
        smfma = [i for i in kernel.instructions if "smfma" in i.mnemonic]

        if regular_mfma and not smfma and len(regular_mfma) > 16:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="compute",
                title="gfx950 Sparse MFMA (SMFMA) Opportunity",
                description=f"Found {len(regular_mfma)} regular MFMA instructions. If the weight matrix "
                            "has structured sparsity (2:4 pattern), gfx950 SMFMA can provide "
                            "2x throughput.",
                suggestion="If model weights support 2:4 structured sparsity, use v_smfma_* "
                           "instructions for doubled compute throughput.",
                metrics={"mfma_count": len(regular_mfma)},
            ))

    def _check_ds_read_write_balance(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check balance between DS reads and writes for pipeline efficiency."""
        ds_reads = sum(1 for i in kernel.instructions
                      if i.is_lds_op and ("read" in i.mnemonic or "load" in i.mnemonic))
        ds_writes = sum(1 for i in kernel.instructions
                       if i.is_lds_op and ("write" in i.mnemonic or "store" in i.mnemonic))

        if ds_reads == 0 and ds_writes == 0:
            return

        total_lds = ds_reads + ds_writes

        if ds_writes > 0 and ds_reads == 0 and total_lds > 4:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="warning",
                category="memory",
                title="LDS Writes Without Reads",
                description=f"Found {ds_writes} LDS writes but no LDS reads. "
                            "Data written to LDS is never read back, wasting LDS bandwidth.",
                suggestion="Either add LDS reads to consume the written data, or eliminate "
                           "LDS writes if the data can be consumed directly from registers.",
                metrics={"ds_writes": ds_writes, "ds_reads": ds_reads},
            ))

        if ds_reads > ds_writes * 8 and ds_writes > 0:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="memory",
                title="High LDS Read-to-Write Ratio",
                description=f"LDS read/write ratio is {ds_reads}:{ds_writes} ({ds_reads/ds_writes:.0f}:1). "
                            "High read ratio indicates good data reuse from LDS.",
                suggestion="No action needed. High read-to-write ratio indicates efficient LDS data reuse.",
                metrics={"ds_reads": ds_reads, "ds_writes": ds_writes,
                         "ratio": ds_reads / max(ds_writes, 1)},
            ))

    def _check_partial_waitcnt_effectiveness(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check effectiveness of partial waitcnt usage.

        Production kernels average 57% partial waitcnt ratio.
        """
        partial_cnt = 0
        full_cnt = 0
        partial_lines = []

        for instr in kernel.instructions:
            if not instr.is_waitcnt:
                continue
            raw = instr.raw_text
            vmcnt_m = re.search(r'vmcnt\((\d+)\)', raw)
            lgkm_m = re.search(r'lgkmcnt\((\d+)\)', raw)

            is_partial = False
            if vmcnt_m and int(vmcnt_m.group(1)) > 0:
                is_partial = True
            if lgkm_m and int(lgkm_m.group(1)) > 0:
                is_partial = True

            if is_partial:
                partial_cnt += 1
                partial_lines.append(instr.line_number)
            else:
                full_cnt += 1

        total = partial_cnt + full_cnt
        if total < 3:
            return

        partial_ratio = partial_cnt / total
        # Production baseline: 57% partial
        if partial_ratio > 0.5:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="pipeline",
                title=f"Good Partial Waitcnt Usage ({partial_ratio:.0%})",
                description=f"Using partial waitcnts for {partial_cnt}/{total} waits ({partial_ratio:.0%}). "
                            "Production kernels average 57%. Partial waits allow memory operations "
                            "to remain in-flight, hiding latency.",
                suggestion="No action needed. Partial waitcnt usage is at or above production levels.",
                metrics={"partial_count": partial_cnt, "full_count": full_cnt,
                         "partial_ratio": partial_ratio, "production_avg": 0.57},
            ))

    def _check_nop_near_mfma(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check for NOP placement near MFMA (hazard avoidance patterns)."""
        nop_near_mfma = 0
        nop_not_near_mfma = 0

        for i, instr in enumerate(kernel.instructions):
            if "s_nop" not in instr.mnemonic:
                continue

            near_mfma = False
            for j in range(max(0, i - 3), min(len(kernel.instructions), i + 4)):
                if kernel.instructions[j].is_mfma:
                    near_mfma = True
                    break

            if near_mfma:
                nop_near_mfma += 1
            else:
                nop_not_near_mfma += 1

        if nop_not_near_mfma > 3 and kernel.nop_count > 5:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="pipeline",
                title=f"NOPs Not Adjacent to MFMA ({nop_not_near_mfma}/{kernel.nop_count})",
                description=f"{nop_not_near_mfma} of {kernel.nop_count} NOPs are not near MFMA instructions. "
                            "NOPs near MFMA serve as hardware hazard avoidance, but NOPs elsewhere "
                            "may be unnecessary compiler-inserted delays.",
                suggestion="NOPs away from MFMA may indicate the compiler couldn't find useful instructions "
                           "to fill slots. Consider restructuring the code to provide more independent "
                           "instructions for the compiler to schedule.",
                metrics={"nop_near_mfma": nop_near_mfma, "nop_not_near_mfma": nop_not_near_mfma},
            ))

    def _check_global_vs_buffer_load(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check global_load vs buffer_load usage strategy.

        buffer_load uses scalar descriptors (more efficient for structured access).
        global_load uses flat addressing (more flexible but potentially less efficient).
        """
        global_loads = sum(1 for i in kernel.instructions if "global_load" in i.mnemonic)
        buffer_loads = sum(1 for i in kernel.instructions if "buffer_load" in i.mnemonic)

        if global_loads == 0 and buffer_loads == 0:
            return

        total_loads = global_loads + buffer_loads

        if global_loads > 0 and buffer_loads == 0 and total_loads > 8:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="memory",
                title="Only global_load Used (Consider buffer_load)",
                description=f"All {global_loads} loads use global_load (flat addressing). "
                            "buffer_load with scalar descriptors can be more efficient for "
                            "structured memory access patterns, as it uses fewer scalar registers "
                            "for base address computation.",
                suggestion="For structured access patterns (e.g., strided, regular), consider "
                           "buffer_load with buffer resource descriptors. This can improve "
                           "address computation efficiency and enable hardware bounds checking.",
                metrics={"global_loads": global_loads, "buffer_loads": buffer_loads},
            ))

    def _check_instruction_density(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check instruction density (useful instructions vs overhead)."""
        if kernel.total_instructions < 20:
            return

        useful = (kernel.valu_count + kernel.mfma_count + kernel.vmem_count +
                  kernel.lds_count + kernel.smem_count)
        overhead = kernel.nop_count + kernel.waitcnt_count + kernel.barrier_count
        total = kernel.total_instructions

        useful_pct = 100 * useful / total
        overhead_pct = 100 * overhead / total

        if overhead_pct > 25:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="warning",
                category="pipeline",
                title=f"High Instruction Overhead ({overhead_pct:.0f}%)",
                description=f"Only {useful_pct:.0f}% of instructions are useful compute/memory "
                            f"({useful} useful, {overhead} overhead out of {total} total). "
                            "NOPs, waitcnts, and barriers comprise the rest.",
                suggestion="Reduce overhead by: (1) using partial waitcnts, (2) eliminating "
                           "unnecessary barriers, (3) restructuring code to reduce NOPs.",
                metrics={"useful_pct": useful_pct, "overhead_pct": overhead_pct,
                         "useful": useful, "overhead": overhead},
            ))

    def _check_agpr_partitioning(self, kernel: ParsedKernel, arch: str, result: AnalysisResult):
        """Check register file partitioning between VGPR and AccVGPR.

        On CDNA3, VGPRs and AccVGPRs share the physical register file.
        Total = max(vgpr, agpr) for occupancy calculation, not vgpr+agpr.
        """
        if arch not in ("gfx940", "gfx941", "gfx942", "gfx950"):
            return

        vgpr = kernel.register_usage.max_vgpr + 1
        agpr = kernel.register_usage.max_agpr + 1

        if vgpr == 1 or agpr == 1:
            return

        total_regs = max(vgpr, agpr)  # Correct for CDNA3 shared register file

        if vgpr > 200 and agpr > 200:
            effective_vgpr = max(vgpr, agpr)
            vgprs_per_simd = 512
            waves = min(8, vgprs_per_simd // effective_vgpr)

            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="warning",
                category="resource",
                title=f"Heavy VGPR + AccVGPR Usage (Effective: {effective_vgpr})",
                description=f"VGPR: {vgpr}, AccVGPR: {agpr}. On CDNA3, physical registers = max(VGPR, AccVGPR) = {effective_vgpr}. "
                            f"This allows {waves} wave(s) per SIMD.",
                suggestion="On CDNA3, VGPRs and AccVGPRs share the physical register file. "
                           "Reduce the larger of the two to improve occupancy. Consider "
                           "reducing tile size or using register-efficient accumulation.",
                metrics={"vgpr": vgpr, "agpr": agpr, "effective": effective_vgpr, "waves": waves},
            ))

    def _check_mfma_type_optimization(self, kernel: ParsedKernel, arch: str, result: AnalysisResult):
        """Check if optimal MFMA variant is used for the architecture.

        Different architectures support different MFMA variants:
        - gfx942: v_mfma_f32_16x16x32_fp8_fp8 (most common in production)
        - gfx950: v_mfma_f32_16x16x128_f8f6f4 (new wider K dimension)
        """
        mfma_types = {}
        for instr in kernel.instructions:
            if instr.is_mfma:
                mfma_types[instr.mnemonic] = mfma_types.get(instr.mnemonic, 0) + 1

        if not mfma_types:
            return

        # Check for old-style MFMA on newer architectures
        if arch in ("gfx942", "gfx950"):
            old_style = sum(v for k, v in mfma_types.items() if "4x4x4" in k)
            new_style = sum(v for k, v in mfma_types.items() if "16x16" in k or "32x32" in k)

            if old_style > 0 and new_style == 0:
                result.findings.append(Finding(
                    finding_id=self._next_id(),
                    severity="warning",
                    category="compute",
                    title=f"Legacy MFMA Tiles on {arch}",
                    description=f"Using {old_style} small-tile (4x4) MFMA instructions. "
                                f"Production kernels on {arch} primarily use 16x16x32 "
                                "(1M FP8 MFMA) or 16x16x16 (122K FP16/BF16 MFMA).",
                    suggestion=f"Upgrade to 16x16 or 32x32 MFMA tiles for significantly "
                               "higher throughput. Production baseline: v_mfma_f32_16x16x32_fp8_fp8.",
                    metrics={"mfma_types": mfma_types},
                ))

    def _check_bf16_vs_fp16_mfma(self, kernel: ParsedKernel, arch: str, result: AnalysisResult):
        """Check BF16 vs FP16 MFMA usage for optimal throughput."""
        bf16_mfma = sum(1 for i in kernel.instructions
                       if i.is_mfma and "bf16" in i.mnemonic)
        fp16_mfma = sum(1 for i in kernel.instructions
                       if i.is_mfma and "f16" in i.mnemonic and "bf16" not in i.mnemonic)

        if bf16_mfma > 0 and fp16_mfma > 0:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="precision",
                title="Mixed BF16 and FP16 MFMA",
                description=f"Kernel uses both BF16 ({bf16_mfma}) and FP16 ({fp16_mfma}) MFMA. "
                            "Mixing precision types may require format conversion instructions.",
                suggestion="Consider standardizing on one format. BF16 is preferred for training "
                           "(larger dynamic range), FP16 for inference (higher precision).",
                metrics={"bf16_mfma": bf16_mfma, "fp16_mfma": fp16_mfma},
            ))

    def _check_load_store_symmetry(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check load vs store ratio for kernel type classification."""
        total_loads = sum(1 for i in kernel.instructions
                        if i.is_memory_op and "load" in i.mnemonic)
        total_stores = sum(1 for i in kernel.instructions
                         if i.is_memory_op and "store" in i.mnemonic)

        if total_loads == 0 and total_stores == 0:
            return

        if total_stores > total_loads * 2 and total_stores > 8:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="memory",
                title="Store-Heavy Kernel",
                description=f"Store-to-load ratio is {total_stores}:{total_loads}. "
                            "This kernel writes significantly more than it reads, "
                            "which may indicate initialization or data generation patterns.",
                suggestion="For store-heavy kernels, ensure coalesced writes and consider "
                           "using async store operations where supported.",
                metrics={"loads": total_loads, "stores": total_stores},
            ))

    def _check_loop_structure(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check for loop structure quality based on basic blocks."""
        if len(kernel.basic_blocks) < 2:
            return

        # Count back-edge branches (loops)
        branch_instrs = [i for i in kernel.instructions
                        if i.mnemonic.startswith("s_cbranch") or i.mnemonic == "s_branch"]
        loop_branches = [i for i in branch_instrs
                        if "scc" in i.raw_text.lower() or "vcc" in i.raw_text.lower()]

        if kernel.total_instructions > 100 and len(branch_instrs) == 0:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="structure",
                title="Fully Unrolled Kernel (No Branches)",
                description=f"Kernel has {kernel.total_instructions} instructions with no branch instructions. "
                            "This indicates full loop unrolling, which increases code size but "
                            "eliminates branch overhead.",
                suggestion="Full unrolling is good for small loops but can cause I-cache pressure "
                           "for very large kernels. Production CK kernels use `static_for` unrolling.",
                metrics={"total_instructions": kernel.total_instructions, "branches": 0},
            ))

    def _check_wavefront_size_alignment(self, kernel: ParsedKernel, arch: str, result: AnalysisResult):
        """Check if work items align with wavefront size (64 on CDNA)."""
        if arch not in ("gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"):
            return

        # Check for potential divergent branches
        exec_mask_ops = sum(1 for i in kernel.instructions
                          if "exec" in i.raw_text.lower() and
                          ("s_mov" in i.mnemonic or "s_and" in i.mnemonic or "s_or" in i.mnemonic))

        if exec_mask_ops > 10:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="warning",
                category="divergence",
                title=f"Frequent EXEC Mask Changes ({exec_mask_ops})",
                description=f"Found {exec_mask_ops} EXEC mask manipulations, indicating significant "
                            "thread divergence. On AMD GPUs, all 64 lanes in a wavefront must "
                            "execute in lockstep; inactive lanes waste compute resources.",
                suggestion="Minimize divergent branches within wavefronts. Restructure data to "
                           "ensure threads in the same wavefront follow the same path.",
                metrics={"exec_mask_ops": exec_mask_ops},
            ))

    def _check_flat_vs_global_addressing(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check for flat memory access which may be slower than global/buffer."""
        flat_ops = sum(1 for i in kernel.instructions if "flat_" in i.mnemonic)
        global_ops = sum(1 for i in kernel.instructions
                        if "global_" in i.mnemonic or "buffer_" in i.mnemonic)

        if flat_ops > 4 and global_ops < flat_ops:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="warning",
                category="memory",
                title=f"Flat Memory Access ({flat_ops} ops)",
                description=f"Found {flat_ops} flat memory operations vs {global_ops} global/buffer ops. "
                            "Flat addressing must resolve the address space at runtime, which adds "
                            "latency compared to explicitly global or LDS addressing.",
                suggestion="Replace flat_load/store with global_load/store when the address space is "
                           "known to be global. Use ds_read/write for LDS. This avoids the runtime "
                           "address space resolution overhead.",
                metrics={"flat_ops": flat_ops, "global_ops": global_ops},
            ))

    def _check_mfma_utilization_ratio(self, kernel: ParsedKernel, result: AnalysisResult):
        """Check MFMA utilization as a fraction of total compute instructions."""
        if kernel.mfma_count == 0:
            return

        total_compute = kernel.valu_count + kernel.mfma_count
        if total_compute < 10:
            return

        mfma_ratio = kernel.mfma_count / total_compute

        if mfma_ratio < 0.3 and kernel.mfma_count > 4:
            result.findings.append(Finding(
                finding_id=self._next_id(),
                severity="info",
                category="compute",
                title=f"Low MFMA Utilization ({mfma_ratio:.0%} of compute)",
                description=f"MFMA instructions are {mfma_ratio:.0%} of total compute ({kernel.mfma_count} MFMA, "
                            f"{kernel.valu_count} VALU). In GEMM-type kernels, MFMA should dominate compute.",
                suggestion="High VALU count alongside MFMA may indicate address computation or "
                           "data preparation overhead. Consider pre-computing indices or using "
                           "more efficient index arithmetic.",
                metrics={"mfma_ratio": mfma_ratio, "mfma_count": kernel.mfma_count,
                         "valu_count": kernel.valu_count},
            ))

    def _compute_summary(self, kernel: ParsedKernel, arch: str, result: AnalysisResult):
        """Compute summary statistics for the analysis."""
        vgpr_count = kernel.metadata.vgpr_count or (kernel.register_usage.max_vgpr + 1)

        # Estimate occupancy
        if arch in ("gfx940", "gfx941", "gfx942"):
            waves = min(8, 512 // max(vgpr_count, 1)) if vgpr_count > 0 else 8
        else:
            waves = min(10, 256 // max(vgpr_count, 1)) if vgpr_count > 0 else 10

        result.summary = {
            "total_instructions": kernel.total_instructions,
            "valu_instructions": kernel.valu_count,
            "salu_instructions": kernel.salu_count,
            "vmem_instructions": kernel.vmem_count,
            "smem_instructions": kernel.smem_count,
            "lds_instructions": kernel.lds_count,
            "mfma_instructions": kernel.mfma_count,
            "branch_instructions": kernel.branch_count,
            "waitcnt_instructions": kernel.waitcnt_count,
            "barrier_instructions": kernel.barrier_count,
            "nop_instructions": kernel.nop_count,
            "vgpr_count": vgpr_count,
            "sgpr_count": kernel.metadata.sgpr_count,
            "lds_size_bytes": kernel.metadata.lds_size,
            "scratch_size_bytes": kernel.metadata.scratch_size,
            "estimated_occupancy_waves": waves,
            "basic_blocks": len(kernel.basic_blocks),
            "findings_critical": result.critical_count,
            "findings_warning": result.warning_count,
            "findings_info": result.info_count,
        }
