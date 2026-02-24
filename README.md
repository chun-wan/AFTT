# AFTT - ASM Fine-Tuning Tool

A static analysis and **binary-level optimization** tool for AMD GPU (AMDGPU) assembly kernels. Compiles C++/HIP code to ASM, analyzes it against a knowledge base of production-grade optimization patterns extracted from aiter/CK kernels, and applies automated ASM-level edits to improve kernel performance on AMD Instinct GPUs.

## Features

- **Automated ASM Optimization**: 8 binary-patchable optimization strategies that directly modify `.co` kernel objects
  - NOP elimination, precise waitcnt relaxation (register-level dependency tracking)
  - Redundant barrier removal, waitcnt splitting
  - LDS-to-DPP conversion, vectorized load merging
  - MFMA-VMEM interleaving for latency hiding
- **ASM Pattern Analysis**: Deep pattern matching against production FMHA, GEMM, and normalization kernels
- **DPP/Cross-lane Optimization**: Detects suboptimal reductions and suggests DPP-based replacements
- **Pipeline-aware Cycle Estimation**: CDNA3/CDNA4 multi-unit pipeline modeling (MFMA, VALU, VMEM, LDS, SALU)
- **Register Pressure Estimation**: VGPR/AGPR/SGPR occupancy impact analysis
- **ISA Instruction Database**: Multi-generation MI GPU instruction reference (gfx900-gfx950)
- **Compiler Flag Comparison**: Diff ASM output across optimization levels
- **Algorithm Classification**: Auto-classifies kernels (GEMM, FMHA, RMSNorm, Softmax, etc.)
- **C++ Template Engine**: HIP kernel templates with parameter substitution for pair generation

## Supported Architectures

| Architecture | GPU | Notes |
|---|---|---|
| gfx900 | MI25 | Vega 10 |
| gfx906 | MI50/MI60 | Vega 20 |
| gfx908 | MI100 | CDNA |
| gfx90a | MI200 | CDNA2 |
| gfx940 | MI300A | CDNA3 |
| gfx941 | MI300A | CDNA3 variant |
| gfx942 | MI300X/MI325X | CDNA3 |
| gfx950 | MI350 | CDNA4 |

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Analyze a HIP kernel file
aftt analyze kernel.cpp --arch gfx942

# Optimize a compiled .co kernel object (binary patching)
aftt optimize kernel.co --arch gfx942

# Full pipeline: analyze → classify → template match → ASM transform → optimize
aftt transform kernel.cpp --arch gfx942

# Compare compiler flag effects
aftt compile-compare kernel.cpp --arch gfx942 --flags "-O2,-O3,-Ofast"

# Look up an ISA instruction
aftt isa v_mfma_f32_32x32x8f16 --arch gfx942

# Get optimization suggestions for existing ASM
aftt suggest kernel.s --arch gfx942

# Show knowledge base statistics
aftt stats
```

## Project Structure

```
AFTT/
  db/                        # Knowledge base
    isa/                     # ISA instruction definitions per arch
    patterns/                # Optimization patterns (DPP, FMHA, etc.)
    compiler_flags/          # Compiler flag effect mappings
    cpp_asm_pairs/           # C++ <-> ASM paired corpus
    profiling_rules/         # Performance anti-pattern rules
    training_data/           # Exported datasets for model training
    disassembly/             # Disassembled .co kernels (not in git)
  src/                       # Core library
    compiler.py              # Wrapper for amdclang++ compilation
    parser.py                # ASM output parser
    instruction.py           # Unified AMDGPU instruction model
    analyzer.py              # Pattern matching engine
    asm_optimizer.py         # 8-strategy ASM optimization engine
    asm_editor.py            # ASM binary editor (llvm-mc + ld.lld)
    pattern_replacer.py      # Multi-level ASM pattern replacement
    cycle_estimator.py       # Pipeline-aware cycle estimation
    algorithm_classifier.py  # Kernel algorithm classification
    template_matcher.py      # Production kernel similarity matching
    cpp_template_engine.py   # HIP C++ template management
    pipeline.py              # Full v2 optimization pipeline
    kernel_validator.py      # HIP runtime validation
    reporter.py              # Report generator
    isa_db.py                # ISA instruction lookup
    knowledge_base.py        # Unified KB access layer
  collectors/                # Data collection scripts
    isa_collector.py         # Parse AMD ISA docs per gfx arch
    isa_deep_collector.py    # Deep ISA instruction extraction
    asm_pair_generator.py    # Generate C++ -> ASM pairs via compiler
    pattern_extractor.py     # Extract patterns from aiter/CK code
    ck_deep_analyzer.py      # Deep CK pipeline pattern analysis
    profiling_collector.py   # Build profiling rule set
    compiler_flag_tester.py  # Test different compiler flags
    trtllm_analyzer.py       # TensorRT-LLM kernel algorithm mapping
    trtllm_mapping.py        # TRT-LLM to AMD pattern bridge
    dataset_exporter.py      # Export training data for GLM
    co_disassembler.py       # .co file disassembly
    co_analyzer.py           # Code object metadata analysis
  templates/                 # HIP kernel templates for pair generation
  tests/                     # Unit tests
  reports/                   # Generated analysis reports
  cli.py                     # CLI entry point
  e2e_optimize.py            # End-to-end kernel optimization pipeline
```

## Optimization Strategies

| # | Strategy | Scope | Safety |
|---|----------|-------|--------|
| 1 | NOP elimination | All kernels | Safe |
| 2 | Precise waitcnt relaxation | GEMM/FMHA | Register-dep aware |
| 3 | Redundant barrier removal | FMHA | Barrier-aware |
| 4 | waitcnt splitting | All kernels | Safe |
| 5 | LDS→DPP conversion | Reduction kernels | Validated patterns |
| 6 | Vectorized load merging | GEMM/FMHA | VGPR-consecutive check |
| 7 | MFMA-VMEM interleaving | Compute-bound | Dependency verified |
| 8 | Pattern dedup (replacer) | All kernels | Conflict resolution |

## Requirements

- ROCm 6.0+ with `amdclang++` and `llvm-mc`
- Python 3.10+
- AMD Instinct GPU (MI100/MI200/MI300/MI325X) for runtime validation

## License

MIT
