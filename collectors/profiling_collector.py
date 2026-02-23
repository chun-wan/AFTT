#!/usr/bin/env python3
"""Profiling Rule Collector.

Utility to expand profiling rules with additional detail and validate them
against the rocprof-compute counter database.
"""

import json
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "db" / "profiling_rules"

ARCH_SPECS = {
    "gfx942": {
        "name": "MI300X / MI325X",
        "generation": "CDNA3",
        "cus": 304,
        "simds_per_cu": 4,
        "max_waves_per_simd": 8,
        "vgprs_per_simd": 512,
        "sgprs_per_simd": 800,
        "lds_per_cu_bytes": 65536,
        "l2_cache_mb": 256,
        "hbm_bandwidth_gbps": 5300,
        "peak_fp16_tflops": 1307,
        "peak_fp8_tflops": 2614,
        "peak_fp32_tflops": 163,
        "peak_fp64_tflops": 81,
        "wavefront_size": 64,
    },
    "gfx90a": {
        "name": "MI200 (MI210/MI250/MI250X)",
        "generation": "CDNA2",
        "cus": 220,
        "simds_per_cu": 4,
        "max_waves_per_simd": 8,
        "vgprs_per_simd": 512,
        "sgprs_per_simd": 800,
        "lds_per_cu_bytes": 65536,
        "l2_cache_mb": 16,
        "hbm_bandwidth_gbps": 3200,
        "peak_fp16_tflops": 383,
        "peak_fp32_tflops": 47.9,
        "peak_fp64_tflops": 47.9,
        "wavefront_size": 64,
    },
    "gfx908": {
        "name": "MI100",
        "generation": "CDNA",
        "cus": 120,
        "simds_per_cu": 4,
        "max_waves_per_simd": 8,
        "vgprs_per_simd": 256,
        "sgprs_per_simd": 800,
        "lds_per_cu_bytes": 65536,
        "l2_cache_mb": 8,
        "hbm_bandwidth_gbps": 1200,
        "peak_fp16_tflops": 184.6,
        "peak_fp32_tflops": 23.1,
        "peak_fp64_tflops": 11.5,
        "wavefront_size": 64,
    },
}


def main():
    out_file = DB_DIR / "arch_specs.json"
    with open(out_file, "w") as f:
        json.dump(ARCH_SPECS, f, indent=2)
    print(f"Wrote arch specs to {out_file}")

    for arch, spec in ARCH_SPECS.items():
        flops_per_byte = spec.get("peak_fp16_tflops", 0) * 1000 / max(spec["hbm_bandwidth_gbps"], 1)
        print(f"  {arch} ({spec['name']}): {spec['cus']} CUs, "
              f"FP16={spec.get('peak_fp16_tflops', 'N/A')} TFLOPS, "
              f"HBM={spec['hbm_bandwidth_gbps']} GB/s, "
              f"arithmetic intensity threshold={flops_per_byte:.1f} FLOP/byte")


if __name__ == "__main__":
    main()
