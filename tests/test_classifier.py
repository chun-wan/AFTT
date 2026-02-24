import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.algorithm_classifier import AlgorithmClassifier, AlgorithmInfo, ASMFeatureFingerprint, AlgorithmType
from src.instruction import Instruction


@pytest.fixture
def classifier():
    return AlgorithmClassifier()


def test_classify_gemm_hip(classifier):
    source = '''
    __global__ void gemm_kernel(const hip_bfloat16* A, const hip_bfloat16* B, float* C, int M, int N, int K) {
        __shared__ hip_bfloat16 tile_A[32][32];
        __shared__ hip_bfloat16 tile_B[32][32];
        for (int k = 0; k < K; k += 32) {
            // Load tiles
            tile_A[ty][tx] = A[row * K + k + tx];
            tile_B[ty][tx] = B[(k + ty) * N + col];
            __syncthreads();
            for (int kk = 0; kk < 32; kk++) {
                sum += tile_A[ty][kk] * tile_B[kk][tx];
            }
            __syncthreads();
        }
        C[row * N + col] = sum;
    }
    '''
    info = classifier.classify_from_hip(source)
    assert info.algo_type == AlgorithmType.GEMM
    assert info.confidence > 0.5


def test_classify_rmsnorm_hip(classifier):
    source = '''
    __global__ void rmsnorm_kernel(float* output, const float* input, const float* weight,
                                    float epsilon, int hidden_size) {
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            sum_sq += input[row * hidden_size + i] * input[row * hidden_size + i];
        }
        // Warp reduction
        sum_sq = warp_reduce_sum(sum_sq);
        float rms = rsqrtf(sum_sq / hidden_size + epsilon);
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            output[row * hidden_size + i] = input[row * hidden_size + i] * rms * weight[i];
        }
    }
    '''
    info = classifier.classify_from_hip(source)
    assert info.algo_type == AlgorithmType.RMSNORM
    assert info.confidence > 0.5


def test_classify_elementwise_hip(classifier):
    source = '''
    __global__ void vector_add(const float* A, const float* B, float* C, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            C[idx] = A[idx] + B[idx];
        }
    }
    '''
    info = classifier.classify_from_hip(source)
    assert info.algo_type == AlgorithmType.ELEMENTWISE


def test_classify_from_asm_mfma(classifier):
    instrs = [
        Instruction.from_disassembly(0, b"\x00" * 4, "v_mfma_f32_16x16x16_bf16", "a[0:3], v[0:1], v[2:3], a[0:3]"),
        Instruction.from_disassembly(4, b"\x00" * 4, "v_mfma_f32_16x16x16_bf16", "a[4:7], v[0:1], v[2:3], a[4:7]"),
        Instruction.from_disassembly(8, b"\x00" * 4, "v_mfma_f32_16x16x16_bf16", "a[8:11], v[0:1], v[2:3], a[8:11]"),
        Instruction.from_disassembly(12, b"\x00" * 4, "v_mfma_f32_16x16x16_bf16", "a[12:15], v[0:1], v[2:3], a[12:15]"),
        Instruction.from_disassembly(16, b"\x00" * 4, "buffer_load_dword", "v0, v1, s[0:3], 0"),
        Instruction.from_disassembly(20, b"\x00" * 4, "ds_read_b32", "v0, v1"),
        Instruction.from_disassembly(24, b"\x00" * 4, "s_waitcnt", "vmcnt(0) lgkmcnt(0)"),
    ]
    info = classifier.classify_from_asm(instrs)
    assert info.algo_type == AlgorithmType.GEMM
    assert "bf16" in info.sub_type


def test_build_fingerprint(classifier):
    instrs = [
        Instruction.from_disassembly(0, b"\x00" * 4, "v_add_f32", "v0, v1, v2"),
        Instruction.from_disassembly(4, b"\x00" * 4, "s_nop", "0"),
        Instruction.from_disassembly(8, b"\x00" * 4, "ds_read_b32", "v0, v1"),
    ]
    fp = classifier.build_fingerprint(instrs)
    assert fp.total_instructions == 3
    assert fp.valu_count == 1
    assert fp.lds_reads == 1
    assert fp.nop_count == 1


def test_extract_parameters(classifier):
    source = "void gemm(float* A, float* B, float* C, int M, int N, int K) {}"
    params = classifier.extract_parameters(source, "GEMM")
    assert isinstance(params, dict)
