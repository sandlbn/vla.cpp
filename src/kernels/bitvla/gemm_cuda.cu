// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file gemm_cuda.cu
 * @brief gemm.h on CUDA, via cuBLAS.
 *
 * The same calls the three drivers used to make inline, moved behind gemm.h so
 * the drivers themselves stop being CUDA sources. Nothing about the GEMMs
 * changed: same handle, same @c CUBLAS_COMPUTE_32F, same
 * @c CUBLAS_GEMM_DEFAULT, so the CUDA numbers before and after this refactor
 * are bit-identical - which is the point, and what ci/slurm/cuda_reference
 * re-checks.
 *
 * ## Row-major to column-major
 *
 * cuBLAS is column-major and everything here is row-major. Rather than
 * transpose anything, use the standard identity: a row-major `C = A*B` is the
 * column-major `C^T = B^T * A^T`, and a row-major (r, c) buffer already *is*
 * the column-major (c, r) buffer. So every call below passes B where cuBLAS
 * wants A, A where it wants B, and N where it wants m - and the leading
 * dimensions come out as the row lengths that were there all along.
 *
 * Worked through for each shape at its call site, because this is exactly the
 * kind of index algebra that is easier to re-derive than to trust.
 */

#include "kernels/bitvla/cuda_compat.h"
#include "kernels/bitvla/gemm.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>

namespace {

/**
 * @brief One process-wide cuBLAS handle.
 *
 * The three drivers used to keep one each, which meant three copies of cuBLAS's
 * workspace for no reason - they never run concurrently, and the stream is set
 * per call anyway.
 */
cublasHandle_t handle() {
    static cublasHandle_t h = [] {
        cublasHandle_t tmp = nullptr;
        if (cublasCreate(&tmp) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla(bitvla_gemm_cuda): cublasCreate failed\n");
            return (cublasHandle_t) nullptr;
        }
        return tmp;
    }();
    return h;
}

int check(const char * which, cublasStatus_t st) {
    if (st == CUBLAS_STATUS_SUCCESS) return 0;
    std::fprintf(stderr, "vla(bitvla_gemm_cuda): %s failed (%d)\n", which, (int) st);
    return -1;
}

}  // namespace

extern "C" int vla_gemm_bf16_nt(const vla_bf16 * A, const vla_bf16 * B, vla_bf16 * C,
                                int M, int N, int K, vla_stream stream_) {
    cublasHandle_t h = handle();
    if (!h) return -1;
    cudaStream_t stream = vla_cu_stream(stream_);
    cublasSetStream(h, stream);

    const float alpha = 1.0f, beta = 0.0f;
    // Row-major C[M][N] = A[M][K] * B[N][K]^T.
    // Column-major: C^T[N][M] = (B[N][K])[K][N]^T ... i.e. op(B)=T, op(A)=N,
    // with m=N, n=M, k=K and both leading dimensions equal to K.
    return check("bf16_nt", cublasGemmEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        vla_cu_bf(B), CUDA_R_16BF, K,
        vla_cu_bf(A), CUDA_R_16BF, K,
        &beta,
        vla_cu_bf(C), CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

extern "C" int vla_gemm_bf16_nt_batched(const vla_bf16 * A, const vla_bf16 * B, vla_bf16 * C,
                                        int M, int N, int K, int batch,
                                        long long stride_a, long long stride_b,
                                        long long stride_c, vla_stream stream_) {
    cublasHandle_t h = handle();
    if (!h) return -1;
    cublasSetStream(h, vla_cu_stream(stream_));

    const float alpha = 1.0f, beta = 0.0f;
    // As above per batch element; A and B swap, so their strides swap with them.
    return check("bf16_nt_batched", cublasGemmStridedBatchedEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        vla_cu_bf(B), CUDA_R_16BF, K, stride_b,
        vla_cu_bf(A), CUDA_R_16BF, K, stride_a,
        &beta,
        vla_cu_bf(C), CUDA_R_16BF, N, stride_c,
        batch,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

extern "C" int vla_gemm_bf16_nn_batched(const vla_bf16 * A, const vla_bf16 * B, vla_bf16 * C,
                                        int M, int N, int K, int batch,
                                        long long stride_a, long long stride_b,
                                        long long stride_c, vla_stream stream_) {
    cublasHandle_t h = handle();
    if (!h) return -1;
    cublasSetStream(h, vla_cu_stream(stream_));

    const float alpha = 1.0f, beta = 0.0f;
    // Row-major C[M][N] = A[M][K] * B[K][N]: neither operand is transposed, and
    // B's leading dimension is its row length N rather than K.
    return check("bf16_nn_batched", cublasGemmStridedBatchedEx(
        h, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        vla_cu_bf(B), CUDA_R_16BF, N, stride_b,
        vla_cu_bf(A), CUDA_R_16BF, K, stride_a,
        &beta,
        vla_cu_bf(C), CUDA_R_16BF, N, stride_c,
        batch,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

extern "C" int vla_gemm_f32_nt(const float * A, const float * B, float * C,
                               int M, int N, int K, vla_stream stream_) {
    cublasHandle_t h = handle();
    if (!h) return -1;
    cublasSetStream(h, vla_cu_stream(stream_));

    const float alpha = 1.0f, beta = 0.0f;
    return check("f32_nt", cublasSgemm(
        h, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, K,
        A, K,
        &beta,
        C, N));
}
