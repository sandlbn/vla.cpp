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
 * @file gemm.h
 * @brief The dense GEMMs BitVLA's drivers need, stated row-major.
 *
 * The ternary projections are BitVLA's own kernels, but attention and the
 * fp32 action head are ordinary dense matmuls, and on CUDA they are cuBLAS
 * calls. Those call sites are the last vendor-specific thing in the three
 * forward drivers; behind this header the drivers become plain C++ that either
 * backend compiles, which is the difference between porting BitVLA once and
 * maintaining two copies of its attention loop.
 *
 * Everything here is **row-major**, because everything in vla.cpp is row-major
 * and the column-major spelling is where the bugs live. Translating the four
 * existing cuBLAS calls gave exactly three shapes, all with contiguous
 * operands:
 *
 *   * @ref vla_gemm_bf16_nt - `C[M][N] = A[M][K] * B[N][K]^T`. A linear layer
 *     against a row-major (N, K) weight matrix. Patch embedding, the multimodal
 *     projector, and (in f32) every layer of the action head.
 *   * @ref vla_gemm_bf16_nt_batched - the same, per head: Q * K^T.
 *   * @ref vla_gemm_bf16_nn_batched - `C[M][N] = A[M][K] * B[K][N]`, per head:
 *     softmax(scores) * V.
 *
 * ## Accumulation
 *
 * bf16 in, bf16 out, **f32 accumulate** - matching `CUBLAS_COMPUTE_32F` in the
 * calls these replace. That is not a tunable: a bf16 accumulator over a
 * 256-term reduction loses most of the answer, and the tolerance the ported
 * ops are held to assumes the f32 sum.
 *
 * Unlike the ternary GEMM these are *not* bit-exact across backends. Both sides
 * sum f32 products in an order the library picks, so the results differ in the
 * last place or two, and the gate is the tolerance in docs/backend/sycl.md
 * rather than equality. The ternary path is where exactness is claimed and
 * checked; see tests/test_bitvla_gemm_gpu.cpp.
 *
 * All functions return 0 on success and non-zero on failure, having already
 * written a diagnostic to stderr.
 */

#pragma once

#include "kernels/bitvla/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief `C[M][N] = A[M][K] * B[N][K]^T`, f32 accumulate.
 * @param A Row-major (M, K).
 * @param B Row-major (N, K) - the weight matrix in its natural layout.
 * @param C Row-major (M, N).
 */
int vla_gemm_bf16_nt(const vla_bf16 * A, const vla_bf16 * B, vla_bf16 * C,
                     int M, int N, int K, vla_stream stream);

/**
 * @brief `C[b][M][N] = A[b][M][K] * B[b][N][K]^T` for b in [0, batch).
 *
 * The Q*K^T of one attention layer: @p A is Q for one head, @p B is K for the
 * same head, both (seq, head_dim), and the strides walk the head axis.
 */
int vla_gemm_bf16_nt_batched(const vla_bf16 * A, const vla_bf16 * B, vla_bf16 * C,
                             int M, int N, int K, int batch,
                             long long stride_a, long long stride_b, long long stride_c,
                             vla_stream stream);

/**
 * @brief `C[b][M][N] = A[b][M][K] * B[b][K][N]` for b in [0, batch).
 *
 * The scores*V of one attention layer.
 */
int vla_gemm_bf16_nn_batched(const vla_bf16 * A, const vla_bf16 * B, vla_bf16 * C,
                             int M, int N, int K, int batch,
                             long long stride_a, long long stride_b, long long stride_c,
                             vla_stream stream);

/**
 * @brief `C[M][N] = A[M][K] * B[N][K]^T` in f32.
 *
 * The action head keeps f32 end to end - it is small, it is the last thing
 * before a robot command, and BitVLA's reference implementation does the same.
 */
int vla_gemm_f32_nt(const float * A, const float * B, float * C,
                    int M, int N, int K, vla_stream stream);

#ifdef __cplusplus
}
#endif
