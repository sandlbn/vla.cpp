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
 * @file bitvla_fused.h
 * @brief Fused row kernels for the BitVLA towers - one launch where the
 *        unfused drivers issue two to four.
 *
 * Written for integrated GPUs (Panther Lake / Xe3), where a profile of the
 * unfused LM showed every elementwise and row kernel already running at the
 * LPDDR5x ceiling: what costs time there is not the arithmetic but writing an
 * intermediate bf16 plane out to DRAM and reading it straight back. Each entry
 * point below computes a chain the drivers used to issue as separate kernels,
 * keeping the intermediates on chip.
 *
 * ## Bit-identity with the unfused chain
 *
 * Every intermediate the unfused chain rounded to bf16 is rounded to bf16 here
 * too, at the same point, and every reduction keeps the unfused partition
 * (@c ROW_WG work-items, @c k = tid; @c k += ROW_WG) and the same
 * @c reduce_over_group. The outputs are therefore the same bits as the chain
 * they replace - @c tests/test_bitvla_ops_gpu.cpp holds each one to exactly
 * that, not to a tolerance.
 *
 * Only built where a backend defines them (SYCL today); a driver compiled
 * without @c VLA_BITVLA_FUSED_OPS keeps issuing the unfused chain.
 */

#pragma once

#include "kernels/bitvla/device.h"

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Optional residual add, RMSNorm, then int8 absmax quantisation.
 *
 * Replaces @c bitvla_add_bf16 (when @p delta is non-null) + @c bitvla_rmsnorm_bf16
 * + @c bitvla_act_quant_cuda. With @p delta, @p h is updated in place to
 * @c bf16(h + delta), exactly as the residual add wrote it.
 *
 * @param h      [M, K] bf16 residual stream (read, and written when @p delta)
 * @param delta  [M, K] bf16 to add first, or null
 * @param w      [K] RMSNorm weight
 * @param out    [M, K] int8 quantised normalised rows
 * @param scales [M] per-row quantisation scale (127/absmax)
 */
void bitvla_add_rmsnorm_quant_bf16(vla_bf16* h, const vla_bf16* delta,
                                   const vla_bf16* w, int8_t* out, float* scales,
                                   float eps, int M, int K, vla_stream stream);

/**
 * @brief Squared-ReLU gate times up, RMSNorm, then int8 quantisation.
 *
 * Replaces @c gate_up_fused_sqrelu_mul_bf16 + the in-place sub-norm
 * @c bitvla_rmsnorm_bf16 + @c bitvla_act_quant_cuda on the LM's FFN.
 *
 * @param gu  [seq, 2*ffn] fused gate|up GEMM output
 */
void bitvla_sqrelu_rmsnorm_quant_bf16(const vla_bf16* gu, const vla_bf16* w,
                                      int8_t* out, float* scales, float eps,
                                      int seq, int ffn, vla_stream stream);

/**
 * @brief Optional biased residual add, LayerNorm, then int8 quantisation.
 *
 * Replaces @c bitvla_add_bias_bf16 (on @p delta) + @c bitvla_add_bf16 +
 * @c bitvla_layernorm_bf16 + @c bitvla_act_quant_cuda on the ViT. When @p delta
 * is non-null, @p h becomes @c bf16(h + bf16(delta + bias)).
 */
void bitvla_add_layernorm_quant_bf16(vla_bf16* h, const vla_bf16* delta,
                                     const vla_bf16* delta_bias,
                                     const vla_bf16* w, const vla_bf16* b,
                                     int8_t* out, float* scales, float eps,
                                     int M, int K, vla_stream stream);

/**
 * @brief Bias, GELU-tanh, then int8 quantisation into a zero-padded stride.
 *
 * Replaces @c bitvla_add_bias_bf16 + @c bitvla_gelu_tanh_bf16 +
 * @c bitvla_act_quant_pad_cuda on the ViT's fc1 output.
 */
void bitvla_bias_gelu_quant_pad_bf16(const vla_bf16* x, const vla_bf16* bias,
                                     int8_t* out, float* scales,
                                     int M, int K_in, int K_out, vla_stream stream);

/**
 * @brief @c h = bf16(h + bf16(delta + bias)) - the ViT's last residual, which
 *        has no following LayerNorm to fuse into.
 */
void bitvla_bias_residual_bf16(vla_bf16* h, const vla_bf16* delta,
                               const vla_bf16* bias, int M, int K,
                               vla_stream stream);

/**
 * @brief NeoX RoPE applied in place to the Q and K projection planes, in their
 *        row-major [S, H*hd] layout - one launch for both.
 *
 * Replaces the head-major transposes and the two @c bitvla_rope_neox_bf16
 * calls ahead of the fused attention. The per-element arithmetic is
 * @c bitvla_rope_neox_bf16's, so each rotated value is the same bits; only
 * where it lives changes.
 */
void bitvla_rope_neox_qk_rows_bf16(vla_bf16* q, vla_bf16* k, const float* cos_tab,
                                   const float* sin_tab, int S, int n_q, int n_kv,
                                   int hd, int q_ld, int k_ld, vla_stream stream);

/**
 * @brief Three ternary projections of the same activations as one GEMM.
 *
 * out = [A W0^T | A W1^T | A W2^T] with the bitlinear epilogue (divide by the
 * row's activation scale, multiply by the column's weight scale) and, when
 * @p bias[0] is non-null, + bias - written as one [M, N0+N1+N2] row-major
 * plane. The weights are unpacked and concatenated once, on first use.
 *
 * Without bias the three column blocks are the same bits as three
 * @c bitlinear_int8xint2_m calls: int32 accumulation is exact and each column
 * sees the same epilogue. With bias the add happens before the single bf16
 * rounding instead of after a first one, so it is one rounding more accurate
 * than @c bitlinear_int8xint2_m followed by @c bitvla_add_bias_bf16.
 *
 * Replaces the LM's q/k/v projections (where k and v alone are 640 wide and
 * run at half the XMX rate of the wide GEMMs) and the ViT's q/k/v + biases.
 */
void bitvla_ternary_gemm_cat3(const int8_t* A, const float* s, int M, int K,
                              int8_t* const B[3], float* const ws[3],
                              const vla_bf16* const bias[3], const int N[3],
                              vla_bf16* out, vla_stream stream);

#ifdef __cplusplus
}
#endif
