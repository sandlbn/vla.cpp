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
 * @file attention.h
 * @brief Fused, unmasked scaled-dot-product attention over row-major
 *        projection planes - no head transposes, no repeated KV, no scores
 *        plane in memory.
 *
 * The unfused towers split each projection into head-major planes, repeat the
 * KV heads per query head, write an [H, S, S] bf16 score plane, softmax it in
 * place and read it back for the second GEMM, then transpose the result back.
 * On a bandwidth-starved iGPU every one of those passes is a DRAM round trip;
 * this reads Q, K and V where the projections left them and writes the merged
 * [S, H*hd] output the o-projection's quantiser wants.
 *
 * Numerics differ from the unfused chain, and by design: scores and softmax
 * stay in f32 inside the kernel instead of being rounded to bf16 between
 * steps. It is closer to the exact result, not further, but it is not the same
 * bits - which is why it is a separate entry point and @c VLA_BITVLA_UNFUSED=1
 * restores the old chain.
 *
 * SYCL only (oneDNN Graph's SDPA micro-kernel), behind @c VLA_BITVLA_FUSED_OPS.
 */

#pragma once

#include "kernels/bitvla/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief out = softmax(Q K^T * scale) V, per head, grouped-query.
 *
 * Head @c h of Q occupies columns [h*hd, (h+1)*hd) of a [S, q_ld] row-major
 * plane, and query head h reads KV head h / (n_q/n_kv). Same convention for
 * K, V (leading dimension @p kv_ld) and the output (leading dimension
 * @p out_ld).
 *
 * @return 0 on success; non-zero if the fused kernel is unavailable for this
 *         shape, in which case nothing was written and the caller must fall
 *         back.
 */
int vla_attention_bf16(const vla_bf16* q, const vla_bf16* k, const vla_bf16* v,
                       vla_bf16* out, int S, int n_q, int n_kv, int hd,
                       int q_ld, int kv_ld, int out_ld, float scale,
                       vla_stream stream);

#ifdef __cplusplus
}
#endif
