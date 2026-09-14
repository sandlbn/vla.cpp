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
 * @file fp32_ops.h
 * @brief The fp32 elementwise and norm ops behind BitVLA's action head.
 *
 * Separate from the bf16 set in @c bitvla_lm_cuda.h because the head is
 * separate: proprioception and the action MLP stay in fp32 end to end while the
 * LM and ViT run bf16 activations over ternary weights. The head is small -
 * chunk x hidden, once per step - so the precision is nearly free, and the
 * actions it emits are what the robot executes, which is the wrong place to
 * spend a few bits.
 *
 * CUDA implements these in @c bitvla_fp32_ops_cuda.cu, SYCL in
 * @c sycl/bitvla_fp32_ops_sycl.cpp. Dense fp32 GEMMs come from @c gemm.h
 * instead; @ref vla_add_bias_fp32 exists because the GEMM does not fuse bias.
 */

#pragma once

#include "kernels/bitvla/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Elementwise exact GELU, @c 0.5*x*(1+erf(x/sqrt(2))), over @p N floats.
 * @param out May alias @p in.
 */
void vla_gelu_erf_fp32(const float* in, float* out, int N, vla_stream stream);

/** @brief In-place elementwise ReLU over @p N floats. */
void vla_relu_fp32(float* inout, int N, vla_stream stream);

/**
 * @brief Row-wise LayerNorm with per-channel affine, on an (M x K) matrix.
 * @param w   Per-channel scale (length K).
 * @param b   Per-channel bias  (length K).
 * @param out May alias @p x.
 * @param eps Added to the variance before the reciprocal square root.
 */
void vla_layernorm_fp32(const float* x, const float* w, const float* b, float* out,
                        float eps, int M, int K, vla_stream stream);

/**
 * @brief Broadcast-add a per-channel bias (length @p K) to an (M x K) matrix.
 * @param out May alias @p x.
 */
void vla_add_bias_fp32(const float* x, const float* bias, float* out,
                       int M, int K, vla_stream stream);

/** @brief Elementwise @c out = a + b over @p N floats; @p out may alias either. */
void vla_add_fp32(const float* a, const float* b, float* out, int N, vla_stream stream);

#ifdef __cplusplus
}
#endif
