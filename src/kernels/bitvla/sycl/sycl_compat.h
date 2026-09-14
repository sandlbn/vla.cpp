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
 * @file sycl_compat.h
 * @brief Bridge between @c device.h's vendor-neutral types and SYCL's own.
 *
 * The twin of @c kernels/bitvla/cuda_compat.h. @c device.h states bf16 as
 * storage so the engine needs neither @c cuda_bf16.h nor @c sycl.hpp; the
 * conversion to the vendor type happens here, at the entry point.
 *
 * Also collects the numeric helpers the ports share, so that "reproduce the
 * CUDA formula literally" is a decision made once rather than per kernel.
 */

#pragma once

#include "kernels/bitvla/device.h"

#include <sycl/sycl.hpp>

namespace vla {
namespace bitvla {

/// The device-side bf16 type. Same 16 bits as @ref vla_bf16, but conversions
/// compile to Xe's native @c cvt instructions instead of shifts.
using bf16 = sycl::ext::oneapi::bfloat16;

static_assert(sizeof(bf16) == sizeof(vla_bf16),
              "vla_bf16 must be the same 16 bits sycl bfloat16 is");
static_assert(alignof(bf16) == alignof(vla_bf16),
              "vla_bf16 and sycl bfloat16 must agree on alignment");

/// @brief Reinterpret a neutral bf16 buffer as SYCL's bf16 type.
inline bf16 * as_bf(vla_bf16 * p) { return reinterpret_cast<bf16 *>(p); }

/// @copydoc as_bf
inline const bf16 * as_bf(const vla_bf16 * p) { return reinterpret_cast<const bf16 *>(p); }

/**
 * @brief bf16 -> f32, exact.
 * @note The CUDA side spells this @c __bfloat162float. Widening bf16 to f32 is
 *       lossless on both, so this is one of the few places where "the same
 *       precision" needs no argument.
 */
inline float to_f32(const bf16 & v) { return static_cast<float>(v); }

/**
 * @brief f32 -> bf16, round-to-nearest-even.
 * @note Matches CUDA's @c __float2bfloat16 bit for bit: both are the hardware
 *       RNE convert, and @c --use_fast_math does not touch conversions. Every
 *       bf16 result the ported kernels write therefore rounds identically to
 *       the reference, and any deviation that does appear came from the f32
 *       arithmetic before this point - which is where to look.
 */
inline bf16 to_bf16(float v) { return bf16(v); }

/**
 * @brief @p a / @p b, correctly rounded, without relying on a build option.
 *
 * SYCL permits float division up to 2.5 ULP of error and Battlemage takes it:
 * @c act_quant's `127.0f/amax` came out exactly 1 ULP high (0x42295556 against
 * the correct 0x42295555) on 58 of 61 rows. Not one quantised int8 changed, but
 * the ternary GEMM's epilogue divides by the same scale, and the claim there is
 * bit-exactness against host arithmetic - so a 1 ULP float error is the whole
 * claim, not a rounding detail.
 *
 * Neither @c -foffload-fp32-prec-div nor the runtime's
 * @c -cl-fp32-correctly-rounded-divide-sqrt changed the result on this stack, so
 * the fix is in the source, where it cannot be undone by a build that forgets a
 * flag. This is Markstein's division: refine the hardware reciprocal with one
 * Newton step, form the quotient, then correct it by its own exact residual.
 * Both @c fma calls are single instructions and exact, so the last one lands on
 * the correctly-rounded result.
 *
 * Only valid for the finite, normal, positive-denominator inputs these kernels
 * produce - @c amax is floored at 1e-5 and the GEMM's divisor is that same
 * scale. It is not a general-purpose divide and is not trying to be.
 *
 * Cost is two extra FMAs, once per row for the scale and once per output
 * element for the epilogue, against a K-length reduction. Not measurable.
 */
inline float exact_div(float a, float b) {
    float y = 1.0f / b;                     // hardware reciprocal, within ~2.5 ULP
    y       = sycl::fma(y, sycl::fma(-b, y, 1.0f), y);  // Newton: now within ~0.5 ULP
    const float q = a * y;
    return sycl::fma(sycl::fma(-b, q, a), y, q);        // residual correction
}

/**
 * @brief Work-group size for the row-per-group kernels.
 *
 * The CUDA kernels use 256 threads per row and reduce with a warp shuffle plus
 * a 32-entry shared-memory stage. Here @c reduce_over_group does both steps, so
 * the only thing that has to carry over is the number of work-items sharing a
 * row - which sets how much of the row each one accumulates in f32, and hence
 * the shape of the summation tree.
 */
constexpr int ROW_WG = 256;

/**
 * @brief Round @p n up to a whole number of work-groups of @p wg items.
 */
inline size_t round_up(size_t n, size_t wg) { return ((n + wg - 1) / wg) * wg; }

}  // namespace bitvla
}  // namespace vla
