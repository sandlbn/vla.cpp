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
 * @file cuda_compat.h
 * @brief Bridge between @c device.h's vendor-neutral types and CUDA's own.
 *
 * @c device.h states bf16 as storage (@ref vla_bf16) and streams as an opaque
 * handle (@ref vla_stream) so that @c src/models/bitvla.cpp has one set of call
 * sites for both backends. Inside a @c .cu the values are @c __nv_bfloat16 and
 * @c cudaStream_t again, and the conversion happens here - at the entry point,
 * in one line, statically checked - rather than by hoping two translation units
 * agree on a typedef.
 *
 * It also carries the one arithmetic guarantee the two backends have to state
 * separately because each has its own way of losing it (@ref vla_exact_div,
 * below). @c sycl/sycl_compat.h is the twin of this file and holds the same
 * pair of responsibilities.
 *
 * CUDA sources only; the SYCL tree has its own equivalent.
 */

#pragma once

#include "kernels/bitvla/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

static_assert(sizeof(vla_bf16) == sizeof(__nv_bfloat16),
              "vla_bf16 must be the same 16 bits __nv_bfloat16 is");
static_assert(alignof(vla_bf16) == alignof(__nv_bfloat16),
              "vla_bf16 and __nv_bfloat16 must agree on alignment");

/// @brief Reinterpret a neutral bf16 buffer as CUDA's bf16 type.
__host__ __device__ inline __nv_bfloat16 * vla_cu_bf(vla_bf16 * p) {
    return reinterpret_cast<__nv_bfloat16 *>(p);
}

/// @copydoc vla_cu_bf
__host__ __device__ inline const __nv_bfloat16 * vla_cu_bf(const vla_bf16 * p) {
    return reinterpret_cast<const __nv_bfloat16 *>(p);
}

/**
 * @brief Reinterpret an opaque stream handle as a CUDA stream.
 * @note A null handle is the default stream, which is what CUDA means by 0 -
 *       the same convention @c device.h documents.
 */
inline cudaStream_t vla_cu_stream(vla_stream s) {
    return static_cast<cudaStream_t>(s);
}

/**
 * @brief @p a / @p b, correctly rounded, regardless of the build's fast-math.
 *
 * This target compiles with @c --use_fast_math (CMakeLists.txt), which implies
 * @c -prec-div=false: the @c / operator becomes an approximate reciprocal and
 * multiply. @c act_quant's `127.0f/amax` then came out exactly 1 ULP high -
 * 0x42295556 against the correct 0x42295555 - on every row of all eleven shapes
 * @c test_bitvla_gemm_gpu dispatches. Not one quantised int8 moved, because the
 * error is far below a bf16 step and @c nearbyintf swallowed it; the scale
 * itself is stored as raw f32 and handed to the GEMM epilogue, which divides by
 * it again, and the claim there is bit-exactness against host arithmetic. A 1
 * ULP float error is that whole claim, not a rounding detail.
 *
 * The SYCL port hit the identical bit patterns for an unrelated reason - SYCL
 * permits 2.5 ULP on float division and Battlemage takes it - and fixed it in
 * the source rather than with a flag (@c exact_div in @c sycl/sycl_compat.h,
 * Markstein division). CUDA has a correctly-rounded divide in hardware, so here
 * it is one instruction: @c __fdiv_rn is an intrinsic, and @c --use_fast_math
 * does not rewrite intrinsics. Same guarantee, both backends.
 *
 * @note This is not free, and the first version of this comment claimed it was -
 *       from counting call sites rather than from a benchmark. The scale divide
 *       is once per row and does not matter; the GEMM epilogue divides once per
 *       *output element*, order 238M times per request.
 *       @c ci/slurm/h100_exact_div_ab.sbatch measured it on H100 NVL: 23.4 ms
 *       per request against 22.0 ms for the incorrect fast divide, 6.4%, of
 *       which 0.6 ms is the ViT. Worth paying, but do not add call sites
 *       believing it is free.
 *
 * @note The Markstein sequence @c sycl_compat.h uses was tried here and is not
 *       worth it on CUDA: 23.3 ms, inside the noise of @c __fdiv_rn's 23.4. The
 *       hope was that its reciprocal refinement, which depends only on @p b,
 *       would hoist out of the epilogue's row loop and leave three FMAs per
 *       element. It did not show up as a win, so the one-instruction form stays.
 *       SYCL keeps Markstein because Battlemage has no correctly-rounded divide
 *       to call - different hardware, same guarantee, different route to it.
 */
__device__ __forceinline__ float vla_exact_div(float a, float b) {
    return __fdiv_rn(a, b);
}
