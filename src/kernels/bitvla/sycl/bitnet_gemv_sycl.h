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
 * @file bitnet_gemv_sycl.h
 * @brief The M=1 ternary GEMV, reading the 2-bit pack directly.
 *
 * Separate from @c bitnet_sycl.cpp because it is a different kind of thing: that
 * file is the oneDNN reference path and exists to be obviously correct, this one
 * is the shipping kernel for LM decode and exists to be fast. They are held to
 * the same bit-exact standard by the same test.
 *
 * Included by SYCL sources only.
 */

#pragma once

#include "kernels/bitvla/device.h"

#include <cstdint>

namespace vla {
namespace bitvla {

/**
 * @brief out = dequant(A x W^T) for a single activation row, W ladder-packed.
 *
 * @param A       Quantised activations, K int8 values.
 * @param packed  Ladder int2 weights, N*K/4 bytes. Read in place - no unpack.
 * @param out     N bf16 outputs.
 * @param s       The single activation scale (device pointer, one float).
 * @param ws      Weight scales, @p ws_num of them, one per column group.
 * @param stream  Null for the process queue.
 *
 * @return false if the shape is not one this kernel handles, having done
 *         nothing, so the caller can fall back to the oneDNN path. Requires
 *         @c K%32==0 and @c N%16==0, which every BitVLA shape satisfies.
 */
bool ternary_gemv_packed(const int8_t * A, const int8_t * packed, vla_bf16 * out, const float * s,
                         const float * ws, int N, int K, int ws_num, vla_stream stream);

}  // namespace bitvla
}  // namespace vla
