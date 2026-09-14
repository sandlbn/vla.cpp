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
 * @file weight_cache.h
 * @brief Lifetime hook for the oneDNN path's unpacked weight copies.
 *
 * The reference GEMM expands each ladder-packed weight matrix to row-major s8
 * once and keys the result on the packed device pointer. That is only sound as
 * long as the pointer identifies the matrix, and a freed pointer does not: the
 * SYCL allocator is free to hand the same address back for the next allocation,
 * at which point a cache hit would silently feed the previous tensor's weights
 * into this tensor's GEMM. The output stays finite and plausible, which is the
 * failure mode worth engineering against.
 *
 * In the engine this never arises - BitVLA allocates its weights at load and
 * frees them at shutdown - so it would have been easy to leave the cache
 * unguarded and be right in production. It was @c tests/test_bitvla_gemm_gpu,
 * which allocates and frees a weight matrix per shape, that made the latent bug
 * an actual one. Hence this hook rather than a comment asserting the engine
 * behaves.
 */

#pragma once

namespace vla {

/**
 * @brief Drop any unpacked copy held for @p packed.
 *
 * Called by @c vla_dev_free before the underlying allocation is released. A
 * pointer with no cached copy - which is almost all of them - costs one lookup.
 */
void bitvla_forget_unpacked(const void * packed);

}  // namespace vla
