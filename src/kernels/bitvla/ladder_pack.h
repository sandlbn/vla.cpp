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
 * @file ladder_pack.h
 * @brief The "ladder int2" weight layout, on the host, in both directions.
 *
 * BitVLA stores its ternary weights two bits per value in an interleave shaped
 * for CUDA's @c wmma @c matrix_b col-major lane layout: a 32-bit slot holds 16
 * consecutive values of one output row, and which row and which 16 values is a
 * five-level division of the slot index. @c scripts/convert_bitvla_to_gguf.py
 * writes it and the CUDA kernels read it back with @c lop3.b32.
 *
 * Two callers need it on the host:
 *
 *   * @c src/models/bitvla.cpp packs, when the GGUF carries F32 ternary weights
 *     rather than a pre-packed blob.
 *   * The oneDNN reference GEMM unpacks, because @c dnnl::matmul wants a plain
 *     row-major s8 matrix. The published BitVLA GGUFs *are* pre-packed, so this
 *     direction is not optional.
 *
 * The layout is not going to change - GGUF files have to stay portable across
 * backends, so the converter is fixed and the repack happens at load - but an
 * inverse that has silently drifted from its forward is the kind of bug that
 * shows up as "the model almost works". Hence @ref ladder_slot_addr: both
 * directions get their addressing from one function, and the round-trip is a
 * unit test rather than an argument.
 */

#pragma once

#include <cstdint>

namespace vla {
namespace bitvla {

/// Output rows per interleave block.
constexpr int LADDER_N_BLOCK = 16;
/// Work-items collaborating along K in the CUDA kernel this layout was cut for.
constexpr int LADDER_K_BLOCK = 8;
/// Ternary values in one 32-bit slot.
constexpr int LADDER_K_PER_LOOP = 16;
/// K extent of one wmma tile.
constexpr int LADDER_WMMA_K = 32;
/// K consumed by one iteration of the kernel's outer loop.
constexpr int LADDER_K_PER_ITER = LADDER_K_PER_LOOP * LADDER_K_BLOCK;

/**
 * @brief Where slot @p slot lives in the logical (N, K) weight matrix.
 *
 * @param slot     Slot index, in [0, N*K/16).
 * @param K        Reduction dimension of the weight matrix.
 * @param n_global Out: the output row this slot belongs to.
 * @param k_base   Out: the first of the 16 consecutive K positions it holds.
 *
 * The divisions peel the slot index apart in the order the kernel's index
 * arithmetic composes it: block of 16 rows, then K iteration, then wmma tile,
 * then which half of the 16 rows, then which of the two 16-value sub-tiles.
 */
inline void ladder_slot_addr(int64_t slot, int64_t K, int64_t & n_global, int64_t & k_base) {
    const int64_t slots_per_block = (LADDER_N_BLOCK * K) / 16;

    const int64_t n_block  = slot / slots_per_block;
    const int64_t in_block = slot % slots_per_block;
    const int64_t k_0      = in_block / 128;
    const int64_t in_k0    = in_block % 128;
    const int64_t major_k  = in_k0 / 32;
    const int64_t in_major = in_k0 % 32;
    const int64_t y_half   = in_major / 16;
    const int64_t in_yhalf = in_major % 16;
    const int64_t sub_k    = in_yhalf / 8;
    const int64_t y_in_h   = in_yhalf % 8;

    n_global = n_block * LADDER_N_BLOCK + y_half * 8 + y_in_h;
    k_base   = k_0 * LADDER_K_PER_ITER + major_k * LADDER_WMMA_K + sub_k * LADDER_K_PER_LOOP;
}

/**
 * @brief Position of ternary value @p t within a slot's four bytes.
 *
 * The 16 values are transposed across the four bytes rather than laid down in
 * order: value @c t sits in byte @c t%4 at bit pair @c t/4. That is what puts
 * four consecutive values in the four lanes of one 32-bit word, which is what
 * lets the decode be a single masked @c lop3 and the multiply a single
 * @c dp4a.
 */
inline void ladder_slot_bit_pos(int t, int & byte_i, int & shift) {
    byte_i = t % 4;
    shift  = 2 * (t / 4);
}

/**
 * @brief Pack a row-major ternary matrix into the ladder int2 layout.
 *
 * @param W   Row-major (N, K) values in {-1, 0, +1}.
 * @param N   Output rows; must be a multiple of @ref LADDER_N_BLOCK.
 * @param K   Reduction dimension; must be a multiple of @ref LADDER_K_PER_ITER.
 * @param out Destination, @c N*K/4 bytes, zeroed by this call.
 */
inline void ladder_pack_int2(const int8_t * W, int64_t N, int64_t K, uint8_t * out) {
    const int64_t n_slots = N * K / 16;
    for (int64_t s = 0; s < n_slots; ++s) {
        int64_t n_global, k_base;
        ladder_slot_addr(s, K, n_global, k_base);

        out[s * 4 + 0] = out[s * 4 + 1] = out[s * 4 + 2] = out[s * 4 + 3] = 0;
        for (int t = 0; t < 16; ++t) {
            int byte_i, shift;
            ladder_slot_bit_pos(t, byte_i, shift);
            // +2 maps {-1,0,1} onto {1,2,3}; the kernel's __vsubss4 takes it
            // back off four lanes at a time.
            const uint8_t enc = (uint8_t) ((int) W[n_global * K + k_base + t] + 2) & 0x3u;
            out[s * 4 + byte_i] |= (uint8_t) (enc << shift);
        }
    }
}

/**
 * @brief Exact inverse of @ref ladder_pack_int2.
 *
 * @param packed Ladder-packed weights, @c N*K/4 bytes.
 * @param N,K    Logical shape, same constraints as the pack.
 * @param out    Destination, @c N*K row-major values in {-1, 0, +1}.
 *
 * Every logical (n, k) is covered exactly once - the slot decomposition is a
 * bijection - so @p out needs no initialisation.
 */
inline void ladder_unpack_int2(const uint8_t * packed, int64_t N, int64_t K, int8_t * out) {
    const int64_t n_slots = N * K / 16;
    for (int64_t s = 0; s < n_slots; ++s) {
        int64_t n_global, k_base;
        ladder_slot_addr(s, K, n_global, k_base);

        for (int t = 0; t < 16; ++t) {
            int byte_i, shift;
            ladder_slot_bit_pos(t, byte_i, shift);
            const uint8_t enc = (uint8_t) ((packed[s * 4 + byte_i] >> shift) & 0x3u);
            out[n_global * K + k_base + t] = (int8_t) ((int) enc - 2);
        }
    }
}

}  // namespace bitvla
}  // namespace vla
