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
 * @file test_ladder_pack.cpp
 * @brief The ladder int2 layout is a bijection, and the unpack is its inverse.
 *
 * The oneDNN reference GEMM reads BitVLA's published weights by unpacking them
 * to row-major s8, so a wrong unpack would show up as a model that produces
 * plausible but wrong actions - the worst failure mode available. Three
 * properties pin it down, on the real weight shapes:
 *
 *   1. Round trip. pack then unpack is the identity on arbitrary ternary data.
 *   2. Coverage. The slot decomposition touches every (n, k) exactly once; if
 *      it did not, (1) could still pass on data that happens to agree.
 *   3. Agreement with the converter. One hand-computed slot, checked against
 *      the addressing in scripts/convert_bitvla_to_gguf.py, so a change made in
 *      lockstep to both directions here still has to answer to the file format.
 *
 * Pure CPU, no GPU and no checkpoint, so it runs in ctest everywhere.
 */

#include "kernels/bitvla/ladder_pack.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using vla::bitvla::ladder_pack_int2;
using vla::bitvla::ladder_slot_addr;
using vla::bitvla::ladder_unpack_int2;

namespace {

int g_failures = 0;

void expect(bool cond, const std::string & what) {
    std::printf("  %-52s %s\n", what.c_str(), cond ? "OK" : "FAIL");
    if (!cond) ++g_failures;
}

/// Ternary values in {-1,0,+1} from a fixed LCG.
std::vector<int8_t> ternary(int64_t n, uint32_t seed) {
    std::vector<int8_t> v((size_t) n);
    uint32_t            s = seed;
    for (int64_t i = 0; i < n; ++i) {
        s    = s * 1664525u + 1013904223u;
        v[(size_t) i] = (int8_t) ((int) ((s >> 16) % 3u) - 1);
    }
    return v;
}

/// The real BitVLA shapes: LM q/o, LM gate_up, LM down, and the ViT's qkv.
const int64_t SHAPES[][2] = {
    {2560, 2560}, {13824, 2560}, {2560, 6912}, {3456, 1152}, {16, 128},
};

void test_round_trip() {
    for (const auto & sh : SHAPES) {
        const int64_t N = sh[0], K = sh[1];
        const auto    W = ternary(N * K, (uint32_t) (N * 31 + K));

        std::vector<uint8_t> packed((size_t) (N * K / 4));
        ladder_pack_int2(W.data(), N, K, packed.data());

        std::vector<int8_t> back((size_t) (N * K), 127);
        ladder_unpack_int2(packed.data(), N, K, back.data());

        int64_t bad = 0;
        for (int64_t i = 0; i < N * K; ++i)
            if (back[(size_t) i] != W[(size_t) i]) ++bad;
        expect(bad == 0, "round trip " + std::to_string(N) + "x" + std::to_string(K) + " (" +
                             std::to_string(bad) + " wrong)");
    }
}

void test_coverage() {
    // A slot addressing that skipped an element would leave a hole; one that
    // aliased would leave a count of 2 and a hole elsewhere. Both show here.
    for (const auto & sh : SHAPES) {
        const int64_t N = sh[0], K = sh[1];

        std::vector<uint8_t> hits((size_t) (N * K), 0);
        for (int64_t s = 0; s < N * K / 16; ++s) {
            int64_t n_global, k_base;
            ladder_slot_addr(s, K, n_global, k_base);
            bool in_range = n_global >= 0 && n_global < N && k_base >= 0 && k_base + 16 <= K;
            if (!in_range) {
                expect(false, "slot " + std::to_string(s) + " addresses outside the matrix");
                return;
            }
            for (int t = 0; t < 16; ++t) ++hits[(size_t) (n_global * K + k_base + t)];
        }
        int64_t not_once = 0;
        for (int64_t i = 0; i < N * K; ++i)
            if (hits[(size_t) i] != 1) ++not_once;
        expect(not_once == 0, "coverage " + std::to_string(N) + "x" + std::to_string(K) + " (" +
                                  std::to_string(not_once) + " not hit exactly once)");
    }
}

void test_addressing_matches_converter() {
    // Slot 0 is the first 16 K positions of row 0, and within it value t lands
    // in byte t%4 at bit pair t/4 - the transpose that makes four consecutive
    // values one dp4a lane. Encode 16 distinguishable values and read the bytes
    // back by hand rather than through the unpack.
    const int64_t N = 16, K = 128;
    std::vector<int8_t> W((size_t) (N * K), 0);
    for (int t = 0; t < 16; ++t) W[(size_t) t] = (int8_t) ((t % 3) - 1);

    std::vector<uint8_t> packed((size_t) (N * K / 4));
    ladder_pack_int2(W.data(), N, K, packed.data());

    bool ok = true;
    for (int t = 0; t < 16; ++t) {
        const uint8_t enc = (uint8_t) ((packed[(size_t) (t % 4)] >> (2 * (t / 4))) & 0x3u);
        if ((int) enc - 2 != (int) W[(size_t) t]) ok = false;
    }
    expect(ok, "slot 0 byte/bit layout matches the converter");

    // Hand-walk the first block for K=256, against the divisions in the
    // converter. The order is: eight rows, then the second 16-value sub-tile of
    // those same rows, then the other eight rows, then the next wmma tile.
    const int64_t K2 = 256;
    struct Expect { int64_t slot, n, k; const char * what; };
    const Expect cases[] = {
        {  1, 1,   0, "slot 1 is row 1, k 0 (consecutive slots walk rows)"        },
        {  8, 0,  16, "slot 8 is row 0, k 16 (sub_k, the second half of a tile)"  },
        { 16, 8,   0, "slot 16 is row 8, k 0 (y_half, the upper eight rows)"      },
        { 32, 0,  32, "slot 32 is row 0, k 32 (major_k steps one wmma tile)"      },
        {128, 0, 128, "slot 128 is row 0, k 128 (k_0 steps K_PER_ITER)"           },
        {256, 16,  0, "slot 256 is row 16, k 0 (the next 16-row block)"           },
    };
    for (const auto & c : cases) {
        int64_t n_global, k_base;
        ladder_slot_addr(c.slot, K2, n_global, k_base);
        expect(n_global == c.n && k_base == c.k, c.what);
    }
}

}  // namespace

int main() {
    std::printf("ladder int2 pack/unpack\n");
    test_round_trip();
    test_coverage();
    test_addressing_matches_converter();
    if (g_failures) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
