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
 * @file test_bitvla_quant.cpp
 * @brief scripts/bitvla_quant.py, read back by the implementation that matters.
 *
 * The Python encoders turn BitVLA's ternary weights into Q8_0 and Q4_0 blocks,
 * and ci/test_bitvla_quant.py already round-trips them through gguf-py. That
 * proves gguf-py agrees with itself. It does not prove gguf-py agrees with
 * ggml, and the two disagreeing is exactly the failure this file exists for:
 *
 *   Q4_0 packs element j in the low nibble and element **j + 16** in the high
 *   nibble, not element j + 1. Pair them the obvious way and every weight still
 *   round-trips through Python, every block is the right size, every scale is
 *   right - and the model computes a permuted GEMM. On BitVLA that surfaces as
 *   plausible-but-wrong actions, which no action-level test on this model can
 *   attribute (see docs: the act_quant decorrelation floor).
 *
 * So: ci/test_bitvla_quant.py writes tests/data/bitvla_q_blocks.bin from the
 * encoders and byte-compares it on every run, and this test dequantises that
 * same file through `ggml_get_type_traits(...)->to_float` - the C code the
 * loader itself reaches - and demands {-s, 0, +s} back. Neither side can drift
 * without the other noticing.
 *
 * The fixture is self-describing (type, shape, scale and the source ternary all
 * travel with the blocks), so nothing about the encoders is hardcoded here.
 *
 * Pure CPU, no GPU and no checkpoint, so it runs in ctest everywhere.
 */

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void expect(bool cond, const std::string & what) {
    std::printf("  %-56s %s\n", what.c_str(), cond ? "OK" : "FAIL");
    if (!cond) ++g_failures;
}

/// Little-endian reader over the fixture. Refuses to run off the end rather
/// than reading whatever follows it, so a truncated file fails as a truncated
/// file instead of as a numerical mismatch.
struct Reader {
    const uint8_t * p   = nullptr;
    const uint8_t * end = nullptr;
    bool            ok  = true;

    bool take(void * dst, size_t n) {
        if (!ok || (size_t) (end - p) < n) {
            ok = false;
            return false;
        }
        std::memcpy(dst, p, n);
        p += n;
        return true;
    }
    uint32_t u32() {
        uint32_t v = 0;
        take(&v, sizeof v);
        return v;
    }
    float f32() {
        float v = 0;
        take(&v, sizeof v);
        return v;
    }
    const uint8_t * bytes(size_t n) {
        if (!ok || (size_t) (end - p) < n) {
            ok = false;
            return nullptr;
        }
        const uint8_t * b = p;
        p += n;
        return b;
    }
};

bool read_file(const char * path, std::vector<uint8_t> & out) {
    std::FILE * f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n < 0) {
        std::fclose(f);
        return false;
    }
    out.resize((size_t) n);
    const size_t got = n ? std::fread(out.data(), 1, (size_t) n, f) : 0;
    std::fclose(f);
    return got == (size_t) n;
}

}  // namespace

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : VLA_BITVLA_QUANT_FIXTURE;

    std::vector<uint8_t> raw;
    if (!read_file(path, raw)) {
        // Not a skip. The fixture is checked in; if it is gone, the Python half
        // of this pair is not being run either.
        std::printf("FAIL cannot read fixture %s\n"
                    "     regenerate with: python3 ci/test_bitvla_quant.py --update-fixture\n",
                    path);
        return 1;
    }

    Reader r{raw.data(), raw.data() + raw.size(), true};

    char magic[4] = {};
    r.take(magic, 4);
    if (!r.ok || std::memcmp(magic, "BVQF", 4) != 0) {
        std::printf("FAIL %s is not a BVQF fixture\n", path);
        return 1;
    }
    const uint32_t version = r.u32();
    const uint32_t n_cases = r.u32();
    if (version != 1) {
        std::printf("FAIL fixture version %u, this test knows version 1\n", version);
        return 1;
    }
    std::printf("%s: %u cases\n", path, n_cases);
    expect(n_cases > 0, "fixture carries at least one case");

    std::vector<float> deq;

    for (uint32_t c = 0; c < n_cases && r.ok; ++c) {
        const uint32_t type_u  = r.u32();
        const uint32_t rows    = r.u32();
        const uint32_t k       = r.u32();
        const float    scale   = r.f32();
        const uint32_t n_bytes = r.u32();
        if (!r.ok) break;

        const ggml_type type   = (ggml_type) type_u;
        const uint8_t * blocks = r.bytes(n_bytes);
        const int8_t *  tern   = (const int8_t *) r.bytes((size_t) rows * k);
        if (!r.ok) break;

        char tag[96];
        std::snprintf(tag, sizeof tag, "%s %ux%u s=%g",
                      ggml_type_name(type), rows, k, (double) scale);

        // The encoder's own idea of how big a row is, against ggml's.
        const size_t row_bytes = ggml_row_size(type, k);
        if (row_bytes * rows != n_bytes) {
            expect(false, std::string(tag) + ": row size");
            continue;
        }

        const ggml_type_traits * tr = ggml_get_type_traits(type);
        if (!tr || !tr->to_float) {
            expect(false, std::string(tag) + ": ggml has no dequantiser");
            continue;
        }

        // Every value must come back as exactly one of {-s, 0, +s}, up to the
        // f16 rounding of the one scale each block stores. That bound is the
        // whole error budget: the codes land on grid points exactly, so any
        // real slack here means a layout disagreement, not accumulated error.
        const float tol      = std::fabs(scale) * (1.0f / 1024.0f);
        double      worst    = 0.0;
        int64_t     n_wrong  = 0;
        int64_t     first_at = -1;

        deq.assign((size_t) rows * k, 0.0f);
        for (uint32_t i = 0; i < rows; ++i)
            tr->to_float(blocks + (size_t) i * row_bytes, deq.data() + (size_t) i * k, k);

        for (size_t i = 0; i < deq.size(); ++i) {
            const double want = (double) tern[i] * (double) scale;
            const double err  = std::fabs((double) deq[i] - want);
            if (err > worst) worst = err;
            if (err > tol) {
                if (first_at < 0) first_at = (int64_t) i;
                ++n_wrong;
            }
        }

        char what[160];
        std::snprintf(what, sizeof what, "%s: ggml dequant, worst %.3g", tag, worst);
        expect(n_wrong == 0, what);
        if (n_wrong) {
            const size_t i = (size_t) first_at;
            std::printf("      %lld/%lld elements off; first at row %lld col %lld: "
                        "ternary %d -> want %.9g, ggml gave %.9g\n",
                        (long long) n_wrong, (long long) deq.size(),
                        (long long) (i / k), (long long) (i % k),
                        (int) tern[i], (double) tern[i] * scale, (double) deq[i]);
            if (type == GGML_TYPE_Q4_0)
                std::printf("      Q4_0 pairs element j with j+16, not j+1 - check "
                            "pack_ternary_q4_0's nibble packing first.\n");
        }
    }

    if (!r.ok) {
        std::printf("FAIL fixture is truncated\n");
        ++g_failures;
    }

    std::printf("%s\n", g_failures ? "FAILED" : "PASSED");
    return g_failures ? 1 : 0;
}
