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
 * @file test_bitvla_gemm_gpu.cpp
 * @brief The ternary GEMM against an exact CPU reference, on every real shape.
 *
 * @c tests/bitvla_gemm_check.cu compares BitVLA's two CUDA tilings to each
 * other, which catches a broken tiling but not a systematically wrong one. This
 * compares against arithmetic done on the host, and it can afford to demand
 * *equality* rather than a tolerance, because nothing in the GEMM is
 * approximate:
 *
 *   * The accumulator is int32. |acc| <= 127 * K <= 3.5e6 against a 2.1e9 range,
 *     so no term overflows and no ordering of the sum differs from any other -
 *     integer addition is associative. A GPU that tiles K differently to the
 *     host loop still has to produce the identical int32.
 *   * The epilogue is (float)acc / s[m] * ws[g], three IEEE operations with no
 *     contraction opportunity between them, rounded once to bf16 by RNE.
 *
 * So a single wrong bit is a bug, not noise, and that is the gate below. It is
 * what lets the oneDNN path serve as the oracle for the hand-written DPAS
 * kernel: both are being held to the same external answer, not to each other.
 *
 * One backend-specific caveat. The CUDA kernels are built with
 * @c --use_fast_math, which lowers float division to a reciprocal approximation
 * good to ~2 ULP. Those 2 ULP are far below a bf16 mantissa step, so they are
 * invisible *unless* the exact quotient sits within 2 ULP of a bf16 rounding
 * tie. @ref MAX_STEPS therefore allows one bf16 step on CUDA and demands zero on
 * SYCL, which is compiled without fast-math and has no excuse.
 *
 * Quantisation is checked separately and first, against the same formula on the
 * host, and the GEMM reference is then built from the int8 the device actually
 * produced. A quantisation bug reports as a quantisation bug rather than
 * reappearing downstream as a GEMM failure.
 *
 * Skips itself (exit 0) when no GPU is present, so ctest can run it anywhere.
 */

#include "kernels/bitvla/device.h"
#include "kernels/bitvla/ladder_pack.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
void bitlinear_int8xint2(int8_t * A, int8_t * B, vla_bf16 * out, float * s, float * ws, int M,
                         int N, int K, vla_stream stream);
void bitlinear_int8xint2_m(int8_t * A, int8_t * B, vla_bf16 * out, float * s, float * ws, int M,
                           int N, int K, vla_stream stream);
void bitvla_act_quant_cuda(const vla_bf16 * in, int8_t * out, float * scales, int M, int K,
                           vla_stream stream);
}

namespace {

#ifdef VLA_GEMM_CHECK_FAST_MATH
/// CUDA builds with --use_fast_math; see the file comment.
constexpr int32_t MAX_STEPS = 1;
#else
constexpr int32_t MAX_STEPS = 0;
#endif

int g_failures = 0;

// --- bf16 on the host -------------------------------------------------------

uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;  // round-nearest-even, matching every bf16 convert here
    return (uint16_t) (u >> 16);
}

float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float          f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// bf16 bit patterns as a signed magnitude ordering, so adjacent values differ
/// by 1 and "how wrong" is countable rather than relative.
int32_t bf_ord(uint16_t b) {
    const int32_t mag = (int32_t) (b & 0x7fffu);
    return (b & 0x8000u) ? -mag : mag;
}

int32_t bf_steps(uint16_t a, uint16_t b) {
    const int32_t d = bf_ord(a) - bf_ord(b);
    return d < 0 ? -d : d;
}

// --- device buffers ---------------------------------------------------------

template <typename T> struct DevBuf {
    T *    p = nullptr;
    size_t n = 0;

    explicit DevBuf(size_t count) : n(count) {
        p = (T *) vla_dev_malloc(count * sizeof(T));
        if (!p) {
            std::fprintf(stderr, "device allocation of %zu bytes failed: %s\n",
                         count * sizeof(T), vla_dev_error());
            std::exit(1);
        }
    }
    ~DevBuf() { vla_dev_free(p); }
    DevBuf(const DevBuf &)             = delete;
    DevBuf & operator=(const DevBuf &) = delete;

    void put(const std::vector<T> & h) { vla_dev_memcpy_h2d(p, h.data(), h.size() * sizeof(T)); }
    std::vector<T> get() const {
        std::vector<T> h(n);
        vla_dev_memcpy_d2h(h.data(), p, n * sizeof(T));
        return h;
    }
};

// --- deterministic inputs ---------------------------------------------------

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed ? seed : 1u) {}
    uint32_t next() {
        s = s * 1664525u + 1013904223u;
        return s;
    }
    /// Uniform in [-1, 1), then snapped to bf16 so the host and the device are
    /// reading the identical value.
    uint16_t bf(float scale) {
        const float f = ((float) (next() >> 8) / (float) (1u << 24)) * 2.0f - 1.0f;
        return f2bf(f * scale);
    }
    int8_t ternary() { return (int8_t) ((int) (next() >> 16) % 3 - 1); }
};

// --- the check ---------------------------------------------------------------

struct Shape {
    int         M, N, K;
    const char * name;
};

void check(const Shape & sh) {
    const int M = sh.M, N = sh.N, K = sh.K;
    const int ws_num = [&] {
        // Whatever the kernel uses, the reference has to use too. Derive it from
        // the shape the same way both dispatch tables do.
        if (N == 3840 && K == 2560) return 3;
        if (N == 13824 && K == 2560) return 2;
        if (N == 4800 && K == 3200) return 6;
        if (N == 20480 && K == 3200) return 2;
        return 1;
    }();

    Rng rng((uint32_t) (M * 7919 + N * 131 + K));

    // Weights: ternary, packed the way a GGUF carries them.
    std::vector<int8_t> W((size_t) N * K);
    for (auto & w : W) w = rng.ternary();
    std::vector<uint8_t> packed((size_t) N * K / 4);
    vla::bitvla::ladder_pack_int2(W.data(), N, K, packed.data());

    // Activations, and a per-group weight scale in the range absmean produces.
    std::vector<vla_bf16> A_bf((size_t) M * K);
    for (auto & a : A_bf) a = rng.bf(3.0f);
    std::vector<float> ws((size_t) ws_num);
    for (auto & w : ws) w = 0.01f + (float) (rng.next() >> 20) / 1.0e6f;

    DevBuf<vla_bf16> d_A_bf(A_bf.size());
    DevBuf<int8_t>   d_A_q((size_t) M * K);
    DevBuf<float>    d_s((size_t) M);
    DevBuf<int8_t>   d_W((size_t) N * K / 4);
    DevBuf<float>    d_ws(ws.size());
    DevBuf<vla_bf16> d_out((size_t) M * N);

    d_A_bf.put(A_bf);
    vla_dev_memcpy_h2d(d_W.p, packed.data(), packed.size());
    d_ws.put(ws);

    // --- quantisation ---
    bitvla_act_quant_cuda(d_A_bf.p, d_A_q.p, d_s.p, M, K, nullptr);
    vla_dev_sync(nullptr);

    const std::vector<int8_t> A_q = d_A_q.get();
    const std::vector<float>  s   = d_s.get();

    int64_t     q_bad = 0, s_bad = 0;
    std::string s_detail;
    for (int m = 0; m < M; ++m) {
        float amax = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float v = std::fabs(bf2f(A_bf[(size_t) m * K + k]));
            if (v > amax) amax = v;
        }
        if (amax < 1e-5f) amax = 1e-5f;
        const float scale = 127.0f / amax;
        if (s[(size_t) m] != scale) {
            ++s_bad;
            if (s_detail.empty()) {
                // Print the bits: a scale that is one ULP out prints identically
                // to the exact one in decimal, and that is the likely failure.
                uint32_t g, w;
                std::memcpy(&g, &s[(size_t) m], 4);
                std::memcpy(&w, &scale, 4);
                char buf[128];
                std::snprintf(buf, sizeof buf, " first m=%d got %.9g (0x%08x) want %.9g (0x%08x)",
                              m, (double) s[(size_t) m], g, (double) scale, w);
                s_detail = buf;
            }
        }
        for (int k = 0; k < K; ++k) {
            float q = std::rint(bf2f(A_bf[(size_t) m * K + k]) * scale);
            if (q > 127.0f) q = 127.0f;
            if (q < -128.0f) q = -128.0f;
            if (A_q[(size_t) m * K + k] != (int8_t) q) ++q_bad;
        }
    }
    if (q_bad || s_bad) {
        std::printf("  %-34s FAIL  act_quant: %lld value(s), %lld scale(s) wrong%s\n", sh.name,
                    (long long) q_bad, (long long) s_bad, s_detail.c_str());
        ++g_failures;
        return;
    }

    // --- GEMM ---
    if (M == 1)
        bitlinear_int8xint2(d_A_q.p, d_W.p, d_out.p, d_s.p, d_ws.p, M, N, K, nullptr);
    else
        bitlinear_int8xint2_m(d_A_q.p, d_W.p, d_out.p, d_s.p, d_ws.p, M, N, K, nullptr);
    vla_dev_sync(nullptr);

    const std::vector<vla_bf16> out = d_out.get();

    // Reference from the int8 the device produced, so this measures the GEMM.
    const int   per_group = N / ws_num;
    int32_t     worst     = 0;
    int64_t     over      = 0;
    std::string first_bad;
    for (int m = 0; m < M; ++m) {
        const int8_t * a = A_q.data() + (size_t) m * K;
        for (int n = 0; n < N; ++n) {
            // Both operands walk K contiguously: W is (N, K) row-major, so this
            // is the plain dot product and stays in cache on the 13824x2560
            // shapes, where the transposed order would not.
            const int8_t * w   = W.data() + (size_t) n * K;
            int32_t        acc = 0;
            for (int k = 0; k < K; ++k) acc += (int32_t) a[k] * (int32_t) w[k];

            const uint16_t want = f2bf((float) acc / s[(size_t) m] * ws[(size_t) (n / per_group)]);
            const uint16_t got  = out[(size_t) m * N + n];
            const int32_t  d    = bf_steps(got, want);
            if (d > worst) worst = d;
            if (d > MAX_STEPS) {
                ++over;
                if (first_bad.empty())
                    first_bad = " first at m=" + std::to_string(m) + " n=" + std::to_string(n) +
                                " acc=" + std::to_string(acc) + " got " +
                                std::to_string(bf2f(got)) + " want " + std::to_string(bf2f(want));
            }
        }
    }

    const bool ok = over == 0;
    std::printf("  %-34s %s  worst %d step(s) of %d, %lld over%s\n", sh.name, ok ? "OK  " : "FAIL",
                (int) worst, (int) MAX_STEPS, (long long) over, first_bad.c_str());
    if (!ok) ++g_failures;
}

/// Every shape both CUDA dispatch tables carry that BitVLA-libero actually
/// reaches, with M values that exercise the ragged tail (M % 8 != 0) as well as
/// a prefill-sized batch.
/// The shapes the engine dispatches, per VLA_BITVLA_LOG_SHAPES, plus the ones
/// only a kernel covers.
///
/// The first two groups are real: M is the sequence length because both drivers
/// project the whole sequence in one call. M=330 is 41 full 8-row tiles and a
/// 2-row tail, so it exercises the ragged edge the M>1 tiling has to get right;
/// M=256 divides evenly and does not.
///
/// The last group is not dispatched by this engine. It is here because
/// bitlinear_int8xint2 (M=1) has a hand-written packed implementation on SYCL
/// and an untested kernel is worse than no kernel - the CUDA dispatch chain
/// still routes BitNet LLM decoding through it, and vla.cpp exports it.
const Shape SHAPES[] = {
    {330, 2560, 2560, "lm      q_o       330x2560x2560"},
    {330, 640, 2560, "lm      kv        330x640x2560"},
    {330, 13824, 2560, "lm      gate_up ws=2 330x13824x2560"},
    {330, 2560, 6912, "lm      down      330x2560x6912"},

    {256, 1152, 1152, "vit     qkvo      256x1152x1152"},
    {256, 4304, 1152, "vit     fc1       256x4304x1152"},
    {256, 1152, 4352, "vit     fc2       256x1152x4352"},

    {1, 2560, 2560, "M=1     o_proj      1x2560x2560"},
    {1, 3840, 2560, "M=1     qkv ws=3    1x3840x2560"},
    {1, 13824, 2560, "M=1     gate_up ws=2 1x13824x2560"},
    {1, 2560, 6912, "M=1     down       1x2560x6912"},
};

}  // namespace

int main() {
    if (vla_dev_count() <= 0) {
        std::printf("no GPU device, skipping\n");
        return 0;
    }
    if (vla_dev_set(0) != 0) {
        std::fprintf(stderr, "vla_dev_set(0) failed: %s\n", vla_dev_error());
        return 1;
    }

    std::printf("bitvla ternary GEMM vs exact CPU reference (gate: %d bf16 step(s))\n",
                (int) MAX_STEPS);
    for (const auto & sh : SHAPES) check(sh);

    if (g_failures) {
        std::printf("FAILED: %d shape(s)\n", g_failures);
        return 1;
    }
    std::printf("all shapes within gate\n");
    return 0;
}
