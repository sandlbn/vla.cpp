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
 * @file test_bitvla_ops_gpu.cpp
 * @brief CPU replay for the BitVLA bf16 ops, against whichever backend is
 *        linked in.
 *
 * The ops are declared in @c bitvla_lm_cuda.h in vendor-neutral terms, so this
 * file has no @c #ifdef in it: build it against @c bitvla_sycl_kernels and it
 * tests the SYCL port, against @c bitvla_cuda_kernels and it tests CUDA. Same
 * source, same thresholds - which is the point, because the port's contract is
 * stated relative to what CUDA achieves.
 *
 * ## What "same precision" is checked as
 *
 * Not equality against the other GPU. The reference here is the *exact*
 * computation: each op is replayed on the host in @c double from the same bf16
 * inputs, and the result rounded once to bf16. The gate is then a distance in
 * bf16 steps between that and what the device produced.
 *
 * This is stricter and more portable than a float-tolerance comparison. bf16
 * has 8 significand bits, so one step is ~0.4% relative - an op that lands
 * within one step of the exact answer has *no* accuracy left to lose, whatever
 * the backend did internally. It also means the copy-shaped ops (transposes,
 * gather, repeat_kv, zero_tail) can be held to zero steps rather than a
 * tolerance, which catches an indexing bug that a loose epsilon would hide.
 *
 * Skips itself with exit 0 when there is no GPU, matching test_bf16_cuda_ops.
 */

#include "kernels/bitvla/bitvla_lm_cuda.h"
#ifdef VLA_BITVLA_FUSED_OPS
#include "kernels/bitvla/attention.h"
#include "kernels/bitvla/bitvla_fused.h"
#include "kernels/bitvla/gemm.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" void gate_up_fused_sqrelu_mul_bf16(const vla_bf16* gu, vla_bf16* out,
                                              int seq, int ffn, vla_stream stream);
extern "C" void bitvla_act_quant_cuda(const vla_bf16* in, int8_t* out, float* scales,
                                      int M, int K, vla_stream stream);
extern "C" void bitvla_act_quant_pad_cuda(const vla_bf16* in, int8_t* out, float* scales,
                                          int M, int K_in, int K_out, vla_stream stream);

namespace {

int g_failures = 0;

// --- bf16 on the host -------------------------------------------------------

/// Round-to-nearest-even f32 -> bf16, the same rounding both device backends do.
uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t) (u >> 16);
}

float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float          f;
    std::memcpy(&f, &u, 4);
    return f;
}

/**
 * @brief Map a bf16 bit pattern onto a monotonically increasing integer.
 *
 * Adjacent keys are adjacent representable values, so subtracting two keys
 * counts the bf16 values between them. Negative zero and positive zero both map
 * to 0, which is what we want: they are the same number.
 */
int32_t bf_ord(uint16_t b) {
    const int32_t mag = (int32_t) (b & 0x7fffu);
    return (b & 0x8000u) ? -mag : mag;
}

/// Distance in representable bf16 values. 0 means bit-identical.
int32_t bf_steps(uint16_t a, uint16_t b) {
    const int32_t d = bf_ord(a) - bf_ord(b);
    return d < 0 ? -d : d;
}

// --- deterministic inputs ---------------------------------------------------

/// A fixed LCG, so a failure is reproducible without carrying a data file.
struct Rng {
    uint32_t s = 0x1234567u;
    float    next(float lo, float hi) {
        s = s * 1664525u + 1013904223u;
        const float u = (float) ((s >> 8) & 0xffffffu) / (float) 0x1000000u;
        return lo + u * (hi - lo);
    }
};

std::vector<uint16_t> rand_bf(size_t n, float lo, float hi, uint32_t seed) {
    Rng                   r{seed};
    std::vector<uint16_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = f2bf(r.next(lo, hi));
    return v;
}

// --- device plumbing --------------------------------------------------------

/// Owning device buffer; the ops take raw pointers, so this only handles life.
template <typename T>
struct DevBuf {
    T *    p = nullptr;
    size_t n = 0;
    explicit DevBuf(size_t count) : n(count) {
        p = (T *) vla_dev_malloc(count * sizeof(T));
        if (!p) {
            std::fprintf(stderr, "vla_dev_malloc failed: %s\n", vla_dev_error());
            std::exit(1);
        }
    }
    DevBuf(const std::vector<T> & h) : DevBuf(h.size()) { upload(h); }
    ~DevBuf() { vla_dev_free(p); }
    DevBuf(const DevBuf &)             = delete;
    DevBuf & operator=(const DevBuf &) = delete;

    void upload(const std::vector<T> & h) { vla_dev_memcpy_h2d(p, h.data(), h.size() * sizeof(T)); }
    std::vector<T> download() const {
        std::vector<T> h(n);
        vla_dev_sync(nullptr);
        vla_dev_memcpy_d2h(h.data(), p, n * sizeof(T));
        return h;
    }
};

/**
 * @brief Compare a device result against the exact reference and record it.
 *
 * @param max_steps Largest bf16 distance tolerated. 0 demands bit-identity,
 *                  which is what the copy-shaped ops are held to.
 */
void check(const char * name, const std::vector<uint16_t> & got,
           const std::vector<double> & exact, int32_t max_steps) {
    int32_t worst   = 0;
    size_t  worst_i = 0;
    int64_t n_over  = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const int32_t d = bf_steps(got[i], f2bf((float) exact[i]));
        if (d > worst) { worst = d; worst_i = i; }
        if (d > max_steps) ++n_over;
    }
    const bool ok = n_over == 0;
    std::printf("  %-34s %-4s worst %d step(s) of %d, %lld over\n", name, ok ? "OK" : "FAIL",
                (int) worst, (int) max_steps, (long long) n_over);
    if (!ok) {
        ++g_failures;
        std::printf("      first magnitude at [%zu]: got %.9g, exact %.9g\n", worst_i,
                    (double) bf2f(got[worst_i]), exact[worst_i]);
    }
}

// --- the ops ----------------------------------------------------------------

void test_rmsnorm() {
    // 2560 is the LM hidden width; 1 and 137 exercise a single row and a row
    // length that is not a multiple of the work-group.
    const int shapes[3][2] = {{7, 2560}, {1, 137}, {3, 64}};
    for (const auto & shape : shapes) {
        const int M = shape[0], K = shape[1];
        const float eps = 1e-5f;

        auto x = rand_bf((size_t) M * K, -3.0f, 3.0f, 11u);
        auto w = rand_bf((size_t) K, 0.2f, 1.8f, 23u);

        DevBuf<uint16_t> dx(x), dw(w), dout((size_t) M * K);
        bitvla_rmsnorm_bf16(dx.p, dw.p, dout.p, eps, M, K, nullptr);

        std::vector<double> ref((size_t) M * K);
        for (int m = 0; m < M; ++m) {
            double ss = 0.0;
            for (int k = 0; k < K; ++k) {
                const double v = bf2f(x[(size_t) m * K + k]);
                ss += v * v;
            }
            const double scale = 1.0 / std::sqrt(ss / K + eps);
            for (int k = 0; k < K; ++k)
                ref[(size_t) m * K + k] = bf2f(x[(size_t) m * K + k]) * scale * bf2f(w[k]);
        }
        check(("rmsnorm " + std::to_string(M) + "x" + std::to_string(K)).c_str(), dout.download(),
              ref, 1);
    }
}

void test_layernorm() {
    const int   M = 5, K = 1152;  // 1152 is the ViT width.
    const float eps = 1e-6f;

    auto x = rand_bf((size_t) M * K, -2.0f, 2.0f, 31u);
    auto w = rand_bf((size_t) K, 0.5f, 1.5f, 37u);
    auto b = rand_bf((size_t) K, -0.3f, 0.3f, 41u);

    DevBuf<uint16_t> dx(x), dw(w), db(b), dout((size_t) M * K);
    bitvla_layernorm_bf16(dx.p, dw.p, db.p, dout.p, eps, M, K, nullptr);

    std::vector<double> ref((size_t) M * K);
    for (int m = 0; m < M; ++m) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) sum += bf2f(x[(size_t) m * K + k]);
        const double mean = sum / K;
        double       vsum = 0.0;
        for (int k = 0; k < K; ++k) {
            const double v = bf2f(x[(size_t) m * K + k]) - mean;
            vsum += v * v;
        }
        const double inv_std = 1.0 / std::sqrt(vsum / K + eps);
        for (int k = 0; k < K; ++k)
            ref[(size_t) m * K + k] =
                (bf2f(x[(size_t) m * K + k]) - mean) * inv_std * bf2f(w[k]) + bf2f(b[k]);
    }
    check("layernorm 5x1152", dout.download(), ref, 1);
}

void test_rope() {
    const int H = 4, S = 9, D = 64, half = D / 2;

    auto               io = rand_bf((size_t) H * S * D, -1.5f, 1.5f, 53u);
    std::vector<float> cs((size_t) S * half), sn((size_t) S * half);
    for (int s = 0; s < S; ++s) {
        for (int k = 0; k < half; ++k) {
            const double th = (double) s / std::pow(10000.0, (double) k / half);
            cs[(size_t) s * half + k] = (float) std::cos(th);
            sn[(size_t) s * half + k] = (float) std::sin(th);
        }
    }

    DevBuf<uint16_t> dio(io);
    DevBuf<float>    dc(cs), ds(sn);
    bitvla_rope_neox_bf16(dio.p, dc.p, ds.p, H, S, D, nullptr);

    std::vector<double> ref((size_t) H * S * D);
    for (int h = 0; h < H; ++h) {
        for (int s = 0; s < S; ++s) {
            const size_t base = ((size_t) h * S + s) * D;
            for (int k = 0; k < half; ++k) {
                const double c = cs[(size_t) s * half + k];
                const double v = sn[(size_t) s * half + k];
                const double a = bf2f(io[base + k]);
                const double b = bf2f(io[base + k + half]);
                ref[base + k]        = a * c - b * v;
                ref[base + k + half] = b * c + a * v;
            }
        }
    }
    check("rope_neox 4x9x64", dio.download(), ref, 1);
}

void test_softmax() {
    // 137 and 512 straddle the work-group size; the first leaves a ragged tail.
    for (int S : {137, 512}) {
        const int   n_rows = 6;
        const float scale  = 0.125f;

        auto             io = rand_bf((size_t) n_rows * S, -4.0f, 4.0f, 61u + (uint32_t) S);
        DevBuf<uint16_t> dio(io);
        bitvla_softmax_scaled_bf16(dio.p, scale, n_rows, S, nullptr);

        std::vector<double> ref((size_t) n_rows * S);
        for (int r = 0; r < n_rows; ++r) {
            double mx = -INFINITY;
            for (int i = 0; i < S; ++i)
                mx = std::fmax(mx, (double) bf2f(io[(size_t) r * S + i]) * scale);
            double sum = 0.0;
            for (int i = 0; i < S; ++i)
                sum += std::exp((double) bf2f(io[(size_t) r * S + i]) * scale - mx);
            for (int i = 0; i < S; ++i)
                ref[(size_t) r * S + i] =
                    std::exp((double) bf2f(io[(size_t) r * S + i]) * scale - mx) / sum;
        }
        check(("softmax_scaled 6x" + std::to_string(S)).c_str(), dio.download(), ref, 1);
    }
}

void test_elementwise() {
    const int N = 4097;  // Deliberately not a multiple of any work-group size.

    auto a = rand_bf((size_t) N, -2.0f, 2.0f, 71u);
    auto b = rand_bf((size_t) N, -2.0f, 2.0f, 73u);

    {
        DevBuf<uint16_t> da(a), db(b), dout((size_t) N);
        bitvla_add_bf16(da.p, db.p, dout.p, N, nullptr);
        std::vector<double> ref((size_t) N);
        for (int i = 0; i < N; ++i) ref[i] = (double) bf2f(a[i]) + bf2f(b[i]);
        check("add 4097", dout.download(), ref, 1);
    }
    {
        DevBuf<uint16_t> da(a), db(b), dout((size_t) N);
        bitvla_squared_relu_mul_bf16(da.p, db.p, dout.p, N, nullptr);
        std::vector<double> ref((size_t) N);
        for (int i = 0; i < N; ++i) {
            double g = bf2f(a[i]);
            if (g < 0.0) g = 0.0;
            ref[i] = g * g * bf2f(b[i]);
        }
        check("squared_relu_mul 4097", dout.download(), ref, 1);
    }
    {
        DevBuf<uint16_t> da(a), dout((size_t) N);
        bitvla_gelu_tanh_bf16(da.p, dout.p, N, nullptr);
        std::vector<double> ref((size_t) N);
        for (int i = 0; i < N; ++i) {
            const double x  = bf2f(a[i]);
            const double kA = 0.7978845608028654, kB = 0.044715;
            ref[i] = 0.5 * x * (1.0 + std::tanh(kA * (x + kB * x * x * x)));
        }
        check("gelu_tanh 4097", dout.download(), ref, 1);
    }
}

void test_gate_up_fused() {
    const int seq = 5, ffn = 1301;

    auto             gu = rand_bf((size_t) seq * 2 * ffn, -2.0f, 2.0f, 83u);
    DevBuf<uint16_t> dgu(gu), dout((size_t) seq * ffn);
    gate_up_fused_sqrelu_mul_bf16(dgu.p, dout.p, seq, ffn, nullptr);

    std::vector<double> ref((size_t) seq * ffn);
    for (int s = 0; s < seq; ++s) {
        const size_t row = (size_t) s * 2 * ffn;
        for (int k = 0; k < ffn; ++k) {
            double g = bf2f(gu[row + k]);
            if (g < 0.0) g = 0.0;
            ref[(size_t) s * ffn + k] = g * g * bf2f(gu[row + ffn + k]);
        }
    }
    check("gate_up_fused 5x1301", dout.download(), ref, 1);
}

void test_add_bias() {
    const int M = 6, K = 777;

    auto x    = rand_bf((size_t) M * K, -1.0f, 1.0f, 89u);
    auto bias = rand_bf((size_t) K, -0.5f, 0.5f, 97u);

    DevBuf<uint16_t> dx(x), db(bias), dout((size_t) M * K);
    bitvla_add_bias_bf16(dx.p, db.p, dout.p, M, K, nullptr);

    std::vector<double> ref((size_t) M * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            ref[(size_t) m * K + k] = (double) bf2f(x[(size_t) m * K + k]) + bf2f(bias[k]);
    check("add_bias 6x777", dout.download(), ref, 1);
}

// The remaining four only move bf16 values around, so any deviation at all is
// an indexing bug rather than rounding: they are held to zero steps.

void test_zero_tail() {
    const int M = 4, cols = 300, start = 197;

    auto             x = rand_bf((size_t) M * cols, -1.0f, 1.0f, 101u);
    DevBuf<uint16_t> dx(x);
    bitvla_zero_tail_bf16(dx.p, M, cols, start, nullptr);

    std::vector<double> ref((size_t) M * cols);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < cols; ++k)
            ref[(size_t) m * cols + k] = k < start ? bf2f(x[(size_t) m * cols + k]) : 0.0;
    check("zero_tail 4x300@197", dx.download(), ref, 0);
}

void test_repeat_kv() {
    const int n_q = 8, n_kv = 2, seq = 7, hd = 96;

    auto             in = rand_bf((size_t) n_kv * seq * hd, -1.0f, 1.0f, 103u);
    DevBuf<uint16_t> din(in), dout((size_t) n_q * seq * hd);
    bitvla_repeat_kv_bf16(din.p, dout.p, n_q, n_kv, seq, hd, nullptr);

    std::vector<double> ref((size_t) n_q * seq * hd);
    for (int q = 0; q < n_q; ++q) {
        const int kv = q * n_kv / n_q;
        for (int s = 0; s < seq; ++s)
            for (int k = 0; k < hd; ++k)
                ref[((size_t) q * seq + s) * hd + k] = bf2f(in[((size_t) kv * seq + s) * hd + k]);
    }
    check("repeat_kv 2->8 x7x96", dout.download(), ref, 0);
}

void test_transposes() {
    const int S = 11, N = 5, hd = 70;
    auto      in = rand_bf((size_t) S * N * hd, -1.0f, 1.0f, 107u);

    DevBuf<uint16_t> din(in), dmid((size_t) S * N * hd), dback((size_t) S * N * hd);
    bitvla_transpose_sNhd_to_NshHd_bf16(din.p, dmid.p, S, N, hd, nullptr);

    std::vector<double> ref((size_t) S * N * hd);
    for (int h = 0; h < N; ++h)
        for (int s = 0; s < S; ++s)
            for (int k = 0; k < hd; ++k)
                ref[((size_t) h * S + s) * hd + k] = bf2f(in[((size_t) s * N + h) * hd + k]);
    check("transpose sNhd->NShd 11x5x70", dmid.download(), ref, 0);

    // Round-tripping is the sharper test of the inverse: it has to land back on
    // the original bits, not merely on something self-consistent.
    bitvla_transpose_NshHd_to_sNhd_bf16(dmid.p, dback.p, N, S, hd, nullptr);
    std::vector<double> orig((size_t) S * N * hd);
    for (size_t i = 0; i < orig.size(); ++i) orig[i] = bf2f(in[i]);
    check("transpose NShd->sNhd roundtrip", dback.download(), orig, 0);
}

void test_gather_rows() {
    const int rows = 9, n_out = 6, K = 130;

    auto                 in = rand_bf((size_t) rows * K, -1.0f, 1.0f, 109u);
    std::vector<int32_t> ids{8, 0, 3, 3, 7, 1};

    DevBuf<uint16_t> din(in), dout((size_t) n_out * K);
    DevBuf<int32_t>  dids(ids);
    bitvla_gather_rows_bf16(din.p, dout.p, dids.p, n_out, K, nullptr);

    std::vector<double> ref((size_t) n_out * K);
    for (int m = 0; m < n_out; ++m)
        for (int k = 0; k < K; ++k)
            ref[(size_t) m * K + k] = bf2f(in[(size_t) ids[(size_t) m] * K + k]);
    check("gather_rows 6 of 9 x130", dout.download(), ref, 0);
}

}  // namespace

/**
 * @brief The padded quantiser must equal restride-then-quantise, exactly.
 *
 * Unlike every other case in this file, the reference here is not an exact
 * host replay - it is the code path this one replaced. The ViT's fc2 wants K
 * padded to a multiple of 128, and fc1's output used to reach it via a 2D copy
 * into a zero-tailed bf16 buffer which was then quantised at the padded width.
 * That copy was 4.4 ms of a 27.2 ms request on B70, because SYCL has no native
 * 2D copy over Level Zero and fell back to a byte-granularity kernel; the
 * quantiser now widens the row itself and the copy is gone.
 *
 * The claim made for that change is bit-identity, not closeness, so this asserts
 * bit-identity: same int8 bytes across the full padded width, same float scales.
 * It holds for a reason worth stating, because it is what makes the change safe
 * rather than merely fast - the scale is a row absmax and the appended tail is
 * zeros, so the reduction sees the same maximum (max is exactly associative, so
 * even the differing reduction lengths cannot perturb it), every real column
 * keeps the scale it had, and the padding quantises to the 0 it was memset to.
 * The ternary GEMM downstream is exact and would deliver any drift here intact
 * to the action vector.
 */
void test_act_quant_pad() {
    // The first shape is the live one: BitSigLIP's ffn is 4304 and ffn_pad 4352.
    // The second has K_in == K_out, which is the path every other call takes and
    // must be undisturbed. The third pads a row that is not a multiple of the
    // work-group, where an off-by-one in the tail loop would show.
    const int shapes[3][3] = {{256, 4304, 4352}, {5, 1152, 1152}, {3, 197, 300}};

    for (const auto & shape : shapes) {
        const int M = shape[0], K_in = shape[1], K_out = shape[2];

        auto x = rand_bf((size_t) M * K_in, -4.0f, 4.0f, 97u);
        DevBuf<uint16_t> dx(x);
        DevBuf<int8_t>   q_old((size_t) M * K_out), q_new((size_t) M * K_out);
        DevBuf<float>    s_old((size_t) M),         s_new((size_t) M);

        // Old path: zeroed padded buffer, 2D restride, quantise at K_out.
        DevBuf<uint16_t> dpad((size_t) M * K_out);
        vla_dev_memset(dpad.p, 0, (size_t) M * K_out * sizeof(uint16_t));
        vla_dev_memcpy2d_d2d(dpad.p, (size_t) K_out * sizeof(uint16_t),
                             dx.p,   (size_t) K_in  * sizeof(uint16_t),
                             (size_t) K_in * sizeof(uint16_t), (size_t) M, nullptr);
        bitvla_act_quant_cuda(dpad.p, q_old.p, s_old.p, M, K_out, nullptr);

        // New path: one kernel, no copy, no padded bf16 buffer.
        bitvla_act_quant_pad_cuda(dx.p, q_new.p, s_new.p, M, K_in, K_out, nullptr);

        const auto a = q_old.download(), b = q_new.download();
        const auto sa = s_old.download(), sb = s_new.download();

        int64_t bad_q = 0, bad_s = 0;
        size_t  first  = 0;
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) { if (!bad_q) first = i; ++bad_q; }
        for (size_t i = 0; i < sa.size(); ++i)
            if (std::memcmp(&sa[i], &sb[i], sizeof(float)) != 0) ++bad_s;

        char name[64];
        std::snprintf(name, sizeof(name), "act_quant_pad %dx%d@%d", M, K_in, K_out);
        const bool ok = (bad_q == 0 && bad_s == 0);
        std::printf("  %-34s %-4s %lld int8, %lld scale(s) differ\n", name, ok ? "OK" : "FAIL",
                    (long long) bad_q, (long long) bad_s);
        if (!ok) {
            ++g_failures;
            std::printf("      first int8 at [%zu]: old %d, new %d\n", first,
                        (int) a[first], (int) b[first]);
        }
    }
}


#ifdef VLA_BITVLA_FUSED_OPS
/// Byte-compare two downloads and record the result. The fused kernels claim
/// bit-identity with the chains they replace, so that is the gate.
template <typename T>
int64_t n_differ(const std::vector<T> & a, const std::vector<T> & b) {
    int64_t n = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) ++n;
    return n;
}

void report_fused(const char * name, int64_t bad_q, int64_t bad_s, int64_t bad_h) {
    const bool ok = bad_q == 0 && bad_s == 0 && bad_h == 0;
    std::printf("  %-34s %-4s %lld int8, %lld scale(s), %lld bf16 differ\n", name,
                ok ? "OK" : "FAIL", (long long) bad_q, (long long) bad_s, (long long) bad_h);
    if (!ok) ++g_failures;
}

/// add + rmsnorm + act_quant against the three unfused kernels, with and
/// without the residual, at the LM's widths (hidden 2560) and an odd one.
void test_fused_add_rmsnorm_quant() {
    const int shapes[3][2] = {{330, 2560}, {7, 1301}, {330, 6912}};
    for (const auto & sh : shapes) {
        const int M = sh[0], K = sh[1];
        for (int with_delta = 0; with_delta < 2; ++with_delta) {
            const auto h0 = rand_bf((size_t) M * K, -6.0f, 6.0f, 11u + K);
            const auto d0 = rand_bf((size_t) M * K, -3.0f, 3.0f, 12u + K);
            const auto w0 = rand_bf((size_t) K, 0.2f, 1.8f, 13u + K);
            DevBuf<uint16_t> h_a(h0), h_b(h0), d(d0), w(w0), hn((size_t) M * K);
            DevBuf<int8_t>   q_a((size_t) M * K), q_b((size_t) M * K);
            DevBuf<float>    s_a((size_t) M), s_b((size_t) M);

            if (with_delta) bitvla_add_bf16(h_a.p, d.p, h_a.p, M * K, nullptr);
            bitvla_rmsnorm_bf16(h_a.p, w.p, hn.p, 1e-6f, M, K, nullptr);
            bitvla_act_quant_cuda(hn.p, q_a.p, s_a.p, M, K, nullptr);

            bitvla_add_rmsnorm_quant_bf16(h_b.p, with_delta ? d.p : nullptr, w.p, q_b.p, s_b.p,
                                          1e-6f, M, K, nullptr);

            char name[64];
            std::snprintf(name, sizeof(name), "fused %srmsnorm_quant %dx%d",
                          with_delta ? "add_" : "", M, K);
            report_fused(name, n_differ(q_a.download(), q_b.download()),
                         n_differ(s_a.download(), s_b.download()),
                         n_differ(h_a.download(), h_b.download()));
        }
    }
}

void test_fused_sqrelu_rmsnorm_quant() {
    const int shapes[2][2] = {{330, 6912}, {5, 1301}};
    for (const auto & sh : shapes) {
        const int seq = sh[0], ffn = sh[1];
        const auto gu0 = rand_bf((size_t) seq * 2 * ffn, -3.0f, 3.0f, 21u + ffn);
        const auto w0  = rand_bf((size_t) ffn, 0.2f, 1.8f, 22u + ffn);
        DevBuf<uint16_t> gu(gu0), w(w0), mid((size_t) seq * ffn);
        DevBuf<int8_t>   q_a((size_t) seq * ffn), q_b((size_t) seq * ffn);
        DevBuf<float>    s_a((size_t) seq), s_b((size_t) seq);

        gate_up_fused_sqrelu_mul_bf16(gu.p, mid.p, seq, ffn, nullptr);
        bitvla_rmsnorm_bf16(mid.p, w.p, mid.p, 1e-6f, seq, ffn, nullptr);
        bitvla_act_quant_cuda(mid.p, q_a.p, s_a.p, seq, ffn, nullptr);

        bitvla_sqrelu_rmsnorm_quant_bf16(gu.p, w.p, q_b.p, s_b.p, 1e-6f, seq, ffn, nullptr);

        char name[64];
        std::snprintf(name, sizeof(name), "fused sqrelu_rmsnorm_quant %dx%d", seq, ffn);
        report_fused(name, n_differ(q_a.download(), q_b.download()),
                     n_differ(s_a.download(), s_b.download()), 0);
    }
}

void test_fused_add_layernorm_quant() {
    const int shapes[2][2] = {{256, 1152}, {3, 777}};
    for (const auto & sh : shapes) {
        const int M = sh[0], K = sh[1];
        for (int with_delta = 0; with_delta < 2; ++with_delta) {
            const auto h0  = rand_bf((size_t) M * K, -6.0f, 6.0f, 31u + K);
            const auto d0  = rand_bf((size_t) M * K, -3.0f, 3.0f, 32u + K);
            const auto db0 = rand_bf((size_t) K, -1.0f, 1.0f, 33u + K);
            const auto w0  = rand_bf((size_t) K, 0.2f, 1.8f, 34u + K);
            const auto b0  = rand_bf((size_t) K, -0.5f, 0.5f, 35u + K);
            DevBuf<uint16_t> h_a(h0), h_b(h0), d(d0), d_a(d0), db(db0), w(w0), b(b0),
                hn((size_t) M * K);
            DevBuf<int8_t> q_a((size_t) M * K), q_b((size_t) M * K);
            DevBuf<float>  s_a((size_t) M), s_b((size_t) M);

            if (with_delta) {
                bitvla_add_bias_bf16(d_a.p, db.p, d_a.p, M, K, nullptr);
                bitvla_add_bf16(h_a.p, d_a.p, h_a.p, M * K, nullptr);
            }
            bitvla_layernorm_bf16(h_a.p, w.p, b.p, hn.p, 1e-6f, M, K, nullptr);
            bitvla_act_quant_cuda(hn.p, q_a.p, s_a.p, M, K, nullptr);

            bitvla_add_layernorm_quant_bf16(h_b.p, with_delta ? d.p : nullptr,
                                            with_delta ? db.p : nullptr, w.p, b.p, q_b.p, s_b.p,
                                            1e-6f, M, K, nullptr);

            char name[64];
            std::snprintf(name, sizeof(name), "fused %slayernorm_quant %dx%d",
                          with_delta ? "add_" : "", M, K);
            report_fused(name, n_differ(q_a.download(), q_b.download()),
                         n_differ(s_a.download(), s_b.download()),
                         n_differ(h_a.download(), h_b.download()));
        }
    }
}

void test_fused_bias_gelu_quant_pad() {
    const int shapes[2][3] = {{256, 4304, 4352}, {3, 197, 300}};
    for (const auto & sh : shapes) {
        const int M = sh[0], K_in = sh[1], K_out = sh[2];
        const auto x0 = rand_bf((size_t) M * K_in, -4.0f, 4.0f, 41u + K_in);
        const auto b0 = rand_bf((size_t) K_in, -1.0f, 1.0f, 42u + K_in);
        DevBuf<uint16_t> x(x0), b(b0);
        DevBuf<int8_t>   q_a((size_t) M * K_out), q_b((size_t) M * K_out);
        DevBuf<float>    s_a((size_t) M), s_b((size_t) M);

        bitvla_add_bias_bf16(x.p, b.p, x.p, M, K_in, nullptr);
        bitvla_gelu_tanh_bf16(x.p, x.p, M * K_in, nullptr);
        bitvla_act_quant_pad_cuda(x.p, q_a.p, s_a.p, M, K_in, K_out, nullptr);

        x.upload(x0);
        bitvla_bias_gelu_quant_pad_bf16(x.p, b.p, q_b.p, s_b.p, M, K_in, K_out, nullptr);

        char name[64];
        std::snprintf(name, sizeof(name), "fused bias_gelu_quant %dx%d@%d", M, K_in, K_out);
        report_fused(name, n_differ(q_a.download(), q_b.download()),
                     n_differ(s_a.download(), s_b.download()), 0);
    }
}

void test_fused_bias_residual() {
    const int M = 256, K = 1152;
    const auto h0 = rand_bf((size_t) M * K, -6.0f, 6.0f, 51u);
    const auto d0 = rand_bf((size_t) M * K, -3.0f, 3.0f, 52u);
    const auto b0 = rand_bf((size_t) K, -1.0f, 1.0f, 53u);
    DevBuf<uint16_t> h_a(h0), h_b(h0), d(d0), d_a(d0), b(b0);

    bitvla_add_bias_bf16(d_a.p, b.p, d_a.p, M, K, nullptr);
    bitvla_add_bf16(h_a.p, d_a.p, h_a.p, M * K, nullptr);
    bitvla_bias_residual_bf16(h_b.p, d.p, b.p, M, K, nullptr);
    report_fused("fused bias_residual 256x1152", 0, 0, n_differ(h_a.download(), h_b.download()));
}

/// Row-major Q/K RoPE against transpose-to-head-major, rope, transpose back.
void test_fused_rope_qk_rows() {
    const int S = 37, n_q = 20, n_kv = 5, hd = 128, half = hd / 2;
    std::vector<float> cs((size_t) S * half), sn((size_t) S * half);
    for (int s = 0; s < S; ++s)
        for (int k = 0; k < half; ++k) {
            const float ang = (float) s / std::pow(500000.0f, (float) (2 * k) / hd);
            cs[(size_t) s * half + k] = std::cos(ang);
            sn[(size_t) s * half + k] = std::sin(ang);
        }
    DevBuf<float> dcs(cs), dsn(sn);
    const auto q0 = rand_bf((size_t) S * n_q * hd, -2.0f, 2.0f, 61u);
    const auto k0 = rand_bf((size_t) S * n_kv * hd, -2.0f, 2.0f, 62u);
    DevBuf<uint16_t> qa(q0), ka(k0), qb(q0), kb(k0), qh(q0.size()), kh(k0.size());

    bitvla_transpose_sNhd_to_NshHd_bf16(qa.p, qh.p, S, n_q, hd, nullptr);
    bitvla_transpose_sNhd_to_NshHd_bf16(ka.p, kh.p, S, n_kv, hd, nullptr);
    bitvla_rope_neox_bf16(qh.p, dcs.p, dsn.p, n_q, S, hd, nullptr);
    bitvla_rope_neox_bf16(kh.p, dcs.p, dsn.p, n_kv, S, hd, nullptr);
    bitvla_transpose_NshHd_to_sNhd_bf16(qh.p, qa.p, n_q, S, hd, nullptr);
    bitvla_transpose_NshHd_to_sNhd_bf16(kh.p, ka.p, n_kv, S, hd, nullptr);

    bitvla_rope_neox_qk_rows_bf16(qb.p, kb.p, dcs.p, dsn.p, S, n_q, n_kv, hd, nullptr);
    report_fused("fused rope_qk_rows 37x(20+5)x128", 0, 0,
                 n_differ(qa.download(), qb.download()) + n_differ(ka.download(), kb.download()));
}

/**
 * @brief Fused attention against an exact host evaluation, next to the
 *        unfused chain it replaces scored against the same reference.
 *
 * The fused kernel keeps scores and softmax in f32 where the chain rounds both
 * to bf16, so it is not bit-identical and is not meant to be; the claim is that
 * it is no less accurate. Both are scored by absolute error against the exact
 * output (bf16 steps are the wrong ruler here: attention outputs straddle zero,
 * where one step is a vanishing magnitude), and the fused path must be no worse
 * than the chain in RMS or at the worst element.
 */
void test_fused_attention() {
    struct Case { int S, n_q, n_kv, hd; };
    const Case cases[] = {{330, 20, 5, 128}, {256, 16, 16, 72}};
    for (const auto & c : cases) {
        const int S = c.S, n_q = c.n_q, n_kv = c.n_kv, hd = c.hd, rep = n_q / n_kv;
        const int hq = n_q * hd, hkv = n_kv * hd;
        const auto q0 = rand_bf((size_t) S * hq, -2.0f, 2.0f, 71u + S);
        const auto k0 = rand_bf((size_t) S * hkv, -2.0f, 2.0f, 72u + S);
        const auto v0 = rand_bf((size_t) S * hkv, -2.0f, 2.0f, 73u + S);
        const float scale = 1.0f / std::sqrt((float) hd);

        // Exact reference in double, from the same bf16 inputs.
        std::vector<double> exact((size_t) S * hq);
        std::vector<double> p(S);
        for (int h = 0; h < n_q; ++h) {
            const int g = h / rep;
            for (int i = 0; i < S; ++i) {
                double mx = -1e300;
                for (int j = 0; j < S; ++j) {
                    double d = 0;
                    for (int t = 0; t < hd; ++t)
                        d += (double) bf2f(q0[(size_t) i * hq + h * hd + t]) *
                             (double) bf2f(k0[(size_t) j * hkv + g * hd + t]);
                    p[j] = d * scale;
                    if (p[j] > mx) mx = p[j];
                }
                double sum = 0;
                for (int j = 0; j < S; ++j) { p[j] = std::exp(p[j] - mx); sum += p[j]; }
                for (int t = 0; t < hd; ++t) {
                    double o = 0;
                    for (int j = 0; j < S; ++j) o += p[j] * (double) bf2f(v0[(size_t) j * hkv + g * hd + t]);
                    exact[(size_t) i * hq + h * hd + t] = o / sum;
                }
            }
        }

        DevBuf<uint16_t> dq(q0), dk(k0), dv(v0), dout((size_t) S * hq);
        if (vla_attention_bf16(dq.p, dk.p, dv.p, dout.p, S, n_q, n_kv, hd, hq, hkv, hq, scale,
                               nullptr) != 0) {
            std::printf("  fused attention S=%d unavailable\n", S);
            ++g_failures;
            continue;
        }
        const auto fused = dout.download();

        // The unfused chain, exactly as the drivers issue it.
        DevBuf<uint16_t> qh((size_t) S * hq), kh((size_t) S * hkv), vh((size_t) S * hkv),
            kr((size_t) S * hq), vr((size_t) S * hq), sc((size_t) n_q * S * S),
            ao((size_t) S * hq), merged((size_t) S * hq);
        bitvla_transpose_sNhd_to_NshHd_bf16(dq.p, qh.p, S, n_q, hd, nullptr);
        bitvla_transpose_sNhd_to_NshHd_bf16(dk.p, kh.p, S, n_kv, hd, nullptr);
        bitvla_transpose_sNhd_to_NshHd_bf16(dv.p, vh.p, S, n_kv, hd, nullptr);
        bitvla_repeat_kv_bf16(kh.p, kr.p, n_q, n_kv, S, hd, nullptr);
        bitvla_repeat_kv_bf16(vh.p, vr.p, n_q, n_kv, S, hd, nullptr);
        vla_gemm_bf16_nt_batched(qh.p, kr.p, sc.p, S, S, hd, n_q, (long long) S * hd,
                                 (long long) S * hd, (long long) S * S, nullptr);
        bitvla_softmax_scaled_bf16(sc.p, scale, n_q * S, S, nullptr);
        vla_gemm_bf16_nn_batched(sc.p, vr.p, ao.p, S, hd, S, n_q, (long long) S * S,
                                 (long long) S * hd, (long long) S * hd, nullptr);
        bitvla_transpose_NshHd_to_sNhd_bf16(ao.p, merged.p, n_q, S, hd, nullptr);
        const auto chain = merged.download();

        double worst_f = 0, worst_c = 0, sq_f = 0, sq_c = 0;
        for (size_t i = 0; i < exact.size(); ++i) {
            const double df = std::fabs(bf2f(fused[i]) - exact[i]);
            const double dc = std::fabs(bf2f(chain[i]) - exact[i]);
            worst_f = std::max(worst_f, df);
            worst_c = std::max(worst_c, dc);
            sq_f += df * df;
            sq_c += dc * dc;
        }
        const double rms_f = std::sqrt(sq_f / exact.size()), rms_c = std::sqrt(sq_c / exact.size());
        const bool   ok    = rms_f <= rms_c && worst_f <= worst_c;
        char name[64];
        std::snprintf(name, sizeof(name), "fused attention %d %dq/%dkv x%d", S, n_q, n_kv, hd);
        std::printf("  %-34s %-4s max|err| %.3g (chain %.3g), rms %.3g (chain %.3g)\n", name,
                    ok ? "OK" : "FAIL", worst_f, worst_c, rms_f, rms_c);
        if (!ok) ++g_failures;
    }
}
#endif


int main() {
    const int n_dev = vla_dev_count();
    if (n_dev <= 0) {
        std::printf("no GPU device (%s) - skipping\n", vla_dev_error());
        return 0;
    }
    if (vla_dev_set(0) != 0) {
        std::printf("vla_dev_set(0) failed (%s) - skipping\n", vla_dev_error());
        return 0;
    }
    std::printf("bitvla bf16 ops, %d device(s), gate is bf16 steps from the exact value\n", n_dev);

    test_rmsnorm();
    test_layernorm();
    test_rope();
    test_softmax();
    test_elementwise();
    test_gate_up_fused();
    test_add_bias();
    test_zero_tail();
    test_repeat_kv();
    test_transposes();
    test_gather_rows();
    test_act_quant_pad();
#ifdef VLA_BITVLA_FUSED_OPS
    test_fused_add_rmsnorm_quant();
    test_fused_sqrelu_rmsnorm_quant();
    test_fused_add_layernorm_quant();
    test_fused_bias_gelu_quant_pad();
    test_fused_bias_residual();
    test_fused_rope_qk_rows();
    test_fused_attention();
#endif

    if (g_failures) {
        std::printf("FAILED: %d op(s)\n", g_failures);
        return 1;
    }
    std::printf("all ops within tolerance\n");
    return 0;
}
