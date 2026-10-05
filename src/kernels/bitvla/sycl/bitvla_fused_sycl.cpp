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
 * @file bitvla_fused_sycl.cpp
 * @brief SYCL implementation of @c bitvla_fused.h.
 *
 * ## Why these exist
 *
 * On Panther Lake's Xe3 iGPU a unitrace of one BitVLA request put softmax,
 * act_quant, rmsnorm, the residual adds, the bias adds and the FFN gate at
 * ~40% of device time between them, and every one of those kernels was moving
 * bytes at 75-83 GB/s - the measured LPDDR5x read ceiling of the part. They
 * were not slow kernels, they were kernels doing nothing but round-tripping a
 * bf16 plane through DRAM so the next kernel could read it. Fusing the chains
 * removes the round trips; the row is re-read from L1/L2 instead, since a row
 * is at most 2*6912 bf16 = 27 KB.
 *
 * ## Why they are bit-identical to the chains they replace
 *
 * Each kernel recomputes the unfused intermediate - rounded to bf16 at the
 * same point - every time it needs it, rather than carrying an unrounded f32
 * forward. The reductions keep the unfused partition, the same group size and
 * the same @c reduce_over_group, and the sub-group width is pinned to the 32
 * the unfused kernels compiled to (a different width is a different combining
 * tree, and therefore different bits). The f32 expressions are copied from
 * @c bitvla_ops_sycl.cpp and @c bitnet_sycl.cpp token for token, so that
 * contraction into FMAs - which @c -fp-model=precise still permits - happens
 * the same way in both.
 *
 * Recomputation is cheaper than it sounds: the work per element is a handful
 * of FLOPs against a DRAM read the unfused chain paid two to four times.
 */

#include "kernels/bitvla/bitvla_fused.h"
#include "kernels/bitvla/sycl/queue_sycl.h"
#include "kernels/bitvla/sycl/sycl_compat.h"

using vla::bitvla::as_bf;
using vla::bitvla::bf16;
using vla::bitvla::exact_div;
using vla::bitvla::ROW_WG;
using vla::bitvla::to_bf16;
using vla::bitvla::to_f32;

namespace {

struct k_add_rmsnorm_quant {};
struct k_sqrelu_rmsnorm_quant {};
struct k_add_layernorm_quant {};
struct k_bias_gelu_quant_pad {};
struct k_bias_residual {};
struct k_rope_qk_rows {};

/// The width the unfused row kernels compiled to (unitrace: SIMD 32). Pinned
/// so the reduction trees, and with them the bits, match.
constexpr int ROW_SG = 32;

inline sycl::nd_range<2> rows_range(int rows) {
    return sycl::nd_range<2>(sycl::range<2>((size_t) rows, (size_t) ROW_WG),
                             sycl::range<2>(1, (size_t) ROW_WG));
}

/// act_quant's tail, shared: absmax over a row of bf16 values produced by
/// @p val(k), then the scale and the rounded, clamped int8 - the expressions of
/// @c bitvla_act_quant_pad_cuda verbatim.
template <typename Val>
inline void quant_row(sycl::nd_item<2> it, int tid, int K, int K_out, Val val,
                      int8_t * row_out, float * scale_out) {
    float local_max = 0.0f;
    for (int k = tid; k < K; k += ROW_WG) {
        const float v = sycl::fabs(to_f32(val(k)));
        if (v > local_max) local_max = v;
    }
    const float row_max = sycl::reduce_over_group(it.get_group(), local_max, sycl::maximum<float>());

    const float amax  = row_max < 1e-5f ? 1e-5f : row_max;
    const float scale = exact_div(127.0f, amax);
    if (tid == 0) *scale_out = scale;

    for (int k = tid; k < K; k += ROW_WG) {
        float q_ = sycl::rint(to_f32(val(k)) * scale);
        if (q_ > 127.0f) q_ = 127.0f;
        if (q_ < -128.0f) q_ = -128.0f;
        row_out[k] = (int8_t) q_;
    }
    for (int k = K + tid; k < K_out; k += ROW_WG)
        row_out[k] = 0;
}

}  // namespace

extern "C" void bitvla_add_rmsnorm_quant_bf16(vla_bf16* h, const vla_bf16* delta,
                                              const vla_bf16* w, int8_t* out, float* scales,
                                              float eps, int M, int K, vla_stream stream) {
    if (M <= 0 || K <= 0) return;
    sycl::queue & q  = vla::bitvla_sycl_queue(stream);
    bf16 *        hs = as_bf(h);
    const bf16 *  ds = as_bf(delta);
    const bf16 *  ws = as_bf(w);
    q.parallel_for<k_add_rmsnorm_quant>(
        rows_range(M), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(ROW_SG)]] {
            const int m   = (int) it.get_group(0);
            const int tid = (int) it.get_local_id(1);
            bf16 *    row = hs + (size_t) m * K;

            // Residual add, written back: each work-item only ever re-reads the
            // elements it wrote itself, so no barrier is needed before reuse.
            float ss = 0.0f;
            if (ds) {
                const bf16 * drow = ds + (size_t) m * K;
                for (int k = tid; k < K; k += ROW_WG) {
                    const bf16 s = to_bf16(to_f32(row[k]) + to_f32(drow[k]));
                    row[k]       = s;
                    float v      = to_f32(s);
                    ss += v * v;
                }
            } else {
                for (int k = tid; k < K; k += ROW_WG) {
                    float v = to_f32(row[k]);
                    ss += v * v;
                }
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());

            const float mean  = ss / (float) K;
            const float scale = sycl::rsqrt(mean + eps);

            auto normed = [&](int k) {
                float v  = to_f32(row[k]) * scale;
                float wv = to_f32(ws[k]);
                return to_bf16(v * wv);
            };
            quant_row(it, tid, K, K, normed, out + (size_t) m * K, scales + m);
        });
}

extern "C" void bitvla_sqrelu_rmsnorm_quant_bf16(const vla_bf16* gu, const vla_bf16* w,
                                                 int8_t* out, float* scales, float eps,
                                                 int seq, int ffn, vla_stream stream) {
    if (seq <= 0 || ffn <= 0) return;
    sycl::queue & q   = vla::bitvla_sycl_queue(stream);
    const bf16 *  gus = as_bf(gu);
    const bf16 *  ws  = as_bf(w);
    const int     K   = ffn;
    q.parallel_for<k_sqrelu_rmsnorm_quant>(
        rows_range(seq), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(ROW_SG)]] {
            const int    m   = (int) it.get_group(0);
            const int    tid = (int) it.get_local_id(1);
            const bf16 * g_r = gus + (size_t) m * 2 * K;
            const bf16 * u_r = g_r + K;

            // gate_up_fused_sqrelu_mul_bf16's element, rounded where it rounded.
            auto gated = [&](int k) {
                float g = to_f32(g_r[k]);
                float u = to_f32(u_r[k]);
                if (g < 0.0f) g = 0.0f;
                return to_bf16(g * g * u);
            };

            float ss = 0.0f;
            for (int k = tid; k < K; k += ROW_WG) {
                float v = to_f32(gated(k));
                ss += v * v;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());

            const float mean  = ss / (float) K;
            const float scale = sycl::rsqrt(mean + eps);

            auto normed = [&](int k) {
                float v  = to_f32(gated(k)) * scale;
                float wv = to_f32(ws[k]);
                return to_bf16(v * wv);
            };
            quant_row(it, tid, K, K, normed, out + (size_t) m * K, scales + m);
        });
}

extern "C" void bitvla_add_layernorm_quant_bf16(vla_bf16* h, const vla_bf16* delta,
                                                const vla_bf16* delta_bias,
                                                const vla_bf16* w, const vla_bf16* b,
                                                int8_t* out, float* scales, float eps,
                                                int M, int K, vla_stream stream) {
    if (M <= 0 || K <= 0) return;
    sycl::queue & q   = vla::bitvla_sycl_queue(stream);
    bf16 *        hs  = as_bf(h);
    const bf16 *  ds  = as_bf(delta);
    const bf16 *  dbs = as_bf(delta_bias);
    const bf16 *  ws  = as_bf(w);
    const bf16 *  bs  = as_bf(b);
    q.parallel_for<k_add_layernorm_quant>(
        rows_range(M), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(ROW_SG)]] {
            const int m   = (int) it.get_group(0);
            const int tid = (int) it.get_local_id(1);
            bf16 *    row = hs + (size_t) m * K;

            float sum = 0.0f;
            if (ds) {
                const bf16 * drow = ds + (size_t) m * K;
                for (int k = tid; k < K; k += ROW_WG) {
                    // add_bias on the projection, then the residual add.
                    const bf16 d = dbs ? to_bf16(to_f32(drow[k]) + to_f32(dbs[k])) : drow[k];
                    const bf16 s = to_bf16(to_f32(row[k]) + to_f32(d));
                    row[k]       = s;
                    sum += to_f32(s);
                }
            } else {
                for (int k = tid; k < K; k += ROW_WG)
                    sum += to_f32(row[k]);
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            const float mean = sum / (float) K;

            float vsum = 0.0f;
            for (int k = tid; k < K; k += ROW_WG) {
                float v = to_f32(row[k]) - mean;
                vsum += v * v;
            }
            vsum = sycl::reduce_over_group(it.get_group(), vsum, sycl::plus<float>());
            const float inv_std = sycl::rsqrt(vsum / (float) K + eps);

            auto normed = [&](int k) {
                float v  = (to_f32(row[k]) - mean) * inv_std;
                float wv = to_f32(ws[k]);
                float bv = to_f32(bs[k]);
                return to_bf16(v * wv + bv);
            };
            quant_row(it, tid, K, K, normed, out + (size_t) m * K, scales + m);
        });
}

extern "C" void bitvla_bias_gelu_quant_pad_bf16(const vla_bf16* x, const vla_bf16* bias,
                                                int8_t* out, float* scales,
                                                int M, int K_in, int K_out, vla_stream stream) {
    if (M <= 0 || K_in <= 0 || K_out < K_in) return;
    sycl::queue & q  = vla::bitvla_sycl_queue(stream);
    const bf16 *  xs = as_bf(x);
    const bf16 *  bs = as_bf(bias);
    q.parallel_for<k_bias_gelu_quant_pad>(
        rows_range(M), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(ROW_SG)]] {
            const int    m   = (int) it.get_group(0);
            const int    tid = (int) it.get_local_id(1);
            const bf16 * row = xs + (size_t) m * K_in;

            // add_bias then bitvla_gelu_tanh_bf16, each rounded as it was.
            auto act = [&](int k) {
                const float x_ = to_f32(to_bf16(to_f32(row[k]) + to_f32(bs[k])));

                const float kAlpha = 0.7978845608028654f;
                const float kBeta  = 0.044715f;
                float       u      = kAlpha * (x_ + kBeta * x_ * x_ * x_);
                float       t      = sycl::tanh(u);
                return to_bf16(0.5f * x_ * (1.0f + t));
            };
            quant_row(it, tid, K_in, K_out, act, out + (size_t) m * K_out, scales + m);
        });
}

extern "C" void bitvla_bias_residual_bf16(vla_bf16* h, const vla_bf16* delta,
                                          const vla_bf16* bias, int M, int K,
                                          vla_stream stream) {
    if (M <= 0 || K <= 0) return;
    constexpr int WG = 256;
    sycl::queue & q  = vla::bitvla_sycl_queue(stream);
    bf16 *        hs = as_bf(h);
    const bf16 *  ds = as_bf(delta);
    const bf16 *  bs = as_bf(bias);
    q.parallel_for<k_bias_residual>(
        sycl::nd_range<2>(sycl::range<2>((size_t) M, vla::bitvla::round_up((size_t) K, WG)),
                          sycl::range<2>(1, WG)),
        [=](sycl::nd_item<2> it) {
            const int m = (int) it.get_group(0);
            const int k = (int) it.get_global_id(1);
            if (k >= K) return;
            const size_t i = (size_t) m * K + k;
            const bf16   d = to_bf16(to_f32(ds[i]) + to_f32(bs[k]));
            hs[i]          = to_bf16(to_f32(hs[i]) + to_f32(d));
        });
}

extern "C" void bitvla_rope_neox_qk_rows_bf16(vla_bf16* q, vla_bf16* k, const float* cos_tab,
                                              const float* sin_tab, int S, int n_q, int n_kv,
                                              int hd, vla_stream stream) {
    if (S <= 0 || hd <= 1) return;
    const int     half    = hd / 2;
    const int     n_heads = n_q + n_kv;
    sycl::queue & q_      = vla::bitvla_sycl_queue(stream);
    bf16 *        qs      = as_bf(q);
    bf16 *        ks      = as_bf(k);
    // One work-item per rotated pair: (s, head, k < half). Heads [0, n_q) are
    // Q's, the rest K's.
    const size_t total = (size_t) S * n_heads * half;
    constexpr int WG   = 256;
    q_.parallel_for<k_rope_qk_rows>(
        sycl::nd_range<1>(vla::bitvla::round_up(total, WG), WG), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            if (i >= total) return;
            const int kk = (int) (i % half);
            const int hh = (int) ((i / half) % n_heads);
            const int s  = (int) (i / ((size_t) half * n_heads));

            bf16 * row = hh < n_q ? qs + ((size_t) s * n_q + hh) * hd
                                  : ks + ((size_t) s * n_kv + (hh - n_q)) * hd;
            const float c  = cos_tab[(size_t) s * half + kk];
            const float si = sin_tab[(size_t) s * half + kk];
            const float a  = to_f32(row[kk]);
            const float b  = to_f32(row[kk + half]);
            row[kk]        = to_bf16(a * c - b * si);
            row[kk + half] = to_bf16(b * c + a * si);
        });
}
