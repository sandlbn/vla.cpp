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
 * @file bitvla_ops_sycl.cpp
 * @brief SYCL twins of the bf16 elementwise/norm/RoPE/softmax ops declared in
 *        @c kernels/bitvla/bitvla_lm_cuda.h.
 *
 * Same @c extern @c "C" symbols as @c bitvla_lm_cuda.cu, so the LM and ViT
 * drivers call one set of names and the link picks the backend. The ops are
 * shared between the two towers, which is why they live apart from either
 * driver.
 *
 * ## Reproducing the arithmetic
 *
 * The precision contract for these ops is a tolerance, not bit-identity: the
 * CUDA build compiles with @c --use_fast_math, so its @c expf / @c tanhf /
 * division are already approximations, and there is nothing to be bit-equal
 * *to*. What the port owes is that its deviation from an exact evaluation is no
 * larger than CUDA's. Three rules get that:
 *
 *   1. The f32 expression is copied literally - same operand order, same
 *      constants, same clamp-then-square rather than square-then-clamp.
 *   2. No @c -ffast-math on this translation unit. It is a global switch over
 *      denormals and NaNs, and the CUDA side only ever opted individual
 *      functions in. @c sycl::exp / @c tanh / @c rsqrt are the correctly-rounded
 *      library calls, which is the *tighter* side of the contract.
 *   3. Every store rounds through @c to_bf16, which is RNE on both backends.
 *
 * The one structural difference is the reductions. CUDA hand-rolls a 32-lane
 * shuffle plus a shared-memory stage; here @c reduce_over_group does both, and
 * on a sub-group of 16 or 32 it will not combine the partials in the same
 * order. Each work-item's own partial *is* bit-identical - the strided
 * partition @c (k = tid; k < K; k += WG) is carried over exactly - so the
 * difference is confined to the shape of the summation tree over 256 values,
 * which is the well-conditioned part of the sum.
 *
 * ## Shape of the launches
 *
 * The grid mapping is carried over as-is: one work-group per row for the
 * reductions, one work-item per element for the elementwise ops. That is not
 * yet a tuned mapping for Xe - the elementwise ops move 2 bytes per work-item
 * and will not saturate B70's bandwidth - but the ternary GEMM dominates the
 * layer by an order of magnitude, so these get revisited against a profile
 * rather than on principle. @ref vla::bitvla::ROW_WG is the single knob.
 */

#include "kernels/bitvla/bitvla_lm_cuda.h"
#include "kernels/bitvla/sycl/queue_sycl.h"
#include "kernels/bitvla/sycl/sycl_compat.h"

#include <limits>

using vla::bitvla::as_bf;
using vla::bitvla::bf16;
using vla::bitvla::ROW_WG;
using vla::bitvla::to_bf16;
using vla::bitvla::to_f32;

namespace {

/// Kernel name tags. SYCL would accept unnamed lambdas, but a profile of this
/// port is going to be read op by op, and mangled lambda types do not read.
struct k_rmsnorm {};
struct k_rope_neox {};
struct k_softmax_scaled {};
struct k_squared_relu_mul {};
struct k_add {};
struct k_repeat_kv {};
struct k_transpose_sNhd {};
struct k_transpose_NsHd {};
struct k_gather_rows {};
struct k_layernorm {};
struct k_gelu_tanh {};
struct k_gelu_erf {};
struct k_add_bias {};
struct k_zero_tail {};
struct k_gate_up_fused {};

/// One work-group per row, @ref ROW_WG work-items along the row.
inline sycl::nd_range<2> rows_range(int rows, int wg = ROW_WG) {
    return sycl::nd_range<2>(sycl::range<2>((size_t) rows, (size_t) wg),
                             sycl::range<2>(1, (size_t) wg));
}

/// Flat one-work-item-per-element grid, padded up to whole work-groups.
inline sycl::nd_range<1> flat_range(int n, int wg = ROW_WG) {
    return sycl::nd_range<1>(sycl::range<1>(vla::bitvla::round_up((size_t) n, (size_t) wg)),
                             sycl::range<1>((size_t) wg));
}

}  // namespace

extern "C" void bitvla_rmsnorm_bf16(const vla_bf16* x, const vla_bf16* w,
                                    vla_bf16* out, float eps, int M, int K,
                                    vla_stream stream) {
    if (M <= 0 || K <= 0)
        return;
    sycl::queue &   q  = vla::bitvla_sycl_queue(stream);
    const bf16 *    xs = as_bf(x);
    const bf16 *    ws = as_bf(w);
    bf16 *          os = as_bf(out);
    q.parallel_for<k_rmsnorm>(rows_range(M), [=](sycl::nd_item<2> it) {
        const int    m   = (int) it.get_group(0);
        const int    tid = (int) it.get_local_id(1);
        const bf16 * row = xs + (size_t) m * K;
        bf16 *       o   = os + (size_t) m * K;

        float ss = 0.0f;
        for (int k = tid; k < K; k += ROW_WG) {
            float v = to_f32(row[k]);
            ss += v * v;
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());

        const float mean  = ss / (float) K;
        const float scale = sycl::rsqrt(mean + eps);

        for (int k = tid; k < K; k += ROW_WG) {
            float v  = to_f32(row[k]) * scale;
            float wv = to_f32(ws[k]);
            o[k]     = to_bf16(v * wv);
        }
    });
}

extern "C" void bitvla_rope_neox_bf16(vla_bf16* inout, const float* cos_tab,
                                      const float* sin_tab, int H, int S, int D,
                                      vla_stream stream) {
    if (H <= 0 || S <= 0 || D <= 1)
        return;
    constexpr int WG = 128;
    sycl::queue & q  = vla::bitvla_sycl_queue(stream);
    bf16 *        io = as_bf(inout);
    q.parallel_for<k_rope_neox>(
        sycl::nd_range<3>(sycl::range<3>((size_t) H, (size_t) S, WG),
                          sycl::range<3>(1, 1, WG)),
        [=](sycl::nd_item<3> it) {
            const int h    = (int) it.get_group(0);
            const int s    = (int) it.get_group(1);
            const int tid  = (int) it.get_local_id(2);
            const int half = D / 2;

            bf16 *        row   = io + ((size_t) h * S + s) * D;
            const float * c_row = cos_tab + (size_t) s * half;
            const float * s_row = sin_tab + (size_t) s * half;
            for (int k = tid; k < half; k += WG) {
                float c  = c_row[k];
                float si = s_row[k];
                float a  = to_f32(row[k]);
                float b  = to_f32(row[k + half]);
                row[k]        = to_bf16(a * c - b * si);
                row[k + half] = to_bf16(b * c + a * si);
            }
        });
}

extern "C" void bitvla_softmax_scaled_bf16(vla_bf16* inout, float scale,
                                           int n_rows, int S, vla_stream stream) {
    if (n_rows <= 0 || S <= 0)
        return;
    sycl::queue & q       = vla::bitvla_sycl_queue(stream);
    bf16 *        io      = as_bf(inout);
    const float   neg_inf = -std::numeric_limits<float>::infinity();
    q.parallel_for<k_softmax_scaled>(rows_range(n_rows), [=](sycl::nd_item<2> it) {
        const int row = (int) it.get_group(0);
        const int tid = (int) it.get_local_id(1);
        bf16 *    r   = io + (size_t) row * S;

        // Max is exactly associative, so unlike the two sums this stage is
        // bit-identical to CUDA's whatever order the group combines in.
        float mx = neg_inf;
        for (int i = tid; i < S; i += ROW_WG) {
            float v = to_f32(r[i]) * scale;
            if (v > mx)
                mx = v;
        }
        const float max_v = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());

        float s_sum = 0.0f;
        for (int i = tid; i < S; i += ROW_WG)
            s_sum += sycl::exp(to_f32(r[i]) * scale - max_v);
        s_sum = sycl::reduce_over_group(it.get_group(), s_sum, sycl::plus<float>());

        const float inv_sum = 1.0f / s_sum;
        for (int i = tid; i < S; i += ROW_WG) {
            float v = sycl::exp(to_f32(r[i]) * scale - max_v) * inv_sum;
            r[i]    = to_bf16(v);
        }
    });
}

extern "C" void bitvla_squared_relu_mul_bf16(const vla_bf16* g, const vla_bf16* u,
                                             vla_bf16* out, int N, vla_stream stream) {
    if (N <= 0)
        return;
    sycl::queue &  q  = vla::bitvla_sycl_queue(stream);
    const bf16 *   gs = as_bf(g);
    const bf16 *   us = as_bf(u);
    bf16 *         os = as_bf(out);
    q.parallel_for<k_squared_relu_mul>(flat_range(N), [=](sycl::nd_item<1> it) {
        const int i = (int) it.get_global_id(0);
        if (i >= N)
            return;
        float gv = to_f32(gs[i]);
        if (gv < 0.0f)
            gv = 0.0f;
        os[i] = to_bf16(gv * gv * to_f32(us[i]));
    });
}

extern "C" void bitvla_add_bf16(const vla_bf16* a, const vla_bf16* b,
                                vla_bf16* out, int N, vla_stream stream) {
    if (N <= 0)
        return;
    sycl::queue &  q  = vla::bitvla_sycl_queue(stream);
    const bf16 *   as_ = as_bf(a);
    const bf16 *   bs = as_bf(b);
    bf16 *         os = as_bf(out);
    q.parallel_for<k_add>(flat_range(N), [=](sycl::nd_item<1> it) {
        const int i = (int) it.get_global_id(0);
        if (i >= N)
            return;
        os[i] = to_bf16(to_f32(as_[i]) + to_f32(bs[i]));
    });
}

extern "C" void bitvla_repeat_kv_bf16(const vla_bf16* in, vla_bf16* out,
                                      int n_q, int n_kv, int seq, int hd,
                                      vla_stream stream) {
    if (n_q <= 0 || seq <= 0 || hd <= 0)
        return;
    constexpr int WG = 128;
    sycl::queue & q  = vla::bitvla_sycl_queue(stream);
    const bf16 *  is = as_bf(in);
    bf16 *        os = as_bf(out);
    q.parallel_for<k_repeat_kv>(
        sycl::nd_range<3>(sycl::range<3>((size_t) n_q, (size_t) seq, WG),
                          sycl::range<3>(1, 1, WG)),
        [=](sycl::nd_item<3> it) {
            const int q_h  = (int) it.get_group(0);
            const int s    = (int) it.get_group(1);
            const int tid  = (int) it.get_local_id(2);
            const int kv_h = q_h * n_kv / n_q;
            for (int k = tid; k < hd; k += WG)
                os[((size_t) q_h * seq + s) * hd + k] = is[((size_t) kv_h * seq + s) * hd + k];
        });
}

extern "C" void bitvla_transpose_sNhd_to_NshHd_bf16(const vla_bf16* in, vla_bf16* out,
                                                    int S, int N, int hd,
                                                    vla_stream stream) {
    if (S <= 0 || N <= 0 || hd <= 0)
        return;
    constexpr int WG = 64;
    sycl::queue & q  = vla::bitvla_sycl_queue(stream);
    const bf16 *  is = as_bf(in);
    bf16 *        os = as_bf(out);
    q.parallel_for<k_transpose_sNhd>(
        sycl::nd_range<3>(sycl::range<3>((size_t) N, (size_t) S, WG),
                          sycl::range<3>(1, 1, WG)),
        [=](sycl::nd_item<3> it) {
            const int h   = (int) it.get_group(0);
            const int s   = (int) it.get_group(1);
            const int tid = (int) it.get_local_id(2);
            for (int k = tid; k < hd; k += WG)
                os[((size_t) h * S + s) * hd + k] = is[((size_t) s * N + h) * hd + k];
        });
}

extern "C" void bitvla_transpose_NshHd_to_sNhd_bf16(const vla_bf16* in, vla_bf16* out,
                                                    int N, int S, int hd,
                                                    vla_stream stream) {
    if (S <= 0 || N <= 0 || hd <= 0)
        return;
    constexpr int WG = 64;
    sycl::queue & q  = vla::bitvla_sycl_queue(stream);
    const bf16 *  is = as_bf(in);
    bf16 *        os = as_bf(out);
    q.parallel_for<k_transpose_NsHd>(
        sycl::nd_range<3>(sycl::range<3>((size_t) N, (size_t) S, WG),
                          sycl::range<3>(1, 1, WG)),
        [=](sycl::nd_item<3> it) {
            const int h   = (int) it.get_group(0);
            const int s   = (int) it.get_group(1);
            const int tid = (int) it.get_local_id(2);
            for (int k = tid; k < hd; k += WG)
                os[((size_t) s * N + h) * hd + k] = is[((size_t) h * S + s) * hd + k];
        });
}

extern "C" void bitvla_gather_rows_bf16(const vla_bf16* in, vla_bf16* out,
                                        const int32_t* row_ids, int n_rows, int K,
                                        vla_stream stream) {
    if (n_rows <= 0 || K <= 0)
        return;
    constexpr int WG = 128;
    sycl::queue & q  = vla::bitvla_sycl_queue(stream);
    const bf16 *  is = as_bf(in);
    bf16 *        os = as_bf(out);
    q.parallel_for<k_gather_rows>(rows_range(n_rows, WG), [=](sycl::nd_item<2> it) {
        const int m   = (int) it.get_group(0);
        const int r   = row_ids[m];
        const int tid = (int) it.get_local_id(1);
        for (int k = tid; k < K; k += WG)
            os[(size_t) m * K + k] = is[(size_t) r * K + k];
    });
}

extern "C" void bitvla_layernorm_bf16(const vla_bf16* x, const vla_bf16* w,
                                      const vla_bf16* b, vla_bf16* out,
                                      float eps, int M, int K, vla_stream stream) {
    if (M <= 0 || K <= 0)
        return;
    sycl::queue &  q  = vla::bitvla_sycl_queue(stream);
    const bf16 *   xs = as_bf(x);
    const bf16 *   ws = as_bf(w);
    const bf16 *   bs = as_bf(b);
    bf16 *         os = as_bf(out);
    q.parallel_for<k_layernorm>(rows_range(M), [=](sycl::nd_item<2> it) {
        const int    m   = (int) it.get_group(0);
        const int    tid = (int) it.get_local_id(1);
        const bf16 * row = xs + (size_t) m * K;
        bf16 *       o   = os + (size_t) m * K;

        // Two passes, mean then variance - the CUDA kernel's choice, kept
        // because the one-pass Welford/sum-of-squares alternative is a
        // different rounding profile, not just a different schedule.
        float sum = 0.0f;
        for (int k = tid; k < K; k += ROW_WG)
            sum += to_f32(row[k]);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        const float mean = sum / (float) K;

        float vsum = 0.0f;
        for (int k = tid; k < K; k += ROW_WG) {
            float v = to_f32(row[k]) - mean;
            vsum += v * v;
        }
        vsum = sycl::reduce_over_group(it.get_group(), vsum, sycl::plus<float>());
        const float inv_std = sycl::rsqrt(vsum / (float) K + eps);

        for (int k = tid; k < K; k += ROW_WG) {
            float v  = (to_f32(row[k]) - mean) * inv_std;
            float wv = to_f32(ws[k]);
            float bv = to_f32(bs[k]);
            o[k]     = to_bf16(v * wv + bv);
        }
    });
}

extern "C" void bitvla_gelu_tanh_bf16(const vla_bf16* x, vla_bf16* out,
                                      int N, vla_stream stream) {
    if (N <= 0)
        return;
    sycl::queue &  q  = vla::bitvla_sycl_queue(stream);
    const bf16 *   xs = as_bf(x);
    bf16 *         os = as_bf(out);
    q.parallel_for<k_gelu_tanh>(flat_range(N), [=](sycl::nd_item<1> it) {
        const int i = (int) it.get_global_id(0);
        if (i >= N)
            return;
        float x_ = to_f32(xs[i]);

        const float kAlpha = 0.7978845608028654f;
        const float kBeta  = 0.044715f;
        float       u      = kAlpha * (x_ + kBeta * x_ * x_ * x_);
        float       t      = sycl::tanh(u);
        os[i]              = to_bf16(0.5f * x_ * (1.0f + t));
    });
}

extern "C" void bitvla_gelu_erf_bf16(const vla_bf16* x, vla_bf16* out,
                                     int N, vla_stream stream) {
    if (N <= 0)
        return;
    sycl::queue &  q  = vla::bitvla_sycl_queue(stream);
    const bf16 *   xs = as_bf(x);
    bf16 *         os = as_bf(out);
    q.parallel_for<k_gelu_erf>(flat_range(N), [=](sycl::nd_item<1> it) {
        const int i = (int) it.get_global_id(0);
        if (i >= N)
            return;
        const float x_        = to_f32(xs[i]);
        const float inv_sqrt2 = 0.7071067811865475f;
        // sycl::erf against CUDA's erff: both are the library function, and CUDA
        // compiles this one with --use_fast_math, which does not redirect erff
        // (there is no __erff intrinsic). So this is the same computation, not a
        // substitute for one.
        os[i] = to_bf16(0.5f * x_ * (1.0f + sycl::erf(x_ * inv_sqrt2)));
    });
}

extern "C" void bitvla_add_bias_bf16(const vla_bf16* x, const vla_bf16* bias,
                                     vla_bf16* out, int M, int K, vla_stream stream) {
    if (M <= 0 || K <= 0)
        return;
    sycl::queue &  q    = vla::bitvla_sycl_queue(stream);
    const bf16 *   xs   = as_bf(x);
    const bf16 *   bias_= as_bf(bias);
    bf16 *         os   = as_bf(out);
    q.parallel_for<k_add_bias>(
        sycl::nd_range<2>(sycl::range<2>((size_t) M, vla::bitvla::round_up((size_t) K, ROW_WG)),
                          sycl::range<2>(1, ROW_WG)),
        [=](sycl::nd_item<2> it) {
            const int m = (int) it.get_group(0);
            const int k = (int) it.get_global_id(1);
            if (k >= K)
                return;
            const size_t i = (size_t) m * K + k;
            os[i] = to_bf16(to_f32(xs[i]) + to_f32(bias_[k]));
        });
}

extern "C" void bitvla_zero_tail_bf16(vla_bf16* x, int M, int total_cols,
                                      int start_col, vla_stream stream) {
    if (M <= 0 || start_col >= total_cols)
        return;
    constexpr int WG  = 128;
    const int     len = total_cols - start_col;
    sycl::queue & q   = vla::bitvla_sycl_queue(stream);
    bf16 *        xs  = as_bf(x);
    q.parallel_for<k_zero_tail>(
        sycl::nd_range<2>(sycl::range<2>((size_t) M, vla::bitvla::round_up((size_t) len, WG)),
                          sycl::range<2>(1, WG)),
        [=](sycl::nd_item<2> it) {
            const int m = (int) it.get_group(0);
            const int k = start_col + (int) it.get_global_id(1);
            if (k >= total_cols)
                return;
            xs[(size_t) m * total_cols + k] = to_bf16(0.0f);
        });
}

/**
 * @brief Squared-ReLU gate against the up-projection, reading both halves out
 *        of one fused gate+up GEMM result.
 *
 * Declared here rather than in a header for the same reason the CUDA file
 * declares it locally: only the LM driver calls it.
 */
extern "C" void gate_up_fused_sqrelu_mul_bf16(const vla_bf16* gu, vla_bf16* out,
                                              int seq, int ffn, vla_stream stream) {
    if (seq <= 0 || ffn <= 0)
        return;
    const int      total = seq * ffn;
    sycl::queue &  q     = vla::bitvla_sycl_queue(stream);
    const bf16 *   gus   = as_bf(gu);
    bf16 *         os    = as_bf(out);
    q.parallel_for<k_gate_up_fused>(flat_range(total), [=](sycl::nd_item<1> it) {
        const int idx = (int) it.get_global_id(0);
        if (idx >= total)
            return;
        const int    s        = idx / ffn;
        const int    k        = idx % ffn;
        const size_t row_base = (size_t) s * 2 * ffn;
        float        g        = to_f32(gus[row_base + k]);
        float        u        = to_f32(gus[row_base + ffn + k]);
        if (g < 0.0f)
            g = 0.0f;
        os[(size_t) idx] = to_bf16(g * g * u);
    });
}
