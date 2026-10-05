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
#include "env_flag.h"

#include <cstdio>
#include <cstdlib>

using vla::bitvla::as_bf;
using vla::bitvla::bf16;
using vla::bitvla::exact_div;
using vla::bitvla::ROW_WG;
using vla::bitvla::to_bf16;
using vla::bitvla::to_f32;

namespace {

template <int NI> struct k_add_rmsnorm_quant {};
template <int NI> struct k_sqrelu_rmsnorm_quant {};
template <int NI> struct k_add_layernorm_quant {};
template <int NI> struct k_bias_gelu_quant_pad {};
struct k_bias_residual {};
struct k_rope_qk_rows {};

/// The width the unfused row kernels compiled to (unitrace: SIMD 32). Pinned
/// so the reduction trees, and with them the bits, match.
constexpr int ROW_SG = 32;

inline sycl::nd_range<2> rows_range(int rows) {
    return sycl::nd_range<2>(sycl::range<2>((size_t) rows, (size_t) ROW_WG),
                             sycl::range<2>(1, (size_t) ROW_WG));
}

/**
 * @brief A work-item's slice of one row - elements tid, tid+WG, tid+2*WG, ... -
 *        held in registers.
 *
 * The point of it is the memory system, not the arithmetic. Measured on Xe3, a
 * kernel issuing one small load per work-item per loop trip reaches ~29 GB/s;
 * the same bytes requested as a burst reach the ~77 GB/s LPDDR5x ceiling. So the
 * slice is loaded once, up front, every load independent of the last, and each
 * later pass over the row (sum, max, quantise) reads registers. The partition
 * and the per-work-item accumulation order are those of the uncached loops, so
 * the results are the same bits.
 *
 * @p NI is the slice length, ceil(K / WG) rounded up to a bucket; elements past
 * K are never read or written, and contribute nothing to the reductions.
 */
template <int NI> struct Slice {
    float v[NI];

    template <typename F> void fill(int tid, int K, F && f) {
#pragma unroll
        for (int i = 0; i < NI; ++i) {
            const int k = tid + i * ROW_WG;
            v[i]        = k < K ? f(k) : 0.0f;
        }
    }

    /// Visit (i, k, value) in the uncached loop's order.
    template <typename F> void each(int tid, int K, F && f) const {
#pragma unroll
        for (int i = 0; i < NI; ++i) {
            const int k = tid + i * ROW_WG;
            if (k < K) f(k, v[i]);
        }
    }
};

/// act_quant's tail over a cached slice: absmax, the scale, and the rounded,
/// clamped int8 - the expressions of @c bitvla_act_quant_pad_cuda verbatim.
/// @p normed(k, x) maps a cached value to the bf16 the quantiser reads.
template <int NI, typename Norm>
inline void quant_slice(sycl::nd_item<2> it, int tid, int K, int K_out, const Slice<NI> & sl,
                        Norm normed, int8_t * row_out, float * scale_out) {
    // The normalised bf16 values are needed twice; keep them too.
    Slice<NI> nv;
    float     local_max = 0.0f;
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int k = tid + i * ROW_WG;
        nv.v[i]     = 0.0f;
        if (k < K) {
            nv.v[i]       = to_f32(normed(k, sl.v[i]));
            const float a = sycl::fabs(nv.v[i]);
            if (a > local_max) local_max = a;
        }
    }
    const float row_max = sycl::reduce_over_group(it.get_group(), local_max, sycl::maximum<float>());

    const float amax  = row_max < 1e-5f ? 1e-5f : row_max;
    const float scale = exact_div(127.0f, amax);
    if (tid == 0) *scale_out = scale;

    nv.each(tid, K, [&](int k, float x) {
        float q_ = sycl::rint(x * scale);
        if (q_ > 127.0f) q_ = 127.0f;
        if (q_ < -128.0f) q_ = -128.0f;
        row_out[k] = (int8_t) q_;
    });
    for (int k = K + tid; k < K_out; k += ROW_WG)
        row_out[k] = 0;
}

/// Run @p launch with the smallest slice bucket that holds a row of @p K.
/// Rows wider than the largest bucket (28 * 256 = 7168, against the LM's
/// widest 6912) would need too many registers; none exist in BitVLA, and they
/// abort rather than silently take a slow path.
template <template <int> class Launch, typename... Args>
void dispatch_slice(int K, Args &&... args) {
    const int ni = (K + ROW_WG - 1) / ROW_WG;
    if (ni <= 4)       Launch<4>::run(args...);
    else if (ni <= 8)  Launch<8>::run(args...);
    else if (ni <= 12) Launch<12>::run(args...);
    else if (ni <= 20) Launch<20>::run(args...);
    else if (ni <= 28) Launch<28>::run(args...);
    else {
        std::fprintf(stderr, "vla(bitvla): fused row kernel: K=%d wider than the 7168 the "
                     "register slices are sized for\n", K);
        std::abort();
    }
}

/**
 * @brief The contiguous counterpart of @ref Slice: each work-item owns whole
 *        8-element chunks - chunk c = tid + i*WG covers elements [8c, 8c+8) -
 *        loaded as one 16-byte vector and quantised out as one 8-byte store.
 *
 * Why it exists next to Slice: on Xe3 the strided partition caps these kernels
 * at ~38 GB/s however the loads are scheduled, because a SIMD32 load of 32
 * strided-by-WG bf16 is only a 64-byte request, and the int8 results leave as
 * single-byte stores. Here a SIMD32 instruction moves 512 contiguous bytes in
 * and 256 out.
 *
 * The cost is bit-identity with the unfused chain: each work-item now sums a
 * different subset of the row, in a different order, so the f32 reductions
 * (and through them the norm scale) can differ in the last bit. That is the
 * same contract the SYCL reductions already have with CUDA's; the per-element
 * arithmetic, every bf16 rounding point and the residual stream written back
 * are unchanged and still exact. Rows whose width is not a multiple of 8 take
 * the strided kernels.
 */
constexpr int VEC = 8;
using vbf    = sycl::vec<uint16_t, VEC>;
using vbytes = sycl::vec<int8_t, VEC>;

inline float bf_bits_to_f32(uint16_t b) { return sycl::bit_cast<float>((uint32_t) b << 16); }
inline uint16_t f32_to_bf_bits(float f) { return sycl::bit_cast<uint16_t>(to_bf16(f)); }

template <int NC> struct VSlice {
    float v[NC][VEC];

    /// @p f(chunk_base, j) -> value of element chunk_base + j.
    template <typename F> void fill(int tid, int n_chunks, F && f) {
#pragma unroll
        for (int i = 0; i < NC; ++i) {
            const int c = tid + i * ROW_WG;
            if (c < n_chunks) f(c * VEC, v[i]);
            else
#pragma unroll
                for (int j = 0; j < VEC; ++j) v[i][j] = 0.0f;
        }
    }
};

template <int NC, typename Norm>
inline void quant_vslice(sycl::nd_item<2> it, int tid, int n_chunks, int K, int K_out,
                         VSlice<NC> & sl, Norm normed, int8_t * row_out, float * scale_out) {
    float local_max = 0.0f;
#pragma unroll
    for (int i = 0; i < NC; ++i) {
        const int c = tid + i * ROW_WG;
        if (c < n_chunks) {
            vbf w8 = normed.load_w(c * VEC);
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                sl.v[i][j]    = to_f32(normed(sl.v[i][j], w8, j, c * VEC + j));
                const float a = sycl::fabs(sl.v[i][j]);
                if (a > local_max) local_max = a;
            }
        }
    }
    const float row_max = sycl::reduce_over_group(it.get_group(), local_max, sycl::maximum<float>());

    const float amax  = row_max < 1e-5f ? 1e-5f : row_max;
    const float scale = exact_div(127.0f, amax);
    if (tid == 0) *scale_out = scale;

#pragma unroll
    for (int i = 0; i < NC; ++i) {
        const int c = tid + i * ROW_WG;
        if (c < n_chunks) {
            vbytes o;
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                float q_ = sycl::rint(sl.v[i][j] * scale);
                if (q_ > 127.0f) q_ = 127.0f;
                if (q_ < -128.0f) q_ = -128.0f;
                o[j] = (int8_t) q_;
            }
            *reinterpret_cast<vbytes *>(row_out + (size_t) c * VEC) = o;
        }
    }
    for (int k = K + tid; k < K_out; k += ROW_WG)
        row_out[k] = 0;
}

inline vbf load8(const bf16 * p) { return *reinterpret_cast<const vbf *>(p); }
inline void store8(bf16 * p, const vbf & v) { *reinterpret_cast<vbf *>(p) = v; }

/// RMSNorm's normalised element: bf16(x * scale * w), with w loaded per chunk.
struct RmsNorm {
    const bf16 * w;
    float        scale;
    vbf          load_w(int base) const { return load8(w + base); }
    bf16         operator()(float x, const vbf & w8, int j, int) const {
        float v  = x * scale;
        float wv = bf_bits_to_f32(w8[j]);
        return to_bf16(v * wv);
    }
};

template <int NC> struct k_v_add_rmsnorm_quant {};
template <int NC> struct k_v_sqrelu_rmsnorm_quant {};
template <int NC> struct k_v_add_layernorm_quant {};
template <int NC> struct k_v_bias_gelu_quant_pad {};

/// Smallest chunk bucket holding @p K / VEC chunks per WG.
template <template <int> class Launch, typename... Args>
void dispatch_vslice(int K, Args &&... args) {
    const int nc = (K / VEC + ROW_WG - 1) / ROW_WG;
    if (nc <= 1)      Launch<1>::run(args...);
    else if (nc <= 2) Launch<2>::run(args...);
    else if (nc <= 3) Launch<3>::run(args...);
    else if (nc <= 4) Launch<4>::run(args...);
    else {
        std::fprintf(stderr, "vla(bitvla): fused row kernel: K=%d wider than the vector "
                     "slices are sized for\n", K);
        std::abort();
    }
}

template <int NC> struct VAddRmsnormQuant {
    static void run(sycl::queue & q, bf16 * hs, const bf16 * ds, const bf16 * ws, int8_t * out,
                    float * scales, float eps, int M, int K) {
        const int n_chunks = K / VEC;
        q.parallel_for<k_v_add_rmsnorm_quant<NC>>(rows_range(M), [=](sycl::nd_item<2> it) {
            const int m   = (int) it.get_group(0);
            const int tid = (int) it.get_local_id(1);
            bf16 *    row = hs + (size_t) m * K;

            VSlice<NC> sl;
            if (ds) {
                const bf16 * drow = ds + (size_t) m * K;
                sl.fill(tid, n_chunks, [&](int base, float * v) {
                    const vbf a = load8(row + base), d = load8(drow + base);
                    vbf       s;
#pragma unroll
                    for (int j = 0; j < VEC; ++j) {
                        s[j] = f32_to_bf_bits(bf_bits_to_f32(a[j]) + bf_bits_to_f32(d[j]));
                        v[j] = bf_bits_to_f32(s[j]);
                    }
                    store8(row + base, s);
                });
            } else {
                sl.fill(tid, n_chunks, [&](int base, float * v) {
                    const vbf a = load8(row + base);
#pragma unroll
                    for (int j = 0; j < VEC; ++j) v[j] = bf_bits_to_f32(a[j]);
                });
            }

            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < NC; ++i)
#pragma unroll
                for (int j = 0; j < VEC; ++j) ss += sl.v[i][j] * sl.v[i][j];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());

            const float mean = ss / (float) K;
            RmsNorm     nrm{ws, sycl::rsqrt(mean + eps)};
            quant_vslice<NC>(it, tid, n_chunks, K, K, sl, nrm, out + (size_t) m * K, scales + m);
        });
    }
};

template <int NC> struct VSqreluRmsnormQuant {
    static void run(sycl::queue & q, const bf16 * gus, const bf16 * ws, int8_t * out,
                    float * scales, float eps, int seq, int K) {
        const int n_chunks = K / VEC;
        q.parallel_for<k_v_sqrelu_rmsnorm_quant<NC>>(rows_range(seq), [=](sycl::nd_item<2> it) {
            const int    m   = (int) it.get_group(0);
            const int    tid = (int) it.get_local_id(1);
            const bf16 * g_r = gus + (size_t) m * 2 * K;
            const bf16 * u_r = g_r + K;

            VSlice<NC> sl;
            sl.fill(tid, n_chunks, [&](int base, float * v) {
                const vbf g8 = load8(g_r + base), u8 = load8(u_r + base);
#pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    float g = bf_bits_to_f32(g8[j]);
                    float u = bf_bits_to_f32(u8[j]);
                    if (g < 0.0f) g = 0.0f;
                    v[j] = to_f32(to_bf16(g * g * u));
                }
            });

            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < NC; ++i)
#pragma unroll
                for (int j = 0; j < VEC; ++j) ss += sl.v[i][j] * sl.v[i][j];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());

            const float mean = ss / (float) K;
            RmsNorm     nrm{ws, sycl::rsqrt(mean + eps)};
            quant_vslice<NC>(it, tid, n_chunks, K, K, sl, nrm, out + (size_t) m * K, scales + m);
        });
    }
};

struct LayerNorm {
    const bf16 * w;
    const bf16 * b;
    float        mean, inv_std;
    vbf  load_w(int base) const { return load8(w + base); }
    bf16 operator()(float x, const vbf & w8, int j, int k) const {
        float v  = (x - mean) * inv_std;
        float wv = bf_bits_to_f32(w8[j]);
        float bv = to_f32(b[k]);
        return to_bf16(v * wv + bv);
    }
};

template <int NC> struct VAddLayernormQuant {
    static void run(sycl::queue & q, bf16 * hs, const bf16 * ds, const bf16 * dbs,
                    const bf16 * ws, const bf16 * bs, int8_t * out, float * scales, float eps,
                    int M, int K) {
        const int n_chunks = K / VEC;
        q.parallel_for<k_v_add_layernorm_quant<NC>>(rows_range(M), [=](sycl::nd_item<2> it) {
            const int m   = (int) it.get_group(0);
            const int tid = (int) it.get_local_id(1);
            bf16 *    row = hs + (size_t) m * K;

            VSlice<NC> sl;
            if (ds) {
                const bf16 * drow = ds + (size_t) m * K;
                sl.fill(tid, n_chunks, [&](int base, float * v) {
                    const vbf a = load8(row + base), d = load8(drow + base);
                    vbf       s;
#pragma unroll
                    for (int j = 0; j < VEC; ++j) {
                        float dv = bf_bits_to_f32(d[j]);
                        if (dbs) dv = to_f32(to_bf16(dv + to_f32(dbs[base + j])));
                        s[j] = f32_to_bf_bits(bf_bits_to_f32(a[j]) + dv);
                        v[j] = bf_bits_to_f32(s[j]);
                    }
                    store8(row + base, s);
                });
            } else {
                sl.fill(tid, n_chunks, [&](int base, float * v) {
                    const vbf a = load8(row + base);
#pragma unroll
                    for (int j = 0; j < VEC; ++j) v[j] = bf_bits_to_f32(a[j]);
                });
            }

            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < NC; ++i)
#pragma unroll
                for (int j = 0; j < VEC; ++j) sum += sl.v[i][j];
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            const float mean = sum / (float) K;

            float vsum = 0.0f;
#pragma unroll
            for (int i = 0; i < NC; ++i) {
                if (tid + i * ROW_WG >= n_chunks) continue;
#pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    float v = sl.v[i][j] - mean;
                    vsum += v * v;
                }
            }
            vsum = sycl::reduce_over_group(it.get_group(), vsum, sycl::plus<float>());

            LayerNorm nrm{ws, bs, mean, sycl::rsqrt(vsum / (float) K + eps)};
            quant_vslice<NC>(it, tid, n_chunks, K, K, sl, nrm, out + (size_t) m * K, scales + m);
        });
    }
};

struct Identity {
    vbf  load_w(int) const { return vbf(0); }
    bf16 operator()(float x, const vbf &, int, int) const { return to_bf16(x); }
};

template <int NC> struct VBiasGeluQuantPad {
    static void run(sycl::queue & q, const bf16 * xs, const bf16 * bs, int8_t * out,
                    float * scales, int M, int K_in, int K_out) {
        const int n_chunks = K_in / VEC;
        q.parallel_for<k_v_bias_gelu_quant_pad<NC>>(rows_range(M), [=](sycl::nd_item<2> it) {
            const int    m   = (int) it.get_group(0);
            const int    tid = (int) it.get_local_id(1);
            const bf16 * row = xs + (size_t) m * K_in;

            VSlice<NC> sl;
            sl.fill(tid, n_chunks, [&](int base, float * v) {
                const vbf x8 = load8(row + base), b8 = load8(bs + base);
#pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    const float x_ = to_f32(to_bf16(bf_bits_to_f32(x8[j]) + bf_bits_to_f32(b8[j])));

                    const float kAlpha = 0.7978845608028654f;
                    const float kBeta  = 0.044715f;
                    float       u      = kAlpha * (x_ + kBeta * x_ * x_ * x_);
                    float       t      = sycl::tanh(u);
                    v[j]               = to_f32(to_bf16(0.5f * x_ * (1.0f + t)));
                }
            });
            quant_vslice<NC>(it, tid, n_chunks, K_in, K_out, sl, Identity{},
                             out + (size_t) m * K_out, scales + m);
        });
    }
};

/// The vector kernels need whole 8-element chunks and 16-byte-aligned rows
/// (8-byte for the int8 output, which K_out % 8 == 0 gives).
inline bool vec_ok(int K, int K_out, const void * a, const void * b = nullptr) {
    auto al = [](const void * p) { return p == nullptr || ((uintptr_t) p % 16) == 0; };
    return K % VEC == 0 && K_out % VEC == 0 && al(a) && al(b);
}

/// @c VLA_BITVLA_STRIDED_ROWS=1 keeps the strided, bit-identical-to-the-chain
/// row kernels, to bisect against.
inline bool strided_only() {
    static const bool on = vla::env_flag("VLA_BITVLA_STRIDED_ROWS");
    return on;
}

template <int NI> struct AddRmsnormQuant {
    static void run(sycl::queue & q, bf16 * hs, const bf16 * ds, const bf16 * ws, int8_t * out,
                    float * scales, float eps, int M, int K) {
        q.parallel_for<k_add_rmsnorm_quant<NI>>(
            rows_range(M), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(ROW_SG)]] {
                const int m   = (int) it.get_group(0);
                const int tid = (int) it.get_local_id(1);
                bf16 *    row = hs + (size_t) m * K;

                // The residual add, written back once, as bitvla_add_bf16 wrote it.
                Slice<NI> sl;
                if (ds) {
                    const bf16 * drow = ds + (size_t) m * K;
                    sl.fill(tid, K, [&](int k) {
                        const bf16 s = to_bf16(to_f32(row[k]) + to_f32(drow[k]));
                        row[k]       = s;
                        return to_f32(s);
                    });
                } else {
                    sl.fill(tid, K, [&](int k) { return to_f32(row[k]); });
                }

                float ss = 0.0f;
                sl.each(tid, K, [&](int, float v) { ss += v * v; });
                ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());

                const float mean  = ss / (float) K;
                const float scale = sycl::rsqrt(mean + eps);

                quant_slice<NI>(it, tid, K, K, sl,
                                [&](int k, float x) {
                                    float v  = x * scale;
                                    float wv = to_f32(ws[k]);
                                    return to_bf16(v * wv);
                                },
                                out + (size_t) m * K, scales + m);
            });
    }
};

template <int NI> struct SqreluRmsnormQuant {
    static void run(sycl::queue & q, const bf16 * gus, const bf16 * ws, int8_t * out,
                    float * scales, float eps, int seq, int K) {
        q.parallel_for<k_sqrelu_rmsnorm_quant<NI>>(
            rows_range(seq), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(ROW_SG)]] {
                const int    m   = (int) it.get_group(0);
                const int    tid = (int) it.get_local_id(1);
                const bf16 * g_r = gus + (size_t) m * 2 * K;
                const bf16 * u_r = g_r + K;

                // gate_up_fused_sqrelu_mul_bf16's element, rounded where it rounded.
                Slice<NI> sl;
                sl.fill(tid, K, [&](int k) {
                    float g = to_f32(g_r[k]);
                    float u = to_f32(u_r[k]);
                    if (g < 0.0f) g = 0.0f;
                    return to_f32(to_bf16(g * g * u));
                });

                float ss = 0.0f;
                sl.each(tid, K, [&](int, float v) { ss += v * v; });
                ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());

                const float mean  = ss / (float) K;
                const float scale = sycl::rsqrt(mean + eps);

                quant_slice<NI>(it, tid, K, K, sl,
                                [&](int k, float x) {
                                    float v  = x * scale;
                                    float wv = to_f32(ws[k]);
                                    return to_bf16(v * wv);
                                },
                                out + (size_t) m * K, scales + m);
            });
    }
};

template <int NI> struct AddLayernormQuant {
    static void run(sycl::queue & q, bf16 * hs, const bf16 * ds, const bf16 * dbs,
                    const bf16 * ws, const bf16 * bs, int8_t * out, float * scales, float eps,
                    int M, int K) {
        q.parallel_for<k_add_layernorm_quant<NI>>(
            rows_range(M), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(ROW_SG)]] {
                const int m   = (int) it.get_group(0);
                const int tid = (int) it.get_local_id(1);
                bf16 *    row = hs + (size_t) m * K;

                Slice<NI> sl;
                if (ds) {
                    const bf16 * drow = ds + (size_t) m * K;
                    sl.fill(tid, K, [&](int k) {
                        // add_bias on the projection, then the residual add.
                        const bf16 d = dbs ? to_bf16(to_f32(drow[k]) + to_f32(dbs[k])) : drow[k];
                        const bf16 s = to_bf16(to_f32(row[k]) + to_f32(d));
                        row[k]       = s;
                        return to_f32(s);
                    });
                } else {
                    sl.fill(tid, K, [&](int k) { return to_f32(row[k]); });
                }

                float sum = 0.0f;
                sl.each(tid, K, [&](int, float v) { sum += v; });
                sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
                const float mean = sum / (float) K;

                float vsum = 0.0f;
                sl.each(tid, K, [&](int, float x) {
                    float v = x - mean;
                    vsum += v * v;
                });
                vsum = sycl::reduce_over_group(it.get_group(), vsum, sycl::plus<float>());
                const float inv_std = sycl::rsqrt(vsum / (float) K + eps);

                quant_slice<NI>(it, tid, K, K, sl,
                                [&](int k, float x) {
                                    float v  = (x - mean) * inv_std;
                                    float wv = to_f32(ws[k]);
                                    float bv = to_f32(bs[k]);
                                    return to_bf16(v * wv + bv);
                                },
                                out + (size_t) m * K, scales + m);
            });
    }
};

template <int NI> struct BiasGeluQuantPad {
    static void run(sycl::queue & q, const bf16 * xs, const bf16 * bs, int8_t * out,
                    float * scales, int M, int K_in, int K_out) {
        q.parallel_for<k_bias_gelu_quant_pad<NI>>(
            rows_range(M), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(ROW_SG)]] {
                const int    m   = (int) it.get_group(0);
                const int    tid = (int) it.get_local_id(1);
                const bf16 * row = xs + (size_t) m * K_in;

                // add_bias then bitvla_gelu_tanh_bf16, each rounded as it was -
                // and now evaluated once per element instead of once per pass.
                Slice<NI> sl;
                sl.fill(tid, K_in, [&](int k) {
                    const float x_ = to_f32(to_bf16(to_f32(row[k]) + to_f32(bs[k])));

                    const float kAlpha = 0.7978845608028654f;
                    const float kBeta  = 0.044715f;
                    float       u      = kAlpha * (x_ + kBeta * x_ * x_ * x_);
                    float       t      = sycl::tanh(u);
                    return to_f32(to_bf16(0.5f * x_ * (1.0f + t)));
                });
                quant_slice<NI>(it, tid, K_in, K_out, sl,
                                [](int, float x) { return to_bf16(x); },
                                out + (size_t) m * K_out, scales + m);
            });
    }
};

}  // namespace

extern "C" void bitvla_add_rmsnorm_quant_bf16(vla_bf16* h, const vla_bf16* delta,
                                              const vla_bf16* w, int8_t* out, float* scales,
                                              float eps, int M, int K, vla_stream stream) {
    if (M <= 0 || K <= 0) return;
    if (!strided_only() && vec_ok(K, K, h, delta) && vec_ok(K, K, w)) {
        dispatch_vslice<VAddRmsnormQuant>(K, vla::bitvla_sycl_queue(stream), as_bf(h),
                                          as_bf(delta), as_bf(w), out, scales, eps, M, K);
        return;
    }
    dispatch_slice<AddRmsnormQuant>(K, vla::bitvla_sycl_queue(stream), as_bf(h), as_bf(delta),
                                    as_bf(w), out, scales, eps, M, K);
}

extern "C" void bitvla_sqrelu_rmsnorm_quant_bf16(const vla_bf16* gu, const vla_bf16* w,
                                                 int8_t* out, float* scales, float eps,
                                                 int seq, int ffn, vla_stream stream) {
    if (seq <= 0 || ffn <= 0) return;
    if (!strided_only() && vec_ok(ffn, ffn, gu, w)) {
        dispatch_vslice<VSqreluRmsnormQuant>(ffn, vla::bitvla_sycl_queue(stream), as_bf(gu),
                                             as_bf(w), out, scales, eps, seq, ffn);
        return;
    }
    dispatch_slice<SqreluRmsnormQuant>(ffn, vla::bitvla_sycl_queue(stream), as_bf(gu), as_bf(w),
                                       out, scales, eps, seq, ffn);
}

extern "C" void bitvla_add_layernorm_quant_bf16(vla_bf16* h, const vla_bf16* delta,
                                                const vla_bf16* delta_bias,
                                                const vla_bf16* w, const vla_bf16* b,
                                                int8_t* out, float* scales, float eps,
                                                int M, int K, vla_stream stream) {
    if (M <= 0 || K <= 0) return;
    if (!strided_only() && vec_ok(K, K, h, delta) && vec_ok(K, K, w)) {
        dispatch_vslice<VAddLayernormQuant>(K, vla::bitvla_sycl_queue(stream), as_bf(h),
                                            as_bf(delta), as_bf(delta_bias), as_bf(w), as_bf(b),
                                            out, scales, eps, M, K);
        return;
    }
    dispatch_slice<AddLayernormQuant>(K, vla::bitvla_sycl_queue(stream), as_bf(h), as_bf(delta),
                                      as_bf(delta_bias), as_bf(w), as_bf(b), out, scales, eps, M,
                                      K);
}

extern "C" void bitvla_bias_gelu_quant_pad_bf16(const vla_bf16* x, const vla_bf16* bias,
                                                int8_t* out, float* scales,
                                                int M, int K_in, int K_out, vla_stream stream) {
    if (M <= 0 || K_in <= 0 || K_out < K_in) return;
    if (!strided_only() && vec_ok(K_in, K_out, x, bias)) {
        dispatch_vslice<VBiasGeluQuantPad>(K_in, vla::bitvla_sycl_queue(stream), as_bf(x),
                                           as_bf(bias), out, scales, M, K_in, K_out);
        return;
    }
    dispatch_slice<BiasGeluQuantPad>(K_in, vla::bitvla_sycl_queue(stream), as_bf(x), as_bf(bias),
                                     out, scales, M, K_in, K_out);
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
                                              int hd, int q_ld, int k_ld, vla_stream stream) {
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

            bf16 * row = hh < n_q ? qs + (size_t) s * q_ld + (size_t) hh * hd
                                  : ks + (size_t) s * k_ld + (size_t) (hh - n_q) * hd;
            const float c  = cos_tab[(size_t) s * half + kk];
            const float si = sin_tab[(size_t) s * half + kk];
            const float a  = to_f32(row[kk]);
            const float b  = to_f32(row[kk + half]);
            row[kk]        = to_bf16(a * c - b * si);
            row[kk + half] = to_bf16(b * c + a * si);
        });
}
