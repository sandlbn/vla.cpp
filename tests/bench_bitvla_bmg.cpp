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
 * @file bench_bitvla_bmg.cpp
 * @brief BitVLA's ternary GEMM against Battlemage's measured ceilings.
 *
 * "How fast is it" is not a useful question on its own; "how much of the
 * machine is it using" is. This measures both halves of that ratio on the same
 * device in the same process, so the answer does not depend on a marketing
 * figure being right.
 *
 * The ceilings are measured, not quoted:
 *
 *   * @ref measure_read_bw / @ref measure_copy_bw - achievable HBM/GDDR
 *     bandwidth, read-only and read+write. Quoting the spec sheet would
 *     overstate the target by the usual 10-20%, and the point of a roofline is
 *     to know when to stop optimising, which a ceiling you cannot reach does
 *     not tell you.
 *   * @ref measure_xmx_int8 / @ref measure_xmx_bf16 - XMX throughput from a
 *     dependency-free @c joint_matrix loop. Eight independent accumulators,
 *     because one is latency-bound and would measure the pipeline depth instead
 *     of the issue rate.
 *
 * Both ceilings matter because BitVLA sits on opposite sides of the ridge
 * depending on M. Decode is M=1: one activation row against a whole weight
 * matrix, arithmetic intensity ~2 ops/byte, nowhere near the ridge, and the
 * only lever is bytes. Prefill and the ViT run M=7..61 and start to reach for
 * the XMX ceiling instead. A single "% of peak" number averaged over both would
 * describe neither.
 *
 * That distinction is also the argument for Stage 4. The oneDNN path unpacks
 * the ternary weights to int8 before the matmul, so it streams four bytes where
 * the checkpoint stores one. At M=1 that is a straight 4x on the term that
 * dominates - which this bench reports as a separate roofline (@c packed)
 * rather than as a claim.
 *
 * Emits a human table on stdout and, with --csv <path>, the same numbers for
 * scripts/plot_bmg_roofline.py.
 *
 * Skips itself (exit 0) when no GPU is present.
 */

#include "kernels/bitvla/device.h"
#include "kernels/bitvla/sycl/queue_sycl.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
void bitlinear_int8xint2(int8_t * input0, int8_t * input1, vla_bf16 * output0, float * s,
                         float * ws, int M, int N, int K, vla_stream stream);
void bitlinear_int8xint2_m(int8_t * input0, int8_t * input1, vla_bf16 * output0, float * s,
                           float * ws, int M, int N, int K, vla_stream stream);
}

namespace {

using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int SG = 16;  ///< Battlemage sub-group width the XMX probes require.

// ---------------------------------------------------------------------------
// timing
// ---------------------------------------------------------------------------

/// Seconds per iteration of @p body, after @p warmup untimed passes.
/// Best-of rather than mean: the interesting quantity is what the hardware can
/// do, and a slow pass is always something else on the machine, never the GPU
/// getting faster than it is.
template <typename F> double best_of(int reps, int warmup, F && body) {
    for (int i = 0; i < warmup; ++i) {
        body();
    }
    vla_dev_sync(nullptr);

    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        body();
        vla_dev_sync(nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

// ---------------------------------------------------------------------------
// memory ceiling
// ---------------------------------------------------------------------------

struct k_read {};
struct k_copy {};
struct k_fill {};

/**
 * @brief Fill @p bytes of device memory with incompressible pseudorandom data.
 *
 * Not a detail. Xe implements lossless compression of memory surfaces, so a
 * buffer memset to a constant byte moves a small fraction of its nominal size
 * across the bus. Filling this way first is the difference between measuring
 * DRAM and measuring the compressor: with @c memset the read probe reported
 * 2176 GB/s at 64 MiB and 1370 GB/s at 256 MiB - both above the bus width, and
 * not even monotonic, which is what gave it away.
 *
 * The same applies to the GEMM operands below. Weights memset to a constant are
 * not the weights the model has.
 */
void fill_random(void * p, size_t bytes, bool ternary = false) {
    sycl::queue & q = vla::bitvla_sycl_queue(nullptr);
    const size_t  n = bytes / sizeof(uint32_t);
    auto *        w = (uint32_t *) p;
    q.parallel_for<k_fill>(sycl::range<1>(n), [=](sycl::id<1> id) {
        // splitmix32: cheap, and its output has no byte-level structure for the
        // compressor to find.
        uint32_t x = (uint32_t) id[0] * 0x9e3779b9u + 0x85ebca6bu;
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        if (ternary) {
            // Every byte reduced to one of four values, matching what the
            // unpack kernel writes: (code & 3) - 2, i.e. {-2,-1,0,1}.
            uint32_t t = 0;
            for (int b = 0; b < 4; ++b) {
                const uint32_t v = (uint32_t) (uint8_t) (int8_t) (((x >> (2 * b)) & 3u) - 2u);
                t |= v << (8 * b);
            }
            x = t;
        }
        w[id[0]] = x;
    });
    q.wait();
}

/// Streaming-read bandwidth in GB/s, over @p bytes of device memory.
///
/// Reads as @c uint4 (16 B per work-item per step) with a grid-stride loop: one
/// work-item per element would issue four times the requests for the same
/// traffic and measure the request rate instead of the bus.
double measure_read_bw(size_t bytes, bool ternary = false) {
    sycl::queue & q = vla::bitvla_sycl_queue(nullptr);

    const size_t n4  = bytes / sizeof(sycl::uint4);
    auto *       buf = (sycl::uint4 *) vla_dev_malloc(n4 * sizeof(sycl::uint4));
    auto *       out = (uint32_t *) vla_dev_malloc(sizeof(uint32_t));
    if (!buf || !out) {
        vla_dev_free(buf);
        vla_dev_free(out);
        return 0.0;
    }
    fill_random(buf, n4 * sizeof(sycl::uint4), ternary);

    const size_t threads = 1u << 20;
    const double secs    = best_of(5, 2, [&] {
        q.parallel_for<k_read>(
            sycl::nd_range<1>(sycl::range<1>(threads), sycl::range<1>(256)),
            [=](sycl::nd_item<1> it) {
                const size_t gid    = it.get_global_id(0);
                const size_t stride = it.get_global_range(0);

                // UNROLL independent accumulators, so UNROLL loads per thread
                // can be in flight at once. A single accumulator makes the loop
                // a dependency chain: the next load does not issue until the
                // previous add retires, and the probe then measures memory
                // latency times occupancy rather than bandwidth. That read 596
                // GB/s here while the GEMM below was demonstrably sustaining
                // more, which is what exposed it.
                constexpr int UNROLL = 8;
                sycl::uint4   acc[UNROLL];
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    acc[u] = sycl::uint4{ 0, 0, 0, 0 };
                }

                for (size_t i = gid; i < n4; i += stride * UNROLL) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        const size_t j = i + (size_t) u * stride;
                        if (j < n4) {
                            acc[u] += buf[j];
                        }
                    }
                }

                sycl::uint4 sum = { 0, 0, 0, 0 };
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    sum += acc[u];
                }
                // Never true, but the compiler cannot prove it, so the loop
                // above has to happen. All four lanes are read: leave one out
                // and the load can legally narrow, which would measure three
                // quarters of the traffic being counted.
                if (sum.x() == 0xdeadbeefu && sum.y() == 0u && sum.z() == 1u &&
                    sum.w() == 2u) {
                    *out = sum.z();
                }
            });
    });

    vla_dev_free(buf);
    vla_dev_free(out);
    return (double) (n4 * sizeof(sycl::uint4)) / secs / 1e9;
}

struct k_null {};

/**
 * @brief Seconds to submit a one-item kernel and see it complete.
 *
 * The floor under every row of the table below. The smallest ViT shapes finish
 * in 14-18 us, which is the same order as this, so reporting them as a fraction
 * of a bandwidth ceiling describes the dispatch path and not the kernel. Having
 * the number measured means the plot can say so rather than imply the ViT GEMM
 * is twenty times worse than the LM's.
 */
double measure_launch_overhead() {
    sycl::queue & q = vla::bitvla_sycl_queue(nullptr);
    auto *        out = (uint32_t *) vla_dev_malloc(sizeof(uint32_t));
    if (!out) {
        return 0.0;
    }
    const double secs = best_of(200, 20, [&] {
        q.parallel_for<k_null>(sycl::range<1>(1), [=](sycl::id<1> id) {
            if (id[0] == 0xffffffffu) {
                *out = 1;
            }
        });
    });
    vla_dev_free(out);
    return secs;
}

/// Copy bandwidth in GB/s, counting both the read and the write.
double measure_copy_bw(size_t bytes) {
    sycl::queue & q = vla::bitvla_sycl_queue(nullptr);

    const size_t n4  = bytes / sizeof(sycl::uint4);
    auto *       src = (sycl::uint4 *) vla_dev_malloc(n4 * sizeof(sycl::uint4));
    auto *       dst = (sycl::uint4 *) vla_dev_malloc(n4 * sizeof(sycl::uint4));
    if (!src || !dst) {
        vla_dev_free(src);
        vla_dev_free(dst);
        return 0.0;
    }
    fill_random(src, n4 * sizeof(sycl::uint4));

    const size_t threads = 1u << 20;
    const double secs    = best_of(5, 2, [&] {
        q.parallel_for<k_copy>(
            sycl::nd_range<1>(sycl::range<1>(threads), sycl::range<1>(256)),
            [=](sycl::nd_item<1> it) {
                const size_t  gid    = it.get_global_id(0);
                const size_t  stride = it.get_global_range(0);
                constexpr int UNROLL = 8;
                // Same reason as the read probe: independent loads in flight.
                for (size_t i = gid; i < n4; i += stride * UNROLL) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        const size_t j = i + (size_t) u * stride;
                        if (j < n4) {
                            dst[j] = src[j];
                        }
                    }
                }
            });
    });

    vla_dev_free(src);
    vla_dev_free(dst);
    return 2.0 * (double) (n4 * sizeof(sycl::uint4)) / secs / 1e9;
}

// ---------------------------------------------------------------------------
// XMX ceiling
// ---------------------------------------------------------------------------

struct k_xmx_i8 {};
struct k_xmx_bf {};

/// Independent accumulator chains per sub-group. One would measure XMX latency.
constexpr int NACC  = 8;
constexpr int ITERS = 512;

/// Peak int8 XMX throughput in TOPS, from the 8x16x32 s8 combination the
/// preflight reports on this part.
double measure_xmx_int8(int sub_groups) {
    sycl::queue & q = vla::bitvla_sycl_queue(nullptr);

    auto * out = (int32_t *) vla_dev_malloc((size_t) sub_groups * NACC * 8 * 16 * sizeof(int32_t));
    if (!out) {
        return 0.0;
    }

    const size_t wg      = 128;  // 8 sub-groups of 16
    const size_t threads = (size_t) sub_groups * SG;
    const double secs    = best_of(5, 2, [&] {
        q.parallel_for<k_xmx_i8>(
            sycl::nd_range<1>(sycl::range<1>(threads), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = it.get_sub_group();

                joint_matrix<sycl::sub_group, int8_t, use::a, 8, 32, layout::row_major> a;
                joint_matrix<sycl::sub_group, int8_t, use::b, 32, 16, layout::ext_intel_packed> b;
                joint_matrix<sycl::sub_group, int32_t, use::accumulator, 8, 16> c[NACC];

                joint_matrix_fill(sg, a, (int8_t) 1);
                joint_matrix_fill(sg, b, (int8_t) 1);
                for (int i = 0; i < NACC; ++i) {
                    joint_matrix_fill(sg, c[i], 0);
                }

                for (int t = 0; t < ITERS; ++t) {
#pragma unroll
                    for (int i = 0; i < NACC; ++i) {
                        joint_matrix_mad(sg, c[i], a, b, c[i]);
                    }
                }

                // Every chain is stored. Storing only c[0] would let the
                // compiler delete the other seven while this function went on
                // counting their work - an 8x overstatement of the ceiling.
                const size_t base = (size_t) (it.get_global_linear_id() / SG) * NACC * 8 * 16;
                for (int i = 0; i < NACC; ++i) {
                    joint_matrix_store(sg, c[i],
                                       sycl::address_space_cast<sycl::access::address_space::global_space,
                                                                sycl::access::decorated::no>(
                                           out + base + (size_t) i * 8 * 16),
                                       16, layout::row_major);
                }
            });
    });

    vla_dev_free(out);
    // 2 ops per MAC, 8*16*32 MACs per joint_matrix_mad.
    const double ops = 2.0 * 8 * 16 * 32 * NACC * ITERS * (double) sub_groups;
    return ops / secs / 1e12;
}

/// Peak bf16 XMX throughput in TFLOPS, from the 8x16x16 bf16 combination.
double measure_xmx_bf16(int sub_groups) {
    sycl::queue & q = vla::bitvla_sycl_queue(nullptr);

    auto * out = (float *) vla_dev_malloc((size_t) sub_groups * NACC * 8 * 16 * sizeof(float));
    if (!out) {
        return 0.0;
    }

    const size_t wg      = 128;
    const size_t threads = (size_t) sub_groups * SG;
    const double secs    = best_of(5, 2, [&] {
        q.parallel_for<k_xmx_bf>(
            sycl::nd_range<1>(sycl::range<1>(threads), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = it.get_sub_group();

                joint_matrix<sycl::sub_group, bf16, use::a, 8, 16, layout::row_major> a;
                joint_matrix<sycl::sub_group, bf16, use::b, 16, 16, layout::ext_intel_packed> b;
                joint_matrix<sycl::sub_group, float, use::accumulator, 8, 16> c[NACC];

                joint_matrix_fill(sg, a, bf16(1.0f));
                joint_matrix_fill(sg, b, bf16(1.0f));
                for (int i = 0; i < NACC; ++i) {
                    joint_matrix_fill(sg, c[i], 0.0f);
                }

                for (int t = 0; t < ITERS; ++t) {
#pragma unroll
                    for (int i = 0; i < NACC; ++i) {
                        joint_matrix_mad(sg, c[i], a, b, c[i]);
                    }
                }

                const size_t base = (size_t) (it.get_global_linear_id() / SG) * NACC * 8 * 16;
                for (int i = 0; i < NACC; ++i) {
                    joint_matrix_store(sg, c[i],
                                       sycl::address_space_cast<sycl::access::address_space::global_space,
                                                                sycl::access::decorated::no>(
                                           out + base + (size_t) i * 8 * 16),
                                       16, layout::row_major);
                }
            });
    });

    vla_dev_free(out);
    const double ops = 2.0 * 8 * 16 * 16 * NACC * ITERS * (double) sub_groups;
    return ops / secs / 1e12;
}

// ---------------------------------------------------------------------------
// the shapes BitVLA actually dispatches
// ---------------------------------------------------------------------------

struct Shape {
    int          M, N, K;
    const char * stage;  ///< decode | prefill | vit
    const char * name;
};

/// Measured, not assumed: this is the histogram VLA_BITVLA_LOG_SHAPES prints for
/// one `vla-bench -hf vrfai/bitvla-libero-gguf --images 1 --size 224`, with
/// @c calls the number of dispatches per request.
///
/// An earlier version of this table was transcribed from the dispatch chain in
/// bitnet_kernels.cu and had M of 1, 3, 5, 7, 9, 17 and 61. Every one of those
/// was wrong. That chain is a superset written for BitNet LLM decoding, and this
/// engine decodes nothing incrementally - both drivers project the whole
/// sequence in a single call, so M is the sequence length and never small. The
/// consequence was not a cosmetic one: at M=1 these shapes are bandwidth-bound
/// and the obvious win is to stop unpacking the weights, while at M=256 they are
/// compute-bound and unpacking to int8 for XMX is the right thing to do. A
/// packed M=1 GEMV was built on the strength of the wrong table, and it is
/// correct, bit-exact and 1.39x faster on shapes the engine does not issue.
const Shape SHAPES[] = {
    // LM, 30 layers x 330 tokens (256 image + text + action chunk).
    { 330,  2560, 2560, "lm",  "q_o"     },  // x60: q and o projections
    { 330,   640, 2560, "lm",  "kv"      },  // x60: k and v, 5 kv heads x 128
    { 330, 13824, 2560, "lm",  "gate_up" },  // x30
    { 330,  2560, 6912, "lm",  "down"    },  // x30

    // ViT, 26 layers x 256 patches.
    { 256,  1152, 1152, "vit", "qkvo"    },  // x104
    { 256,  4304, 1152, "vit", "fc1"     },  // x26
    { 256,  1152, 4352, "vit", "fc2"     },  // x26
};

/// Dispatches per request, index-matched to @ref SHAPES, so the bench can report
/// where the time goes rather than only how fast each shape is. A shape that is
/// twice as slow but issued a tenth as often is not the one to work on.
const int SHAPE_CALLS[] = { 60, 60, 30, 30, 104, 26, 26 };

static_assert(sizeof(SHAPE_CALLS) / sizeof(*SHAPE_CALLS) == sizeof(SHAPES) / sizeof(*SHAPES),
              "every shape needs its dispatch count");

struct Result {
    Shape  sh;
    double ms;         ///< one call, synced - latency
    double ms_pipe;    ///< amortised over a back-to-back burst - throughput
    double tops;
    double tops_pipe;
    double ai_int8;    ///< arithmetic intensity if the weights are unpacked int8
    double ai_packed;  ///< ... and if they stay 2-bit
    bool   packed;     ///< which of those two this shape actually dispatched to
    int    copies;     ///< distinct weight buffers rotated to defeat the L2
};

/// @brief Whether a shape reaches the packed GEMV rather than the oneDNN path.
///
/// Mirrors the dispatch in @c bitnet_sycl.cpp - M=1 goes to
/// @c ternary_gemv_packed, everything else unpacks to int8 for oneDNN. It has to
/// be mirrored because the two paths move different numbers of bytes, so a
/// single traffic model would report one of them against a roof that is not its
/// own. Charging M=1 the unpacked traffic it no longer moves is what produced a
/// "117% of ceiling" row, which is not a good result, it is a wrong denominator.
bool uses_packed_path(const Shape & sh) { return sh.M == 1; }

/// Bytes of DRAM traffic for one call, counting each operand once.
///
/// @p w_bytes_per_val is 1 for the unpacked oneDNN path and 0.25 for the packed
/// kernel. The activation and output terms are the same either way and at M=1
/// they are noise next to N*K, which is the whole point.
double traffic(const Shape & sh, double w_bytes_per_val) {
    const double weights = (double) sh.N * sh.K * w_bytes_per_val;
    const double act     = (double) sh.M * sh.K;      // int8
    const double out     = (double) sh.M * sh.N * 2;  // bf16
    return weights + act + out;
}

/// Device-resident weight bytes to cycle through, chosen to exceed this part's
/// L2 by more than an order of magnitude. The sweep in main() puts L2 somewhere
/// between 16 and 64 MiB.
constexpr size_t COLD_SET_BYTES = 384ull << 20;

Result time_shape(const Shape & sh) {
    const int ws_num = (sh.N == 3840 && sh.K == 2560) ? 3 : (sh.N == 13824 && sh.K == 2560) ? 2 : 1;

    // Timing one weight matrix repeatedly measures it from L2, which is not
    // where BitVLA reads its weights: the LM alone unpacks to ~2 GB, so every
    // matrix arrives cold, once per token. Rotating over enough distinct copies
    // to blow the cache between reuses restores that. Without this, decode
    // gate_up measured 845 GB/s of weight traffic against a 600 GB/s bus and
    // duly reported 142% of its own roofline ceiling.
    const size_t unpacked = (size_t) sh.N * sh.K;  // what the oneDNN path streams
    const int    copies   = (int) std::min<size_t>(24, std::max<size_t>(2, (COLD_SET_BYTES + unpacked - 1) / unpacked));

    auto * A   = (int8_t *) vla_dev_malloc((size_t) sh.M * sh.K);
    auto * out = (vla_bf16 *) vla_dev_malloc((size_t) sh.M * sh.N * sizeof(vla_bf16));
    auto * s   = (float *) vla_dev_malloc((size_t) sh.M * sizeof(float));
    auto * ws  = (float *) vla_dev_malloc((size_t) ws_num * sizeof(float));

    std::vector<int8_t *> B(copies, nullptr);
    bool                  ok = A && out && s && ws;
    for (int i = 0; i < copies && ok; ++i) {
        B[i] = (int8_t *) vla_dev_malloc(unpacked / 4);
        ok   = ok && B[i];
    }

    Result r{ sh, 0, 0, 0, 0, 0, 0, false, copies };
    if (!ok) {
        std::fprintf(stderr, "alloc failed for %s %s\n", sh.stage, sh.name);
    } else {
        // Values are irrelevant to timing, but s must not be zero: the epilogue
        // divides by it, and a denormal result would be a different kernel to
        // the one that runs in production.
        std::vector<float> ones_m((size_t) sh.M, 1.0f);
        std::vector<float> ones_w((size_t) ws_num, 1.0f);
        fill_random(A, (size_t) sh.M * sh.K);
        for (int i = 0; i < copies; ++i) {
            fill_random(B[i], unpacked / 4);
        }
        vla_dev_memcpy_h2d(s, ones_m.data(), ones_m.size() * sizeof(float));
        vla_dev_memcpy_h2d(ws, ones_w.data(), ones_w.size() * sizeof(float));

        int  turn = 0;
        auto call = [&] {
            int8_t * b = B[turn++ % copies];
            if (sh.M == 1) {
                bitlinear_int8xint2(A, b, out, s, ws, sh.M, sh.N, sh.K, nullptr);
            } else {
                bitlinear_int8xint2_m(A, b, out, s, ws, sh.M, sh.N, sh.K, nullptr);
            }
        };
        // Each copy is unpacked and cached on its first call, so every copy must
        // be touched during warmup or that one-off cost lands in a timed rep.
        const double secs = best_of(4 * copies, copies + 2, call);

        // The same calls issued back-to-back with one sync at the end, which is
        // what the model actually does: a decode step fires ~120 of these into an
        // in-order queue and looks at the result once. Syncing after every call,
        // as above, adds a full round-trip to each and measures latency. The gap
        // between the two is the part of the per-call overhead the queue can hide
        // on its own - and it is not worth optimising what is already hidden.
        const double burst = best_of(4, 2, [&] {
            for (int i = 0; i < copies; ++i) {
                call();
            }
        }) / copies;

        r.ms        = secs * 1e3;
        r.ms_pipe   = burst * 1e3;
        r.tops      = 2.0 * sh.M * sh.N * sh.K / secs / 1e12;
        r.tops_pipe = 2.0 * sh.M * sh.N * sh.K / burst / 1e12;
        r.ai_int8   = 2.0 * sh.M * sh.N * sh.K / traffic(sh, 1.0);
        r.ai_packed = 2.0 * sh.M * sh.N * sh.K / traffic(sh, 0.25);
        r.packed    = uses_packed_path(sh);
    }

    vla_dev_free(A);
    for (int8_t * b : B) {
        vla_dev_free(b);
    }
    vla_dev_free(out);
    vla_dev_free(s);
    vla_dev_free(ws);
    return r;
}

}  // namespace

int main(int argc, char ** argv) {
    // Both spellings, and anything else is an error rather than a shrug. The
    // loop used to accept only "--csv path": "--csv=path" was dropped in silence
    // and the run exited 0 having written nothing, so a stale CSV got plotted
    // and read as current. A bench that silently ignores its arguments is a
    // bench that silently reports the wrong measurement.
    const char * csv_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (std::strncmp(argv[i], "--csv=", 6) == 0) {
            csv_path = argv[i] + 6;
        } else {
            std::fprintf(stderr, "usage: %s [--csv PATH]\nunrecognised argument: %s\n", argv[0],
                         argv[i]);
            return 2;
        }
    }

    if (vla_dev_count() <= 0) {
        std::printf("no GPU device, skipping\n");
        return 0;
    }
    if (vla_dev_set(0) != 0) {
        std::fprintf(stderr, "vla_dev_set(0) failed: %s\n", vla_dev_error());
        return 1;
    }

    sycl::queue & q   = vla::bitvla_sycl_queue(nullptr);
    const auto    dev = q.get_device();
    const int     eus = (int) dev.get_info<sycl::info::device::max_compute_units>();
    const int     mhz = (int) dev.get_info<sycl::info::device::max_clock_frequency>();

    std::printf("device : %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("         %d XVE @ %d MHz, %zu MiB\n", eus, mhz,
                (size_t) (dev.get_info<sycl::info::device::global_mem_size>() >> 20));

    // --- ceilings ----------------------------------------------------------
    std::printf("\n=== measured ceilings ===\n");
    const double bw_read = measure_read_bw(1ull << 30);
    const double bw_copy = measure_copy_bw(1ull << 29);
    // The same probe over data drawn from the alphabet the unpack kernel emits.
    // Xe compresses memory surfaces losslessly, and unpacked ternary weights hold
    // four distinct byte values, so they cost less bus time than their size. That
    // is not an artefact of this bench - bitnet_sycl.cpp's unpacked_weights()
    // writes exactly this alphabet and oneDNN streams exactly this buffer - so it
    // is the honest ceiling for the unpacked path, and ignoring it is what made
    // gate_up report 128% of a roofline it cannot exceed.
    const double bw_tern = measure_read_bw(1ull << 30, /*ternary=*/true);
    // 4x the XVE count, so every thread slot is contended and the measurement
    // is of issue rate rather than of occupancy.
    const double tops_i8 = measure_xmx_int8(eus * 4);
    const double tf_bf16 = measure_xmx_bf16(eus * 4);
    const double launch  = measure_launch_overhead();

    std::printf("  DRAM read           %8.1f GB/s\n", bw_read);
    std::printf("  DRAM read (ternary) %8.1f GB/s   (effective, Xe compression)\n", bw_tern);
    std::printf("  DRAM copy (r+w)     %8.1f GB/s\n", bw_copy);
    std::printf("  XMX int8            %8.1f TOPS\n", tops_i8);
    std::printf("  XMX bf16            %8.1f TFLOPS\n", tf_bf16);
    std::printf("  launch round-trip   %8.1f us    (floor under every row below)\n", launch * 1e6);

    // A read ceiling is only a DRAM ceiling if it stops improving once the
    // working set leaves cache. Battlemage has a large L2, and BitVLA's
    // smallest weight matrix is 1.3 MB, so this is not a hypothetical: if the
    // curve is still climbing at the top of the sweep the headline figure is
    // measuring cache and every "% of ceiling" below it is wrong.
    std::printf("\n  read bandwidth vs working set (cache check)\n");
    for (size_t mib : { 4, 16, 64, 256, 1024, 4096 }) {
        std::printf("    %6zu MiB        %8.1f GB/s\n", mib, measure_read_bw(mib << 20));
    }

    // Two different denominators, because the two paths stream two different
    // kinds of bytes. The unpacked path streams the compressible ternary
    // alphabet; the packed path streams 2-bit codes, four to a byte, which is
    // dense and has nothing left for the compressor to find - so it is held to
    // the incompressible figure. Using the compressed rate for both would flatter
    // Stage 4 with an assist it will not get.
    const double bw_unpacked = std::max(bw_tern, bw_read);
    const double bw_packed   = bw_read;

    // --- shapes ------------------------------------------------------------
    std::printf("\n=== ternary GEMM (M=1 packed GEMV, M>1 oneDNN on unpacked int8) ===\n");
    std::printf("%-8s %-8s %-18s %5s %9s %9s %9s %8s %5s %9s %9s\n", "stage", "op", "MxNxK", "path",
                "ms", "ms-pipe", "TOPS-pipe", "AI", "cold", "%ceil", "%ceil-pk");

    std::vector<Result> results;
    for (const auto & sh : SHAPES) {
        const Result r = time_shape(sh);
        results.push_back(r);

        // Roofline: a kernel at intensity AI cannot exceed AI*BW, and nothing
        // exceeds the XMX peak. Which of the two binds is the finding.
        //
        // Each path against its own roof. The packed one moves incompressible
        // 2-bit codes; the unpacked one moves four byte values that Xe's lossless
        // compression sees straight through, so it gets the higher rate. Rows
        // already on the packed path have the two columns coincide - the second
        // is the remaining Stage 4 opportunity, and for them there is none.
        const double ai_now   = r.packed ? r.ai_packed : r.ai_int8;
        const double bw_now   = r.packed ? bw_packed : bw_unpacked;
        const double ceil_now = std::min(tops_i8, ai_now * bw_now / 1e3);
        const double ceil_pk  = std::min(tops_i8, r.ai_packed * bw_packed / 1e3);

        // Reported against the pipelined figure: that is the one the model gets.
        char dims[32];
        std::snprintf(dims, sizeof(dims), "%dx%dx%d", sh.M, sh.N, sh.K);
        std::printf("%-8s %-8s %-18s %5s %9.4f %9.4f %9.2f %8.2f %5d %8.1f%% %8.1f%%\n", sh.stage,
                    sh.name, dims, r.packed ? "pack" : "dnn", r.ms, r.ms_pipe, r.tops_pipe, ai_now,
                    r.copies, 100.0 * r.tops_pipe / ceil_now, 100.0 * r.tops_pipe / ceil_pk);
    }

    // --- where the request actually goes -----------------------------------
    // Per-shape percentages say how well each kernel runs; this says which of
    // them is worth running better. They are not the same ranking.
    double budget = 0.0;
    for (size_t i = 0; i < results.size(); ++i) {
        budget += results[i].ms_pipe * SHAPE_CALLS[i];
    }
    std::printf("\n=== ternary GEMM share of one request ===\n");
    std::printf("%-8s %-8s %6s %9s %9s\n", "stage", "op", "calls", "ms", "% of GEMM");
    for (size_t i = 0; i < results.size(); ++i) {
        const double ms = results[i].ms_pipe * SHAPE_CALLS[i];
        std::printf("%-8s %-8s %6d %9.3f %8.1f%%\n", results[i].sh.stage, results[i].sh.name,
                    SHAPE_CALLS[i], ms, 100.0 * ms / budget);
    }
    std::printf("%-8s %-8s %6s %9.3f\n", "total", "", "", budget);

    if (csv_path) {
        FILE * f = std::fopen(csv_path, "w");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", csv_path);
            return 1;
        }
        // device= gets a line to itself: the name has spaces in it, and the
        // reader splits the multi-key lines on whitespace.
        std::fprintf(f, "# device=%s\n", dev.get_info<sycl::info::device::name>().c_str());
        std::fprintf(f, "# xve=%d mhz=%d launch_us=%.3f\n", eus, mhz, launch * 1e6);
        std::fprintf(f,
                     "# bw_read_GBs=%.3f bw_ternary_GBs=%.3f bw_copy_GBs=%.3f "
                     "xmx_int8_TOPS=%.3f xmx_bf16_TFLOPS=%.3f\n",
                     bw_read, bw_tern, bw_copy, tops_i8, tf_bf16);
        std::fprintf(f, "stage,op,M,N,K,path,ms,ms_pipe,tops,tops_pipe,ai_int8,ai_packed,copies\n");
        for (const auto & r : results) {
            std::fprintf(f, "%s,%s,%d,%d,%d,%s,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%d\n", r.sh.stage,
                         r.sh.name, r.sh.M, r.sh.N, r.sh.K, r.packed ? "packed" : "onednn", r.ms,
                         r.ms_pipe, r.tops, r.tops_pipe, r.ai_int8, r.ai_packed, r.copies);
        }
        std::fclose(f);
        std::printf("\nwrote %s\n", csv_path);
    }

    return 0;
}
