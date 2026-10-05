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
 * @file bitnet_sycl.cpp
 * @brief BitVLA's ternary GEMM on Intel GPUs, oneDNN reference path.
 *
 * Implements the same three @c extern @c "C" entry points as
 * @c bitnet_kernels.cu - @c bitlinear_int8xint2 (M=1),
 * @c bitlinear_int8xint2_m (M>1) and @c bitvla_act_quant_cuda - so the LM, ViT
 * and action-head drivers are shared between backends.
 *
 * ## Why a reference path exists at all
 *
 * The CUDA kernels keep BitVLA's weights in their 2-bit pack all the way into
 * the tensor core, decoding 16 ternary values per 32-bit word with @c lop3.b32
 * on the way to a @c wmma fragment. The Xe equivalent - a @c joint_matrix DPAS
 * kernel over the same pack - is the shipping path and lives in the Stage 4
 * file. This one does the dumb thing instead: unpack the weights to row-major
 * s8 once at first use, and hand them to @c dnnl::matmul.
 *
 * It costs 4x the weight footprint (~5.8 GB for BitVLA-libero, against 32 GB of
 * B70 memory) and reads 4x the bytes per GEMM, so it is not what ships. What it
 * is, is an oracle. int32 accumulation is exact and order-independent, so
 * oneDNN's answer and a hand-written DPAS kernel's answer must agree *bit for
 * bit* - not to a tolerance. That turns "is the new kernel right" from a
 * judgement call into a comparison, and it is why the unpack is unit-tested
 * separately (@c tests/test_ladder_pack.cpp) rather than trusted.
 *
 * Select it at runtime with @c VLA_BITVLA_ONEDNN_GEMM=1, mirroring the existing
 * @c VLA_BITVLA_NARROW_GEMM bisect flag on the CUDA side.
 *
 * ## Arithmetic
 *
 * Bit-exact against CUDA by construction, not by tolerance. Both sides form the
 * same int32 dot product - the summation order cannot matter, because int32
 * addition is associative and the values cannot overflow (K <= 27648 terms of
 * at most 127*1 each, so |acc| <= 3.5e6, against a 2.1e9 range). The epilogue
 * is then the same three f32 operations in the same order, rounded once to bf16
 * by the same RNE convert.
 */

#include "env_flag.h"
#include "kernels/bitvla/device.h"
#include "kernels/bitvla/ladder_pack.h"
#include "kernels/bitvla/sycl/bitnet_gemv_sycl.h"
#include "kernels/bitvla/sycl/queue_sycl.h"
#include "kernels/bitvla/sycl/sycl_compat.h"
#include "kernels/bitvla/sycl/weight_cache.h"

#include "kernels/bitvla/sycl/dnnl_sycl.h"

#include <array>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

using vla::bitvla::as_bf;
using vla::bitvla::bf16;
using vla::bitvla::exact_div;
using vla::bitvla::ladder_slot_addr;
using vla::bitvla::to_bf16;
using vla::bitvla::to_f32;

namespace {

struct k_act_quant {};
struct k_unpack {};
struct k_epilogue {};
struct k_ws_expand {};

/**
 * @brief @c VLA_BITVLA_LOG_SHAPES=1 - histogram of every GEMM the engine issues.
 *
 * Worth carrying permanently. The shape table in @c bench_bitvla_bmg.cpp and
 * @c test_bitvla_gemm_gpu.cpp was transcribed from the CUDA dispatch chain in
 * @c bitnet_kernels.cu, which is a superset written for BitNet LLM decoding, and
 * its M=1 rows are shapes *this* engine never issues: both drivers project the
 * whole sequence in one call, so M is always the sequence length. A packed M=1
 * GEMV was written, proved bit-exact and measured 1.39x faster before an A/B run
 * showed end-to-end unchanged, because nothing called it. Guessing the workload
 * from the kernel's own dispatch table is the mistake this exists to prevent.
 */
struct ShapeLog {
    std::map<std::array<int, 3>, long> hits;
    std::mutex                         mu;

    void add(int M, int N, int K) {
        std::lock_guard<std::mutex> lock(mu);
        ++hits[{ M, N, K }];
    }

    ~ShapeLog() {
        if (hits.empty()) return;
        std::fprintf(stderr, "\nvla(bitvla): ternary GEMM shapes actually dispatched\n");
        std::fprintf(stderr, "  %6s %6s %6s %10s\n", "M", "N", "K", "calls");
        for (const auto & kv : hits) {
            std::fprintf(stderr, "  %6d %6d %6d %10ld\n", kv.first[0], kv.first[1], kv.first[2],
                         kv.second);
        }
    }
};

ShapeLog & shape_log() {
    static ShapeLog log;
    return log;
}

[[noreturn]] void unsupported_shape(const char * which, int M, int N, int K) {
    std::fprintf(stderr,
                 "vla(bitvla): %s has no registered shape (M=%d, N=%d, K=%d). "
                 "Add it to the ws_num table in bitnet_sycl.cpp and rebuild.\n",
                 which, M, N, K);
    std::abort();
}

// --- how many output columns share one weight scale -------------------------

/**
 * @brief @c ws_num for a weight shape: the number of column groups in @p ws.
 *
 * On the CUDA side this is a template parameter, fixed by which branch of the
 * dispatch chain a shape lands in. Here the shape is only known at run time, so
 * the same information has to be a table - and it has to be a *table*, not a
 * default: ws_num is 3 for a fused QKV projection and 2 for a fused gate+up
 * because those tensors were quantised per sub-matrix, and silently assuming 1
 * would apply one projection's absmean to all three. The result would still be
 * finite, plausible actions, which is exactly the bug worth refusing to have.
 *
 * The union of both CUDA dispatch chains (bitnet_kernels.cu), so a shape that
 * runs on CUDA runs here.
 */
int ws_num_for(int N, int K) {
    struct Entry { int N, K, ws_num; };
    static const Entry table[] = {
        // BitVLA-libero LM: fused QKV, q, fused gate+up, down, and the separate
        // k/v projections the M>1 path uses instead of the fusion.
        { 3840,  2560, 3},
        { 2560,  2560, 1},
        {13824,  2560, 2},
        { 2560,  6912, 1},
        {  640,  2560, 1},
        // BitVLA ViT.
        { 1152,  1152, 1},
        { 4304,  1152, 1},
        { 1152,  4352, 1},
        // The larger BitVLA the CUDA M=1 chain also carries.
        { 4800,  3200, 6},
        { 3200,  3200, 1},
        {20480,  3200, 2},
        { 3200, 10240, 1},
        { 5120, 27648, 1},
        {55296,  5120, 1},
    };
    for (const auto & e : table)
        if (e.N == N && e.K == K) return e.ws_num;
    return 0;
}

// --- unpacked weight cache --------------------------------------------------

/// The cache and its lock, shared with vla::bitvla_forget_unpacked below.
std::mutex &                                cache_mutex() { static std::mutex m; return m; }
std::map<const void *, int8_t *> &          weight_cache() {
    static std::map<const void *, int8_t *> c;
    return c;
}

/// Per-output-column weight scales, keyed on the grouped @c ws they came from.
/// Same pointer-identity contract and the same eviction hook as the weight
/// cache, for the same reason.
std::map<const void *, float *> & ws_cache() {
    static std::map<const void *, float *> c;
    return c;
}

/// Concatenated [N0+N1+N2, K] weights, their per-column scales and bias, for
/// bitvla_ternary_gemm_cat3, keyed on the first packed pointer. Same eviction
/// contract as the caches above: forgetting any one of the three source
/// pointers drops the entry.
struct Cat3 {
    const void * src[3];
    int8_t *     w     = nullptr;
    float *      ws    = nullptr;
    float *      bias  = nullptr;
};
std::map<const void *, Cat3> & cat3_cache() {
    static std::map<const void *, Cat3> c;
    return c;
}

/// Blocked copies of the unpacked weights, keyed on the packed pointer and the
/// layout. Evicted with the rest by bitvla_forget_unpacked.
std::map<const void *, std::vector<std::pair<dnnl::memory::desc, int8_t *>>> & blocked_cache() {
    static std::map<const void *, std::vector<std::pair<dnnl::memory::desc, int8_t *>>> c;
    return c;
}

/**
 * @brief @p ws broadcast from @p ws_num groups to one scale per output column.
 *
 * The hand-written epilogue indexes @c ws[n / per_group] and needs nothing more.
 * A oneDNN binary post-op cannot do that indexing - it broadcasts a dimension or
 * matches it - so the fused path needs the grouped vector materialised at length
 * N. It is N floats per weight tensor, built once, against weight matrices of
 * N*K bytes; the largest here is 55 KB against 35 MB.
 *
 * Sound only because @c ws is written at model load and never again. The values
 * are absmean quantisation constants baked into the checkpoint, not anything a
 * forward pass recomputes.
 */
float * expanded_ws(const float * ws, int N, int ws_num, sycl::queue & q) {
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto &                      cache = ws_cache();
    auto                        it    = cache.find(ws);
    if (it != cache.end()) return it->second;

    float * full = (float *) vla_dev_malloc((size_t) N * sizeof(float));
    if (!full) {
        std::fprintf(stderr, "vla(bitvla): could not expand weight scales to %d columns (%s)\n", N,
                     vla_dev_error());
        std::abort();
    }

    const int per_group = N / ws_num;
    q.parallel_for<k_ws_expand>(sycl::range<1>((size_t) N),
                                [=](sycl::id<1> id) { full[id[0]] = ws[id[0] / per_group]; });
    q.wait();

    cache.emplace(ws, full);
    return full;
}

/**
 * @brief Row-major s8 copy of a ladder-packed weight matrix, unpacked once.
 *
 * Keyed on the packed device pointer, so the first GEMM of a given projection
 * pays for the unpack and the rest are lookups. The key is only valid while the
 * pointer is - @c vla_dev_free calls @ref vla::bitvla_forget_unpacked to evict,
 * because the allocator will happily hand the same address to the next
 * allocation and a stale hit would be undetectable in the output.
 */
/// Unpack a ladder-packed (N, K) matrix into row-major s8 at @p dst, which
/// may be a row offset into a larger buffer (the concatenated QKV weights).
void unpack_into(const int8_t * packed, int N, int K, int8_t * w, sycl::queue & q) {
    const int64_t n_slots = (int64_t) N * K / 16;
    const int64_t K64     = K;
    q.parallel_for<k_unpack>(sycl::range<1>((size_t) n_slots), [=](sycl::id<1> id) {
        const int64_t slot = (int64_t) id[0];
        int64_t       n_global, k_base;
        ladder_slot_addr(slot, K64, n_global, k_base);

        // The four bytes of a slot hold the 16 values transposed: value t is in
        // byte t%4 at bit pair t/4. Reading the bytes once and shifting beats
        // four strided byte loads.
        const uint8_t b0 = (uint8_t) packed[slot * 4 + 0];
        const uint8_t b1 = (uint8_t) packed[slot * 4 + 1];
        const uint8_t b2 = (uint8_t) packed[slot * 4 + 2];
        const uint8_t b3 = (uint8_t) packed[slot * 4 + 3];
        const uint8_t bs[4] = {b0, b1, b2, b3};

        int8_t * dst = w + n_global * K64 + k_base;
        for (int t = 0; t < 16; ++t)
            dst[t] = (int8_t) ((int) ((bs[t % 4] >> (2 * (t / 4))) & 0x3u) - 2);
    });
}

int8_t * unpacked_weights(const int8_t * packed, int N, int K, sycl::queue & q) {
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto &                      cache = weight_cache();
    auto                        it    = cache.find(packed);
    if (it != cache.end()) return it->second;

    int8_t * w = (int8_t *) vla_dev_malloc((size_t) N * K);
    if (!w) {
        std::fprintf(stderr,
                     "vla(bitvla): could not allocate %lld bytes to unpack a %dx%d weight "
                     "matrix (%s). The oneDNN reference path needs 4x the packed footprint; "
                     "unset VLA_BITVLA_ONEDNN_GEMM to use the packed kernels.\n",
                     (long long) N * K, N, K, vla_dev_error());
        std::abort();
    }
    unpack_into(packed, N, K, w, q);
    q.wait();

    cache.emplace(packed, w);
    return w;
}

}  // namespace

void vla::bitvla_forget_unpacked(const void * packed) {
    if (!packed) return;

    // A pointer can be a packed weight matrix or a grouped ws vector - never
    // both - so this looks in each cache and releases whatever it finds.
    int8_t * unpacked = nullptr;
    float *  scales   = nullptr;
    Cat3     cat{};
    std::vector<std::pair<dnnl::memory::desc, int8_t *>> blocked, cat_blocked;
    {
        std::lock_guard<std::mutex> lock(cache_mutex());

        auto bit = blocked_cache().find(packed);
        if (bit != blocked_cache().end()) {
            blocked = std::move(bit->second);
            blocked_cache().erase(bit);
        }

        auto & cc = cat3_cache();
        for (auto cit = cc.begin(); cit != cc.end(); ++cit) {
            const Cat3 & e = cit->second;
            if (e.src[0] == packed || e.src[1] == packed || e.src[2] == packed) {
                cat = e;
                cc.erase(cit);
                auto cb = blocked_cache().find(cat.w);
                if (cb != blocked_cache().end()) {
                    cat_blocked = std::move(cb->second);
                    blocked_cache().erase(cb);
                }
                break;
            }
        }

        auto & cache = weight_cache();
        auto   it    = cache.find(packed);
        if (it != cache.end()) {
            unpacked = it->second;
            cache.erase(it);
        }

        auto & wsc = ws_cache();
        auto   wit = wsc.find(packed);
        if (wit != wsc.end()) {
            scales = wit->second;
            wsc.erase(wit);
        }
    }
    for (auto & e : blocked) vla_dev_free(e.second);
    if (cat.w) {
        for (auto & e : cat_blocked) vla_dev_free(e.second);
        vla_dev_free(cat.w);
        vla_dev_free(cat.ws);
        vla_dev_free(cat.bias);
    }
    if (!unpacked && !scales && blocked.empty()) return;

    // Deliberately outside the lock. vla_dev_free calls this function on the
    // pointer it is about to release, so freeing while holding a non-recursive
    // mutex is a self-deadlock - and one that only fires on the path this whole
    // mechanism exists for, the first eviction of a real cached weight. Erasing
    // before releasing also makes that re-entrant call a clean miss rather than
    // a second free of the same address.
    //
    // Freeing through the engine's own entry point, rather than sycl::free
    // directly, keeps the allocation and release symmetric.
    vla_dev_free(unpacked);
    vla_dev_free(scales);
}

namespace {

// --- s32 accumulator scratch ------------------------------------------------

/// Grown, never shrunk: the shapes repeat every layer, so it settles at the
/// largest one after the first forward pass.
int32_t * acc_scratch(size_t elems) {
    static int32_t * buf      = nullptr;
    static size_t    capacity = 0;
    if (elems > capacity) {
        vla_dev_free(buf);
        buf = (int32_t *) vla_dev_malloc(elems * sizeof(int32_t));
        if (!buf) {
            std::fprintf(stderr, "vla(bitvla): accumulator allocation failed (%s)\n",
                         vla_dev_error());
            std::abort();
        }
        capacity = elems;
    }
    return buf;
}

// --- weight layout autotuning ----------------------------------------------

/**
 * @brief A fused matmul built twice - once reading the unpacked weights in
 *        place, once against a blocked layout of oneDNN's choosing - and which
 *        of the two this device runs faster.
 *
 * The weights are fixed for the life of the model, so letting oneDNN pick their
 * layout (@c format_tag::any) and reordering them into it once costs nothing per
 * call. Whether it pays is a property of the silicon and the shape, not a
 * constant: measured on Panther Lake's Xe3 it nearly halves some GEMMs (the
 * 3840-wide QKV at M=330: 0.40 -> 0.23 ms; the ViT's fc1: 0.18 -> 0.09 ms) and
 * slows others by ~15% (the ViT's fc2). So each shape is timed both ways the
 * first time it runs and keeps the winner.
 *
 * Correctness does not depend on the choice: int32 accumulation is exact and
 * the epilogue is the same post-op chain, so both produce the same bits.
 * @c VLA_BITVLA_GEMM_LAYOUT=plain|blocked pins it.
 */
struct Tuned {
    dnnl::matmul       blocked;
    dnnl::memory::desc blocked_w;
    int                choice = -1;  ///< -1 undecided, 0 plain, 1 blocked
};

int8_t * blocked_weights(const void * key, const dnnl::memory::desc & wd, int8_t * plain,
                         const dnnl::memory::desc & plain_md, dnnl::engine & eng,
                         dnnl::stream & strm, sycl::queue & q) {
    {
        std::lock_guard<std::mutex> lock(cache_mutex());
        for (auto & e : blocked_cache()[key])
            if (e.first == wd) return e.second;
    }
    int8_t * w = (int8_t *) vla_dev_malloc(wd.get_size());
    if (!w) {
        std::fprintf(stderr, "vla(bitvla): blocked weight copy: allocation failed (%s)\n",
                     vla_dev_error());
        std::abort();
    }
    dnnl::memory src(plain_md, eng, plain), dst(wd, eng, w);
    dnnl::reorder(src, dst).execute(strm, src, dst);
    q.wait();
    std::lock_guard<std::mutex> lock(cache_mutex());
    blocked_cache()[key].emplace_back(wd, w);
    return w;
}

int layout_override() {
    static const int v = [] {
        const char * e = std::getenv("VLA_BITVLA_GEMM_LAYOUT");
        if (!e) return -1;
        if (std::string(e) == "plain") return 0;
        if (std::string(e) == "blocked") return 1;
        return -1;
    }();
    return v;
}

/**
 * @brief Execute @p plain or the tuned blocked variant, deciding on first use.
 * @param args everything but DNNL_ARG_WEIGHTS
 */
void exec_tuned(Tuned & t, const dnnl::matmul & plain, const dnnl::matmul::primitive_desc & any_pd_proto,
                const void * wkey, int8_t * W, const dnnl::memory::desc & b_md,
                std::unordered_map<int, dnnl::memory> args, vla_stream stream) {
    (void) any_pd_proto;
    dnnl::engine & eng  = vla::bitvla_dnnl_engine(stream);
    dnnl::stream & strm = vla::bitvla_dnnl_stream(stream);
    sycl::queue &  q    = vla::bitvla_sycl_queue(stream);

    dnnl::memory w_plain(b_md, eng, W);
    if (t.choice == 0 || !t.blocked) {
        args[DNNL_ARG_WEIGHTS] = w_plain;
        plain.execute(strm, args);
        return;
    }
    dnnl::memory w_blk(t.blocked_w, eng, blocked_weights(wkey, t.blocked_w, W, b_md, eng, strm, q));

    if (t.choice < 0) {
        const int forced = layout_override();
        if (forced >= 0) {
            t.choice = forced;
        } else {
            // Same work both ways, so the results are identical and it does not
            // matter that the real output is overwritten while timing.
            auto time = [&](const dnnl::matmul & p, const dnnl::memory & w) {
                args[DNNL_ARG_WEIGHTS] = w;
                p.execute(strm, args);
                q.wait();
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 3; ++i) p.execute(strm, args);
                q.wait();
                return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            };
            const double tp = time(plain, w_plain);
            const double tb = time(t.blocked, w_blk);
            t.choice        = tb < tp ? 1 : 0;
            if (vla::env_flag("VLA_BITVLA_LOG_SHAPES"))
                std::fprintf(stderr, "vla(bitvla): GEMM layout %s (plain %.1f us, blocked %.1f us)\n",
                             t.choice ? "blocked" : "plain", tp * 1e6 / 3, tb * 1e6 / 3);
        }
        if (t.choice == 0) {
            args[DNNL_ARG_WEIGHTS] = w_plain;
            plain.execute(strm, args);
            return;
        }
    }
    args[DNNL_ARG_WEIGHTS] = w_blk;
    t.blocked.execute(strm, args);
}

/// Build the blocked twin of a fused matmul. Leaves @p t.blocked empty if
/// oneDNN picks the plain layout anyway, which makes exec_tuned skip tuning.
void build_blocked(Tuned & t, const dnnl::engine & eng, const dnnl::memory::desc & a_md,
                   const dnnl::memory::desc & b_md, const dnnl::memory::desc & c_md,
                   const dnnl::primitive_attr & attr) {
    if (layout_override() == 0) return;
    const auto dims = b_md.get_dims();
    dnnl::memory::desc any(dims, dnnl::memory::data_type::s8, dnnl::memory::format_tag::any);
    try {
        dnnl::matmul::primitive_desc pd(eng, a_md, any, c_md, attr);
        if (pd.weights_desc() == b_md) return;
        t.blocked   = dnnl::matmul(pd);
        t.blocked_w = pd.weights_desc();
    } catch (const dnnl::error &) {
        // No blocked implementation for this shape: plain it is.
    }
}

// --- primitive cache --------------------------------------------------------

/// A built matmul and the descriptors its arguments must be bound with.
/// @c s_md and @c ws_md are only populated on the fused variant.
struct Plan {
    dnnl::memory::desc a_md, b_md, c_md, s_md, ws_md;
    dnnl::matmul       prim;
    Tuned              tuned;  ///< fused variant only: plain vs oneDNN-chosen weight layout
};

/**
 * @brief Whether to fold the epilogue into the matmul as post-ops.
 *
 * Worth a flag because it trades a property for a property. Unfused, the
 * epilogue is three f32 operations written out longhand in a kernel this file
 * owns, which is what makes the bit-exactness claim checkable by reading it.
 * Fused, the same three operations are oneDNN's to implement, and whether its
 * @c binary_div is the IEEE divide @c exact_div guarantees or a fast reciprocal
 * is not something the API promises either way.
 *
 * So it is not asserted, it is gated: @c tests/test_bitvla_gemm_gpu holds both
 * settings to the same bit-exact standard against host arithmetic, and the flag
 * exists to bisect if a future oneDNN changes its mind. Default on because it
 * passes today and saves a kernel launch per GEMM - which at these shapes is
 * 25-45% of the call.
 */
bool fuse_epilogue(int M, int N, int K) {
    static const bool on = !vla::env_flag("VLA_BITVLA_SPLIT_EPILOGUE");
    if (!on) return false;

    // One exception, and it is a mechanism rather than a tuning constant. A
    // post-op runs on the finished sum, so fusing forbids oneDNN from splitting
    // the K loop across work-groups - a split would need each partial result
    // scaled, which is not what the epilogue means. The unfused path has no such
    // constraint: its dst is s32, so a K-split can accumulate with integer
    // atomics and get parallelism that N alone does not provide.
    //
    // That only matters when N is too small to fill the machine and K is long
    // enough to be worth splitting, which of BitVLA's shapes is exactly the LM's
    // down projection at M=1 (2560 wide, 6912 deep). Measured on B70, fused vs
    // unfused, microseconds per call amortised over a back-to-back burst:
    //
    //     o_proj  1x2560x2560    9.8  vs 13.7     down  1x2560x6912  30.2 vs 27.4
    //     qkv     1x3840x2560   11.7  vs 15.5     down  5x2560x6912  25.3 vs 26.6
    //     gate_up 1x13824x2560  35.4  vs 40.3     qkvo 61x1152x1152   5.6 vs  8.3
    //
    // Fusing wins everywhere the reduction is no longer than the output is wide,
    // and loses by 10% where it is not. M>1 supplies its own parallelism and
    // never needs the split.
    return !(M == 1 && K > N);
}

/**
 * @brief The matmul for one (M, N, K), built once.
 *
 * oneDNN caches the generated kernel internally, but @c primitive_desc
 * construction still walks the implementation list and hashes a key on every
 * call, and a BitVLA decode step issues one of these per projection per layer -
 * 100+ times for a single token. Only a handful of distinct shapes ever appear
 * (the dispatch table above, crossed with the sequence lengths a run uses), so
 * the map stays small and stops growing after the first forward pass.
 */
Plan & plan_for(const dnnl::engine & eng, int M, int N, int K, bool fused) {
    using dt   = dnnl::memory::data_type;
    using dims = dnnl::memory::dims;

    struct Key {
        int  M, N, K;
        bool fused;
        bool operator<(const Key & o) const {
            if (M != o.M) return M < o.M;
            if (N != o.N) return N < o.N;
            if (K != o.K) return K < o.K;
            return fused < o.fused;
        }
    };
    static std::mutex          mu;
    static std::map<Key, Plan> cache;

    std::lock_guard<std::mutex> lock(mu);
    const Key                   key{M, N, K, fused};
    auto                        it = cache.find(key);
    if (it != cache.end()) return it->second;

    // W is (N, K) row-major and the GEMM wants B as (K, N). Rather than
    // transpose it, describe the same bytes with swapped strides - oneDNN takes
    // arbitrary strides, so B is W^T for free.
    dnnl::memory::desc a_md({M, K}, dt::s8, dims{K, 1});
    dnnl::memory::desc b_md({K, N}, dt::s8, dims{1, K});

    if (!fused) {
        dnnl::memory::desc c_md({M, N}, dt::s32, dims{N, 1});
        dnnl::matmul::primitive_desc pd(eng, a_md, b_md, c_md);
        return cache.emplace(key, Plan{a_md, b_md, c_md, {}, {}, dnnl::matmul(pd)}).first->second;
    }

    // Fused: dst is bf16 straight out, and the two scalings ride along as binary
    // post-ops in the order the CUDA epilogue applies them - divide by the
    // per-row activation scale, then multiply by the per-column weight scale.
    // Both operands are described with a broadcast dimension so oneDNN reads one
    // value per row and per column respectively rather than a full M*N plane.
    dnnl::memory::desc c_md({M, N}, dt::bf16, dims{N, 1});
    dnnl::memory::desc s_md({M, 1}, dt::f32, dims{1, 1});
    dnnl::memory::desc ws_md({1, N}, dt::f32, dims{N, 1});

    dnnl::post_ops po;
    po.append_binary(dnnl::algorithm::binary_div, s_md);
    po.append_binary(dnnl::algorithm::binary_mul, ws_md);

    dnnl::primitive_attr attr;
    attr.set_post_ops(po);

    dnnl::matmul::primitive_desc pd(eng, a_md, b_md, c_md, attr);
    Plan p{a_md, b_md, c_md, s_md, ws_md, dnnl::matmul(pd), {}};
    build_blocked(p.tuned, eng, a_md, b_md, c_md, attr);
    return cache.emplace(key, p).first->second;
}

// --- the GEMM ---------------------------------------------------------------

void ternary_matmul(const char * which, int8_t * A, int8_t * B_packed, vla_bf16 * out, float * s,
                    float * ws, int M, int N, int K, vla_stream stream) {
    const int ws_num = ws_num_for(N, K);
    if (ws_num == 0) unsupported_shape(which, M, N, K);

    static const bool log_shapes = vla::env_flag("VLA_BITVLA_LOG_SHAPES");
    if (log_shapes) shape_log().add(M, N, K);

    sycl::queue &  q   = vla::bitvla_sycl_queue(stream);

    // LM decode. Reads the 2-bit pack in place, so it never touches the unpacked
    // copy below and never allocates it. Bit-identical to the oneDNN path by
    // construction (int32 accumulation, one rounding), and the env flag is here
    // to bisect that claim rather than to choose a policy.
    if (M == 1 && !vla::env_flag("VLA_BITVLA_ONEDNN_GEMM") &&
        vla::bitvla::ternary_gemv_packed(A, B_packed, out, s, ws, N, K, ws_num, stream)) {
        return;
    }

    dnnl::engine & eng = vla::bitvla_dnnl_engine(stream);

    int8_t *     W     = unpacked_weights(B_packed, N, K, q);
    const bool   fused = fuse_epilogue(M, N, K);
    Plan &       plan  = plan_for(eng, M, N, K, fused);

    dnnl::memory a_mem(plan.a_md, eng, A);
    dnnl::memory b_mem(plan.b_md, eng, W);

    if (fused) {
        // One submission, and no M*N int32 plane written only to be read back by
        // the next kernel. The accumulator stays inside the matmul.
        dnnl::memory c_mem(plan.c_md, eng, out);
        dnnl::memory s_mem(plan.s_md, eng, s);
        dnnl::memory ws_mem(plan.ws_md, eng, expanded_ws(ws, N, ws_num, q));

        exec_tuned(plan.tuned, plan.prim, {}, B_packed, W, plan.b_md,
                   {{DNNL_ARG_SRC, a_mem},
                    {DNNL_ARG_DST, c_mem},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, s_mem},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, ws_mem}},
                   stream);
        return;
    }

    int32_t *    acc = acc_scratch((size_t) M * N);
    dnnl::memory c_mem(plan.c_md, eng, acc);

    plan.prim.execute(vla::bitvla_dnnl_stream(stream),
                      {{DNNL_ARG_SRC, a_mem}, {DNNL_ARG_WEIGHTS, b_mem}, {DNNL_ARG_DST, c_mem}});

    // Epilogue, separate rather than a post-op: the CUDA kernels compute
    // (float)acc / s[m] * ws[g] in that order and round once, and writing it out
    // longhand is what makes the bit-exactness claim checkable by reading.
    const int      per_group = N / ws_num;
    bf16 *         o         = as_bf(out);
    const int32_t * acc_ro   = acc;
    q.parallel_for<k_epilogue>(
        sycl::nd_range<2>(sycl::range<2>((size_t) M, vla::bitvla::round_up((size_t) N, 256)),
                          sycl::range<2>(1, 256)),
        [=](sycl::nd_item<2> it) {
            const int m = (int) it.get_group(0);
            const int n = (int) it.get_global_id(1);
            if (n >= N) return;
            o[(size_t) m * N + n] =
                to_bf16(exact_div((float) acc_ro[(size_t) m * N + n], s[m]) * ws[n / per_group]);
        });
}

}  // namespace

extern "C" void bitlinear_int8xint2(int8_t* input0, int8_t* input1, vla_bf16* output0,
                                    float* s, float* ws, int M, int N, int K,
                                    vla_stream stream) {
    ternary_matmul("bitlinear_int8xint2", input0, input1, output0, s, ws, M, N, K, stream);
}

extern "C" void bitlinear_int8xint2_m(int8_t* input0, int8_t* input1, vla_bf16* output0,
                                      float* s, float* ws, int M, int N, int K,
                                      vla_stream stream) {
    ternary_matmul("bitlinear_int8xint2_m", input0, input1, output0, s, ws, M, N, K, stream);
}

/**
 * @brief Row-wise absmax int8 quantisation of a bf16 activation matrix.
 *
 * A literal translation of @c act_quant_kernel. Two details carry over and both
 * matter: the 1e-5 floor on the row maximum, which keeps an all-zero row from
 * producing an infinite scale, and @c rint's round-half-to-even, which is what
 * CUDA's @c nearbyintf does under the default rounding mode. Rounding halves
 * away from zero instead would shift roughly one value in 256 by one int8 step,
 * and the GEMM below it is exact, so that error would arrive intact at the
 * output.
 *
 * @p K_out widens the output row and zero-fills the tail, which lets the ViT's
 * fc1 output land straight in the padded stride fc2 needs. See the CUDA twin in
 * @c bitnet_kernels.h for why that is worth doing and why it is bit-identical:
 * in short, the restride used to be an @c ext_oneapi_memcpy2d, Level Zero has no
 * native 2D copy behind it, and the byte-wise fallback kernel cost 4.4 ms of a
 * 27.2 ms request at 6% of achievable bandwidth.
 */
extern "C" void bitvla_act_quant_pad_cuda(const vla_bf16* in, int8_t* out, float* scales,
                                          int M, int K_in, int K_out, vla_stream stream) {
    if (M <= 0 || K_in <= 0 || K_out < K_in) return;
    constexpr int WG = 256;
    const int K = K_in;

    sycl::queue & q  = vla::bitvla_sycl_queue(stream);
    const bf16 *  is = as_bf(in);
    q.parallel_for<k_act_quant>(
        sycl::nd_range<2>(sycl::range<2>((size_t) M, WG), sycl::range<2>(1, WG)),
        [=](sycl::nd_item<2> it) {
            const int    m       = (int) it.get_group(0);
            const int    tid     = (int) it.get_local_id(1);
            const bf16 * row_in  = is + (size_t) m * K_in;
            int8_t *     row_out = out + (size_t) m * K_out;

            float local_max = 0.0f;
            for (int k = tid; k < K; k += WG) {
                const float v = sycl::fabs(to_f32(row_in[k]));
                if (v > local_max) local_max = v;
            }
            const float row_max =
                sycl::reduce_over_group(it.get_group(), local_max, sycl::maximum<float>());

            const float amax  = row_max < 1e-5f ? 1e-5f : row_max;
            const float scale = exact_div(127.0f, amax);
            if (tid == 0) scales[m] = scale;

            for (int k = tid; k < K; k += WG) {
                float q_ = sycl::rint(to_f32(row_in[k]) * scale);
                if (q_ > 127.0f) q_ = 127.0f;
                if (q_ < -128.0f) q_ = -128.0f;
                row_out[k] = (int8_t) q_;
            }

            // Zero tail. Iterates zero times when K_out == K_in, which is every
            // call but the ViT's fc1, so the unpadded path is unaffected.
            for (int k = K_in + tid; k < K_out; k += WG)
                row_out[k] = 0;
        });
}

extern "C" void bitvla_act_quant_cuda(const vla_bf16* in, int8_t* out, float* scales,
                                      int M, int K, vla_stream stream) {
    bitvla_act_quant_pad_cuda(in, out, scales, M, K, K, stream);
}

namespace {

struct k_cat_ws {};
struct k_cat_bias {};

/// Build (once) the concatenated weights, scales and optional bias.
const Cat3 & cat3_for(int8_t * const B[3], float * const ws[3], const vla_bf16 * const bias[3],
                      const int N[3], int K, sycl::queue & q) {
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto &                      cc = cat3_cache();
    auto                        it = cc.find(B[0]);
    if (it != cc.end()) return it->second;

    const int Nt = N[0] + N[1] + N[2];
    Cat3      e;
    e.src[0] = B[0]; e.src[1] = B[1]; e.src[2] = B[2];
    e.w      = (int8_t *) vla_dev_malloc((size_t) Nt * K);
    e.ws     = (float *) vla_dev_malloc((size_t) Nt * sizeof(float));
    if (bias[0]) e.bias = (float *) vla_dev_malloc((size_t) Nt * sizeof(float));
    if (!e.w || !e.ws || (bias[0] && !e.bias)) {
        std::fprintf(stderr, "vla(bitvla): concatenated QKV weights: allocation failed (%s)\n",
                     vla_dev_error());
        std::abort();
    }

    int off = 0;
    for (int i = 0; i < 3; ++i) {
        const int ws_num = ws_num_for(N[i], K);
        if (ws_num == 0) unsupported_shape("bitvla_ternary_gemm_cat3", 0, N[i], K);
        unpack_into(B[i], N[i], K, e.w + (size_t) off * K, q);

        const int     per_group = N[i] / ws_num;
        const float * w_i       = ws[i];
        float *       dst       = e.ws + off;
        q.parallel_for<k_cat_ws>(sycl::range<1>((size_t) N[i]),
                                 [=](sycl::id<1> id) { dst[id[0]] = w_i[id[0] / per_group]; });
        if (bias[0]) {
            const bf16 * b_i = as_bf(bias[i]);
            float *      bd  = e.bias + off;
            q.parallel_for<k_cat_bias>(sycl::range<1>((size_t) N[i]),
                                       [=](sycl::id<1> id) { bd[id[0]] = to_f32(b_i[id[0]]); });
        }
        off += N[i];
    }
    q.wait();
    return cc.emplace(B[0], e).first->second;
}

struct Cat3Plan {
    dnnl::memory::desc a_md, b_md, c_md, s_md, ws_md, bias_md;
    dnnl::matmul       prim;
    Tuned              tuned;
};

Cat3Plan & cat3_plan(const dnnl::engine & eng, int M, int Nt, int K, bool with_bias) {
    using dt   = dnnl::memory::data_type;
    using dims = dnnl::memory::dims;
    static std::mutex                                         mu;
    static std::map<std::array<int, 4>, Cat3Plan>             cache;
    std::lock_guard<std::mutex>                               lock(mu);
    const std::array<int, 4>                                  key{M, Nt, K, with_bias ? 1 : 0};
    auto                                                      it = cache.find(key);
    if (it != cache.end()) return it->second;

    Cat3Plan p;
    p.a_md  = dnnl::memory::desc({M, K}, dt::s8, dims{K, 1});
    p.b_md  = dnnl::memory::desc({K, Nt}, dt::s8, dims{1, K});
    p.c_md  = dnnl::memory::desc({M, Nt}, dt::bf16, dims{Nt, 1});
    p.s_md  = dnnl::memory::desc({M, 1}, dt::f32, dims{1, 1});
    p.ws_md = dnnl::memory::desc({1, Nt}, dt::f32, dims{Nt, 1});
    dnnl::post_ops po;
    po.append_binary(dnnl::algorithm::binary_div, p.s_md);
    po.append_binary(dnnl::algorithm::binary_mul, p.ws_md);
    if (with_bias) {
        p.bias_md = p.ws_md;
        po.append_binary(dnnl::algorithm::binary_add, p.bias_md);
    }
    dnnl::primitive_attr attr;
    attr.set_post_ops(po);
    p.prim = dnnl::matmul(dnnl::matmul::primitive_desc(eng, p.a_md, p.b_md, p.c_md, attr));
    build_blocked(p.tuned, eng, p.a_md, p.b_md, p.c_md, attr);
    return cache.emplace(key, p).first->second;
}

}  // namespace

extern "C" void bitvla_ternary_gemm_cat3(const int8_t* A, const float* s, int M, int K,
                                         int8_t* const B[3], float* const ws[3],
                                         const vla_bf16* const bias[3], const int N[3],
                                         vla_bf16* out, vla_stream stream) {
    static const bool log_shapes = vla::env_flag("VLA_BITVLA_LOG_SHAPES");
    if (log_shapes)
        for (int i = 0; i < 3; ++i) shape_log().add(M, N[i], K);

    sycl::queue &    q    = vla::bitvla_sycl_queue(stream);
    dnnl::engine &   eng  = vla::bitvla_dnnl_engine(stream);
    const Cat3 &     cat  = cat3_for(B, ws, bias, N, K, q);
    const int        Nt   = N[0] + N[1] + N[2];
    Cat3Plan &       plan = cat3_plan(eng, M, Nt, K, cat.bias != nullptr);

    dnnl::memory a_mem(plan.a_md, eng, const_cast<int8_t *>(A));
    dnnl::memory c_mem(plan.c_md, eng, out);
    dnnl::memory s_mem(plan.s_md, eng, const_cast<float *>(s));
    dnnl::memory ws_mem(plan.ws_md, eng, cat.ws);
    std::unordered_map<int, dnnl::memory> args{
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_DST, c_mem},
        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, s_mem},
        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, ws_mem}};
    if (cat.bias)
        args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                     dnnl::memory(plan.bias_md, eng, cat.bias));
    exec_tuned(plan.tuned, plan.prim, {}, cat.w, cat.w, plan.b_md, args, stream);
}
