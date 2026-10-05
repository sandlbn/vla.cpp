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

// BF16 activation kernels for the SYCL backend - the twin of
// src/cuda/vla_cuda_bf16.cu, and deliberately a quarter of its size.
//
// What ggml-sycl already does, and this file therefore does not
// -----------------------------------------------------------
// ggml-sycl is further along on BF16 than ggml-cuda, which is worth stating as
// a measured fact about the pinned tag rather than an assumption:
//
//   ADD / MUL   binbcast.cpp instantiates BF16 x BF16 -> BF16 and
//               BF16 x F32 -> BF16. ggml-cuda has neither, which is most of
//               why vla_cuda_bf16.cu is 700 lines.
//   UNARY       dispatch_ggml_sycl_op_unary covers BF16 behind
//               GGML_SYCL_HAS_BF16, which icpx defines - and our CMake refuses
//               to configure with any other compiler, so it is always on here.
//               SILU, RELU, GELU and GELU_ERF all route through it.
//
// Re-implementing those would mean carrying a second, less-tested copy of a
// kernel that already works, for no measured gain. So the entry point below
// declines them by simply not listing them, and they run on ggml-sycl's own
// kernels. If a profile ever says those are the bottleneck, that is a decision
// to revisit with numbers in hand.
//
// What is left is what ggml-sycl genuinely cannot do:
//
//   NORM        norm.cpp asserts F32 on src and dst.
//   RMS_NORM    same.
//   SCALE       ggml_sycl_op_scale asserts F32 on src and dst.
//   MUL_MAT     ggml_sycl_op_mul_mat_sycl has a bf16 fast path, but it hands
//               oneDNN a to_dt<float>() destination unconditionally. For a BF16
//               dst that writes the wrong type into twice the bytes the
//               allocator reserved.
//
// ggml offers no way to register kernels for built-in ops (GGML_OP_CUSTOM is
// CPU-only), so the fetched ggml carries one hook - a function pointer consulted
// in ggml_sycl_compute_forward - and everything else lives here, in tree, as
// ordinary first-party code. See scripts/patch_ggml_sycl_ext_hook.py; it is the
// entire ggml modification.
//
// This file deliberately depends only on the PUBLIC ggml header. It never
// includes ggml-sycl internals (common.hpp and friends), so a llama.cpp bump
// cannot break it the way an anchored source patch would.
//
// Every entry point returns false for anything it does not handle, and ggml then
// runs the op exactly as it would have. Nothing here changes the F32 path.
//
// Accumulation is float throughout: only operand and result *storage* is BF16,
// never a reduction.
//
// Precision: this is the tolerance-gated half of the port's precision contract,
// not the bit-exact half (that is BitVLA's ternary GEMM). Each op reproduces the
// CUDA formula literally in fp32 - same variance identity, same order of
// operations - so SYCL-vs-F32 deviation is held to the same band CUDA-vs-F32 is.
// tests/test_bf16_sycl_ops.cpp is the gate.

#include "ggml.h"
#include "env_flag.h"

#include <sycl/sycl.hpp>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

// Must match the typedef the hook patch inserts into ggml-sycl.cpp.
extern "C" {
typedef bool (*ggml_sycl_ext_forward_t)(struct ggml_tensor * dst, void * stream);
extern ggml_sycl_ext_forward_t ggml_sycl_ext_forward;
}

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

// One work-group per row for the norms; also the flat block for scale. Matches
// the CUDA file's BLOCK so the two backends split a row the same number of ways
// and therefore build the same shape of summation tree.
constexpr int WG = 256;

// VLA_SYCL_BF16_OFF=1 makes every entry point decline, which hands the graph
// back to stock ggml-sycl. NORM/RMS_NORM/SCALE then abort on their F32 asserts,
// so this is a bisecting tool for the MUL_MAT path, not a supported mode.
bool disabled() {
    static const bool v = vla::env_flag("VLA_SYCL_BF16_OFF");
    return v;
}

// ---------------------------------------------------------------------------
// oneDNN engine and primitive cache, bound to ggml-sycl's own queue
// ---------------------------------------------------------------------------
//
// Not shared with vla::bitvla_dnnl_engine(): that one wraps the queue
// device_sycl.cpp created for the ternary kernels, and a dnnl::stream has to
// wrap the queue its work is ordered against. BitVLA pins its ggml graph to the
// CPU backend anyway, so the two never coexist in one process.

struct Ctx {
    dnnl::engine engine;
    dnnl::stream stream;

    explicit Ctx(sycl::queue & q) :
        engine(dnnl::sycl_interop::make_engine(q.get_device(), q.get_context())),
        stream(dnnl::sycl_interop::make_stream(engine, q)) {}
};

// ggml-sycl uses one queue per device and vla only ever brings up one device,
// so a single lazily-built context covers the process. Keyed anyway, so a
// second queue gets its own rather than silently reusing the wrong one.
Ctx & ctx_for(sycl::queue & q) {
    static std::mutex                    mu;
    static std::map<sycl::queue *, Ctx*> cache;

    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(&q);
    if (it == cache.end()) {
        it = cache.emplace(&q, new Ctx(q)).first;
    }
    return *it->second;
}

using dt   = dnnl::memory::data_type;
using ddims = dnnl::memory::dims;

// Everything that distinguishes one matmul from another, and nothing else.
struct Key {
    long long M, N, K, batch, wbatch;

    bool operator<(const Key & o) const {
        if (M != o.M) return M < o.M;
        if (N != o.N) return N < o.N;
        if (K != o.K) return K < o.K;
        if (batch != o.batch) return batch < o.batch;
        return wbatch < o.wbatch;
    }
};

struct Plan {
    dnnl::memory::desc a_md, b_md, c_md;
    dnnl::matmul       prim;
};

// ---------------------------------------------------------------------------
// mul_mat: BF16 x BF16 -> BF16
// ---------------------------------------------------------------------------
//
// ggml's own bf16 path converts src1 into a pool allocation on the way in and
// writes f32 out, because ggml_mul_mat's result is F32 by definition. When the
// caller asked for a BF16 result (vla::mul_mat_t) and both operands are already
// BF16 there is nothing to convert: describe the bytes to oneDNN and run.
//
// The shapes are expressed as strides rather than a transpose, the same way
// kernels/bitvla/sycl/gemm_sycl.cpp does it: src0 is (N, K) row-major, so
// describing it as {K, N} with strides {1, K} is B^T for free - no repacking and
// no transpose kernel, and the primitive sees a plain NN matmul it can pick its
// best Xe implementation for. Accumulation is oneDNN's f32 default, matching the
// CUBLAS_COMPUTE_32F the CUDA twin asks cuBLAS for.

bool mul_mat(ggml_tensor * dst, sycl::queue & q) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (!src0 || !src1) return false;
    if (dst->type != GGML_TYPE_BF16 || src0->type != GGML_TYPE_BF16 || src1->type != GGML_TYPE_BF16) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // src0 is either shared across the whole batch or batched 1:1 with src1
    const bool bcast = src0->ne[2] == 1 && src0->ne[3] == 1;
    const bool batch_ok = bcast || (src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3]);
    if (!batch_ok) return false;

    const long long K     = src1->ne[0];
    const long long M     = src1->ne[1];
    const long long N     = src0->ne[1];
    const long long batch = src1->ne[2]*src1->ne[3];
    if (src0->ne[0] != K) return false;
    if (dst->ne[0] != N || dst->ne[1] != M) return false;

    // Broadcasting a batch is expressed as an extent of 1 on the weights, which
    // is oneDNN's documented rule, rather than as a zero stride.
    const long long wbatch = bcast ? 1 : batch;

    const Key key{ M, N, K, batch, wbatch };

    static std::mutex          mu;
    static std::map<Key, Plan> cache;

    try {
        Ctx & c = ctx_for(q);

        const dnnl::memory::desc a_md({ batch, M, K }, dt::bf16, ddims{ M*K, K, 1 });
        const dnnl::memory::desc b_md({ wbatch, K, N }, dt::bf16, ddims{ N*K, 1, K });
        const dnnl::memory::desc c_md({ batch, M, N }, dt::bf16, ddims{ M*N, N, 1 });

        const Plan * plan = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu);
            auto it = cache.find(key);
            if (it == cache.end()) {
                dnnl::matmul::primitive_desc pd(c.engine, a_md, b_md, c_md);
                it = cache.emplace(key, Plan{ a_md, b_md, c_md, dnnl::matmul(pd) }).first;
            }
            plan = &it->second;
        }

        // oneDNN's handles are non-const; the primitive only reads A and B.
        dnnl::memory a_mem(plan->a_md, c.engine, src1->data);
        dnnl::memory b_mem(plan->b_md, c.engine, src0->data);
        dnnl::memory c_mem(plan->c_md, c.engine, dst->data);

        plan->prim.execute(c.stream,
                           { { DNNL_ARG_SRC, a_mem }, { DNNL_ARG_WEIGHTS, b_mem }, { DNNL_ARG_DST, c_mem } });
        return true;
    } catch (const dnnl::error & e) {
        // Declining is not an option once we are this far: ggml-sycl's own path
        // would write f32 into a bf16 dst. Say so and let the assert that
        // follows point at the real cause.
        std::fprintf(stderr, "vla(sycl_bf16): matmul %lldx%lldx%lld (batch %lld) failed: %s\n",
                     M, N, K, batch, e.what());
        return false;
    }
}

// ---------------------------------------------------------------------------
// scale: dst = x*scale + bias
// ---------------------------------------------------------------------------

bool scale(ggml_tensor * dst, sycl::queue & q) {
    const ggml_tensor * src0 = dst->src[0];
    if (!src0) return false;
    if (dst->type != GGML_TYPE_BF16 || src0->type != GGML_TYPE_BF16) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) return false;

    float s = 1.0f, b = 0.0f;
    std::memcpy(&s, (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&b, (const float *) dst->op_params + 1, sizeof(float));

    const int64_t n = ggml_nelements(dst);
    if (n <= 0) return true;

    const bf16 * x = (const bf16 *) src0->data;
    bf16 *       d = (bf16 *) dst->data;

    const size_t global = (size_t) ((n + WG - 1)/WG)*WG;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
                   [=](sycl::nd_item<1> it) {
                       const int64_t i = (int64_t) it.get_global_id(0);
                       if (i < n) d[i] = bf16(s*(float) x[i] + b);
                   });
    return true;
}

// ---------------------------------------------------------------------------
// norm / rms_norm - one work-group per row, float reduction
// ---------------------------------------------------------------------------
//
// The CUDA twin reduces through a shared-memory tree over its 256 threads;
// reduce_over_group is the same reduction expressed once, and with the same
// number of work-items per row the partial sums each one carries are identical.
// The trees differ in shape, so this is not bit-identical to CUDA - it is not
// claimed to be. Both sum a row of a bf16 tensor in f32, and the gate is the
// tolerance test.
//
// Both formulas are the CUDA ones transcribed: rms uses rsqrt(mean(x^2) + eps),
// and norm uses rsqrt(E[x^2] - E[x]^2 + eps). That second identity is not how
// ggml's CPU norm computes the variance (it accumulates (x - mean)^2), and it is
// the less numerically stable of the two, but reproducing the CUDA path is the
// whole point of the exercise: the two backends have to agree with each other
// before either is compared to the reference.

template <bool rms> class k_norm_bf16;

template <bool rms>
bool norm(ggml_tensor * dst, sycl::queue & q) {
    const ggml_tensor * src0 = dst->src[0];
    if (!src0) return false;
    if (dst->type != GGML_TYPE_BF16 || src0->type != GGML_TYPE_BF16) return false;
    // Rows must be dense; higher dims are handled by flattening into the row index.
    if (src0->nb[0] != ggml_type_size(src0->type)) return false;
    if (dst->nb[0]  != ggml_type_size(dst->type))  return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) return false;

    float eps = 0.0f;
    std::memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ncols > 0 ? ggml_nelements(src0)/ncols : 0;
    if (ncols <= 0 || nrows <= 0) return true;

    const bf16 * x = (const bf16 *) src0->data;
    bf16 *       d = (bf16 *) dst->data;

    q.parallel_for<k_norm_bf16<rms>>(
        sycl::nd_range<1>(sycl::range<1>((size_t) nrows*WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) {
            const int64_t row = (int64_t) it.get_group(0);
            const int64_t lid = (int64_t) it.get_local_id(0);

            const bf16 * xr = x + row*ncols;
            bf16 *       dr = d + row*ncols;

            float sum = 0.0f, sumsq = 0.0f;
            for (int64_t c = lid; c < ncols; c += WG) {
                const float v = (float) xr[c];
                sumsq += v*v;
                if (!rms) sum += v;
            }

            auto g = it.get_group();
            if (rms) {
                const float ms  = sycl::reduce_over_group(g, sumsq, sycl::plus<float>())/(float) ncols;
                const float inv = sycl::rsqrt(ms + eps);
                for (int64_t c = lid; c < ncols; c += WG) dr[c] = bf16((float) xr[c]*inv);
            } else {
                // Two reductions, both reached by every work-item in the group:
                // the branch is on a template parameter, so it is uniform.
                const float mean   = sycl::reduce_over_group(g, sum,   sycl::plus<float>())/(float) ncols;
                const float meansq = sycl::reduce_over_group(g, sumsq, sycl::plus<float>())/(float) ncols;
                const float inv    = sycl::rsqrt(meansq - mean*mean + eps);
                for (int64_t c = lid; c < ncols; c += WG) dr[c] = bf16(((float) xr[c] - mean)*inv);
            }
        });
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// hook entry point
// ---------------------------------------------------------------------------

namespace vla {
bool sycl_flash_attn_ext(ggml_tensor * dst, sycl::queue & q);  // vla_sycl_attn.cpp
}

extern "C" bool vla_sycl_bf16_forward(ggml_tensor * dst, void * stream_v) {
    if (!dst || !stream_v || disabled()) return false;

    sycl::queue & q = *static_cast<sycl::queue *>(stream_v);

    // Flash attention is claimed for F32 results too: it is not a BF16 gap but
    // a faster kernel for every activation dtype (vla_sycl_attn.cpp).
    if (dst->op == GGML_OP_FLASH_ATTN_EXT) return vla::sycl_flash_attn_ext(dst, q);

    // Only BF16 results can be ours, and checking once here keeps the F32 graph
    // off every branch below.
    if (dst->type != GGML_TYPE_BF16) return false;

    switch (dst->op) {
        case GGML_OP_MUL_MAT:  return mul_mat(dst, q);
        case GGML_OP_SCALE:    return scale(dst, q);
        case GGML_OP_NORM:     return norm<false>(dst, q);
        case GGML_OP_RMS_NORM: return norm<true>(dst, q);
        // ADD, MUL and UNARY are deliberately absent: ggml-sycl instantiates all
        // three for BF16 already. See the header comment.
        default: return false;
    }
}

namespace vla {

// Called once, after the SYCL backend is up. Idempotent.
//
// There is only one pointer to install, unlike CUDA's two: ggml-sycl has no
// multi-node ADD/MUL fusion for a fused node to bypass the forward hook.
void sycl_register_bf16_ops() {
    ggml_sycl_ext_forward = vla_sycl_bf16_forward;
}

}  // namespace vla
