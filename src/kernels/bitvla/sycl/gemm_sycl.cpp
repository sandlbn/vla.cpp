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
 * @file gemm_sycl.cpp
 * @brief gemm.h on Intel GPUs, via dnnl::matmul. Also owns the oneDNN engine.
 *
 * These are the dense GEMMs BitVLA's attention and action head need, the ones
 * that were cuBLAS. There is no hand-written alternative planned for them and
 * there should not be: oneDNN's bf16 matmul already targets Battlemage's DPAS
 * units and is tuned per shape, which is more than a from-scratch attention
 * GEMM would be worth here - the ternary projections, not attention, are where
 * BitVLA's time goes.
 *
 * Every shape in gemm.h is expressed as strides rather than a transpose. oneDNN
 * takes arbitrary strides on any operand, so `B[N][K]^T` is the same bytes
 * described as `{K, N}` with strides `{1, K}` - no repacking, no transpose
 * kernel, and the primitive sees a plain NN matmul it can pick its best
 * implementation for.
 */

#include "kernels/bitvla/gemm.h"
#include "kernels/bitvla/sycl/dnnl_sycl.h"
#include "kernels/bitvla/sycl/queue_sycl.h"

#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include <cstdio>
#include <map>
#include <mutex>

namespace vla {

namespace {

void * graph_malloc(size_t size, size_t alignment, const void * dev, const void * ctx) {
    return sycl::aligned_alloc_device(alignment, size, *static_cast<const sycl::device *>(dev),
                                      *static_cast<const sycl::context *>(ctx));
}

void graph_free(void * ptr, const void *, const void * ctx, void * event) {
    if (event) static_cast<sycl::event *>(event)->wait();
    sycl::free(ptr, *static_cast<const sycl::context *>(ctx));
}

/// Built once per device, lazily, because nothing exists before vla_dev_set.
///
/// The engine carries a graph allocator so that the same engine also serves
/// the fused attention partitions in attention_sycl.cpp - oneDNN Graph needs
/// one to place its scratchpads, and a second engine would mean a second JIT
/// cache. Primitives ignore the allocator.
struct Ctx {
    dnnl::engine engine;
    dnnl::stream stream;

    explicit Ctx(sycl::queue & q)
        : engine(dnnl::graph::sycl_interop::make_engine_with_allocator(
              q.get_device(), q.get_context(),
              dnnl::graph::sycl_interop::make_allocator(graph_malloc, graph_free))),
          stream(dnnl::sycl_interop::make_stream(engine, q)) {}
};

Ctx & ctx_for(vla_stream s) {
    static Ctx c(vla::bitvla_sycl_queue(s));
    return c;
}

}  // namespace

dnnl::engine & bitvla_dnnl_engine(vla_stream stream) { return ctx_for(stream).engine; }
dnnl::stream & bitvla_dnnl_stream(vla_stream stream) { return ctx_for(stream).stream; }

}  // namespace vla

namespace {

using dt   = dnnl::memory::data_type;
using dims = dnnl::memory::dims;

/// A built matmul plus the descriptors its arguments must be bound with.
struct Plan {
    dnnl::memory::desc a_md, b_md, c_md;
    dnnl::matmul       prim;
};

/// Everything that distinguishes one problem from another, and nothing else.
struct Key {
    int       kind, M, N, K, batch;
    long long sa, sb, sc;

    bool operator<(const Key & o) const {
        if (kind != o.kind) return kind < o.kind;
        if (M != o.M) return M < o.M;
        if (N != o.N) return N < o.N;
        if (K != o.K) return K < o.K;
        if (batch != o.batch) return batch < o.batch;
        if (sa != o.sa) return sa < o.sa;
        if (sb != o.sb) return sb < o.sb;
        return sc < o.sc;
    }
};

/**
 * @brief The matmul for one problem, built once.
 *
 * A BitVLA forward pass issues these in a loop over 30 LM layers and 26 ViT
 * layers, hitting the same handful of shapes every time. @c primitive_desc
 * construction walks the implementation list on every call even when oneDNN's
 * own kernel cache hits, so the map in front of it is what keeps that off the
 * critical path. The key is the problem parameters rather than the serialised
 * descriptors, since @c memory::desc::get_blob allocates - which would put the
 * cost back, in a less obvious place.
 */
int run(const char * which, const Key & key, const dnnl::memory::desc & a_md,
        const dnnl::memory::desc & b_md, const dnnl::memory::desc & c_md, const void * A,
        const void * B, void * C, vla_stream stream) {
    static std::mutex          mu;
    static std::map<Key, Plan> cache;

    try {
        dnnl::engine & eng = vla::bitvla_dnnl_engine(stream);

        const Plan * plan = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu);
            auto                        it = cache.find(key);
            if (it == cache.end()) {
                dnnl::matmul::primitive_desc pd(eng, a_md, b_md, c_md);
                it = cache.emplace(key, Plan{a_md, b_md, c_md, dnnl::matmul(pd)}).first;
            }
            plan = &it->second;
        }
        const Plan & p = *plan;

        // oneDNN's handles are non-const; the primitive only reads A and B.
        dnnl::memory a_mem(p.a_md, eng, const_cast<void *>(A));
        dnnl::memory b_mem(p.b_md, eng, const_cast<void *>(B));
        dnnl::memory c_mem(p.c_md, eng, C);

        p.prim.execute(vla::bitvla_dnnl_stream(stream),
                       {{DNNL_ARG_SRC, a_mem}, {DNNL_ARG_WEIGHTS, b_mem}, {DNNL_ARG_DST, c_mem}});
        return 0;
    } catch (const dnnl::error & e) {
        std::fprintf(stderr, "vla(bitvla_gemm_sycl): %s failed: %s\n", which, e.what());
        return -1;
    }
}

}  // namespace

extern "C" int vla_gemm_bf16_nt(const vla_bf16 * A, const vla_bf16 * B, vla_bf16 * C,
                                int M, int N, int K, vla_stream stream) {
    return run("bf16_nt", Key{0, M, N, K, 1, 0, 0, 0},
               dnnl::memory::desc({M, K}, dt::bf16, dims{K, 1}),
               // B is (N, K) row-major; swapped strides make it B^T for free.
               dnnl::memory::desc({K, N}, dt::bf16, dims{1, K}),
               dnnl::memory::desc({M, N}, dt::bf16, dims{N, 1}),
               A, B, C, stream);
}

extern "C" int vla_gemm_bf16_nt_batched(const vla_bf16 * A, const vla_bf16 * B, vla_bf16 * C,
                                        int M, int N, int K, int batch,
                                        long long stride_a, long long stride_b,
                                        long long stride_c, vla_stream stream) {
    return run("bf16_nt_batched", Key{1, M, N, K, batch, stride_a, stride_b, stride_c},
               dnnl::memory::desc({batch, M, K}, dt::bf16, dims{stride_a, K, 1}),
               dnnl::memory::desc({batch, K, N}, dt::bf16, dims{stride_b, 1, K}),
               dnnl::memory::desc({batch, M, N}, dt::bf16, dims{stride_c, N, 1}),
               A, B, C, stream);
}

extern "C" int vla_gemm_bf16_nn_batched(const vla_bf16 * A, const vla_bf16 * B, vla_bf16 * C,
                                        int M, int N, int K, int batch,
                                        long long stride_a, long long stride_b,
                                        long long stride_c, vla_stream stream) {
    return run("bf16_nn_batched", Key{2, M, N, K, batch, stride_a, stride_b, stride_c},
               dnnl::memory::desc({batch, M, K}, dt::bf16, dims{stride_a, K, 1}),
               dnnl::memory::desc({batch, K, N}, dt::bf16, dims{stride_b, N, 1}),
               dnnl::memory::desc({batch, M, N}, dt::bf16, dims{stride_c, N, 1}),
               A, B, C, stream);
}

extern "C" int vla_gemm_f32_nt(const float * A, const float * B, float * C,
                               int M, int N, int K, vla_stream stream) {
    return run("f32_nt", Key{3, M, N, K, 1, 0, 0, 0},
               dnnl::memory::desc({M, K}, dt::f32, dims{K, 1}),
               dnnl::memory::desc({K, N}, dt::f32, dims{1, K}),
               dnnl::memory::desc({M, N}, dt::f32, dims{N, 1}),
               A, B, C, stream);
}
