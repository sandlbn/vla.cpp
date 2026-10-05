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

// GGML_OP_FLASH_ATTN_EXT on oneDNN Graph's fused SDPA, through the same
// ggml-sycl forward hook as vla_sycl_bf16.cpp.
//
// Why: ggml-sycl's own flash-attention kernel is slower on Intel iGPUs than
// the unfused graph it replaces (pi0 on Panther Lake: 508 ms vs 487 ms), and
// the unfused graph is itself dominated by attention - on the same part an f32
// softmax over the score plane, f32 oneMKL GEMMs either side of it and the
// permute/convert copies between them came to ~20% of pi0's device time. The
// SDPA micro-kernel oneDNN ships for Xe2/Xe3 does the whole thing in one pass
// with the scores on chip.
//
// Semantics, ggml's (ggml.h, ggml_flash_attn_ext):
//   q    [D, S,    H,    1]   any of F32/F16/BF16, arbitrary strides
//   k, v [D, S_kv, H_kv, 1]   H % H_kv == 0 (grouped-query)
//   mask [S_kv, S_pad, 1, 1]  F16, additive, rows >= S ignored; or null
//   dst  [D, H, S, 1]         F32 or BF16, contiguous - note H and S swap
//   op_params: scale, max_bias, logit_softcap
//
// Precision: Q, K and V are converted to F16 on the way in - which is what
// ggml's own FA kernels do with K/V - and the scores, softmax and accumulation
// are F32 inside the kernel. F16 rather than BF16 because it keeps three more
// mantissa bits, and attention operands are post-norm activations well inside
// F16's range.
//
// Declines (returns false, ggml runs its own kernel) for ALiBi (max_bias != 0),
// logit soft-capping, batch > 1, or a pattern oneDNN will not fuse.

#include "ggml.h"
#include "env_flag.h"

#include <sycl/sycl.hpp>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

namespace {

using lt   = dnnl::graph::logical_tensor;
using gdt  = lt::data_type;
using half = sycl::half;

void * graph_malloc(size_t size, size_t alignment, const void * dev, const void * ctx) {
    return sycl::aligned_alloc_device(alignment, size, *static_cast<const sycl::device *>(dev),
                                      *static_cast<const sycl::context *>(ctx));
}

void graph_free(void * ptr, const void *, const void * ctx, void * event) {
    if (event) static_cast<sycl::event *>(event)->wait();
    sycl::free(ptr, *static_cast<const sycl::context *>(ctx));
}

/// Engine + stream on ggml-sycl's queue, with a graph allocator for the SDPA
/// partitions' scratch. One per queue.
struct Ctx {
    dnnl::engine engine;
    dnnl::stream stream;
    explicit Ctx(sycl::queue & q) :
        engine(dnnl::graph::sycl_interop::make_engine_with_allocator(
            q.get_device(), q.get_context(),
            dnnl::graph::sycl_interop::make_allocator(graph_malloc, graph_free))),
        stream(dnnl::sycl_interop::make_stream(engine, q)) {}
};

Ctx & ctx_for(sycl::queue & q) {
    static std::mutex                     mu;
    static std::map<sycl::queue *, Ctx *> cache;
    std::lock_guard<std::mutex>           lock(mu);
    auto                                  it = cache.find(&q);
    if (it == cache.end()) it = cache.emplace(&q, new Ctx(q)).first;
    return *it->second;
}

struct Key {
    long long S, S_kv, H, H_kv, D, mask_ld;
    bool      has_mask, out_bf16;
    float     scale;
    auto      tie() const { return std::tie(S, S_kv, H, H_kv, D, mask_ld, has_mask, out_bf16, scale); }
    bool      operator<(const Key & o) const { return tie() < o.tie(); }
};

struct Plan {
    bool                            ok = false;
    lt                              q, k, v, scale, mask, out;
    dnnl::graph::compiled_partition cp;
    float *                         d_scale = nullptr;
};

Plan build(const Key & key, sycl::queue & q, Ctx & c) {
    Plan      p;
    const long long rep = key.H / key.H_kv, S = key.S, Skv = key.S_kv, D = key.D;
    size_t    id  = 0;

    // F16 scratch layouts, written by the converter below: q as [H][S][D],
    // k and v as [H_kv][S_kv][D] - i.e. ggml's contiguous order.
    p.q = lt(id++, gdt::f16, lt::dims{1, key.H_kv, rep, S, D},
             lt::dims{key.H * S * D, rep * S * D, S * D, D, 1});
    p.k = lt(id++, gdt::f16, lt::dims{1, key.H_kv, 1, Skv, D},
             lt::dims{key.H_kv * Skv * D, Skv * D, Skv * D, D, 1});
    p.v = lt(id++, gdt::f16, lt::dims{1, key.H_kv, 1, Skv, D},
             lt::dims{key.H_kv * Skv * D, Skv * D, Skv * D, D, 1});
    p.scale = lt(id++, gdt::f32, lt::dims{1}, lt::layout_type::strided);
    // ggml's mask is [S_pad rows of S_kv], broadcast over heads; only the first
    // S rows are read.
    p.mask = lt(id++, gdt::f16, lt::dims{1, 1, 1, S, Skv},
                lt::dims{S * key.mask_ld, S * key.mask_ld, S * key.mask_ld, key.mask_ld, 1});
    // dst is [D, H, S]: element (h, s, d) at s*H*D + h*D + d.
    p.out = lt(id++, key.out_bf16 ? gdt::bf16 : gdt::f32, lt::dims{1, key.H_kv, rep, S, D},
               lt::dims{S * key.H * D, rep * D, D, key.H * D, 1});

    const lt::dims score{1, key.H_kv, rep, S, Skv};
    lt s0(id++, gdt::f32, score, lt::layout_type::strided);
    lt s1(id++, gdt::f32, score, lt::layout_type::strided);
    lt s2(id++, gdt::f32, score, lt::layout_type::strided);
    lt pr(id++, gdt::f16, score, lt::layout_type::strided);

    using op = dnnl::graph::op;
    op bmm1(id++, op::kind::MatMul, "qk");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({p.q, p.k});
    bmm1.add_outputs({s0});
    op mul(id++, op::kind::Multiply, "scale");
    mul.add_inputs({s0, p.scale});
    mul.add_outputs({s1});
    op add(id++, op::kind::Add, "mask");
    add.add_inputs({s1, p.mask});
    add.add_outputs({s2});
    op sm(id++, op::kind::SoftMax, "softmax");
    sm.set_attr<int64_t>(op::attr::axis, -1);
    // A fully masked row is -inf everywhere; inf_as_zero makes it zeros, which
    // is what ggml's kernels produce, instead of NaN.
    sm.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    sm.add_inputs({key.has_mask ? s2 : s1});
    sm.add_outputs({pr});
    op bmm2(id++, op::kind::MatMul, "pv");
    bmm2.add_inputs({pr, p.v});
    bmm2.add_outputs({p.out});

    try {
        dnnl::graph::graph g(dnnl::engine::kind::gpu);
        g.add_op(bmm1);
        g.add_op(mul);
        if (key.has_mask) g.add_op(add);
        g.add_op(sm);
        g.add_op(bmm2);
        g.finalize();
        auto parts = g.get_partitions();
        if (parts.size() != 1 || !parts[0].is_supported()) {
            std::fprintf(stderr, "vla(sycl): fused SDPA unavailable for S=%lld kv=%lld %lldq/%lldkv x%lld; "
                         "using ggml's flash attention\n", S, Skv, key.H, key.H_kv, D);
            return p;
        }
        std::vector<lt> ins = key.has_mask ? std::vector<lt>{p.q, p.k, p.scale, p.mask, p.v}
                                           : std::vector<lt>{p.q, p.k, p.scale, p.v};
        p.cp = parts[0].compile(ins, {p.out}, c.engine);
    } catch (const dnnl::error & e) {
        std::fprintf(stderr, "vla(sycl): fused SDPA compile failed (%s); using ggml's flash attention\n",
                     e.what());
        return p;
    }
    p.d_scale = sycl::malloc_device<float>(1, q);
    q.memcpy(p.d_scale, &key.scale, sizeof(float)).wait();
    p.ok = true;
    return p;
}

/// Grown, never shrunk: one region each for the F16 q, k and v.
half * scratch(sycl::queue & q, int slot, size_t elems) {
    static std::mutex mu;
    static half *     buf[3] = {nullptr, nullptr, nullptr};
    static size_t     cap[3] = {0, 0, 0};
    std::lock_guard<std::mutex> lock(mu);
    if (elems > cap[slot]) {
        if (buf[slot]) {
            q.wait();
            sycl::free(buf[slot], q);
        }
        buf[slot] = sycl::malloc_device<half>(elems, q);
        cap[slot] = elems;
    }
    return buf[slot];
}

struct k_to_f16 {};

/// [D, S, H] tensor of any float type, any strides -> contiguous F16 [H][S][D].
bool to_f16(const ggml_tensor * t, half * dst, sycl::queue & q) {
    const long long D = t->ne[0], S = t->ne[1], H = t->ne[2];
    const size_t    nb0 = t->nb[0], nb1 = t->nb[1], nb2 = t->nb[2];
    const char *    src = static_cast<const char *>(t->data);
    const int       type = t->type;
    const size_t    n = (size_t) (D * S * H);
    q.parallel_for<k_to_f16>(sycl::range<1>(n), [=](sycl::id<1> i) {
        const long long d = (long long) (i[0] % D);
        const long long s = (long long) ((i[0] / D) % S);
        const long long h = (long long) (i[0] / (D * S));
        const char *    p = src + d * nb0 + s * nb1 + h * nb2;
        float           v;
        if (type == GGML_TYPE_F32)       v = *reinterpret_cast<const float *>(p);
        else if (type == GGML_TYPE_F16)  v = (float) *reinterpret_cast<const half *>(p);
        else                             v = (float) *reinterpret_cast<const sycl::ext::oneapi::bfloat16 *>(p);
        dst[i[0]] = (half) v;
    });
    return true;
}

bool float_type(int t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16; }

}  // namespace

namespace vla {

bool sycl_flash_attn_ext(ggml_tensor * dst, sycl::queue & q) {
    static const bool off = vla::env_flag("VLA_SYCL_NO_SDPA");
    if (off) return false;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * M = dst->src[3];
    if (!Q || !K || !V) return false;

    float scale = 1.0f, max_bias = 0.0f, softcap = 0.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) return false;

    if (!float_type(Q->type) || !float_type(K->type) || !float_type(V->type)) return false;
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_BF16) return false;
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) return false;
    if (M && (M->type != GGML_TYPE_F16 || M->ne[2] != 1 || M->ne[3] != 1 ||
              M->nb[0] != sizeof(half) || M->ne[1] < Q->ne[1])) return false;
    if (!ggml_is_contiguous(dst)) return false;

    const long long D = Q->ne[0], S = Q->ne[1], H = Q->ne[2];
    const long long Skv = K->ne[1], Hkv = K->ne[2];
    if (K->ne[0] != D || V->ne[0] != D || V->ne[1] != Skv || V->ne[2] != Hkv || Hkv == 0 ||
        H % Hkv != 0)
        return false;

    const Key key{S, Skv, H, Hkv, D, M ? (long long) (M->nb[1] / sizeof(half)) : Skv, M != nullptr,
                  dst->type == GGML_TYPE_BF16, scale};

    Ctx & c = ctx_for(q);
    static std::mutex          mu;
    static std::map<Key, Plan> plans;
    Plan *                     p;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto                        it = plans.find(key);
        if (it == plans.end()) it = plans.emplace(key, build(key, q, c)).first;
        p = &it->second;
    }
    if (!p->ok) return false;

    half * q16 = scratch(q, 0, (size_t) (D * S * H));
    half * k16 = scratch(q, 1, (size_t) (D * Skv * Hkv));
    half * v16 = scratch(q, 2, (size_t) (D * Skv * Hkv));
    to_f16(Q, q16, q);
    to_f16(K, k16, q);
    to_f16(V, v16, q);

    using dnnl::graph::tensor;
    tensor tq(p->q, c.engine, q16), tk(p->k, c.engine, k16), tv(p->v, c.engine, v16);
    tensor ts(p->scale, c.engine, p->d_scale), to(p->out, c.engine, dst->data);
    try {
        if (M) {
            tensor tm(p->mask, c.engine, M->data);
            p->cp.execute(c.stream, {tq, tk, ts, tm, tv}, {to});
        } else {
            p->cp.execute(c.stream, {tq, tk, ts, tv}, {to});
        }
    } catch (const dnnl::error & e) {
        std::fprintf(stderr, "vla(sycl): fused SDPA execute failed: %s\n", e.what());
        return false;
    }
    return true;
}

}  // namespace vla
