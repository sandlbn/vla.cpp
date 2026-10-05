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
 * @file attention_sycl.cpp
 * @brief @c attention.h on oneDNN Graph's fused SDPA.
 *
 * The graph is the textbook pattern - MatMul(Q, K^T), Multiply(scale),
 * SoftMax, MatMul(., V) - which oneDNN's pattern matcher lowers to a single
 * @c ocl:micro SDPA kernel on Xe2/Xe3. The point of describing it to the graph
 * API rather than calling the pieces is the layouts:
 *
 *  - Q, K, V and the output are 5-D views [1, n_kv, rep, S, hd] with strides
 *    straight into the row-major [S, H*hd] projection planes, so there is no
 *    head-major transpose on either side.
 *  - K and V carry a size-1 @c rep dimension, which is how oneDNN spells
 *    grouped-query attention: query head (g, r) reads KV head g, and the
 *    repeated KV planes never exist.
 *
 * Measured standalone on Panther Lake (Xe3, 4 cores), per layer: LM (S=330,
 * 20q/5kv x 128) 0.35 ms, ViT (S=256, 16 x 72) 0.10 ms - about 3.2 TFLOPS.
 */

#include "kernels/bitvla/attention.h"
#include "kernels/bitvla/sycl/dnnl_sycl.h"
#include "kernels/bitvla/sycl/queue_sycl.h"

#include "oneapi/dnnl/dnnl_graph.hpp"

#include <cstdio>
#include <map>
#include <mutex>
#include <tuple>

namespace {

using lt = dnnl::graph::logical_tensor;
using dt = lt::data_type;

struct Key {
    int   S, n_q, n_kv, hd, q_ld, kv_ld, out_ld;
    float scale;
    auto  tie() const { return std::tie(S, n_q, n_kv, hd, q_ld, kv_ld, out_ld, scale); }
    bool  operator<(const Key & o) const { return tie() < o.tie(); }
};

struct Plan {
    bool                                ok = false;
    lt                                  q, k, v, scale, out;
    dnnl::graph::compiled_partition     cp;
    float *                             d_scale = nullptr;  ///< device copy of the scalar
};

/// A [1, n_kv, rep, S, hd] view of a row-major [S, ld] plane in which head
/// (g, r) starts at column (g*rep + r)*hd. @p rep_stride is 0 for K and V,
/// whose rep dimension is 1 and only there to broadcast.
lt view(size_t id, int S, int n_kv, int rep, int hd, int ld) {
    return lt(id, dt::bf16, lt::dims{1, n_kv, rep, S, hd},
              lt::dims{(int64_t) S * ld, (int64_t) rep * hd, hd, ld, 1});
}

Plan build(const Key & key, vla_stream stream) {
    Plan p;
    const int rep = key.n_q / key.n_kv;
    size_t    id  = 0;

    p.q   = view(id++, key.S, key.n_kv, rep, key.hd, key.q_ld);
    p.k   = view(id++, key.S, key.n_kv, 1, key.hd, key.kv_ld);
    p.v   = view(id++, key.S, key.n_kv, 1, key.hd, key.kv_ld);
    p.out = view(id++, key.S, key.n_kv, rep, key.hd, key.out_ld);
    p.scale = lt(id++, dt::f32, lt::dims{1}, lt::layout_type::strided);

    const lt::dims score_dims{1, key.n_kv, rep, key.S, key.S};
    lt score(id++, dt::f32, score_dims, lt::layout_type::strided);
    lt scaled(id++, dt::f32, score_dims, lt::layout_type::strided);
    lt probs(id++, dt::bf16, score_dims, lt::layout_type::strided);

    using op = dnnl::graph::op;
    op bmm1(id++, op::kind::MatMul, "qk");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({p.q, p.k});
    bmm1.add_outputs({score});
    op mul(id++, op::kind::Multiply, "scale");
    mul.add_inputs({score, p.scale});
    mul.add_outputs({scaled});
    op sm(id++, op::kind::SoftMax, "softmax");
    sm.set_attr<int64_t>(op::attr::axis, -1);
    sm.add_inputs({scaled});
    sm.add_outputs({probs});
    op bmm2(id++, op::kind::MatMul, "pv");
    bmm2.add_inputs({probs, p.v});
    bmm2.add_outputs({p.out});

    try {
        dnnl::graph::graph g(dnnl::engine::kind::gpu);
        g.add_op(bmm1);
        g.add_op(mul);
        g.add_op(sm);
        g.add_op(bmm2);
        g.finalize();
        auto parts = g.get_partitions();
        // One partition is the fused kernel. Anything else means the matcher
        // split the pattern, which would run it as the same unfused pieces we
        // are trying to replace - report unavailable and let the caller keep
        // its own chain.
        if (parts.size() != 1 || !parts[0].is_supported()) {
            std::fprintf(stderr, "vla(bitvla): fused attention unavailable for S=%d %dq/%dkv x%d "
                         "(%zu partitions); using the unfused chain\n",
                         key.S, key.n_q, key.n_kv, key.hd, parts.size());
            return p;
        }
        p.cp = parts[0].compile({p.q, p.k, p.scale, p.v}, {p.out}, vla::bitvla_dnnl_engine(stream));
    } catch (const dnnl::error & e) {
        std::fprintf(stderr, "vla(bitvla): fused attention compile failed (%s); "
                     "using the unfused chain\n", e.what());
        return p;
    }

    sycl::queue & q = vla::bitvla_sycl_queue(stream);
    p.d_scale       = sycl::malloc_device<float>(1, q);
    q.memcpy(p.d_scale, &key.scale, sizeof(float)).wait();
    p.ok = true;
    return p;
}

}  // namespace

extern "C" int vla_attention_bf16(const vla_bf16* q, const vla_bf16* k, const vla_bf16* v,
                                  vla_bf16* out, int S, int n_q, int n_kv, int hd,
                                  int q_ld, int kv_ld, int out_ld, float scale,
                                  vla_stream stream) {
    if (S <= 0 || n_kv <= 0 || n_q % n_kv != 0) return 1;

    static std::mutex          mu;
    static std::map<Key, Plan> plans;

    const Key key{S, n_q, n_kv, hd, q_ld, kv_ld, out_ld, scale};
    Plan *    p;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto                        it = plans.find(key);
        if (it == plans.end()) it = plans.emplace(key, build(key, stream)).first;
        p = &it->second;
    }
    if (!p->ok) return 1;

    dnnl::engine & eng = vla::bitvla_dnnl_engine(stream);
    using dnnl::graph::tensor;
    tensor tq(p->q, eng, const_cast<vla_bf16 *>(q));
    tensor tk(p->k, eng, const_cast<vla_bf16 *>(k));
    tensor tv(p->v, eng, const_cast<vla_bf16 *>(v));
    tensor ts(p->scale, eng, p->d_scale);
    tensor to(p->out, eng, out);
    try {
        p->cp.execute(vla::bitvla_dnnl_stream(stream), {tq, tk, ts, tv}, {to});
    } catch (const dnnl::error & e) {
        std::fprintf(stderr, "vla(bitvla): fused attention execute failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
