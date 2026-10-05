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
 * @file bitvla_lm_driver.cpp
 * @brief Forward pass for BitVLA's BitNet language model.
 *
 * Backend-neutral, like @c bitvla_vit_driver.cpp beside it: device memory comes
 * from @c device.h, the two attention GEMMs from @c gemm.h, and everything else
 * from the @c extern @c "C" ops that @c bitvla_ops_cuda.cu and
 * @c sycl/bitvla_ops_sycl.cpp each implement. This file used to be the second
 * half of @c bitvla_lm_cuda.cu; the kernels stayed there, the orchestration
 * moved here so it would not have to be written twice.
 *
 * The per-layer order below is BitNet's, not a choice - in particular the two
 * sub-norms (after attention output, after the FFN gate) are what distinguishes
 * BitNet b1.58 from a conventional transformer block, and dropping them is a
 * silent accuracy loss rather than a crash.
 */

#include "bitvla_lm_cuda.h"
#include "kernels/bitvla/device.h"
#include "kernels/bitvla/gemm.h"
#ifdef VLA_BITVLA_FUSED_OPS
#include "env_flag.h"
#include "kernels/bitvla/attention.h"
#include "kernels/bitvla/bitvla_fused.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" void bitlinear_int8xint2_m(int8_t* A, int8_t* B, vla_bf16* out,
                                      float* s, float* ws,
                                      int M, int N, int K, vla_stream stream);
extern "C" void bitvla_act_quant_cuda(const vla_bf16* in, int8_t* out,
                                      float* scales,
                                      int M, int K, vla_stream stream);

#define DEV_OK(call) do { if ((call) != 0) { \
    std::fprintf(stderr, "vla(bitvla_lm): %s @ %s:%d\n", vla_dev_error(), __FILE__, __LINE__); \
    return -1; } } while (0)

#define DEV_OKV(assign) do { if (!(assign)) { \
    std::fprintf(stderr, "vla(bitvla_lm): %s @ %s:%d\n", vla_dev_error(), __FILE__, __LINE__); \
    bitvla_lm_cuda_free(ctx); return nullptr; } } while (0)

struct bitvla_lm_cuda_ctx {
    int hidden, n_q, n_kv, head_dim, ffn, n_layers, max_seq;
    float rope_base, rms_eps;
    int hidden_q, hidden_kv;

    std::vector<bitvla_lm_layer_cuda> layers;
    vla_bf16* output_norm_w = nullptr;

    float* d_cos = nullptr;
    float* d_sin = nullptr;

    // Workspaces are vla_bf16 so the driver below can hand them straight to the
    // entry points; the GEMMs take them the same way, and only the kernel
    // sources ever reinterpret them as a native bf16 type.
    vla_bf16*      d_h         = nullptr;
    vla_bf16*      d_h_norm    = nullptr;
    int8_t*        d_act_int8_h= nullptr;
    float*         d_act_s     = nullptr;
    vla_bf16*      d_qkv       = nullptr;
    vla_bf16*      d_q_HShd    = nullptr;
    vla_bf16*      d_k_HShd    = nullptr;
    vla_bf16*      d_v_HShd    = nullptr;
    vla_bf16*      d_k_rep     = nullptr;
    vla_bf16*      d_v_rep     = nullptr;
    vla_bf16*      d_scores    = nullptr;
    vla_bf16*      d_attn_out  = nullptr;
    vla_bf16*      d_attn_merged = nullptr;
    vla_bf16*      d_o_out     = nullptr;
    int8_t*        d_act_int8_ffn = nullptr;
    vla_bf16*      d_gate_up   = nullptr;
    vla_bf16*      d_gate_sq_up= nullptr;
    vla_bf16*      d_down_out  = nullptr;
};

extern "C" bitvla_lm_cuda_ctx* bitvla_lm_cuda_init(int hidden, int n_q, int n_kv, int head_dim,
                                                   int ffn, int n_layers,
                                                   float rope_base, float rms_eps, int max_seq)
{
    auto* ctx = new bitvla_lm_cuda_ctx{};
    ctx->hidden = hidden; ctx->n_q = n_q; ctx->n_kv = n_kv; ctx->head_dim = head_dim;
    ctx->ffn = ffn; ctx->n_layers = n_layers;
    ctx->rope_base = rope_base; ctx->rms_eps = rms_eps; ctx->max_seq = max_seq;
    ctx->hidden_q  = n_q  * head_dim;
    ctx->hidden_kv = n_kv * head_dim;
    ctx->layers.resize(n_layers);

    // RoPE tables are built on the host and uploaded once. max_seq is a few
    // hundred, so the cost is nothing, and computing them on device would mean a
    // second trig implementation to keep numerically aligned with the CPU path -
    // and a per-backend one at that.
    const int half = head_dim/2;
    std::vector<float> h_cos((size_t)max_seq * half), h_sin((size_t)max_seq * half);
    for (int s=0; s<max_seq; ++s) {
        for (int k=0; k<half; ++k) {
            float freq = 1.0f/std::pow(rope_base, (float)(2*k)/(float)head_dim);
            float ang  = (float)s * freq;
            h_cos[(size_t)s * half+k] = std::cos(ang);
            h_sin[(size_t)s * half+k] = std::sin(ang);
        }
    }

    auto dev_bf16 = [](size_t n) { return (vla_bf16*) vla_dev_malloc(n * sizeof(vla_bf16)); };
    auto dev_f32  = [](size_t n) { return (float*)    vla_dev_malloc(n * sizeof(float)); };

    DEV_OKV(ctx->d_cos = dev_f32((size_t)max_seq * half));
    DEV_OKV(ctx->d_sin = dev_f32((size_t)max_seq * half));
    if (vla_dev_memcpy_h2d(ctx->d_cos, h_cos.data(), h_cos.size()*sizeof(float)) != 0 ||
        vla_dev_memcpy_h2d(ctx->d_sin, h_sin.data(), h_sin.size()*sizeof(float)) != 0) {
        std::fprintf(stderr, "vla(bitvla_lm): rope table upload: %s\n", vla_dev_error());
        bitvla_lm_cuda_free(ctx);
        return nullptr;
    }

    DEV_OKV(ctx->d_h            = dev_bf16((size_t)max_seq * hidden));
    DEV_OKV(ctx->d_h_norm       = dev_bf16((size_t)max_seq * hidden));
    DEV_OKV(ctx->d_act_int8_h   = (int8_t*) vla_dev_malloc((size_t)max_seq * hidden));
    DEV_OKV(ctx->d_act_s        = dev_f32((size_t)max_seq));
    DEV_OKV(ctx->d_qkv          = dev_bf16((size_t)max_seq * (ctx->hidden_q+2*ctx->hidden_kv)));
    DEV_OKV(ctx->d_q_HShd       = dev_bf16((size_t)n_q  * max_seq * head_dim));
    DEV_OKV(ctx->d_k_HShd       = dev_bf16((size_t)n_kv * max_seq * head_dim));
    DEV_OKV(ctx->d_v_HShd       = dev_bf16((size_t)n_kv * max_seq * head_dim));
    DEV_OKV(ctx->d_k_rep        = dev_bf16((size_t)n_q  * max_seq * head_dim));
    DEV_OKV(ctx->d_v_rep        = dev_bf16((size_t)n_q  * max_seq * head_dim));
    DEV_OKV(ctx->d_scores       = dev_bf16((size_t)n_q  * max_seq * max_seq));
    DEV_OKV(ctx->d_attn_out     = dev_bf16((size_t)n_q  * max_seq * head_dim));
    DEV_OKV(ctx->d_attn_merged  = dev_bf16((size_t)max_seq * ctx->hidden_q));
    DEV_OKV(ctx->d_o_out        = dev_bf16((size_t)max_seq * hidden));
    DEV_OKV(ctx->d_act_int8_ffn = (int8_t*) vla_dev_malloc((size_t)max_seq * ffn));
    DEV_OKV(ctx->d_gate_up      = dev_bf16((size_t)max_seq * 2 * ffn));
    DEV_OKV(ctx->d_gate_sq_up   = dev_bf16((size_t)max_seq * ffn));
    DEV_OKV(ctx->d_down_out     = dev_bf16((size_t)max_seq * hidden));
    return ctx;
}

extern "C" void bitvla_lm_cuda_free(bitvla_lm_cuda_ctx* ctx) {
    if (!ctx)
        return;
    vla_dev_free(ctx->d_cos);  vla_dev_free(ctx->d_sin);
    vla_dev_free(ctx->d_h);    vla_dev_free(ctx->d_h_norm);
    vla_dev_free(ctx->d_act_int8_h);    vla_dev_free(ctx->d_act_s);
    vla_dev_free(ctx->d_qkv);
    vla_dev_free(ctx->d_q_HShd); vla_dev_free(ctx->d_k_HShd); vla_dev_free(ctx->d_v_HShd);
    vla_dev_free(ctx->d_k_rep);  vla_dev_free(ctx->d_v_rep);
    vla_dev_free(ctx->d_scores); vla_dev_free(ctx->d_attn_out); vla_dev_free(ctx->d_attn_merged);
    vla_dev_free(ctx->d_o_out);
    vla_dev_free(ctx->d_act_int8_ffn);
    vla_dev_free(ctx->d_gate_up); vla_dev_free(ctx->d_gate_sq_up); vla_dev_free(ctx->d_down_out);
    delete ctx;
}

extern "C" void bitvla_lm_cuda_set_layer(bitvla_lm_cuda_ctx* ctx, int L,
                                         const bitvla_lm_layer_cuda* layer) {
    ctx->layers[L] = *layer;
}
extern "C" void bitvla_lm_cuda_set_output_norm(bitvla_lm_cuda_ctx* ctx, const vla_bf16* w) {
    ctx->output_norm_w = const_cast<vla_bf16*>(w);
}

namespace {

// Dump a bf16 device buffer as fp32, for the per-op comparisons in
// ci/reference/l0. Widening on the host keeps this file free of any bf16
// arithmetic, and the reference files are fp32 anyway.
void dump_bf16(const char* dir, const char* name, const vla_bf16* d_ptr, size_t n,
               vla_stream stream) {
    if (!dir)
        return;
    vla_dev_sync(stream);
    std::vector<vla_bf16> tmp(n);
    std::vector<float>    f32(n);
    if (vla_dev_memcpy_d2h(tmp.data(), d_ptr, n * sizeof(vla_bf16)) != 0)
        return;
    for (size_t i=0; i<n; ++i) {
        uint32_t b = ((uint32_t) tmp[i]) << 16;
        float f; std::memcpy(&f, &b, 4);
        f32[i] = f;
    }
    std::string path = std::string(dir) + "/" + name + ".bin";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f) {
        std::fwrite(f32.data(), sizeof(float), n, f);
        std::fclose(f);
    }
}

}  // namespace

static int run_layer(bitvla_lm_cuda_ctx* ctx, int L, int seq, vla_stream stream) {
    const auto& lr = ctx->layers[L];
    const int hidden = ctx->hidden, n_q = ctx->n_q, n_kv = ctx->n_kv, hd = ctx->head_dim;
    const int ffn = ctx->ffn, hq = ctx->hidden_q, hkv = ctx->hidden_kv;

    const char* l0_dir = (L == 0) ? std::getenv("VLA_BITVLA_DUMP_L0") : nullptr;
    auto l0_dump = [&](const char* name, const vla_bf16* d_ptr, size_t n) {
        dump_bf16(l0_dir, name, d_ptr, n, stream);
    };

    bitvla_rmsnorm_bf16(ctx->d_h, lr.attn_norm_w, ctx->d_h_norm, ctx->rms_eps, seq, hidden, stream);
    l0_dump("L0_01_attn_norm", ctx->d_h_norm, (size_t)seq * hidden);

    bitvla_act_quant_cuda(ctx->d_h_norm, ctx->d_act_int8_h, ctx->d_act_s, seq, hidden, stream);

    vla_bf16* q_dense = ctx->d_qkv;
    vla_bf16* k_dense = ctx->d_qkv+(size_t)seq * hq;
    vla_bf16* v_dense = ctx->d_qkv+(size_t)seq * (hq+hkv);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.q_packed, q_dense,
                          ctx->d_act_s, lr.q_ws, seq, hq,  hidden, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.k_packed, k_dense,
                          ctx->d_act_s, lr.k_ws, seq, hkv, hidden, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.v_packed, v_dense,
                          ctx->d_act_s, lr.v_ws, seq, hkv, hidden, stream);
    l0_dump("L0_02_qkv_proj", ctx->d_qkv, (size_t)seq * (hq+2*hkv));

    // Split the interleaved [seq, H*hd] projections into head-major [H, seq, hd]
    // with one kernel per tensor instead of a strided copy per head (30 tiny
    // transfers per layer). Same transpose the ViT head-split already uses.
    bitvla_transpose_sNhd_to_NshHd_bf16(q_dense, ctx->d_q_HShd, seq, n_q,  hd, stream);
    bitvla_transpose_sNhd_to_NshHd_bf16(k_dense, ctx->d_k_HShd, seq, n_kv, hd, stream);
    bitvla_transpose_sNhd_to_NshHd_bf16(v_dense, ctx->d_v_HShd, seq, n_kv, hd, stream);

    bitvla_rope_neox_bf16(ctx->d_q_HShd, ctx->d_cos, ctx->d_sin, n_q,  seq, hd, stream);
    bitvla_rope_neox_bf16(ctx->d_k_HShd, ctx->d_cos, ctx->d_sin, n_kv, seq, hd, stream);

    bitvla_repeat_kv_bf16(ctx->d_k_HShd, ctx->d_k_rep, n_q, n_kv, seq, hd, stream);
    bitvla_repeat_kv_bf16(ctx->d_v_HShd, ctx->d_v_rep, n_q, n_kv, seq, hd, stream);

    // scores[h] = Q[h] (seq x hd) * K[h]^T, one batch entry per query head. The
    // KV heads were already materialised per query head by repeat_kv above, so
    // both operands share a stride and this is a plain batched GEMM.
    if (vla_gemm_bf16_nt_batched(ctx->d_q_HShd, ctx->d_k_rep, ctx->d_scores,
                                 seq, seq, hd, n_q,
                                 (long long)seq * hd, (long long)seq * hd,
                                 (long long)seq * seq, stream) != 0) {
        std::fprintf(stderr, "vla(bitvla_lm): QK^T gemm failed @L%d\n", L);
        return -1;
    }

    const float scl = 1.0f/std::sqrt((float)hd);
    bitvla_softmax_scaled_bf16(ctx->d_scores, scl, n_q * seq, seq, stream);

    if (vla_gemm_bf16_nn_batched(ctx->d_scores, ctx->d_v_rep, ctx->d_attn_out,
                                 seq, hd, seq, n_q,
                                 (long long)seq * seq, (long long)seq * hd,
                                 (long long)seq * hd, stream) != 0) {
        std::fprintf(stderr, "vla(bitvla_lm): attn@V gemm failed @L%d\n", L);
        return -1;
    }

    bitvla_transpose_NshHd_to_sNhd_bf16(ctx->d_attn_out, ctx->d_attn_merged, n_q, seq, hd, stream);

    l0_dump("L0_03_attn_merged_pre_subnorm", ctx->d_attn_merged, (size_t)seq * hq);

    bitvla_rmsnorm_bf16(ctx->d_attn_merged, lr.attn_sub_norm_w, ctx->d_h_norm, ctx->rms_eps, seq, hq, stream);
    l0_dump("L0_04_attn_sub_norm", ctx->d_h_norm, (size_t)seq * hq);

    bitvla_act_quant_cuda(ctx->d_h_norm, ctx->d_act_int8_h, ctx->d_act_s, seq, hq, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.o_packed, ctx->d_o_out,
                          ctx->d_act_s, lr.o_ws, seq, hidden, hq, stream);

    l0_dump("L0_05_o_proj_out", ctx->d_o_out, (size_t)seq * hidden);

    bitvla_add_bf16(ctx->d_h, ctx->d_o_out, ctx->d_h, seq * hidden, stream);
    l0_dump("L0_06_after_attn_residual", ctx->d_h, (size_t)seq * hidden);

    bitvla_rmsnorm_bf16(ctx->d_h, lr.ffn_norm_w, ctx->d_h_norm, ctx->rms_eps, seq, hidden, stream);
    l0_dump("L0_07_ffn_norm", ctx->d_h_norm, (size_t)seq * hidden);
    bitvla_act_quant_cuda(ctx->d_h_norm, ctx->d_act_int8_h, ctx->d_act_s, seq, hidden, stream);

    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.gate_up_packed, ctx->d_gate_up,
                          ctx->d_act_s, lr.gate_up_ws, seq, 2*ffn, hidden, stream);

    gate_up_fused_sqrelu_mul_bf16(ctx->d_gate_up, ctx->d_gate_sq_up, seq, ffn, stream);

    bitvla_rmsnorm_bf16(ctx->d_gate_sq_up, lr.ffn_sub_norm_w, ctx->d_gate_sq_up,
                        ctx->rms_eps, seq, ffn, stream);

    l0_dump("L0_08_ffn_sub_norm", ctx->d_gate_sq_up, (size_t)seq * ffn);

    bitvla_act_quant_cuda(ctx->d_gate_sq_up, ctx->d_act_int8_ffn, ctx->d_act_s, seq, ffn, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_ffn, lr.down_packed, ctx->d_down_out,
                          ctx->d_act_s, lr.down_ws, seq, hidden, ffn, stream);
    l0_dump("L0_09_down_out", ctx->d_down_out, (size_t)seq * hidden);

    bitvla_add_bf16(ctx->d_h, ctx->d_down_out, ctx->d_h, seq * hidden, stream);
    l0_dump("L0_10_final", ctx->d_h, (size_t)seq * hidden);

    return 0;
}

#ifdef VLA_BITVLA_FUSED_OPS
// The two norm sites before attention and the FFN, the attention sub-norm and
// the FFN gate+sub-norm are each one row kernel (see bitvla_fused.h), and the
// FFN's residual add is deferred into the *next* layer's input norm - so on
// return d_h is still missing this layer's down projection, which sits in
// d_down_out. The caller folds it in.
//
// Same bits as run_layer: every fused kernel reproduces the chain it replaces
// exactly. The attention block in between is shared code.
static int attention_block(bitvla_lm_cuda_ctx* ctx, const bitvla_lm_layer_cuda& lr, int L,
                           int seq, vla_stream stream);

static int run_layer_fused(bitvla_lm_cuda_ctx* ctx, int L, int seq, vla_stream stream) {
    const auto& lr = ctx->layers[L];
    const int hidden = ctx->hidden, ffn = ctx->ffn, hq = ctx->hidden_q;

    bitvla_add_rmsnorm_quant_bf16(ctx->d_h, L == 0 ? nullptr : ctx->d_down_out, lr.attn_norm_w,
                                  ctx->d_act_int8_h, ctx->d_act_s, ctx->rms_eps, seq, hidden,
                                  stream);

    if (attention_block(ctx, lr, L, seq, stream) != 0)
        return -1;

    bitvla_add_rmsnorm_quant_bf16(ctx->d_attn_merged, nullptr, lr.attn_sub_norm_w,
                                  ctx->d_act_int8_h, ctx->d_act_s, ctx->rms_eps, seq, hq, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.o_packed, ctx->d_o_out,
                          ctx->d_act_s, lr.o_ws, seq, hidden, hq, stream);

    bitvla_add_rmsnorm_quant_bf16(ctx->d_h, ctx->d_o_out, lr.ffn_norm_w, ctx->d_act_int8_h,
                                  ctx->d_act_s, ctx->rms_eps, seq, hidden, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.gate_up_packed, ctx->d_gate_up,
                          ctx->d_act_s, lr.gate_up_ws, seq, 2*ffn, hidden, stream);

    bitvla_sqrelu_rmsnorm_quant_bf16(ctx->d_gate_up, lr.ffn_sub_norm_w, ctx->d_act_int8_ffn,
                                     ctx->d_act_s, ctx->rms_eps, seq, ffn, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_ffn, lr.down_packed, ctx->d_down_out,
                          ctx->d_act_s, lr.down_ws, seq, hidden, ffn, stream);
    return 0;
}

// QKV projections through the merged attention output, from d_act_int8_h /
// d_act_s. Identical to the middle of run_layer.
static int attention_block(bitvla_lm_cuda_ctx* ctx, const bitvla_lm_layer_cuda& lr, int L,
                           int seq, vla_stream stream) {
    const int hidden = ctx->hidden, n_q = ctx->n_q, n_kv = ctx->n_kv, hd = ctx->head_dim;
    const int hq = ctx->hidden_q, hkv = ctx->hidden_kv;

    vla_bf16* q_dense = ctx->d_qkv;
    vla_bf16* k_dense = ctx->d_qkv+(size_t)seq * hq;
    vla_bf16* v_dense = ctx->d_qkv+(size_t)seq * (hq+hkv);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.q_packed, q_dense,
                          ctx->d_act_s, lr.q_ws, seq, hq,  hidden, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.k_packed, k_dense,
                          ctx->d_act_s, lr.k_ws, seq, hkv, hidden, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.v_packed, v_dense,
                          ctx->d_act_s, lr.v_ws, seq, hkv, hidden, stream);

    // RoPE in place on the row-major planes, Q and K in one launch. The fused
    // attention then reads Q, K and V where the projections left them and
    // writes the merged [seq, hq] rows the sub-norm wants: no head-major
    // transposes, no repeated KV, no scores plane (attention.h).
    bitvla_rope_neox_qk_rows_bf16(q_dense, k_dense, ctx->d_cos, ctx->d_sin, seq, n_q, n_kv, hd,
                                  stream);

    const float scl = 1.0f/std::sqrt((float)hd);
    static const bool sdpa = !vla::env_flag("VLA_BITVLA_NO_SDPA") && !vla::env_flag("VLA_BITVLA_NO_SDPA_LM");
    if (sdpa && vla_attention_bf16(q_dense, k_dense, v_dense, ctx->d_attn_merged, seq, n_q, n_kv,
                                   hd, hq, hkv, hq, scl, stream) == 0)
        return 0;

    // Fallback: the unfused chain, minus the RoPE already applied above.
    bitvla_transpose_sNhd_to_NshHd_bf16(q_dense, ctx->d_q_HShd, seq, n_q,  hd, stream);
    bitvla_transpose_sNhd_to_NshHd_bf16(k_dense, ctx->d_k_HShd, seq, n_kv, hd, stream);
    bitvla_transpose_sNhd_to_NshHd_bf16(v_dense, ctx->d_v_HShd, seq, n_kv, hd, stream);

    bitvla_repeat_kv_bf16(ctx->d_k_HShd, ctx->d_k_rep, n_q, n_kv, seq, hd, stream);
    bitvla_repeat_kv_bf16(ctx->d_v_HShd, ctx->d_v_rep, n_q, n_kv, seq, hd, stream);

    if (vla_gemm_bf16_nt_batched(ctx->d_q_HShd, ctx->d_k_rep, ctx->d_scores,
                                 seq, seq, hd, n_q,
                                 (long long)seq * hd, (long long)seq * hd,
                                 (long long)seq * seq, stream) != 0) {
        std::fprintf(stderr, "vla(bitvla_lm): QK^T gemm failed @L%d\n", L);
        return -1;
    }

    bitvla_softmax_scaled_bf16(ctx->d_scores, scl, n_q * seq, seq, stream);

    if (vla_gemm_bf16_nn_batched(ctx->d_scores, ctx->d_v_rep, ctx->d_attn_out,
                                 seq, hd, seq, n_q,
                                 (long long)seq * seq, (long long)seq * hd,
                                 (long long)seq * hd, stream) != 0) {
        std::fprintf(stderr, "vla(bitvla_lm): attn@V gemm failed @L%d\n", L);
        return -1;
    }

    bitvla_transpose_NshHd_to_sNhd_bf16(ctx->d_attn_out, ctx->d_attn_merged, n_q, seq, hd, stream);
    return 0;
}

/// @c VLA_BITVLA_UNFUSED=1 forces the per-op chain, to bisect against.
static bool use_fused() {
    static const bool on = !vla::env_flag("VLA_BITVLA_UNFUSED");
    return on;
}
#endif

extern "C" int bitvla_lm_cuda_forward(bitvla_lm_cuda_ctx* ctx,
                                      const vla_bf16* d_in,
                                      vla_bf16* d_out,
                                      int seq, vla_stream stream)
{
    if (seq > ctx->max_seq) {
        std::fprintf(stderr, "vla(bitvla_lm): seq=%d > max_seq=%d\n", seq, ctx->max_seq);
        return -1;
    }

    const char* dump_dir = std::getenv("VLA_BITVLA_DUMP_LM_LAYERS");

    DEV_OK(vla_dev_memcpy_d2d(ctx->d_h, d_in, (size_t)seq * ctx->hidden*sizeof(vla_bf16), stream));
    dump_bf16(dump_dir, "lm_layer_input", ctx->d_h, (size_t)seq * ctx->hidden, stream);

#ifdef VLA_BITVLA_FUSED_OPS
    // The per-layer dumps need d_h complete after every layer, which the
    // deferred residual is not; dumping is a debugging mode, so it gets the
    // per-op chain.
    if (use_fused() && !dump_dir && !std::getenv("VLA_BITVLA_DUMP_L0")) {
        for (int L=0; L<ctx->n_layers; ++L) {
            if (run_layer_fused(ctx, L, seq, stream) != 0)
                return -1;
        }
        bitvla_add_bf16(ctx->d_h, ctx->d_down_out, ctx->d_h, seq * ctx->hidden, stream);
        bitvla_rmsnorm_bf16(ctx->d_h, ctx->output_norm_w, d_out, ctx->rms_eps, seq, ctx->hidden, stream);
        return 0;
    }
#endif

    for (int L=0; L<ctx->n_layers; ++L) {
        int rc = run_layer(ctx, L, seq, stream);
        if (rc != 0)
            return rc;
        if (dump_dir) {
            char name[64]; std::snprintf(name, sizeof(name), "lm_layer_%d", L);
            dump_bf16(dump_dir, name, ctx->d_h, (size_t)seq * ctx->hidden, stream);
        }
    }

    bitvla_rmsnorm_bf16(ctx->d_h, ctx->output_norm_w, d_out, ctx->rms_eps, seq, ctx->hidden, stream);
    dump_bf16(dump_dir, "lm_final", d_out, (size_t)seq * ctx->hidden, stream);
    return 0;
}
