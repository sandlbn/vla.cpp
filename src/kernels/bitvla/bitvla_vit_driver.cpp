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
 * @file bitvla_vit_driver.cpp
 * @brief Forward pass for BitVLA's BitSigLIP vision tower.
 *
 * Plain C++: no CUDA, no SYCL, no bf16 arithmetic. Everything this file does to
 * the GPU it does through three vendor-neutral headers - @c device.h to
 * allocate and copy, @c gemm.h for the four dense GEMMs, and the op
 * declarations in @c bitvla_lm_cuda.h for the ternary GEMM and the elementwise
 * kernels. Which backend answers is a link-time question.
 *
 * It used to be @c bitvla_vit_cuda.cu. The port to Battlemage would otherwise
 * have meant a second copy of this orchestration, and a driver is exactly the
 * wrong thing to duplicate: the layer order, the workspace aliasing and the
 * padding rule below are model properties, identical on every backend, and two
 * copies of them would be two things to keep in agreement for no gain.
 */

#include "bitvla_vit_cuda.h"
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
#include <vector>

extern "C" void bitlinear_int8xint2_m(int8_t* A, int8_t* B, vla_bf16* out,
                                      float* s, float* ws,
                                      int M, int N, int K, vla_stream stream);
extern "C" void bitvla_act_quant_cuda(const vla_bf16* in, int8_t* out,
                                      float* scales,
                                      int M, int K, vla_stream stream);
extern "C" void bitvla_act_quant_pad_cuda(const vla_bf16* in, int8_t* out,
                                          float* scales,
                                          int M, int K_in, int K_out, vla_stream stream);

// Every device.h call reports 0/non-zero and stashes its own message, so these
// differ only in what the failing caller has to return. DEV_OKV additionally
// unwinds: a half-built context has to be freed, and the nulls in it make
// vla_dev_free a no-op on the members that were never reached.
#define DEV_OK(call) do { if ((call) != 0) { \
    std::fprintf(stderr, "vla(bitvla_vit): %s @ %s:%d\n", vla_dev_error(), __FILE__, __LINE__); \
    return -1; } } while (0)

#define DEV_OKV(assign) do { if (!(assign)) { \
    std::fprintf(stderr, "vla(bitvla_vit): %s @ %s:%d\n", vla_dev_error(), __FILE__, __LINE__); \
    bitvla_vit_cuda_free(ctx); return nullptr; } } while (0)

struct bitvla_vit_cuda_ctx {
    int n_layers, hidden, n_heads, head_dim, ffn, n_patches, patch_flat, mm_out;
    float ln_eps;
    int ffn_pad;

    vla_bf16* patch_w   = nullptr;
    vla_bf16* patch_b   = nullptr;
    vla_bf16* pos_emb   = nullptr;

    vla_bf16* mm_W1     = nullptr;
    vla_bf16* mm_b1     = nullptr;
    vla_bf16* mm_W2     = nullptr;
    vla_bf16* mm_b2     = nullptr;

    std::vector<bitvla_vit_layer_cuda> layers;

    vla_bf16* d_h            = nullptr;
    vla_bf16* d_h_norm       = nullptr;
    int8_t*   d_act_int8_h   = nullptr;
    int8_t*   d_act_int8_ffn = nullptr;
    float*    d_act_s        = nullptr;
    vla_bf16* d_q_proj       = nullptr;
    vla_bf16* d_k_proj       = nullptr;
    vla_bf16* d_v_proj       = nullptr;
    vla_bf16* d_q_HShd       = nullptr;
    vla_bf16* d_k_HShd       = nullptr;
    vla_bf16* d_v_HShd       = nullptr;
    vla_bf16* d_scores       = nullptr;
    vla_bf16* d_attn_out     = nullptr;
    vla_bf16* d_attn_merged  = nullptr;
    vla_bf16* d_o_out        = nullptr;
    vla_bf16* d_fc1_dense    = nullptr;
    vla_bf16* d_fc2_out      = nullptr;
    vla_bf16* d_mm_h1        = nullptr;
};

bitvla_vit_cuda_ctx* bitvla_vit_cuda_init(int n_layers, int hidden, int n_heads, int ffn,
                                          int n_patches, int patch_flat,
                                          float ln_eps, int mm_out)
{
    auto* ctx = new bitvla_vit_cuda_ctx{};
    ctx->n_layers = n_layers;
    ctx->hidden   = hidden;
    ctx->n_heads  = n_heads;
    ctx->head_dim = hidden/n_heads;
    ctx->ffn      = ffn;
    ctx->n_patches  = n_patches;
    ctx->patch_flat = patch_flat;
    ctx->ln_eps   = ln_eps;
    ctx->mm_out   = mm_out;

    ctx->ffn_pad = ((ffn+127)/128)*128;
    ctx->layers.resize(n_layers);

    auto dev_bf16 = [](size_t n) { return (vla_bf16*) vla_dev_malloc(n * sizeof(vla_bf16)); };

    DEV_OKV(ctx->d_h            = dev_bf16((size_t) n_patches * hidden));
    DEV_OKV(ctx->d_h_norm       = dev_bf16((size_t) n_patches * hidden));
    DEV_OKV(ctx->d_act_int8_h   = (int8_t*) vla_dev_malloc((size_t) n_patches * hidden));
    DEV_OKV(ctx->d_act_int8_ffn = (int8_t*) vla_dev_malloc((size_t) n_patches * ctx->ffn_pad));
    DEV_OKV(ctx->d_act_s        = (float*)  vla_dev_malloc((size_t) n_patches * sizeof(float)));
    // 3x: the fused path writes one concatenated [seq, 3*hidden] QKV plane here.
    DEV_OKV(ctx->d_q_proj       = dev_bf16((size_t) n_patches * 3 * hidden));
    DEV_OKV(ctx->d_k_proj       = dev_bf16((size_t) n_patches * hidden));
    DEV_OKV(ctx->d_v_proj       = dev_bf16((size_t) n_patches * hidden));
    DEV_OKV(ctx->d_q_HShd       = dev_bf16((size_t) n_heads * n_patches * ctx->head_dim));
    DEV_OKV(ctx->d_k_HShd       = dev_bf16((size_t) n_heads * n_patches * ctx->head_dim));
    DEV_OKV(ctx->d_v_HShd       = dev_bf16((size_t) n_heads * n_patches * ctx->head_dim));
    DEV_OKV(ctx->d_scores       = dev_bf16((size_t) n_heads * n_patches * n_patches));
    DEV_OKV(ctx->d_attn_out     = dev_bf16((size_t) n_heads * n_patches * ctx->head_dim));
    DEV_OKV(ctx->d_attn_merged  = dev_bf16((size_t) n_patches * hidden));
    DEV_OKV(ctx->d_o_out        = dev_bf16((size_t) n_patches * hidden));
    DEV_OKV(ctx->d_fc1_dense    = dev_bf16((size_t) n_patches * ffn));
    DEV_OKV(ctx->d_fc2_out      = dev_bf16((size_t) n_patches * hidden));
    DEV_OKV(ctx->d_mm_h1        = dev_bf16((size_t) n_patches * mm_out));

    // No bf16 buffer in the padded width, and no memset to zero its tail: the
    // quantiser writes the [ffn, ffn_pad) tail of d_act_int8_ffn itself, so the
    // padding is materialised once per layer in int8 rather than held as a
    // second bf16 copy of the whole fc1 output.
    return ctx;
}

void bitvla_vit_cuda_free(bitvla_vit_cuda_ctx* ctx) {
    if (!ctx)
        return;
    vla_dev_free(ctx->d_h);   vla_dev_free(ctx->d_h_norm);
    vla_dev_free(ctx->d_act_int8_h);   vla_dev_free(ctx->d_act_int8_ffn);   vla_dev_free(ctx->d_act_s);
    vla_dev_free(ctx->d_q_proj); vla_dev_free(ctx->d_k_proj); vla_dev_free(ctx->d_v_proj);
    vla_dev_free(ctx->d_q_HShd); vla_dev_free(ctx->d_k_HShd); vla_dev_free(ctx->d_v_HShd);
    vla_dev_free(ctx->d_scores); vla_dev_free(ctx->d_attn_out); vla_dev_free(ctx->d_attn_merged);
    vla_dev_free(ctx->d_o_out);
    vla_dev_free(ctx->d_fc1_dense); vla_dev_free(ctx->d_fc2_out);
    vla_dev_free(ctx->d_mm_h1);
    delete ctx;
}

void bitvla_vit_cuda_set_layer(bitvla_vit_cuda_ctx* ctx, int L,
                               const bitvla_vit_layer_cuda* layer) {
    ctx->layers[L] = *layer;
}
void bitvla_vit_cuda_set_embed(bitvla_vit_cuda_ctx* ctx,
                               const vla_bf16* patch_w,
                               const vla_bf16* patch_b,
                               const vla_bf16* pos_emb) {
    ctx->patch_w = const_cast<vla_bf16*>(patch_w);
    ctx->patch_b = const_cast<vla_bf16*>(patch_b);
    ctx->pos_emb = const_cast<vla_bf16*>(pos_emb);
}
void bitvla_vit_cuda_set_mmproj(bitvla_vit_cuda_ctx* ctx,
                                const vla_bf16* W1, const vla_bf16* b1,
                                const vla_bf16* W2, const vla_bf16* b2) {
    ctx->mm_W1 = const_cast<vla_bf16*>(W1);
    ctx->mm_b1 = const_cast<vla_bf16*>(b1);
    ctx->mm_W2 = const_cast<vla_bf16*>(W2);
    ctx->mm_b2 = const_cast<vla_bf16*>(b2);
}

static int run_vit_layer(bitvla_vit_cuda_ctx* ctx, int L, vla_stream stream) {
    auto& lr = ctx->layers[L];
    const int seq = ctx->n_patches, H = ctx->hidden, n_heads = ctx->n_heads;
    const int hd  = ctx->head_dim, ffn = ctx->ffn, ffn_pad = ctx->ffn_pad;

    bitvla_layernorm_bf16(ctx->d_h, lr.ln1_w, lr.ln1_b, ctx->d_h_norm, ctx->ln_eps, seq, H, stream);
    bitvla_act_quant_cuda(ctx->d_h_norm, ctx->d_act_int8_h, ctx->d_act_s, seq, H, stream);

    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.q_packed, ctx->d_q_proj, ctx->d_act_s, lr.q_ws, seq, H, H, stream);
    bitvla_add_bias_bf16(ctx->d_q_proj, lr.q_b, ctx->d_q_proj, seq, H, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.k_packed, ctx->d_k_proj, ctx->d_act_s, lr.k_ws, seq, H, H, stream);
    bitvla_add_bias_bf16(ctx->d_k_proj, lr.k_b, ctx->d_k_proj, seq, H, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.v_packed, ctx->d_v_proj, ctx->d_act_s, lr.v_ws, seq, H, H, stream);
    bitvla_add_bias_bf16(ctx->d_v_proj, lr.v_b, ctx->d_v_proj, seq, H, stream);

    bitvla_transpose_sNhd_to_NshHd_bf16(ctx->d_q_proj, ctx->d_q_HShd, seq, n_heads, hd, stream);
    bitvla_transpose_sNhd_to_NshHd_bf16(ctx->d_k_proj, ctx->d_k_HShd, seq, n_heads, hd, stream);
    bitvla_transpose_sNhd_to_NshHd_bf16(ctx->d_v_proj, ctx->d_v_HShd, seq, n_heads, hd, stream);

    // scores[h] = Q[h] (seq x hd) * K[h]^T, one batch entry per head.
    if (vla_gemm_bf16_nt_batched(ctx->d_q_HShd, ctx->d_k_HShd, ctx->d_scores,
                                 seq, seq, hd, n_heads,
                                 (long long) seq * hd, (long long) seq * hd,
                                 (long long) seq * seq, stream) != 0) {
        std::fprintf(stderr, "vla(bitvla_vit): QK^T gemm @L%d failed\n", L);
        return -1;
    }

    const float scl = 1.0f/std::sqrt((float) hd);
    bitvla_softmax_scaled_bf16(ctx->d_scores, scl, n_heads * seq, seq, stream);

    // attn_out[h] = scores[h] (seq x seq) * V[h] (seq x hd); V is not transposed.
    if (vla_gemm_bf16_nn_batched(ctx->d_scores, ctx->d_v_HShd, ctx->d_attn_out,
                                 seq, hd, seq, n_heads,
                                 (long long) seq * seq, (long long) seq * hd,
                                 (long long) seq * hd, stream) != 0) {
        std::fprintf(stderr, "vla(bitvla_vit): attn@V gemm @L%d failed\n", L);
        return -1;
    }

    bitvla_transpose_NshHd_to_sNhd_bf16(ctx->d_attn_out, ctx->d_attn_merged, n_heads, seq, hd, stream);

    bitvla_act_quant_cuda(ctx->d_attn_merged, ctx->d_act_int8_h, ctx->d_act_s, seq, H, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.o_packed, ctx->d_o_out, ctx->d_act_s, lr.o_ws, seq, H, H, stream);
    bitvla_add_bias_bf16(ctx->d_o_out, lr.o_b, ctx->d_o_out, seq, H, stream);

    bitvla_add_bf16(ctx->d_h, ctx->d_o_out, ctx->d_h, seq * H, stream);

    bitvla_layernorm_bf16(ctx->d_h, lr.ln2_w, lr.ln2_b, ctx->d_h_norm, ctx->ln_eps, seq, H, stream);
    bitvla_act_quant_cuda(ctx->d_h_norm, ctx->d_act_int8_h, ctx->d_act_s, seq, H, stream);

    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.fc1_packed, ctx->d_fc1_dense,
                          ctx->d_act_s, lr.fc1_ws, seq, ffn, H, stream);
    bitvla_add_bias_bf16(ctx->d_fc1_dense, lr.fc1_b, ctx->d_fc1_dense, seq, ffn, stream);

    bitvla_gelu_tanh_bf16(ctx->d_fc1_dense, ctx->d_fc1_dense, seq * ffn, stream);

    // fc2's K is the padded width, so the dense fc1 rows have to reach it in the
    // wider stride. The quantiser does that widening itself rather than a copy
    // doing it first: this is the only pass that touches every element of the
    // tensor anyway, so the restride rides along for free. It used to be a
    // vla_dev_memcpy2d_d2d, and on SYCL that is ext_oneapi_memcpy2d, which Level
    // Zero has no native 2D copy behind - it fell back to a byte-granularity
    // kernel that moved 2.2 MB per block at 26 GB/s and cost 4.4 ms of a 27.2 ms
    // request, the largest single non-GEMM item in the profile. CUDA never
    // showed it because cudaMemcpy2D runs on the copy engine.
    bitvla_act_quant_pad_cuda(ctx->d_fc1_dense, ctx->d_act_int8_ffn, ctx->d_act_s,
                              seq, ffn, ffn_pad, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_ffn, lr.fc2_packed, ctx->d_fc2_out,
                          ctx->d_act_s, lr.fc2_ws, seq, H, ffn_pad, stream);
    bitvla_add_bias_bf16(ctx->d_fc2_out, lr.fc2_b, ctx->d_fc2_out, seq, H, stream);

    bitvla_add_bf16(ctx->d_h, ctx->d_fc2_out, ctx->d_h, seq * H, stream);
    return 0;
}

#ifdef VLA_BITVLA_FUSED_OPS
// run_vit_layer with its row kernels fused (bitvla_fused.h). The fc2 output's
// bias and residual are deferred into the next layer's ln1, so on return d_h
// still lacks this layer's MLP branch - it sits in d_fc2_out - and the caller
// folds it in. Same bits as run_vit_layer.
static int run_vit_layer_fused(bitvla_vit_cuda_ctx* ctx, int L, vla_stream stream) {
    auto& lr = ctx->layers[L];
    const int seq = ctx->n_patches, H = ctx->hidden, n_heads = ctx->n_heads;
    const int hd  = ctx->head_dim, ffn = ctx->ffn, ffn_pad = ctx->ffn_pad;

    const bool first = L == 0;
    bitvla_add_layernorm_quant_bf16(ctx->d_h, first ? nullptr : ctx->d_fc2_out,
                                    first ? nullptr : ctx->layers[L - 1].fc2_b,
                                    lr.ln1_w, lr.ln1_b, ctx->d_act_int8_h, ctx->d_act_s,
                                    ctx->ln_eps, seq, H, stream);

    const float scl = 1.0f/std::sqrt((float) hd);
    static bool sdpa = !vla::env_flag("VLA_BITVLA_NO_SDPA") && !vla::env_flag("VLA_BITVLA_NO_SDPA_VIT");
    bool done = false;
    if (sdpa) {
        // One QKV GEMM with the biases folded into its epilogue, then the fused
        // attention straight off the concatenated plane.
        int8_t* const         B[3]    = {lr.q_packed, lr.k_packed, lr.v_packed};
        float* const          ws[3]   = {lr.q_ws, lr.k_ws, lr.v_ws};
        const vla_bf16* const bias[3] = {lr.q_b, lr.k_b, lr.v_b};
        const int             N[3]    = {H, H, H};
        bitvla_ternary_gemm_cat3(ctx->d_act_int8_h, ctx->d_act_s, seq, H, B, ws, bias, N,
                                 ctx->d_q_proj, stream);
        const vla_bf16* qkv = ctx->d_q_proj;
        done = vla_attention_bf16(qkv, qkv + H, qkv + 2*H, ctx->d_attn_merged, seq, n_heads,
                                  n_heads, hd, 3*H, 3*H, H, scl, stream) == 0;
        if (!done) sdpa = false;
    }
    if (!done) {
        bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.q_packed, ctx->d_q_proj, ctx->d_act_s, lr.q_ws, seq, H, H, stream);
        bitvla_add_bias_bf16(ctx->d_q_proj, lr.q_b, ctx->d_q_proj, seq, H, stream);
        bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.k_packed, ctx->d_k_proj, ctx->d_act_s, lr.k_ws, seq, H, H, stream);
        bitvla_add_bias_bf16(ctx->d_k_proj, lr.k_b, ctx->d_k_proj, seq, H, stream);
        bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.v_packed, ctx->d_v_proj, ctx->d_act_s, lr.v_ws, seq, H, H, stream);
        bitvla_add_bias_bf16(ctx->d_v_proj, lr.v_b, ctx->d_v_proj, seq, H, stream);

        bitvla_transpose_sNhd_to_NshHd_bf16(ctx->d_q_proj, ctx->d_q_HShd, seq, n_heads, hd, stream);
        bitvla_transpose_sNhd_to_NshHd_bf16(ctx->d_k_proj, ctx->d_k_HShd, seq, n_heads, hd, stream);
        bitvla_transpose_sNhd_to_NshHd_bf16(ctx->d_v_proj, ctx->d_v_HShd, seq, n_heads, hd, stream);

        if (vla_gemm_bf16_nt_batched(ctx->d_q_HShd, ctx->d_k_HShd, ctx->d_scores,
                                     seq, seq, hd, n_heads,
                                     (long long) seq * hd, (long long) seq * hd,
                                     (long long) seq * seq, stream) != 0) {
            std::fprintf(stderr, "vla(bitvla_vit): QK^T gemm @L%d failed\n", L);
            return -1;
        }
        bitvla_softmax_scaled_bf16(ctx->d_scores, scl, n_heads * seq, seq, stream);
        if (vla_gemm_bf16_nn_batched(ctx->d_scores, ctx->d_v_HShd, ctx->d_attn_out,
                                     seq, hd, seq, n_heads,
                                     (long long) seq * seq, (long long) seq * hd,
                                     (long long) seq * hd, stream) != 0) {
            std::fprintf(stderr, "vla(bitvla_vit): attn@V gemm @L%d failed\n", L);
            return -1;
        }
        bitvla_transpose_NshHd_to_sNhd_bf16(ctx->d_attn_out, ctx->d_attn_merged, n_heads, seq, hd, stream);
    }

    bitvla_act_quant_cuda(ctx->d_attn_merged, ctx->d_act_int8_h, ctx->d_act_s, seq, H, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.o_packed, ctx->d_o_out, ctx->d_act_s, lr.o_ws, seq, H, H, stream);

    bitvla_add_layernorm_quant_bf16(ctx->d_h, ctx->d_o_out, lr.o_b, lr.ln2_w, lr.ln2_b,
                                    ctx->d_act_int8_h, ctx->d_act_s, ctx->ln_eps, seq, H, stream);

    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.fc1_packed, ctx->d_fc1_dense,
                          ctx->d_act_s, lr.fc1_ws, seq, ffn, H, stream);
    bitvla_bias_gelu_quant_pad_bf16(ctx->d_fc1_dense, lr.fc1_b, ctx->d_act_int8_ffn,
                                    ctx->d_act_s, seq, ffn, ffn_pad, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_ffn, lr.fc2_packed, ctx->d_fc2_out,
                          ctx->d_act_s, lr.fc2_ws, seq, H, ffn_pad, stream);
    return 0;
}
#endif

int bitvla_vit_cuda_forward(bitvla_vit_cuda_ctx* ctx,
                            const vla_bf16* d_patches,
                            vla_bf16* d_out,
                            vla_stream stream)
{
    const int seq = ctx->n_patches, H = ctx->hidden, P = ctx->patch_flat, M = ctx->mm_out;

    // Patch embedding: (seq x patch_flat) against the (hidden x patch_flat)
    // conv weight flattened to a matrix, so the weight is the transposed side.
    if (vla_gemm_bf16_nt(d_patches, ctx->patch_w, ctx->d_h, seq, H, P, stream) != 0) {
        std::fprintf(stderr, "vla(bitvla_vit): patch_embed gemm failed\n");
        return -1;
    }

    bitvla_add_bias_bf16(ctx->d_h, ctx->patch_b, ctx->d_h, seq, H, stream);

    bitvla_add_bf16(ctx->d_h, ctx->pos_emb, ctx->d_h, seq * H, stream);

#ifdef VLA_BITVLA_FUSED_OPS
    static const bool fused = !vla::env_flag("VLA_BITVLA_UNFUSED");
    if (fused) {
        for (int L=0; L<ctx->n_layers; ++L) {
            int rc = run_vit_layer_fused(ctx, L, stream);
            if (rc != 0)
                return rc;
        }
        bitvla_bias_residual_bf16(ctx->d_h, ctx->d_fc2_out, ctx->layers[ctx->n_layers - 1].fc2_b,
                                  seq, H, stream);
    } else
#endif
    for (int L=0; L<ctx->n_layers; ++L) {
        int rc = run_vit_layer(ctx, L, stream);
        if (rc != 0)
            return rc;
    }

    if (vla_gemm_bf16_nt(ctx->d_h, ctx->mm_W1, ctx->d_mm_h1, seq, M, H, stream) != 0) {
        std::fprintf(stderr, "vla(bitvla_vit): MM linear_1 gemm failed\n");
        return -1;
    }
    bitvla_add_bias_bf16(ctx->d_mm_h1, ctx->mm_b1, ctx->d_mm_h1, seq, M, stream);
    bitvla_gelu_erf_bf16(ctx->d_mm_h1, ctx->d_mm_h1, seq * M, stream);

    if (vla_gemm_bf16_nt(ctx->d_mm_h1, ctx->mm_W2, d_out, seq, M, M, stream) != 0) {
        std::fprintf(stderr, "vla(bitvla_vit): MM linear_2 gemm failed\n");
        return -1;
    }
    bitvla_add_bias_bf16(d_out, ctx->mm_b2, d_out, seq, M, stream);
    return 0;
}
