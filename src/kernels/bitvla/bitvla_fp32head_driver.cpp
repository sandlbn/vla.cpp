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
 * @file bitvla_fp32head_driver.cpp
 * @brief Proprioception encoder and action head, in fp32, backend-neutral.
 *
 * Two short forward passes that bracket the LM: the proprio MLP turns the robot
 * state into one LM-width token, and the action head turns the LM's last hidden
 * states into the normalised action chunk. Both stay in fp32 - see fp32_ops.h
 * for why - and both are small enough that the host round trip at either end
 * dominates the arithmetic.
 *
 * Kernels come from @c fp32_ops.h, GEMMs from @c gemm.h, memory from
 * @c device.h. Formerly the host half of @c bitvla_fp32head_cuda.cu.
 */

#include "bitvla_fp32head_cuda.h"
#include "kernels/bitvla/device.h"
#include "kernels/bitvla/fp32_ops.h"
#include "kernels/bitvla/gemm.h"

#include <cstdio>
#include <cstring>

#define DEV_OK_RET(c) do { if ((c) != 0) { \
    std::fprintf(stderr, "vla(bitvla_fp32head): %s at %s:%d\n", vla_dev_error(), __FILE__, __LINE__); \
    return -1; } } while (0)
#define DEV_OK_NULL(assign) do { if (!(assign)) { \
    std::fprintf(stderr, "vla(bitvla_fp32head): %s at %s:%d\n", vla_dev_error(), __FILE__, __LINE__); \
    bitvla_fp32head_cuda_free(ctx); return nullptr; } } while (0)

struct bitvla_fp32head_cuda_ctx {
    int proprio_dim;
    int lm_hidden;
    int chunk;
    int action_dim;
    int ah_in_dim;
    float ah_ln_eps;

    float* pp_fc1_w; float* pp_fc1_b;
    float* pp_fc2_w; float* pp_fc2_b;

    float* d_state;
    float* d_pp_h1;
    float* d_pp_out;

    float* ah_ln1_w; float* ah_ln1_b;
    float* ah_fc1_w; float* ah_fc1_b;
    float* ah_b0_lnw; float* ah_b0_lnb; float* ah_b0_w; float* ah_b0_b;
    float* ah_b1_lnw; float* ah_b1_lnb; float* ah_b1_w; float* ah_b1_b;
    float* ah_ln2_w; float* ah_ln2_b;
    float* ah_fc2_w; float* ah_fc2_b;

    float* d_ah_in;
    float* d_ah_norm_big;
    float* d_ah_h;
    float* d_ah_tmp;
    float* d_ah_tmp2;
    float* d_ah_out;
};

namespace {

float* upload_f32(const float* h, size_t n) {
    float* d = (float*) vla_dev_malloc(n * sizeof(float));
    if (!d) {
        std::fprintf(stderr, "vla(bitvla_fp32head): alloc failed (%zu): %s\n", n, vla_dev_error());
        return nullptr;
    }
    if (vla_dev_memcpy_h2d(d, h, n * sizeof(float)) != 0) {
        std::fprintf(stderr, "vla(bitvla_fp32head): H2D failed (%zu): %s\n", n, vla_dev_error());
        vla_dev_free(d);
        return nullptr;
    }
    return d;
}

/**
 * @brief out(M x N_out) = x(M x K_in) * W(N_out x K_in)^T, then bias.
 *
 * Every linear layer in this file has that shape - weights stored output-major,
 * which is how both the checkpoint and @ref vla_gemm_f32_nt want them - so the
 * whole head is this call with different operands. Bias stays a separate kernel
 * because the GEMM does not fuse one.
 */
int linear_bias_fp32(const float* W, const float* bias, const float* x, float* out,
                     int M, int N_out, int K_in, vla_stream stream) {
    if (vla_gemm_f32_nt(x, W, out, M, N_out, K_in, stream) != 0) {
        std::fprintf(stderr, "vla(bitvla_fp32head): sgemm failed M=%d N=%d K=%d\n", M, N_out, K_in);
        return -1;
    }
    if (bias)
        vla_add_bias_fp32(out, bias, out, M, N_out, stream);
    return 0;
}

}  // namespace

extern "C" bitvla_fp32head_cuda_ctx* bitvla_fp32head_cuda_init(
    int proprio_dim, int lm_hidden, int chunk, int action_dim,
    float ah_ln_eps,
    const float* pp_fc1_w, const float* pp_fc1_b,
    const float* pp_fc2_w, const float* pp_fc2_b,
    const float* ah_ln1_w, const float* ah_ln1_b,
    const float* ah_fc1_w, const float* ah_fc1_b,
    const float* ah_b0_lnw, const float* ah_b0_lnb,
    const float* ah_b0_w,   const float* ah_b0_b,
    const float* ah_b1_lnw, const float* ah_b1_lnb,
    const float* ah_b1_w,   const float* ah_b1_b,
    const float* ah_ln2_w,  const float* ah_ln2_b,
    const float* ah_fc2_w,  const float* ah_fc2_b)
{
    auto* ctx = new bitvla_fp32head_cuda_ctx;
    std::memset(ctx, 0, sizeof(*ctx));
    ctx->proprio_dim = proprio_dim;
    ctx->lm_hidden   = lm_hidden;
    ctx->chunk       = chunk;
    ctx->action_dim  = action_dim;
    ctx->ah_in_dim   = action_dim * lm_hidden;
    ctx->ah_ln_eps   = ah_ln_eps;

    ctx->pp_fc1_w = upload_f32(pp_fc1_w, (size_t)lm_hidden * proprio_dim);
    ctx->pp_fc1_b = upload_f32(pp_fc1_b, (size_t)lm_hidden);
    ctx->pp_fc2_w = upload_f32(pp_fc2_w, (size_t)lm_hidden * lm_hidden);
    ctx->pp_fc2_b = upload_f32(pp_fc2_b, (size_t)lm_hidden);

    ctx->ah_ln1_w = upload_f32(ah_ln1_w, (size_t)ctx->ah_in_dim);
    ctx->ah_ln1_b = upload_f32(ah_ln1_b, (size_t)ctx->ah_in_dim);
    ctx->ah_fc1_w = upload_f32(ah_fc1_w, (size_t)lm_hidden * ctx->ah_in_dim);
    ctx->ah_fc1_b = upload_f32(ah_fc1_b, (size_t)lm_hidden);

    ctx->ah_b0_lnw = upload_f32(ah_b0_lnw, (size_t)lm_hidden);
    ctx->ah_b0_lnb = upload_f32(ah_b0_lnb, (size_t)lm_hidden);
    ctx->ah_b0_w   = upload_f32(ah_b0_w,   (size_t)lm_hidden * lm_hidden);
    ctx->ah_b0_b   = upload_f32(ah_b0_b,   (size_t)lm_hidden);

    ctx->ah_b1_lnw = upload_f32(ah_b1_lnw, (size_t)lm_hidden);
    ctx->ah_b1_lnb = upload_f32(ah_b1_lnb, (size_t)lm_hidden);
    ctx->ah_b1_w   = upload_f32(ah_b1_w,   (size_t)lm_hidden * lm_hidden);
    ctx->ah_b1_b   = upload_f32(ah_b1_b,   (size_t)lm_hidden);

    ctx->ah_ln2_w = upload_f32(ah_ln2_w, (size_t)lm_hidden);
    ctx->ah_ln2_b = upload_f32(ah_ln2_b, (size_t)lm_hidden);
    ctx->ah_fc2_w = upload_f32(ah_fc2_w, (size_t)action_dim * lm_hidden);
    ctx->ah_fc2_b = upload_f32(ah_fc2_b, (size_t)action_dim);

    DEV_OK_NULL(ctx->d_state  = (float*) vla_dev_malloc((size_t)proprio_dim * sizeof(float)));
    DEV_OK_NULL(ctx->d_pp_h1  = (float*) vla_dev_malloc((size_t)lm_hidden  * sizeof(float)));
    DEV_OK_NULL(ctx->d_pp_out = (float*) vla_dev_malloc((size_t)lm_hidden  * sizeof(float)));

    DEV_OK_NULL(ctx->d_ah_in       = (float*) vla_dev_malloc((size_t)chunk * ctx->ah_in_dim*sizeof(float)));
    DEV_OK_NULL(ctx->d_ah_norm_big = (float*) vla_dev_malloc((size_t)chunk * ctx->ah_in_dim*sizeof(float)));
    DEV_OK_NULL(ctx->d_ah_h        = (float*) vla_dev_malloc((size_t)chunk * lm_hidden  * sizeof(float)));
    DEV_OK_NULL(ctx->d_ah_tmp      = (float*) vla_dev_malloc((size_t)chunk * lm_hidden  * sizeof(float)));
    DEV_OK_NULL(ctx->d_ah_tmp2     = (float*) vla_dev_malloc((size_t)chunk * lm_hidden  * sizeof(float)));
    DEV_OK_NULL(ctx->d_ah_out      = (float*) vla_dev_malloc((size_t)chunk * action_dim * sizeof(float)));

    // The uploads above report failure by returning null rather than unwinding,
    // so they are checked together here; the workspace allocations unwind on the
    // spot through DEV_OK_NULL.
    void* required[] = {
        ctx->pp_fc1_w, ctx->pp_fc1_b, ctx->pp_fc2_w, ctx->pp_fc2_b,
        ctx->ah_ln1_w, ctx->ah_ln1_b, ctx->ah_fc1_w, ctx->ah_fc1_b,
        ctx->ah_b0_lnw, ctx->ah_b0_lnb, ctx->ah_b0_w, ctx->ah_b0_b,
        ctx->ah_b1_lnw, ctx->ah_b1_lnb, ctx->ah_b1_w, ctx->ah_b1_b,
        ctx->ah_ln2_w, ctx->ah_ln2_b, ctx->ah_fc2_w, ctx->ah_fc2_b,
    };
    for (void* p : required) if (!p) {
        std::fprintf(stderr, "vla(bitvla_fp32head): a weight upload returned null; init failed\n");
        bitvla_fp32head_cuda_free(ctx);
        return nullptr;
    }
    return ctx;
}

extern "C" int bitvla_fp32head_proprio_forward(
    bitvla_fp32head_cuda_ctx* ctx,
    const float* host_state,
    float* host_out,
    vla_stream stream)
{
    DEV_OK_RET(vla_dev_memcpy_h2d_async(ctx->d_state, host_state,
                                        (size_t)ctx->proprio_dim*sizeof(float), stream));

    if (linear_bias_fp32(ctx->pp_fc1_w, ctx->pp_fc1_b,
                         ctx->d_state, ctx->d_pp_h1,
                         1, ctx->lm_hidden, ctx->proprio_dim, stream) != 0) return -1;

    vla_gelu_erf_fp32(ctx->d_pp_h1, ctx->d_pp_h1, ctx->lm_hidden, stream);

    if (linear_bias_fp32(ctx->pp_fc2_w, ctx->pp_fc2_b,
                         ctx->d_pp_h1, ctx->d_pp_out,
                         1, ctx->lm_hidden, ctx->lm_hidden, stream) != 0) return -1;

    DEV_OK_RET(vla_dev_memcpy_d2h_async(host_out, ctx->d_pp_out,
                                        (size_t)ctx->lm_hidden*sizeof(float), stream));
    DEV_OK_RET(vla_dev_sync(stream));
    return 0;
}

extern "C" int bitvla_fp32head_action_forward(
    bitvla_fp32head_cuda_ctx* ctx,
    const float* host_ah_input,
    float* host_norm_actions,
    vla_stream stream)
{
    const int M  = ctx->chunk;
    const int Di = ctx->ah_in_dim;
    const int H  = ctx->lm_hidden;
    const int A  = ctx->action_dim;

    DEV_OK_RET(vla_dev_memcpy_h2d_async(ctx->d_ah_in, host_ah_input,
                                        (size_t)M * Di * sizeof(float), stream));

    vla_layernorm_fp32(ctx->d_ah_in, ctx->ah_ln1_w, ctx->ah_ln1_b, ctx->d_ah_norm_big,
                       ctx->ah_ln_eps, M, Di, stream);

    if (linear_bias_fp32(ctx->ah_fc1_w, ctx->ah_fc1_b,
                         ctx->d_ah_norm_big, ctx->d_ah_h, M, H, Di, stream) != 0) return -1;
    vla_relu_fp32(ctx->d_ah_h, M * H, stream);

    // Two residual blocks, each norm -> linear -> relu -> add into d_ah_h.
    vla_layernorm_fp32(ctx->d_ah_h, ctx->ah_b0_lnw, ctx->ah_b0_lnb, ctx->d_ah_tmp,
                       ctx->ah_ln_eps, M, H, stream);
    if (linear_bias_fp32(ctx->ah_b0_w, ctx->ah_b0_b,
                         ctx->d_ah_tmp, ctx->d_ah_tmp2, M, H, H, stream) != 0) return -1;
    vla_relu_fp32(ctx->d_ah_tmp2, M * H, stream);
    vla_add_fp32(ctx->d_ah_h, ctx->d_ah_tmp2, ctx->d_ah_h, M * H, stream);

    vla_layernorm_fp32(ctx->d_ah_h, ctx->ah_b1_lnw, ctx->ah_b1_lnb, ctx->d_ah_tmp,
                       ctx->ah_ln_eps, M, H, stream);
    if (linear_bias_fp32(ctx->ah_b1_w, ctx->ah_b1_b,
                         ctx->d_ah_tmp, ctx->d_ah_tmp2, M, H, H, stream) != 0) return -1;
    vla_relu_fp32(ctx->d_ah_tmp2, M * H, stream);
    vla_add_fp32(ctx->d_ah_h, ctx->d_ah_tmp2, ctx->d_ah_h, M * H, stream);

    vla_layernorm_fp32(ctx->d_ah_h, ctx->ah_ln2_w, ctx->ah_ln2_b, ctx->d_ah_tmp,
                       ctx->ah_ln_eps, M, H, stream);
    if (linear_bias_fp32(ctx->ah_fc2_w, ctx->ah_fc2_b,
                         ctx->d_ah_tmp, ctx->d_ah_out, M, A, H, stream) != 0) return -1;

    DEV_OK_RET(vla_dev_memcpy_d2h_async(host_norm_actions, ctx->d_ah_out,
                                        (size_t)M * A * sizeof(float), stream));
    DEV_OK_RET(vla_dev_sync(stream));
    return 0;
}

extern "C" void bitvla_fp32head_cuda_free(bitvla_fp32head_cuda_ctx* ctx) {
    if (!ctx)
        return;
    float* ws[] = {
        ctx->pp_fc1_w, ctx->pp_fc1_b, ctx->pp_fc2_w, ctx->pp_fc2_b,
        ctx->ah_ln1_w, ctx->ah_ln1_b, ctx->ah_fc1_w, ctx->ah_fc1_b,
        ctx->ah_b0_lnw, ctx->ah_b0_lnb, ctx->ah_b0_w, ctx->ah_b0_b,
        ctx->ah_b1_lnw, ctx->ah_b1_lnb, ctx->ah_b1_w, ctx->ah_b1_b,
        ctx->ah_ln2_w, ctx->ah_ln2_b, ctx->ah_fc2_w, ctx->ah_fc2_b,
        ctx->d_state, ctx->d_pp_h1, ctx->d_pp_out,
        ctx->d_ah_in, ctx->d_ah_norm_big, ctx->d_ah_h, ctx->d_ah_tmp, ctx->d_ah_tmp2, ctx->d_ah_out,
    };
    for (float* p : ws)
        vla_dev_free(p);
    delete ctx;
}
