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
 * @file bitvla_ops_cuda.cu
 * @brief CUDA half of the bf16 op set declared in bitvla_lm_cuda.h.
 *
 * Elementwise, norm, RoPE, softmax and transpose kernels, shared by the LM and
 * the ViT. Nothing here orchestrates a forward pass - the drivers that do live
 * in the neutral @c *_driver.cpp files and reach these through the extern "C"
 * entry points, which is what lets one driver serve both backends.
 * @c sycl/bitvla_ops_sycl.cpp is the other half.
 */

#include "bitvla_lm_cuda.h"
#include "bitnet_kernels.h"
#include "cuda_compat.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

// The kernels below still speak __nv_bfloat16; only the entry points and these
// cross-file declarations use the neutral spelling, because those are what the
// engine and the other backend see.

template <int BLOCK>
__global__ void rmsnorm_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ w,
                                     __nv_bfloat16* __restrict__ out,
                                     float eps, int K)
{
    const int m   = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const __nv_bfloat16* row = x+m * K;
    __nv_bfloat16*       o   = out+m * K;

    float ss = 0.0f;
    for (int k=tid; k<K; k += BLOCK) {
        float v = __bfloat162float(row[k]);
        ss += v * v;
    }
    for (int off=16; off>0; off >>= 1)
        ss += __shfl_down_sync(0xffffffff, ss, off);
    __shared__ float smem[32];
    if ((tid & 31) == 0)
        smem[tid >> 5] = ss;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < (BLOCK+31)/32) ? smem[tid] : 0.0f;
        for (int off=16; off>0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0)
            smem[0] = v;
    }
    __syncthreads();
    const float mean  = smem[0]/(float)K;
    const float scale = rsqrtf(mean+eps);

    for (int k=tid; k<K; k += BLOCK) {
        float v  = __bfloat162float(row[k])*scale;
        float wv = __bfloat162float(w[k]);
        o[k] = __float2bfloat16(v * wv);
    }
}

template <int BLOCK>
__global__ void rope_neox_bf16_kernel(__nv_bfloat16* __restrict__ inout,
                                       const float* __restrict__ cos_tab,
                                       const float* __restrict__ sin_tab,
                                       int S, int D)
{
    const int h    = (int)blockIdx.x;
    const int s    = (int)blockIdx.y;
    const int tid  = (int)threadIdx.x;
    const int half = D/2;
    __nv_bfloat16* row = inout+((size_t)h * S+s)*D;
    const float* c_row = cos_tab+(size_t)s * half;
    const float* s_row = sin_tab+(size_t)s * half;
    for (int k=tid; k<half; k += BLOCK) {
        float c = c_row[k];
        float si= s_row[k];
        float a = __bfloat162float(row[k]);
        float b = __bfloat162float(row[k+half]);
        row[k]        = __float2bfloat16(a * c-b * si);
        row[k+half] = __float2bfloat16(b * c+a * si);
    }
}

template <int BLOCK>
__global__ void softmax_scaled_bf16_kernel(__nv_bfloat16* __restrict__ inout,
                                            float scale, int S)
{
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    __nv_bfloat16* r = inout+(size_t)row * S;

    float mx = -INFINITY;
    for (int i=tid; i<S; i += BLOCK) {
        float v = __bfloat162float(r[i])*scale;
        if (v > mx)
            mx = v;
    }
    for (int off=16; off>0; off >>= 1) {
        float other = __shfl_down_sync(0xffffffff, mx, off);
        if (other > mx)
            mx = other;
    }
    __shared__ float smem[32];
    if ((tid & 31) == 0)
        smem[tid >> 5] = mx;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < (BLOCK+31)/32) ? smem[tid] : -INFINITY;
        for (int off=16; off>0; off >>= 1) {
            float other = __shfl_down_sync(0xffffffff, v, off);
            if (other > v)
                v = other;
        }
        if (tid == 0)
            smem[0] = v;
    }
    __syncthreads();
    const float max_v = smem[0];

    float s_sum = 0.0f;
    for (int i=tid; i<S; i += BLOCK) {
        s_sum += expf(__bfloat162float(r[i])*scale-max_v);
    }
    for (int off=16; off>0; off >>= 1)
        s_sum += __shfl_down_sync(0xffffffff, s_sum, off);
    if ((tid & 31) == 0)
        smem[tid >> 5] = s_sum;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < (BLOCK+31)/32) ? smem[tid] : 0.0f;
        for (int off=16; off>0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0)
            smem[0] = v;
    }
    __syncthreads();
    const float inv_sum = 1.0f/smem[0];

    for (int i=tid; i<S; i += BLOCK) {
        float v = expf(__bfloat162float(r[i])*scale-max_v)*inv_sum;
        r[i] = __float2bfloat16(v);
    }
}

__global__ void squared_relu_mul_bf16_kernel(const __nv_bfloat16* g,
                                              const __nv_bfloat16* u,
                                              __nv_bfloat16* out, int N) {
    const int i = (int)(blockIdx.x*blockDim.x+threadIdx.x);
    if (i >= N)
        return;
    float gv = __bfloat162float(g[i]);
    if (gv < 0.0f)
        gv = 0.0f;
    out[i] = __float2bfloat16(gv * gv * __bfloat162float(u[i]));
}

__global__ void add_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                 __nv_bfloat16* out, int N) {
    const int i = (int)(blockIdx.x*blockDim.x+threadIdx.x);
    if (i >= N)
        return;
    out[i] = __float2bfloat16(__bfloat162float(a[i])+__bfloat162float(b[i]));
}

__global__ void repeat_kv_bf16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out,
                                       int n_q, int n_kv, int seq, int hd) {
    const int q_h = (int)blockIdx.x;
    const int s   = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int kv_h = q_h * n_kv/n_q;
    for (int k=tid; k<hd; k += blockDim.x) {
        out[((size_t)q_h * seq+s)*hd+k] = in[((size_t)kv_h * seq+s)*hd+k];
    }
}

__global__ void transpose_sHhd_to_HShd_bf16_kernel(const __nv_bfloat16* in,
                                                    __nv_bfloat16* out,
                                                    int S, int H, int hd) {
    const int s   = (int)blockIdx.y;
    const int h   = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    for (int k=tid; k<hd; k += blockDim.x) {
        out[((size_t)h * S+s)*hd+k] = in[((size_t)s * H+h)*hd+k];
    }
}

__global__ void transpose_HShd_to_sHhd_bf16_kernel(const __nv_bfloat16* in,
                                                    __nv_bfloat16* out,
                                                    int H, int S, int hd) {
    const int h   = (int)blockIdx.x;
    const int s   = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    for (int k=tid; k<hd; k += blockDim.x) {
        out[((size_t)s * H+h)*hd+k] = in[((size_t)h * S+s)*hd+k];
    }
}

__global__ void gather_rows_bf16_kernel(const __nv_bfloat16* in,
                                         __nv_bfloat16* out,
                                         const int32_t* row_ids,
                                         int K) {
    const int m = (int)blockIdx.x;
    const int r = row_ids[m];
    const int tid = (int)threadIdx.x;
    for (int k=tid; k<K; k += blockDim.x) {
        out[(size_t)m * K+k] = in[(size_t)r * K+k];
    }
}

extern "C" void bitvla_rmsnorm_bf16(const vla_bf16* x, const vla_bf16* w,
                                     vla_bf16* out, float eps, int M, int K,
                                     vla_stream stream) {
    constexpr int B = 256;
    rmsnorm_bf16_kernel<B><<<dim3(M, 1, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(x), vla_cu_bf(w), vla_cu_bf(out), eps, K);
}
extern "C" void bitvla_rope_neox_bf16(vla_bf16* inout, const float* cos_tab,
                                       const float* sin_tab, int H, int S, int D,
                                       vla_stream stream) {
    constexpr int B = 128;
    rope_neox_bf16_kernel<B><<<dim3(H, S, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(inout), cos_tab, sin_tab, S, D);
}
extern "C" void bitvla_softmax_scaled_bf16(vla_bf16* inout, float scale,
                                            int n_rows, int S, vla_stream stream) {
    constexpr int B = 256;
    softmax_scaled_bf16_kernel<B><<<dim3(n_rows, 1, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(inout), scale, S);
}
extern "C" void bitvla_squared_relu_mul_bf16(const vla_bf16* g, const vla_bf16* u,
                                              vla_bf16* out, int N, vla_stream stream) {
    constexpr int B = 256;
    squared_relu_mul_bf16_kernel<<<dim3((N+B-1)/B, 1, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(g), vla_cu_bf(u), vla_cu_bf(out), N);
}
extern "C" void bitvla_add_bf16(const vla_bf16* a, const vla_bf16* b,
                                 vla_bf16* out, int N, vla_stream stream) {
    constexpr int B = 256;
    add_bf16_kernel<<<dim3((N+B-1)/B, 1, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(a), vla_cu_bf(b), vla_cu_bf(out), N);
}
extern "C" void bitvla_repeat_kv_bf16(const vla_bf16* in, vla_bf16* out,
                                       int n_q, int n_kv, int seq, int hd, vla_stream stream) {
    constexpr int B = 128;
    repeat_kv_bf16_kernel<<<dim3(n_q, seq, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(in), vla_cu_bf(out), n_q, n_kv, seq, hd);
}
extern "C" void bitvla_transpose_sNhd_to_NshHd_bf16(const vla_bf16* in, vla_bf16* out,
                                                     int S, int N, int hd, vla_stream stream) {
    constexpr int B = 64;
    transpose_sHhd_to_HShd_bf16_kernel<<<dim3(N, S, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(in), vla_cu_bf(out), S, N, hd);
}
extern "C" void bitvla_transpose_NshHd_to_sNhd_bf16(const vla_bf16* in, vla_bf16* out,
                                                     int N, int S, int hd, vla_stream stream) {
    constexpr int B = 64;
    transpose_HShd_to_sHhd_bf16_kernel<<<dim3(N, S, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(in), vla_cu_bf(out), N, S, hd);
}
extern "C" void bitvla_gather_rows_bf16(const vla_bf16* in, vla_bf16* out,
                                         const int32_t* row_ids, int n_rows, int K,
                                         vla_stream stream) {
    constexpr int B = 128;
    gather_rows_bf16_kernel<<<dim3(n_rows, 1, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(in), vla_cu_bf(out), row_ids, K);
}

__global__ void gate_up_fused_sqrelu_mul_bf16_kernel(const __nv_bfloat16* __restrict__ gu,
                                                      __nv_bfloat16* __restrict__ out,
                                                      int seq, int ffn) {
    const int idx = (int)(blockIdx.x*blockDim.x+threadIdx.x);
    const int total = seq * ffn;
    if (idx >= total)
        return;
    const int s = idx/ffn;
    const int k = idx%ffn;
    const size_t row_base = (size_t)s*2*ffn;
    float g = __bfloat162float(gu[row_base+k]);
    float u = __bfloat162float(gu[row_base+ffn+k]);
    if (g < 0.0f)
        g = 0.0f;
    out[(size_t)idx] = __float2bfloat16(g * g * u);
}
extern "C" void gate_up_fused_sqrelu_mul_bf16(const vla_bf16* gu, vla_bf16* out,
                                               int seq, int ffn, vla_stream stream) {
    const int total = seq * ffn;
    const int B = 256;
    gate_up_fused_sqrelu_mul_bf16_kernel<<<dim3((total+B-1)/B, 1, 1),
                                            dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(gu), vla_cu_bf(out), seq, ffn);
}

template <int BLOCK>
__global__ void layernorm_bias_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ w,
                                            const __nv_bfloat16* __restrict__ b,
                                            __nv_bfloat16* __restrict__ out,
                                            float eps, int K)
{
    const int m   = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const __nv_bfloat16* row = x+(size_t)m * K;
    __nv_bfloat16*       o   = out+(size_t)m * K;

    float sum = 0.0f;
    for (int k=tid; k<K; k += BLOCK)
        sum += __bfloat162float(row[k]);
    for (int off=16; off>0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    __shared__ float smem[32];
    if ((tid & 31) == 0)
        smem[tid >> 5] = sum;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < (BLOCK+31)/32) ? smem[tid] : 0.0f;
        for (int off=16; off>0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0)
            smem[0] = v;
    }
    __syncthreads();
    const float mean = smem[0]/(float)K;

    float vsum = 0.0f;
    for (int k=tid; k<K; k += BLOCK) {
        float v = __bfloat162float(row[k])-mean;
        vsum += v * v;
    }
    for (int off=16; off>0; off >>= 1)
        vsum += __shfl_down_sync(0xffffffff, vsum, off);
    if ((tid & 31) == 0)
        smem[tid >> 5] = vsum;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < (BLOCK+31)/32) ? smem[tid] : 0.0f;
        for (int off=16; off>0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0)
            smem[0] = v;
    }
    __syncthreads();
    const float inv_std = rsqrtf(smem[0]/(float)K+eps);

    for (int k=tid; k<K; k += BLOCK) {
        float v = (__bfloat162float(row[k])-mean)*inv_std;
        float wv = __bfloat162float(w[k]);
        float bv = __bfloat162float(b[k]);
        o[k] = __float2bfloat16(v * wv+bv);
    }
}

__global__ void gelu_tanh_bf16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, int N) {
    const int i = (int)(blockIdx.x*blockDim.x+threadIdx.x);
    if (i >= N)
        return;
    float x = __bfloat162float(in[i]);

    const float kAlpha = 0.7978845608028654f;
    const float kBeta  = 0.044715f;
    float u = kAlpha * (x+kBeta * x * x * x);
    float t = tanhf(u);
    out[i] = __float2bfloat16(0.5f * x * (1.0f+t));
}

__global__ void gelu_erf_bf16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, int N) {
    const int i = (int)(blockIdx.x*blockDim.x+threadIdx.x);
    if (i >= N)
        return;
    const float x = __bfloat162float(in[i]);
    const float inv_sqrt2 = 0.7071067811865475f;
    out[i] = __float2bfloat16(0.5f * x * (1.0f+erff(x * inv_sqrt2)));
}

__global__ void add_bias_bf16_kernel(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                                      __nv_bfloat16* out, int M, int K) {
    const int m = (int)blockIdx.x;
    const int k = (int)(blockIdx.y*blockDim.x+threadIdx.x);
    if (k >= K)
        return;
    const size_t i = (size_t)m * K+k;
    out[i] = __float2bfloat16(__bfloat162float(x[i])+__bfloat162float(bias[k]));
}

__global__ void zero_tail_bf16_kernel(__nv_bfloat16* x, int total_cols, int start_col) {
    const int m = (int)blockIdx.x;
    const int k = (int)(start_col+blockIdx.y*blockDim.x+threadIdx.x);
    if (k >= total_cols)
        return;
    x[(size_t)m * total_cols+k] = __float2bfloat16(0.0f);
}

extern "C" void bitvla_layernorm_bf16(const vla_bf16* x, const vla_bf16* w,
                                       const vla_bf16* b, vla_bf16* out,
                                       float eps, int M, int K, vla_stream stream) {
    constexpr int B = 256;
    layernorm_bias_bf16_kernel<B><<<dim3(M, 1, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(x), vla_cu_bf(w), vla_cu_bf(b), vla_cu_bf(out), eps, K);
}
extern "C" void bitvla_gelu_tanh_bf16(const vla_bf16* x, vla_bf16* out,
                                       int N, vla_stream stream) {
    constexpr int B = 256;
    gelu_tanh_bf16_kernel<<<dim3((N+B-1)/B, 1, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(x), vla_cu_bf(out), N);
}
extern "C" void bitvla_gelu_erf_bf16(const vla_bf16* x, vla_bf16* out,
                                      int N, vla_stream stream) {
    constexpr int B = 256;
    gelu_erf_bf16_kernel<<<dim3((N+B-1)/B, 1, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(x), vla_cu_bf(out), N);
}
extern "C" void bitvla_add_bias_bf16(const vla_bf16* x, const vla_bf16* bias,
                                      vla_bf16* out, int M, int K, vla_stream stream) {
    constexpr int B = 256;
    const int n_kb = (K+B-1)/B;
    add_bias_bf16_kernel<<<dim3(M, n_kb, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(x), vla_cu_bf(bias), vla_cu_bf(out), M, K);
}
extern "C" void bitvla_zero_tail_bf16(vla_bf16* x, int M, int total_cols,
                                       int start_col, vla_stream stream) {
    if (start_col >= total_cols)
        return;
    constexpr int B = 128;
    const int len = total_cols-start_col;
    const int n_kb = (len+B-1)/B;
    zero_tail_bf16_kernel<<<dim3(M, n_kb, 1), dim3(B, 1, 1), 0, vla_cu_stream(stream)>>>(
        vla_cu_bf(x), total_cols, start_col);
}
