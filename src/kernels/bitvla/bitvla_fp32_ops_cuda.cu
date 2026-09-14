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
 * @file bitvla_fp32_ops_cuda.cu
 * @brief CUDA half of fp32_ops.h - the action head's elementwise and norm kernels.
 *
 * Lifted unchanged out of bitvla_fp32head_cuda.cu when the driver above them
 * became backend-neutral. sycl/bitvla_fp32_ops_sycl.cpp is the other half, and
 * the two are held to the same output by tests/test_bitvla_ops_gpu.
 */

#include "kernels/bitvla/cuda_compat.h"
#include "kernels/bitvla/fp32_ops.h"

#include <cuda_runtime.h>

__global__ void gelu_erf_fp32_kernel(const float* __restrict__ in, float* __restrict__ out, int N) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N)
        return;
    float x = in[i];
    out[i] = 0.5f * x * (1.0f+erff(x*0.70710678118654752440f));
}

__global__ void relu_fp32_kernel(float* __restrict__ inout, int N) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N)
        return;
    float v = inout[i];
    if (v < 0.0f)
        inout[i] = 0.0f;
}

template <int BLOCK>
__global__ void layernorm_fp32_kernel(const float* __restrict__ x,
                                       const float* __restrict__ w,
                                       const float* __restrict__ b,
                                       float* __restrict__ out,
                                       float eps, int K) {
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row = x+(size_t)m * K;
    float*       o   = out+(size_t)m * K;

    float sum = 0.0f;
    for (int k=tid; k<K; k += BLOCK)
        sum += row[k];
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
    const float mean = vla_exact_div(smem[0], (float)K);

    float vsum = 0.0f;
    for (int k=tid; k<K; k += BLOCK) {
        float v = row[k]-mean;
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
    const float inv_std = rsqrtf(vla_exact_div(smem[0], (float)K)+eps);

    for (int k=tid; k<K; k += BLOCK) {
        o[k] = (row[k]-mean)*inv_std * w[k]+b[k];
    }
}

__global__ void add_bias_fp32_kernel(const float* x, const float* bias,
                                      float* out, int M, int K) {
    const int m = blockIdx.x;
    const int k = blockIdx.y*blockDim.x+threadIdx.x;
    if (k >= K)
        return;
    const size_t i = (size_t)m * K+k;
    out[i] = x[i]+bias[k];
}

__global__ void add_fp32_kernel(const float* a, const float* b, float* out, int N) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N)
        return;
    out[i] = a[i]+b[i];
}

extern "C" void vla_gelu_erf_fp32(const float* in, float* out, int N, vla_stream stream) {
    constexpr int B = 256;
    gelu_erf_fp32_kernel<<<dim3((N+B-1)/B), dim3(B), 0, (cudaStream_t) stream>>>(in, out, N);
}

extern "C" void vla_relu_fp32(float* inout, int N, vla_stream stream) {
    constexpr int B = 256;
    relu_fp32_kernel<<<dim3((N+B-1)/B), dim3(B), 0, (cudaStream_t) stream>>>(inout, N);
}

extern "C" void vla_layernorm_fp32(const float* x, const float* w, const float* b, float* out,
                                   float eps, int M, int K, vla_stream stream) {
    constexpr int B = 256;
    layernorm_fp32_kernel<B><<<dim3(M), dim3(B), 0, (cudaStream_t) stream>>>(x, w, b, out, eps, K);
}

extern "C" void vla_add_bias_fp32(const float* x, const float* bias, float* out,
                                  int M, int K, vla_stream stream) {
    constexpr int B = 256;
    const int n_kb = (K+B-1)/B;
    add_bias_fp32_kernel<<<dim3(M, n_kb, 1), dim3(B, 1, 1), 0, (cudaStream_t) stream>>>(
        x, bias, out, M, K);
}

extern "C" void vla_add_fp32(const float* a, const float* b, float* out, int N, vla_stream stream) {
    constexpr int B = 256;
    add_fp32_kernel<<<dim3((N+B-1)/B), dim3(B), 0, (cudaStream_t) stream>>>(a, b, out, N);
}
