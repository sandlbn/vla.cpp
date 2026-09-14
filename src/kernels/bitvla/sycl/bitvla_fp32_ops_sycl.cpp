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
 * @file bitvla_fp32_ops_sycl.cpp
 * @brief SYCL twins of the five fp32 ops in @c kernels/bitvla/fp32_ops.h.
 *
 * Same @c extern @c "C" names as @c bitvla_fp32_ops_cuda.cu, so
 * @c bitvla_fp32head_driver.cpp links against either. Everything the bf16 op
 * file says about reproducing CUDA's arithmetic applies here and is not
 * repeated - with one difference worth stating, since this is the path that
 * produces the numbers the robot acts on.
 *
 * These ops never leave fp32. There is no bf16 rounding to absorb a last-place
 * difference, so a divide that is 2 ULP off stays 2 ULP off all the way to the
 * action. @ref vla::bitvla::exact_div is used for both divisions below for that
 * reason; see sycl_compat.h for what the hardware does without it.
 */

#include "kernels/bitvla/fp32_ops.h"
#include "kernels/bitvla/sycl/queue_sycl.h"
#include "kernels/bitvla/sycl/sycl_compat.h"

using vla::bitvla::exact_div;
using vla::bitvla::ROW_WG;

namespace {

struct k_gelu_erf_f32 {};
struct k_relu_f32 {};
struct k_layernorm_f32 {};
struct k_add_bias_f32 {};
struct k_add_f32 {};

inline sycl::nd_range<1> flat_range(int n, int wg = ROW_WG) {
    return sycl::nd_range<1>(sycl::range<1>(vla::bitvla::round_up((size_t) n, (size_t) wg)),
                             sycl::range<1>((size_t) wg));
}

inline sycl::nd_range<2> rows_range(int rows, int wg = ROW_WG) {
    return sycl::nd_range<2>(sycl::range<2>((size_t) rows, (size_t) wg),
                             sycl::range<2>(1, (size_t) wg));
}

}  // namespace

extern "C" void vla_gelu_erf_fp32(const float* in, float* out, int N, vla_stream stream) {
    if (N <= 0)
        return;
    sycl::queue & q = vla::bitvla_sycl_queue(stream);
    q.parallel_for<k_gelu_erf_f32>(flat_range(N), [=](sycl::nd_item<1> it) {
        const int i = (int) it.get_global_id(0);
        if (i >= N)
            return;
        const float x = in[i];
        out[i] = 0.5f * x * (1.0f + sycl::erf(x * 0.70710678118654752440f));
    });
}

extern "C" void vla_relu_fp32(float* inout, int N, vla_stream stream) {
    if (N <= 0)
        return;
    sycl::queue & q = vla::bitvla_sycl_queue(stream);
    q.parallel_for<k_relu_f32>(flat_range(N), [=](sycl::nd_item<1> it) {
        const int i = (int) it.get_global_id(0);
        if (i >= N)
            return;
        // Read-compare-conditionally-write, as CUDA does, rather than an
        // unconditional fmax store: same result, and it keeps the write traffic
        // identical for anyone profiling the two side by side.
        const float v = inout[i];
        if (v < 0.0f)
            inout[i] = 0.0f;
    });
}

extern "C" void vla_layernorm_fp32(const float* x, const float* w, const float* b, float* out,
                                   float eps, int M, int K, vla_stream stream) {
    if (M <= 0 || K <= 0)
        return;
    sycl::queue & q = vla::bitvla_sycl_queue(stream);
    q.parallel_for<k_layernorm_f32>(rows_range(M), [=](sycl::nd_item<2> it) {
        const int     m   = (int) it.get_group(0);
        const int     tid = (int) it.get_local_id(1);
        const float * row = x + (size_t) m * K;
        float *       o   = out + (size_t) m * K;

        float sum = 0.0f;
        for (int k = tid; k < K; k += ROW_WG)
            sum += row[k];
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        const float mean = exact_div(sum, (float) K);

        float vsum = 0.0f;
        for (int k = tid; k < K; k += ROW_WG) {
            const float v = row[k] - mean;
            vsum += v * v;
        }
        vsum = sycl::reduce_over_group(it.get_group(), vsum, sycl::plus<float>());
        const float inv_std = sycl::rsqrt(exact_div(vsum, (float) K) + eps);

        for (int k = tid; k < K; k += ROW_WG)
            o[k] = (row[k] - mean) * inv_std * w[k] + b[k];
    });
}

extern "C" void vla_add_bias_fp32(const float* x, const float* bias, float* out,
                                  int M, int K, vla_stream stream) {
    if (M <= 0 || K <= 0)
        return;
    sycl::queue & q = vla::bitvla_sycl_queue(stream);
    q.parallel_for<k_add_bias_f32>(
        sycl::nd_range<2>(sycl::range<2>((size_t) M, vla::bitvla::round_up((size_t) K, ROW_WG)),
                          sycl::range<2>(1, ROW_WG)),
        [=](sycl::nd_item<2> it) {
            const int m = (int) it.get_group(0);
            const int k = (int) it.get_global_id(1);
            if (k >= K)
                return;
            const size_t i = (size_t) m * K + k;
            out[i] = x[i] + bias[k];
        });
}

extern "C" void vla_add_fp32(const float* a, const float* b, float* out, int N, vla_stream stream) {
    if (N <= 0)
        return;
    sycl::queue & q = vla::bitvla_sycl_queue(stream);
    q.parallel_for<k_add_f32>(flat_range(N), [=](sycl::nd_item<1> it) {
        const int i = (int) it.get_global_id(0);
        if (i >= N)
            return;
        out[i] = a[i] + b[i];
    });
}
