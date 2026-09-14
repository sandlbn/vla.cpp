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
 * @file device.h
 * @brief The GPU surface the BitVLA kernels present to the engine.
 *
 * BitVLA is the one architecture that does not run on a ggml backend: it pins
 * its graph to the CPU and offloads the LM, the ViT and the action head through
 * hand-written kernels. That leaves @c src/models/bitvla.cpp holding the
 * allocate/upload/free glue itself, and a second backend must not mean a second
 * copy of it.
 *
 * So the kernel API is stated once, in terms that name no vendor: the entry
 * points below plus @ref vla_bf16 and @ref vla_stream. CUDA implements them in
 * @c device_cuda.cu, SYCL in @c sycl/device_sycl.cpp, and the engine calls them
 * without knowing which it got.
 *
 * The dense GEMMs the forward drivers need are stated separately, in
 * @c gemm.h - they are a library call rather than a runtime call, and on each
 * backend a different library answers.
 *
 * Every function returning @c int returns 0 on success and non-zero on failure;
 * @ref vla_dev_error describes the most recent one.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A 16-bit brain-float as it crosses the kernel API.
 *
 * Deliberately an unsigned integer rather than the backend's own bf16 class.
 * The engine never does arithmetic on these values - it widens from F32 into a
 * @c std::vector<uint16_t>, hands over the pointer, and reads the result back
 * the same way. Naming the storage instead of the vendor type keeps
 * @c bitvla.cpp free of @c cuda_bf16.h and @c sycl.hpp both, which is what lets
 * one set of call sites drive either backend. Kernel sources reinterpret to
 * their native bf16 at the entry point, where the layout is identical.
 */
typedef uint16_t vla_bf16;

/**
 * @brief Opaque stream/queue handle.
 *
 * CUDA reads it as a @c cudaStream_t, SYCL as a @c sycl::queue*. A null handle
 * selects the backend's own default in both cases, which is what every call
 * site in the engine passes today.
 */
typedef void * vla_stream;

/** @brief Number of usable devices; 0 if none or if the runtime failed to load. */
int vla_dev_count(void);

/** @brief Make @p device current for subsequent allocations and launches. */
int vla_dev_set(int device);

/** @brief Allocate @p bytes of device memory, or return @c NULL. */
void * vla_dev_malloc(size_t bytes);

/** @brief Release a pointer from @ref vla_dev_malloc; @c NULL is a no-op. */
void vla_dev_free(void * p);

/** @brief Blocking host-to-device copy. */
int vla_dev_memcpy_h2d(void * dst, const void * src, size_t bytes);

/** @brief Blocking device-to-host copy. */
int vla_dev_memcpy_d2h(void * dst, const void * src, size_t bytes);

/**
 * @brief Blocking fill of @p bytes at @p dst with the low byte of @p value.
 *
 * @c memset semantics, not @c std::fill: the value is a byte pattern. Used once
 * at init to zero a padded workspace, so a blocking call is the right shape.
 */
int vla_dev_memset(void * dst, int value, size_t bytes);

/**
 * @brief Device-to-device copy enqueued on @p stream.
 *
 * Unlike the two blocking copies above this one is ordered, not awaited: the
 * drivers issue it between kernels that already run on @p stream and rely on
 * the stream to sequence them.
 */
int vla_dev_memcpy_d2d(void * dst, const void * src, size_t bytes, vla_stream stream);

/**
 * @brief Strided device-to-device copy of @p height rows of @p width bytes.
 *
 * @param dpitch Bytes between the starts of consecutive destination rows.
 * @param spitch The same for the source.
 *
 * This is how a dense (M x N) matrix is scattered into a wider (M x N_pad)
 * buffer without a kernel. @c cudaMemcpy2DAsync and SYCL's
 * @c ext_oneapi_memcpy2d take the same six arguments in the same order.
 */
int vla_dev_memcpy2d_d2d(void * dst, size_t dpitch, const void * src, size_t spitch,
                         size_t width, size_t height, vla_stream stream);

/** @brief Host-to-device copy enqueued on @p stream rather than awaited. */
int vla_dev_memcpy_h2d_async(void * dst, const void * src, size_t bytes, vla_stream stream);

/** @brief Device-to-host copy enqueued on @p stream; pair with @ref vla_dev_sync. */
int vla_dev_memcpy_d2h_async(void * dst, const void * src, size_t bytes, vla_stream stream);

/** @brief Wait for @p stream to drain; a null handle waits on the default. */
int vla_dev_sync(vla_stream stream);

/** @brief Text for the most recent failure. Never @c NULL. */
const char * vla_dev_error(void);

#ifdef __cplusplus
}
#endif
