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

// CUDA implementation of src/kernels/bitvla/device.h.
//
// A thin pass-through: every call was a direct cudaMalloc/cudaMemcpy in
// src/models/bitvla.cpp until the SYCL port needed the same call sites, so the
// only behaviour added here is remembering the last error string, which the
// engine now prints instead of cudaGetErrorString.

#include "kernels/bitvla/device.h"

#include <cuda_runtime.h>

namespace {

const char * g_last_error = "no error";

bool ok(cudaError_t e) {
    if (e != cudaSuccess) {
        g_last_error = cudaGetErrorString(e);
        return false;
    }
    return true;
}

}  // namespace

extern "C" {

int vla_dev_count(void) {
    int n = 0;
    if (!ok(cudaGetDeviceCount(&n))) return 0;
    return n;
}

int vla_dev_set(int device) { return ok(cudaSetDevice(device)) ? 0 : 1; }

void * vla_dev_malloc(size_t bytes) {
    void * p = nullptr;
    if (!ok(cudaMalloc(&p, bytes))) return nullptr;
    return p;
}

void vla_dev_free(void * p) {
    if (p) cudaFree(p);
}

int vla_dev_memcpy_h2d(void * dst, const void * src, size_t bytes) {
    return ok(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice)) ? 0 : 1;
}

int vla_dev_memcpy_d2h(void * dst, const void * src, size_t bytes) {
    return ok(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost)) ? 0 : 1;
}

int vla_dev_memset(void * dst, int value, size_t bytes) {
    return ok(cudaMemset(dst, value, bytes)) ? 0 : 1;
}

int vla_dev_memcpy_d2d(void * dst, const void * src, size_t bytes, vla_stream stream) {
    return ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                              static_cast<cudaStream_t>(stream))) ? 0 : 1;
}

int vla_dev_memcpy2d_d2d(void * dst, size_t dpitch, const void * src, size_t spitch,
                         size_t width, size_t height, vla_stream stream) {
    return ok(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                cudaMemcpyDeviceToDevice,
                                static_cast<cudaStream_t>(stream))) ? 0 : 1;
}

int vla_dev_memcpy_h2d_async(void * dst, const void * src, size_t bytes, vla_stream stream) {
    return ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                              static_cast<cudaStream_t>(stream))) ? 0 : 1;
}

int vla_dev_memcpy_d2h_async(void * dst, const void * src, size_t bytes, vla_stream stream) {
    return ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                              static_cast<cudaStream_t>(stream))) ? 0 : 1;
}

int vla_dev_sync(vla_stream stream) {
    return ok(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))) ? 0 : 1;
}

const char * vla_dev_error(void) { return g_last_error; }

}  // extern "C"
