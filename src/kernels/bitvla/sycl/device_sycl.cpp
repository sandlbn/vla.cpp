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

// SYCL implementation of src/kernels/bitvla/device.h.
//
// One in-order queue for the whole process, created on first use. In-order is
// the point: the CUDA kernels this mirrors are launched back-to-back on a
// single stream and rely on that ordering for correctness, so the SYCL side
// must not reorder them either. Where the CUDA API takes a null stream, the
// null vla_stream lands on this queue.

#include "kernels/bitvla/device.h"
#include "kernels/bitvla/sycl/queue_sycl.h"
#include "kernels/bitvla/sycl/weight_cache.h"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <string>
#include <vector>

namespace {

std::string & last_error() {
    static std::string e = "no error";
    return e;
}

// Every GPU the platform list offers, Level Zero only, so VLA_DEVICE selects
// the same physical part here as in ggml-sycl.
//
// The filter is not a preference, it is a de-duplication. One Battlemage card
// shows up twice on this cluster - once under Level Zero and once under the
// OpenCL NEO runtime - so an unfiltered walk reports two devices for one GPU
// and makes VLA_DEVICE=1 mean "the same card, through the slower runtime"
// instead of "the second card". ggml-sycl restricts itself to Level Zero for
// the same reason, and the two indexings have to agree or the ggml graph and
// the hand-written kernels end up on different contexts.
//
// Only if no Level Zero GPU exists at all does the walk fall back, which keeps
// a machine that has just the OpenCL runtime installed working rather than
// reporting no device.
const std::vector<sycl::device> & gpu_devices() {
    static const std::vector<sycl::device> devs = [] {
        std::vector<sycl::device> l0, other;
        try {
            for (const auto & p : sycl::platform::get_platforms()) {
                const bool is_l0 = p.get_backend() == sycl::backend::ext_oneapi_level_zero;
                for (const auto & d : p.get_devices()) {
                    if (!d.is_gpu()) continue;
                    (is_l0 ? l0 : other).push_back(d);
                }
            }
        } catch (const sycl::exception & e) {
            last_error() = std::string("device enumeration: ") + e.what();
        }
        return l0.empty() ? other : l0;
    }();
    return devs;
}

int g_device_index = 0;

struct Ctx {
    sycl::device device;
    sycl::context context;
    sycl::queue  queue;

    explicit Ctx(const sycl::device & d)
        : device(d), context(d),
          queue(context, d, sycl::property::queue::in_order{}) {}
};

Ctx * g_ctx = nullptr;

}  // namespace

namespace vla {

// The kernel sources need the live queue, not a void*; this is how they get it
// without each one re-deriving the singleton.
sycl::queue & bitvla_sycl_queue(vla_stream stream) {
    if (stream) return *static_cast<sycl::queue *>(stream);
    return g_ctx->queue;
}

bool bitvla_sycl_ready() { return g_ctx != nullptr; }

}  // namespace vla

extern "C" {

int vla_dev_count(void) { return (int) gpu_devices().size(); }

int vla_dev_set(int device) {
    const auto & devs = gpu_devices();
    if (device < 0 || device >= (int) devs.size()) {
        last_error() = "device index out of range";
        return 1;
    }
    if (g_ctx && device == g_device_index) return 0;
    try {
        // Replacing the context invalidates every pointer allocated against the
        // old one, so this is only legitimate before any allocation - which is
        // how the engine uses it: one vla_dev_set at load, then uploads.
        delete g_ctx;
        g_ctx = new Ctx(devs[(size_t) device]);
        g_device_index = device;
        // One line, once: the rest of the load banner says which ggml backend
        // came up, and this says which card the hand-written kernels took - the
        // two are chosen separately and a mismatch is worth being able to see.
        //
        // The runtime is named too, and not for decoration. One B70 shows up
        // twice, once under Level Zero and once under OpenCL NEO, and
        // gpu_devices() prefers Level Zero; printing which one was taken is how
        // that stays checkable rather than assumed.
        const auto backend = g_ctx->device.get_backend();
        std::fprintf(stderr, "vla(bitvla): SYCL kernels on device %d: %s [%s]\n", device,
                     g_ctx->device.get_info<sycl::info::device::name>().c_str(),
                     backend == sycl::backend::ext_oneapi_level_zero ? "level_zero" : "other");
    } catch (const sycl::exception & e) {
        last_error() = std::string("device init: ") + e.what();
        g_ctx = nullptr;
        return 1;
    }
    return 0;
}

void * vla_dev_malloc(size_t bytes) {
    if (!g_ctx && vla_dev_set(g_device_index) != 0) return nullptr;
    try {
        void * p = sycl::malloc_device(bytes, g_ctx->device, g_ctx->context);
        if (!p) last_error() = "malloc_device returned null";
        return p;
    } catch (const sycl::exception & e) {
        last_error() = std::string("malloc_device: ") + e.what();
        return nullptr;
    }
}

void vla_dev_free(void * p) {
    if (!p || !g_ctx) return;
    // Before the address is released, not after: the oneDNN path may be holding
    // an unpacked copy of this weight matrix keyed on exactly this pointer, and
    // the allocator is free to reissue it. See sycl/weight_cache.h.
    vla::bitvla_forget_unpacked(p);
    try {
        sycl::free(p, g_ctx->context);
    } catch (const sycl::exception & e) {
        last_error() = std::string("free: ") + e.what();
    }
}

int vla_dev_memcpy_h2d(void * dst, const void * src, size_t bytes) {
    if (!g_ctx) { last_error() = "no device context"; return 1; }
    try {
        g_ctx->queue.memcpy(dst, src, bytes).wait_and_throw();
    } catch (const sycl::exception & e) {
        last_error() = std::string("memcpy h2d: ") + e.what();
        return 1;
    }
    return 0;
}

int vla_dev_memcpy_d2h(void * dst, const void * src, size_t bytes) {
    if (!g_ctx) { last_error() = "no device context"; return 1; }
    try {
        g_ctx->queue.memcpy(dst, src, bytes).wait_and_throw();
    } catch (const sycl::exception & e) {
        last_error() = std::string("memcpy d2h: ") + e.what();
        return 1;
    }
    return 0;
}

int vla_dev_memset(void * dst, int value, size_t bytes) {
    if (!g_ctx) { last_error() = "no device context"; return 1; }
    try {
        g_ctx->queue.memset(dst, value, bytes).wait_and_throw();
    } catch (const sycl::exception & e) {
        last_error() = std::string("memset: ") + e.what();
        return 1;
    }
    return 0;
}

int vla_dev_memcpy_d2d(void * dst, const void * src, size_t bytes, vla_stream stream) {
    if (!g_ctx) { last_error() = "no device context"; return 1; }
    try {
        // Not awaited: the queue is in-order, so enqueueing is already enough to
        // order this against the kernels on either side of it.
        vla::bitvla_sycl_queue(stream).memcpy(dst, src, bytes);
    } catch (const sycl::exception & e) {
        last_error() = std::string("memcpy d2d: ") + e.what();
        return 1;
    }
    return 0;
}

int vla_dev_memcpy2d_d2d(void * dst, size_t dpitch, const void * src, size_t spitch,
                         size_t width, size_t height, vla_stream stream) {
    if (!g_ctx) { last_error() = "no device context"; return 1; }
    try {
        // Argument order matches cudaMemcpy2DAsync's, which is what device.h
        // documents; the extension spells the element type as a template
        // parameter, so bytes means unsigned char here.
        vla::bitvla_sycl_queue(stream).ext_oneapi_memcpy2d(
            dst, dpitch, src, spitch, width, height);
    } catch (const sycl::exception & e) {
        last_error() = std::string("memcpy2d d2d: ") + e.what();
        return 1;
    }
    return 0;
}

int vla_dev_memcpy_h2d_async(void * dst, const void * src, size_t bytes, vla_stream stream) {
    if (!g_ctx) { last_error() = "no device context"; return 1; }
    try {
        vla::bitvla_sycl_queue(stream).memcpy(dst, src, bytes);
    } catch (const sycl::exception & e) {
        last_error() = std::string("memcpy h2d async: ") + e.what();
        return 1;
    }
    return 0;
}

int vla_dev_memcpy_d2h_async(void * dst, const void * src, size_t bytes, vla_stream stream) {
    if (!g_ctx) { last_error() = "no device context"; return 1; }
    try {
        vla::bitvla_sycl_queue(stream).memcpy(dst, src, bytes);
    } catch (const sycl::exception & e) {
        last_error() = std::string("memcpy d2h async: ") + e.what();
        return 1;
    }
    return 0;
}

int vla_dev_sync(vla_stream stream) {
    if (!g_ctx) { last_error() = "no device context"; return 1; }
    try {
        vla::bitvla_sycl_queue(stream).wait_and_throw();
    } catch (const sycl::exception & e) {
        last_error() = std::string("sync: ") + e.what();
        return 1;
    }
    return 0;
}

const char * vla_dev_error(void) { return last_error().c_str(); }

}  // extern "C"
