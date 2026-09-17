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

// Device probe for the Battlemage SYCL port. Prints KEY: value lines for the
// facts the kernel design turns on and that no datasheet settles for this exact
// part: memory ceiling, sub-group widths, SLM budget, and the int8 DPAS shapes
// joint_matrix will actually accept.
//
// Built ad hoc by ci/slurm/bmg_preflight.sbatch; not part of the CMake build.
//
//   icpx -fsycl -O2 -o bmg_probe ci/slurm/bmg_probe.cpp && ./bmg_probe

#include <sycl/sycl.hpp>
// dot_acc is sycl::ext::oneapi, not ::experimental, and sycl.hpp does not pull
// it in. This is the dp4a the M=1 ternary decode kernel reduces with.
#include <sycl/ext/oneapi/dot_product.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#define VLA_HAVE_MATRIX_HEADER 1
#endif

namespace {

const char * type_name(int t) {
#ifdef VLA_HAVE_MATRIX_HEADER
    using namespace sycl::ext::oneapi::experimental::matrix;
    switch (static_cast<matrix_type>(t)) {
        case matrix_type::bf16:  return "bf16";
        case matrix_type::fp16:  return "fp16";
        case matrix_type::tf32:  return "tf32";
        case matrix_type::fp32:  return "fp32";
        case matrix_type::fp64:  return "fp64";
        case matrix_type::sint8: return "s8";
        case matrix_type::sint16:return "s16";
        case matrix_type::sint32:return "s32";
        case matrix_type::sint64:return "s64";
        case matrix_type::uint8: return "u8";
        case matrix_type::uint16:return "u16";
        case matrix_type::uint32:return "u32";
        case matrix_type::uint64:return "u64";
        default: return "?";
    }
#else
    (void) t;
    return "?";
#endif
}

// dp4a is what the M=1 ternary decode kernel reduces with. Its presence is a
// compile-time question (does this DPC++ carry the extension) and a runtime one
// (does the result match a scalar replay), so ask both.
int32_t dp4a_ref(int32_t a, int32_t b, int32_t acc) {
    for (int i = 0; i < 4; ++i) {
        const int8_t ai = static_cast<int8_t>((a >> (8 * i)) & 0xff);
        const int8_t bi = static_cast<int8_t>((b >> (8 * i)) & 0xff);
        acc += static_cast<int32_t>(ai) * static_cast<int32_t>(bi);
    }
    return acc;
}

}  // namespace

int main() {
    std::vector<sycl::device> gpus;
    for (const auto & p : sycl::platform::get_platforms()) {
        for (const auto & d : p.get_devices()) {
            if (d.is_gpu()) gpus.push_back(d);
        }
    }
    std::printf("GPU_COUNT: %zu\n", gpus.size());
    if (gpus.empty()) {
        std::printf("PROBE: no SYCL GPU device\n");
        return 1;
    }

    const sycl::device dev = gpus.front();
    std::printf("DEVICE_NAME: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("DEVICE_VENDOR: %s\n", dev.get_info<sycl::info::device::vendor>().c_str());
    std::printf("DRIVER_VERSION: %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("GLOBAL_MEM_MB: %llu\n",
                (unsigned long long) (dev.get_info<sycl::info::device::global_mem_size>() >> 20));
    std::printf("MAX_ALLOC_MB: %llu\n",
                (unsigned long long) (dev.get_info<sycl::info::device::max_mem_alloc_size>() >> 20));
    std::printf("LOCAL_MEM_KB: %llu\n",
                (unsigned long long) (dev.get_info<sycl::info::device::local_mem_size>() >> 10));
    std::printf("MAX_COMPUTE_UNITS: %u\n", dev.get_info<sycl::info::device::max_compute_units>());
    std::printf("MAX_WORK_GROUP_SIZE: %zu\n", dev.get_info<sycl::info::device::max_work_group_size>());
    std::printf("MAX_CLOCK_MHZ: %u\n", dev.get_info<sycl::info::device::max_clock_frequency>());

    std::printf("SUB_GROUP_SIZES:");
    for (size_t s : dev.get_info<sycl::info::device::sub_group_sizes>()) std::printf(" %zu", s);
    std::printf("\n");

    // Matrix (DPAS) combinations. The port's M>1 GEMM needs s8 x s8 -> s32; the
    // dense ViT/head GEMMs want bf16 x bf16 -> fp32. Print every combination the
    // runtime advertises rather than the two we hope for, so an absent one is
    // visible as an absence and not as a silent fallback later.
#ifdef VLA_HAVE_MATRIX_HEADER
    try {
        const auto combos =
            dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        std::printf("MATRIX_COMBINATIONS: %zu\n", combos.size());
        for (const auto & c : combos) {
            std::printf("MATRIX: a=%s b=%s c=%s d=%s  M=%d N=%d K=%d  maxM=%d maxN=%d maxK=%d\n",
                        type_name((int) c.atype), type_name((int) c.btype),
                        type_name((int) c.ctype), type_name((int) c.dtype),
                        (int) c.msize, (int) c.nsize, (int) c.ksize,
                        (int) c.max_msize, (int) c.max_nsize, (int) c.max_ksize);
        }
    } catch (const sycl::exception & e) {
        std::printf("MATRIX_COMBINATIONS: query failed (%s)\n", e.what());
    }
#else
    std::printf("MATRIX_COMBINATIONS: header absent in this DPC++\n");
#endif

    // dp4a, end to end.
    try {
        sycl::queue q(dev);
        constexpr int N = 256;
        int32_t * a = sycl::malloc_shared<int32_t>(N, q);
        int32_t * b = sycl::malloc_shared<int32_t>(N, q);
        int32_t * o = sycl::malloc_shared<int32_t>(N, q);
        for (int i = 0; i < N; ++i) {
            a[i] = (int32_t) (0x7f0102ff ^ (i * 2654435761u));
            b[i] = (int32_t) (0x01ff80fe ^ (i * 40503u));
            o[i] = 0;
        }
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
             o[i] = sycl::ext::oneapi::dot_acc(a[i], b[i], 0);
         }).wait();
        int bad = 0;
        for (int i = 0; i < N; ++i) {
            if (o[i] != dp4a_ref(a[i], b[i], 0)) ++bad;
        }
        std::printf("DP4A: %s (%d/%d mismatched)\n", bad ? "MISMATCH" : "ok", bad, N);
        sycl::free(a, q); sycl::free(b, q); sycl::free(o, q);
    } catch (const sycl::exception & e) {
        std::printf("DP4A: unavailable (%s)\n", e.what());
    }

    // Achieved read bandwidth. The M=1 ternary GEMM is bandwidth bound on the
    // 2-bit weight stream, so this number, not a FLOP peak, is what the decode
    // kernel gets measured against.
    try {
        sycl::queue q(dev, sycl::property::queue::in_order{});
        constexpr size_t BYTES = 512ull << 20;
        constexpr size_t N4 = BYTES / sizeof(sycl::vec<uint32_t, 4>);
        auto * src = sycl::malloc_device<sycl::vec<uint32_t, 4>>(N4, q);
        auto * dst = sycl::malloc_device<uint32_t>(1024, q);
        q.memset(src, 1, BYTES).wait();
        auto run = [&] {
            q.parallel_for(sycl::nd_range<1>(1024 * 256, 256), [=](sycl::nd_item<1> it) {
                 const size_t gid = it.get_global_id(0);
                 const size_t stride = it.get_global_range(0);
                 uint32_t acc = 0;
                 for (size_t i = gid; i < N4; i += stride) {
                     const auto v = src[i];
                     acc += v[0] + v[1] + v[2] + v[3];
                 }
                 if (acc == 0xdeadbeefu) dst[gid % 1024] = acc;  // never taken; defeats DCE
             }).wait();
        };
        run();  // warm
        const auto t0 = std::chrono::steady_clock::now();
        constexpr int ITERS = 5;
        for (int i = 0; i < ITERS; ++i) run();
        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count() / ITERS;
        std::printf("READ_BW_GBPS: %.1f\n", (double) BYTES / secs / 1e9);
        sycl::free(src, q); sycl::free(dst, q);
    } catch (const sycl::exception & e) {
        std::printf("READ_BW_GBPS: unavailable (%s)\n", e.what());
    }

    return 0;
}
