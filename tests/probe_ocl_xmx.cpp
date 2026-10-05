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
//
// probe_ocl_xmx -- can an OpenVINO custom op reach XMX on this B70?
//
// The fused 4D int8 matmul is going in as an OpenVINO GPU CustomLayer, and that
// mechanism takes OpenCL C source, not SYCL and not Level Zero. (This repo
// prefers Level Zero everywhere it has the choice; here it does not. The GPU
// plugin reaches the device through OpenCL by its own architecture and compiles
// custom layers with the same IGC, so a custom op is OpenCL C or it is nothing.)
//
// That makes one unverified assumption load-bearing: that the systolic array is
// addressable from OpenCL C at all. tests/bench_bitvla_bmg.cpp measured 251-258
// TOPS of int8 XMX on this exact device, but it measured it through SYCL's
// joint_matrix. The OpenCL route is cl_intel_subgroup_matrix_multiply_accumulate,
// a different front door to the same hardware, and a driver can ship one without
// the other. If it is missing, the custom op tops out at whatever the vector ALUs
// do -- roughly an eighth of the XMX figure -- and the whole plan is wrong.
//
// Extension-string presence is a claim the driver makes about itself, so this
// does not stop there. It compiles a kernel that actually calls the intrinsic,
// for each M tile and each subgroup size, and reports what IGC accepted. A build
// log beats a feature string: the string says "supported", the build says "this
// exact signature, at this exact subgroup size, on this driver".
//
// Output is a design input, not a pass/fail: which (M, subgroup size) pairs
// compile is what the tiling of the real kernel gets written against.
//
//     icpx -O2 -o probe_ocl_xmx tests/probe_ocl_xmx.cpp -lOpenCL
//     OCL_ICD_VENDORS=/swtools/intel-gpu/latest/neo ./probe_ocl_xmx
//
// Exit code is 0 if the int8 intrinsic compiled at any tile, 1 otherwise.

#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// From cl_ext.h's cl_intel_required_subgroup_size, defined defensively: a header
// predating the extension would otherwise fail to compile against a driver that
// has it, which is the wrong way round for a capability probe.
#ifndef CL_DEVICE_SUB_GROUP_SIZES_INTEL
#define CL_DEVICE_SUB_GROUP_SIZES_INTEL 0x4108
#endif

static std::string dev_str(cl_device_id d, cl_device_info k) {
    size_t n = 0;
    if (clGetDeviceInfo(d, k, 0, nullptr, &n) != CL_SUCCESS || n == 0) {
        return "";
    }
    std::string s(n, '\0');
    clGetDeviceInfo(d, k, n, s.data(), nullptr);
    if (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    return s;
}

template <typename T> static T dev_val(cl_device_id d, cl_device_info k) {
    T v = T();
    clGetDeviceInfo(d, k, sizeof(v), &v, nullptr);
    return v;
}

// One kernel per (M, subgroup size). The operands are read from and written to
// global memory so that nothing can be folded away: a kernel whose result is
// unused is a kernel IGC is free to delete, and a deleted kernel compiles
// whatever the intrinsic does.
//
// Shapes follow the extension: for i8 the K dimension is 32 and B is always
// int8 (8 dwords = 32 bytes per lane); A and the accumulator scale with M.
static std::string i8_src(int m, int sg) {
    const char * vec = (m == 1) ? "int" : (m == 2) ? "int2" : (m == 4) ? "int4" : "int8";
    char buf[1400];
    snprintf(buf, sizeof(buf),
             "__attribute__((intel_reqd_sub_group_size(%d)))\n"
             "__kernel void probe(__global %s * a, __global int8 * b,\n"
             "                    __global %s * c) {\n"
             "    int i = get_global_id(0);\n"
             "    %s acc = c[i];\n"
             "    acc = intel_sub_group_i8_i8_matrix_mad_k32(a[i], b[i], acc);\n"
             "    c[i] = acc;\n"
             "}\n",
             sg, vec, vec, vec);
    return buf;
}

// The fallback the design falls back *to*, so it is worth knowing whether it is
// there. K is 16 for bf16 (half the i8 K, same 32 bytes per lane), operands are
// carried as short/ushort and the accumulator is float.
static std::string bf16_src(int m, int sg) {
    const char * av = (m == 1) ? "short" : (m == 2) ? "short2" : (m == 4) ? "short4" : "short8";
    const char * cv = (m == 1) ? "float" : (m == 2) ? "float2" : (m == 4) ? "float4" : "float8";
    char buf[1400];
    snprintf(buf, sizeof(buf),
             "__attribute__((intel_reqd_sub_group_size(%d)))\n"
             "__kernel void probe(__global %s * a, __global int8 * b,\n"
             "                    __global %s * c) {\n"
             "    int i = get_global_id(0);\n"
             "    %s acc = c[i];\n"
             "    acc = intel_sub_group_bf16_bf16_matrix_mad_k16(a[i], b[i], acc);\n"
             "    c[i] = acc;\n"
             "}\n",
             sg, av, cv, cv);
    return buf;
}

// Returns true if IGC built it. `log` receives the compiler's own words on
// failure -- "which signature does it want" is only ever answered there.
static bool try_build(cl_context ctx, cl_device_id dev, const std::string & src,
                      const char * opts, std::string & log) {
    const char * s   = src.c_str();
    size_t       len = src.size();
    cl_int       err = CL_SUCCESS;

    cl_program p = clCreateProgramWithSource(ctx, 1, &s, &len, &err);
    if (err != CL_SUCCESS) {
        log = "clCreateProgramWithSource failed: " + std::to_string(err);
        return false;
    }
    err = clBuildProgram(p, 1, &dev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t n = 0;
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        log.assign(n ? n - 1 : 0, '\0');
        if (n) {
            clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        }
    }
    clReleaseProgram(p);
    return err == CL_SUCCESS;
}

int main() {
    cl_uint np = 0;
    if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) {
        printf("PROBE: no OpenCL platforms. Is OCL_ICD_VENDORS pointed at the NEO driver?\n");
        return 1;
    }
    std::vector<cl_platform_id> plats(np);
    clGetPlatformIDs(np, plats.data(), nullptr);

    cl_device_id dev = nullptr;
    for (cl_uint i = 0; i < np && !dev; i++) {
        char pn[256] = { 0 };
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof(pn), pn, nullptr);
        cl_uint nd = 0;
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || !nd) {
            printf("platform %u: %s -- no GPU devices\n", i, pn);
            continue;
        }
        std::vector<cl_device_id> devs(nd);
        clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
        printf("platform %u: %s -- %u GPU device(s)\n", i, pn, nd);
        for (cl_uint j = 0; j < nd; j++) {
            printf("  device %u: %s\n", j, dev_str(devs[j], CL_DEVICE_NAME).c_str());
        }
        dev = devs[0];
    }
    if (!dev) {
        printf("PROBE: no GPU device on any platform.\n");
        return 1;
    }

    printf("\n=== device ===\n");
    printf("  name            : %s\n", dev_str(dev, CL_DEVICE_NAME).c_str());
    printf("  driver          : %s\n", dev_str(dev, CL_DRIVER_VERSION).c_str());
    printf("  OpenCL C        : %s\n", dev_str(dev, CL_DEVICE_OPENCL_C_VERSION).c_str());
    printf("  compute units   : %u\n", dev_val<cl_uint>(dev, CL_DEVICE_MAX_COMPUTE_UNITS));
    printf("  max clock       : %u MHz\n", dev_val<cl_uint>(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY));
    printf("  max wg size     : %zu\n", dev_val<size_t>(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE));
    printf("  local mem       : %llu KiB\n",
           (unsigned long long) (dev_val<cl_ulong>(dev, CL_DEVICE_LOCAL_MEM_SIZE) / 1024));
    printf("  global mem      : %llu MiB\n",
           (unsigned long long) (dev_val<cl_ulong>(dev, CL_DEVICE_GLOBAL_MEM_SIZE) >> 20));

    // Which subgroup sizes exist decides the kernel's whole lane layout, so read
    // it rather than assuming the 16 that Xe usually reports.
    size_t sgn = 0;
    clGetDeviceInfo(dev, CL_DEVICE_SUB_GROUP_SIZES_INTEL, 0, nullptr, &sgn);
    std::vector<size_t> sgs(sgn / sizeof(size_t));
    if (!sgs.empty()) {
        clGetDeviceInfo(dev, CL_DEVICE_SUB_GROUP_SIZES_INTEL, sgn, sgs.data(), nullptr);
        printf("  subgroup sizes  :");
        for (size_t s : sgs) {
            printf(" %zu", s);
        }
        printf("\n");
    } else {
        printf("  subgroup sizes  : (not reported)\n");
        sgs = { 8, 16 };
    }

    const std::string ext = dev_str(dev, CL_DEVICE_EXTENSIONS);
    printf("\n=== extensions that matter ===\n");
    const char * want[] = {
        "cl_intel_subgroup_matrix_multiply_accumulate",
        "cl_intel_subgroup_split_matrix_multiply_accumulate",
        "cl_intel_subgroups",
        "cl_intel_subgroups_short",
        "cl_intel_subgroups_char",
        "cl_intel_required_subgroup_size",
        "cl_khr_fp16",
        "cl_intel_unified_shared_memory",
    };
    bool ext_mma = false;
    for (const char * w : want) {
        bool have = ext.find(w) != std::string::npos;
        printf("  %-52s %s\n", w, have ? "yes" : "NO");
        if (!strcmp(w, "cl_intel_subgroup_matrix_multiply_accumulate")) {
            ext_mma = have;
        }
    }

    cl_int     err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        printf("PROBE: clCreateContext failed: %d\n", err);
        return 1;
    }

    // The part that is evidence rather than testimony.
    printf("\n=== does IGC compile the intrinsic? ===\n");
    printf("  %-8s %-6s %-10s %s\n", "dtype", "M", "subgroup", "result");
    bool        any_i8 = false;
    std::string log, first_fail;
    for (size_t sg : sgs) {
        if (sg != 8 && sg != 16) {
            // Not a gap in this probe. cl_intel_subgroup_matrix_multiply_accumulate
            // is specified for sub-group sizes 8 and 16 only; DPAS is not issued
            // from a SIMD32 dispatch, so a device reporting 32 reports it for
            // ordinary kernels. Printed rather than skipped in silence, because an
            // untested row and an unsupported one look identical in a table.
            printf("  %-8s %-6s %-10zu not applicable (MMA is defined for SIMD8/16)\n", "-", "-", sg);
            continue;
        }
        for (int m : { 1, 2, 4, 8 }) {
            bool ok = try_build(ctx, dev, i8_src(m, (int) sg), "-cl-std=CL2.0", log);
            printf("  %-8s %-6d %-10zu %s\n", "i8", m, sg, ok ? "BUILD OK" : "build failed");
            if (ok) {
                any_i8 = true;
            } else if (first_fail.empty()) {
                first_fail = log;
            }
        }
        for (int m : { 1, 8 }) {
            bool ok = try_build(ctx, dev, bf16_src(m, (int) sg), "-cl-std=CL2.0", log);
            printf("  %-8s %-6d %-10zu %s\n", "bf16", m, sg, ok ? "BUILD OK" : "build failed");
        }
    }

    if (!any_i8 && !first_fail.empty()) {
        printf("\n--- first build log ---\n%s\n", first_fail.c_str());
    }

    printf("\n=== verdict ===\n");
    if (any_i8) {
        printf("XMX REACHABLE FROM OPENCL C. The custom op can issue DPAS directly;\n");
        printf("  write the tiling against the (M, subgroup) pairs marked BUILD OK above.\n");
        if (!ext_mma) {
            printf("  Note: the intrinsic builds but the extension is NOT advertised. It\n");
            printf("  works, but it is undocumented-for-this-driver -- pin the driver\n");
            printf("  version in whatever ships.\n");
        }
    } else {
        printf("XMX NOT REACHABLE FROM OPENCL C on this driver.\n");
        printf("  An OpenVINO CustomLayer would run on the vector ALUs, roughly an\n");
        printf("  eighth of the 251-258 TOPS measured through SYCL joint_matrix, which\n");
        printf("  is slower than the cldnn gemm_kernel it would replace. The custom-op\n");
        printf("  route is wrong as designed: either the kernel goes in as a Level Zero\n");
        printf("  or SYCL op outside OpenVINO's graph, or the frontend emits a shape\n");
        printf("  the plugin's own int8 path accepts.\n");
    }

    clReleaseContext(ctx);
    return any_i8 ? 0 : 1;
}
