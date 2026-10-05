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
// ============================================================================
// ggml_ov_meansq.cl -- mean of squares along the last axis, accumulated in f32.
//
// This is RMSNorm's reduction and nothing else: it replaces the Multiply(x, x)
// and ReduceMean(axis=-1) pair that openvino/op/rms_norm.cpp emits, and leaves
// Add(eps) / Sqrt / Divide / Multiply(x) downstream exactly as they are. eps
// therefore never enters this kernel, which is what lets the whole binding be
// two static files instead of a runtime-generated descriptor per eps value.
//
// WHY IT EXISTS, in one paragraph. OpenVINO's own rms_gpu_bfyx_opt is the right
// algorithm and the wrong accumulator. Its reduction is
//
//     ACCUMULATOR_TYPE rms = ACCUMULATOR_VAL_ZERO;
//     rms += native_powr(tmp, 2);
//
// and ACCUMULATOR_TYPE is a JIT constant the plugin derives from the tensor
// dtype. At f16 that sums 2560 (LM attn) or 6912 (FFN) squares into a type whose
// ceiling is 65504, so it does not drift, it overflows -- which is why job
// 372854 saw the action chunk collapse rather than degrade. No graph-level
// attribute reaches a JIT constant: an explicit Convert (mode 2),
// mark_as_precision_sensitive (mode 3, and mode 6 at 3.2x the runtime), a MatMul
// with an XMX accumulator (mode 4) and three scaling schemes (modes 7, 8) all
// failed, five of them by graph surgery and the sixth because jit:gemm:any__f16
// at N = 1 returns zeros on this plugin (job 372887).
//
// So stop arguing with the JIT and own the accumulator. Structure below is
// rms_gpu_bfyx_opt's -- one work-group per row, sub-group block reads,
// sub_group_reduce_add, then an SLM tree across sub-groups -- with three
// deliberate differences:
//
//   1. THE ACCUMULATOR IS float, UNCONDITIONALLY. The entire point.
//   2. t * t RATHER THAN native_powr(t, 2). native_powr is a low-precision
//      builtin; for an exponent of 2 it is strictly worse than a multiply and
//      no faster. fma() then keeps the product unrounded into the sum.
//   3. NO PRIVATE data[] CACHE. The plugin's kernel keeps the whole row in
//      registers (ACCUMULATOR_TYPE data[STACK_SIZE]) because it also does the
//      normalise pass. We only reduce, so we spend no registers on it and the
//      occupancy is not capped by row width.
//
// VARIANTS. The GPU plugin's CustomLayer built-in defines are not documented for
// this build and guessing them costs a whole job, so all three candidate
// contracts are compiled here and tests/probe_ov_custom_rms.cpp reports which
// bind. They compute the same thing; only where they learn K and the element
// types differs. MS_VARIANT selects one, and each CustomLayer entry in
// ggml_ov_meansq.xml defines it.
//
//   1  INPUT0_DIMS + INPUT0_TYPE/OUTPUT0_TYPE  -- the documented contract
//   2  INPUT0_DIMS, element types hardcoded    -- if only the dims survive
//   3  MS_K_CONST from a <Define>, all static  -- the control; cannot fail for
//      a reason we would not already know about, and would force the descriptor
//      to be generated per distinct K.
//
// Every variant is bandwidth-bound by construction: it reads the row once and
// writes one float. At 2560 halves per row that is the floor for this operation.

// EXTENSIONS MUST BE ASKED FOR. Job 372897 bound all three variants -- the
// plugin matched the op type name, parsed the descriptor and accepted the
// WorkSizes formula -- and then failed identically at clBuildProgram with -11,
// V3 included, which is the variant that uses no built-in define at all. The
// plugin's own kernels do not carry these pragmas because cldnn prepends a JIT
// preamble that enables them; a CustomLayer source is compiled without it and
// starts from bare OpenCL C.
//
//   cl_khr_fp16             half, and the (float) conversions off it
//   cl_intel_subgroups      get_sub_group_id / get_num_sub_groups /
//                           get_sub_group_local_id / sub_group_reduce_add
//   cl_intel_subgroups_short  intel_sub_group_block_read_us8 and ushort8
//
// Job 372836 confirmed all three are advertised by this driver.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define MS_SG   16   // sub-group width. Job 372836 read 16 and 32 off this B70;
                     // 16 is the one the block-read widths below are stated for.
#define MS_VEC  8    // halves per lane per block read -> MS_SG*MS_VEC = 128
                     // contiguous halves per sub-group per message.

// LWS arrives as a <Define> from the descriptor, which also writes the same
// number into <WorkSizes local=...>. Both come from one variable in
// scripts/patch_ggml_openvino.py, because a kernel whose __local array is sized
// for a different work-group than the one dispatched is a silent wrong answer.
#ifndef LWS
#    define LWS 256
#endif

#define MS_NUM_SG (LWS / MS_SG)

// MS_EPS_T is ggml_ov_rrms's second input, a single element. It is read through
// a (float) cast, so the only thing that has to be right is its width.
//
// On variant 1 it names INPUT1_TYPE, which the plugin injects only when the
// descriptor declares a second input tensor. That is why MS_ENTRY == 1 must not
// reference it: a one-input descriptor compiled against entry 2 dies at
// "unknown type name 'INPUT1_TYPE'", which job 372913 also produced.
#if MS_VARIANT == 1
#    define MS_IN_T    INPUT0_TYPE
#    define MS_EPS_T   INPUT1_TYPE
#    define MS_OUT_T   OUTPUT0_TYPE
#    define MS_K       ((uint) INPUT0_DIMS[3])
#elif MS_VARIANT == 2
#    define MS_IN_T    half
#    define MS_EPS_T   float
#    define MS_OUT_T   float
#    define MS_K       ((uint) INPUT0_DIMS[3])
#elif MS_VARIANT == 3
#    define MS_IN_T    half
#    define MS_EPS_T   float
#    define MS_OUT_T   float
#    define MS_K       ((uint) MS_K_CONST)
#else
#    error "ggml_ov_meansq.cl: MS_VARIANT must be 1, 2 or 3"
#endif

// EXACTLY ONE __kernel MAY BE DEFINED HERE, and that is a plugin contract, not a
// style rule. cldnn builds one program per CustomLayer and then asserts
// `kernels.size() == batch.kernels_counter` (kernels_cache.cpp:314); a second
// entry point in the same source makes every descriptor pointing at this file
// fail to bind, with an error that names neither the file nor the extra kernel.
// Job 372913 is that failure: the source was valid OpenCL C, all three variants
// compiled standalone, and all four descriptors died at ProgramBuilder.
//
// So the entry point is chosen at compile time. 2 is what ships; the descriptor
// does not have to say so.
#ifndef MS_ENTRY
#    define MS_ENTRY 2
#endif

// The block-read path is stated for 16-bit input: intel_sub_group_block_read_us8
// moves 8 ushorts per lane. A wider element would need a different message, so
// that case has to fall back to the scalar loop for the whole row.
//
// Which path applies is decided in C below, as `sizeof(MS_IN_T) == 2`, NOT in
// the preprocessor -- sizeof is not available to #if, which is what job 372898
// died on in all three variants. Deciding it in C rather than in the descriptor
// is what makes this kernel correct for whatever element type the plugin ends up
// handing it: sizeof is a compile-time constant, IGC folds the dead branch away,
// and there is no define anyone can set inconsistently with the data.

// The row reduction, shared by both entry points below. Returns the mean of
// squares IN FLOAT, and returns it meaningfully only on lid == 0 -- the caller
// is the one that knows what to store and in what type.
//
// Splitting it out is what lets ggml_ov_rrms exist without a second copy of the
// arithmetic. The two kernels must not drift: they are the same reduction, and
// only the epilogue differs.
inline float ggml_ov_meansq_row(const __global MS_IN_T * input, uint row, uint K, __local float * slm) {
    const uint lid = get_local_id(0);

    const __global MS_IN_T * in = input + (size_t) row * (size_t) K;

    const uint sg      = get_sub_group_id();
    const uint num_sg  = get_num_sub_groups();
    const uint chunk   = MS_SG * MS_VEC;

    float acc = 0.0f;
    uint  done = 0;

    if (sizeof(MS_IN_T) == 2) {
        // Whole 128-half chunks, one sub-group block read each. c is uniform
        // within a sub-group (c = sg + n*num_sg), which is what the block read
        // requires. The cast is a reinterpret of the buffer, so this branch also
        // compiles when MS_IN_T is wider -- it is simply never taken there.
        const uint nchunk = K / chunk;
        for (uint c = sg; c < nchunk; c += num_sg) {
            ushort8 raw = intel_sub_group_block_read_us8((const __global ushort *) (in + (size_t) c * chunk));
            __attribute__((opencl_unroll_hint(MS_VEC)))
            for (int j = 0; j < MS_VEC; ++j) {
                float t = (float) as_half(raw[j]);
                acc = fma(t, t, acc);
            }
        }
        done = nchunk * chunk;
    }

    // Ragged tail, spread across the whole work-group rather than the sub-group
    // that happened to finish last. 2560 and 6912 leave none of it; 1152 and the
    // ViT shapes can.
    for (uint i = done + lid; i < K; i += LWS) {
        float t = (float) in[i];
        acc = fma(t, t, acc);
    }

    acc = sub_group_reduce_add(acc);

    if (get_sub_group_local_id() == 0) {
        slm[sg] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // MS_NUM_SG is 16 at LWS=256. A serial sum by one lane beats a barrier-per-
    // level tree at that width, and -- the reason it is written this way -- it
    // sums in a fixed order that does not depend on how many sub-groups the
    // driver actually launched, so the result is reproducible run to run.
    float s = 0.0f;
    if (lid == 0) {
        __attribute__((opencl_unroll_hint(MS_NUM_SG)))
        for (uint i = 0; i < MS_NUM_SG; ++i) {
            s += slm[i];
        }
    }
    return s / (float) K;
}

// ENTRY POINT 1: the mean of squares itself.
//
// KEPT FOR THE PROBE, NOT SHIPPED. Job 372912 measured why: on the real model
// mean(x^2) OVERFLOWS f16. The chunk spread went to exactly 0.00000, which is
// the signature of mean = inf reaching 1/sqrt(inf) = 0, and it is the direct
// confirmation of the ceiling job 372883 first suspected -- mean(x^2) passes
// 65504 once rms(x) > 256, and BitVLA's norms do. Storing the mean therefore
// forces an f32 output, and job 372911 measured what THAT costs: 726 opaque f32
// custom ops in an f16 graph disrupt the plugin's precision plan badly enough to
// add 8.77 SECONDS of full-size reorders, against the 47 ms being chased.
//
// Both measurements point the same way, which is why ggml_ov_rrms below is what
// the descriptor actually binds.
#if MS_ENTRY == 1
__attribute__((intel_reqd_sub_group_size(MS_SG)))
__attribute__((reqd_work_group_size(LWS, 1, 1)))
__kernel void ggml_ov_meansq(const __global MS_IN_T * input, __global MS_OUT_T * output) {
    // One work-group per row. <WorkSizes global="B*F*Y*X*LWS,1,1"> makes the
    // group count the *output* element count, and the output is {..., rows, 1},
    // so this is the row index however the plugin assigned the axes.
    const uint row = get_group_id(0);
    __local float slm[MS_NUM_SG];
    const float   mean = ggml_ov_meansq_row(input, row, MS_K, slm);
    if (get_local_id(0) == 0) {
        output[row] = (MS_OUT_T) mean;
    }
}
#endif  // MS_ENTRY == 1

// ENTRY POINT 2: the reciprocal RMS. THIS IS THE ONE THAT SHIPS.
//
// Storing 1/sqrt(mean + eps) rather than the mean is not a convenience, it is
// the only quantity in this chain that is safe in f16 at BOTH ends. It runs
// ~1e-3 to 1e-1 for this model: nowhere near the 65504 that sinks the mean, and
// nowhere near f16's 6e-8 subnormal floor either. Being f16-representable is
// what lets the output follow the input type, and following the input type is
// what keeps the op from becoming an f32 island the plugin has to reorder
// around.
//
// It is also what the plugin's own rms_gpu_bfyx_opt does, for the same reason:
//
//     slm_buf[0] = native_powr(sqrt(rms + TO_ACCUMULATOR_TYPE(EPSILON)), -1);
//
// The differences from that line are deliberate. The sum reaching it here was
// accumulated in float rather than in ACCUMULATOR_TYPE, which is the entire
// point of this file. And the reciprocal is a real divide rather than
// native_powr(x, -1) or rsqrt(): both of those are low-precision builtins, this
// value multiplies EVERY element of the row, and the brief is that the port keep
// the precision. One divide per row is not a cost worth trading it for.
//
// eps arrives as a one-element INPUT rather than a <Define> so that a single
// descriptor still covers every norm in the graph. As a define it would have to
// be baked per eps value, which is a descriptor generated at runtime and a
// temporary directory to put it in.
#if MS_ENTRY == 2
__attribute__((intel_reqd_sub_group_size(MS_SG)))
__attribute__((reqd_work_group_size(LWS, 1, 1)))
__kernel void ggml_ov_rrms(const __global MS_IN_T * input, const __global MS_EPS_T * eps,
                           __global MS_OUT_T * output) {
    const uint row = get_group_id(0);
    __local float slm[MS_NUM_SG];
    const float   mean = ggml_ov_meansq_row(input, row, MS_K, slm);
    if (get_local_id(0) == 0) {
        output[row] = (MS_OUT_T) (1.0f / sqrt(mean + (float) eps[0]));
    }
}
#endif  // MS_ENTRY == 2
