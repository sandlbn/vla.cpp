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
// probe_ov_custom_rms.cpp -- can an OpenVINO GPU CustomLayer replace RMSNorm's
// reduction, and does it beat the reference reduce it replaces?
//
// WHAT THIS IS FOR. Stage E spent five modes and eight jobs discovering that the
// f16 accumulator in rms_gpu_bfyx_opt is a JIT constant no graph-level change
// can reach, and that the one graph shape with a hardware-wide accumulator --
// a MatMul, mode 4 -- is 11.7x faster on the norm and returns exact zeros,
// because jit:gemm:any__f16 at N = 1 does not compute on this plugin (372887).
// The remaining route is to own the accumulator: ship the reduction as our own
// OpenCL C kernel. See ci/kernels/ggml_ov_meansq.cl.
//
// Wiring that into 726 nodes of a real model is a day. The GPU plugin's
// CustomLayer contract, though, has several parts this build does not document,
// and each wrong guess would cost a full job:
//
//   - does binding by op type name work for an op the frontend invents?
//   - which built-in defines are injected, and under what prefix? (the binary
//     has NUM_INPUTS, GLOBAL_WORKSIZE, LOCAL_WORKSIZE and the _DIMS / _PITCHES /
//     _LOWER_PADDING / _UPPER_PADDING suffixes, but not the prefix spelling)
//   - does the WorkSizes formula parser accept B*F*Y*X*256?
//   - may a custom op's output element type differ from its input's?
//   - does an unknown op survive the plugin's transformation pipeline at all?
//
// So this answers all of them in one short job, against a reference computed in
// double on the host, and times the result against the ReduceMean it would
// replace. It builds the smallest model that can carry the question --
// Parameter -> custom op -> Result -- rather than a model.
//
// Each variant gets its own ov::Core and its own single-CustomLayer descriptor,
// written here rather than shipped: a descriptor is three lines of XML, the
// variants differ only in defines, and if variant 3 is the one that binds then
// mode 9 has to generate a descriptor per distinct K anyway and this is the code
// that will do it.
//
//   probe_ov_custom_rms <kernel.cl> [rows] [K]
//
// Exit status is 0 if at least one variant is both bound and correct, 1
// otherwise, so the sbatch can gate on it.
//
// STAGE 3 WAS ADDED AFTER THE FIRST TWO ANSWERS CAME BACK. The variants above
// all bind the MEAN of the squares, and the mean is the wrong thing to hand the
// graph -- jobs 372911 and 372912 measured both of its failure modes on the real
// model (f32 output: 8.77 s of reorders; f16 output: overflow, spread 0.00000).
// What ships is the reciprocal RMS instead, which is what the plugin's own
// rms_gpu_bfyx_opt stores and for the same reason. That form asks two questions
// these variants cannot:
//
//   - will a CustomLayer take TWO inputs, the second a folded Constant? eps has
//     to arrive as a tensor rather than a <Define> or the descriptor has to be
//     regenerated per eps value.
//   - is an f16 output of 1/sqrt(mean + eps) accurate enough, given that this is
//     the type the mean could not survive?
//
// Both are cheap to ask here and expensive to get wrong on a model job.

#include <openvino/openvino.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/op.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_mean.hpp>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

// Shared by the descriptor below and by the standalone OpenCL build stage, so
// the diagnostic compiles the kernel the same way the plugin will. -cl-std=CL2.0
// is not decoration: the sub-group builtins the reduction is written against
// need it, and job 372897 failed at clBuildProgram without it.
const char * const kCompilerOptions = "-cl-mad-enable -cl-std=CL2.0";

// The op the CustomLayer binds to. The plugin matches CustomLayer name= against
// get_type_info().name and tries CreateCustomOp before its own factory, so the
// name here and the name in the descriptor are the whole contract. Everything
// else this class does is for the *other* consumers of the graph -- shape and
// type inference have to be right or the transformation pipeline rejects the
// model long before the plugin sees it.
//
// One class per variant name, without three copies of the body: the type name is
// a template parameter and get_type_info() returns a per-instantiation static.
template <int V> class MeanSq : public ov::op::Op {
  public:
    MeanSq() = default;

    explicit MeanSq(const ov::Output<ov::Node> & arg) : ov::op::Op({arg}) { constructor_validate_and_infer_types(); }

    static const char * type_name() {
        static const char * names[] = {"GgmlMeanSqV1", "GgmlMeanSqV2", "GgmlMeanSqV3"};
        return names[V - 1];
    }

    const ov::Node::type_info_t & get_type_info() const override {
        static ov::Node::type_info_t info{type_name(), "vla"};
        return info;
    }

    void validate_and_infer_types() override {
        auto ps = get_input_partial_shape(0);
        // {..., rows, K} -> {..., rows, 1}. Same rank on purpose: it is what
        // ReduceMean(keep_dims=true) produced, so the Add/Sqrt/Divide chain
        // downstream of the real translator needs no reshape.
        if (ps.rank().is_static()) {
            ps[ps.rank().get_length() - 1] = 1;
        }
        // f32 unconditionally -- this is the entire point of the exercise, and
        // saying it here is what keeps ConvertPrecision from compressing the
        // chain that consumes it.
        set_output_type(0, ov::element::f32, ps);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector & new_args) const override {
        return std::make_shared<MeanSq<V>>(new_args.at(0));
    }

    bool visit_attributes(ov::AttributeVisitor &) override { return true; }
};

// A single-CustomLayer descriptor. global is the output element count times LWS
// -- one work-group per row -- written as B*F*Y*X so it does not depend on which
// axis the plugin decided carries rows. local must equal both the kernel's
// reqd_work_group_size and the LWS define, because the __local array is sized
// from the define while the dispatch comes from here, and a mismatch between
// them is a wrong answer rather than a build error.
std::string descriptor_for(const char * type_name, int variant, const std::string & cl_name, int lws, int k_const) {
    std::string x;
    x += "<CustomLayer name=\"";
    x += type_name;
    x += "\" type=\"SimpleGPU\" version=\"1\">\n";
    x += "  <Kernel entry=\"ggml_ov_meansq\">\n";
    x += "    <Source filename=\"" + cl_name + "\"/>\n";
    x += "    <Define name=\"MS_VARIANT\" type=\"int\" default=\"" + std::to_string(variant) + "\"/>\n";
    x += "    <Define name=\"LWS\" type=\"int\" default=\"" + std::to_string(lws) + "\"/>\n";
    // Entry 1. The source holds two __kernel bodies and cldnn asserts a program
    // contains exactly one (kernels_cache.cpp:314), so the one that is NOT being
    // bound has to be preprocessed away. Job 372913 is what happens otherwise:
    // every descriptor here failed at ProgramBuilder while the source compiled
    // fine standalone.
    x += "    <Define name=\"MS_ENTRY\" type=\"int\" default=\"1\"/>\n";
    if (variant == 3) {
        x += "    <Define name=\"MS_K_CONST\" type=\"int\" default=\"" + std::to_string(k_const) + "\"/>\n";
    }
    x += "  </Kernel>\n";
    x += "  <Buffers>\n";
    x += "    <Tensor arg-index=\"0\" type=\"input\"  port-index=\"0\" format=\"BFYX\"/>\n";
    x += "    <Tensor arg-index=\"1\" type=\"output\" port-index=\"0\" format=\"BFYX\"/>\n";
    x += "  </Buffers>\n";
    x += "  <CompilerOptions options=\"" + std::string(kCompilerOptions) + "\"/>\n";
    x += "  <WorkSizes global=\"B*F*Y*X*" + std::to_string(lws) + ",1,1\" local=\"" + std::to_string(lws) +
         ",1,1\"/>\n";
    x += "</CustomLayer>\n";
    return x;
}

// The shipped op: (x, eps) -> 1/sqrt(mean(x^2) + eps). Two inputs, and the
// output follows the INPUT type rather than being pinned f32 as MeanSq is --
// which is the whole difference, and the thing job 372911 says has to be true.
class RRms : public ov::op::Op {
  public:
    RRms() = default;

    RRms(const ov::Output<ov::Node> & arg, const ov::Output<ov::Node> & eps) : ov::op::Op({arg, eps}) {
        constructor_validate_and_infer_types();
    }

    // Deliberately the same type name the shipped descriptor and rms_norm.cpp
    // use. If this probe binds, ci/kernels/ggml_ov_meansq.xml is proven as
    // written rather than as approximated.
    static const char * type_name() { return "GgmlRmsRecip"; }

    const ov::Node::type_info_t & get_type_info() const override {
        static ov::Node::type_info_t info{type_name(), "vla"};
        return info;
    }

    void validate_and_infer_types() override {
        auto ps = get_input_partial_shape(0);
        if (ps.rank().is_static()) {
            ps[ps.rank().get_length() - 1] = 1;
        }
        set_output_type(0, get_input_element_type(0), ps);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector & new_args) const override {
        return std::make_shared<RRms>(new_args.at(0), new_args.at(1));
    }

    bool visit_attributes(ov::AttributeVisitor &) override { return true; }
};

// Three buffers rather than two, and eps is the middle one. arg-index is the
// kernel's parameter position and port-index is the op's input position; they
// happen to coincide here, but they are different numbering spaces and the
// output's arg-index 2 is what makes that visible.
std::string descriptor_rrms(const std::string & cl_name, int lws) {
    std::string x;
    x += "<CustomLayer name=\"";
    x += RRms::type_name();
    x += "\" type=\"SimpleGPU\" version=\"1\">\n";
    x += "  <Kernel entry=\"ggml_ov_rrms\">\n";
    x += "    <Source filename=\"" + cl_name + "\"/>\n";
    x += "    <Define name=\"MS_VARIANT\" type=\"int\" default=\"1\"/>\n";
    x += "    <Define name=\"MS_ENTRY\" type=\"int\" default=\"2\"/>\n";
    x += "    <Define name=\"LWS\" type=\"int\" default=\"" + std::to_string(lws) + "\"/>\n";
    x += "  </Kernel>\n";
    x += "  <Buffers>\n";
    x += "    <Tensor arg-index=\"0\" type=\"input\"  port-index=\"0\" format=\"BFYX\"/>\n";
    x += "    <Tensor arg-index=\"1\" type=\"input\"  port-index=\"1\" format=\"BFYX\"/>\n";
    x += "    <Tensor arg-index=\"2\" type=\"output\" port-index=\"0\" format=\"BFYX\"/>\n";
    x += "  </Buffers>\n";
    x += "  <CompilerOptions options=\"" + std::string(kCompilerOptions) + "\"/>\n";
    x += "  <WorkSizes global=\"B*F*Y*X*" + std::to_string(lws) + ",1,1\" local=\"" + std::to_string(lws) +
         ",1,1\"/>\n";
    x += "</CustomLayer>\n";
    return x;
}

// f16 <-> f32 without pulling in a dependency. ov::float16 is in the runtime we
// already link, so use it rather than hand-rolling the bit twiddling.
using f16 = ov::float16;

struct Result {
    bool        bound = false;
    bool        correct = false;
    double      max_rel = 0.0;
    double      ms = 0.0;
    std::string note;
};

// The reference, in double, on the host. Not float: the question on the table is
// precisely whether a narrow accumulator loses the sum, and a float reference
// would be the same experiment as the one under test.
std::vector<double> reference(const std::vector<f16> & x, size_t rows, size_t K) {
    std::vector<double> out(rows);
    for (size_t r = 0; r < rows; ++r) {
        double s = 0.0;
        for (size_t c = 0; c < K; ++c) {
            double t = (double) (float) x[r * K + c];
            s += t * t;
        }
        out[r] = s / (double) K;
    }
    return out;
}

template <int V>
Result run_variant(const std::string & cl_path, const std::string & cl_dir, const std::string & cl_name,
                   const std::vector<f16> & x, const std::vector<double> & ref, size_t rows, size_t K, int lws) {
    Result res;
    const char * tname = MeanSq<V>::type_name();

    // The descriptor has to sit beside the .cl: <Source filename=> is resolved
    // relative to the descriptor's own directory.
    const std::string xml_path = cl_dir + "/probe_" + tname + ".xml";
    {
        std::ofstream f(xml_path);
        if (!f) {
            res.note = "cannot write " + xml_path;
            return res;
        }
        f << descriptor_for(tname, V, cl_name, lws, (int) K);
    }
    (void) cl_path;

    try {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, rows, K});
        auto node  = std::make_shared<MeanSq<V>>(param);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{node->output(0)}, ov::ParameterVector{param},
                                                 std::string("probe_") + tname);

        // A fresh Core per variant. CONFIG_FILE is plugin-global state and a
        // descriptor naming an op type the model does not contain is harmless,
        // but a *failed* variant must not be able to explain a later one's
        // result -- that is how job 372831 got a wrong answer from a real
        // measurement.
        ov::Core core;
        core.set_property("GPU", {{"CONFIG_FILE", xml_path}});

        auto compiled = core.compile_model(model, "GPU");
        res.bound     = true;

        auto req = compiled.create_infer_request();
        ov::Tensor in(ov::element::f16, ov::Shape{1, 1, rows, K});
        std::memcpy(in.data(), x.data(), x.size() * sizeof(f16));
        req.set_input_tensor(in);

        req.infer();  // warm up: first infer carries the OpenCL build
        const int iters = 200;
        auto      t0    = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            req.infer();
        }
        auto t1 = std::chrono::steady_clock::now();
        res.ms  = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        auto   out = req.get_output_tensor();
        auto * p   = out.data<float>();
        if (out.get_size() != rows) {
            res.note = "output has " + std::to_string(out.get_size()) + " elements, expected " + std::to_string(rows);
            return res;
        }
        double worst = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            double denom = std::fabs(ref[r]) > 1e-30 ? std::fabs(ref[r]) : 1e-30;
            worst        = std::max(worst, std::fabs((double) p[r] - ref[r]) / denom);
        }
        res.max_rel = worst;
        // The input is f16, so each term already carries up to 2^-11 relative
        // error before the kernel sees it; summing K of them in f32 adds
        // essentially nothing. 1e-3 is loose enough not to trip on the input's
        // own quantisation and far tighter than any of the failures this is
        // looking for -- an f16 accumulator overflows to inf, and the mode 4
        // path returned exact zeros.
        res.correct = worst < 1e-3;
        if (!res.correct) {
            res.note = "numerics: max rel " + std::to_string(worst);
        }
    } catch (const std::exception & e) {
        std::string what = e.what();
        if (what.size() > 400) {
            what = what.substr(0, 400) + " ...";
        }
        res.note = what;
    }
    return res;
}

// Stage 3: the form that ships. Same kernel file, different entry point, two
// inputs, f16 in and f16 out.
//
// `dynamic` is the question job 372916 forced. Everything this probe had
// measured used a STATIC input shape, and so does the ggml frontend's naive
// path (get_graph_input_shape returns the tensor's own shape when m_naive).
// The LM graphs are over the 20-node naive threshold and take the dynamic path
// instead, where every activation is PartialShape{1,1,-1,-1}. A CustomLayer's
// K arrives as the JIT constant INPUT0_DIMS[3] and its dispatch as the
// WorkSizes formula, and BOTH are resolved by the plugin, not by us -- so
// whether they track a shape that is only known at infer time is a property of
// cldnn that nothing here had ever asked about.
Result run_rrms(const std::string & cl_dir, const std::string & cl_name, const std::vector<f16> & x,
                const std::vector<double> & mean_ref, size_t rows, size_t K, int lws, float eps,
                bool dynamic) {
    Result res;

    const std::string xml_path = cl_dir + (dynamic ? "/probe_GgmlRmsRecip_dyn.xml" : "/probe_GgmlRmsRecip.xml");
    {
        std::ofstream f(xml_path);
        if (!f) {
            res.note = "cannot write " + xml_path;
            return res;
        }
        f << descriptor_rrms(cl_name, lws);
    }

    try {
        const ov::PartialShape pshape =
            dynamic ? ov::PartialShape{1, 1, ov::Dimension::dynamic(), ov::Dimension::dynamic()}
                    : ov::PartialShape{1, 1, (int64_t) rows, (int64_t) K};
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, pshape);
        // eps as a Constant of the INPUT's type, exactly as rms_norm.cpp builds
        // it. If the plugin refuses to give a folded constant its own buffer,
        // this is where it says so, and it says so before a model job pays for
        // the answer.
        auto eps_c = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {eps});
        auto node  = std::make_shared<RRms>(param, eps_c);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{node->output(0)}, ov::ParameterVector{param},
                                                 dynamic ? "probe_rrms_dyn" : "probe_rrms");

        ov::Core core;
        core.set_property("GPU", {{"CONFIG_FILE", xml_path}});

        auto compiled = core.compile_model(model, "GPU");
        res.bound     = true;

        auto req = compiled.create_infer_request();
        ov::Tensor in(ov::element::f16, ov::Shape{1, 1, rows, K});
        std::memcpy(in.data(), x.data(), x.size() * sizeof(f16));
        req.set_input_tensor(in);

        req.infer();
        const int iters = 200;
        auto      t0    = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            req.infer();
        }
        auto t1 = std::chrono::steady_clock::now();
        res.ms  = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        auto out = req.get_output_tensor();
        if (out.get_size() != rows) {
            res.note = "output has " + std::to_string(out.get_size()) + " elements, expected " + std::to_string(rows);
            return res;
        }
        if (out.get_element_type() != ov::element::f16) {
            // Not a failure in itself, but it means the plugin overrode the
            // output type we declared, and the whole point of the rrms form is
            // that it does not have to.
            res.note = "output type is " + out.get_element_type().get_type_name() + ", not f16; ";
        }
        double worst = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            const double want = 1.0 / std::sqrt(mean_ref[r] + (double) eps);
            const double got  = out.get_element_type() == ov::element::f16 ? (double) (float) out.data<f16>()[r]
                                                                          : (double) out.data<float>()[r];
            worst = std::max(worst, std::fabs(got - want) / std::fabs(want));
        }
        res.max_rel = worst;
        // Looser than the mean's 1e-3 for one reason: the result is STORED in
        // f16, whose own spacing is 2^-11 = 4.9e-4. 2e-3 is four f16 ulps, which
        // catches every failure worth catching here -- overflow to zero, a
        // narrow accumulator, a wrong eps -- without tripping on the storage.
        res.correct = worst < 2e-3;
        if (!res.correct) {
            res.note += "numerics: max rel " + std::to_string(worst);
        }
    } catch (const std::exception & e) {
        std::string what = e.what();
        if (what.size() > 400) {
            what = what.substr(0, 400) + " ...";
        }
        res.note = what;
    }
    return res;
}

// What it has to beat: Multiply(x, x) -> ReduceMean(-1, keepdims), which is what
// openvino/op/rms_norm.cpp emits today and what the profile attributes 47.12 ms
// to across 726 nodes. Same Core settings, same shapes, same timing loop.
Result run_reference_graph(const std::vector<f16> & x, const std::vector<double> & ref, size_t rows, size_t K) {
    Result res;
    try {
        auto param  = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, rows, K});
        auto square = std::make_shared<ov::op::v1::Multiply>(param, param);
        auto axes   = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean   = std::make_shared<ov::op::v1::ReduceMean>(square, axes, true);
        auto model  = std::make_shared<ov::Model>(ov::OutputVector{mean->output(0)}, ov::ParameterVector{param},
                                                  "probe_reduce_mean");

        ov::Core core;
        auto     compiled = core.compile_model(model, "GPU");
        res.bound         = true;

        auto req = compiled.create_infer_request();
        ov::Tensor in(ov::element::f16, ov::Shape{1, 1, rows, K});
        std::memcpy(in.data(), x.data(), x.size() * sizeof(f16));
        req.set_input_tensor(in);

        req.infer();
        const int iters = 200;
        auto      t0    = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            req.infer();
        }
        auto t1 = std::chrono::steady_clock::now();
        res.ms  = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        auto   out   = req.get_output_tensor();
        double worst = 0.0;
        // The control graph's output type is whatever the plugin chose, which is
        // the thing under investigation -- read it as f16 or f32 accordingly
        // rather than assuming.
        const bool is_f16 = out.get_element_type() == ov::element::f16;
        for (size_t r = 0; r < rows; ++r) {
            double v     = is_f16 ? (double) (float) out.data<f16>()[r] : (double) out.data<float>()[r];
            double denom = std::fabs(ref[r]) > 1e-30 ? std::fabs(ref[r]) : 1e-30;
            worst        = std::max(worst, std::fabs(v - ref[r]) / denom);
        }
        res.max_rel = worst;
        res.correct = worst < 1e-3;
        res.note    = std::string("output element type ") + out.get_element_type().get_type_name();
    } catch (const std::exception & e) {
        res.note = e.what();
    }
    return res;
}

// ---------------------------------------------------------------------------
// The diagnostic stage. The GPU plugin reports a failed kernel build as
// "clBuildProgram, error code: -11" and throws the log away, and this build has
// no OV_GPU_* debug variables compiled in to get it back (Stage D established
// that: zero matching strings in the 44 MB plugin). Without the log, every fix
// is a guess costing a whole job -- which is the shape of mistake this probe
// exists to avoid in the first place.
//
// So compile the same source, with the same options, through plain OpenCL
// first, and print clGetProgramBuildInfo. The defines a CustomLayer would have
// injected are prepended as text rather than passed as -D, because INPUT0_DIMS
// is an array initialiser and quoting one through a build-options string is its
// own source of error. This stage answers "is my kernel valid OpenCL C"; the
// OpenVINO stage after it answers "does the plugin's contract match what I
// assumed". Keeping them apart is the whole point.
std::string ocl_preamble(int variant, int entry, size_t rows, size_t K, int lws) {
    std::string p;
    p += "#define LWS " + std::to_string(lws) + "\n";
    p += "#define MS_VARIANT " + std::to_string(variant) + "\n";
    p += "#define MS_ENTRY " + std::to_string(entry) + "\n";
    if (variant == 1) {
        // INPUT1_TYPE only exists when the descriptor declares a second input,
        // so entry 2 is the only build that may name it. Entry 1 compiled
        // against a two-input source is exactly how job 372913's variant 1 died.
        p += "#define INPUT0_TYPE half\n#define OUTPUT0_TYPE float\n";
        if (entry == 2) {
            p += "#define INPUT1_TYPE half\n";
        }
    }
    if (variant == 1 || variant == 2) {
        // No __constant: a compound literal at function scope may not carry an
        // address space, which is what job 372898's first two variants died on.
        // That was this guess being wrong, not the plugin -- the real spelling
        // is still unknown, which is why mode 9 is planned on variant 3.
        p += "#define INPUT0_DIMS ((int[]){1,1," + std::to_string(rows) + "," + std::to_string(K) + "})\n";
    }
    if (variant == 3) {
        p += "#define MS_K_CONST " + std::to_string(K) + "\n";
    }
    return p;
}

// Returns true if the build succeeded. Prints the log either way -- a warning
// on a kernel that does run is worth seeing too.
bool ocl_try_build(const std::string & src, int variant, int entry, size_t rows, size_t K, int lws) {
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) {
        printf("    (no OpenCL platform; skipping the build log)\n");
        return false;
    }
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);

    cl_device_id dev = nullptr;
    for (auto p : plats) {
        char vendor[256] = {0};
        clGetPlatformInfo(p, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr);
        if (!strstr(vendor, "Intel")) {
            continue;
        }
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr) == CL_SUCCESS && dev) {
            break;
        }
        dev = nullptr;
    }
    if (!dev) {
        printf("    (no Intel GPU via OpenCL; skipping the build log)\n");
        return false;
    }

    cl_int  err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (!ctx) {
        printf("    (clCreateContext failed: %d)\n", (int) err);
        return false;
    }

    const std::string full = ocl_preamble(variant, entry, rows, K, lws) + src;
    const char *      cstr = full.c_str();
    size_t            len  = full.size();
    cl_program        prog = clCreateProgramWithSource(ctx, 1, &cstr, &len, &err);
    bool              ok   = false;
    if (prog) {
        err = clBuildProgram(prog, 1, &dev, kCompilerOptions, nullptr, nullptr);
        ok  = (err == CL_SUCCESS);
        size_t loglen = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &loglen);
        if (loglen > 1) {
            std::vector<char> log(loglen + 1, 0);
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, loglen, log.data(), nullptr);
            printf("    --- build log ---\n");
            // Indent so the log cannot be mistaken for the probe's own output.
            std::string line;
            for (char c : std::string(log.data())) {
                if (c == '\n') {
                    printf("    | %s\n", line.c_str());
                    line.clear();
                } else {
                    line += c;
                }
            }
            if (!line.empty()) {
                printf("    | %s\n", line.c_str());
            }
        }
        clReleaseProgram(prog);
    }
    clReleaseContext(ctx);
    printf("    OpenCL build: %s\n", ok ? "OK" : "FAILED");
    return ok;
}

// Stage 4: the op WITH ITS CONSUMER, which is the one thing no stage above has
// ever built. Every earlier arm ends at a Result, so the custom op's output is
// read by nothing; rms_norm.cpp feeds it to Multiply(input_node, reciprocal),
// and job 372916's whole 8.9 SECONDS is 30 distinct
// "Multiply_*_compressed_to_f16" reorders at ~23 ms each. The op itself was
// innocent in that profile -- "rmsrecip_*_decompressed_to_f32" totals 0.6 ms
// over 720 nodes -- so the question is whether an f16 tensor produced by an
// UNKNOWN op stops its consumer from being compressed to f16.
//
// The answer is in the runtime model, not in the numbers: ov::Model after
// compile_model carries each node's chosen precision and kernel, and that is
// what this stage prints.
Result run_rrms_chain(const std::string & cl_dir, const std::string & cl_name, const std::vector<f16> & x,
                      const std::vector<double> & mean_ref, size_t rows, size_t K, int lws, float eps) {
    Result res;

    const std::string xml_path = cl_dir + "/probe_GgmlRmsRecip_chain.xml";
    {
        std::ofstream f(xml_path);
        if (!f) {
            res.note = "cannot write " + xml_path;
            return res;
        }
        f << descriptor_rrms(cl_name, lws);
    }

    try {
        // Dynamic, because that is what the LM graphs are and 372918 showed it
        // costs the op nothing.
        auto param = std::make_shared<ov::op::v0::Parameter>(
            ov::element::f16, ov::PartialShape{1, 1, ov::Dimension::dynamic(), ov::Dimension::dynamic()});
        auto eps_c = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {eps});
        auto node  = std::make_shared<RRms>(param, eps_c);
        // THE LINE UNDER TEST. Same op, same operand order as rms_norm.cpp.
        auto mul   = std::make_shared<ov::op::v1::Multiply>(param, node);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{mul->output(0)}, ov::ParameterVector{param},
                                                 "probe_rrms_chain");

        ov::Core core;
        core.set_property("GPU", {{"CONFIG_FILE", xml_path}});

        auto compiled = core.compile_model(model, "GPU");
        res.bound     = true;

        // What the plugin actually chose. One line per node: type, the kernel it
        // picked, and the precision it runs at. A Convert here that is not in
        // the source model is the plugin inserting a reorder, and that is the
        // 23 ms node from job 372916 reproduced at 1/726th the scale.
        printf("    -- runtime model --\n");
        int converts = 0;
        for (const auto & op : compiled.get_runtime_model()->get_ordered_ops()) {
            const auto & rt = op->get_rt_info();
            auto         g  = [&](const char * k) -> std::string {
                auto it = rt.find(k);
                return it == rt.end() ? std::string("?") : it->second.as<std::string>();
            };
            const std::string type = g("layerType");
            if (type == "Convert" || type == "Reorder") {
                ++converts;
            }
            printf("      %-22s %-30s %s\n", type.c_str(), g("execType").c_str(), g("runtimePrecision").c_str());
        }

        auto req = compiled.create_infer_request();
        ov::Tensor in(ov::element::f16, ov::Shape{1, 1, rows, K});
        std::memcpy(in.data(), x.data(), x.size() * sizeof(f16));
        req.set_input_tensor(in);
        req.infer();

        auto   out = req.get_output_tensor();
        double worst = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            const double recip = 1.0 / std::sqrt(mean_ref[r] + (double) eps);
            for (size_t c = 0; c < K; ++c) {
                const double want = (double) (float) x[r * K + c] * recip;
                const double got  = out.get_element_type() == ov::element::f16 ? (double) (float) out.data<f16>()[r * K + c]
                                                                              : (double) out.data<float>()[r * K + c];
                const double den  = std::fabs(want) > 1e-3 ? std::fabs(want) : 1e-3;
                worst = std::max(worst, std::fabs(got - want) / den);
            }
        }
        res.max_rel = worst;
        res.correct = worst < 4e-3;
        res.note    = std::to_string(converts) + " Convert/Reorder in the runtime model";
    } catch (const std::exception & e) {
        std::string what = e.what();
        if (what.size() > 400) {
            what = what.substr(0, 400) + " ...";
        }
        res.note = what;
    }
    return res;
}

void report(const char * label, const Result & r) {
    printf("  %-14s  bound %-3s  correct %-3s  max_rel %-12.3e  %8.4f ms   %s\n", label, r.bound ? "yes" : "NO",
           r.correct ? "yes" : "NO", r.max_rel, r.ms, r.note.c_str());
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path/to/ggml_ov_meansq.cl> [rows] [K]\n", argv[0]);
        return 2;
    }
    const std::string cl_path = argv[1];
    const size_t      rows    = argc > 2 ? (size_t) atoll(argv[2]) : 256;
    const size_t      K       = argc > 3 ? (size_t) atoll(argv[3]) : 2560;
    const int         lws     = 256;

    const size_t slash   = cl_path.find_last_of('/');
    const std::string dir  = slash == std::string::npos ? std::string(".") : cl_path.substr(0, slash);
    const std::string name = slash == std::string::npos ? cl_path : cl_path.substr(slash + 1);

    printf("probe_ov_custom_rms: rows=%zu K=%zu LWS=%d  kernel=%s\n", rows, K, lws, cl_path.c_str());
    std::string cl_src;
    {
        std::ifstream f(cl_path);
        if (!f) {
            fprintf(stderr, "cannot read %s\n", cl_path.c_str());
            return 2;
        }
        cl_src.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }

    // Stage 0: is it valid OpenCL C at all? Run this before touching OpenVINO,
    // because the plugin's -11 tells you nothing and this tells you everything.
    printf("\n-- stage 0: does the kernel compile, and what does the driver say? --\n");
    bool any_ocl = false;
    for (int v = 1; v <= 3; ++v) {
        for (int e = 1; e <= 2; ++e) {
            printf("  variant %d, entry %d (%s):\n", v, e, e == 1 ? "ggml_ov_meansq" : "ggml_ov_rrms");
            any_ocl |= ocl_try_build(cl_src, v, e, rows, K, lws);
        }
    }
    if (!any_ocl) {
        printf("\n  No variant compiles as plain OpenCL C. Fix the kernel before reading\n"
               "  anything below -- the OpenVINO stage can only repeat this failure.\n");
    }

    // Activations at the scale that actually breaks the f16 accumulator. BitVLA
    // reaches |x| ~ 10 in the LM residual stream, and 2560 squares of that sum
    // to ~2.6e5 against f16's 65504 ceiling -- so this input is not a stress
    // test, it is the operating point, and a variant that returns inf here is
    // reproducing the bug this kernel exists to fix.
    std::vector<f16> x(rows * K);
    uint32_t         s = 12345u;
    for (size_t i = 0; i < x.size(); ++i) {
        s        = s * 1664525u + 1013904223u;
        float u  = (float) ((s >> 8) & 0xFFFF) / 65535.0f;  // [0,1)
        x[i]     = f16((u * 2.0f - 1.0f) * 10.0f);
    }
    const auto ref = reference(x, rows, K);

    printf("\n-- the control: what mode 9 has to beat --\n");
    auto ctrl = run_reference_graph(x, ref, rows, K);
    report("Multiply+RM", ctrl);

    printf("\n-- custom kernel variants --\n");
    auto v1 = run_variant<1>(cl_path, dir, name, x, ref, rows, K, lws);
    report("V1 dims+type", v1);
    auto v2 = run_variant<2>(cl_path, dir, name, x, ref, rows, K, lws);
    report("V2 dims only", v2);
    auto v3 = run_variant<3>(cl_path, dir, name, x, ref, rows, K, lws);
    report("V3 all static", v3);

    // The consumer. Everything above ends at a Result, and job 372916 says the
    // damage is not in the op but in what reads it.
    printf("\n-- the op inside the chain rms_norm.cpp actually builds --\n");
    auto rrc = run_rrms_chain(dir, name, x, ref, rows, K, lws, 1e-5f);
    report("rrms->Multiply", rrc);

    // The form that ships. eps is BitVLA's own: 1e-5, as ggml stores it.
    printf("\n-- the form that ships: two inputs, f16 out --\n");
    auto rr = run_rrms(dir, name, x, ref, rows, K, lws, 1e-5f, false);
    report("rrms static", rr);

    // The arm that matters for the LM. Same everything, dynamic input shape.
    auto rrd = run_rrms(dir, name, x, ref, rows, K, lws, 1e-5f, true);
    report("rrms dynamic", rrd);

    const Result * cands[] = {&v1, &v2, &v3};
    const char *   cnames[] = {"V1", "V2", "V3"};
    const Result * win = nullptr;
    const char *   wn  = nullptr;
    for (int i = 0; i < 3; ++i) {
        if (cands[i]->bound && cands[i]->correct && (!win || cands[i]->ms < win->ms)) {
            win = cands[i];
            wn  = cnames[i];
        }
    }

    printf("\n-- verdict --\n");
    if (!win) {
        printf("  NO VARIANT BOUND AND CORRECT. The CustomLayer route is not\n"
               "  available as written; read the notes above before changing the kernel.\n");
        return 1;
    }
    printf("  WINNER: %s at %.4f ms, max rel %.3e\n", wn, win->ms, win->max_rel);
    if (ctrl.bound) {
        printf("  against the control's %.4f ms -> %.2fx\n", ctrl.ms, ctrl.ms / win->ms);
        if (!ctrl.correct) {
            printf("  NOTE: the control is itself wrong here (max rel %.3e). That is the\n"
                   "  f16 accumulator, reproduced at this shape -- which is the bug.\n",
                   ctrl.max_rel);
        }
    }

    // Do NOT read the ms columns above as speed. Job 372908 fitted two row
    // counts and got 16.3 GB/s marginal, which is PCIe: set_input_tensor on a
    // host-allocated ov::Tensor copies the whole input across the bus every
    // infer, and at K = 2560 that copy dwarfs the reduction. This probe answers
    // binding and numerics only. The norm bucket in bmg_ov_profile_ops.sbatch
    // is the speed measurement.
    if (!rr.bound || !rr.correct) {
        printf("\n  THE SHIPPED FORM DID NOT PASS. Mode 9 as wired in rms_norm.cpp is\n"
               "  ggml_ov_rrms, not ggml_ov_meansq, so a green V1 above does not cover\n"
               "  it. Fix this before spending a model job.\n");
        return 1;
    }
    if (!rrd.bound || !rrd.correct) {
        printf("\n  STATIC PASSES AND DYNAMIC DOES NOT. Mode 9 is then usable only on\n"
               "  the frontend's naive path, whose shapes come from the ggml tensors;\n"
               "  every graph above GGML_OPENVINO_NAIVE_GRAPH_SIZE is dynamic, which\n"
               "  is the LM and most of the 726 norms. Do not spend a model job on it\n"
               "  in this state -- job 372916 is what that costs.\n");
        return 1;
    }
    return 0;
}
