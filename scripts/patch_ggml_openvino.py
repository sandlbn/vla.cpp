#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fourteen fixes to the fetched ggml OpenVINO backend.

ggml-openvino is written against llama.cpp's graphs: one decoder-only
transformer, one position input, an F16 KV cache. vla.cpp drives it with vision
towers and action experts instead, which is legal ggml but nothing the backend
has seen. Eight of the hunks below narrow an llama.cpp-shaped assumption back to
the ggml contract, two fill gaps in the op table, and two are about the path
those graphs take. Together they are what lets SmolVLA and pi0.5 run end to end
on the CPU, GPU and NPU plugins. Numbers 4 and 5 are the ones worth upstreaming.

A hunk that applies is not a hunk that runs. Whenever VLA_LLAMA_TAG moves, check
that each patched function still has a live caller, not just that its anchor
still matches: b10729 moved the position-input naming out of the GgmlOvDecoder
member that fix 4 patches and into a free function, and the fix went silently
dead while applying cleanly.

See docs/backend/ov.md for the measured results and for what is still blocked.

  1. ggml-openvino-extra.cpp - pick the *Intel* OpenCL platform.
     `GGML_OPENVINO_DEVICE=GPU` builds an OpenVINO remote context on an OpenCL
     queue and takes the first platform the ICD loader reports. With more than
     one runtime installed (an NVIDIA card next to the Intel iGPU, POCL,
     Rusticl) that is whichever `/etc/OpenCL/vendors/*.icd` sorted first, and
     the GPU plugin only accepts an Intel context -- it aborts at startup with
     "Incompatible OpenCL runtime: program is not in expected ELF format".
     Selecting by `CL_PLATFORM_VENDOR` leaves single-runtime boxes unchanged.

  2. ggml-decoder.cpp - narrow RESHAPE op_case 3.
     Case 3 is the KV-cache flatten, `[512,1024,1,1] -> [1,524288,1,1]`, and it
     emits a shape with `-1` in dim 2 and 1 in dim 3. Its guard only tests
     `src->ne[0]*ne[1]*ne[2] == node->ne[1]`, which also matches the kernel
     reshape inside `ggml_conv_2d` (`[16,16,3,768] -> [768,768]`) and mangles
     it. The real case always has `node->ne[0] == 1`; requiring that sends the
     conv kernel to case 6, the plain reshape.

  3. openvino/op/flash_attn_ext.cpp - convert K/V to F16 with Q.
     The translator converts Q, the mask and the scale to F16 because
     llama.cpp's KV cache already is. vla.cpp keeps K/V in F32, and OpenVINO's
     SDPA rejects mixed input types ("Mixed input types are not supported").
     Converting K/V too matches the precision the translator already chose.

  4. ggml-decoder.{h,cpp} - stop distinct position inputs aliasing each other.
     Every tensor feeding a ROPE's second input is renamed to one graph
     parameter called "inp_pos", because an llama.cpp graph has exactly one.
     add_rope_sin_cos() then builds a single shared sin/cos table from it. Every
     vla.cpp arch has several position tensors -- SmolVLA passes a prefill, a
     full and a rebased one -- so they alias each other and every ROPE takes the
     table built from whichever won, which fails shape inference:
     "Multiply (Split[1]:f32[1,113,5,32], Multiply[0]:f32[1,50,1,32])
      Argument shapes are inconsistent."
     When the graph has more than one, keep each tensor's own name. Nothing is
     then called "inp_pos", so translate_rope() falls back to building sin/cos
     per op from its own position input. Single-position graphs are untouched.
     This one is what carries an arch through to a full prediction.

     Patch the free function get_tensor_graph_input_ov_name() in
     ggml-decoder.cpp, not just the GgmlOvDecoder member. b10729 relocated the
     naming into that free function and left the member behind with no callers;
     patching only the member applies cleanly and does nothing at all. Both are
     patched here so the fix survives whichever one upstream keeps.

  5. utils.{h,cpp} - cache what the naive path compiles.
     The dynamic and static paths keep a `graph_key`-indexed cache of the
     decoder and the compiled infer request; the naive path has none, so it
     rebuilt the decoder, re-converted the model and called compile_model() on
     every single ggml_backend_graph_compute. That is the dominant cost once a
     real graph goes through it: SmolVLA on the CPU plugin drops from 22.7 s to
     1.4 s per prediction with the cache in place. A hit rebinds the cached
     decoder to the new graph through the existing update_io(), which is how the
     dynamic path already handles freshly built tensors.
     The key is `naive_key`, not `graph_key`: the latter is n_nodes plus the
     first and last node name, which two graphs of the same size can share, and
     a compiled model is bound to the shapes it was built for. Reusing one
     across a shape change returns another graph's answer with no error, so the
     key mixes in every node's op and shape. The map is bounded; see the comment
     on the flush.

  6. openvino/op_table.cpp - get both GELU flavours right.
     ggml has two: GGML_UNARY_OP_GELU is the tanh approximation, GGML_UNARY_OP_
     GELU_ERF is the exact one. The table mapped GELU onto ov's Gelu, which
     defaults to erf, and had no entry for GELU_ERF at all. So the tanh op was
     computed as erf, and an arch using the erf op could not run at all - and
     with no per-op CPU fallback in the core, "could not run" means the whole
     prediction. Small per node, but a vision tower has dozens and the error
     compounds: setting the mode explicitly moved VLA-JEPA from 5.5e-3 to 1.1e-4
     and GR00T N1.5 from 2.7e-2 to 6.0e-4, turning both from "runs but drifts"
     into supported. The highest-yield single fix here.
     RELU, NEG and SQR were missing here too; upstream added all three in b10729.

  7. openvino/utils.cpp - give a folded weight its full rank before slicing.
     A ggml tensor that is 2-D folds in as a rank-2 ov constant, which is what a
     GEMM operand wants, but process_view_input_new() indexes a viewed tensor at
     its full ggml rank, so the slice axis lands outside it:
     "Slice (Constant aex.blk.0.attn_in.weight[0]:bf16[2688,896], ...)
      Axis 2 out of the tensor rank range [-2, 1]."
     Evo-1 hits it by viewing Q, K and V out of one fused attn_in weight.
     Left-pad the input with 1s, which is the shape ggml gave it anyway.

  8. openvino/op/concat.cpp - align input ranks.
     The same rank-2 constants reach CONCAT, which unlike the broadcasting
     elementwise ops needs both inputs at the graph's rank or the axis falls
     outside them. Evo-1 concatenates a CLS weight onto 4-D patch embeddings;
     SmolVLA concatenates a precomputed time tile onto a 4-D activation. Padding
     here is what saves each arch from working around it, e.g. by moving
     SmolVLA's time tiles into a buffer of their own.

  9. openvino/utils.cpp - bound the interleaved-mrope sector cycle by sections.
     ggml's IMROPE cycles t/h/w by sector % 3, but only while the sector is
     inside 3 * sections[k]; past that it falls through to the fourth position
     stream (ggml_rope_cache_init in ggml/src/ggml-cpu/ops.cpp). The translator
     cycled unconditionally, so with sections {24,20,20,0} and n_dims 128
     sectors 61 and 62 took h and w instead of e. NOTE: this matches the ggml
     reference but had no measurable effect on any arch tested here, because the
     fourth stream happens to carry the same positions as the first. Kept
     because it removes a real divergence from the reference, not because a
     measurement demanded it.

 10. utils.cpp - make the naive-path graph-size threshold settable.
     Graphs under 20 nodes bypass the LLM decoder and translate literally, with
     static shapes and no KV-cache inference. That literal path is the one that
     suits a vision tower, but a vision tower is ~450 nodes. The constant
     becomes `GGML_OPENVINO_NAIVE_GRAPH_SIZE`; src/backend.h defaults it high
     for vla.cpp and an explicit setting still wins. Parsed with strtol, because
     atoi turns junk into 0 and that would send every graph down the LLM builder
     with nothing said.

 11. ggml-decoder.cpp - require a ROPE before taking PERMUTE op_case 2.
     op_case 2 rewrites the tensor as [n_seq, -1, n_heads, head_size] and only
     then transposes, which is correct for llama.cpp's rope'd query and nothing
     else. The classifier reached it for ANY permute whose source is a view of a
     non-leaf, so GR00T N1.7's DiT cross-attention V - ggml_permute(view, 1,2,0,3)
     over a fused KV projection - was reshaped into a shape unrelated to it and
     came out with its elements rearranged: V was 139% wrong while K, which uses
     permute(0,2,1,3) and happened to survive the same rewrite, was 0.04%. That
     turned into a 27% error in the final actions. Walking the view/reshape/cont
     chain and requiring a ROPE at the end sends every other permute to op_case 1,
     the plain transpose. This is what makes GR00T N1.7 correct (27% -> 0.005%).

 12. ggml-decoder.cpp + openvino/op/add.cpp - do not stack two elementwise adds
     on a GEMM. The GPU plugin folds elementwise ops into the preceding GEMM as
     post-ops. Given ADD(ADD(residual, GEMM), graph_input) it folds both, and the
     second operand is silently lost: the result equals the inner add, as though
     the outer one never ran. Nothing is logged. An llama.cpp graph never builds
     that chain - one residual add per sub-block - but a VLA does, wherever a
     tower's features are added on top of an FFN residual. It cost VLA-JEPA,
     GR00T N1.7 and pi0 their GPU support, while every other arch was unaffected.
     Addition is associative, so re-hang the outer add on the inner one's
     non-GEMM operand: the GEMM keeps exactly one post-op and the arithmetic is
     unchanged. op_case 2/3 records which of the inner add's operands is the
     GEMM, because ggml's operand order is not fixed.
     Found by bisecting VLA-JEPA with GGML_OPENVINO_DEBUG_NODE: its ViT and DiT
     graphs matched the CPU plugin to 0.2%, the VLM prefill was already wrong at
     the end of layer 0, and within that layer the divergence was one node - the
     FFN residual add under the deepstack add. The same fusion path already has a
     known defect with broadcast DIV; see ggml_backend_openvino_supports_op.

 13. ggml-openvino-extra.cpp - expose the GPU plugin's inference precision.
     The GPU plugin computes in f16 unless told otherwise, which is most of why
     it is fast, and for nearly every graph that is the right trade. It is not
     the right trade for a graph carrying a long serial chain - a flow-matching
     denoise loop unrolled inside one graph - because the error compounds across
     every step with nothing to reset it, and a saturating output channel can
     then cross its threshold in the wrong place. pi0 does exactly that: on the
     GPU its continuous action dims land 4e-2 from an F32 reference and its
     bistable gripper flips one step early, which is a 1.7 max|delta| on a metric
     and a late grasp on a robot. GGML_OPENVINO_GPU_PRECISION=f32 puts it at
     6.5e-5. Exposed rather than forced: f32 costs about 3x on this plugin, so
     src/backend.h defaults it for pi0 alone and an explicit setting still wins.

 14. openvino/op_table.cpp + ggml-openvino.cpp - translate BitVLA's activation
     quantiser, and only that one. Ten of vla.cpp's eleven architectures build
     graphs out of stock ggml ops; BitVLA builds one op of its own, the BitNet
     per-row int8 fake-quant, and that single gap is the whole reason it pinned
     itself to the CPU backend and never reached this one (docs/backend/ov.md).
     It is six OpenVINO nodes and no new numerics -- see translate_bitvla_act_quant
     for the two steps in it that look redundant and are not.
     The awkward half is that it arrives as GGML_OP_MAP_CUSTOM1, which is not an
     op so much as a slot: the kernel is a host function pointer, invisible to any
     backend, so a second model using the slot for something else would be
     mistranslated into a fake-quant rather than rejected. Hence the name test, in
     the translator and again in ggml_backend_openvino_device_supports_op -- the
     latter because that function derives its supported set from the op table's
     keys, so the table entry alone would have this backend claim every custom op
     in existence and then throw on it instead of leaving it to the CPU.

Idempotent - re-running on a patched tree is a no-op, so a reconfigure that
re-populates the FetchContent source dir is safe either way.

Usage: scripts/patch_ggml_openvino.py [<llama-src-dir>]
"""

import pathlib
import re
import sys

HELPER = """// vla.cpp: select the Intel OpenCL platform. With several OpenCL runtimes
// installed the first platform is not always Intel's, and the OpenVINO GPU
// plugin only accepts an Intel context. Cached: the ICD list cannot change
// under a running process.
static cl_platform_id ggml_openvino_get_intel_platform() {
    static cl_platform_id platform = nullptr;
    static bool searched = false;
    if (searched) {
        return platform;
    }
    searched = true;

    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0) {
        return nullptr;
    }
    std::vector<cl_platform_id> platforms(n_platforms);
    if (clGetPlatformIDs(n_platforms, platforms.data(), nullptr) != CL_SUCCESS) {
        return nullptr;
    }

    for (cl_platform_id p : platforms) {
        char vendor[256] = "";
        if (clGetPlatformInfo(p, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr) != CL_SUCCESS) {
            continue;
        }
        if (strstr(vendor, "Intel") != nullptr) {
            platform = p;
            break;
        }
    }
    return platform;
}

"""

USM_LOOKUP = """        cl_platform_id platform;
        if (clGetPlatformIDs(1, &platform, nullptr) == CL_SUCCESS) {
            fn = (%s_fn) clGetExtensionFunctionAddressForPlatform(platform, "%s");
"""

USM_LOOKUP_NEW = """        cl_platform_id platform = ggml_openvino_get_intel_platform();
        if (platform != nullptr) {
            fn = (%s_fn) clGetExtensionFunctionAddressForPlatform(platform, "%s");
"""

NAIVE_COMPUTE_OLD = """enum ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config) {
    if (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        return GGML_STATUS_SUCCESS;
    }

    bool naive = true;
    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
    auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
    auto model = ov::frontend::ggml::FrontEnd::convert(input_model, naive);
    if (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    auto remote_context = ggml_openvino_get_remote_context();
    if (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
    } else {
        core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    }
    if (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), config).create_infer_request());
    } else {
        infer_request =
            std::make_shared<ov::InferRequest>(core.compile_model(model, device, config).create_infer_request());
    }

    auto ov_params = model->get_parameters();"""

NAIVE_COMPUTE_NEW = """enum ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config,
                               std::shared_ptr<ov_runtime_context> r_ctx) {
    if (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        return GGML_STATUS_SUCCESS;
    }

    // vla.cpp: reuse the decoder, the converted model and the compiled infer
    // request across calls on the same graph, the way the dynamic and static
    // paths already do. Conversion plus compile_model dominates a naive call, so
    // without this every graph_compute pays it again.
    static const bool cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    const naive_key key(cgraph);

    std::shared_ptr<naive_runtime_ctx> entry;
    bool cache_hit = false;
    if (cache_enabled && r_ctx != nullptr) {
        std::lock_guard<std::mutex> lock(r_ctx->ctx_mutex);
        auto it = r_ctx->naive_cache.find(key);
        if (it != r_ctx->naive_cache.end()) {
            entry = it->second;
            cache_hit = true;
        } else {
            // Each entry holds a compiled model, so this cannot grow forever.
            // Flush rather than evict: a caller sees a handful of shapes, and an
            // LRU is only worth its bookkeeping once that stops being true.
            if (r_ctx->naive_cache.size() >= 32) {
                r_ctx->naive_cache.clear();
            }
            entry = std::make_shared<naive_runtime_ctx>();
            r_ctx->naive_cache[key] = entry;
        }
    } else {
        entry = std::make_shared<naive_runtime_ctx>();
    }

    // One graph at a time: an ov::InferRequest is not re-entrant, and a hit
    // rebinds the decoder to this cgraph.
    std::lock_guard<std::mutex> entry_lock(entry->mutex);

    bool naive = true;
    std::shared_ptr<GgmlOvDecoder> decoder;
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::InferRequest> infer_request;

    if (cache_hit && entry->infer_request != nullptr) {
        decoder = entry->decoder;
        model = entry->model;
        infer_request = entry->infer_request;
        // Same shapes, new tensors: point the decoder at this call's graph.
        decoder->update_io(cgraph);
    } else {
        auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
        decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
        auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
        model = ov::frontend::ggml::FrontEnd::convert(input_model, naive);
        if (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
            ov::serialize(model, "IR_naive.xml");
        }

        auto remote_context = ggml_openvino_get_remote_context();
        if (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
            // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
            core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
        } else {
            core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
        }
        if (remote_context.has_value()) {
            infer_request = std::make_shared<ov::InferRequest>(
                core.compile_model(model, remote_context.value(), config).create_infer_request());
        } else {
            infer_request =
                std::make_shared<ov::InferRequest>(core.compile_model(model, device, config).create_infer_request());
        }

        entry->decoder = decoder;
        entry->model = model;
        entry->infer_request = infer_request;
    }

    auto ov_params = model->get_parameters();"""

# file -> [(anchor, replacement), ...]. Every anchor must match exactly once.
EDITS = {
    "ggml/src/ggml-openvino/ggml-openvino-extra.cpp": [
        ("#include <optional>\n", "#include <optional>\n#include <vector>\n"),
        ("void ggml_openvino_device_config::init() {", HELPER + "void ggml_openvino_device_config::init() {"),
        (
            """        cl_int err;
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to get OpenCL platform: %d\\n", err);
            return;
        }
""",
            """        cl_int err;
        cl_platform_id platform = ggml_openvino_get_intel_platform();
        if (platform == nullptr) {
            GGML_LOG_ERROR("Failed to find an Intel OpenCL platform\\n");
            return;
        }
""",
        ),
        (USM_LOOKUP % (("clEnqueueMemFillINTEL",) * 2), USM_LOOKUP_NEW % (("clEnqueueMemFillINTEL",) * 2)),
        (USM_LOOKUP % (("clEnqueueMemcpyINTEL",) * 2), USM_LOOKUP_NEW % (("clEnqueueMemcpyINTEL",) * 2)),
        (
            """        "GGML_OPENVINO_LOG_UNSUPPORTED_OPS",
    };""",
            """        "GGML_OPENVINO_LOG_UNSUPPORTED_OPS",
        // vla.cpp: f16 (default) or f32 for the GPU plugin's inference precision.
        "GGML_OPENVINO_GPU_PRECISION",
    };""",
            '// vla.cpp: f16 (default) or f32 for the GPU plugin\'s inference precision.',
        ),
        # Anchored on the hunk above rather than folded into it. Edits are applied
        # in order to one buffer, so this matches whether the tree is pristine (the
        # hunk above just produced this text) or was already patched by an earlier
        # checkout (that text is still there and the hunk above was skipped). Folding
        # a new entry into an applied hunk's replacement instead strands its anchor,
        # which cannot be recovered from without restoring the llama.cpp mirror.
        (
            """        // vla.cpp: f16 (default) or f32 for the GPU plugin's inference precision.
        "GGML_OPENVINO_GPU_PRECISION",
    };""",
            """        // vla.cpp: f16 (default) or f32 for the GPU plugin's inference precision.
        "GGML_OPENVINO_GPU_PRECISION",
        // vla.cpp: group size for the GPU plugin's dynamic activation
        // quantisation. Empty (default) leaves the plugin's own behaviour alone;
        // "max" means per-token, which is the granularity BitVLA's act_quant
        // already uses. See the compile_config hunk below.
        "GGML_OPENVINO_DYNAMIC_QUANT_GROUP",
    };""",
            '// vla.cpp: group size for the GPU plugin\'s dynamic activation',
        ),
        # Same anchoring rule as above: a separate tuple hanging off the previous
        # hunk's output, never an edit to the previous hunk's replacement text.
        (
            """        // vla.cpp: group size for the GPU plugin's dynamic activation
        // quantisation. Empty (default) leaves the plugin's own behaviour alone;
        // "max" means per-token, which is the granularity BitVLA's act_quant
        // already uses. See the compile_config hunk below.
        "GGML_OPENVINO_DYNAMIC_QUANT_GROUP",
    };""",
            """        // vla.cpp: group size for the GPU plugin's dynamic activation
        // quantisation. Empty (default) leaves the plugin's own behaviour alone;
        // "max" means per-token, which is the granularity BitVLA's act_quant
        // already uses. See the compile_config hunk below.
        "GGML_OPENVINO_DYNAMIC_QUANT_GROUP",
        // vla.cpp: number of infer calls to dump per-op profiling for. 0 = off.
        "GGML_OPENVINO_PROFILE_OPS",
    };""",
            '// vla.cpp: number of infer calls to dump per-op profiling for.',
        ),
        (
            """        // vla.cpp: number of infer calls to dump per-op profiling for. 0 = off.
        "GGML_OPENVINO_PROFILE_OPS",
    };""",
            """        // vla.cpp: number of infer calls to dump per-op profiling for. 0 = off.
        "GGML_OPENVINO_PROFILE_OPS",
        // vla.cpp: emit RMSNorm's square as Power(x,2) so ov::pass::RMSFusion
        // matches it. See openvino/op/rms_norm.cpp.
        "GGML_OPENVINO_RMS_FUSION",
    };""",
            '// vla.cpp: emit RMSNorm\'s square as Power(x,2) so ov::pass::RMSFusion',
        ),
        (
            """        // vla.cpp: emit RMSNorm's square as Power(x,2) so ov::pass::RMSFusion
        // matches it. See openvino/op/rms_norm.cpp.
        "GGML_OPENVINO_RMS_FUSION",
    };""",
            """        // vla.cpp: emit RMSNorm's square as Power(x,2) so ov::pass::RMSFusion
        // matches it. See openvino/op/rms_norm.cpp.
        "GGML_OPENVINO_RMS_FUSION",
        // vla.cpp: squeeze leading unit axes off MUL_MAT operands so the GPU
        // plugin's DynamicQuantizeFullyConnected pass will accept them. See
        // openvino/op/mulmat.cpp.
        "GGML_OPENVINO_FC_RANK3",
    };""",
            '// vla.cpp: squeeze leading unit axes off MUL_MAT operands so the GPU',
        ),
        (
            """        // vla.cpp: squeeze leading unit axes off MUL_MAT operands so the GPU
        // plugin's DynamicQuantizeFullyConnected pass will accept them. See
        // openvino/op/mulmat.cpp.
        "GGML_OPENVINO_FC_RANK3",
    };""",
            """        // vla.cpp: squeeze leading unit axes off MUL_MAT operands so the GPU
        // plugin's DynamicQuantizeFullyConnected pass will accept them. See
        // openvino/op/mulmat.cpp.
        "GGML_OPENVINO_FC_RANK3",
        // vla.cpp: path to the GPU CustomLayer descriptor that supplies mode 9's
        // RMSNorm kernel. Required by GGML_OPENVINO_RMS_FUSION=9 and ignored
        // otherwise. See ci/kernels/ggml_ov_meansq.xml.
        "GGML_OPENVINO_CUSTOM_KERNELS",
    };""",
            '// vla.cpp: path to the GPU CustomLayer descriptor that supplies mode 9',
        ),
        (
            """        // vla.cpp: path to the GPU CustomLayer descriptor that supplies mode 9's
        // RMSNorm kernel. Required by GGML_OPENVINO_RMS_FUSION=9 and ignored
        // otherwise. See ci/kernels/ggml_ov_meansq.xml.
        "GGML_OPENVINO_CUSTOM_KERNELS",
    };""",
            """        // vla.cpp: path to the GPU CustomLayer descriptor that supplies mode 9's
        // RMSNorm kernel. Required by GGML_OPENVINO_RMS_FUSION=9 and ignored
        // otherwise. See ci/kernels/ggml_ov_meansq.xml.
        "GGML_OPENVINO_CUSTOM_KERNELS",
        // vla.cpp: translate BitVLA's activation quantiser to a pass-through and
        // let the GPU plugin's own dynamic quantiser do the work instead. ONLY
        // MEANINGFUL TOGETHER WITH GGML_OPENVINO_FC_RANK3 -- see
        // openvino/op_table.cpp.
        "GGML_OPENVINO_ACT_QUANT_ELIDE",
        // vla.cpp: serialise ov::CompiledModel::get_runtime_model() -- the graph
        // AFTER the plugin's transformations -- one numbered file per compiled
        // graph, into the CWD. See utils.cpp.
        "GGML_OPENVINO_DUMP_RUNTIME",
    };""",
            '"GGML_OPENVINO_DUMP_RUNTIME",',
        ),
        (
            """    } else if (cache_dir && strlen(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    }""",
            """    } else if (cache_dir && strlen(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    }

    // vla.cpp: the GPU plugin computes in f16 unless told otherwise, which is why
    // it is fast. A model whose graph carries a long serial chain -- a denoise
    // loop unrolled inside one graph -- compounds that across every step, and a
    // saturating output channel can then cross its threshold in the wrong place.
    // Exposed rather than forced: f32 costs roughly 3x on this plugin.
    if (device_name == "GPU") {
        const char * gpu_prec = ggml_openvino_getenv_str("GGML_OPENVINO_GPU_PRECISION", "f16");
        if (strcmp(gpu_prec, "f32") == 0) {
            compile_config.insert(ov::hint::inference_precision(ov::element::f32));
        } else if (strcmp(gpu_prec, "f16") != 0) {
            GGML_LOG_WARN("GGML OpenVINO Backend: GGML_OPENVINO_GPU_PRECISION=%s is not f16 or f32, ignoring\\n",
                          gpu_prec);
        }
    }""",
        ),
        # Anchored on the hunk above, for the reason given at the env-var table.
        (
            """        } else if (strcmp(gpu_prec, "f16") != 0) {
            GGML_LOG_WARN("GGML OpenVINO Backend: GGML_OPENVINO_GPU_PRECISION=%s is not f16 or f32, ignoring\\n",
                          gpu_prec);
        }
    }""",
            """        } else if (strcmp(gpu_prec, "f16") != 0) {
            GGML_LOG_WARN("GGML OpenVINO Backend: GGML_OPENVINO_GPU_PRECISION=%s is not f16 or f32, ignoring\\n",
                          gpu_prec);
        }
    }

    // vla.cpp: BitVLA is natively W1.58A8. bitvla_act_quant_op has already
    // rounded the activations onto an int8 lattice by the time the GEMM sees
    // them (src/models/bitvla.cpp, and translate_bitvla_act_quant here), but it
    // hands them on as floats, so the BitLinear GEMMs run f16 x f16 carrying
    // int8-valued data and the int8 XMX pipeline goes unused. This property is
    // how the GPU plugin is told to quantise activations at runtime and pick an
    // int8 kernel for a FullyConnected with compressed weights.
    //
    // "max" is per-token, which is exactly act_quant's own granularity: it emits
    // q * (amax/127) with q integral and amax the row absmax, so the row absmax
    // of that is again amax and re-quantising per-row recovers the same q. For
    // this model the coarsest setting is therefore also the most faithful one,
    // which is the opposite of the usual accuracy/speed dial.
    //
    // Off by default. It changes which kernel the plugin selects, and that is a
    // decision worth making explicitly rather than inheriting.
    if (device_name == "GPU") {
        const char * dq = ggml_openvino_getenv_str("GGML_OPENVINO_DYNAMIC_QUANT_GROUP", "");
        if (dq && strlen(dq) > 0) {
            // ~0 rather than UINT64_MAX or std::numeric_limits: this file
            // includes <cstdlib> and <cstring> but neither <cstdint> nor
            // <limits>, and a patch hunk that needs a new #include is a patch
            // hunk that breaks when the upstream include list moves.
            const uint64_t group = (strcmp(dq, "max") == 0) ? ~(uint64_t) 0 : (uint64_t) atoll(dq);
            compile_config.insert(ov::hint::dynamic_quantization_group_size(group));
            GGML_LOG_INFO("GGML OpenVINO Backend: dynamic quantization group size = %s\\n", dq);
        }
    }""",
        ),
        (
            """            compile_config.insert(ov::hint::dynamic_quantization_group_size(group));
            GGML_LOG_INFO("GGML OpenVINO Backend: dynamic quantization group size = %s\\n", dq);
        }
    }""",
            """            compile_config.insert(ov::hint::dynamic_quantization_group_size(group));
            GGML_LOG_INFO("GGML OpenVINO Backend: dynamic quantization group size = %s\\n", dq);
        }
    }

    // vla.cpp: without this the plugin keeps no per-node timings and
    // get_profiling_info() comes back empty. Gated on the same env var that asks
    // for the dump, because profiling makes the plugin instrument every
    // primitive -- a measurement that changes what it measures if left on.
    if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILE_OPS") > 0) {
        compile_config.insert(ov::enable_profiling(true));
    }""",
        ),
        (
            """    if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILE_OPS") > 0) {
        compile_config.insert(ov::enable_profiling(true));
    }""",
            """    if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILE_OPS") > 0) {
        compile_config.insert(ov::enable_profiling(true));
    }

    // vla.cpp: point the GPU plugin at our CustomLayer descriptor, which is how
    // GGML_OPENVINO_RMS_FUSION=9 gets its kernel onto the device. The plugin
    // binds a CustomLayer by op TYPE NAME, so this file is what turns the
    // GgmlRmsRecip node openvino/op/rms_norm.cpp emits into something executable.
    //
    // "CONFIG_FILE" is passed as a raw string on purpose. It is a GPU-plugin
    // config key with no ov::Property<> wrapper exposed in the public headers of
    // this build, so there is nothing typed to use instead.
    //
    // Deliberately NOT defaulted to a path inside the repo. The runtime has no
    // reliable way to locate the source tree, and a default that silently
    // resolves to nothing would make mode 9 fall back to the plugin's own f16
    // kernel -- the exact silent-no-op failure that cost job 372752 a run. Unset
    // means mode 9 throws at compile_model, which is the loud failure to want.
    if (device_name == "GPU") {
        const char * custom_kernels = ggml_openvino_getenv_str("GGML_OPENVINO_CUSTOM_KERNELS", "");
        if (custom_kernels && strlen(custom_kernels) > 0) {
            compile_config["CONFIG_FILE"] = std::string(custom_kernels);
            GGML_LOG_INFO("GGML OpenVINO Backend: GPU custom kernels from %s\\n", custom_kernels);
        }
    }""",
            'GPU custom kernels from %s',
        ),
        (
            """        cl_queue = clCreateCommandQueueWithProperties(cl_ctx, cl_device, nullptr, &err);""",
            """        // vla.cpp: the GPU path compiles against a remote context that shares this
        // queue, and a plugin cannot turn profiling on for a queue it did not
        // create -- ov::enable_profiling(true) then throws "could not reset
        // profiling" out of get_profiling_info(). CL_QUEUE_PROFILING_ENABLE is
        // settable only here, at creation, so the two have to be decided together.
        //
        // Gated on the same env var rather than always on, so that what gets
        // measured is what ships: same remote context, same shared queue, same
        // zero-copy tensors, with only the queue's profiling bit different. (A
        // profiled queue timestamps every enqueue, which is a real cost -- small
        // next to enable_profiling's per-primitive instrumentation, but not zero,
        // and not something to inflict on every run for a measurement nobody
        // asked for.)
        cl_queue_properties prof_qprops[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        const bool want_queue_profiling = ggml_openvino_getenv_int("GGML_OPENVINO_PROFILE_OPS") > 0;
        cl_queue = clCreateCommandQueueWithProperties(cl_ctx, cl_device,
                                                      want_queue_profiling ? prof_qprops : nullptr, &err);""",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/op/rms_norm.cpp": [
        (
            """#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
""",
            """#include "../../ggml-openvino-extra.h"
#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
""",
        ),
        (
            """    auto square = std::make_shared<ov::op::v1::Multiply>(input_node, input_node);

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);""",
            """    // vla.cpp: x*x and pow(x,2) compute the same thing and cost the plugin
    // wildly different amounts, because only one of them is a pattern it knows.
    //
    // ov::pass::RMSFusion collapses the whole Power/ReduceMean/Add/Sqrt/Divide/
    // Multiply decomposition below into a single RMS primitive with a dedicated
    // kernel (rms_gpu_bfyx_opt). Its pattern is written against Power(x, 2) --
    // the shape PyTorch and every ONNX exporter produce -- so the Multiply(x, x)
    // spelling here never matches, the decomposition survives into the compiled
    // graph, and the ReduceMean lands on reduce_ref: OpenVINO's *generic
    // reference* reduction, in f32, while the tensors around it are f16.
    //
    // Job 372841 measured what that costs on BitVLA: ReduceMean is 47.4 ms,
    // 20.25% of device time, 42-119 us per node over 726 nodes. The same run
    // contains the control -- the ViT's LayerNorm, which OpenVINO *does*
    // recognise, runs on mvn_gpu_bfyx_opt at 3 us per node while computing a mean
    // AND a variance over comparable data. Same reduction, ~15-40x apart, and the
    // only difference is whether the plugin could name it.
    //
    // Off by default and measured before it is trusted: if RMSFusion does not fire
    // for some other reason, this leaves a Power node where a Multiply was, and
    // Power is the slower and less exact of the two when it has to stand alone.
    // The OVPROF table says which happened -- an RMS node type means it fused.
    ov::Output<ov::Node> square;
    if (ggml_openvino_getenv_int("GGML_OPENVINO_RMS_FUSION")) {
        square = std::make_shared<ov::op::v1::Power>(
            input_node, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2.0f}));
    } else {
        square = std::make_shared<ov::op::v1::Multiply>(input_node, input_node);
    }

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);""",
            '// The OVPROF table says which happened -- an RMS node type means it fused.',
        ),
        # Mode 2, anchored into mode 1's replacement, which is why mode 1 above
        # carries an explicit marker: its replacement text no longer survives.
        (
            """#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>""",
            """#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/divide.hpp>""",
        ),
        (
            """    ov::Output<ov::Node> square;
    if (ggml_openvino_getenv_int("GGML_OPENVINO_RMS_FUSION")) {""",
            """    //
    // MODE 2 KEEPS THE REDUCTION WIDE, and job 372854 is why it has to.
    //
    // Mode 1 works and is wrong. RMSFusion itself is correct -- at f32 the fused
    // and unfused graphs agree to max 0.088 over the 56 normalised actions, which
    // is *below* the 0.116 that an already-accepted f16-vs-f32 change moves them.
    // But the kernel it unlocks, rms_gpu_bfyx_opt__f16, accumulates a 2560- and
    // 6912-wide mean of squares in f16, and the model does not survive that: the
    // 8-step action chunk collapses from 0.023 of per-dimension spread to 0.0036,
    // i.e. the policy emits nearly the same action eight times.
    //
    // That collapse is the evidence, not the 0.94 action delta beside it.
    // BitVLA's decorrelation floor (memory/bitvla-action-bar-cannot-bind.md) means
    // a large delta proves nothing -- one f32 ULP into act_quant already moves the
    // actions 0.068. What decorrelation does *not* do is flatten the chunk;
    // it moves actions while preserving their statistics. A dead chunk is the norm
    // failing, and the two f32 arms and the unfused f16 arm all sit at 0.022-0.024.
    //
    // reduce_ref__f32 was accidentally immune. OpenVINO's generic reference
    // reduction has only an f32 implementation, so the plugin ran it in f32 while
    // every tensor around it was f16 -- the slow path was buying accuracy nobody
    // had asked for and nobody had noticed. Naming the primitive took the
    // accumulator down with it.
    //
    // So ask for f32 explicitly instead of inheriting it by accident. RMSFusion
    // matches on op types rather than element types, so it should still fire, and
    // the plugin should then pick the f32 variant of the same fast kernel. Whether
    // it honours this or folds the Converts away during f16 compression is a
    // question for the OVPROF table, not for this comment:
    // rms_gpu_bfyx_opt__f32 means it held, __f16 means it did not.
    const int rms_mode = ggml_openvino_getenv_int("GGML_OPENVINO_RMS_FUSION");
    const auto rms_in_type = input_node.get_element_type();
    const bool rms_widen = rms_mode >= 2 && rms_in_type != ov::element::f32;
    if (rms_widen) {
        input_node = std::make_shared<ov::op::v0::Convert>(input_node, ov::element::f32);
    }

    ov::Output<ov::Node> square;
    if (rms_mode) {""",
            '// MODE 2 KEEPS THE REDUCTION WIDE, and job 372854 is why it has to.',
        ),
        (
            """    auto res = std::make_shared<ov::op::v1::Multiply>(input_node, reciprocal);

    return rename_outputs_with_suffix({res}, context.get_name());""",
            """    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(input_node, reciprocal);

    // vla.cpp: back to the type the rest of the graph expects. Converting the
    // *result* rather than letting the f32 leak onward keeps this hunk local --
    // the norm is wide, everything downstream is unchanged.
    if (rms_widen) {
        res = std::make_shared<ov::op::v0::Convert>(res, rms_in_type);
    }

    return rename_outputs_with_suffix({res}, context.get_name());""",
            '// the norm is wide, everything downstream is unchanged.',
        ),
        # Mode 3, anchored into mode 2's replacement, which is why the two hunks
        # above carry explicit markers.
        (
            """#include <openvino/op/sqrt.hpp>""",
            """#include <openvino/op/sqrt.hpp>
#include <openvino/op/util/precision_sensitive_attribute.hpp>""",
        ),
        (
            """    const bool rms_widen = rms_mode >= 2 && rms_in_type != ov::element::f32;""",
            """    const bool rms_widen = rms_mode == 2 && rms_in_type != ov::element::f32;""",
            # Mode 5 widens this test again. Match on the "== 2" that mode 3
            # introduces and every later mode keeps, not on the whole line.
            'rms_mode == 2',
        ),
        (
            """    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(input_node, reciprocal);""",
            """    auto res_mul = std::make_shared<ov::op::v1::Multiply>(input_node, reciprocal);

    // vla.cpp: MODE 3 -- the supported way to say what mode 2 tried to say by
    // hand. Job 372856 refuted mode 2: inserting explicit f32 Converts around the
    // norm changed nothing the device ran (still rms_gpu_bfyx_opt__f16, still
    // 7.19 ms against mode 1's 7.20, chunk spread still collapsed at 0.0028
    // against the unfused 0.0218). The plugin's f16 compression pass simply folded
    // the Converts away, which is what it is for -- an element type in the graph is
    // a request, and ConvertPrecision is entitled to decline it.
    //
    // ov::mark_as_precision_sensitive is the request it does not decline. Its own
    // header: "marks input to an operation as a precision sensitive and disables
    // compression to FP16 of the subgraph before this input". Marking the
    // reciprocal input names exactly the subgraph that must stay wide -- Power,
    // ReduceMean, Add(eps), Sqrt, Divide -- and leaves the tensor path through
    // input_node alone, so the f16 activations stay f16 and only the reduction
    // widens.
    //
    // Two things could still defeat it, and the OVPROF table is what says which:
    //
    //   FUSION MAY DROP THE MARK. PrecisionSensitive is is_copyable() == false, so
    //   it does not survive being copied onto a new node, and RMSFusion replaces
    //   this whole Multiply with an RMS primitive. If the mark is consumed before
    //   the fusion it holds; if after, it is gone. Reading rms_gpu_bfyx_opt__f32
    //   means it held.
    //
    //   THE MARK MAY PROPAGATE TOO FAR. It disables compression on the subgraph
    //   *before* the input, and that walk does not stop at the norm by itself. If
    //   it runs back through the residual stream, the arm gets slower rather than
    //   wrong, and total device time is what shows it.
    if (rms_mode >= 3) {
        ov::mark_as_precision_sensitive(res_mul->input(1));
    }

    ov::Output<ov::Node> res = res_mul;""",
            # Mode 4 narrows the "rms_mode >= 3" inside this replacement, so the
            # replacement text is not a usable marker. Anchor on the line mode 3
            # introduces and nothing later rewrites.
            'auto res_mul = std::make_shared<ov::op::v1::Multiply>(input_node, reciprocal);',
        ),
        # Mode 4, anchored into modes 1-3. It is the first mode that does not want
        # RMSFusion to fire at all, so it also has to narrow the two mode tests
        # above from ">=" to an exact match.
        (
            """#include <memory>""",
            """#include <cstdio>
#include <memory>
#include <string>
#include <vector>""",
        ),
        (
            """#include <openvino/op/divide.hpp>
#include <openvino/op/multiply.hpp>""",
            """#include <openvino/op/divide.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>""",
        ),
        (
            """    ov::Output<ov::Node> square;
    if (rms_mode) {""",
            """    ov::Output<ov::Node> square;
    if (rms_mode >= 1 && rms_mode <= 3) {""",
        ),
        (
            """    if (rms_mode >= 3) {
        ov::mark_as_precision_sensitive(res_mul->input(1));""",
            """    if (rms_mode == 3) {
        ov::mark_as_precision_sensitive(res_mul->input(1));""",
            # Mode 6 appends "|| rms_mode == 6" to this test, so match only the
            # part of it mode 4 introduces and mode 6 keeps.
            'if (rms_mode == 3',
        ),
        (
            """    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);""",
            """    // vla.cpp: MODE 4 -- express the reduction as a GEMM, because that is the
    // only way to get a wide accumulator the plugin cannot take back.
    //
    // Modes 1-3 all failed identically and for one reason, found after job
    // 372857: rms_kernel_bfyx_opt accumulates into ACCUMULATOR_TYPE, a JIT
    // constant the plugin derives from the tensor dtype, and squares its terms
    // with native_powr, a low-precision builtin on top of it. Summing 2560 (LM
    // attn) or 6912 (FFN) squares in f16 does not drift, it *overflows* -- the
    // f16 ceiling is 65504 and |x| ~ 10 is enough -- which is why the chunk
    // spread collapses rather than degrades. No graph-level attribute reaches a
    // JIT constant, so an explicit Convert (mode 2, job 372856) and
    // mark_as_precision_sensitive (mode 3, job 372857) were both arguing with
    // the wrong layer of the stack, and both produced the identical
    // rms_gpu_bfyx_opt__f16 at the identical 7.18-7.20 ms.
    //
    // MatMul does not have that problem: XMX/DPAS accumulates f16 x f16 into f32
    // in hardware. The wide accumulator comes from the silicon rather than from
    // an attribute ConvertPrecision is entitled to fold away. The elementwise
    // square stays f16 and is safe there -- x*x overflows f16 only above
    // |x| = 255, and it is the sum, never the square, that overflows today.
    //
    // Folding 1/K into the constant makes the mean's division free, so this is
    // one primitive replacing one primitive; the {K,1} operand gives a
    // {..., rows, 1} result, exactly the keepdims=true shape ReduceMean had, so
    // Add(eps)/Sqrt/Divide/Multiply below are untouched. RMSFusion is
    // deliberately NOT invited -- its pattern wants a ReduceMean, so it cannot
    // match and the f16 RMS kernel never enters the graph.
    //
    // 1/K rounds in f16 to ~5e-4 relative, and that reaches the output as a
    // *uniform* scale on the row. act_quant, which is what consumes this,
    // normalises by the row absmax and is exactly invariant to it.
    //
    // The prize is the same 44.8-47.2 ms / ~19% of device time that ReduceMean
    // costs on reduce_ref__f32, OpenVINO's generic *reference* reduction -- which
    // is simultaneously why it is slow and, by accident, why it is currently
    // correct, since the reference has only an f32 implementation.
    ov::Output<ov::Node> mean;
    const auto & rms_ps = square.get_partial_shape();
    const bool rms_gemm = (rms_mode == 4 || rms_mode == 5) && rms_ps.rank().is_static() &&
                          rms_ps[rms_ps.rank().get_length() - 1].is_static();
    if (rms_gemm) {
        const int64_t rms_k = rms_ps[rms_ps.rank().get_length() - 1].get_length();
        std::vector<float> rms_recip((size_t) rms_k, 1.0f / (float) rms_k);
        auto rms_matmul = std::make_shared<ov::op::v0::MatMul>(
            square, ov::op::v0::Constant::create(square.get_element_type(), ov::Shape{(size_t) rms_k, 1}, rms_recip),
            false, false);
        // Name it, because node type can no longer find it. The reduction is now
        // one MatMul among ~2600, and this change's first gate is "what replaced
        // ReduceMean costs materially less than the 44.8-47.2 ms it cost". That
        // needs a handle on exactly these nodes in the OVPROF stream, and the
        // friendly name is the only field that can carry one.
        rms_matmul->set_friendly_name("rmsgemm_" + context.get_name());
        mean = rms_matmul;
        // MODE 5 = this GEMM with the chain kept f32 (rms_widen above), and job
        // 372878 is why mode 4 is not the end of it. Mode 4 is fast -- 47.12 ms
        // of ReduceMean became 4.03 ms of FullyConnected/jit:gemm:any__f16 and
        // total device time fell 24.0% -- and it is still broken at f16: the
        // chunk spread goes to exactly 0.00000, the policy emitting one identical
        // action eight times.
        //
        // The wide accumulator was necessary and not sufficient. Job 372877's own
        // table shows why: Multiply(x, x) ran eltwise_simple_vload8__f32 in the
        // control arm and __f16 in mode 4. The plugin had been keeping the square
        // in f32 only because its consumer reduce_ref has an f32-only kernel;
        // giving it an f16-capable MatMul instead removed that reason, and
        // ConvertPrecision compressed the square. x*x overflows f16 above
        // |x| = 255 and the comment above used to assert BitVLA stays well below
        // that. It does not. XMX accumulates the *sum* in f32, but the terms
        // handed to it had already saturated.
        //
        // So the terms have to be wide as well, which is what mode 2's Converts
        // did. Mode 2 failed because RMSFusion then replaced the subgraph and
        // ConvertPrecision folded the Converts out from under it; mode 5 never
        // invites RMSFusion, so there is nothing to fold them into. If they are
        // folded anyway, the profile says so in one line -- the square goes back
        // to eltwise_simple_vload8__f16 -- and the spread collapses again.
    } else {
        // A dynamic feature axis would leave mode 4 silently doing nothing, and
        // a mode that silently does nothing is how job 372752 burned a run. Say
        // so once; the OVPROF table then agrees or disagrees with this line.
        static bool rms_warned = false;
        if ((rms_mode == 4 || rms_mode == 5) && !rms_warned) {
            rms_warned = true;
            fprintf(stderr, "GGML_OPENVINO_RMS_FUSION=%d: feature axis is dynamic, keeping ReduceMean\\n",
                    rms_mode);
        }
        mean = std::make_shared<ov::op::v1::ReduceMean>(
            square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);
    }""",
            # Mode 6 rewrites both "(rms_mode == 4 || rms_mode == 5)" tests inside
            # this replacement, so the replacement text is not a usable marker.
            # Anchor on the friendly name, which is this hunk's alone and which
            # nothing later touches.
            'rms_matmul->set_friendly_name("rmsgemm_',
        ),
        # Mode 5: mode 4's GEMM plus mode 2's wide chain. Anchored into mode 3's
        # rewrite of this line, which is why that hunk's marker is the bare
        # "rms_mode == 2" rather than the whole line.
        (
            """    const bool rms_widen = rms_mode == 2 && rms_in_type != ov::element::f32;""",
            """    const bool rms_widen = (rms_mode == 2 || rms_mode == 5) && rms_in_type != ov::element::f32;""",
            # Mode 6 appends itself to this disjunction, so match the part of it
            # that mode 5 introduces and mode 6 keeps.
            'rms_mode == 2 || rms_mode == 5',
        ),
        # Mode 6 = mode 5's Converts, held in place by mode 3's mark. Job 372880
        # is what makes this a targeted change rather than a fifth guess: mode 5's
        # Convert count rose by exactly 726 (one per norm, so rms_widen did fire)
        # while the square still ran eltwise_simple_vload8__f16. Only the
        # *output*-side Convert survived; ConvertPrecision folded the input-side
        # one, which is its job -- an element type in the graph is a request.
        #
        # So mode 2 never failed because "RMSFusion ate the Converts", and neither
        # did mode 5. ConvertPrecision declines an f32 Convert in an f16 inference
        # graph on its own, with no fusion involved.
        #
        # mark_as_precision_sensitive is the request it does not decline: "disables
        # compression to FP16 of the subgraph before this input". Mode 3 already
        # tried it and lost it, but for a reason that does not apply here --
        # PrecisionSensitive is is_copyable() == false and RMSFusion replaced the
        # node it was attached to. The GEMM never invites RMSFusion, so nothing
        # replaces the marked node.
        #
        # The mark and the Convert are both needed and neither is sufficient. The
        # mark only prevents f32 from being narrowed; it cannot widen an f16 input,
        # and rms_in_type is f16 here (proven by the Convert firing at all). Mode 5
        # supplies the f32; mode 6 stops it being taken back.
        (
            """    const bool rms_widen = (rms_mode == 2 || rms_mode == 5) && rms_in_type != ov::element::f32;""",
            """    const bool rms_widen =
        (rms_mode == 2 || rms_mode == 5 || rms_mode == 6) && rms_in_type != ov::element::f32;""",
            'rms_mode == 6) && rms_in_type',
        ),
        (
            """    const bool rms_gemm = (rms_mode == 4 || rms_mode == 5) && rms_ps.rank().is_static() &&
                          rms_ps[rms_ps.rank().get_length() - 1].is_static();""",
            """    const bool rms_wants_gemm = rms_mode >= 4 && rms_mode <= 6;
    const bool rms_gemm = rms_wants_gemm && rms_ps.rank().is_static() &&
                          rms_ps[rms_ps.rank().get_length() - 1].is_static();""",
            'rms_wants_gemm = rms_mode >= 4',
        ),
        (
            """        if ((rms_mode == 4 || rms_mode == 5) && !rms_warned) {""",
            """        if (rms_wants_gemm && !rms_warned) {""",
        ),
        (
            """    if (rms_mode == 3) {
        ov::mark_as_precision_sensitive(res_mul->input(1));""",
            """    // Mode 6 marks the same input for the opposite reason to mode 3. Mode 3
    // wanted the mark to widen the reduction on its own and it could not, because
    // the mark does not widen anything -- it only stops ConvertPrecision
    // narrowing what is already f32. Mode 6 hands it something f32 to protect:
    // the Convert that rms_widen inserted above, which job 372880 watched get
    // folded away when it was left unmarked.
    //
    // The risk the mode 3 comment named is still the one to watch, and it is now
    // the only one left: the walk back from this input does not stop at the norm
    // by itself. If it runs through the residual stream the whole model widens,
    // and that shows up as total device time going *up*, not as a wrong answer.
    // 175 ms is the number to beat; anything near the 230 ms control means the
    // mark propagated and mode 6 is as dead as the rest.
    if (rms_mode == 3 || rms_mode == 6) {
        ov::mark_as_precision_sensitive(res_mul->input(1));""",
            'if (rms_mode == 3 || rms_mode == 6)',
        ),
        # Mode 7 -- stop asking the plugin for f32 and make the f16 arithmetic
        # safe instead. Modes 2 and 5 asked with a Convert and ConvertPrecision
        # folded it (job 372880: +726 Converts inserted, square still __f16).
        # Mode 3 asked with an attribute and RMSFusion dropped it. Mode 6 asked
        # with the attribute where nothing could drop it, and it worked and cost
        # everything: job 372882 got jit:gemm:any__f32 and
        # eltwise_simple_vload8__f32 exactly as ordered, at 735.77 ms of total
        # device time against a 230.20 ms control -- the mark walks back from the
        # norm and does not stop at it, so the residual stream widened too.
        #
        # That is the whole "ask for f32" family refuted, four different ways. The
        # arithmetic route asks for nothing.
        #
        # x*x overflows f16 above |x| = 255 and a pre-norm residual stream reaches
        # that in the later blocks -- which is what the failure looked like all
        # along: overflow -> mean = inf -> 1/sqrt(inf) = 0 -> the norm outputs
        # zeros -> the head emits one constant action, which is a chunk spread of
        # exactly 0.00000 rather than a merely small one.
        #
        # So square a scaled copy. With y = x * 2^-s,
        #   mean(x^2) = sum(y^2) * 2^(2s) / K
        # and 2^(2s)/K folds into the GEMM's constant, which was already 1/K. The
        # scale-back is therefore FREE -- eps, the Sqrt, the reciprocal and the
        # gamma Multiply below are all untouched, and this stays one primitive
        # replacing one primitive. The only new work is one Multiply by a
        # constant.
        #
        # Two properties make this exact rather than a trade:
        #   - 2^-s is a power of two, so y = x * 2^-s is an exponent shift with no
        #     mantissa rounding at all in f16.
        #   - the GEMM still accumulates in f32 on XMX, so the sum is as wide as it
        #     was in mode 4, and the output is a *mean* -- bounded by the average
        #     square, not the max -- so it does not overflow on the way back out.
        # Elements below ~1e-3 go subnormal when shifted, but their squares are
        # ~1e-6 and contribute nothing to a mean the large terms dominate.
        #
        # s = 4 by default buys headroom to |x| = 4080 and costs 2^8/K on the
        # constant, which is 0.1 at K = 2560. Tunable because the right answer is
        # a property of this checkpoint's activations, not of the arithmetic.
        (
            """#include <cstdio>
#include <memory>""",
            """#include <cmath>
#include <cstdio>
#include <memory>""",
        ),
        (
            """    ov::Output<ov::Node> square;
    if (rms_mode >= 1 && rms_mode <= 3) {""",
            """    // MODE 7: square a scaled copy so the square itself cannot overflow f16.
    // rms_gain carries 2^(2s) into the GEMM constant below, where it costs
    // nothing; see the long note there for why this is the only route left.
    ov::Output<ov::Node> rms_sq_in = input_node;
    float rms_gain = 1.0f;
    float rms_post = 1.0f;
    float rms_eps_scale = 1.0f;
    if (rms_mode == 7 || rms_mode == 8) {
        int rms_shift = ggml_openvino_getenv_int("GGML_OPENVINO_RMS_SCALE_LOG2");
        if (rms_shift <= 0) {
            rms_shift = 4;
        }
        rms_sq_in = std::make_shared<ov::op::v1::Multiply>(
            input_node, ov::op::v0::Constant::create(input_node.get_element_type(), ov::Shape{1},
                                                    {std::ldexp(1.0f, -rms_shift)}));
        if (rms_mode == 7) {
            // Mode 7 undoes the scale inside the GEMM constant, which is exactly
            // why it could not work: see MODE 8 below.
            rms_gain = std::ldexp(1.0f, 2 * rms_shift);
        } else {
            rms_post = std::ldexp(1.0f, -rms_shift);
            rms_eps_scale = std::ldexp(1.0f, -2 * rms_shift);
        }
    }

    ov::Output<ov::Node> square;
    if (rms_mode >= 1 && rms_mode <= 3) {""",
            'float rms_gain = 1.0f;',
        ),
        (
            """        square = std::make_shared<ov::op::v1::Multiply>(input_node, input_node);""",
            """        square = std::make_shared<ov::op::v1::Multiply>(rms_sq_in, rms_sq_in);""",
            'Multiply>(rms_sq_in, rms_sq_in)',
        ),
        # Name the upper bound rather than inlining it: mode 8 moves it again, and
        # a literal "<= 7" is both the thing that changes and the only thing that
        # could mark this hunk. "rms_gemm_max" survives every later mode.
        (
            """    const bool rms_wants_gemm = rms_mode >= 4 && rms_mode <= 6;""",
            """    const int rms_gemm_max = 7;
    const bool rms_wants_gemm = rms_mode >= 4 && rms_mode <= rms_gemm_max;""",
            'rms_gemm_max',
        ),
        # MODE 8 -- keep the reduction scaled all the way to the reciprocal.
        #
        # Mode 7 tested one half of the overflow story and job 372883 came back
        # 0.00000 again, which is more informative than it looks. Mode 7 folds
        # 2^(2s) back into the GEMM constant, so its *output* is bit-for-bit the
        # quantity mode 4 computed; it only ever protected the per-element square.
        # The profile (job 372884) confirms the pre-scale really was in the graph
        # -- Multiply 4734 -> 5460, +726, square still eltwise_simple_vload8__f16
        # -- so the square was 256x smaller and the model died anyway.
        #
        # That leaves exactly one place for it to die: the GEMM's f16 OUTPUT.
        # mean(x^2) exceeds 65504 once rms(x) > 256, which is ordinary for a
        # pre-norm residual stream in the later blocks, and it is invisible in the
        # control arm because reduce_ref writes f32. Mode 4 was never losing the
        # sum -- XMX accumulates in f32 - it was losing the answer on the way out.
        #
        # So do not scale back into the mean at all. Carry 2^-2s through the
        # reduction and cancel it once, in the two constants that are already
        # there:
        #   mean'      = mean(x^2) * 2^-2s          (small, cannot overflow)
        #   eps'       = eps * 2^-2s
        #   reciprocal = 2^-s / sqrt(mean' + eps')  == 1 / sqrt(mean(x^2) + eps)
        # Both constants are constant-folded, so mode 8 costs exactly what mode 7
        # cost -- one Multiply -- and nothing downstream changes.
        #
        # eps' can go subnormal in f16 (eps = 1e-5 at s = 4 gives 3.9e-8) and that
        # is harmless: eps only guards an all-zero row, and on an all-zero row the
        # numerator is zero too.
        (
            """    const int rms_gemm_max = 7;""",
            """    const int rms_gemm_max = 8;""",
            'rms_gemm_max = 8',
        ),
        (
            """        std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));""",
            """        std::make_shared<ov::op::v1::Add>(
            mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps * rms_eps_scale})));""",
            'eps * rms_eps_scale',
        ),
        (
            """        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);""",
            """        std::make_shared<ov::op::v1::Divide>(
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {rms_post}), rms);""",
            'ov::Shape{1}, {rms_post})',
        ),
        (
            """        std::vector<float> rms_recip((size_t) rms_k, 1.0f / (float) rms_k);""",
            """        // rms_gain is 1.0 for modes 4-6 and 2^(2s) for mode 7, which is what
        // makes mode 7's pre-scale free: the reduction constant was already
        // there and only its value changes.
        std::vector<float> rms_recip((size_t) rms_k, rms_gain / (float) rms_k);""",
            'rms_gain / (float) rms_k',
        ),
        (
            """#include <openvino/op/negative.hpp>""",
            """#include <openvino/op/negative.hpp>
// vla.cpp: ov::op::Op, the base class mode 9's GgmlRmsRecip derives from. It
// arrives transitively through every other op header here, but a custom op
// declaration that compiles only by accident of someone else's include list is
// one upstream tidy-up away from breaking.
#include <openvino/op/op.hpp>""",
            '#include <openvino/op/op.hpp>',
        ),
        # --- MODE 9: the custom kernel ---------------------------------------
        #
        # Modes 1-8 all tried to make the PLUGIN's kernel accumulate wide, from
        # the graph. They could not: rms_gpu_bfyx_opt sums into ACCUMULATOR_TYPE,
        # a JIT constant derived from the tensor dtype, and no attribute reaches a
        # JIT constant. Mode 9 stops arguing and supplies the kernel.
        #
        # The op declaration below and ci/kernels/ggml_ov_meansq.xml are ONE
        # CONTRACT: the plugin binds a CustomLayer by matching
        # op->get_type_info().name against the descriptor's name=, so "GgmlRmsRecip"
        # has to be spelled identically in both, and changing one alone does not
        # fail loudly -- it silently leaves an op the plugin cannot execute.
        (
            """namespace op {""",
            """namespace op {

// vla.cpp: MODE 9's op -- the reciprocal RMS of the last axis, computed by
// ci/kernels/ggml_ov_meansq.cl (entry ggml_ov_rrms) instead of by the plugin.
//
// This is deliberately a type the plugin has never heard of. Job 372899
// established that an unknown op survives the whole transformation pipeline and
// that CreateCustomOp runs BEFORE the plugin's own op factory, so a CustomLayer
// named GgmlRmsRecip claims it. If the descriptor is missing, compile_model throws
// rather than producing wrong numbers -- which is the failure mode to want.
//
// It replaces FIVE nodes per norm, not one: Multiply(x, x), ReduceMean, Add(eps),
// Sqrt and the Divide. eps is therefore an input rather than a <Define>, which is
// what keeps the binding at two static files instead of a descriptor regenerated
// per eps value.
class GgmlRmsRecip : public ov::op::Op {
  public:
    GgmlRmsRecip() = default;

    // (x, eps) -> 1/sqrt(mean(x^2) + eps) along the last axis. eps is an input
    // rather than an attribute so that ci/kernels/ggml_ov_meansq.xml stays ONE
    // descriptor for the whole graph; as a <Define> it would have to be baked
    // per eps value and generated at runtime.
    GgmlRmsRecip(const ov::Output<ov::Node> & arg, const ov::Output<ov::Node> & eps) : ov::op::Op({arg, eps}) {
        constructor_validate_and_infer_types();
    }

    const ov::Node::type_info_t & get_type_info() const override {
        // "vla" is the version_id, and it is not decoration: DiscreteTypeInfo
        // compares both fields, so it is what keeps this from ever colliding with
        // an OpenVINO op of the same name.
        static ov::Node::type_info_t info{"GgmlRmsRecip", "vla"};
        return info;
    }

    void validate_and_infer_types() override {
        auto ps = get_input_partial_shape(0);
        // {..., rows, K} -> {..., rows, 1}. Rank is preserved on purpose: this is
        // exactly the shape ReduceMean(keep_dims=true) produced, so nothing
        // downstream needs a reshape and mode 9 is a one-for-one swap.
        if (ps.rank().is_static() && ps.rank().get_length() > 0) {
            ps[ps.rank().get_length() - 1] = 1;
        }
        // THE OUTPUT TYPE MUST FOLLOW THE INPUT, and job 372910 is what that
        // costs otherwise. Hardcoding f32 here does not just make this one
        // tensor wide: it pins the whole Add(eps)/Sqrt/Divide/Multiply chain
        // below at f32, so ConvertPrecision has to compress each norm's FULL
        // {rows, K} output back to f16 on the way out. That is 726 extra
        // reorder_data_fast_b1__f32 nodes at 22.9 ms each -- 8.75 SECONDS, 97%
        // of device time, against the 47 ms this whole exercise is trying to
        // win. The kernel was fine; the declaration was not.
        //
        // Nothing is lost by following the input, because of WHAT is stored.
        // The accumulator that matters is the float one INSIDE the kernel, and
        // it is unaffected by the type of the single value that leaves. That
        // value is 1/sqrt(mean + eps), not the mean: about 1e-3 to 1e-1 here,
        // five orders from f16's 65504 ceiling and five from its 6e-8 subnormal
        // floor. Storing the MEAN instead does not fit -- job 372912 measured
        // it overflowing, chunk spread exactly 0.00000 from 1/sqrt(inf).
        //
        // The headroom is believed sufficient and is NOT assumed:
        // bmg_ov_rms_fusion_check.sbatch's chunk spread is the gate on it.
        set_output_type(0, get_input_element_type(0), ps);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector & new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<GgmlRmsRecip>(new_args.at(0), new_args.at(1));
    }

    bool visit_attributes(ov::AttributeVisitor &) override { return true; }
};""",
            'class GgmlRmsRecip : public ov::op::Op',
        ),
        (
            """    ov::Output<ov::Node> mean;
    const auto & rms_ps = square.get_partial_shape();""",
            """    ov::Output<ov::Node> mean;
    // vla.cpp: MODE 9 -- hand the whole reduction to our own OpenCL kernel.
    //
    // Note it consumes input_node, NOT square: the kernel squares each term as
    // it reads it, so the f16 square tensor that broke mode 4 never exists. Job
    // 372878 is the reason that matters. Mode 4 got the wide accumulator from
    // XMX and still collapsed, because ConvertPrecision compressed Multiply(x,x)
    // to f16 once its consumer stopped being an f32-only reference kernel, and
    // x*x saturates f16 above |x| = 255. There is no intermediate left to
    // compress here.
    //
    // The `square` built above is therefore not consumed on this path. It hangs
    // off nothing, and a node no Result reaches is not in the model, so it costs
    // one host-side allocation and no device work.
    //
    // Numerics, job 372899: max rel 2.815e-07 against a double reference, where
    // the Multiply+ReduceMean this replaces measures 4.886e-04.
    const bool rms_custom = rms_mode == 9;
    // eps is read here rather than below because mode 9's kernel consumes it:
    // it folds Add(eps), Sqrt and the reciprocal into its own epilogue, exactly
    // as the plugin's rms_gpu_bfyx_opt does.
    float eps;
    memcpy(&eps, context.get_output_op_params(), sizeof(float));
    ov::Output<ov::Node> rms_recip;
    const auto & rms_ps = square.get_partial_shape();""",
            'const bool rms_custom = rms_mode == 9;',
        ),
        (
            """                          rms_ps[rms_ps.rank().get_length() - 1].is_static();
    if (rms_gemm) {""",
            """                          rms_ps[rms_ps.rank().get_length() - 1].is_static();
    if (rms_custom) {
        // Mode 9 does not produce a mean at all, so `mean` stays unset on this
        // path and the Add/Sqrt/Divide below is skipped. Job 372912 is why:
        // storing the MEAN needs an f32 output, because mean(x^2) overflows f16
        // on this model (spread collapsed to exactly 0.00000, which is
        // 1/sqrt(inf)), and job 372911 measured that an f32 output then costs
        // 8.77 s of reorders because 726 opaque f32 ops in an f16 graph wreck
        // the plugin's precision plan. The reciprocal RMS is ~1e-3 to 1e-1 --
        // safe in f16 at both ends -- so it can follow the input type and no
        // precision boundary appears.
        auto rms_ms = std::make_shared<GgmlRmsRecip>(
            input_node, ov::op::v0::Constant::create(input_node.get_element_type(), ov::Shape{1}, {eps}));
        // Same reason mode 4's GEMM is named: node type can no longer find this.
        // The first gate on mode 9 is "what replaced ReduceMean costs materially
        // less than its 47.12 ms", and the friendly name is the only field in the
        // OVPROF stream that can carry a handle on exactly these 726 nodes.
        rms_ms->set_friendly_name("rmsrecip_" + context.get_name());
        rms_recip = rms_ms;
    } else if (rms_gemm) {""",
            'if (rms_custom) {',
        ),
        # Skip the Add(eps)/Sqrt/Divide chain on mode 9: the kernel already did
        # all three, in f32 registers, before it ever wrote a result. Also drops
        # the second `float eps` now that the hunk above reads it earlier.
        (
            """    float eps;
    memcpy(&eps, context.get_output_op_params(), sizeof(float));

    auto rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::Add>(
            mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps * rms_eps_scale})));

    auto reciprocal =
        std::make_shared<ov::op::v1::Divide>(
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {rms_post}), rms);

    auto res_mul = std::make_shared<ov::op::v1::Multiply>(input_node, reciprocal);""",
            """    // vla.cpp: MODE 9 skips all three of these. ggml_ov_rrms returns
    // 1/sqrt(mean + eps) directly, which is not an optimisation but the whole
    // reason the mode works at all, and it is the same thing the plugin's own
    // rms_gpu_bfyx_opt stores. Two jobs measured why the mean cannot be the
    // boundary value instead:
    //
    //   372912 -- with an f16 output the chunk spread went to exactly 0.00000,
    //   the signature of mean = inf reaching 1/sqrt(inf) = 0. mean(x^2) passes
    //   f16's 65504 on this model, so the mean is not f16-storable.
    //
    //   372911 -- with an f32 output it is storable and costs 8.77 SECONDS of
    //   reorders, because 726 opaque f32 ops inside an f16 graph disrupt the
    //   plugin's precision plan. Convert went 230.80 ms to 8971.05 ms.
    //
    // 1/sqrt(mean + eps) runs about 1e-3 to 1e-1 here: nowhere near either f16
    // limit, so it follows the input type and neither failure has anything to
    // attach to.
    //
    // rms_post and rms_eps_scale are both 1.0 on this path. They are mode 7/8's
    // pre-scale, and rms_wants_gemm caps at mode 8, so 9 never sees them; if a
    // later mode changes that, this is the line that has to change with it.
    ov::Output<ov::Node> reciprocal;
    if (rms_custom) {
        reciprocal = rms_recip;
    } else {
        auto rms = std::make_shared<ov::op::v0::Sqrt>(
            std::make_shared<ov::op::v1::Add>(
                mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps * rms_eps_scale})));

        reciprocal =
            std::make_shared<ov::op::v1::Divide>(
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {rms_post}), rms);
    }

    auto res_mul = std::make_shared<ov::op::v1::Multiply>(input_node, reciprocal);""",
            'ov::Output<ov::Node> reciprocal;',
        ),
        # Mode 10: an RMSNorm that cannot overflow in f16, for devices that run
        # the reduction at f16 (the NPU). Default there; opt-in elsewhere.
        (
            "#include <openvino/op/reduce_mean.hpp>",
            "#include <openvino/op/abs.hpp>\n#include <openvino/op/maximum.hpp>\n#include <openvino/op/reduce_max.hpp>\n#include <openvino/op/reduce_mean.hpp>",
            "#include <openvino/op/reduce_max.hpp>",
        ),
        (
            """    const bool rms_widen =
        (rms_mode == 2 || rms_mode == 5 || rms_mode == 6) && rms_in_type != ov::element::f32;""",
            """    // vla.cpp, MODE 10: SCALE BY THE ROW'S OWN MAXIMUM.
    //
    // On Panther Lake's NPU the unmodified graph turns BitVLA's LM into NaNs:
    // the NPU compiler runs the whole norm at f16, and a 2560- or 6912-wide sum
    // of squares overflows 65504 once activations reach |x| ~ 10 (the same
    // failure mode 1 hit on the GPU's rms_gpu_bfyx_opt__f16; the GPU default only
    // survives because reduce_ref happens to be f32-only). Modes 7/8 tried a
    // fixed scale and collapsed: no constant fits every row of every layer.
    //
    // A per-row scale does: with a = max|x| over the row,
    //     x / sqrt(mean(x^2) + eps) = x / (a * sqrt(mean((x/a)^2) + eps/a^2))
    // and every term (x/a)^2 is <= 1, so the mean is <= 1 whatever the
    // activation range - nothing in the chain can overflow at f16, and the
    // squares keep f16's full relative precision instead of losing their small
    // terms under one huge one. a is floored at 1e-4, so an all-zero row still
    // divides by a finite number, and eps/a^2 stays <= 1e3.
    //
    // Selected by GGML_OPENVINO_RMS_FUSION=10, and by default whenever the
    // device is the NPU and no mode is set.
    const char * rms_dev = std::getenv("GGML_OPENVINO_DEVICE");
    const bool rms_amax = rms_mode == 10 ||
                          (rms_mode == 0 && rms_dev && std::string(rms_dev).rfind("NPU", 0) == 0);
    if (rms_amax) {
        float eps10;
        memcpy(&eps10, context.get_output_op_params(), sizeof(float));
        const auto et   = input_node.get_element_type();
        auto       axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto amax = std::make_shared<ov::op::v1::Maximum>(
            std::make_shared<ov::op::v1::ReduceMax>(std::make_shared<ov::op::v0::Abs>(input_node), axis, true),
            ov::op::v0::Constant::create(et, ov::Shape{1}, {1e-4f}));
        auto xs   = std::make_shared<ov::op::v1::Divide>(input_node, amax);
        auto ms   = std::make_shared<ov::op::v1::ReduceMean>(std::make_shared<ov::op::v1::Multiply>(xs, xs), axis, true);
        auto eps_s = std::make_shared<ov::op::v1::Divide>(
            ov::op::v0::Constant::create(et, ov::Shape{1}, {eps10}), std::make_shared<ov::op::v1::Multiply>(amax, amax));
        auto denom = std::make_shared<ov::op::v1::Multiply>(
            amax, std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(ms, eps_s)));
        ov::Output<ov::Node> res10 = std::make_shared<ov::op::v1::Divide>(input_node, denom);
        return rename_outputs_with_suffix({res10}, context.get_name());
    }

    const bool rms_widen =
        (rms_mode == 2 || rms_mode == 5 || rms_mode == 6) && rms_in_type != ov::element::f32;""",
            "// vla.cpp, MODE 10: SCALE BY THE ROW'S OWN MAXIMUM.",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/op/mulmat.cpp": [
        (
            """#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
""",
            """#include "../../ggml-openvino-extra.h"
#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
""",
        ),
        (
            """#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>""",
            """#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/transpose.hpp>""",
        ),
        (
            "#include <memory>",
            "#include <algorithm>\n#include <cstdio>\n#include <memory>\n#include <numeric>\n#include <set>\n#include <tuple>",
        ),
        (
            """    bool transpose_b = true;
    res = std::make_shared<ov::op::v0::MatMul>(A, B, false, transpose_b);""",
            """    bool transpose_b = true;

    // vla.cpp: hand the plugin a rank-<=3 MatMul, because that is the only kind
    // its int8 path will look at.
    //
    // DynamicQuantizeFullyConnected is what turns a compressed FullyConnected
    // into an int8 one. It is registered on this device -- it needs
    // supports_immad && use_onednn, both true on BMG -- but its
    // transformation_callback then skips the node when input_rank > 3, logging
    // "input rank is not supported"; the plugin also carries the string
    // "[GPU] Dynamic quantization for 4D matmul is not implemented". The ggml
    // frontend describes *everything* as rank 4 (ggml-decoder.cpp: {1,1,1,len},
    // {1,1,prefill_chunk,ctx}, {1,1,-1,-1}), so no FullyConnected in this model
    // has ever been a candidate, at any group size. The profiled graph contains
    // no DynamicQuantize node of any kind while FullyConnectedCompressed is
    // 47.4% of device time -- that, and not the hint, is what job 372831's
    // "ENGAGEMENT: NO" was measuring.
    //
    // Dropping leading axes that are statically 1 is a pure relabelling: same
    // elements, same order, and Reshape does not appear in any timed bucket of
    // the profile. Off by default -- it changes which kernel the plugin picks,
    // and that is a decision worth making explicitly.
    int64_t fc_squeeze = 0;
    if (ggml_openvino_getenv_int("GGML_OPENVINO_FC_RANK3")) {
        const auto & A_ps = A.get_partial_shape();
        const auto & B_ps = B.get_partial_shape();
        if (A_ps.rank().is_static() && B_ps.rank().is_static()) {
            const int64_t rank_a = A_ps.rank().get_length();
            const int64_t rank_b = B_ps.rank().get_length();
            // Only the FullyConnected case, and rank_b == 2 is the test for it:
            // B is the weight operand, and a weight the frontend folded in as a
            // constant arrives already rank 2, with no leading axes to lose.
            // Anything else is activation x activation -- attention's KQ and
            // KQV -- which DynamicQuantizeFullyConnected never looks at.
            //
            // Job 372875 used the looser test "the leading axis is a static 1"
            // and it was not enough. 180 of the activation x activation MUL_MATs
            // do carry a static 1 at axis 0: they squeezed to rank 3, moved off
            // the batched Gemm primitive onto MatMul, and cost ~2x the time per
            // node (+3.9 ms across MatMul/Permute/Transpose) for a benefit they
            // were never eligible for. Only the trailing two axes are the GEMM,
            // so squeezing leading static-1 axes off A alone is a relabelling:
            // same elements, same order, and the Unsqueeze below puts the ggml
            // rank back.
            const int64_t limit = (rank_b == 2) ? rank_a - 2 : 0;
            while (fc_squeeze < limit && A_ps[fc_squeeze].is_static() && A_ps[fc_squeeze].get_length() == 1) {
                ++fc_squeeze;
            }

            // One line per distinct shape family -- 2040 MUL_MAT nodes collapse
            // to a handful. Without it "the squeeze never fired" and "the
            // squeeze fired and the plugin still refused" look identical in the
            // profile, and they have completely different next steps. Only
            // reachable with the env var set, so it costs a normal run nothing.
            static std::set<std::tuple<int64_t, int64_t, int64_t>> fc_seen;
            if (fc_seen.emplace(rank_a, rank_b, fc_squeeze).second) {
                fprintf(stderr, "GGML_OPENVINO_FC_RANK3: MUL_MAT rank_a=%ld rank_b=%ld -> squeeze %ld\\n",
                        (long) rank_a, (long) rank_b, (long) fc_squeeze);
            }
        }
    }
    auto fc_axes = [&fc_squeeze]() {
        std::vector<int64_t> axes(static_cast<size_t>(fc_squeeze));
        std::iota(axes.begin(), axes.end(), (int64_t) 0);
        return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
    };
    if (fc_squeeze > 0) {
        // B is rank 2 by the gate above, so it is never squeezed; MatMul
        // broadcasts it exactly as it did before and the result rank drops by
        // precisely fc_squeeze.
        A = std::make_shared<ov::op::v0::Squeeze>(A, fc_axes());
    }

    res = std::make_shared<ov::op::v0::MatMul>(A, B, false, transpose_b);

    if (fc_squeeze > 0) {
        res = std::make_shared<ov::op::v0::Unsqueeze>(res, fc_axes());
    }""",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/op/add.cpp": [
        (
            """    auto input_0 = process_view_input_new(context, 0);
    auto input_1 = process_view_input_new(context, 1);
    auto res = std::make_shared<ov::op::v1::Add>(input_0, input_1);""",
            """    auto input_0 = process_view_input_new(context, 0);
    auto input_1 = process_view_input_new(context, 1);

    // vla.cpp: re-hang the outer add on the inner one's non-GEMM operand so the
    // GEMM is left with a single post-op. Addition is associative.
    const int oc = context.get_op_case();
    if (oc == 2 || oc == 3) {
        auto inner = input_0.get_node_shared_ptr();
        if (inner->get_input_size() == 2) {
            const size_t keep = (oc == 2) ? 0 : 1;
            const size_t fold = 1 - keep;
            auto folded = std::make_shared<ov::op::v1::Add>(inner->input_value(fold), input_1);
            auto res2 = std::make_shared<ov::op::v1::Add>(inner->input_value(keep), folded);
            return rename_outputs_with_suffix({res2}, context.get_name());
        }
    }

    auto res = std::make_shared<ov::op::v1::Add>(input_0, input_1);""",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/op/concat.cpp": [
        (
            """#include <openvino/op/concat.hpp>
#include <openvino/op/convert.hpp>""",
            """#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/unsqueeze.hpp>""",
        ),
        ("#include <memory>", "#include <memory>\n#include <numeric>"),
        (
            """    const auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{input_0, input_1}, axis);""",
            """    // vla.cpp: a weight that is 2-D in ggml is folded in as a rank-2 constant,
    // because that is what a GEMM operand wants. Concat is the one op where that
    // matters: it needs both inputs at the graph's rank, or the axis computed
    // below falls outside them. Left-pad the shorter one with 1s, which is the
    // shape ggml gave it anyway.
    auto align_rank = [rank](ov::Output<ov::Node> in) {
        const auto & ps = in.get_partial_shape();
        if (ps.rank().is_dynamic() || ps.rank().get_length() >= rank) {
            return in;
        }
        std::vector<int64_t> axes(rank - ps.rank().get_length());
        std::iota(axes.begin(), axes.end(), 0);
        auto axes_node = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
        return ov::Output<ov::Node>(std::make_shared<ov::op::v0::Unsqueeze>(in, axes_node));
    };
    input_0 = align_rank(input_0);
    input_1 = align_rank(input_1);

    const auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{input_0, input_1}, axis);""",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/utils.cpp": [
        ("#include <memory>", "#include <memory>\n#include <numeric>"),
        (
            """        std::vector<int64_t> gather_indices(n_dims_half);
        for (size_t j = 0; j < n_dims_half; j++) {
            gather_indices[j] = j % 3;
            factor[j] = std::pow(theta_scale, j);
        }""",
            """        // vla.cpp: ggml's interleaved mrope cycles t/h/w by sector % 3, but only
        // while the sector is still inside 3 * sections[k]; past that it falls
        // through to the fourth position stream. Ignoring the bound sends the
        // tail sectors to the wrong stream -- with sections {24,20,20,0} and
        // n_dims 128, sectors 61 and 62 take h and w instead of e. See
        // ggml_rope_cache_init in ggml/src/ggml-cpu/ops.cpp.
        const int32_t * sections = rope_params + 11;
        std::vector<int64_t> gather_indices(n_dims_half);
        for (size_t j = 0; j < n_dims_half; j++) {
            const int sector = (int) j;
            int64_t stream = 3;
            if (sector % 3 == 1 && sector < 3 * sections[1]) {
                stream = 1;
            } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
                stream = 2;
            } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
                stream = 0;
            }
            gather_indices[j] = stream;
            factor[j] = std::pow(theta_scale, j);
        }""",
        ),
        (
            "#include <openvino/op/transpose.hpp>",
            "#include <openvino/op/transpose.hpp>\n#include <openvino/op/unsqueeze.hpp>",
        ),
        (
            """    size_t view_input_size = context.get_view_input_size(input_index);
    if (view_input_size == 0) {
        // No view inputs, return the input as is
        return input;
    }
""",
            """    size_t view_input_size = context.get_view_input_size(input_index);
    if (view_input_size == 0) {
        // No view inputs, return the input as is
        return input;
    }

    // vla.cpp: a ggml tensor that is 2-D folds in as a rank-2 ov constant, which
    // is what a GEMM operand wants, but every slice below indexes the tensor at
    // its full ggml rank. Left-pad with 1s so the axes line up. Evo-1 hits this
    // by viewing Q/K/V out of one fused attn_in weight.
    {
        const auto src_ggml_shape = context.get_view_input_src_ggml_shape(input_index, 0);
        const auto & in_ps = input.get_partial_shape();
        if (in_ps.rank().is_static() && (size_t) in_ps.rank().get_length() < src_ggml_shape.size()) {
            std::vector<int64_t> axes(src_ggml_shape.size() - (size_t) in_ps.rank().get_length());
            std::iota(axes.begin(), axes.end(), 0);
            input = std::make_shared<ov::op::v0::Unsqueeze>(
                input, ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes));
        }
    }
""",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/op_table.cpp": [
        (
            # vla.cpp: ggml_openvino_getenv_int is declared here, and
            # translate_bitvla_act_quant needs it for GGML_OPENVINO_ACT_QUANT_ELIDE.
            # Note the path is one level shallower than the op/*.cpp files use:
            # op_table.cpp sits in openvino/, not openvino/op/.
            """#include "op_table.h\"""",
            """#include "../ggml-openvino-extra.h"

#include "op_table.h\"""",
            '#include "../ggml-openvino-extra.h"',
        ),
        (
            """namespace ov {
namespace frontend {
namespace ggml {

std::unordered_map<std::string, CreatorFunction> get_supported_ops() {""",
            """namespace ov {
namespace frontend {
namespace ggml {

namespace op {
// vla.cpp: ggml's GGML_UNARY_OP_GELU is the *tanh* approximation (its CPU kernel
// additionally reads an fp16 lookup table); ov::op::v7::Gelu defaults to the
// exact erf formulation. Mapping the tanh op onto erf is a real approximation
// mismatch -- small per node, but a vision tower has dozens of them and the
// error compounds through the whole encoder.
static OutputVector translate_gelu_tanh(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    auto input = process_view_input_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(input, ov::op::GeluApproximationMode::TANH);
    return rename_outputs_with_suffix({res}, context.get_name());
}
}  // namespace op

std::unordered_map<std::string, CreatorFunction> get_supported_ops() {""",
        ),
        (
            """        {"GGML_UNARY_OP_GELU",      op::translate_1to1_match_1_input<v7::Gelu>     },""",
            """        {"GGML_UNARY_OP_GELU",      op::translate_gelu_tanh                        },
        // vla.cpp: tanh approximation for GELU, exact erf for GELU_ERF. ov's Gelu
        // defaults to erf, so the tanh variant must set its mode explicitly.
        {"GGML_UNARY_OP_GELU_ERF",  op::translate_1to1_match_1_input<v7::Gelu>     },""",
        ),
        (
            # Its own namespace block on the far side of the include list, rather than
            # sharing the one the GELU hunk above opens. Hunks are skipped by testing
            # whether their replacement is already present, so inserting into another
            # hunk's replacement text makes that hunk look unapplied on the next run
            # and the whole patch fails on a re-configure.
            """#include <openvino/op/tanh.hpp>
""",
            """#include <openvino/op/tanh.hpp>

#include <openvino/frontend/exception.hpp>
#include <openvino/op/abs.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/reduce_max.hpp>
#include <openvino/op/round.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// vla.cpp: BitVLA's BitNet activation quantiser, the one op in that model's graph
// that is not stock ggml. It arrives as GGML_OP_MAP_CUSTOM1, ggml's generic escape
// hatch: such an op identifies its kernel by a host function pointer, which means
// nothing to another backend, and get_op_type() stringifies only the op enum -- so
// every custom op anyone writes is indistinguishable here. The tensor name is the
// only discriminator, and vla.cpp sets it in act_quant() (src/models/bitvla.cpp,
// BITVLA_ACT_QUANT_NAME). Keep the two in step.
//
// Substring rather than equality, because both sides may decorate the name with a
// "<n>#" disambiguator: vla::graph_unique_names() prepends one in src/backend.h,
// get_tensor_ov_name() appends one here.
//
// The op is a fake-quant -- round to int8 against a per-row scale, then undo that
// scale -- so it is shape- and type-preserving, and the arithmetic mirrors
// bitvla_act_quant_op line for line:
//
//     amax  = max(reduce_max(|x|, axis=-1), 1e-5)
//     s     = 127 / amax
//     inv_s = 1 / s
//     out   = clamp(rint(x * s), -128, 127) * inv_s
//
// Two steps there are load-bearing and must not be tidied up. Rounding is
// HALF_TO_EVEN because the C uses nearbyintf under the default rounding mode, and
// Round is a threshold: a value landing on the wrong side of a .5 boundary changes
// an integer, not a low-order bit. And the last line multiplies by a separately
// rounded 1/s rather than dividing by s, again because that is what the C does --
// the two differ by one rounding, and a 1-ULP error in exactly this scale was a
// shipping bug on both the CUDA and the SYCL kernels (see vla_exact_div in
// src/kernels/bitvla/cuda_compat.h).
static OutputVector translate_bitvla_act_quant(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    FRONT_END_OP_CONVERSION_CHECK(context.get_name().find("bitvla.act_quant") != std::string::npos,
                                  "MAP_CUSTOM1 is a generic ggml custom op and only vla.cpp's "
                                  "bitvla.act_quant is recognised here; got '", context.get_name(), "'");

    auto input = process_view_input_new(context, 0);

    // vla.cpp: GGML_OPENVINO_ACT_QUANT_ELIDE -- drop the chain and hand the raw
    // activations downstream, so the GPU plugin's DynamicQuantize does the
    // quantisation instead of redoing it.
    //
    // WHY THIS IS NOT A NUMERICS CHANGE. The plugin's dynamic quantiser is
    // per-token symmetric int8: scale = rowmax(|x|)/127, codes = rint(x*127/rowmax).
    // That is the same operation the chain above spells out, so eliding one and
    // enabling the other computes the same codes rather than different ones. It
    // is NOT a claim that the result is bit-identical: the plugin rounds with its
    // own mode and applies the scale on the accumulator side, and the two differ
    // by roundings of exactly the kind that are load-bearing here. Which is why
    // the gate is LIBERO task success and never action equality
    // (memory/bitvla-action-bar-cannot-bind.md).
    //
    // ONLY MEANINGFUL WITH GGML_OPENVINO_FC_RANK3. On its own this removes the
    // model's quantisation and puts nothing in its place: the frontend emits rank
    // 4 everywhere, DynamicQuantizeFullyConnected rejects input_rank > 3, and no
    // DynamicQuantize node is ever created. The two ship together or not at all,
    // and the structure gate is what proves the second half happened -- count
    // DynamicQuantize nodes, do not infer them from a latency change.
    if (ggml_openvino_getenv_int("GGML_OPENVINO_ACT_QUANT_ELIDE")) {
        return rename_outputs_with_suffix({input}, context.get_name());
    }

    auto row_max = std::make_shared<ov::op::v1::ReduceMax>(std::make_shared<ov::op::v0::Abs>(input),
                                                          ov::op::v0::Constant::create(ov::element::i64,
                                                                                       ov::Shape{1}, {-1}),
                                                          true);
    auto amax = std::make_shared<ov::op::v1::Maximum>(
        row_max, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1e-5f}));

    auto s = std::make_shared<ov::op::v1::Divide>(
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {127.0f}), amax);
    auto inv_s = std::make_shared<ov::op::v1::Divide>(
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), s);

    auto rounded = std::make_shared<ov::op::v5::Round>(std::make_shared<ov::op::v1::Multiply>(input, s),
                                                      ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto q = std::make_shared<ov::op::v0::Clamp>(rounded, -128.0, 127.0);

    auto res = std::make_shared<ov::op::v1::Multiply>(q, inv_s);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
""",
        ),
        (
            """        {"GGML_OP_ROLL",            op::translate_roll                             },""",
            """        {"GGML_OP_ROLL",            op::translate_roll                             },
        // vla.cpp: BitVLA's activation quantiser. Unlike every other entry here this
        // one is not "translate this op" but "translate this op when the node carries
        // a particular name" -- MAP_CUSTOM1 is a slot, not an operation. The name is
        // checked in translate_bitvla_act_quant and, so the scheduler can still route
        // an unrecognised custom op to the CPU rather than reach a translator that
        // will throw, again in ggml_backend_openvino_device_supports_op.
        {"GGML_OP_MAP_CUSTOM1",     op::translate_bitvla_act_quant                 },""",
        ),
    ],
    "ggml/src/ggml-openvino/ggml-openvino.cpp": [
        (
            """    default: {
        auto supported = supported_ops.find(op->op) != supported_ops.end();""",
            """    case GGML_OP_MAP_CUSTOM1: {
        // vla.cpp: the op table's supported set is derived from its keys, so adding a
        // MAP_CUSTOM1 translator makes this backend claim *every* MAP_CUSTOM1 node --
        // and a custom op names its kernel with a host function pointer, so one
        // translator cannot possibly be right for all of them. Without this the next
        // model to use the escape hatch for something else would get a
        // FRONT_END_OP_CONVERSION_CHECK at translation time instead of the graceful
        // CPU fallback the scheduler is there to provide. Same name test as the
        // translator, for the same reason; see translate_bitvla_act_quant.
        if (std::string(op->name).find("bitvla.act_quant") == std::string::npos) {
            return {false, "MAP_CUSTOM1 is a custom op this backend does not recognise"};
        }
        if (op->src[0] == nullptr || op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
            return {false, "bitvla.act_quant is only translated for F32"};
        }
        break;
    }
    default: {
        auto supported = supported_ops.find(op->op) != supported_ops.end();""",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/op/flash_attn_ext.cpp": [
        (
            """    auto q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);""",
            """    auto q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    // vla.cpp: Q, the mask and the scale below are all forced to F16 because
    // llama.cpp's KV cache already is. K/V that arrive as F32 have to come along
    // or SDPA rejects the mix.
    if (k.get_element_type() != ov::element::f16) {
        k = std::make_shared<ov::op::v0::Convert>(k, ov::element::f16);
    }
    if (v.get_element_type() != ov::element::f16) {
        v = std::make_shared<ov::op::v0::Convert>(v, ov::element::f16);
    }""",
        ),
    ],
    "ggml/src/ggml-openvino/ggml-decoder.h": [
        (
            """    std::string get_graph_input_ov_name(const ggml_tensor * tensor, const ggml_tensor * op) const {
        if (is_inp_pos(tensor, op)) {
            return "inp_pos";
        }""",
            """    // vla.cpp: only collapse ROPE position inputs onto one "inp_pos" parameter
    // when the graph really has one. See scripts/patch_ggml_openvino.py.
    bool has_multiple_inp_pos() const;

    std::string get_graph_input_ov_name(const ggml_tensor * tensor, const ggml_tensor * op) const {
        if (is_inp_pos(tensor, op)) {
            return has_multiple_inp_pos() ? std::string(tensor->name) : std::string("inp_pos");
        }""",
        ),
        (
            "    ggml_cgraph * m_cgraph = nullptr;",
            "    ggml_cgraph * m_cgraph = nullptr;\n"
            "    mutable int m_multi_inp_pos = -1;  // vla.cpp: lazily computed, -1 = unknown",
        ),
    ],
    "ggml/src/ggml-openvino/ggml-decoder.cpp": [
        (
            """    case GGML_OP_ADD: {
        if (is_moe_expert_sum_add(node)) {""",
            """    case GGML_OP_ADD: {
        // vla.cpp: ADD(ADD(.., GEMM), graph-input). Two elementwise ops chained on a
        // GEMM; the GPU plugin folds both in as post-ops and loses the second operand.
        // op_case 2 = the inner add's input 0 is the GEMM, 3 = its input 1 is.
        {
            const ggml_tensor * inner = node->src[0];
            const ggml_tensor * other = node->src[1];
            if (inner && other && inner->op == GGML_OP_ADD && other->op == GGML_OP_NONE) {
                for (int k = 0; k < 2; k++) {
                    const ggml_tensor * s = inner->src[k];
                    while (s && s->src[0] &&
                           (s->op == GGML_OP_VIEW || s->op == GGML_OP_RESHAPE || s->op == GGML_OP_CONT)) {
                        s = s->src[0];
                    }
                    if (s && s->op == GGML_OP_MUL_MAT) {
                        op_case = (k == 0) ? 2 : 3;
                    }
                }
            }
        }
        if (is_moe_expert_sum_add(node)) {""",
        ),
        (
            """    if (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        return "inp_pos";
    }""",
            """    if (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        // vla.cpp: this free function is the live naming path -- the
        // GgmlOvDecoder member of the same intent is unreferenced at b10729.
        // Only collapse ROPE position inputs onto one "inp_pos" parameter when
        // the graph really has one. See scripts/patch_ggml_openvino.py.
        return decoder->has_multiple_inp_pos() ? get_tensor_ov_name(cgraph, tensor) : std::string("inp_pos");
    }""",
        ),
        (
            """        } else {
            // rope'ed query tensor
            op_case = 2;""",
            """        } else {
            // vla.cpp: op_case 2 rewrites the tensor as [n_seq, -1, n_heads,
            // head_size] before transposing, which is only correct for
            // llama.cpp's rope'd query. It was reached by ANY permute whose
            // source is a view of a non-leaf, so a DiT head split -- GR00T
            // N1.7's cross-attention V, ggml_permute(view, 1,2,0,3) -- was
            // reshaped into a shape that has nothing to do with it and came out
            // with its elements rearranged. Require the rope.
            const ggml_tensor * prod = node->src[0];
            while (prod && prod->src[0] &&
                   (prod->op == GGML_OP_VIEW || prod->op == GGML_OP_RESHAPE || prod->op == GGML_OP_CONT)) {
                prod = prod->src[0];
            }
            op_case = (prod && prod->op == GGML_OP_ROPE) ? 2 : 1;""",
        ),
        (
            """        } else if (src->ne[0] * src->ne[1] * src->ne[2] == node->ne[1]) {
            op_case = 3;""",
            """            // vla.cpp: case 3 is the KV-cache flatten, whose result is always
            // [1, n, 1, 1]. Without the ne[0] test it also swallows the kernel
            // reshape ggml_conv_2d emits and rewrites it to the wrong shape.
        } else if (src->ne[0] * src->ne[1] * src->ne[2] == node->ne[1] && node->ne[0] == 1) {
            op_case = 3;""",
        ),
        (
            "int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) const {",
            """bool GgmlOvDecoder::has_multiple_inp_pos() const {
    if (m_cgraph == nullptr) {
        return false;
    }
    if (m_multi_inp_pos < 0) {
        std::set<const ggml_tensor *> seen;
        for (int i = 0; i < m_cgraph->n_nodes && seen.size() < 2; i++) {
            const ggml_tensor * node = m_cgraph->nodes[i];
            for (int j = 0; j < GGML_MAX_SRC && node->src[j] != nullptr; j++) {
                if (is_inp_pos(node->src[j], node)) {
                    seen.insert(node->src[j]);
                }
            }
        }
        m_multi_inp_pos = seen.size() > 1 ? 1 : 0;
    }
    return m_multi_inp_pos == 1;
}

int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) const {""",
        ),
    ],
    "ggml/src/ggml-openvino/utils.h": [
        (
            "struct decoder_runtime_ctx {",
            """// vla.cpp: what naive_compute() reuses across calls on the same graph. Without
// it that path rebuilt the decoder, re-converted the model and called
// compile_model() on every ggml_backend_graph_compute, which dominated runtime.
struct naive_runtime_ctx {
    std::mutex mutex;
    std::shared_ptr<GgmlOvDecoder> decoder;
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::InferRequest> infer_request;
};

// vla.cpp: graph_key is {n_nodes, first name, last name}, which two graphs of the
// same size can share. A compiled model is bound to the shapes it was built for,
// so reusing one across a shape change returns another graph's answer with no
// error. Mix the ops and shapes in as well.
inline uint64_t naive_graph_sig(const ggml_cgraph * cgraph) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        mix((uint64_t) node->op);
        mix((uint64_t) node->type);
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            mix((uint64_t) node->ne[d]);
        }
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * src = node->src[s];
            mix(src ? (uint64_t) src->type + 1 : 0ull);
            for (int d = 0; src && d < GGML_MAX_DIMS; d++) {
                mix((uint64_t) src->ne[d]);
            }
        }
    }
    return h;
}

struct naive_key {
    graph_key base;
    uint64_t  sig;

    naive_key(const ggml_cgraph * cgraph) : base(cgraph), sig(naive_graph_sig(cgraph)) {}

    bool operator==(const naive_key & other) const { return sig == other.sig && base == other.base; }
};

struct naive_key_hash {
    size_t operator()(const naive_key & key) const {
        size_t h = graph_key_hash{}(key.base);
        return h ^ (std::hash<uint64_t>{}(key.sig) + 0x9e3779b9 + (h << 6) + (h >> 2));
    }
};

struct decoder_runtime_ctx {""",
        ),
        (
            "    std::unordered_map<graph_key, std::shared_ptr<decoder_runtime_ctx>, graph_key_hash> decoder_cache;",
            "    std::unordered_map<graph_key, std::shared_ptr<decoder_runtime_ctx>, graph_key_hash> decoder_cache;\n"
            "    std::unordered_map<naive_key, std::shared_ptr<naive_runtime_ctx>, naive_key_hash> naive_cache;",
        ),
        (
            """        decoder_cache.clear();
        infer_request_cache.clear();""",
            """        decoder_cache.clear();
        naive_cache.clear();
        infer_request_cache.clear();""",
        ),
        (
            """enum ggml_status naive_compute(struct ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config);""",
            """enum ggml_status naive_compute(struct ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config,
                               std::shared_ptr<ov_runtime_context> r_ctx);""",
        ),
    ],
    "ggml/src/ggml-openvino/utils.cpp": [
        (
            # vla.cpp: for vla_dump_runtime_model's sequence counter, which is
            # reached from both the dynamic and the naive path.
            """#include <algorithm>""",
            """#include <algorithm>
#include <atomic>""",
            "#include <atomic>",
        ),
        # Anchored inside naive_compute, which is the path vla.cpp's graphs take
        # (is_naive fires on everything that is not a decoder-only LLM). The
        # dynamic and static paths have their own infer() calls; they are not
        # patched because nothing here reaches them.
        (
            """    infer_request->infer();

    auto ov_results = model->get_results();""",
            """    infer_request->infer();

    // vla.cpp: per-op device attribution. unitrace names the kernels the plugin
    // ran -- gemm_kernel, reduce_ref, generic_eltwise_ref -- but not which ggml
    // op each came from, and the two do not map one to one: a single reduce_ref
    // row covers act_quant's row absmax, RMSNorm's mean-of-squares and every
    // other reduction in the graph at once. Job 372831 could therefore size the
    // GEMMs (52% of device time, solid across three weight formats) but could
    // not say what the other 32% is, and "fuse the elementwise work" is not an
    // instruction until that is answered. ov::ProfilingInfo is the only place
    // carrying both the op name and the device time.
    //
    // The value is how many infer calls to dump: one step issues many graphs and
    // dumping all of them buries the run. Printed as fixed-field lines rather
    // than a formatted table because the consumer is a script, and node names
    // contain spaces, so the name goes last.
    {
        static const int prof_ops = ggml_openvino_getenv_int("GGML_OPENVINO_PROFILE_OPS");
        static int prof_calls = 0;
        if (prof_ops > 0 && prof_calls < prof_ops) {
            prof_calls++;
            const auto prof = infer_request->get_profiling_info();
            GGML_LOG_INFO("OVPROF_BEGIN call=%d nodes=%zu\\n", prof_calls, prof.size());
            for (const auto & p : prof) {
                const char * st = p.status == ov::ProfilingInfo::Status::EXECUTED       ? "EXEC" :
                                  p.status == ov::ProfilingInfo::Status::OPTIMIZED_OUT  ? "OPTOUT" :
                                                                                          "NOTRUN";
                // real_time is the node's device duration. cpu_time is the host
                // side of the same node and would double-count if added to it.
                GGML_LOG_INFO("OVPROF %lld %s %s %s %s\\n", (long long) p.real_time.count(), st,
                              p.node_type.c_str(), p.exec_type.c_str(), p.node_name.c_str());
            }
            GGML_LOG_INFO("OVPROF_END\\n");
        }
    }

    auto ov_results = model->get_results();""",
        ),
        (
            """        if (!model_is_splitted) {
            return naive_compute(cgraph, core, device, config);
        }""",
            """        if (!model_is_splitted) {
            return naive_compute(cgraph, core, device, config, r_ctx);
        }""",
        ),
        (
            """    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config);
    }""",
            """    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config, r_ctx);
    }""",
        ),
        (
            NAIVE_COMPUTE_OLD,
            NAIVE_COMPUTE_NEW,
            # Explicit marker. The default marker is the replacement text, and
            # the GGML_OPENVINO_DUMP_RUNTIME hunk at the end of this group
            # rewrites part of it -- so on a second apply the default marker
            # would not be found, this hunk would try again, and its anchor is
            # long gone. Key on a line no later hunk touches instead -- and one
            # that is unique to the naive path, since the cache_enabled line
            # this first reached for already exists upstream on the non-naive
            # one, which would skip this hunk on a clean tree.
            "            r_ctx->naive_cache[key] = entry;",
        ),
        (
            """bool is_naive(ggml_cgraph * cgraph) {
    constexpr int naive_graph_size_threshold = 20;""",
            """bool is_naive(ggml_cgraph * cgraph) {
    // vla.cpp: the literal translation path suits any graph that is not a
    // decoder-only LLM, so let the caller raise the bar it is chosen under.
    // Junk parses to 0 under atoi, which would silently send every graph down
    // the LLM builder, so reject anything that is not a whole positive number.
    static const int naive_graph_size_threshold = [] {
        const char * env = getenv("GGML_OPENVINO_NAIVE_GRAPH_SIZE");
        if (env == nullptr || *env == '\\0') {
            return 20;
        }
        char *     end = nullptr;
        const long val = strtol(env, &end, 10);
        const int  n   = (int) val;
        if (*end != '\\0' || val <= 0 || (long) n != val) {
            GGML_LOG_WARN("GGML OpenVINO Backend: ignoring GGML_OPENVINO_NAIVE_GRAPH_SIZE='%s'\\n", env);
            return 20;
        }
        return n;
    }();""",
        ),
        (
            # vla.cpp: GGML_OPENVINO_DUMP_RUNTIME -- the *transformed* graph.
            #
            # GGML_OPENVINO_DUMP_IR serialises the frontend's output, which is the
            # INPUT to every plugin decision and so answers none of them.
            # get_runtime_model() is the only view of what the plugin actually
            # did: which primitives fused, what precision each runs at, and -- the
            # question this was added for (job 372927) -- which FullyConnected
            # nodes got a DynamicQuantize in front of them and which did not.
            #
            # The helper goes in ahead of the first caller; the two call sites
            # follow as their own hunks. One file per compiled graph, numbered,
            # because IR_naive.xml is a single fixed name that every graph
            # overwrites -- which is how jobs 372752 and 372755 came to read the
            # 8x17920 action head and take it for the model.
            """enum ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {""",
            """// vla.cpp: serialise the post-transformation graph when
// GGML_OPENVINO_DUMP_RUNTIME is set. Written to the CWD, like IR_naive.xml --
// the caller cds. Numbered, one file per compiled graph.
static void vla_dump_runtime_model(const ov::CompiledModel & compiled_model) {
    if (!ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_RUNTIME")) {
        return;
    }
    static std::atomic<int> dump_seq{0};
    const std::string dump_path = "runtime_model_" + std::to_string(dump_seq++) + ".xml";
    auto runtime_model = compiled_model.get_runtime_model();
    ov::serialize(runtime_model, dump_path);
    GGML_LOG_INFO("GGML OpenVINO Backend: runtime model -> %s (%zu ops)\\n",
                  dump_path.c_str(), runtime_model->get_ops().size());
}

enum ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {""",
            "static void vla_dump_runtime_model(",
        ),
        (
            # Call site 1: ov_graph_compute_dynamic. THIS is the one that matters
            # -- the 2040-FullyConnectedCompressed models job 372927 profiled come
            # through here, not through naive_compute.
            """                compile_end_time = ggml_time_us();""",
            """                vla_dump_runtime_model(compiled_model);
                compile_end_time = ggml_time_us();""",
            # 16-space indent, so this cannot be confused with call site 2's
            # 8-space one (of which it would otherwise be a superstring).
            "                vla_dump_runtime_model(compiled_model);",
        ),
        (
            # Call site 2: naive_compute. Small per-op graphs, kept for symmetry
            # so a dump is never silently missing for whichever path a graph took.
            # LAST in this group on purpose: it anchors on text an earlier hunk
            # here produces, so it must run after that one.
            """        if (remote_context.has_value()) {
            infer_request = std::make_shared<ov::InferRequest>(
                core.compile_model(model, remote_context.value(), config).create_infer_request());
        } else {
            infer_request =
                std::make_shared<ov::InferRequest>(core.compile_model(model, device, config).create_infer_request());
        }""",
            """        ov::CompiledModel compiled_model;
        if (remote_context.has_value()) {
            compiled_model = core.compile_model(model, remote_context.value(), config);
        } else {
            compiled_model = core.compile_model(model, device, config);
        }
        infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
        vla_dump_runtime_model(compiled_model);""",
            # Indent-sensitive: the marker test is a substring test, and BOTH the
            # bare call and the create_infer_request() line already appear here
            # at deeper indents (utils.cpp:488, :729 upstream), so either alone
            # would skip this hunk on a clean tree. Two lines that only ever sit
            # together on the naive path.
            """        ov::CompiledModel compiled_model;
        if (remote_context.has_value()) {""",
        ),
    ],
}


# A duplicate key in EDITS silently drops the earlier entry's hunks -- Python keeps
# the last one and nothing complains. That happened once and took the whole Intel
# OpenCL fix out of the patch, so compare the literal keys against the dict.
_keys = re.findall(r'^    "([^"]+)": \[$', pathlib.Path(__file__).read_text(), re.M)
if len(_keys) != len(EDITS):
    _dup = sorted({k for k in _keys if _keys.count(k) > 1})
    sys.exit(f"patch_ggml_openvino: EDITS has {len(_keys)} literal keys but {len(EDITS)} entries; "
             f"duplicate key(s): {_dup}")


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

    pending = []
    for rel, edits in EDITS.items():
        src = root / rel
        if not src.is_file():
            print(f"patch_ggml_openvino: {src} not found", file=sys.stderr)
            return 1
        original = src.read_text()
        text = original
        for anchor, replacement, *rest in edits:
            # Per hunk, not per file. The hunk list grows between commits, so a
            # tree patched by an older checkout still needs the newer ones and a
            # file-wide marker would skip them, leaving a build that runs and is
            # quietly wrong.
            #
            # "Already applied" normally means the replacement is present, but
            # that test fails for a hunk another hunk is anchored *into*. Adding
            # an entry to the env-var allowlist rewrites the region ending in
            # "};", so each new entry destroys the previous hunk's replacement
            # text and the previous hunk then looks unapplied and fails on a
            # missing anchor. An optional third element names a marker to look
            # for instead: something the hunk introduces that later hunks leave
            # alone. Hunks that are only ever appended *after* do not need one.
            marker = rest[0] if rest else replacement
            if marker in text:
                continue
            n = text.count(anchor)
            if n != 1:
                print(f"patch_ggml_openvino: anchor matched {n} times in {rel}, expected 1:\n{anchor}\n"
                      f"A tree patched by an older checkout reads like this. Delete the fetched "
                      f"llama.cpp (rm -rf <build>/_deps) and reconfigure. If the tree is clean, the "
                      f"anchor no longer matches VLA_LLAMA_TAG and needs re-targeting.",
                      file=sys.stderr)
                return 1
            text = text.replace(anchor, replacement, 1)
        if text != original:
            pending.append((src, text, rel))

    # Every anchor matched, so nothing was half-written on the way here.
    for src, text, rel in pending:
        src.write_text(text)
        print(f"patch_ggml_openvino: patched {rel}")
    if not pending:
        print("patch_ggml_openvino: already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
