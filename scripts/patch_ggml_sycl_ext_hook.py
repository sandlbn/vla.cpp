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

"""Add one extension hook to the fetched ggml SYCL backend.

The twin of scripts/patch_ggml_cuda_ext_hook.py, and deliberately smaller than
it. The kernels it enables live in src/sycl/vla_sycl_bf16.cpp as ordinary
in-tree code, depending only on the public ggml header, so a GIT_TAG bump
cannot break them.

How much less is needed here than on CUDA
-----------------------------------------
ggml-sycl is further along on BF16 than ggml-cuda, and the difference is worth
stating precisely, because it is the reason this script makes two edits where
the CUDA one makes four:

  * ADD / MUL already have BF16 instantiations - both BF16 x BF16 -> BF16 and
    BF16 x F32 -> BF16 (ggml/src/ggml-sycl/binbcast.cpp). ggml-cuda has none.
  * The unary dispatcher already covers BF16 behind GGML_SYCL_HAS_BF16, which
    icpx defines (ggml/src/ggml-sycl/element_wise.cpp,
    dispatch_ggml_sycl_op_unary). ggml-cuda instantiates F32/F16 only.
  * ggml-sycl has no multi-node ADD/MUL fusion at all. ggml_sycl_fuse() is
    topk-moe only, and the two binary fusions it does have are RMS_NORM+MUL and
    UNARY+MUL. So there is no fused-binbcast run to intercept, and the CUDA
    hook's second function pointer has no counterpart.
  * The UNARY+MUL fusion check already *declines* on a non-F32/F16 type rather
    than asserting, so a BF16 silu+mul falls through to the unfused path on its
    own. That costs a fusion, not a crash, and buying it back would mean
    patching a kernel rather than a guard - out of scope for a hook.

What is left over is what ggml-sycl genuinely cannot do, and it is exactly the
set src/sycl/vla_sycl_bf16.cpp implements: NORM, RMS_NORM and SCALE assert F32
outright, and ggml_sycl_op_mul_mat_sycl hands oneDNN a to_dt<float>() dst
unconditionally (for a BF16 dst that is both wrong and twice the bytes the
allocator reserved). There is no way to register kernels for built-in ops from
outside: GGML_OP_CUSTOM is CPU-only. So the backend has to offer one place where
an external implementation gets first refusal.

What it changes
---------------
  1. ggml/src/ggml-sycl/ggml-sycl.cpp: one exported function pointer, null by
     default, and one call to it in ggml_sycl_compute_forward. Returning false
     means "not mine", and ggml runs the op exactly as before. The call sits
     after the g_sycl_loaded check and initialize_sycl_begining(), so the
     extension never runs against an uninitialised device, and it is handed
     ctx.stream() - the same sycl::queue the surrounding ops submit to, which
     is what orders the extension's kernels against them.
  2. ggml/src/ggml-sycl/fusion.cpp: the RMS_NORM+MUL fusion check GGML_ASSERTs
     F32 rather than declining, so a BF16 rms_norm aborts the process before
     dispatch is ever reached. Those two asserts become a return, which is what
     the three checks immediately below them already do for an unsupported
     mul type.

With the pointer left null this is a no-op, so an unpatched-but-hooked ggml
behaves identically to a stock one.

Usage: scripts/patch_ggml_sycl_ext_hook.py [<llama-src-dir>]
"""

import pathlib
import sys

MARKER = "vla.cpp: SYCL extension hook"

# (relative path, anchor, replacement)
EDITS = [
    (
        "ggml/src/ggml-sycl/ggml-sycl.cpp",
        """static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {
    if (!g_sycl_loaded) return false;
    initialize_sycl_begining();
""",
        """// vla.cpp: SYCL extension hook. Null unless vla::sycl_register_bf16_ops() ran;
// see src/sycl/vla_sycl_bf16.cpp, which holds every kernel behind it. The void*
// is the ggml_backend_sycl_context's sycl::queue*, so the extension submits into
// the same in-order queue and needs no synchronisation of its own.
extern "C" {
typedef bool (*ggml_sycl_ext_forward_t)(struct ggml_tensor * dst, void * stream);
__attribute__((visibility("default"))) ggml_sycl_ext_forward_t ggml_sycl_ext_forward = nullptr;
}

static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {
    if (!g_sycl_loaded) return false;
    initialize_sycl_begining();

    // vla.cpp: SYCL extension hook - first refusal, after the device is up.
    if (ggml_sycl_ext_forward && ggml_sycl_ext_forward(dst, (void *) ctx.stream())) {
        return true;
    }
""",
    ),
    (
        "ggml/src/ggml-sycl/fusion.cpp",
        """        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
""",
        """        // vla.cpp: SYCL extension hook - decline instead of aborting, so a BF16
        // rms_norm falls through to the unfused path (and then to the hook).
        if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32) {
            return false;
        }
""",
    ),
]


def main():
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if not (src / "ggml/src/ggml-sycl/ggml-sycl.cpp").exists():
        raise SystemExit(f"not a llama.cpp source tree: {src}")

    for rel, old, new in EDITS:
        path = src / rel
        text = path.read_text()
        if MARKER in text:
            continue  # idempotent: re-configure over an already-patched tree
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"{path}: anchor found {n} times, expected 1. The pinned llama.cpp "
                f"probably moved; re-check this anchor against the new tag.\n"
                f"---\n{old[:400]}\n---"
            )
        path.write_text(text.replace(old, new))


if __name__ == "__main__":
    main()
