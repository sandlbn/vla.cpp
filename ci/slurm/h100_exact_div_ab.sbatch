#!/bin/bash
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
# =============================================================================
# h100_exact_div_ab.sbatch -- what does a correctly-rounded divide cost?
#
# vla_exact_div fixed a 1 ULP error in the activation scale that --use_fast_math
# had introduced (see docs/backend/sycl.md). The comment written with it claimed
# the cost was "not measurable". That claim was made from the call-site count and
# never checked, and the first CUDA bench after the change read 23.7 ms against a
# recorded 22.3 - so it needs checking, because the epilogue runs that divide
# once per output element, roughly 238M times per request, and an IEEE divide on
# NVIDIA is ~10 cycles against ~4 for the approximate one. That is ~0.9 ms of
# arithmetic, the right order for the gap.
#
# Three builds, same node, back to back, so the only variable is the divide:
#
#   fdiv    __fdiv_rn        correctly rounded, one instruction. Shipping today.
#   fast    a / b            what --use_fast_math does. NOT correct - it is here
#                            only to price the guarantee, never to ship.
#   mark    Markstein FMAs   correctly rounded via reciprocal + Newton +
#                            residual, the same sequence sycl_compat.h uses.
#                            5 cheap ops, and the reciprocal refinement depends
#                            only on the divisor, so it can hoist out of a row.
#
# The point is not to pick the fastest. `fast` will probably win and is
# disqualified regardless; the question is whether `mark` recovers what `fdiv`
# costs while keeping the result bit-identical, which ctest checks per build.
#
# The source is patched in place and restored by an EXIT trap, so a job that
# dies partway does not leave the tree rewritten. Each patch is asserted to have
# matched - silently benching the same binary three times is the one failure mode
# that would look like a clean result.
#
#   sbatch ci/slurm/h100_exact_div_ab.sbatch
#
# Knobs: CUDA_VER, ARCH, REPS, WARMUP, JOBS.
# =============================================================================
#SBATCH --job-name=h100_exact_div_ab
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm_h100_exact_div_ab_%j.out

set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
HDR="$REPO_DIR/src/kernels/bitvla/cuda_compat.h"
BAK="$(mktemp)"
JOBS="${JOBS:-$(nproc)}"
ARCH="${ARCH:-90}"
REPS="${REPS:-10}"
WARMUP="${WARMUP:-3}"
MODEL="${MODEL:-vrfai/bitvla-libero-gguf}"

cp -a "$HDR" "$BAK"
restore() { cp -a "$BAK" "$HDR"; rm -f "$BAK"; echo "[restored $HDR]"; }
trap restore EXIT

echo "NODE: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader

CUDA_VER="${CUDA_VER:-12.9.0}"
for d in "/swtools/cuda_x86_64/$CUDA_VER" /swtools/cuda_x86_64/12.9.0 \
         /swtools/cuda_x86_64/12.6.3 /usr/local/cuda; do
  if [ -x "$d/bin/nvcc" ]; then CUDA_HOME="$d"; break; fi
done
[ -n "$CUDA_HOME" ] || { echo "VERDICT: FAILED no nvcc"; exit 1; }
export CUDA_HOME CUDAToolkit_ROOT="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Rewrite just the body of vla_exact_div. Asserts a match, so a source edit that
# invalidates the pattern stops the job instead of quietly producing three
# identical binaries.
patch_div() {
python3 - "$BAK" "$HDR" "$1" <<'PY'
import re, sys
src, dst, variant = sys.argv[1], sys.argv[2], sys.argv[3]
bodies = {
  "fdiv": "    return __fdiv_rn(a, b);",
  "fast": "    return a / b;",
  "mark": ("    float y = __frcp_rn(b);\n"
           "    y = __fmaf_rn(y, __fmaf_rn(-b, y, 1.0f), y);\n"
           "    const float q = a * y;\n"
           "    return __fmaf_rn(__fmaf_rn(-b, q, a), y, q);"),
}
t = open(src).read()
pat = re.compile(r"(__device__ __forceinline__ float vla_exact_div\(float a, float b\) \{\n)(.*?)(\n\})", re.S)
m = pat.search(t)
if not m:
    sys.exit("ASSERT FAILED: vla_exact_div body not found - pattern is stale")
out = pat.sub(lambda _: m.group(1) + bodies[variant] + m.group(3), t, count=1)
if out == t and variant != "fdiv":
    sys.exit(f"ASSERT FAILED: patching {variant} changed nothing")
open(dst, "w").write(out)
print(f"[patched vla_exact_div -> {variant}]")
PY
}

for V in fdiv fast mark; do
  echo
  echo "############################## $V ##############################"
  patch_div "$V" || { echo "VERDICT: FAILED patch $V"; exit 1; }
  grep -A5 "float vla_exact_div" "$HDR" | head -7

  B="$REPO_DIR/build-cuda-div-$V"
  if [ ! -f "$B/CMakeCache.txt" ]; then
    cmake -B "$B" -S "$REPO_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
        -DVLA_BUILD_TESTS=ON -DVLA_BUILD_SERVERS=OFF >/dev/null 2>&1 \
      || { echo "VERDICT: FAILED configure $V"; exit 1; }
  fi
  cmake --build "$B" -j"$JOBS" 2>&1 | tail -3 || { echo "VERDICT: FAILED build $V"; exit 1; }

  # The correctness gate. `fast` is expected to fail here; that failure is the
  # evidence the guarantee is doing something, not a problem with the job.
  echo "--- bitvla_gemm_gpu ($V) ---"
  "$B/tests/test_bitvla_gemm_gpu" 2>&1 | tail -13
  echo "GEMM_RC[$V]: $?"

  echo "--- bench ($V) ---"
  "$B/vla-bench" -hf "$MODEL" --images 1 --size 224 \
      --warmup "$WARMUP" --reps "$REPS" --markdown 2>&1 | tail -3
done

echo
echo "DONE"
