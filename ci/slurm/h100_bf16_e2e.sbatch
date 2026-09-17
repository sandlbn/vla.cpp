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
# h100_bf16_e2e.sbatch -- the CUDA half of the BF16 precision claim.
#
# ci/slurm/bmg_bf16_e2e.sbatch measures how far --act-dtype bf16 moves the action
# chunk on Battlemage. On its own that number has no meaning: BF16 activations
# are a lossy choice on any backend, so some deviation is the point of the
# feature rather than a defect in the port.
#
# This job runs the identical two arms through the CUDA BF16 kernels
# (src/cuda/vla_cuda_bf16.cu), which are the implementation the SYCL ones were
# written from. The gate the port is actually held to is the comparison of the
# two deviations: SYCL-vs-F32 must be no worse than CUDA-vs-F32 on the same
# checkpoint and the same fixed inputs. Both sides are scored by the same
# ci/compare_actions.py so the comparison is of the numbers, not of two scripts.
#
# That comparison is only honest with the cpu arm below, because the two
# backends' F32 baselines are not the same precision. ggml-cuda sets
# CUBLAS_TF32_TENSOR_OP_MATH on every handle it creates (ggml-cuda/common.cuh),
# unconditionally and with no env guard, so "F32" here means TF32 tensor cores at
# a 10-bit mantissa. ggml-sycl's matching f16 fpmath hint is behind
# GGML_SYCL_F16, which is off in that build, so its F32 is true fp32. Scoring
# each backend's bf16 arm against its own f32 arm therefore measures two
# different things. The cpu arm is the common reference that fixes it.
#
#   sbatch ci/slurm/h100_bf16_e2e.sbatch
#   MODEL=vrfai/pi0-libero-finetuned-v044-gguf sbatch ci/slurm/h100_bf16_e2e.sbatch
#
# Knobs: BUILD_DIR, MODEL, IMG_SIZE, TOL, ITERS, ARCH, CUDA_VER, JOBS.
# =============================================================================
#SBATCH --job-name=h100_bf16_e2e
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm_h100_bf16_e2e_%j.out

set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-cuda}"
JOBS="${JOBS:-$(nproc)}"
ARCH="${ARCH:-90}"
MODEL="${MODEL:-vrfai/evo1-libero-gguf}"
TOL="${TOL:-0.06}"   # same floor-derived value as the Battlemage job; see its header
ITERS="${ITERS:-10}"
IMG_SIZE="${IMG_SIZE:-448}"
OUT="${SLURM_SUBMIT_DIR:-$PWD}/h100_bf16_$$"

echo "NODE:  $(hostname)"
echo "MODEL: $MODEL"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

# The per-arm scratch files are removed on every exit path, not just the happy
# one: an arm that fails leaves them behind in the submit directory otherwise.
trap 'rm -f "$OUT".*.txt "$OUT".*.err' EXIT

# --- toolchain --------------------------------------------------------------
CUDA_VER="${CUDA_VER:-12.9.0}"
for d in "/swtools/cuda_x86_64/$CUDA_VER" /swtools/cuda_x86_64/12.9.0 \
         /swtools/cuda_x86_64/12.6.3 /usr/local/cuda; do
  if [ -x "$d/bin/nvcc" ]; then CUDA_HOME="$d"; break; fi
done
[ -n "$CUDA_HOME" ] || fail "TOOLCHAIN no nvcc under /swtools/cuda_x86_64"
export CUDA_HOME CUDAToolkit_ROOT="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# --- build ------------------------------------------------------------------
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  cmake -B "$BUILD_DIR" -S "$REPO_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      -DVLA_BUILD_TESTS=ON \
      -DVLA_BUILD_SERVERS=OFF 2>&1 || fail "CONFIGURE"
fi
cmake --build "$BUILD_DIR" -j"$JOBS" 2>&1 | tail -5 || fail "BUILD"
echo "BUILD: ok"

# --- the CUDA unit gate -----------------------------------------------------
# test_bf16_cuda_ops is the twin of test_bf16_sycl_ops: same chain, same
# constants, same tolerance.
if [ -x "$BUILD_DIR/tests/test_bf16_cuda_ops" ]; then
  echo
  echo "=== test_bf16_cuda_ops ==="
  "$BUILD_DIR/tests/test_bf16_cuda_ops" 2>&1 | tail -4
  echo "OPS_RC: ${PIPESTATUS[0]}"
fi

CKPT="$(ls "${VLA_CACHE:-$HOME/.cache/vla}/$MODEL"/*.gguf 2>/dev/null | head -1)"
[ -n "$CKPT" ] || fail "no .gguf for $MODEL under ${VLA_CACHE:-$HOME/.cache/vla}"
echo "CKPT:  $CKPT"

# --- the two arms -----------------------------------------------------------
run_arm() {   # $1 = tag, $2.. = extra args; VLA_DEVICE/ITERS may be preset
  local tag="$1"; shift
  echo
  echo "=== predict_check [$tag] ==="
  # The empty second argument is load-bearing: predict_check reads argv[2] as an
  # mmproj path whenever it is non-empty and not a flag.
  VLA_IMG_SIZE="$IMG_SIZE" VLA_BENCH_ITERS="$ITERS" \
      "$BUILD_DIR/tests/vla_predict_check" "$CKPT" "" 1 "$@" \
      > "$OUT.$tag.txt" 2> "$OUT.$tag.err"
  local rc=$?
  grep -hE "activations =|BF16_ACT|predict\(\) over|last split|error|failed" \
      "$OUT.$tag.txt" "$OUT.$tag.err" | sort -u | head -6
  echo "RC[$tag]: $rc"
  return $rc
}

# Each dtype twice: the engine is not bit-reproducible run to run, and a
# dtype comparison below that floor is measuring the scheduler, not the kernels.
run_arm f32                     || fail "RUN f32"
run_arm f32b                    || fail "RUN f32b"
run_arm bf16 --act-dtype bf16   || fail "RUN bf16"
run_arm bf16b --act-dtype bf16  || fail "RUN bf16b"

# The independent reference; see the TF32 note in the header for why the f32 arm
# above cannot serve as one. ggml_backend_cuda_init rejects an out-of-range
# device and returns nullptr (ggml-cuda.cu), and src/backend.h then falls back to
# CPU -- so the same binary yields an fp32 reference with no cuBLAS math mode in
# it at all. Slow, hence one iteration and no timing loop.
if [ "${CPU_REF:-1}" = "1" ]; then
  # Subshell + export: run_arm passes VLA_IMG_SIZE/VLA_BENCH_ITERS explicitly but
  # inherits VLA_DEVICE, and the subshell keeps both off any later arm.
  ( export VLA_DEVICE=99; ITERS=0; run_arm cpu ) || fail "RUN cpu"
fi

grep -qh "activations = BF16" "$OUT.bf16.txt" "$OUT.bf16.err" ||
    fail "BF16 ARM DID NOT ENGAGE"

# --- compare ----------------------------------------------------------------
DIMS="$(sed -n 's/.*max_action=\([0-9]*\).*/\1/p' "$OUT.f32.err" | head -1)"

# The controls, which have to be read before the gate below means anything:
# predict_check builds its images, tokens, state and noise from fixed formulas,
# so whatever a repeat run fails to reproduce is the engine. The f32 pair assigns
# blame -- it never enters src/cuda/vla_cuda_bf16.cu.
echo
echo "=== f32 vs f32 (run-to-run control, stock ggml-cuda) ==="
python3 "$REPO_DIR/ci/compare_actions.py" "$OUT.f32.txt" "$OUT.f32b.txt" \
    --tol 1.0 --dims "${DIMS:-0}" --label-a run1 --label-b run2 | head -2
echo
echo "=== bf16 vs bf16 (run-to-run control, vla kernels) ==="
python3 "$REPO_DIR/ci/compare_actions.py" "$OUT.bf16.txt" "$OUT.bf16b.txt" \
    --tol 1.0 --dims "${DIMS:-0}" --label-a run1 --label-b run2 | head -2

echo
echo "=== f32 vs bf16 action chunk (CUDA) ==="
python3 "$REPO_DIR/ci/compare_actions.py" "$OUT.f32.txt" "$OUT.bf16.txt" \
    --tol "$TOL" --dims "${DIMS:-0}" --label-a f32 --label-b bf16
cmp_rc=$?
echo "COMPARE_RC: $cmp_rc"

# Against the CPU fp32 reference. This is the pair that says which arm is closer
# to the truth rather than merely how far apart the two are -- and, on this
# backend specifically, how much of the f32 arm's own error is TF32.
if [ -f "$OUT.cpu.txt" ]; then
  echo
  echo "=== cpu-f32 vs gpu-f32/TF32 (reference) ==="
  python3 "$REPO_DIR/ci/compare_actions.py" "$OUT.cpu.txt" "$OUT.f32.txt" \
      --tol 1.0 --dims "${DIMS:-0}" --label-a cpu --label-b gpuf32 | head -2
  echo
  echo "=== cpu-f32 vs gpu-bf16 (reference) ==="
  python3 "$REPO_DIR/ci/compare_actions.py" "$OUT.cpu.txt" "$OUT.bf16.txt" \
      --tol 1.0 --dims "${DIMS:-0}" --label-a cpu --label-b gpubf16 | head -2
fi

ART="$REPO_DIR/ci/reference/bf16"
mkdir -p "$ART"
tag="$(basename "$MODEL")"
cp "$OUT.f32.txt"  "$ART/$tag-cuda-f32.txt"
cp "$OUT.bf16.txt" "$ART/$tag-cuda-bf16.txt"
[ -f "$OUT.cpu.txt" ] && cp "$OUT.cpu.txt" "$ART/$tag-cuda-cpu-f32.txt"

# --- cross-backend ----------------------------------------------------------
# Only if the Battlemage job has already deposited its side.
if [ -f "$ART/$tag-sycl-bf16.txt" ]; then
  echo
  echo "=== cuda-bf16 vs sycl-bf16 (cross-backend) ==="
  python3 "$REPO_DIR/ci/compare_actions.py" \
      "$ART/$tag-cuda-bf16.txt" "$ART/$tag-sycl-bf16.txt" \
      --tol "$TOL" --dims "${DIMS:-0}" --label-a cuda --label-b sycl
  echo "CROSS_RC: $?"
else
  echo "no sycl artefact at $ART/$tag-sycl-bf16.txt -- run bmg_bf16_e2e.sbatch first"
fi

[ "$cmp_rc" = "0" ] && echo "VERDICT: OK" || echo "VERDICT: FAILED COMPARE"
echo "DONE"
