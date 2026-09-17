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
# cuda_reference.sbatch -- build the CUDA path on h100 and capture the numbers
# the Intel port is scored against.
#
# BitVLA publishes only int2-packed GGUFs, and model_load refuses those without
# hand-written GPU kernels, so there is no CPU run to compare against: CUDA *is*
# the reference. Two things come out of this job.
#
#   * ci/reference/bitvla-<task>-cuda.txt -- predict_check's action vector.
#     predict_check synthesises its images, tokens, state and noise from fixed
#     formulas, so the same binary on Battlemage must produce the same numbers
#     from nothing but the checkpoint.
#   * ci/reference/l0/ -- VLA_BITVLA_DUMP_L0 tensor dumps, one file per op in
#     layer 0. These are what the per-op SYCL unit tests compare against, which
#     is finer-grained than an end-to-end action diff and localises a bad port
#     to a single kernel.
#
# It doubles as the "CUDA build still works" gate for the device.h refactor:
# the headers now say vla_bf16/vla_stream, and only nvcc can prove the casts.
#
#   sbatch ci/slurm/cuda_reference.sbatch
#
# Knobs: BUILD_DIR, CLEAN=1, TASK (libero_object), ARCH (90), JOBS.
# =============================================================================
#SBATCH --job-name=cuda_ref
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=slurm_cuda_ref_%j.out

set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-cuda}"
JOBS="${JOBS:-$(nproc)}"
TASK="${TASK:-libero_object}"
ARCH="${ARCH:-90}"
REF_DIR="$REPO_DIR/ci/reference"

echo "NODE: $(hostname)"
echo "BUILD_DIR: $BUILD_DIR"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

# --- toolchain --------------------------------------------------------------
# The cluster keeps its toolkits under /swtools, not /usr/local; CUDA_VER pins a
# 12.x by default because that is what the vendored llama.cpp tag is tested
# against, and `latest` moves.
CUDA_VER="${CUDA_VER:-12.9.0}"
for d in "/swtools/cuda_x86_64/$CUDA_VER" /swtools/cuda_x86_64/12.9.0 \
         /swtools/cuda_x86_64/12.6.3 /usr/local/cuda; do
  if [ -x "$d/bin/nvcc" ]; then CUDA_HOME="$d"; break; fi
done
[ -n "$CUDA_HOME" ] || fail "TOOLCHAIN no nvcc under /swtools/cuda_x86_64 or /usr/local/cuda"
export CUDA_HOME CUDAToolkit_ROOT="$CUDA_HOME"
echo "CUDA_HOME: $CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
echo "NVCC: $(nvcc --version | tail -2 | head -1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# --- build ------------------------------------------------------------------
[ "${CLEAN:-0}" = "1" ] && rm -rf "$BUILD_DIR"
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  cmake -B "$BUILD_DIR" -S "$REPO_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      -DVLA_BUILD_TESTS=ON \
      -DVLA_BUILD_SERVERS=OFF 2>&1 || fail "CONFIGURE"
fi
cmake --build "$BUILD_DIR" -j"$JOBS" 2>&1 || fail "BUILD"
echo "BUILD: ok"

# --- the ternary GEMM's own A/B harness -------------------------------------
# Holds the narrow and wide CUDA tilings to bit-identity over every dispatched
# shape. The SYCL kernels get held to the same standard against these shapes.
if [ -x "$BUILD_DIR/tests/bitvla_gemm_check" ]; then
  echo
  echo "=== bitvla_gemm_check ==="
  "$BUILD_DIR/tests/bitvla_gemm_check" 2>&1 | tail -30
  echo "GEMM_CHECK_RC: ${PIPESTATUS[0]}"
fi

# --- reference actions + layer-0 tensors ------------------------------------
CKPT="$HOME/.cache/vla/vrfai/bitvla-libero-gguf/$TASK/bitvla-${TASK//_/-}.gguf"
if [ ! -f "$CKPT" ]; then
  # The cache layout is <task>/bitvla-<task-with-dashes>.gguf; find it rather
  # than guessing if the naming ever changes.
  CKPT=$(find "$HOME/.cache/vla/vrfai/bitvla-libero-gguf" -name '*.gguf' | sort | head -1)
fi
[ -f "$CKPT" ] || fail "no BitVLA checkpoint in the cache -- run ci/slurm/bmg_run.sbatch once to populate it"
echo "CKPT: $CKPT"

mkdir -p "$REF_DIR/l0"
echo
echo "=== predict_check (reference actions) ==="
"$BUILD_DIR/tests/vla_predict_check" "$CKPT" "" 1 > "$REF_DIR/bitvla-${TASK}-cuda.txt" 2> "$REF_DIR/bitvla-${TASK}-cuda.log"
rc=$?
echo "PREDICT_RC: $rc"
head -12 "$REF_DIR/bitvla-${TASK}-cuda.txt"
tail -5 "$REF_DIR/bitvla-${TASK}-cuda.log"

echo
echo "=== layer-0 op dumps ==="
VLA_BITVLA_DUMP_L0="$REF_DIR/l0" \
  "$BUILD_DIR/tests/vla_predict_check" "$CKPT" "" 1 >/dev/null 2>&1
ls -la "$REF_DIR/l0" | tail -15

# --- perf baseline ----------------------------------------------------------
echo
echo "=== bench (CUDA baseline) ==="
"$BUILD_DIR/vla-bench" -hf vrfai/bitvla-libero-gguf --images 1 --size 224 \
    --warmup 3 --reps 10 --markdown 2>&1 | tail -8

echo
echo "VERDICT: OK"
echo "DONE"
