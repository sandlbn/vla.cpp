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
# bmg_run.sbatch -- run the built binaries on a Battlemage node.
#
# Stage 0 of the port uses this to establish two baselines before any SYCL
# kernel exists:
#   * that the stock ggml-sycl path really works on Xe2 (the backend docs were
#     written against an Arc A380, which is Xe-HPG and a different part),
#   * BitVLA's own numbers, which on this build come off the CPU because the
#     ternary kernels are CUDA-only -- that is the reference the port is scored
#     against.
#
#   sbatch ci/slurm/bmg_run.sbatch
#   MODELS="vrfai/smolvla-libero-gguf" sbatch ci/slurm/bmg_run.sbatch
#
# Knobs: BUILD_DIR, MODELS (space-separated -hf repos), REPS, WARMUP, SIZE, CTEST=0.
# =============================================================================
#SBATCH --job-name=bmg_run
#SBATCH --partition=b70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm_bmg_run_%j.out

# No `set -u`: the vendor's intel_gpu_vars.sh and oneAPI's setvars.sh both read
# unset variables, so nounset halts before the toolchain is on PATH.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-sycl}"

echo "NODE: $(hostname)"
echo "BUILD_DIR: $BUILD_DIR"

source /swtools/intel-gpu/latest/intel_gpu_vars.sh
source /swtools/intel/setvars.sh --force >/dev/null 2>&1

[ -x "$BUILD_DIR/vla-bench" ] || { echo "VERDICT: FAILED no vla-bench in $BUILD_DIR"; echo "DONE"; exit 1; }

echo "GPU: $(sycl-ls 2>/dev/null | grep -m1 level_zero)"

# --- unit tests -------------------------------------------------------------
# Cheap, and they are the only thing here that fails loudly on a bad refactor of
# the shared headers rather than on a missing checkpoint.
if [ "${CTEST:-1}" = "1" ]; then
  echo
  echo "=== ctest ==="
  ( cd "$BUILD_DIR" && ctest --output-on-failure 2>&1 | tail -25 )
  echo "CTEST_RC: ${PIPESTATUS[0]}"
fi

# --- benchmarks -------------------------------------------------------------
# -hf pulls the checkpoint on first use and caches it under $VLA_CACHE, so the
# first job of a session is slower than the rest.
MODELS="${MODELS:-vrfai/smolvla-libero-gguf vrfai/bitvla-libero-gguf}"
REPS="${REPS:-10}"
WARMUP="${WARMUP:-3}"

for repo in $MODELS; do
  echo
  echo "=== bench $repo ==="
  size="${SIZE:-224}"
  case "$repo" in *smolvla*) size="${SIZE:-512}";; esac
  "$BUILD_DIR/vla-bench" -hf "$repo" --images 1 --size "$size" \
      --warmup "$WARMUP" --reps "$REPS" --markdown 2>&1
  echo "BENCH_RC[$repo]: $?"
done

echo
echo "DONE"
