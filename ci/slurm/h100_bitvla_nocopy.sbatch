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
# h100_bitvla_nocopy.sbatch -- the CUDA half of the fc1 restride change.
#
# The restride was removed because of a SYCL pathology: ext_oneapi_memcpy2d has
# no native 2D copy behind it on Level Zero and fell back to a byte-granularity
# kernel worth 4.4 ms of a 27.2 ms request. CUDA does not have that problem -
# cudaMemcpy2D runs on the copy engine - but the driver is shared, so CUDA takes
# the change whether or not it benefits, and that is exactly why it needs
# checking here. A shared driver means a SYCL-motivated edit can only be called
# safe once the other backend has compiled and run it.
#
# Two things to establish, in order of importance:
#   1. it still builds, and act_quant_kernel's new K_in/K_out signature did not
#      break the CUDA instantiation or the tests that use the entry point;
#   2. it is still bit-identical on CUDA. test_bitvla_ops_gpu carries the A/B
#      inside it - old path (memset, memcpy2d, quantise at K_out) against the
#      new one - so the same source proves the same property on both backends.
#
# The bench is reported but not gated: the expectation on CUDA is "no worse",
# since the copy it removes was nearly free here to begin with.
#
#   sbatch ci/slurm/h100_bitvla_nocopy.sbatch
#
# Knobs: BUILD_DIR, CUDA_VER, ARCH, REPS, WARMUP, JOBS.
# =============================================================================
#SBATCH --job-name=h100_bitvla_nocopy
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=slurm_h100_bitvla_nocopy_%j.out

set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-cuda}"
JOBS="${JOBS:-$(nproc)}"
ARCH="${ARCH:-90}"
REPS="${REPS:-10}"
WARMUP="${WARMUP:-3}"

echo "NODE: $(hostname)"
fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

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
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

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
echo
echo "=== build ==="
cmake --build "$BUILD_DIR" -j"$JOBS" 2>&1 | tail -20 || fail "BUILD"
echo "BUILD_RC: ok"

# --- the bit-identity A/B, on CUDA ------------------------------------------
echo
echo "=== test_bitvla_ops_gpu ==="
"$BUILD_DIR/tests/test_bitvla_ops_gpu" 2>&1
ops_rc=$?
echo "OPS_RC: $ops_rc"

echo
echo "=== ctest ==="
# pipefail is on, so the subshell's status is ctest's whenever ctest is the one
# that failed - tail cannot mask it.
( cd "$BUILD_DIR" && ctest --output-on-failure 2>&1 | tail -25 )
ctest_rc=$?
echo "CTEST_RC: $ctest_rc"

# --- bench, reported not gated ----------------------------------------------
echo
echo "=== vla-bench bitvla ==="
"$BUILD_DIR/vla-bench" -hf vrfai/bitvla-libero-gguf --images 1 --size 224 \
    --warmup "$WARMUP" --reps "$REPS" --markdown 2>&1 | tail -3

echo
if [ "$ops_rc" = "0" ] && [ "$ctest_rc" = "0" ]; then
  echo "VERDICT: OK  CUDA builds, all tests pass, act_quant stays bit-identical"
else
  echo "VERDICT: FAILED  ops_rc=$ops_rc ctest_rc=$ctest_rc"
fi
echo "DONE"
