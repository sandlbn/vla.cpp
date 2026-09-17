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
# h100_bf16_bench.sbatch -- the CUDA twin of ci/slurm/bmg_bf16_bench.sbatch.
#
# Same two architectures, same sizes, same rep counts, same p50 statistic, so the
# speedup columns can be put side by side. What it cannot do is put the *latency*
# columns side by side: an H100 and an Arc Pro B70 are different parts at
# different power budgets, and a ms-for-ms table would be read as a comparison
# nobody ran. The number this job exists to produce is how much --act-dtype bf16
# is worth on each backend relative to that backend's own F32.
#
# One asymmetry to keep in mind when reading the two speedups: ggml-cuda sets
# CUBLAS_TF32_TENSOR_OP_MATH on every handle (ggml-cuda/common.cuh),
# unconditionally, so the F32 denominator here is already tensor-core work at a
# 10-bit mantissa. ggml-sycl's equivalent hint is behind GGML_SYCL_F16, off in
# that build, so its denominator is true fp32. CUDA's bf16 speedup is therefore
# measured against a stronger baseline and will read lower for that reason alone.
#
#   sbatch ci/slurm/h100_bf16_bench.sbatch
#   MODELS="vrfai/evo1-libero-gguf" REPS=40 sbatch ci/slurm/h100_bf16_bench.sbatch
#
# Knobs: BUILD_DIR, MODELS, REPS, WARMUP, IMAGES, SIZE, ARCH, CUDA_VER, JOBS.
# =============================================================================
#SBATCH --job-name=h100_bf16_bench
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=slurm_h100_bf16_bench_%j.out

set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-cuda}"
MODELS="${MODELS:-vrfai/evo1-libero-gguf vrfai/pi0-libero-finetuned-v044-gguf}"
REPS="${REPS:-20}"
WARMUP="${WARMUP:-3}"
IMAGES="${IMAGES:-1}"
JOBS="${JOBS:-$(nproc)}"
ARCH="${ARCH:-90}"
OUT="${SLURM_SUBMIT_DIR:-$PWD}/h100_bench_$$"

echo "NODE: $(hostname)"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }
trap 'rm -f "$OUT".*.txt' EXIT

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
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader

# --- build ------------------------------------------------------------------
# Usually a no-op: h100_bf16_e2e.sbatch populates the same build directory.
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  cmake -B "$BUILD_DIR" -S "$REPO_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      -DVLA_BUILD_TESTS=ON \
      -DVLA_BUILD_SERVERS=OFF 2>&1 || fail "CONFIGURE"
fi
# vla-bench sits outside the VLA_BUILD_SERVERS guard in CMakeLists.txt, so it is
# present whether or not the e2e job configured this tree with the servers off.
cmake --build "$BUILD_DIR" -j"$JOBS" 2>&1 | tail -5 || fail "BUILD"
[ -x "$BUILD_DIR/vla-bench" ] || fail "no vla-bench in $BUILD_DIR"

# --- arms -------------------------------------------------------------------
run_bench() {   # $1 = tag, $2 = ckpt, $3 = label, $4 = size, $5.. = extra args
  local tag="$1" ckpt="$2" label="$3" size="$4"; shift 4
  echo
  echo "=== vla-bench [$tag] ==="
  "$BUILD_DIR/vla-bench" --ckpt "$ckpt" --label "$label" \
      --images "$IMAGES" --size "$size" \
      --warmup "$WARMUP" --reps "$REPS" "$@" 2>&1 | tee "$OUT.$tag.txt"
  local rc=${PIPESTATUS[0]}
  echo "RC[$tag]: $rc"
  return $rc
}

for repo in $MODELS; do
  tag="$(basename "$repo")"

  ckpt="$(ls "${VLA_CACHE:-$HOME/.cache/vla}/$repo"/*.gguf 2>/dev/null | head -1)"
  [ -n "$ckpt" ] || fail "no .gguf for $repo under ${VLA_CACHE:-$HOME/.cache/vla} -- pre-pull it"

  # Must match the Battlemage job exactly or the two speedup columns are not
  # measuring the same graph: evo1's SigLIP tower is 448, pi0 is 224.
  size="${SIZE:-224}"
  case "$repo" in *evo1*) size="${SIZE:-448}";; esac
  echo
  echo "##### $repo (size $size, $IMAGES image, $REPS reps) #####"

  run_bench "$tag-f32"  "$ckpt" "$tag-f32"  "$size"                  || fail "BENCH $tag f32"
  run_bench "$tag-bf16" "$ckpt" "$tag-bf16" "$size" --act-dtype bf16 || fail "BENCH $tag bf16"

  # Without this a model that ignores the flag reports a flattering 1.00x while
  # benchmarking the F32 graph against itself.
  grep -q "activations = BF16" "$OUT.$tag-bf16.txt" ||
      fail "BF16 ARM DID NOT ENGAGE for $repo"
done

# --- summary ----------------------------------------------------------------
# p50, not min: min is the one rep that dodged every scheduling hiccup.
echo
echo "=== summary (p50 ms, lower is better) ==="
printf '%-34s %10s %10s %8s\n' model f32 bf16 speedup
for repo in $MODELS; do
  tag="$(basename "$repo")"
  p50() { sed -n 's/.*p50[= ]*\([0-9.]*\).*/\1/p' "$1" | head -1; }
  a="$(p50 "$OUT.$tag-f32.txt")"
  b="$(p50 "$OUT.$tag-bf16.txt")"
  if [ -n "$a" ] && [ -n "$b" ]; then
    printf '%-34s %10s %10s %8s\n' "$tag" "$a" "$b" \
        "$(awk -v x="$a" -v y="$b" 'BEGIN{ if (y>0) printf "%.3fx", x/y; else print "n/a" }')"
  else
    # Not fatal: the per-arm output above is the record.
    printf '%-34s %10s %10s %8s\n' "$tag" "${a:-?}" "${b:-?}" "n/a"
  fi
done

echo
echo "VERDICT: OK"
echo "DONE"
