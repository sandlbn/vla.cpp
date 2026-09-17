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
# bmg_bitvla_profile.sbatch -- attribute a BitVLA request on Battlemage.
#
# The state of the port is that the ternary GEMM is 9.5 ms of a 27.2 ms request,
# about 35%. Driving all seven dispatched shapes to the XMX peak would save
# 3.3 ms - 12% of the request - and nobody reaches the peak. The other ~65% is
# RMS norm, RoPE, softmax, the transposes, attention, the fp32 action head and
# whatever host-side cost sits between them, and none of it has ever been
# measured. Writing another GEMM kernel before measuring that is optimising the
# part we already know is not the problem.
#
# So this job measures before anything gets written. unitrace ships in the
# vendor GPU runtime (/swtools/intel-gpu/latest/unitrace) and reports per-kernel
# device time over Level Zero, with shapes under -v. That attributes the device
# side by kernel name without a line of instrumentation in our own sources.
#
# Three arms, and the first two are what make the third readable:
#
#   1. bench, no profiler       the wall clock the attribution has to add up to,
#                               and the number the docs already quote.
#   2. bench, under unitrace    the same run, so arm 1 vs arm 2 is the profiler's
#                               own overhead. A kernel breakdown taken from a run
#                               that is 3x slower than the real one describes a
#                               workload we do not ship.
#   3. --host-timing            total device time never equals wall time. What is
#                               left is submit cost, waits and host work, which
#                               is the "per-call overheads between them" half of
#                               the question and is invisible to -d.
#
# Reports are written under outputs/bitvla_profile_$SLURM_JOB_ID/ as well as to
# stdout, because the useful next step is to bucket the kernel table by op
# family and that is done off the saved file rather than by rerunning the job.
#
#   sbatch ci/slurm/bmg_bitvla_profile.sbatch
#
# Knobs: BUILD_DIR, MODEL, SIZE, REPS, WARMUP, PROF_REPS.
# =============================================================================
#SBATCH --job-name=bmg_bitvla_profile
#SBATCH --partition=b70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm_bmg_bitvla_profile_%j.out

# No `set -u`: the vendor's intel_gpu_vars.sh and oneAPI's setvars.sh both read
# unset variables, so nounset halts before the toolchain is on PATH.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-sycl}"
OUT_DIR="$REPO_DIR/outputs/bitvla_profile_${SLURM_JOB_ID:-local}"
UNITRACE="${UNITRACE:-/swtools/intel-gpu/latest/unitrace/bin/unitrace}"

MODEL="${MODEL:-vrfai/bitvla-libero-gguf}"
SIZE="${SIZE:-224}"
REPS="${REPS:-10}"
WARMUP="${WARMUP:-3}"
# Fewer reps under the profiler: unitrace writes a record per kernel launch and
# BitVLA launches hundreds per request, so the trace grows fast and the extra
# reps buy no resolution that five do not already give.
PROF_REPS="${PROF_REPS:-5}"

mkdir -p "$OUT_DIR"

echo "NODE: $(hostname)"
echo "BUILD_DIR: $BUILD_DIR"
echo "OUT_DIR: $OUT_DIR"

source /swtools/intel-gpu/latest/intel_gpu_vars.sh
source /swtools/intel/setvars.sh --force >/dev/null 2>&1

[ -x "$BUILD_DIR/vla-bench" ] || { echo "VERDICT: FAILED no vla-bench in $BUILD_DIR"; echo "DONE"; exit 1; }
[ -x "$UNITRACE" ] || { echo "VERDICT: FAILED no unitrace at $UNITRACE"; echo "DONE"; exit 1; }

echo "GPU: $(sycl-ls 2>/dev/null | grep -m1 level_zero)"
echo "UNITRACE: $("$UNITRACE" --version 2>&1 | head -1)"

bench_args=( -hf "$MODEL" --images 1 --size "$SIZE" --markdown )

# --- arm 1: the unprofiled number -------------------------------------------
echo
echo "=== [1/3] bench, no profiler ==="
"$BUILD_DIR/vla-bench" "${bench_args[@]}" --warmup "$WARMUP" --reps "$REPS" \
    2>&1 | tee "$OUT_DIR/bench_plain.txt"
echo "RC[plain]: ${PIPESTATUS[0]}"

# --- arm 2: device timing, per kernel ---------------------------------------
# -d  per-kernel device execution time, already aggregated and sorted
# -v  kernel shapes, so the seven GEMM dispatches stay distinguishable from each
#     other instead of collapsing into one row per template instantiation
echo
echo "=== [2/3] unitrace --device-timing ==="
"$UNITRACE" -d -v -o "$OUT_DIR/device" \
    "$BUILD_DIR/vla-bench" "${bench_args[@]}" --warmup 1 --reps "$PROF_REPS" \
    2>&1 | tee "$OUT_DIR/bench_device.txt"
echo "RC[device]: ${PIPESTATUS[0]}"

# --- arm 3: host API timing -------------------------------------------------
# The gap between summed device time and wall time is the whole question for a
# model that launches this many small kernels, and -d cannot see it.
echo
echo "=== [3/3] unitrace --host-timing ==="
"$UNITRACE" -h -o "$OUT_DIR/host" \
    "$BUILD_DIR/vla-bench" "${bench_args[@]}" --warmup 1 --reps "$PROF_REPS" \
    2>&1 | tee "$OUT_DIR/bench_host.txt"
echo "RC[host]: ${PIPESTATUS[0]}"

# --- collect ----------------------------------------------------------------
# unitrace appends the pid to -o, so glob rather than name the files.
echo
echo "=== saved reports ==="
ls -la "$OUT_DIR"

for f in "$OUT_DIR"/device.*; do
  [ -f "$f" ] || continue
  echo
  echo "########## $f ##########"
  cat "$f"
done

for f in "$OUT_DIR"/host.*; do
  [ -f "$f" ] || continue
  echo
  echo "########## $f (top 40) ##########"
  head -50 "$f"
done

echo
echo "DONE"
