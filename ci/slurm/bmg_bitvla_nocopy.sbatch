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
# bmg_bitvla_nocopy.sbatch -- A/B the ViT fc1 restride removal.
#
# What changed: the ViT's fc1 output was restrided into a wider zero-padded
# buffer by vla_dev_memcpy2d_d2d so fc2 could consume it at K=ffn_pad. On SYCL
# that is ext_oneapi_memcpy2d, Level Zero has no native 2D copy behind it, and
# the byte-granularity fallback kernel moved 2.2 MB per block at 26 GB/s - 6% of
# this part's bandwidth. Job 371998 measured it at 4.41 ms of a 27.2 ms request,
# 18% of device time and the largest single non-GEMM item. The quantiser that
# reads the buffer next now does the widening itself, so the copy and the padded
# bf16 buffer are both gone. CUDA never saw this cost (cudaMemcpy2D runs on the
# copy engine) but takes the same change, because the driver is shared.
#
# The claim is bit-identical, not "close", so the gate has to be an exact A/B
# against the same binary before the change - not against the stored CUDA
# reference, which differs for unrelated reasons and would hide a small drift.
# Only libvla_core.so changes (the ggml libraries are untouched), so a copy of
# that plus the two executables is a complete "before" and costs no rebuild.
#
# Order matters: snapshot before building, or the thing being compared against
# is the thing under test.
#
#   sbatch ci/slurm/bmg_bitvla_nocopy.sbatch
#
# Knobs: BUILD_DIR, MODEL, REPS, WARMUP, PROF_REPS, SKIP_PROFILE=1.
# =============================================================================
#SBATCH --job-name=bmg_bitvla_nocopy
#SBATCH --partition=b70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=slurm_bmg_bitvla_nocopy_%j.out

# No `set -u`: the vendor's intel_gpu_vars.sh and oneAPI's setvars.sh both read
# unset variables, so nounset halts before the toolchain is on PATH.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-sycl}"
OUT="$REPO_DIR/outputs/bitvla_nocopy_${SLURM_JOB_ID:-local}"
BEFORE="$OUT/before"
UNITRACE="${UNITRACE:-/swtools/intel-gpu/latest/unitrace/bin/unitrace}"

MODEL="${MODEL:-vrfai/bitvla-libero-gguf}"
REPS="${REPS:-10}"
WARMUP="${WARMUP:-3}"
PROF_REPS="${PROF_REPS:-12}"

mkdir -p "$BEFORE"
echo "NODE: $(hostname)"
echo "OUT:  $OUT"

# --- snapshot "before", before touching anything ----------------------------
have_before=1
for f in vla-bench libvla_core.so tests/vla_predict_check; do
  if [ -e "$BUILD_DIR/$f" ]; then
    cp -a "$BUILD_DIR/$f" "$BEFORE/$(basename "$f")"
  else
    echo "WARN: no $BUILD_DIR/$f to snapshot - A/B will be skipped"
    have_before=0
  fi
done

source /swtools/intel-gpu/latest/intel_gpu_vars.sh
source /swtools/intel/setvars.sh --force >/dev/null 2>&1
echo "GPU: $(sycl-ls 2>/dev/null | grep -m1 level_zero)"

# --- build ------------------------------------------------------------------
echo
echo "=== build ==="
cmake --build "$BUILD_DIR" -j "${JOBS:-$(nproc)}" 2>&1 | tail -25
build_rc=${PIPESTATUS[0]}
echo "BUILD_RC: $build_rc"
[ "$build_rc" = "0" ] || { echo "VERDICT: FAILED build"; echo "DONE"; exit 1; }

# --- unit tests -------------------------------------------------------------
echo
echo "=== ctest ==="
( cd "$BUILD_DIR" && ctest --output-on-failure 2>&1 | tail -30 )
echo "CTEST_RC: ${PIPESTATUS[0]}"

# --- the numbers ------------------------------------------------------------
# vla_predict_check prints one float per line, so compare_actions.py can diff two
# runs directly. The checkpoint resolves through the same -hf cache the bench
# uses; pass the resolved path once so both binaries read the same file.
# Globbed, not spelled out: -hf lays the repo down under a per-suite directory
# (.../bitvla-libero-gguf/libero_object/bitvla-libero-object.gguf) and hardcoding
# that shape silently skips the A/B when it does not match, which is the one
# check in this job that cannot be re-derived later.
if [ -z "$CKPT" ]; then
  CKPT=$(find "${VLA_CACHE:-$HOME/.cache/vla}/vrfai/bitvla-libero-gguf" \
              -name '*.gguf' 2>/dev/null | sort | head -1)
fi
if [ ! -f "$CKPT" ]; then
  echo "WARN: no bitvla .gguf found - fetching via vla-bench first"
  "$BUILD_DIR/vla-bench" -hf "$MODEL" --images 1 --size 224 --warmup 0 --reps 1 >/dev/null 2>&1
  CKPT=$(find "${VLA_CACHE:-$HOME/.cache/vla}/vrfai/bitvla-libero-gguf" \
              -name '*.gguf' 2>/dev/null | sort | head -1)
fi
echo "CKPT: ${CKPT:-<none>}"

if [ "$have_before" = "1" ] && [ -f "$CKPT" ]; then
  echo
  echo "=== actions: before vs after (must be bit-identical) ==="
  # The old executable needs the old libvla_core.so ahead of the rebuilt one.
  LD_LIBRARY_PATH="$BEFORE:$BUILD_DIR:$BUILD_DIR/bin:$LD_LIBRARY_PATH" \
      "$BEFORE/vla_predict_check" "$CKPT" "" 1 > "$OUT/actions.before.txt" 2> "$OUT/actions.before.log"
  echo "PREDICT_RC[before]: $?"
  "$BUILD_DIR/tests/vla_predict_check" "$CKPT" "" 1 > "$OUT/actions.after.txt" 2> "$OUT/actions.after.log"
  echo "PREDICT_RC[after]:  $?"

  # --tol 0: anything but an exact match fails, which is the claim being made.
  python3 "$REPO_DIR/ci/compare_actions.py" "$OUT/actions.before.txt" "$OUT/actions.after.txt" \
      --tol 0 --dims 7 --label-a before --label-b after
  echo "IDENTICAL_RC: $?"
fi

# --- speed ------------------------------------------------------------------
echo
echo "=== bench: before ==="
if [ "$have_before" = "1" ]; then
  LD_LIBRARY_PATH="$BEFORE:$BUILD_DIR:$BUILD_DIR/bin:$LD_LIBRARY_PATH" \
      "$BEFORE/vla-bench" -hf "$MODEL" --images 1 --size 224 \
      --warmup "$WARMUP" --reps "$REPS" --markdown 2>&1 | tail -3
fi
echo
echo "=== bench: after ==="
"$BUILD_DIR/vla-bench" -hf "$MODEL" --images 1 --size 224 \
    --warmup "$WARMUP" --reps "$REPS" --markdown 2>&1 | tail -3

# --- profile ----------------------------------------------------------------
# Same instrument as job 371998, so the buckets are directly comparable and the
# usm-copy-2d row should now be absent rather than merely smaller.
if [ "${SKIP_PROFILE:-0}" != "1" ]; then
  echo
  echo "=== unitrace --device-timing (after) ==="
  "$UNITRACE" -d -v -o "$OUT/device" \
      "$BUILD_DIR/vla-bench" -hf "$MODEL" --images 1 --size 224 \
      --warmup 1 --reps "$PROF_REPS" 2>&1 | tail -3
  for f in "$OUT"/device.*; do
    [ -f "$f" ] || continue
    echo "########## $f ##########"
    cat "$f"
  done
fi

echo
echo "DONE"
