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
# bench_ov.sh -- vla-bench latency on the OpenVINO GPU plugin.
# Local, no-profiling equivalent of the timing half of
# ci/slurm/bmg_ov_quant_why_flat.sbatch.
#
#   ci/local/build.sh ov
#   ci/local/bench_ov.sh                                  # BitVLA, 4 arms
#   ARMS="bf16:f32 bf16:f16" ci/local/bench_ov.sh         # precision only
#   ARMS= MODELS="vrfai/evo1-libero-gguf" ci/local/bench_ov.sh   # another arch
#
# An arm is WEIGHTS:PRECISION for BitVLA:
#   WEIGHTS    bf16 | q8_0 | q4_0   checkpoint flavour, transcoded on first use
#   PRECISION  f32 | f16            GGML_OPENVINO_GPU_PRECISION, the GPU plugin's
#                                   compute precision
# MODELS adds Hub checkpoints, each run at every precision in PRECISIONS.
#
# This is the honest latency number: ov::enable_profiling is off. Per-op
# profiles (bmg_ov_profile_ops) perturb what they measure and only their
# within-run ratios mean anything.
#
# Knobs: BUILD_DIR (build-ov), ARMS, MODELS, PRECISIONS (f32 f16), DEVICE (GPU),
# REPS (20), WARMUP (3), IMAGES (1), SIZE (224), OUT.
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-ov}"
ARMS="${ARMS-bf16:f32 bf16:f16 q8_0:f16 q4_0:f16}"
MODELS="${MODELS:-}"
PRECISIONS="${PRECISIONS:-f32 f16}"
DEVICE="${DEVICE:-GPU}"
REPS="${REPS:-20}"
WARMUP="${WARMUP:-3}"
IMAGES="${IMAGES:-1}"
SIZE="${SIZE:-224}"
OUT="${OUT:-$REPO_DIR/outputs/local/bench_ov_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

load_openvino
[ -x "$BUILD_DIR/vla-bench" ] || fail "no vla-bench in $BUILD_DIR -- ci/local/build.sh ov"
echo "DEVICE:    $DEVICE"
echo "OUT:       $OUT"

TAGS=()
run_bench() {   # $1 = tag, $2 = precision, $3.. = model source args
  local tag="$1" prec="$2"; shift 2
  local log="$OUT/$tag.txt"
  echo
  echo "=== vla-bench [$tag]  device=$DEVICE precision=$prec ==="
  env GGML_OPENVINO_DEVICE="$DEVICE" GGML_OPENVINO_GPU_PRECISION="$prec" \
      "$BUILD_DIR/vla-bench" "$@" --label "$tag" \
          --images "$IMAGES" --size "$SIZE" \
          --warmup "$WARMUP" --reps "$REPS" 2>&1 | tee "$log"
  [ "${PIPESTATUS[0]}" = 0 ] || return 1
  # ggml falls back to CPU without failing when the device is missing, which
  # gives a perfectly plausible -- and wrong -- latency. Both lines must appear.
  grep -q "backend = OPENVINO" "$log" || { echo "  $tag did not use OpenVINO"; return 1; }
  grep -q "using device $DEVICE" "$log" ||
      { echo "  $tag did not reach the $DEVICE plugin (fell back?)"; return 1; }
  TAGS+=("$tag")
}

for arm in $ARMS; do
  w="${arm%%:*}"; p="${arm#*:}"
  case "$w" in bf16|q8_0|q4_0) ;; *) fail "arm $arm: weights must be bf16|q8_0|q4_0" ;; esac
  case "$p" in f32|f16) ;;           *) fail "arm $arm: precision must be f32|f16" ;; esac
  ckpt="$(bitvla_ckpt "$w")" || fail "CHECKPOINT $w"
  run_bench "bitvla-$w-$p" "$p" --ckpt "$ckpt" || fail "BENCH $arm"
done

for repo in $MODELS; do
  ckpt="$(find_ckpt "$repo")"
  for p in $PRECISIONS; do
    if [ -n "$ckpt" ]; then
      run_bench "$(basename "$repo")-$p" "$p" --ckpt "$ckpt" || fail "BENCH $repo $p"
    else
      run_bench "$(basename "$repo")-$p" "$p" -hf "$repo" || fail "BENCH $repo $p"
    fi
  done
done

[ ${#TAGS[@]} -gt 0 ] || fail "nothing to run -- set ARMS and/or MODELS"

# Only vla-bench's own result line ("<label>: min X ms  mean X ms  p50 X ms ..."),
# so a stray "mean" or "p50" elsewhere in the log cannot be picked up instead.
field() { grep -h "^$2: min " "$1" | sed -n "s/.* $3 \([0-9.]*\) ms.*/\1/p" | head -1; }
echo
echo "=== summary (ms, lower is better) ==="
printf '%-44s %10s %10s\n' arm p50 mean
for t in "${TAGS[@]}"; do
  printf '%-44s %10s %10s\n' "$t" "$(field "$OUT/$t.txt" "$t" p50)" "$(field "$OUT/$t.txt" "$t" mean)"
done
echo
echo "Raw logs: $OUT"
echo "VERDICT: OK"
echo "DONE"
