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
# bmg_bf16_bench.sbatch -- what --act-dtype bf16 is worth on Battlemage.
#
# ci/slurm/bmg_bf16_e2e.sbatch answers "is it still correct". This answers "is it
# any faster", which until now has rested on the ten-iteration timing loop inside
# vla_predict_check -- a number good enough to notice a regression and not good
# enough to publish. vla-bench is the right instrument: it sorts the reps and
# reports min/p50/p90/mean, so the claim can be made on p50 with the spread
# visible next to it.
#
# It runs the whole model -- vision tower, LM, action head, real checkpoint
# weights, the same vla::predict() the server calls. It does not run the
# workload: inputs are a procedural pixel ramp and fixed state/noise, there is no
# simulator and no task success. A regression in *accuracy* will not show up
# here; that is what the e2e job and eval/ are for.
#
# evo1 and pi0 are the only two architectures that opt into the BF16 activation
# path, which is why the list is not longer. Each is run twice, f32 then bf16,
# and the bf16 arm is checked for the model's own banner rather than trusting
# that the flag was accepted -- an arch that declines --act-dtype bf16 runs the
# identical F32 graph and would otherwise report a very convincing 1.00x.
#
#   sbatch ci/slurm/bmg_bf16_bench.sbatch
#   MODELS="vrfai/evo1-libero-gguf" REPS=40 sbatch ci/slurm/bmg_bf16_bench.sbatch
#
# Knobs: BUILD_DIR, MODELS, REPS, WARMUP, IMAGES, SIZE (overrides the per-arch
# default below).
# =============================================================================
#SBATCH --job-name=bmg_bf16_bench
#SBATCH --partition=b70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=slurm_bmg_bf16_bench_%j.out

# No `set -u`: both vendor scripts read unset variables.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-sycl}"
MODELS="${MODELS:-vrfai/evo1-libero-gguf vrfai/pi0-libero-finetuned-v044-gguf}"
REPS="${REPS:-20}"
WARMUP="${WARMUP:-3}"
IMAGES="${IMAGES:-1}"
OUT="${SLURM_SUBMIT_DIR:-$PWD}/bf16_bench_$$"

echo "NODE: $(hostname)"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }
trap 'rm -f "$OUT".*.txt' EXIT

source /swtools/intel-gpu/latest/intel_gpu_vars.sh || fail "TOOLCHAIN intel_gpu_vars.sh"
source /swtools/intel/setvars.sh --force >/dev/null 2>&1 || fail "TOOLCHAIN setvars.sh"

[ -x "$BUILD_DIR/vla-bench" ] || fail "no vla-bench in $BUILD_DIR"
echo "GPU:  $(sycl-ls 2>/dev/null | grep -m1 level_zero)"

# --- arms -------------------------------------------------------------------
# One tagged file per (model, dtype) so the summary at the end can be assembled
# from what actually ran rather than re-parsed out of the interleaved log.
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

  # Pre-pulled into $VLA_CACHE from the login node: -hf would shell out to
  # `hf download` here, and a compute node may have no route out.
  ckpt="$(ls "${VLA_CACHE:-$HOME/.cache/vla}/$repo"/*.gguf 2>/dev/null | head -1)"
  [ -n "$ckpt" ] || fail "no .gguf for $repo under ${VLA_CACHE:-$HOME/.cache/vla} -- pre-pull it"

  # evo1's SigLIP tower is configured for 448 and refuses a mismatched view;
  # pi0 wants 224. A wrong size does not error out of vla-bench, it just makes
  # the two architectures incomparable to their own published rows.
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
# Speedup off p50, not min: min is the one rep that dodged every scheduling
# hiccup, and on a 20-rep sample it is the least stable statistic on offer.
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
    # Not fatal: the per-arm output above is the record, and a changed vla-bench
    # summary line should not fail a job whose benchmarks all completed.
    printf '%-34s %10s %10s %8s\n' "$tag" "${a:-?}" "${b:-?}" "n/a"
  fi
done

echo
echo "VERDICT: OK"
echo "DONE"
