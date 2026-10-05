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
# bench_sycl.sh -- vla-bench latency on the SYCL backend, f32 vs bf16
# activations. Local equivalent of ci/slurm/bmg_bf16_bench.sbatch.
#
#   ci/local/build.sh sycl
#   ci/local/bench_sycl.sh
#   MODELS="vrfai/evo1-libero-gguf" REPS=50 ci/local/bench_sycl.sh
#
# Missing checkpoints are fetched from the Hub on first use (-hf) and cached
# under $VLA_CACHE, so the first run needs network and the rest do not.
#
# Knobs: BUILD_DIR (build-sycl), MODELS, REPS (20), WARMUP (3), IMAGES (1),
# SIZE (224, 448 for evo1), OUT (outputs/local/bench_sycl_<timestamp>).
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-sycl}"
MODELS="${MODELS:-vrfai/evo1-libero-gguf vrfai/pi0-libero-finetuned-v044-gguf}"
REPS="${REPS:-20}"
WARMUP="${WARMUP:-3}"
IMAGES="${IMAGES:-1}"
OUT="${OUT:-$REPO_DIR/outputs/local/bench_sycl_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

load_oneapi
[ -x "$BUILD_DIR/vla-bench" ] || fail "no vla-bench in $BUILD_DIR -- ci/local/build.sh sycl"
echo "OUT:       $OUT"

run_bench() {   # $1 = tag, $2 = model source args (--ckpt X | -hf X), $3 = size, $4.. = extra
  local tag="$1" src="$2" size="$3"; shift 3
  echo
  echo "=== vla-bench [$tag] ==="
  # shellcheck disable=SC2086
  "$BUILD_DIR/vla-bench" $src --label "$tag" \
      --images "$IMAGES" --size "$size" \
      --warmup "$WARMUP" --reps "$REPS" "$@" 2>&1 | tee "$OUT/$tag.txt"
  return "${PIPESTATUS[0]}"
}

for repo in $MODELS; do
  tag="$(basename "$repo")"
  ckpt="$(find_ckpt "$repo")"
  if [ -n "$ckpt" ]; then src="--ckpt $ckpt"; else src="-hf $repo"; fi
  size="${SIZE:-224}"
  case "$repo" in *evo1*) size="${SIZE:-448}" ;; esac
  echo
  echo "##### $repo (size $size, $IMAGES image, $REPS reps) #####"
  run_bench "$tag-f32"  "$src" "$size"                  || fail "BENCH $tag f32"
  run_bench "$tag-bf16" "$src" "$size" --act-dtype bf16 || fail "BENCH $tag bf16"
  # An arch that ignores --act-dtype would report a perfect, meaningless tie.
  grep -q "activations = BF16" "$OUT/$tag-bf16.txt" || fail "BF16 ARM DID NOT ENGAGE for $repo"
done

# Only vla-bench's own result line ("<label>: min X ms  mean X ms  p50 X ms ...").
p50() { grep -h "^$2: min " "$1" | sed -n 's/.* p50 \([0-9.]*\) ms.*/\1/p' | head -1; }
echo
echo "=== summary (p50 ms, lower is better) ==="
printf '%-40s %10s %10s %8s\n' model f32 bf16 speedup
for repo in $MODELS; do
  tag="$(basename "$repo")"
  a="$(p50 "$OUT/$tag-f32.txt" "$tag-f32")"; b="$(p50 "$OUT/$tag-bf16.txt" "$tag-bf16")"
  printf '%-40s %10s %10s %8s\n' "$tag" "${a:-?}" "${b:-?}" \
      "$(awk -v x="$a" -v y="$b" 'BEGIN{ if (y>0) printf "%.2fx", x/y; else print "n/a" }')"
done
echo
echo "Raw logs: $OUT"
echo "VERDICT: OK"
echo "DONE"
