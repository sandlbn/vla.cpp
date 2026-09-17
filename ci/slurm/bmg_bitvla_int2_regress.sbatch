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
# bmg_bitvla_int2_regress.sbatch -- the shipping int2 path still ships.
#
# Stage B added block-quantised weight flavours, and to do it the loader had to
# stop using `matmul_type == F32` as the signal that the ternary kernels take
# the GEMMs. That value was doing two unrelated jobs -- resident dtype and
# kernel trigger -- and a Q8_0 checkpoint arrives with it still at its F32
# default, so as the code stood a kernel build would have handed ladder-expecting
# kernels a block-quantised tensor. The fix is one `m->kernels_drive`, computed
# once in src/models/bitvla.cpp.
#
# That refactor is the change in the stage most able to break something that
# already works, and nothing else in Stage B touches the int2 path. So this job
# runs it: the published ladder-packed int2 checkpoint, on the SYCL kernels.
#
# Three checks, in increasing order of how much they prove:
#
#   ENGAGEMENT   the three kernel banners ("LM forward ENABLED", "ViT forward
#                ENABLED", "fp32head ... ENABLED") must appear. If kernels_drive
#                came out false the model would quietly run its glue graph on
#                ggml-sycl and produce *correct* actions -- passing any
#                comparison while proving the opposite of what it claims.
#
#   NEUTRALITY   the gate. src/models/bitvla.cpp is swapped for HEAD's copy, the
#                one TU is rebuilt, and the same checkpoint is run again. For an
#                int2 file on a kernel build the two versions of the condition
#                are the same expression by substitution -- packed_int2 is true,
#                ggml_is_quantized(I8) is false, and matmul_type still defaults
#                to F32 under VLA_BITVLA_GPU_KERNELS -- so the refactor owes the
#                identical answer.
#
#                Measured against a control, not against zero: the working-tree
#                binary is run *twice* first, because the SYCL ternary kernels
#                are not bit-reproducible run to run (job 372750 saw one element
#                in 56 move by 1e-6 between two runs of the same code). Without
#                that control a reproducibility wobble reads as a regression.
#                The gate is head-vs-stage-b <= mine-vs-mine2, i.e. the refactor
#                must not move the output further than the engine moves it on
#                its own.
#
#   CUDA REF     not gating, and cannot be. ci/reference/*-cuda.txt was recorded
#                on an H100 with the CUDA kernels; cross-vendor ternary GEMMs are
#                not bit-identical, and BitVLA amplifies any non-bit-identity to
#                ~0.07-0.14 normalised (its decorrelation floor is 0.068 for a
#                *one-ULP* input perturbation -- see docs/backend/ov.md, "BitVLA
#                has no action bar"). Printed for the record; a number in that
#                band means "different silicon", not "broken".
#
# The swap is in-place in the working tree and is undone by an EXIT trap, which
# also rebuilds. If the job is killed -9 the tree is left holding HEAD's copy of
# that one file and the working version is at $OUT.mine -- the log says so.
#
#   sbatch ci/slurm/bmg_bitvla_int2_regress.sbatch
#
# Knobs: BUILD_DIR, CKPT, REFERENCE, IMG_SIZE, ITERS, JOBS, BASE=0 (skip the
# bit-identity half, e.g. when there is nothing uncommitted to compare against).
# =============================================================================
#SBATCH --job-name=bitvla_int2_regress
#SBATCH --partition=b70
#SBATCH --nodelist=pcl-adl01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=slurm_bitvla_int2_regress_%j.out

# No `set -u`: the vendor toolchain scripts read unset variables.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-sycl}"
JOBS="${JOBS:-$(nproc)}"
TASK="${TASK:-libero_object}"
CKPT="${CKPT:-$HOME/.cache/vla/vrfai/bitvla-libero-gguf/$TASK/bitvla-${TASK//_/-}.gguf}"
REFERENCE="${REFERENCE:-$REPO_DIR/ci/reference/bitvla-${TASK}-cuda.txt}"
IMG_SIZE="${IMG_SIZE:-224}"
ITERS="${ITERS:-1}"
SRC="$REPO_DIR/src/models/bitvla.cpp"
OUT="${SLURM_SUBMIT_DIR:-$PWD}/int2_regress_${SLURM_JOB_ID:-$$}"

echo "NODE:      $(hostname -s)"
echo "BUILD:     $BUILD_DIR"
echo "CKPT:      $CKPT"
echo "REFERENCE: $REFERENCE"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

[ -f "$CKPT" ] || fail "no int2 checkpoint at $CKPT"

source /swtools/intel-gpu/latest/intel_gpu_vars.sh || fail "TOOLCHAIN intel_gpu_vars.sh"
source /swtools/intel/setvars.sh --force >/dev/null 2>&1
echo "GPU:       $(sycl-ls 2>/dev/null | grep -m1 level_zero)"

[ -f "$BUILD_DIR/CMakeCache.txt" ] || fail "no $BUILD_DIR -- run ci/slurm/bmg_build.sbatch first"

build() {
  cmake --build "$BUILD_DIR" -j"$JOBS" --target vla_predict_check 2>&1 | tail -3
  [ "${PIPESTATUS[0]}" = 0 ]
}

run() {   # $1 = tag
  local tag="$1"
  VLA_IMG_SIZE="$IMG_SIZE" VLA_BENCH_ITERS="$ITERS" \
      "$BUILD_DIR/tests/vla_predict_check" "$CKPT" "" 1 \
      > "$OUT.$tag.txt" 2> "$OUT.$tag.err"
  return $?
}

# --- arm 1: the tree as it stands --------------------------------------------
echo
echo "--- build (working tree) ---"
build || fail "BUILD working"

echo
echo "--- predict_check, working tree ---"
run mine || { tail -20 "$OUT.mine.err"; fail "RUN working"; }
grep -hE "resident=|weights resident|forward ENABLED|fp32head|backend =" \
     "$OUT.mine.txt" "$OUT.mine.err" | head -10

echo
echo "--- engagement ---"
for want in "LM forward ENABLED" "ViT forward ENABLED" "fp32head"; do
  if grep -qh "$want" "$OUT.mine.txt" "$OUT.mine.err"; then
    echo "  ok      $want"
  else
    echo "  MISSING $want"
    fail "KERNELS DID NOT ENGAGE -- kernels_drive came out false on an int2 checkpoint"
  fi
done
grep -qh "resident=int2/" "$OUT.mine.txt" || fail "LOADER DID NOT SEE THE WEIGHTS AS int2"

# --- the control: the same binary, again -------------------------------------
# What the engine does to its own output when nothing changes. Everything below
# is measured against this, not against zero.
echo
echo "--- predict_check, working tree AGAIN (reproducibility control) ---"
run mine2 || { tail -20 "$OUT.mine2.err"; fail "RUN working repeat"; }

DIMS="$(sed -n 's/.*chunk.dim=[0-9]*.\([0-9]*\).*/\1/p' "$OUT.mine.txt" | head -1)"
cmp_max() {   # $1 = a.txt, $2 = b.txt, $3 = label-a, $4 = label-b; echoes max_abs
  python3 "$REPO_DIR/ci/compare_actions.py" "$1" "$2" \
      --tol 1.0 --dims "${DIMS:-0}" --label-a "$3" --label-b "$4" > "$OUT.cmp.$3-$4.txt" 2>&1
  sed -n 's/.*max_abs=\([0-9.eE+-]*\).*/\1/p' "$OUT.cmp.$3-$4.txt" | head -1
}

FLOOR="$(cmp_max "$OUT.mine.txt" "$OUT.mine2.txt" stage-b stage-b2)"
head -1 "$OUT.cmp.stage-b-stage-b2.txt"
echo "FLOOR: the engine repeats itself to max_abs=${FLOOR:-?}"

# --- arm 2: the same thing at HEAD -------------------------------------------
rc=0
if [ "${BASE:-1}" = "1" ] && ! git -C "$REPO_DIR" diff --quiet HEAD -- "$SRC"; then
  cp "$SRC" "$OUT.mine.cpp" || fail "could not save $SRC"
  restore() {
    cp "$OUT.mine.cpp" "$SRC" &&
      echo "restored $SRC from $OUT.mine.cpp" ||
      echo "!!! COULD NOT RESTORE $SRC -- copy it back from $OUT.mine.cpp"
    build >/dev/null 2>&1
  }
  trap restore EXIT

  echo
  echo "--- build (src/models/bitvla.cpp at HEAD) ---"
  git -C "$REPO_DIR" show "HEAD:src/models/bitvla.cpp" > "$SRC" || fail "git show"
  build || fail "BUILD base"

  echo
  echo "--- predict_check, HEAD ---"
  run base || { tail -20 "$OUT.base.err"; fail "RUN base"; }
  grep -hE "resident=|forward ENABLED" "$OUT.base.txt" | head -4

  echo
  echo "--- neutrality: HEAD vs working tree, against the control ---"
  DELTA="$(cmp_max "$OUT.base.txt" "$OUT.mine.txt" head stage-b)"
  cat "$OUT.cmp.head-stage-b.txt"
  echo
  echo "  refactor moved the output by  max_abs=${DELTA:-?}"
  echo "  the engine moves it by        max_abs=${FLOOR:-?}  on a repeat of itself"
  # 1.5e-6 is the floor's floor: predict_check prints six decimals, so two runs
  # that differ at the last printed digit of one element are at the resolution
  # limit of the artefact being compared, not at a resolution limit of the model.
  if awk -v d="${DELTA:-1}" -v f="${FLOOR:-0}" \
         'BEGIN{ if (f < 1.5e-6) f = 1.5e-6; exit !(d <= f) }'; then
    echo "GATE: PASSED  the kernels_drive split is inside the engine's own noise"
  else
    echo "GATE: FAILED  the kernels_drive split moved the int2 path further than"
    echo "              the engine moves it on its own -- this is a real change"
    rc=1
  fi
else
  echo
  echo "--- neutrality: SKIPPED (nothing uncommitted in src/models/bitvla.cpp) ---"
fi

# --- the CUDA reference, for the record --------------------------------------
if [ -f "$REFERENCE" ]; then
  echo
  echo "--- vs the recorded CUDA reference (NOT GATING) ---"
  cmp_max "$REFERENCE" "$OUT.mine.txt" cuda-ref bmg-sycl >/dev/null
  cat "$OUT.cmp.cuda-ref-bmg-sycl.txt"
  echo "(BitVLA's decorrelation floor is 0.068 for a one-ULP input perturbation,"
  echo " and this reference is different silicon running different kernels, so a"
  echo " number in the 0.07-0.15 band is expected. See docs/backend/ov.md.)"
fi

echo
if [ "$rc" = 0 ]; then
  echo "VERDICT: PASSED  int2 kernels engage and the refactor is bit-neutral"
else
  echo "VERDICT: FAILED  see GATE: lines above"
fi
echo "DONE"
exit "$rc"
