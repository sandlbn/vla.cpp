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
# bmg_bf16_e2e.sbatch -- end-to-end gate for the SYCL BF16 op extension.
#
# test_bf16_sycl_ops proves the four kernels on an eight-node graph. This proves
# the thing that actually matters: a real checkpoint, loaded and run on the same
# device in stock F32 and with --act-dtype bf16, produces the same action chunk.
# The BF16 run is the only one that reaches the hook, so a clean diff here is a
# statement about the whole extension, not about one kernel.
#
# Four arms, not two. Each dtype is run twice, because the engine turns out not
# to be bit-reproducible run to run and a dtype comparison below that floor means
# nothing. The f32 pair is the one that assigns blame: it never enters
# src/sycl/vla_sycl_bf16.cpp, so whatever it fails to reproduce is upstream of
# this port. See the TOL comment below for the measured numbers.
#
# evo1 and pi0 are the only architectures that opt into the BF16 activation path,
# so the model is not a knob to widen casually: any other repo would run both
# arms through identical code and pass without testing anything. The job checks
# for the model's own "activations = BF16" banner rather than trusting the flag.
#
#   sbatch ci/slurm/bmg_bf16_e2e.sbatch
#   MODEL=vrfai/pi0-libero-finetuned-v044-gguf IMG_SIZE=224 sbatch ci/slurm/bmg_bf16_e2e.sbatch
#
# The CUDA half of the precision claim is ci/slurm/h100_bf16_e2e.sbatch; run this
# one first so it has a SYCL artefact to diff against.
#
# Knobs: BUILD_DIR, MODEL, IMG_SIZE, TOL (normalised action deviation), ITERS,
# CTEST=0.
# =============================================================================
#SBATCH --job-name=bmg_bf16_e2e
#SBATCH --partition=b70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=slurm_bmg_bf16_e2e_%j.out

# No `set -u`: both vendor scripts read unset variables.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-sycl}"
MODEL="${MODEL:-vrfai/evo1-libero-gguf}"
# 0.06 is sized against the two controls this job prints, not picked to make the
# test pass. The engine is not bit-reproducible run to run (stock ggml-sycl's own
# f32 path repeats to ~0.007 normalised on pi0), and a saturating action
# dimension - LIBERO's gripper - turns that floor into a visibly larger output
# swing at a handful of steps. Below that floor the number says nothing about the
# dtype. Read max_abs together with rms: rms is the stable statistic here and
# sits around 1e-3 on both models.
TOL="${TOL:-0.06}"
ITERS="${ITERS:-10}"
# evo1's SigLIP tower is configured for 448; a mismatched view is refused by the
# model and predict() returns empty, which reads as a failure rather than a skip.
IMG_SIZE="${IMG_SIZE:-448}"
OUT="${SLURM_SUBMIT_DIR:-$PWD}/bf16_e2e_$$"

echo "NODE:  $(hostname)"
echo "MODEL: $MODEL"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

# The per-arm scratch files are removed on every exit path, not just the happy
# one: an arm that fails leaves them behind in the submit directory otherwise.
trap 'rm -f "$OUT".*.txt "$OUT".*.err' EXIT

source /swtools/intel-gpu/latest/intel_gpu_vars.sh || fail "TOOLCHAIN intel_gpu_vars.sh"
source /swtools/intel/setvars.sh --force >/dev/null 2>&1 || fail "TOOLCHAIN setvars.sh"

[ -x "$BUILD_DIR/tests/vla_predict_check" ] || fail "no vla_predict_check in $BUILD_DIR"

# The checkpoint is pre-pulled into $VLA_CACHE from the login node; -hf would
# shell out to `hf download` here, and a compute node may have no route out.
CKPT="$(ls "${VLA_CACHE:-$HOME/.cache/vla}/$MODEL"/*.gguf 2>/dev/null | head -1)"
[ -n "$CKPT" ] || fail "no .gguf for $MODEL under ${VLA_CACHE:-$HOME/.cache/vla} -- pre-pull it"
echo "CKPT:  $CKPT"

# --- unit tests -------------------------------------------------------------
if [ "${CTEST:-1}" = "1" ]; then
  echo
  echo "=== ctest ==="
  ( cd "$BUILD_DIR" && ctest --output-on-failure 2>&1 | tail -30 )
  echo "CTEST_RC: ${PIPESTATUS[0]}"
fi

# --- the two arms -----------------------------------------------------------
# Same binary, same fixed images/state/noise, same device; the only difference is
# the activation dtype. stdout is `action_len=N` then N floats.
run_arm() {   # $1 = tag, $2.. = extra args; VLA_DEVICE/ITERS may be preset
  local tag="$1"; shift
  echo
  echo "=== predict_check [$tag] ==="
  # The empty second argument is load-bearing: predict_check reads argv[2] as an
  # mmproj path whenever it is non-empty and not a flag, so `<ckpt> 1` would make
  # "1" the mmproj and leave n_images at its default.
  VLA_IMG_SIZE="$IMG_SIZE" VLA_BENCH_ITERS="$ITERS" \
      "$BUILD_DIR/tests/vla_predict_check" "$CKPT" "" 1 "$@" \
      > "$OUT.$tag.txt" 2> "$OUT.$tag.err"
  local rc=$?
  # predict_check prints one adapts/notes line per iteration, so filter rather
  # than tail: the lines worth seeing are the dtype banner and the timings.
  # Both streams: the model's dtype banner is a printf (stdout) while the
  # timings are fprintf(stderr).
  grep -hE "activations =|BF16_ACT|predict\(\) over|last split|error|failed" \
      "$OUT.$tag.txt" "$OUT.$tag.err" | sort -u | head -6
  echo "RC[$tag]: $rc"
  return $rc
}

run_arm f32                     || fail "RUN f32"
run_arm f32b                    || fail "RUN f32b"
run_arm bf16 --act-dtype bf16   || fail "RUN bf16"
run_arm bf16b --act-dtype bf16  || fail "RUN bf16b"

# The independent reference. Both GPU arms above are scored against each other,
# and that only measures the *change* a dtype makes - it cannot say which one is
# closer to the truth, because the F32 arm is not itself exact. It matters here
# that the two backends' F32 arms are not even the same precision: ggml-cuda sets
# CUBLAS_TF32_TENSOR_OP_MATH on every handle (common.cuh), so CUDA's "F32" GEMMs
# run on TF32 tensor cores at a 10-bit mantissa, while ggml-sycl's f16 fpmath
# hint is behind GGML_SYCL_F16, which is off in this build - so SYCL's F32 is
# true fp32. Comparing the two GPU bf16 arms to each other mixes that asymmetry
# in with the kernels.
#
# VLA_DEVICE out of range falls back to CPU (src/backend.h), so the same binary
# gives an fp32 reference with no vendor math mode in it at all. It is slow, so
# one iteration and no timing loop.
if [ "${CPU_REF:-1}" = "1" ]; then
  # Subshell + export: run_arm passes VLA_IMG_SIZE/VLA_BENCH_ITERS explicitly but
  # inherits VLA_DEVICE, and the subshell keeps both off the later arms.
  ( export VLA_DEVICE=99; ITERS=0; run_arm cpu ) || fail "RUN cpu"
fi

# Without this the comparison is vacuous: if --act-dtype bf16 were ignored (wrong
# weight type, a backend that is neither CUDA nor SYCL) both arms would run the
# identical F32 graph and agree perfectly while testing nothing.
grep -qh "activations = BF16" "$OUT.bf16.txt" "$OUT.bf16.err" ||
    fail "BF16 ARM DID NOT ENGAGE -- model declined --act-dtype bf16"

# --- compare ----------------------------------------------------------------
# max_action_dim comes off predict_check's own loaded: line, so the per-dimension
# table is keyed on what the checkpoint actually declares.
DIMS="$(sed -n 's/.*max_action=\([0-9]*\).*/\1/p' "$OUT.f32.err" | head -1)"

# The two controls, and they have to be read before the gate below means
# anything. predict_check builds its images, tokens, state and noise from fixed
# formulas, so a repeated invocation should reproduce exactly; whatever it does
# not reproduce is the engine, and it sets the floor under which an f32-vs-bf16
# number cannot be interpreted.
#
# The f32 pair is the one that assigns blame. It runs entirely on stock
# ggml-sycl - none of src/sycl/vla_sycl_bf16.cpp is reachable without
# --act-dtype bf16 - so a non-zero f32 control is upstream of this port.
echo
echo "=== f32 vs f32 (run-to-run control, stock ggml-sycl) ==="
python3 "$REPO_DIR/ci/compare_actions.py" "$OUT.f32.txt" "$OUT.f32b.txt" \
    --tol 1.0 --dims "${DIMS:-0}" --label-a run1 --label-b run2 | head -2
echo
echo "=== bf16 vs bf16 (run-to-run control, vla kernels) ==="
python3 "$REPO_DIR/ci/compare_actions.py" "$OUT.bf16.txt" "$OUT.bf16b.txt" \
    --tol 1.0 --dims "${DIMS:-0}" --label-a run1 --label-b run2 | head -2
echo

echo "=== f32 vs bf16 action chunk ==="
python3 "$REPO_DIR/ci/compare_actions.py" "$OUT.f32.txt" "$OUT.bf16.txt" \
    --tol "$TOL" --dims "${DIMS:-0}" --label-a f32 --label-b bf16
cmp_rc=$?
echo "COMPARE_RC: $cmp_rc"

# Against the CPU fp32 reference. This is the pair that says which arm is closer
# to the truth rather than merely how far apart the two are.
if [ -f "$OUT.cpu.txt" ]; then
  echo
  echo "=== cpu-f32 vs gpu-f32 (reference) ==="
  python3 "$REPO_DIR/ci/compare_actions.py" "$OUT.cpu.txt" "$OUT.f32.txt" \
      --tol 1.0 --dims "${DIMS:-0}" --label-a cpu --label-b gpuf32 | head -2
  echo
  echo "=== cpu-f32 vs gpu-bf16 (reference) ==="
  python3 "$REPO_DIR/ci/compare_actions.py" "$OUT.cpu.txt" "$OUT.bf16.txt" \
      --tol 1.0 --dims "${DIMS:-0}" --label-a cpu --label-b gpubf16 | head -2
  cp "$OUT.cpu.txt" "$REPO_DIR/ci/reference/bf16/$(basename "$MODEL")-cpu-f32.txt" 2>/dev/null
fi

# Kept, not deleted: the action vectors are the artefact the CUDA reference run
# is diffed against, and re-running to get them back costs a node.
ART="$REPO_DIR/ci/reference/bf16"
mkdir -p "$ART"
tag="$(basename "$MODEL")"
cp "$OUT.f32.txt"  "$ART/$tag-sycl-f32.txt"
cp "$OUT.bf16.txt" "$ART/$tag-sycl-bf16.txt"
echo "artefacts: $ART/$tag-sycl-{f32,bf16}.txt"

[ "$cmp_rc" = "0" ] && echo "VERDICT: OK" || echo "VERDICT: FAILED COMPARE"
echo "DONE"
