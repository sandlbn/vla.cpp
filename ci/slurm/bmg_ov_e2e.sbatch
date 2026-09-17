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
# bmg_ov_e2e.sbatch -- BitVLA end to end on OpenVINO, and what precision it needs.
#
# Stage A of the OpenVINO port: the dense bf16 checkpoint through the ordinary
# ggml graph, on the CPU plugin and on the Arc Pro B70, scored against ggml's own
# CPU backend on identical fixed inputs.
#
# It also answers a question the act_quant unit test cannot. That test measures
# one op's bit fidelity and says f16 is bad: the BitNet quantiser rounds against
# 127/amax and Round is a threshold, so at f16 whole integers move -- 2215 of
# 394752 elements, a full quantisation step each. That is a real measurement and
# it is still the wrong bar to set a default from, because a flip only happens
# where x*s sits within f16 rounding of a .5 boundary, which is exactly where the
# quantiser was already undecided. The dequantised value moves one step, but the
# error against the *unquantised* input only goes from +0.5 step to -0.5 step.
# It is noise the size of the quantiser's own noise, not a bias, and whether it
# survives thirty layers is a question about the model, not about the op.
#
# So: run the GPU at f16 and at f32 and compare both to the reference. If f16 is
# inside the bar, backend.h's bitvla entry in the GGML_OPENVINO_GPU_PRECISION
# default comes out -- its own comment says so -- and the port runs at the
# precision OpenVINO is built around. If it is not, the entry has stopped being
# a prediction and become a measurement.
#
# Two build directories, both per-host, both built here. GGML_NATIVE bakes in
# -march=native and the b70 nodes are Alder Lake / Arrow Lake / Zen 4 while the
# login node is a Xeon 8580, so nothing may be carried across. The reference arm
# needs a build with no OpenVINO compiled into it at all: vla::backend_init has
# no env knob that declines the backend once GGML_USE_OPENVINO is defined, and a
# reference that shares a backend with the thing it is scoring is not one.
#
#   sbatch ci/slurm/bmg_ov_e2e.sbatch
#   CKPT=/path/to/other-bf16.gguf sbatch ci/slurm/bmg_ov_e2e.sbatch
#
# Knobs: CKPT, TASK, OV_BUILD, CPU_BUILD, JOBS, OV_ROOT, TOL, IMG_SIZE, ITERS.
# =============================================================================
#SBATCH --job-name=bmg_ov_e2e
#SBATCH --partition=b70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=slurm_bmg_ov_e2e_%j.out

# No `set -u`: intel_gpu_vars.sh and OpenVINO's setupvars.sh both read unset
# variables, so nounset halts before the toolchain is on PATH.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
HOST="$(hostname -s)"
OV_BUILD="${OV_BUILD:-$REPO_DIR/build-ov-$HOST}"
CPU_BUILD="${CPU_BUILD:-$REPO_DIR/build-cpu-$HOST}"
JOBS="${JOBS:-$(nproc)}"
OV_ROOT="${OV_ROOT:-$HOME/intel/openvino}"
LLAMA_SRC_MIRROR="${LLAMA_SRC_MIRROR:-$HOME/vla-deps/llama-src}"
TASK="${TASK:-libero_object}"
CKPT="${CKPT:-$HOME/.cache/vla/vrfai/bitvla-libero-gguf/$TASK/bitvla-${TASK//_/-}-bf16.gguf}"
# The bar the other ten architectures are held to (docs/backend/ov.md).
TOL="${TOL:-0.0029}"
IMG_SIZE="${IMG_SIZE:-224}"
ITERS="${ITERS:-0}"
OUT="${OUT:-$REPO_DIR/outputs/ov_e2e_${SLURM_JOB_ID:-0}}"

mkdir -p "$(dirname "$OUT")"

echo "NODE: $HOST"
echo "CPU: $(lscpu | sed -n 's/^Model name: *//p' | head -1)"
echo "CKPT: $CKPT"
echo "TOL: $TOL"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

[ -f "$CKPT" ] || fail "no checkpoint at $CKPT -- run ci/slurm/bitvla_bf16_gguf.sbatch first"

# --- toolchain --------------------------------------------------------------
source /swtools/intel-gpu/latest/intel_gpu_vars.sh || fail "TOOLCHAIN intel_gpu_vars.sh"
[ -f "$OV_ROOT/setupvars.sh" ] || fail "no OpenVINO runtime at $OV_ROOT"
source "$OV_ROOT/setupvars.sh" >/dev/null 2>&1 || fail "TOOLCHAIN setupvars.sh"
export OCL_ICD_VENDORS="${OCL_ICD_VENDORS:-/swtools/intel-gpu/latest/neo}"
# Cached OpenVINO blobs reload a graph that computes the wrong thing; see
# docs/backend/ov.md. Never set this to make first-compile latency look better.
unset GGML_OPENVINO_CACHE_DIR

# --- builds -----------------------------------------------------------------
if [ ! -d "$LLAMA_SRC_MIRROR" ]; then
  for d in "$REPO_DIR"/build-*/_deps/llama-src; do
    [ -d "$d" ] || continue
    echo "MIRROR: seeding $LLAMA_SRC_MIRROR from $d"
    mkdir -p "$(dirname "$LLAMA_SRC_MIRROR")"
    cp -a "$d" "$LLAMA_SRC_MIRROR" || fail "MIRROR copy"
    break
  done
fi
FC_ARGS=()
[ -d "$LLAMA_SRC_MIRROR" ] && FC_ARGS=(-DFETCHCONTENT_SOURCE_DIR_LLAMA="$LLAMA_SRC_MIRROR")

build_one() {  # $1 = dir, $2.. = extra cmake args
  local bd="$1"; shift
  if [ ! -f "$bd/CMakeCache.txt" ]; then
    echo "CONFIGURE: $bd"
    cmake -B "$bd" -S "$REPO_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DVLA_BUILD_TESTS=ON \
        -DVLA_BUILD_SERVERS=OFF \
        "${FC_ARGS[@]}" "$@" >/dev/null 2>&1 || return 1
  else
    echo "CONFIGURE: reusing $bd"
  fi
  echo "BUILD: $bd -j$JOBS"
  cmake --build "$bd" -j"$JOBS" --target vla_predict_check 2>&1 | tail -5
  return "${PIPESTATUS[0]}"
}

build_one "$OV_BUILD"  -DGGML_OPENVINO=ON || fail "BUILD ov"
build_one "$CPU_BUILD"                    || fail "BUILD cpu"

# --- arms -------------------------------------------------------------------
# Same binary family, same fixed images/state/noise; the only difference is which
# backend is under the identical ggml graph. stdout is `action_len=N` then N floats.
run_arm() {   # $1 = tag, $2 = build dir
  local tag="$1" bd="$2"
  echo
  echo "=== predict_check [$tag] ==="
  # The empty second argument is load-bearing: predict_check reads argv[2] as an
  # mmproj path whenever it is non-empty and not a flag, so `<ckpt> 1` would make
  # "1" the mmproj and leave n_images at its default.
  VLA_IMG_SIZE="$IMG_SIZE" VLA_BENCH_ITERS="$ITERS" \
      "$bd/tests/vla_predict_check" "$CKPT" "" 1 \
      > "$OUT.$tag.txt" 2> "$OUT.$tag.err"
  local rc=$?
  grep -hE "backend =|using device|packed int2|weights =|error|failed" \
      "$OUT.$tag.txt" "$OUT.$tag.err" | sort -u | head -8
  echo "RC[$tag]: $rc"
  return $rc
}

run_arm cpu  "$CPU_BUILD" || fail "RUN cpu"
run_arm cpu2 "$CPU_BUILD" || fail "RUN cpu2"
( export GGML_OPENVINO_DEVICE=CPU
  run_arm ovcpu "$OV_BUILD" ) || fail "RUN ovcpu"
( export GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_GPU_PRECISION=f32
  run_arm ovgpu32 "$OV_BUILD" ) || fail "RUN ovgpu32"
( export GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_GPU_PRECISION=f16
  run_arm ovgpu16 "$OV_BUILD" ) || fail "RUN ovgpu16"

# Without these the comparison is vacuous: an OpenVINO arm that quietly fell back
# would agree with the reference perfectly while testing nothing, and a reference
# that is itself OpenVINO is not a reference.
grep -qh "backend = OPENVINO" "$OUT.cpu.txt" "$OUT.cpu.err" &&
    fail "REFERENCE ARM USED OPENVINO -- CPU_BUILD has the backend compiled in"
for tag in ovcpu ovgpu32 ovgpu16; do
  grep -qh "backend = OPENVINO" "$OUT.$tag.txt" "$OUT.$tag.err" ||
      fail "$tag DID NOT USE OPENVINO"
done
for tag in ovgpu32 ovgpu16; do
  grep -qh "using device GPU" "$OUT.$tag.txt" "$OUT.$tag.err" ||
      fail "$tag DID NOT REACH THE GPU PLUGIN"
done

# --- compare ----------------------------------------------------------------
DIMS="$(sed -n 's/.*max_action=\([0-9]*\).*/\1/p' "$OUT.cpu.err" | head -1)"
echo; echo "DIMS: ${DIMS:-unknown}"

cmp_arm() {   # $1 = tag, $2 = tol, $3 = gating(1/0)
  local tag="$1" tol="$2" gating="$3"
  echo
  echo "--- ggml-CPU vs $tag (tol $tol)${gating:+}$([ "$gating" = 0 ] && echo ' [not gating]') ---"
  python3 "$REPO_DIR/ci/compare_actions.py" "$OUT.cpu.txt" "$OUT.$tag.txt" \
      --tol "$tol" --dims "${DIMS:-0}" --label-a cpu --label-b "$tag"
  return $?
}

rc=0
# predict_check builds its inputs from fixed formulas, so a repeat must reproduce
# exactly. If this fails nothing below it means anything.
cmp_arm cpu2 0.0 1 || fail "REFERENCE IS NOT DETERMINISTIC"

cmp_arm ovcpu   "$TOL" 1 || { echo "GATE: ovcpu FAILED";   rc=1; }
cmp_arm ovgpu32 "$TOL" 1 || { echo "GATE: ovgpu32 FAILED"; rc=1; }

# Not gating: this is the measurement that decides the default, not a pass/fail.
if cmp_arm ovgpu16 "$TOL" 0; then
  echo "DECISION: GPU f16 is inside $TOL -- drop bitvla from the"
  echo "          GGML_OPENVINO_GPU_PRECISION=f32 default in src/backend.h."
else
  echo "DECISION: GPU f16 is outside $TOL -- keep the f32 default in src/backend.h,"
  echo "          and it is now a measurement rather than a prediction."
fi

[ "$rc" = "0" ] || fail "GATE"
echo
echo "VERDICT: PASSED (ovcpu, ovgpu32 within $TOL of ggml-CPU)"
echo "DONE"
