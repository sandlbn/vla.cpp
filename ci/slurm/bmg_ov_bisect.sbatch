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
# bmg_ov_bisect.sbatch -- which of BitVLA's four graphs does OpenVINO get wrong?
#
# bmg_ov_e2e.sbatch scored the final action chunk and every OpenVINO arm missed
# the 2.9e-3 bar by two orders of magnitude, the CPU plugin at f32 included
# (0.1336). That rules precision out as the story: the CPU plugin at f32 against
# ggml's CPU backend, on a graph where both sides load the same weights at F32,
# should agree to about 1e-6. Something is being translated wrongly.
#
# Two facts make this cheap to localise. First, BitVLA's predict() is four
# separate ggml_backend_graph_compute calls (vision, proprio, LM prefill, action
# head) with host memory between them, not one fused graph. Second,
# VLA_BITVLA_DUMP_DIR already writes the tensor at each of those seams. So
# running the same binary family twice and diffing the dumps names the graph at
# fault without adding a line of instrumentation.
#
# `inputs_embeds` is the control in the middle: it is assembled on the host from
# mm_proj_out, proprio_features and the token-embedding fetch, so if the two
# stages before it agree and it does not, the fault is in the embedding fetch and
# not in any backend at all.
#
# Pinned to the node the failing measurement was taken on. The builds are
# per-host because GGML_NATIVE bakes in -march=native and the b70 partition is
# three different microarchitectures; pinning also means this reuses
# build-ov-pcl-adl01 / build-cpu-pcl-adl01 rather than spending twenty minutes
# rebuilding them, and compares against the same silicon that produced 0.1336.
#
#   sbatch ci/slurm/bmg_ov_bisect.sbatch
#
# Knobs: CKPT, TASK, ARMS (default "ovcpu ovgpu32"), TOL, OV_ROOT, IMG_SIZE.
# =============================================================================
#SBATCH --job-name=bmg_ov_bisect
#SBATCH --partition=b70
#SBATCH --nodelist=pcl-adl01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=slurm_bmg_ov_bisect_%j.out

# No `set -u`: intel_gpu_vars.sh and OpenVINO's setupvars.sh both read unset
# variables, so nounset halts before the toolchain is on PATH.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
HOST="$(hostname -s)"
OV_BUILD="${OV_BUILD:-$REPO_DIR/build-ov-$HOST}"
CPU_BUILD="${CPU_BUILD:-$REPO_DIR/build-cpu-$HOST}"
OV_ROOT="${OV_ROOT:-$HOME/intel/openvino}"
TASK="${TASK:-libero_object}"
CKPT="${CKPT:-$HOME/.cache/vla/vrfai/bitvla-libero-gguf/$TASK/bitvla-${TASK//_/-}-bf16.gguf}"
TOL="${TOL:-0.0029}"
IMG_SIZE="${IMG_SIZE:-224}"
ARMS="${ARMS:-ovcpu ovgpu32}"
OUT="${OUT:-$REPO_DIR/outputs/ov_bisect_${SLURM_JOB_ID:-0}}"

echo "NODE: $HOST"
echo "CKPT: $CKPT"
echo "TOL:  $TOL"
echo "ARMS: $ARMS"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

[ -f "$CKPT" ] || fail "no checkpoint at $CKPT"
# Reuse only. A rebuild here would mean the pin did not hold and the numbers
# would no longer be the ones bmg_ov_e2e measured.
[ -x "$OV_BUILD/tests/vla_predict_check" ]  || fail "no vla_predict_check in $OV_BUILD -- run bmg_ov_e2e.sbatch first"
[ -x "$CPU_BUILD/tests/vla_predict_check" ] || fail "no vla_predict_check in $CPU_BUILD -- run bmg_ov_e2e.sbatch first"

# --- toolchain --------------------------------------------------------------
source /swtools/intel-gpu/latest/intel_gpu_vars.sh || fail "TOOLCHAIN intel_gpu_vars.sh"
[ -f "$OV_ROOT/setupvars.sh" ] || fail "no OpenVINO runtime at $OV_ROOT"
source "$OV_ROOT/setupvars.sh" >/dev/null 2>&1 || fail "TOOLCHAIN setupvars.sh"
export OCL_ICD_VENDORS="${OCL_ICD_VENDORS:-/swtools/intel-gpu/latest/neo}"
# Cached blobs reload a graph that computes the wrong thing (docs/backend/ov.md).
unset GGML_OPENVINO_CACHE_DIR

mkdir -p "$(dirname "$OUT")"

# --- arms -------------------------------------------------------------------
run_arm() {   # $1 = tag, $2 = build dir
  # Three separate `local`s, not one. Bash expands the whole command line before
  # the builtin assigns, so `local tag="$1" dd="$OUT.$tag.dump"` reads $tag while
  # it is still unset and every arm lands in the same directory, silently
  # overwriting the arm before it.
  local tag="$1"
  local bd="$2"
  local dd="$OUT.$tag.dump"
  rm -rf "$dd"; mkdir -p "$dd"
  echo
  echo "=== predict_check [$tag] -> $dd ==="
  # The empty second argument is load-bearing: predict_check reads argv[2] as an
  # mmproj path whenever it is non-empty and not a flag.
  VLA_BITVLA_DUMP_DIR="$dd" VLA_IMG_SIZE="$IMG_SIZE" \
      "$bd/tests/vla_predict_check" "$CKPT" "" 1 \
      > "$OUT.$tag.txt" 2> "$OUT.$tag.err"
  local rc=$?
  grep -hE "backend =|using device|resident=|error|failed" \
      "$OUT.$tag.txt" "$OUT.$tag.err" | cut -c1-160 | sort -u | head -6
  echo "RC[$tag]: $rc  stages: $(ls "$dd"/*.bin 2>/dev/null | wc -l)"
  return $rc
}

run_arm cpu "$CPU_BUILD" || fail "RUN cpu"
grep -qh "backend = OPENVINO" "$OUT.cpu.txt" "$OUT.cpu.err" &&
    fail "REFERENCE ARM USED OPENVINO"

for arm in $ARMS; do
  case "$arm" in
    ovcpu)   ( export GGML_OPENVINO_DEVICE=CPU
               run_arm ovcpu "$OV_BUILD" ) ;;
    ovgpu32) ( export GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_GPU_PRECISION=f32
               run_arm ovgpu32 "$OV_BUILD" ) ;;
    ovgpu16) ( export GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_GPU_PRECISION=f16
               run_arm ovgpu16 "$OV_BUILD" ) ;;
    *) fail "unknown arm $arm" ;;
  esac
  [ $? = 0 ] || fail "RUN $arm"
  grep -qh "backend = OPENVINO" "$OUT.$arm.txt" "$OUT.$arm.err" ||
      fail "$arm DID NOT USE OPENVINO"
done

# --- compare ----------------------------------------------------------------
echo
echo "DUMP DIRS:"
ls -d "$OUT".*.dump | while read -r d; do
  echo "  $d  ($(ls "$d"/*.bin 2>/dev/null | wc -l) stages)"
done

for arm in $ARMS; do
  echo
  echo "======================================================================"
  echo "--- stage bisect: ggml-CPU vs $arm (tol $TOL) ---"
  echo "======================================================================"
  # rc is not gated on: a divergence is the result this job exists to find, not
  # a failure. An empty comparison is a different matter and compare_dumps.py
  # returns non-zero with NO STAGES COMPARED for it, which is worth seeing.
  python3 "$REPO_DIR/ci/compare_dumps.py" "$OUT.cpu.dump" "$OUT.$arm.dump" \
      --tol "$TOL" --label-a cpu --label-b "$arm"
done

echo
echo "VERDICT: BISECT COMPLETE (see FIRST DIVERGENCE above)"
echo "DONE"
