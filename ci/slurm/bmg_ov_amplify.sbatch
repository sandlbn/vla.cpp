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
# bmg_ov_amplify.sbatch -- is BitVLA's LM amplifying, or is OpenVINO wrong?
#
# bmg_ov_bisect.sbatch (job 372735) localised the disagreement:
#
#   proprio_features   3.9e-07   exact        pure MLP, no act_quant
#   mm_proj_out        6.3e-03   0.12% of elements   ViT, 26 layers
#   ah_input           2.5e-01   91.7% of elements   LM prefill, 30 layers
#
# The proprio stage being exact says the backend, its GEMM and its elementwise
# ops are sound. The ViT error is sparse -- a tenth of a percent of elements --
# which is the signature of a threshold flip rather than a systematic bias:
# act_quant rounds against 127/amax, Round is a threshold, and a sub-ULP
# disagreement moves an integer instead of a low bit. The LM then turns 6.3e-03
# into 2.5e-01 across nearly every element.
#
# That leaves two readings, and they call for opposite responses:
#
#   (a) the LM is amplifying. ~200 hard thresholds in the forward pass, and a
#       perturbation the size of ordinary backend disagreement grows through
#       them. Then no backend that is not bit-exact with ggml-CPU can hold
#       2.9e-3 on this architecture, the bar is the wrong instrument, and task
#       success rate is what the port has to be judged on.
#   (b) the LM translation is wrong, independently of the ViT.
#
# predict_check's VLA_PRECOMP_IMG_EMB separates them, because it lets the LM be
# fed a chosen embedding with the vision tower held out:
#
#   AMPLIFICATION  cpu_cpuemb vs cpu_ovemb -- one backend, ggml-CPU, for both
#                  runs; the only difference is a ViT output that differs by
#                  6.3e-03. No OpenVINO anywhere in the measurement. Whatever
#                  this shows is the model's own sensitivity.
#   LM FIDELITY    cpu_ovemb vs ov_ovemb -- identical input embeddings, one
#                  through ggml-CPU and one through OpenVINO. This is the LM
#                  translation on its own, with the ViT difference removed.
#   CONTROL        cpu vs cpu_cpuemb -- replaying ggml-CPU's own embeddings must
#                  reproduce ggml-CPU's own actions. If it does not, the
#                  precomputed-embedding path itself perturbs the model and
#                  nothing below it means anything.
#
# Pinned to pcl-adl01: per-host builds (GGML_NATIVE bakes in -march=native and
# the b70 partition is three microarchitectures), and this is the node the
# failing numbers were measured on.
#
#   sbatch ci/slurm/bmg_ov_amplify.sbatch
#
# Knobs: CKPT, TASK, TOL, OV_ROOT, IMG_SIZE.
# =============================================================================
#SBATCH --job-name=bmg_ov_amplify
#SBATCH --partition=b70
#SBATCH --nodelist=pcl-adl01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm_bmg_ov_amplify_%j.out

# No `set -u`: intel_gpu_vars.sh and OpenVINO's setupvars.sh both read unset
# variables, so nounset halts before the toolchain is on PATH.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
HOST="$(hostname -s)"
OV_BUILD="${OV_BUILD:-$REPO_DIR/build-ov-$HOST}"
CPU_BUILD="${CPU_BUILD:-$REPO_DIR/build-cpu-$HOST}"
JOBS="${JOBS:-$(nproc)}"
OV_ROOT="${OV_ROOT:-$HOME/intel/openvino}"
TASK="${TASK:-libero_object}"
CKPT="${CKPT:-$HOME/.cache/vla/vrfai/bitvla-libero-gguf/$TASK/bitvla-${TASK//_/-}-bf16.gguf}"
TOL="${TOL:-0.0029}"
IMG_SIZE="${IMG_SIZE:-224}"
OUT="${OUT:-$REPO_DIR/outputs/ov_amp_${SLURM_JOB_ID:-0}}"

echo "NODE: $HOST"
echo "CKPT: $CKPT"
echo "TOL:  $TOL"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

[ -f "$CKPT" ] || fail "no checkpoint at $CKPT"
[ -f "$OV_BUILD/CMakeCache.txt" ]  || fail "no $OV_BUILD -- run bmg_ov_e2e.sbatch first"
[ -f "$CPU_BUILD/CMakeCache.txt" ] || fail "no $CPU_BUILD -- run bmg_ov_e2e.sbatch first"

# --- toolchain --------------------------------------------------------------
source /swtools/intel-gpu/latest/intel_gpu_vars.sh || fail "TOOLCHAIN intel_gpu_vars.sh"
[ -f "$OV_ROOT/setupvars.sh" ] || fail "no OpenVINO runtime at $OV_ROOT"
source "$OV_ROOT/setupvars.sh" >/dev/null 2>&1 || fail "TOOLCHAIN setupvars.sh"
export OCL_ICD_VENDORS="${OCL_ICD_VENDORS:-/swtools/intel-gpu/latest/neo}"
unset GGML_OPENVINO_CACHE_DIR

# predict_check gained VLA_PRECOMP_IMG_EMB for this job; both builds need it.
for bd in "$OV_BUILD" "$CPU_BUILD"; do
  echo "BUILD: $bd"
  cmake --build "$bd" -j"$JOBS" --target vla_predict_check 2>&1 | tail -3
  [ "${PIPESTATUS[0]}" = 0 ] || fail "BUILD $bd"
done

mkdir -p "$(dirname "$OUT")"

# --- arms -------------------------------------------------------------------
# Separate `local`s on purpose: bash expands the whole command line before the
# builtin assigns, so `local tag="$1" dd="$OUT.$tag.dump"` reads an unset $tag.
run_arm() {   # $1 = tag, $2 = build dir, $3 = precomputed emb file or ""
  local tag="$1"
  local bd="$2"
  local emb="$3"
  local dd="$OUT.$tag.dump"
  rm -rf "$dd"; mkdir -p "$dd"
  echo
  echo "=== predict_check [$tag] ${emb:+<- $(basename "$(dirname "$emb")")/$(basename "$emb")} ==="
  # The empty second argument is load-bearing: predict_check reads argv[2] as an
  # mmproj path whenever it is non-empty and not a flag.
  env VLA_BITVLA_DUMP_DIR="$dd" VLA_IMG_SIZE="$IMG_SIZE" \
      ${emb:+VLA_PRECOMP_IMG_EMB="$emb"} ${emb:+VLA_PRECOMP_VIEWS=1} \
      "$bd/tests/vla_predict_check" "$CKPT" "" 1 \
      > "$OUT.$tag.txt" 2> "$OUT.$tag.err"
  local rc=$?
  grep -hE "backend =|using device|precomputed img emb|error|failed" \
      "$OUT.$tag.txt" "$OUT.$tag.err" | cut -c1-140 | sort -u | head -5
  echo "RC[$tag]: $rc"
  return $rc
}

# Stage 1: the two vision towers, to get the two embeddings to replay.
run_arm cpu "$CPU_BUILD" "" || fail "RUN cpu"
( export GGML_OPENVINO_DEVICE=CPU; run_arm ovcpu "$OV_BUILD" "" ) || fail "RUN ovcpu"

grep -qh "backend = OPENVINO" "$OUT.cpu.txt" "$OUT.cpu.err" && fail "REFERENCE ARM USED OPENVINO"
grep -qh "backend = OPENVINO" "$OUT.ovcpu.txt" "$OUT.ovcpu.err" || fail "ovcpu DID NOT USE OPENVINO"

CPU_EMB="$OUT.cpu.dump/mm_proj_out.bin"
OV_EMB="$OUT.ovcpu.dump/mm_proj_out.bin"
[ -s "$CPU_EMB" ] || fail "no $CPU_EMB"
[ -s "$OV_EMB" ]  || fail "no $OV_EMB"
echo
echo "EMBEDDINGS: cpu=$(stat -c%s "$CPU_EMB") bytes  ovcpu=$(stat -c%s "$OV_EMB") bytes"

# Stage 2: replay both embeddings through ggml-CPU. No OpenVINO in either run.
run_arm cpu_cpuemb "$CPU_BUILD" "$CPU_EMB" || fail "RUN cpu_cpuemb"
run_arm cpu_ovemb  "$CPU_BUILD" "$OV_EMB"  || fail "RUN cpu_ovemb"

# Stage 3: the OpenVINO LM on the same embeddings ggml-CPU just saw.
( export GGML_OPENVINO_DEVICE=CPU; run_arm ov_ovemb "$OV_BUILD" "$OV_EMB" ) || fail "RUN ov_ovemb"
grep -qh "backend = OPENVINO" "$OUT.ov_ovemb.txt" "$OUT.ov_ovemb.err" || fail "ov_ovemb DID NOT USE OPENVINO"

# --- compare ----------------------------------------------------------------
DIMS="$(sed -n 's/.*max_action=\([0-9]*\).*/\1/p' "$OUT.cpu.err" | head -1)"
echo; echo "DIMS: ${DIMS:-unknown}"

cmp2() {   # $1 = title, $2 = tag a, $3 = tag b
  echo
  echo "======================================================================"
  echo "$1"
  echo "  actions: $2 vs $3"
  echo "======================================================================"
  python3 "$REPO_DIR/ci/compare_actions.py" "$OUT.$2.txt" "$OUT.$3.txt" \
      --tol "$TOL" --dims "${DIMS:-0}" --label-a "$2" --label-b "$3"
  echo "  stages:"
  python3 "$REPO_DIR/ci/compare_dumps.py" "$OUT.$2.dump" "$OUT.$3.dump" \
      --tol "$TOL" --label-a "$2" --label-b "$3" | sed 's/^/    /'
}

cmp2 "CONTROL -- does replaying an embedding change the answer? (want: no)" \
     cpu cpu_cpuemb
cmp2 "AMPLIFICATION -- ggml-CPU both sides, ViT outputs differ by 6.3e-03" \
     cpu_cpuemb cpu_ovemb
cmp2 "LM FIDELITY -- identical embeddings, ggml-CPU vs OpenVINO" \
     cpu_ovemb ov_ovemb

echo
echo "READ IT LIKE THIS:"
echo "  CONTROL large          -> the precomp path perturbs the model; ignore the rest."
echo "  AMPLIFICATION ~ 0.1-0.2 and LM FIDELITY small"
echo "                         -> reading (a): BitVLA amplifies, OpenVINO is faithful,"
echo "                            and the 2.9e-3 action bar cannot bind on this arch."
echo "  AMPLIFICATION small and LM FIDELITY large"
echo "                         -> reading (b): the LM translation is wrong. Bisect it."
echo "DONE"
