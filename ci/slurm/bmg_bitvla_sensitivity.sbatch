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
# bmg_bitvla_sensitivity.sbatch -- how much does BitVLA amplify a small input change?
#
# This job contains no OpenVINO at all. It runs one build, ggml's CPU backend,
# against itself. That is the point: it measures a property of the model, and a
# property of the model is not allowed to depend on which backend measured it.
#
# Why it is needed. bmg_ov_amplify.sbatch (job 372736) showed
#
#   CONTROL        cpu vs cpu_cpuemb    0.0000   replay is faithful
#   AMPLIFICATION  cpu_cpuemb/cpu_ovemb 0.1395   ggml-CPU both sides, ViT in differs 6.3e-03
#   LM FIDELITY    cpu_ovemb/ov_ovemb   0.0702   identical input, CPU vs OpenVINO
#
# so a 6.3e-03 difference at the LM's input becomes 0.14 at its output with no
# OpenVINO involved. But that does not yet settle whether the 0.0702 is the LM
# being mistranslated or the same amplification acting on ordinary op-level
# noise, because the amplification factor at *small* perturbations was never
# measured -- 6.3e-03 is not small.
#
# So: perturb the LM's input embeddings by a known relative eps and sweep it
# across seven decades. The shape of the curve is the answer.
#
#   LINEAR      output deviation tracks eps (1e-6 in -> ~1e-5 out). The model is
#               well conditioned, amplification is a modest constant, and
#               OpenVINO's 0.0702 on identical inputs is a real defect that has
#               to be found and fixed.
#   SATURATING  output deviation is ~0.1 for every eps down to 1e-6 or below.
#               Then ~200 Round thresholds have decorrelated the result and any
#               two implementations that are not bit-exact land this far apart.
#               The 2.9e-3 action bar is then measuring the model's conditioning
#               rather than the port's fidelity, and the port has to be judged on
#               task success instead.
#
# The eps ladder brackets the quantities that matter: 1.2e-7 is one f32 ULP,
# 1e-3 is roughly bf16's resolution, and 6.3e-3 is the ViT difference actually
# observed, included so the sweep reproduces the number it is explaining.
#
# Pinned to pcl-adl01 to reuse build-cpu-pcl-adl01; GGML_NATIVE bakes in
# -march=native and the b70 partition is three microarchitectures.
#
#   sbatch ci/slurm/bmg_bitvla_sensitivity.sbatch
#
# Knobs: CKPT, TASK, EPS_LIST, CPU_BUILD, IMG_SIZE.
# =============================================================================
#SBATCH --job-name=bitvla_sens
#SBATCH --partition=b70
#SBATCH --nodelist=pcl-adl01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm_bitvla_sens_%j.out

set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
HOST="$(hostname -s)"
CPU_BUILD="${CPU_BUILD:-$REPO_DIR/build-cpu-$HOST}"
JOBS="${JOBS:-$(nproc)}"
TASK="${TASK:-libero_object}"
CKPT="${CKPT:-$HOME/.cache/vla/vrfai/bitvla-libero-gguf/$TASK/bitvla-${TASK//_/-}-bf16.gguf}"
IMG_SIZE="${IMG_SIZE:-224}"
# 1.2e-07 is one f32 ULP; 1e-03 is about bf16's resolution; 6.3e-03 is the ViT
# difference job 372735 measured between ggml-CPU and the OpenVINO CPU plugin.
EPS_LIST="${EPS_LIST:-0 1.2e-07 1e-06 1e-05 1e-04 1e-03 6.3e-03}"
OUT="${OUT:-$REPO_DIR/outputs/bitvla_sens_${SLURM_JOB_ID:-0}}"

echo "NODE: $HOST"
echo "CKPT: $CKPT"
echo "EPS:  $EPS_LIST"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

[ -f "$CKPT" ] || fail "no checkpoint at $CKPT"
[ -f "$CPU_BUILD/CMakeCache.txt" ] || fail "no $CPU_BUILD -- run bmg_ov_e2e.sbatch first"

echo "BUILD: $CPU_BUILD"
cmake --build "$CPU_BUILD" -j"$JOBS" --target vla_predict_check 2>&1 | tail -3
[ "${PIPESTATUS[0]}" = 0 ] || fail "BUILD"

BIN="$CPU_BUILD/tests/vla_predict_check"
mkdir -p "$(dirname "$OUT")"

# --- baseline: run the real vision tower once to get the embeddings ----------
BASE_DUMP="$OUT.base.dump"
rm -rf "$BASE_DUMP"; mkdir -p "$BASE_DUMP"
echo
echo "=== baseline (real vision tower, ggml-CPU) ==="
VLA_BITVLA_DUMP_DIR="$BASE_DUMP" VLA_IMG_SIZE="$IMG_SIZE" \
    "$BIN" "$CKPT" "" 1 > "$OUT.base.txt" 2> "$OUT.base.err" || fail "RUN base"
grep -qh "backend = OPENVINO" "$OUT.base.txt" "$OUT.base.err" &&
    fail "BASELINE USED OPENVINO -- $CPU_BUILD has the backend compiled in"
EMB="$BASE_DUMP/mm_proj_out.bin"
[ -s "$EMB" ] || fail "no $EMB"
DIMS="$(sed -n 's/.*max_action=\([0-9]*\).*/\1/p' "$OUT.base.err" | head -1)"
echo "EMB: $(stat -c%s "$EMB") bytes   DIMS: ${DIMS:-unknown}"

# --- sweep -------------------------------------------------------------------
# eps=0 is the control: a perturbation of nothing must reproduce the baseline
# exactly, which also proves the replay path is not itself perturbing anything.
echo
printf '%-12s %-14s %-14s %s\n' "eps" "in_normalised" "out_normalised" "amplification"
printf '%s\n' "----------------------------------------------------------------"

for eps in $EPS_LIST; do
  pert="$OUT.eps$eps.bin"
  in_norm="$(python3 "$REPO_DIR/ci/perturb_emb.py" "$EMB" "$pert" "$eps" \
             | sed -n 's/.*normalised=\([0-9.e+-]*\).*/\1/p')"
  [ -n "$in_norm" ] || fail "PERTURB $eps"

  dd="$OUT.eps$eps.dump"; rm -rf "$dd"; mkdir -p "$dd"
  VLA_BITVLA_DUMP_DIR="$dd" VLA_IMG_SIZE="$IMG_SIZE" \
      VLA_PRECOMP_IMG_EMB="$pert" VLA_PRECOMP_VIEWS=1 \
      "$BIN" "$CKPT" "" 1 > "$OUT.eps$eps.txt" 2> "$OUT.eps$eps.err" || fail "RUN eps=$eps"

  # Scored against the baseline actions, which came from the unperturbed
  # embeddings through the same LM.
  out_norm="$(python3 "$REPO_DIR/ci/compare_actions.py" "$OUT.base.txt" "$OUT.eps$eps.txt" \
              --tol 1.0 --dims "${DIMS:-0}" --label-a base --label-b "eps$eps" \
              | sed -n 's/.*normalised=\([0-9.e+-]*\).*/\1/p' | head -1)"

  amp="$(python3 -c "
i=float('${in_norm:-0}'); o=float('${out_norm:-0}')
print('n/a' if i == 0 else f'{o/i:.1f}x')")"
  printf '%-12s %-14s %-14s %s\n' "$eps" "${in_norm:-?}" "${out_norm:-?}" "$amp"
  rm -f "$pert"
done

echo
echo "READ IT LIKE THIS:"
echo "  out tracks eps (amplification roughly constant and small)"
echo "     -> BitVLA is well conditioned; OpenVINO's 0.0702 on identical inputs"
echo "        is a real defect and must be bisected further."
echo "  out is ~0.1 even at eps=1e-06 (amplification exploding as eps shrinks)"
echo "     -> BitVLA decorrelates under any non-bit-exact arithmetic; the 2.9e-3"
echo "        action bar measures conditioning, not fidelity, and task success"
echo "        rate is the instrument the port has to be judged with."
echo "DONE"
