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
# plot_ov_quant.sbatch -- redraw the BitVLA weight-flavour figure.
#
# ci/slurm/bmg_ov_libero.sbatch draws this figure at the end of every sweep, so
# this job exists for the other case: changing how the figure looks, or adding an
# arm measured by an earlier sweep, without spending a day re-measuring what is
# already on disk. It reads outputs/libero_ov_b70 and writes docs/img.
#
# It holds the B70 for a few seconds and computes nothing -- it is a job only
# because measurement and the plots drawn from it are not done on the login node.
#
#   sbatch ci/slurm/plot_ov_quant.sbatch
#   ARMS="q8_0:bitvla-ov-q8_0-f16 q4_0:bitvla-ov-q4_0-f16" sbatch ci/slurm/plot_ov_quant.sbatch
#
# ARMS is "label:dirname" in ladder order, widest weights first. The default is
# the full ladder: the two dense arms from Stage A, then the three block-quantised
# ones, which is the comparison the figure is for.
#
# Knobs: ARMS, OUTPUT_ROOT, OUT, TITLE.
# =============================================================================
#SBATCH --job-name=plot_ov_quant
#SBATCH --partition=b70
#SBATCH --nodelist=pcl-adl01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm_plot_ov_quant_%j.out

set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/libero_ov_b70}"
VENV_PY="$REPO_DIR/eval/sim/libero/libero_uv/.venv/bin/python"
OUT="${OUT:-$REPO_DIR/docs/img/bitvla_ov_quant_b70}"
TITLE="${TITLE:-BitVLA weight flavours on OpenVINO GPU, LIBERO object, Arc Pro B70}"
ARMS="${ARMS:-f32:bitvla-ov-f32 f16:bitvla-ov-f16 bf16:bitvla-ov-bf16-f16 q8_0:bitvla-ov-q8_0-f16 q4_0:bitvla-ov-q4_0-f16}"

echo "NODE: $(hostname -s)"
echo "ROOT: $OUTPUT_ROOT"
echo "OUT:  $OUT"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

[ -x "$VENV_PY" ] || fail "no LIBERO venv at $VENV_PY"

PLOT_ARMS=()
for a in $ARMS; do
  label="${a%%:*}"
  dir="$OUTPUT_ROOT/${a##*:}"
  # Skipped rather than fatal: the usual reason an arm is missing is that this
  # machine has not run that sweep, and a figure of the arms that do exist is
  # more use than no figure at all. Named, so the gap is not silent.
  if [ -d "$dir" ]; then
    PLOT_ARMS+=(--arm "$label=$dir")
  else
    echo "  skipping $label -- no $dir"
  fi
done
[ "${#PLOT_ARMS[@]}" -gt 0 ] || fail "no arms found under $OUTPUT_ROOT"

echo
"$VENV_PY" "$REPO_DIR/scripts/plot_ov_quant.py" "${PLOT_ARMS[@]}" \
    --out "$OUT" --title "$TITLE" || fail "PLOT"

echo
echo "VERDICT: OK"
echo "DONE"
