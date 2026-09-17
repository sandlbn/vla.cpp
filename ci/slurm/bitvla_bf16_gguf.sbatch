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
# bitvla_bf16_gguf.sbatch -- dense or block-quantised BitVLA GGUFs, from int2.
#
# The OpenVINO port needs a BitVLA GGUF whose weights are not ladder-packed
# int2. The supported way to make one is `convert_bitvla_to_gguf.py
# --weight-dtype ...`, which needs torch and the original HF checkpoint; this
# cluster has neither and no package manager to get them.
# `transcode_bitvla_int2.py` reaches the same files from the published int2
# GGUF instead -- see its docstring for why that is exact rather than
# approximate, for all three flavours.
#
#   bf16   2.0000 bytes/weight   the dense one every backend reads
#   q8_0   1.0625               23x more faithful than bf16, and half the size
#   q4_0   0.5625               d = s/7, so {-s,0,+s} land on q = {1,8,15}
#
# Pure numpy and no GPU, so this runs on a plain CPU partition. Each flavour is
# a few minutes of index arithmetic over ~2.5G ternary values.
#
#   sbatch ci/slurm/bitvla_bf16_gguf.sbatch
#   DTYPES="bf16 q8_0 q4_0" sbatch ci/slurm/bitvla_bf16_gguf.sbatch
#   TASK=libero_spatial sbatch ci/slurm/bitvla_bf16_gguf.sbatch
#
# Knobs: TASK (default libero_object), DTYPES (default bf16), SRC, GGUF_PY.
# DST is derived per flavour and cannot be overridden when DTYPES names more
# than one; a single flavour still honours it.
# =============================================================================
#SBATCH --job-name=bitvla_bf16_gguf
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=slurm_bitvla_bf16_gguf_%j.out

set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
TASK="${TASK:-libero_object}"
DTYPES="${DTYPES:-bf16}"
CACHE="$HOME/.cache/vla/vrfai/bitvla-libero-gguf/$TASK"
SRC="${SRC:-$CACHE/bitvla-${TASK//_/-}.gguf}"

echo "NODE:   $(hostname)"
echo "TASK:   $TASK"
echo "DTYPES: $DTYPES"
echo "SRC:    $SRC"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

[ -f "$SRC" ] || fail "no source checkpoint at $SRC"

# The `gguf` package is not installed here, but llama.cpp vendors it as pure
# python over numpy and any configured build tree has a copy. Prefer an
# installed one if there is somehow one on the path.
if ! python3 -c "import gguf" >/dev/null 2>&1; then
  for d in "$REPO_DIR"/build-*/_deps/llama-src/gguf-py; do
    [ -d "$d" ] && { GGUF_PY="${GGUF_PY:-$d}"; break; }
  done
  [ -n "${GGUF_PY:-}" ] || fail "no gguf module and no vendored gguf-py under $REPO_DIR/build-*"
  export PYTHONPATH="$GGUF_PY${PYTHONPATH:+:$PYTHONPATH}"
  echo "GGUF_PY: $GGUF_PY"
fi
python3 -c "import gguf, numpy; print('gguf ok, numpy', numpy.__version__)" || fail "gguf import"

df -h "$CACHE" | tail -1

n_dtypes=$(echo $DTYPES | wc -w)
written=()

for dtype in $DTYPES; do
  # bf16 keeps the historical name so nothing that already points at it breaks.
  dst="$CACHE/bitvla-${TASK//_/-}-$dtype.gguf"
  [ "$n_dtypes" = 1 ] && dst="${DST:-$dst}"

  echo
  echo "--- transcode $dtype -> $dst ---"
  python3 "$REPO_DIR/scripts/transcode_bitvla_int2.py" \
      --weight-dtype "$dtype" "$SRC" "$dst" 2>&1 || fail "TRANSCODE $dtype"
  written+=("$dst")
done

echo
echo "--- sizes ---"
ls -l "$SRC" "${written[@]}" | awk '{printf "  %-12s %s\n", $5, $9}'

echo "VERDICT: WROTE ${#written[@]} file(s)"
for f in "${written[@]}"; do echo "  $f"; done
echo "DONE"
