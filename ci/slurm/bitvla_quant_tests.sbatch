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
# bitvla_quant_tests.sbatch -- the Q8_0/Q4_0 ternary encoders, both directions.
#
# Stage B teaches BitVLA to keep its weights small: ternary weights go into
# Q8_0 (1.0625 bytes/weight) or Q4_0 (0.5625) blocks instead of being carried
# in bf16 containers. Two encoders in scripts/bitvla_quant.py do it, and they
# are the only new numerics in the stage, so they get checked from both ends:
#
#   ci/test_bitvla_quant.py      round-trips through gguf-py, pins ggml's own
#                                Q4_0 quantiser as known-bad on ternary (12.5%,
#                                structural: d = max/-8 clips one sign), and
#                                writes tests/data/bitvla_q_blocks.bin.
#   ctest -R bitvla_quant        dequantises that same fixture through ggml's C
#                                to_float. gguf-py agreeing with itself says
#                                nothing about the Q4_0 nibble pairing, which
#                                puts element j next to j+16, not j+1.
#
# The fixture is generated here rather than checked in from a workstation, so
# the bytes under test are the bytes this cluster's numpy produces.
#
# This job is also the first compile of the Stage B loader change (the
# kernels_drive / bit_type split in src/models/bitvla.cpp), so it builds the
# whole CPU tree and runs the full ctest suite, not just the two new tests.
#
# Pinned to pcl-adl01 to reuse build-cpu-pcl-adl01; GGML_NATIVE bakes in
# -march=native and the partition is three microarchitectures.
#
#   sbatch ci/slurm/bitvla_quant_tests.sbatch
#
# Knobs: CPU_BUILD, JOBS, GGUF_PY.
# =============================================================================
#SBATCH --job-name=bitvla_quant
#SBATCH --partition=b70
#SBATCH --nodelist=pcl-adl01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_bitvla_quant_%j.out

set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
HOST="$(hostname -s)"
CPU_BUILD="${CPU_BUILD:-$REPO_DIR/build-cpu-$HOST}"
JOBS="${JOBS:-$(nproc)}"

echo "NODE:  $HOST"
echo "BUILD: $CPU_BUILD"

rc=0
fail() { echo "FAILED: $1"; rc=1; }

# --- python half -------------------------------------------------------------
# `gguf` is not installed here; llama.cpp vendors it as pure python over numpy
# and any configured build tree has a copy.
if ! python3 -c "import gguf" >/dev/null 2>&1; then
  for d in "$REPO_DIR"/build-*/_deps/llama-src/gguf-py; do
    [ -d "$d" ] && { GGUF_PY="${GGUF_PY:-$d}"; break; }
  done
  if [ -z "${GGUF_PY:-}" ]; then
    echo "VERDICT: FAILED no gguf module and no vendored gguf-py under $REPO_DIR/build-*"
    echo "DONE"
    exit 1
  fi
  export PYTHONPATH="$GGUF_PY${PYTHONPATH:+:$PYTHONPATH}"
  echo "GGUF_PY: $GGUF_PY"
fi
python3 -c "import gguf, numpy; print('gguf ok, numpy', numpy.__version__)" \
  || { echo "VERDICT: FAILED gguf import"; echo "DONE"; exit 1; }

echo
echo "--- regenerate the fixture ---"
python3 "$REPO_DIR/ci/test_bitvla_quant.py" --update-fixture 2>&1 || fail "FIXTURE"

echo
echo "--- ci/test_bitvla_quant.py ---"
python3 "$REPO_DIR/ci/test_bitvla_quant.py" 2>&1 || fail "PY_ENCODERS"

# --- c++ half ----------------------------------------------------------------
echo
echo "--- configure + build ---"
if [ ! -f "$CPU_BUILD/CMakeCache.txt" ]; then
  cmake -S "$REPO_DIR" -B "$CPU_BUILD" -DCMAKE_BUILD_TYPE=Release \
        -DVLA_BUILD_TESTS=ON 2>&1 | tail -5
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "VERDICT: FAILED CONFIGURE"; echo "DONE"; exit 1; }
fi
# The test is new, so an existing cache has never seen it; reconfigure cheaply.
cmake -S "$REPO_DIR" -B "$CPU_BUILD" -DVLA_BUILD_TESTS=ON 2>&1 | tail -3
[ "${PIPESTATUS[0]}" = 0 ] || { echo "VERDICT: FAILED RECONFIGURE"; echo "DONE"; exit 1; }

cmake --build "$CPU_BUILD" -j"$JOBS" 2>&1 | grep -Ei "error|warning: .*bitvla|warning: .*backend\.h" | head -40
[ "${PIPESTATUS[0]}" = 0 ] || fail "BUILD"

echo
echo "--- ctest ---"
( cd "$CPU_BUILD" && ctest --output-on-failure ) 2>&1 | tail -40
[ "${PIPESTATUS[0]}" = 0 ] || fail "CTEST"

echo
if [ "$rc" = 0 ]; then
  echo "VERDICT: PASSED  encoders agree with ggml, tree builds, ctest green"
else
  echo "VERDICT: FAILED  see FAILED: lines above"
fi
echo "DONE"
exit "$rc"
