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
# bmg_ops.sbatch -- the inner loop for porting a kernel to Battlemage.
#
# Builds one target and runs one binary on a b70 node, and nothing else. The
# full bmg_build + bmg_run pair takes minutes and fetches checkpoints; while a
# kernel is being written what you want back is a pass/fail and a number, so
# this is the script to iterate against.
#
#   sbatch ci/slurm/bmg_ops.sbatch
#   TARGET=test_bitvla_ops_gpu sbatch ci/slurm/bmg_ops.sbatch
#
# Knobs: BUILD_DIR, TARGET (also the binary to run), BIN (if it differs from
# TARGET), ARGS, JOBS.
# =============================================================================
#SBATCH --job-name=bmg_ops
#SBATCH --partition=b70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH --output=slurm_bmg_ops_%j.out

# No `set -u`: both vendor scripts read unset variables.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-sycl}"
JOBS="${JOBS:-$(nproc)}"
TARGET="${TARGET:-test_bitvla_ops_gpu}"
BIN="${BIN:-tests/$TARGET}"

echo "NODE: $(hostname)"
echo "TARGET: $TARGET"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

source /swtools/intel-gpu/latest/intel_gpu_vars.sh || fail "TOOLCHAIN intel_gpu_vars.sh"
source /swtools/intel/setvars.sh --force >/dev/null 2>&1 || fail "TOOLCHAIN setvars.sh"
command -v icpx >/dev/null 2>&1 || fail "TOOLCHAIN icpx not on PATH"

[ -f "$BUILD_DIR/CMakeCache.txt" ] || fail "no $BUILD_DIR -- run ci/slurm/bmg_build.sbatch first"

# Re-run cmake so a new source or test registration is picked up without a
# reconfigure from scratch.
cmake -B "$BUILD_DIR" -S "$REPO_DIR" >/dev/null 2>&1 || fail "RECONFIGURE"
cmake --build "$BUILD_DIR" -j"$JOBS" --target "$TARGET" 2>&1 || fail "BUILD"
echo "BUILD: ok"

echo
echo "=== $BIN $ARGS ==="
# shellcheck disable=SC2086  # ARGS is an intentional word-split list
"$BUILD_DIR/$BIN" $ARGS 2>&1
rc=$?
echo "RUN_RC: $rc"

[ "$rc" = "0" ] && echo "VERDICT: OK" || echo "VERDICT: FAILED RUN"
echo "DONE"
