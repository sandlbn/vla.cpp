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
# bmg_gdb.sbatch -- run one binary on b70 under gdb and print a backtrace.
#
# A SIGSEGV from a batch job is otherwise just "exit 139" in the log, and the
# interesting frames are in a .so that is not on this login node. This runs the
# binary in batch-mode gdb so the stack comes back in the job output.
#
#   BIN=tests/vla_predict_check ARGS="$CKPT '' 1" sbatch ci/slurm/bmg_gdb.sbatch
#
# Knobs: BUILD_DIR, BIN, ARGS, ENV (extra VAR=VAL exports, space-separated).
# =============================================================================
#SBATCH --job-name=bmg_gdb
#SBATCH --partition=b70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_bmg_gdb_%j.out

# No `set -u`: both vendor scripts read unset variables.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-sycl}"
BIN="${BIN:-vla-bench}"

echo "NODE: $(hostname)"
echo "BIN: $BIN $ARGS"

source /swtools/intel-gpu/latest/intel_gpu_vars.sh
source /swtools/intel/setvars.sh --force >/dev/null 2>&1

# shellcheck disable=SC2086  # ENV is an intentional word-split list
[ -n "$ENV" ] && export $ENV

command -v gdb >/dev/null 2>&1 || { echo "VERDICT: FAILED no gdb"; echo "DONE"; exit 1; }

# SYCL defers JIT to first submit, so a crash there reports inside the runtime
# unless the kernels are built up front.
export SYCL_CACHE_PERSISTENT=1

# shellcheck disable=SC2086  # ARGS is an intentional word-split list
gdb --batch \
    -ex "set pagination off" \
    -ex "run" \
    -ex "echo \n==== backtrace ====\n" \
    -ex "bt 40" \
    -ex "echo \n==== locals ====\n" \
    -ex "info locals" \
    --args "$BUILD_DIR/$BIN" $ARGS 2>&1 | tail -80

echo "DONE"
