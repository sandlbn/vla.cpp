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
# bmg_preflight.sbatch -- what the Battlemage node actually offers.
#
# Establishes the ground truth the SYCL port is written against, before any of
# it exists: which device, how much of it, which sub-group sizes, whether the
# int8 DPAS path `joint_matrix` needs is present, and which oneDNN/DPC++ the
# /swtools toolchain hands us. Everything is printed as KEY: value so a driving
# agent reads keys rather than prose, and the script ends on DONE.
#
#   sbatch ci/slurm/bmg_preflight.sbatch
#
# Knobs: VLA_GPU_VARS, VLA_ONEAPI_VARS to point at a different toolchain.
# =============================================================================
#SBATCH --job-name=bmg_preflight
#SBATCH --partition=b70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=slurm_bmg_preflight_%j.out

# No `set -u`: the vendor's intel_gpu_vars.sh and oneAPI's setvars.sh both read
# unset variables (CPATH and friends), so nounset halts before the GPU is found.
set -o pipefail

echo "NODE: $(hostname)"
echo "DATE: $(date -Is)"

# --- toolchain from /swtools ------------------------------------------------
# Driver first: level-zero is how the GPU is found at all.
GPU_VARS="${VLA_GPU_VARS:-/swtools/intel-gpu/latest/intel_gpu_vars.sh}"
if [ -f "$GPU_VARS" ]; then
  # shellcheck disable=SC1090
  source "$GPU_VARS"
  echo "GPU_UMD: $GPU_VARS -> $(readlink -f "$(dirname "$GPU_VARS")")"
else
  echo "GPU_UMD: ABSENT at $GPU_VARS"
fi

ONEAPI="${VLA_ONEAPI_VARS:-/swtools/intel/setvars.sh}"
if [ -f "$ONEAPI" ]; then
  # shellcheck disable=SC1090
  source "$ONEAPI" --force >/dev/null 2>&1 && echo "ONEAPI: $ONEAPI"
else
  echo "ONEAPI: ABSENT at $ONEAPI"
fi

echo "ICPX: $(command -v icpx || echo none)"
icpx --version 2>/dev/null | head -2 | sed 's/^/ICPX_VERSION: /'

# --- what the runtime sees --------------------------------------------------
echo
echo "=== sycl-ls ==="
sycl-ls 2>&1 || echo "SYCL_LS: failed"

echo
echo "=== render nodes ==="
ls -l /dev/dri/ 2>&1 | sed 's/^/DRI: /'

# --- oneDNN -----------------------------------------------------------------
# Located rather than assumed: the port's dense GEMMs and the reference ternary
# path both link it, and the header/library pair has to come from the same
# oneAPI as icpx.
# ONEAPI_ROOT is the parent of the component tree (/data/swtools/intel), not a
# version directory, and there is no `latest` symlink beside it -- oneDNN sits in
# a per-release unified dir ($ONEAPI_ROOT/2026.0). Search rather than guess, and
# prefer the release whose version matches the icpx that setvars just put on PATH.
echo "ONEAPI_ROOT: ${ONEAPI_ROOT:-unset}"
DNNL_H=$(find "${ONEAPI_ROOT:-/swtools/intel}" -maxdepth 3 -name dnnl_version.h 2>/dev/null \
         | sort -V | tail -1)
if [ -n "$DNNL_H" ]; then
  DNNL_ROOT="$(dirname "$(dirname "$DNNL_H")")"
  echo "ONEDNN_ROOT: $DNNL_ROOT"
  echo "ONEDNN_HEADER: $DNNL_H"
  grep -E "define DNNL_VERSION_(MAJOR|MINOR|PATCH)" "$DNNL_H" 2>/dev/null | sed 's/^/ONEDNN_/'
  echo "LIBDNNL: $(ls "$DNNL_ROOT"/lib/libdnnl.so.*.* 2>/dev/null | head -1 || echo none)"
else
  echo "ONEDNN_HEADER: not found under ${ONEAPI_ROOT:-/swtools/intel}"
  echo "LIBDNNL: none"
fi

# --- device capability probe ------------------------------------------------
# The three facts the kernel design depends on and that no doc can settle for
# this exact part: global memory (does the checkpoint fit), the supported
# sub-group sizes (the M=1 decode kernel pins one), and whether the int8 DPAS
# combination joint_matrix needs is actually in the device's matrix list.
PROBE_DIR="${SLURM_SUBMIT_DIR:-$PWD}/ci/slurm"
PROBE_SRC="$PROBE_DIR/bmg_probe.cpp"
PROBE_BIN="${TMPDIR:-/tmp}/bmg_probe.$$"

if [ ! -f "$PROBE_SRC" ]; then
  echo "PROBE: ABSENT at $PROBE_SRC"
elif ! command -v icpx >/dev/null 2>&1; then
  echo "PROBE: SKIPPED -- no icpx on PATH"
else
  echo
  echo "=== device probe ==="
  if icpx -fsycl -O2 -o "$PROBE_BIN" "$PROBE_SRC" 2>&1 | sed 's/^/PROBE_BUILD: /'; then
    "$PROBE_BIN" 2>&1
    rm -f "$PROBE_BIN"
  else
    echo "PROBE: build failed"
  fi
fi

echo
echo "DONE"
