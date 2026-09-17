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
# bmg_ov.sbatch -- build the OpenVINO backend and test it, CPU plugin then GPU.
#
# This is the OpenVINO port's counterpart to bmg_build.sbatch. The backends are
# mutually exclusive at CMakeLists.txt, so GGML_OPENVINO gets its own build
# directory and neither GGML_SYCL nor GGML_CUDA is set; the compiler is the
# stock one rather than icpx, because nothing here is SYCL.
#
# The runtime is the archive unpacked under $HOME -- scripts/install_ov.sh wants
# Ubuntu, sudo and /opt, none of which apply on these nodes.
#
# The two plugins are run back to back on one node: the CPU plugin is the
# fidelity reference and the B70 is the performance target. The GPU plugin
# reaches the device through OpenCL rather than Level Zero -- that is the
# plugin's own architecture, not a choice made here, and it is why this cluster's
# second, non-Intel OpenCL platform has to be steered around with
# OCL_ICD_VENDORS.
#
# The build directory is per-host and that is not fussiness. GGML_NATIVE is ON,
# so ggml compiles with -march=native, resolved on whichever machine configures.
# The b70 partition is pcl-adl01 / pcl-arl01 / pcl-zen4 -- Alder Lake, Arrow Lake
# and a Zen 4 -- while the login node is a Xeon 8580 with AVX-512 and AMX. A
# build carried across from there does not merely run mistuned, it takes SIGILL
# on the first AVX-512 instruction, and a build shared between two of these nodes
# is just as wrong in the other direction. Hence build-ov-<host>.
#
#   sbatch ci/slurm/bmg_ov.sbatch
#   CLEAN=1 sbatch ci/slurm/bmg_ov.sbatch
#
# The GPU is run at both precisions on purpose. The plugin defaults to f16, and
# BitVLA's activation quantiser rounds against a 127/amax scale -- Round is a
# threshold, so a scale that is a little off does not perturb a low-order bit, it
# moves an integer. f16 carries ~11 bits of mantissa and that is not enough: it
# flips whole quantisation steps. f32 is what backend.h defaults bitvla to, and
# the f16 run is kept as the measurement that justifies the default rather than
# an assertion that it is needed.
#
# Knobs: BUILD_DIR, CLEAN=1, JOBS, OV_ROOT, LLAMA_SRC_MIRROR,
#        CONFIGS (default "CPU GPU:f32 GPU:f16~"; a trailing ~ means report but
#        do not gate).
# =============================================================================
#SBATCH --job-name=bmg_ov
#SBATCH --partition=b70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm_bmg_ov_%j.out

# No `set -u`: the vendor's intel_gpu_vars.sh and OpenVINO's setupvars.sh both
# read unset variables, so nounset halts before the toolchain is on PATH.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
HOST="$(hostname -s)"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-ov-$HOST}"
JOBS="${JOBS:-$(nproc)}"
OV_ROOT="${OV_ROOT:-$HOME/intel/openvino}"
CONFIGS="${CONFIGS:-CPU GPU:f32 GPU:f16~}"
LLAMA_SRC_MIRROR="${LLAMA_SRC_MIRROR:-$HOME/vla-deps/llama-src}"

echo "NODE: $HOST"
echo "CPU: $(lscpu | sed -n 's/^Model name: *//p' | head -1)"
echo "BUILD_DIR: $BUILD_DIR"
echo "OV_ROOT: $OV_ROOT"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

# --- toolchain --------------------------------------------------------------
source /swtools/intel-gpu/latest/intel_gpu_vars.sh || fail "TOOLCHAIN intel_gpu_vars.sh"
[ -f "$OV_ROOT/setupvars.sh" ] || fail "no OpenVINO runtime at $OV_ROOT"
source "$OV_ROOT/setupvars.sh" >/dev/null 2>&1 || fail "TOOLCHAIN setupvars.sh"

# Select the Intel OpenCL platform explicitly. clpeak's results directory on this
# cluster records a second, non-Intel platform being present, and the GPU plugin
# will happily bind to the wrong one; hunk 1 of scripts/patch_ggml_openvino.py
# fixes the same problem inside ggml by selecting on CL_PLATFORM_VENDOR.
export OCL_ICD_VENDORS="${OCL_ICD_VENDORS:-/swtools/intel-gpu/latest/neo}"

# Caching OpenVINO's compiled blobs produces silently wrong actions (docs/backend/ov.md),
# and backend.h strips the variable and warns unless VLA_ALLOW_OV_CACHE=1. Do not
# set it here to make first-compile latency look better.
unset GGML_OPENVINO_CACHE_DIR

echo "OV: $(ls "$OV_ROOT" | tr '\n' ' ')"
echo "CLINFO:"; clinfo -l 2>&1 | sed 's/^/  /' | head -10

# --- configure --------------------------------------------------------------
if [ "${CLEAN:-0}" = "1" ]; then
  echo "CLEAN: wiping $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

# A per-host build directory means a first run would re-fetch llama.cpp, and
# these compute nodes are not assumed to have a route to github. Any already
# configured tree has the sources, so seed a mirror from one and point
# FetchContent at it; the tag check in CMakeLists still applies, and the
# ggml-openvino patch script is idempotent so re-running it over the mirror is
# safe. Falls through to a real fetch if there is nothing to copy.
if [ ! -d "$LLAMA_SRC_MIRROR" ]; then
  for d in "$REPO_DIR"/build-*/_deps/llama-src; do
    if [ -d "$d" ]; then
      echo "MIRROR: seeding $LLAMA_SRC_MIRROR from $d"
      mkdir -p "$(dirname "$LLAMA_SRC_MIRROR")"
      cp -a "$d" "$LLAMA_SRC_MIRROR" || fail "MIRROR copy"
      break
    fi
  done
fi
FC_ARGS=()
if [ -d "$LLAMA_SRC_MIRROR" ]; then
  echo "MIRROR: using $LLAMA_SRC_MIRROR"
  FC_ARGS=(-DFETCHCONTENT_SOURCE_DIR_LLAMA="$LLAMA_SRC_MIRROR")
else
  echo "MIRROR: none, will fetch"
fi

# VLA_BUILD_SERVERS=OFF: vla-server needs protobuf, libzmq and cppzmq, and this
# node has no package manager or root to add them. Nothing the OpenVINO port
# touches needs them.
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  echo "CONFIGURE: fresh"
  cmake -B "$BUILD_DIR" -S "$REPO_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_OPENVINO=ON \
      -DVLA_BUILD_TESTS=ON \
      -DVLA_BUILD_SERVERS=OFF \
      "${FC_ARGS[@]}" 2>&1 || fail "CONFIGURE"
else
  echo "CONFIGURE: reusing $BUILD_DIR"
fi

echo "BUILD: -j$JOBS"
cmake --build "$BUILD_DIR" -j"$JOBS" 2>&1 | tail -20 || fail "BUILD"

# --- the device-independent suite -------------------------------------------
# Everything that does not care which plugin is up: the graph-naming pass, the
# ladder round-trip, the config guard. Cheap, and they are what hunk 14 could
# plausibly have broken by accident.
echo
echo "--- ctest (device-independent) ---"
( cd "$BUILD_DIR" && ctest --output-on-failure -E bitvla_act_quant_ov 2>&1 ) || fail "CTEST"

# --- the translator, on each plugin -----------------------------------------
ACT_QUANT="$BUILD_DIR/tests/test_bitvla_act_quant_ov"
[ -x "$ACT_QUANT" ] || fail "no $ACT_QUANT -- did GGML_OPENVINO stick?"

rc=0
for cfg in $CONFIGS; do
  # A trailing ~ means measure and report but do not gate.
  nogate=0
  case "$cfg" in *'~') nogate=1; cfg="${cfg%\~}" ;; esac
  dev="${cfg%%:*}"
  prec=""
  case "$cfg" in *:*) prec="${cfg#*:}" ;; esac

  echo
  # ${nogate:+...} would expand for "0" too -- it tests for non-empty, not truth.
  label=""; [ "$nogate" = "1" ] && label=" (not gated)"
  echo "--- act_quant translator: device=$dev precision=${prec:-default}$label ---"
  # The test calls ggml_backend_openvino_init directly rather than
  # vla::backend_init -- on purpose, so a failure to start OpenVINO cannot fall
  # back to the CPU backend and quietly become a CPU-against-CPU comparison. The
  # cost is that backend.h's GGML_OPENVINO_GPU_PRECISION default does not apply,
  # so it is set explicitly here.
  if env GGML_OPENVINO_DEVICE="$dev" \
         ${prec:+GGML_OPENVINO_GPU_PRECISION="$prec"} \
         "$ACT_QUANT" 2>&1; then
    echo "RESULT: $dev ${prec:-default} ok"
  elif [ "$nogate" = "1" ]; then
    echo "RESULT: $dev ${prec:-default} outside gate (expected; recorded, not gating)"
  else
    echo "RESULT: $dev ${prec:-default} FAILED"
    rc=1
  fi
done

[ "$rc" = "0" ] || fail "ACT_QUANT"
echo
echo "VERDICT: PASSED ($CONFIGS)"
echo "DONE"
