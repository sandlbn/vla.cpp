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
# bmg_build.sbatch -- configure and build the SYCL backend on a Battlemage node.
#
# ggml's SYCL sources only compile under the oneAPI DPC++ driver and
# CMAKE_CXX_COMPILER is global, so the whole project is built by icpx. The build
# directory is therefore per-compiler and must be fresh the first time; it lives
# under the checkout so successive jobs rebuild incrementally rather than
# re-fetching llama.cpp every time.
#
# The two RPC servers are off: protobuf, libzmq and cppzmq are not installed on
# these nodes and there is no package manager or root to add them. Nothing the
# kernel work touches needs them.
#
#   sbatch ci/slurm/bmg_build.sbatch
#   BUILD_DIR=build-sycl-fresh CLEAN=1 sbatch ci/slurm/bmg_build.sbatch
#
# Knobs: BUILD_DIR, CLEAN=1 (wipe first), TARGETS (default: all), JOBS.
# =============================================================================
#SBATCH --job-name=bmg_build
#SBATCH --partition=b70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm_bmg_build_%j.out

# No `set -u`: the vendor's intel_gpu_vars.sh and oneAPI's setvars.sh both read
# unset variables, so nounset halts before the toolchain is on PATH.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-sycl}"
JOBS="${JOBS:-$(nproc)}"

echo "NODE: $(hostname)"
echo "REPO_DIR: $REPO_DIR"
echo "BUILD_DIR: $BUILD_DIR"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

# --- toolchain --------------------------------------------------------------
source /swtools/intel-gpu/latest/intel_gpu_vars.sh || fail "TOOLCHAIN intel_gpu_vars.sh"
source /swtools/intel/setvars.sh --force >/dev/null 2>&1 || fail "TOOLCHAIN setvars.sh"
command -v icpx >/dev/null 2>&1 || fail "TOOLCHAIN icpx not on PATH"
echo "ICPX: $(icpx --version | head -1)"
echo "GPU: $(sycl-ls 2>/dev/null | grep -m1 level_zero)"

# --- configure --------------------------------------------------------------
if [ "${CLEAN:-0}" = "1" ]; then
  echo "CLEAN: wiping $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

# vla-server needs protobuf, libzmq and cppzmq, none of which this cluster ships;
# ci/build_server_deps.sh puts them in a local prefix. Off by default so the
# ordinary kernel-development build does not depend on that prefix existing, but
# the LIBERO eval harness drives the model over ZMQ and cannot run without it.
SERVERS="${SERVERS:-OFF}"
DEPS_PREFIX="${DEPS_PREFIX:-$HOME/opt/vla-deps}"
if [ "$SERVERS" = "ON" ] && [ ! -f "$DEPS_PREFIX/include/zmq.hpp" ]; then
  fail "SERVERS=ON but no cppzmq under $DEPS_PREFIX -- run ci/build_server_deps.sh on the login node"
fi

# A cache built with the other SERVERS setting has to be reconfigured, not
# reused: the option is read at configure time, so the target simply would not
# exist. cmake updates the cache in place, so this costs a configure and not a
# rebuild of everything that already compiled.
cached_servers="$(sed -n 's/^VLA_BUILD_SERVERS:BOOL=//p' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null)"
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ] || [ "$cached_servers" != "$SERVERS" ]; then
  [ -f "$BUILD_DIR/CMakeCache.txt" ] &&
      echo "CONFIGURE: VLA_BUILD_SERVERS ${cached_servers:-unset} -> $SERVERS" ||
      echo "CONFIGURE: fresh"
  cmake -B "$BUILD_DIR" -S "$REPO_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_SYCL=ON \
      -DCMAKE_C_COMPILER=icx \
      -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_PREFIX_PATH="$DEPS_PREFIX" \
      -DVLA_BUILD_TESTS=ON \
      -DVLA_BUILD_SERVERS="$SERVERS" 2>&1 || fail "CONFIGURE"
else
  echo "CONFIGURE: reusing $BUILD_DIR"
fi

# --- build ------------------------------------------------------------------
echo "BUILD: -j$JOBS ${TARGETS:-all}"
# shellcheck disable=SC2086  # TARGETS is an intentional word-split list
if [ -n "${TARGETS:-}" ]; then
  cmake --build "$BUILD_DIR" -j"$JOBS" --target $TARGETS 2>&1 || fail "BUILD"
else
  cmake --build "$BUILD_DIR" -j"$JOBS" 2>&1 || fail "BUILD"
fi

echo
echo "ARTIFACTS:"
for f in vla-bench vla-cli vla-server tests/vla_predict_check; do
  [ -x "$BUILD_DIR/$f" ] && echo "  $BUILD_DIR/$f"
done

echo "VERDICT: BUILT $BUILD_DIR"
echo "DONE"
