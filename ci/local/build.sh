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
# build.sh -- build vla.cpp for an Intel GPU on a workstation.
# Local equivalent of ci/slurm/bmg_build.sbatch (SYCL) and the build half of
# ci/slurm/bmg_ov.sbatch (OpenVINO).
#
#   ci/local/build.sh sycl                 # -> build-sycl/  (vla-bench, tests)
#   ci/local/build.sh ov                   # -> build-ov/
#   SERVERS=ON ci/local/build.sh sycl      # + vla-server, needed for LIBERO
#   CLEAN=1 ci/local/build.sh ov           # wipe the build dir first
#   TARGETS="vla-bench" ci/local/build.sh sycl
#   SYCL_AOT=ptl-u,ptl-h ci/local/build.sh sycl   # native Xe3 code, no first-run JIT
#
# Knobs: BUILD_DIR, SERVERS (OFF), CLEAN (0), TARGETS (all), JOBS, SYCL_AOT
# (ocloc device list -> VLA_SYCL_DEVICE_ARCH; empty = JIT), plus the
# toolchain paths documented in ci/local/common.sh.
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKEND="${1:-}"
case "$BACKEND" in
  sycl|ov) ;;
  *) echo "usage: $0 sycl|ov"; exit 2 ;;
esac
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-$BACKEND}"
SERVERS="${SERVERS:-OFF}"

echo "BACKEND:   $BACKEND"
echo "BUILD_DIR: $BUILD_DIR"
echo "SERVERS:   $SERVERS"

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DVLA_BUILD_TESTS=ON -DVLA_BUILD_SERVERS="$SERVERS")
if [ "$BACKEND" = sycl ]; then
  load_oneapi
  CMAKE_ARGS+=(-DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx)
  [ -n "${SYCL_AOT:-}" ] && CMAKE_ARGS+=(-DVLA_SYCL_DEVICE_ARCH="$SYCL_AOT")
  echo "SYCL_AOT:  ${SYCL_AOT:-none (JIT)}"
else
  load_openvino
  # The ggml-openvino patches (scripts/patch_ggml_openvino.py) are applied by
  # CMake when it fetches llama.cpp, so a fresh build dir needs nothing extra.
  CMAKE_ARGS+=(-DGGML_OPENVINO=ON)
fi
command -v ninja >/dev/null 2>&1 && CMAKE_ARGS+=(-G Ninja)

if [ "$SERVERS" = ON ]; then
  check_server_deps
  dep_arg="$(server_deps_cmake_args)"
  [ -n "$dep_arg" ] && CMAKE_ARGS+=("$dep_arg")
fi

if [ "${CLEAN:-0}" = 1 ]; then
  echo "CLEAN:     wiping $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

# Reconfigure when SERVERS changed: CMake keeps the cached value otherwise, and
# a LIBERO run would then find no vla-server in a dir that "built fine".
cached="$(sed -n 's/^VLA_BUILD_SERVERS:BOOL=//p' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null)"
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ] || [ "$cached" != "$SERVERS" ]; then
  if [ -n "$cached" ]; then
    echo "CONFIGURE: VLA_BUILD_SERVERS $cached -> $SERVERS"
  else
    echo "CONFIGURE: fresh"
  fi
  cmake -B "$BUILD_DIR" -S "$REPO_DIR" "${CMAKE_ARGS[@]}" || fail "CONFIGURE"
else
  echo "CONFIGURE: reusing $BUILD_DIR"
fi

echo "BUILD:     -j$JOBS ${TARGETS:-all}"
if [ -n "${TARGETS:-}" ]; then
  # shellcheck disable=SC2086
  cmake --build "$BUILD_DIR" -j"$JOBS" --target $TARGETS || fail "BUILD"
else
  cmake --build "$BUILD_DIR" -j"$JOBS" || fail "BUILD"
fi

echo
echo "ARTIFACTS:"
for f in vla-bench vla-cli vla-server tests/vla_predict_check; do
  [ -x "$BUILD_DIR/$f" ] && echo "  $BUILD_DIR/$f"
done
echo "VERDICT: BUILT $BUILD_DIR"
echo "DONE"
