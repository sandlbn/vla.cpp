#!/usr/bin/env bash
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
# build_server_deps.sh -- protobuf, libzmq and cppzmq into a local prefix.
#
# vla-server needs all three (CMakeLists.txt), and this cluster has none of them:
# /usr/lib64 carries libprotobuf-c, which is the C library and not what
# find_package(Protobuf) looks for. The LIBERO eval harness drives the model over
# ZMQ, so no server means no task-success numbers at all.
#
# Run on the login node -- it fetches sources. Then point the vla.cpp build at
# the prefix:
#
#   bash ci/build_server_deps.sh
#   cmake -B build-sycl -DCMAKE_PREFIX_PATH=$HOME/opt/vla-deps ... -DVLA_BUILD_SERVERS=ON
#
# protobuf is pinned to 21.12 deliberately: it is the last release before the
# hard Abseil dependency, so it builds standalone in a couple of minutes instead
# of dragging in a second source tree. vla.proto uses nothing newer.
#
# Knobs: PREFIX, JOBS, PB_TAG, ZMQ_TAG, CPPZMQ_TAG.
# =============================================================================
set -euo pipefail

PREFIX="${PREFIX:-$HOME/opt/vla-deps}"
SRC="${SRC:-$PREFIX/src}"
JOBS="${JOBS:-$(nproc)}"
PB_TAG="${PB_TAG:-v21.12}"
ZMQ_TAG="${ZMQ_TAG:-v4.3.5}"
CPPZMQ_TAG="${CPPZMQ_TAG:-v4.10.0}"

mkdir -p "$SRC"
echo "[deps] prefix=$PREFIX jobs=$JOBS"

# --- protobuf ---------------------------------------------------------------
if [ ! -f "$PREFIX/lib64/libprotobuf.a" ] && [ ! -f "$PREFIX/lib/libprotobuf.a" ]; then
  echo "[deps] protobuf $PB_TAG"
  [ -d "$SRC/protobuf" ] || git clone --depth 1 -b "$PB_TAG" \
      https://github.com/protocolbuffers/protobuf "$SRC/protobuf"
  cmake -S "$SRC/protobuf" -B "$SRC/protobuf/build" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -Dprotobuf_BUILD_TESTS=OFF \
      -Dprotobuf_BUILD_SHARED_LIBS=OFF \
      -Dprotobuf_ABSL_PROVIDER=module
  cmake --build "$SRC/protobuf/build" -j"$JOBS"
  cmake --install "$SRC/protobuf/build"
else
  echo "[deps] protobuf already present"
fi

# --- libzmq -----------------------------------------------------------------
if [ ! -f "$PREFIX/lib/pkgconfig/libzmq.pc" ] && [ ! -f "$PREFIX/lib64/pkgconfig/libzmq.pc" ]; then
  echo "[deps] libzmq $ZMQ_TAG"
  [ -d "$SRC/libzmq" ] || git clone --depth 1 -b "$ZMQ_TAG" \
      https://github.com/zeromq/libzmq "$SRC/libzmq"
  # Tests off and draft API off: the server uses only the stable socket types,
  # and the test suite is the bulk of the build time.
  cmake -S "$SRC/libzmq" -B "$SRC/libzmq/build" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DBUILD_TESTS=OFF \
      -DENABLE_DRAFTS=OFF \
      -DWITH_DOC=OFF
  cmake --build "$SRC/libzmq/build" -j"$JOBS"
  cmake --install "$SRC/libzmq/build"
else
  echo "[deps] libzmq already present"
fi

# --- cppzmq (header only) ---------------------------------------------------
if [ ! -f "$PREFIX/include/zmq.hpp" ]; then
  echo "[deps] cppzmq $CPPZMQ_TAG"
  [ -d "$SRC/cppzmq" ] || git clone --depth 1 -b "$CPPZMQ_TAG" \
      https://github.com/zeromq/cppzmq "$SRC/cppzmq"
  # Its CMake config insists on finding libzmq and running tests; the project
  # only ever #includes the header, so install it directly.
  install -m 0644 "$SRC/cppzmq/zmq.hpp" "$PREFIX/include/zmq.hpp"
  [ -f "$SRC/cppzmq/zmq_addon.hpp" ] &&
      install -m 0644 "$SRC/cppzmq/zmq_addon.hpp" "$PREFIX/include/zmq_addon.hpp"
else
  echo "[deps] cppzmq already present"
fi

echo
echo "[deps] done:"
ls "$PREFIX"/lib*/libprotobuf.a "$PREFIX"/lib*/libzmq.so* "$PREFIX/include/zmq.hpp" 2>/dev/null
echo "[deps] configure vla.cpp with -DCMAKE_PREFIX_PATH=$PREFIX"
