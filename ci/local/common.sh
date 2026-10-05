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
# common.sh -- shared setup for the ci/local/ scripts. Sourced, never run.
#
# The ci/slurm/ jobs take their toolchain from the cluster's /swtools tree. These
# scripts are the same jobs for a workstation with a stock Intel install:
#
#   oneAPI     /opt/intel/oneapi/setvars.sh       (apt / offline installer default)
#   OpenVINO   /opt/intel/openvino/setupvars.sh   (scripts/install_ov.sh default)
#   GPU driver system packages (intel-opencl-icd, libze1, ...), nothing to source
#
# Every path is an env override, so a non-default install needs no edits:
#   INTEL_ROOT   /opt/intel
#   ONEAPI_ROOT  $INTEL_ROOT/oneapi
#   OV_ROOT      $INTEL_ROOT/openvino   (falls back to the newest openvino_* beside it)
#   VLA_CACHE    $HOME/.cache/vla       (where -hf downloads land)
#   DEPS_PREFIX  $HOME/opt/vla-deps     (protobuf/zmq from ci/build_server_deps.sh;
#                                        optional if the distro packages are installed)
# =============================================================================

# No `set -u` anywhere in ci/local: setvars.sh and setupvars.sh both read unset
# variables and abort under it.
set -o pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
INTEL_ROOT="${INTEL_ROOT:-/opt/intel}"
ONEAPI_ROOT="${ONEAPI_ROOT:-$INTEL_ROOT/oneapi}"
VLA_CACHE="${VLA_CACHE:-$HOME/.cache/vla}"
DEPS_PREFIX="${DEPS_PREFIX:-$HOME/opt/vla-deps}"
JOBS="${JOBS:-$(nproc)}"
VENV_PY="$REPO_DIR/eval/sim/libero/libero_uv/.venv/bin/python"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

# --- toolchains ---------------------------------------------------------------
# SYCL builds and runs need oneAPI (icpx, the SYCL runtime, oneDNN, oneMKL).
load_oneapi() {
  [ -f "$ONEAPI_ROOT/setvars.sh" ] ||
      fail "no oneAPI at $ONEAPI_ROOT -- install the Base Toolkit or set ONEAPI_ROOT"
  # shellcheck disable=SC1091
  source "$ONEAPI_ROOT/setvars.sh" --force >/dev/null 2>&1 || fail "TOOLCHAIN setvars.sh"
  command -v icpx >/dev/null 2>&1 || fail "TOOLCHAIN icpx not on PATH after setvars.sh"
  echo "ICPX:      $(icpx --version | head -1)"
  echo "GPU:       $(sycl-ls 2>/dev/null | grep -m1 level_zero || echo 'no Level Zero device')"
}

# OpenVINO builds and runs need only OpenVINO. Do NOT also source oneAPI in the
# same shell: its lib directory carries its own libOpenCL.so.1 and TBB, and the
# GPU plugin reaches the device through OpenCL -- shadowing the loader it was
# built against changes what you are measuring.
load_openvino() {
  if [ -z "${OV_ROOT:-}" ]; then
    OV_ROOT="$INTEL_ROOT/openvino"
    if [ ! -f "$OV_ROOT/setupvars.sh" ]; then
      local d
      d="$(ls -d "$INTEL_ROOT"/openvino_* 2>/dev/null | sort -V | tail -1)"
      [ -n "$d" ] && OV_ROOT="$d"
    fi
  fi
  [ -f "$OV_ROOT/setupvars.sh" ] ||
      fail "no OpenVINO at $OV_ROOT -- run scripts/install_ov.sh or set OV_ROOT"
  # shellcheck disable=SC1091
  source "$OV_ROOT/setupvars.sh" >/dev/null 2>&1 || fail "TOOLCHAIN setupvars.sh"
  echo "OPENVINO:  $OV_ROOT"
  command -v clinfo >/dev/null 2>&1 &&
      echo "OPENCL:    $(clinfo -l 2>/dev/null | grep -m1 -i 'device' | sed 's/^ *//')"
  # Cached blobs have reloaded graphs that compute the wrong thing on the GPU
  # plugin (docs/backend/ov.md). Never inherit one from the caller's shell.
  unset GGML_OPENVINO_CACHE_DIR
  # The experimental graph switches stay off unless a script sets them per arm,
  # or a "baseline" arm silently stops being one.
  unset GGML_OPENVINO_DYNAMIC_QUANT_GROUP GGML_OPENVINO_FC_RANK3 \
        GGML_OPENVINO_ACT_QUANT_ELIDE GGML_OPENVINO_RMS_FUSION
}

# --- server dependencies --------------------------------------------------------
# vla-server needs protobuf + ZeroMQ. Either the distro packages
#   sudo apt install protobuf-compiler libprotobuf-dev libzmq3-dev cppzmq-dev
# or a private prefix from ci/build_server_deps.sh. Prefer the prefix when it
# exists: it is the pinned protobuf the CI builds against.
server_deps_cmake_args() {
  if [ -f "$DEPS_PREFIX/include/zmq.hpp" ]; then
    export PATH="$DEPS_PREFIX/bin:$PATH"
    echo "-DCMAKE_PREFIX_PATH=$DEPS_PREFIX"
  fi
}

check_server_deps() {
  [ -f "$DEPS_PREFIX/include/zmq.hpp" ] && export PATH="$DEPS_PREFIX/bin:$PATH"
  command -v protoc >/dev/null 2>&1 ||
      fail "protoc not found -- apt install protobuf-compiler libprotobuf-dev libzmq3-dev cppzmq-dev, or bash ci/build_server_deps.sh"
}

# --- checkpoints ----------------------------------------------------------------
# First .gguf under $VLA_CACHE/<user/repo>. Pull a model once with
#   <build>/vla-bench -hf <user/repo> --reps 1
# or any other -hf aware tool; they all cache to the same place.
find_ckpt() {   # $1 = user/repo
  ls "$VLA_CACHE/$1"/*.gguf 2>/dev/null | head -1
}

# --- LIBERO / MuJoCo --------------------------------------------------------------
setup_mujoco() {
  [ -x "$VENV_PY" ] || fail "no LIBERO venv at $VENV_PY -- bash eval/sim/libero/setup_libero.sh"
  # egl renders headless on the GPU; a desktop session can also use glfw, and a
  # machine with no GL at all can fall back to osmesa (slow, CPU).
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="$MUJOCO_GL"
  echo "MUJOCO_GL: $MUJOCO_GL"
}

# One reset and one frame, before hours of episodes are spent. A renderer that
# returns blank frames makes every episode a silent failure, and the success
# rate then reads as a model regression.
render_smoke() {
  [ "${SMOKE:-1}" = "1" ] || return 0
  echo
  echo "=== MuJoCo render smoke test ==="
  ( cd "$REPO_DIR/eval" && "$VENV_PY" - <<'PY'
import sys
import numpy as np
try:
    import sim.libero  # noqa: F401  registers the gymnasium envs
    import gymnasium as gym
    env = gym.make("libero_object/task_0")
    obs, _ = env.reset()
    def images(node, prefix=""):
        out = {}
        if isinstance(node, dict):
            for k, v in node.items():
                out.update(images(v, f"{prefix}{k}/"))
        elif hasattr(node, "shape") and np.asarray(node).ndim == 3:
            out[prefix.rstrip("/")] = np.asarray(node)
        return out
    imgs = images(obs) if isinstance(obs, dict) else {}
    if not imgs:
        print("SMOKE: reset ok but no image in obs")
        sys.exit(3)
    k, img = sorted(imgs.items())[0]
    if int(img.max()) == int(img.min()):
        print(f"SMOKE: FAILED {k} is uniform -- the renderer produced a blank frame")
        sys.exit(4)
    print(f"SMOKE: ok, {k} {img.shape} {img.dtype}, range {img.min()}-{img.max()}")
    env.close()
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"SMOKE: FAILED {type(e).__name__}: {e}")
    sys.exit(1)
PY
  ) || fail "RENDER SMOKE TEST (MUJOCO_GL=$MUJOCO_GL) -- try MUJOCO_GL=glfw on a desktop, osmesa as a last resort"
}

# Comma-separated chunk roots for one arm; plot_libero_bf16.py sums them.
arm_paths() {   # $1 = root holding chunk0..chunkN
  local d p=""
  for d in "$1"/chunk*; do [ -d "$d" ] && p="${p:+$p,}$d"; done
  echo "$p"
}

# Episodes run in chunks of EPISODES_PER_PROC client processes. robosuite leaks
# an offscreen GL context per reset and some EGL stacks abort after a dozen or
# so; ten per process has never hit it. Chunks are independent Bernoulli trials,
# so 5 x 10 is the same 50 episodes as 1 x 50.
EPISODES_PER_PROC="${EPISODES_PER_PROC:-10}"

run_libero_chunked() {   # $1 = output root, $2 = model (-m), $3 = build dir; env passes through
  local out="$1" model="$2" build="$3"
  local remaining="$N_EPISODES" chunk=0 n
  # Cleared, not merged into: run_libero.sh never removes an old summary.txt,
  # so a re-run at a different -n would mix two episode counts in one arm.
  rm -rf "$out"
  while [ "$remaining" -gt 0 ]; do
    n="$EPISODES_PER_PROC"
    [ "$n" -gt "$remaining" ] && n="$remaining"
    echo "--- $model chunk $chunk: $n episodes/task -> $out/chunk$chunk ---"
    SKIP_BUILD=1 BUILD_DIR="$build" \
        bash "$REPO_DIR/eval/run_libero.sh" \
            -i "$MODELS_ROOT" -o "$out/chunk$chunk" -n "$n" -m "$model" ||
        return 1
    remaining=$((remaining - n))
    chunk=$((chunk + 1))
  done
}

# --- BitVLA dense flavours (OpenVINO) -------------------------------------------
# The published BitVLA file is int2-packed, which only the CUDA/SYCL ternary
# kernels read. OpenVINO needs it transcoded to bf16 / q8_0 / q4_0 -- a lossless
# lookup on the 2-bit codes, CPU only, a minute or two per flavour. Same tool and
# file names as ci/slurm/bitvla_bf16_gguf.sbatch, so both setups share a cache.
TASK="${TASK:-libero_object}"
BITVLA_DIR="${BITVLA_DIR:-$VLA_CACHE/vrfai/bitvla-libero-gguf/$TASK}"
BITVLA_STEM="bitvla-${TASK//_/-}"

bitvla_ckpt() {   # $1 = bf16 | q8_0 | q4_0 -> path, transcoding it if missing
  local dtype="$1"
  local dst="$BITVLA_DIR/$BITVLA_STEM-$dtype.gguf"
  local src="$BITVLA_DIR/$BITVLA_STEM.gguf"
  if [ ! -f "$dst" ]; then
    [ -f "$src" ] || {
      echo "no BitVLA int2 checkpoint at $src -- fetch it with" >&2
      echo "  huggingface-cli download vrfai/bitvla-libero-gguf --local-dir $VLA_CACHE/vrfai/bitvla-libero-gguf" >&2
      return 1
    }
    # gguf-py ships inside the llama.cpp tree CMake fetched; no pip install needed.
    if ! python3 -c "import gguf" >/dev/null 2>&1; then
      local d
      for d in "$REPO_DIR"/build-*/_deps/llama-src/gguf-py; do
        [ -d "$d" ] && { export PYTHONPATH="$d${PYTHONPATH:+:$PYTHONPATH}"; break; }
      done
    fi
    echo "TRANSCODE: $dtype -> $dst" >&2
    python3 "$REPO_DIR/scripts/transcode_bitvla_int2.py" \
        --weight-dtype "$dtype" "$src" "$dst" >&2 || return 1
  fi
  echo "$dst"
}

# A python with matplotlib for the plots: the system one if it has it, else the
# LIBERO venv. Empty if neither -- the caller then skips the plot, not the run.
plot_python() {
  local py
  for py in python3 "$VENV_PY"; do
    "$py" -c "import matplotlib, numpy" >/dev/null 2>&1 && { echo "$py"; return; }
  done
}
