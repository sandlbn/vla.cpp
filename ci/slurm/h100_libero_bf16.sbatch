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
# h100_libero_bf16.sbatch -- the CUDA half of the task-success claim.
#
# ci/slurm/bmg_libero_bf16.sbatch ran libero_object on B70 and produced four
# numbers: evo1 96/94 and pi0 83/76 out of 100, f32 against bf16. Read on their
# own those numbers cannot answer the question anyone actually has, which is
# whether the Intel port costs task success. They are all one backend. If SYCL
# f32 already scored below CUDA f32, that is a port defect and it would be
# invisible here -- the bf16 arm would be scored against an already-degraded
# baseline and look fine.
#
# This is the same mistake the TF32 finding exposed in the precision work
# (see the header of ci/slurm/h100_bf16_e2e.sbatch): a backend compared against
# itself measures nothing about the backend. The fix is the same. Run the
# identical sweep -- same suite, same task ids, same episode count, same chunk
# replay -- through the CUDA backend, and read the 2x2x2 table across.
#
# What each comparison answers:
#   CUDA f32  vs SYCL f32    does the port lose task success at all?
#   CUDA bf16 vs CUDA f32    is the bf16 delta a property of the dtype...
#   SYCL bf16 vs SYCL f32    ...or of this backend's kernels?
#
# Note this sweep is NOT paired in seeds. LIBERO draws initial states per
# episode and neither backend is being asked to replay the other's rollouts, so
# every cell is an independent binomial sample. At ten episodes per task that is
# a wide interval; N_EPISODES is the knob that buys resolution, and pi0 runs
# ~10 min per 100 episodes, so buying it is cheap.
#
# Prerequisites (login node, one-time, shared with the B70 job via NFS $HOME):
#   bash ci/build_server_deps.sh          # protobuf + libzmq + cppzmq
#   bash eval/sim/libero/setup_libero.sh  # LIBERO + its venv
#
#   sbatch ci/slurm/h100_libero_bf16.sbatch
#   N_EPISODES=50 MODELS=pi0 sbatch ci/slurm/h100_libero_bf16.sbatch
#
# Knobs: BUILD_DIR, MODELS_ROOT, N_EPISODES, MODELS, OUTPUT_ROOT, SMOKE=0,
#        ARCH, CUDA_VER, JOBS, DEPS_PREFIX.
# =============================================================================
#SBATCH --job-name=h100_libero_bf16
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=slurm_h100_libero_bf16_%j.out

# No `set -u`: the eval scripts read unset variables.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-cuda-srv}"
MODELS_ROOT="${MODELS_ROOT:-${VLA_CACHE:-$HOME/.cache/vla}/vrfai}"
N_EPISODES="${N_EPISODES:-10}"
MODELS="${MODELS:-evo1 pi0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/libero_h100}"
VENV_PY="$REPO_DIR/eval/sim/libero/libero_uv/.venv/bin/python"
JOBS="${JOBS:-$(nproc)}"
ARCH="${ARCH:-90}"

echo "NODE:      $(hostname)"
echo "EPISODES:  $N_EPISODES per task, models: $MODELS"
echo "OUTPUT:    $OUTPUT_ROOT"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

# --- toolchain --------------------------------------------------------------
CUDA_VER="${CUDA_VER:-12.9.0}"
for d in "/swtools/cuda_x86_64/$CUDA_VER" /swtools/cuda_x86_64/12.9.0 \
         /swtools/cuda_x86_64/12.6.3 /usr/local/cuda; do
  if [ -x "$d/bin/nvcc" ]; then CUDA_HOME="$d"; break; fi
done
[ -n "$CUDA_HOME" ] || fail "TOOLCHAIN no nvcc under /swtools/cuda_x86_64"
export CUDA_HOME CUDAToolkit_ROOT="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
echo "CUDA:      $("$CUDA_HOME/bin/nvcc" --version | tail -2 | head -1)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

[ -x "$VENV_PY" ] || fail "no LIBERO venv -- run eval/sim/libero/setup_libero.sh on the login node"
[ -d "$MODELS_ROOT" ] || fail "no MODELS_ROOT at $MODELS_ROOT"

# The client shells out to protoc at import time to generate its bindings
# (eval/client/vla_cpp_client.py), so the compiler has to be on PATH and not
# merely linked into vla-server. Same local prefix the B70 job uses.
DEPS_PREFIX="${DEPS_PREFIX:-$HOME/opt/vla-deps}"
export PATH="$DEPS_PREFIX/bin:$PATH"
command -v protoc >/dev/null 2>&1 ||
    fail "protoc not on PATH -- run ci/build_server_deps.sh on the login node"
[ -f "$DEPS_PREFIX/include/zmq.hpp" ] ||
    fail "no cppzmq under $DEPS_PREFIX -- run ci/build_server_deps.sh on the login node"
echo "PROTOC:    $(protoc --version)"

# --- build ------------------------------------------------------------------
# A separate build dir from the e2e/bench jobs on purpose: those configure with
# VLA_BUILD_SERVERS=OFF, and the option is read at configure time, so sharing a
# tree would mean reconfiguring back and forth and relinking on every job.
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  echo "CONFIGURE: fresh $BUILD_DIR"
  cmake -B "$BUILD_DIR" -S "$REPO_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      -DCMAKE_PREFIX_PATH="$DEPS_PREFIX" \
      -DVLA_BUILD_SERVERS=ON 2>&1 | tail -5 || fail "CONFIGURE"
else
  echo "CONFIGURE: reusing $BUILD_DIR"
fi
cmake --build "$BUILD_DIR" -j"$JOBS" 2>&1 | tail -5 || fail "BUILD"
[ -x "$BUILD_DIR/vla-server" ] || fail "no vla-server in $BUILD_DIR after build"

# --- headless rendering -----------------------------------------------------
# MuJoCo picks a renderer at import time and neither run_libero.sh nor
# libero_env.py sets one; on a headless compute node the default lands on GLFW
# and dies without a display.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="$MUJOCO_GL"
# Always on, not just when debugging: the client's failure mode here is a native
# abort inside the renderer, which exits silently and leaves the sweep log with
# nothing but "Aborted (core dumped)". Job 371832 cost a whole extra scheduling
# round for exactly this. A traceback naming the extension module is free.
export PYTHONFAULTHANDLER=1
echo "MUJOCO_GL: $MUJOCO_GL"

if [ "${SMOKE:-1}" = "1" ]; then
  echo
  echo "=== render smoke test ==="
  ( cd "$REPO_DIR/eval" && "$VENV_PY" - <<'PY'
import sys
import numpy as np
try:
    import sim.libero  # noqa: F401  registers the gymnasium envs
    import gymnasium as gym
    env = gym.make("libero_object/task_0")   # "<suite>/task_<id>"
    obs, _ = env.reset()
    # obs["pixels"] is itself a dict of camera views (multi-view env), so this
    # has to walk down rather than filter the top level on ndim.
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
        print("SMOKE: reset ok but no image in obs:",
              list(obs) if isinstance(obs, dict) else type(obs))
        sys.exit(3)
    k, img = sorted(imgs.items())[0]
    # A uniform frame is what a broken EGL context yields: reset succeeds and
    # every episode then fails for no visible reason.
    if int(img.max()) == int(img.min()):
        print(f"SMOKE: FAILED {k} is uniform ({img.min()}) -- renderer produced a blank frame")
        sys.exit(4)
    print(f"SMOKE: ok, {k} {img.shape} {img.dtype}, range {img.min()}-{img.max()}")
    env.close()
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"SMOKE: FAILED {type(e).__name__}: {e}")
    sys.exit(1)
PY
  ) || fail "RENDER SMOKE TEST (MUJOCO_GL=$MUJOCO_GL) -- see the SMOKE line above"
fi

# --- the arms ---------------------------------------------------------------
# Episodes are split across several client processes rather than run in one.
# robosuite rebuilds its offscreen GL context on every reset and the old one is
# only reclaimed by the cyclic collector (eval/client/run_sim_client_direct.py
# has the detail), so contexts accumulate within a process. NVIDIA's EGL runs
# out and aborts mid-sweep; the threshold is not fixed - job 371995 died at 12
# episodes where an identical 20-episode run survived - so the fix is to stay
# well under it rather than to find it. Ten per process is the size evo1 ran a
# full sweep at without a single abort.
#
# Splitting costs nothing statistically: episodes are independent and LIBERO
# draws its initial state per episode, so 5x10 is the same 50 Bernoulli trials
# as 1x50. The chunks are summed at plot time.
EPISODES_PER_PROC="${EPISODES_PER_PROC:-10}"

run_arm() {   # $1 = dtype tag, $2.. = extra server args
  local dtype="$1"; shift
  for model in $MODELS; do
    echo
    echo "########## $model / $dtype ##########"
    local out="$OUTPUT_ROOT/$model-$dtype"
    # Cleared, not merged into: run_libero.sh writes one summary.txt per task
    # and never removes an old one, so a re-run at a different -n leaves the
    # previous episode count in place for every task it has not reached.
    rm -rf "$out"
    local remaining="$N_EPISODES" chunk=0
    while [ "$remaining" -gt 0 ]; do
      local n="$EPISODES_PER_PROC"
      [ "$n" -gt "$remaining" ] && n="$remaining"
      echo "--- $model/$dtype chunk $chunk: $n episodes/task ---"
      SKIP_BUILD=1 BUILD_DIR="$BUILD_DIR" EXTRA_SERVER_ARGS="$*" \
          bash "$REPO_DIR/eval/run_libero.sh" \
              -i "$MODELS_ROOT" -o "$out/chunk$chunk" -n "$n" -m "$model" \
          || fail "SWEEP $model/$dtype chunk $chunk"
      remaining=$((remaining - n))
      chunk=$((chunk + 1))
    done
  done
}

# Comma-separated chunk roots for one arm; the plot script sums them.
arm_paths() {   # $1 = root holding chunk0..chunkN
  local d p=""
  for d in "$1"/chunk*; do [ -d "$d" ] && p="${p:+$p,}$d"; done
  echo "$p"
}

run_arm f32
run_arm bf16 --act-dtype bf16

# An arch that declined --act-dtype bf16 would produce a perfect, meaningless
# agreement with its own f32 arm.
for model in $MODELS; do
  grep -qh "activations = BF16" \
      "$OUTPUT_ROOT/$model-bf16/"chunk*/_server_logs/*.log 2>/dev/null ||
      fail "BF16 ARM DID NOT ENGAGE for $model"
done

# --- collect and plot -------------------------------------------------------
echo
echo "=== summary (CUDA) ==="
ARMS=()
for model in $MODELS; do
  for dtype in f32 bf16; do
    ARMS+=(--arm "$model:$dtype=$(arm_paths "$OUTPUT_ROOT/$model-$dtype")")
  done
done
PLOT_TAG="$(echo $MODELS | tr ' ' '_')"
python3 "$REPO_DIR/scripts/plot_libero_bf16.py" "${ARMS[@]}" \
    --out "$REPO_DIR/docs/img/libero_bf16_h100-$PLOT_TAG" \
    --title "LIBERO object, H100 (CUDA), $N_EPISODES episodes/task" \
    || fail "PLOT"

# --- the cross-backend table, which is the point of the job -----------------
# Only emitted if the B70 sweep's outputs are present; the job is still useful
# without them, it just cannot draw the comparison itself.
B70_ROOT="${B70_ROOT:-$REPO_DIR/outputs/libero_b70}"
if [ -d "$B70_ROOT" ]; then
  echo
  echo "=== CUDA vs SYCL ==="
  XARMS=()
  for model in $MODELS; do
    for dtype in f32 bf16; do
      [ -d "$B70_ROOT/$model-$dtype" ] &&
          XARMS+=(--arm "$model:b70-$dtype=$B70_ROOT/$model-$dtype")
      XARMS+=(--arm "$model:h100-$dtype=$(arm_paths "$OUTPUT_ROOT/$model-$dtype")")
    done
  done
  python3 "$REPO_DIR/scripts/plot_libero_bf16.py" "${XARMS[@]}" \
      --out "$REPO_DIR/docs/img/libero_bf16_xbackend-$PLOT_TAG" \
      --title "LIBERO object, B70 (SYCL) vs H100 (CUDA), $N_EPISODES episodes/task" \
      || echo "WARNING: cross-backend plot failed (the per-backend one above is still valid)"
else
  echo "NOTE: no $B70_ROOT, skipping the cross-backend comparison"
fi

echo
echo "VERDICT: OK"
echo "DONE"
