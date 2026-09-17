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
# bmg_libero_bf16.sbatch -- does --act-dtype bf16 still solve the task?
#
# Everything else measured about the BF16 activation path is engine-side: action
# chunks scored against a CPU fp32 reference (ci/slurm/bmg_bf16_e2e.sbatch) and
# latency (ci/slurm/bmg_bf16_bench.sbatch). Both can look fine while the policy
# quietly stops picking the object up. This closes the loop: libero_object, ten
# task ids, N episodes each, evo1 and pi0, F32 and BF16 activations, all on B70.
#
# Prerequisites, both one-time and both on the LOGIN node (this job has no route
# out and will not install anything):
#   bash ci/build_server_deps.sh          # protobuf + libzmq + cppzmq
#   bash eval/sim/libero/setup_libero.sh  # LIBERO + its venv
#   sbatch --export=ALL,SERVERS=ON ci/slurm/bmg_build.sbatch
#
# Budget: four arms x 10 tasks x N_EPISODES. At N=10 that is 400 episodes and
# several hours, which is why the walltime is what it is.
#
#   sbatch ci/slurm/bmg_libero_bf16.sbatch
#   N_EPISODES=3 sbatch ci/slurm/bmg_libero_bf16.sbatch     # a quicker shakedown
#
# Knobs: BUILD_DIR, MODELS_ROOT, N_EPISODES, MODELS, OUTPUT_ROOT, SMOKE=0.
# =============================================================================
#SBATCH --job-name=bmg_libero_bf16
#SBATCH --partition=b70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --output=slurm_bmg_libero_bf16_%j.out

# No `set -u`: both vendor scripts read unset variables.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-sycl}"
MODELS_ROOT="${MODELS_ROOT:-${VLA_CACHE:-$HOME/.cache/vla}/vrfai}"
N_EPISODES="${N_EPISODES:-10}"
MODELS="${MODELS:-evo1 pi0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/libero_b70}"
VENV_PY="$REPO_DIR/eval/sim/libero/libero_uv/.venv/bin/python"

echo "NODE:      $(hostname)"
echo "EPISODES:  $N_EPISODES per task, models: $MODELS"
echo "OUTPUT:    $OUTPUT_ROOT"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

source /swtools/intel-gpu/latest/intel_gpu_vars.sh || fail "TOOLCHAIN intel_gpu_vars.sh"
source /swtools/intel/setvars.sh --force >/dev/null 2>&1 || fail "TOOLCHAIN setvars.sh"

[ -x "$BUILD_DIR/vla-server" ] || fail "no vla-server in $BUILD_DIR -- SERVERS=ON ci/slurm/bmg_build.sbatch"
[ -x "$VENV_PY" ]              || fail "no LIBERO venv -- run eval/sim/libero/setup_libero.sh on the login node"
[ -d "$MODELS_ROOT" ]          || fail "no MODELS_ROOT at $MODELS_ROOT"
echo "GPU:       $(sycl-ls 2>/dev/null | grep -m1 level_zero)"

# The client generates its protobuf bindings at import time by shelling out to
# protoc (eval/client/vla_cpp_client.py), so the compiler has to be on PATH and
# not merely linked into vla-server. It comes from the same local prefix.
DEPS_PREFIX="${DEPS_PREFIX:-$HOME/opt/vla-deps}"
export PATH="$DEPS_PREFIX/bin:$PATH"
command -v protoc >/dev/null 2>&1 ||
    fail "protoc not on PATH -- run ci/build_server_deps.sh on the login node"
echo "PROTOC:    $(protoc --version)"

# MuJoCo picks a renderer at import time and neither run_libero.sh nor
# libero_env.py sets one; on a headless compute node the default lands on GLFW
# and dies without a display. egl renders on the GPU; osmesa is the software
# fallback if the Intel EGL stack refuses a surface.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="$MUJOCO_GL"
echo "MUJOCO_GL: $MUJOCO_GL"

# --- render smoke test ------------------------------------------------------
# The single thing here that cannot be checked from a login node, and the
# failure mode is ugly: the sweep would run for minutes per arm and produce ten
# empty summaries. Fail in thirty seconds instead.
if [ "${SMOKE:-1}" = "1" ]; then
  echo
  echo "=== render smoke test ==="
  ( cd "$REPO_DIR/eval" && "$VENV_PY" - <<'PY'
import os, sys
import numpy as np
try:
    import sim.libero  # noqa: F401  registers the gymnasium envs
    import gymnasium as gym
    # Registered as "<suite>/task_<id>" by sim/libero/libero_env.py.
    env = gym.make("libero_object/task_0")
    obs, _ = env.reset()
    # obs["pixels"] is itself a dict of camera views (the env is built with
    # video_view_mode="multi-view"), so this has to walk one level down.
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
    # An all-black frame is what a broken EGL context yields: reset succeeds and
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
  ) || fail "RENDER SMOKE TEST (MUJOCO_GL=$MUJOCO_GL) -- see the SMOKE line above for which stage"
fi

# --- the four arms ----------------------------------------------------------
# SKIP_BUILD=1: the tree was built by ci/slurm/bmg_build.sbatch under icpx, and
# run_libero.sh's plain `cmake --build` would relink with whatever compiler the
# cache happens to name.
run_arm() {   # $1 = dtype tag, $2.. = extra server args
  local dtype="$1"; shift
  for model in $MODELS; do
    echo
    echo "########## $model / $dtype ##########"
    local out="$OUTPUT_ROOT/$model-$dtype"
    # Cleared, not merged into: run_libero.sh writes one summary.txt per task and
    # never removes an old one, so a re-run at a different -n leaves the previous
    # episode count in place for every task it has not reached. The plot then
    # silently averages two different sample sizes together.
    rm -rf "$out"
    SKIP_BUILD=1 BUILD_DIR="$BUILD_DIR" EXTRA_SERVER_ARGS="$*" \
        bash "$REPO_DIR/eval/run_libero.sh" \
            -i "$MODELS_ROOT" -o "$out" -n "$N_EPISODES" -m "$model" \
        || fail "SWEEP $model/$dtype"
  done
}

run_arm f32
run_arm bf16 --act-dtype bf16

# The bf16 arms have to have actually engaged; an arch that declined the flag
# would produce a perfect, meaningless agreement with its own f32 arm.
for model in $MODELS; do
  grep -qh "activations = BF16" \
      "$OUTPUT_ROOT/$model-bf16/_server_logs/"*.log 2>/dev/null ||
      fail "BF16 ARM DID NOT ENGAGE for $model"
done

# --- collect and plot -------------------------------------------------------
echo
echo "=== summary ==="
ARMS=()
for model in $MODELS; do
  for dtype in f32 bf16; do
    ARMS+=(--arm "$model:$dtype=$OUTPUT_ROOT/$model-$dtype")
  done
done
# Suffixed by the model list: a run over one model would otherwise overwrite a
# previous run's figure with a plot that silently drops the other model's arms.
PLOT_TAG="$(echo $MODELS | tr ' ' '_')"
python3 "$REPO_DIR/scripts/plot_libero_bf16.py" "${ARMS[@]}" \
    --out "$REPO_DIR/docs/img/libero_bf16_b70-$PLOT_TAG" \
    --title "LIBERO object, Arc Pro B70 (SYCL), $N_EPISODES episodes/task" \
    || fail "PLOT"

echo
echo "VERDICT: OK"
echo "DONE"
