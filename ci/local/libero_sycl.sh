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
# libero_sycl.sh -- LIBERO-object task success in MuJoCo, SYCL backend, f32 vs
# bf16 activations. Local equivalent of ci/slurm/bmg_libero_bf16.sbatch.
#
#   bash eval/sim/libero/setup_libero.sh          # once: simulator + venv
#   SERVERS=ON ci/local/build.sh sycl             # vla-server is the model end
#   N_EPISODES=2 MODELS=evo1 ci/local/libero_sycl.sh   # quick look, ~10 min
#   ci/local/libero_sycl.sh                       # evo1 + pi0, 10 ep/task
#
# What runs: vla-server serves the model on the GPU; eval/run_libero.sh drives
# the LIBERO MuJoCo client against it over ZeroMQ, 10 tasks x N_EPISODES.
#
# Read the result as a success RATE with its Wilson interval, which the plot
# draws: 10 episodes per task is 10 coin flips, and 9/10 vs 8/10 is not a
# difference. Do not compare actions between f32 and bf16 -- per-step action
# deltas do not predict task success on these models.
#
# Knobs: BUILD_DIR (build-sycl), MODELS (evo1 pi0), N_EPISODES (10),
# EPISODES_PER_PROC (10), MODELS_ROOT ($VLA_CACHE/vrfai), OUTPUT_ROOT,
# MUJOCO_GL (egl), SMOKE (1).
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-sycl}"
MODELS="${MODELS:-evo1 pi0}"
N_EPISODES="${N_EPISODES:-10}"
MODELS_ROOT="${MODELS_ROOT:-$VLA_CACHE/vrfai}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/local/libero_sycl}"

echo "MODELS:    $MODELS, $N_EPISODES episodes/task"
echo "OUTPUT:    $OUTPUT_ROOT"

load_oneapi
[ -x "$BUILD_DIR/vla-server" ] || fail "no vla-server in $BUILD_DIR -- SERVERS=ON ci/local/build.sh sycl"
[ -d "$MODELS_ROOT" ] || fail "no MODELS_ROOT at $MODELS_ROOT -- pull the evo1/pi0 GGUFs first (see ci/local/README.md)"
check_server_deps
setup_mujoco
render_smoke

for dtype in f32 bf16; do
  extra=""
  [ "$dtype" = bf16 ] && extra="--act-dtype bf16"
  for model in $MODELS; do
    echo
    echo "########## $model / SYCL / $dtype activations ##########"
    EXTRA_SERVER_ARGS="$extra" \
        run_libero_chunked "$OUTPUT_ROOT/$model-$dtype" "$model" "$BUILD_DIR" ||
        fail "SWEEP $model/$dtype"
  done
done

# An arch that declined --act-dtype bf16 would agree perfectly with its own f32
# arm, and that agreement would mean nothing.
for model in $MODELS; do
  grep -qh "activations = BF16" "$OUTPUT_ROOT/$model-bf16/"chunk*/_server_logs/*.log 2>/dev/null ||
      fail "BF16 ARM DID NOT ENGAGE for $model"
done

echo
echo "=== results ==="
ARMS=()
for model in $MODELS; do
  for dtype in f32 bf16; do
    ARMS+=(--arm "$model:$dtype=$(arm_paths "$OUTPUT_ROOT/$model-$dtype")")
  done
done
PY="$(plot_python)"
if [ -n "$PY" ]; then
  "$PY" "$REPO_DIR/scripts/plot_libero_bf16.py" "${ARMS[@]}" \
      --out "$OUTPUT_ROOT/libero_sycl" \
      --title "LIBERO object, $(hostname -s) (SYCL), $N_EPISODES episodes/task" ||
      echo "  (plot failed -- the sweep outputs above are intact)"
else
  echo "  no python with matplotlib -- skipping the plot; summaries are under $OUTPUT_ROOT"
fi
echo "VERDICT: OK"
echo "DONE"
