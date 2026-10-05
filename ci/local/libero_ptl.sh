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
# libero_ptl.sh -- LIBERO-object task success for the numerics-changing
# Panther Lake optimisations (docs/backend/ptl.md), each next to a control arm
# on the same machine and build:
#
#   bit:  fused kernels + fused attention (default)  vs  VLA_BITVLA_UNFUSED=1
#   pi0:  bf16 + --flash-attn 1                      vs  bf16
#   evo1: bf16 + --flash-attn 1                      vs  bf16
#
#   bash eval/sim/libero/setup_libero.sh            # once
#   SERVERS=ON ci/local/build.sh sycl
#   MODELS=bit ci/local/libero_ptl.sh               # one model, 50 episodes/arm
#
# Read the result as a success rate with its Wilson interval: the question is
# whether the optimised arm is indistinguishable from its control, not whether
# its actions match.
#
# Knobs: BUILD_DIR (build-sycl), MODELS (bit pi0 evo1), N_EPISODES_ALL (per
# task; default 10, and 5 for bit to match the B70 sweep), EPISODES_PER_PROC,
# MODELS_ROOT, OUTPUT_ROOT,
# MUJOCO_GL, SMOKE.
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-sycl}"
MODELS="${MODELS:-bit pi0 evo1}"
MODELS_ROOT="${MODELS_ROOT:-$VLA_CACHE/vrfai}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/local/libero_ptl}"

echo "MODELS:    $MODELS"
echo "OUTPUT:    $OUTPUT_ROOT"

load_oneapi
[ -x "$BUILD_DIR/vla-server" ] || fail "no vla-server in $BUILD_DIR -- SERVERS=ON ci/local/build.sh sycl"
check_server_deps
setup_mujoco
render_smoke

# arm name -> "env assignments|server args"
arms_for() {
  case "$1" in
    bit)  echo "opt||" ; echo "ctl|VLA_BITVLA_UNFUSED=1|" ;;
    pi0)  echo "opt||--act-dtype bf16 --flash-attn 1" ; echo "ctl||--act-dtype bf16" ;;
    evo1) echo "opt||--act-dtype bf16 --flash-attn 1" ; echo "ctl||--act-dtype bf16" ;;
  esac
}

ARMS=()
for model in $MODELS; do
  default_n=10
  [ "$model" = bit ] && default_n=5
  N_EPISODES="${N_EPISODES_ALL:-$default_n}"
  while IFS='|' read -r arm envs args; do
    out="$OUTPUT_ROOT/$model-$arm"
    echo
    echo "########## $model / $arm  (env: ${envs:-none}  args: ${args:-none}) ##########"
    (
      # shellcheck disable=SC2086
      [ -n "$envs" ] && export $envs
      export EXTRA_SERVER_ARGS="$args" N_EPISODES
      run_libero_chunked "$out" "$model" "$BUILD_DIR"
    ) || fail "SWEEP $model/$arm"
    ARMS+=(--arm "$model:$arm=$(arm_paths "$out")")
  done < <(arms_for "$model")
done

echo
echo "=== results ==="
for d in "$OUTPUT_ROOT"/*/; do
  echo "--- $(basename "$d")"
  cat "$d"/chunk*/summary.txt 2>/dev/null | grep -iE "success|total" | tail -3
done
PY="$(plot_python)"
if [ -n "$PY" ]; then
  "$PY" "$REPO_DIR/scripts/plot_libero_bf16.py" "${ARMS[@]}" \
      --out "$OUTPUT_ROOT/libero_ptl" \
      --title "LIBERO object, $(hostname -s) (Panther Lake, SYCL)" ||
      echo "  (plot failed -- the sweep outputs above are intact)"
fi
echo "VERDICT: OK"
echo "DONE"
