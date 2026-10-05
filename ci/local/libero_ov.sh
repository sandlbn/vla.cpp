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
# libero_ov.sh -- BitVLA LIBERO-object task success in MuJoCo on the OpenVINO
# GPU plugin. Local equivalent of ci/slurm/bmg_ov_libero.sbatch.
#
#   bash eval/sim/libero/setup_libero.sh                     # once
#   SERVERS=ON ci/local/build.sh ov
#   N_EPISODES=1 ARMS=q4_0:f16 ci/local/libero_ov.sh          # smoke, ~5 min
#   ci/local/libero_ov.sh                                     # bf16/q8_0/q4_0 at f16
#   ARMS="bf16:f32 bf16:f16" ci/local/libero_ov.sh            # precision only
#
# An arm is WEIGHTS:PRECISION[:DEVICE], as in bench_ov.sh:
#   WEIGHTS    bf16 | q8_0 | q4_0   checkpoint flavour, transcoded on first use
#   PRECISION  f32 | f16            GPU plugin compute precision (the NPU is f16)
#   DEVICE     GPU (default) | NPU  OpenVINO device, e.g. ARMS="q8_0:f16:NPU"
#
# Reference, Arc Pro B70, 5 ep/task (jobs 372740/372759): every arm 98-100%,
# f16 ~16-17 ms/step against f32's 37.5. A quantised arm that drops well below
# that is a bug, not a precision limit. Judge on success RATE: BitVLA's
# activation quantiser turns a 1-ULP input change into a visibly different
# action, so action-by-action comparison between arms says nothing.
#
# Knobs: BUILD_DIR (build-ov), ARMS, N_EPISODES (5), EPISODES_PER_PROC (10),
# TASK (libero_object), MODELS_ROOT, OUTPUT_ROOT, MUJOCO_GL (egl), SMOKE (1).
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-ov}"
ARMS="${ARMS:-bf16:f16 q8_0:f16 q4_0:f16}"
N_EPISODES="${N_EPISODES:-5}"
MODELS_ROOT="${MODELS_ROOT:-$VLA_CACHE/vrfai}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/local/libero_ov}"

echo "ARMS:      $ARMS, $N_EPISODES episodes/task"
echo "OUTPUT:    $OUTPUT_ROOT"

load_openvino
[ -x "$BUILD_DIR/vla-server" ] || fail "no vla-server in $BUILD_DIR -- SERVERS=ON ci/local/build.sh ov"
check_server_deps

# Validate and materialise every checkpoint before the first episode, so a typo
# in the last arm does not surface hours in.
declare -A CKPT
arm_w()   { echo "${1%%:*}"; }
arm_p()   { local r="${1#*:}"; echo "${r%%:*}"; }
arm_dev() { local r="${1#*:}"; [ "$r" = "${r#*:}" ] && echo GPU || echo "${r#*:}"; }
for arm in $ARMS; do
  w="$(arm_w "$arm")"; p="$(arm_p "$arm")"
  case "$w" in bf16|q8_0|q4_0) ;; *) fail "arm $arm: weights must be bf16|q8_0|q4_0" ;; esac
  case "$p" in f32|f16) ;;           *) fail "arm $arm: precision must be f32|f16" ;; esac
  case "$(arm_dev "$arm")" in GPU|NPU) ;; *) fail "arm $arm: device must be GPU|NPU" ;; esac
  CKPT[$arm]="$(bitvla_ckpt "$w")" || fail "CHECKPOINT $w"
done

setup_mujoco
render_smoke

arm_out() { echo "$OUTPUT_ROOT/bitvla-ov-$(arm_w "$1")-$(arm_p "$1")$([ "$(arm_dev "$1")" = GPU ] || echo "-$(arm_dev "$1")")"; }

for arm in $ARMS; do
  out="$(arm_out "$arm")"
  echo
  dev="$(arm_dev "$arm")"
  echo "########## bitvla / OpenVINO $dev / $(arm_w "$arm") weights / $(arm_p "$arm") compute ##########"
  echo "CKPT: ${CKPT[$arm]}  ($(du -h "${CKPT[$arm]}" | cut -f1))"
  GGML_OPENVINO_DEVICE="$dev" GGML_OPENVINO_GPU_PRECISION="$(arm_p "$arm")" \
  BITVLA_CKPT="${CKPT[$arm]}" \
      run_libero_chunked "$out" bit "$BUILD_DIR" || fail "SWEEP $arm"

  # A CPU fallback still completes tasks -- slowly, and with a latency that would
  # be reported as the GPU's. Both lines must be in the server log.
  grep -qh "backend = OPENVINO" "$out"/chunk*/_server_logs/*.log 2>/dev/null ||
      fail "$arm DID NOT USE OPENVINO"
  grep -qh "using device $dev" "$out"/chunk*/_server_logs/*.log 2>/dev/null ||
      fail "$arm DID NOT REACH THE $dev PLUGIN"
  grep -h "weights resident in" "$out"/chunk*/_server_logs/*.log 2>/dev/null | tail -1
done

echo
echo "=== results ==="
PLOT_ARMS=()
for arm in $ARMS; do
  PLOT_ARMS+=(--arm "bitvla:$(arm_w "$arm") w / $(arm_p "$arm") / $(arm_dev "$arm")=$(arm_paths "$(arm_out "$arm")")")
done
PY="$(plot_python)"
if [ -n "$PY" ]; then
  "$PY" "$REPO_DIR/scripts/plot_libero_bf16.py" "${PLOT_ARMS[@]}" \
      --out "$OUTPUT_ROOT/libero_ov" \
      --title "LIBERO object, $(hostname -s) (OpenVINO GPU), weights / compute, $N_EPISODES ep/task" ||
      echo "  (plot failed -- the sweep outputs above are intact)"
else
  echo "  no python with matplotlib -- skipping the plot; summaries are under $OUTPUT_ROOT"
fi
echo "VERDICT: OK"
echo "DONE"
