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
# bmg_ov_libero.sbatch -- BitVLA on OpenVINO, judged on task success.
#
# This is the gate for the OpenVINO port of BitVLA, and it is a task-success
# gate rather than an action-equality one because action equality was measured
# and shown to be unusable on this architecture.
#
# The evidence (ggml-CPU only, no OpenVINO in the measurement; job 372738,
# ci/slurm/bmg_bitvla_sensitivity.sbatch): perturbing the LM's input embeddings
# by one f32 ULP moves the action chunk by 0.068 normalised, and the curve is
# flat from there down --
#
#     eps 1.2e-07 -> 0.0682      eps 1e-05 -> 0.0747      eps 1e-03 -> 0.1361
#     eps 1e-06   -> 0.0675      eps 1e-04 -> 0.0837      eps 6.3e-03 -> 0.1427
#
# act_quant's nearbyintf is a hard threshold evaluated ~200 times per forward
# pass, so a sub-ULP disagreement moves an integer rather than a low bit. BitVLA
# therefore has a decorrelation floor around 0.068, which is 23x the 2.9e-3 bar
# the other ten architectures are held to: no implementation that is not
# bit-identical to ggml-CPU can pass it, including a perfectly correct one.
# Confirming this, the OpenVINO LM on byte-identical inputs scores 0.0702 --
# sitting on the floor (job 372736).
#
# Above that floor the numbers are draws from a decorrelated distribution and do
# not even rank configurations: ovgpu32 scored 0.1092 against ovcpu's 0.1336,
# which means nothing. So neither f32-vs-f16 nor bf16-vs-q4_0 can be decided by
# comparing action vectors, and this job decides them by running the policy.
#
# An arm is `weight:precision` -- which checkpoint flavour, at which GPU compute
# precision -- because the two are independent and both need answering:
#
#   weight     bf16 | q8_0 | q4_0   how the ternary weights are stored on disk.
#              2.0000 / 1.0625 / 0.5625 bytes per weight. The block formats are
#              *more* faithful than bf16, not less: bf16 spends its mantissa
#              re-representing a per-tensor scale that a block stores once per
#              32 weights (max rel err 1.34e-03 / 5.89e-05 / 1.78e-04).
#              Produced by ci/slurm/bitvla_bf16_gguf.sbatch.
#   precision  f16 | f32            what the GPU plugin computes in.
#
# Stage A settled the precision axis: f16 and f32 both scored 100.0% over 10
# tasks x 10 episodes, at 16.0 vs 37.5 ms/step (job 372740), so backend.h's
# GGML_OPENVINO_GPU_PRECISION=f32 default for vla(bitvla) came out as its own
# comment promised, and every arm here is f16. Stage B is the weight axis, and
# its reference is that same 100.0% / 16.0 ms/step.
#
# The gate is success rate; the footprint each arm reports (the "weights
# resident in" line) is the thing being bought with it.
#
# Prerequisites, one-time, on the LOGIN node (this job has no route out):
#   bash ci/build_server_deps.sh          # protobuf + libzmq + cppzmq
#   bash eval/sim/libero/setup_libero.sh  # LIBERO + its venv
#   DTYPES="bf16 q8_0 q4_0" sbatch ci/slurm/bitvla_bf16_gguf.sbatch
#
#   sbatch ci/slurm/bmg_ov_libero.sbatch
#   N_EPISODES=2 sbatch ci/slurm/bmg_ov_libero.sbatch    # quicker shakedown
#   ARMS="bf16:f16 bf16:f32" sbatch ci/slurm/bmg_ov_libero.sbatch   # Stage A
#   ARMS="q4_0:f16 q4_0:f16:onaq" sbatch ci/slurm/bmg_ov_libero.sbatch
#       the act_quant elision against its own baseline. This is the ONLY
#       instrument that can price it: job 372927 showed the elision also costs
#       104 of 340 BitLinears their dynamic quantiser, so 31% of them run
#       activations unquantised, and neither the op profile nor the chunk-spread
#       screen can say whether that matters. Reference is q4_0's 50/50.
#
# Knobs: N_EPISODES, ARMS, CKPT_DIR, BUILD_DIR, OV_ROOT, OUTPUT_ROOT, SMOKE=0.
# =============================================================================
#SBATCH --job-name=bmg_ov_libero
#SBATCH --partition=b70
#SBATCH --nodelist=pcl-adl01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --output=slurm_bmg_ov_libero_%j.out

# No `set -u`: the vendor toolchain scripts read unset variables.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
HOST="$(hostname -s)"
# Its own build directory: build-ov-$HOST is configured VLA_BUILD_SERVERS=OFF and
# is reused by the other OpenVINO jobs, and flipping that option forces a
# reconfigure that would churn their cache.
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-ovsrv-$HOST}"
JOBS="${JOBS:-$(nproc)}"
OV_ROOT="${OV_ROOT:-$HOME/intel/openvino}"
DEPS_PREFIX="${DEPS_PREFIX:-$HOME/opt/vla-deps}"
LLAMA_SRC_MIRROR="${LLAMA_SRC_MIRROR:-$HOME/vla-deps/llama-src}"
MODELS_ROOT="${MODELS_ROOT:-${VLA_CACHE:-$HOME/.cache/vla}/vrfai}"
TASK="${TASK:-libero_object}"
CKPT_DIR="${CKPT_DIR:-$MODELS_ROOT/bitvla-libero-gguf/$TASK}"
CKPT_STEM="${CKPT_STEM:-bitvla-${TASK//_/-}}"
N_EPISODES="${N_EPISODES:-5}"
ARMS="${ARMS:-bf16:f16 q8_0:f16 q4_0:f16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/libero_ov_b70}"
VENV_PY="$REPO_DIR/eval/sim/libero/libero_uv/.venv/bin/python"

echo "NODE:      $HOST"
echo "CKPT_DIR:  $CKPT_DIR"
echo "EPISODES:  $N_EPISODES per task"
echo "ARMS:      $ARMS"
echo "OUTPUT:    $OUTPUT_ROOT"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

# A bare arm is a precision on the bf16 checkpoint -- what ARMS meant before the
# weight axis existed, so Stage A's invocations still mean what they meant.
# A third field is the graph variant: weight:precision:variant, default "base".
#   base  the shipping graph
#   fc3   GGML_OPENVINO_FC_RANK3 -- the rank squeeze (Stage D, closed negative)
#   onaq  fc3 plus GGML_OPENVINO_ACT_QUANT_ELIDE -- act_quant elided and the
#         plugin's own dynamic quantiser left to derive the codes instead
arm_weight() { case "$1" in *:*) echo "${1%%:*}";; *) echo "bf16";; esac; }
arm_prec()   { local r="${1#*:}"; case "$1" in *:*) echo "${r%%:*}";; *) echo "$1";; esac; }
arm_var()    { case "$1" in *:*:*) echo "${1##*:}";; *) echo "base";; esac; }
# Callers splice this into `env`. It is invoked inside $( ), so it CANNOT fail
# the job from here -- a `fail` would exit only the subshell and leave the caller
# running a mislabelled base arm. Validation is in the pre-flight loop below,
# which runs in the main shell; keep the two lists in step.
arm_env() {
  case "$(arm_var "$1")" in
    fc3)  echo "GGML_OPENVINO_FC_RANK3=1" ;;
    onaq) echo "GGML_OPENVINO_FC_RANK3=1 GGML_OPENVINO_ACT_QUANT_ELIDE=1" ;;
  esac
}
arm_ckpt()   { echo "$CKPT_DIR/$CKPT_STEM-$(arm_weight "$1").gguf"; }
# The one place an arm's output directory is spelled. The variant is appended
# only when it is not "base", so every earlier sweep's directory keeps its name
# and PLOT_EXTRA_ARMS still resolves.
arm_out() {
  local o="$OUTPUT_ROOT/bitvla-ov-$(arm_weight "$1")-$(arm_prec "$1")"
  [ "$(arm_var "$1")" = base ] || o="$o-$(arm_var "$1")"
  echo "$o"
}
# Plot label: the weight flavour alone, unless a variant needs telling apart.
arm_label() {
  case "$(arm_var "$1")" in
    base) arm_weight "$1" ;;
    *)    echo "$(arm_weight "$1")/$(arm_var "$1")" ;;
  esac
}

# Every checkpoint up front: an hour into a sweep is a bad time to learn that
# the q4_0 transcode was never made.
for arm in $ARMS; do
  c="$(arm_ckpt "$arm")"
  [ -f "$c" ] || fail "arm $arm: no checkpoint at $c -- run DTYPES=\"$(arm_weight "$arm")\" sbatch ci/slurm/bitvla_bf16_gguf.sbatch"
  case "$(arm_var "$arm")" in
    base|fc3|onaq) ;;
    *) fail "arm $arm: unknown graph variant '$(arm_var "$arm")' (want: base fc3 onaq)" ;;
  esac
done

[ -x "$VENV_PY" ] || fail "no LIBERO venv -- run eval/sim/libero/setup_libero.sh on the login node"
[ -f "$DEPS_PREFIX/include/zmq.hpp" ] || fail "no cppzmq under $DEPS_PREFIX -- run ci/build_server_deps.sh on the login node"

# --- toolchain --------------------------------------------------------------
source /swtools/intel-gpu/latest/intel_gpu_vars.sh || fail "TOOLCHAIN intel_gpu_vars.sh"
[ -f "$OV_ROOT/setupvars.sh" ] || fail "no OpenVINO runtime at $OV_ROOT"
source "$OV_ROOT/setupvars.sh" >/dev/null 2>&1 || fail "TOOLCHAIN setupvars.sh"
export OCL_ICD_VENDORS="${OCL_ICD_VENDORS:-/swtools/intel-gpu/latest/neo}"
# Cached OpenVINO blobs reload a graph that computes the wrong thing
# (docs/backend/ov.md); backend.h strips this unless VLA_ALLOW_OV_CACHE=1.
unset GGML_OPENVINO_CACHE_DIR
# The graph-variant vars come from the arm, never from the submitting shell: a
# stale export would silently turn a "base" control into something else.
unset GGML_OPENVINO_FC_RANK3
unset GGML_OPENVINO_ACT_QUANT_ELIDE

# The client generates its protobuf bindings at import time by shelling out to
# protoc, so the compiler has to be on PATH and not merely linked into vla-server.
export PATH="$DEPS_PREFIX/bin:$PATH"
command -v protoc >/dev/null 2>&1 || fail "protoc not on PATH"
echo "PROTOC:    $(protoc --version)"

# MuJoCo picks a renderer at import time; on a headless node the default lands on
# GLFW and dies without a display.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="$MUJOCO_GL"
echo "MUJOCO_GL: $MUJOCO_GL"

# --- build ------------------------------------------------------------------
FC_ARGS=()
[ -d "$LLAMA_SRC_MIRROR" ] && FC_ARGS=(-DFETCHCONTENT_SOURCE_DIR_LLAMA="$LLAMA_SRC_MIRROR")

if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  echo "CONFIGURE: $BUILD_DIR (OPENVINO=ON, SERVERS=ON)"
  cmake -B "$BUILD_DIR" -S "$REPO_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_OPENVINO=ON \
      -DVLA_BUILD_TESTS=ON \
      -DVLA_BUILD_SERVERS=ON \
      -DCMAKE_PREFIX_PATH="$DEPS_PREFIX" \
      "${FC_ARGS[@]}" 2>&1 | tail -20 || fail "CONFIGURE"
else
  echo "CONFIGURE: reusing $BUILD_DIR"
fi
echo "BUILD: -j$JOBS"
cmake --build "$BUILD_DIR" -j"$JOBS" 2>&1 | tail -10
[ "${PIPESTATUS[0]}" = 0 ] || fail "BUILD"
[ -x "$BUILD_DIR/vla-server" ] || fail "no vla-server in $BUILD_DIR"

# --- render smoke test ------------------------------------------------------
# The one thing that cannot be checked from a login node, and the failure mode is
# ugly: the sweep runs for hours and produces empty summaries. Fail in thirty
# seconds instead.
if [ "${SMOKE:-1}" = "1" ]; then
  echo
  echo "=== render smoke test ==="
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
        print("SMOKE: reset ok but no image in obs"); sys.exit(3)
    k, img = sorted(imgs.items())[0]
    # An all-black frame is what a broken EGL context yields.
    if int(img.max()) == int(img.min()):
        print(f"SMOKE: FAILED {k} is uniform ({img.min()})"); sys.exit(4)
    print(f"SMOKE: ok, {k} {img.shape}, range {img.min()}-{img.max()}")
    env.close()
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"SMOKE: FAILED {type(e).__name__}: {e}")
    sys.exit(5)
PY
  ) || fail "SMOKE"
fi

# --- the arms ---------------------------------------------------------------
# BITVLA_CKPT points run_libero.sh at a transcode of the published ladder-packed
# int2 file, which only the CUDA and SYCL kernels read.
# SKIP_BUILD=1: the tree is built above; run_libero.sh's plain `cmake --build`
# would relink with whatever compiler the cache happens to name.
run_arm() {   # $1 = arm, weight:precision
  local arm="$1"
  local prec weight var CKPT out
  prec="$(arm_prec "$arm")"
  weight="$(arm_weight "$arm")"
  var="$(arm_var "$arm")"
  CKPT="$(arm_ckpt "$arm")"
  # The variant is in the path only when it is not the shipping graph, so every
  # earlier job's output directory keeps the name it had.
  out="$(arm_out "$arm")"
  # shellcheck disable=SC2207
  local venv=($(arm_env "$arm"))
  echo
  echo "########## bitvla / OpenVINO GPU / $weight weights / $prec compute / graph $var ##########"
  echo "CKPT: $CKPT  ($(du -h "$CKPT" | cut -f1))"
  [ ${#venv[@]} -eq 0 ] || echo "GRAPH ENV: ${venv[*]}"
  # Cleared, not merged into: run_libero.sh writes one summary.txt per task and
  # never removes an old one, so a re-run at a different -n leaves the previous
  # episode count in place for every task it did not reach.
  rm -rf "$out"
  env GGML_OPENVINO_DEVICE=GPU \
      GGML_OPENVINO_GPU_PRECISION="$prec" \
      "${venv[@]}" \
      BITVLA_CKPT="$CKPT" \
      SKIP_BUILD=1 BUILD_DIR="$BUILD_DIR" \
      bash "$REPO_DIR/eval/run_libero.sh" \
          -i "$MODELS_ROOT" -o "$out" -n "$N_EPISODES" -m bit \
      || fail "SWEEP $arm"

  # An arm that quietly fell back to the CPU backend would produce a perfectly
  # respectable success rate while testing nothing this job exists to test.
  grep -qh "backend = OPENVINO" "$out/_server_logs/"*.log 2>/dev/null ||
      fail "$arm ARM DID NOT USE OPENVINO"
  grep -qh "using device GPU" "$out/_server_logs/"*.log 2>/dev/null ||
      fail "$arm ARM DID NOT REACH THE GPU PLUGIN"

  # The footprint this arm was run to buy, straight from the loader's own line.
  # Also the only place a silently-demoted weight flavour shows up: if the
  # loader read a q4_0 file as bf16, the success rate would be perfect and the
  # resident figure would be wrong.
  grep -h "weights resident in" "$out/_server_logs/"*.log 2>/dev/null | tail -1
}

for arm in $ARMS; do run_arm "$arm"; done

# --- summary ----------------------------------------------------------------
echo
echo "=== success rates ==="
for arm in $ARMS; do
  out="$(arm_out "$arm")"
  printf '%-18s ' "$arm"
  "$VENV_PY" - "$out" <<'PY'
import glob, os, re, sys
root = sys.argv[1]
tot = suc = 0
ms = []
# Anchored on the "Success rate:" line. A bare (\d+)/(\d+) would also match the
# "Skipped (terminated mid-step): 0/10" line in the same file.
rate = re.compile(r"Success rate:.*?\((\d+)\s*/\s*(\d+)\)")
lat = re.compile(r"Average inference time per step:\s*([0-9.]+)\s*ms")
files = sorted(glob.glob(os.path.join(root, "**", "summary.txt"), recursive=True))
for f in files:
    txt = open(f).read()
    m = rate.search(txt)
    if m:
        suc += int(m.group(1)); tot += int(m.group(2))
    m = lat.search(txt)
    if m:
        ms.append(float(m.group(1)))
if tot:
    lat_s = f"  {sum(ms)/len(ms):6.1f} ms/step" if ms else ""
    print(f"{suc:3d}/{tot:3d} = {100.0*suc/tot:5.1f}%   over {len(files)} tasks{lat_s}")
else:
    print(f"no summaries found under {root}")
PY
done

echo
echo "=== footprint ==="
for arm in $ARMS; do
  out="$(arm_out "$arm")"
  printf '%-18s ' "$arm"
  grep -h "weights resident in" "$out/_server_logs/"*.log 2>/dev/null | tail -1 ||
      echo "(no loader line)"
done

echo
echo "=== plot ==="
# Arms in the order given, which is the ladder order: the panels are read left to
# right as decreasing weight width, and sorting them by score would destroy that.
# Extra arms from earlier sweeps under the same OUTPUT_ROOT can be prepended with
# PLOT_EXTRA_ARMS, which is how the dense f32/f16 reference bars get into the
# figure without re-running them.
PLOT_ARMS=()
for a in ${PLOT_EXTRA_ARMS:-}; do PLOT_ARMS+=(--arm "$a"); done
for arm in $ARMS; do
  PLOT_ARMS+=(--arm "$(arm_label "$arm")=$(arm_out "$arm")")
done
"$VENV_PY" "$REPO_DIR/scripts/plot_ov_quant.py" "${PLOT_ARMS[@]}" \
    --out "$REPO_DIR/docs/img/bitvla_ov_quant_b70" \
    --title "BitVLA weight flavours on OpenVINO GPU, LIBERO object, Arc Pro B70" ||
    echo "  (plot failed -- the numbers above still stand)"

echo
echo "REFERENCE: bf16:f16 scored 100.0% over 10 tasks at 16.0 ms/step (job 372740),"
echo "           with the weights resident dense. A block-quantised arm that holds"
echo "           that success rate at a smaller resident figure is the whole point;"
echo "           one that drops it is a layout bug, not a precision limit -- the"
echo "           encoders are more faithful than bf16 (ci/test_bitvla_quant.py)."
echo "DONE"
