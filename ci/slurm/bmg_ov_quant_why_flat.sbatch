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
# bmg_ov_quant_why_flat.sbatch -- why is q4_0 no faster than bf16?
#
# Job 372759 measured 16.6 / 16.2 / 17.1 ms/step for bf16 / q8_0 / q4_0 across a
# 2.4x range of resident weight size. docs/backend/ov.md and the Stage B plan
# both currently read that flat curve as "decode at this model size is not
# weight-bandwidth-bound", and the plan then uses it to argue *against* native
# element::u2. That is a conclusion drawn from an absence, and exactly one other
# mechanism produces the same absence:
#
#   (a) COMPUTE-BOUND.  A VLA step is prefill-shaped -- a ViT over 256 patches
#       and a prefill over hundreds of tokens, then an 8x7 action chunk. At
#       M ~ 300 rows the GEMMs have enough arithmetic intensity that weight bytes
#       stop being the limit, so shrinking them buys nothing. Weight-only
#       quantisation is a *decode* optimisation and this is not a decode.
#       Corroborated already by f32 -> f16: 37.5 -> 16.0 ms/step from halving
#       compute precision with the weights byte-identical in both arms.
#
#   (b) THE PLUGIN DECOMPRESSED THEM.  The plan's own stated Risk:
#       "ConvertFullyConnectedToFullyConnectedCompressed may not fire on every
#       BitLinear shape, in which case the plugin decompresses to f16 and runs a
#       dense GEMM: correct, same footprint win, no bandwidth win." The frontend
#       hands the plugin u4/i8 constants (that much is settled -- see the smoke
#       job's header), but what the plugin does with them afterwards is its own
#       decision, and utils.cpp:571 is explicit that "the plugin holds its own
#       device copy of every weight after compile".
#
# Everything measured so far is host-side. "weights resident" is the ggml host
# buffer and peak RSS is host RSS, and both shrink under (b) exactly as they do
# under (a). Device memory is the discriminator and it has never been read on
# this platform: eval/client/benchmark.py samples VRAM via nvidia-smi, so every
# *.mem.json here carries peak_vram_mib: null.
#
# The two hypotheses predict device figures 2.4x apart:
#
#     arm     (a) compressed on device    (b) decompressed to f16
#     bf16          ~5.4 GiB                     ~5.4 GiB
#     q8_0          ~3.3 GiB                     ~5.4 GiB
#     q4_0          ~2.2 GiB                     ~5.4 GiB
#
# So this job measures, per arm:
#
#   DEVICE MEM  scripts/xpu_mem_sample.py, reading per-process DRM fdinfo. If
#               q4_0 holds what it says it holds, (a). If all three hold the
#               same, (b) and the footprint win is host-only.
#
#   GPU KERNELS unitrace --device-timing: every kernel that ran, its total GPU
#               time and call count. Under (b) the three arms execute the *same*
#               kernels for the same durations. This also localises the 16 ms --
#               if the GEMMs are a minority of it, (a) holds regardless.
#
#   SPLIT       predict_check's own "vision=.. inference=.. total=.." line plus
#               GGML_OPENVINO_PROFILING, which separates OV raw infer time from
#               the ggml-side graph handling around it.
#
# Two runs per arm: one clean (timing that means something) and one under
# unitrace (kernel breakdown, tracer overhead and all). The unitrace half is
# best-effort -- if it fails, the device-memory half still decides the question.
#
#   sbatch ci/slurm/bmg_ov_quant_why_flat.sbatch
#   FLAVOURS="bf16 q4_0" sbatch ci/slurm/bmg_ov_quant_why_flat.sbatch
#
# Knobs: FLAVOURS, TASK, OV_BUILD, OV_ROOT, PRECISION, IMG_SIZE, ITERS,
#        TRACE_ITERS, UNITRACE=0, JOBS.
#
# --- RESULT, job 372827 ------------------------------------------------------
#
#   arm    host weights   device VRAM   device GTT   ms/step
#   bf16       5.39 GiB      4.99 GiB     4.77 GiB      53.5
#   q8_0       3.34 GiB      2.93 GiB     2.51 GiB      55.3
#   q4_0       2.24 GiB      1.85 GiB     1.32 GiB      57.0
#
# 2.69x spread on device memory: (a). The weights are executed compressed,
# ConvertFullyConnectedToFullyConnectedCompressed does fire, and the flat latency
# curve is the workload being compute-bound rather than an artefact.
#
# The latency column is the second, unplanned confirmation: it rises
# *monotonically* with compression, +6.6% bf16 -> q4_0. That is the nibble
# unpacking in the GEMM inner loop, which costs nothing on a bandwidth-bound
# kernel because it hides behind memory stalls, and costs this much here because
# there are none to hide in. q4_0 is not merely no faster, it is slightly slower,
# which is what a FLOP-limited kernel doing extra ALU work has to look like.
#
# These ms are ~3x the LIBERO sweep's 16-17 because predict_check runs a full
# forward per iteration with no KV reuse. Only the across-arm comparison is
# meaningful -- and it reproduces the sweep's flat curve on a second harness.
#
# Two instruments in here produced nothing on that run and were not needed:
#
#   unitrace                  rc=127, "libxptifw.so: cannot open shared object
#                             file". The binary at /swtools/intel-gpu/latest is
#                             built with BUILD_WITH_XPTI=1 and wants the SYCL
#                             runtime's XPTI framework on LD_LIBRARY_PATH, which
#                             this OpenVINO-only environment does not source.
#                             Sourcing /swtools/intel/2026.0 setvars would fix it.
#                             Left failing-but-non-fatal on purpose: it is the
#                             corroborating instrument, not the deciding one.
#
#   GGML_OPENVINO_PROFILING   emitted no "infer time" lines at all on this path,
#                             so the OV ms column reads 0.0 throughout. Not
#                             chased; device memory had already decided it.
# =============================================================================
#SBATCH --job-name=bmg_ov_why_flat
#SBATCH --partition=b70
#SBATCH --nodelist=pcl-adl01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm_bmg_ov_why_flat_%j.out

# No `set -u`: the vendor toolchain scripts read unset variables.
set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
HOST="$(hostname -s)"
OV_BUILD="${OV_BUILD:-$REPO_DIR/build-ov-$HOST}"
JOBS="${JOBS:-$(nproc)}"
OV_ROOT="${OV_ROOT:-$HOME/intel/openvino}"
LLAMA_SRC_MIRROR="${LLAMA_SRC_MIRROR:-$HOME/vla-deps/llama-src}"
TASK="${TASK:-libero_object}"
CKPT_DIR="${CKPT_DIR:-$HOME/.cache/vla/vrfai/bitvla-libero-gguf/$TASK}"
CKPT_STEM="${CKPT_STEM:-bitvla-${TASK//_/-}}"
FLAVOURS="${FLAVOURS:-bf16 q8_0 q4_0}"
PRECISION="${PRECISION:-f16}"
IMG_SIZE="${IMG_SIZE:-224}"
ITERS="${ITERS:-10}"
TRACE_ITERS="${TRACE_ITERS:-3}"
UNITRACE_BIN="${UNITRACE_BIN:-/swtools/intel-gpu/latest/unitrace/bin/unitrace}"
OUT="${OUT:-$REPO_DIR/outputs/ov_why_flat_${SLURM_JOB_ID:-0}}"

mkdir -p "$OUT"

echo "NODE:      $HOST"
echo "FLAVOURS:  $FLAVOURS"
echo "PRECISION: $PRECISION"
echo "ITERS:     $ITERS (clean) / $TRACE_ITERS (traced)"
echo "OUT:       $OUT"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

for f in $FLAVOURS; do
  [ -f "$CKPT_DIR/$CKPT_STEM-$f.gguf" ] ||
      fail "no $f checkpoint -- DTYPES=\"$f\" sbatch ci/slurm/bitvla_bf16_gguf.sbatch"
done

PY="$(command -v python3)" || fail "no python3"

# --- toolchain --------------------------------------------------------------
source /swtools/intel-gpu/latest/intel_gpu_vars.sh || fail "TOOLCHAIN intel_gpu_vars.sh"
[ -f "$OV_ROOT/setupvars.sh" ] || fail "no OpenVINO runtime at $OV_ROOT"
source "$OV_ROOT/setupvars.sh" >/dev/null 2>&1 || fail "TOOLCHAIN setupvars.sh"
export OCL_ICD_VENDORS="${OCL_ICD_VENDORS:-/swtools/intel-gpu/latest/neo}"
# Cached blobs reload a graph that computes the wrong thing; docs/backend/ov.md.
unset GGML_OPENVINO_CACHE_DIR

# --- build ------------------------------------------------------------------
FC_ARGS=()
[ -d "$LLAMA_SRC_MIRROR" ] && FC_ARGS=(-DFETCHCONTENT_SOURCE_DIR_LLAMA="$LLAMA_SRC_MIRROR")
if [ ! -f "$OV_BUILD/CMakeCache.txt" ]; then
  echo "CONFIGURE: $OV_BUILD"
  cmake -B "$OV_BUILD" -S "$REPO_DIR" \
      -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON \
      -DVLA_BUILD_TESTS=ON -DVLA_BUILD_SERVERS=OFF \
      "${FC_ARGS[@]}" >/dev/null 2>&1 || fail "CONFIGURE"
else
  echo "CONFIGURE: reusing $OV_BUILD"
fi
cmake --build "$OV_BUILD" -j"$JOBS" --target vla_predict_check 2>&1 | tail -5
[ "${PIPESTATUS[0]}" = 0 ] || fail "BUILD"

# --- arms -------------------------------------------------------------------
declare -A DEV_MIB HOST_GIB AVG_MS OV_MS

# The empty second argument is load-bearing: predict_check reads argv[2] as an
# mmproj path whenever it is non-empty and not a flag.
common_env() {
  echo GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_GPU_PRECISION="$PRECISION" \
       VLA_IMG_SIZE="$IMG_SIZE"
}

run_flavour() {   # $1 = flavour
  local f="$1"
  local ckpt="$CKPT_DIR/$CKPT_STEM-$f.gguf"
  echo
  echo "########## $f ##########"

  # --- clean run: device memory + honest timing -----------------------------
  # Backgrounded so the sampler can attach to the pid. The sampler polls until
  # the process exits, so starting it after launch cannot miss the peak, which
  # arrives long after compile.
  env $(common_env) GGML_OPENVINO_PROFILING=1 VLA_BENCH_ITERS="$ITERS" \
      "$OV_BUILD/tests/vla_predict_check" "$ckpt" "" 1 \
      > "$OUT/$f.clean.txt" 2> "$OUT/$f.clean.err" &
  local app_pid=$!
  "$PY" "$REPO_DIR/scripts/xpu_mem_sample.py" --pid "$app_pid" \
      --interval 0.2 --summary-key resident --out "$OUT/$f.mem.json" \
      > "$OUT/$f.mem.txt" 2>&1 &
  local smp_pid=$!
  wait "$app_pid"; local run_rc=$?
  wait "$smp_pid" 2>/dev/null

  if [ "$run_rc" != 0 ]; then
    tail -20 "$OUT/$f.clean.err"
    echo "  FAILED $f: predict_check rc=$run_rc"
    return 1
  fi
  grep -qh "backend = OPENVINO" "$OUT/$f.clean.txt" "$OUT/$f.clean.err" ||
      { echo "  FAILED $f: did not use OpenVINO"; return 1; }
  grep -qh "using device GPU" "$OUT/$f.clean.txt" "$OUT/$f.clean.err" ||
      { echo "  FAILED $f: did not reach the GPU plugin"; return 1; }

  HOST_GIB[$f]="$(sed -n 's/.*weights resident in \([0-9.]*\) GiB.*/\1/p' "$OUT/$f.clean.txt" | head -1)"
  AVG_MS[$f]="$(sed -n 's/.*avg=\([0-9.]*\) ms.*/\1/p' "$OUT/$f.clean.err" | head -1)"
  DEV_MIB[$f]="$(sed -n 's/^xpu_mem: PEAK \([0-9.]*\) MiB.*/\1/p' "$OUT/$f.mem.txt" | head -1)"

  echo "  host weights : ${HOST_GIB[$f]:-?} GiB"
  sed -n 's/^xpu_mem: /  device mem   : /p' "$OUT/$f.mem.txt" | head -12
  echo "  avg step     : ${AVG_MS[$f]:-?} ms over $ITERS iters"
  grep -h "last split:" "$OUT/$f.clean.err" | sed 's/^/  /'

  # OV's own accounting. Summed rather than averaged: PROFILING prints one line
  # per compute call and a step is many graphs, so a single line is a fragment
  # of a step and comparing fragments across arms compares different graphs.
  OV_MS[$f]="$(awk '/OV raw infer time/ {gsub(/[^0-9.]/,"",$NF); s+=$NF} END{printf "%.1f", s}' \
               "$OUT/$f.clean.err" 2>/dev/null)"
  echo "  OV raw infer : ${OV_MS[$f]:-?} ms summed over all compute calls"

  # --- traced run: what actually executed on the GPU ------------------------
  if [ "${UNITRACE:-1}" = "1" ] && [ -x "$UNITRACE_BIN" ]; then
    # --opencl because the OpenVINO GPU plugin reaches the device through
    # OpenCL by its own architecture, whatever this repo prefers elsewhere.
    "$UNITRACE_BIN" --opencl --device-timing --demangle \
        --output "$OUT/$f.unitrace.txt" \
        env $(common_env) VLA_BENCH_ITERS="$TRACE_ITERS" \
        "$OV_BUILD/tests/vla_predict_check" "$ckpt" "" 1 \
        > "$OUT/$f.trace.txt" 2> "$OUT/$f.trace.err"
    local tr_rc=$?
    local tf
    tf="$(ls -1 "$OUT/$f.unitrace.txt"* 2>/dev/null | head -1)"
    [ -z "$tf" ] && tf="$OUT/$f.trace.err"    # some builds report to stderr
    if [ "$tr_rc" = 0 ] && grep -qi "Total .*Time\|Kernel," "$tf" 2>/dev/null; then
      cp "$tf" "$OUT/$f.kernels.txt" 2>/dev/null
      # Printed, not parsed into a number. unitrace's device-timing table is a
      # formatted text report whose column layout is its own business, and a
      # total scraped out of it with a guessed field separator would be a number
      # nobody could check. The table itself answers the question: same kernel
      # names at the same durations across arms means one dense GEMM everywhere.
      echo "  GPU kernels by total device time (head):"
      head -20 "$OUT/$f.kernels.txt" | sed 's/^/    /'
    else
      echo "  (unitrace produced no device-timing table, rc=$tr_rc -- non-fatal)"
      tail -5 "$OUT/$f.trace.err" | sed 's/^/    /'
    fi
  fi
  return 0
}

rc=0
for f in $FLAVOURS; do run_flavour "$f" || rc=1; done

# --- the discriminator ------------------------------------------------------
echo
echo "=== host vs device ==="
printf '%-8s %14s %14s %12s %12s\n' arm "host GiB" "device GiB" "avg ms" "OV ms"
for f in $FLAVOURS; do
  d="${DEV_MIB[$f]}"
  dg="-"; [ -n "$d" ] && dg="$(awk -v m="$d" 'BEGIN{printf "%.2f", m/1024}')"
  printf '%-8s %14s %14s %12s %12s\n' \
      "$f" "${HOST_GIB[$f]:--}" "$dg" "${AVG_MS[$f]:--}" "${OV_MS[$f]:--}"
done

echo
have_dev=1
for f in $FLAVOURS; do [ -n "${DEV_MIB[$f]}" ] || have_dev=0; done

if [ "$have_dev" = 0 ]; then
  echo "INCONCLUSIVE: this driver publishes no per-process DRM fdinfo memory keys,"
  echo "  so device memory could not be read and neither hypothesis is excluded."
  echo "  See $OUT/*.mem.txt for what fdinfo did expose. The kernel tables in"
  echo "  $OUT/*.kernels.txt are then the only evidence: identical kernel names and"
  echo "  durations across arms would still indicate the plugin decompressed."
else
  # Ratio over whatever arms were actually run, so FLAVOURS="bf16 q4_0" works.
  lo=""; hi=""
  for f in $FLAVOURS; do
    d="${DEV_MIB[$f]}"
    [ -z "$lo" ] && { lo="$d"; hi="$d"; }
    lo="$(awk -v a="$lo" -v b="$d" 'BEGIN{print (b<a)?b:a}')"
    hi="$(awk -v a="$hi" -v b="$d" 'BEGIN{print (b>a)?b:a}')"
  done
  ratio="$(awk -v h="$hi" -v l="$lo" 'BEGIN{printf "%.2f", (l>0)?h/l:0}')"
  echo "device memory spread across arms: ${ratio}x  (min ${lo} MiB, max ${hi} MiB)"
  if awk -v r="$ratio" 'BEGIN{exit !(r < 1.3)}'; then
    echo
    echo "VERDICT: DECOMPRESSED -- every arm holds the same device memory."
    echo "  The plugin is expanding the u4/i8 constants to f16 at compile time, so"
    echo "  all three arms run the identical dense GEMM and the flat latency curve"
    echo "  is an artefact, not a property of the workload. The footprint win is"
    echo "  real but HOST-ONLY. docs/backend/ov.md's 'not weight-bandwidth-bound'"
    echo "  claim must be withdrawn, and native element::u2 is back on the table."
  else
    echo
    echo "VERDICT: COMPRESSED -- device memory tracks the weight format."
    echo "  The weights really are executed compressed, so the flat latency is the"
    echo "  workload telling us it is compute-bound, not bandwidth-bound: a VLA"
    echo "  step is prefill-shaped and weight-only quantisation is a decode"
    echo "  optimisation. docs/backend/ov.md stands, and native element::u2 would"
    echo "  buy footprint only -- as the plan concluded."
  fi
fi

echo
[ "$rc" = 0 ] || echo "NOTE: at least one arm failed to run; see FAILED lines above."
echo "DONE"
exit "$rc"
