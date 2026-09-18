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
#   DQ_MODES=off sbatch ci/slurm/bmg_ov_quant_why_flat.sbatch   # job 372827's shape
#
# Knobs: FLAVOURS, DQ_MODES, TASK, OV_BUILD, OV_ROOT, PRECISION, IMG_SIZE,
#        ITERS, TRACE_ITERS, UNITRACE=0, JOBS.
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
#
# --- Stage C: the second axis ------------------------------------------------
#
# With (a) established the question moves from *how wide the weights are* to
# *what the compute is doing*, and the answer is floating point on a model
# designed for integers. BitVLA is natively W1.58A8: bitvla_act_quant_op has
# already rounded the activations onto an int8 lattice by the time a GEMM sees
# them (src/models/bitvla.cpp:71-98), but line 95 multiplies them straight back
# out to float, and translate_bitvla_act_quant mirrors that into OpenVINO as nine
# float nodes with no i8 tensor anywhere. The BitLinear GEMMs therefore run
# f16 x f16 carrying int8-valued data, and the int8 XMX pipeline goes unused.
#
# DQ_MODES adds ov::hint::dynamic_quantization_group_size as a second axis:
# "off" is every job before this one, "max" is per-token, which is act_quant's
# own granularity -- act_quant emits q*(amax/127) with q integral and amax the
# row absmax, so re-quantising per row recovers exactly q. Unusually, the
# coarsest group size is here also the most faithful.
#
# The prediction, so that this can fail: today bf16 < q8_0 < q4_0. If int8
# compute engages the order should INVERT, q8_0 fastest (symmetric i8, no unpack,
# no zero point). bf16 is the CONTROL -- dense weights produce no compressed
# FullyConnected, so there is nothing for the hint to attach to and it must not
# move. If bf16 moves, the reading is suspect.
#
# Latency alone is not engagement evidence, which is why the run under unitrace
# now has to work: a kernel-name change between the two dq arms is the signal,
# and the latency table is only interpretable once that signal exists.
#
# --- RESULT, job 372831 ------------------------------------------------------
#
# The XPTI shim works and unitrace produced its device-timing table on all six
# arms. (The job still printed ENGAGEMENT: UNDECIDED, because the glob looking
# for the table was wrong -- unitrace writes "<name>.<pid>.txt", inserting the
# pid before the extension, so nothing matched "<name>.txt*". Fixed above; the
# numbers below are read from the files that run left on disk.)
#
#   arm      device ms   gemm ms   gemm %   avg ms/step
#   bf16/off     406.8     144.6     35.6          53.7
#   bf16/max     407.1     144.7     35.5          54.0
#   q8_0/off     313.9     151.2     48.2          56.3
#   q8_0/max     314.7     150.8     47.9          55.4
#   q4_0/off     323.3     156.2     48.3          57.2
#   q4_0/max     323.2     156.1     48.3          57.1
#
# ENGAGEMENT: NO. Identical kernel name sets on and off (zero-line diff on all
# three flavours), total device time within 0.2%, gemm device time within 0.2%,
# and the predicted inversion did not appear. ov::hint::dynamic_quantization_-
# group_size is accepted -- the hunk logs it, so the property really is set --
# and changes nothing the device executes.
#
# WHY it did not engage is still open. This block originally answered it --
# "there is no fully_connected kernel in any arm, so the hint has nothing to
# attach to" -- and JOB 372841 DISPROVED THAT. Per-op profiling, which reports
# node identity rather than kernel name, shows FullyConnectedCompressed at 2040
# nodes and 47.3% of device time. ConvertMatMulToFC fires; the compressed FC the
# hint wants is there; the hint still changes nothing.
#
# The mistake was reading a kernel name as a primitive name. A KERNEL NAME
# IDENTIFIES AN IMPLEMENTATION, NEVER A PRIMITIVE, AND THE MAPPING IS MANY-TO-ONE.
# FullyConnectedCompressed and plain Gemm are both implemented by oneDNN's
# jit:gemm:any micro-kernel, whose generated OpenCL kernel object unitrace reports
# as `gemm_kernel`. Seeing only gemm_kernel is consistent with every matmul being
# an FC and tells us nothing either way. This is the structural reason unitrace
# could not answer the question, and it is why ci/slurm/bmg_ov_profile_ops.sbatch
# exists -- ov::InferRequest::get_profiling_info() is the only source carrying
# both op identity and device time.
#
# ENGAGEMENT: NO survives as a measurement. Only its explanation was wrong, and
# the replacement is not yet known.
#
# Two further findings from the same table. The first held; the second did not:
#
#   THE WEIGHTS REALLY ARE COMPRESSED, a second independent confirmation of job
#   372827. reorder_data is 132.4 ms (32.5%) on bf16 against 25.2 ms (8.0%) on
#   q8_0 over the *same* 389 calls. Reorder cost tracks bytes moved, so a plugin
#   that had decompressed q8_0 to f16 at compile time would reorder the same
#   bytes as bf16. It moves 5.3x fewer. This one stands.
#
#   THE FAKE-QUANT CHAIN IS THE LARGEST NON-GEMM COST -- ALSO WRONG, same cause.
#   The claim was ~30% of device time, from reduce_ref 54.6 ms + generic_eltwise_-
#   ref 32.6 ms + eltwise_simple_vload8 6.6 ms, attributed to act_quant's nine-node
#   chain by shape and call count. The paragraph flagged its own weakness
#   ("attribution is by shape and call count, not by node identity -- worth
#   confirming ... before it is spent against") and the flag was correct.
#
#   Job 372841 attributed it by node identity: act_quant is ~6% (Abs 1.00% +
#   ReduceMax 1.49% + Power 0.81% + Maximum 0.40% + a share of Multiply),
#   consistent with the SYCL backend's independent 8.0% for the same bucket. The
#   reduce_ref time was almost all RMSNorm, not ReduceMax -- 726 ReduceMean nodes
#   at 20.5% of device time, falling onto OpenVINO's *generic reference* reduction
#   because the frontend spelled the square Multiply(x, x) and ov::pass::RMSFusion
#   matches Power(x, 2). See ci/slurm/bmg_ov_rms_fusion_check.sbatch for what
#   happened when that one word was changed.
#
# So Stage C's premise holds -- the compute is floating point on an integer
# model -- but the lever the plan chose does not reach it, and the larger prize
# is the ~30% spent *simulating* quantisation rather than the int8 GEMM win the
# plan went looking for.
# =============================================================================
#SBATCH --job-name=bmg_ov_why_flat
#SBATCH --partition=b70
#SBATCH --nodelist=pcl-adl01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
# 6 arms now instead of 3, and the traced half actually runs this time.
#SBATCH --time=04:00:00
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
# Stage C's second axis: the GPU plugin's dynamic activation quantisation.
# "off" leaves GGML_OPENVINO_DYNAMIC_QUANT_GROUP unset (today's behaviour);
# "max" is per-token, which is act_quant's own granularity.
#
# The axis has since widened into "which graph variant", because the later
# experiments are graph changes rather than hint changes and they need the same
# thing from this job: an HONEST ms/step. Every other instrument in the tree runs
# under ov::enable_profiling(true), which perturbs what it measures and licenses
# only within-run ratios -- so this is the ONLY place a latency claim may come
# from. Values:
#
#   off   today's behaviour, nothing set
#   max   GGML_OPENVINO_DYNAMIC_QUANT_GROUP=max (Stage C's hint)
#   fc3   GGML_OPENVINO_FC_RANK3=1 (Stage D's rank squeeze)
#   onaq  fc3 + GGML_OPENVINO_ACT_QUANT_ELIDE=1 (act_quant elided)
#
# --- RESULT, job 372938: q4_0 x {off, fc3, onaq} -----------------------------
#
#   arm    host GiB   device GiB   avg ms
#   off        2.24         1.86   57.134
#   fc3        2.24         1.85   56.968     -0.3%
#   onaq       2.24         1.85   50.032    -12.4%
#
# The decomposition is the point. fc3 -- the whole of Stage D, the rank squeeze
# that engages the plugin's int8 GEMM -- buys 0.17 ms, i.e. nothing, which is
# Stage D's negative verdict reproduced on honest timing instead of a profiled
# total. ALL of the win is the act_quant elision. Device time agrees and is
# quieter: 324.8 -> 274.2 ms, with gemm 157.6 -> 128.7.
#
# -12.4% IS A CEILING, NOT THE DELIVERABLE. Job 372927 showed this onaq build
# also costs 104 of 340 compressed FCs their DynamicQuantize, so 31% of the
# BitLinears here run activations quantised by nobody. Skipping both quantisers
# is cheaper than running one, so the fixed version lands at or below -12.4%.
#
# Two pieces of this job's own output were NOT applicable to this invocation and
# must not be quoted from it:
#   - "VERDICT: DECOMPRESSED". FLAVOURS was q4_0 alone, so the spread compared
#     one arm to itself and read 1.00x by construction. Job 372827's 2.69x over
#     bf16/q8_0/q4_0 stands. The block now refuses to render below two flavours.
#   - "ENGAGEMENT: YES ... proceed to the accuracy sweep". That prose is Stage C's
#     and is about the dynamic-quantisation hint. The kernel-name diff is real,
#     but on a graph-variant arm it is describing a different change.
DQ_MODES="${DQ_MODES:-off max}"
PRECISION="${PRECISION:-f16}"
IMG_SIZE="${IMG_SIZE:-224}"
ITERS="${ITERS:-10}"
TRACE_ITERS="${TRACE_ITERS:-3}"
UNITRACE_BIN="${UNITRACE_BIN:-/swtools/intel-gpu/latest/unitrace/bin/unitrace}"
# libxptifw.so and its four transitive deps live here. See the XPTI shim below
# for why this directory is never put on LD_LIBRARY_PATH wholesale.
ONEAPI_LIB="${ONEAPI_LIB:-/swtools/intel/2026.0/lib}"
OUT="${OUT:-$REPO_DIR/outputs/ov_why_flat_${SLURM_JOB_ID:-0}}"

mkdir -p "$OUT"

echo "NODE:      $HOST"
echo "FLAVOURS:  $FLAVOURS"
echo "DQ_MODES:  $DQ_MODES"
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
# Leave the dynamic-quantisation knob unset by default; run_arm sets it per arm.
unset GGML_OPENVINO_DYNAMIC_QUANT_GROUP
# Same reason for the graph variants: common_env is the only thing allowed to
# set these, or the "off" control silently stops being a control.
unset GGML_OPENVINO_FC_RANK3
unset GGML_OPENVINO_ACT_QUANT_ELIDE

# --- XPTI shim --------------------------------------------------------------
# unitrace is built with BUILD_WITH_XPTI=1 and dlopens libxptifw.so, which job
# 372827 did not have (rc=127) because this job sources the GPU runtime and
# OpenVINO but not the compiler.
#
# Do NOT fix that by prepending $ONEAPI_LIB to LD_LIBRARY_PATH: that directory
# also holds libOpenCL.so.1 and TBB, and shadowing the OpenCL loader OpenVINO is
# already using would change what we are measuring in order to measure it. Five
# symlinks into a directory of their own gets unitrace what it needs and nothing
# else -- and is applied only to the traced invocation.
XPTI_SHIM="$OUT/xptishim"
mkdir -p "$XPTI_SHIM"
XPTI_OK=1
for l in libxptifw.so libsvml.so libirng.so libimf.so libintlc.so.5; do
  if [ -e "$ONEAPI_LIB/$l" ]; then
    ln -sf "$ONEAPI_LIB/$l" "$XPTI_SHIM/$l"
  else
    echo "  XPTI shim: $l missing from $ONEAPI_LIB -- tracing will be skipped"
    XPTI_OK=0
  fi
done
[ "$XPTI_OK" = 1 ] && echo "XPTI shim: $XPTI_SHIM (5 symlinks)"

# --- build ------------------------------------------------------------------
# Apply the ggml-openvino hunks explicitly. CMake runs the patcher as a
# FetchContent PATCH_COMMAND, i.e. once at populate time, so a build directory
# that already exists will never see a hunk added since it was configured -- and
# GGML_OPENVINO_DYNAMIC_QUANT_GROUP is exactly such a hunk. The patcher skips any
# hunk whose replacement is already present, so running it here is idempotent.
if [ -d "$LLAMA_SRC_MIRROR" ]; then
  "$PY" "$REPO_DIR/scripts/patch_ggml_openvino.py" "$LLAMA_SRC_MIRROR" || fail "PATCH"
fi

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
# Keyed on "flavour/dqmode" so the two axes share one set of tables.
declare -A DEV_MIB HOST_GIB AVG_MS OV_MS KERN_SIG DEV_NS GEMM_NS

# The empty second argument is load-bearing: predict_check reads argv[2] as an
# mmproj path whenever it is non-empty and not a flag.
#
# $1 = dq mode. "off" emits nothing at all rather than an empty value: the hunk
# treats empty as off, but not setting the variable is what every job before
# Stage C did, and the control arm should be that exact environment.
common_env() {
  echo GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_GPU_PRECISION="$PRECISION" \
       VLA_IMG_SIZE="$IMG_SIZE"
  case "$1" in
    off)  ;;
    fc3)  echo GGML_OPENVINO_FC_RANK3=1 ;;
    onaq) echo GGML_OPENVINO_FC_RANK3=1 GGML_OPENVINO_ACT_QUANT_ELIDE=1 ;;
    *)    echo GGML_OPENVINO_DYNAMIC_QUANT_GROUP="$1" ;;
  esac
}

run_arm() {   # $1 = flavour, $2 = dq mode
  local f="$1" dq="$2"
  local k="$f/$dq"
  local tag="$f.dq-$dq"
  local ckpt="$CKPT_DIR/$CKPT_STEM-$f.gguf"
  echo
  echo "########## $f  |  dynamic quant: $dq ##########"

  # --- clean run: device memory + honest timing -----------------------------
  # Backgrounded so the sampler can attach to the pid. The sampler polls until
  # the process exits, so starting it after launch cannot miss the peak, which
  # arrives long after compile.
  env $(common_env "$dq") GGML_OPENVINO_PROFILING=1 VLA_BENCH_ITERS="$ITERS" \
      "$OV_BUILD/tests/vla_predict_check" "$ckpt" "" 1 \
      > "$OUT/$tag.clean.txt" 2> "$OUT/$tag.clean.err" &
  local app_pid=$!
  "$PY" "$REPO_DIR/scripts/xpu_mem_sample.py" --pid "$app_pid" \
      --interval 0.2 --summary-key resident --out "$OUT/$tag.mem.json" \
      > "$OUT/$tag.mem.txt" 2>&1 &
  local smp_pid=$!
  wait "$app_pid"; local run_rc=$?
  wait "$smp_pid" 2>/dev/null

  if [ "$run_rc" != 0 ]; then
    tail -20 "$OUT/$tag.clean.err"
    echo "  FAILED $k: predict_check rc=$run_rc"
    return 1
  fi
  grep -qh "backend = OPENVINO" "$OUT/$tag.clean.txt" "$OUT/$tag.clean.err" ||
      { echo "  FAILED $k: did not use OpenVINO"; return 1; }
  grep -qh "using device GPU" "$OUT/$tag.clean.txt" "$OUT/$tag.clean.err" ||
      { echo "  FAILED $k: did not reach the GPU plugin"; return 1; }

  # The hunk logs the group size it inserted. On a "max" arm its absence means
  # the binary predates hunk 16, which would silently turn this into a duplicate
  # of the "off" arm -- the one failure mode that looks exactly like "no effect".
  # Only the group-size values go through hunk 16; the graph variants are
  # different hunks with no such log line, and asserting on it would fail them.
  case "$dq" in off|fc3|onaq) hunk16=0 ;; *) hunk16=1 ;; esac
  if [ "$hunk16" = 1 ]; then
    if grep -qh "dynamic quantization group size" "$OUT/$tag.clean.txt" "$OUT/$tag.clean.err"; then
      grep -h "dynamic quantization group size" "$OUT/$tag.clean.err" "$OUT/$tag.clean.txt" |
          head -1 | sed 's/^/  property     : /'
    else
      echo "  FAILED $k: the binary never logged a dynamic quantization group size"
      echo "    -- hunk 16 is not in this build; rebuild after patch_ggml_openvino.py"
      return 1
    fi
  fi

  HOST_GIB[$k]="$(sed -n 's/.*weights resident in \([0-9.]*\) GiB.*/\1/p' "$OUT/$tag.clean.txt" | head -1)"
  AVG_MS[$k]="$(sed -n 's/.*avg=\([0-9.]*\) ms.*/\1/p' "$OUT/$tag.clean.err" | head -1)"
  DEV_MIB[$k]="$(sed -n 's/^xpu_mem: PEAK \([0-9.]*\) MiB.*/\1/p' "$OUT/$tag.mem.txt" | head -1)"

  echo "  host weights : ${HOST_GIB[$k]:-?} GiB"
  sed -n 's/^xpu_mem: /  device mem   : /p' "$OUT/$tag.mem.txt" | head -12
  echo "  avg step     : ${AVG_MS[$k]:-?} ms over $ITERS iters"
  grep -h "last split:" "$OUT/$tag.clean.err" | sed 's/^/  /'

  # OV's own accounting. Summed rather than averaged: PROFILING prints one line
  # per compute call and a step is many graphs, so a single line is a fragment
  # of a step and comparing fragments across arms compares different graphs.
  OV_MS[$k]="$(awk '/OV raw infer time/ {gsub(/[^0-9.]/,"",$NF); s+=$NF} END{printf "%.1f", s}' \
               "$OUT/$tag.clean.err" 2>/dev/null)"
  echo "  OV raw infer : ${OV_MS[$k]:-?} ms summed over all compute calls"

  # --- traced run: what actually executed on the GPU ------------------------
  if [ "${UNITRACE:-1}" = "1" ] && [ -x "$UNITRACE_BIN" ] && [ "$XPTI_OK" = 1 ]; then
    # --opencl because the OpenVINO GPU plugin reaches the device through
    # OpenCL by its own architecture, whatever this repo prefers elsewhere.
    #
    # The shim goes on LD_LIBRARY_PATH for this command only, in a subshell, so
    # the clean run above and the next arm both keep the library resolution they
    # were measured with.
    (
      export LD_LIBRARY_PATH="$XPTI_SHIM:$LD_LIBRARY_PATH"
      "$UNITRACE_BIN" --opencl --device-timing --demangle \
          --output "$OUT/$tag.unitrace.txt" \
          env $(common_env "$dq") VLA_BENCH_ITERS="$TRACE_ITERS" \
          "$OV_BUILD/tests/vla_predict_check" "$ckpt" "" 1 \
          > "$OUT/$tag.trace.txt" 2> "$OUT/$tag.trace.err"
    )
    local tr_rc=$?
    local tf
    # unitrace inserts its pid *before* the extension -- asked for
    # "bf16.dq-off.unitrace.txt" it writes "bf16.dq-off.unitrace.2638060.txt".
    # Globbing on the name we passed finds nothing, which job 372831 reported as
    # "unitrace produced no device-timing table" while the table sat on disk.
    tf="$(ls -1 "$OUT/$tag.unitrace."* 2>/dev/null | head -1)"
    [ -z "$tf" ] && tf="$OUT/$tag.trace.err"    # some builds report to stderr
    if [ "$tr_rc" = 0 ] && grep -qi "Total .*Time\|Kernel," "$tf" 2>/dev/null; then
      cp "$tf" "$OUT/$tag.kernels.txt" 2>/dev/null
      # Printed, not parsed into a number. unitrace's device-timing table is a
      # formatted text report whose column layout is its own business, and a
      # total scraped out of it with a guessed field separator would be a number
      # nobody could check. The table itself answers the question: same kernel
      # names at the same durations across arms means one dense GEMM everywhere.
      echo "  GPU kernels by total device time (head):"
      head -25 "$OUT/$tag.kernels.txt" | sed 's/^/    /'

      # The table IS parseable after all -- it is a CSV with a header line
      # "Kernel, Calls, Time (ns), Time (%), ..." -- so take the name column
      # exactly, not a regex over the whole file that would also pick up the
      # per-kernel hash suffixes and the prose in the banner.
      awk -F, 'NF>5 && $1 !~ /Kernel/ {gsub(/^ +| +$/,"",$1); print $1}' \
          "$OUT/$tag.kernels.txt" | sort -u > "$OUT/$tag.kernelnames.txt"
      KERN_SIG[$k]="$OUT/$tag.kernelnames.txt"

      # Total and GEMM device time. This is the Amdahl number the Stage C plan
      # asked for -- gemm_kernel is every matmul, BitLinear and attention alike,
      # so the BitLinear share is at most this.
      DEV_NS[$k]="$(awk -F: '/Total Device Time/{gsub(/[^0-9]/,"",$2); print $2}' \
                    "$OUT/$tag.kernels.txt")"
      GEMM_NS[$k]="$(awk -F, '$1 ~ /gemm_kernel/ {gsub(/ /,"",$3); print $3}' \
                     "$OUT/$tag.kernels.txt" | head -1)"
      echo "  device time  : $(awk -v n="${DEV_NS[$k]:-0}" 'BEGIN{printf "%.1f", n/1e6}') ms" \
           "| gemm $(awk -v n="${GEMM_NS[$k]:-0}" 'BEGIN{printf "%.1f", n/1e6}') ms"
    else
      echo "  (unitrace produced no device-timing table, rc=$tr_rc -- non-fatal)"
      tail -5 "$OUT/$tag.trace.err" | sed 's/^/    /'
    fi
  fi
  return 0
}

rc=0
for f in $FLAVOURS; do
  for dq in $DQ_MODES; do
    run_arm "$f" "$dq" || rc=1
  done
done

# --- the discriminator ------------------------------------------------------
echo
echo "=== host vs device ==="
printf '%-8s %5s %14s %14s %12s %12s\n' arm dq "host GiB" "device GiB" "avg ms" "OV ms"
for f in $FLAVOURS; do
  for dq in $DQ_MODES; do
    k="$f/$dq"
    d="${DEV_MIB[$k]}"
    dg="-"; [ -n "$d" ] && dg="$(awk -v m="$d" 'BEGIN{printf "%.2f", m/1024}')"
    printf '%-8s %5s %14s %14s %12s %12s\n' \
        "$f" "$dq" "${HOST_GIB[$k]:--}" "$dg" "${AVG_MS[$k]:--}" "${OV_MS[$k]:--}"
  done
done

# The compression question is about the weight format and is settled on a single
# dq axis; use the first mode listed so FLAVOURS/DQ_MODES subsets still work.
REF_DQ="${DQ_MODES%% *}"
echo
echo "(compression verdict below reads the dq=$REF_DQ arms)"

echo
have_dev=1
for f in $FLAVOURS; do [ -n "${DEV_MIB[$f/$REF_DQ]}" ] || have_dev=0; done

# The verdict is a spread ACROSS WEIGHT FLAVOURS. With one flavour the spread is
# 1.00x by definition and the DECOMPRESSED branch fires on a tautology -- which
# it did in job 372938 (FLAVOURS=q4_0, run for its latency axis), contradicting
# job 372827's measured 2.69x for no reason but the arm list. Refuse to render.
if [ "$(echo $FLAVOURS | wc -w)" -lt 2 ]; then
  echo "NOT APPLICABLE: the compression verdict is a spread across weight flavours"
  echo "  and only one was run ($FLAVOURS). A single arm compares to itself and"
  echo "  would always read 1.00x. Job 372827 measured 2.69x over bf16/q8_0/q4_0"
  echo "  and that stands; re-run with FLAVOURS=\"bf16 q8_0 q4_0\" to revisit it."
elif [ "$have_dev" = 0 ]; then
  echo "INCONCLUSIVE: this driver publishes no per-process DRM fdinfo memory keys,"
  echo "  so device memory could not be read and neither hypothesis is excluded."
  echo "  See $OUT/*.mem.txt for what fdinfo did expose. The kernel tables in"
  echo "  $OUT/*.kernels.txt are then the only evidence: identical kernel names and"
  echo "  durations across arms would still indicate the plugin decompressed."
else
  # Ratio over whatever arms were actually run, so FLAVOURS="bf16 q4_0" works.
  lo=""; hi=""
  for f in $FLAVOURS; do
    d="${DEV_MIB[$f/$REF_DQ]}"
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

# --- Stage C: did dynamic quantisation engage? ------------------------------
# Deliberately separate from the verdict above and deliberately not a latency
# test. Stage B's lesson was that a flat curve has more than one cause, so a
# faster number here would not by itself mean int8 compute. Three signals are
# printed and each is labelled with what it can and cannot establish.
if [ "$(echo $DQ_MODES | wc -w)" -ge 2 ]; then
  ON_DQ="$(echo $DQ_MODES | awk '{print $NF}')"
  echo
  echo "=== Stage C: dynamic quantisation, $REF_DQ vs $ON_DQ ==="

  echo
  echo "-- signal 1: kernel name set (strongest) --"
  kern_changed=0; kern_seen=0
  for f in $FLAVOURS; do
    a="${KERN_SIG[$f/$REF_DQ]}"; b="${KERN_SIG[$f/$ON_DQ]}"
    if [ -n "$a" ] && [ -n "$b" ] && [ -s "$a" ] && [ -s "$b" ]; then
      kern_seen=1
      n="$(comm -3 "$a" "$b" | wc -l)"
      if [ "$n" = 0 ]; then
        printf '  %-8s identical kernel names\n' "$f"
      else
        kern_changed=1
        printf '  %-8s %s kernel names differ:\n' "$f" "$n"
        comm -3 "$a" "$b" | head -20 | sed 's/^/      /'
      fi
    else
      printf '  %-8s no kernel table on one or both arms\n' "$f"
    fi
  done
  [ "$kern_seen" = 0 ] && echo "  (unitrace produced nothing comparable -- this signal is unavailable,"
  [ "$kern_seen" = 0 ] && echo "   which is not the same as 'the kernels did not change')"

  echo
  echo "-- signal 1b: device time (stronger than wall clock: no host noise) --"
  printf '  %-8s %12s %12s %12s %12s %8s\n' arm "dev $REF_DQ" "dev $ON_DQ" \
      "gemm $REF_DQ" "gemm $ON_DQ" "gemm %"
  for f in $FLAVOURS; do
    do_ms() { awk -v n="${1:-0}" 'BEGIN{printf "%.1f", n/1e6}'; }
    gp="-"
    [ -n "${GEMM_NS[$f/$REF_DQ]}" ] && [ -n "${DEV_NS[$f/$REF_DQ]}" ] &&
        gp="$(awk -v g="${GEMM_NS[$f/$REF_DQ]}" -v d="${DEV_NS[$f/$REF_DQ]}" \
              'BEGIN{printf "%.1f", (d>0)?100*g/d:0}')"
    printf '  %-8s %12s %12s %12s %12s %8s\n' "$f" \
        "$(do_ms "${DEV_NS[$f/$REF_DQ]}")" "$(do_ms "${DEV_NS[$f/$ON_DQ]}")" \
        "$(do_ms "${GEMM_NS[$f/$REF_DQ]}")" "$(do_ms "${GEMM_NS[$f/$ON_DQ]}")" "$gp"
  done
  echo "  gemm % is the Amdahl ceiling: an int8 win applies to at most this much,"
  echo "  and less, because gemm_kernel also carries attention and the action head."

  echo
  echo "-- signal 2: latency delta, with bf16 as the control --"
  printf '  %-8s %10s %10s %10s\n' arm "$REF_DQ" "$ON_DQ" "delta %"
  for f in $FLAVOURS; do
    o="${AVG_MS[$f/$REF_DQ]}"; n="${AVG_MS[$f/$ON_DQ]}"
    d="-"
    [ -n "$o" ] && [ -n "$n" ] &&
        d="$(awk -v a="$o" -v b="$n" 'BEGIN{printf "%+.1f", (a>0)?100*(b-a)/a:0}')"
    printf '  %-8s %10s %10s %10s\n' "$f" "${o:--}" "${n:--}" "$d"
  done
  echo "  bf16 has dense weights, so there is no compressed FullyConnected for the"
  echo "  property to attach to and its delta should be ~0. If bf16 moves, the"
  echo "  property is changing something other than BitLinear GEMM precision and"
  echo "  every other row here is suspect."

  echo
  echo "-- signal 3: arm ordering --"
  echo "  Stage B measured bf16 < q8_0 < q4_0 (more compression, slightly slower)."
  echo "  If int8 compute engages this should invert: q8_0 fastest -- symmetric i8,"
  echo "  no nibble unpack, no zero point -- then q4_0, with bf16 unchanged."
  for dq in "$REF_DQ" "$ON_DQ"; do
    line="  dq=$dq:"
    for f in $FLAVOURS; do line="$line  $f=${AVG_MS[$f/$dq]:--}"; done
    echo "$line"
  done

  echo
  if [ "$kern_changed" = 1 ]; then
    echo "ENGAGEMENT: YES -- the plugin selected different kernels with the property"
    echo "  set. This is the signal Stage C asked for; the latency table above can"
    echo "  now be read as an int8-compute result. Proceed to the accuracy sweep."
  elif [ "$kern_seen" = 1 ]; then
    echo "ENGAGEMENT: NO -- identical kernel names on both arms. The property was"
    echo "  accepted (it is a hint, so it is always accepted) and changed nothing the"
    echo "  device executed. Any latency difference above is run-to-run noise and"
    echo "  must not be reported as int8 compute. Stage C falls back to patching"
    echo "  translate_bitvla_act_quant to emit a real i8 path (plan section 4)."
  else
    echo "ENGAGEMENT: UNDECIDED -- the deciding signal (kernel names) is missing"
    echo "  because unitrace produced no table. Do not promote the latency deltas"
    echo "  above to an engagement claim; fix the tracer first."
  fi
fi

echo
[ "$rc" = 0 ] || echo "NOTE: at least one arm failed to run; see FAILED lines above."
echo "DONE"
exit "$rc"
