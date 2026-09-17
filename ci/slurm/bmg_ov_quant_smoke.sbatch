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
# bmg_ov_quant_smoke.sbatch -- do the quantised BitVLA checkpoints work at all?
#
# ci/slurm/bmg_ov_libero.sbatch is the real gate for Stage B, and at three arms
# x 10 tasks x 10 episodes it is most of a day. This is the twenty minutes that
# decide whether that day is worth spending: load bf16, q8_0 and q4_0 on the
# OpenVINO GPU plugin, run one forward each, and check the three things that
# would make the long job a waste.
#
#   FOOTPRINT   the loader's own "weights resident in %.2f GiB" line. This is
#               what the stage is for. Expect 5.39 / 3.34 / 2.24 GiB -- i.e. the
#               plan's 5.78 / 3.58 / 2.40 GB, which is the same three numbers in
#               decimal units -- against the 10.77 GiB (11.57 GB) a bf16 file
#               used to occupy when the loader upconverted it to F32.
#               Job 372752 measured 5.39 / 3.34 / 2.24 exactly.
#
#   RESIDENCY   the banner names the BitLinear type and the dense type
#               separately ("resident=q4_0/bf16"). A checkpoint that was
#               silently read back as bf16 would produce perfect actions at the
#               wrong size, and the footprint line is the only thing that says
#               so -- which is why both are checked rather than just the first.
#
#   NO FALLBACK the one that decides the day. ggml's scheduler asks the backend
#               about every op, and an op it refuses runs on ggml-CPU instead --
#               correct actions, footprint win intact, latency ruined, and
#               nothing in the output says so. GGML_OPENVINO_LOG_UNSUPPORTED_OPS
#               makes each refusal a log line; a MUL_MAT among them on a
#               quantised arm means the GEMMs this stage exists to shrink are
#               being done on the CPU.
#
#               Reading ggml-openvino says they should not be: Q8_0 and Q4_0 are
#               both in `supported_types` (ggml-openvino.cpp:1356) and the only
#               quantised MUL_MAT the GPU path excludes is one named benchmark
#               shape (:1231). The check is here because "should" is an argument
#               and a fallback is silent.
#
# What this job deliberately does NOT check any more is whether the IR carries
# i8/u4 constants. Job 372755 chased that and found the question already settled
# upstream of any dump: create_weight_node (ggml-decoder.cpp:1290) routes every
# quantised tensor through process_weight_tensor -> extract_q{4,8}_0_data, and
# has no dequantise-to-f16 branch at all. The dump also cannot answer it here --
# vla.cpp's graphs take the frontend's *naive* path, whose dump is a single
# fixed-name IR_naive.xml that each graph overwrites, so what lands on disk is
# whichever graph happened to compile last (it was the 8x17920 action head, all
# f32/bf16 by nature). GGML_OPENVINO_DUMP_IR is still set, and the element types
# still printed, but for the record only -- DUMP_IR=0 turns it off.
#
# Whether the plugin then fuses those constants into FullyConnectedCompressed is
# a separate, latency-shaped question, answered by vla-bench and not by this job.
#
# Actions are printed but not compared. BitVLA has no usable action bar (see
# docs/backend/ov.md, "BitVLA has no action bar"); the block encoders are
# measurably *more* faithful than bf16, so a wrong number here would be a layout
# bug, and a layout bug is what the footprint and residency checks catch.
#
#   sbatch ci/slurm/bmg_ov_quant_smoke.sbatch
#   FLAVOURS="q4_0" sbatch ci/slurm/bmg_ov_quant_smoke.sbatch
#
# Knobs: FLAVOURS, TASK, OV_BUILD, OV_ROOT, PRECISION, IMG_SIZE, JOBS, DUMP_IR=0.
# =============================================================================
#SBATCH --job-name=bmg_ov_quant_smoke
#SBATCH --partition=b70
#SBATCH --nodelist=pcl-adl01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm_bmg_ov_quant_smoke_%j.out

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
OUT="${OUT:-$REPO_DIR/outputs/ov_quant_smoke_${SLURM_JOB_ID:-0}}"

mkdir -p "$OUT"

echo "NODE:      $HOST"
echo "FLAVOURS:  $FLAVOURS"
echo "PRECISION: $PRECISION"
echo "OUT:       $OUT"

fail() { echo "VERDICT: FAILED $1"; echo "DONE"; exit 1; }

for f in $FLAVOURS; do
  [ -f "$CKPT_DIR/$CKPT_STEM-$f.gguf" ] ||
      fail "no $f checkpoint -- DTYPES=\"$f\" sbatch ci/slurm/bitvla_bf16_gguf.sbatch"
done

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
rc=0
declare -A RESIDENT

run_flavour() {   # $1 = flavour
  local f="$1"
  local ckpt="$CKPT_DIR/$CKPT_STEM-$f.gguf"
  local ir="$OUT/ir-$f"
  echo
  echo "########## $f ##########"
  echo "CKPT: $ckpt  ($(du -h "$ckpt" | cut -f1) on disk)"
  mkdir -p "$ir"

  # GGML_OPENVINO_DUMP_IR is read with ggml_openvino_getenv_int -- it is a
  # boolean, and ov::serialize writes model_<ts>.xml into the *current working
  # directory*. Job 372752 passed it a path, atoi("/path") returned 0, and the
  # dump silently never happened while the check read the empty dir as "the
  # frontend decompressed the weights". So: set it to 1 and cd into the dump dir.
  local dump=()
  [ "${DUMP_IR:-1}" = "1" ] && dump=(GGML_OPENVINO_DUMP_IR=1)

  # The empty second argument is load-bearing: predict_check reads argv[2] as an
  # mmproj path whenever it is non-empty and not a flag.
  ( cd "$ir" &&
    env GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_GPU_PRECISION="$PRECISION" \
        GGML_OPENVINO_LOG_UNSUPPORTED_OPS=1 \
        VLA_IMG_SIZE="$IMG_SIZE" VLA_BENCH_ITERS=1 "${dump[@]}" \
        "$OV_BUILD/tests/vla_predict_check" "$ckpt" "" 1 \
        > "$OUT/$f.txt" 2> "$OUT/$f.err" )
  local run_rc=$?

  grep -hE "resident=|weights resident|backend =|using device" \
       "$OUT/$f.txt" "$OUT/$f.err" | head -6
  if [ "$run_rc" != 0 ]; then
    tail -20 "$OUT/$f.err"
    echo "  FAILED $f: predict_check rc=$run_rc"
    rc=1
    return
  fi

  # Engagement. An arm that fell back would pass everything else below.
  grep -qh "backend = OPENVINO" "$OUT/$f.txt" "$OUT/$f.err" ||
      { echo "  FAILED $f: did not use OpenVINO"; rc=1; return; }
  grep -qh "using device GPU" "$OUT/$f.txt" "$OUT/$f.err" ||
      { echo "  FAILED $f: did not reach the GPU plugin"; rc=1; return; }

  # Residency: the loader's own account of what it kept the weights as.
  if grep -qh "resident=$f/" "$OUT/$f.txt"; then
    echo "  ok      banner says resident=$f/..."
  else
    echo "  FAILED $f: banner does not say resident=$f/ -- read back as something else"
    rc=1
  fi
  RESIDENT[$f]="$(sed -n 's/.*weights resident in \([0-9.]*\) GiB.*/\1/p' "$OUT/$f.txt" | head -1)"

  # No fallback: every op the backend refused, and whether any GEMM is among them.
  # ggml's default log callback writes to stderr, but grep both streams rather
  # than depend on vla.cpp not having installed one of its own.
  local n_unsup n_mm
  grep -h "OpenVINO op unsupported" "$OUT/$f.txt" "$OUT/$f.err" > "$OUT/$f.unsup" 2>/dev/null
  n_unsup=$(wc -l < "$OUT/$f.unsup")
  n_mm=$(grep -c "(MUL_MAT" "$OUT/$f.unsup")
  if [ "$n_mm" -gt 0 ]; then
    echo "  FAILED $f: $n_mm MUL_MAT refused by the backend -- the GEMMs ran on ggml-CPU"
    grep "(MUL_MAT" "$OUT/$f.unsup" | sed 's/^/    /' | sort -u | head -5
    rc=1
  else
    echo "  ok      no MUL_MAT fell back ($n_unsup ops refused in total)"
  fi
  if [ "$n_unsup" -gt 0 ]; then
    echo "  refused ops by reason:"
    sed -n 's/.*op .[^ ]* (\([A-Z_]*\)).*: \(.*\)/    \1: \2/p' "$OUT/$f.unsup" |
        sort | uniq -c | sort -rn | head -8
  fi

  # For the record only; see the header for why this cannot gate anything.
  if [ "${DUMP_IR:-1}" = "1" ] && ls "$ir"/*.xml >/dev/null 2>&1; then
    echo "  IR (last naive graph only, not gating) constants by element type:"
    grep -roh 'element_type="[a-z0-9]*"' "$ir"/*.xml |
        sort | uniq -c | sort -rn | sed 's/^/    /'
    # The .bin beside each .xml is a full copy of the weights; the element types
    # live in the .xml. Tens of GiB per flavour otherwise.
    rm -f "$ir"/*.bin
  fi

  # Printed, not compared: there is no action bar on this architecture.
  sed -n '/^action_len=/,+4p' "$OUT/$f.txt"
}

for f in $FLAVOURS; do run_flavour "$f"; done

echo
echo "=== resident footprint ==="
printf '%-8s %10s %10s\n' flavour GiB "vs bf16"
base="${RESIDENT[bf16]}"
for f in $FLAVOURS; do
  g="${RESIDENT[$f]:-?}"
  if [ -n "$base" ] && [ "$g" != "?" ]; then
    printf '%-8s %10s %9sx\n' "$f" "$g" "$(awk -v b="$base" -v g="$g" 'BEGIN{printf "%.2f", b/g}')"
  else
    printf '%-8s %10s %10s\n' "$f" "$g" "-"
  fi
done
echo "(expect 5.39 / 3.34 / 2.24 GiB = the plan's 5.78 / 3.58 / 2.40 GB in decimal"
echo " units; 10.77 GiB is what a bf16 file cost before the loader stopped"
echo " upconverting it to F32.)"

echo
if [ "$rc" = 0 ]; then
  echo "VERDICT: PASSED  all flavours load, stay compressed and reach the GPU plugin"
  echo "NEXT:    sbatch ci/slurm/bmg_ov_libero.sbatch"
else
  echo "VERDICT: FAILED  see FAILED lines above"
fi
echo "DONE"
exit "$rc"
