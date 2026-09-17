#!/bin/bash
# =============================================================================
# h100_libero_repro.sh -- isolate the client SIGABRT seen in job 371832.
#
# In that job the CUDA side was healthy: vla-server built, loaded evo1, and
# served 51 requests at 85 ms with no complaint. The *client* died - three
# episode mp4s written, then a silent SIGABRT with no message in the log. A
# silent abort is a native library calling abort(), not a Python exception, so
# the sweep's stdout had nothing to show.
#
# Two things this changes over the sweep:
#   PYTHONFAULTHANDLER=1  a fatal signal now prints a Python traceback, which
#                         names the extension module that aborted.
#   TASK_IDS=0, N=5       one task, five episodes - it died on the fourth, so
#                         five is enough and a full sweep is not.
#
# And it tests both renderers in one allocation, because the h100 partition
# stays fully allocated for hours at a time and a second queue wait to try the
# obvious alternative is a bad trade. The leading hypothesis is a per-episode
# EGL context leak on the NVIDIA stack - the same venv does ten episodes a task
# on Intel without trouble - so osmesa (software, no GPU context at all) is the
# natural control. If egl aborts and osmesa does not, that is the answer.
#
# Submitted rather than srun'd on purpose: an srun dies with the shell that
# launched it, which is exactly how the first attempt at this was lost.
#
#   sbatch ci/slurm/h100_libero_repro.sh
#
# Knobs: TASK_IDS, N, M, RENDERERS, BUILD_DIR.
# =============================================================================
#SBATCH --job-name=h100_libero_repro
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=slurm_h100_libero_repro_%j.out

set -o pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-cuda-srv}"
export PATH="$HOME/opt/vla-deps/bin:$PATH"
export PYTHONFAULTHANDLER=1      # turn a fatal signal into a named traceback
export PYTHONUNBUFFERED=1        # ...that actually reaches the log before we die
ulimit -c unlimited

echo "NODE: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[ -x "$BUILD_DIR/vla-server" ] || { echo "VERDICT: FAILED no vla-server in $BUILD_DIR"; echo DONE; exit 1; }

rc_egl=""; rc_osmesa=""
for gl in ${RENDERERS:-egl osmesa}; do
  echo
  echo "########## MUJOCO_GL=$gl ##########"
  export MUJOCO_GL="$gl" PYOPENGL_PLATFORM="$gl"
  out="$REPO_DIR/outputs/libero_h100_repro/$gl"
  rm -rf "$out"
  SKIP_BUILD=1 BUILD_DIR="$BUILD_DIR" TASK_IDS="${TASK_IDS:-0}" \
      bash "$REPO_DIR/eval/run_libero.sh" \
          -i "${VLA_CACHE:-$HOME/.cache/vla}/vrfai" \
          -o "$out" -n "${N:-5}" -m "${M:-evo1}"
  rc=$?
  echo "EXIT[$gl]=$rc  episodes_written=$(find "$out" -name '*.mp4' 2>/dev/null | wc -l)"
  eval "rc_$gl=$rc"
done

echo
echo "=== verdict ==="
echo "egl=$rc_egl osmesa=$rc_osmesa  (rc 134 = SIGABRT)"
# An arm that wrote no episodes at all did not render once, so it is not a
# control - osmesa exits 1 at import on a node with no libOSMesa, which says
# nothing about the crash under test. Only an arm that actually rendered can
# acquit or convict a renderer.
osmesa_eps=$(find "$REPO_DIR/outputs/libero_h100_repro/osmesa" -name '*.mp4' 2>/dev/null | wc -l)
if [ "$rc_egl" = "0" ]; then
  echo "VERDICT: EGL SURVIVED N=${N:-5} on task ${TASK_IDS:-0}"
elif [ "$rc_osmesa" = "0" ]; then
  echo "VERDICT: EGL IS THE FAULT -- osmesa completed the same run"
elif [ "$osmesa_eps" = "0" ]; then
  echo "VERDICT: EGL FAILED, osmesa UNAVAILABLE (no frames rendered) -- no control, read the traceback"
else
  echo "VERDICT: BOTH RENDERERS FAILED -- not a renderer problem"
fi
echo "DONE"
