#!/usr/bin/env python3
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
"""Plot BitVLA's weight flavours on the OpenVINO backend.

The claim this stage makes is three-dimensional and only one dimension is the
interesting one, so all three are drawn together: the weights get 4.8x smaller,
the robot keeps completing the task, and the latency does not move. Any two of
those alone would be misleading -- a footprint plot without success rates is a
compression benchmark, and a success plot without footprints says nothing at all.

    python3 scripts/plot_ov_quant.py \
        --arm f32=outputs/libero_ov_b70/bitvla-ov-f32 \
        --arm f16=outputs/libero_ov_b70/bitvla-ov-f16 \
        --arm bf16=outputs/libero_ov_b70/bitvla-ov-bf16-f16 \
        --arm q8_0=outputs/libero_ov_b70/bitvla-ov-q8_0-f16 \
        --arm q4_0=outputs/libero_ov_b70/bitvla-ov-q4_0-f16 \
        --out docs/img/bitvla_ov_quant_b70

Pass the arms in ladder order (widest weights first); the panels keep the order
given, because these are an ordered sequence rather than a set of alternatives
and sorting them by score would destroy the only reading that matters.

Each arm path is a sweep root, as written by ci/slurm/bmg_ov_libero.sbatch:

    <root>/**/libero_object/task_<id>/summary.txt   success + latency
    <root>/_server_logs/*.log                       "weights resident in N GiB"
    <root>/_server_logs/*.mem.json                  sampled peak process RSS

The two memory numbers come from different places on purpose. The banner figure
is the loader's own account of what it allocated for weights; peak RSS is what
the OS watched the process do. Plotting only the first would be taking the
program's word for it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a compute node
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reuse the sibling plot's loaders rather than restating them. load_arm carries
# the warnings that matter here -- stale summaries from a re-run at a different
# episode count, and missing tasks -- and a second copy would drift.
_spec = importlib.util.spec_from_file_location(
    "plot_libero_bf16", REPO_ROOT / "scripts" / "plot_libero_bf16.py"
)
_sib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sib)
load_arm, wilson, totals = _sib.load_arm, _sib.wilson, _sib.totals

# Slots 1 and 2 of the validated categorical palette, in fixed order. Only two
# are needed: arm identity lives on the x axis, shared by all three panels, so
# the only thing colour has to separate is the two parts of the memory stack.
C_WEIGHTS = "#2a78d6"   # blue   - the part this stage shrinks
C_REST = "#eb6834"      # orange - the part it does not
INK = "#0b0b0b"
INK_MUTED = "#52514e"

MIB = 1024.0


def resident_gib(root: Path) -> float | None:
    """The loader's own 'weights resident in N GiB', from the server log."""
    for log in sorted(root.glob("_server_logs/*.log")):
        m = re.search(r"weights resident in ([0-9.]+) GiB", log.read_text(errors="replace"))
        if m:
            return float(m.group(1))
    return None


def peak_rss_gib(root: Path) -> float | None:
    """Sampled peak RSS, written by the eval harness's memory sampler."""
    for js in sorted(root.glob("_server_logs/*.mem.json")):
        try:
            v = json.loads(js.read_text()).get("peak_rss_mib")
        except json.JSONDecodeError:
            continue
        if v:
            return float(v) / MIB
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=PATH",
                    help="e.g. q4_0=outputs/libero_ov_b70/bitvla-ov-q4_0-f16 (repeatable, "
                         "order preserved)")
    ap.add_argument("--out", default="docs/img/bitvla_ov_quant_b70",
                    help="output path prefix; .png and .svg are written")
    ap.add_argument("--title",
                    default="BitVLA weight flavours on OpenVINO GPU, LIBERO object, Arc Pro B70")
    args = ap.parse_args()

    labels: list[str] = []
    arms: dict[str, dict] = {}
    for spec in args.arm:
        try:
            label, path = spec.split("=", 1)
        except ValueError:
            raise SystemExit(f"--arm wants LABEL=PATH, got {spec!r}")
        root = Path(path)
        labels.append(label)
        arms[label] = {
            "per_task": load_arm(path, label),
            "resident": resident_gib(root),
            "rss": peak_rss_gib(root),
        }

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(4.6 * 3, 4.6))

    # --- 1. task success ----------------------------------------------------
    # One series, so no legend: the title names it. Wilson rather than the normal
    # approximation because at 50 trials with p at 1.0 the latter puts the upper
    # bound above 1 and the lower bound on top of the point estimate.
    ax = axes[0]
    vals, errs, ticks = [], [], []
    for label in labels:
        succ, eps, _ = totals(arms[label]["per_task"])
        p = succ / eps if eps else 0.0
        lo, hi = wilson(succ, eps)
        vals.append(p)
        errs.append((p - lo, hi - p))
        # The episode count belongs on the tick, not in the caption: these arms
        # were not all swept at the same n, and the only other clue that they
        # differ is that one arm's interval is narrower than its neighbour's.
        ticks.append(f"{label}\nn={eps}")
    ax.bar(x, vals, 0.62, color=C_WEIGHTS,
           yerr=np.array(errs).T, capsize=3, ecolor=INK_MUTED, error_kw={"lw": 1.0})
    for i, (v, (lo_e, hi_e)) in enumerate(zip(vals, errs)):
        ax.text(i, v + hi_e + 0.015, f"{v * 100:.0f}%", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("task success rate")
    ax.set_title("task success (Wilson 95%)", color=INK)

    # --- 2. memory ----------------------------------------------------------
    # Stacked rather than grouped: peak RSS *contains* the weights, so two bars
    # side by side would invite reading them as independent measures and hide the
    # one fact the stack makes obvious -- everything above the weights is a
    # constant ~2 GiB that no weight format touches.
    ax = axes[1]
    resident = [arms[l]["resident"] for l in labels]
    rss = [arms[l]["rss"] for l in labels]
    have_mem = [r is not None for r in resident]
    base = np.array([r if r is not None else 0.0 for r in resident])
    rest = np.array([(s - r) if (s is not None and r is not None and s > r) else 0.0
                     for s, r in zip(rss, resident)])
    ax.bar(x, base, 0.62, color=C_WEIGHTS, label="weights resident")
    ax.bar(x, rest, 0.62, bottom=base, color=C_REST, label="rest of process (peak RSS)")
    for i, (b, r) in enumerate(zip(base, rest)):
        if not have_mem[i]:
            continue
        ax.text(i, b / 2, f"{b:.2f}", ha="center", va="center", fontsize=9, color="white")
        if r:
            ax.text(i, b + r + 0.12, f"{b + r:.1f}", ha="center", va="bottom",
                    fontsize=8, color=INK_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("GiB")
    ax.set_title("memory footprint", color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    # --- 3. latency ---------------------------------------------------------
    # Whiskers are the spread across the ten tasks, not a confidence interval.
    # Without them a flat set of bars looks like a result that was arranged; with
    # them it is visibly a measurement whose differences sit inside task-to-task
    # variation.
    ax = axes[2]
    means, lo_err, hi_err = [], [], []
    for label in labels:
        per_task = arms[label]["per_task"]
        _, _, inf = totals(per_task)
        spread = [t["inf_ms"] for t in per_task.values() if t["n_episodes"]]
        means.append(inf)
        lo_err.append(inf - min(spread) if spread else 0.0)
        hi_err.append(max(spread) - inf if spread else 0.0)
    ax.bar(x, means, 0.62, color=C_WEIGHTS,
           yerr=np.array([lo_err, hi_err]), capsize=3, ecolor=INK_MUTED,
           error_kw={"lw": 1.0})
    for i, (v, h) in enumerate(zip(means, hi_err)):
        ax.text(i, v + h, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("inference ms / step")
    ax.set_title("latency (bar = mean, whisker = per-task range)", color=INK)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(args.title, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
        print(f"wrote {out.with_suffix('.' + ext)}")

    # The table the plot is drawn from, so a reader can check the bars. It is
    # also the table that goes in the docs -- transcribing from a chart is how
    # published numbers stop matching the run that produced them.
    print()
    print(f"{'arm':<8} {'successes':>11} {'SR':>7} {'Wilson 95%':>18} "
          f"{'resident':>10} {'peak RSS':>10} {'ms/step':>9}")
    for label in labels:
        a = arms[label]
        succ, eps, inf = totals(a["per_task"])
        lo, hi = wilson(succ, eps)
        res = f"{a['resident']:.2f}" if a["resident"] is not None else "-"
        rs = f"{a['rss']:.2f}" if a["rss"] is not None else "-"
        print(f"{label:<8} {f'{succ}/{eps}':>11} {succ / eps if eps else 0:>7.3f} "
              f"{f'[{lo:.3f}, {hi:.3f}]':>18} {res:>10} {rs:>10} {inf:>9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
