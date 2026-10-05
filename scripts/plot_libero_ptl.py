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
"""Per-task LIBERO results for the Panther Lake sweep (ci/local/libero_ptl.sh).

One row per model, tasks down the side, two panels sharing them:

  success rate per task, one dot per configuration with its Wilson 95% interval
  inference latency per step, per task and configuration

Each configuration's overall success rate and Wilson interval is in the legend.

    python scripts/plot_libero_ptl.py \\
        --arm "BitVLA:unfused (control)=outputs/local/libero_ptl/bit-ctl" \\
        --arm "BitVLA:fused (ptl-xe3)=outputs/local/libero_ptl/bit-opt" \\
        --out docs/img/libero_ptl

Arms are drawn in the order given within each model: the first in blue, the
second in orange. Paths are sweep roots (comma-separate chunk roots to sum
them), as plot_libero_bf16.py takes.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("plot_libero_bf16", REPO_ROOT / "scripts" / "plot_libero_bf16.py")
_lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lb)
wilson, load_arm, totals = _lb.wilson, _lb.load_arm, _lb.totals

# LIBERO-object, in task-id order.
TASKS = ["alphabet soup", "cream cheese", "salad dressing", "bbq sauce", "ketchup",
         "tomato sauce", "butter", "milk", "chocolate pudding", "orange juice"]

# What each model computes in, shown under its name. --precision MODEL=TEXT
# overrides or adds one.
PRECISION = {
    "BitVLA": "weights 1.58-bit ternary (int2-packed, unpacked to int8 for the GEMMs) · "
              "activations int8 into the GEMMs, bf16 elsewhere",
    "π0": "weights bf16 · activations bf16",
    "Evo-1": "weights bf16 · activations bf16",
}

# Reference categorical palette, slots 1-2 (validated pair: CVD dE 24.7).
COLOURS = ["#2a78d6", "#eb6834"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="MODEL:LABEL=PATH")
    ap.add_argument("--out", default="docs/img/libero_ptl")
    ap.add_argument("--title", default="LIBERO-object on Panther Lake (Core Ultra 7 356H), SYCL backend")
    ap.add_argument("--precision", action="append", default=[], metavar="MODEL=TEXT")
    args = ap.parse_args()
    precision = dict(PRECISION)
    for spec in args.precision:
        m, text = spec.split("=", 1)
        precision[m] = text

    models: dict[str, list[tuple[str, dict]]] = {}
    for spec in args.arm:
        key, path = spec.split("=", 1)
        model, label = key.split(":", 1)
        models.setdefault(model, []).append((label, load_arm(path, f"{model}/{label}")))

    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(11, 4.2 * n), squeeze=False,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(SURFACE)

    for r, (model, arms) in enumerate(models.items()):
        ax_sr, ax_ms = axes[r]
        style(ax_sr)
        style(ax_ms)
        k = len(arms)
        offs = [(j - (k - 1) / 2) * 0.32 for j in range(k)]
        handles, labels = [], []
        eps_total = set()
        for j, (label, per_task) in enumerate(arms):
            c = COLOURS[j % len(COLOURS)]
            ys, sr, lo, hi, ms = [], [], [], [], []
            for tid in sorted(per_task):
                t = per_task[tid]
                if not t["n_episodes"]:
                    continue
                p = t["successes"] / t["n_episodes"]
                l, h = wilson(t["successes"], t["n_episodes"])
                ys.append(len(TASKS) - 1 - tid + offs[j])
                sr.append(p)
                lo.append(max(0.0, p - l))
                hi.append(max(0.0, h - p))
                ms.append(t["inf_ms"])
            ax_sr.errorbar(sr, ys, xerr=[lo, hi], fmt="none", ecolor=c, elinewidth=2, capsize=0)
            ax_sr.scatter(sr, ys, s=56, color=c, edgecolors=SURFACE, linewidths=2, zorder=3)
            ax_ms.barh(ys, ms, height=0.3, color=c, linewidth=0)
            for y, v in zip(ys, ms):
                ax_ms.text(v, y, f" {v:.0f}", va="center", fontsize=7.5, color=INK_2)
            succ, eps, inf = totals(per_task)
            eps_total.add(eps)
            ol, oh = wilson(succ, eps)
            handles.append(plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                                      markerfacecolor=c, markeredgecolor=SURFACE))
            n_tasks = sum(1 for t in per_task.values() if t["n_episodes"])
            partial = "" if n_tasks == len(TASKS) else f", {n_tasks}/10 tasks so far"
            labels.append(f"{label}: {succ}/{eps} = {succ / eps:.0%} [{ol:.0%}-{oh:.0%}], "
                          f"{inf:.0f} ms/step{partial}")

        for ax in (ax_sr, ax_ms):
            ax.set_yticks(range(len(TASKS)))
            ax.set_yticklabels(list(reversed(TASKS)), color=INK_2)
            ax.set_ylim(-0.6, len(TASKS) - 0.4)
        ax_ms.set_yticklabels([])
        ax_sr.set_xlim(-0.02, 1.02)
        ax_sr.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax_sr.set_xlabel("task success (Wilson 95% interval)", color=INK_2, fontsize=9)
        ax_ms.set_xlim(0, max(t["inf_ms"] for _, pt in arms for t in pt.values()) * 1.18)
        ax_ms.set_xlabel("model inference per step, ms (server side)", color=INK_2, fontsize=9)
        ax_sr.set_title(model, loc="left", fontsize=12, color=INK, fontweight="bold", pad=50)
        if model in precision:
            # Out of the layout: tight_layout would otherwise widen the left panel
            # to fit the line and push the latency panel off to the right.
            note = ax_sr.annotate(precision[model], xy=(0, 1), xycoords="axes fraction",
                                  xytext=(0, 38), textcoords="offset points", fontsize=9,
                                  color=INK_2, va="bottom", annotation_clip=False)
            note.set_in_layout(False)
        ax_sr.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False,
                     fontsize=8.5, ncol=1, borderaxespad=0.2, handletextpad=0.3)

    fig.suptitle(args.title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=150, facecolor=SURFACE)
    print(f"wrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
