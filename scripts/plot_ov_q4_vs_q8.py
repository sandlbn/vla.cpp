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
"""Plot q4_0 against q8_0: the shipping flavour runs the same graph.

Every RMSNorm and int8 measurement on this backend was taken on q8_0, because
both gate scripts default to it, while Stage B's recommendation is q4_0 (2.24
GiB, 50/50 LIBERO tasks). This figure is the answer to "does any of it
transfer?", and the answer is that the two flavours are the same graph.

    python3 scripts/plot_ov_q4_vs_q8.py \
        --arm q8_0=outputs/ov_prof_ops_372916/q8_0.rms-off.fc-off.bucket.txt \
        --arm q4_0=outputs/ov_prof_ops_372919/q4_0.rms-off.fc-off.bucket.txt \
        --numerics outputs/../slurm_bmg_ov_rms_chk_372879.out \
        --numerics outputs/../slurm_bmg_ov_rms_chk_372921.out \
        --out docs/img/bitvla_ov_q4_vs_q8_b70

THE LEFT PANEL IS A PERCENTAGE AND THAT IS NOT A PRESENTATION CHOICE. Both
profiles were taken with enable_profiling on, which perturbs absolute device
time, and they come from two different jobs. Totals are comparable only WITHIN a
job, so the only honest cross-job encoding is each bucket's share of its own
run's total. Plotting the ms columns side by side would invite a reading of the
114.78-vs-109.09 ms difference that the measurement cannot support. The ms
values are printed in the caption strip for provenance, not drawn as bars.

The right panels come from ci/slurm/bmg_ov_rms_fusion_check.sbatch, which
reports two different things from the same runs: the chunk spread (a screen for
a collapsed norm -- see memory/bitvla-action-bar-cannot-bind.md for why action
equality cannot be a gate on this architecture) and the f16-vs-f32 action
deviation. Both are per-flavour and neither had been measured on q4_0 before
job 372921.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a compute node
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# --- the validated palette ---------------------------------------------------
# Categorical slots 1 and 2 of the reference theme, checked with the dataviz
# validator against this surface before use: lightness band, chroma floor,
# adjacent-pair CVD separation (worst dE 24.7 protan / 32.7 tritan), the
# normal-vision floor (33.6) and 3:1 contrast all pass. Colour follows the
# FLAVOUR and nothing else, so adding a precision axis on the right-hand panels
# reuses these two hues rather than introducing a third.
SURFACE = "#fcfcfb"
SERIES = {"q8_0": "#2a78d6", "q4_0": "#eb6834"}
INK = "#1a1a19"
INK_SOFT = "#5c5c58"
INK_MUTED = "#8a8a84"
GRID = "#e4e4e0"
BAND = "#d8ecdf"  # the healthy chunk-spread band, a recessive fill

# Buckets below this share fold into "other". 1.5% keeps the four that carry the
# argument (FC, ReduceMean, Multiply, Convert = 84%) plus the near-misses, and
# folds the long tail of sub-1% eltwise nodes that no optimisation targets.
FOLD_BELOW_PCT = 1.5


def parse_bucket(path: Path) -> tuple[dict[str, tuple[float, float, int]], float]:
    """Read the '=== by node type ===' table out of a .bucket.txt.

    Returns {bucket: (ms, pct, nodes)} and the attributed device-time total.
    The node-type table is taken deliberately rather than the kernel table: a
    kernel name identifies an implementation and the mapping to primitives is
    many-to-one (FullyConnectedCompressed, Gemm and MatMul all execute
    jit:gemm:any__f16), so a kernel-keyed plot cannot show an Amdahl split.
    """
    text = path.read_text()

    m = re.search(r"([\d.]+)\s*ms of device time attributed", text)
    if not m:
        sys.exit(f"{path}: no 'ms of device time attributed' line")
    total = float(m.group(1))

    block = re.search(
        r"=== by node type.*?===\n.*?\n(.*?)(?:\n\s*\n|\n===)", text, re.S
    )
    if not block:
        sys.exit(f"{path}: no '=== by node type ===' table")

    rows: dict[str, tuple[float, float, int]] = {}
    for line in block.group(1).splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            ms, pct, nodes = float(parts[-3]), float(parts[-2]), int(parts[-1])
        except ValueError:
            continue
        rows[" ".join(parts[:-3])] = (ms, pct, nodes)

    if not rows:
        sys.exit(f"{path}: node-type table parsed empty")
    return rows, total


def parse_numerics(paths: list[Path]) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Pull chunk spreads and f16-vs-f32 deviations out of rms_chk job logs."""
    spread: dict[str, float] = {}
    dev: dict[str, tuple[float, float]] = {}
    for p in paths:
        text = p.read_text()
        for fl, mode, fc, prec, val in re.findall(
            r"^\s*(\w+)/rms-(\w+)/fc-(\w+)/(f\d+)\s+([\d.]+)\s*$", text, re.M
        ):
            if mode == "off":
                spread[f"{fl}/{prec}"] = float(val)
        for fl, mx, mn in re.findall(
            r"^\s*(\w+)/fc-\w+ off:\s+f16 vs f32\s+max\s+([\d.]+)\s+mean\s+([\d.]+)",
            text,
            re.M,
        ):
            dev[fl] = (float(mx), float(mn))
    return spread, dev


def style(ax):
    """Recessive grid and axes; the marks carry the chart, not the furniture."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=8.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="FLAVOUR=BUCKET.TXT",
                    help="pass q8_0 first, then q4_0; panel order follows")
    ap.add_argument("--numerics", action="append", default=[], metavar="JOBLOG.out")
    ap.add_argument("--out", required=True, help="output stem; .png and .svg are written")
    args = ap.parse_args()

    arms: dict[str, tuple[dict, float]] = {}
    for spec in args.arm:
        if "=" not in spec:
            sys.exit(f"--arm wants FLAVOUR=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        arms[name] = parse_bucket(Path(path))
    names = list(arms)
    if len(names) != 2:
        sys.exit("this figure compares exactly two flavours")
    a, b = names
    unknown = [n for n in names if n not in SERIES]
    if unknown:
        sys.exit(f"no validated colour slot for {unknown}; extend SERIES and re-run the validator")

    spread, dev = parse_numerics([Path(p) for p in args.numerics])

    # --- fold the tail, keeping both arms' bucket sets aligned ---------------
    keys = [k for k in arms[a][0] if max(arms[x][0].get(k, (0, 0, 0))[1] for x in names) >= FOLD_BELOW_PCT]
    keys.sort(key=lambda k: -arms[a][0][k][1])
    folded = {x: 100.0 - sum(arms[x][0].get(k, (0, 0, 0))[1] for k in keys) for x in names}
    n_folded = len(set(arms[a][0]) | set(arms[b][0])) - len(keys)

    labels = keys + [f"other ({n_folded} types)"]
    vals = {x: [arms[x][0].get(k, (0, 0, 0))[1] for k in keys] + [folded[x]] for x in names}

    fig = plt.figure(figsize=(13.2, 6.6), facecolor=SURFACE)
    gs = GridSpec(2, 2, width_ratios=[1.85, 1.0], height_ratios=[1, 1],
                  hspace=0.52, wspace=0.26, left=0.150, right=0.975, top=0.775, bottom=0.115)

    # =========================== PANEL A ====================================
    axA = fig.add_subplot(gs[:, 0])
    style(axA)
    y = np.arange(len(labels))
    h = 0.37  # two bars + a surface gap inside each slot
    for i, x in enumerate(names):
        off = (i - 0.5) * (h + 0.035)
        axA.barh(y + off, vals[x], height=h, color=SERIES[x], label=x, zorder=3)
    axA.set_yticks(y)
    axA.set_yticklabels(labels, fontsize=9)
    axA.invert_yaxis()
    axA.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    axA.set_axisbelow(True)
    axA.set_xlabel("share of attributed device time  (%)", fontsize=9, color=INK_SOFT)
    axA.set_xlim(0, max(max(vals[a]), max(vals[b])) * 1.16)

    # Selective direct labels: the three buckets that carry 80% of the time and
    # are the only ones any optimisation has targeted. A number on all 18 marks
    # would be noise, and the axis already carries the rest.
    for j in range(min(3, len(labels))):
        for i, x in enumerate(names):
            off = (i - 0.5) * (h + 0.035)
            axA.text(vals[x][j] + 0.5, y[j] + off, f"{vals[x][j]:.2f}%",
                     va="center", ha="left", fontsize=8, color=INK_SOFT)

    axA.set_title("The Amdahl split is flavour-invariant",
                  fontsize=11.5, color=INK, loc="left", pad=26, weight="bold")
    axA.text(0, 1.012,
             f"same kernel (jit:gemm:any__f16), same node counts; "
             f"{a} {arms[a][1]:.1f} ms · {b} {arms[b][1]:.1f} ms attributed",
             transform=axA.transAxes, fontsize=8.4, color=INK_MUTED, va="bottom")

    # =========================== PANEL B ====================================
    axB = fig.add_subplot(gs[0, 1])
    style(axB)
    precs = ["f16", "f32"]
    xb = np.arange(len(precs))
    w = 0.33
    axB.axhspan(0.022, 0.024, color=BAND, zorder=0)
    axB.text(1.42, 0.0230, "healthy\nband", fontsize=7.6, color="#3f7a57",
             va="center", ha="left", linespacing=1.25)
    for i, x in enumerate(names):
        hv = [spread.get(f"{x}/{p}", np.nan) for p in precs]
        axB.bar(xb + (i - 0.5) * (w + 0.03), hv, width=w, color=SERIES[x], zorder=3)
        for k, v in enumerate(hv):
            if not np.isnan(v):
                axB.text(xb[k] + (i - 0.5) * (w + 0.03), v + 0.0006, f"{v:.5f}",
                         ha="center", fontsize=7.4, color=INK_SOFT)
    axB.set_xticks(xb)
    axB.set_xticklabels(precs, fontsize=9)
    axB.set_xlim(-0.6, 2.25)
    axB.set_ylim(0, 0.031)
    axB.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    axB.set_axisbelow(True)
    axB.set_ylabel("chunk spread", fontsize=8.5, color=INK_SOFT)
    axB.set_title("The screen transfers to q4_0",
                  fontsize=10.5, color=INK, loc="left", pad=7, weight="bold")

    # =========================== PANEL C ====================================
    axC = fig.add_subplot(gs[1, 1])
    style(axC)
    stats = ["max", "mean"]
    xc = np.arange(len(stats))
    for i, x in enumerate(names):
        mx, mn = dev.get(x, (np.nan, np.nan))
        axC.bar(xc + (i - 0.5) * (w + 0.03), [mx, mn], width=w, color=SERIES[x], zorder=3)
        for k, v in enumerate((mx, mn)):
            if not np.isnan(v):
                axC.text(xc[k] + (i - 0.5) * (w + 0.03), v + 0.002, f"{v:.4f}",
                         ha="center", fontsize=7.4, color=INK_SOFT)
    axC.set_xticks(xc)
    axC.set_xticklabels(stats, fontsize=9)
    axC.set_xlim(-0.6, 2.25)
    axC.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    axC.set_axisbelow(True)
    axC.set_ylabel("f16 vs f32 action dev.", fontsize=8.5, color=INK_SOFT)
    axC.set_title("q4_0 is half as precision-sensitive",
                  fontsize=10.5, color=INK, loc="left", pad=7, weight="bold")

    # --- figure furniture ---------------------------------------------------
    fig.text(0.150, 0.945, "BitVLA on OpenVINO / Arc Pro B70 — q4_0 runs the same graph as q8_0",
             fontsize=13.5, color=INK, weight="bold", ha="left")
    fig.text(0.150, 0.905,
             "Shares of device time, not milliseconds: enable_profiling perturbs absolute time and these are two separate jobs,\n"
             "so only within-run shares compare. Right-hand panels are the numerics screen, run on q4_0 for the first time (372921).",
             fontsize=8.6, color=INK_MUTED, ha="left", va="top", linespacing=1.6)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[x]) for x in names]
    axA.legend(handles, names, loc="lower right", bbox_to_anchor=(0.995, 0.055),
               frameon=False, fontsize=10, handlelength=1.1, handleheight=1.1,
               labelcolor=INK_SOFT)

    fig.text(0.150, 0.026,
             "q8_0 job 372916 · q4_0 job 372919 · numerics jobs 372879 / 372921 · LIBERO-object, GPU plugin f16",
             fontsize=7.6, color=INK_MUTED, ha="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURFACE)
        print(f"wrote {out}.{ext}")


if __name__ == "__main__":
    main()
