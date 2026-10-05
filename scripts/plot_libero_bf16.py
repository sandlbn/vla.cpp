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
"""Plot the LIBERO f32-vs-bf16 sweep.

Everything measured for the BF16 activation path so far is engine-side: action
chunks compared against a CPU fp32 reference, and vla-bench latency. Neither says
whether the robot still completes the task. This turns a libero_object sweep into
the three plots that do.

The success-rate panels carry Wilson intervals, and they are not decoration. Ten
episodes per task is 10 Bernoulli trials: 7/10 against 8/10 is not a difference,
and a bare bar chart invites reading it as one. If the intervals overlap, the
honest statement is that the sweep did not resolve a difference at this episode
count -- not that the two are equal.

    python3 scripts/plot_libero_bf16.py \
        --arm evo1:f32=outputs/libero_b70/evo1-f32 \
        --arm evo1:bf16=outputs/libero_b70/evo1-bf16 \
        --arm pi0:f32=outputs/libero_b70/pi0-f32 \
        --arm pi0:bf16=outputs/libero_b70/pi0-bf16 \
        --out docs/img/libero_bf16_b70

Each arm path is a sweep root holding **/libero_object/task_<id>/summary.txt.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a compute node
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
N_TASKS = 10

# Import the collector's parsers rather than restating its regexes here: the
# summary.txt format is its business, and two copies would drift the first time
# the client's output changes.
_spec = importlib.util.spec_from_file_location(
    "collect_libero_results", REPO_ROOT / "eval" / "collect_libero_results.py"
)
_collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_collect)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Normal approximation is useless at n=10 and p near
    0 or 1 -- it produces bounds outside [0, 1)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def merge(parts: list[dict[int, dict]]) -> dict[int, dict]:
    """Sum several sweeps of the same arm task-by-task.

    Used for chunked runs: robosuite leaks a GL context per reset, so a client
    process that does fifty episodes can abort partway, while ten is reliable.
    Five ten-episode processes are the same five hundred Bernoulli trials as one
    fifty-episode process - episodes are independent and the seeds are drawn per
    episode either way - so summing them loses nothing.
    """
    out: dict[int, dict] = {}
    for part in parts:
        for tid, t in part.items():
            acc = out.setdefault(tid, {"successes": 0, "n_episodes": 0, "inf_ms": 0.0})
            acc["successes"] += t["successes"]
            acc["n_episodes"] += t["n_episodes"]
            # Episode-weighted, so chunks of unequal size do not skew it.
            acc["inf_ms"] += t["inf_ms"] * t["n_episodes"]
    for t in out.values():
        t["inf_ms"] = t["inf_ms"] / t["n_episodes"] if t["n_episodes"] else 0.0
    return out


def load_arm(spec: str, label: str) -> dict[int, dict]:
    # Comma-separated roots are chunks of one arm, to be summed.
    roots = [Path(p) for p in str(spec).split(",") if p]
    parts = []
    for r in roots:
        p = _collect.collect_model(r)
        if not p:
            raise SystemExit(f"no {_collect.TASK_SUITE} summaries under {r}")
        parts.append(p)
    per_task = merge(parts)
    root = roots[0] if len(roots) == 1 else f"{len(roots)} chunks under {roots[0].parent}"
    if not per_task:
        raise SystemExit(f"no {_collect.TASK_SUITE} summaries under {root}")

    # A sweep writes summary.txt per task into a directory it does not clear, so
    # a re-run at a different episode count leaves the old files in place for
    # every task it has not reached yet. The plot then mixes them without
    # complaint - and it is invisible in the bars, showing up only as error bars
    # of two different widths.
    counts = {t["n_episodes"] for t in per_task.values()}
    if len(counts) > 1:
        by_count: dict[int, list[int]] = {}
        for tid, t in sorted(per_task.items()):
            by_count.setdefault(t["n_episodes"], []).append(tid)
        detail = "; ".join(f"n={n}: tasks {ids}" for n, ids in sorted(by_count.items()))
        print(f"WARNING: {label} mixes episode counts under {root} -- {detail}. "
              f"Almost certainly stale summaries from an earlier run at a "
              f"different -n; delete the arm directory and re-run.", file=sys.stderr)

    missing = [t for t in range(N_TASKS) if t not in per_task]
    if missing:
        print(f"WARNING: {label} has no summary for task(s) {missing} under {root}",
              file=sys.stderr)
    return per_task


def tick_label(model: str, dtype: str) -> str:
    """Arm label under a bar. An arm named "weights / compute" (the OpenVINO
    figure's arms) is too wide for one line once there are five bars in a 1.3-wide
    panel, and the neighbours overprint it -- so it breaks at the slash."""
    return f"{model}\n" + dtype.replace(" / ", "\n")


def totals(per_task: dict[int, dict]) -> tuple[int, int, float]:
    succ = sum(t["successes"] for t in per_task.values())
    eps = sum(t["n_episodes"] for t in per_task.values())
    inf = (
        sum(t["inf_ms"] * t["n_episodes"] for t in per_task.values()) / eps
        if eps
        else 0.0
    )
    return succ, eps, inf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="MODEL:DTYPE=PATH",
                    help="e.g. evo1:bf16=outputs/libero_b70/evo1-bf16 (repeatable)")
    ap.add_argument("--out", default="docs/img/libero_bf16",
                    help="output path prefix; .png and .svg are written")
    ap.add_argument("--title", default="LIBERO object, Arc Pro B70 (SYCL)")
    args = ap.parse_args()

    arms: dict[tuple[str, str], dict[int, dict]] = {}
    for spec in args.arm:
        try:
            key, path = spec.split("=", 1)
            model, dtype = key.split(":", 1)
        except ValueError:
            raise SystemExit(f"--arm wants MODEL:DTYPE=PATH, got {spec!r}")
        arms[(model, dtype)] = load_arm(path, f"{model}/{dtype}")

    models = sorted({m for m, _ in arms})
    # f32 and bf16 first in that order, then anything else (a repeat arm such as
    # f32r, used to measure the harness's own spread) in the order given. Not a
    # fixed two-element list: silently dropping an arm the caller passed would
    # hide exactly the control the caller added it to see.
    seen = list(dict.fromkeys(d for _, d in arms))
    dtypes = [d for d in ("f32", "bf16") if d in seen] + \
             [d for d in seen if d not in ("f32", "bf16")]
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    colours = {"f32": palette[0], "bf16": palette[1]}
    for d in dtypes:
        colours.setdefault(d, palette[len(colours) % len(palette)])

    fig = plt.figure(figsize=(6.2 * max(len(models), 1) + 5.0, 4.6))
    gs = fig.add_gridspec(1, len(models) + 2, width_ratios=[2.2] * len(models) + [1.3, 1.3])

    # --- per-task success, one panel per model ------------------------------
    for i, model in enumerate(models):
        ax = fig.add_subplot(gs[0, i])
        width = 0.8 / max(len(dtypes), 1)
        for j, dtype in enumerate(dtypes):
            per_task = arms.get((model, dtype))
            if per_task is None:
                continue
            ids = sorted(per_task)
            sr = [per_task[t]["successes"] / per_task[t]["n_episodes"]
                  if per_task[t]["n_episodes"] else 0.0 for t in ids]
            lo_hi = [wilson(per_task[t]["successes"], per_task[t]["n_episodes"]) for t in ids]
            # Clamped: at p = 1 the Wilson upper bound is 1 only up to rounding,
            # and matplotlib rejects the resulting -1e-17 bar length outright.
            err = np.array([[max(0.0, s - lo) for s, (lo, _) in zip(sr, lo_hi)],
                            [max(0.0, hi - s) for s, (_, hi) in zip(sr, lo_hi)]])
            x = np.arange(len(ids)) + (j - (len(dtypes) - 1) / 2) * width
            ax.bar(x, sr, width, label=dtype, color=colours.get(dtype),
                   yerr=err, capsize=2, ecolor="#444444", error_kw={"lw": 1.0})
            ax.set_xticks(np.arange(len(ids)))
            ax.set_xticklabels([str(t) for t in ids])
        # Episode count in the panel title, not just the figure title: a figure
        # may hold one model swept at n=10 beside another re-swept at n=50, and
        # the only other clue is that one panel's error bars are narrower.
        eps = {sum(t["n_episodes"] for t in arms[(model, d)].values())
               for d in dtypes if (model, d) in arms}
        ax.set_title(f"{model} (n={eps.pop()})" if len(eps) == 1 else model)
        ax.set_xlabel("task id")
        ax.set_ylim(0, 1.05)
        if i == 0:
            ax.set_ylabel("success rate")
        ax.grid(axis="y", alpha=0.3)

    # --- overall success ----------------------------------------------------
    ax = fig.add_subplot(gs[0, len(models)])
    labels, vals, errs, cols = [], [], [], []
    for model in models:
        for dtype in dtypes:
            per_task = arms.get((model, dtype))
            if per_task is None:
                continue
            succ, eps, _ = totals(per_task)
            lo, hi = wilson(succ, eps)
            labels.append(tick_label(model, dtype))
            vals.append(succ / eps if eps else 0.0)
            errs.append((max(0.0, vals[-1] - lo), max(0.0, hi - vals[-1])))
            cols.append(colours.get(dtype))
    err = np.array(errs).T if errs else None
    ax.bar(range(len(vals)), vals, 0.6, color=cols, yerr=err, capsize=3,
           ecolor="#444444", error_kw={"lw": 1.0})
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("overall (Wilson 95%)")
    ax.grid(axis="y", alpha=0.3)

    # --- inference latency --------------------------------------------------
    ax = fig.add_subplot(gs[0, len(models) + 1])
    labels, vals, cols = [], [], []
    for model in models:
        for dtype in dtypes:
            per_task = arms.get((model, dtype))
            if per_task is None:
                continue
            _, _, inf = totals(per_task)
            labels.append(tick_label(model, dtype))
            vals.append(inf)
            cols.append(colours.get(dtype))
    ax.bar(range(len(vals)), vals, 0.6, color=cols)
    # One decimal, not zero: evo1 and pi0 differ by an order of magnitude, so a
    # shared linear axis flattens pi0's two bars into the same 4 mm of ink and
    # "4 vs 4" would hide the very difference the panel exists to show.
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("inference ms / step")
    ax.set_title("client-side latency")
    ax.grid(axis="y", alpha=0.3)

    # One figure-level legend rather than one per panel: an in-axes legend has
    # nowhere to sit here. Success rates cluster at the top of the panel and the
    # error bars reach the bottom, so every corner is occupied by data.
    #
    # Title and legend get a row each, in a band the panels are kept out of.
    # Both used to be pinned to the top edge -- the legend left, the title
    # centred -- and once the legend grew to four arms (b70/h100 x f32/bf16) it
    # ran underneath the title. Sizes are in inches so the band does not scale
    # with the figure height.
    fig_h = fig.get_figheight()
    title_h, legend_h, pad = 0.34, 0.34, 0.10
    top = 1.0 - (title_h + legend_h + pad) / fig_h
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.suptitle(args.title, y=1.0 - 0.04 / fig_h, va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours.get(d)) for d in dtypes]
    fig.legend(handles, dtypes, loc="upper center",
               bbox_to_anchor=(0.5, 1.0 - title_h / fig_h),
               ncol=len(dtypes), frameon=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
        print(f"wrote {out.with_suffix('.' + ext)}")

    # The table the plot is drawn from, so a reader can check the bars.
    print()
    print(f"{'arm':<14} {'successes':>11} {'SR':>8} {'Wilson 95%':>18} {'inf ms':>8}")
    for model in models:
        for dtype in dtypes:
            per_task = arms.get((model, dtype))
            if per_task is None:
                continue
            succ, eps, inf = totals(per_task)
            lo, hi = wilson(succ, eps)
            print(f"{model + '/' + dtype:<14} {f'{succ}/{eps}':>11} "
                  f"{succ / eps if eps else 0:>8.3f} {f'[{lo:.3f}, {hi:.3f}]':>18} {inf:>8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
