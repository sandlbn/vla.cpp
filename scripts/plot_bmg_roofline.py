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
"""Plot BitVLA's ternary GEMM against the ceilings measured on Battlemage.

Consumes the CSV written by ``tests/bench_bitvla_bmg.cpp --csv`` and produces
two figures:

``bmg_roofline.png``
    The classic log-log roofline. Every dispatched shape is a point at its own
    arithmetic intensity. The second, dashed set of markers is where the same
    shape would sit once the weights stay 2-bit packed: same FLOPs, a quarter of
    the weight traffic, so the point slides right along a line of constant work
    and the memory roof stops binding it as early.

``bmg_ceiling.png``
    The number that actually answers "how close are we" - achieved throughput as
    a percentage of that shape's own roofline ceiling, which is not the same
    ceiling for every shape. Averaging a single "% of peak" across decode and
    prefill would describe neither.

Usage:
    scripts/plot_bmg_roofline.py ci/reference/bmg_roofline.csv -o docs/img
"""

import argparse
import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# One colour per pipeline stage; the stages sit in very different places on the
# roofline and that is the story the plot is told to carry. "decode"/"prefill"
# are the pre-VLA_BITVLA_LOG_SHAPES labels, kept so older CSVs still plot.
STAGE_COLOUR = {"lm": "#d1495b", "vit": "#edae49",
                "decode": "#d1495b", "prefill": "#00798c"}


def read_csv(path):
    """Return (meta, rows). Meta comes from the leading ``#`` comment lines."""
    meta, rows = {}, []
    with open(path, newline="") as fh:
        lines = fh.readlines()

    body = []
    for line in lines:
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            # A line carrying a single key takes the whole rest as its value, so
            # that values with spaces in them - the device name - survive.
            if text.count("=") == 1:
                k, v = text.split("=", 1)
                meta[k.strip()] = v.strip()
                continue
            for tok in text.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k] = v
        else:
            body.append(line)

    for row in csv.DictReader(body):
        for key in ("M", "N", "K"):
            row[key] = int(row[key])
        for key in ("ms", "ms_pipe", "tops", "tops_pipe", "ai_int8", "ai_packed"):
            if key in row:
                row[key] = float(row[key])
        # Plot the pipelined figure: the model issues ~120 of these back to back
        # into an in-order queue, so the per-call sync the latency column pays is
        # not a cost it incurs. Older CSVs only carry the synced number.
        row.setdefault("tops_pipe", row["tops"])
        row.setdefault("ms_pipe", row["ms"])
        # Which kernel the row actually ran on, and therefore which roof applies.
        # CSVs from before the packed GEMV landed are all-oneDNN.
        row.setdefault("path", "onednn")
        rows.append(row)
    return meta, rows


def row_roof(row, bw_unpacked, bw_packed, peak):
    """(ceiling of the path that ran, ceiling of the packed path)."""
    packed = row["path"] == "packed"
    now = roofline(row["ai_packed"] if packed else row["ai_int8"],
                   bw_packed if packed else bw_unpacked, peak)
    return now, roofline(row["ai_packed"], bw_packed, peak)


def roofline(ai, bw_GBs, peak_TOPS):
    """Ceiling in TOPS at arithmetic intensity ``ai`` (ops/byte)."""
    return min(peak_TOPS, ai * bw_GBs / 1e3)


def bandwidths(meta):
    """Return (unpacked, packed) GB/s - the two paths do not share a roof.

    Xe compresses memory surfaces losslessly and the unpacked ternary weights
    hold four distinct byte values, so the oneDNN path gets an effective read
    rate above the incompressible one. The 2-bit codes it would replace are
    dense, so the packed path is held to the plain figure. Pre-compression CSVs
    have no ``bw_ternary_GBs``; fall back rather than refuse to plot them.
    """
    plain = max(float(meta["bw_read_GBs"]), float(meta["bw_copy_GBs"]))
    return max(float(meta.get("bw_ternary_GBs", plain)), plain), plain


def plot_roofline(meta, rows, out_path):
    bw_unpacked, bw_packed = bandwidths(meta)
    peak = float(meta["xmx_int8_TOPS"])
    ridge = peak * 1e3 / bw_unpacked

    fig, ax = plt.subplots(figsize=(9, 6))

    # Two roofs, because the two paths stream different kinds of bytes: a
    # memory-bound ramp of slope BW up to the ridge, flat at the XMX peak after
    # it. The upper one is the compressed rate the unpacked weights actually
    # achieve; the lower is what incompressible 2-bit codes will get.
    xs = [0.05, 5000]
    for bw, style, name in ((bw_unpacked, "-", "unpacked int8, compressed"),
                            (bw_packed, "--", "incompressible")):
        pts = sorted(set(xs + [peak * 1e3 / bw]))
        ax.plot(pts, [roofline(x, bw, peak) for x in pts], color="black", lw=2,
                ls=style, zorder=3, alpha=1.0 if style == "-" else 0.45)
        ax.text(0.07, roofline(0.07, bw, peak) * 1.15, f"{bw:.0f} GB/s  ({name})",
                fontsize=8, rotation=34)
    ax.axvline(ridge, color="grey", ls=":", lw=1, zorder=1)
    ax.text(ridge * 1.06, peak * 0.045, f"ridge\n{ridge:.0f} ops/B", fontsize=8, color="grey")
    ax.text(1200, peak * 1.08, f"XMX int8 {peak:.0f} TOPS", fontsize=9, ha="right")

    seen = set()
    for row in rows:
        colour = STAGE_COLOUR[row["stage"]]
        label = row["stage"] if row["stage"] not in seen else None
        seen.add(row["stage"])

        # A row sits at the intensity of the kernel that ran. The packed rows
        # have already made the move to the right; the oneDNN rows still have it
        # in front of them, and the arrow is that remaining distance.
        packed = row["path"] == "packed"
        ax.scatter(row["ai_packed"] if packed else row["ai_int8"], row["tops_pipe"],
                   s=64, color=colour, zorder=5, edgecolor="white", linewidth=0.6,
                   label=label, marker="D" if packed else "o")
        if not packed:
            ax.scatter(row["ai_packed"], row["tops_pipe"], s=44, facecolor="none",
                       edgecolor=colour, linewidth=1.1, linestyle="--", zorder=4)
            ax.annotate("", xy=(row["ai_packed"], row["tops_pipe"]),
                        xytext=(row["ai_int8"], row["tops_pipe"]),
                        arrowprops=dict(arrowstyle="->", color=colour, lw=0.7, alpha=0.5))

    ax.scatter([], [], s=52, color="grey", marker="D", label="packed GEMV (M=1)")
    ax.scatter([], [], s=44, facecolor="none", edgecolor="grey", linestyle="--",
               label="oneDNN row, where packing would put it")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (ops / byte of DRAM traffic)")
    ax.set_ylabel("throughput (TOPS)")
    paths = {row["path"] for row in rows}
    subtitle = ("oneDNN over unpacked int8" if paths == {"onednn"} else
                "packed GEMV" if paths == {"packed"} else
                "M=1 on the packed GEMV, M>1 on oneDNN over unpacked int8")
    ax.set_title(f"BitVLA ternary GEMM roofline - {meta.get('device', 'B70')}\n{subtitle}",
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def plot_ceiling(meta, rows, out_path):
    bw_unpacked, bw_packed = bandwidths(meta)
    peak = float(meta["xmx_int8_TOPS"])

    # Two bars per shape against two different ceilings. The solid one is the
    # measured time against the roof of the kernel that ran, and is the progress
    # report; the hollow one is the same time against the roof the 2-bit path
    # has, and is the remaining opportunity. For the M=1 rows those are now the
    # same kernel and the bars coincide - that is what "done" looks like here.
    # Showing only one of the two would misdirect the next piece of work, since a
    # shape can be near-done on the first and still have 2x left on the second.
    labels, pct, pct_pk, colours = [], [], [], []
    for row in rows:
        mark = "*" if row["path"] == "packed" else ""
        labels.append(f"{row['stage']}\n{row['op']}{mark}\nM={row['M']}")
        now, pk = row_roof(row, bw_unpacked, bw_packed, peak)
        pct.append(100.0 * row["tops_pipe"] / now)
        pct_pk.append(100.0 * row["tops_pipe"] / pk)
        colours.append(STAGE_COLOUR[row["stage"]])

    fig, ax = plt.subplots(figsize=(11, 5.4))
    idx = range(len(pct))
    bars = ax.bar(idx, pct, color=colours, edgecolor="white", zorder=2,
                  label="of the ceiling of the kernel that ran")
    ax.bar(idx, pct_pk, color="none", edgecolor="grey", linewidth=0.9, linestyle="--",
           zorder=3, label="of the 2-bit packed ceiling (* = already there)")
    for bar, value in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}%",
                ha="center", fontsize=8)

    ax.axhline(100, color="black", lw=1.2)
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("% of ceiling")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.set_title("How much of the machine each dispatched shape reaches\n"
                 "(ceiling is per-shape: min(XMX peak, intensity x measured bandwidth))",
                 fontsize=11)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="output of bench_bitvla_bmg --csv")
    ap.add_argument("-o", "--outdir", default="docs/img", help="directory for the PNGs")
    args = ap.parse_args()

    meta, rows = read_csv(args.csv)
    if not rows:
        sys.exit(f"{args.csv}: no data rows")

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_roofline(meta, rows, outdir / "bmg_roofline.png")
    plot_ceiling(meta, rows, outdir / "bmg_ceiling.png")


if __name__ == "__main__":
    main()
