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
"""Plot the Panther Lake (Xe3 iGPU) SYCL results of docs/backend/ptl.md.

    python scripts/plot_ptl_sycl.py [--out docs/img]

Writes ptl_sycl_models.png, ptl_bitvla_steps.png and ptl_bandwidth.png. The
numbers are inline, each with where it came from: vla-bench p50 on a Core Ultra
7 356H (20 reps, 3 warmup; 10/5 reps for pi0/Evo-1), unless noted.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette, slots 1-2 in fixed order (validated: CVD dE 24.7,
# normal-vision dE 33.6, both >= 3:1 on the surface).
BEFORE = "#2a78d6"
AFTER = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# (model, [(config, before ms, after ms)]). "before" is the bmg-sycl-kernels
# branch as it arrived, on the same machine; "after" is ptl-xe3. pi0/Evo-1
# "after" is with --flash-attn 1.
MODELS = [
    ("BitVLA", [("SYCL ternary kernels", 128.5, 83.8)]),
    ("π0", [("f32 activations", 481.5, 379.4), ("bf16 activations", 453.3, 346.0)]),
    ("Evo-1 (448 px)", [("f32 activations", 1443.5, 582.6), ("bf16 activations", 1314.0, 512.2)]),
]

# BitVLA p50 after each change, in commit order (git log bmg-sycl-kernels..ptl-xe3).
BITVLA_STEPS = [
    ("Baseline (B70-tuned port)", 128.5),
    ("Fused norm / quant row kernels", 115.7),
    ("Fused attention (oneDNN SDPA)", 95.7),
    ("One QKV GEMM", 94.3),
    ("Vectorised row kernels", 86.9),
    ("Weight-layout autotune", 85.7),
    ("Vectorised RoPE", 83.8),
]

# Achieved read bandwidth on the iGPU, random data (compressible data reads
# faster and lies). Microbenchmarks from a scratch SYCL kernel over 1 GiB;
# kernel rows from unitrace (bytes moved / kernel time).
BANDWIDTH = [
    ("Microbenchmark: 4 B load per work-item", 28.9),
    ("Microbenchmark: 16 B vector loads", 77.3),
    ("sqReLU+norm+quant, strided (B70 layout)", 38.0),
    ("sqReLU+norm+quant, 8-wide contiguous", 61.0),
]
CEILING_GBS = 77.3


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def hbar(ax, y, w, color, h):
    ax.barh(y, w, height=h, color=color, linewidth=0)


def plot_models(out):
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(8, 5.6),
                             gridspec_kw={"height_ratios": [len(c) for _, c in MODELS]})
    fig.patch.set_facecolor(SURFACE)
    h = 0.32
    for ax, (model, cfgs) in zip(axes, MODELS):
        style(ax)
        top = max(b for _, b, _ in cfgs)
        for i, (cfg, b, a) in enumerate(cfgs):
            y = len(cfgs) - 1 - i
            hbar(ax, y + h / 2 + 0.02, b, BEFORE, h)
            hbar(ax, y - h / 2 - 0.02, a, AFTER, h)
            ax.text(b + top * 0.01, y + h / 2 + 0.02, f"{b:,.0f} ms", va="center", fontsize=9, color=INK_2)
            ax.text(a + top * 0.01, y - h / 2 - 0.02, f"{a:,.0f} ms  ({(a / b - 1) * 100:+.0f}%)",
                    va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(cfgs)))
        ax.set_yticklabels([c for c, _, _ in reversed(cfgs)], color=INK_2)
        ax.set_xlim(0, top * 1.28)
        ax.set_ylim(-0.6, len(cfgs) - 0.4)
        ax.set_title(model, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[-1].set_xlabel("latency per step, ms (p50, lower is better)", color=INK_2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BEFORE), plt.Rectangle((0, 0), 1, 1, color=AFTER)]
    fig.legend(handles, ["before (bmg-sycl-kernels)", "after (ptl-xe3)"], loc="upper left",
               frameon=False, fontsize=9, ncol=2, bbox_to_anchor=(0.012, 0.955))
    fig.suptitle("vla.cpp on Panther Lake (Core Ultra 7 356H, Xe3 iGPU), SYCL backend",
                 x=0.02, y=0.99, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(os.path.join(out, "ptl_sycl_models.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_steps(out):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    h = 0.55
    n = len(BITVLA_STEPS)
    for i, (label, ms) in enumerate(BITVLA_STEPS):
        y = n - 1 - i
        hbar(ax, y, ms, BEFORE if i == 0 else AFTER, h)
        ax.text(ms + 1.5, y, f"{ms:.1f} ms", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels([l for l, _ in reversed(BITVLA_STEPS)], color=INK_2)
    ax.set_xlim(0, 150)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("BitVLA latency per step, ms (p50)", color=INK_2, fontsize=9)
    ax.set_title("BitVLA on the Xe3 iGPU, change by change: 128.5 → 83.8 ms (−35%)",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ptl_bitvla_steps.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_bandwidth(out):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    h = 0.55
    n = len(BANDWIDTH)
    for i, (label, gbs) in enumerate(BANDWIDTH):
        y = n - 1 - i
        hbar(ax, y, gbs, BEFORE if i % 2 == 0 else AFTER, h)
        ax.text(gbs + 1, y, f"{gbs:.0f} GB/s", va="center", fontsize=9, color=INK)
    ax.axvline(CEILING_GBS, color=INK_2, linewidth=1)
    ax.text(CEILING_GBS + 1, n - 0.45, "measured ceiling", fontsize=8, color=INK_2, va="bottom")
    ax.set_yticks(range(n))
    ax.set_yticklabels([l for l, _ in reversed(BANDWIDTH)], color=INK_2)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_xlabel("achieved DRAM read bandwidth, GB/s (random data)", color=INK_2, fontsize=9)
    ax.set_title("Small loads reach ~⅓ of the iGPU's DRAM bandwidth", loc="left", fontsize=11,
                 color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BEFORE), plt.Rectangle((0, 0), 1, 1, color=AFTER)]
    ax.legend(handles, ["B70-style access", "Xe3-tuned access"], loc="upper center",
              bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ptl_bandwidth.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/img")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plot_models(args.out)
    plot_steps(args.out)
    plot_bandwidth(args.out)
    print(f"wrote ptl_sycl_models.png, ptl_bitvla_steps.png, ptl_bandwidth.png to {args.out}")


if __name__ == "__main__":
    main()
