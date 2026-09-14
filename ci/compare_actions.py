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

"""Compare two vla_predict_check action chunks.

Used by the BF16 activation gates on both backends (ci/slurm/bmg_bf16_e2e.sbatch
and its h100 twin) so the two are scored by identical code - the port's precision
claim is "SYCL deviates from F32 no more than CUDA does", and that is only a
claim if both sides are measured the same way.

A chunk is (n_suffix, action_dim) flattened row-major, so the per-dimension
breakdown below is the useful view: a deviation concentrated in one action
dimension is a different thing from one spread across all of them. Gripper-like
dimensions saturate, and a saturating dimension amplifies a small upstream
difference into a large output one without anything being wrong.

  compare_actions.py <a.txt> <b.txt> --tol 0.02 [--dims N] [--label-a f32]
"""

import argparse
import sys


def load(path):
    """Actions from predict_check stdout: the `action_len=N` marker, then N floats.

    The model loader logs to stdout as well, so anchor on the marker and take
    exactly the declared count rather than every line that parses as a number.
    """
    lines = open(path).read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("action_len="):
            n = int(line.split("=", 1)[1])
            vals = lines[i + 1 : i + 1 + n]
            if len(vals) != n:
                sys.exit(f"{path}: action_len={n} but only {len(vals)} lines follow")
            return [float(x) for x in vals]
    sys.exit(f"{path}: no action_len= marker")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--dims", type=int, default=0, help="action_dim, for the per-dim table")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    if len(a) != len(b):
        sys.exit(f"MISMATCH len {len(a)} vs {len(b)}")

    diff = [abs(x - y) for x, y in zip(a, b)]
    # Normalised against the chunk's own peak, the same way the unit tests score
    # themselves: an action chunk spans a couple of decades, so a bare max-abs is
    # really a statement about its largest component alone.
    scale = max(max(abs(x) for x in a), 1e-6)
    mx = max(diff)
    rms = (sum(d * d for d in diff) / len(diff)) ** 0.5

    print(
        f"n={len(a)} max|{args.label_a}|={scale:.4f} max_abs={mx:.6f} rms={rms:.6f} "
        f"normalised={mx / scale:.4f} (tol {args.tol})"
    )

    if args.dims > 0 and len(a) % args.dims == 0:
        print(f"per action dim (chunk is {len(a) // args.dims} x {args.dims}):")
        for d in range(args.dims):
            col = diff[d :: args.dims]
            peak = max(abs(x) for x in a[d :: args.dims])
            if max(col) == 0.0 and peak == 0.0:
                continue  # padding dimension: never written
            print(
                f"  dim {d:2d}  max_abs={max(col):.6f}  "
                f"rms={(sum(c * c for c in col) / len(col)) ** 0.5:.6f}  "
                f"max|{args.label_a}|={peak:.4f}"
            )

    print("worst elements:")
    for i in sorted(range(len(a)), key=lambda i: -diff[i])[:5]:
        step, dim = (i // args.dims, i % args.dims) if args.dims > 0 else (-1, -1)
        where = f" step={step} dim={dim}" if args.dims > 0 else ""
        print(
            f"  [{i:5d}]{where}  {args.label_a}={a[i]:+.6f}  "
            f"{args.label_b}={b[i]:+.6f}  d={diff[i]:.6f}"
        )

    return 0 if mx / scale <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
