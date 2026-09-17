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
"""Compare two VLA_BITVLA_DUMP_DIR trees stage by stage.

`ci/compare_actions.py` scores the final action vector, which tells you *that*
a backend disagrees but not *where*. BitVLA's predict() already writes five
intermediate tensors when VLA_BITVLA_DUMP_DIR is set, and they happen to sit at
exactly the seams between its four graph computes:

    mm_proj_out       after the ViT + multimodal projector   (graph 1)
    proprio_features  after the proprio MLP                  (graph 2)
    inputs_embeds     the assembled LM input                 (host-side concat)
    ah_input          LM prefill output at the action slots  (graph 3)
    ah_norm_actions   the action head                        (graph 4)

So the first stage that diverges names the graph at fault, and `inputs_embeds`
is a useful control: it is assembled on the host from mm_proj_out,
proprio_features and token embeddings, so if the two before it agree and it does
not, the problem is in the embedding fetch rather than in any backend.

The metric matches compare_actions.py -- max absolute difference over
max(max|a|, 1e-6) -- so a number here is directly comparable to the 2.9e-3 bar
the final actions are held to.

    compare_dumps.py <dir_a> <dir_b> [--tol 0.0029] [--label-a cpu] [--label-b ov]

Exit status is 0 when every stage present in both trees is within tol.
"""

import argparse
import os
import sys

import numpy as np

# Evaluation order, which is also the order they are written. Anything else
# found in the manifest is compared afterwards, so adding a _dump_bin call
# upstream does not silently drop out of this report.
STAGES = [
    "mm_proj_out",
    "proprio_features",
    "inputs_embeds",
    "ah_input",
    "ah_norm_actions",
]


def load(d, name):
    p = os.path.join(d, name + ".bin")
    if not os.path.exists(p):
        return None
    return np.fromfile(p, dtype=np.float32)


def manifest_names(d):
    p = os.path.join(d, "manifest.txt")
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            parts = line.split()
            if parts:
                out.append(parts[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--tol", type=float, default=0.0029)
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    args = ap.parse_args()

    names = list(STAGES)
    for n in manifest_names(args.dir_a) + manifest_names(args.dir_b):
        if n not in names:
            names.append(n)

    print(f"{'stage':<20} {'n':>10} {'max|d|':>12} {'normalised':>12}  verdict")
    print("-" * 70)

    rc = 0
    first_bad = None
    compared = 0
    for name in names:
        a = load(args.dir_a, name)
        b = load(args.dir_b, name)
        if a is None and b is None:
            continue
        if a is None or b is None:
            where = args.label_a if a is None else args.label_b
            print(f"{name:<20} {'-':>10} {'-':>12} {'-':>12}  MISSING in {where}")
            rc = 1
            continue
        if a.size != b.size:
            print(f"{name:<20} {a.size:>10} {'-':>12} {'-':>12}  SIZE MISMATCH ({b.size})")
            rc = 1
            continue

        # NaN/Inf would make the max meaningless, and a backend that produces
        # them is broken in a way worth naming rather than scoring.
        bad_a = int(np.count_nonzero(~np.isfinite(a)))
        bad_b = int(np.count_nonzero(~np.isfinite(b)))
        if bad_a or bad_b:
            print(f"{name:<20} {a.size:>10} {'-':>12} {'-':>12}  "
                  f"NON-FINITE ({args.label_a}={bad_a} {args.label_b}={bad_b})")
            rc = 1
            continue

        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        mx = float(d.max()) if d.size else 0.0
        scale = max(float(np.abs(a).max()) if a.size else 0.0, 1e-6)
        norm = mx / scale
        compared += 1
        ok = norm <= args.tol
        if not ok:
            rc = 1
            if first_bad is None:
                first_bad = name
        print(f"{name:<20} {a.size:>10} {mx:>12.6g} {norm:>12.6g}  {'ok' if ok else 'FAIL'}")

        if not ok:
            i = int(d.argmax())
            print(f"{'':<20} worst element [{i}]: "
                  f"{args.label_a}={a[i]:.6g} {args.label_b}={b[i]:.6g}")
            # How widespread: a handful of outliers is a different bug from a
            # uniformly shifted tensor.
            over = int(np.count_nonzero(d > args.tol * scale))
            print(f"{'':<20} {over} of {a.size} elements over tol "
                  f"({100.0 * over / a.size:.2f}%)")

    print()
    # A comparison that compared nothing must not report success. Two empty
    # directories agree on every stage they both have, which is none of them.
    if compared == 0:
        print(f"NO STAGES COMPARED -- {args.dir_a} and {args.dir_b} have no "
              f"overlapping .bin dumps. Was VLA_BITVLA_DUMP_DIR set?")
        return 1
    if first_bad is not None:
        print(f"FIRST DIVERGENCE: {first_bad}")
    elif rc == 0:
        print(f"ALL {compared} STAGES within {args.tol}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
