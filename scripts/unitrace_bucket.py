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
"""Bucket a unitrace --device-timing report into op families.

unitrace reports one row per kernel *instantiation*, which for a BitVLA request
is ninety-odd rows: oneDNN emits a separate `gemm_kernel` per shape, and each of
our elementwise kernels appears once per work-group geometry it was launched
with. That is the right granularity for tuning one kernel and the wrong one for
answering "where does the request go", which is the question on the table.

Two runs at different --reps separate per-request cost from one-time cost. The
weight unpack is the case that forces this: it looks like 10 ms of device time
and it is a load-time cost paid once, so dividing the single-run total by the
number of requests would charge every request for 1.7 ms it never pays. With two
points the split is just a line - slope is per-request, intercept is startup:

    t(reps) = intercept + slope * reps

Usage:
    unitrace_bucket.py REPORT_A REPS_A [REPORT_B REPS_B]

With one report, everything is divided by REPS_A and reported as an upper bound,
which is honest but blunt. With two, the split is exact.
"""

import re
import sys
from pathlib import Path

# Ordered: first match wins, so the specific patterns precede the general ones.
# `gemm_kernel` is oneDNN's matmul under every shape - ternary bitlinear, the two
# batched attention GEMMs and the fp32 head all land there, and the device report
# cannot tell them apart. The per-shape split lives in docs/backend/sycl.md,
# measured separately with VLA_BITVLA_LOG_SHAPES=1.
BUCKETS = [
    ("gemm",        r"gemm_kernel|::gemm|xetla|brgemm"),
    ("usm-copy-2d", r"commonUSMCopy2DFallbackKernel"),
    ("memcpy",      r"zeCommandListAppendMemoryCopy|zeCommandListAppendMemoryFill"),
    ("unpack",      r"k_unpack|k_ws_expand|ladder|pack"),
    ("act_quant",   r"k_act_quant"),
    ("norm",        r"k_rmsnorm|k_layernorm"),
    ("softmax",     r"k_softmax"),
    ("ffn-act",     r"k_gate_up_fused|k_gelu|k_relu|k_sqrelu"),
    ("transpose",   r"k_transpose|k_repeat_kv"),
    ("rope",        r"k_rope"),
    ("elementwise", r"k_add|k_mul|k_scale|k_cast|k_copy"),
]


def bucket_of(name: str) -> str:
    for label, pat in BUCKETS:
        if re.search(pat, name):
            return label
    return "other"


def parse(path: Path) -> dict[str, tuple[int, int]]:
    """-> {bucket: (calls, ns)} from a unitrace device-timing report."""
    rows: dict[str, tuple[int, int]] = {}
    in_table = False
    for line in path.read_text(errors="replace").splitlines():
        if "Device Timing Summary" in line or "== L0 Backend ==" in line:
            in_table = True
            continue
        if not in_table:
            continue
        # The table ends at the next section header.
        if line.startswith("==="):
            break
        # Kernel names contain commas (C++ signatures), so split from the right:
        # the last five fields are always Calls, Time, Time%, Average, Min, Max.
        parts = [p.strip() for p in line.rsplit(",", 6)]
        if len(parts) != 7:
            continue
        name, calls, ns = parts[0].strip('" '), parts[1], parts[2]
        if not calls.isdigit() or not ns.isdigit():
            continue          # header row, or a name that happened to have 6 commas
        b = bucket_of(name)
        c, t = rows.get(b, (0, 0))
        rows[b] = (c + int(calls), t + int(ns))
    if not rows:
        raise SystemExit(f"no device-timing rows parsed from {path}")
    return rows


def main() -> None:
    av = sys.argv[1:]
    if len(av) not in (2, 4):
        raise SystemExit(__doc__)
    a, reps_a = parse(Path(av[0])), int(av[1])
    b, reps_b = (parse(Path(av[2])), int(av[3])) if len(av) == 4 else (None, 0)

    keys = sorted(set(a) | set(b or {}), key=lambda k: -a.get(k, (0, 0))[1])
    print(f"{'bucket':<14} {'calls/req':>10} {'ms/req':>9} {'% dev':>7} {'startup ms':>11}")
    print("-" * 56)

    per_req, startup = {}, {}
    for k in keys:
        ca, ta = a.get(k, (0, 0))
        if b is not None:
            cb, tb = b.get(k, (0, 0))
            d = reps_b - reps_a
            slope_t = (tb - ta) / d
            slope_c = (cb - ca) / d
            # Clamp: a bucket that is purely startup regresses to a tiny negative
            # slope on measurement noise, and a negative ms/req is not a thing.
            per_req[k] = (max(slope_c, 0.0), max(slope_t, 0.0))
            startup[k] = max(ta - slope_t * reps_a, 0.0)
        else:
            per_req[k] = (ca / reps_a, ta / reps_a)
            startup[k] = 0.0

    total = sum(t for _, t in per_req.values())
    for k in keys:
        c, t = per_req[k]
        pct = 100.0 * t / total if total else 0.0
        print(f"{k:<14} {c:>10.1f} {t/1e6:>9.3f} {pct:>6.1f}% {startup[k]/1e6:>11.1f}")
    print("-" * 56)
    tc = sum(c for c, _ in per_req.values())
    print(f"{'TOTAL':<14} {tc:>10.1f} {total/1e6:>9.3f} {100.0:>6.1f}% "
          f"{sum(startup.values())/1e6:>11.1f}")


if __name__ == "__main__":
    main()
