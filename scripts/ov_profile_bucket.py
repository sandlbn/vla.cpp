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
"""Attribute OpenVINO GPU device time to ops, from GGML_OPENVINO_PROFILE_OPS.

unitrace answers "which kernels ran and for how long" and job 372831 used it to
establish that `gemm_kernel` is 52.4-52.7% of steady device time across three
weight formats. What it cannot answer is which *op* each kernel came from, and
the mapping is many-to-one in the direction that matters: one `reduce_ref` row
covers act_quant's row absmax, RMSNorm's mean-of-squares and every other
reduction in the graph at once. That left ~32% of device time as "activation-side
elementwise work of mixed provenance", which is not something you can fuse.

This reads the OVPROF lines the profiling hunk prints and buckets them three ways:

    by node type     what kind of op it is        (the Amdahl split)
    by kernel        which cldnn kernel ran it    (joins back to unitrace)
    by node x kernel which op used which kernel   (the attribution itself)

The third table is the point. A `reduce_ref` row that splits 70/30 between
ReduceMax and ReduceMean says how much of that 17% is act_quant and how much is
normalisation, and those two want completely different treatment.

    python3 scripts/ov_profile_bucket.py run.err
    python3 scripts/ov_profile_bucket.py run.err --name-filter act_quant

Times are ov::ProfilingInfo::real_time, i.e. per-node device duration in
microseconds, summed over every dumped infer call. Nodes the plugin fused or
folded away report OPTIMIZED_OUT and cost nothing; they are counted separately
rather than dropped, because a large optimised-out population is itself the
finding that the graph was written for a fusion that did happen.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# "OVPROF <us> <status> <node_type> <exec_type> <node name, may contain spaces>"
_LINE = re.compile(r"^OVPROF (\d+) (EXEC|OPTOUT|NOTRUN) (\S+) (\S+) (.*)$")
_BEGIN = re.compile(r"^OVPROF_BEGIN call=(\d+) nodes=(\d+)$")


def parse(text: str):
    rows = []
    calls = 0
    for line in text.splitlines():
        line = line.strip()
        # GGML_LOG_INFO may prefix the line; find our marker wherever it starts.
        i = line.find("OVPROF")
        if i < 0:
            continue
        line = line[i:]
        if _BEGIN.match(line):
            calls += 1
            continue
        m = _LINE.match(line)
        if m:
            rows.append((int(m.group(1)), m.group(2), m.group(3), m.group(4), m.group(5).strip()))
    return rows, calls


def table(title: str, totals: dict, counts: dict, denom: int, limit: int) -> None:
    print(f"\n=== {title} ===")
    print(f"{'bucket':<46} {'ms':>10} {'%':>7} {'nodes':>8}")
    for k in sorted(totals, key=lambda k: -totals[k])[:limit]:
        pct = 100.0 * totals[k] / denom if denom else 0.0
        print(f"{k:<46} {totals[k] / 1000.0:10.2f} {pct:7.2f} {counts[k]:8d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="+", help="run log(s) containing OVPROF lines")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--name-filter", help="only count nodes whose name contains this")
    args = ap.parse_args()

    rows, calls = [], 0
    for p in args.log:
        r, c = parse(Path(p).read_text(errors="replace"))
        rows += r
        calls += c

    if not rows:
        # Saying so beats printing a table of zeros: the usual cause is that the
        # build predates the profiling hunk, which is indistinguishable from
        # "nothing ran" in any summary that treats missing as empty.
        print("ov_profile_bucket: no OVPROF lines found.")
        print("  Either the build predates the profiling hunk (rebuild after")
        print("  scripts/patch_ggml_openvino.py) or GGML_OPENVINO_PROFILE_OPS was unset.")
        return 1

    if args.name_filter:
        rows = [r for r in rows if args.name_filter in r[4]]
        print(f"filtered to {len(rows)} nodes whose name contains {args.name_filter!r}")

    by_type, n_type = defaultdict(int), defaultdict(int)
    by_kern, n_kern = defaultdict(int), defaultdict(int)
    by_pair, n_pair = defaultdict(int), defaultdict(int)
    optout = 0
    total = 0

    for us, status, ntype, etype, _name in rows:
        if status != "EXEC":
            optout += 1
            continue
        total += us
        by_type[ntype] += us
        n_type[ntype] += 1
        by_kern[etype] += us
        n_kern[etype] += 1
        by_pair[f"{ntype:<22} {etype}"] += us
        n_pair[f"{ntype:<22} {etype}"] += 1

    print(f"OVPROF: {calls} infer call(s), {len(rows)} node records, "
          f"{optout} not executed (fused or folded)")
    print(f"        {total / 1000.0:.2f} ms of device time attributed")

    table("by node type -- the Amdahl split", by_type, n_type, total, args.limit)
    table("by kernel -- joins back to unitrace", by_kern, n_kern, total, args.limit)
    table("by node type x kernel -- the attribution", by_pair, n_pair, total, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
