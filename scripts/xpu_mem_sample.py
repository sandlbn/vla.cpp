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
"""Peak GPU memory of a process on an Intel GPU, from DRM fdinfo.

eval/client/benchmark.py samples device memory by shelling out to nvidia-smi,
so on this cluster every `*.mem.json` carries `peak_vram_mib: null`. That gap is
why a 4.8x *host* footprint win could not be checked against what the GPU
actually holds -- which is the one measurement that separates "the plugin runs
the weights compressed" from "the plugin decompressed them to f16 at compile
time and the win is host-only".

The kernel publishes per-client GPU memory through the DRM fdinfo interface:
every open render node under /proc/<pid>/fdinfo/<fd> carries `drm-driver` plus
some set of memory keys. This is the same source intel_gpu_top reads for its
per-client column, it needs no root and no vendor tool, and it is per-process
rather than device-wide -- so a second tenant on the node cannot inflate it.

The key names are driver- and generation-specific (i915 names local memory
`local0`, xe names it `vram0`, integrated parts report `system0` only), so this
deliberately does not look for one blessed key. It records every `drm-*` key it
finds and reports each one's peak, leaving the reader to pick. `--summary-key`
names the substring that gets promoted to the headline number.

    python3 scripts/xpu_mem_sample.py --pid 12345 --out mem.json
    python3 scripts/xpu_mem_sample.py --pid 12345 --summary-key resident-vram

Values are summed across the fds of the process that expose a `drm-driver` line
and tracked as a peak over time. Summing is right when a process holds one
render node, which is the normal case; if it holds several the per-fd count is
printed so the double-count is visible rather than silent.

Exits when the process does, or on --timeout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# "drm-resident-vram0:	1234 KiB" -- the unit suffix is optional and the kernel
# emits KiB for memory keys and bare integers for counters like drm-total-cycles.
_LINE = re.compile(r"^(drm-[a-z0-9-]+):\s+(\d+)\s*([KMG]iB)?\s*$")
_UNIT = {None: 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}

MIB = 1024.0 ** 2


def sample_once(pid: int) -> tuple[dict[str, int], int]:
    """Sum every drm-* memory key across this pid's DRM fds. (values, n_fds)"""
    totals: dict[str, int] = {}
    n_fds = 0
    fdinfo = Path(f"/proc/{pid}/fdinfo")
    try:
        entries = list(fdinfo.iterdir())
    except OSError:
        return totals, 0  # process gone, or not ours to read
    for entry in entries:
        try:
            text = entry.read_text()
        except OSError:
            continue  # fds come and go while we walk them; skip, do not fail
        if "drm-driver" not in text:
            continue
        n_fds += 1
        for line in text.splitlines():
            m = _LINE.match(line.strip())
            if not m:
                continue
            key, val, unit = m.group(1), int(m.group(2)), m.group(3)
            # Only the keys carrying a byte unit are memory; the rest are
            # counters (drm-total-cycles and friends) that must not be added to
            # a memory total just because they share the prefix.
            if unit is None:
                continue
            totals[key] = totals.get(key, 0) + val * _UNIT[unit]
    return totals, n_fds


def alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--interval", type=float, default=0.2, help="seconds between samples")
    ap.add_argument("--timeout", type=float, default=0.0, help="0 = until the process exits")
    ap.add_argument("--out", help="write the peak table here as JSON")
    ap.add_argument("--summary-key", default="resident",
                    help="substring of the key to promote to the headline (default: resident)")
    args = ap.parse_args()

    peaks: dict[str, int] = {}
    max_fds = 0
    n_samples = 0
    t0 = time.monotonic()

    while alive(args.pid):
        vals, n_fds = sample_once(args.pid)
        if vals:
            n_samples += 1
            max_fds = max(max_fds, n_fds)
            for k, v in vals.items():
                peaks[k] = max(peaks.get(k, 0), v)
        if args.timeout and (time.monotonic() - t0) >= args.timeout:
            break
        time.sleep(args.interval)

    # A zero-sample run is a real finding, not a crash: it means this driver
    # publishes no fdinfo memory keys, and saying so is more use than a 0.
    if not peaks:
        print(f"xpu_mem: NO DRM fdinfo memory keys for pid {args.pid} "
              f"({n_samples} samples over {time.monotonic() - t0:.1f}s)")
        print("xpu_mem: device memory is unavailable from this driver -- "
              "do not read the absence as zero")
    else:
        print(f"xpu_mem: {n_samples} samples, {max_fds} DRM fd(s), peaks:")
        for k in sorted(peaks, key=lambda k: -peaks[k]):
            print(f"xpu_mem:   {k:<28} {peaks[k] / MIB:10.1f} MiB "
                  f"({peaks[k] / 1024 ** 3:.2f} GiB)")

    headline = None
    cands = {k: v for k, v in peaks.items() if args.summary_key in k}
    if cands:
        headline = max(cands.values())
        print(f"xpu_mem: PEAK {headline / MIB:.1f} MiB "
              f"({headline / 1024 ** 3:.2f} GiB)  [key match '{args.summary_key}']")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "pid": args.pid,
            "n_samples": n_samples,
            "n_drm_fds": max_fds,
            "peak_bytes": peaks,
            "peak_mib": {k: round(v / MIB, 1) for k, v in peaks.items()},
            "headline_key_substring": args.summary_key,
            "peak_vram_mib": round(headline / MIB, 1) if headline is not None else None,
        }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
