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

"""scripts/bitvla_quant.py, pinned.

Three things are checked, and the third is the reason this file exists.

1. Both encoders round-trip ternary x scale through `gguf.quants.dequantize`
   to within the f16 rounding of one block scale, at every shape and scale
   BitVLA actually uses.

2. ggml's own Q4_0 quantiser is pinned as **known bad** at 12.5%. It picks
   `d = max/-8`, which reaches `+s` at `q = 0` but needs `q = 16` for `-s`, so
   whichever sign loses gets clipped to `-7s/8`. Anyone who later "simplifies"
   pack_ternary_q4_0 into `quants.quantize(x, Q4_0)` fails here instead of
   shipping a 12.5% weight error that no action-level test on this model could
   ever attribute (see memory: BitVLA's decorrelation floor).

3. The fixture `tests/data/bitvla_q_blocks.bin` is regenerated and byte-compared.
   tests/test_bitvla_quant.cpp reads that same file and dequantises it with
   ggml's C implementation. Together the two directions close the loop: this
   test catches the encoder drifting, and the C++ one catches gguf-py and ggml
   disagreeing about a layout -- which they would have to, for the nibble
   pairing to be wrong in a way a Python-only round-trip could not see.

    python3 ci/test_bitvla_quant.py [--update-fixture]

Needs `gguf` on the path; a configured build tree has a vendored copy:
    PYTHONPATH=build-ov/_deps/llama-src/gguf-py python3 ci/test_bitvla_quant.py
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from bitvla_quant import (  # noqa: E402
    NotBlockable,
    Q4_0_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES,
    pack_ternary_q4_0,
    pack_ternary_q8_0,
)

FIXTURE = ROOT / "tests" / "data" / "bitvla_q_blocks.bin"
MAGIC = b"BVQF"
FIXTURE_VERSION = 1

# f16 carries 11 significand bits, so rounding one block scale costs at most
# 2^-11 relative. The encoders add nothing on top of that -- the integers land
# on grid points exactly -- so this is the whole error budget.
F16_SCALE_TOL = 2.0 ** -11

# ggml's own Q4_0 on ternary. Pinned so the failure is loud, not a surprise.
GGML_Q4_0_TERNARY_ERROR = 0.125

# The shapes BitVLA reduces over, from the tensor census of the published
# checkpoint. 4304 is vit.*.fc2 and is the one that does not divide 32.
K_VALUES = (2560, 6912, 1152, 4304, 32)
SCALES = (0.0123456, 1.0, 3.7e-3, 0.25, 2.5e-2)


def _gguf():
    try:
        from gguf import quants
        from gguf.constants import GGMLQuantizationType
    except ImportError:
        print("error: no `gguf` module. Try\n"
              "  PYTHONPATH=build-ov/_deps/llama-src/gguf-py python3 "
              "ci/test_bitvla_quant.py", file=sys.stderr)
        raise SystemExit(2)
    return quants, GGMLQuantizationType


def check_roundtrip() -> int:
    quants, T = _gguf()
    rng = np.random.default_rng(20260916)
    bad = 0
    for k in K_VALUES:
        n = 8
        tern = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
        for scale in SCALES:
            x = tern.astype(np.float32) * np.float32(scale)
            for name, fn, blk, qtype in (
                ("q8_0", pack_ternary_q8_0, Q8_0_BLOCK_BYTES, T.Q8_0),
                ("q4_0", pack_ternary_q4_0, Q4_0_BLOCK_BYTES, T.Q4_0),
            ):
                try:
                    packed = fn(tern, scale, f"{n}x{k}")
                except NotBlockable:
                    if k % 32 == 0:
                        print(f"FAIL {name} K={k}: refused a blockable shape")
                        bad += 1
                    continue
                if k % 32:
                    print(f"FAIL {name} K={k}: accepted a shape that is "
                          f"{k % 32} mod 32")
                    bad += 1
                    continue
                if packed.shape != (n, k // 32 * blk) or packed.dtype != np.uint8:
                    print(f"FAIL {name} K={k}: packed {packed.shape} {packed.dtype}")
                    bad += 1
                    continue
                got = quants.dequantize(packed, qtype).astype(np.float32)
                err = float(np.abs(got - x).max()) / float(scale)
                if err > F16_SCALE_TOL:
                    print(f"FAIL {name} K={k} scale={scale:g}: rel err {err:.3g} "
                          f"> {F16_SCALE_TOL:.3g}")
                    bad += 1
    if not bad:
        print(f"ok   round-trip: both encoders within {F16_SCALE_TOL:.3g} "
              f"at {len(K_VALUES)} shapes x {len(SCALES)} scales")
    return bad


def check_ggml_q4_0_is_unusable() -> int:
    """The trap. If this ever passes, ggml changed and the comment is stale."""
    quants, T = _gguf()
    rng = np.random.default_rng(1)
    tern = rng.integers(-1, 2, size=(8, 2560)).astype(np.int8)
    scale = np.float32(0.0123456)
    x = tern.astype(np.float32) * scale
    got = quants.dequantize(quants.quantize(x, T.Q4_0), T.Q4_0).astype(np.float32)
    err = float(np.abs(got - x).max()) / float(scale)
    if abs(err - GGML_Q4_0_TERNARY_ERROR) > 1e-3:
        print(f"FAIL ggml Q4_0 on ternary: {err:.4g}, expected "
              f"{GGML_Q4_0_TERNARY_ERROR} (d = max/-8 clipping one sign to -7s/8)."
              f"\n     If ggml's quantiser changed, re-derive pack_ternary_q4_0's "
              f"d = s/7 before trusting this.")
        return 1
    print(f"ok   ggml's own Q4_0 still loses {err:.3f} on ternary "
          f"-- pack_ternary_q4_0 is still required")
    return 0


def fixture_bytes() -> bytes:
    """Cases for tests/test_bitvla_quant.cpp, self-describing so C hardcodes nothing.

    header  magic "BVQF", u32 version, u32 n_cases
    case    u32 ggml_type, u32 rows, u32 k, f32 scale, u32 n_payload
            u8[n_payload] blocks, i8[rows*k] source ternary
    """
    quants, T = _gguf()
    rng = np.random.default_rng(4242)
    cases = []
    for k in (32, 64, 1152, 2560):
        for name, fn, qtype in (("q8_0", pack_ternary_q8_0, T.Q8_0),
                                ("q4_0", pack_ternary_q4_0, T.Q4_0)):
            rows = 6
            tern = rng.integers(-1, 2, size=(rows, k)).astype(np.int8)
            # Three rows pinned: both saturated signs (ggml's own Q4_0 clips one
            # of them) and an all-zero block (where it would pick d = 0). The
            # rows left random are what catches a wrong nibble pairing -- a
            # constant row survives any permutation of its own elements.
            tern[0, :] = 1
            tern[1, :] = -1
            tern[rows - 1, :] = 0
            scale = np.float32(0.0123456 if k % 64 else 0.25)
            packed = fn(tern, scale, f"fixture-{name}-{k}")
            cases.append((int(qtype), rows, k, float(scale),
                          packed.tobytes(), tern.tobytes()))

    out = bytearray(MAGIC + struct.pack("<II", FIXTURE_VERSION, len(cases)))
    for qtype, rows, k, scale, payload, src in cases:
        out += struct.pack("<IIIfI", qtype, rows, k, scale, len(payload))
        out += payload
        out += src
    return bytes(out)


def check_fixture(update: bool) -> int:
    want = fixture_bytes()
    if update:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_bytes(want)
        print(f"ok   wrote {FIXTURE.relative_to(ROOT)} ({len(want)} bytes)")
        return 0
    if not FIXTURE.exists():
        print(f"FAIL {FIXTURE.relative_to(ROOT)} is missing. Regenerate with\n"
              f"     python3 ci/test_bitvla_quant.py --update-fixture")
        return 1
    if FIXTURE.read_bytes() != want:
        print(f"FAIL {FIXTURE.relative_to(ROOT)} does not match the encoders.\n"
              f"     Either an encoder changed (regenerate with "
              f"--update-fixture, and re-run\n"
              f"     the ctest bitvla_quant so ggml gets a say), or it changed "
              f"by accident.")
        return 1
    print(f"ok   {FIXTURE.relative_to(ROOT)} matches the encoders "
          f"({len(want)} bytes)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update-fixture", action="store_true",
                    help="rewrite tests/data/bitvla_q_blocks.bin from the encoders")
    args = ap.parse_args()

    bad = check_roundtrip()
    bad += check_ggml_q4_0_is_unusable()
    bad += check_fixture(args.update_fixture)
    print("FAILED" if bad else "PASSED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
