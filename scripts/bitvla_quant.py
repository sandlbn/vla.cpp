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

"""Ternary BitLinear weights -> ggml Q8_0 / Q4_0 blocks, exactly.

Shared by `convert_bitvla_to_gguf.py` and `transcode_bitvla_int2.py`. These two
encoders are the only new numerics in the low-precision work, so the whole
argument for why they are lossless lives here.

BitVLA's BitLinear weights are ternary with a *per-tensor* scale:
`weight_quant_to_ternary` returns `(tern in {-1,0,+1}, absmean)`, one float for
the whole matrix. A dense flavour stores `tern * scale`, which spends a full
mantissa per element re-representing one number. A ggml block format stores the
scale once per 32 weights and the integers exactly, so it is both smaller and
*more* faithful. Measured, ternary x scale, max relative error:

    bf16                   2.0000 bytes/weight   1.34e-03
    Q8_0                   1.0625               5.89e-05
    Q4_0 (this encoder)    0.5625               1.78e-04
    Q4_0 (ggml's own)      0.5625               1.25e-01   <-- do not use

The residual in the two middle rows is the f16 rounding of the block scale and
nothing else; the integers land on grid points with no error at all.

**Q8_0** dequantises as `d * q`. Take `d = s/127` and `{-s, 0, +s}` are `q =
{-127, 0, 127}`. ggml's own quantiser already finds this (it picks
`d = amax/127`, and amax is `s` in every block that has a nonzero), so Q8_0
could equally be produced by `gguf.quants.quantize`. It is written out here
anyway so both formats are read off the same table.

**Q4_0** dequantises as `d * (q - 8)` with `q` in `[0, 15]`, and this is where
ggml's quantiser cannot be used. It picks `d = max/-8`, which places `+s` at
`q = 0` but needs `q = 16` for `-s`; that clips to `-7s/8`, the 12.5% above, on
whichever sign loses. It is not a tuning failure, it is the grid being offset.
Taking `d = s/7` instead puts `{-s, 0, +s}` on `q = {1, 8, 15}`, all three
exact, and gives up only the two extreme codes -- which a ternary matrix has no
use for.

Both encoders emit a *uniform* `d` for every block, including all-zero ones.
ggml would pick `d = 0` there; either way every element dequantises to 0, so the
blocks agree on the values and only disagree on an unused scale.
"""

from __future__ import annotations

import numpy as np

# ggml's block length for Q4_0 and Q8_0 alike. Both formats lead with a 2-byte
# f16 scale; Q8_0 then carries 32 int8 and Q4_0 16 packed nibble pairs.
QK = 32
Q8_0_BLOCK_BYTES = 2 + QK        # 34
Q4_0_BLOCK_BYTES = 2 + QK // 2   # 18

# Indexed by tern + 1, so -1 -> [0], 0 -> [1], +1 -> [2]. A lookup rather than
# arithmetic: it is the same three values for every weight in the model.
_Q8_LEVELS = np.array([-127, 0, 127], dtype=np.int8)
_Q4_LEVELS = np.array([1, 8, 15], dtype=np.uint8)   # dequant d*(q-8) -> -7d, 0, +7d


class NotBlockable(ValueError):
    """The reduction dimension does not divide ggml's 32-element block."""


def _prepare(tern: np.ndarray, scale: float, levels_per_side: float, what: str):
    """Common checks, and the f16 block scale both formats lead with.

    Returns (tern as int8 (N, K), nb, the f16 scale). `levels_per_side` is how
    many integer steps the format spends reaching +s from 0: 127 for Q8_0, 7 for
    Q4_0.
    """
    tern = np.ascontiguousarray(tern)
    if tern.ndim != 2:
        raise ValueError(f"{what}: expected a 2-D (N, K) weight, got {tern.shape}")
    if tern.dtype != np.int8:
        raise ValueError(f"{what}: expected int8 ternary, got {tern.dtype}")
    lo, hi = int(tern.min(initial=0)), int(tern.max(initial=0))
    if lo < -1 or hi > 1:
        raise ValueError(f"{what}: values outside {{-1,0,+1}} (min {lo}, max {hi})")

    _, k = tern.shape
    if k % QK:
        # The caller decides the fallback -- for BitVLA that is bf16 for
        # vit.*.fc2, whose K is 4304 and 4304 % 32 == 16. Refusing here rather
        # than padding keeps the shape the loader asserts against.
        raise NotBlockable(f"{what}: K={k} is not a multiple of {QK}")

    s = np.float32(scale)
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"{what}: scale {scale!r} is not a positive finite float")
    d = np.float16(s / np.float32(levels_per_side))
    # f16 bottoms out at 6e-8 subnormal and tops out at 65504. A scale that
    # falls off either end would dequantise to zeros or infinities, silently.
    if d == 0 or not np.isfinite(d):
        raise ValueError(f"{what}: scale {scale!r} does not survive f16 as d={float(d)!r}")
    return tern, k // QK, d


def _lead_with_scale(d: np.float16, n: int, nb: int, qs: np.ndarray) -> np.ndarray:
    """Prepend the 2-byte f16 block scale to each block's quantised payload."""
    dbytes = np.frombuffer(np.full(n * nb, d, dtype=np.float16).tobytes(),
                           dtype=np.uint8).reshape(n, nb, 2)
    return np.concatenate([dbytes, qs], axis=2).reshape(n, -1)


def pack_ternary_q8_0(tern: np.ndarray, scale: float, what: str = "tensor") -> np.ndarray:
    """(N, K) ternary int8 -> (N, K/32 * 34) uint8 of ggml Q8_0 blocks."""
    tern, nb, d = _prepare(tern, scale, 127.0, what)
    n = tern.shape[0]
    qs = _Q8_LEVELS[tern + 1].reshape(n, nb, QK).view(np.uint8)
    return _lead_with_scale(d, n, nb, qs)


def pack_ternary_q4_0(tern: np.ndarray, scale: float, what: str = "tensor") -> np.ndarray:
    """(N, K) ternary int8 -> (N, K/32 * 18) uint8 of ggml Q4_0 blocks.

    ggml packs element `j` in the low nibble and element `j + 16` -- not
    `j + 1` -- in the high nibble (`qs[j] = xi0 | (xi1 << 4)` in
    quantize_row_q4_0_ref). Getting that pairing wrong still round-trips through
    a matching unpacker, so it is checked against ggml's own C dequantiser in
    tests/test_bitvla_quant.cpp, not only against gguf-py.
    """
    tern, nb, d = _prepare(tern, scale, 7.0, what)
    n = tern.shape[0]
    nib = _Q4_LEVELS[tern + 1].reshape(n, nb, QK)
    qs = (nib[:, :, :QK // 2] | (nib[:, :, QK // 2:] << 4)).astype(np.uint8)
    return _lead_with_scale(d, n, nb, qs)


PACKERS = {"q8_0": pack_ternary_q8_0, "q4_0": pack_ternary_q4_0}
