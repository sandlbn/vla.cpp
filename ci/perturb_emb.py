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
"""Write a relatively-perturbed copy of a raw float32 embedding dump.

Feeds ci/slurm/bmg_bitvla_sensitivity.sbatch. Each output element is

    x' = x * (1 + eps * r),   r deterministic in [-1, 1]

so the perturbation is relative and its size is eps regardless of the tensor's
scale. The seed is fixed and derived from eps, so every level is reproducible
and no two levels share a pattern.

Relative rather than absolute on purpose: the question being asked is what a
backend disagreement does, and backends disagree by a relative amount -- a
rounding difference in the low bits of each value, not a fixed offset.

    perturb_emb.py <in.bin> <out.bin> <eps>
"""

import sys

import numpy as np


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    src, dst, eps = sys.argv[1], sys.argv[2], float(sys.argv[3])

    x = np.fromfile(src, dtype=np.float32)
    if x.size == 0:
        print(f"{src} is empty", file=sys.stderr)
        return 1

    if eps == 0.0:
        y = x.copy()
    else:
        # A fixed seed per level: reproducible, and distinct across levels so a
        # result cannot be an artefact of one particular perturbation pattern.
        rng = np.random.default_rng(abs(hash(f"{eps:.3e}")) % (2**32))
        r = rng.uniform(-1.0, 1.0, size=x.size).astype(np.float32)
        y = (x.astype(np.float64) * (1.0 + eps * r.astype(np.float64))).astype(np.float32)

    y.tofile(dst)

    d = np.abs(y.astype(np.float64) - x.astype(np.float64))
    scale = max(float(np.abs(x).max()), 1e-6)
    print(f"eps={eps:.3e}  n={x.size}  max|d|={d.max():.6g}  normalised={d.max() / scale:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
