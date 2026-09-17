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

"""Published int2 BitVLA GGUF -> a dense or block-quantised flavour, no torch.

`convert_bitvla_to_gguf.py --weight-dtype <t>` is the way to make these files and
it needs the original HF checkpoint, torch, and a download. This script is the
same destination reached from the published int2 GGUF instead, which is useful
on a cluster node that has neither -- and it is not an approximation of the
converter, it is the identical function applied to the identical intermediate.

The reason is that quantisation has already happened by the time the int2 file
exists. `weight_quant_to_ternary` returns `(tern, absmean)` and the int2 flavour
stores exactly those two things: the ladder-packed ternary values and the scale.
The bf16 flavour stores `_dense_from_ternary(tern, scale)`, which is
`tern.float() * scale`. So the transcode is an unpack and a multiply over inputs
that are already final, and the bytes it writes are the bytes the converter
would have written.

That multiply is worth being precise about, because it is exact and that is what
makes this safe. `tern` is one of -1, 0, +1, so `tern * scale` in f32 is one of
-scale, 0, +scale with no rounding anywhere -- multiplying by +-1 is exact in
any floating-point format, and so is multiplying by zero. The only rounding in
the whole pipeline is the f32->bf16 of `scale` itself, and it happens once per
tensor rather than once per element. A whole dense weight matrix therefore holds
three distinct bit patterns, so the conversion below is a four-entry lookup on
the two-bit code rather than any arithmetic on the values.

`--weight-dtype q8_0` and `q4_0` keep that property and sharpen it. They are the
same lookup with a different table -- the ternary integers go into a ggml block
verbatim and the scale is stored once per 32 weights rather than folded into
every element, so they are *more* faithful than bf16 while being half its size
or less (bf16 1.34e-03 max relative error, Q8_0 5.89e-05, Q4_0 1.78e-04; see
`bitvla_quant.py`, which holds both encoders and the argument for them).

One tensor cannot take them. `vit.blk.N.fc2.weight` reduces over K = 4304 and
4304 % 32 == 16, so it does not divide ggml's block; it stays bf16 and the run
says so rather than quietly shrinking less than it claims.

Three asymmetries between the two flavours, all of them the loader's rules and
all handled here:

  * K padding comes off. `_add_bit(..., ffn_pad)` pads the ViT fc2 reduction
    dimension up to a multiple of 128 so K divides the ladder tile. A dense GEMM
    has no such constraint and `bitvla.cpp` asserts the unpadded shape.
  * gate and up come apart. The kernels want one fused projection so a single
    GEMM covers both halves; `bitvla.cpp` reads `ffn_gate_up` only when the
    checkpoint is int2-packed and looks for `ffn_gate` / `ffn_up` otherwise.
    They carry a scale each, in the order they were concatenated.
  * The `.scale` tensors go away, and so does `bitvla.quant.int2_packed` -- the
    scale is folded into the weights and there is no GEMM epilogue left to apply
    it in. The key is dropped rather than set to 0 because the loader reads it as
    `g.has(...) && g.u32(...) != 0` and absent is the form every checkpoint
    without it already takes.

Everything else -- norms, biases, embeddings, the projector and the action head
-- is already dense bf16 or f32 in the int2 file and is copied through byte for
byte.

Usage:
    python3 scripts/transcode_bitvla_int2.py IN.gguf OUT.gguf [--weight-dtype T]

`gguf` is imported from wherever it is installed; the vendored copy inside a
configured build tree works too:
    PYTHONPATH=build-ov/_deps/llama-src/gguf-py python3 scripts/transcode_...
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.constants import GGMLQuantizationType

from bitvla_quant import PACKERS, NotBlockable

ARCH = "bitvla"
WEIGHT_DTYPES = ("bf16", *PACKERS)

# Mirrors src/kernels/bitvla/ladder_pack.h. Named rather than inlined so the two
# can be diffed by eye; they are a pack/unpack pair and must not drift.
LADDER_N_BLOCK  = 16
LADDER_K_BLOCK  = 8
LADDER_K_PER_LOOP = 16
LADDER_WMMA_K   = 32
LADDER_K_PER_ITER = LADDER_K_PER_LOOP * LADDER_K_BLOCK   # 128


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 bit patterns, round-to-nearest-even.

    The same rounding `torch.Tensor.to(torch.bfloat16)` does, which is what the
    converter uses, spelled out here because numpy has no bfloat16. Adding
    0x7FFF plus the low bit of the surviving mantissa rounds half to even; it is
    the standard formulation and it is exact for every finite input.

    Only ever called on scales, one value at a time -- see the module docstring
    for why the elements themselves need no arithmetic.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite value in a BitVLA scale")
    u = x.view(np.uint32).astype(np.uint64)          # uint64: the +0x7FFF cannot wrap
    u += 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    """Inverse of the above, for the self-check. Exact: bf16 is a prefix of f32."""
    return (np.asarray(b, dtype=np.uint32) << 16).view(np.float32)


def encode_bf16(codes: np.ndarray, scale: float):
    """Two-bit codes -> bf16, by lookup. 1 -> -scale, 2 -> 0, 3 -> +scale."""
    b = int(f32_to_bf16_bits(np.array([scale], dtype=np.float32))[0])
    lut = np.array([0, b ^ 0x8000, 0, b], dtype=np.uint16)
    return lut[codes], GGMLQuantizationType.BF16, f"bf16 {bf16_bits_to_f32(np.array([b]))[0]:.6g}"


def encode(name: str, codes: np.ndarray, scale: float, want: str):
    """Two-bit codes -> (payload, ggml type, note), in the requested flavour.

    A tensor whose K does not divide ggml's 32-element block falls back to bf16
    and says so in its note; that is `vit.*.fc2` and nothing else in this model.
    """
    if want == "bf16":
        return encode_bf16(codes, scale)
    # code - 2 is the ternary value, and the codes were already range-checked.
    tern = codes.astype(np.int8) - 2
    try:
        packed = PACKERS[want](tern, scale, name)
    except NotBlockable as e:
        data, dtype, note = encode_bf16(codes, scale)
        return data, dtype, f"{note}  <-- not {want}: {str(e).split(': ', 1)[-1]}"
    return packed, GGMLQuantizationType[want.upper()], f"{want} d={scale:.6g}/level"


def ladder_unpack_codes(packed: np.ndarray, N: int, K: int) -> np.ndarray:
    """Ladder int2 -> row-major (N, K) two-bit codes.

    A transcription of `ladder_unpack_int2`, vectorised over slots. The code is
    left undecoded (0..3, where the stored value is code - 2) so the caller can
    turn it into bf16 with a lookup instead of arithmetic.

    The slot decomposition is a bijection over the logical matrix, so every
    output element is written exactly once and the buffer needs no fill.
    """
    if N % LADDER_N_BLOCK or K % LADDER_K_PER_ITER:
        raise ValueError(f"({N}, {K}) does not divide the ladder tile")
    n_slots = N * K // 16
    if packed.size != n_slots * 4:
        raise ValueError(f"packed size {packed.size} != {n_slots * 4} for ({N}, {K})")

    slots = packed.reshape(n_slots, 4)
    s = np.arange(n_slots, dtype=np.int64)

    # slots_per_block is (LADDER_N_BLOCK * K) / 16, which is K -- kept as the
    # expression rather than the answer so it tracks the constant.
    slots_per_block = (LADDER_N_BLOCK * K) // 16
    n_block,  in_block = np.divmod(s,        slots_per_block)
    k_0,      in_k0    = np.divmod(in_block, 128)
    major_k,  in_major = np.divmod(in_k0,    32)
    y_half,   in_yhalf = np.divmod(in_major, 16)
    sub_k,    y_in_h   = np.divmod(in_yhalf, 8)

    n_global = n_block * LADDER_N_BLOCK + y_half * 8 + y_in_h
    k_base   = k_0 * LADDER_K_PER_ITER + major_k * LADDER_WMMA_K + sub_k * LADDER_K_PER_LOOP
    base = n_global * K + k_base
    del s, n_block, in_block, k_0, in_k0, major_k, in_major, y_half, in_yhalf, sub_k, y_in_h
    del n_global, k_base

    out = np.empty(N * K, dtype=np.uint8)
    for t in range(16):
        # Value t lives in byte t%4 at bit pair t/4: four consecutive values in
        # the four lanes of one 32-bit word, which is what made the CUDA decode a
        # single lop3. See ladder_slot_bit_pos.
        out[base + t] = (slots[:, t % 4] >> (2 * (t // 4))) & 0x3
    return out.reshape(N, K)


def bit_tensor_shape(base: str, cfg: dict):
    """(N, K_packed, K_logical, split) for a ternary tensor, or None if not one.

    The shape table is `load_bit`'s call sites in src/models/bitvla.cpp, derived
    from the checkpoint's own KV rather than hardcoded. `split` is None except
    for the fused gate/up projection, where it names the two halves in the order
    `pack_fused_projection` concatenated them.
    """
    H, LI = cfg["lm_hidden"], cfg["lm_inter"]
    V, VI = cfg["vit_hidden"], cfg["vit_inter"]
    hq  = cfg["lm_q_heads"]  * cfg["lm_head_dim"]
    hkv = cfg["lm_kv_heads"] * cfg["lm_head_dim"]
    ffn_pad = ((VI + 127) // 128) * 128

    m = re.fullmatch(r"vit\.blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_o|fc1|fc2)", base)
    if m:
        what = m.group(2)
        if what == "fc1":
            return (VI, V, V, None)
        if what == "fc2":
            return (V, ffn_pad, VI, None)      # the one tensor that is K-padded
        return (V, V, V, None)

    m = re.fullmatch(r"lm\.blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_o|ffn_gate_up|ffn_down)", base)
    if m:
        L, what = m.group(1), m.group(2)
        if what == "attn_q":      return (hq,  H, H, None)
        if what in ("attn_k", "attn_v"): return (hkv, H, H, None)
        if what == "attn_o":      return (H,   H, H, None)
        if what == "ffn_down":    return (H,  LI, LI, None)
        return (2 * LI, H, H, [f"lm.blk.{L}.ffn_gate", f"lm.blk.{L}.ffn_up"])

    return None


def read_config(reader: GGUFReader) -> dict:
    def u32(k):
        f = reader.fields.get(f"{ARCH}.{k}")
        if f is None:
            raise KeyError(f"{ARCH}.{k} missing; is this a BitVLA checkpoint?")
        return int(f.contents())
    return {
        "lm_hidden":   u32("lm.hidden"),
        "lm_inter":    u32("lm.inter"),
        "lm_q_heads":  u32("lm.q_heads"),
        "lm_kv_heads": u32("lm.kv_heads"),
        "lm_head_dim": u32("lm.head_dim"),
        "vit_hidden":  u32("vit.hidden"),
        "vit_inter":   u32("vit.inter"),
    }


# Reader-side pseudo-fields describing the container, not model metadata.
_CONTAINER_KEYS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}


def copy_kv(reader: GGUFReader, writer: GGUFWriter) -> int:
    """Every key across, minus the one that no longer applies.

    GGUFWriter's constructor has already written general.architecture, so that
    one is skipped rather than duplicated (a duplicate is a hard error there).
    Anything whose type this does not understand stops the run: silently losing
    a key would produce a file that loads and then behaves differently.
    """
    n = 0
    for key, field in reader.fields.items():
        if key in _CONTAINER_KEYS or key == "general.architecture":
            continue
        if key == f"{ARCH}.quant.int2_packed":
            continue
        if not field.types:
            raise ValueError(f"{key}: no type information")
        # types[-1] for the element type, matching how ReaderField.contents reads it.
        vtype, sub = field.types[0], (field.types[-1] if len(field.types) > 1 else None)
        if vtype is GGUFValueType.ARRAY and sub is None:
            raise ValueError(f"{key}: array with no element type")
        writer.add_key_value(key, field.contents(), vtype, sub_type=sub)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path, help="published int2 BitVLA GGUF")
    ap.add_argument("dst", type=Path, help="GGUF to write")
    ap.add_argument(
        "--weight-dtype",
        choices=WEIGHT_DTYPES,
        default="bf16",
        help="how the BitLinear weights are stored. 'bf16' folds the scale into "
             "every element; 'q8_0' and 'q4_0' keep the ternary integers and "
             "store the scale once per 32 weights, which is both smaller and "
             "more faithful (default: %(default)s)"
    )
    args = ap.parse_args()

    reader = GGUFReader(str(args.src))
    arch_f = reader.fields.get("general.architecture")
    if arch_f is None or str(arch_f.contents()) != ARCH:
        print(f"error: {args.src} is not a {ARCH} checkpoint", file=sys.stderr)
        return 1
    packed_f = reader.fields.get(f"{ARCH}.quant.int2_packed")
    if packed_f is None or int(packed_f.contents()) == 0:
        print(f"error: {args.src} is not int2-packed; nothing to transcode", file=sys.stderr)
        return 1

    cfg = read_config(reader)
    tensors = {t.name: t for t in reader.tensors}

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(args.dst), arch=ARCH)
    n_kv = copy_kv(reader, writer)
    print(f"{args.src.name}: {len(tensors)} tensors, {n_kv} keys copied "
          f"(dropped {ARCH}.quant.int2_packed) -> --weight-dtype {args.weight_dtype}")

    n_dense, n_copied, n_params = 0, 0, 0
    n_quant, n_demoted, n_bytes = 0, 0, 0
    for name, t in tensors.items():
        if name.endswith(".scale"):
            continue                              # folded into the weights below

        base = name[: -len(".weight")] if name.endswith(".weight") else None
        shape = bit_tensor_shape(base, cfg) if base else None
        if shape is None:
            # An I8 tensor the table does not know is a ternary weight this
            # script would copy through still packed, having already dropped its
            # scale -- a file that loads and is wrong. Stop instead.
            if int(t.tensor_type) == int(GGMLQuantizationType.I8):
                raise ValueError(f"{name}: packed I8 tensor with no entry in the shape table")
            # Already dense. The reader hands back raw bytes for bf16, and
            # add_tensor_info turns a uint8 byte shape into the logical shape
            # when raw_dtype says what the bytes mean.
            writer.add_tensor(name, t.data, raw_dtype=GGMLQuantizationType(int(t.tensor_type)))
            n_copied += 1
            continue

        if int(t.tensor_type) != int(GGMLQuantizationType.I8):
            raise ValueError(f"{name}: expected packed I8, got type {t.tensor_type}")
        N, K_packed, K_logical, split = shape

        scale_t = tensors.get(base + ".scale")
        if scale_t is None:
            raise ValueError(f"{name}: no sibling {base}.scale")
        scales = np.asarray(scale_t.data, dtype=np.float32).reshape(-1)
        want = 2 if split else 1
        if scales.size != want:
            raise ValueError(f"{base}.scale: {scales.size} values, expected {want}")

        codes = ladder_unpack_codes(t.data.view(np.uint8), N, K_packed)
        if K_logical != K_packed:
            codes = codes[:, :K_logical]          # drop the ladder's K padding
        if codes.min() == 0:
            raise ValueError(f"{name}: code 0 decodes to -2, outside the ternary set")

        halves = [(split[i], scales[i], codes[i * (N // want):(i + 1) * (N // want)])
                  for i in range(want)] if split else [(name, scales[0], codes)]

        for out_name, scale, blk in halves:
            if not out_name.endswith(".weight"):
                out_name += ".weight"
            data, dtype, note = encode(out_name, blk, scale, args.weight_dtype)
            if dtype is not GGMLQuantizationType.BF16:
                n_quant += 1
            elif args.weight_dtype != "bf16":
                n_demoted += 1
            # add_tensor_info turns a uint8 byte shape back into the logical
            # (N, K) when raw_dtype says how many weights a byte holds.
            writer.add_tensor(out_name, data, raw_shape=list(data.shape),
                              raw_dtype=dtype)
            n_dense += 1
            n_params += blk.size
            n_bytes += data.nbytes
            print(f"  {out_name:38s} {blk.shape[0]:6d} x {blk.shape[1]:5d}  "
                  f"scale {scale:.6g} -> {note}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    mib = args.dst.stat().st_size / (1024 * 1024)
    print(f"done. {args.dst} ({mib:.1f} MiB)")
    print(f"  {n_dense} BitLinear weights ({n_params / 1e6:.1f}M values) at "
          f"{n_bytes / max(n_params, 1):.4f} bytes/weight, {n_copied} copied through")
    if n_demoted:
        # Named rather than counted silently: a footprint that quietly comes in
        # above what the flavour promises is how this regresses.
        print(f"  {n_quant} in {args.weight_dtype}, {n_demoted} demoted to bf16 "
              f"(K not a multiple of 32 -- see the notes above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
