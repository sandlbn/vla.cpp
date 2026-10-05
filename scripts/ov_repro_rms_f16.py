"""Minimal reproducer: an f16 overflow that OpenVINO 2026.3's RMSFusion exposes.

BitVLA's FFN computes gu = relu(g)^2 * u (here pre-scaled by 1/32 as vla.cpp
does on OpenVINO) and feeds it to a weighted RMSNorm. gu reaches ~1e5 - past
f16's 65504. On the GPU at f16:

  2026.2: the RMSNorm stays unfused and its reduction runs on the f32-only
          reduce_ref, which keeps the producer in f32 - correct, by accident.
  2026.3: RMSFusion turns it into rms_gpu_bfyx_opt__f16, the producer is now
          f16, gu overflows to inf and the norm returns NaN/garbage.

ACTIVATIONS_SCALE_FACTOR (OpenVINO's knob for f16 activation overflow) fixes it.

    python repro_rms_f16.py
"""
import numpy as np
import openvino as ov
import openvino.opset13 as ops

ROWS, K, EPS, GATE = 320, 6912, 1e-5, 1.0 / 32.0

g = ops.parameter([1, ROWS, K], np.float32, name="g")
u = ops.parameter([1, ROWS, K], np.float32, name="u")
r = ops.multiply(ops.relu(g), np.array([GATE], dtype=np.float32))
gu = ops.multiply(ops.multiply(r, r), u)                       # the FFN gate product
mean = ops.reduce_mean(ops.multiply(gu, gu), np.array([-1], dtype=np.int64), keep_dims=True)
inv = ops.divide(np.array([1.0], dtype=np.float32),
                 ops.sqrt(ops.add(mean, np.array([EPS * GATE ** 4], dtype=np.float32))))
gamma = np.random.default_rng(1).uniform(0.5, 1.5, K).astype(np.float32)
y = ops.multiply(ops.multiply(gu, inv), gamma.reshape(1, 1, K))  # weighted RMSNorm
model = ov.Model([y], [g, u], "ffn_gate_subnorm")

# Magnitudes as measured in BitVLA's LM (gate ReLU outputs up to ~300, up-proj
# values in the hundreds on a few channels).
rng = np.random.default_rng(0)
gd = (rng.standard_normal((1, ROWS, K)) * 20.0).astype(np.float32)
ud = (rng.standard_normal((1, ROWS, K)) * 20.0).astype(np.float32)
gd[..., :8] = 300.0
ud[..., :8] = 800.0
g64, u64 = gd.astype(np.float64), ud.astype(np.float64)
gu64 = (np.maximum(g64, 0) * GATE) ** 2 * u64
ref = gu64 / np.sqrt((gu64 * gu64).mean(-1, keepdims=True) + EPS * GATE ** 4) * gamma
print(f"openvino {ov.__version__}   max |gu| = {np.abs(gu64).max():.3g}  (f16 max 65504)")

core = ov.Core()
for name, cfg in [("GPU f32", {"INFERENCE_PRECISION_HINT": "f32"}),
                  ("GPU f16", {"INFERENCE_PRECISION_HINT": "f16"}),
                  ("GPU f16 + ACTIVATIONS_SCALE_FACTOR=64",
                   {"INFERENCE_PRECISION_HINT": "f16", "ACTIVATIONS_SCALE_FACTOR": "64"})]:
    cm = core.compile_model(model, "GPU", cfg)
    out = cm.create_infer_request().infer({0: gd, 1: ud})[cm.outputs[0]].astype(np.float64)
    kernels = sorted({n.get_rt_info()["primitiveType"].astype(str) for n in cm.get_runtime_model().get_ordered_ops()
                      if any(t in n.get_rt_info()["primitiveType"].astype(str) for t in ("rms", "reduce"))})
    bad = ~np.isfinite(out)
    err = np.max(np.abs(np.where(bad, 0, out) - ref)) / np.abs(ref).max()
    print(f"  {name:40s} max rel err {err:8.4f}  non-finite {int(bad.sum()):8d}  kernels {kernels}")
