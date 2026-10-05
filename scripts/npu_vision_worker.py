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
"""Prototype: serve pi0's vision encoder on the Intel NPU (docs/backend/ptl.md).

    python scripts/npu_vision_worker.py --ir pi0_vision.xml --sock /tmp/pi0_npu.sock
    VLA_PI0_NPU_VISION=/tmp/pi0_npu.sock build-sycl/vla-server ...  # 2+ views

The IR is the vision graph ggml-openvino builds for pi0 (SigLIP + projector,
CHW float image in, [n_img_tokens, hidden] out). Export it once with the
OpenVINO build:

    GGML_OPENVINO_DEVICE=NPU GGML_OPENVINO_DUMP_IR=1 VLA_IMG_SIZE=224 \\
        build-ov/tests/vla_predict_check pi0.gguf "" 1     # writes IR_naive_0.xml/.bin

Protocol (native endianness), one request per connection: u32 n_floats +
float32 CHW image -> u32 n_floats + float32 embedding (0 on failure).
"""
import argparse
import os
import socket
import struct
import sys
import time

import numpy as np
import openvino as ov


def recv_all(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True, help="pi0 vision IR (.xml with its .bin)")
    ap.add_argument("--sock", default="/tmp/pi0_npu.sock")
    ap.add_argument("--device", default="NPU")
    args = ap.parse_args()

    core = ov.Core()
    model = core.read_model(args.ir)
    t0 = time.time()
    compiled = core.compile_model(model, args.device)
    req = compiled.create_infer_request()
    in_shape = [d.get_length() for d in model.inputs[0].get_partial_shape()]
    n_in = int(np.prod(in_shape))
    print(f"npu_vision_worker: {args.ir} on {args.device}, compiled in {time.time() - t0:.1f}s, "
          f"input {in_shape}", flush=True)

    if os.path.exists(args.sock):
        os.unlink(args.sock)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.sock)
    srv.listen(4)
    print(f"npu_vision_worker: listening on {args.sock}", flush=True)

    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                (n,) = struct.unpack("=I", recv_all(conn, 4))
                img = np.frombuffer(recv_all(conn, 4 * n), dtype=np.float32)
                if n != n_in:
                    raise ValueError(f"expected {n_in} floats, got {n}")
                out = req.infer({0: img.reshape(in_shape)})[compiled.outputs[0]]
                out = np.ascontiguousarray(out, dtype=np.float32).ravel()
                conn.sendall(struct.pack("=I", out.size) + out.tobytes())
            except Exception as e:  # report and keep serving
                print(f"npu_vision_worker: request failed: {e}", file=sys.stderr, flush=True)
                try:
                    conn.sendall(struct.pack("=I", 0))
                except OSError:
                    pass


if __name__ == "__main__":
    main()
