#!/usr/bin/env python3
"""Read an OpenVINO *runtime* model dump and say which FullyConnected nodes got a
DynamicQuantize in front of them -- and, for the ones that did not, what they are
fed by instead.

This exists because job 372927 measured the defect without being able to explain
it: with GGML_OPENVINO_ACT_QUANT_ELIDE=1 the act_quant chain goes away as
intended, but 104 of 340 compressed FullyConnected nodes lose their
DynamicQuantize at the same time, so 31% of the BitLinears run activations
quantised by nobody. The node-type inventory that found this is an aggregate: it
counts DynamicQuantize nodes, not which FC each one belongs to. The runtime model
is the per-node view.

Input is what `GGML_OPENVINO_DUMP_RUNTIME=1` writes (utils.cpp,
vla_dump_runtime_model): one numbered runtime_model_N.xml per compiled graph.
That is get_runtime_model(), i.e. the graph AFTER the plugin's transformations --
not GGML_OPENVINO_DUMP_IR, which serialises the frontend's output and therefore
predates every decision this asks about.

Usage:
    ov_runtime_fc_inputs.py runtime_model_*.xml [--per-node] [--top N]

Reported per FC node, walking back through nodes that are pure plumbing
(Reshape/Squeeze/Unsqueeze/Convert/Transpose and any not-executed node), because
the plugin leaves those between a producer and its consumer and they are not what
decides whether dynamic quantisation fired.
"""

import argparse
import collections
import glob
import os
import sys
import xml.etree.ElementTree as ET

# Walked through when looking for the "real" producer of an FC's activations.
# Not-executed nodes are skipped as well, regardless of type -- they were fused
# or folded and do not exist at runtime.
PLUMBING = {
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Transpose",
    "Convert",
    "Broadcast",
    "ShapeOf",
}

FC_TYPES = ("FullyConnected", "MatMul", "Gemm")


def load(path):
    """-> (layers, producers) where producers[(layer_id, in_port)] = layer_id."""
    root = ET.parse(path).getroot()
    layers = {}
    for ly in root.iter("layer"):
        data = ly.find("data")
        attrs = dict(data.attrib) if data is not None else {}
        out_shapes = []
        outp = ly.find("output")
        if outp is not None:
            for port in outp.findall("port"):
                out_shapes.append("x".join(d.text for d in port.findall("dim")))
        layers[ly.get("id")] = {
            "name": ly.get("name", ""),
            "type": ly.get("type", ""),
            "attrs": attrs,
            "out": out_shapes,
        }
    producers = {}
    for e in root.iter("edge"):
        producers[(e.get("to-layer"), e.get("to-port"))] = e.get("from-layer")
    return layers, producers


def executed(layer):
    t = layer["attrs"].get("execTimeMcs", "")
    return t != "not_executed"


def real_producer(layers, producers, lid, port="0"):
    """Walk back past plumbing and not-executed nodes. Returns a layer dict."""
    seen = set()
    cur = producers.get((lid, port))
    while cur is not None and cur not in seen:
        seen.add(cur)
        ly = layers.get(cur)
        if ly is None:
            return None
        if ly["type"] in PLUMBING or not executed(ly):
            nxt = producers.get((cur, "0"))
            if nxt is None:
                return ly
            cur = nxt
            continue
        return ly
    return layers.get(cur) if cur else None


def describe(layer):
    if layer is None:
        return "<no producer>"
    a = layer["attrs"]
    return "%s [%s]" % (layer["type"], a.get("runtimePrecision", a.get("outputPrecisions", "?")))


def analyse(paths, per_node=False, top=12):
    tot_fc = 0
    tot_dq = 0
    with_dq = collections.Counter()
    without_dq = collections.Counter()
    per_file = []

    for path in paths:
        try:
            layers, producers = load(path)
        except ET.ParseError as exc:
            print("  %-28s PARSE ERROR: %s" % (os.path.basename(path), exc))
            continue

        fcs = [(i, l) for i, l in layers.items()
               if any(k in l["type"] for k in FC_TYPES) and executed(l)]
        dqs = [l for l in layers.values()
               if "DynamicQuantize" in l["type"] and executed(l)]

        f_with = f_without = 0
        for lid, ly in fcs:
            src = real_producer(layers, producers, lid, "0")
            key = describe(src)
            if src is not None and "DynamicQuantize" in src["type"]:
                with_dq[ly["type"]] += 1
                f_with += 1
            else:
                without_dq[(ly["type"], key)] += 1
                f_without += 1
                if per_node:
                    print("    no-DQ  %-26s <- %-34s  %s"
                          % (ly["type"], key, ly["name"][:70]))

        tot_fc += len(fcs)
        tot_dq += len(dqs)
        per_file.append((os.path.basename(path), len(layers), len(fcs), len(dqs),
                         f_with, f_without))

    print()
    print("  %-28s %7s %7s %7s %8s %9s" %
          ("file", "layers", "FC", "DQ", "FC+DQ", "FC no-DQ"))
    for row in per_file:
        print("  %-28s %7d %7d %7d %8d %9d" % row)
    print("  %-28s %7s %7d %7d %8d %9d" %
          ("TOTAL", "-", tot_fc, tot_dq, sum(with_dq.values()),
           sum(without_dq.values())))

    if without_dq:
        print()
        print("  FullyConnected nodes with NO DynamicQuantize on their activation")
        print("  input, by what actually feeds them:")
        for (fct, src), n in without_dq.most_common(top):
            print("    %5d  %-30s <- %s" % (n, fct, src))
    if with_dq:
        print()
        print("  ...and the ones that kept it, by FC type:")
        for fct, n in with_dq.most_common(top):
            print("    %5d  %s" % (n, fct))
    return 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", nargs="+")
    ap.add_argument("--per-node", action="store_true",
                    help="print one line per FC that lost its DynamicQuantize")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args(argv)

    paths = []
    for p in args.xml:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("no runtime model XML found", file=sys.stderr)
        return 1
    return analyse(paths, per_node=args.per_node, top=args.top)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
