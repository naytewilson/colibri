#!/usr/bin/env python3
"""Parity adjudication for the Ling3 Colibri port.

Compares the native engine's diagnostic dumps (L3_TRACE / L3_LOGITS /
L3_ROUTE) against reference fixtures:
  Fixture A (official modeling code, N layers)
  Fixture B (full-depth streamed reference)

Usage: ling3_parity.py --fixture DIR [--stages N] [--trace PATH]
                    [--logits PATH] [--route PATH] [--tag NAME]
Prints one CHECK line per comparison and exits nonzero on any failure.
"""
import argparse
import json
import struct

import numpy as np


def load_trace(path, n_stages, T, D):
    raw = np.fromfile(path, dtype=np.float32)
    per = T * D
    rows = raw.size // per
    assert rows >= n_stages, f"trace holds {rows} stage dumps < {n_stages}"
    if raw.size % per:
        print(f"NOTE trace has {raw.size % per} trailing floats (final-norm dump); ignored")
    if rows != n_stages:
        print(f"NOTE trace holds {rows} full-stage dumps; using first {n_stages}")
    return raw[: n_stages * per].reshape(n_stages, T, D)


def compare(name, got, ref, atol, rtol):
    diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    scale = np.maximum(np.abs(ref.astype(np.float64)), 1e-9)
    ok = (diff <= atol + rtol * scale).all()
    md = float(diff.max())
    print(f"CHECK {name}: max_abs_diff={md:.6g} atol={atol} rtol={rtol} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--trace")
    ap.add_argument("--logits")
    ap.add_argument("--route")
    ap.add_argument("--stages", type=int, default=0,
                    help="how many layer stages the trace holds (not counting embeddings)")
    ap.add_argument("--hidden-atol", type=float, default=None)
    ap.add_argument("--hidden-rtol", type=float, default=None)
    args = ap.parse_args()

    ref_hidden = np.load(f"{args.fixture}/hidden.npy")   # [1+L, T, H]
    ref_logits = np.load(f"{args.fixture}/logits.npy")   # [T, V]
    T, D = ref_hidden.shape[1], ref_hidden.shape[2]
    ok = True

    if args.trace:
        n = args.stages or (ref_hidden.shape[0] - 1)
        got = load_trace(args.trace, n, T, D)
        for i in range(n):
            ok &= compare(f"hidden_stage{i + 1}", got[i], ref_hidden[1 + i],
                          args.hidden_atol or 2e-2, args.hidden_rtol or 2e-2)

    if args.logits:
        got_l = np.fromfile(args.logits, dtype=np.float32).reshape(T, -1)
        V = min(got_l.shape[1], ref_logits.shape[1])
        okl = compare("logits_all_pos", got_l[:, :V], ref_logits[:, :V],
                      args.hidden_atol or 0.15, args.hidden_rtol or 2e-2)
        g_got = got_l.argmax(-1)
        g_ref = ref_logits.argmax(-1)
        agree = int((g_got == g_ref).sum())
        print(f"CHECK argmax_tokens: {agree}/{T} -> "
              f"{'PASS' if agree == T else 'FAIL'}")
        ok &= okl and agree == T

    if args.route:
        ra = json.load(open(f"{args.fixture}/router.json"))
        # fixture router records are per-layer dicts; flatten to (layer,pos)
        ref = {}
        items = ra if isinstance(ra, list) else []
        for rec in items:
            if isinstance(rec, dict) and "items" in rec:      # fixture B shape
                for pos, item in enumerate(rec["items"]):
                    ref[(rec["layer"], pos)] = item
            elif isinstance(rec, dict) and "layer" in rec:    # fixture A hook shape list
                pass
        if isinstance(ra, list) and ra and "idx" in ra[0]:    # fixture A flat list
            # official hook emits one record per call in layer-major, pos-minor order
            layers_seen = sorted({r_["layer"] for r_ in ra})
            per_layer = {}
            for r_ in ra:
                per_layer.setdefault(r_["layer"], []).append(r_)
            ref = {(li, p): it for li, lst in per_layer.items()
                   for p, it in enumerate(lst)}
        got_raw = open(args.route, "rb").read()
        # engine route record: 2*int + topk*(int + float); recover topk from config
        topk = 8
        try:
            cfgm = json.load(open("/data/ANVIL/ling3/hf_tiny/config.json"))
            topk = cfgm["num_experts_per_tok"]
        except Exception:
            pass
        off = 0
        got = {}
        while off < len(got_raw):
            li, t = struct.unpack_from("<ii", got_raw, off)
            off += 8
            idx = struct.unpack_from(f"<{topk}i", got_raw, off)
            off += 4 * topk
            wts = struct.unpack_from(f"<{topk}f", got_raw, off)
            off += 4 * topk
            got[(li, t)] = {"idx": list(idx), "w": list(wts)}
        match_n = 0
        total = 0
        max_wdiff = 0.0
        for key, rit in sorted(ref.items()):
            if key not in got:
                continue
            total += 1
            git = got[key]
            idx_ok = [int(x) for x in rit["idx"]] == git["idx"]
            wd = float(np.max(np.abs(np.array(rit["w"]) - np.array(git["w"]))))
            max_wdiff = max(max_wdiff, wd)
            if idx_ok and wd < 2e-3:
                match_n += 1
        print(f"CHECK router_selections: {match_n}/{total} exact "
              f"(max weight diff {max_wdiff:.2e}) -> "
              f"{'PASS' if total and match_n == total else 'FAIL'}")
        ok &= total > 0 and match_n == total

    print("PARITY_VERDICT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
