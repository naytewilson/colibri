#!/usr/bin/env python3
"""VERIFIER TOOL (ling3-wave1 independent verification).

Per-stage parity comparator between native engine diagnostic dumps and the
official-reference dumps. Produces a machine-readable JSON summary plus a
human table; never passes/fails silently (every number it prints is computed
from both sides).

Comparisons available:
  --trace   engine per-layer hidden chain [L*T*D (+ trailing D)] vs fixture
            hidden.npy stages 1..L (chained mode) — or, when --teacher-forced,
            engine layer outputs under reference-injected inputs.
  --att     engine att_L{i}.f32 vs refhooks attn_out_L{i}.f32
  --mlp     engine mlp_L{i}.f32 vs refhooks mlp_out_L{i}.f32
  --logits  engine logits dump vs fixture logits.npy
"""
import argparse
import json

import numpy as np


def load_f32(path):
    return np.fromfile(path, dtype=np.float32)


def stats(got, ref):
    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    return float(d.max()), float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, help="refgen output dir")
    ap.add_argument("--refhooks", help="v_ling3_hookdump output dir")
    ap.add_argument("--engine", required=True, help="engine dump dir")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--D", type=int, default=1536)
    ap.add_argument("--mode", choices=["chained", "teacher-forced"],
                    default="chained")
    ap.add_argument("--json-out", help="write summary JSON here")
    args = ap.parse_args()

    ref_hidden = np.load(f"{args.fixture}/hidden.npy")  # [1+L, T, H]
    T, D = args.T, args.D
    summary = {"mode": args.mode, "layers": args.layers, "T": T, "D": D,
               "stages": {}}

    trace = load_f32(f"{args.engine}/trace.f32")
    per = T * D
    n_st = trace.size // per
    print(f"engine trace: {n_st} full stage dumps + {trace.size % per} trailing")

    print("\n== hidden stage map (max_abs / mean_abs) ==")
    print(f"{'stage':>6} {'layer':>5} {'max':>12} {'mean':>12}")
    worst = {}
    for s in range(min(args.layers, n_st)):
        got = trace[s * per:(s + 1) * per].reshape(T, D)
        ref = ref_hidden[s + 1]          # [T,D]  (npy is [stages,T,D], no batch)
        mx, mn = stats(got, ref)
        tag = f"L{s}" if s > 0 else "L0"
        worst[tag] = {"max": mx, "mean": mn}
        print(f"{s:>6} {tag:>5} {mx:12.6g} {mn:12.6g}")
    if trace.size % per == D:
        fin = trace[n_st * per:]            # engine: last-position row only
        if args.refhooks:
            ref_fin = load_f32(f"{args.refhooks}/finln_out.f32").reshape(T, D)
            mx, mn = stats(fin.reshape(1, D), ref_fin[T - 1:T, :])
            worst["post_final_norm_lastpos"] = {"max": mx, "mean": mn}
            print(f"   post-final-norm: engine last-position row vs reference"
                  f" last position: max {mx:.6g} mean {mn:.6g}")
    summary["hidden"] = worst

    if args.refhooks:
        print("\n== sublayer deltas (teacher-forced: identical inputs) ==")
        sub = {}
        for i in range(args.layers):
            ea = load_f32(f"{args.engine}/att_L{i}.f32").reshape(T, D)
            ra = load_f32(f"{args.refhooks}/attn_out_L{i}.f32").reshape(T, D)
            amx, amn = stats(ea, ra)
            em = load_f32(f"{args.engine}/mlp_L{i}.f32").reshape(T, D)
            rm = load_f32(f"{args.refhooks}/mlp_out_L{i}.f32").reshape(T, D)
            mmx, mmn = stats(em, rm)
            sub[f"L{i}"] = {"attn_max": amx, "attn_mean": amn,
                            "mlp_max": mmx, "mlp_mean": mmn}
            print(f"L{i}: attn max {amx:.6g} mean {amn:.6g} | "
                  f"mlp max {mmx:.6g} mean {mmn:.6g}")
        summary["sublayers"] = sub

    print("\n== logits ==")
    eng_lo = load_f32(f"{args.engine}/logits.f32")
    ref_lo = np.load(f"{args.fixture}/logits.npy").reshape(-1, eng_lo.size // T)
    V = min(eng_lo.size // T, ref_lo.shape[1])
    eng_lo = eng_lo.reshape(T, -1)[:, :V]
    ref_lo = ref_lo[:, :V]
    lmx, lmn = stats(eng_lo, ref_lo)
    ag = int((eng_lo.argmax(1) == ref_lo.argmax(1)).sum())
    print(f"logits max {lmx:.6g} mean {lmn:.6g} argmax {ag}/{T}")
    summary["logits"] = {"max": lmx, "mean": lmn, "argmax_agree": ag,
                         "argmax_total": T}

    if args.json_out:
        json.dump(summary, open(args.json_out, "w"), indent=1)
        print("summary ->", args.json_out)


if __name__ == "__main__":
    main()
