#!/usr/bin/env python3
"""VERIFIER TOOL (ling3-wave1): Reference B generator — official code path with
ROUTED-EXPERT WEIGHTS REPLACED by independent INT4-dequantized values from the
official INT4 checkpoint.

Everything else (embeddings, KDA/MLA attention, router gate + bias, shared
expert, dense MLP, norms, LM head) stays the OFFICIAL BF16 snapshot — exactly
matching what the native engine consumes (only routed experts are quantized in
the official artifact).

Dumps the same artifacts as tools/ling3_refgen_official.py so ling3_check and
the verifier comparator can adjudicate engine-vs-BF16ref against
engine-vs-INT4DEQREF on identical teacher-forced inputs.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("LING3_SHIM_DIR") or _HERE)
import ling3_flacpu_shim  # noqa: E402

ling3_flacpu_shim.install()

sys.path.insert(0, _HERE)
from v_ling3_int4_dequant import (  # noqa: E402
    iter_int4_expert_tensors,
    load_tensor_index,
    unpack_i32Nibbles,  # noqa: F401  (re-exported for row-check callers)
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/ANVIL/ling3/hf_tiny",
                    help="official BF16 snapshot dir")
    ap.add_argument("--int4-model", default="/data/ANVIL/ling3/hf_int4",
                    help="official INT4 snapshot dir (routed experts source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--row-check", type=int, default=6,
                    help="expert rows cross-checked against official BF16")
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x != ""]
    pkg = types.ModuleType("hfmodel")
    pkg.__path__ = [args.model]
    sys.modules["hfmodel"] = pkg

    def _load(modname):
        spec = importlib.util.spec_from_file_location(
            f"hfmodel.{modname}", f"{args.model}/{modname}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"hfmodel.{modname}"] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("configuration_bailing_moe_v3")
    modeling = _load("modeling_bailing_moe_v3")

    _orig_eager = modeling.eager_attention_forward

    def _causal_eager(module, q, k, v, attention_mask=None, dropout=0.0, **kw):
        Tq, Tk = q.shape[2], k.shape[2]
        m = torch.full((Tq, Tk), torch.finfo(q.dtype).min, dtype=q.dtype,
                       device=q.device).triu(1)
        kw.pop("scaling", None)
        return _orig_eager(module, q, k, v, m[None, None],
                           dropout=dropout, scaling=module.scaling, **kw)

    modeling.eager_attention_forward = _causal_eager

    cfg = modeling.BailingMoeV3Config.from_pretrained(args.model)
    cfg.num_hidden_layers = args.layers
    cfg.num_nextn_predict_layers = 0
    cfg._attn_implementation = "eager"
    cfg._attn_implementation_autoset = False
    with torch.device("meta"):
        model = modeling.BailingMoeV3ForCausalLM(cfg)
    model = model.to_empty(device="cpu").to(torch.float32)
    model.eval()

    rope = model.model.rotary_emb
    inv = 1.0 / (cfg.rope_theta ** (
        torch.arange(0, cfg.qk_rope_head_dim, 2, dtype=torch.float32)
        / cfg.qk_rope_head_dim))
    with torch.no_grad():
        rope.inv_freq.copy_(inv)

    from safetensors import safe_open
    wm_bf = load_tensor_index(args.model)
    is_expert = lambda n: ".mlp.experts." in n
    want = ("model.word_embeddings.", "model.norm.", "lm_head.") + tuple(
        f"model.layers.{i}." for i in range(args.layers))
    nonexpert = [n for n in wm_bf if n.startswith(want) and not is_expert(n)]
    expert_names = [n for n in wm_bf if n.startswith(want) and is_expert(n)
                    and n.endswith(".weight")]
    byfile = {}
    for n in nonexpert:
        byfile.setdefault(wm_bf[n], []).append(n)
    loaded = set()
    cosines = []
    import random
    rnd = random.Random(20260824)
    rowcheck_targets = sorted(rnd.sample(
        [n[:-len(".weight")] for n in expert_names],
        min(args.row_check, len(expert_names)))) if args.row_check else []

    for fn, names in sorted(byfile.items()):
        with safe_open(f"{args.model}/{fn}", framework="pt", device="cpu") as st:
            for n in names:
                model.load_state_dict({n: st.get_tensor(n)}, strict=False)
                loaded.add(n)

    # routed experts: independent dequantization from the OFFICIAL INT4 files
    nexp = 0
    for name, w in iter_int4_expert_tensors(args.int4_model, expert_names):
        if name not in wm_bf or not name.startswith(want):
            continue
        model.load_state_dict({name: w}, strict=False)
        loaded.add(name)
        nexp += 1

    missing = [k for k, p in model.named_parameters()
               if k not in loaded and p.numel() > 0]
    if missing:
        print("MISSING WEIGHTS:", missing[:20])
        raise SystemExit(1)

    # ---- row-level sanity: dequant rows vs official BF16 rows -------------
    if rowcheck_targets:
        print("== independent-dequant row checks vs official BF16 ==")
        for base in rowcheck_targets:
            fnb = wm_bf[f"{base}.weight"]
            with safe_open(f"{args.model}/{fnb}", framework="pt", device="cpu") as st:
                wbf = st.get_tensor(f"{base}.weight").float().numpy()
            for nm, w in iter_int4_expert_tensors(
                    args.int4_model, [f"{base}.weight"]):
                wdq = w.numpy()
            r = rnd.randrange(wdq.shape[0])
            a, b = wdq[r], wbf[r]
            cos = float(np.dot(a, b) /
                        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
            rel = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))
            cosines.append(cos)
            print(f"  {base} row{r}: cos={cos:.6f} maxrel={rel:.4f}")

    Tm = len(ids)
    causal4d = torch.zeros(1, 1, Tm, Tm)
    causal4d.masked_fill_(torch.triu(torch.ones(Tm, Tm, dtype=torch.bool), 1),
                          torch.finfo(torch.float32).min)

    captured = {}
    mlp_in = {}
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "gate"):
            def pre(mod, fargs, fkwargs, _i=i):
                mlp_in[_i] = fargs[0].detach().float().clone()
            layer.mlp.register_forward_pre_hook(pre, with_kwargs=True)

    def gate_hook(layer_idx):
        def h(mod, inputs, fkwargs, output):
            topk_idx, topk_weight, logits = output
            captured.setdefault("router", []).append({
                "layer": layer_idx,
                "idx": topk_idx.to(torch.int32).tolist(),
                "w": [[round(float(x), 7) for x in row]
                      for row in topk_weight.float().tolist()],
            })
        return h

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.register_forward_hook(gate_hook(i), with_kwargs=True)

    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), attention_mask=causal4d,
                    output_hidden_states=True, use_cache=False)
        hidden_states = [h[0].float() for h in out.hidden_states]
        logits = out.logits[0].float()

    os.makedirs(args.out, exist_ok=True)
    hs_np = torch.stack(hidden_states).numpy().astype(np.float32)
    lo_np = logits.numpy().astype(np.float32)
    np.save(f"{args.out}/hidden.npy", hs_np)
    hs_np.tofile(f"{args.out}/hidden.f32")
    np.save(f"{args.out}/logits.npy", lo_np)
    lo_np.tofile(f"{args.out}/logits.f32")
    json.dump(ids, open(f"{args.out}/prompt_ids.json", "w"))
    json.dump(captured.get("router", []), open(f"{args.out}/router.json", "w"))
    os.makedirs(f"{args.out}/moe_in", exist_ok=True)
    for li, ten in mlp_in.items():
        ten[0].numpy().tofile(f"{args.out}/moe_in/x_mlp_L{li}.f32")
    cfg_digest = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True)
                                .encode()).hexdigest()[:16]
    json.dump({"config_digest16": cfg_digest, "layers": args.layers, "T": Tm,
               "note": "OFFICIAL code; routed experts = INDEPENDENT int4-g32 "
                       "dequant of official INT4 checkpoint",
               "dequant_row_cosine_min": min(cosines) if cosines else None},
              open(f"{args.out}/meta.json", "w"), indent=1)
    print("FIXTURE_INT4DEQ_OK", args.layers, "layers", Tm, "tokens",
          "experts_dequantized", nexp)


if __name__ == "__main__":
    main()
