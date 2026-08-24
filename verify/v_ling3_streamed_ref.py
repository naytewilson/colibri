#!/usr/bin/env python3
"""VERIFIER TOOL (ling3-wave1): streamed full-depth official reference.

Runs the OFFICIAL modeling code over all 24 layers WITHOUT holding the whole
BF16 snapshot resident: per decoder layer, only that layer's tensors are read
from safetensors (lazy per-tensor), used, and released before advancing.

Chain semantics identical to tools/ling3_refgen_official.py: explicit 4D
additive causal mask, eager attention wrapped for strict causality, fla CPU
shim, fp32.

Outputs <out>/streamed_ref.npz containing per layer i (0..23):
    attn_out_L{i}  [T,H]   attention block output (pre-residual)
    mlp_out_L{i}   [T,H]   MLP/MoE block output (pre-residual)
    out_L{i}       [T,H]   residual-stream output of layer i
plus finln_out [T,H], logits [T,V], router.json, meta.json.
"""
import argparse
import gc
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/ANVIL/ling3/hf_tiny")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--int4-model",
                    help="if set, routed-expert weights are replaced by "
                         "independent int4-g32 dequantization of this "
                         "official INT4 snapshot (Reference B)")
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x != ""]
    T = len(ids)
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
    NL = cfg.num_hidden_layers
    cfg.num_nextn_predict_layers = 0
    cfg._attn_implementation = "eager"
    cfg._attn_implementation_autoset = False

    from safetensors import safe_open
    wm = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]

    # streaming build: only GLOBAL tensors + one decoder layer resident at a
    # time (the 15 GB snapshot is NEVER materialized whole)
    emb = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    finln_mod = modeling.BailingMoeV3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def load_tensor(name, module):
        fn = wm[name]
        with safe_open(f"{args.model}/{fn}", framework="pt", device="cpu") as st:
            t = st.get_tensor(name)
        with torch.no_grad():
            if isinstance(module, torch.nn.Embedding) or \
                    isinstance(module, torch.nn.Linear):
                module.weight.copy_(t)
            else:
                module.weight.copy_(t)

    load_tensor("model.word_embeddings.weight", emb)
    load_tensor("model.norm.weight", finln_mod)
    load_tensor("lm_head.weight", head)

    rope = modeling.BailingMoeV3RotaryEmbedding(cfg)
    inv = 1.0 / (cfg.rope_theta ** (
        torch.arange(0, cfg.qk_rope_head_dim, 2, dtype=torch.float32)
        / cfg.qk_rope_head_dim))
    with torch.no_grad():
        rope.inv_freq.copy_(inv)

    def load_into(module, prefix):
        byfile = {}
        pairs = [(n[len(prefix):], n) for n in wm if n.startswith(prefix)]
        if args.int4_model:
            from v_ling3_int4_dequant import iter_int4_expert_tensors
            full_names = [n for _, n in pairs]
            exp_full = {n for n in full_names
                        if ".mlp.experts." in n and n.endswith(".weight")}
            sd = {}
            for name, w in iter_int4_expert_tensors(args.int4_model,
                                                    sorted(exp_full)):
                sd[name[len(prefix):]] = w
            module.load_state_dict(sd, strict=False)
            pairs = [(s, n) for s, n in pairs if n not in exp_full]
        for short, n in pairs:
            byfile.setdefault(wm[n], []).append((short, n))
        for fn, items in sorted(byfile.items()):
            with safe_open(f"{args.model}/{fn}", framework="pt", device="cpu") as st:
                sd = {short: st.get_tensor(n) for short, n in items}
            module.load_state_dict(sd, strict=False)

    # capture buffers
    caps = {}

    def mk_hooks(i):
        def attn_h(mod, fargs, fkwargs, out):
            t = out[0] if isinstance(out, tuple) else out
            caps[f"attn_out_L{i}"] = t.detach().float().clone()
        def mlp_h(mod, fargs, fkwargs, out):
            t = out[0] if isinstance(out, tuple) else out
            caps[f"mlp_out_L{i}"] = t.detach().float().clone()
        return attn_h, mlp_h

    hs = {}
    causal4d = torch.zeros(1, 1, T, T)
    causal4d.masked_fill_(torch.triu(torch.ones(T, T, dtype=torch.bool), 1),
                          torch.finfo(torch.float32).min)
    ids_t = torch.tensor([ids])

    with torch.no_grad():
        hidden = emb(ids_t)[0].detach().float().clone()
        hs["embeddings"] = hidden.numpy().astype(np.float32)

        pos_ids = torch.arange(T).unsqueeze(0)
        pos_emb = rope(hidden, pos_ids)

        for i in range(NL):
            # fresh single-layer module per stage: dropping it releases the
            # layer's tensors (streaming contract; never hold two layers)
            prefix = f"model.layers.{i}."
            layer = modeling.BailingMoeV3DecoderLayer(cfg, layer_idx=i)
            layer.eval()
            ah, mh = mk_hooks(i)
            h1 = layer.attention.register_forward_hook(ah, with_kwargs=True)
            h2 = layer.mlp.register_forward_hook(mh, with_kwargs=True)
            load_into(layer, prefix)
            with torch.no_grad():
                out = layer(hidden.unsqueeze(0),
                            attention_mask=causal4d,
                            position_ids=pos_ids,
                            position_embeddings=pos_emb,
                            use_cache=False)
                hidden = out[0][0].detach().float().clone()
            h1.remove()
            h2.remove()
            for k in list(caps):
                if k.endswith(f"_L{i}"):
                    hs[k] = caps.pop(k).numpy().astype(np.float32)
            hs[f"out_L{i}"] = hidden.numpy().astype(np.float32)
            print(f"layer {i}: |out|max {np.abs(hs[f'out_L{i}']).max():.4f}",
                  flush=True)
            del layer, out, h1, h2
            gc.collect()

        # final norm + head
        finln = finln_mod(hidden.unsqueeze(0))[0].detach().float()
        hs["finln_out"] = finln.numpy().astype(np.float32)
        logits = head(finln.unsqueeze(0))[0].detach().float()

    os.makedirs(args.out, exist_ok=True)
    np.savez(f"{args.out}/streamed_ref.npz", **hs)
    lo = logits.numpy().astype(np.float32)
    lo.tofile(f"{args.out}/logits.f32")
    np.save(f"{args.out}/logits.npy", lo.reshape(T, -1))
    json.dump({"layers": NL, "T": T,
               "note": "streamed per-layer official reference, fp32, "
                       "explicit causal mask"},
              open(f"{args.out}/meta.json", "w"), indent=1)
    print("STREAMED_REF_OK layers", NL, "tokens", T)


if __name__ == "__main__":
    main()
