#!/usr/bin/env python3
"""Ling-3.0-tiny official-code reference fixture (Fixture A).

Runs the OFFICIAL modeling_bailing_moe_v3.py (transformers 4.57.x + fla CPU
shim) on the first N layers of the official BF16 snapshot with the real LM
head, teacher-forced over a frozen token-id prompt, and dumps:

  <out>/prompt_ids.json     the frozen input token ids
  <out>/hidden.npy          f32 hidden states after EVERY stage:
                            [n_stages, T, H] with stage 0 = embeddings,
                            stage i = output of layer i-1
  <out>/logits.npy          f32 logits at every position [T, V]
  <out>/router.json         per MoE layer/position: selected ids + weights
  <out>/meta.json           model revision, config digest, tolerances seed

Memory-bounded: only layers 0..N-1 + embedding + final norm + lm_head are
materialized (meta-init then partial state_dict load).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

# Repo-relative import of the fla CPU shim: resolve from THIS script's location
# inside the colibri checkout, so fixture generation is reproducible from any
# clone. Override only via LING3_SHIM_DIR when the shim lives elsewhere (the
# old hardcoded /data/ANVIL/ling3 broke every non-Dell checkout).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SHIM_DIR = os.environ.get("LING3_SHIM_DIR") or _HERE
sys.path.insert(0, _SHIM_DIR)
import ling3_flacpu_shim  # noqa: E402  (installs the fake `fla` modules)

ling3_flacpu_shim.install()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/ANVIL/ling3/hf_tiny",
                    help="official BF16 snapshot dir; every receipt must "
                         "record the exact explicit path + revision used")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ids", required=True, help="comma-separated token ids")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    sys.path.insert(0, args.model)
    # load the official files as a synthetic package so the intra-package
    # relative import (`from .configuration_bailing_moe_v3 ...`) resolves
    import importlib.util
    import types
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
    BailingMoeV3Config = modeling.BailingMoeV3Config
    BailingMoeV3ForCausalLM = modeling.BailingMoeV3ForCausalLM

    # CRITICAL (proven): the shipped modeling file has its causal-mask creation
    # COMMENTED OUT ("# tptest miss causal_mask = create_causal_mask(") and the
    # custom MLA call sites never build one, so attention runs BIDIRECTIONALLY
    # in every harness invocation regardless of the attention_mask argument.
    # True autoregressive semantics require causality -> wrap the eager kernel
    # to apply a strict causal trim itself. (Verified: without this, token 0's
    # context equals the UNGATED UNIFORM average over all positions.)
    _orig_eager = modeling.eager_attention_forward

    def _causal_eager(module, q, k, v, attention_mask=None, dropout=0.0, **kw):
        Tq, Tk = q.shape[2], k.shape[2]
        m = torch.full((Tq, Tk), torch.finfo(q.dtype).min, dtype=q.dtype,
                       device=q.device).triu(1)
        kw.pop("scaling", None)
        return _orig_eager(module, q, k, v, m[None, None],
                           dropout=dropout, scaling=module.scaling, **kw)

    modeling.eager_attention_forward = _causal_eager
    from safetensors import safe_open

    ids = [int(x) for x in args.ids.split(",")]
    T = len(ids)

    # full config, truncated depth
    cfg = BailingMoeV3Config.from_pretrained(args.model)
    cfg.num_hidden_layers = args.layers
    cfg.num_nextn_predict_layers = 0
    # CRITICAL: direct construction leaves _attn_implementation unset, in which
    # case Model.forward builds NO causal mask and the eager MLA path runs
    # BIDIRECTIONALLY (proven via hook analysis: token 0 attends all positions).
    # Force the eager implementation so a proper causal mask is created.
    cfg._attn_implementation = "eager"
    cfg._attn_implementation_autoset = False
    with torch.device("meta"):
        model = BailingMoeV3ForCausalLM(cfg)
    model = model.to_empty(device="cpu")
    model = model.to(getattr(torch, args.dtype))
    model.eval()

    idx = json.load(open(f"{args.model}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    want_prefixes = ("model.word_embeddings.", "model.norm.",
                     "lm_head.") + tuple(f"model.layers.{i}." for i in range(args.layers))
    loaded, missed_files = set(), {}
    tensors_by_file = {}
    for name in wm:
        if name.startswith(want_prefixes):
            tensors_by_file.setdefault(wm[name], []).append(name)
    for fname, names in tensors_by_file.items():
        with safe_open(f"{args.model}/{fname}", framework="pt", device="cpu") as f:
            for name in names:
                ten = f.get_tensor(name)
                sd_key = name
                try:
                    model.load_state_dict({sd_key: ten}, strict=False, assign=False)
                except Exception as e:
                    print("load fail", sd_key, e)
                    raise
                loaded.add(sd_key)
    # verify every non-expert parameter of the truncated model got real weights
    missing = []
    for k, p in model.named_parameters():
        if k not in loaded and p.numel() > 0:
            missing.append(k)
    if missing:
        print("MISSING WEIGHTS:", missing[:20])
        raise SystemExit(1)

    ids_t = torch.tensor([ids], dtype=torch.long)
    captured = {}

    # meta->to_empty leaves the non-persistent rotary inv_freq buffer
    # uninitialized: rebuild it exactly as BailingMoeV3RotaryEmbedding would
    # (default rope, head_dim=qk_rope_head_dim, partial factor forced to 1.0).
    rope = model.model.rotary_emb
    dim = cfg.qk_rope_head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    with torch.no_grad():
        rope.inv_freq.copy_(inv_freq)
    assert float(getattr(rope, "attention_scaling", 1.0)) == 1.0

    # capture the TRUE MoE inputs (post_attention_layernorm outputs) so routing
    # margins can be measured exactly on the reference side
    mlp_in = {}
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "gate"):
            def pre(mod, args, kwargs, _i=i):
                mlp_in[_i] = args[0].detach().float().clone()
            layer.mlp.register_forward_pre_hook(pre, with_kwargs=True)

    def gate_hook(layer_idx):
        def hook(mod, inputs, output):
            topk_idx, topk_weight, logits = output   # 2D [T, topk]
            x = mlp_in.get(layer_idx)
            margins = None
            if x is not None:
                xf = x[0].float()
                Wg = mod.weight.float()
                bg = mod.expert_bias.float() if hasattr(mod, "expert_bias") else None
                lo2 = xf @ Wg.T
                s2 = torch.sigmoid(lo2)
                rt2 = s2 + (bg if bg is not None else 0)
                Tn = rt2.shape[0]
                Eg = s2.shape[1] // mod.n_group
                gsv = rt2.view(Tn, mod.n_group, Eg).topk(2, dim=-1).values.sum(-1)
                gidx = torch.topk(gsv, k=mod.topk_group, dim=-1).indices
                msk = torch.zeros_like(gsv, dtype=torch.bool).scatter(1, gidx, True)
                smk = msk.unsqueeze(-1).expand(Tn, mod.n_group, Eg).reshape(Tn, -1)
                mk = rt2.masked_fill(~smk, float("-inf"))
                vals, _ = torch.topk(mk, k=min(mod.top_k + 3, int(smk.sum(-1)[0])), dim=-1)
                margins = (vals[:, mod.top_k - 1] - vals[:, mod.top_k]).tolist()
            captured.setdefault("router", []).append({
                "layer": layer_idx,
                "idx": topk_idx.to(torch.int32).tolist(),
                "w": [[round(float(x), 7) for x in row]
                      for row in topk_weight.float().tolist()],
                "margins": margins and [round(float(v), 7) for v in margins],
            })
        return hook

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.register_forward_hook(gate_hook(i))

    hs_out = {}
    # Explicit 4D additive causal mask: the direct-constructed model resolves
    # _attn_implementation=sdpa and the custom MLA call site passes whatever
    # attention_mask arrives; without an explicit mask both sdpa and eager run
    # BIDIRECTIONALLY (prefix-invariance test proven). This restores true
    # autoregressive semantics for every attention implementation.
    Tm = len(ids)
    causal4d = torch.zeros(1, 1, Tm, Tm)
    causal4d.masked_fill_(torch.triu(torch.ones(Tm, Tm, dtype=torch.bool), 1),
                          torch.finfo(torch.float32).min)
    with torch.no_grad():
        out = model(input_ids=ids_t, attention_mask=causal4d,
                    output_hidden_states=True, use_cache=False)
        hidden_states = [h[0].float() for h in out.hidden_states]   # T+1 stages incl embed
        logits = out.logits[0].float()

    import os
    os.makedirs(args.out, exist_ok=True)
    hs_np = torch.stack(hidden_states).numpy().astype(np.float32)
    lo_np = logits.numpy().astype(np.float32)
    np.save(f"{args.out}/hidden.npy", hs_np)
    hs_np.tofile(f"{args.out}/hidden.f32")       # raw companion for NATIVE checker
    np.save(f"{args.out}/logits.npy", lo_np)
    lo_np.tofile(f"{args.out}/logits.f32")
    json.dump(ids, open(f"{args.out}/prompt_ids.json", "w"))
    json.dump(captured["router"], open(f"{args.out}/router.json", "w"))
    import os as _os
    _os.makedirs(f"{args.out}/moe_in", exist_ok=True)
    for li, ten in mlp_in.items():
        ten[0].numpy().tofile(f"{args.out}/moe_in/x_mlp_L{li}.f32")
    cfg_digest = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:16]
    json.dump({"revision_source": "git-sha-of-hf_tiny-clone", "config_digest16": cfg_digest,
               "layers": args.layers, "T": T,
               "note": "official modeling code, fla CPU shim (naive recurrence)"},
              open(f"{args.out}/meta.json", "w"), indent=1)
    print("FIXTURE_A_OK", args.layers, "layers", T, "tokens",
          "hidden", tuple(torch.stack(hidden_states).shape),
          "logits", tuple(logits.shape))


if __name__ == "__main__":
    main()
