#!/usr/bin/env python3
"""VERIFIER TOOL (ling3-wave1 independent verification).

Captures OFFICIAL-reference sublayer outputs for Ling-3.0-tiny under the same
conditions as tools/ling3_refgen_official.py (explicit causal semantics):

    <out>/attn_out_L<i>.f32   f32 [T,H]  decoder layer i attention output
                                          (o_proj / KDA out_proj result,
                                           BEFORE residual add)
    <out>/mlp_out_L<i>.f32    f32 [T,H]  MLP / sparse-MoE block output
                                          (BEFORE residual add)
    <out>/finln_out.f32       f32 [T,H]  model.norm output (post-final-norm)

Complements ling3_refgen_official.py (layer inputs + hidden chain + logits +
router) so the verifier can adjudicate every sublayer boundary without
modifying candidate runtime source.
"""
import argparse
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
    ap.add_argument("--layers", type=int, default=24)
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

    cfg = modeling.BailingMoeV3Config.from_pretrained(args.model)
    cfg.num_hidden_layers = args.layers
    cfg.num_nextn_predict_layers = 0
    cfg._attn_implementation = "eager"
    cfg._attn_implementation_autoset = False
    with torch.device("meta"):
        model = modeling.BailingMoeV3ForCausalLM(cfg)
    model = model.to_empty(device="cpu").to(torch.float32)
    model.eval()

    from safetensors import safe_open
    idx = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
    want = ("model.word_embeddings.", "model.norm.", "lm_head.") + tuple(
        f"model.layers.{i}." for i in range(args.layers))
    byfile = {}
    for n in idx:
        if n.startswith(want):
            byfile.setdefault(idx[n], []).append(n)
    for fn, names in byfile.items():
        with safe_open(f"{args.model}/{fn}", framework="pt", device="cpu") as f:
            for n in names:
                model.load_state_dict({n: f.get_tensor(n)}, strict=False)

    rope = model.model.rotary_emb
    inv = 1.0 / (cfg.rope_theta ** (
        torch.arange(0, cfg.qk_rope_head_dim, 2, dtype=torch.float32)
        / cfg.qk_rope_head_dim))
    with torch.no_grad():
        rope.inv_freq.copy_(inv)

    Tm = len(ids)
    causal4d = torch.zeros(1, 1, Tm, Tm)
    causal4d.masked_fill_(
        torch.triu(torch.ones(Tm, Tm, dtype=torch.bool), 1),
        torch.finfo(torch.float32).min)

    os.makedirs(args.out, exist_ok=True)
    caps = {}

    def attn_hook(i):
        def h(mod, fargs, fkwargs, out):
            t = out[0] if isinstance(out, tuple) else out
            caps[f"attn_out_L{i}"] = t.detach().float().clone()
        return h

    def mlp_hook(i):
        def h(mod, fargs, fkwargs, out):
            t = out[0] if isinstance(out, tuple) else out
            caps[f"mlp_out_L{i}"] = t.detach().float().clone()
        return h

    def finln_hook(mod, fargs, fkwargs, out):
        t = out[0] if isinstance(out, tuple) else out
        caps["finln_out"] = t.detach().float().clone()

    for i, lay in enumerate(model.model.layers):
        lay.attention.register_forward_hook(attn_hook(i), with_kwargs=True)
        lay.mlp.register_forward_hook(mlp_hook(i), with_kwargs=True)
    model.model.norm.register_forward_hook(finln_hook, with_kwargs=True)

    with torch.no_grad():
        model(input_ids=torch.tensor([ids]), attention_mask=causal4d,
              use_cache=False)

    for k, ten in sorted(caps.items()):
        ten[0].numpy().tofile(f"{args.out}/{k}.f32")
    print("REFHOOKS_OK", sorted(caps), "rows", Tm)


if __name__ == "__main__":
    main()
