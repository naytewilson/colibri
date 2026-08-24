#!/usr/bin/env python3
"""Dump OFFICIAL per-layer attention inputs for Ling-3.0-tiny (XLDIR companion).

Runs the official modeling code (transformers 4.57.x + fla CPU shim) over the
frozen prompt with an explicit 4D additive causal mask (the direct-constructed
model resolves sdpa/eager and runs BIDIRECTIONALLY without it — proven via
prefix-invariance), capturing each DecoderLayer's input tensor:

    <xl-dir>/xin_L<i>.f32   f32 [T, H] attention-stage input of layer i

Together with <fixture>/moe_in/x_mlp_L<i>.f32 (written by
ling3_refgen_official.py) this gives the engine BIT-IDENTICAL inputs to every
sublayer, which is what makes first-N-layer parity adjudication immune to
routing-chaos amplification.
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
    ap.add_argument("--model", default="/data/ANVIL/ling3/hf_tiny",
                    help="official BF16 snapshot dir; record path + revision "
                         "in every receipt")
    ap.add_argument("--xl-dir", required=True)
    ap.add_argument("--ids", required=True,
                    help="comma-separated token ids (same frozen prompt as "
                         "the refgen fixture)")
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x != ""]
    sys.path.insert(0, args.model)
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

    os.makedirs(args.xl_dir, exist_ok=True)
    caps = {}

    def pre(i):
        def h(mod, fargs, fkwargs):
            caps[i] = fargs[0].detach().float().clone()
        return h

    for i, lay in enumerate(model.model.layers):
        lay.register_forward_pre_hook(pre(i), with_kwargs=True)

    with torch.no_grad():
        model(input_ids=torch.tensor([ids]), attention_mask=causal4d,
              output_hidden_states=True, use_cache=False)

    for i, ten in caps.items():
        ten[0].numpy().tofile(f"{args.xl_dir}/xin_L{i}.f32")
    print("XLDIR_OK", sorted(caps), "rows", Tm)


if __name__ == "__main__":
    main()
