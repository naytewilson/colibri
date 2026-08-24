#!/usr/bin/env python3
"""Capture official layer-0 KDA intermediates via forward hooks."""
import importlib.util
import json
import sys
import types

import numpy as np
import torch

sys.path.insert(0, "/data/ANVIL/ling3")
import ling3_flacpu_shim

ling3_flacpu_shim.install()

M = "/data/ANVIL/ling3/hf_tiny"
pkg = types.ModuleType("hfmodel")
pkg.__path__ = [M]
sys.modules["hfmodel"] = pkg


def _load(modname):
    spec = importlib.util.spec_from_file_location(f"hfmodel.{modname}", f"{M}/{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"hfmodel.{modname}"] = mod
    spec.loader.exec_module(mod)
    return mod


_load("configuration_bailing_moe_v3")
modeling = _load("modeling_bailing_moe_v3")

cfg = modeling.BailingMoeV3Config.from_pretrained(M)
cfg.num_hidden_layers = 4
cfg.num_nextn_predict_layers = 0
with torch.device("meta"):
    model = modeling.BailingMoeV3ForCausalLM(cfg)
model = model.to_empty(device="cpu").to(torch.float32)
model.eval()

from safetensors import safe_open
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
want = ("model.word_embeddings.", "model.norm.", "lm_head.") + tuple(
    f"model.layers.{i}." for i in range(4))
loaded = set()
byfile = {}
for name in idx:
    if name.startswith(want):
        byfile.setdefault(idx[name], []).append(name)
for fname, names in byfile.items():
    with safe_open(f"{M}/{fname}", framework="pt", device="cpu") as f:
        for name in names:
            model.load_state_dict({name: f.get_tensor(name)}, strict=False)
            loaded.add(name)
missing = [k for k, p in model.named_parameters() if p.numel() and k not in loaded]
assert not missing, missing[:10]

rope = model.model.rotary_emb
inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.qk_rope_head_dim, 2,
                                              dtype=torch.float32) / cfg.qk_rope_head_dim))
with torch.no_grad():
    rope.inv_freq.copy_(inv)

ids = [151644, 8948, 198, 2610, 525, 264, 10950, 17871, 13, 151645, 198,
       151644, 872, 198, 9707, 11, 151645, 198, 151644, 77091, 198]
cap = {}
att0 = model.model.layers[0].attention


def mk(tag):
    def h(mod, args, kwargs, out):
        ins = []
        for a in args[:2]:
            if torch.is_tensor(a):
                ins.append(a.detach().float().clone())
        cap[tag] = (ins, out.detach().float().clone() if torch.is_tensor(out) else None)
    return h


for name in ["q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d",
             "f_proj", "b_proj", "g_proj", "o_norm", "o_proj"]:
    if hasattr(att0, name):
        getattr(att0, name).register_forward_hook(mk("L0." + name),
                                                  with_kwargs=True)
model.model.layers[0].input_layernorm.register_forward_hook(
    mk("L0.in_ln"), with_kwargs=True)

with torch.no_grad():
    out = model(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False)

np.save("/data/ANVIL/ling3/hook_meta.npy", np.array([0]))
for tag, (ins, o2) in sorted(cap.items()):
    for j, ten in enumerate(ins):
        np.save(f"/data/ANVIL/ling3/hook_{tag.replace('.', '_')}_in{j}.npy", ten.numpy())
    if o2 is not None:
        np.save(f"/data/ANVIL/ling3/hook_{tag.replace('.', '_')}_out.npy", o2.numpy())
    print("HOOK", tag, "n_in", len(ins),
          "out", tuple(o2.shape) if o2 is not None else None)
print("HOOKS_OK")
