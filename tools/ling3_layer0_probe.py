#!/usr/bin/env python3
"""Layer-0 KDA bisection: official-style forward vs engine-formula forward."""
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

M = "/data/ANVIL/ling3/hf_tiny"
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
cache = {}


def get(name):
    if name not in cache:
        cache[name] = safe_open(f"{M}/{idx[name]}", framework="pt", device="cpu").get_tensor(name)
    return cache[name].float()


ids = [151644, 8948, 198, 2610, 525, 264, 10950, 17871, 13, 151645, 198,
       151644, 872, 198, 9707, 11, 151645, 198, 151644, 77091, 198]
T = len(ids)
eps = 1e-6
x0 = get("model.word_embeddings.weight")[ids]                     # [T,D]
ln0 = get("model.layers.0.input_layernorm.weight")
p = "model.layers.0.attention."


def rmsnorm(x, w):
    ms = x.float().pow(2).mean(-1, keepdim=True)
    return x.float() / torch.sqrt(ms + eps) * w


xn = rmsnorm(x0, ln0)

# --- official-style KDA (mirrors modeling code + shim semantics) ---
def l2n(v):
    return v / torch.sqrt((v * v).sum(-1, keepdim=True) + eps)


q = xn @ get(p + "q_proj.weight").T
k = xn @ get(p + "k_proj.weight").T
v = xn @ get(p + "v_proj.weight").T
gp = xn @ get(p + "g_proj.weight").T
graw = xn @ get(p + "f_proj.weight").T
beta_raw = xn @ get(p + "b_proj.weight").T

import torch.nn.functional as F


def causal_conv(xw, w):           # x [T,D], w [D,K]
    K = w.shape[-1]
    xp = F.pad(xw.T.unsqueeze(0), (K - 1, 0))[0]                  # [D,T+K-1]
    out = torch.zeros_like(xw)
    for t in range(xw.shape[0]):
        out[t] = (xp[:, t:t + K] * w).sum(-1)
    return F.silu(out)


cq, ck_, cv_ = get(p + "q_conv1d.weight")[:, 0, :], get(p + "k_conv1d.weight")[:, 0, :], get(p + "v_conv1d.weight")[:, 0, :]
q, k, v = causal_conv(q, cq), causal_conv(k, ck_), causal_conv(v, cv_)

H, hd = 16, 128
P = H * hd
q = l2n(q.reshape(T, H, hd)) * hd ** -0.5
k = l2n(k.reshape(T, H, hd))
v = v.reshape(T, H, hd)
gp = gp.reshape(T, H, hd)
A = torch.exp(get(p + "A_log"))
dt = get(p + "dt_bias").reshape(H, hd)
bt = torch.sigmoid(beta_raw)                                       # [T,H]
S = torch.zeros(H, hd, hd)
o = torch.zeros(T, H, hd)
for t in range(T):
    z = graw[t].reshape(H, hd) + dt
    alpha = torch.exp(-5.0 * torch.sigmoid(A[:, None] * z))
    S = S * alpha[..., None]
    delta = (S * k[t][..., None]).sum(-2)                          # wait: dims
    # correct: delta[v] = sum_k S[k,v]*k[k]
    delta = (S * k[t].reshape(H, hd, 1)).sum(1)
    vt = (v[t] - delta) * bt[t][:, None]
    S = S + k[t].reshape(H, hd, 1) * vt[:, None, :]
    o[t] = (S * q[t].reshape(H, hd, 1)).sum(1)
on = o / torch.sqrt(o.pow(2).mean(-1, keepdim=True) + eps) * get(p + "o_norm.weight") * torch.sigmoid(gp)
attn_out = on.reshape(T, P) @ get(p + "o_proj.weight").T
h1_official_style = x0 + attn_out

# dense MLP
mlp_g = get("model.layers.0.mlp.gate_proj.weight")
mlp_u = get("model.layers.0.mlp.up_proj.weight")
mlp_d = get("model.layers.0.mlp.down_proj.weight")
pln = get("model.layers.0.post_attention_layernorm.weight")
h1 = rmsnorm(h1_official_style, pln)
mlp_out = (F.silu(h1 @ mlp_g.T) * (h1 @ mlp_u.T)) @ mlp_d.T
h2 = h1_official_style + mlp_out

ref = np.load("/data/ANVIL/ling3/fixture_a/hidden.npy")            # [5,T,D]
r1 = torch.tensor(ref[1])
r2 = torch.tensor(ref[2])
print("official-style h1 vs fixture[1]: max diff",
      (h1_official_style - r1).abs().max().item())
print("official-style h2 vs fixture[2]: max diff",
      (h2 - r2).abs().max().item())

# --- sub-step probes against the fixture ---
# embedding check
print("embed vs fixture[0]:", (x0 - torch.tensor(ref[0])).abs().max().item())
