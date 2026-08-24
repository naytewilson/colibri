#!/usr/bin/env python3
"""Compare engine-formula replication against official hook dumps, and
brute-force the ShortConvolution semantics until the recurrence matches."""
import json

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

M = "/data/ANVIL/ling3/hf_tiny"
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
cache = {}


def get(name):
    if name not in cache:
        cache[name] = safe_open(f"{M}/{idx[name]}", framework="pt", device="cpu").get_tensor(name)
    return cache[name].float()


def hook(tag):
    return np.load(f"/data/ANVIL/ling3/hook_{tag.replace('.', '_')}_out.npy")[0]


def hook_in(tag, j=0):
    return np.load(f"/data/ANVIL/ling3/hook_{tag.replace('.', '_')}_in{j}.npy")[0]


xn = torch.tensor(hook_in("L0.in_ln"))          # wait: in_ln hook captured its OUTPUT as 'out'; input stored too
# in_ln hook: in0 = embed, out = normalized -> use out as xn
xn = torch.tensor(hook("L0.in_ln"))
T = xn.shape[0]
p = "model.layers.0.attention."
H, hd = 16, 128
P = H * hd
eps = 1e-6


def l2n(v):
    return v / torch.sqrt((v * v).sum(-1, keepdim=True) + eps)


print("q_proj match:", np.abs((xn @ get(p + "q_proj.weight").T).numpy()
                              - hook("L0.q_proj")).max())
beta_raw = xn @ get(p + "b_proj.weight").T
print("b_proj match:", np.abs(beta_raw.numpy() - hook("L0.b_proj")).max())
graw = xn @ get(p + "f_proj.weight").T
print("f_proj match:", np.abs(graw.numpy() - hook("L0.f_proj")).max())
gp = xn @ get(p + "g_proj.weight").T
print("g_proj match:", np.abs(gp.numpy() - hook("L0.g_proj")).max())

qp_out = torch.tensor(hook("L0.q_proj"))
kp_out = torch.tensor(hook("L0.k_proj"))
vp_out = torch.tensor(hook("L0.v_proj"))


def causal_conv(xw, w, order="tap_newest"):
    K = w.shape[-1]
    xp = F.pad(xw.T.unsqueeze(0), (K - 1, 0))[0].float()
    out = torch.zeros_like(xw)
    for t in range(xw.shape[0]):
        win = xp[:, t:t + K]                      # oldest..newest
        ww = w.float()
        if order == "tap_newest":                 # w[j]*win[j], j=K-1 newest
            out[t] = (win * ww).sum(-1)
        elif order == "tap_oldest":
            out[t] = (win.flip(1) * ww).sum(-1)
    return F.silu(out)


cq = get(p + "q_conv1d.weight")[:, 0, :]
ckc = get(p + "k_conv1d.weight")[:, 0, :]
cvv = get(p + "v_conv1d.weight")[:, 0, :]
A = torch.exp(get(p + "A_log"))
dt = get(p + "dt_bias").reshape(H, hd)

on_ref = torch.tensor(hook_in("L0.o_norm", 0))     # kernel o [T,H,hd]
gate_ref = torch.tensor(hook_in("L0.o_norm", 1))   # g [T,H,hd]

for order in ("tap_newest", "tap_oldest"):
    q = causal_conv(qp_out, cq, order)
    k = causal_conv(kp_out, ckc, order)
    v = causal_conv(vp_out, cvv, order)
    q2 = l2n(q.reshape(T, H, hd)) * hd ** -0.5
    k2 = l2n(k.reshape(T, H, hd))
    v2 = v.reshape(T, H, hd)
    bt = torch.sigmoid(torch.tensor(hook("L0.b_proj")))
    S = torch.zeros(H, hd, hd)
    o = torch.zeros(T, H, hd)
    for t in range(T):
        z = graw[t].reshape(H, hd) + dt
        alpha = torch.exp(-5.0 * torch.sigmoid(A[:, None] * z))
        S = S * alpha[..., None]
        delta = (S * k2[t].reshape(H, hd, 1)).sum(1)
        vt = (v2[t] - delta) * bt[t][:, None]
        S = S + k2[t].reshape(H, hd, 1) * vt[:, None, :]
        o[t] = (S * q2[t].reshape(H, hd, 1)).sum(1)
    d = (o - on_ref).abs().max().item()
    print(f"recurrence vs o_norm.in0 [{order}]: max diff {d:.6e}")

print("my gate vs o_norm.in1:", np.abs(gp.numpy() - gate_ref.numpy()).max())
