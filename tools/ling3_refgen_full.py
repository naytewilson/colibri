#!/usr/bin/env python3
"""Ling-3.0-tiny full-depth streamed reference (Fixture B).

Pure-torch reimplementation of BailingMoeV3ForCausalLM semantics (verified
line-by-line against modeling_bailing_moe_v3.py @ b61f4338 and fla naive
kernels). Streams one layer at a time off the BF16 snapshot so a 24-layer
7.9B model fits in ~14 GiB RAM. Produces:

  <out>/prompt_ids.json
  <out>/hidden.npy     [n_stages, T, H] f32 (stage 0 = embeddings)
  <out>/logits.npy     [T, V] f32 teacher-forced logits at every position
  <out>/greedy.json    greedy continuation token ids (ngen)
  <out>/router.json    router decisions per layer/position

Cross-validation: this tool's stage hiddens for layers 0..N must match
ling3_refgen_official.py's Fixture A within tolerance before use.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def load_cfg(model):
    cfg = json.load(open(f"{model}/config.json"))
    return cfg


class ShardReader:
    def __init__(self, model):
        self.model = model
        self.idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
        self.open_file = None
        self.open_name = None

    def get(self, name):
        shard = self.idx[name]
        if self.open_name != shard:
            if self.open_file is not None:
                del self.open_file          # safe_open has no close(); drop ref
            self.open_file = safe_open(f"{self.model}/{shard}", framework="pt", device="cpu")
            self.open_name = shard
        return self.open_file.get_tensor(name)


def l2norm(x, eps=1e-6):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def silu(x):
    return x / (1 + torch.exp(-x))


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def rmsnorm(x, w, eps):
    ms = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() / torch.sqrt(ms + eps)) * w.float()


class KDALayer:
    def __init__(self, r, li):
        p = f"model.layers.{li}.attention."
        self.q = r.get(p + "q_proj.weight").float()
        self.k = r.get(p + "k_proj.weight").float()
        self.v = r.get(p + "v_proj.weight").float()
        self.o = r.get(p + "o_proj.weight").float()
        self.gp = r.get(p + "g_proj.weight").float()
        self.fp = r.get(p + "f_proj.weight").float()
        self.cq = r.get(p + "q_conv1d.weight").float()[:, 0, :]   # [D,K]
        self.ck = r.get(p + "k_conv1d.weight").float()[:, 0, :]
        self.cv = r.get(p + "v_conv1d.weight").float()[:, 0, :]
        self.A = torch.exp(r.get(p + "A_log").float())            # [H]
        self.dt = r.get(p + "dt_bias").float()                    # [P]
        self.bp = r.get(p + "b_proj.weight").float()              # [H,D]
        self.onw = r.get(p + "o_norm.weight").float()             # [hd]
        self.H = self.bp.shape[0]
        self.hd = self.onw.shape[0]

    def forward(self, x, state):
        # x: [T,D] f32 ; state: dict(conv=[3,T?,...] rolling windows, S=[H,hd,hd])
        T, D = x.shape
        P = self.q.shape[0]
        H, hd = self.H, self.hd
        q = x @ self.q.T
        k = x @ self.k.T
        v = x @ self.v.T
        gp = x @ self.gp.T
        graw = x @ self.fp.T
        beta_raw = x @ self.bp.T
        convw = state["conv"]                     # list of [P, K] tensors
        outs = [q, k, v]
        for ci in range(3):
            cur = outs[ci]
            win = convw[ci]                       # [P,K] oldest..newest
            taps = [self.cq, self.ck, self.cv][ci]
            new = torch.empty_like(cur)
            for t in range(T):
                win = torch.cat([win[:, 1:], cur[t][:, None]], dim=1)
                convw[ci] = win
                acc = (win * taps).sum(1)
                new[t] = silu(acc)
            outs[ci] = new
        q, k, v = outs
        q = l2norm(q) * hd ** -0.5
        k = l2norm(k)
        S = state["S"]
        o = torch.zeros_like(v)
        lb = -5.0
        for t in range(T):
            z = graw[t] + self.dt
            alpha = torch.exp(lb * sigmoid(torch.outer(self.A, z)))   # [H,P]->[H,hd] via view
            alpha = alpha.reshape(H, hd)
            bt = sigmoid(beta_raw[t])                                  # [H]
            S = S * alpha[..., None]                                   # S'[h,k,v]
            delta = (S * k[t].reshape(H, hd, 1)).sum(-2)               # [H,hd(v)]
            vt = (v[t].reshape(H, hd) - delta) * bt[:, None]
            S = S + k[t].reshape(H, hd, 1) * vt[:, None, :]
            o[t] = (S * q[t].reshape(H, hd, 1)).sum(-2).reshape(P)
        on = o.float() / torch.sqrt(o.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        on = on * self.onw * sigmoid(gp.float())
        return on @ self.o.T


def rope_interleave(x, pos, theta, half):
    # x: [T,H,half*2]; returns rotated copy
    inv = theta ** (-torch.arange(half, dtype=torch.float32) * 2 / (half * 2))
    ang = pos[:, None].float() * inv[None, :]
    cos, sin = ang.cos(), ang.sin()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    out = torch.stack([r0, r1], dim=-1).flatten(-2)
    return out


class MLALayer:
    def __init__(self, r, li, cfg):
        p = f"model.layers.{li}.attention."
        self.qa = r.get(p + "q_a_proj.weight").float()
        self.qaln = r.get(p + "q_a_layernorm.weight").float()
        self.qb = r.get(p + "q_b_proj.weight").float()
        self.kva = r.get(p + "kv_a_proj_with_mqa.weight").float()
        self.kvaln = r.get(p + "kv_a_layernorm.weight").float()
        self.kvb = r.get(p + "kv_b_proj.weight").float()
        self.dense = r.get(p + "dense.weight").float()
        self.gp = r.get(p + "g_proj.weight").float()
        self.cfg = cfg
        self.scale = (cfg["qk_head_dim"]) ** -0.5

    def forward(self, x, cache):
        H = self.cfg["num_attention_heads"]
        nope = self.cfg["qk_nope_head_dim"]
        rope = self.cfg["qk_rope_head_dim"]
        vh = self.cfg["v_head_dim"]
        kvr = self.cfg["kv_lora_rank"]
        theta = self.cfg["rope_theta"]
        T, D = x.shape
        qa = rmsnorm(x @ self.qa.T, self.qaln, self.cfg["rms_norm_eps"])
        q = (qa @ self.qb.T).reshape(T, H, nope + rope)
        ckv = x @ self.kva.T
        L = rmsnorm(ckv[:, :kvr], self.kvaln, self.cfg["eps"])
        krot_new = ckv[:, kvr:]
        kvb_out = (L @ self.kvb.T).reshape(T, H, nope + vh)
        kp, vp = kvb_out[..., :nope], kvb_out[..., nope:]
        half = rope // 2
        pos = torch.arange(cache["pos"], cache["pos"] + T)
        q_rot = rope_interleave(q[:, :, nope:], pos, theta, half)
        Lc = cache["L"]
        Rc = cache["R"]
        for t in range(T):
            Lc[cache["pos"] + t] = L[t]
            kr = rope_interleave(krot_new[t:t, None, :], pos[t:t], theta, half)[0]
            Rc[cache["pos"] + t] = kr
        gate = sigmoid((x @ self.gp.T).float())                       # [T,H] head-wise
        ctx = torch.zeros(T, H, vh)
        ntot = cache["pos"] + T
        for tt in range(T):
            nt = cache["pos"] + tt + 1
            qp = q[tt, :, :nope]
            qr_ = q_rot[tt]
            sc = torch.zeros(nt)
            for t2 in range(nt):
                kp_t = (Lc[t2][None, :] @ self.kvb.T).reshape(H, nope + vh)[:, :nope]
                s = (qp * kp_t).sum(-1) + (qr_ * Rc[t2]).sum(-1)
                sc[t2] = s * self.scale
            sc = torch.softmax(sc, -1)
            lat = torch.einsum("t,t k->k", sc[:nt], Lc[:nt])           # absorbed latent ctx
            for h in range(H):
                vd_h = self.kvb[h * (nope + vh) + nope: (h + 1) * (nope + vh)]
                ctx[tt, h] = (vd_h @ lat) * gate[tt, h]
        cache["pos"] += T
        return ctx.reshape(T, H * vh) @ self.dense.T


class MLP:
    def __init__(self, r, prefix):
        self.gate = r.get(prefix + ".gate_proj.weight").float()
        self.up = r.get(prefix + ".up_proj.weight").float()
        self.down = r.get(prefix + ".down_proj.weight").float()

    def forward(self, x):
        return (silu(x @ self.gate.T) * (x @ self.up.T)) @ self.down.T


class MoE:
    def __init__(self, r, li, cfg, reader):
        self.gate_w = r.get(f"model.layers.{li}.mlp.gate.weight").float()
        self.bias = r.get(f"model.layers.{li}.mlp.gate.expert_bias").float()
        self.shared = MLP(r, f"model.layers.{li}.mlp.shared_experts")
        self.E = cfg["num_experts"]
        self.topk = cfg["num_experts_per_tok"]
        self.n_group = cfg["n_group"]
        self.topk_group = cfg["topk_group"]
        self.rscale = cfg["routed_scaling_factor"]
        self.experts = []
        for e in range(self.E):
            p = f"model.layers.{li}.mlp.experts.{e}."
            self.experts.append((r.get(p + "gate_proj.weight").float(),
                                 r.get(p + "up_proj.weight").float(),
                                 r.get(p + "down_proj.weight").float()))

    def forward(self, x, capture=None):
        T, D = x.shape
        logits = x.float() @ self.gate_w.T
        scores = sigmoid(logits)
        rt = scores + self.bias
        Eg = scores.shape[-1] // self.n_group
        gs = rt.reshape(T, self.n_group, Eg).topk(2, dim=-1).values.sum(-1)
        gidx = torch.topk(gs, k=self.topk_group, dim=-1).indices
        mask = torch.zeros_like(gs, dtype=torch.bool).scatter(1, gidx, True)
        score_mask = mask.unsqueeze(-1).expand(T, self.n_group, Eg).reshape(T, -1)
        masked = rt.masked_fill(~score_mask, float("-inf"))
        _, topk_idx = torch.topk(masked, k=self.topk, dim=-1)
        topk_scores = torch.gather(scores, 1, topk_idx)
        w = topk_scores / (topk_scores.sum(-1, keepdim=True) + 1e-20)
        w = w * self.rscale
        y = torch.zeros(T, D)
        for t in range(T):
            for kk in range(self.topk):
                e = int(topk_idx[t, kk])
                gw, uw, dw = self.experts[e]
                y[t] += float(w[t, kk]) * ((silu(x[t] @ gw.T) * (x[t] @ uw.T)) @ dw.T)
        if capture is not None:
            capture.append({"idx": topk_idx.tolist(),
                            "w": [[round(float(v), 7) for v in row] for row in w]})
        return y + self.shared.forward(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/ANVIL/ling3/hf_tiny")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ngen", type=int, default=8)
    args = ap.parse_args()

    cfg = load_cfg(args.model)
    r = ShardReader(args.model)
    ids = [int(x) for x in args.ids.split(",")]
    eps = cfg["rms_norm_eps"]

    embed_w = r.get("model.word_embeddings.weight")
    final_norm = r.get("model.norm.weight")
    head_w = r.get("lm_head.weight")

    layers = []
    for li in range(cfg["num_hidden_layers"]):
        if (li + 1) % cfg["layer_group_size"] == 0 or li >= cfg["num_hidden_layers"] // cfg["layer_group_size"] * cfg["layer_group_size"]:
            attn = MLALayer(r, li, cfg)
            state = {"L": torch.zeros(4096, cfg["kv_lora_rank"]),
                     "R": torch.zeros(4096, cfg["qk_rope_head_dim"]), "pos": 0}
        else:
            attn = KDALayer(r, li)
            state = {"conv": [torch.zeros(attn.q.shape[0], cfg["short_conv_kernel_size"]) for _ in range(3)],
                     "S": torch.zeros(attn.H, attn.hd, attn.hd)}
        mlp = MoE(r, li, cfg, r) if li >= cfg["first_k_dense_replace"] else \
            MLP(r, f"model.layers.{li}.mlp")
        iln = r.get(f"model.layers.{li}.input_layernorm.weight").float()
        pln = r.get(f"model.layers.{li}.post_attention_layernorm.weight").float()
        layers.append({"attn": attn, "mlp": mlp, "iln": iln, "pln": pln,
                       "state": state, "moe": not isinstance(mlp, MLP)})
        print(f"loaded layer {li}", flush=True)

    def full_forward(ids_list, capture_router=False):
        T = len(ids_list)
        hiddens = [embed_w[ids_list].float()]
        x = hiddens[0]
        caps = {}
        for li, L in enumerate(layers):
            xn = rmsnorm(x, L["iln"], eps)
            a = L["attn"].forward(xn, L["state"])
            x = x + a
            xn2 = rmsnorm(x, L["pln"], eps)
            cap = [] if (capture_router and L["moe"]) else None
            m = L["mlp"].forward(xn2, capture=cap) if L["moe"] else L["mlp"].forward(xn2)
            if cap is not None:
                caps[li] = cap
            x = x + m
            hiddens.append(x.clone())
        logits = (rmsnorm(x, final_norm, eps) @ head_w.T).float()
        return hiddens, logits, caps

    hiddens, logits, caps = full_forward(ids, capture_router=True)
    import os
    os.makedirs(args.out, exist_ok=True)
    np.save(f"{args.out}/hidden.npy", torch.stack(hiddens).numpy().astype(np.float32))
    np.save(f"{args.out}/logits.npy", logits.numpy().astype(np.float32))
    json.dump(ids, open(f"{args.out}/prompt_ids.json", "w"))
    json.dump([{"layer": k, "items": v} for k, v in caps.items()],
              open(f"{args.out}/router.json", "w"))
    gen = []
    for _ in range(args.ngen):
        nxt = int(logits[-1].argmax())
        gen.append(nxt)
        hiddens, logits, _ = full_forward(ids + gen)
    json.dump(gen, open(f"{args.out}/greedy.json", "w"))
    print("FIXTURE_B_OK", len(layers), "layers", len(ids), "tokens", "greedy", gen)


if __name__ == "__main__":
    main()
