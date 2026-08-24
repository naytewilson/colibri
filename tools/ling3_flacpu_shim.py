#!/usr/bin/env python3
"""CPU-runnable `fla` shim for the official Ling-3.0-tiny modeling code.

Implements the EXACT semantics of the fla kernels the model calls, using
fla's own naive/reference formulas (fla/ops/kda/naive.py) plus the
fused_recurrent_kda gate preparation (lower_bound * sigmoid(exp(A_log)*(g+dt_bias))):

    chunk_kda / fused_recurrent_kda(q,k,v,g,beta,A_log,dt_bias,...,
        use_qk_l2norm_in_kernel=True, safe_gate, lower_bound)

This shim exists because the shipped fla kernels are Triton/GPU-only and the
reference host is CPU-only. It is used ONLY for reference fixture generation,
never by the deployed runtime.
"""
import math

import torch
from einops import rearrange


def _l2norm(x, eps=1e-6):
    # fla l2norm eps sits INSIDE the sqrt: x / sqrt(sum(x^2)+eps)
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


def _kda_core(q, k, v, gk, beta, initial_state=None, output_final_state=False):
    """fla naive_recurrent_kda recurrence with precomputed log gates gk."""
    B, T, H, K, V = *q.shape, v.shape[-1]
    S = q.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state.to(S)
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i = q[:, i], k[:, i], v[:, i]
        S = S * gk[:, i][..., None].exp()
        delta = (k_i[..., None] * S).sum(-2)
        S = S + torch.einsum("b h k, b h v -> b h k v", beta[:, i, :, None] * k_i, v_i - delta)
        o[:, i] = torch.einsum("b h k, b h k v -> b h v", q_i, S)
    return (o, S) if output_final_state else (o, None)


def _prepare_gate(g, A_log, dt_bias, safe_gate, lower_bound):
    # g: [B,T,H,K]; dt_bias: [H*K] flattened per (head,dim) in the checkpoint
    g = g.float()
    if dt_bias is not None:
        H, K = g.shape[2], g.shape[3]
        g = g + dt_bias.float().reshape(H, K)
    if lower_bound is not None:            # USE_LOWER_BOUND branch (Ling: safe_gate)
        A = torch.exp(A_log.float()) if A_log is not None else 1.0
        return lower_bound * torch.sigmoid(A.reshape(-1, 1) * g)
    return -torch.exp(A_log.float()).reshape(-1, 1) * torch.nn.functional.softplus(g)


def fused_recurrent_kda(q, k, v, g, beta, A_log=None, dt_bias=None,
                        initial_state=None, output_final_state=True,
                        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                        safe_gate=False, lower_bound=None, cu_seqlens=None, **kw):
    B, T, H, K = q.shape
    scale = K ** -0.5
    if use_qk_l2norm_in_kernel:
        q, k = _l2norm(q.float()), _l2norm(k.float())
    q = q * scale
    gk = _prepare_gate(g, A_log, dt_bias, safe_gate, lower_bound) if use_gate_in_kernel \
        else g.float()
    beta = beta.float().sigmoid()
    o, S = _kda_core(q.to(v.dtype), k.to(v.dtype), v, gk.to(v.dtype),
                     beta.to(v.dtype), initial_state, output_final_state)
    return o, S


def chunk_kda(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None,
              initial_state=None, output_final_state=True,
              use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
              safe_gate=False, lower_bound=None, cu_seqlens=None, **kw):
    # Sequential recurrence == chunk math up to float association; the fixture
    # protocol pins T<=64 so the official code itself takes fused_recurrent.
    return fused_recurrent_kda(q, k, v, g, beta, A_log, dt_bias, initial_state,
                               output_final_state, use_qk_l2norm_in_kernel,
                               use_gate_in_kernel, safe_gate, lower_bound, cu_seqlens)


class ShortConvolution(torch.nn.Conv1d):
    """fla.modules.ShortConvolution: depthwise causal conv + SiLU."""

    def __init__(self, hidden_size, kernel_size=4, activation="silu", bias=False):
        super().__init__(hidden_size, hidden_size, kernel_size,
                         groups=hidden_size, bias=bias,
                         padding=kernel_size - 1)
        self.activation = activation

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, **kw):
        # x: [B, T, D]; causal conv over time, trim future padding
        y = super().forward(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        if self.activation == "silu":
            y = torch.nn.functional.silu(y)
        final_state = None
        if output_final_state:
            K = self.kernel_size[0]
            final_state = x[:, -K + 1:].transpose(1, 2) if T_last_ok(x, K) else None
        return (y, final_state) if output_final_state else (y, cache)


def T_last_ok(x, K):
    return x.shape[1] >= K - 1


class FusedRMSNormGated(torch.nn.Module):
    """RMSNorm over head_dim, multiplied by sigmoid(gate) (activation='sigmoid')."""

    def __init__(self, head_dim, eps=1e-6, activation="sigmoid"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(head_dim))
        self.eps = eps
        self.activation = activation

    def forward(self, o, g):
        ms = o.float().pow(2).mean(-1, keepdim=True)
        on = o.float() / torch.sqrt(ms + self.eps)
        on = on * self.weight.float()
        if self.activation == "sigmoid":
            on = on * torch.sigmoid(g.float())
        return on.to(o.dtype)


# --- names imported by the official file but unused on the CPU fixture path ---
def fused_recurrent_simple_gla(*a, **k):
    raise NotImplementedError("simple_gla not needed for Ling fixtures")


def chunk_simple_gla(*a, **k):
    raise NotImplementedError("simple_gla not needed for Ling fixtures")


def prepare_cu_seqlens_from_mask(*a, **k):
    raise NotImplementedError("unpadding not exercised (mask=None)")


def prepare_lens_from_mask(*a, **k):
    raise NotImplementedError


def tensor_cache(*a, **k):
    def deco(f):
        return f
    return deco if not callable(a[0] if a else None) else a[0]


def install():
    import sys
    import types
    flaN = types.ModuleType("fla")
    mods = types.ModuleType("fla.modules")
    ops = types.ModuleType("fla.ops")
    opskda = types.ModuleType("fla.ops.kda")
    sglaf = types.ModuleType("fla.ops.simple_gla.fused_recurrent")
    sglac = types.ModuleType("fla.ops.simple_gla.chunk")
    idxm = types.ModuleType("fla.ops.utils.index")
    utim = types.ModuleType("fla.utils")
    mods.FusedRMSNormGated = FusedRMSNormGated
    mods.ShortConvolution = ShortConvolution
    opskda.chunk_kda = chunk_kda
    opskda.fused_recurrent_kda = fused_recurrent_kda
    sglaf.fused_recurrent_simple_gla = fused_recurrent_simple_gla
    sglac.chunk_simple_gla = chunk_simple_gla
    idxm.prepare_cu_seqlens_from_mask = prepare_cu_seqlens_from_mask
    idxm.prepare_lens_from_mask = prepare_lens_from_mask
    utim.tensor_cache = tensor_cache
    for name, mod in [("fla", flaN), ("fla.modules", mods), ("fla.ops", ops),
                      ("fla.ops.kda", opskda),
                      ("fla.ops.simple_gla.fused_recurrent", sglaf),
                      ("fla.ops.simple_gla.chunk", sglac),
                      ("fla.ops.utils.index", idxm), ("fla.utils", utim)]:
        sys.modules[name] = mod
