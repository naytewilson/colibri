#!/usr/bin/env python3
"""Ling-3.0-tiny INT4 compressed-tensors packing discriminator.

Dequantizes an expert weight from the official INT4 checkpoint under both
candidate nibble orders (LSB-first / MSB-first within each int32 word) and
compares against the same tensor from the official BF16 checkpoint.
The order with far lower relative error is the container's true layout.
"""
import json
import struct
import sys

import numpy as np


def load_tensor(repo_dir, index_path, name):
    idx = json.load(open(index_path))
    shard = idx["weight_map"][name]
    path = f"{repo_dir}/{shard}"
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        h = hdr[name]
        s, e = h["data_offsets"]
        f.seek(base + s)
        raw = f.read(e - s)
        return h["dtype"], h["shape"], raw


def bf16_to_f32(raw):
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return np.frombuffer(u.tobytes(), dtype=np.float32)


def main(bf16_dir, int4_dir, tensor="model.layers.9.mlp.experts.0.gate_proj.weight"):
    dt, shape, raw = load_tensor(int4_dir, f"{int4_dir}/model.safetensors.index.json",
                                 tensor + "_packed")
    assert dt == "I32", dt
    packed = np.frombuffer(raw, dtype="<u4").reshape(shape)

    _, sc_shape, sc_raw = load_tensor(int4_dir, f"{int4_dir}/model.safetensors.index.json",
                                      tensor + "_scale")
    scales = bf16_to_f32(sc_raw).reshape(sc_shape).astype(np.float32)
    _, sh_shape, sh_raw = load_tensor(int4_dir, f"{int4_dir}/model.safetensors.index.json",
                                      tensor + "_shape")
    O, I = np.frombuffer(sh_raw, dtype="<i8").tolist()

    shifts_lo = (np.arange(8, dtype=np.uint32) * 4).reshape(1, 1, 8)
    shifts_hi = ((7 - np.arange(8, dtype=np.uint32)) * 4).reshape(1, 1, 8)
    b = packed.reshape(*packed.shape, 1)

    expanded = np.repeat(scales.astype(np.float32), 32, axis=1)
    ref_dt, ref_shape, ref_raw = load_tensor(bf16_dir, f"{bf16_dir}/model.safetensors.index.json",
                                             tensor)
    W = bf16_to_f32(ref_raw).reshape(ref_shape).astype(np.float64)
    print(f"tensor {tensor}: declared shape {sh_shape} -> [{O},{I}], bf16 {ref_shape}")

    for tag, shifts in (("LSB-first-nibble", shifts_lo), ("MSB-first-nibble", shifts_hi)):
        q = ((b >> shifts) & 0xF).reshape(packed.shape[0], -1).astype(np.int16)
        q = np.where(q >= 8, q - 16, q).astype(np.float64)
        Wq = q * expanded.astype(np.float64)[: W.shape[0], : W.shape[1]]
        err = np.abs(Wq - W).mean()
        rel = err / max(np.abs(W).mean(), 1e-30)
        cos = (Wq.ravel() @ W.ravel()) / max((np.linalg.norm(Wq) * np.linalg.norm(W)), 1e-30)
        print(f"{tag}: mean_abs_err={err:.6e} rel={rel:.4f} cosine={cos:.5f} "
              f"sample {np.round(Wq[0, :4], 4)} vs bf16 {np.round(W[0, :4], 4)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
