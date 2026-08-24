#!/usr/bin/env python3
"""VERIFIER TOOL (ling3-wave1): independent compressed-tensors INT4 dequantizer.

Implements the pack-quantized int4-g32 symmetric contract DIRECTLY from the
checkpoint tensors (no native-C code involved):

    weight_packed I32 [O, I/8] : 8 unsigned nibbles per I32, LSB-first,
                                 value = q + 8, q in [-8, 7]
    weight_scale BF16 [O, I/32]: one symmetric group scale per 32 input cols
    weight_shape I64 [2]       : original [O, I]

    W[o, i] = (nibble(o, i) - 8) * scale[o, i // 32]
"""
import json

import numpy as np
import torch


def load_tensor_index(model_dir):
    return json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]


def unpack_i32Nibbles(packed_i32):
    """[O, I/8] int32 -> [O, I] int16 values in [-8, 7] (LSB-first nibbles)."""
    p = packed_i32.astype(np.uint32)
    O, W = p.shape
    out = np.empty((O, W * 8), dtype=np.int16)
    for k in range(8):
        nib = ((p >> (4 * k)) & 0xF).astype(np.int16)
        out[:, k::8] = nib - 8
    return out


def iter_int4_expert_tensors(model_dir, wanted_names):
    """Yield ('<base>.weight', f32 torch [O, I]) for each requested matrix.

    wanted_names use BF16-style '<base>.weight' suffixes; the INT4 index
    stores the same matrices under '<base>.weight_packed'.
    """
    from safetensors import safe_open
    wm = load_tensor_index(model_dir)
    byfile = {}
    for n in wanted_names:
        byfile.setdefault(wm[f"{n}_packed"], []).append(n)
    for fname, names in sorted(byfile.items()):
        with safe_open(f"{model_dir}/{fname}", framework="pt", device="cpu") as st:
            for n in sorted(names):
                pk = st.get_tensor(f"{n}_packed")
                sc = st.get_tensor(f"{n}_scale").float()
                shp = st.get_tensor(f"{n}_shape")
                O, I = int(shp[0]), int(shp[1])
                q = unpack_i32Nibbles(pk.numpy())
                assert q.shape == (O, I) and sc.shape == (O, I // 32), \
                    f"geometry mismatch {q.shape} {sc.shape} vs {(O, I)}"
                w = q.astype(np.float32) * np.repeat(sc.numpy(), 32, axis=1)
                yield n, torch.from_numpy(w)
