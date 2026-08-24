# FORGE F1 — Parity Container Pinning (Phase 0)

Date: 2026-08-23 · Agent: ox-alpha (Kilo) · Task: `colibri-forge-f1-ooc-moe-runtime`

## Resolution of container identity

Frozen-campaign provenance cross-check:

- `anvil.events #54964` (2026-08-22): baseline reproduced at tournament-p1 tip
  `5dc15f4`, **4.98 tok/s warm median, mixed-low cap128 EP** — protocol
  `CORPUS_FILE=/home/nayte/prompts/corpus_persistent_serving.txt`, cap=128,
  ebits=4, OMP=8, PILOT=1, COLI_DENSE_I8=1, COLI_EXPERT_PARALLEL=1, TEMP=0,
  N_NEW=64, 4 turns × 64 tokens, warm median = median(turns 2–4).
- Receipt: ANVIL repo `docs/experiments/colibri-qwen36-a3b-node/QWEN36_ASYNC_ADMISSION_RECEIPT_20260822.md`
  (worktree `~/ANVIL-worktrees/colibri-qwen36-a3b-node-20260820`) names the
  container explicitly: **`/home/nayte/models/qwen36_mixed_low`**
  (MIXED-LOW-15pct-INT4-DIRECT-BF16; 1536 fmt-INT4 / 8704 INT3 experts).
- Promgate factorial receipt (`QWEN36_FUSED_BATCH_PROMGATE_RECEIPT_20260823.md`,
  tracked at ANVIL commit `2f41cdc`) ran on the same binary/container class;
  arms A–D totals consistent with the 4.98 anchor lineage.

**PINNED PARITY CONTAINER**: `/home/nayte/models/qwen36_mixed_low` on
`anvil-node-02 (ANVIL Dell Node)`. Do not copy to Mac.

## SHA256 pins (measured 2026-08-23 on node-02)

- `config.json` = `7606939241b64e82308369b308af8abe85d24e70846704c41ed4e0b59687a674`
- `allocation_manifest.json` = `9df927beb35ef8f28583e857687d0a3f9dba3239e09d72ad305319a8fed3a2e4`
- `tokenizer.json` = `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- Corpus `/home/nayte/prompts/corpus_persistent_serving.txt` (489 B)
  = `831b27539e462cf841b3898dd635f3d3945b458f9a4495269c17028e8f56983a`
- All 41 shards hashed: node-side copy at
  `/tmp/forge_f1_mixed_low_hashes.txt` on node-02 (regenerate if absent):
  `cd /home/nayte/models/qwen36_mixed_low && sha256sum model-*.safetensors tokenizer.json`

First/last shard samples (full list in the file above):
- `model-00000.safetensors` = `eff8fd16f452c5e7e05112eac3c0fff633118b079bb36f05097edd0492388cc9`
- `model-globals.safetensors` = `c3fe3ca2c44f539bf3fd9c140451c3ab4f2aeeacf953714aa8ca4fdbf0fb95b4`

## Other candidates observed (NOT pinned)

| Path | Size | Role |
|---|---|---|
| `/home/nayte/models/qwen36_i3_gs64_clean` | 18G | alternate precision tier |
| `/home/nayte/models/qwen36_i4_gs64` | 22G | i4 tier (docs/qwen36.md quickstart) |
| `/home/nayte/models/qwen36_mixed_med` | 19G | alternate mix ratio |

## Recoverability note for Phase 8 task 20

The "missing" `QWEN36_FUSED_BATCH_PROMGATE_RECEIPT_20260823.md` is NOT missing:
it is tracked in the ANVIL repo at commit `2f41cdc`
(`docs(qwen36): promotion-gate factorial receipt + verifier handoff`), path
`docs/experiments/colibri-qwen36-a3b-node/QWEN36_FUSED_BATCH_PROMGATE_RECEIPT_20260823.md`.
It will be copied into the colibri tree during Phase 8 with attribution.
