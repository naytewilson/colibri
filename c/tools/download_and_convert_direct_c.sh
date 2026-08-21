#!/bin/bash
set -e

SOURCE_DIR="/data/ANVIL/models/hf/source_shards"
DEST_DIR="/data/ANVIL/models/qwen36_i3_gs64_clean"
CONVERTER="/home/nayte/ane-hot/colibri-qwen36/c/tools/convert_qwen36_direct_c"
TOKEN_FILE="$HOME/.cache/huggingface/token"
HF_REPO="https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main"

mkdir -p "$SOURCE_DIR"
mkdir -p "$DEST_DIR"

if [ ! -f "$TOKEN_FILE" ]; then
    echo "Error: Token file $TOKEN_FILE not found."
    exit 1
fi
HF_TOKEN=$(cat "$TOKEN_FILE")

echo "=== 1. Downloading Metadata & Configs ==="
curl -s -L -H "Authorization: Bearer $HF_TOKEN" -C - "$HF_REPO/config.json" -o "$SOURCE_DIR/config.json"
curl -s -L -H "Authorization: Bearer $HF_TOKEN" -C - "$HF_REPO/tokenizer.json" -o "$SOURCE_DIR/tokenizer.json"
curl -s -L -H "Authorization: Bearer $HF_TOKEN" -C - "$HF_REPO/model.safetensors.index.json" -o "$SOURCE_DIR/model.safetensors.index.json"

cp "$SOURCE_DIR/config.json" "$DEST_DIR/config.json"
cp "$SOURCE_DIR/tokenizer.json" "$DEST_DIR/tokenizer.json"

# Write qwen36_meta.json
cat << 'EOF' > "$DEST_DIR/qwen36_meta.json"
{
  "hidden_size": 2048,
  "num_hidden_layers": 40,
  "vocab_size": 248320,
  "num_experts": 256,
  "topk": 8,
  "moe_inter": 512,
  "shared_inter": 512,
  "q_heads": 16,
  "kv_heads": 2,
  "head_dim": 256,
  "q_head_dim": 512,
  "k_head_dim": 256,
  "v_head_dim": 256,
  "o_in": 4096,
  "qk_rope_head_dim": 64,
  "partial_rotary_factor": 0.25,
  "rope_theta": 10000000.0,
  "rms_eps": 0.000001,
  "attn_output_gate": 1,
  "has_qk_norm": 1,
  "dn_vheads": 32,
  "dn_kheads": 16,
  "dn_kdim": 128,
  "dn_vdim": 128,
  "dn_convk": 4,
  "dn_conv_dim": 8192,
  "ebits": 3,
  "expert_fmt": 5,
  "expert_gs": 64
}
EOF

echo "=== 2. Downloading 26 Source BF16 Shards & Converting in C ==="
for i in $(seq 1 26); do
    SHARD=$(printf "model-%05d-of-00026.safetensors" $i)
    TARGET_FILE="$SOURCE_DIR/$SHARD"
    echo "[download] fetching $SHARD..."
    curl -L -H "Authorization: Bearer $HF_TOKEN" -C - "$HF_REPO/$SHARD" -o "$TARGET_FILE"
    echo "[download] $SHARD complete ($(ls -lh "$TARGET_FILE" | awk '{print $5}'))"
done

echo "=== 3. Executing Pure Native C Direct BF16 -> INT3-g64 Conversion ==="
"$CONVERTER" "$SOURCE_DIR" "$DEST_DIR"

echo "=== 4. Cleaning Up Source BF16 Shards ==="
rm -rf "$SOURCE_DIR"/*.safetensors

echo "=== 5. Running Full Evaluation & Benchmark Campaign ==="
/home/nayte/ane-hot/colibri-qwen36/c/tools/run_full_eval_campaign.sh

echo "=== PIPELINE COMPLETE ==="
