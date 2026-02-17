#!/bin/bash
# Run Dual-Head Attention Model
# Optimized for RTX 4080 Super (16GB VRAM)

cd "$(dirname "$0")"

echo "=================================="
echo "Dual-Head Attention Pooling Model"
echo "Target: 0.81 → 0.90+ F1"
echo "=================================="
echo ""

python dual.py \
    --train_file ../train_rehydrated.jsonl \
    --dev_file ../dev_public.jsonl \
    --test_file ../test_rehydrated.jsonl \
    --output_dir outputs/dual_head \
    --predictions_file submission.jsonl

echo ""
echo "✅ Training complete!"
echo "Results in: outputs/dual_head/"
