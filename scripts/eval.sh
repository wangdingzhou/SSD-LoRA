#!/usr/bin/env bash
# HA-LoRA Evaluation Launch Script
# Usage: bash scripts/eval.sh [config_path] [checkpoint_path]

set -e

CONFIG="${1:-src/config.yaml}"
CHECKPOINT="${2:-}"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research

echo "=== HA-LoRA Evaluation ==="
echo "Config: $CONFIG"

CMD="python src/eval.py --config $CONFIG"
if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
    echo "Checkpoint: $CHECKPOINT"
fi

echo ""

eval $CMD
