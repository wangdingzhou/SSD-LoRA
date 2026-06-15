#!/usr/bin/env bash
# HA-LoRA Training Launch Script
# Usage: bash scripts/train.sh [config_path]

set -e

CONFIG="${1:-src/config.yaml}"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research

echo "=== HA-LoRA Training ==="
echo "Config: $CONFIG"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU')"
echo "Start time: $(date)"
echo ""

# Run inside screen for long training
if [ -z "$STY" ]; then
    echo "Not in a screen session. Starting one..."
    screen -dmS halora_train bash -c "python src/train.py --config $CONFIG 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log"
    echo "Training started in screen session 'halora_train'"
    echo "Attach with: screen -r halora_train"
else
    python src/train.py --config "$CONFIG" 2>&1 | tee "logs/train_$(date +%Y%m%d_%H%M%S).log"
fi
