#!/bin/bash
# Run ANE training on Apple Neural Engine (Apple Silicon M4)
# This uses reverse-engineered ANE private APIs for transformer training.
# WARNING: Uses undocumented APIs that may break with macOS updates.

set -e

# Setup
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Ensure venv
command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate

# Build ANE bridge if needed
ANE_BRIDGE="../ane-training/bridge/libane_bridge.dylib"
if [ ! -f "$ANE_BRIDGE" ]; then
    echo "Building ANE bridge..."
    make -C ../ane-training/bridge
fi

# Train a tiny model on ANE with synthetic data
# depth=1, dim=64 is the smallest feasible config
# The compile budget (~90 kernels) limits training length
python -m scripts.ane_train \
    --depth=1 \
    --dim=64 \
    --heads=2 \
    --seq-len=32 \
    --vocab-size=256 \
    --lr=3e-4 \
    --num-batches=6 \
    --accum-steps=10

echo ""
echo "To train with real data, first tokenize:"
echo "  python -m nanochat.dataset -n 8"
echo "  python -m scripts.tok_train --max-chars=2000000000"
echo "Then:"
echo "  python -m scripts.ane_train --depth=2 --dim=128 --data-path=\$NANOCHAT_BASE_DIR/tok/data.bin"
