#!/usr/bin/env bash
set -euo pipefail
COSMOS_DIR="${COSMOS_DIR:-$HOME/work/Cosmos-Transfer2.5}"
COSMOS_INPUT_DIR="${COSMOS_INPUT_DIR:-$HOME/work/cosmos-outputs/episode000000_front_clip090}"
# Cosmos checkpoint downloads shell out to `uvx`, which lives in ~/.local/bin.
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/.cache/so101"
cd "$COSMOS_DIR"
export COSMOS_EXPERIMENTAL_CHECKPOINTS=1
# Same lock as GR00T training; downloads/setup should happen before this command.
exec flock "$HOME/.cache/so101/gpu.lock" .venv/bin/python examples/inference.py \
  -i "$COSMOS_INPUT_DIR/spec.json" -o "$COSMOS_INPUT_DIR/generated" --model=edge/distilled
