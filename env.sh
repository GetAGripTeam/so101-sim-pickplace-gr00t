#!/bin/bash
# Shared settings for the Isaac Lab SO-101 block pick-and-place -> GR00T N1.7 post-training pipeline.
#   source ~/Desktop/nvidia/so101_sim_pickplace/env.sh

# Resolve this checkout, with optional machine-specific overrides in env.local.sh.
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$PROJECT_DIR/env.local.sh" ]; then
    source "$PROJECT_DIR/env.local.sh"
fi
export GR00T_REPO="${GR00T_REPO:-$HOME/work/Isaac-GR00T}"
export ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/work/IsaacLab}"
export ISAACLAB_PY="${ISAACLAB_PY:-$HOME/work/env_isaaclab/bin/python}"

# Isaac Sim needs these (same as ~/Desktop/isaac/launch_isaacsim.sh)
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Task + dataset
# Block / pad colours must match the sim scene (CUBE_COLOR / PAD_COLOR in pickplace_env_cfg.py) and the real station.
export CUBE_COLOR="${CUBE_COLOR:-red}"
export PAD_COLOR="${PAD_COLOR:-green}"
export PAD_NAME="${PAD_NAME:-$PAD_COLOR}"   # word used in the task text (PAD_COLOR may be an "r,g,b" triple)
export TASK="${TASK:-pick up the $CUBE_COLOR block and place it on the $PAD_NAME pad}"
export DATASETS_DIR="${DATASETS_DIR:-$PROJECT_DIR/datasets}"
export DATASET_NAME="${DATASET_NAME:-so101_sim_pickplace}"
export DATASET_ROOT="${DATASET_ROOT:-$DATASETS_DIR/local/$DATASET_NAME}"
export NUM_EPISODES="${NUM_EPISODES:-50}"

# GR00T
export BASE_MODEL="${BASE_MODEL:-$GR00T_REPO/checkpoints/GR00T-N1.7-3B}"
export MODALITY_CONFIG=$PROJECT_DIR/gr00t/so101_sim_config.py
export EMBODIMENT_TAG=NEW_EMBODIMENT
export FINETUNE_OUTPUT="${FINETUNE_OUTPUT:-$PROJECT_DIR/finetune/$DATASET_NAME}"
export CHECKPOINT="${CHECKPOINT:-}"           # set to $FINETUNE_OUTPUT/checkpoint-<N> for start_server.sh
export POLICY_HOST="${POLICY_HOST:-localhost}"
export POLICY_PORT="${POLICY_PORT:-5555}"

# N1.7 loads the gated nvidia/Cosmos-Reason2-2B backbone from the Hub (do not set HF_HUB_OFFLINE)
unset HF_HUB_OFFLINE
export TOKENIZERS_PARALLELISM=false

gr00t_activate() {
    cd "$GR00T_REPO" || return 1
    source .venv/bin/activate || return 1
    if [ "$(uname -m)" = "aarch64" ] && [ -f scripts/activate_spark.sh ]; then
        source scripts/activate_spark.sh >/dev/null
        export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
}
