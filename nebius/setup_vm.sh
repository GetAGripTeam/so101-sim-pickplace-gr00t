#!/bin/bash
# One-shot environment setup on a fresh Nebius GPU VM (Ubuntu 24.04 + CUDA image, x86_64, L40S or
# RTX PRO 6000). Installs Isaac Sim 6.1 + Isaac Lab 3.0 (pip, headless-capable) and Isaac-GR00T N1.7,
# then clones this repo. Run as the VM user:  bash setup_vm.sh
# Versions match the machine this project was developed on (Isaac Sim 6.1.0, Isaac Lab 3.0.0, Python 3.12).
set -euo pipefail
WORK="${WORK:-$HOME/work}"
REPO_URL="${REPO_URL:-https://github.com/ddiddi/so101-sim-pickplace-gr00t.git}"
ISAACLAB_REF="${ISAACLAB_REF:-472d78801a80d2c80a2c318fb3e9cdf6e187f9dd}"
GR00T_REF="${GR00T_REF:-dac54b313160c201c450f97df0fd74ebd379c892}"
if [ "$(uname -s)" != Linux ] || ! command -v nvidia-smi >/dev/null; then
    echo "Run this installer on the Linux NVIDIA GPU VM." >&2
    exit 1
fi
mkdir -p "$WORK" && cd "$WORK"

sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg python3.12-venv python3.12-dev build-essential cmake jq \
  libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 libvulkan1 vulkan-tools
git lfs install

# --- Isaac Sim 6.1 + compatible Isaac Lab source snapshot --------------------------------------
# This commit includes SO-101 contrib tasks and still targets Isaac Sim 6.1.
# Newer release/3.0.0 commits use 6.2; the published beta tag predates SO-101.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
[ -d IsaacLab/.git ] || git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout "$ISAACLAB_REF"
UV_PROJECT_ENVIRONMENT="$WORK/env_isaaclab" uv sync --python 3.12 --extra isaacsim
uv pip install --python "$WORK/env_isaaclab/bin/python" --no-deps -e source/isaaclab_teleop
uv pip install --python "$WORK/env_isaaclab/bin/python" pyzmq msgpack msgpack-numpy h5py imageio pyarrow
cd ..

# --- Isaac-GR00T N1.7 --------------------------------------------------------------------------
[ -d Isaac-GR00T/.git ] || git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
git checkout "$GR00T_REF"
git submodule update --init --recursive
uv sync --python 3.12
if uv run hf auth whoami >/dev/null 2>&1; then
    uv run hf download nvidia/GR00T-N1.7-3B --local-dir checkpoints/GR00T-N1.7-3B
    uv run hf download nvidia/Cosmos-Reason2-2B
else
    echo "Model download pending: run uv run hf auth login, then uv run hf download nvidia/GR00T-N1.7-3B --local-dir checkpoints/GR00T-N1.7-3B"
fi
cd ..

# --- this project --------------------------------------------------------------------------------
[ -d so101_sim_pickplace/.git ] || git clone "$REPO_URL" so101_sim_pickplace
cat > so101_sim_pickplace/env.local.sh <<EOT
export ISAACLAB_PY="$WORK/env_isaaclab/bin/python"
export ISAACLAB_DIR="$WORK/IsaacLab"
export GR00T_REPO="$WORK/Isaac-GR00T"
EOT
printf '\nStack installed. Project: %s/so101_sim_pickplace\n' "$WORK"
