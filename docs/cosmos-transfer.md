# Cosmos Transfer 2.5 on the hackathon VM

The upstream repository is isolated at `/home/ubuntu/work/Cosmos-Transfer2.5` on `ubuntu@<VM-IP>`. It uses its own `.venv`, leaving Isaac Lab and GR00T environments intact.

Installation uses the official CUDA 12.8 option with **Python 3.10** (the upstream flash-attention wheel has ABI cp310):

```bash
cd ~/work/Cosmos-Transfer2.5
~/.local/bin/uv sync --python 3.10 --extra cu128
```

Install log: `~/work/cosmos-install.log`.

## Access prerequisites

The cached Hugging Face account must accept the model licenses:

- https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B
- https://huggingface.co/nvidia/Cosmos-Guardrail1
- https://huggingface.co/nvidia/Cosmos-Predict2.5-2B (required VAE/tokenizer)

The Transfer2.5 license is now accepted and the distilled checkpoint is cached. Guardrail1 and Reason1 text encoder snapshots are also cached. The remaining access blocker is the Predict2.5 VAE (`tokenizer.pth`), which currently returns HTTP 403 (`GatedRepoError`). Access to Cosmos Reason does not grant access to Transfer or Predict.

## Smoke augmentation

The prepared clip is at `~/work/cosmos-outputs/episode000000_front_clip090/input.mp4`. It contains exactly 93 consecutive frames, original episode indices 90–182, with the original 30fps timing. `provenance.json` records source SHA256, frame indices/timestamps, checkpoint revision, prompt and seed. The original dataset and actions are untouched.

```bash
bash ~/work/so101_sim_pickplace/scripts/cosmos_smoke.sh
```

The wrapper selects `--model=edge/distilled` and four diffusion steps, enables the upstream experimental checkpoint catalog, and takes `~/.cache/so101/gpu.lock`, shared with GR00T training. Edge conditioning is computed from the source clip by the model pipeline. Generated results stay in the separate `generated` directory.

This smoke clip is not a complete augmented episode and is not included in training. Before creating a training dataset, process full episodes with frame-aligned chunks, verify geometry/motion preservation, preserve both camera schemas and action indices, and record counts in the dataset manifest. The appearance prompt is generic until a real station reference is available.

Official docs: https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md
