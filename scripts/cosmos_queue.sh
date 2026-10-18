#!/usr/bin/env bash
set -euo pipefail
COSMOS_DIR="${COSMOS_DIR:-$HOME/work/Cosmos-Transfer2.5}"
COSMOS_INPUT_DIR="${COSMOS_INPUT_DIR:-$HOME/work/cosmos-outputs/episode000000_front_clip090}"
export COSMOS_DIR COSMOS_INPUT_DIR
mkdir -p "$COSMOS_INPUT_DIR"
status="$COSMOS_INPUT_DIR/queue-status.txt"
for attempt in $(seq 1 30); do
  date -Is > "$status"
  printf 'Checking VAE access, attempt %s/30\n' "$attempt" >> "$status"
  if "$COSMOS_DIR/.venv/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError
try:
    print(hf_hub_download('nvidia/Cosmos-Predict2.5-2B','tokenizer.pth',revision='f176dc95b4a70f53ce01c4b302851595e7322b00'))
except GatedRepoError:
    print('VAE access not yet granted')
    raise SystemExit(2)
PY
  then
    printf 'Weights ready; waiting for shared GPU lock / starting inference\n' >> "$status"
    if bash "$HOME/work/so101_sim_pickplace/scripts/cosmos_smoke.sh"; then
      printf 'INFERENCE SUCCEEDED %s\n' "$(date -Is)" >> "$status"
      exit 0
    else
      printf 'INFERENCE FAILED %s; see queue.log\n' "$(date -Is)" >> "$status"
      exit 1
    fi
  fi
  sleep 60
done
printf 'Timed out waiting for VAE access after30 attempts %s\n' "$(date -Is)" >> "$status"
exit 2
