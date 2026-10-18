#!/usr/bin/env bash
# Collect a target total of demos, validate them, then train GR00T.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/env.sh"
mkdir -p "$PROJECT_DIR/logs"
# Prevent two runs from writing the same dataset or training output.
exec 9>"$PROJECT_DIR/logs/train-pipeline.lock"
flock -n 9 || { echo "A training pipeline is already running." >&2; exit 1; }
STATUS_FILE="$PROJECT_DIR/logs/train-pipeline.status"
trap 'result=$?; if [ "$result" -ne 0 ]; then printf "failed (exit %s) at %s\n" "$result" "$(date -Is)" > "$STATUS_FILE"; fi' EXIT
stage() {
    printf '%s at %s\n' "$1" "$(date -Is)" | tee "$STATUS_FILE"
}
cd "$PROJECT_DIR"
TARGET_EPISODES="${TARGET_EPISODES:-100}"
CURRENT_EPISODES=$(python3 - "$DATASET_ROOT/meta/info.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
print(json.loads(path.read_text())["total_episodes"] if path.exists() else 0)
PY
)
if (( CURRENT_EPISODES < TARGET_EPISODES )); then
    stage collecting
    RESUME=()
    if (( CURRENT_EPISODES > 0 )); then RESUME=(--resume); fi
    NUM_EPISODES=$((TARGET_EPISODES - CURRENT_EPISODES)) ./collect.sh "${RESUME[@]}"
fi
python3 - "$DATASET_ROOT/meta/info.json" "$TARGET_EPISODES" <<'PY'
import json, sys
count = json.load(open(sys.argv[1]))["total_episodes"]
if count < int(sys.argv[2]):
    raise SystemExit(f"Only {count} episodes collected; target is {sys.argv[2]}")
PY
stage preparing
./prepare_dataset.sh
if [ "${TRAIN_AFTER_COLLECTION:-0}" != 1 ]; then
    stage awaiting_mixed_data
    echo "Collection validated. Add Cosmos and real data before main training."
    echo "For an explicit sim-only baseline, set TRAIN_AFTER_COLLECTION=1."
    exit 0
fi
stage training
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}" MAX_STEPS="${MAX_STEPS:-4000}" ./finetune.sh
stage complete
