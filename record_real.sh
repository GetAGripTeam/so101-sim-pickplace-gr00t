#!/bin/bash
# Record real SO-101 demos (leader arm + front webcam) into a NEW LeRobot dataset folder on every start.
#   ./record_real.sh                 # 20 episodes x 20 s
#   NUM_EPISODES=50 EPISODE_TIME_S=25 ./record_real.sh
# Data lands in ~/so101_real_data/<date_time>/ (v3.0 format; convert with Isaac-GR00T's
# scripts/lerobot_conversion/convert_v3_to_v2.py before training). Nothing is pushed to Hugging Face.
# A Rerun window opens with the live camera image and the leader/follower joint curves.
# Keys while recording: Right arrow = next episode, Left arrow = discard + redo current episode, Esc = stop
# (best during the reset pause; Esc mid-episode also SAVES the running episode). Ctrl+C keeps finished episodes.
# A failed start (motor handshake hiccup, camera busy) leaves no data behind and is retried automatically.
set -uo pipefail
# Ctrl+C stops everything (otherwise the "start failed, retry" loop below would restart the recorder).
trap 'echo; echo "Interrupted - stopped."; exit 130' INT
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./robot.local.sh

RECORD_BIN="${RECORD_BIN:-.venv/bin/lerobot-record}"
OUT_BASE="${OUT_BASE:-$HOME/so101_real_data}"
NUM_EPISODES="${NUM_EPISODES:-20}"
EPISODE_TIME_S="${EPISODE_TIME_S:-20}"
RESET_TIME_S="${RESET_TIME_S:-10}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
MAX_TRIES="${MAX_TRIES:-5}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"   # live viewer (Rerun): camera image + joint curves; DISPLAY_DATA=false to turn off
TASK_TEXT="${TASK_TEXT:-pick up the blue block and place it on the pink pad}"  # must match the sim task text
mkdir -p "$OUT_BASE"

# Delete leftovers of failed starts: folders that only contain meta/ and no recorded data.
for d in "$OUT_BASE"/* "$HOME"/so101_real_[0-9]*; do
    [ -d "$d" ] || continue
    [ -z "$(find "$d" -type f ! -path '*/meta/*' | head -1)" ] && [ "$(find "$d" -type f | wc -l)" -le 1 ] && rm -r "$d" && echo "removed empty leftover $d"
done

for try in $(seq 1 "$MAX_TRIES"); do
    ROOT="$OUT_BASE/$(date +%Y%m%d_%H%M%S)"
    echo "== attempt $try/$MAX_TRIES -> $ROOT"
    "$RECORD_BIN" \
        --robot.type=so101_follower --robot.port="$ROBOT_PORT" --robot.id="$ROBOT_ID" \
        --robot.max_relative_target=30.0 --robot.use_degrees=false \
        --robot.cameras="{front: {type: opencv, index_or_path: $CAMERA_INDEX, width: 640, height: 480, fps: 30}}" \
        --teleop.type=so101_leader --teleop.port="$TELEOP_PORT" --teleop.id="$TELEOP_ID" --teleop.use_degrees=false \
        --dataset.repo_id=local/so101_real_pickplace --dataset.root="$ROOT" \
        --dataset.push_to_hub=false --dataset.single_task="$TASK_TEXT" \
        --dataset.num_episodes="$NUM_EPISODES" --dataset.episode_time_s="$EPISODE_TIME_S" \
        --dataset.reset_time_s="$RESET_TIME_S" --display_data=$DISPLAY_DATA
    status=$?
    if [ -n "$(find "$ROOT" -type f ! -path '*/meta/*' 2>/dev/null | head -1)" ]; then
        echo "Recorded data is in $ROOT (exit $status)"; exit "$status"
    fi
    rm -rf "$ROOT"
    [ "$status" -eq 0 ] && { echo "Finished without recording any episode."; exit 0; }
    echo "start failed (exit $status), no data recorded - retrying in 3 s"; sleep 3
done
echo "Giving up after $MAX_TRIES attempts. Check the follower's power/cable and the camera." >&2
exit 1
