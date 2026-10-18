#!/bin/bash
# Local web GUI for recording real SO-101 demos: live camera, buttons, joint bars, episode list, sim progress.
#   ./record_gui.sh                       # real arms + webcam, opens http://localhost:8765
#   ./record_gui.sh --demo                # simulated arms/camera (try the GUI without hardware)
#   TASK_TEXT="..." EPISODE_TIME_S=45 NUM_EPISODES=50 ./record_gui.sh
# Run it from Terminal.app (camera permission). Data: ~/so101_real_data/<date_time>/ (LeRobot v3.0).
# Set SIM_HOST=ubuntu@<vm-ip> (e.g. in robot.local.sh, which is not in git) to see the sim collection progress.
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./robot.local.sh
export PATH="$PWD/.venv/bin:$PATH"
export TASK_TEXT="${TASK_TEXT:-stack the medium, then the big, then the small blue block on the pink pad}"
exec .venv/bin/python scripts/record_gui.py "$@"
