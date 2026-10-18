#!/bin/bash
# Run the fine-tuned policy on the real SO-101 (needs the VM server + an ssh tunnel, see scripts/run_real_policy.py).
#   ./run_policy.sh                 # live, 2 minutes max, Ctrl+C stops and parks the arm
#   ./run_policy.sh --no-robot      # only test the connection to the server
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./robot.local.sh
export PATH="$PWD/.venv/bin:$PATH"
exec .venv/bin/python scripts/run_real_policy.py "$@"
