#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "A3 Isaac Sim training requires Linux (Ubuntu 22.04 or newer)." >&2
  exit 1
fi

source "$ROOT_DIR/scripts/source_isaacsim_setup.sh"

NUM_ENVS=${NUM_ENVS:-4096}
SEED=${SEED:-1}

python "$ROOT_DIR/src/holosoma/holosoma/train_agent.py" \
  exp:a3-t3d0-wbt-fast-sac \
  --training.num-envs "$NUM_ENVS" \
  --training.seed "$SEED" \
  "$@"
