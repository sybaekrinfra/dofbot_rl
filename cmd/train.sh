#!/bin/bash

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_PARENT="$(dirname -- "$PROJECT_DIR")"
NUM_ENVS="${NUM_ENVS:-16}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"

export PYTHONPATH="$PROJECT_PARENT${PYTHONPATH:+:$PYTHONPATH}"

cd "$ISAACLAB_DIR" || exit 1

./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Dofbot-Reach-IK-Direct-v0 \
  --external_callback dofbot_rl.tasks.register_tasks \
  --device cuda:0 \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --viz none
