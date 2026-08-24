#!/bin/bash
set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_PARENT="$(dirname -- "$PROJECT_DIR")"
STAGE="${STAGE:-reach}"
NUM_ENVS="${NUM_ENVS:-64}"
VIZ_MODE="${VIZ_MODE:-none}"

case "$STAGE" in
  reach)
    TASK="Dofbot-V2-PickPlace-Reach-Direct-v0"
    DEFAULT_ITERATIONS=1000
    DEFAULT_RESUME=0
    DEFAULT_LOAD_RUN=""
    ;;
  lift)
    TASK="Dofbot-V2-PickPlace-Lift-Direct-v0"
    DEFAULT_ITERATIONS=2000
    DEFAULT_RESUME=1
    DEFAULT_LOAD_RUN=".*_reach"
    ;;
  pick_place)
    TASK="Dofbot-V2-PickPlace-Direct-v0"
    DEFAULT_ITERATIONS=4000
    DEFAULT_RESUME=1
    DEFAULT_LOAD_RUN=".*_lift"
    ;;
  *)
    echo "STAGE must be one of: reach, lift, pick_place" >&2
    exit 2
    ;;
esac

MAX_ITERATIONS="${MAX_ITERATIONS:-$DEFAULT_ITERATIONS}"
RESUME="${RESUME:-$DEFAULT_RESUME}"
LOAD_RUN="${LOAD_RUN:-$DEFAULT_LOAD_RUN}"
CHECKPOINT_PATTERN="${CHECKPOINT_PATTERN:-model_.*.pt}"

export PYTHONPATH="$PROJECT_PARENT${PYTHONPATH:+:$PYTHONPATH}"

TRAIN_ARGS=(
  --rl_library rsl_rl
  --task "$TASK"
  --external_callback dofbot_rl.tasks.register_tasks
  --device cuda:0
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERATIONS"
  --run_name "$STAGE"
  --viz "$VIZ_MODE"
)

if [[ "$RESUME" == "1" ]]; then
  TRAIN_ARGS+=(--resume --load_run "$LOAD_RUN" --checkpoint "$CHECKPOINT_PATTERN")
fi

cd "$ISAACLAB_DIR"
./isaaclab.sh train "${TRAIN_ARGS[@]}" "$@"
