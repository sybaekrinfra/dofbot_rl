#!/bin/bash
set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_PARENT="$(dirname -- "$PROJECT_DIR")"
STAGE="${STAGE:-reach}"
NUM_ENVS="${NUM_ENVS:-1024}"
VIZ_MODE="${VIZ_MODE:-none}"

if [[ ! "$NUM_ENVS" =~ ^[0-9]+$ ]] || (( NUM_ENVS < 1 || NUM_ENVS > 1024 )); then
  echo "NUM_ENVS must be an integer in the safe range 1..1024 (got: $NUM_ENVS)" >&2
  exit 2
fi

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
    DEFAULT_LOAD_RUN=".*_reach$"
    ;;
  pick_place)
    TASK="Dofbot-V2-PickPlace-Direct-v0"
    DEFAULT_ITERATIONS=4000
    DEFAULT_RESUME=1
    DEFAULT_LOAD_RUN=".*_lift$"
    ;;
  *)
    echo "STAGE must be one of: reach, lift, pick_place" >&2
    exit 2
    ;;
esac

MAX_ITERATIONS="${MAX_ITERATIONS:-$DEFAULT_ITERATIONS}"
RESUME="${RESUME:-$DEFAULT_RESUME}"
LOAD_RUN="${LOAD_RUN:-$DEFAULT_LOAD_RUN}"
CHECKPOINT_PATTERN="${CHECKPOINT_PATTERN:-model_[0-9]+[.]pt$}"
RUN_NAME="${RUN_NAME:-$STAGE}"

if [[ ! "$MAX_ITERATIONS" =~ ^[0-9]+$ ]] || (( MAX_ITERATIONS < 1 )); then
  echo "MAX_ITERATIONS must be a positive integer (got: $MAX_ITERATIONS)" >&2
  exit 2
fi
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "RESUME must be 0 or 1 (got: $RESUME)" >&2
  exit 2
fi
if [[ "$RESUME" == "1" && -z "$LOAD_RUN" ]]; then
  echo "LOAD_RUN is required when RESUME=1" >&2
  exit 2
fi

# Keep the validated environment count, task, stage lineage and logging path
# authoritative.  argparse accepts duplicate options and the final occurrence
# wins, so forwarding one of these through "$@" would otherwise bypass the
# checks above (for example: --num_envs 2048).
for extra_arg in "$@"; do
  case "$extra_arg" in
    --rl_library|--rl_library=*|\
    --task|--task=*|\
    --agent|--agent=*|\
    --external_callback|--external_callback=*|\
    --device|--device=*|\
    --num_envs|--num_envs=*|*num_envs=*|\
    --max_iterations|--max_iterations=*|*max_iterations=*|\
    --experiment_name|--experiment_name=*|*experiment_name=*|\
    --run_name|--run_name=*|*run_name=*|\
    --resume|--resume=*|*resume=*|\
    --load_run|--load_run=*|*load_run=*|\
    --checkpoint|--checkpoint=*|*load_checkpoint=*|\
    --viz|--viz=*|--headless)
      echo "Protected training option must be set through STAGE/NUM_ENVS/MAX_ITERATIONS/RESUME/RUN_NAME/VIZ_MODE: $extra_arg" >&2
      exit 2
      ;;
  esac
done

export PYTHONPATH="$PROJECT_PARENT${PYTHONPATH:+:$PYTHONPATH}"

TRAIN_ARGS=(
  --rl_library rsl_rl
  --task "$TASK"
  --external_callback dofbot_rl.tasks.register_tasks
  --device cuda:0
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERATIONS"
  --run_name "$RUN_NAME"
  --viz "$VIZ_MODE"
)

if [[ "$RESUME" == "1" ]]; then
  "$ISAACLAB_DIR/env_isaaclab/bin/python" \
    "$PROJECT_DIR/cmd/check_resume_checkpoint.py" \
    --log-root "$ISAACLAB_DIR/logs/rsl_rl/dofbot_v2_pick_place" \
    --load-run "$LOAD_RUN" \
    --checkpoint-pattern "$CHECKPOINT_PATTERN" \
    --expected-learning-rate 0.0003 \
    --expected-observations 70 \
    --expected-actions 6
  TRAIN_ARGS+=(--resume --load_run "$LOAD_RUN" --checkpoint "$CHECKPOINT_PATTERN")
fi

cd "$ISAACLAB_DIR"
train_log="$(mktemp /tmp/dofbot_train.XXXXXX.log)"
set +e
./isaaclab.sh train "${TRAIN_ARGS[@]}" "$@" 2>&1 | tee "$train_log"
simulator_status="${PIPESTATUS[0]}"
set -e
if (( simulator_status != 0 )) || ! grep -Fq "Training time:" "$train_log"; then
  echo "Training was interrupted or did not complete normally." >&2
  echo "Training log preserved at: $train_log" >&2
  exit 1
fi
rm -f "$train_log"
