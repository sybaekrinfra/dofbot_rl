#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/dofbot_rl}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
LOG_DIR="${LOG_DIR:-$ISAACLAB_DIR/logs/rsl_rl/dofbot_v2_pick_place}"
NUM_ENVS="${NUM_ENVS:-2048}"
PLAY_AFTER_EACH="${PLAY_AFTER_EACH:-1}"
PLAY_VIZ_MODE="${PLAY_VIZ_MODE:-kit}"
RUN_SANITY="${RUN_SANITY:-1}"

# Iterations added by each stage.  With RSL-RL's resume indexing these produce
# model_999 -> model_2998 -> model_6997.
REACH_ITERATIONS="${REACH_ITERATIONS:-1000}"
LIFT_ITERATIONS="${LIFT_ITERATIONS:-2000}"
PICK_PLACE_ITERATIONS="${PICK_PLACE_ITERATIONS:-4000}"

if [[ ! "$NUM_ENVS" =~ ^[0-9]+$ ]] || (( NUM_ENVS < 1 || NUM_ENVS > 2048 )); then
  echo "NUM_ENVS must be an integer in the safe range 1..2048 (got: $NUM_ENVS)" >&2
  exit 2
fi
for iterations in "$REACH_ITERATIONS" "$LIFT_ITERATIONS" "$PICK_PLACE_ITERATIONS"; do
  if [[ ! "$iterations" =~ ^[0-9]+$ ]] || (( iterations < 1 )); then
    echo "Every stage iteration count must be a positive integer (got: $iterations)" >&2
    exit 2
  fi
done
if [[ "$PLAY_AFTER_EACH" != "0" && "$PLAY_AFTER_EACH" != "1" ]]; then
  echo "PLAY_AFTER_EACH must be 0 or 1" >&2
  exit 2
fi
if [[ "$RUN_SANITY" != "0" && "$RUN_SANITY" != "1" ]]; then
  echo "RUN_SANITY must be 0 or 1" >&2
  exit 2
fi

stage_runs() {
  local stage="$1"
  if [[ ! -d "$LOG_DIR" ]]; then
    return 1
  fi
  find "$LOG_DIR" -mindepth 1 -maxdepth 1 -type d -name "*_${stage}" -printf '%f\n' \
    | sort -V
}

checkpoint_iteration() {
  local checkpoint_name
  checkpoint_name="$(basename -- "$1")"
  if [[ ! "$checkpoint_name" =~ ^model_([0-9]+)[.]pt$ ]]; then
    echo "Invalid numbered checkpoint name: $checkpoint_name" >&2
    return 1
  fi
  printf '%s\n' "${BASH_REMATCH[1]}"
}

run_deterministic_preflight() {
  local sanity_log
  if [[ "$RUN_SANITY" != "1" ]]; then
    echo "[sanity] skipped by RUN_SANITY=0"
    return
  fi

  sanity_log="$(mktemp /tmp/dofbot_sanity.XXXXXX.log)"
  echo "[sanity] deterministic full PickPlace preflight"
  set +e
  "$ISAACLAB_DIR/isaaclab.sh" -p "$PROJECT_DIR/cmd/sanity_grasp_lift.py" \
    --full_pick_place --device cuda:0 --viz none --log_interval 120 \
    2>&1 | tee "$sanity_log"
  local simulator_status="${PIPESTATUS[0]}"
  set -e
  if (( simulator_status != 0 )) || ! grep -Fq "[SANITY:PASS] PICK_PLACE" "$sanity_log"; then
    echo "[sanity] deterministic PickPlace failed; PPO will not start." >&2
    echo "[sanity] log: $sanity_log" >&2
    exit 1
  fi
  rm -f "$sanity_log"
}

play_stage() {
  local stage="$1"
  local task="$2"
  local checkpoint="$3"

  if [[ "$PLAY_AFTER_EACH" != "1" ]]; then
    return
  fi
  echo
  echo "[$stage] play: $checkpoint"
  echo "[$stage] Close the Isaac Sim window to continue to the next stage."
  TASK="$task" CHECKPOINT_PATH="$checkpoint" VIZ_MODE="$PLAY_VIZ_MODE" \
    bash "$PROJECT_DIR/cmd/play_pick_place.sh" --debug_interval 30
}

# Outputs are returned through these globals so the training process remains
# attached to the terminal instead of being hidden in command substitution.
TRAINED_RUN=""
TRAINED_CHECKPOINT=""

train_stage() {
  local stage="$1"
  local task="$2"
  local iterations="$3"
  local resume="$4"
  local source_run="${5:-}"
  local source_checkpoint="${6:-}"
  local after_run
  local source_iteration=0
  local expected_iteration
  local load_run_pattern=""
  local checkpoint_pattern=""
  local candidate
  local -a new_runs=()
  local -A runs_before=()

  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] && runs_before["$candidate"]=1
  done < <(stage_runs "$stage" || true)
  if [[ "$resume" == "1" ]]; then
    if [[ ! -d "$source_run" || ! -f "$source_checkpoint" ]]; then
      echo "[$stage] exact source run/checkpoint is missing" >&2
      exit 1
    fi
    source_iteration="$(checkpoint_iteration "$source_checkpoint")"
    load_run_pattern="^$(basename -- "$source_run")$"
    checkpoint_pattern="^model_${source_iteration}[.]pt$"
    expected_iteration=$(( source_iteration + iterations - 1 ))
  else
    expected_iteration=$(( iterations - 1 ))
  fi

  echo
  echo "[$stage] train: envs=$NUM_ENVS iterations=$iterations resume=$resume"
  if [[ "$resume" == "1" ]]; then
    echo "[$stage] exact parent: $source_checkpoint"
    STAGE="$stage" NUM_ENVS="$NUM_ENVS" MAX_ITERATIONS="$iterations" RESUME=1 \
      RUN_NAME="$stage" LOAD_RUN="$load_run_pattern" \
      CHECKPOINT_PATTERN="$checkpoint_pattern" VIZ_MODE=none \
      bash "$PROJECT_DIR/cmd/train_pick_place.sh"
  else
    STAGE="$stage" NUM_ENVS="$NUM_ENVS" MAX_ITERATIONS="$iterations" RESUME=0 \
      RUN_NAME="$stage" VIZ_MODE=none \
      bash "$PROJECT_DIR/cmd/train_pick_place.sh"
  fi

  while IFS= read -r candidate; do
    if [[ -n "$candidate" && ! -v 'runs_before[$candidate]' ]]; then
      new_runs+=("$candidate")
    fi
  done < <(stage_runs "$stage" || true)
  if (( ${#new_runs[@]} != 1 )); then
    echo "[$stage] expected exactly one new canonical run, found ${#new_runs[@]}: ${new_runs[*]:-(none)}" >&2
    exit 1
  fi
  after_run="${new_runs[0]}"
  TRAINED_RUN="$LOG_DIR/$after_run"
  TRAINED_CHECKPOINT="$TRAINED_RUN/model_${expected_iteration}.pt"
  if [[ ! -f "$TRAINED_CHECKPOINT" ]]; then
    echo "[$stage] training was interrupted or incomplete." >&2
    echo "[$stage] expected final checkpoint: $TRAINED_CHECKPOINT" >&2
    exit 1
  fi

  "$ISAACLAB_DIR/env_isaaclab/bin/python" \
    "$PROJECT_DIR/cmd/check_stage_training.py" \
    --run-dir "$TRAINED_RUN" \
    --stage "$stage" \
    --checkpoint "$TRAINED_CHECKPOINT" \
    --expected-iteration "$expected_iteration"

  play_stage "$stage" "$task" "$TRAINED_CHECKPOINT"
}

cd "$PROJECT_DIR"
run_deterministic_preflight

train_stage reach Dofbot-V2-PickPlace-Reach-Direct-v0 "$REACH_ITERATIONS" 0
reach_run="$TRAINED_RUN"
reach_checkpoint="$TRAINED_CHECKPOINT"

train_stage lift Dofbot-V2-PickPlace-Lift-Direct-v0 "$LIFT_ITERATIONS" 1 \
  "$reach_run" "$reach_checkpoint"
lift_run="$TRAINED_RUN"
lift_checkpoint="$TRAINED_CHECKPOINT"

train_stage pick_place Dofbot-V2-PickPlace-Direct-v0 "$PICK_PLACE_ITERATIONS" 1 \
  "$lift_run" "$lift_checkpoint"

echo
echo "Reach -> Lift -> PickPlace completed with exact checkpoint lineage."
echo "Final checkpoint: $TRAINED_CHECKPOINT"
