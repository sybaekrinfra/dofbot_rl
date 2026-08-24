#!/bin/bash
set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TASK="${TASK:-Dofbot-V2-PickPlace-Direct-v0}"
VIZ_MODE="${VIZ_MODE:-kit}"

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  echo "Set CHECKPOINT_PATH to a trained model_*.pt file." >&2
  exit 2
fi

cd "$ISAACLAB_DIR"
./isaaclab.sh -p "$PROJECT_DIR/play_pick_place.py" \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT_PATH" \
  --num_envs 1 \
  --device cuda:0 \
  --viz "$VIZ_MODE" \
  "$@"
