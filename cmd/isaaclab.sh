#!/bin/bash

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

export ROS_DOMAIN_ID=32

cd "$ISAACLAB_DIR" || exit 1

PLAY_ARGS=(
  --task Dofbot-Reach-IK-Direct-v0
  --use-ros-target
  --ros-target-topic /bottle/position
  --ros-target-msg pose_stamped
  --ros-target-frame env
  --ros-target-camera-is-base
  --ros-target-swap-xy
  --ros-target-flip-x
  --ros-target-xy-scale 1.0 1.0
  --ros-target-fixed-z 0.1
  --ros-target-clamp-range -0.25 0.25 -0.10 0.80 0.03 1.20
  --real-time
)

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  PLAY_ARGS+=(--checkpoint "$CHECKPOINT_PATH")
fi

./isaaclab.sh -p "$PROJECT_DIR/play_rsl_rl.py" "${PLAY_ARGS[@]}"
