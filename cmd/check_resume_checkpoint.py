#!/usr/bin/env python3
"""Resolve and reject incompatible or low-LR RSL-RL resume checkpoints."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch


def natural_key(name: str) -> list[int | str]:
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", name)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--load-run", required=True)
    parser.add_argument("--checkpoint-pattern", required=True)
    parser.add_argument("--expected-learning-rate", required=True, type=float)
    parser.add_argument("--expected-observations", required=True, type=int)
    parser.add_argument("--expected-actions", required=True, type=int)
    args = parser.parse_args()

    if not args.log_root.is_dir():
        raise SystemExit(f"[RESUME:FAIL] log root does not exist: {args.log_root}")
    try:
        run_pattern = re.compile(args.load_run)
        checkpoint_pattern = re.compile(args.checkpoint_pattern)
    except re.error as exc:
        raise SystemExit(f"[RESUME:FAIL] invalid regex: {exc}") from exc

    runs = sorted(
        (
            path
            for path in args.log_root.iterdir()
            if path.is_dir() and run_pattern.match(path.name)
        ),
        key=lambda path: natural_key(path.name),
    )
    if not runs:
        raise SystemExit(
            f"[RESUME:FAIL] no run matches {args.load_run!r} under {args.log_root}. "
            "Start a fresh Reach stage first."
        )
    run_dir = runs[-1]
    checkpoints = sorted(
        (
            path
            for path in run_dir.iterdir()
            if path.is_file() and checkpoint_pattern.match(path.name)
        ),
        key=lambda path: natural_key(path.name),
    )
    if not checkpoints:
        raise SystemExit(
            f"[RESUME:FAIL] no checkpoint matches {args.checkpoint_pattern!r} in {run_dir}"
        )
    checkpoint = checkpoints[-1]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    iteration = payload.get("iter")
    numbered_checkpoint = re.fullmatch(r"model_(\d+)[.]pt", checkpoint.name)
    if numbered_checkpoint is None:
        raise SystemExit(
            f"[RESUME:FAIL] checkpoint must have a numbered model_N.pt name: {checkpoint}"
        )
    filename_iteration = int(numbered_checkpoint.group(1))
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration != filename_iteration:
        raise SystemExit(
            "[RESUME:FAIL] checkpoint filename/internal iteration mismatch: "
            f"path={checkpoint}, filename_iter={filename_iteration}, payload_iter={iteration}"
        )
    optimizer = payload.get("optimizer_state_dict", {})
    learning_rates = [group.get("lr") for group in optimizer.get("param_groups", [])]
    if not learning_rates or any(
        lr is None
        or not math.isfinite(float(lr))
        or not math.isclose(
            float(lr), args.expected_learning_rate, rel_tol=1.0e-5, abs_tol=1.0e-8
        )
        for lr in learning_rates
    ):
        raise SystemExit(
            "[RESUME:FAIL] checkpoint optimizer LR is incompatible: "
            f"path={checkpoint}, lr={learning_rates}, expected={args.expected_learning_rate}. "
            "Do not resume an old adaptive-LR validation run; retrain fresh Reach."
        )

    actor = payload.get("actor_state_dict", {})
    mean = actor.get("obs_normalizer._mean")
    std_param = actor.get("distribution.std_param")
    observation_count = int(mean.shape[-1]) if hasattr(mean, "shape") else None
    action_count = int(std_param.shape[-1]) if hasattr(std_param, "shape") else None
    if observation_count != args.expected_observations or action_count != args.expected_actions:
        raise SystemExit(
            "[RESUME:FAIL] checkpoint policy shape is incompatible: "
            f"obs={observation_count}, actions={action_count}, "
            f"expected=({args.expected_observations}, {args.expected_actions}), path={checkpoint}"
        )

    print(
        f"[RESUME:PASS] checkpoint={checkpoint} iteration={iteration} "
        f"lr={learning_rates[0]:.8f} obs={observation_count} actions={action_count}"
    )


if __name__ == "__main__":
    main()
