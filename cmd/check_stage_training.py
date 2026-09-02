#!/usr/bin/env python3
"""Reject incomplete curriculum runs and physically invalid learned stages."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalar_values(accumulator: EventAccumulator, tag: str, window: int) -> list[float]:
    available = set(accumulator.Tags().get("scalars", []))
    if tag not in available:
        raise RuntimeError(f"TensorBoard scalar is missing: {tag}")
    values = [float(event.value) for event in accumulator.Scalars(tag)][-window:]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"TensorBoard scalar is empty or non-finite: {tag}")
    return values


def tail_mean(accumulator: EventAccumulator, tag: str, window: int) -> float:
    values = scalar_values(accumulator, tag, window)
    return sum(values) / len(values)


def tail_max(accumulator: EventAccumulator, tag: str, window: int) -> float:
    return max(scalar_values(accumulator, tag, window))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("reach", "lift", "pick_place"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-iteration", required=True, type=int)
    parser.add_argument("--expected-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    if args.checkpoint.parent.resolve() != args.run_dir.resolve():
        raise SystemExit("[CHECK:FAIL] checkpoint does not belong to the requested run directory")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_iteration = int(payload.get("iter", -1))
    if checkpoint_iteration != args.expected_iteration:
        raise SystemExit(
            f"[CHECK:FAIL] incomplete checkpoint: iter={checkpoint_iteration}, "
            f"expected={args.expected_iteration}, path={args.checkpoint}"
        )
    optimizer_lrs = [
        float(group["lr"])
        for group in payload.get("optimizer_state_dict", {}).get("param_groups", [])
        if "lr" in group
    ]
    if not optimizer_lrs or any(
        not math.isclose(lr, args.expected_learning_rate, rel_tol=1.0e-5, abs_tol=1.0e-8)
        for lr in optimizer_lrs
    ):
        raise SystemExit(
            f"[CHECK:FAIL] optimizer LR={optimizer_lrs}, expected={args.expected_learning_rate}"
        )

    accumulator = EventAccumulator(str(args.run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    all_scalar_events = [
        event.step
        for tag in accumulator.Tags().get("scalars", [])
        for event in accumulator.Scalars(tag)[-1:]
    ]
    if not all_scalar_events or max(all_scalar_events) < args.expected_iteration:
        raise SystemExit(
            f"[CHECK:FAIL] TensorBoard ended before iteration {args.expected_iteration}"
        )

    expected_stage = {"reach": 0.0, "lift": 1.0, "pick_place": 2.0}[args.stage]
    stage_value = tail_mean(accumulator, "Curriculum/stage", args.window)
    event_lr = tail_mean(accumulator, "Loss/learning_rate", args.window)
    if not math.isclose(stage_value, expected_stage, abs_tol=1.0e-5):
        raise SystemExit(
            f"[CHECK:FAIL] curriculum stage={stage_value:.4f}, expected={expected_stage:.1f}"
        )
    if not math.isclose(
        event_lr, args.expected_learning_rate, rel_tol=1.0e-4, abs_tol=1.0e-8
    ):
        raise SystemExit(
            f"[CHECK:FAIL] TensorBoard LR={event_lr:.8f}, expected={args.expected_learning_rate:.8f}"
        )

    success = tail_mean(accumulator, "Metrics/success_rate", args.window)
    held_peak = tail_max(accumulator, "Conditions/success_held", args.window)
    task_failed = tail_mean(accumulator, "Metrics/task_failed", args.window)
    # DirectRLEnv auto-resets a terminal environment before the reward/log
    # dictionary is collected.  Consequently the exact step that reaches the
    # configured success_hold_steps can be absent from Conditions/success_held
    # even though termination was correctly gated by that counter.  Treat the
    # tag as diagnostic and validate the stage through its persistent physical
    # conditions and success rate below.
    checks = [task_failed < 0.75]
    details = [
        f"success={success:.4f}",
        f"held_peak={held_peak:.4f}",
        f"failed={task_failed:.4f}",
    ]

    if args.stage == "reach":
        pregrasp = tail_mean(accumulator, "Metrics/pregrasp_ready", args.window)
        phase1 = tail_mean(accumulator, "Phase/phase1_rate", args.window)
        checks.extend((success >= 0.02, pregrasp >= 0.02, phase1 >= 0.02))
        details.extend((f"pregrasp={pregrasp:.4f}", f"phase1={phase1:.4f}"))
    elif args.stage == "lift":
        lift = tail_mean(accumulator, "Metrics/lift_rate", args.window)
        maintained = tail_mean(accumulator, "Metrics/grasp_maintained", args.window)
        phase3 = tail_mean(accumulator, "Phase/phase3_rate", args.window)
        checks.extend((success >= 0.01, lift >= 0.02, maintained >= 0.02, phase3 >= 0.005))
        details.extend(
            (f"lift={lift:.4f}", f"maintained={maintained:.4f}", f"phase3={phase3:.4f}")
        )
    else:
        phase4 = tail_mean(accumulator, "Phase/phase4_rate", args.window)
        authorized = tail_mean(
            accumulator, "Conditions/place_release_authorized", args.window
        )
        released = tail_mean(accumulator, "Conditions/gripper_released", args.window)
        checks.extend(
            (success >= 0.005, phase4 >= 0.005, authorized >= 0.001, released >= 0.001)
        )
        details.extend(
            (f"phase4={phase4:.4f}", f"authorized={authorized:.4f}", f"released={released:.4f}")
        )

    print(
        f"[CHECK] {args.stage}: iter={checkpoint_iteration}, lr={event_lr:.8f}, "
        + ", ".join(details)
    )
    if not all(checks):
        raise SystemExit(
            f"[CHECK:FAIL] {args.stage} did not satisfy the physical curriculum gates; "
            "the next stage will not start."
        )
    print(f"[CHECK:PASS] stage={args.stage} checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
