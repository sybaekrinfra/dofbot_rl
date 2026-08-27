from __future__ import annotations

"""GPU forward-kinematics search for a safe DOFBOT pre-grasp IK seed."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Search a vertical pre-grasp seed with batched FK.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=32)
parser.add_argument("--object_x", type=float, default=0.0)
parser.add_argument("--object_y", type=float, default=0.13)
parser.add_argument(
    "--height_offset",
    type=float,
    default=0.055,
    help="Target height above the resting cube center (0.055=pre-grasp, 0=grasp).",
)
parser.add_argument(
    "--approach_sign",
    choices=("either", "up", "down"),
    default="either",
    help="Constrain Wrist_Twist local +Z to either world sign, or accept both.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply

import dofbot_rl.tasks  # noqa: F401
from dofbot_rl.tasks import PICK_PLACE_LIFT_ENV_ID
from dofbot_rl.tasks.dofbot_pick_place_cfg import DofbotPickPlaceLiftEnvCfg


def main() -> None:
    cfg = DofbotPickPlaceLiftEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args_cli.num_envs
    cfg.enable_finger_contact_sensors = False
    cfg.robot_cfg.spawn.activate_contact_sensors = False
    env = gym.make(PICK_PLACE_LIFT_ENV_ID, cfg=cfg, render_mode=None)
    u = env.unwrapped
    env.reset()

    generator = torch.Generator(device=u.device)
    generator.manual_seed(42)
    lower = u._controlled_lower[:4]
    upper = u._controlled_upper[:4]
    target_local = torch.tensor(
        [
            args_cli.object_x,
            args_cli.object_y,
            u.cfg.table_top_z + 0.5 * u.cfg.object_size + args_cli.height_offset,
        ],
        device=u.device,
    )
    target_w = u.scene.env_origins + target_local

    best_cost = float("inf")
    best_q = torch.zeros((4,), device=u.device)
    best_values: tuple[float, float, list[float]] | None = None
    best_valid_position = float("inf")
    best_valid_q = torch.zeros((4,), device=u.device)
    best_valid_values: tuple[float, float, list[float]] | None = None
    best_pure_position = float("inf")
    best_pure_values: tuple[float, float, list[float], list[float]] | None = None
    best_vertical_at_target = -1.0
    best_vertical_values: tuple[float, float, list[float], list[float]] | None = None
    joint_pos = u.robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)

    for iteration in range(args_cli.iterations):
        if iteration < 8 or best_cost == float("inf"):
            samples = lower + (upper - lower) * torch.rand(
                (u.num_envs, 4), generator=generator, device=u.device
            )
        else:
            progress = (iteration - 8) / max(args_cli.iterations - 9, 1)
            sigma = 0.45 * (0.025 / 0.45) ** progress
            samples = best_q.unsqueeze(0) + sigma * torch.randn(
                (u.num_envs, 4), generator=generator, device=u.device
            )
            samples = torch.clamp(samples, lower, upper)
        # Symmetric target x=0 strongly favors joint1 near zero.  Retain some
        # exploration while spending most samples on the arm's planar joints.
        samples[:, 0] *= 0.35

        joint_pos[:, u._arm_joint_ids] = samples
        joint_pos[:, u._wrist_joint_id] = 0.0
        joint_pos[:, u._gripper_driver_joint_id] = 0.0
        joint_pos[:, u._gripper_mimic_joint_id] = 0.0
        u.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        u.sim.forward()
        u.scene.update(0.0)

        grasp_pos = u._gripper_center_w()
        body_quat = u.robot.data.body_quat_w.torch[:, u._grasp_body_id]
        local_z = torch.zeros((u.num_envs, 3), device=u.device)
        local_z[:, 2] = 1.0
        approach = quat_apply(body_quat, local_z)
        position_error = torch.linalg.norm(grasp_pos - target_w, dim=-1)
        horizontal_axis_error = torch.linalg.norm(approach[:, :2], dim=-1)
        if args_cli.approach_sign == "up":
            vertical = approach[:, 2]
        elif args_cli.approach_sign == "down":
            vertical = -approach[:, 2]
        else:
            vertical = torch.abs(approach[:, 2])
        # A 0.20-m virtual lever gives vertical orientation comparable weight
        # to Cartesian error without hiding the actual position metric.
        cost = (
            position_error
            + 0.03 * horizontal_axis_error
            + 0.20 * torch.relu(0.70 - vertical)
        )
        index = int(torch.argmin(cost).item())
        candidate_cost = float(cost[index].item())
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_q = samples[index].clone()
            best_values = (
                float(position_error[index].item()),
                float(vertical[index].item()),
                (grasp_pos[index] - u.scene.env_origins[index]).tolist(),
            )
        pure_index = int(torch.argmin(position_error).item())
        pure_position = float(position_error[pure_index].item())
        if pure_position < best_pure_position:
            best_pure_position = pure_position
            best_pure_values = (
                pure_position,
                float(vertical[pure_index].item()),
                samples[pure_index].tolist(),
                (grasp_pos[pure_index] - u.scene.env_origins[pure_index]).tolist(),
            )
        valid = vertical >= 0.70
        if bool(valid.any()):
            valid_position = torch.where(
                valid, position_error, torch.full_like(position_error, float("inf"))
            )
            valid_index = int(torch.argmin(valid_position).item())
            candidate_position = float(valid_position[valid_index].item())
            if candidate_position < best_valid_position:
                best_valid_position = candidate_position
                best_valid_q = samples[valid_index].clone()
                best_valid_values = (
                    candidate_position,
                    float(vertical[valid_index].item()),
                    (grasp_pos[valid_index] - u.scene.env_origins[valid_index]).tolist(),
                )
        at_target = position_error <= 0.008
        if bool(at_target.any()):
            constrained_vertical = torch.where(
                at_target, vertical, torch.full_like(vertical, -1.0)
            )
            vertical_index = int(torch.argmax(constrained_vertical).item())
            candidate_vertical = float(constrained_vertical[vertical_index].item())
            if candidate_vertical > best_vertical_at_target:
                best_vertical_at_target = candidate_vertical
                best_vertical_values = (
                    float(position_error[vertical_index].item()),
                    candidate_vertical,
                    samples[vertical_index].tolist(),
                    (grasp_pos[vertical_index] - u.scene.env_origins[vertical_index]).tolist(),
                )
        if iteration % 4 == 0 or iteration == args_cli.iterations - 1:
            assert best_values is not None
            valid_text = "none"
            if best_valid_values is not None:
                valid_text = (
                    f"{1000.0 * best_valid_values[0]:.2f}mm@{best_valid_values[1]:.4f} "
                    f"q_deg={[round(v, 2) for v in torch.rad2deg(best_valid_q).tolist()]}"
                )
            assert best_pure_values is not None
            vertical_text = "none"
            if best_vertical_values is not None:
                vertical_text = (
                    f"{best_vertical_values[1]:.4f}@{1000.0 * best_vertical_values[0]:.2f}mm "
                    f"q_deg={[round(v, 2) for v in torch.rad2deg(torch.tensor(best_vertical_values[2])).tolist()]}"
                )
            print(
                f"[IK SEARCH] iteration={iteration:02d} cost={1000.0 * best_cost:.2f} "
                f"position={1000.0 * best_values[0]:.2f}mm vertical={best_values[1]:.5f} "
                f"q_rad={[round(v, 6) for v in best_q.tolist()]} "
                f"q_deg={[round(v, 3) for v in torch.rad2deg(best_q).tolist()]} "
                f"point_local={[round(v, 6) for v in best_values[2]]} "
                f"best_valid={valid_text} "
                f"best_pure={1000.0 * best_pure_values[0]:.2f}mm@{best_pure_values[1]:.4f} "
                f"max_vertical_at_8mm={vertical_text}",
                flush=True,
            )

    if best_valid_values is None or best_valid_values[0] > 0.008:
        raise RuntimeError(
            "No valid vertical pre-grasp seed: "
            + (
                "none"
                if best_valid_values is None
                else f"position={1000.0 * best_valid_values[0]:.2f}mm "
                f"vertical={best_valid_values[1]:.4f}"
            )
        )
    print(
        f"[IK SEARCH:PASS] q_rad={best_valid_q.tolist()} "
        f"position={1000.0 * best_valid_values[0]:.2f}mm "
        f"vertical={best_valid_values[1]:.5f}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
