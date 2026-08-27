from __future__ import annotations

"""GPU FK search for a continuous, high-clearance DOFBOT transport path."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Search smooth joint waypoints that keep the grasp TCP above the table."
)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--iterations", type=int, default=36)
parser.add_argument("--min_vertical", type=float, default=0.88)
parser.add_argument(
    "--z_offset",
    type=float,
    default=0.0,
    help="Diagnostic height added to every nominal TCP waypoint in meters.",
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

    # Begin from the measured settled-lift branch, then move outward while
    # keeping the grasp point safely above the table.  The final TCP z=62 mm
    # corresponds to a cube center 15 mm above its resting height because the
    # calibrated physical grasp point is 4.5 mm above the cube center.
    targets_yz = (
        (0.130, 0.070),
        (0.142, 0.068),
        (0.154, 0.066),
        (0.166, 0.064),
        (0.176, 0.062),
        (0.184, 0.062),
    )
    previous_q = torch.tensor(
        [-0.005, 0.374, -1.480, -1.570], device=u.device
    )
    lower = u._controlled_lower[:4]
    upper = u._controlled_upper[:4]
    joint_pos = u.robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    generator = torch.Generator(device=u.device)
    generator.manual_seed(20260827)
    solutions: list[list[float]] = []

    print(
        f"[TRANSPORT_IK:START] num_envs={u.num_envs} "
        f"iterations={args_cli.iterations} z_offset={args_cli.z_offset:.4f}",
        flush=True,
    )

    for waypoint_index, (target_y, nominal_target_z) in enumerate(targets_yz):
        target_z = nominal_target_z + args_cli.z_offset
        print(
            f"[TRANSPORT_IK:SEARCH] waypoint={waypoint_index} "
            f"target=[0.0,{target_y:.3f},{target_z:.3f}]",
            flush=True,
        )
        target_local = torch.tensor([0.0, target_y, target_z], device=u.device)
        target_w = u.scene.env_origins + target_local
        best_q = previous_q.clone()
        best_cost = float("inf")
        best_position_error = float("inf")
        best_vertical = 0.0
        best_tcp = torch.zeros(3, device=u.device)

        for iteration in range(args_cli.iterations):
            progress = iteration / max(args_cli.iterations - 1, 1)
            sigma = 0.24 * (0.003 / 0.24) ** progress
            center = previous_q if iteration < 4 else best_q
            samples = center.unsqueeze(0) + sigma * torch.randn(
                (u.num_envs, 4), generator=generator, device=u.device
            )
            samples[0] = center
            samples[:, 0] *= 0.25
            # Stay on the requested negative-joint3 branch and retain a small
            # margin from the joint4 hard stop whenever kinematics permit it.
            samples[:, 2] = torch.minimum(samples[:, 2], torch.full_like(samples[:, 2], -0.10))
            samples = torch.clamp(samples, lower, upper)

            joint_pos[:, u._arm_joint_ids] = samples
            joint_pos[:, u._wrist_joint_id] = 0.0
            joint_pos[:, u._gripper_driver_joint_id] = 0.0
            joint_pos[:, u._gripper_mimic_joint_id] = 0.0
            u.robot.write_joint_state_to_sim(joint_pos, joint_vel)
            u.sim.forward()
            u.scene.update(0.0)

            tcp_w = u._gripper_center_w()
            body_quat = u.robot.data.body_quat_w.torch[:, u._grasp_body_id]
            local_z = torch.zeros((u.num_envs, 3), device=u.device)
            local_z[:, 2] = 1.0
            approach = quat_apply(body_quat, local_z)
            vertical = torch.abs(approach[:, 2])
            position_error = torch.linalg.norm(tcp_w - target_w, dim=-1)
            smoothness = torch.linalg.norm(samples - previous_q.unsqueeze(0), dim=-1)
            hard_vertical_cost = 0.30 * torch.relu(args_cli.min_vertical - vertical)
            # Position dominates; a very small smoothness term selects the
            # continuous solution when multiple IK branches are equivalent.
            cost = position_error + hard_vertical_cost + 0.0005 * smoothness
            index = int(torch.argmin(cost).item())
            candidate_cost = float(cost[index].item())
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_q = samples[index].clone()
                best_position_error = float(position_error[index].item())
                best_vertical = float(vertical[index].item())
                best_tcp = (tcp_w[index] - u.scene.env_origins[index]).clone()

        if best_position_error > 0.003 or best_vertical < args_cli.min_vertical:
            print(
                f"[TRANSPORT_IK:FAIL] waypoint={waypoint_index} "
                f"position={1000.0 * best_position_error:.2f}mm "
                f"vertical={best_vertical:.4f}",
                flush=True,
            )
            raise RuntimeError(
                f"waypoint {waypoint_index} infeasible: "
                f"position={1000.0 * best_position_error:.2f}mm "
                f"vertical={best_vertical:.4f}"
            )
        solutions.append(best_q.tolist())
        print(
            f"[TRANSPORT_IK:PASS] waypoint={waypoint_index} "
            f"target=[0.0,{target_y:.3f},{target_z:.3f}] "
            f"error={1000.0 * best_position_error:.2f}mm "
            f"vertical={best_vertical:.5f} "
            f"q={[round(float(v), 8) for v in best_q.tolist()]} "
            f"tcp={[round(float(v), 6) for v in best_tcp.tolist()]}",
            flush=True,
        )
        previous_q = best_q

    print(f"[TRANSPORT_IK:RESULT] {solutions}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
