from __future__ import annotations

"""Probe the local DOFBOT lift kinematics around the collision-free grasp pose."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply, skew_symmetric_matrix

import dofbot_rl.tasks  # noqa: F401
from dofbot_rl.tasks import PICK_PLACE_LIFT_ENV_ID
from dofbot_rl.tasks.dofbot_pick_place_cfg import DofbotPickPlaceLiftEnvCfg


def main() -> None:
    cfg = DofbotPickPlaceLiftEnvCfg()
    cfg.scene.num_envs = 8
    cfg.enable_finger_contact_sensors = False
    cfg.robot_cfg.spawn.activate_contact_sensors = False
    env = gym.make(PICK_PLACE_LIFT_ENV_ID, cfg=cfg, render_mode=None)
    u = env.unwrapped
    env.reset()

    q0 = torch.tensor(
        [-0.00534522, -0.01464671, -1.31024933, -1.57079633],
        device=u.device,
    )
    perturbations = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.05],
            [0.0, 0.0, 0.0, 0.10],
            [0.0, 0.05, 0.0, 0.0],
            [0.0, 0.0, 0.05, 0.0],
            [0.0, 0.05, -0.05, 0.05],
            [0.0, -0.05, 0.05, 0.05],
            [0.0, -0.05, -0.05, 0.10],
        ],
        device=u.device,
    )
    q = q0.unsqueeze(0) + perturbations
    joint_pos = u.robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, u._arm_joint_ids] = q
    joint_pos[:, u._wrist_joint_id] = 0.0
    joint_pos[:, u._gripper_driver_joint_id] = 0.0
    joint_pos[:, u._gripper_mimic_joint_id] = 0.0
    u.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    u.sim.forward()
    u.scene.update(0.0)

    tcp = u._gripper_center_w() - u.scene.env_origins
    quat = u.robot.data.body_quat_w.torch[:, u._grasp_body_id]
    local_z = torch.zeros((u.num_envs, 3), device=u.device)
    local_z[:, 2] = 1.0
    approach = quat_apply(quat, local_z)
    angle = torch.atan2(-approach[:, 1], approach[:, 2])
    for index in range(u.num_envs):
        print(
            f"[KIN] i={index} q={[round(v, 6) for v in q[index].tolist()]} "
            f"tcp={[round(v, 6) for v in tcp[index].tolist()]} "
            f"approach={[round(v, 6) for v in approach[index].tolist()]} "
            f"angle={float(angle[index]):+.6f} vertical={abs(float(approach[index, 2])):.6f}",
            flush=True,
        )

    arm_ids = list(u._arm_joint_ids)
    columns = [joint_id + u.robot.num_base_dofs for joint_id in arm_ids]
    jacobian_body = u._grasp_body_id - 1 if u.robot.is_fixed_base else u._grasp_body_id
    jacobian = u.robot.data.body_link_jacobian_w.torch[
        0:1, jacobian_body, :, columns
    ].clone()
    body_pos = u.robot.data.body_pos_w.torch[0:1, u._grasp_body_id]
    offset_w = tcp[0:1] + u.scene.env_origins[0:1] - body_pos
    jacobian[:, :3, :] -= torch.bmm(
        skew_symmetric_matrix(offset_w), jacobian[:, 3:, :]
    )
    print(f"[KIN:JPOS] {jacobian[0, :3].tolist()}", flush=True)
    print(f"[KIN:JANG_X] {jacobian[0, 3].tolist()}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
