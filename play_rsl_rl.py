from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for source_dir in ("isaaclab", "isaaclab_rl", "isaaclab_tasks"):
    source_path = PROJECT_ROOT / "source" / source_dir
    if source_path.is_dir() and str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def _ros_args_requested() -> bool:
    ros_flags = (
        "--use-ros-target",
        "--ros-target-topic",
        "--ros-publish-joint-states",
        "--ros-publish-gripper-control",
    )
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv for flag in ros_flags)


def _selected_ros_distro() -> str:
    """Return the ROS distribution supported by this Isaac Sim installation."""
    ros_distro = os.environ.get("ROS_DISTRO", "jazzy")
    return ros_distro if ros_distro in ("jazzy", "humble") else "jazzy"


def _find_isaacsim_ros2_core_root(ros_distro: str) -> Path | None:
    isaacsim_path = os.environ.get("ISAACSIM_PATH")
    candidates = []
    if isaacsim_path:
        candidates.append(Path(isaacsim_path) / "exts" / "isaacsim.ros2.core" / ros_distro)
    candidates.extend(
        (
            Path.home() / "isaacsim" / "exts" / "isaacsim.ros2.core" / ros_distro,
            Path.home() / "IsaacLab" / "_isaac_sim" / "exts" / "isaacsim.ros2.core" / ros_distro,
        )
    )
    for bridge_root in candidates:
        if (bridge_root / "rclpy").is_dir() and (bridge_root / "lib").is_dir():
            return bridge_root
    return None


def _prepare_ros2_env_before_python_start() -> None:
    """Re-exec with Isaac Sim 6's ROS2 core paths visible to Python and the linker."""
    if not _ros_args_requested() or os.environ.get("DOFBOT_ROS2_ENV_READY") == "1":
        return

    for idx, arg in enumerate(sys.argv):
        if arg == "--ros-domain-id" and idx + 1 < len(sys.argv):
            os.environ["ROS_DOMAIN_ID"] = sys.argv[idx + 1]
            break
        if arg.startswith("--ros-domain-id="):
            os.environ["ROS_DOMAIN_ID"] = arg.split("=", 1)[1]
            break

    ros_distro = _selected_ros_distro()
    bridge_root = _find_isaacsim_ros2_core_root(ros_distro)
    if bridge_root is None:
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        system_python_path = Path(f"/opt/ros/{ros_distro}/lib/{python_version}/site-packages")
        system_lib_path = Path(f"/opt/ros/{ros_distro}/lib")
        if not (system_python_path / "rclpy").is_dir():
            raise RuntimeError(
                "ROS 2 was requested, but neither Isaac Sim 6's internal ROS 2 core nor system rclpy was found. "
                "Expected isaacsim.ros2.core/<distro> or /opt/ros/<distro>."
            )
        rclpy_path = str(system_python_path)
        lib_path = str(system_lib_path)
    else:
        rclpy_path = str(bridge_root / "rclpy")
        lib_path = str(bridge_root / "lib")

    env = os.environ.copy()
    env["DOFBOT_ROS2_ENV_READY"] = "1"
    env["DOFBOT_ROS2_PYTHON_PATH"] = rclpy_path
    env["ROS_DISTRO"] = ros_distro
    env.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    for key in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        entries = [entry for entry in env.get(key, "").split(":") if entry]
        preferred = rclpy_path if key == "PYTHONPATH" else lib_path
        if preferred in entries:
            entries.remove(preferred)
        env[key] = ":".join([preferred] + entries)

    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_prepare_ros2_env_before_python_start()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained DOFBOT reach policy with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Gym task id. Defaults to the joint-delta reach task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained RSL-RL checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many steps. Use 0 to run until closed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--demo-mode",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Disable episode auto-reset and target randomization for live demos.",
)
parser.add_argument(
    "--use-ros-target",
    action="store_true",
    default=False,
    help="Subscribe to the configured ROS2 target topic. Uses the environment default if --ros-target-topic is omitted.",
)
parser.add_argument(
    "--ros-target-topic",
    type=str,
    default=None,
    help="Optional ROS2 topic that streams target xyz positions for the Isaac target prim.",
)
parser.add_argument(
    "--ros-target-msg",
    type=str,
    default="pose_stamped",
    choices=("auto", "point", "point_stamped", "pose_stamped", "vector3"),
    help="ROS2 message type used by --ros-target-topic.",
)
parser.add_argument(
    "--ros-target-frame",
    type=str,
    default="env",
    choices=("env", "world"),
    help="Interpret ROS target xyz as environment-local robot-base coordinates or world coordinates.",
)
parser.add_argument(
    "--ros-target-base-pos-in-camera",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help=(
        "Optional base_link origin expressed in the incoming camera frame. "
        "When set, incoming camera xyz is transformed into base_link coordinates before applying --ros-target-frame."
    ),
)
parser.add_argument(
    "--ros-target-camera-is-base",
    action="store_true",
    default=False,
    help=(
        "Treat the incoming camera frame origin as the DOFBOT base origin. "
        "This ignores --ros-target-base-pos-in-camera and maps bottle-relative camera xyz directly to the target frame."
    ),
)
parser.add_argument(
    "--ros-target-base-quat-in-camera",
    type=float,
    nargs=4,
    default=(0.0, 0.0, 0.0, 1.0),
    metavar=("X", "Y", "Z", "W"),
    help=(
        "Optional base_link orientation expressed in the incoming camera frame as ROS xyzw. "
        "Used with --ros-target-base-pos-in-camera. Defaults to identity."
    ),
)
parser.add_argument(
    "--ros-target-timeout",
    type=float,
    default=0.5,
    help="Seconds before a stale ROS target is ignored. Use <=0 to never expire.",
)
parser.add_argument(
    "--ros-target-alpha",
    type=float,
    default=1.0,
    help="Low-pass filter alpha for ROS target updates. 1.0 disables filtering.",
)
parser.add_argument(
    "--ros-target-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="XYZ offset in meters applied after camera/base transform and before target clamping.",
)
parser.add_argument(
    "--ros-target-scale",
    type=float,
    nargs=3,
    default=(1.0, 1.0, 1.0),
    metavar=("X", "Y", "Z"),
    help="XYZ scale applied after camera/base transform and before offsets.",
)
parser.add_argument(
    "--ros-target-swap-xy",
    action="store_true",
    default=False,
    help="Swap target X and Y after camera/base transform and before scaling/offsets.",
)
parser.add_argument(
    "--ros-target-flip-x",
    action="store_true",
    default=False,
    help="Flip the sign of target X after optional X/Y swap and before scaling/offsets.",
)
parser.add_argument(
    "--ros-target-flip-y",
    action="store_true",
    default=False,
    help="Flip the sign of target Y after optional X/Y swap and before scaling/offsets.",
)
parser.add_argument(
    "--ros-target-flip-z",
    action="store_true",
    default=False,
    help="Flip the sign of target Z after optional X/Y swap and before scaling/offsets.",
)
parser.add_argument(
    "--ros-target-xy-scale",
    type=float,
    nargs=2,
    default=None,
    metavar=("X", "Y"),
    help="Convenience XY scale override. Applied with --ros-target-scale before offsets.",
)
parser.add_argument(
    "--ros-target-scale-origin",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Origin/pivot in meters for target scaling in the selected target frame.",
)
parser.add_argument(
    "--ros-target-z-offset",
    type=float,
    default=0.0,
    help="Extra Z offset in meters applied after --ros-target-offset.",
)
parser.add_argument(
    "--ros-target-min-z",
    type=float,
    default=None,
    help="Optional minimum target Z in the selected target frame, applied before range clamping.",
)
parser.add_argument(
    "--ros-target-max-z",
    type=float,
    default=None,
    help="Optional maximum target Z in the selected target frame, applied before range clamping.",
)
parser.add_argument(
    "--ros-target-fixed-z",
    type=float,
    default=None,
    help="Optional fixed target Z in the selected target frame, useful for quick tabletop tuning.",
)
parser.add_argument(
    "--ros-target-clamp-range",
    type=float,
    nargs=6,
    default=None,
    metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    help="Optional clamp range override for ROS targets. Useful when the trained target range compresses live Y motion.",
)
parser.add_argument(
    "--no-ros-target-clamp",
    action="store_true",
    default=False,
    help="Disable clamping ROS targets to the configured target sampling ranges.",
)
parser.add_argument(
    "--ros-publish-joint-states",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Publish simulated DOFBOT joint states on ROS2 for robot_state_publisher/TF.",
)
parser.add_argument(
    "--ros-namespace",
    type=str,
    default="dofbot",
    help="ROS2 namespace for optional target subscriber and joint-state publisher. Use '' to disable.",
)
parser.add_argument(
    "--ros-domain-id",
    type=int,
    default=None,
    help="Set ROS_DOMAIN_ID for this process before creating rclpy nodes.",
)
parser.add_argument(
    "--ros-joint-state-topic",
    type=str,
    default="joint_command",
    help="ROS2 topic for sensor_msgs/JointState when --ros-publish-joint-states is enabled.",
)
parser.add_argument(
    "--ros-joint-state-rate",
    type=float,
    default=30.0,
    help="Max publish rate in Hz for ROS2 JointState messages.",
)
parser.add_argument(
    "--ros-publish-gripper-control",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Publish an initial std_msgs/Float64 gripper command when play starts.",
)
parser.add_argument(
    "--ros-gripper-control-topic",
    type=str,
    default="gripper_control",
    help="ROS2 topic for std_msgs/Float64 gripper commands.",
)
parser.add_argument(
    "--ros-gripper-start-value",
    type=float,
    default=1.0,
    help="Initial gripper control value to publish at play startup.",
)
parser.add_argument(
    "--ros-gripper-start-duration",
    type=float,
    default=1.0,
    help="Seconds to repeat the initial gripper command so late-discovered subscribers receive it.",
)
parser.add_argument(
    "--ros-gripper-start-rate",
    type=float,
    default=10.0,
    help="Rate in Hz for repeating the initial gripper command.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.ros_domain_id is not None:
    os.environ["ROS_DOMAIN_ID"] = str(args_cli.ros_domain_id)
if (
    args_cli.use_ros_target
    or args_cli.ros_target_topic
    or args_cli.ros_publish_joint_states
    or args_cli.ros_publish_gripper_control
):
    print(
        f"[INFO] ROS_DISTRO={os.environ.get('ROS_DISTRO', 'unset')} "
        f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')} "
        f"RMW_IMPLEMENTATION={os.environ.get('RMW_IMPLEMENTATION', 'default')}",
        flush=True,
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from dofbot_rl.tasks import ENV_ID, IK_ENV_ID
from dofbot_rl.tasks.agents.rsl_rl_ppo_cfg import DofbotReachIKPPORunnerCfg, DofbotReachPPORunnerCfg
from dofbot_rl.tasks.dofbot_reach_cfg import DofbotReachEnvCfg, DofbotReachIKEnvCfg
import dofbot_rl.tasks  # noqa: F401


def prefer_isaacsim_ros2_python() -> None:
    """Ensure the ROS 2 Python path selected before AppLauncher stays first."""
    rclpy_path = os.environ.get("DOFBOT_ROS2_PYTHON_PATH")
    if rclpy_path and rclpy_path in sys.path:
        sys.path.remove(rclpy_path)
    if rclpy_path:
        sys.path.insert(0, rclpy_path)


_RCLPY_INITIALIZED_BY_PLAY = False


def ensure_rclpy_initialized(rclpy) -> None:
    """Initialize one shared rclpy context for all play publishers/subscribers."""
    global _RCLPY_INITIALIZED_BY_PLAY
    if not rclpy.ok():
        rclpy.init(args=None)
        _RCLPY_INITIALIZED_BY_PLAY = True


def shutdown_rclpy() -> None:
    """Shutdown the shared rclpy context after every node has been destroyed."""
    if not _RCLPY_INITIALIZED_BY_PLAY:
        return
    import rclpy

    if rclpy.ok():
        rclpy.shutdown()


def normalize_ros_namespace(namespace: str) -> str:
    namespace = namespace.strip()
    if not namespace:
        return ""
    return "/" + namespace.strip("/")


def normalize_ros_topic(topic_name: str, namespace: str = "", *, apply_namespace: bool = False) -> str:
    """Return an absolute ROS topic name with predictable namespace behavior."""
    topic_name = topic_name.strip()
    if not topic_name:
        raise ValueError("ROS topic name must not be empty.")
    if topic_name.startswith("/"):
        return "/" + topic_name.strip("/")
    if apply_namespace and namespace:
        return f"{namespace}/{topic_name.strip('/')}"
    return "/" + topic_name.strip("/")


def quat_xyzw_to_matrix(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert a ROS xyzw quaternion to a 3x3 rotation matrix."""
    quat_xyzw = quat_xyzw / torch.linalg.norm(quat_xyzw).clamp_min(1.0e-8)
    x, y, z, w = quat_xyzw.unbind()
    return torch.stack(
        (
            torch.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w))),
            torch.stack((2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w))),
            torch.stack((2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y))),
        )
    )


class RosTargetSubscriber:
    """Small optional ROS2 subscriber for streaming target xyz into the play loop."""

    def __init__(self, topic_name: str, msg_type: str, namespace: str):
        prefer_isaacsim_ros2_python()
        try:
            import rclpy
            from geometry_msgs.msg import Point, PointStamped, PoseStamped, Vector3
        except ImportError as exc:
            raise ImportError(
                "ROS target streaming requires rclpy and geometry_msgs in the Isaac Python environment. "
                "For Isaac Sim 6.0.1, use its isaacsim.ros2.core/jazzy Python 3.12 packages or system ROS 2 Jazzy."
            ) from exc

        self.rclpy = rclpy
        ensure_rclpy_initialized(rclpy)

        msg_classes = {
            "point": Point,
            "point_stamped": PointStamped,
            "pose_stamped": PoseStamped,
            "vector3": Vector3,
        }
        self.node = rclpy.create_node("dofbot_rl_target_subscriber", namespace=namespace)
        if msg_type == "auto":
            type_to_msg = {
                "geometry_msgs/msg/Point": "point",
                "geometry_msgs/msg/PointStamped": "point_stamped",
                "geometry_msgs/msg/PoseStamped": "pose_stamped",
                "geometry_msgs/msg/Vector3": "vector3",
            }
            published_types = {
                info.topic_type for info in self.node.get_publishers_info_by_topic(topic_name)
            }
            for candidate_type in (
                "geometry_msgs/msg/PoseStamped",
                "geometry_msgs/msg/Point",
                "geometry_msgs/msg/PointStamped",
                "geometry_msgs/msg/Vector3",
            ):
                if candidate_type in published_types:
                    msg_type = type_to_msg[candidate_type]
                    break
            else:
                msg_type = "pose_stamped"
        self.msg_type = msg_type
        self._latest_xyz: tuple[float, float, float] | None = None
        self._latest_time = 0.0
        self.received_count = 0
        self.node.create_subscription(msg_classes[msg_type], topic_name, self._callback, 10)
        print(
            f"[INFO] ROS2 target subscriber enabled: namespace={namespace or '/'} topic={topic_name}, msg={msg_type}",
            flush=True,
        )

    def _callback(self, msg, *_) -> None:
        if hasattr(msg, "point"):
            point = msg.point
        elif hasattr(msg, "pose"):
            point = msg.pose.position
        else:
            point = msg
        self._latest_xyz = (float(point.x), float(point.y), float(point.z))
        self._latest_time = time.monotonic()
        self.received_count += 1

    def poll(self, timeout_s: float) -> tuple[float, float, float] | None:
        self.rclpy.spin_once(self.node, timeout_sec=0.0)
        if self._latest_xyz is None:
            return None
        if timeout_s > 0.0 and time.monotonic() - self._latest_time > timeout_s:
            return None
        return self._latest_xyz

    def close(self) -> None:
        self.node.destroy_node()


class RosJointStatePublisher:
    """Optional ROS2 publisher for streaming simulated joint states to robot_state_publisher."""

    def __init__(self, topic_name: str, joint_names: list[str], publish_rate_hz: float, namespace: str):
        prefer_isaacsim_ros2_python()
        try:
            import rclpy
            from sensor_msgs.msg import JointState
        except ImportError as exc:
            raise ImportError(
                "JointState publishing requires rclpy and sensor_msgs in the Isaac Python environment. "
                "For Isaac Sim 6.0.1, use its isaacsim.ros2.core/jazzy Python 3.12 packages or system ROS 2 Jazzy."
            ) from exc

        self.rclpy = rclpy
        self.JointState = JointState
        ensure_rclpy_initialized(rclpy)

        self.node = rclpy.create_node("dofbot_rl_joint_state_publisher", namespace=namespace)
        self.publisher = self.node.create_publisher(JointState, topic_name, 10)
        self.joint_names = list(joint_names)
        self.period_s = 1.0 / publish_rate_hz if publish_rate_hz > 0.0 else 0.0
        self._last_publish_time = 0.0
        self.published_count = 0
        print(
            f"[INFO] ROS2 JointState publisher enabled: namespace={namespace or '/'} topic={topic_name}",
            flush=True,
        )

    def publish(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor | None = None) -> None:
        now_monotonic = time.monotonic()
        if self.period_s > 0.0 and now_monotonic - self._last_publish_time < self.period_s:
            return
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(value) for value in joint_pos.detach().cpu().tolist()]
        if joint_vel is not None:
            msg.velocity = [float(value) for value in joint_vel.detach().cpu().tolist()]
        self.publisher.publish(msg)
        self._last_publish_time = now_monotonic
        self.published_count += 1

    def close(self) -> None:
        self.node.destroy_node()


class RosFloat64Publisher:
    """Small ROS2 Float64 publisher used for gripper startup commands."""

    def __init__(self, node_name: str, topic_name: str, namespace: str):
        prefer_isaacsim_ros2_python()
        try:
            import rclpy
            from std_msgs.msg import Float64
        except ImportError as exc:
            raise ImportError(
                "Float64 publishing requires rclpy and std_msgs in the Isaac Python environment. "
                "For Isaac Sim 6.0.1, use its isaacsim.ros2.core/jazzy Python 3.12 packages or system ROS 2 Jazzy."
            ) from exc

        self.rclpy = rclpy
        self.Float64 = Float64
        ensure_rclpy_initialized(rclpy)

        self.node = rclpy.create_node(node_name, namespace=namespace)
        self.publisher = self.node.create_publisher(Float64, topic_name, 10)
        self.published_count = 0
        print(
            f"[INFO] ROS2 Float64 publisher enabled: namespace={namespace or '/'} topic={topic_name}",
            flush=True,
        )

    def publish(self, value: float) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=0.0)
        msg = self.Float64()
        msg.data = float(value)
        self.publisher.publish(msg)
        self.published_count += 1

    def close(self) -> None:
        self.node.destroy_node()


def get_task_cfgs(task_id: str):
    if task_id == IK_ENV_ID:
        return DofbotReachIKEnvCfg, DofbotReachIKPPORunnerCfg
    if task_id == ENV_ID:
        return DofbotReachEnvCfg, DofbotReachPPORunnerCfg
    raise ValueError(f"Unsupported DOFBOT task id: {task_id}")


def apply_external_target(unwrapped_env, xyz: tuple[float, float, float], filtered_target: torch.Tensor | None):
    """Write an external target position to the Isaac target object and return the filtered local target."""
    target_local = torch.tensor(xyz, dtype=torch.float32, device=unwrapped_env.device).unsqueeze(0)
    if args_cli.ros_target_base_pos_in_camera is not None and not args_cli.ros_target_camera_is_base:
        base_pos_in_camera = torch.tensor(
            args_cli.ros_target_base_pos_in_camera,
            dtype=torch.float32,
            device=unwrapped_env.device,
        ).unsqueeze(0)
        base_quat_in_camera = torch.tensor(
            args_cli.ros_target_base_quat_in_camera,
            dtype=torch.float32,
            device=unwrapped_env.device,
        )
        camera_rot_base = quat_xyzw_to_matrix(base_quat_in_camera)
        base_rot_camera = camera_rot_base.transpose(0, 1)
        target_local = torch.matmul(target_local - base_pos_in_camera, base_rot_camera.T)

    if args_cli.ros_target_swap_xy:
        target_local = target_local[:, [1, 0, 2]]
    if args_cli.ros_target_flip_x:
        target_local[:, 0] *= -1.0
    if args_cli.ros_target_flip_y:
        target_local[:, 1] *= -1.0
    if args_cli.ros_target_flip_z:
        target_local[:, 2] *= -1.0

    target_scale = torch.tensor(args_cli.ros_target_scale, dtype=torch.float32, device=unwrapped_env.device)
    if args_cli.ros_target_xy_scale is not None:
        target_scale[:2] = torch.tensor(args_cli.ros_target_xy_scale, dtype=torch.float32, device=unwrapped_env.device)
    target_scale_origin = torch.tensor(
        args_cli.ros_target_scale_origin,
        dtype=torch.float32,
        device=unwrapped_env.device,
    ).unsqueeze(0)
    target_local = (target_local - target_scale_origin) * target_scale.unsqueeze(0) + target_scale_origin

    target_offset = torch.tensor(args_cli.ros_target_offset, dtype=torch.float32, device=unwrapped_env.device)
    target_local = target_local + target_offset.unsqueeze(0)
    target_local[:, 2] += args_cli.ros_target_z_offset
    if args_cli.ros_target_fixed_z is not None:
        target_local[:, 2] = args_cli.ros_target_fixed_z
    if args_cli.ros_target_min_z is not None:
        target_local[:, 2] = torch.clamp(target_local[:, 2], min=args_cli.ros_target_min_z)
    if args_cli.ros_target_max_z is not None:
        target_local[:, 2] = torch.clamp(target_local[:, 2], max=args_cli.ros_target_max_z)

    if not args_cli.no_ros_target_clamp:
        cfg = unwrapped_env.cfg
        if args_cli.ros_target_clamp_range is None:
            clamp_range = (
                cfg.target_x_range[0],
                cfg.target_x_range[1],
                cfg.target_y_range[0],
                cfg.target_y_range[1],
                cfg.target_z_range[0],
                cfg.target_z_range[1],
            )
        else:
            clamp_range = args_cli.ros_target_clamp_range
        lower = torch.tensor(
            [clamp_range[0], clamp_range[2], clamp_range[4]],
            dtype=torch.float32,
            device=unwrapped_env.device,
        ).unsqueeze(0)
        upper = torch.tensor(
            [clamp_range[1], clamp_range[3], clamp_range[5]],
            dtype=torch.float32,
            device=unwrapped_env.device,
        ).unsqueeze(0)
        target_local = torch.clamp(target_local, lower, upper)

    alpha = min(max(args_cli.ros_target_alpha, 0.0), 1.0)
    if filtered_target is None or alpha >= 1.0:
        filtered_target = target_local
    else:
        filtered_target = alpha * target_local + (1.0 - alpha) * filtered_target

    unwrapped_env.set_target_position(filtered_target, frame=args_cli.ros_target_frame)
    return filtered_target


def find_latest_checkpoint() -> Path:
    task_id = args_cli.task or ENV_ID
    _, runner_cfg_cls = get_task_cfgs(task_id)
    isaaclab_dir = Path(os.environ.get("ISAACLAB_DIR", PROJECT_ROOT / "IsaacLab")).expanduser()
    root = isaaclab_dir / "logs" / "rsl_rl" / runner_cfg_cls().experiment_name
    candidates = sorted(root.rglob("model_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No RSL-RL checkpoints found under '{root}'.")
    return candidates[0]


def main():
    task_id = args_cli.task or ENV_ID
    env_cfg_cls, runner_cfg_cls = get_task_cfgs(task_id)
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve() if args_cli.checkpoint else find_latest_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.demo_mode = args_cli.demo_mode
    env_cfg.training_mode = not args_cli.demo_mode
    env_cfg.use_ros_target = bool(args_cli.use_ros_target or args_cli.ros_target_topic)
    env_cfg.disable_auto_reset = bool(env_cfg.disable_auto_reset or args_cli.demo_mode)
    play_device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.device = play_device
    env_cfg.log_dir = str(checkpoint.parent)
    ros_namespace = normalize_ros_namespace(args_cli.ros_namespace)
    ros_target_topic = args_cli.ros_target_topic
    if env_cfg.use_ros_target and ros_target_topic is None:
        ros_target_topic = env_cfg.ros_target_topic
    if ros_target_topic is not None:
        ros_target_topic = normalize_ros_topic(ros_target_topic, ros_namespace, apply_namespace=False)
    ros_joint_state_topic = normalize_ros_topic(
        args_cli.ros_joint_state_topic,
        ros_namespace,
        apply_namespace=True,
    )
    ros_gripper_control_topic = normalize_ros_topic(
        args_cli.ros_gripper_control_topic,
        ros_namespace,
        apply_namespace=True,
    )

    agent_cfg = runner_cfg_cls()
    agent_cfg.device = play_device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped_env = env.unwrapped

    print(f"[INFO] Loading RSL-RL checkpoint: {checkpoint}", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_dir = checkpoint.parent / "exported"
    runner.export_policy_to_jit(path=str(export_dir), filename="policy.pt")
    runner.export_policy_to_onnx(path=str(export_dir), filename="policy.onnx")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    ros_target = None
    joint_state_pub = None
    gripper_control_pub = None
    filtered_target = None
    last_ros_debug_time = 0.0
    gripper_start_begin_time = time.monotonic()
    gripper_start_last_publish_time = 0.0
    gripper_start_period = 1.0 / args_cli.ros_gripper_start_rate if args_cli.ros_gripper_start_rate > 0.0 else 0.0
    try:
        if env_cfg.use_ros_target and ros_target_topic:
            ros_target = RosTargetSubscriber(ros_target_topic, args_cli.ros_target_msg, ros_namespace)
        if args_cli.ros_publish_joint_states:
            joint_state_pub = RosJointStatePublisher(
                ros_joint_state_topic,
                unwrapped_env.robot.data.joint_names,
                args_cli.ros_joint_state_rate,
                ros_namespace,
            )
        if args_cli.ros_publish_gripper_control:
            gripper_control_pub = RosFloat64Publisher(
                "dofbot_rl_gripper_control_publisher",
                ros_gripper_control_topic,
                ros_namespace,
            )
        while simulation_app.is_running():
            start_time = time.time()
            if ros_target is not None:
                xyz = ros_target.poll(args_cli.ros_target_timeout)
                if xyz is not None:
                    filtered_target = apply_external_target(unwrapped_env, xyz, filtered_target)
                    obs = env.get_observations()
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if joint_state_pub is not None:
                joint_state_pub.publish(
                    unwrapped_env.robot.data.joint_pos.torch[0],
                    unwrapped_env.robot.data.joint_vel.torch[0],
                )
            now = time.monotonic()
            if (
                gripper_control_pub is not None
                and now - gripper_start_begin_time <= args_cli.ros_gripper_start_duration
                and (
                    gripper_start_period <= 0.0
                    or now - gripper_start_last_publish_time >= gripper_start_period
                )
            ):
                gripper_control_pub.publish(args_cli.ros_gripper_start_value)
                gripper_start_last_publish_time = now
            if now - last_ros_debug_time >= 1.0:
                if ros_target is not None:
                    target_debug = None
                    if filtered_target is not None:
                        target_debug = [round(float(value), 4) for value in filtered_target[0].detach().cpu().tolist()]
                    print(
                        f"[ROS] target received={ros_target.received_count} "
                        f"raw={ros_target._latest_xyz} applied={target_debug}",
                        flush=True,
                    )
                if joint_state_pub is not None:
                    print(f"[ROS] joint published={joint_state_pub.published_count}", flush=True)
                if gripper_control_pub is not None:
                    print(f"[ROS] gripper published={gripper_control_pub.published_count}", flush=True)
                last_ros_debug_time = now
            timestep += 1
            if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
                break
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0.0:
                time.sleep(sleep_time)
    finally:
        if ros_target is not None:
            ros_target.close()
        if joint_state_pub is not None:
            joint_state_pub.close()
        if gripper_control_pub is not None:
            gripper_control_pub.close()
        shutdown_rclpy()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
