import re

import carb
import omni.kit.app
import omni.timeline
import omni.usd

from isaacsim.robot.poser import (
    apply_pose_by_name,
    list_named_poses,
    validate_robot_schema,
)


ROBOT_PRIM_PATH = "/World/dofbot"
POSE_INTERVAL_SECONDS = 1.0


# Script Editor에서 코드를 다시 실행했을 때 기존 구독 제거
try:
    pose_player.stop()
except (NameError, AttributeError):
    pass


class NamedPosePlayer:
    def __init__(
        self,
        robot_prim_path: str,
        interval_seconds: float = 1.0,
    ):
        self.robot_prim_path = robot_prim_path
        self.interval_seconds = interval_seconds

        self._stage = None
        self._robot_prim = None
        self._pose_names = []

        self._pose_index = 0
        self._elapsed = 0.0
        self._subscription = None

    def start(self):
        carb.log_info("[Pose Player] 초기화 시작")

        self._stage = omni.usd.get_context().get_stage()

        if self._stage is None:
            raise RuntimeError(
                "열려 있는 USD Stage가 없습니다."
            )

        self._robot_prim = self._stage.GetPrimAtPath(
            self.robot_prim_path
        )

        if not self._robot_prim.IsValid():
            raise RuntimeError(
                f"Robot Prim이 없습니다: "
                f"{self.robot_prim_path}"
            )

        if not validate_robot_schema(self._robot_prim):
            raise RuntimeError(
                f"IsaacRobotAPI가 적용된 Robot Prim이 아닙니다: "
                f"{self.robot_prim_path}"
            )

        self._refresh_pose_names()

        if not self._pose_names:
            raise RuntimeError(
                "Pose_1, Pose_2, ... 형식으로 등록된 "
                "Named Pose가 없습니다."
            )

        update_stream = (
            omni.kit.app.get_app()
            .get_update_event_stream()
        )

        self._subscription = (
            update_stream.create_subscription_to_pop(
                self._on_update,
                name="dofbot_named_pose_player",
            )
        )

        carb.log_info(
            f"[Pose Player] 시작: {self._pose_names}"
        )

        # 시작하자마자 Pose_1 적용
        self._apply_current_pose()

    def _refresh_pose_names(self):
        pattern = re.compile(r"^Pose_(\d+)$", re.IGNORECASE)

        numbered_poses = []

        all_pose_names = list_named_poses(
            self._stage,
            self._robot_prim,
        )

        carb.log_info(
            f"[Pose Player] 등록된 전체 Pose: "
            f"{all_pose_names}"
        )

        for pose_name in all_pose_names:
            match = pattern.fullmatch(pose_name)

            if match:
                numbered_poses.append(
                    (int(match.group(1)), pose_name)
                )

        numbered_poses.sort(key=lambda item: item[0])

        self._pose_names = [
            pose_name
            for _, pose_name in numbered_poses
        ]

    def _apply_current_pose(self):
        pose_name = self._pose_names[self._pose_index]

        result = apply_pose_by_name(
            self._stage,
            self._robot_prim,
            pose_name,
        )

        carb.log_info(
            f"[Pose Player] Apply {pose_name}: {result}"
        )

        if not result:
            carb.log_error(
                f"[Pose Player] Pose 적용 실패: {pose_name}"
            )

    def _on_update(self, event):
        try:
            # UpdateEvent payload의 dt는 프레임 경과 시간
            dt = event.payload.get("dt", 0.0)

            if dt is None:
                return

            self._elapsed += float(dt)

            if self._elapsed < self.interval_seconds:
                return

            # 프레임 지연 시 누적 오차를 줄이기 위해 빼기 사용
            self._elapsed -= self.interval_seconds

            self._pose_index = (
                self._pose_index + 1
            ) % len(self._pose_names)

            self._apply_current_pose()

        except Exception as error:
            carb.log_error(
                f"[Pose Player] 업데이트 오류: "
                f"{type(error).__name__}: {error}"
            )
            self.stop()
            raise

    def stop(self):
        self._subscription = None
        carb.log_info("[Pose Player] 중지")


pose_player = NamedPosePlayer(
    ROBOT_PRIM_PATH,
    POSE_INTERVAL_SECONDS,
)

pose_player.start()