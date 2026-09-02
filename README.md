# DOFBOT Reach RL

이 프로젝트는 Isaac Sim 6.0.1 / Isaac Lab 3.0.0-beta2의 direct workflow와 RSL-RL을 사용해
DOFBOT 로봇이 3D 공간의 목표 위치를 따라가도록 학습하는 예제입니다.

## License

이 저장소의 코드, 설정 파일, 학습/추론 스크립트는 [MIT License](LICENSE)에 따라 사용할 수 있습니다.

- 제작자: SooYoun Bae
- 소속: KRINFRA (주식회사 한국인프라)

단, Isaac Lab, Isaac Sim, ROS 2, RL 관련 패키지 및 외부에서 가져온 3D 자산(예: USD/URDF/mesh 파일) 은 해당 프로젝트의 라이선스와 재배포 조건이 따로 적용될 수 있습니다. 외부 자산을 사용하는 경우, 해당 파일의 라이선스와 사용 조건을 별도로 확인해 주시기 바랍니다.

## 기능

- DOFBOT 로봇을 Isaac Lab scene에 구성
- reset마다 무작위 3D 목표 위치 생성
- PPO로 end-effector가 목표에 가까워지도록 학습
- 학습된 체크포인트로 play / inference 실행

## 프로젝트 구성

- `tasks/dofbot_reach_cfg.py` - Isaac Lab `configclass` 기반 환경 및 scene 설정
- `tasks/dofbot_reach_env.py` - `DirectRLEnv` 환경 구현
- `cmd/train.sh` - RSL-RL PPO 학습 실행 스크립트
- `play_rsl_rl.py` - RSL-RL 체크포인트를 불러와 추론하는 스크립트
- `smoke_test.py` - 환경 reset/step 확인용 스모크 테스트
- `assets/dofbot/README.md` - DOFBOT USD / URDF 자산 준비 안내

## 설치

1. Isaac Sim 6.0.1 / Isaac Lab 3.0.0-beta2 환경을 준비합니다.
2. Isaac Lab을 설치합니다.
3. DOFBOT USD 또는 URDF를 아래 경로에 준비합니다.

   - 권장: `dofbot_rl/assets/dofbot/dofbot.usd`
   - 대안: `dofbot_rl/assets/dofbot/dofbot_info/urdf/dofbot.urdf`

4. 실행 환경에서 `isaaclab`, `isaaclab_rl`, `rsl_rl`, `gymnasium`을 사용할 수 있어야 합니다.

## 스모크 테스트

```bash
cd "$HOME/IsaacLab"
./isaaclab.sh -p "$HOME/dofbot_rl/smoke_test.py" \
  --task Dofbot-Reach-IK-Direct-v0 \
  --device cuda:0 \
  --num_envs 1 \
  --num_steps 20 \
  --viz none
```

## 학습 실행

```bash
cd "$HOME/dofbot_rl"
bash cmd/train.sh
```

환경 수와 iteration 수는 환경 변수로 바꿀 수 있습니다.

```bash
NUM_ENVS=64 MAX_ITERATIONS=2000 bash cmd/train.sh
```

## 추론 실행

```bash
cd "$HOME/IsaacLab"
./isaaclab.sh -p "$HOME/dofbot_rl/play_rsl_rl.py" \
  --task Dofbot-Reach-IK-Direct-v0 \
  --checkpoint logs/rsl_rl/dofbot_reach_ik_direct_refine/<run>/model_999.pt \
  --viz kit
```

`--checkpoint`를 지정하지 않으면 `play_rsl_rl.py`가 해당 task의 최신 `model_*.pt`를 자동으로 찾습니다.
일반 play에서는 ROS 2를 불러오지 않습니다. ROS 2 Jazzy target 입력과 joint/gripper 출력을 함께 사용하려면
체크포인트 경로를 지정한 뒤 프로젝트 실행 스크립트를 사용합니다.

```bash
cd "$HOME/dofbot_rl"
CHECKPOINT_PATH="$HOME/IsaacLab/logs/rsl_rl/dofbot_reach_ik_direct_refine/<run>/model_999.pt" \
  bash cmd/isaaclab.sh
```

이 스크립트는 Isaac Sim 6.0.1의 `isaacsim.ros2.core/jazzy`(Python 3.12)를 우선 사용하고,
없을 때 `/opt/ros/jazzy`를 사용합니다. `ROS_DOMAIN_ID` 기본값은 `32`입니다.

## DOFBOT_V2 Pick–Place 학습

`assets/dofbot_v2/dofbot.usd`의 4축 arm, wrist, 물리 그리퍼를 사용하는 Direct-RL 환경이 포함되어 있습니다.
정책 action은 `joint1~joint4`의 누적 position delta 4개, 0° safety-hold wrist 슬롯 1개와 오른쪽 손가락 명령 1개로 총
6차원입니다. 왼쪽 1번 손가락은 USD의 PhysX mimic 관계를 그대로 사용하므로 직접 명령하지 않습니다.
정책 관측은 70차원이며 `reach → lift → pick_place` 순서로 체크포인트를 이어받습니다.

사용자가 지정한 `Finger_Right_02` 로컬 점 `(0.00004, 0.020316, 0.019236) m`은 TCP 보정의
원본 측정값입니다. 이 링크는 그리퍼가 닫힐 때 움직이므로, 런타임 정책은 이 점에서 보정한
`Wrist_Twist` 고정 프레임의 jaw 중앙점을 사용합니다. 정책은 아래 5단계 one-hot 상태를 관측합니다.

1. 열린 그리퍼를 물체 중심보다 `0.055 m` 높은 pre-grasp 위치로 이동하고 수직 정렬
2. 보정된 grasp point를 물체에 내린 뒤 그리퍼 닫기
3. 물체의 자세를 유지하며 부드럽게 들어 올리기
4. 물체를 목표점보다 `0.015 m` 높은 위치로 운반
5. 물체를 테이블에서 `2 mm` 이내까지 내리고 안정화한 뒤 천천히 그리퍼 열기

운반 중에는 action 변화량, 물체 각속도와 기울기에 penalty가 적용됩니다. 물체의 up-axis와
각속도, 그리퍼 접근축, 현재 단계, 단계별 연속-hold 상태, 실제 양쪽 접촉력을 포함하므로
여기에 position target과 실제 joint 위치의 servo 오차 및 현재 phase가 실제로
추적해야 하는 목표 오차를 포함하므로 최종 정책 관측은 70차원입니다.

모든 episode는 `joint1, joint2, joint3, joint4, wrist, right finger = (0, 0, 0, 0, 0, 0)`에서
시작합니다. arm/wrist 5축은 모두 `-90°~+90°`, 오른쪽 finger driver는 `-57°~+33°`이고
음수가 열림, 양수가 닫힘입니다. reset은 0°를 유지하되 phase 0에서 driver를 `-0.50 rad`로
열고, grasp/carry/release gate는 wrist가 0°±8° 안에 있을 때만 통과합니다. 왼쪽 finger는
USD mimic 관계만 사용합니다. 이
6-action/70-observation 환경은
이전 4-action/35-observation, 6-action/39-observation, 6-action/53-observation 및
6-action/62-observation, 6-action/65-observation 모델과
호환되지 않으므로
기존 Pick–Place 모델을 resume하지 말고 Reach 단계부터 새로 학습해야 합니다.

커리큘럼에서 Reach는 1번의 안전한 pre-grasp 위치를 연속 유지하면 종료합니다. 큐브 쪽으로
수직 하강하고 gripper를 닫는 동작은 Lift 단계에서 pre-grasp를 다시 수행한 뒤 시작합니다.

단계 전환은 한 프레임 조건이 아닙니다. pre-grasp, grasp, lift, transport gate를 각각
`4, 6, 8, 8` control step 연속 만족해야 다음 phase로 갑니다. Place pose는 위치·높이·선속도·각속도·
기울기를 12 step 유지해야 release가 허용되고, 최종 성공은 실제 finger joint/gap으로 열린 상태와
안정된 cube pose를 20 step 유지해야 합니다. 허가 전에는 phase 4에서도 물리 그립을 강제로 유지합니다.

모든 task 좌표는 DOFBOT의 `base_link=(0, 0, 0)` 기준입니다. 로봇은 identity quaternion
`(x, y, z, w)=(0, 0, 0, 1)`로 배치되고, object와 goal 위치 및 정책 관측도 실제 `base_link`
world position을 빼서 계산합니다. 스모크 테스트는 base 오차가 `1e-5 m`를 넘으면 실패합니다.

먼저 64개 환경에서 모든 actuator와 mimic을 검사합니다.

```bash
cd "$HOME/IsaacLab"
./isaaclab.sh -p "$HOME/dofbot_rl/smoke_test.py" \
  --task Dofbot-V2-PickPlace-Direct-v0 \
  --num_envs 64 --num_steps 20 --actuator_test --device cuda:0 --viz none
```

PPO 전에 checkpoint 없이 전체 물리 경로를 결정론적으로 검사할 수 있습니다.

```bash
cd "$HOME/IsaacLab"
./isaaclab.sh -p "$HOME/dofbot_rl/cmd/sanity_grasp_lift.py" \
  --full_pick_place --device cuda:0 --viz none --log_interval 120
```

권장 실행은 아래 end-to-end 명령 하나입니다. deterministic sanity가 먼저 통과해야 Reach가 시작되며,
각 단계의 정확한 새 checkpoint만 다음 단계에 전달합니다. 중단됐거나 요구 iteration이 없는 run,
학습률이 `3e-4`가 아닌 과거 adaptive-LR checkpoint, 물리 gate 기준에 미달한 run은 자동 중단됩니다.

```bash
cd "$HOME/dofbot_rl"
NUM_ENVS=1024 PLAY_AFTER_EACH=1 bash cmd/end-to-end.sh
```

기본 curriculum은 `1000 → 2000 → 4000` iteration을 추가하며 체크포인트 번호는 RSL-RL resume
인덱스 규칙에 따라 `model_999.pt → model_2998.pt → model_6997.pt`가 됩니다. 환경 수는 시스템
안전을 위해 모든 실행 스크립트에서 `1..1024`로 제한합니다.

각 단계를 수동 실행하려면 아래 순서를 사용합니다. 기본 자동 검색은 이름이 정확히 `_reach`,
`_lift`로 끝나는 canonical run만 허용합니다.

```bash
cd "$HOME/dofbot_rl"

STAGE=reach NUM_ENVS=1024 MAX_ITERATIONS=1000 bash cmd/train_pick_place.sh
STAGE=lift NUM_ENVS=1024 MAX_ITERATIONS=2000 bash cmd/train_pick_place.sh
STAGE=pick_place NUM_ENVS=1024 MAX_ITERATIONS=4000 bash cmd/train_pick_place.sh
```

환경·보상·관측이 변경됐으므로 현재는 반드시 `RESUME=0`인 새 Reach부터 시작해야 합니다.
특정 checkpoint를 수동 지정할 때는 예를 들어 `LOAD_RUN='^2026-..._reach$'`와
`CHECKPOINT_PATTERN='^model_999[.]pt$'`처럼 양 끝을 고정합니다.

최종 정책을 GUI에서 재생합니다.

```bash
cd "$HOME/dofbot_rl"
CHECKPOINT_PATH="$HOME/IsaacLab/logs/rsl_rl/dofbot_v2_pick_place/<run>/<model>.pt" \
  bash cmd/play_pick_place.sh
```

## DOFBOT_V2 관측 / 행동 / 보상

### 관측

70차원 관측은 아래 항목으로 구성됩니다.

- arm/wrist position·velocity 10개, joint target servo 오차 5개, finger position·velocity 4개
- base_link 기준 gripper/object/goal position 9개
- grasp/goal delta 6개와 object 선속도 3개
- gripper command 1개, previous action 6개, 현재 phase target 오차 3개
- grasp 접근축 3개, phase one-hot 5개, object up-axis·각속도 6개
- phase/place/success hold, release authorization, grasp-loss 상태 5개
- reset 위치 대비 cube XY 변위 2개와 실제 좌·우 fingertip 접촉력 2개

### 행동

- `joint1~joint4` 누적 position increment 4개와 0° safety-hold wrist 슬롯 1개
- 오른쪽 finger 열기/닫기 명령 1개

### 보상

- 각 phase의 signed distance/progress와 한 번만 지급되는 phase 완료 보상
- 수직 정렬, 안정적 lift/transport/place, 실제 bilateral grasp 보상
- action rate, 바닥 훑기, cube 밀기, grasp 손실, 각속도·기울기 penalty
- 안정된 실제 release를 20 step 유지했을 때의 성공 보너스

### 종료 조건

- stage별 held physical success, timeout
- cube 낙하/작업영역 이탈, TCP 바닥 충돌, capture 전 cube 과다 이동
- lift/transport/place 중 grasp 손실, release 허가 전 조기 개방, joint limit 오류

## 자산 준비

기본적으로 `dofbot.usd`를 우선 사용하고, 파일이 없으면 URDF를 Isaac Lab의 URDF importer로 변환해서 사용합니다.

## cmd 폴더

`cmd/` 폴더에는 로컬 Isaac Lab / ROS 환경을 실행하기 위한 보조 스크립트와 desktop launcher가 들어 있습니다.

- `cmd/train.sh` - Isaac Lab 학습 실행
- `cmd/isaaclab.sh` - 체크포인트를 이용한 추론 / play 실행
- `cmd/startup.sh` - ROS, camera, Isaac ROS, TensorBoard용 tmux 워크플로우 실행
- `cmd/*.desktop` - `~/Desktop`에서 각 shell script를 호출하는 바로가기

스크립트들은 고정 사용자명 대신 `$HOME` 기반 경로를 사용하도록 정리되어 있습니다. 아래 경로들이 실제 환경과 맞아야 합니다.

- `"$HOME/IsaacLab"`
- Conda hook 경로, 예: `"$HOME/anaconda3/etc/profile.d/conda.sh"` 또는 `"$HOME/miniconda3/etc/profile.d/conda.sh"`
- `cmd/isaaclab.sh`에서 사용하는 체크포인트 경로, 필요하면 `CHECKPOINT_PATH`로 덮어쓸 수 있음

예시:

```bash
CHECKPOINT_PATH="$HOME/IsaacLab/logs/rsl_rl/<run>/model_999.pt" bash cmd/isaaclab.sh
```

## `get_bottle_coord_ws` / `bottle_position` 노드

`get_bottle_coord_ws` 안에는 ROS 2 패키지 `bottle_position`이 들어 있습니다.
이 노드는 검출 결과와 깊이 이미지, 카메라 정보를 이용해서 병의 3D 위치를 계산한 뒤,
지정한 좌표계로 변환해서 `PoseStamped` 메시지로 퍼블리시합니다.

### 동작 방식

- 입력:
  - `/detections_output` (`vision_msgs/Detection2DArray`)
  - `/Orbbec/depth/image_raw` (`sensor_msgs/Image`)
  - `/Orbbec/depth/camera_info` (`sensor_msgs/CameraInfo`)
- 출력:
  - `/bottle/position` (`geometry_msgs/PoseStamped`)
- 기본 좌표계:
  - `world`

### 주요 파라미터

- `target_class_id`: 추적할 클래스 ID, 기본값은 `39`
- `target_frame`: 최종 변환할 좌표계, 기본값은 `world`
- `detections_topic`: 검출 결과 토픽
- `depth_image_topic`: 깊이 이미지 토픽
- `camera_info_topic`: 카메라 정보 토픽
- `position_topic`: 병 위치를 publish할 토픽
- `search_radius`: 중심 픽셀에서 유효한 깊이를 찾기 위한 탐색 반경

### 실행 예시

```bash
source get_bottle_coord_ws/install/setup.bash
ros2 run bottle_position bottle_position_node
```

### 참고

- 검출 박스 중심 픽셀의 깊이가 0이거나 유효하지 않으면, 주변 픽셀을 `search_radius` 범위 안에서 탐색합니다.
- 유효한 깊이를 찾지 못하면 해당 프레임에서는 위치를 publish하지 않습니다.
- TF 변환이 가능한 경우에만 `target_frame` 기준 위치를 publish합니다.

## Isaac ROS CLI 설정

이 프로젝트의 Isaac ROS 실행은 `isaac-ros` CLI 설정을 전제로 합니다.
처음 세팅할 때 아래 위치와 파일 배치를 꼭 맞춰야 합니다.

- `config.yaml` 파일은 반드시 `$HOME/.config/isaac-ros-cli/config.yaml` 에 있어야 합니다.
- `Dockerfile.dofbot` 은 반드시 `/etc/isaac-ros-cli/docker/` 폴더 안에 있어야 합니다.
- Isaac ROS는 `isaac-ros --build-local` 명령으로 실행해야 합니다.

### 왜 필요한가

- `config.yaml`은 Isaac ROS CLI가 사용할 빌드/실행 설정을 읽는 기준 파일입니다.
- `Dockerfile.dofbot`이 지정된 폴더에 있어야 `isaac-ros` CLI가 로컬 빌드용 Docker 설정을 찾을 수 있습니다.
- `--build-local` 옵션을 사용해야 이 레포의 로컬 Docker 설정과 연결된 Isaac ROS 환경이 정상적으로 올라옵니다.

### 실무 체크리스트

1. `config.yaml`을 `$HOME/.config/isaac-ros-cli/config.yaml`에 둡니다.
2. `Dockerfile.dofbot`을 `/etc/isaac-ros-cli/docker/Dockerfile.dofbot`에 둡니다.
3. Isaac ROS 실행 전에 `isaac-ros --build-local` 흐름이 동작하는지 확인합니다.
4. `cmd/startup.sh`에서 Isaac ROS 관련 창도 같은 전제를 기준으로 실행합니다.

## 실행 구조 요약

이 프로젝트는 이 레포 하나만으로 끝나는 구조가 아니라, 아래 구성요소가 함께 맞물려 동작합니다.

- `Dockerfile.dofbot`
  - Isaac ROS 기반 이미지를 사용합니다.
  - `isaac-ros-yolov8` 등 YOLO 추론에 필요한 패키지를 설치합니다.
  - 즉, 검출 결과를 만드는 쪽을 담당합니다.
- `cmd/startup.sh`
  - Orbbec 카메라를 띄웁니다.
  - YOLO 검출 결과를 만들고, `bottle_position` 노드로 병의 3D 위치를 계산합니다.
  - 최종적으로 `/bottle/position` 토픽을 만들어 RL 쪽으로 넘깁니다.
- `cmd/isaaclab.sh`
  - Isaac Lab에서 정책을 실행합니다.
  - `/bottle/position`을 입력으로 받아 로봇이 목표 위치를 따라가게 합니다.

### 처음 보는 사람이 이해해야 할 전제

- Orbbec 카메라 드라이버가 ROS 2 토픽을 publish할 수 있어야 합니다.
- Isaac ROS YOLOv8이 `detections_output` 같은 검출 토픽을 publish하고 있어야 합니다.
- `bottle_position`은 검출 결과만으로는 동작하지 않고, 깊이 이미지와 `camera_info`도 함께 필요합니다.
- `cmd/startup.sh`는 `~/ros2_ws/install/setup.bash`를 source 하므로, `bottle_position` 패키지는 그 ROS 워크스페이스에 빌드되어 있어야 합니다.
- `cmd/isaaclab.sh`는 `$HOME/IsaacLab`를 기준으로 실행되므로, Isaac Lab도 해당 경로에 준비되어 있어야 합니다.

### 실행 흐름

1. Orbbec 카메라를 켭니다.
2. YOLOv8이 병을 검출해서 결과를 냅니다.
3. `bottle_position`이 검출 박스와 depth를 이용해 3D 좌표를 계산합니다.
4. 계산된 좌표가 `/bottle/position`으로 publish됩니다.
5. Isaac Lab 정책이 그 좌표를 받아 로봇 행동을 만듭니다.

### 주의할 점

- 검출은 color 기준, 위치 계산은 depth 기준이므로 두 영상의 정렬 상태가 중요합니다.
- TF 변환이 맞지 않으면 `bottle_position`은 좌표를 publish하지 못할 수 있습니다.
- `Dockerfile.dofbot`만으로 Orbbec 드라이버까지 자동 설치되는 구조는 아니므로, 카메라 쪽은 별도 환경 준비가 필요합니다.
