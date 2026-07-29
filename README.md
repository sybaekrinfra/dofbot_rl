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
  --checkpoint logs/rsl_rl/dofbot_reach_ik_direct_refine/<run>/model_999.pt
```

`--checkpoint`를 지정하지 않으면 `play_rsl_rl.py`가 해당 task의 최신 `model_*.pt`를 자동으로 찾습니다.

## 관측 / 행동 / 보상

### 관측

관측값은 아래 항목으로 구성됩니다.

- 5개 joint position
- 5개 joint velocity
- 3개 end-effector position
- 3개 target position
- 5개 previous action

### 행동

- 5차원 연속 제어
- 각 action은 joint position increment로 해석

### 보상

- end-effector와 target 사이 거리 기반 보상
- action penalty
- 성공 보너스

### 종료 조건

- end-effector가 target에 충분히 가까워지면 성공 종료
- 샘플 수가 길어지면 timeout 종료

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
