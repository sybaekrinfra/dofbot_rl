#!/bin/bash

SESSION="yolo_bottle"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"

tmux has-session -t $SESSION 2>/dev/null

if [ $? != 0 ]; then
    # 0: status check
    tmux new-session -d -s $SESSION -n no_working
    tmux send-keys -t $SESSION:0 '
nvidia-smi
' C-m

    # 1: Orbbec camera
    tmux new-window -t $SESSION -n orbbec_camera
    tmux send-keys -t $SESSION:1 '
source /opt/ros/jazzy/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py camera_name:=Orbbec
' C-m

    # 2: camera_info relay
    tmux new-window -t $SESSION -n camera_info
    tmux send-keys -t $SESSION:2 '
sleep 2
source /opt/ros/jazzy/setup.bash
ros2 run topic_tools relay /Orbbec/color/camera_info /camera_info_rect
' C-m

    # 3: image relay
    tmux new-window -t $SESSION -n image_rect
    tmux send-keys -t $SESSION:3 '
sleep 2
source /opt/ros/jazzy/setup.bash
ros2 run topic_tools relay /Orbbec/color/image_raw /image_rect
' C-m

    # 4: Isaac ROS YOLOv8
    tmux new-window -t $SESSION -n isaac_ros
    tmux send-keys -t $SESSION:4 '
isaac-ros activate --build-local
cd /workspaces/isaac_ros-dev
ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
  launch_fragments:=yolov8 \
  interface_specs_file:=${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_yolov8/quickstart_interface_specs.json \
  engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/yolov8s.plan
' C-m

    # 5: YOLO visualizer
    tmux new-window -t $SESSION -n yolov8_visualizer
    tmux send-keys -t $SESSION:5 '
sleep 5
isaac-ros activate --build-local
ros2 run isaac_ros_yolov8 isaac_ros_yolov8_visualizer.py
' C-m

    # 6: bottle position
    tmux new-window -t $SESSION -n bottle_position
    tmux send-keys -t $SESSION:6 '
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run bottle_position bottle_position_node --ros-args -p target_frame:=Orbbec_link
' C-m

    # 7: rqt image view
    tmux new-window -t $SESSION -n rqt_image
    tmux send-keys -t $SESSION:7 '
source /opt/ros/jazzy/setup.bash
ros2 run rqt_image_view rqt_image_view /yolov8_processed_image
' C-m
git -C "$HOME/dofbot_rl" remote get-url origin
git -C "$HOME/dofbot_rl" status --short
git -C "$HOME/dofbot_rl" rev-parse HEAD
git -C "$HOME/dofbot_rl" describe --tags --always
    # 8: tensorboard
    tmux new-window -t $SESSION -n tensorboard
    tmux send-keys -t $SESSION:8 "
source \"$CONDA_SH\"
conda activate env_isaaclab
cd \"$ISAACLAB_DIR\"
tensorboard --logdir logs/rsl_rl --port 6006
" C-m

fi

tmux attach -t $SESSION
