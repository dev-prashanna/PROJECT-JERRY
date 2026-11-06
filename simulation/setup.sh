#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")/.."
source /opt/ros/jazzy/setup.bash
command -v gz >/dev/null
python3 simulation/ros2_ws/src/vbmd_description/scripts/generate_models.py
python3 simulation/ros2_ws/src/vbmd_description/scripts/generate_gazebo.py
cd simulation/ros2_ws
colcon build --symlink-install --packages-select vbmd_description
