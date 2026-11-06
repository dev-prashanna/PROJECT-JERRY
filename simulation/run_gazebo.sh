#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")/.."
source /opt/ros/jazzy/setup.bash
source simulation/gui_environment.sh
if [ ! -f simulation/ros2_ws/install/setup.bash ]; then
  echo 'ROS workspace not built. Run simulation/setup.sh first.' >&2
  exit 1
fi
source simulation/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-86}"
export GZ_PARTITION="${GZ_PARTITION:-vbmd_dev}"
export GZ_IP="${GZ_IP:-127.0.0.1}"
exec ros2 launch vbmd_description gazebo.launch.py "$@"
