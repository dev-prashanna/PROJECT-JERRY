#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")/.."
source /opt/ros/jazzy/setup.bash
source simulation/ros2_ws/install/setup.bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-87}"
export GZ_PARTITION="${GZ_PARTITION:-jerry_integrated}"
export GZ_IP="${GZ_IP:-127.0.0.1}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/jerry-ros}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source simulation/gui_environment.sh
jerry_python=.venv-jerry/bin/python
exec "$jerry_python" -m jerry.launcher "$@"
