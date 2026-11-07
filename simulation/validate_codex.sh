#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")/.."
source /opt/ros/jazzy/setup.bash
source simulation/ros2_ws/install/setup.bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-87}"
export GZ_PARTITION="${GZ_PARTITION:-jerry_integrated}"
export GZ_IP="${GZ_IP:-127.0.0.1}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/jerry-codex-validation-ros}"
exec .venv-jerry/bin/python simulation/tests/validate_codex_robot.py
