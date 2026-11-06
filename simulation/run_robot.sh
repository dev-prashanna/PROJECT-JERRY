#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")/.."
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-86}"
export GZ_PARTITION="${GZ_PARTITION:-vbmd_dev}"
export GZ_IP="${GZ_IP:-127.0.0.1}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/vbmd-robot-ros}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source simulation/gui_environment.sh
vbmd_python=python3
if [ -x .venv-jerry/bin/python ]; then vbmd_python=.venv-jerry/bin/python; fi
exec "$vbmd_python" simulation/robot_session.py "$@"
