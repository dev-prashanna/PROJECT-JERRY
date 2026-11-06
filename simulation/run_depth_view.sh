#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")/.."
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-86}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source simulation/gui_environment.sh
vbmd_python=python3
if [ -x .venv-jerry/bin/python ]; then vbmd_python=.venv-jerry/bin/python; fi
exec "$vbmd_python" realtime_depth_app/robot_live.py "$@"
