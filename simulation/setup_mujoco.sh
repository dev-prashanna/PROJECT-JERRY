#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")/.."
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
"$UV_BIN" venv --system-site-packages --allow-existing .venv-vbmd-sim
"$UV_BIN" pip install --python .venv-vbmd-sim/bin/python -r simulation/requirements.txt
source /opt/ros/jazzy/setup.bash
cd simulation/ros2_ws
colcon build --symlink-install --packages-select vbmd_description
