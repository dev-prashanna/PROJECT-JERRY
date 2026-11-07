#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")/.."
exec .venv-jerry/bin/python simulation/tests/check_robot_mcp.py
