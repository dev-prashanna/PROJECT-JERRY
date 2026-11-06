#!/usr/bin/env python3
"""Project launcher, using the isolated simulation environment."""
from pathlib import Path
import os,sys
root=Path(__file__).resolve().parents[1]
python=root/'.venv-vbmd-sim/bin/python'
script=root/'simulation/ros2_ws/src/vbmd_description/scripts/run_sim.py'
if not python.exists():raise SystemExit('Run simulation/setup.sh first.')
os.execv(str(python),[str(python),str(script),*sys.argv[1:]])
