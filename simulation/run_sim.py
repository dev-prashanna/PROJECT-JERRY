#!/usr/bin/env python3
"""Launch the project's primary simulator: Gazebo Harmonic with ROS 2."""
from pathlib import Path
import os,sys
script=Path(__file__).resolve().with_name('run_gazebo.sh')
os.execv('/bin/bash',['bash',str(script),*sys.argv[1:]])
