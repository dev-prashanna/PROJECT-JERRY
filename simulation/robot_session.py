#!/usr/bin/env python3
"""Reuse or start Gazebo, then run one curious robot dashboard and controller."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Range
from rosgraph_msgs.msg import Clock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connect-only',action='store_true',help='Require an existing simulator')
    parser.add_argument('--server-headless',action='store_true',help='Start Gazebo without its 3D window')
    parser.add_argument('--monitor',action='store_true',help='Wait for Start movement instead of arming automatically')
    parser.add_argument('--mode',choices=['explore','avoid'],default='explore')
    args,extra=parser.parse_known_args()
    root=Path(__file__).resolve().parents[1]
    rclpy.init();node=Node('vbmd_session_launcher')
    seen={}
    topics=[('/clock',Clock),('/camera/image_raw',Image)]+[('/ultrasonic/'+s,Range) for s in ['front','left','rear','right']]
    subscriptions=[node.create_subscription(kind,topic,lambda msg,t=topic:seen.__setitem__(t,time.monotonic()),qos_profile_sensor_data) for topic,kind in topics]
    def poll(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            rclpy.spin_once(node,timeout_sec=.1)
    gazebo=None;vision=None
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        poll(2.)
        names=node.get_node_names()
        if 'vbmd_depth_avoidance' in names:
            raise RuntimeError('A robot dashboard/controller is already running. Close it before starting this session.')
        existing='vbmd_gazebo_adapter' in names or node.count_publishers('/clock')>0
        if not existing:
            if args.connect_only:
                raise RuntimeError('No Gazebo bridge found in ROS domain '+os.environ.get('ROS_DOMAIN_ID','86'))
            cmd=['bash',str(root/'simulation/run_gazebo.sh')]
            if args.server_headless:cmd.append('headless:=true')
            print('Starting Gazebo and its sensor/command bridge…',flush=True)
            gazebo=subprocess.Popen(cmd,cwd=root,start_new_session=True)
        else:
            print('Using the running Gazebo bridge.',flush=True)
        until=time.monotonic()+(5. if existing else 45.)
        while not all(time.monotonic()-seen.get(t,-1e9)<1. for t,_ in topics):
            if gazebo and gazebo.poll() is not None:
                raise RuntimeError('Gazebo exited before the robot sensors became ready')
            if time.monotonic()>until:
                missing=[t for t,_ in topics if time.monotonic()-seen.get(t,-1e9)>=1.]
                if existing and '--headless' not in extra:
                    print('Some sensors are paused/missing: '+', '.join(missing)+'. Opening the dashboard; use Resume Gazebo.',flush=True)
                    break
                raise RuntimeError('Gazebo paused or sensor bridge incomplete: '+', '.join(missing))
            poll(.2)
        print('Robot sensors connected. Loading vision and SLAM; movement waits for valid inputs.',flush=True)
        cmd=[sys.executable,str(root/'realtime_depth_app/robot_live.py'),'--mode',args.mode]
        if not args.monitor:cmd.append('--auto')
        cmd.extend(extra)
        vision=subprocess.Popen(cmd,cwd=root,start_new_session=True)
        return vision.wait()
    finally:
        # Stop only processes this launcher owns. Reused Gazebo sessions stay open.
        for process in [vision,gazebo]:
            if process and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit(130)
    except RuntimeError as exc:print(str(exc),file=sys.stderr);sys.exit(1)
