"""Observe a live exploration run. Scene poses are evaluation-only, never published.

Run after starting run_robot.sh, in its ROS domain and Gazebo partition.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,Range
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from gz.transport13 import Node as GazeboNode
from gz.msgs10.pose_v_pb2 import Pose_V


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration',type=float,default=90.)
    parser.add_argument('--require-objects',action='store_true')
    parser.add_argument('--require-completed-goals',type=int,default=0)
    parser.add_argument('--max-tracking-p95-ms',type=float)
    parser.add_argument('--output',type=Path,default=Path('simulation/artifacts/curiosity/check.json'))
    args=parser.parse_args()
    os.environ.setdefault('GZ_PARTITION','vbmd_dev');os.environ.setdefault('GZ_IP','127.0.0.1')
    positions=[];pose_lock=threading.Lock();last_pose_time=[-1e9];autonomy_active=[False]
    def poses(msg):
        now=time.monotonic()
        if now-last_pose_time[0]<.2:return
        for p in msg.pose:
            if p.name=='vbmd':
                with pose_lock:
                    positions.append([p.position.x,p.position.y,p.position.z,float(autonomy_active[0])]);last_pose_time[0]=now
                break
    transport=GazeboNode();transport.subscribe(Pose_V,'/world/vbmd_lab/dynamic_pose/info',poses)
    rclpy.init();node=Node('vbmd_curiosity_evaluation')
    counts=Counter();states=Counter();areas=[];moving=Counter();errors=[];last={}
    destinations=[];premature_changes=[];last_destination=[None];previous_command=[None]
    max_acceleration=[0.];max_yaw_acceleration=[0.]
    tracking_times=[];localization_states=Counter();recovery_transitions=[0];last_recovery=[False]
    states_by_goal={}
    depth_ages=[];inference_times=[];stop_reasons=Counter();stationary_since=[None];longest_stationary=[0.]
    def status(msg):
        s=json.loads(msg.data);states[s['state']]+=1
        autonomy_active[0]=bool(s.get('active') and s.get('mode')=='explore')
        slam=s.get('slam',{});goal=slam.get('goal');goal_id=slam.get('goal_id')
        if goal_id:states_by_goal.setdefault(str(goal_id),Counter())[s['state']]+=1
        if slam.get('processing_ms') is not None:tracking_times.append(slam['processing_ms'])
        if s.get('inference_ms') is not None and slam.get('keyframes',0)>0:
            depth_ages.append(s['depth_age_wall_s']);inference_times.append(s['inference_ms'])
            if s['state']=='STOP':stop_reasons[s.get('reason','')]+=1
            stationary=s.get('active') and abs(s['linear_m_s'])<.005 and abs(s['yaw_rad_s'])<.02
            if stationary:
                if stationary_since[0] is None:stationary_since[0]=time.monotonic()
                longest_stationary[0]=max(longest_stationary[0],time.monotonic()-stationary_since[0])
            else:stationary_since[0]=None
            localization_states[s['state']]+=1
            recovering=s['state'] in ['RELOCALIZE','RECOVERING'] or (s['state']=='STOP' and 'SLAM' in s.get('reason',''))
            if recovering and not last_recovery[0]:recovery_transitions[0]+=1
            last_recovery[0]=recovering
        if goal is not None and goal_id is not None:
            identity=(goal_id,tuple(goal))
            old=last_destination[0]
            if identity!=old:
                entry={'goal_id':goal_id,'goal':goal,'completed_goals':slam.get('completed_goals',0),'simulation_time_s':s.get('simulation_time_s')}
                if old is not None and slam.get('completed_goals',0)<old[0]:premature_changes.append(entry)
                destinations.append(entry);last_destination[0]=identity
        previous=previous_command[0]
        if previous and s.get('active') and previous.get('active'):
            dt=s.get('command_time_s',0)-previous.get('command_time_s',0)
            immediate=['STOP','WAITING','PLANNING','NOTICE','STARTLED','ARRIVING','ARRIVED','TURN','RELOCALIZE','RECOVERING','TRACKING_BRIDGE']
            if dt>1e-5 and s['state'] not in immediate and previous['state'] not in immediate:
                max_acceleration[0]=max(max_acceleration[0],abs(s['linear_m_s']-previous['linear_m_s'])/dt)
                max_yaw_acceleration[0]=max(max_yaw_acceleration[0],abs(s['yaw_rad_s']-previous['yaw_rad_s'])/dt)
        previous_command[0]=s
        if s.get('ros_error'):errors.append(s['ros_error'])
        last.update(s)
    def image(msg,topic):counts[topic]+=1
    def objects(msg):
        data=json.loads(msg.data)
        assert data['coordinate_units']=='image pixels' and 'objects' in data
        counts['yolo_frames']+=1
        counts['object_boxes']+=len(data['objects'])
    def grid(msg):
        counts['map']+=1
        data=np.asarray(msg.data)
        areas.append(float(np.sum(data>=0)*msg.info.resolution**2))
    def command(msg):
        counts['safe_commands']+=1
        if abs(msg.linear.x)>.005:moving['translation']+=1
        if abs(msg.angular.z)>.05:moving['rotation']+=1
    subscriptions=[node.create_subscription(String,'/perception/avoidance_status',status,10),
        node.create_subscription(OccupancyGrid,'/map',grid,1),
        node.create_subscription(Twist,'/sim/cmd_vel_safe',command,10)]
    subscriptions.append(node.create_subscription(String,'/perception/objects',objects,10))
    for topic in ['/camera/image_raw','/perception/depth_relative']:
        subscriptions.append(node.create_subscription(Image,topic,lambda m,t=topic:image(m,t),qos_profile_sensor_data))
    for side in ['front','left','rear','right']:
        topic='/ultrasonic/'+side
        subscriptions.append(node.create_subscription(Range,topic,lambda m,t=topic:counts.update([t]),qos_profile_sensor_data))
    until=time.monotonic()+args.duration
    try:
        while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.1)
    finally:
        node.destroy_node();rclpy.try_shutdown()
    with pose_lock:points=np.asarray(positions)
    distance=float(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1).sum()) if len(points)>1 else 0.
    extent=float(np.linalg.norm(np.ptp(points[:,:2],axis=0))) if len(points)>1 else 0.
    autonomous_distance=0.
    if len(points)>1:
        automatic=(points[:-1,3]>0)&(points[1:,3]>0)
        autonomous_distance=float(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1)[automatic].sum())
    report={'topics_received':dict(counts),'states':dict(states),'motion_commands':dict(moving),
            'actual_path_length_m':distance,'actual_exploration_extent_m':extent,
            'autonomous_curiosity_path_length_m':autonomous_distance,
            'states_by_destination':{k:dict(v) for k,v in states_by_goal.items()},
            'destinations':destinations,'destination_changes_without_arrival':premature_changes,
            'max_normal_linear_acceleration_m_s2':max_acceleration[0],
            'max_normal_yaw_acceleration_rad_s2':max_yaw_acceleration[0],
            'states_after_model_warmup':dict(localization_states),'localization_recovery_entries':recovery_transitions[0],
            'visual_processing_p95_ms':float(np.percentile(tracking_times,95)) if tracking_times else None,
            'visual_processing_max_ms':max(tracking_times) if tracking_times else None,
            'depth_source_age_p95_s':float(np.percentile(depth_ages,95)) if depth_ages else None,
            'depth_inference_p95_ms':float(np.percentile(inference_times,95)) if inference_times else None,
            'stop_reasons_after_model_warmup':dict(stop_reasons),
            'longest_stationary_command_s':longest_stationary[0],
            'observed_area_first_m2':areas[0] if areas else 0.,'observed_area_last_m2':areas[-1] if areas else 0.,
            'last_state':last,'scene_poses_used_only_for_evaluation':True,'errors':errors}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='last_state'},indent=2))
    assert counts['/camera/image_raw']>10 and counts['/perception/depth_relative']>10
    assert all(counts['/ultrasonic/'+s]>10 for s in ['front','left','rear','right'])
    assert counts['map']>5 and moving['translation']>10 and moving['rotation']>10
    assert distance>.25 and extent>.15,'Robot did not explore a meaningful distance'
    assert autonomous_distance>.25,'Insufficient autonomous curiosity motion; manual driving is excluded'
    assert max(areas)-min(areas)>.5,'Occupancy map did not grow'
    assert not premature_changes,'An exploration destination changed before arrival'
    if args.require_objects:assert counts['yolo_frames']>5,'Live YOLO results were not published'
    assert last.get('slam',{}).get('completed_goals',0)>=args.require_completed_goals,'Too few confirmed destination arrivals'
    if args.max_tracking_p95_ms is not None:
        assert report['visual_processing_p95_ms'] is not None and report['visual_processing_p95_ms']<=args.max_tracking_p95_ms,'Visual tracking exceeded its latency budget'
    if last.get('command_time_s'):
        assert max_acceleration[0]<=.12001 and max_yaw_acceleration[0]<=.60001,'Normal motion exceeded smoothing limits'


if __name__=='__main__':main()
