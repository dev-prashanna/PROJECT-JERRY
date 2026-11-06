"""Live Gazebo integration test. Uses scene poses only for evaluation.
Run with the Gazebo launch active and matching ROS_DOMAIN_ID / GZ_PARTITION / GZ_IP.
Sends a brief low-speed command, then verifies the watchdog stops the robot.
"""
import os,subprocess,json,time,math
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,CameraInfo,Range,JointState
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
import numpy as np,cv2

def scene_pose():
 env=dict(os.environ);env.setdefault('GZ_IP','127.0.0.1');env.setdefault('GZ_PARTITION','vbmd_dev')
 text=subprocess.run(['gz','topic','-e','-n','1','--json-output','-t','/world/vbmd_lab/dynamic_pose/info'],env=env,capture_output=True,text=True,timeout=10,check=True).stdout
 poses=json.loads(text)['pose'];p=next(p for p in poses if p['name']=='vbmd')
 xyz=[p.get('position',{}).get(k,0.) for k in ['x','y','z']]
 return xyz

def main():
 rclpy.init();n=Node('vbmd_gazebo_integration_check');seen={};subs=[]
 topics=[('/camera/image_raw',Image),('/camera/camera_info',CameraInfo),('/joint_states',JointState),('/sim/cmd_vel_safe',Twist),('/clock',Clock)]+[('/ultrasonic/'+s,Range) for s in ['front','left','rear','right']]
 for topic,kind in topics:subs.append(n.create_subscription(kind,topic,lambda m,t=topic:seen.__setitem__(t,m),qos_profile_sensor_data))
 pub=n.create_publisher(Twist,'/cmd_vel',10)
 def spin_until(predicate,timeout=20):
  until=time.monotonic()+timeout
  while not predicate():
   if time.monotonic()>until:raise AssertionError('Timeout waiting for simulator data/state')
   rclpy.spin_once(n,timeout_sec=.03)
 def simtime():
  m=seen['/clock'].clock;return m.sec+m.nanosec*1e-9
 try:
  spin_until(lambda:len(seen)==len(topics))
  img=seen['/camera/image_raw'];info=seen['/camera/camera_info']
  assert img.width==640 and img.height==480
  assert img.header.frame_id==info.header.frame_id=='camera_optical_frame'
  assert len(seen['/joint_states'].name)==4
  for side in ['front','left','rear','right']:
   r=seen['/ultrasonic/'+side];assert r.radiation_type==Range.ULTRASOUND;assert r.header.frame_id==side+'_ultrasonic_frame';assert .2<r.field_of_view<.3
  array=np.frombuffer(img.data,np.uint8).reshape(img.height,img.step)[:,:img.width*3].reshape(img.height,img.width,3)
  assert array.std()>15
  start=scene_pose();t=simtime();until=time.monotonic()+20
  while simtime()-t<1.2:
   assert time.monotonic()<until
   m=Twist();m.linear.x=.06;pub.publish(m);rclpy.spin_once(n,timeout_sec=.02)
  moved=scene_pose();t=simtime()
  # Cease refreshing the command; the adapter must expire it after 0.5 sim seconds.
  spin_until(lambda:simtime()-t>.8)
  safe=seen['/sim/cmd_vel_safe'];assert safe.linear.x==0 and safe.angular.z==0
  stopped=scene_pose();t=simtime();spin_until(lambda:simtime()-t>.4);later=scene_pose()
  distance=math.dist(start[:2],moved[:2]);drift=math.dist(stopped[:2],later[:2])
  assert .02<distance<.12,distance
  assert drift<.005,drift
  assert abs(later[2])<.003,later
  names=[a for a,b in n.get_topic_names_and_types()]
  assert '/scan' not in names and '/odom' not in names and '/sim/ground_truth/pose' not in names
  out=Path(__file__).resolve().parents[1]/'artifacts/gazebo';out.mkdir(exist_ok=True)
  cv2.imwrite(str(out/'esp32_camera.png'),array[:,:,::-1] if img.encoding=='rgb8' else array)
  report={'engine':'Gazebo Harmonic 8.15.0','rgb_size':[img.width,img.height],'camera_frame':img.header.frame_id,'camera_intrinsics':list(info.k),'wheel_joints':list(seen['/joint_states'].name),'ultrasound_topics':4,'forward_distance_m':distance,'command_watchdog_stopped':True,'stopped_drift_m':drift,'settled_base_height_m':later[2],'no_navigation_odometry_lidar_or_groundtruth':True,'scene_poses_used_only_for_test_evaluation':True,'ranges_m':{s:seen['/ultrasonic/'+s].range for s in ['front','left','rear','right']}}
  (out/'integration_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 finally:
  pub.publish(Twist());n.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
