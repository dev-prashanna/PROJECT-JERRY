"""ROS 2 sensor/command adapter. True pose only appears with --ground-truth."""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, Range, JointState
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from rosgraph_msgs.msg import Clock
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time
import mujoco
from sim_core import CAM_WIDTH,CAM_HEIGHT,CAM_FOV_Y,RANGE_MIN,RANGE_MAX,RANGE_FOV,SIDES,WHEELS,RADIUS,TRACK

def stamp(t):
 ns=round(t*1e9);return Time(sec=ns//1000000000,nanosec=ns%1000000000)
class SensorBridge(Node):
 def __init__(self,sim,ground_truth=False):
  super().__init__('vbmd_mujoco');self.sim=sim;self.ground_truth=ground_truth
  self.create_subscription(Twist,'/cmd_vel',lambda m:sim.set_twist(m.linear.x,m.angular.z),10)
  self.clock=self.create_publisher(Clock,'/clock',10)
  self.cam=self.create_publisher(Image,'/camera/image_raw',qos_profile_sensor_data)
  self.info=self.create_publisher(CameraInfo,'/camera/camera_info',qos_profile_sensor_data)
  self.range_pubs={s:self.create_publisher(Range,'/ultrasonic/'+s,10) for s in SIDES}
  self.joints=self.create_publisher(JointState,'/joint_states',10)
  self.debug=self.create_publisher(PoseStamped,'/sim/ground_truth/pose',10) if ground_truth else None
  self.tf=TransformBroadcaster(self) if ground_truth else None
  self.next_camera=0.;self.next_state=0.
  self.get_logger().info('RGB + four sequential ultrasound topics ready. No /odom or /scan. Ground truth '+('ENABLED for debug' if ground_truth else 'disabled')+'.')
 def publish(self,render=True):
  s=self.sim;now=stamp(s.data.time);c=Clock();c.clock=now;self.clock.publish(c)
  side=s.updated_sonar
  if side:
   r=Range();r.header.stamp=now;r.header.frame_id=side+'_ultrasonic_frame';r.radiation_type=Range.ULTRASOUND
   r.field_of_view=RANGE_FOV;r.min_range=RANGE_MIN;r.max_range=RANGE_MAX;r.range=float(s.ranges[side]);self.range_pubs[side].publish(r)
  if s.data.time+1e-9>=self.next_camera and render:
   rgb=s.camera();m=Image();m.header.stamp=now;m.header.frame_id='camera_optical_frame';m.height=CAM_HEIGHT;m.width=CAM_WIDTH;m.encoding='rgb8';m.is_bigendian=0;m.step=CAM_WIDTH*3;m.data=rgb.tobytes();self.cam.publish(m)
   f=CAM_HEIGHT/(2*math.tan(math.radians(CAM_FOV_Y)/2));cx=(CAM_WIDTH-1)/2;cy=(CAM_HEIGHT-1)/2
   k=CameraInfo();k.header=m.header;k.height=CAM_HEIGHT;k.width=CAM_WIDTH;k.distortion_model='plumb_bob';k.d=[0.]*5;k.k=[f,0.,cx,0.,f,cy,0.,0.,1.];k.r=[1.,0.,0.,0.,1.,0.,0.,0.,1.];k.p=[f,0.,cx,0.,0.,f,cy,0.,0.,0.,1.,0.];self.info.publish(k)
   self.next_camera=s.data.time+.1
  if s.data.time+1e-9>=self.next_state:
   # Simulator joint angles serve robot visualization only, never integrated into /odom.
   j=JointState();j.header.stamp=now
   for wh in WHEELS:
    name=wh+'_wheel_joint';i=mujoco.mj_name2id(s.model,mujoco.mjtObj.mjOBJ_JOINT,name)
    j.name.append(name);j.position.append(float(s.data.qpos[s.model.jnt_qposadr[i]]));j.velocity.append(float(s.data.qvel[s.model.jnt_dofadr[i]]))
   self.joints.publish(j)
   if self.debug:
    p=PoseStamped();p.header.stamp=now;p.header.frame_id='sim_world';p.pose.position.x=float(s.data.qpos[0]);p.pose.position.y=float(s.data.qpos[1]);p.pose.position.z=float(s.data.qpos[2]);q=s.data.qpos[3:7]
    p.pose.orientation.w=float(q[0]);p.pose.orientation.x=float(q[1]);p.pose.orientation.y=float(q[2]);p.pose.orientation.z=float(q[3]);self.debug.publish(p)
    tr=TransformStamped();tr.header=p.header;tr.child_frame_id='base_footprint';tr.transform.translation.x=p.pose.position.x;tr.transform.translation.y=p.pose.position.y;tr.transform.translation.z=p.pose.position.z;tr.transform.rotation=p.pose.orientation;self.tf.sendTransform(tr)
   self.next_state=s.data.time+.02
