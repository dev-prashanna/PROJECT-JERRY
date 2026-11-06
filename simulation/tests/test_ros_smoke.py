"""Run while sim.launch.py is running, with matching ROS_DOMAIN_ID."""
import rclpy,time,json
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
from sensor_msgs.msg import Image,CameraInfo,Range,JointState
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage
from pathlib import Path

def main():
 rclpy.init();n=Node('vbmd_sensor_smoke');seen={};subs=[]
 topics=[('/camera/image_raw',Image),('/camera/camera_info',CameraInfo),('/joint_states',JointState)]+[('/ultrasonic/'+s,Range) for s in ['front','rear','left','right']]
 for topic,kind in topics:subs.append(n.create_subscription(kind,topic,lambda m,t=topic:seen.__setitem__(t,m),qos_profile_sensor_data))
 subs.append(n.create_subscription(TFMessage,'/tf_static',lambda m:seen.__setitem__('/tf_static',m),QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)))
 pub=n.create_publisher(Twist,'/cmd_vel',10);start=time.monotonic()
 try:
  while time.monotonic()-start<5:
   rclpy.spin_once(n,timeout_sec=.05)
   if time.monotonic()-start>2:
    m=Twist();m.linear.x=.04;pub.publish(m)
  pub.publish(Twist())
  assert len(seen)==8,seen.keys()
  assert seen['/camera/image_raw'].width==640 and seen['/camera/image_raw'].height==480
  assert len(seen['/joint_states'].name)==4
  assert len(seen['/tf_static'].transforms)==13,len(seen['/tf_static'].transforms)
  names=[t for t,types in n.get_topic_names_and_types()]
  assert '/odom' not in names and '/scan' not in names and '/sim/ground_truth/pose' not in names
  report={'topics_received':list(seen),'rgb_size':[640,480],'joint_count':4,'fixed_tf_count':len(seen['/tf_static'].transforms),'no_fake_odometry_or_lidar':True,'ground_truth_disabled':True,'ultrasonic_ranges_m':{s:seen['/ultrasonic/'+s].range for s in ['front','rear','left','right']}}
  engine='gazebo' if 'vbmd_gazebo_adapter' in n.get_node_names() else 'mujoco'
  report['engine']=engine
  out=Path(__file__).resolve().parents[1]/'artifacts'/engine/'ros_smoke.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 finally:pub.publish(Twist());n.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
