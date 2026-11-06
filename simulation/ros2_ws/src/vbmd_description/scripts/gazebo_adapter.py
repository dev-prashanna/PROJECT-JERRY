#!/usr/bin/env python3
"""Four narrow beam proxies -> ultrasound Range; command watchdog; optional reactive avoidance."""
import math,time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan,Range,Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
from rclpy.parameter import Parameter
from rclpy.clock import Clock as RosClock,ClockType
SIDES=('front','left','rear','right')
def beam_minimum(ranges,minimum,maximum):
 if not ranges or any(math.isnan(r) for r in ranges):return math.nan
 if any(r==-math.inf or 0<=r<minimum for r in ranges):return -math.inf
 hits=[r for r in ranges if math.isfinite(r) and minimum<=r<=maximum]
 if hits:return min(hits)
 return math.inf if all(r>=maximum for r in ranges) else math.nan
class GazeboAdapter(Node):
 def __init__(self):
  super().__init__('vbmd_gazebo_adapter')
  self.declare_parameter('autonomy',False);self.cache={};self.ranges={s:math.inf for s in SIDES};self.index=0
  self.command=Twist();self.last_command=-1e9;self.last_command_wall=-1e9;self.last_scan_wall={};self.avoid_direction=0.;self.avoid_start=0.
  self.imu_output=self.create_publisher(Imu,'/imu/data_raw',qos_profile_sensor_data)
  self.create_subscription(Imu,'/sim/imu_raw',self.imu_raw,qos_profile_sensor_data)
  self.output=self.create_publisher(Twist,'/sim/cmd_vel_safe',10)
  self.pubs={s:self.create_publisher(Range,'/ultrasonic/'+s,10) for s in SIDES}
  self.subs=[self.create_subscription(LaserScan,'/sim/sonar_rays/'+s,lambda m,side=s:self.scan(side,m),qos_profile_sensor_data) for s in SIDES]
  self.create_subscription(Twist,'/cmd_vel',self.receive,10)
  self.create_subscription(Int32,'/sim/keyboard',self.keyboard,10)
  self.create_timer(.06,self.publish_range);self.create_timer(.02,self.drive,clock=RosClock(clock_type=ClockType.STEADY_TIME))
  self.get_logger().info('Gazebo adapter ready: one RGB camera + four ultrasound Range topics; no odometry or lidar scan exposed to navigation.')
 def imu_raw(self,msg):
  # The real MPU-6050 has no absolute orientation output; do not leak simulator attitude.
  msg.orientation.x=msg.orientation.y=msg.orientation.z=0.;msg.orientation.w=1.;msg.orientation_covariance=[-1.,0.,0.,0.,0.,0.,0.,0.,0.]
  self.imu_output.publish(msg)
 def seconds(self):return self.get_clock().now().nanoseconds*1e-9
 def scan(self,side,msg):
  self.cache[side]=msg;self.last_scan_wall[side]=time.monotonic()
 def receive(self,msg):
  if not all(math.isfinite(v) for v in [msg.linear.x,msg.angular.z]):msg=Twist()
  msg.linear.x=max(-.22,min(.22,msg.linear.x));msg.angular.z=max(-2.,min(2.,msg.angular.z))
  self.command=msg;self.last_command=self.seconds();self.last_command_wall=time.monotonic()
 def keyboard(self,msg):
  keys={87:(.10,0.),83:(-.08,0.),65:(0.,.8),68:(0.,-.8),88:(0.,0.),32:(0.,0.),16777235:(.10,0.),16777237:(-.08,0.),16777234:(0.,.8),16777236:(0.,-.8)}
  if msg.data not in keys:return
  self.set_parameters([Parameter('autonomy',value=False)])
  v,w=keys[msg.data];m=Twist();m.linear.x=v;m.angular.z=w;self.receive(m)
 def publish_range(self):
  side=SIDES[self.index%4];self.index+=1
  if side not in self.cache:return
  scan=self.cache[side]
  age=self.seconds()-(scan.header.stamp.sec+scan.header.stamp.nanosec*1e-9)
  if age>.30 or age<-.1:return
  # Keep the source measurement timestamp; never relabel stale data as newly acquired.
  val=beam_minimum(scan.ranges,scan.range_min,scan.range_max);self.ranges[side]=val
  out=Range();out.header=scan.header;out.header.frame_id=side+'_ultrasonic_frame';out.radiation_type=Range.ULTRASOUND
  out.field_of_view=float(scan.angle_max-scan.angle_min);out.min_range=scan.range_min;out.max_range=scan.range_max;out.range=val
  self.pubs[side].publish(out)
 def reactive(self):
  if not all(s in self.cache for s in SIDES):return 0.,0.
  if any(time.monotonic()-self.last_scan_wall.get(s,-1e9)>.5 for s in SIDES):return 0.,0.
  # Stop if camera/range simulation stalls or the current data becomes stale.
  now=self.seconds()
  if any(now-(m.header.stamp.sec+m.header.stamp.nanosec*1e-9)>.30 for m in self.cache.values()):return 0.,0.
  f=self.ranges['front'];l=self.ranges['left'];r=self.ranges['right']
  if self.avoid_direction:
   if f>.40 and now-self.avoid_start>1.:self.avoid_direction=0.
   else:return 0.,self.avoid_direction
  if f<.30:
   self.avoid_direction=-.9 if r>l else .9;self.avoid_start=now;return 0.,self.avoid_direction
  if l<.13:return .035,-.6
  if r<.13:return .035,.6
  return .08,0.
 def drive(self):
  if self.get_parameter('autonomy').value:
   v,w=self.reactive();m=Twist();m.linear.x=v;m.angular.z=w
  else:m=self.command if 0<=self.seconds()-self.last_command<.5 and time.monotonic()-self.last_command_wall<.5 else Twist()
  self.output.publish(m)
def main():
 rclpy.init();node=GazeboAdapter()
 try:rclpy.spin(node)
 except KeyboardInterrupt:pass
 finally:
  if rclpy.ok():node.output.publish(Twist())
  node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
