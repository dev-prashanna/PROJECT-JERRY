"""Push the movable box, then restore robot and box starting poses. Gazebo must be running."""
import os,subprocess,json,time,math
from pathlib import Path
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock

def run_gz(args):
 env=dict(os.environ);env.setdefault('GZ_IP','127.0.0.1');env.setdefault('GZ_PARTITION','vbmd_dev')
 return subprocess.run(['gz',*args],env=env,capture_output=True,text=True,check=True,timeout=10).stdout

def set_pose(name,x,y,z):
 req='name: "'+name+'", position: {x: '+str(x)+', y: '+str(y)+', z: '+str(z)+'}, orientation: {w: 1}'
 result=run_gz(['service','-s','/world/vbmd_lab/set_pose','--reqtype','gz.msgs.Pose','--reptype','gz.msgs.Boolean','--timeout','3000','--req',req])
 assert 'true' in result,result

def poses():
 raw=json.loads(run_gz(['topic','-e','-n','1','--json-output','-t','/world/vbmd_lab/dynamic_pose/info']))
 return {p['name']:[p.get('position',{}).get(k,0.) for k in ['x','y','z']] for p in raw['pose'] if p['name'] in ['vbmd','movable_box']}

def main():
 rclpy.init();n=Node('vbmd_box_push_check');clock=[None]
 sub=n.create_subscription(Clock,'/clock',lambda m:clock.__setitem__(0,m.clock.sec+m.clock.nanosec*1e-9),10)
 pub=n.create_publisher(Twist,'/cmd_vel',10)
 def wait(seconds,velocity=0.):
  end=time.monotonic()+25
  while clock[0] is None:
   assert time.monotonic()<end;rclpy.spin_once(n,timeout_sec=.03)
  start=clock[0]
  while clock[0]-start<seconds:
   assert time.monotonic()<end
   msg=Twist();msg.linear.x=velocity;pub.publish(msg);rclpy.spin_once(n,timeout_sec=.02)
 try:
  set_pose('vbmd',0,0,.003);set_pose('movable_box',.22,0,.061);wait(.7)
  before=poses();wait(2.5,.10);wait(.7);after=poses()
  shift=after['movable_box'][0]-before['movable_box'][0]
  assert shift>.03,(before,after)
  assert abs(after['movable_box'][2]-.06)<.015
  out=Path(__file__).resolve().parents[1]/'artifacts/gazebo/box_interaction.json'
  report={'engine':'Gazebo Harmonic','box_mass_kg':.06,'box_forward_displacement_m':shift,'physical_box_push_passed':True,'before_debug_m':before,'after_debug_m':after,'poses_restored_after_test':True}
  out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 finally:
  pub.publish(Twist());set_pose('movable_box',-.45,-.55,.061);set_pose('vbmd',0,0,.003)
  n.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
