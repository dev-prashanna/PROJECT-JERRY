import math
import numpy as np
from src.robot_attitude import AttitudeFilter,wrap
from src.robot_avoidance import RoamingPolicy,RangeReading,analyze_depth,SIDES

def rotation(roll,pitch,yaw):
 r,p,y=roll,pitch,yaw
 rx=np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
 ry=np.array([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]])
 rz=np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
 return rz@ry@rx

def test_tilt_compensated_heading_on_tilted_robot():
 roll,pitch,yaw=map(math.radians,[12,-8,125]);r=rotation(roll,pitch,yaw)
 accel=r.T@np.array([0,0,9.81]);field=r.T@np.array([0,2.15e-5,-4.27e-5]);f=AttitudeFilter()
 for i in range(100):f.update_imu(i*.01,[0,0,0],accel);f.update_mag(i*.01,field)
 assert abs(f.roll-roll)<1e-5 and abs(f.pitch-pitch)<1e-5
 assert abs(wrap(f.yaw-yaw))<1e-5
 assert abs(f.snapshot()['heading_deg']-325)<1e-4

def test_gyro_integration_and_rejection_of_invalid_fields():
 f=AttitudeFilter();f.update_imu(0,[0,0,0],[0,0,9.81]);f.update_mag(0,[0,2.15e-5,-4.27e-5])
 for i in range(1,101):f.update_imu(i*.01,[0,0,.3],[0,0,9.81])
 assert abs(f.yaw-.3)<.01
 old=f.yaw;f.update_mag(1.1,[np.nan,0,0]);assert f.yaw==old
 assert not AttitudeFilter().snapshot()['ready']

def view():
 d=np.zeros((100,100),np.float32);d[80:]=1;d[20:70,6:94]=.2
 return analyze_depth(d)
def ranges(**change):
 rs={s:RangeReading(1.,.02,.02) for s in SIDES}
 for s,v in change.items():rs[s]=RangeReading(v,.02,.02)
 return rs
def decide(p,t,yaw=0.,rs=None,ready=True):return p.decide(view(),rs or ranges(),t,.02,.02,.02,yaw,ready)

def test_roaming_holds_travel_direction_for_ten_seconds_then_changes():
 p=RoamingPolicy(seed=9);decide(p,0);target=p.target
 decide(p,.1,yaw=target);assert p.phase=='TRAVEL'
 d=decide(p,10.,yaw=target);assert d.state=='ROAM' and p.target==target
 d=decide(p,10.11,yaw=target);assert d.state=='SELECT'
 decide(p,10.16,yaw=target);assert abs(wrap(p.target-target))>.15

def test_obstacle_interrupts_timer_brakes_then_selects_clearer_escape():
 p=RoamingPolicy(seed=9);decide(p,0);decide(p,.1,yaw=p.target)
 d=decide(p,2.,yaw=p.target,rs=ranges(front=.35,left=1.5,right=.27))
 assert d.state=='BRAKE' and d.linear==d.angular==0
 d=decide(p,2.4,yaw=p.target,rs=ranges(front=.35,left=1.5,right=.27))
 assert d.state=='REORIENT' and d.linear==0 and d.angular>0

def test_roaming_needs_fresh_heading_and_preserves_emergency_stop():
 p=RoamingPolicy(seed=9)
 d=decide(p,0,ready=False);assert d.linear==d.angular==0 and d.state=='STOP'
 d=decide(p,1,rs=ranges(front=.1));assert d.linear==d.angular==0

def test_dashboard_fails_closed_if_control_thread_stops_with_old_advance_state():
 import threading,time
 from types import SimpleNamespace
 from robot_live import RobotNode
 from src.robot_avoidance import Decision
 sent=[];now=time.monotonic()
 fake=SimpleNamespace(lock=threading.RLock(),frame=SimpleNamespace(received=now-10),prediction=SimpleNamespace(packet=SimpleNamespace(received=now-10)),clock_received=now-10,readings=lambda _: {},decision=Decision(.08,.1,'ADVANCE','old command'),active=True,roaming=SimpleNamespace(reset=lambda:None),policy=SimpleNamespace(reset=lambda:None),command_pub=SimpleNamespace(publish=sent.append))
 _,_,_,decision,active=RobotNode.snapshot(fake)
 assert decision.state=='STOP' and decision.linear==decision.angular==0 and not active
 assert len(sent)==1 and sent[0].linear.x==0 and sent[0].angular.z==0

def test_flat_floor_visual_translation_uses_geometry_not_commands():
 from src.robot_mapping import GroundVisualOdometry
 vo=GroundVisualOdometry();vo.k=(461.,461.,320.,240.)
 world=np.array([[x,y,0.] for x in np.linspace(.25,1.,10) for y in np.linspace(-.15,.15,6)])
 a0=(0.,0.,.1);a1=(.01,-.01,.14);shift=np.array([.025,-.004])
 def project(base,rpy):
  R=rotation(*rpy);b=(world-np.r_[base,0.])@R-np.array([.0757,0,.072])
  return np.column_stack([320-461*b[:,1]/b[:,0],240-461*b[:,2]/b[:,0]])
 old=project(np.zeros(2),a0);new=project(shift,a1)
 delta,count=vo.translation(old,new,a0,a1,.2)
 assert count>=6 and np.allclose(delta,shift,atol=1e-6)

def test_range_cones_build_unknown_free_occupied_cells_and_frontiers():
 from src.robot_mapping import ExplorationMap
 m=ExplorationMap()
 for _ in range(4):
  for side in SIDES:m.observe((0,0,0),side,1.)
 assert m.logodds[m.cell(.5,0)]<-.3
 assert m.logodds[m.cell(1.08,0)]>.45
 assert m.logodds[m.cell(2.5,2.5)]==0
 assert m.frontiers((0,0,0))
