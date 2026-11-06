"""MuJoCo mechanics and one-camera/four-ultrasonic observation model.
No true pose, depth image or wheel odometry is part of the robot observation.
"""
from pathlib import Path
import math
import mujoco
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
SIDES=('front','left','rear','right')
WHEELS=('front_left','rear_left','front_right','rear_right')
RADIUS=.03;TRACK=.136
CAM_WIDTH=640;CAM_HEIGHT=480;CAM_FOV_Y=55.
RANGE_MIN=.02;RANGE_MAX=4.;RANGE_FOV=math.radians(15)
class RobotSim:
 def __init__(self,scene=None,seed=7,noise=True):
  self.model=mujoco.MjModel.from_xml_path(str(scene or PACKAGE/'mjcf/room.xml'))
  self.data=mujoco.MjData(self.model);self.rng=np.random.default_rng(seed);self.noise=noise
  self.command=(0.,0.);self.last_command=-1e9;self.ranges={s:math.inf for s in SIDES};self.range_times={s:None for s in SIDES}
  self.next_sonar=0.;self.sonar_index=0;self.updated_sonar=None;self.renderer=None;self.avoid_direction=0.;self.avoid_start=0.
  self.actuators={w:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,w+'_wheel_motor') for w in WHEELS}
  self.sites={s:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,s+'_ultrasonic_site') for s in SIDES}
  mujoco.mj_forward(self.model,self.data)
 def set_twist(self,linear,angular):
  if not np.isfinite([linear,angular]).all():linear=angular=0.
  self.command=(float(np.clip(linear,-.22,.22)),float(np.clip(angular,-2.,2.)));self.last_command=self.data.time
 def advance(self,seconds=.01):
  n=max(1,round(seconds/self.model.opt.timestep))
  for _ in range(n):
   v,w=self.command if self.data.time-self.last_command<.5 else (0.,0.)
   l=(v-w*TRACK/2)/RADIUS;r=(v+w*TRACK/2)/RADIUS
   for wheel,i in self.actuators.items():self.data.ctrl[i]=l if 'left' in wheel else r
   mujoco.mj_step(self.model,self.data)
  mujoco.mj_forward(self.model,self.data)
  self.updated_sonar=None
  if self.data.time+1e-9>=self.next_sonar:
   side=SIDES[self.sonar_index%4];self.ranges[side]=self.sample_sonar(side);self.range_times[side]=self.data.time
   self.updated_sonar=side;self.sonar_index+=1;self.next_sonar=self.data.time+.06
 def sample_sonar(self,side):
  i=self.sites[side];pos=self.data.site_xpos[i];rot=self.data.site_xmat[i].reshape(3,3)
  # Nine rays approximate the horizontal ultrasound cone; group 0 contains room surfaces only.
  group=np.array([1,0,0,0,0,0],dtype=np.uint8);hits=[];gid=np.array([-1],dtype=np.int32)
  for yaw in np.linspace(-RANGE_FOV/2,RANGE_FOV/2,9):
   direction=rot@np.array([math.cos(yaw),math.sin(yaw),0.])
   dist=mujoco.mj_ray(self.model,self.data,pos,direction,group,1,-1,gid)
   if dist>=0:hits.append(dist)
  hit=min(hits,default=math.inf)
  if hit>RANGE_MAX:return math.inf
  if hit<RANGE_MIN:return -math.inf
  if self.noise:hit+=self.rng.normal(0,.004)
  return float(np.clip(hit,RANGE_MIN,RANGE_MAX))
 def camera(self):
  if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=CAM_HEIGHT,width=CAM_WIDTH)
  self.renderer.update_scene(self.data,camera='esp32_cam')
  return self.renderer.render().copy()
 def reactive_command(self):
  # Sensor-only local avoidance; explicitly not goal navigation or SLAM.
  front=self.ranges['front'];left=self.ranges['left'];right=self.ranges['right']
  if self.range_times['front'] is None:return 0.,0.
  if self.avoid_direction:
   if front>.40 and self.data.time-self.avoid_start>1.:self.avoid_direction=0.
   else:return 0.,self.avoid_direction
  if front<.30:
   self.avoid_direction=-.9 if right>left else .9;self.avoid_start=self.data.time
   return 0.,self.avoid_direction
  if left<.13:return .035,-.6
  if right<.13:return .035,.6
  return .09,0.
 def close(self):
  if self.renderer is not None:self.renderer.close();self.renderer=None
