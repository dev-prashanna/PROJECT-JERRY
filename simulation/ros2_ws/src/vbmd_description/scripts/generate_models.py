#!/usr/bin/env python3
"""Generate URDF and MuJoCo models from the CAD mesh manifest. SI units."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json, math
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/'meshes/manifest.json').read_text())
def e(parent,tag,**attrs):return ET.SubElement(parent,tag,{k:str(v) for k,v in attrs.items()})
def xyz(values):return ' '.join(f'{v:.9g}' for v in values)
def write(root,path):
 ET.indent(root,space='  ');ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
def inertial(link,mass,diag,com=(0,0,0)):
 i=e(link,'inertial');e(i,'origin',xyz=xyz(com));e(i,'mass',value=mass)
 e(i,'inertia',ixx=diag[0],iyy=diag[1],izz=diag[2],ixy=0,ixz=0,iyz=0)

def origin(parent,pos=(0,0,0),rpy=(0,0,0)):e(parent,'origin',xyz=xyz(pos),rpy=xyz(rpy))
def collision(link,kind,size,pos=(0,0,0),rpy=(0,0,0)):
 c=e(link,'collision');origin(c,pos,rpy);g=e(c,'geometry')
 if kind=='box':e(g,'box',size=xyz(size))
 else:e(g,'cylinder',radius=size[0],length=size[1])

robot=ET.Element('robot',name='vbmd_robot')
mu=e(robot,'mujoco');e(mu,'compiler',meshdir='../meshes',discardvisual='false',fusestatic='false',strippath='true')
root=e(robot,'link',name='base_footprint')
base=e(robot,'link',name='base_link');inertial(base,.55,[.0009,.0013,.0017],(0,0,.047))
j=e(robot,'joint',name='base_fixed',type='fixed');e(j,'parent',link='base_footprint');e(j,'child',link='base_link');origin(j)
collision(base,'box',(.15,.10,.035),(0,0,.0425))
collision(base,'box',(.135,.09,.045),(-.005,0,.0875))
for name,info in MANIFEST['links'].items():
 link=base if name=='base_link' else e(robot,'link',name=name)
 if name!='base_link':
  wheel='wheel' in name; mass=.040 if wheel else .009
  diag=[1.092e-5,1.8e-5,1.092e-5] if wheel else [2.e-6,2.e-6,2.e-6]
  inertial(link,mass,diag)
  joint=e(robot,'joint',name=name.replace('_link','_joint'),type='continuous' if wheel else 'fixed')
  e(joint,'parent',link='base_link');e(joint,'child',link=name)
  origin(joint,[v*.001 for v in info['origin_mm']],(0,0,math.radians(info['yaw_degrees'])))
  if wheel:
   e(joint,'axis',xyz='0 1 0');e(joint,'limit',effort='.10',velocity='30');e(joint,'dynamics',damping='.00005',friction='.00005')
   collision(link,'cylinder',(.03,.024),rpy=(math.pi/2,0,0))
  elif 'ultrasonic' in name:collision(link,'box',(.016,.045,.020),(-.003,0,0))
  else:collision(link,'box',(.012,.027,.04),(-.009,0,.006))
 for idx,v in enumerate(info['visuals']):
  vis=e(link,'visual',name=f'{name}_color_{idx}');origin(vis)
  geom=e(vis,'geometry');e(geom,'mesh',filename='package://vbmd_description/meshes/'+v['mesh'])
  mat=e(vis,'material',name=f'{name}_mat_{idx}');e(mat,'color',rgba=xyz(v['rgba']))
 if 'ultrasonic' in name:
  frame=name.replace('_link','_frame');e(robot,'link',name=frame)
  j=e(robot,'joint',name=frame+'_fixed',type='fixed');e(j,'parent',link=name);e(j,'child',link=frame);origin(j,(.0126,0,0))
# ROS camera_link: +X forward, optical frame: +Z forward, +X right, +Y down.
e(robot,'link',name='camera_optical_frame')
j=e(robot,'joint',name='camera_optical_fixed',type='fixed');e(j,'parent',link='camera_link');e(j,'child',link='camera_optical_frame');origin(j,rpy=(-math.pi/2,0,-math.pi/2))
# Sensor chip frames; CAD module visuals remain part of the base visual meshes.
for name,pos in [('imu_link',(0,-.020,.04955)),('magnetometer_link',(-.028,.043,.1345))]:
 e(robot,'link',name=name)
 j=e(robot,'joint',name=name+'_fixed',type='fixed');e(j,'parent',link='base_link');e(j,'child',link=name);origin(j,pos)
write(robot,ROOT/'urdf/vbmd.urdf')
# MJCF uses the same meshes and fixed link poses, plus explicit collision geoms and actuators.
mj=ET.Element('mujoco',model='VBMD camera and four sonar robot')
e(mj,'compiler',angle='radian',meshdir='../meshes',inertiafromgeom='false',autolimits='true')
e(mj,'option',timestep='.002',gravity='0 0 -9.81',integrator='implicitfast',cone='elliptic',iterations='60')
visual=e(mj,'visual');e(visual,'global',offwidth='1600',offheight='1200');e(visual,'quality',shadowsize='4096',offsamples='4');e(visual,'map',znear='.002',zfar='10')
default=e(mj,'default');e(default,'joint',damping='.00005',armature='.000001');e(default,'geom',solref='.008 1',solimp='.95 .99 .001',friction='.35 .002 .0001')
asset=e(mj,'asset')
e(asset,'texture',name='floor_tex',type='2d',builtin='checker',rgb1='.25 .28 .30',rgb2='.39 .42 .43',width='512',height='512')
e(asset,'material',name='floor_mat',texture='floor_tex',texrepeat='18 14',reflectance='.08',roughness='.8')
for name,info in MANIFEST['links'].items():
 for v in info['visuals']:
  n=Path(v['mesh']).stem;e(asset,'mesh',name=n,file=v['mesh'])
  metal=any(abs(v['rgba'][i]-[.70,.75,.79][i])<.001 for i in range(3))
  e(asset,'material',name=n+'_mat',rgba=xyz(v['rgba']),specular='.65' if metal else '.25',shininess='.5' if metal else '.15',reflectance='.08' if metal else '0')
world=e(mj,'worldbody');e(world,'light',name='key',pos='0 -1 3',dir='0 .2 -1',diffuse='.85 .85 .85',castshadow='true')
e(world,'light',name='fill',pos='-1 1 2',dir='.2 -.2 -1',diffuse='.4 .4 .4',castshadow='false')
e(world,'geom',name='floor',type='plane',size='3 3 .1',material='floor_mat',group='0')
# Enclosed textured test room; asymmetric landmarks support image tracking.
for name,pos,size,color in [('north',(0,1.5,.7),(2,.03,.7),(.65,.68,.7)),('south',(0,-1.5,.7),(2,.03,.7),(.68,.65,.6)),('east',(2,0,.7),(.03,1.5,.7),(.65,.68,.7)),('west',(-2,0,.7),(.03,1.5,.7),(.68,.65,.6))]:
 e(world,'geom',name=name+'_wall',type='box',pos=xyz(pos),size=xyz(size),rgba=xyz([*color,1]),group='0')
for i in range(24):
 y=-1.38+i*.12
 e(world,'geom',name=f'front_landmark_{i}',type='box',pos=xyz([1.965,y,.10+(i%5)*.075]),size=xyz([.006,.025+.005*(i%3),.018+.005*(i%4)]),rgba=xyz([.1+(i%3)*.28,.1+(i%4)*.19,.12+(i%2)*.5,1]),group='0')
for i in range(18):
 x=-1.75+i*.2
 for sy in [-1,1]:
  e(world,'geom',name=f'wall_patch_{i}_{sy}',type='box',pos=xyz([x,sy*1.465,.08+(i%4)*.10]),size=xyz([.025,.006,.025]),rgba=xyz([.12+(i%3)*.24,.15+(i%4)*.17,.18+(i%2)*.48,1]),group='0')
for i,(pos,size,color) in enumerate([((.85,.08,.13),(.12,.15,.13),(.55,.12,.08,1)),((-.65,.50,.10),(.10,.10,.10),(.08,.32,.50,1)),((.20,-.65,.18),(.08,.11,.18),(.58,.42,.05,1)),((1.15,.75,.08),(.16,.12,.08),(.12,.40,.18,1))]):
 e(world,'geom',name=f'obstacle_{i}',type='box',pos=xyz(pos),size=xyz(size),rgba=xyz(color),group='0')
body=e(world,'body',name='base_link',pos='0 0 .001');e(body,'freejoint',name='root')
e(body,'inertial',pos='0 0 .047',mass='.55',diaginertia='.0009 .0013 .0017')
e(body,'geom',name='base_collision',type='box',pos='0 0 .0425',size='.075 .05 .0175',group='3',rgba='.2 .4 .6 .1')
e(body,'geom',name='electronics_collision',type='box',pos='-.005 0 .0875',size='.0675 .045 .0225',group='3',rgba='.2 .4 .6 .1')
act=e(mj,'actuator')
for name,info in MANIFEST['links'].items():
 b=body
 if name!='base_link':
  b=e(body,'body',name=name,pos=xyz([v*.001 for v in info['origin_mm']]),euler=xyz([0,0,math.radians(info['yaw_degrees'])]))
  wh='wheel' in name
  e(b,'inertial',pos='0 0 0',mass='.04' if wh else '.009',diaginertia='1.092e-5 1.8e-5 1.092e-5' if wh else '2e-6 2e-6 2e-6')
  if wh:
   j=name.replace('_link','_joint');e(b,'joint',name=j,type='hinge',axis='0 1 0')
   e(b,'geom',name=name+'_collision',type='cylinder',size='.03 .012',euler=xyz([math.pi/2,0,0]),group='3',condim='6',friction='.35 .002 .0001',rgba='.15 .15 .15 1')
   e(act,'velocity',name=name.replace('_link','_motor'),joint=j,kv='.06',ctrlrange='-25 25',forcerange='-.10 .10')
  else:
   e(b,'geom',name=name+'_collision',type='box',size='.008 .0225 .010' if 'ultrasonic' in name else '.006 .0135 .02',pos='-.003 0 0' if 'ultrasonic' in name else '-.009 0 .006',group='3',rgba='.3 .3 .3 .1')
 if 'ultrasonic' in name:
  # Frame uses +X boresight; custom cone raycaster in sim_core.py.
  e(b,'site',name=name.replace('_link','_site'),pos='.0126 0 0',size='.001',rgba='.1 .7 .9 1',group='4')
 if name=='camera_link':
  # MuJoCo camera looks along local -Z; X image-right points -Y and Y image-up +Z.
  e(b,'camera',name='esp32_cam',pos='0 0 0',xyaxes='0 -1 0 0 0 1',fovy='55',resolution='640 480')
 for idx,v in enumerate(info['visuals']):
  n=Path(v['mesh']).stem;e(b,'geom',name=n+'_visual',type='mesh',mesh=n,material=n+'_mat',contype='0',conaffinity='0',group='2',density='0')
# Include no inertial/laser/depth observation: only RGB + four simulated ultrasound cones.
write(mj,ROOT/'mjcf/room.xml')
print('Generated',ROOT/'urdf/vbmd.urdf','and',ROOT/'mjcf/room.xml')
