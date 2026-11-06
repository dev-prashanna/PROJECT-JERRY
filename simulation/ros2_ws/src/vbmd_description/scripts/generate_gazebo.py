#!/usr/bin/env python3
"""Generate Gazebo Harmonic SDF from the same CAD meshes and URDF inertias."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json,math,copy,yaml
P=Path(__file__).resolve().parents[1]
manifest=json.loads((P/'meshes/manifest.json').read_text())
urdf=ET.parse(P/'urdf/vbmd.urdf').getroot()
def el(p,tag,text=None,**a):
 x=ET.SubElement(p,tag,{k:str(v) for k,v in a.items()});x.text=None if text is None else str(text);return x
def vals(x):return ' '.join(str(v) for v in x)
def write(r,p):ET.indent(r,space='  ');ET.ElementTree(r).write(p,encoding='utf-8',xml_declaration=True)
def mat(parent,rgba):
 m=el(parent,'material');el(m,'ambient',vals(rgba));el(m,'diffuse',vals(rgba));el(m,'specular','0.3 0.3 0.3 1');el(m,'shininess',40)
 return m
sdf=ET.Element('sdf',version='1.9');model=el(sdf,'model',name='vbmd');el(model,'self_collide','false')
for name,info in manifest['links'].items():
 u=urdf.find(f"link[@name='{name}']");l=el(model,'link',name=name)
 el(l,'pose',vals([v*.001 for v in info['origin_mm']]+[0,0,math.radians(info['yaw_degrees'])]))
 i=el(l,'inertial');ui=u.find('inertial');com=ui.find('origin').get('xyz','0 0 0');el(i,'pose',com+' 0 0 0');el(i,'mass',ui.find('mass').get('value'))
 inertia=el(i,'inertia')
 for key,value in ui.find('inertia').attrib.items():el(inertia,key,value)
 for idx,v in enumerate(info['visuals']):
  vis=el(l,'visual',name=f'{name}_visual_{idx}');g=el(vis,'geometry');mesh=el(g,'mesh');el(mesh,'uri','model://vbmd_description/meshes/'+v['mesh'])
  rgba=v['rgba'];mat(vis,rgba);el(vis,'transparency',1-rgba[3]);el(vis,'cast_shadows','true')
 for idx,uc in enumerate(u.findall('collision')):
  c=el(l,'collision',name=name+'_collision_'+str(idx));o=uc.find('origin');el(c,'pose',o.get('xyz','0 0 0')+' '+o.get('rpy','0 0 0'))
  geom=el(c,'geometry');ug=uc.find('geometry')
  for primitive in ug:
   sg=el(geom,primitive.tag)
   for k,v in primitive.attrib.items():el(sg,k,v)
  surface=el(c,'surface');friction=el(surface,'friction');ode=el(friction,'ode');el(ode,'mu','.35');el(ode,'mu2','.35')
 if name!='base_link':
  wh='wheel' in name;j=el(model,'joint',name=name.replace('_link','_joint'),type='revolute' if wh else 'fixed');el(j,'parent','base_link');el(j,'child',name)
  if wh:
   ax=el(j,'axis');el(ax,'xyz','0 1 0');lim=el(ax,'limit');el(lim,'lower','-1e16');el(lim,'upper','1e16');el(lim,'effort','.1');el(lim,'velocity','30')
   dyn=el(ax,'dynamics');el(dyn,'damping','.00005');el(dyn,'friction','.00005')
 if 'ultrasonic' in name:
  side=name.split('_')[0];sen=el(l,'sensor',name=side+'_ultrasonic_proxy',type='gpu_lidar')
  el(sen,'pose','.0126 0 0 0 0 0');el(sen,'topic','/sim/sonar_rays/'+side);el(sen,'gz_frame_id',side+'_ultrasonic_frame');el(sen,'always_on','true');el(sen,'update_rate','20');el(sen,'visualize','false')
  ray=el(sen,'lidar');scan=el(ray,'scan');h=el(scan,'horizontal')
  for k,v in [('samples',9),('resolution',1),('min_angle',-math.pi/24),('max_angle',math.pi/24)]:el(h,k,v)
  v=el(scan,'vertical')
  for k,value in [('samples',1),('resolution',1),('min_angle',0),('max_angle',0)]:el(v,k,value)
  rr=el(ray,'range')
  for k,v in [('min',.02),('max',4.),('resolution',.001)]:el(rr,k,v)
  noise=el(ray,'noise');el(noise,'type','gaussian');el(noise,'mean',0);el(noise,'stddev','.004')
 if name=='camera_link':
  sen=el(l,'sensor',name='esp32_cam',type='camera');el(sen,'topic','/camera/image_raw');el(sen,'gz_frame_id','camera_optical_frame');el(sen,'always_on','true');el(sen,'update_rate',10)
  cam=el(sen,'camera');el(cam,'camera_info_topic','/camera/camera_info');el(cam,'horizontal_fov',2*math.atan(640/480*math.tan(math.radians(55)/2)))
  im=el(cam,'image');el(im,'width',640);el(im,'height',480);el(im,'format','R8G8B8')
  clip=el(cam,'clip');el(clip,'near','.003');el(clip,'far',8)
# Requested navigation sensors at the CAD chip centers. The MPU exposes gyro/accel only.
base=model.find("link[@name='base_link']")
for name,kind,pos,topic,frame,rate in [('mpu6050','imu','0 -.020 .04955 0 0 0','/sim/imu_raw','imu_link',100),('compass','magnetometer','-.028 .043 .1345 0 0 0','/magnetometer/data_raw','magnetometer_link',20)]:
 sen=el(base,'sensor',name=name,type=kind);el(sen,'pose',pos);el(sen,'topic',topic);el(sen,'gz_frame_id',frame);el(sen,'always_on','true');el(sen,'update_rate',rate)
 settings=el(sen,kind)
 if kind=='imu':
  for measurement,std in [('angular_velocity',.002),('linear_acceleration',.03)]:
   axis=el(settings,measurement)
   for direction in ['x','y','z']:
    ax=el(axis,direction);noise=el(ax,'noise',type='gaussian');el(noise,'mean',0);el(noise,'stddev',std)
 else:
  for direction in ['x','y','z']:
   ax=el(settings,direction);noise=el(ax,'noise',type='gaussian');el(noise,'mean',0);el(noise,'stddev',1e-7)
p=el(model,'plugin',filename='gz-sim-diff-drive-system',name='gz::sim::systems::DiffDrive')
for side in ['left','right']:
 for end in ['front','rear']:el(p,side+'_joint',end+'_'+side+'_wheel_joint')
for k,v in [('wheel_separation','.136'),('wheel_radius','.03'),('topic','/model/vbmd/cmd_vel'),('odom_topic','/sim/privileged/wheel_odometry'),('tf_topic','/sim/privileged/odom_tf'),('frame_id','odom'),('child_frame_id','base_footprint'),('odom_publish_frequency',20),('max_linear_velocity','.22'),('min_linear_velocity','-.22'),('max_angular_velocity','2'),('min_angular_velocity','-2'),('max_linear_acceleration','.4'),('max_angular_acceleration','2')]:el(p,k,v)
p=el(model,'plugin',filename='gz-sim-joint-state-publisher-system',name='gz::sim::systems::JointStatePublisher');el(p,'topic','/model/vbmd/joint_state')
for side in ['left','right']:
 for end in ['front','rear']:el(p,'joint_name',end+'_'+side+'_wheel_joint')
write(sdf,P/'models/vbmd.sdf')
# Self-contained robot included in an editable world: no Fuel downloads.
r=ET.Element('sdf',version='1.9');w=el(r,'world',name='vbmd_lab');el(w,'gravity','0 0 -9.81');el(w,'magnetic_field','0 2.15e-5 -4.27e-5')
geo=el(w,'spherical_coordinates');el(geo,'surface_model','EARTH_WGS84');el(geo,'world_frame_orientation','ENU');el(geo,'latitude_deg',0);el(geo,'longitude_deg',100);el(geo,'elevation',0);el(geo,'heading_deg',0)
physics=el(w,'physics',name='1ms',type='ignored');el(physics,'max_step_size','.001');el(physics,'real_time_factor','1')
for suffix,name in [('physics','Physics'),('user-commands','UserCommands'),('scene-broadcaster','SceneBroadcaster'),('sensors','Sensors'),('imu','Imu'),('magnetometer','Magnetometer')]:
 p=el(w,'plugin',filename='gz-sim-'+suffix+'-system',name='gz::sim::systems::'+name)
 if suffix=='sensors':el(p,'render_engine','ogre2')
 if suffix=='magnetometer':el(p,'use_units_gauss','false');el(p,'use_earth_frame_ned','false')
scene=el(w,'scene');el(scene,'ambient','.55 .55 .55 1');el(scene,'background','.78 .84 .9 1');el(scene,'shadows','true')
light=el(w,'light',name='sun',type='directional');el(light,'pose','0 0 4 0 0 0');el(light,'diffuse','.85 .85 .85 1');el(light,'specular','.4 .4 .4 1');el(light,'direction','-.3 -.2 -1');el(light,'cast_shadows','true')
# GUI with a close initial viewpoint, model tree, interaction controls and camera display.
gui=el(w,'gui',fullscreen='false')
for filename,name in [('MinimalScene','3D View'),('GzSceneManager','Scene Manager'),('InteractiveViewControl','View Control'),('EntityContextMenuPlugin','Context Menu'),('TransformControl','Move Models'),('EntityTree','Entity Tree'),('WorldControl','World Control'),('WorldStats','World Stats'),('ComponentInspector','Inspector'),('ImageDisplay','ESP32 Camera'),('KeyPublisher','Keyboard')]:
 pl=el(gui,'plugin',filename=filename,name=name)
 if filename in ['GzSceneManager','InteractiveViewControl','EntityContextMenuPlugin','KeyPublisher']:
  ui=el(pl,'gz-gui');el(ui,'property','floating',type='string',key='state');el(ui,'property','false',type='bool',key='showTitleBar');el(ui,'property',5,type='double',key='width');el(ui,'property',5,type='double',key='height')
 elif filename in ['TransformControl','WorldControl','WorldStats']:
  ui=el(pl,'gz-gui');el(ui,'property','floating',type='string',key='state');el(ui,'property','false',type='bool',key='showTitleBar');el(ui,'property',320 if filename=='TransformControl' else 290,type='double',key='width');el(ui,'property',72 if filename=='WorldControl' else 40,type='double',key='height');el(ui,'property',1,type='double',key='z')
  anchors=el(ui,'anchors',target='3D View');el(anchors,'line',own='left' if filename!='WorldStats' else 'right',target='left' if filename!='WorldStats' else 'right');el(anchors,'line',own='top' if filename=='TransformControl' else 'bottom',target='top' if filename=='TransformControl' else 'bottom')
 elif filename in ['EntityTree','ComponentInspector']:
  # Available from the plugin menu when needed; keep the first view uncluttered.
  gui.remove(pl);continue
 elif filename=='ImageDisplay':
  ui=el(pl,'gz-gui');el(ui,'property','docked',type='string',key='state');el(ui,'property',320,type='double',key='width');el(ui,'property',300,type='double',key='height')
 if filename=='MinimalScene':
  ui=el(pl,'gz-gui');el(ui,'title','3D View');el(ui,'property','false',type='bool',key='showTitleBar');el(ui,'property','docked',type='string',key='state')
  el(pl,'engine','ogre2');el(pl,'scene','scene');el(pl,'camera_pose','.40 -.40 .30 0 .40 2.356');el(pl,'background_color','.78 .84 .9 1')
 if filename=='WorldControl':el(pl,'play_pause','true');el(pl,'step','true');el(pl,'start_paused','false');el(pl,'use_event','true')
 if filename=='ImageDisplay':el(pl,'topic','/camera/image_raw')
# Floor with visual-only checker squares; one planar collision avoids seams.
ground=el(w,'model',name='ground');el(ground,'static','true');l=el(ground,'link',name='floor')
c=el(l,'collision',name='floor_collision');g=el(c,'geometry');plane=el(g,'plane');el(plane,'normal','0 0 1');el(plane,'size','4 3')
surf=el(c,'surface');fr=el(surf,'friction');ode=el(fr,'ode');el(ode,'mu','.35');el(ode,'mu2','.35')
for ix in range(20):
 for iy in range(15):
  vis=el(l,'visual',name=f'tile_{ix}_{iy}');el(vis,'pose',f'{-1.9+ix*.2} {-1.4+iy*.2} -.002 0 0 0');g=el(vis,'geometry');b=el(g,'box');el(b,'size','.2 .2 .004');a=.27 if (ix+iy)%2 else .43;mat(vis,[a,a+.02,a+.03,1])
def block(name,pos,size,color,static=True,mass=1):
 m=el(w,'model',name=name);el(m,'static',str(static).lower());el(m,'pose',vals([*pos,0,0,0]));l=el(m,'link',name='body')
 if not static:
  i=el(l,'inertial');el(i,'mass',mass);d=el(i,'inertia');a,b,c=size
  for key,v in [('ixx',mass*(b*b+c*c)/12),('iyy',mass*(a*a+c*c)/12),('izz',mass*(a*a+b*b)/12)]:el(d,key,v)
 for typ in ['collision','visual']:
  o=el(l,typ,name=name+'_'+typ);g=el(o,'geometry');b=el(g,'box');el(b,'size',vals(size))
  if typ=='visual':mat(o,color)
 return m
for name,pos,size,col in [('north',(0,1.5,.7),(4,.06,1.4),[.67,.7,.73,1]),('south',(0,-1.5,.7),(4,.06,1.4),[.72,.68,.61,1]),('east',(2,0,.7),(.06,3,1.4),[.67,.7,.73,1]),('west',(-2,0,.7),(.06,3,1.4),[.72,.68,.61,1])]:block(name,pos,size,col)
# Wall landmarks: asymmetry and varied heights provide persistent CV features.
for i in range(24):block('landmark_'+str(i),(1.965,-1.38+i*.12,.1+(i%5)*.075),(.012,.05+.01*(i%3),.036+.01*(i%4)),[.1+(i%3)*.28,.1+(i%4)*.19,.12+(i%2)*.5,1])
for i in range(18):
 for sy in [-1,1]:block(f'patch_{i}_{sy}',(-1.75+i*.2,sy*1.465,.08+(i%4)*.1),(.05,.012,.05),[.12+(i%3)*.24,.15+(i%4)*.17,.18+(i%2)*.48,1])
for i,(pos,size,col) in enumerate([((.85,.08,.13),(.24,.3,.26),[.6,.15,.08,1]),((-.65,.5,.1),(.2,.2,.2),[.08,.35,.6,1]),((.2,-.65,.18),(.16,.22,.36),[.65,.45,.06,1]),((1.15,.75,.08),(.32,.24,.16),[.12,.45,.2,1])]):block('obstacle_'+str(i),pos,size,col)
block('movable_box',(-.45,-.55,.06),(.12,.12,.12),[.6,.17,.45,1],False,.06)
# Planned scene zones: open center, furniture left/rear, varied shapes right/front.
# Main corridors are kept open; the corner low obstacle is an explicit difficult case.
def cylinder(name,x,y,radius,height,color):
 m=el(w,'model',name=name);el(m,'static','true');el(m,'pose',vals([x,y,height/2,0,0,0]));l=el(m,'link',name='body')
 for kind in ['collision','visual']:
  o=el(l,kind,name=name+'_'+kind);g=el(o,'geometry');c=el(g,'cylinder');el(c,'radius',radius);el(c,'length',height)
  if kind=='visual':mat(o,color)
 return m
cylinder('barrel',-1.40,-.70,.13,.35,[.12,.34,.65,1])
cylinder('thin_post',1.55,-.95,.035,.48,[.75,.55,.12,1])
block('low_corner_obstacle',(-1.65,1.15,.02),(.28,.18,.04),[.55,.22,.12,1])
block('short_divider',(-.25,.95,.18),(.06,.30,.36),[.28,.46,.42,1])
block('movable_crate',(1.55,.35,.08),(.16,.16,.16),[.55,.18,.38,1],False,.10)
block('table_top',(-1.25,.85,.40),(.60,.40,.04),[.47,.28,.14,1])
for i,(dx,dy) in enumerate([(-.25,-.15),(-.25,.15),(.25,-.15),(.25,.15)]):
 cylinder('table_leg_'+str(i),-1.25+dx,.85+dy,.025,.38,[.22,.23,.25,1])
# Small contrasting face patches provide texture on otherwise uniform obstacle faces.
for obj in w.findall('model'):
 if not obj.get('name').startswith(('obstacle_','movable_','short_divider')):continue
 link=obj.find('link');geometry=link.find('collision/geometry/box/size')
 if geometry is None:continue
 sx,sy,sz=map(float,geometry.text.split())
 for j in range(4):
  patch=el(link,'visual',name='feature_patch_'+str(j));el(patch,'pose',vals([-sx/2-.001,(-.22 if j%2 else .22)*sy,(-.22 if j<2 else .22)*sz,0,0,0]));g=el(patch,'geometry');box=el(g,'box');el(box,'size',vals([.001,min(.04,sy*.16),min(.04,sz*.16)]));mat(patch,[.9,.9,.85,1] if j%2 else [.05,.07,.10,1])
# Inline model makes the world self-contained except package meshes.
robot=copy.deepcopy(model);el(robot,'pose','0 0 .003 0 0 0');w.append(robot)
write(r,P/'worlds/vbmd_lab.sdf')
from collections import deque
import random
import numpy as np

def layout_metrics(world):
 grid=np.zeros((109,149),bool);ys=np.linspace(-1.35,1.35,109);xs=np.linspace(-1.85,1.85,149);xx,yy=np.meshgrid(xs,ys)
 objects=[]
 for obj in world.findall('model'):
  name=obj.get('name')
  if not name.startswith(('obstacle_','movable_','barrel','thin_post','low_corner','short_divider','table_leg')):continue
  pose=list(map(float,obj.findtext('pose').split()));x,y,z=pose[:3];yaw=pose[5]
  geom=obj.find('link/collision/geometry');box=geom.find('box/size');cyl=geom.find('cylinder')
  if box is not None:
   sx,sy,sz=map(float,box.text.split());hx=(abs(math.cos(yaw))*sx+abs(math.sin(yaw))*sy)/2;hy=(abs(math.sin(yaw))*sx+abs(math.cos(yaw))*sy)/2
  elif cyl is not None:hx=hy=float(cyl.findtext('radius'));sz=float(cyl.findtext('length'))
  else:continue
  objects.append({'name':name,'center_m':[x,y,z],'footprint_half_extents_m':[hx,hy],'height_m':sz})
  # Inflate by the robot's ~117 mm corner radius + 25 mm extra margin.
  grid|=(abs(xx-x)<hx+.142)&(abs(yy-y)<hy+.142)
 for i,a in enumerate(objects):
  for b in objects[i+1:]:
   if all(abs(a['center_m'][k]-b['center_m'][k])<a['footprint_half_extents_m'][k]+b['footprint_half_extents_m'][k]+.02 for k in [0,1]):return None
 start=(54,74)
 if grid[start]:return None
 seen={start};queue=deque([start])
 while queue:
  a,b=queue.popleft()
  for u,v in [(a+1,b),(a-1,b),(a,b+1),(a,b-1)]:
   if 0<=u<109 and 0<=v<149 and not grid[u,v] and (u,v) not in seen:seen.add((u,v));queue.append((u,v))
 return {'obstacles':objects,'free_cells':int((~grid).sum()),'reachable_free_fraction':len(seen)/int((~grid).sum()),'footprint_inflation_m':.142,'grid_resolution_m':.025}
layouts={}
for split,seed,jitter in [('practice',7,0.),('validation',19,.08),('test',43,.14)]:
 for attempt in range(50):
  rng=random.Random(seed+attempt);variant=copy.deepcopy(r);vw=variant.find('world');table_shift=[rng.uniform(-jitter,jitter),rng.uniform(-jitter,jitter)]
  for obj in vw.findall('model'):
   name=obj.get('name');pose=obj.find('pose')
   if pose is None or not name.startswith(('obstacle_','movable_','barrel','thin_post','low_corner','short_divider','table_')):continue
   values=list(map(float,pose.text.split()));dx,dy=table_shift if name.startswith('table_') else [rng.uniform(-jitter,jitter),rng.uniform(-jitter,jitter)]
   values[0]+=dx;values[1]+=dy
   if not name.startswith('table_'):values[5]+=rng.uniform(-jitter*2,jitter*2)
   pose.text=vals(values)
   for visual in obj.findall('.//visual'):
    for tag in ['ambient','diffuse']:
     col=visual.find('material/'+tag)
     if col is not None:
      rgba=list(map(float,col.text.split()));col.text=vals([min(1,max(.03,v*rng.uniform(.82,1.15))) for v in rgba[:3]]+[rgba[3]])
  ambient={'practice':'.55 .55 .55 1','validation':'.43 .46 .50 1','test':'.65 .56 .47 1'}[split];vw.find('scene/ambient').text=ambient
  metrics=layout_metrics(vw)
  if metrics and metrics['reachable_free_fraction']>.88:break
 else:raise RuntimeError('Could not generate connected layout: '+split)
 layouts[split]={'seed':seed+attempt,**metrics}
 write(variant,P/'worlds'/('vbmd_'+split+'.sdf'))
 if split=='practice':write(variant,P/'worlds/vbmd_lab.sdf')
(P/'worlds/layouts.json').write_text(json.dumps(layouts,indent=2))

# ROS bridge carries the ray implementation only on explicitly internal simulation topics.
bridge=[]
def br(ros,gz,rt,gt,direction='GZ_TO_ROS'):
 bridge.append({'ros_topic_name':ros,'gz_topic_name':gz,'ros_type_name':rt,'gz_type_name':gt,'direction':direction})
br('/sim/imu_raw','/sim/imu_raw','sensor_msgs/msg/Imu','gz.msgs.IMU')
br('/magnetometer/data_raw','/magnetometer/data_raw','sensor_msgs/msg/MagneticField','gz.msgs.Magnetometer')
br('/clock','/clock','rosgraph_msgs/msg/Clock','gz.msgs.Clock')
br('/camera/image_raw','/camera/image_raw','sensor_msgs/msg/Image','gz.msgs.Image')
br('/camera/camera_info','/camera/camera_info','sensor_msgs/msg/CameraInfo','gz.msgs.CameraInfo')
br('/sim/keyboard','/keyboard/keypress','std_msgs/msg/Int32','gz.msgs.Int32')
br('/joint_states','/model/vbmd/joint_state','sensor_msgs/msg/JointState','gz.msgs.Model')
br('/sim/cmd_vel_safe','/model/vbmd/cmd_vel','geometry_msgs/msg/Twist','gz.msgs.Twist','ROS_TO_GZ')
for side in ['front','left','rear','right']:br('/sim/sonar_rays/'+side,'/sim/sonar_rays/'+side,'sensor_msgs/msg/LaserScan','gz.msgs.LaserScan')
(P/'config/gazebo_bridge.yaml').write_text(yaml.safe_dump(bridge,sort_keys=False))
print('Generated Gazebo robot, interactive world, and ROS bridge configuration.')
