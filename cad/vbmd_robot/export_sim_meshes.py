"""Export color-separated local-link visual meshes from the actual CAD assembly."""
import FreeCAD as App, Part, MeshPart, Mesh
from pathlib import Path
from collections import defaultdict
import json
ROOT=Path('/home/prashanna/Documents/PROJECT VBMD')
OUT=ROOT/'simulation/ros2_ws/src/vbmd_description'
D=App.ActiveDocument
assert D is not None and D.getObject('Wheel_Front_Left'), 'Activate the VBMD robot document first'
link_poses={
 'base_link':([0,0,0],0),
 'front_ultrasonic_link':([72,0,51],0),
 'rear_ultrasonic_link':([-72,0,51],180),
 'left_ultrasonic_link':([0,55,68],90),
 'right_ultrasonic_link':([0,-55,68],-90),
 'camera_link':([75.7,0,72],0),
}
for side,y in [('left',68),('right',-68)]:
 for end,x in [('front',55),('rear',-55)]: link_poses[f'{end}_{side}_wheel_link']=([x,y,30],0)
poses={k:App.Placement(App.Vector(*pos),App.Rotation(App.Vector(0,0,1),yaw)) for k,(pos,yaw) in link_poses.items()}

def link_for(o):
 p=o.getParentGeoFeatureGroup()
 while p:
  if p.Name.startswith('Wheel_'):
   words=p.Name.split('_'); return f'{words[1].lower()}_{words[2].lower()}_wheel_link'
  if p.Name.startswith('HC_SR04_'): return p.Name.split('_')[-1].lower()+'_ultrasonic_link'
  if p.Name=='ESP32_CAM': return 'camera_link'
  p=p.getParentGeoFeatureGroup()
 return 'base_link'

def visible(o):
 if not o.Visibility:return False
 p=o.getParentGeoFeatureGroup()
 while p:
  if not p.Visibility:return False
  p=p.getParentGeoFeatureGroup()
 return True
buckets=defaultdict(list)
for o in D.Objects:
 if not hasattr(o,'Shape') or hasattr(o,'Group') or o.TypeId=='Sketcher::SketchObject':continue
 if not len(o.Shape.Solids) or not visible(o):continue
 link=link_for(o)
 sh=o.Shape.copy()
 sh.Placement=poses[link].inverse().multiply(o.getGlobalPlacement()).multiply(o.Placement.inverse()).multiply(sh.Placement)
 rgb=tuple(round(c,4) for c in o.ViewObject.ShapeColor[:3]); alpha=round(1-o.ViewObject.Transparency/100,3)
 buckets[(link,rgb,alpha)].append(sh)
manifest={'units':'metres','source':'VBMD_Robot.FCStd; color-separated CAD geometry','links':{}}
scale=App.Matrix(); scale.A11=scale.A22=scale.A33=.001
for idx,((link,color,alpha),shapes) in enumerate(buckets.items()):
 mesh=MeshPart.meshFromShape(Shape=Part.makeCompound(shapes),LinearDeflection=.18,AngularDeflection=.22,Relative=False)
 mesh.transform(scale)
 name=f'{link}_{idx:02d}.stl'; mesh.write(str(OUT/'meshes'/name))
 manifest['links'].setdefault(link,{'origin_mm':link_poses[link][0],'yaw_degrees':link_poses[link][1],'visuals':[]})['visuals'].append({'mesh':name,'rgba':[*color,alpha],'triangles':mesh.CountFacets})
(OUT/'meshes/manifest.json').write_text(json.dumps(manifest,indent=2))
used={v['mesh'] for info in manifest['links'].values() for v in info['visuals']}
for old in (OUT/'meshes').glob('*.stl'):
    if old.name not in used: old.unlink()
print('EXPORTED',len(buckets),'color meshes;',len(manifest['links']),'physical links; all vertices in metres.')
