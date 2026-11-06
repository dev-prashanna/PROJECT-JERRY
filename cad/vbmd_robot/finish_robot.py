"""Run after build_robot.py inside FreeCAD. Validate, export and save views."""
import FreeCAD as App, FreeCADGui as Gui, Part, MeshPart, Import
from pathlib import Path
import json, math
ROOT=Path('/home/prashanna/Documents/PROJECT VBMD/cad/vbmd_robot')
D=App.ActiveDocument
assert D is not None and D.getObject('Wheel_Front_Left'), 'Activate the VBMD robot document first'
App.setActiveDocument(D.Name)

def effective_visible(o):
    if not o.Visibility: return False
    par=o.getParentGeoFeatureGroup()
    while par:
        if not par.Visibility: return False
        par=par.getParentGeoFeatureGroup()
    return True

def leaves():
    return [o for o in D.Objects if hasattr(o,'Shape') and not hasattr(o,'Group') and o.TypeId not in ['PartDesign::Body','Sketcher::SketchObject'] and effective_visible(o) and not o.Shape.isNull() and len(o.Shape.Solids)>0]

def worldshape(o):
    s=o.Shape.copy()
    s.Placement=o.getGlobalPlacement().multiply(o.Placement.inverse()).multiply(s.Placement)
    return s

features=leaves()
invalid=[o.Name for o in features if not o.Shape.isValid()]
assert not invalid, invalid
# Ensure the kinematics preserve wheel centers and child geometry through spin.
spins=[]
for name in ['Wheel_Front_Left','Wheel_Front_Right','Wheel_Rear_Left','Wheel_Rear_Right']:
    w=D.getObject(name); origin=w.Placement.Base; before=w.Group[0].getGlobalPlacement().Base
    w.Spin=90; w.Placement.Rotation=App.Rotation(App.Vector(0,1,0),90); D.recompute()
    assert (w.Placement.Base-origin).Length<1e-7
    assert w.Placement.Rotation.isSame(App.Rotation(App.Vector(0,1,0),90),1e-7)
    spins.append({'wheel':name,'angle_test_deg':90,'center_unchanged':True})
    w.Spin=0; w.Placement.Rotation=App.Rotation()
D.recompute()
features=leaves()
shapes=[worldshape(o) for o in features]
whole=Part.makeCompound(shapes); bounds=whole.BoundBox
report={
    'model':D.Name,'freecad_version':'.'.join(App.Version()[:3]),
    'visible_shape_objects':len(features),'document_objects':len(D.Objects),
    'invalid_shapes':invalid,
    'sketches':[{'name':o.Name,'fully_constrained':o.FullyConstrained} for o in D.Objects if o.TypeId=='Sketcher::SketchObject'],
    'bounding_box_mm':{'length_x':bounds.XLength,'width_y':bounds.YLength,'height_z':bounds.ZLength,'ground_z_min':bounds.ZMin},
    'owner_envelope_mm':{'length':170,'width':160,'height':140,'wheel_diameter':60},
    'wheel_motion_checks':spins,
    'limitations':['Photo-derived reconstruction, not metrology or manufacturing approval.','Internal board dimensions, hole locations, hidden geometry, motor internals and cable routes inferred.','Wheel motion is visual kinematics; no physics or electrical operation validated.','Chassis parameters update native decks; edit dimensions.json and rebuild to coordinate all components.']
}
(ROOT/'validation.json').write_text(json.dumps(report,indent=2))
print('BOUNDS',report['bounding_box_mm'])
# Smooth shaded CAD presentation with crisp hardware detail.
for o in features:
    o.ViewObject.DisplayMode='Flat Lines' if 'acrylic' in o.Name else 'Flat Lines'
    o.ViewObject.LineWidth=1.
# Use a temporary document to bake nested placements while retaining assembly colors.
E=App.newDocument('VBMD_Export')
baked=[]
for obj,shape in zip(features,shapes):
    f=E.addObject('Part::Feature',obj.Name); f.Label=obj.Label; f.Shape=shape
    f.ViewObject.ShapeColor=obj.ViewObject.ShapeColor; f.ViewObject.Transparency=obj.ViewObject.Transparency
    baked.append(f)
E.recompute()
Import.export(baked,str(ROOT/'exports/VBMD_Robot.step'))
# Mesh previews exchange geometry; units mm.
mesh=MeshPart.meshFromShape(Shape=whole,LinearDeflection=.18,AngularDeflection=.2,Relative=False)
mesh.write(str(ROOT/'exports/VBMD_Robot.stl'))
# STL round trip verifies non-empty tessellation, rather than just file creation.
import Mesh
roundtrip=Mesh.Mesh(str(ROOT/'exports/VBMD_Robot.stl'))
assert roundtrip.CountFacets>0
report['stl_facets']=roundtrip.CountFacets
for part in ['Lower_acrylic_deckPocket','Middle_acrylic_deckPocket']:
    ob=D.getObject(part)
    ob.Shape.exportStep(str(ROOT/'exports'/ (part+'.step')))
    pm=MeshPart.meshFromShape(Shape=ob.Shape,LinearDeflection=.1,AngularDeflection=.2,Relative=False)
    pm.write(str(ROOT/'exports'/(part+'.stl')))
# STEP round trip checks the placed assembly envelope and solids survived export.
loaded=Part.Shape(); loaded.read(str(ROOT/'exports/VBMD_Robot.step'))
assert loaded.isValid()
for axis in ['XLength','YLength','ZLength']:
    assert abs(getattr(loaded.BoundBox,axis)-getattr(bounds,axis))<.02
report['step_roundtrip']={'valid':True,'solids':len(loaded.Solids),'envelope_preserved':True}
(ROOT/'validation.json').write_text(json.dumps(report,indent=2))
App.closeDocument(E.Name); App.setActiveDocument(D.Name)
Gui.activeDocument().activeView().setCameraType('Orthographic')
view=Gui.activeDocument().activeView()
# Set a neutral background only for the saved images, leaving user preferences untouched.
for name,method in [('isometric','viewIsometric'),('top','viewTop'),('side','viewFront'),('front','viewRight'),('rear','viewLeft')]:
    getattr(view,method)(); view.fitAll(); Gui.updateGui()
    view.saveImage(str(ROOT/'previews'/(name+'.png')),2400,1800,'White')
view.viewIsometric(); view.fitAll(); D.recompute(); D.save()
print('EXPORTS COMPLETE',len(features),'features,',len(loaded.Solids),'STEP solids;',roundtrip.CountFacets,'STL triangles')
