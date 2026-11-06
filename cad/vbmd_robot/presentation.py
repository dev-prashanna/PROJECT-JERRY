"""Save shaded and exploded CAD views, then restore the assembly."""
import FreeCAD as App, FreeCADGui as Gui
from pathlib import Path
ROOT=Path('/home/prashanna/Documents/PROJECT VBMD/cad/vbmd_robot')
D=App.ActiveDocument
assert D is not None and D.getObject('Wheel_Front_Left'), 'Activate the VBMD robot document first'; App.setActiveDocument(D.Name)
view=Gui.activeDocument().activeView()
original=[]
for o in D.Objects:
    if hasattr(o,'Shape') and not hasattr(o,'Group') and len(o.Shape.Solids)>0 and 'Shaded' in o.ViewObject.listDisplayModes():
        original.append((o,o.ViewObject.DisplayMode)); o.ViewObject.DisplayMode='Shaded'
        try:
            mat=o.ViewObject.ShapeMaterial
            mat.Shininess=45. if any(s in o.Name.lower() for s in ['aluminium','shield','can','usb','screw']) else 12.
            o.ViewObject.ShapeMaterial=mat
        except Exception: pass
view.viewIsometric(); view.fitAll(); Gui.updateGui()
view.saveImage(str(ROOT/'previews/shaded.png'),2400,1800,'White')
placements=[]
for name,z in [('Motors',15),('Power',30),('Electronics',100),('Sensors',48)]:
    g=D.getObject(name); placements.append((g,g.Placement.copy())); g.Placement.Base.z=z
D.Wiring.Visibility=False
D.recompute(); view.fitAll(); Gui.updateGui()
view.saveImage(str(ROOT/'previews/exploded.png'),1800,2400,'White')
for g,pl in placements: g.Placement=pl
D.Wiring.Visibility=True
for o,mode in original: o.ViewObject.DisplayMode=mode
D.recompute(); view.viewIsometric(); view.fitAll(); D.save()
print('Saved shaded and exploded CAD views; original assembly restored.')
