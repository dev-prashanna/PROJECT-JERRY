"""Run in FreeCAD Python console: exec(open('/absolute/path/build_robot.py').read()).
Photo reconstruction, dimensions in mm. Build uses a unique document and never closes others.
"""
import FreeCAD as App, FreeCADGui as Gui, Part, Sketcher
import math, json, csv, os
from pathlib import Path
ROOT = Path('/home/prashanna/Documents/PROJECT VBMD/cad/vbmd_robot')
C = json.loads((ROOT/'dimensions.json').read_text())
V = App.Vector
D = App.newDocument('VBMD_Robot')
D.Label = 'VBMD | photo reconstruction | 170 × 160 × 140 mm'
D.addObject('App::DocumentObjectGroup','Documentation').Label='00 · Dimensions and model notes'
S = D.addObject('Spreadsheet::Sheet','Dimensions'); D.Documentation.addObject(S)
S.set('A1','VBMD / OWNER MEASUREMENTS + PHOTO ESTIMATES')
S.set('A2','Dimension'); S.set('B2','mm'); S.set('C2','Source')
for row,(key,val) in enumerate([(k,v) for k,v in C.items() if isinstance(v,(int,float))],3):
    S.set('A'+str(row),key); S.set('B'+str(row),str(val)); S.setAlias('B'+str(row),key)
    S.set('C'+str(row),'Owner, approximate' if key in ['overall_length','overall_width','overall_height','wheel_diameter'] else 'Inferred; verify')
S.setColumnWidth('A',210); S.setColumnWidth('B',85); S.setColumnWidth('C',190)
S.setStyle('A1:C2','bold','add'); S.setBackground('A1:C2',(0.08,0.17,0.23)); S.setForeground('A1:C2',(1.,1.,1.))
notes = D.addObject('App::FeaturePython','ModelNotes'); D.Documentation.addObject(notes)
notes.addProperty('App::PropertyString','Accuracy','Provenance'); notes.Accuracy=C['measurement_note']
notes.addProperty('App::PropertyString','Axes','Provenance'); notes.Axes='+X front / camera; +Y left; +Z up. Ground z=0. Units mm.'
notes.addProperty('App::PropertyString','Scope','Provenance'); notes.Scope='Mechanical layout and wheel kinematics; electronics are visual envelopes. Not a certified manufacturing assembly or powered hardware simulation.'
notes.addProperty('App::PropertyString','Regeneration','Provenance'); notes.Regeneration='Edit dimensions.json and rerun build_robot.py for a coordinated rebuild. Native plate sketches and wheel placements also use Dimensions expressions.'
robot = D.addObject('App::Part','Robot'); robot.Label='VBMD · four-wheel differential-drive robot'
def group(name,label,parent=robot):
    o=D.addObject('App::Part',name); o.Label=label; parent.addObject(o); return o
chassis=group('Chassis','01 · Acrylic chassis and fasteners')
motors=group('Motors','02 · Four yellow geared motors')
power=group('Power','03 · Central two-cell battery holder')
electronics=group('Electronics','04 · Controller, breadboard and motor driver')
sensors=group('Sensors','05 · Front/rear ranging and camera')
wiring=group('Wiring','06 · Jumper wires and rainbow ribbon')
BLACK=(0.025,0.032,0.041); RUBBER=(0.065,0.073,0.084); YELLOW=(1.,0.69,0.035)
SILVER=(0.70,0.75,0.79); GOLD=(0.84,0.62,0.19); PCB=(0.015,0.22,0.31)
BLUE=(0.025,0.27,0.51); RED=(0.64,0.035,0.065); WHITE=(0.88,0.89,0.85)
BOM=[]
def feature(name,shape,color,parent,desc='',trans=0):
    o=D.addObject('PartDesign::Feature' if parent.TypeId=='PartDesign::Body' else 'Part::Feature',name)
    parent.addObject(o); o.Shape=shape; o.Label=name.replace('_',' ')
    o.ViewObject.ShapeColor=color; o.ViewObject.LineColor=(0.09,0.11,0.13)
    o.ViewObject.DisplayMode='Flat Lines' if parent==chassis else 'Flat Lines'
    o.ViewObject.LineWidth=1.; o.ViewObject.AngularDeflection=8.; o.ViewObject.Deviation=0.10
    o.ViewObject.Transparency=trans
    o.addProperty('App::PropertyString','Description','Component'); o.Description=desc or 'Photo-derived visual geometry; dimensions approximate'
    return o

def box(a,b,c,x=0,y=0,z=0): return Part.makeBox(a,b,c,V(x,y,z))
def cyl(r,h,x=0,y=0,z=0,axis=V(0,0,1)): return Part.makeCylinder(r,h,V(x,y,z),axis)
def compound(shapes): return Part.makeCompound(shapes)
def rounded(a,b,h,x,y,z,r=2):
    sh=box(a,b,h,x,y,z)
    es=[e for e in sh.Edges if abs(e.Vertexes[-1].Point.z-e.Vertexes[0].Point.z)>h*.9]
    return sh.makeFillet(r,es)
def ring(ro,ri,h,x,y,z,axis=V(0,0,1)): return cyl(ro,h,x,y,z,axis).cut(cyl(ri,h,x,y,z,axis))
def screw(x,y,z,parent=chassis):
    sh=cyl(1.45,5,x,y,z-5).fuse(cyl(2.7,1.8,x,y,z))
    sh=sh.cut(box(3.8,.65,1.2,x-1.9,y-.325,z+1.1))
    return feature('M3_screw',sh,SILVER,parent,'M3 mounting fastener, estimated location')
def wire(name,pts,color,r=.48):
    # Circular sections swept along an interpolated smooth cable route.
    curve=Part.BSplineCurve(); curve.interpolate([V(*p) for p in pts])
    path=Part.Wire([curve.toShape()]); start=curve.value(curve.FirstParameter)
    tangent=curve.tangent(curve.FirstParameter)[0]
    section=Part.Wire([Part.makeCircle(r,start,tangent)])
    return feature(name,path.makePipeShell([section],True,False),color,wiring,'Visual cable route; not an electrical netlist')

# Native fully constrained rectangular sketches: pads and pocket remain editable.
def plate(name,L,W,z,parent,holes=True):
    b=D.addObject('PartDesign::Body',name); parent.addObject(b)
    sk=D.addObject('Sketcher::SketchObject',name+'Outline'); b.addObject(sk)
    pts=[V(-L/2,-W/2,0),V(L/2,-W/2,0),V(L/2,W/2,0),V(-L/2,W/2,0)]
    for i in range(4): sk.addGeometry(Part.LineSegment(pts[i],pts[(i+1)%4]),False)
    for i in range(4): sk.addConstraint(Sketcher.Constraint('Coincident',i,2,(i+1)%4,1))
    for i in [0,2]: sk.addConstraint(Sketcher.Constraint('Horizontal',i))
    for i in [1,3]: sk.addConstraint(Sketcher.Constraint('Vertical',i))
    c=sk.addConstraint(Sketcher.Constraint('Distance',0,L)); sk.setExpression('Constraints['+str(c)+']','Dimensions.chassis_length')
    c=sk.addConstraint(Sketcher.Constraint('Distance',1,W)); sk.setExpression('Constraints['+str(c)+']','Dimensions.chassis_width')
    c=sk.addConstraint(Sketcher.Constraint('DistanceX',0,1,-L/2)); sk.setExpression('Constraints['+str(c)+']','-Dimensions.chassis_length / 2')
    c=sk.addConstraint(Sketcher.Constraint('DistanceY',0,1,-W/2)); sk.setExpression('Constraints['+str(c)+']','-Dimensions.chassis_width / 2')
    sk.Placement.Base.z=z
    sk.setExpression('Placement.Base.z','Dimensions.'+('lower_deck_z' if z==C['lower_deck_z'] else 'middle_deck_z'))
    pad=b.newObject('PartDesign::Pad',name+'Pad'); pad.Profile=sk; pad.Length=C['plate_thickness']; pad.setExpression('Length','Dimensions.plate_thickness')
    D.recompute(); sk.Visibility=False
    hs=b.newObject('Sketcher::SketchObject',name+'MountingPattern'); hs.Placement.Base.z=z+C['plate_thickness']
    hs.setExpression('Placement.Base.z','Dimensions.'+('lower_deck_z' if z==C['lower_deck_z'] else 'middle_deck_z')+' + Dimensions.plate_thickness')
    for x in [-67,-32,32,67]:
        for y in [-43,43]:
            i=hs.addGeometry(Part.Circle(V(x,y,0),V(0,0,1),1.65),False)
            hs.addConstraint(Sketcher.Constraint('Block',i))
    # Wiring clearance slot, represented as a rectangular pocket.
    pts=[V(-14,-5,0),V(14,-5,0),V(14,5,0),V(-14,5,0)]
    for i in range(4):
        j=hs.addGeometry(Part.LineSegment(pts[i],pts[(i+1)%4]),False); hs.addConstraint(Sketcher.Constraint('Block',j))
    if z==C['middle_deck_z']:
        # Corner apertures around the four vertically mounted motor cans.
        for x in [-55,55]:
            for y in [-46,46]:
                j=hs.addGeometry(Part.Circle(V(x,y,0),V(0,0,1),9),False); hs.addConstraint(Sketcher.Constraint('Block',j))
    cut=b.newObject('PartDesign::Pocket',name+'Pocket'); cut.Profile=hs; cut.Length=C['plate_thickness']; cut.setExpression('Length','Dimensions.plate_thickness')
    D.recompute(); hs.Visibility=False; pad.Visibility=False
    cut.ViewObject.ShapeColor=(.67,.83,.86); cut.ViewObject.Transparency=52; cut.ViewObject.LineColor=(.25,.47,.51)
    b.Label=name.replace('_',' ')+' · editable pad / mounting pockets'
    return b,cut
lower, lower_tip=plate('Lower_acrylic_deck',C['chassis_length'],C['chassis_width'],C['lower_deck_z'],chassis)
middle,middle_tip=plate('Middle_acrylic_deck',C['chassis_length'],C['chassis_width'],C['middle_deck_z'],chassis)
for x in [-32,32]:
    for y in [-43,43]:
        feature('Brass_deck_spacer',ring(2.8,1.55,31.5,x,y,22.5),GOLD,chassis)
        screw(x,y,57)
# UNO supports, positioned away from the motor cans.
for x in [0,48]:
    for y in [-3,39]:
        feature('UNO_aluminium_standoff',ring(2.4,1.25,37.5,x,y,56.5),SILVER,chassis)
# Rear breadboard support mimics the hand-assembled raised white carrier.
feature('Rear_carrier',rounded(49,89,3,-73,-44.5,82.5,2),WHITE,chassis,'Cleaned-up carrier replacing uneven foam/glue; inferred')
for x in [-66,-30]:
    for y in [-34,34]:
        feature('Rear_carrier_post',ring(2.6,1.4,26,x,y,56.5),WHITE,chassis)
        screw(x,y,85.5)
BOM += [('Acrylic deck','2','156 × 110 × 2.5','Inferred; native sketches'),('Deck spacers','4','31.5 high','Inferred'),('UNO standoffs','4','37.5 high','Inferred'),('Rear board carrier','1','49 × 89 × 3','Inferred replacement for handmade support')]

# Motor bodies: the photographs show the cans above the yellow gearcases.
for x in [-C['wheelbase']/2,C['wheelbase']/2]:
    for sign in [-1,1]:
        y=sign*46
        m=group('Motor_'+('Front' if x>0 else 'Rear')+('_Left' if sign>0 else '_Right'),'TT motor · '+str(x)+' / '+str(sign),motors)
        sh=rounded(19,17,31,x-9.5,y-8.5,23,2)
        feature('Yellow_gearbox',sh,YELLOW,m)
        feature('Gearbox_seam',box(19.2,17.2,.7,x-9.6,y-8.6,31),(.75,.46,.03),m)
        feature('DC_motor_can',cyl(7.7,16,x,y,57),SILVER,m)
        feature('Motor_end_cap',cyl(7.8,3,x,y,73),BLACK,m)
        feature('Motor_terminal_tabs',compound([box(2,2.3,2,x-5,y-1,76),box(2,2.3,2,x+3,y-1,76)]),GOLD,m)
        feature('Axle',cyl(2.5,13,x,sign*54,30,V(0,sign,0)),WHITE,m)
        feature('Motor_mount_strap',compound([box(2,20,19,x-11,y-10,25),box(2,20,19,x+9,y-10,25)]),WHITE,m)
BOM += [('TT DC geared motor','4','19 × 17 × 53 visual envelope','Photo-derived; shaft not tolerance modeled')]

# Four independently rotating wheel assemblies, axis +Y for both sides.
R=C['wheel_diameter']/2; WW=C['wheel_width']; TY=(C['overall_width']-WW)/2
wheels=[]
for x,front in [(-C['wheelbase']/2,'Rear'),(C['wheelbase']/2,'Front')]:
    for sy,side in [(1,'Left'),(-1,'Right')]:
        w=group('Wheel_'+front+'_'+side,'Wheel · '+front+' '+side)
        w.addProperty('App::PropertyAngle','Spin','Kinematics'); w.Spin=0
        w.addProperty('App::PropertyString','JointType','Kinematics'); w.JointType='Continuous revolute; axis +Y; local center at origin'
        w.Placement=App.Placement(V(x,sy*TY,R),App.Rotation())
        w.setExpression('Placement.Base.x',('' if x>0 else '-')+'Dimensions.wheelbase / 2')
        w.setExpression('Placement.Base.y',('' if sy>0 else '-')+'(Dimensions.overall_width - Dimensions.wheel_width) / 2')
        w.setExpression('Placement.Base.z','Dimensions.wheel_diameter / 2')
        # Rotation via quaternion components is driven by explicit motion controller for persistence.
        tire=ring(R-.7,R-8,WW,0,-WW/2,0,V(0,1,0))
        feature('Rubber_tyre',tire,RUBBER,w,'60 mm nominal tyre, tread reaches radius 30 mm')
        blocks=[]
        for a in range(0,360,10):
            for j in [-1,0,1]:
                b=box(4,6.7,1.4,-2,j*7.5-3.35,math.sqrt(R*R-4)-1.4)
                b.rotate(V(0,0,0),V(0,1,0),a+(4 if j%2 else 0)); blocks.append(b)
        feature('Tyre_tread_blocks',compound(blocks),BLACK,w)
        # Open five-spoke rim with real windows, wheel geometry is one valid connected solid.
        rim=ring(R-7,R-10,WW-2,0,-WW/2+1,0,V(0,1,0))
        hub=cyl(8.0,WW-3,0,-WW/2+1.5,0,V(0,1,0))
        spokes=[]
        for a in range(0,360,72):
            b=box(6,4,R-9,-3,sy*(WW/2-3)-2,0)
            b.rotate(V(0,0,0),V(0,1,0),a); spokes.append(b)
        rim=rim.multiFuse([hub]+spokes).removeSplitter()
        rim=rim.cut(cyl(2.55,WW+2,0,-WW/2-1,0,V(0,1,0)))
        feature('Yellow_five_spoke_rim',rim,YELLOW,w)
        # Sidewall rings and raised concentric yellow lip, on both visible faces.
        for s in [-1,1]:
            yy=s*(WW/2-.7)
            feature('Sidewall_bead',ring(R-2.4,R-3.1,.6,0,yy,0,V(0,s,0)),BLACK,w)
            feature('Yellow_rim_lip',ring(R-7.1,R-8.2,.8,0,s*(WW/2-1),0,V(0,s,0)),YELLOW,w)
            feature('Hub_cap',cyl(6.4,1,0,s*(WW/2-1.2),0,V(0,s,0)),YELLOW,w)
        wheels.append(w)
BOM += [('Rubber wheel / yellow five-spoke rim','4','Ø60 × 24','Diameter from owner; width and tread inferred')]

# Construction fragment shared by build_robot.py and the live CAD update.
# Requires D, power, feature, rounded, box, cyl, compound and palette from build_robot.py.
# Coordinates mm. Two 18650 envelopes are a provisional assumption, not a measured cell type.
tray=rounded(77,44,2,-38.5,-22,22.5,2)
walls=compound([box(77,2,17,-38.5,-22,24.5),box(77,2,17,-38.5,20,24.5),box(2,40,17,-38.5,-20,24.5),box(2,40,17,36.5,-20,24.5)])
feature('Two_cell_battery_holder',tray.fuse(walls).removeSplitter(),BLACK,power,'Centered lower-deck holder, provisional 2 × 18650 fit; external dimensions 77 × 44 × 19 mm')
for i,y in enumerate([-10,10],1):
    g=group('Li_ion_cell_'+str(i),'Cell '+str(i)+' · 3.7 V Li-ion · provisional 18650',power)
    cell=feature('Cell_insulated_jacket',cyl(9.15,65.2,-32.6,y,33.8,V(1,0,0)),(.025,.28,.36),g,'3.7 V nominal per cell; diameter 18.3 × length 65.2 mm assumed')
    feature('Cell_negative_terminal',cyl(8.4,.35,-32.95,y,33.8,V(1,0,0)),SILVER,g)
    feature('Cell_positive_insulator',cyl(7.7,.35,32.6,y,33.8,V(1,0,0)),WHITE,g)
    feature('Cell_positive_button',cyl(3.8,.8,32.95,y,33.8,V(1,0,0)),SILVER,g)
    # Concentric compression-spring rings are a visual contact envelope.
    springs=[]
    for j in range(5):
        springs.append(ring(3.5-j*.25,3.1-j*.25,.3,-36+j*.55,y,33.8,V(1,0,0)))
    feature('Holder_spring_contact',compound(springs),SILVER,g)
    feature('Holder_positive_contact',box(.5,7,9,35.5,y-3.5,29.3),SILVER,g)
BOM += [('Two-cell holder','1','77 × 44 × 19','Owner reports central lower holder; 18650 form factor assumed'),('Li-ion battery cell','2','Ø18.3 × 65.2; 3.7 V each','Voltage/count from owner; cell size assumed; series/parallel unspecified')]


# Boards are visual assembly models with holes, individual headers, pins, ICs and connectors.
def board(name,L,W,x,y,z,color=PCB,parent=electronics):
    g=group(name,name.replace('_',' '),parent)
    pcb=box(L,W,1.6,x,y,z)
    for xx in [x+3,x+L-3]:
        for yy in [y+3,y+W-3]: pcb=pcb.cut(cyl(1.25,2,xx,yy,z-.2))
    feature(name+'_PCB',pcb,color,g)
    pads=[]
    for xx in [x+3,x+L-3]:
        for yy in [y+3,y+W-3]: pads.append(ring(2,1.3,.15,xx,yy,z+1.6))
    feature('Plated_mounting_rings',compound(pads),GOLD,g)
    return g

def header(g,n,x,y,z,axis='x'):
    L=n*2.54
    sh=box(L,2.5,7,x,y,z) if axis=='x' else box(2.5,L,7,x,y,z)
    holes=[]; pins=[]
    for i in range(n):
        xx=x+1.27+(i*2.54 if axis=='x' else 0); yy=y+1.25+(i*2.54 if axis!='x' else 0)
        holes.append(box(.75,.75,2,xx-.375,yy-.375,z+5.2)); pins.append(box(.5,.5,3,xx-.25,yy-.25,z-3))
    feature('Female_pin_header',sh.cut(compound(holes)),BLACK,g)
    feature('Header_solder_pins',compound(pins),SILVER,g)

uno=board('Arduino_UNO',68.6,53.4,-10,-7,94)
for x,n in [(-7,8),(16,10)]: header(uno,n,x,-6,95.6)
for x,n in [(-7,6),(12,8)]: header(uno,n,x,42.4,95.6)
feature('ATmega_DIP',box(35,10,3.7,5,18,95.6),BLACK,uno)
feature('DIP_legs',compound([box(.7,2,1.3,6+i*2.54,y,95.1) for i in range(13) for y in [16.4,28]]),SILVER,uno)
feature('USB_B_shell',box(15,12,10.5,44,-2,95.6).cut(box(10,9,7,50,-.5,97.1)),SILVER,uno)
feature('USB_B_tongue',box(5,6,3,51,1,99),WHITE,uno)
feature('DC_barrel_jack',box(11,9,10.5,47,30,95.6).cut(cyl(3.1,8,59,34.5,101,V(-1,0,0))),BLACK,uno)
for x in [36,43]: feature('UNO_capacitor',cyl(2.6,6,x,23,95.6),SILVER,uno)
feature('Crystal_oscillator',rounded(9,3,2.5,18,8,95.6,1),SILVER,uno)
feature('UNO_USB_interface_IC',box(6,6,1.5,35,7,95.6),BLACK,uno)
feature('UNO_green_LED',box(2,1.2,1,29,34,95.6),(.15,.85,.20),uno)
feature('UNO_reset_switch',box(6,6,2.8,-7,31,95.6),SILVER,uno)
# Silkscreen made as actual line geometry, hidden from mechanical exports.
def silk(text,x,y,z,parent,size=2.6):
    try:
        import Draft
        font='/run/host/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        if not os.path.exists(font): font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        sh=Draft.make_shapestring(String=text,FontFile=font,Size=size,Tracking=0)
        parent.addObject(sh); sh.Placement.Base=V(x,y,z); sh.ViewObject.ShapeColor=WHITE
        sh.Label='Silkscreen · '+text
    except Exception: pass
silk('UNO',-5,7,95.65,uno,3.5); silk('VBMD',20,33,95.65,uno,2)

for idx,y in enumerate([-10,10],1): silk('3.7 V',-12,y-3,43.05,D.getObject('Li_ion_cell_'+str(idx)),2.4)

# Rear white solderless breadboard (holes modeled, not a texture).
bb=group('Breadboard','Solderless breadboard · raised rear carrier',electronics)
bbbase=rounded(45,84,7,-72,-42,86,1.2)
holes=[]
for row in range(30):
    for col in range(10):
        xx=-68+col*3.6+(2 if col>=5 else 0); yy=-37+row*2.54
        holes.append(box(.9,.9,1.5,xx,yy,91.8))
bbbase=bbbase.cut(compound(holes)).cut(box(1.8,80,2,-50.3,-40,91.2))
feature('Breadboard_perforated_body',bbbase,WHITE,bb)
for x,col in [(-70,RED),(-29,BLUE)]: feature('Breadboard_power_rail',box(.45,77,.15,x,-38.5,93),col,bb)
# ESP32 DevKit near rear edge, oriented across the chassis as in the photographs.
esp=board('ESP32_DevKit',26,49,-74,-32,94,BLACK)
feature('ESP32_RF_shield',box(19,17,2.6,-70,-20,95.6),SILVER,esp)
feature('ESP32_main_IC',box(7,7,1.3,-65,-5,95.6),BLACK,esp)
feature('ESP32_micro_USB',box(7,5,3,-65,-35,95.6).cut(box(5,4,1.5,-64,-35.1,96.2)),SILVER,esp)
for xx in [-73,-50]: header(esp,15,xx,-28,95.6,'y')
feature('ESP32_antenna_trace',compound([box(1,2,0.15,-69+i*2.5,11+(i%2)*2,95.7) for i in range(7)]),GOLD,esp)
# L298N driver centered on middle deck with recognizable finned heatsink.
ld=board('L298N_motor_driver',43,43,-17,-21.5,60,RED)
feature('Driver_IC',box(20,5,10,-6,-5,61.6),BLACK,ld)
feature('Heatsink_base',box(22,16,2,-7,-8,69),BLACK,ld)
feature('Heatsink_fins',compound([box(1.3,16,16,-7+i*3,-8,71) for i in range(8)]),(.12,.14,.17),ld)
for x,y,n in [(-15,-19,3),(16,-17,2),(16,10,2)]:
    term=box(6,n*5,8,x,y,61.6)
    feature('Blue_screw_terminal',term,BLUE,ld)
    feature('Terminal_screw_heads',compound([cyl(1.5,.3,x+3,y+2.5+i*5,69.6) for i in range(n)]),SILVER,ld)
for x,y in [(-10,13),(6,12)]:
    feature('Electrolytic_capacitor',cyl(3.1,8,x,y,61.6),BLACK,ld)
    feature('Capacitor_top',cyl(2.8,.3,x,y,69.6),SILVER,ld)
header(ld,6,-4,-19,61.6)
# Vertical red breakout in rear breadboard; identity is not readable in supplied images.
rb=group('Rear_breakout','Unidentified red rear breakout · visual envelope',electronics)
feature('Red_vertical_PCB',box(1.6,18,32,-50,10,93),RED,rb)
feature('Breakout_module_shield',box(2,10,10,-51.8,14,111),SILVER,rb)
feature('Breakout_IC',box(1.4,7,6,-51.4,15,100),BLACK,rb)
feature('Breakout_solder_pads',compound([cyl(.8,.25,-50.2,y,96+i*2.54,V(-1,0,0)) for y in [12,26] for i in range(10)]),SILVER,rb)
BOM += [('Arduino Uno envelope','1','68.6 × 53.4 PCB','Identified visually; mounting locations inferred'),('ESP32 DevKit envelope','1','26 × 49 PCB','Identified visually; verify revision'),('Breadboard','1','45 × 84 × 7','Photo-derived'),('L298N motor driver','1','43 × 43 PCB','Project README and photographs'),('Rear red breakout','1','18 × 32 PCB','Identity unreadable; visual placeholder')]

# HC-SR04 at BOTH ends, transducers facing outward.
for side in [1,-1]:
    g=group('HC_SR04_'+('Front' if side>0 else 'Rear'),'HC-SR04 · '+('front' if side>0 else 'rear'),sensors)
    x=side*72
    feature('Ultrasonic_PCB',box(1.6,45,20,x if side>0 else x-1.6,-22.5,41),BLUE,g)
    for yy in [-13,13]:
        feature('Transducer_aluminium_can',ring(8,6.7,11,x+side*1.6,yy,51,V(side,0,0)),SILVER,g)
        feature('Transducer_dark_diaphragm',cyl(6.65,.5,x+side*11.9,yy,51,V(side,0,0)),BLACK,g)
        mesh=[]
        for n in range(-5,6):
            # Short wire chords contained within the circular diaphragm.
            h=math.sqrt(max(0,6.1**2-n**2))
            mesh.append(cyl(.09,h*2,x+side*12.45,yy+n,51-h,V(0,0,1)))
        feature('Transducer_grille',compound(mesh),(.3,.34,.36),g)
    feature('Sensor_crystal',box(2,8,3,x+side*2,-4,42),SILVER,g)
    feature('Sensor_mount_bracket',compound([box(8,45,2,x-4,-22.5,38),box(2,45,5,x-1,-22.5,38)]),BLACK,g)
    feature('Sensor_header_pins',compound([box(1,1,6,x,-4+i*2.54,35) for i in range(4)]),GOLD,g)
# Two requested additions: outward-facing side modules above the tyre crowns.
for direction, angle, dy in [('Left',90,-17),('Right',-90,17)]:
    src=D.getObject('HC_SR04_Front')
    dst=group('HC_SR04_'+direction,'HC-SR04 · '+direction.lower()+' · added for navigation',sensors)
    for child in src.Group:
        feature(child.Name+'_'+direction,child.Shape.copy(),child.ViewObject.ShapeColor,dst,'Added side ultrasonic module; mount location inferred')
    dst.Placement=App.Placement(V(0,dy,17),App.Rotation(V(0,0,1),angle))

# Camera upright just behind front ultrasonic board.
cam=group('ESP32_CAM','ESP32-CAM · fixed forward camera',sensors)
feature('Camera_PCB',box(1.6,27,40,62,-13.5,59),(.025,.12,.13),cam)
feature('Camera_RF_shield',box(2,17,14,63.6,-8.5,82),SILVER,cam)
feature('OV2640_camera_housing',box(8,12,12,63.6,-6,66),BLACK,cam)
feature('Lens_barrel',cyl(4.3,3.5,71.6,0,72,V(1,0,0)),BLACK,cam)
feature('Lens_optical_glass',cyl(3.1,.6,75.1,0,72,V(1,0,0)),(.08,.12,.20),cam)
feature('Lens_highlight',cyl(.75,.05,75.72,-.8,73,V(1,0,0)),(.35,.47,.66),cam)
feature('Camera_flash_LED',box(2.6,3,3,63.6,7,90),WHITE,cam)
feature('Camera_mount_foot',box(12,32,2,54,-16,57),BLACK,cam)
BOM += [('HC-SR04 ultrasonic module','4','45 × 20 PCB; Ø16 transducers','Photo-derived front/rear; left/right requested additions'),('ESP32-CAM / OV2640','1','27 × 40 PCB envelope','Project README and front photograph')]

# Exposed harness recreated as tidy, distinct cables with recognizable high loops.
colors=[(.77,.025,.04),(.98,.32,.035),(.95,.77,.03),(.1,.55,.23),(.03,.40,.72),(.36,.13,.55)]
for i,col in enumerate(colors):
    wire('Rainbow_ribbon_'+str(i+1),[(62,6+i*1.25,98),(38,9+i*1.25,126),(-4,9+i*1.25,137-i*.25),(-38,7+i*1.25,101)],col,.53)
for i,(end,col) in enumerate([((3,-4,103),(.65,.65,.64)),((19,-4,103),WHITE),((29,43,103),(.42,.30,.49)),((5,43,103),(.90,.33,.05)),((-48,24,124),(.38,.40,.44)),((40,-4,103),WHITE)]):
    wire('Upper_jumper_'+str(i),[(-63+i*3,-20+i*5,103),(-65+i*4,-18+i*3,123+i*1.5),(-20+i*5,-10+i*3,138-i*.8),end],col,.46)
for i in range(8):
    col=RED if i%2==0 else BLACK
    sx=55 if i<4 else -55; sy=46 if i%4<2 else -46
    wire('Motor_power_'+str(i),[(sx,sy,77),(sx*.7,sy*.7,83),(12,sy*.5,79),(18,-15+i*3,70)],col,.55)
wire('Front_sensor_signal',[(73,-6,40),(64,-23,42),(40,-31,70),(-32,-31,94)],(.0,.49,.51),.65)
wire('Rear_sensor_signal',[(-73,4,40),(-64,27,43),(-40,31,73),(-34,24,94)],(.92,.43,.1),.55)
wire('USB_power_cable',[(54,-11,30),(69,-28,31),(58,-46,52),(13,-43,70),(-68,-33,96)],BLACK,1.7)
for i in range(12):
    feature('Dupont_connector',box(2.1,2.1,6,-66+i*3.1,-30+(i%3)*6,94),BLACK,wiring)
BOM += [('Jumper / ribbon harness','1 set','Photo-routed visual cables','Not an electrical schematic or routed wiring specification')]

# Requested navigation hardware; same construction is used for live updates.
nav_ns={'__name__':'vbmd_navigation_modules'}
exec(compile((ROOT/'add_navigation_modules.py').read_text(),'add_navigation_modules.py','exec'),nav_ns)
nav_ns['add_navigation_modules'](D)
BOM += [('MPU-6050 breakout','1','21 × 16 × 1.6 PCB assumed','Central underside of middle deck; chip envelope from TDK'),('Three-axis magnetometer breakout','1','18 × 16 × 1.6 PCB assumed','Exact chip unspecified; elevated nylon mast'),('Nylon compass mast and platform','1','Ø4 mast; 71.9 high; 22 × 20 platform','Inferred nonmagnetic mount; verify compass with motors on')]

# Export-ready metadata and provenance.
robot.addProperty('App::PropertyString','ScaleSource','Documentation'); robot.ScaleSource='Owner: 170 L × 160 W × 140 H mm, wheel Ø~60 mm. Internal geometry inferred.'
robot.addProperty('App::PropertyString','ControlMapping','Documentation'); robot.ControlMapping='Two drive channels, left pair and right pair, matching esp32-motor-driver differential drive.'
D.recompute()
for g in [chassis,motors,power,electronics,sensors,wiring]: g.Visibility=True
Gui.activeDocument().activeView().viewIsometric(); Gui.activeDocument().activeView().fitAll()
Gui.activeDocument().activeView().setAnimationEnabled(False) if hasattr(Gui.activeDocument().activeView(),'setAnimationEnabled') else None
Gui.activeDocument().activeView().setAxisCross(False)
Gui.activeDocument().activeView().setCameraType('Orthographic')
D.recompute(); D.saveAs(str(ROOT/'VBMD_Robot.FCStd'))
with (ROOT/'bill_of_materials.csv').open('w',newline='') as f:
    out=csv.writer(f); out.writerow(['Component','Quantity','Dimensions_mm','Evidence_or_limitation']); out.writerows(BOM)
print('BUILD COMPLETE:',D.Name,'objects:',len(D.Objects),'file:',D.FileName)
