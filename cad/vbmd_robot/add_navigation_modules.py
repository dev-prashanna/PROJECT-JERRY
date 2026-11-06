"""Add the requested IMU and compass, with explicitly provisional breakout footprints.
The MPU-6050 is below the middle deck near the chassis center. A generic compass
sits on a nylon mast within the existing envelope, away from motor/power hardware.
"""
import FreeCAD as App, Part
import math
V=App.Vector

def add_navigation_modules(doc):
    if doc.getObject('NavigationModules'):
        return doc.NavigationModules
    parent=doc.addObject('App::Part','NavigationModules');doc.Sensors.addObject(parent)
    parent.Label='Navigation · MPU-6050 and elevated magnetometer'
    blue=(.025,.25,.52);green=(.025,.33,.23);black=(.025,.032,.041)
    silver=(.70,.75,.79);gold=(.84,.62,.19);nylon=(.88,.89,.85);white=(.96,.97,.93)
    def part(name,label):
        p=doc.addObject('App::Part',name);parent.addObject(p);p.Label=label;return p
    def feat(name,shape,col,p,desc=''):
        o=doc.addObject('Part::Feature',name);p.addObject(o);o.Shape=shape
        o.ViewObject.ShapeColor=col;o.ViewObject.DisplayMode='Flat Lines';o.ViewObject.Deviation=.1
        o.addProperty('App::PropertyString','Description','Component');o.Description=desc
        return o
    def box(a,b,c,x,y,z):return Part.makeBox(a,b,c,V(x,y,z))
    def cyl(r,h,x,y,z):return Part.makeCylinder(r,h,V(x,y,z))
    def ring(ro,ri,h,x,y,z):return cyl(ro,h,x,y,z).cut(cyl(ri,h,x,y,z))
    def breakout(p,name,L,W,holes,col):
        sh=box(L,W,1.6,-L/2,-W/2,0)
        for x,y in holes:sh=sh.cut(cyl(1.,2,x,y,-.2))
        feat(name+'_PCB',sh,col,p,'Provisional breakout envelope; verify actual purchased board and mounting pattern')
        feat(name+'_plated_holes',Part.makeCompound([ring(1.65,1.05,.08,x,y,1.6) for x,y in holes]),gold,p)
    def header(p,n,y):
        x0=-(n-1)*2.54/2
        feat('I2C_header_housing',box(n*2.54,2.4,2.2,x0-1.27,y-1.2,1.6),black,p)
        feat('I2C_header_pins',Part.makeCompound([box(.5,.5,3,x0+i*2.54-.25,y-.25,1.) for i in range(n)]),gold,p)
    def axis_marks(p,z):
        # Raised white silkscreen arrows: X points along robot +X, Y along +Y.
        shapes=[box(3,.25,.04,-6,2.6,z),box(.25,3,.04,-6,2.6,z)]
        for pts in [[(-2.2,2.725),(-3.1,2.15),(-3.1,3.3)],[(-5.875,6.4),(-6.45,5.5),(-5.3,5.5)]]:
            wire=Part.makePolygon([V(x,y,z) for x,y in pts]+[V(pts[0][0],pts[0][1],z)])
            shapes.append(Part.Face(wire).extrude(V(0,0,.04)))
        feat('Board_XY_axis_silkscreen',Part.makeCompound(shapes),white,p,'X arrow forward, Y arrow left; Z upward; actual breakout die orientation must be checked')
    def metadata(p,device,origin,chip,footprint):
        p.addProperty('App::PropertyString','Device','Navigation');p.Device=device
        p.addProperty('App::PropertyString','AxisConvention','Navigation');p.AxisConvention='Mount frame +X forward, +Y left, +Z up. Check actual chip-to-board axes before configuring software.'
        p.addProperty('App::PropertyVector','ChipCenterInChassis','Navigation');p.ChipCenterInChassis=V(*chip)
        p.addProperty('App::PropertyString','FootprintAssumption','Navigation');p.FootprintAssumption=footprint
        p.Placement.Base=V(*origin)
    imu=part('MPU6050','MPU-6050 · central underside mount')
    metadata(imu,'MPU-6050 accelerometer + gyroscope',(0,-20,47.5),(0,-20,49.55),'GY-521 style 21 × 16 mm PCB, estimated; 4 × 4 × 0.9 mm IC')
    holes=[(-8,5.5),(8,5.5)]
    breakout(imu,'MPU6050',21,16,holes,blue)
    feat('MPU6050_4mm_IC',box(4,4,.9,-2,-2,1.6),black,imu,'MPU-6050 package envelope from TDK product specifications')
    feat('IMU_regulator',box(2.8,1.8,1.,3,-1,1.6),black,imu)
    feat('IMU_passives',Part.makeCompound([box(1.2,.6,.55,x,y,1.6) for x,y in [(-4,-2),(4,2),(2,4),(-2,4)]]),silver,imu)
    header(imu,8,-5.8);axis_marks(imu,1.61)
    for x,y in holes:
        feat('IMU_nylon_underside_spacer',ring(1.65,1.05,4.9,x,y,1.6),nylon,imu,'Rigid nylon spacer to middle-deck underside at z=54 mm')
        feat('IMU_M2_brass_fastener',cyl(.9,9.,x,y,0).fuse(cyl(1.8,1.4,x,y,9.)),gold,imu,'Inferred mounting location; drill pattern must be confirmed')
    compass=part('Magnetometer','Magnetometer · isolated elevated compass mount')
    metadata(compass,'Generic three-axis magnetometer; exact chip unspecified',(-28,43,132.4),(-28,43,134.5),'Generic 18 × 16 mm PCB; replace with measured module footprint')
    holes=[(-6.5,5.5),(6.5,5.5)]
    breakout(compass,'Magnetometer',18,16,holes,green)
    feat('Compass_sensor_IC',box(3,3,1.,-1.5,-1.5,1.6),black,compass,'Generic visual IC; no HMC/QMC/LIS chip identity or register map assumed')
    feat('Compass_passives',Part.makeCompound([box(1.2,.6,.5,x,y,1.6) for x,y in [(3,0),(3,3),(-3,-2)]]),silver,compass)
    header(compass,4,-5.6);axis_marks(compass,1.61)
    for x,y in holes:
        feat('Compass_nylon_spacer',ring(1.6,1.05,.8,x,y,-.8),nylon,compass)
        feat('Compass_M2_brass_fastener',cyl(.9,3.,x,y,-1.1).fuse(cyl(1.7,.9,x,y,1.9)),gold,compass)
    mounts=part('NavigationMounts','Nonmagnetic mounting hardware · nylon and brass')
    feat('Compass_mast_foot',box(8,6,2,-32,46,56.5),nylon,mounts)
    feat('Compass_nylon_mast',cyl(2.,71.9,-28,49,58.5),nylon,mounts,'Nylon mast; no steel hardware near compass')
    feat('Compass_mast_platform',box(22,20,1.2,-39,33,130.4),nylon,mounts)
    for x in [-30,-26]:
        feat('Mast_base_brass_screw',cyl(.9,3,x,49,55.5).fuse(cyl(1.7,.9,x,49,58.5)),gold,mounts)
    parent.addProperty('App::PropertyString','PlacementRationale','Navigation')
    parent.PlacementRationale='IMU centered in X and 20 mm right of geometric center, below middle deck and above batteries; compass elevated to reduce motor/current interference. Separation does not establish magnetic cleanliness.'
    parent.addProperty('App::PropertyString','CalibrationNote','Navigation')
    parent.CalibrationNote='Measure breakout footprints and die axes. Calibrate installed compass hard/soft iron and check motor-on heading error. True CG and magnetic performance require physical measurements.'
    doc.recompute()
    return parent

if __name__=='__main__':
    add_navigation_modules(App.ActiveDocument)
