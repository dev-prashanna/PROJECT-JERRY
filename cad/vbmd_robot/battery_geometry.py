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
