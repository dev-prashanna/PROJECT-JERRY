# VBMD robot — photo-derived CAD assembly

The FreeCAD model reconstructs the supplied six robot photographs, scaled to the owner's approximate measurements: **170 mm long × 160 mm wide × 140 mm high, with 60 mm wheels**. Its computed envelope is 170.0 × 160.0 × 140.6 mm; the small height difference comes from curved jumper wires.

## Open and use

1. Open `VBMD_Robot.FCStd` in FreeCAD.
2. Expand `VBMD · four-wheel differential-drive robot` to inspect chassis, motors, power, electronics, sensors, wiring, and four individual wheel assemblies.
3. To animate, open `Wheel_Motion.FCMacro` with FreeCAD's **Macro → Macros** dialog and execute it. Set left/right speeds or use Forward, Reverse, Turn left, Stop, and Reset. The macro only animates the CAD wheels; it sends no robot commands. The body stays stationary. Speed/yaw readouts use ideal differential-drive kinematics, without tyre slip or skid-steer physics.
4. Use the `Dimensions` spreadsheet to inspect owner measurements and inferred values. Native chassis outline dimensions, pad thicknesses, deck heights, and wheel-center positions have expressions. For a coordinated change to the entire robot, edit `dimensions.json` and rerun `build_robot.py` inside FreeCAD. Electronics, wire routes, tread and hub geometry are built as detailed static solids, not as a fully constrained assembly of manufacturer parts.

The build creates a new document and does not close or overwrite unrelated FreeCAD documents. The output `VBMD_Robot.FCStd` is replaced when you rebuild.

## Files

- `VBMD_Robot.FCStd`: editable native assembly; two sketch/pad/pocket chassis decks, four fully constrained sketches, named component groups and independent wheel rotations.
- `Wheel_Motion.FCMacro`: nonblocking Qt timer and visual left/right drive controls.
- `exports/VBMD_Robot.step`: placed, colored component geometry for CAD exchange. No parametric feature history or motion controls.
- `exports/VBMD_Robot.stl`: full assembly mesh in millimetres, including separate and intersecting visual components. It is **not a single printable solid**.
- `exports/*acrylic_deckPocket.step` and `.stl`: individual plate geometry. Hole sizes and positions are inferred; inspect before machining or printing.
- `previews/`: 2400 × 1800 shaded, isometric, top, side, front and rear CAD images; 1800 × 2400 exploded view. The exploded view is illustrative and does not alter the saved assembly.
- `assembly_reference.svg`: self-contained presentation/reference sheet with orthographic views and dimensions.
- `bill_of_materials.csv`: modeled components with quantities and provenance. Not a procurement-ready BOM.
- `validation.json`: shape, sketch, envelope, wheel motion and STEP/STL read-back results.
- `references/`: all six original photographs.
- `build_robot.py`, `finish_robot.py`, `presentation.py`: construction, validation/export, and rendering scripts. Run them in that order from the FreeCAD Python console.

## What matches the photographs

Four black treaded tyres with yellow five-spoke rims; four yellow vertically mounted gearmotor housings with silver motor cans; transparent lower and middle decks; raised Uno and breadboard; ESP32 DevKit; L298N driver and finned heatsink; fixed ESP32-CAM/OV2640; front, rear, left and right ultrasonic modules; central lower two-cell Li-ion holder with two provisional 18650 cells; vertical red breakout; exposed rainbow ribbon and jumper harnesses.

This is a cleaned-up reconstruction. Irregular hot glue, foam deformation, individual wire tangles and solder joints are simplified. Two 3.7 V cells and their holder were added from the owner’s clarification. The 18650 form factor is assumed, and series/parallel wiring is unspecified. The requested left/right ultrasonic modules are added above the tyre crowns.

The unidentified red breakout is explicitly labeled as a visual placeholder. A servo is not added because its installation is not visible in these photographs; the repository's servo firmware alone does not establish an installed pan/tilt mechanism.

## Accuracy and project use

Only overall length, width, approximate height and approximate wheel diameter were supplied as measurements. Wheel width, wheelbase, deck dimensions, motor and board envelopes, holes, spacers, camera height and cable paths are inferred. Board connectors and component details are illustrative; no manufacturer datasheets or exact PCB revisions were used.

The model is suitable for project visualization, packaging discussions, presentation and visual wheel-motion demonstrations. Before fabrication, measure the deck outline, thickness, wheel width and track, shaft diameter and hub bore, motor mounting geometry, board hole patterns, module revisions, sensor elevations and connector/cable clearances. No collision-free manufacturing fit, stress calculation, electronics functionality, mass/inertia model, or real-world driving behavior is asserted. Some visual components intentionally overlap at mounts and interfaces.

Coordinate system: **+X forward/camera, +Y left, +Z up; ground Z=0; millimetres**. Left/right wheel pairs correspond to the project's two-channel differential-drive control arrangement. The chassis opening is a cable passage, not a validated cable-routing design.

## Rebuild in FreeCAD's Python console

```python
root = '/home/prashanna/Documents/PROJECT VBMD/cad/vbmd_robot/'
exec(compile(open(root + 'build_robot.py').read(), 'build_robot.py', 'exec'))
exec(compile(open(root + 'finish_robot.py').read(), 'finish_robot.py', 'exec'))
exec(compile(open(root + 'presentation.py').read(), 'presentation.py', 'exec'))
```

The scripts use this workspace's absolute output path at their top; adjust `ROOT` if moving the folder. Export and presentation scripts use the active robot document; activate it before running them. The animation macro can be run independently with the native document open.

## Physics and ROS

See [simulation/README.md](../../simulation/README.md) for the CAD-derived URDF, tested Gazebo Harmonic engine, ROS 2 sensor bridge, and sensor-only SLAM development constraints.

## Added IMU and magnetometer

The requested MPU-6050 is placed beneath the middle deck, with its chip center at **(0, -20, 49.55) mm**. This keeps it close to the chassis center, above the cells and below the driver board. A provisional GY-521-style **21 × 16 mm** breakout is modeled with a **4 × 4 × 0.9 mm** MPU-6050 package, nylon spacers and brass mounting hardware. The chip envelope follows [TDK product dimensions](https://product.tdk.com/en/search/sensor/mortion-inertial/imu/info?part_no=MPU-6050); the breakout footprint and holes are assumptions.

A generic **18 × 16 mm** magnetometer breakout sits on a nylon mast, with its chip center at **(-28, 43, 134.5) mm**. No magnetometer chip identity is assumed. This position avoids the side ultrasonic bracket and increases separation from the motors and battery/current wiring while retaining the original outer envelope. The mast and nearby modeled fasteners are nylon and brass. Placement follows [NXP magnetic interference guidance](https://www.nxp.com/docs/en/application-note/AN4247.pdf); it does not establish magnetic cleanliness. Verify heading with the motors energized and calibrate hard/soft-iron effects on the assembled robot.

Both boards have XY axis markers and chip-center metadata. Their mounting frames use +X forward, +Y left, +Z up; verify actual chip-to-board axes and purchased module dimensions before drilling or configuring software. `previews/navigation_layout.png` hides the middle deck and upper electronics to show both additions. The CAD/STEP/STL and simulator visual meshes include the parts; simulated IMU or magnetic-field streams have not been added by this CAD-only change.
