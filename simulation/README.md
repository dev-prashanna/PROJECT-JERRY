# VBMD — Gazebo Harmonic + ROS 2 Jazzy

**Gazebo is the primary simulator for this project.** The CAD-derived robot, interactive world and ROS bridge are installed and built on this machine (Gazebo Harmonic 8.15.0 / ROS 2 Jazzy).

The model includes one ESP32-CAM, four ultrasonic modules (front/rear and requested left/right additions), four driven wheels, and a lower central holder containing two 3.7 V Li-ion cells. The cell envelopes provisionally assume 18650s. Outer dimensions are approximately 170 × 160 × 140 mm, with 60 mm wheels. Visual meshes preserve the CAD detail and component colors. Collision shapes, masses, motor behavior and sensor optics are estimates that need hardware calibration; this is not an exact digital twin.

## Launch

For the integrated Gazebo, live depth, SLAM map and autonomous exploration app:

```bash
simulation/run_robot.sh
```

It reuses a running simulator or starts one, waits for the sensor bridge and opens
the robot dashboard in curiosity/exploration mode. Close older robot controllers
first. Use `--monitor` to start disarmed. The dashboard has camera/depth and map
tabs, connection status, movement stop/start and Gazebo pause/resume controls.
See [curious exploration](../realtime_depth_app/docs/curious_exploration.md).

From the project root:

```bash
simulation/run_gazebo.sh
```

Equivalent: `python3 simulation/run_sim.py`. The launcher starts Gazebo, the ROS bridge, the sensor/command adapter and `robot_state_publisher`. It defaults to ROS domain **86**, Gazebo partition **vbmd_dev**, and Gazebo transport IP **127.0.0.1**. Local transport is explicit because Gazebo otherwise selected an inactive Docker interface on this machine. Set `GZ_IP` to the desired network interface when intentionally using remote clients.

Useful options:

```bash
simulation/run_gazebo.sh rviz:=true
simulation/run_gazebo.sh autonomy:=true
simulation/run_gazebo.sh headless:=true
```

`headless:=true` runs server rendering with EGL and still generates RGB and range data. `autonomy:=true` starts a simple sensor-only obstacle-avoidance controller; it is not SLAM, exploration, or destination navigation. The existing optional MuJoCo prototype is retained separately in [README_MUJOCO.md](README_MUJOCO.md); it is not the default.

If rebuilding or moving the workspace, run `simulation/setup.sh`. It uses the existing system ROS/Gazebo installation and colcon; no MuJoCo environment is required for Gazebo. The pre-exported meshes are included. The ROS package lives in `simulation/ros2_ws/src/vbmd_description`.

## Interact with the robot and world

Focus the Gazebo window and use **W/S** or up/down arrows for forward/reverse, **A/D** or left/right arrows to turn, and **X/space** to stop. Hold/repeat a key to keep moving. Commands expire after **0.5 seconds of simulation time**. A keyboard command also turns off reactive autonomy, returning control to you.

The world contains checker tiles, colored wall landmarks at varied positions, fixed obstacles and a low-mass **movable_box**. Select an object and use Gazebo's transform tool to move it; add Entity Tree or Inspector from the plugin menu if needed. Objects marked static can be repositioned through world editing/services but will not be pushed by contact. The movable box responds to physical contact. Change layouts, camera lighting and textures to make harder vision experiments, then save a separate world rather than overwriting the source generator's output.

Manual ROS driving in another terminal:

```bash
source /opt/ros/jazzy/setup.bash
source simulation/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=86
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.08}, angular: {z: 0.0}}'
```

Stopping the publisher lets the watchdog stop the wheels. Do not simultaneously use manual command publishers and reactive autonomy.

You can toggle local avoidance while running:

```bash
ros2 param set /vbmd_gazebo_adapter autonomy true
ros2 param set /vbmd_gazebo_adapter autonomy false
```

ROS users may launch directly after sourcing the workspace:

```bash
export ROS_DOMAIN_ID=86 GZ_PARTITION=vbmd_dev GZ_IP=127.0.0.1
ros2 launch vbmd_description gazebo.launch.py
```

`ros2 launch vbmd_description sim.launch.py` also selects Gazebo. A custom world can be selected with `world:=/absolute/path/to/world.sdf`; it must contain the robot and matching sensor/controller topics or include the supplied model and systems.

## Sensor and command interfaces

| ROS topic | Type | Meaning |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | User command; X linear and Z angular |
| `/camera/image_raw` | `sensor_msgs/Image` | One RGB camera, 640 × 480 at a target 10 Hz |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Matching ideal pinhole calibration, no distortion |
| `/ultrasonic/front`, `/left`, `/rear`, `/right` | `sensor_msgs/Range` | One scalar reading per ultrasonic module |
| `/joint_states` | `sensor_msgs/JointState` | Simulator wheel angles for robot visualization |
| `/tf_static`, `/tf` | TF | Fixed URDF transforms and animated wheels |
| `/clock` | `rosgraph_msgs/Clock` | Simulation time |

The robot has **no simulated lidar hardware, IMU, depth camera or encoder-based navigation input**. No `/scan` or `/odom` is exposed to navigation. Wheel joint states support visualization only; do not integrate them as if the hardware had encoders.

Gazebo Harmonic does not supply a native sonar sensor implementation here. Four narrow GPU ray fans approximate the ultrasonic beams. Each fan has nine rays across 15° horizontally, range 0.02–4 m, and 4 mm per-ray Gaussian noise. `gazebo_adapter.py` takes the minimum valid hit and publishes `Range` messages with the original measurement timestamp. The adapter emits one direction every 60 ms, approximately 4.17 Hz per direction. The underlying fans render independently at 20 Hz; this staggered output is an interface approximation, not a simulation of sequential acoustic transmission or cross-talk. The fans do not reproduce vertical beam shape, multipath, specular acoustic reflection, material effects, temperature or missed echoes. Because a minimum is taken over noisy rays, aggregate range noise is not exactly zero-mean Gaussian.

Raw implementation messages use explicitly internal `/sim/sonar_rays/*` `LaserScan` topics to bridge Gazebo's ray API. Those nine-ray arrays are **not robot observations** and must not be fed to a SLAM/localization system. Only `/ultrasonic/*` represents the requested hardware interface. The same applies to Gazebo's `/sim/privileged/wheel_odometry` and `/sim/privileged/odom_tf`: they remain on Gazebo transport and are not bridged into ROS navigation. Scene poses available from Gazebo are evaluation/debug data.

The camera uses a 55° vertical field of view (about 69.5° horizontal), a 3 mm near plane, and an 8 m far plane. These are provisional simulator optics. It does not yet model rolling shutter, JPEG degradation, Wi-Fi jitter or ESP32-CAM timestamp errors. Calibrate the actual camera at its operating resolution and update the model before expecting matching real-world behavior.

The chassis convention is +X forward, +Y left, +Z up; the camera optical frame is +Z forward, +X right, +Y down. Range frames use +X boresight. All URDF/SDF/mesh geometry is in **metres**, while FreeCAD uses millimetres.

RViz defaults to `base_footprint`, displaying the model and ranges without an odometry estimator. A future estimator should own `odom → base_footprint`, and global SLAM/localization should own `map → odom`. The simulator deliberately does not fabricate those transforms for the navigation pipeline.

## SLAM and autonomous navigation development

The simulation and ROS interfaces are working infrastructure, not a completed SLAM/navigation algorithm. One monocular camera can support visual SLAM, but metric scale and robust tracking must be established explicitly. Four broad ultrasound cones are useful for local avoidance; a range to an arbitrary object is not automatically the depth of a tracked image feature. The simulator does not resolve that association problem for you.

A practical development sequence using only the requested sensors:

1. Calibrate the real camera's intrinsics/distortion and camera-to-base transform, and measure all four range sensor offsets. Preserve acquisition timestamps.
2. Connect a monocular SLAM/visual-odometry frontend to the RGB stream. ORB-SLAM3 supports monocular operation; its upstream ROS examples use ROS 1, so a ROS 2 integration must be selected or developed. No ORB-SLAM3 installation or successful SLAM run is claimed by this package.
3. Develop and evaluate metric scale recovery, e.g. using a calibrated observable floor plane/camera height or a carefully associated visual/range constraint. These are research assumptions to validate, not guaranteed fixes. Motor PWM and commanded velocity are not measured travel.
4. Publish continuous odometry with uncertainty, handle tracking loss/relocalization, and construct an occupancy representation from estimated poses and timestamped range cones. A sparse visual feature map alone is not a complete 2D navigation map. Unknown space should remain unknown.
5. Integrate the four `Range` topics with Nav2's range sensor costmap layer. `config/nav2_range_layer.yaml` is a mergeable fragment, **not a complete Nav2 bringup**. Global planning still needs valid localization/transforms, map or exploration logic, and tuned controllers. Do not configure lidar-oriented `slam_toolbox` using invented scan data.
6. Record sensor-only runs, vary world layouts, and score localization/navigation against privileged scene poses kept separate from the estimator. Repeat with real recordings. Textureless scenes, blur, lighting changes, pure rotations and ultrasonic ambiguity remain important failure cases.

To record the hardware-equivalent streams in a sourced terminal:

```bash
export ROS_DOMAIN_ID=86
ros2 bag record /camera/image_raw /camera/camera_info /ultrasonic/front /ultrasonic/left /ultrasonic/rear /ultrasonic/right /tf /tf_static /clock
```

Official references: [Gazebo Harmonic sensors](https://gazebosim.org/docs/harmonic/sensors/), [Gazebo sensor support](https://gazebosim.org/docs/harmonic/comparison/), [four-wheel DiffDrive system](https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1DiffDrive.html), [Nav2 sensor interfaces](https://docs.nav2.org/rolling/configuration_and_development/first_time_robot_setup_guide/sensors/setup_sensors/), [Nav2 range layer](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/core_servers/costmap_2d/costmap_plugins/range/), [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3).

## Assets and regeneration

- `urdf/vbmd.urdf`: detailed ROS description, four continuous wheel joints, fixed sensor links and optical/range frames. Keep its package mesh assets with it.
- `meshes/`: 67 CAD-derived color-separated visual meshes and link-pose manifest.
- `models/vbmd.sdf`: Gazebo robot with collision geometry, mass/inertia, wheel drive, camera and ultrasonic proxies.
- `worlds/vbmd_lab.sdf`: full self-contained interactive world; no external Fuel download required.
- `scripts/generate_models.py`: URDF and optional MJCF generator from the CAD mesh manifest.
- `scripts/generate_gazebo.py`: Gazebo model/world/bridge generator from the same manifest and URDF inertias.
- `config/gazebo_bridge.yaml`: explicit ROS/Gazebo topic mapping; command direction is ROS-to-Gazebo, sensors are Gazebo-to-ROS.
- `scripts/gazebo_adapter.py`: range aggregation, staggered output, keyboard/ROS commands, watchdog and optional reactive avoidance.
- `artifacts/gazebo/`: captured camera images and checks of the running Gazebo stack.

To change robot geometry, edit/rebuild the CAD, then export meshes with `cad/vbmd_robot/export_sim_meshes.py` inside FreeCAD with the robot document active. Run `simulation/setup.sh` afterward. Preserve changes to world layouts separately; generated files will be replaced on regeneration.

The physics uses approximate positive inertias and a provisional mass of about 0.755 kg: 0.55 kg base, four 0.04 kg wheels, and five 0.009 kg camera/range links. Actual masses, center of gravity, torque limits, tyre friction and camera calibration need measurement. The battery holder is placed at the lower geometric center; physical CG is not established by appearance.

## Checks

The ROS package builds successfully. Live checks verify 640 × 480 RGB delivery, camera frame IDs/intrinsics, four directional `Range` topics, four wheel joints, fixed transforms, command handling, and the absence of navigation `/scan`, `/odom` or privileged pose topics by default. SDF validation reports the model valid; `gz_frame_id` is a Gazebo extension and produces an SDFormat warning while correctly setting message frame IDs.

The contact-interaction test also drove the robot into the movable box and verified that the box moved, then restored the starting poses.

Run with the simulator active in domain 86:

```bash
ROS_DOMAIN_ID=86 python3 simulation/tests/test_ros_smoke.py
ROS_DOMAIN_ID=86 GZ_IP=127.0.0.1 GZ_PARTITION=vbmd_dev python3 simulation/tests/test_gazebo.py
# Also test pushing a box, then restore both starting poses:
ROS_DOMAIN_ID=86 GZ_IP=127.0.0.1 GZ_PARTITION=vbmd_dev python3 simulation/tests/test_gazebo_interaction.py
```

The Gazebo integration test uses privileged scene poses only to evaluate physical movement and stopping; it never publishes them to the navigation interface. It sends a short low-speed motion command, then stops the robot. Close the simulator with Ctrl-C in the launch terminal; closing only a separately attached GUI may leave its server running.

The CAD also includes the subsequently requested MPU-6050 and elevated generic magnetometer. They are included as visual geometry with the base link. This update adds no IMU/magnetic-field topics; the documented camera/ultrasound-only experimental interface remains as configured. See `cad/vbmd_robot/README.md` for locations, assumptions and die-axis checks.

## Live depth window and depth-assisted avoidance

With Gazebo running, execute `simulation/run_depth_view.sh` from the project root. The separate window runs the repository's existing Depth Anything V2 checkpoint on the robot's RGB camera, shows matched RGB/relative-depth views and all four labeled ultrasonic measurements, and provides Start exploring / STOP controls. It starts in monitor mode. See [robot depth integration](../realtime_depth_app/docs/robot_integration.md) for the controller, age gates, ROS interfaces and live checks. The inferred relative depth is not a metric depth camera.
