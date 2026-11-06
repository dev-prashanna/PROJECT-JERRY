> Optional retained prototype. Gazebo is the project default; see README.md.

# VBMD — CAD-derived URDF, MuJoCo physics and ROS 2 Jazzy

The simulator is installed and tested in this workspace. It uses the detailed FreeCAD assembly, including **one ESP32-CAM, four ultrasonic modules (front/rear plus requested left/right additions), four driven wheels, and a lower central two-cell 3.7 V Li-ion holder**. Robot scale is 170 × 160 × approximately 140 mm, with 60 mm wheels. The cell geometry provisionally assumes 18650 cells; actual cell dimensions, mass, electrical configuration and center of gravity need measurement.

The visual meshes are color-separated exports of the CAD, not a stock robot proxy. Simple collision shapes and explicit approximate inertias keep contact simulation practical. **Geometry and physics are estimates, not an exact digital twin**: motor torque, tyre friction, component mass, lens parameters, ultrasound acoustics and Wi-Fi latency have not been calibrated against the physical robot.

## Start now

From the project root:

```bash
python3 simulation/run_mujoco.py
```

This opens MuJoCo. **W/S** command forward/reverse, **A/D** turn, **X** stop. The 0.5-second command watchdog stops motion unless commands are refreshed. Hold/repeat the key for sustained motion. Ctrl-C or closing the viewer exits.

Run sensor-only local obstacle avoidance:

```bash
python3 simulation/run_mujoco.py --autonomy
```

This is a reactive avoidance demonstration, not goal navigation or SLAM. It uses four noisy ranges, with a latched turn direction to avoid oscillation. It reads no simulator pose or wheel odometry.

Headless smoke run and saved camera/physics images:

```bash
python3 simulation/run_mujoco.py --headless --fast --duration 3 --output simulation/artifacts/smoke
```

The project environment is `.venv-vbmd-sim` (MuJoCo 3.17.0). It inherits ROS/OpenCV/numpy from the host without changing the system Python installation. `simulation/setup_mujoco.sh` recreates the environment and builds the ROS package if needed. The host requires Ubuntu/ROS 2 Jazzy, colcon, uv, OpenCV and an EGL/OpenGL driver; all are available on this machine. A simulator-only launcher does not require the ROS package to be built.

## ROS 2 launch

The ROS package is already built. The launch file starts MuJoCo, its ROS bridge, and `robot_state_publisher`. RViz is optional.

```bash
source /opt/ros/jazzy/setup.bash
source simulation/ros2_ws/install/setup.bash
export VBMD_SIM_PYTHON="$PWD/.venv-vbmd-sim/bin/python"
export ROS_DOMAIN_ID=86
ros2 launch vbmd_description mujoco.launch.py rviz:=true
```

Use `headless:=true` to omit the MuJoCo window. Use `autonomy:=true` for reactive range-based avoidance. Avoid sending `/cmd_vel` while reactive mode is active; its internal controller supplies the commands. Tests use domain 86 to isolate this project; use the same domain in every terminal or choose another consistently.

Manual ROS control in another sourced terminal:

```bash
export ROS_DOMAIN_ID=86
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.08}, angular: {z: 0.0}}'
```

Stop the publisher to trigger the 0.5-second watchdog. MuJoCo models skid steering, so achieved yaw can differ substantially from the ideal differential-drive command, especially at low speeds. Friction, velocity gain and torque limits are illustrative and require tuning to measured behavior.

| Topic | Type | Behavior |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Commands linear X and yaw Z; ±0.22 m/s and ±2 rad/s command limits |
| `/camera/image_raw` | `sensor_msgs/Image` | One RGB camera, 640 × 480, target 10 Hz simulated time |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Ideal pinhole intrinsics matching the simulated 55° vertical field of view |
| `/ultrasonic/front`, `/left`, `/rear`, `/right` | `sensor_msgs/Range` | Four modules, 0.02–4 m, 15° horizontal beam approximation, 4 mm Gaussian noise |
| `/joint_states` | `sensor_msgs/JointState` | True simulator wheel angles for animation; not integrated into odometry |
| `/tf_static` and `/tf` | TF | URDF fixed transforms and animated wheel transforms |
| `/clock` | `rosgraph_msgs/Clock` | Simulation time |

Ultrasound pings are staggered by 60 ms across the four modules: each sensor updates at about 4.17 Hz. Each reading is the nearest room hit from nine horizontal rays, not a lidar scan. The simulation does not model acoustic material response, vertical beam shape, multipath, cross-talk, temperature dependence or missed echoes. Out-of-range readings use infinity; every range message carries its own timestamp.

The simulated camera has no distortion, rolling shutter, JPEG artifacts or stream jitter. Replace its intrinsics and tune its timing/noise model from actual ESP32-CAM calibration before drawing hardware-performance conclusions. Camera optical frame uses standard +Z forward, +X right, +Y down; chassis uses +X forward, +Y left, +Z up.

No `/odom`, `/scan`, IMU or depth topic is supplied by default. `ground_truth:=true` additionally publishes `/sim/ground_truth/pose` and `sim_world → base_footprint` for debugging. Those are privileged simulator data and must not become SLAM inputs. The saved `run_report.json` contains debug positions for test evaluation regardless of ROS mode; it is not a robot sensor stream.

RViz defaults to `base_footprint` so the model and sensor cones work without an odometry estimator. A future estimator must own `odom → base_footprint`; a global localization/SLAM system must own `map → odom`. Do not run competing TF publishers. With explicit ground truth enabled, `sim_world` can be selected as the RViz fixed frame for debugging.

## Your camera + ultrasound SLAM development path

The requested sensor set is preserved: **no lidar, stereo, depth camera, wheel encoders or IMU is assumed**. One monocular camera can run visual SLAM, but its reconstructed map does not inherently establish metric scale. Four coarse sonar cones provide local proximity; a range to an arbitrary object is not automatically the metric depth of a matched visual feature. Robust scale recovery needs additional assumptions or associations, for example calibrated camera height over an observable ground plane or a deliberately designed visual/range association method. These remain development work.

1. Calibrate the actual camera at the exact stream resolution, including distortion; measure the camera-to-chassis transform and sensor offsets. Record synchronized image and range data with capture timestamps.
2. Start with a monocular visual SLAM/odometry frontend on the host computer, such as ORB-SLAM3 in monocular mode. The upstream ROS examples target ROS 1; a compatible ROS 2 integration must be selected or implemented. ORB-SLAM3 is **not installed or claimed operational** by this package.
3. Establish and evaluate metric scale explicitly. Apply the camera-to-base transform, handle tracking loss/relocalization, and produce a continuous odometry estimate with justified uncertainty. Do not infer measured travel from motor PWM or command integration.
4. Build an occupancy representation using estimated poses and timestamped range cones. Keep unknown space unknown; four directions cannot establish full free-space coverage. A sparse visual map alone is not a complete 2D occupancy map.
5. Integrate the four `Range` topics into Nav2's range sensor costmap layer. `config/nav2_range_layer.yaml` is a mergeable fragment for that step; it is **not a complete Nav2 bringup**. Global planning still requires localization, valid transforms, a map or exploration strategy, and tuned controllers. Standard lidar-oriented `slam_toolbox` is not configured for these inputs.
6. Evaluate on held-out trajectories and physical recordings, using simulator truth only for scoring. Expect failures in textureless scenes, motion blur, lighting changes, pure rotations, range ambiguities and limited camera overlap. The current simulator and avoidance demo provide a development starting point, not a validated autonomous navigation stack.

Official references: [MuJoCo Python/viewer](https://mujoco.readthedocs.io/en/stable/python.html), [URDF import and visual geometry](https://github.com/google-deepmind/mujoco/blob/main/doc/modeling.rst), [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3), [Nav2 sensor interfaces](https://docs.nav2.org/rolling/configuration_and_development/first_time_robot_setup_guide/sensors/setup_sensors/), [Nav2 range costmap layer](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/core_servers/costmap_2d/costmap_plugins/range/).

## Model files and regeneration

- `ros2_ws/src/vbmd_description/urdf/vbmd.urdf`: ROS URDF, four continuous wheel joints, sensor links and optical/range frames; package-relative mesh resources. Retain the mesh folder with the URDF.
- `meshes/manifest.json`: local link poses, colors and triangle counts exported from CAD. Mesh vertices are in **metres**, unlike the CAD's millimetres.
- `mjcf/room.xml`: MuJoCo room, obstacles, explicit inertias, wheel actuators, camera, lighting and detailed visuals. The MJCF and URDF share exported geometry; MJCF adds physics and room settings that URDF cannot express fully.
- `scripts/generate_models.py`: regenerate URDF and MJCF after exporting CAD meshes.
- `scripts/sim_core.py`, `scripts/run_sim.py`, `scripts/ros_bridge.py`: physics, viewing, sensor approximation and ROS adapter.
- `config/vbmd.rviz`, `config/nav2_range_layer.yaml`: RViz display and future Nav2 range-layer fragment.
- `artifacts/`: rendered physics/camera views, smoke-test report and ROS topic validation.

To export modified CAD geometry, run `cad/vbmd_robot/export_sim_meshes.py` inside FreeCAD with `VBMD_Robot` open, then run:

```bash
.venv-vbmd-sim/bin/python simulation/ros2_ws/src/vbmd_description/scripts/generate_models.py
```

If you move the workspace, update the FreeCAD export scripts' `ROOT`; meshes, URDF and MJCF are otherwise relative to the package. Rebuild the ROS package with `simulation/setup_mujoco.sh` after moving it. No Gazebo setup is claimed; MuJoCo is the tested physics engine, with ROS 2 communication and optional RViz.

## Verification

```bash
MUJOCO_GL=egl .venv-vbmd-sim/bin/python simulation/tests/test_physics.py
# While the ROS launch is running in the same domain:
ROS_DOMAIN_ID=86 .venv-vbmd-sim/bin/python simulation/tests/test_ros_smoke.py
```

Checks cover URDF import with visual meshes retained, stable ground contact, forward motion and stopping on command timeout, skid-steer rotation without tipping, image rendering, four directionally correct/staggered ranges, ROS topic delivery and transforms, and absence of privileged pose/odometry/lidar topics by default. See `artifacts/ros_smoke.json` for the live ROS check. Physics masses are provisional: base 0.55 kg, four 0.04 kg wheels and five 0.009 kg sensor/camera links, approximately **0.755 kg total**; inertias and CG are estimates.
