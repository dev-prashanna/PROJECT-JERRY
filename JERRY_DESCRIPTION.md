# JERRY — Robot Technical Description

**Project:** PROJECT JERRY / PROJECT VBMD  
**Documentation date:** 6 November 2026  
**Scope:** Physical design, CAD reconstruction, firmware, perception, Gazebo/ROS integration, map-based exploration, user interface, dialogue and memory software, validation, and implementation limits.

This document describes the components and code present in this workspace. A modeled component, an available firmware project, and a feature demonstrated on the physical robot are different levels of implementation. Dimensions reconstructed from photographs and simulator settings are identified as assumptions. Passwords and API keys are intentionally excluded.

## 1. System purpose and implementation status

JERRY is a small four-wheel differential-drive companion-robot project. Its intended behavior is to observe its surroundings, remember explored space, choose unfamiliar reachable areas, avoid obstacles, and support visual and spoken interaction.

The current integrated navigation system connects a CAD-derived Gazebo robot to a host-computer depth and SLAM application. Exploration selects goals using the occupancy map and visit history; **random-direction roaming is not an available dashboard or launch mode**. Curiosity is an explicit information-gain and habituation policy, rather than an emotional state or a learned personality model.

| Subsystem | Present implementation | Qualification |
|---|---|---|
| Mechanical robot | Photo-derived FreeCAD model, four wheels/motors, stacked decks and electronics | Most internal dimensions are inferred |
| Motor control | ESP32/L298N UDP firmware and Python driver | Separate from the ROS/Gazebo command path |
| Physical camera | ESP32-CAM/OV2640 HTTP JPEG/MJPEG firmware and Python receiver | Stream defaults are targets, not guaranteed throughput |
| Physical ultrasonic firmware | One HC-SR04 serial reader | Four-direction hardware acquisition firmware is not present |
| Four-direction ranging | Four sensors in CAD and Gazebo, ROS `Range` output | Gazebo uses narrow ray fans as ultrasonic proxies |
| IMU and compass | CAD mounting, simulated measurements, host attitude estimator | ESP32 acquisition firmware and physical calibration are not established |
| Depth estimation | Locally cached Depth Anything V2 Small on the host | Relative inverse depth, not an absolute depth camera |
| SLAM and exploration | Constrained floor-feature/compass pose estimation, occupancy mapping, keyframes, loop constraints, frontier planning and visit memory | Experimental flat-floor SLAM; not general 3D SLAM |
| Dashboard | PySide6 live map/camera views, labeled status/sensor tables, start/stop and simulator controls | Explicit tracking, planning and freshness states |
| Dialogue and multimodal memory | AURA speech, vision-language, response and SQLite memory modules | Runs as a separate application; not the owner of Gazebo exploration |
| Camera-head servo | Standalone ESP32 servo firmware and controller | Installed pan/tilt mechanics are not confirmed by the photographs |

## 2. Mechanical structure and dimensions

### 2.1 Envelope and chassis

| Parameter | Value | Basis |
|---|---:|---|
| Nominal overall length | 170 mm | Approximate owner measurement |
| Nominal overall width | 160 mm | Approximate owner measurement |
| Nominal overall height | 140 mm | Approximate owner measurement |
| Recorded CAD bounding box | 170.000 × 160.000 × 140.591 mm | `cad/vbmd_robot/validation.json`; wires affect height |
| Wheel diameter | 60 mm | Approximate owner measurement |
| Wheel width | 24 mm | Inferred |
| Front-to-rear wheelbase | 110 mm | Inferred |
| Left-to-right wheel-center separation | 136 mm | CAD wheel centers at Y = ±68 mm |
| Chassis deck outline | 156 × 110 mm | Inferred |
| Acrylic deck thickness | 2.5 mm | Inferred |
| Lower deck elevation | 20 mm | CAD parameter |
| Middle deck elevation | 54 mm | CAD parameter |
| Uno deck elevation | 94 mm | CAD parameter |
| Breadboard elevation | 86 mm | CAD parameter |

The assembly includes two transparent decks, spacers, an upper board carrier, motor housings and cans, four treaded black tyres with yellow five-spoke rims, sensor brackets, and simplified jumper/ribbon wiring. Each wheel is a separate assembly. Left and right wheel pairs provide two-channel differential/skid steering.

The ideal kinematic relationship used for interpretation is:

```text
v = (v_right + v_left) / 2
omega = (v_right - v_left) / wheel_separation
wheel_linear_speed = wheel_radius × wheel_angular_speed
```

Tyre slip, contact geometry, motor asymmetry and driver losses prevent these ideal equations from establishing actual hardware odometry.

### 2.2 Component inventory

| Component | Quantity | Modeled dimensions / role |
|---|---:|---|
| Acrylic deck | 2 | 156 × 110 × 2.5 mm |
| Deck spacers | 4 | Approximately 31.5 mm high |
| Uno standoffs | 4 | Approximately 37.5 mm high |
| Rear board carrier | 1 | 49 × 89 × 3 mm |
| TT-type DC geared motor | 4 | Approximate visual envelope 19 × 17 × 53 mm |
| Wheel and five-spoke rim | 4 | Ø60 × 24 mm |
| Two-cell holder | 1 | 77 × 44 × 19 mm |
| Li-ion cell | 2 | 3.7 V each; provisional Ø18.3 × 65.2 mm geometry |
| Arduino Uno board envelope | 1 | 68.6 × 53.4 mm PCB |
| ESP32 DevKit envelope | 1 | 26 × 49 mm PCB |
| Breadboard | 1 | 45 × 84 × 7 mm |
| L298N driver | 1 | 43 × 43 mm PCB, finned heatsink |
| Unidentified red breakout | 1 | 18 × 32 mm visual placeholder |
| HC-SR04-style module | 4 | 45 × 20 mm PCB, approximately Ø16 mm transducers |
| ESP32-CAM/OV2640 | 1 | Approximate PCB envelope 27 × 40 mm |
| MPU-6050 breakout | 1 | Provisional 21 × 16 × 1.6 mm PCB |
| Generic magnetometer breakout | 1 | Provisional 18 × 16 × 1.6 mm PCB |
| Compass mast/platform | 1 | Nylon support; modeled mast approximately Ø4 × 71.9 mm |
| Harnesses | One set | Illustrative wiring, not an electrical netlist |

Front/rear ultrasonic modules are photo-derived; left/right modules and navigation-sensor mounts are requested additions. The magnetometer chip type is unspecified. The red breakout must not be treated as an identified electrical component.

The two cells and their central lower holder are documented, but series/parallel wiring, battery capacity, protection, regulation, charging, current budget, electrical interfaces and complete wiring remain unspecified. Battery geometry does not establish the robot's center of gravity.

### 2.3 Coordinate systems and sensor mounting

Robot coordinates use **+X forward, +Y left, +Z up**, with ground at Z = 0. CAD construction uses millimetres; URDF, SDF, ROS and exported simulation meshes use metres. Camera optical coordinates use **+Z forward, +X right, +Y down**.

| Item | CAD position (mm) | Orientation / interpretation |
|---|---|---|
| Front-left wheel center | (55, 68, 30) | Rotation about axle |
| Front-right wheel center | (55, −68, 30) | Rotation about axle |
| Rear-left wheel center | (−55, 68, 30) | Rotation about axle |
| Rear-right wheel center | (−55, −68, 30) | Rotation about axle |
| Camera optical center used by SLAM | (75.7, 0, 72) | Fixed camera, forward-facing |
| Front ultrasonic link | (72, 0, 51) | Yaw 0° |
| Rear ultrasonic link | (−72, 0, 51) | Yaw 180° |
| Left ultrasonic link | (0, 55, 68) | Yaw +90° |
| Right ultrasonic link | (0, −55, 68) | Yaw −90° |
| MPU-6050 chip center | (0, −20, 49.55) | Beneath middle deck |
| Magnetometer chip center | (−28, 43, 134.5) | Raised nonmagnetic mast |

Ultrasonic proxy measurement origins are displaced 12.6 mm forward along each sensor's boresight: front/rear at X = ±84.6 mm and left/right at Y = ±67.6 mm. Mapping uses these offsets. Camera and IMU mounting assumptions must match the actual robot before physical navigation is attempted. Compass placement reduces proximity to motors but does not demonstrate freedom from magnetic interference.

## 3. CAD assets and model construction

The native assembly is [VBMD_Robot.FCStd](cad/vbmd_robot/VBMD_Robot.FCStd). Construction parameters are in [dimensions.json](cad/vbmd_robot/dimensions.json); the detailed inventory is [bill_of_materials.csv](cad/vbmd_robot/bill_of_materials.csv).

Available assets include:

- Native sketch/pad/pocket deck features and constrained mounting sketches.
- A dimensions spreadsheet, named component groups and individual wheel rotations.
- STEP assembly export and complete/individual-deck STL exports.
- Orthographic, isometric, exploded and navigation-layout previews.
- An SVG assembly reference, original photographs, and geometry validation results.
- `Wheel_Motion.FCMacro` for visual wheel animation and ideal speed/yaw readouts.
- 67 color-separated STL visual mesh files in the ROS description package.
- A URDF with 18 links and 17 joints, including optical/sensor frames and four wheel joints.

The recorded FreeCAD validation uses version 1.1.4 and reports 576 document objects, 266 visible shape objects, four fully constrained deck sketches, no invalid shapes in that check, and a successful STEP round-trip with preserved envelope. These are geometry checks, not manufacturing approval. The full assembly STL contains multiple/intersecting components and is not a single printable part.

Construction scripts are `build_robot.py`, `finish_robot.py`, `presentation.py`, `add_navigation_modules.py`, and `export_sim_meshes.py` under `cad/vbmd_robot/`. The wheel-animation macro does not send commands to the robot. Coordinated dimension changes require rebuilding; electronics and harness details are not a completely constrained manufacturer-part assembly.

## 4. Embedded electronics and firmware

### 4.1 Motor control: ESP32 and L298N

Source: [esp32-motor-driver/src/main.cpp](esp32-motor-driver/src/main.cpp).

| Motor channel | Enable/PWM | Direction inputs |
|---|---|---|
| A | GPIO25 | GPIO18, GPIO19 |
| B | GPIO14 | GPIO21, GPIO22 |

Firmware behavior:

- ESP32 operates as a Wi-Fi soft access point; the Python driver defaults to `192.168.4.1:8888` over UDP.
- PWM uses Arduino `analogWrite`/LEDC, 5 kHz, 8-bit duty cycle, values 0–255; startup speed setting is 200.
- Commands are `F` forward, `B` backward, `L` left pivot, `R` right pivot, `S` stop, and `V<number>` to set duty cycle.
- Speed changes take effect immediately during an active motion command.
- Motors stop at boot and after 2,000 ms without a received packet while moving.
- GPIO direction levels implement opposing left/right directions for pivoting; channel-to-side wiring must match the physical chassis.

The host [driver.py](src/robot/driver.py) refreshes speed/action every 0.5 seconds during a held motion and sends stop on close. Manual keyboard control does not require an LLM. Its separate language-driven mode uses structured action output through LangChain/Gemini; the configured model string is `gemini-2.5-flash`. A text-to-action call is not a closed-loop visual navigation system.

PWM duty is not a calibrated m/s command, and firmware availability does not prove installed wiring or successful physical autonomous driving.

### 4.2 Ultrasonic acquisition firmware

Source: [esp32-hcsr04/src/main.cpp](esp32-hcsr04/src/main.cpp).

| Setting | Value |
|---|---|
| Trigger | GPIO4 |
| Echo | GPIO2 |
| Trigger pulse | 10 µs, after a 2 µs low interval |
| Echo timeout | 30,000 µs |
| Distance conversion | `distance_cm = duration_us × 0.0343 / 2` |
| Maximum accepted distance | 400 cm |
| Serial rate | 115200 baud |
| Loop delay | 200 ms, plus acquisition time |
| Missing/too-distant echo | Printed as `Distance: out of range` |

This project reads one sensor and prints text over serial. It does not supply a four-sensor multiplexed ESP32 acquisition driver or automatically bridge those serial readings into the simulated ROS pipeline. A complete electrical interface specification is not included.

### 4.3 Servo firmware

Source: [esp32-servo/src/main.cpp](esp32-servo/src/main.cpp).

The standalone servo implementation uses GPIO14, 50 Hz LEDC, a 20,000 µs period, 16-bit duty resolution, a 500–2,500 µs pulse range, and commanded angles 0–180°. It initializes at 90°.

Newline-terminated serial commands are `A<degrees>` or a number, `W<microseconds>`, `S` to toggle sweep, `C` to center, `D` to detach, and `?` for help. Sweep advances one degree every 15 ms when the loop permits; this is a software command cadence, not measured shaft speed. `esp32-servo/servo.py` provides host interaction.

This is a separate firmware target. GPIO14 also appears in the motor firmware; these pin maps must not be combined unchanged on one board. A physical pan/tilt installation is not established by the CAD photographs.

### 4.4 ESP32-CAM/OV2640 camera

Sources: [camera firmware](esp32-cam-streamer/src/main.cpp), [receiver](esp32-cam-streamer/stream.py), and [camera README](esp32-cam-streamer/README.md).

The target board is AI-Thinker ESP32-CAM with OV2640. It uses Wi-Fi station mode, a 20 MHz camera clock, JPEG capture, and these board pin assignments:

| Camera function | GPIO |
|---|---|
| Power down / reset | 32 / no reset pin (`−1`) |
| XCLK | 0 |
| SCCB data / clock | 26 / 27 |
| Y9, Y8, Y7, Y6 | 35, 34, 39, 36 |
| Y5, Y4, Y3, Y2 | 21, 19, 18, 5 |
| VSYNC / HREF / PCLK | 25 / 23 / 22 |
| Board LED | 4 |

With PSRAM, allocation is prepared at UXGA before selecting HD 1280 × 720, JPEG quality 10 and two frame buffers. Without PSRAM, it uses QVGA and one internal-RAM buffer. Automatic JPEG quality adjustment operates between 10 and 25 toward a 20 FPS target; lower numeric quality means larger/higher-quality JPEGs. The target is not a measured guarantee of HD throughput.

| Network interface | Purpose |
|---|---|
| `http://<board-ip>:81/stream` | MJPEG multipart video |
| `http://<board-ip>:80/capture` | Still JPEG |
| `http://<board-ip>:80/status` | Resolution, quality/mode, stream FPS, RSSI and available memory |
| `size` / `quality` query settings | Supported resolution and quality selection; `quality=auto` restores automatic tuning |
| mDNS hostname | `esp32cam` |

The firmware includes asynchronous Wi-Fi association, reconnect attempts, connection/IP/RSSI diagnostics, periodic endpoint reporting, HTTP startup retry, server restart after network changes, synchronized sensor settings, JPEG validation, PSRAM staging buffers and `TCP_NODELAY`. The synchronous streaming handler serves one active stream at a time. The Python receiver decodes HTTP transfer chunks before extracting JPEG frames, reconnects after failures and uses latest-frame buffering.

PlatformIO targets use Arduino and `espressif32@6.13.0`. Motor/ultrasonic/servo targets use `esp32dev`, with 115200-baud monitoring and 921600-baud upload settings. The camera target uses `esp32cam`, `huge_app.csv`, 115200-baud monitoring and 460800-baud upload settings. Configuration files are project build settings, not proof that the latest firmware was flashed.

## 5. Gazebo digital model and ROS 2 integration

### 5.1 Runtime and physics

The primary simulation is Gazebo Harmonic with ROS 2 Jazzy and `ros_gz_bridge`. The repository's recorded Gazebo version is 8.15.0. MuJoCo 3.17.0 remains an optional earlier prototype and does not provide the current dashboard's exploration controller.

The self-contained SDF robot/world use CAD-derived visual meshes, simpler collision geometry and estimated inertias. Key simulator settings are:

| Parameter | Value |
|---|---:|
| Wheel radius / separation | 0.03 m / 0.136 m |
| Maximum linear velocity | ±0.22 m/s |
| Maximum yaw rate | ±2 rad/s |
| Maximum linear acceleration | 0.4 m/s² |
| Maximum angular acceleration | 2 rad/s² |
| Maximum physics step | 0.001 s |
| Requested real-time factor | 1; actual speed depends on host load |
| Gravity | (0, 0, −9.81) m/s² |
| Wheel collision friction coefficients | Approximately 0.35 / 0.35 |
| Provisional mass distribution | 0.55 kg base, 0.04 kg per wheel, 0.009 kg per camera/range link; about 0.755 kg total |
| ROS domain default | 86 |
| Gazebo partition / transport IP defaults | `vbmd_dev` / `127.0.0.1` |

Simulation does not electrically model batteries, current draw, motor-driver losses or thermal behavior. Masses, motor dynamics, friction and optical/acoustic properties are provisional.

The supplied lab, practice, validation and test worlds contain a floor, wall landmarks and obstacles. A low-mass movable box supports contact experiments. The rendering backend is Ogre2; headless sensor rendering uses EGL. The package also provides robot-state publication and an RViz configuration.

### 5.2 Simulated sensor characteristics

| Sensor | Implementation |
|---|---|
| RGB camera | 640 × 480, target 10 Hz, approximately 69.5° horizontal / 55° vertical FOV, clip range 0.003–8 m |
| Four ultrasonic proxies | Nine GPU rays per sensor across 15° horizontal FOV, 0.02–4 m, target 20 Hz raw ray updates, 0.004 m per-ray Gaussian noise |
| ROS ultrasonic output | One side every 60 ms, approximately 4.17 Hz per direction; minimum valid beam return, original acquisition timestamp retained |
| IMU | Target 100 Hz; angular-velocity noise standard deviation 0.002 and acceleration noise standard deviation 0.03 in the corresponding sensor units |
| Magnetometer | Target 20 Hz; per-axis noise standard deviation `1e−7` T |

The IMU adapter removes simulator orientation so the estimator receives gyro/acceleration rather than privileged attitude. Magnetic output is configured in tesla with the world convention handled explicitly. The ultrasonic adapter preserves positive infinity as no echo, negative infinity/below-minimum readings as too close, and invalid readings as invalid.

Ray fans approximate ranging interfaces. They do not reproduce acoustic multipath, material-dependent echoes, crosstalk, vertical beam shape, Wi-Fi delays or actual ESP32-CAM compression/rolling shutter.

### 5.3 Interfaces

| ROS interface | Type | Role |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Robot RGB/BGR image |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Pinhole calibration |
| `/ultrasonic/front`, `/ultrasonic/left`, `/ultrasonic/rear`, `/ultrasonic/right` | `sensor_msgs/Range` | Directional proximity readings |
| `/imu/data_raw` | `sensor_msgs/Imu` | Gyro/acceleration without simulator attitude |
| `/magnetometer/data_raw` | `sensor_msgs/MagneticField` | Compass measurements |
| `/clock` | `rosgraph_msgs/Clock` | Simulation time |
| `/joint_states` | `sensor_msgs/JointState` | Wheel visualization; not navigation odometry |
| `/sim/keyboard` | `std_msgs/Int32` | Manual takeover events |
| `/cmd_vel` | `geometry_msgs/Twist` | Host controller request |
| `/sim/cmd_vel_safe` | `geometry_msgs/Twist` | Adapter-limited, expiring command |
| `/perception/depth_relative` | `sensor_msgs/Image`, `32FC1` | Raw inferred relative inverse depth with source timestamp/frame |
| `/perception/avoidance_status` | `std_msgs/String` containing JSON | Movement request/state, reason, sensor ages, SLAM/plan status |
| `/map` | `nav_msgs/OccupancyGrid` | Estimated occupancy map |
| `/slam/pose` | `geometry_msgs/PoseStamped` | Estimated pose in map coordinates |
| `/exploration/path` | `nav_msgs/Path` | Planned frontier route |
| `/tf`, `/tf_static` | TF transforms | Estimated `map → base_footprint` and robot geometry |
| `/perception/set_avoidance` | `std_srvs/SetBool` | Request start or stop |
| `/world/vbmd_lab/control` | `ros_gz_interfaces/ControlWorld` | Pause/resume supplied Gazebo worlds |

Internal `/sim/sonar_rays/*` messages bridge Gazebo ray scans to scalar ranges; they are not simulated lidar hardware. Privileged Gazebo wheel-odometry/scene-pose information is excluded from the navigation inputs. Scene poses may be read by a separate evaluation script to measure actual motion.

The adapter bounds commands and expires them after 0.5 seconds against both wall and simulation clocks. Its drive timer uses a steady clock. The primary launcher reuses an existing simulator, rejects a second dashboard/controller, and closes owned child processes. Gazebo exit also shuts down its associated bridge processes. The GUI launch scripts remove incompatible Snap GTK/Qt environment overrides while retaining native ROS library paths.

## 6. Depth perception

The active model is `depth-anything/Depth-Anything-V2-Small-hf`, loaded from local cached weights. Host inference supports CUDA with CPU fallback.

| Setting | Standalone depth app | Integrated robot controller |
|---|---|---|
| Default model input | 518 pixels | 392 pixels; configurable |
| Precision | FP32 by default | CUDA autocast FP16, FP32 CPU fallback |
| Warm-up | Three iterations | One iteration |
| Temporal smoothing | EMA enabled, alpha 0.45 | Disabled in the control path |
| Acquisition | Webcam, image or video source | ROS robot camera |
| Display | Relative map and point measurements | Source-aligned RGB, relative map and navigation status |

Depth inference runs in a spawned process with a bounded latest-frame queue, two Torch CPU threads and one OpenCV thread. Its source frame identity and timestamp travel back with each prediction; the GUI, ROS control and visual tracker do not share its Python interpreter. Preprocessing, inference and postprocessing durations are reported separately. Camera/depth freshness limits remain enforced.

The checkpoint predicts relative inverse depth: larger values indicate nearer content, with arbitrary per-image scale and offset. The controller normalizes the 2nd–98th percentile to a 0–1 proximity score. It does not interpret those scores as metres or collision probabilities.

Obstacle analysis uses the middle-height band, Y = 20–70% of the image, and left/center/right X regions 6–34%, 34–66% and 66–94%. Each sector provides its 80th-percentile proximity and fraction of pixels above 0.72. A prediction requires at least 95% finite pixels and sufficient depth variation.

The standalone app additionally supports regional point measurements, camera-intrinsic calibration, undistortion, optional ArUco reference validation, temporal filtering and empirical metric profiles. Implemented fit forms include `D = a·s + b` and `D = a/(s + b)`, where `s` is a raw model score. Profile validation is scene-dependent; it does not make this checkpoint a general metric depth sensor. Exploration uses ultrasound for measured range and floor geometry for visual metric scale.

### 6.1 Live pretrained YOLO object detection

The integrated app includes **Ultralytics YOLO11n**, using the official pretrained `yolo11n.pt` weights and 80 COCO classes. This is object detection, separate from geometric depth and SLAM.

| Setting | Implementation |
|---|---|
| Library | `ultralytics==8.3.228` in the project `.venv-jerry` environment |
| Weights | `models/yolo/yolo11n.pt`, 5,613,764 bytes |
| SHA-256 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| Inference size | 416 pixels |
| Confidence threshold | 0.30, configurable |
| Scheduling | Up to four inferences/second by default, latest-frame buffering and depth-worker priority |
| Device | CUDA/FP16 when available; CPU fallback or explicit detector device |
| Live display | Exact detection source frame with box, class and confidence; Objects table |
| Freshness | Results older than one second are removed from the live overlay/table |
| ROS output | `/perception/objects`: JSON containing source frame/timestamp, class, confidence and `xyxy` pixel coordinates |
| Exports | `objects.png` and `objects_report.json` |

Run `simulation/setup_vision.sh` once to create the project environment and download/verify the official checkpoint. Launch scripts automatically use this environment when present. Runtime inference uses the local checkpoint; it does not download weights silently. `--no-detection`, `--detector-model`, `--detector-fps`, `--detector-confidence` and `--detector-device` control this optional subsystem. The default detector failure is surfaced independently; it is not treated as a motion sensor.

Boxes are never drawn over a different frame or converted into metric depth. They do not replace collision checking or change the committed exploration destination. Pretrained COCO detection can miss synthetic Gazebo props, plain blocks and objects outside its trained categories; zero detections can be a valid result. This integration does not claim custom training, persistent object tracking or a semantic 3D map.

The real checkpoint was checked on Ultralytics' public bus image: it detected a bus and people. The recorded warmed 416-pixel GPU pass took approximately 7.2 ms on this host; that is an isolated inference measurement, not guaranteed end-to-end stream latency. Results are in `simulation/artifacts/yolo/pretrained_check.json` and `pretrained_check.png`.

Model/API references: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), [predict interface](https://docs.ultralytics.com/modes/predict/), and [official pretrained weight release](https://github.com/ultralytics/assets/releases/tag/v8.3.0). Ultralytics' licensing terms apply to its library and weights.

## 7. Attitude estimation and constrained SLAM

### 7.1 Attitude filter

`robot_attitude.py` integrates gyro readings, corrects roll/pitch using the gravity direction, and corrects yaw using tilt-compensated magnetic measurements. It does not consume the simulator orientation quaternion.

The tilt correction has an approximately 0.5 s time constant; magnetic yaw correction uses approximately 0.8 s. Acceleration correction is accepted for norms between 7 and 12.5 m/s². The filter rejects invalid measurements and large integration intervals. Outputs include roll, pitch, yaw, compass heading, acceleration, angular velocity and magnetic field.

Camera processing uses timestamp-associated attitude samples rather than whichever orientation happened to arrive most recently. The host buffers up to 400 attitude samples and associates a camera stamp within 25 ms. This reduces apparent translation during robot rotation.

### 7.2 Visual frontend and metric scale

The frontend assumes a visible, flat, static floor and the calibrated camera mount `(0.0757, 0, 0.072)` m. It projects camera rays onto that floor, compensating roll/pitch, then removes compass-measured yaw from successive observations.

| Frontend setting | Current value |
|---|---|
| Feature region | Lower 48% of image |
| Corner detector | Up to 600 features; quality 0.008; minimum spacing 7 pixels |
| Optical flow | Bidirectional pyramidal Lucas–Kanade, 21 × 21 window, maximum level 3 |
| Forward/backward rejection | Approximately 0.8 pixels |
| Translation consensus | At least 8 inliers and 25% support in the dominant 0.012 m translation cluster; similarly supported conflicting clusters rejected |
| Ground-vector extent | Projected forward range approximately 0.10–1.4 m |
| Temporal recovery | Retains the last trusted frame through a rejected/dropped frame |
| Motion validation | Translation/yaw bounds reject implausible inter-frame jumps |

Nearby floor matches within 0.70 m receive priority over distant features that may belong to walls. A smaller support set is accepted only with at least four points, 60% agreement within 8 mm and two-dimensional spatial covariance above 0.00015 m²; collinear or tightly clustered points cannot activate this fallback. The newest geometrically accepted image remains a recovery reference after long dropouts, subject to heading compatibility and bounded metric displacement.

SLAM translation is derived from observed feature geometry, not integrated motor commands. Gyro/compass heading is a measurement; neither commanded speed nor simulated wheel odometry supplies position.

### 7.3 Keyframes, relocalization and loop constraints

ORB stores up to 1,600 descriptors per view. Keyframes are added after approximately 0.12 m translation or 0.12 rad yaw change, with a current budget of 300. Recent trusted grayscale/flow references are retained for recovery; older compact descriptors remain available.

Candidate matches are constrained by estimated proximity (0.5 m) and heading compatibility (0.45 rad). Descriptor matching uses Hamming distance, a 0.65 nearest-neighbor ratio, at least 20 supporting matches, and geometric checks. Similarity verification rejects scale deviations above approximately 4% and rotation disagreement above approximately 3°. Recent trusted views can also be checked with bidirectional flow for local recovery, independently of whether the current view has enough ORB descriptors. Recovery cycles through distinct remembered headings when one view does not restore tracking.

Accepted older-view loop constraints feed a translation pose graph. A least-squares solve corrects keyframe positions while compass measurements constrain rotations; stored range observations are replayed into a corrected map. This is a constrained 2D estimator, not full six-degree-of-freedom visual-inertial optimization.

Two consecutive accepted frames restore tracking readiness. Recovery checks at most three nearby optical-flow references; descriptor matching checks at most four candidates, every 0.4 seconds while lost or two seconds during healthy tracking. Short dropouts permit at most 0.20 seconds at 0.020 m/s (4 mm), only inside a freshly collision-checked straight corridor with fresh depth, heading and range clearance. Other losses hold translation; remembered-view turning begins only after 0.80 seconds of genuine pose loss. No untrusted observations update the map. Insufficient texture or unsafe surroundings can still require manual repositioning.

Occupancy fan integration batches ray samples and endpoint evidence in NumPy. A 1,000-scan synthetic replay on this host took approximately 0.50 seconds. Loop-closure graph solving and historical scan replay run in a separate process. Tracking continues against the current consistent map while that work runs. At a frame boundary, the result corrects poses, recent recovery anchors and new scans accumulated during the rebuild before replacing the map. A simulator reset discards pending corrections from the previous map epoch.

## 8. Occupancy map and exploration memory

The default map is a fixed **240 × 240 cell** grid at **0.05 m/cell**: a **12 × 12 m** area with origin `(−6, −6)` m.

| Map property | Implementation |
|---|---|
| Unknown / free / occupied | ROS values −1 / 0 / 100 |
| Evidence | Saturated log odds in approximately [−4, +4] |
| Free/occupied thresholds | Below −0.25 / above +0.65 for observed cells |
| Range integration | Fresh, timestamp-associated scalar ultrasonic observations and sensor extrinsics |
| Integration distance cap | 2.5 m |
| Free-space evidence | Beam interior up to its minimum return, with a 0.06 m margin before finite hits |
| Endpoint uncertainty | Possible hits distributed across the 15° beam |
| Robot footprint | Currently occupied footprint marked free |
| Memory of visits | Counts in 0.25 m spatial bins, including nearby-place habituation |
| Saved artifacts | `map.pgm`, `map.yaml`, `slam_report.json`, trajectory and visit information |

The beam-clearing assumption corresponds to the supplied simulator's minimum-over-nine-rays interface. Physical acoustic missed echoes require separate validation. A no-echo return does not create an obstacle at the maximum range.

Current memory is maintained during the running session and exported on save/exit. Automatic cross-session map loading and global relocalization in a previously saved map are not implemented. A simulation-clock reset starts a new mapping epoch so an old route is not used in a reset world.

## 9. Curious exploration algorithm

### 9.1 Goal choice

Exploration is deterministic and memory-driven:

1. Identify observed-free cells adjacent to unknown space.
2. Inflate occupied cells by a nominal 0.13 m radius; require approximately 0.11 m clearance inside known-free space.
3. Search reachable free space with eight-connected Dijkstra, rejecting diagonal corner cutting.
4. Score frontier goals by expected new observations, low visit count, travel effort, turning effort and goal persistence.
5. Lock the chosen destination and its ID until arrival is confirmed; retain a safe route instead of regenerating it at every update.
6. If blocked, retain the same destination and reroute or observe its corridor. Timeouts, changing frontier scores and temporary map loss never select another destination.
7. Choose new destinations at least 0.35 m from mapped obstacles so the 0.12 m arrival tolerance is compatible with forward-sonar caution and the camera/sensor mount offsets. Route search penalizes low-clearance cells to prefer open corridors. Follow with continuous lookahead only when swept segments remain inside inflated known-free space; otherwise replan. Depth/range checks can stop or override motion.

The frontier utility is approximately:

```text
utility = 2.5 × local_unknown_fraction
        + novelty_from_visit_memory
        − 0.18 × route_length_metres
        − 0.12 × required_heading_change_radians
```

Unknown density is evaluated in a 21 × 21 cell neighborhood. Visit habituation includes surrounding bins, so moving one cell sideways does not make the same place completely new. Visit counts are updated on changing bins or after a stationary time interval, rather than once per control tick.

Destination completion requires the trusted estimated position to remain within 0.12 m for at least 0.5 seconds. A 1.5-second arrival observation hold follows before selecting the next destination. An unreachable destination stays selected; its route can be rebuilt without moving the destination. A safe approach position within the arrival tolerance is permitted. There is no elapsed-time abandonment and no random-direction fallback. Stop/resume retains the planning mission; a new simulation epoch resets it.

### 9.2 Observation and recovery behavior

| Behavior | Trigger / action |
|---|---|
| Explore | Follow a known-free route toward an unfamiliar frontier, up to 0.065 m/s |
| Align | Turn toward the route; yaw correction capped at approximately 0.40 rad/s |
| Scan | Rotate at approximately 0.25 rad/s when observations are insufficient to form a route and clearances permit |
| Arrival observation | Hold after confirmed destination arrival; no periodic travel interruption |
| Rescan blocked corridor | Observe around the same destination bearing while holding translation and retaining the target |
| Relocalize | Turn toward a remembered view; hold translation until tracking is recovered |
| Back away | Bounded recovery after a front emergency or prolonged escape turn, only with valid rear/side clearance |
| Wait / stop | Missing data, stale route, tracking loss without safe recovery, insufficient clearance or explicit user stop |

The planner removes grid staircases using line-of-sight links up to 0.8 m, validating known-free inflated cells and diagonal corner crossings. The follower projects the robot onto the retained route and uses a continuous 0.07–0.40 m lookahead with swept-segment validation. Stationary alignment requires both target error above 1.05 rad and route-tangent error above 0.70 rad; it releases below 0.40 rad target error or 0.45 rad tangent error. Small heading and cross-track errors are corrected while moving. Pure-pursuit steering varies continuously with heading and speed, and speed reduces near arrival. Issued commands are limited to 0.06 m/s² acceleration, 0.12 m/s² normal deceleration and 0.60 rad/s² yaw acceleration. Safety/data-loss/user stops remain immediate; reversing brakes through zero. The GUI does not inject competing stop commands while the control loop is healthy.

Backing begins after a 0.4 s pause, requires rear clearance above 0.35 m and both sides above 0.12 m, and commands −0.035 m/s. It ends after 0.10 m of visually measured movement or four simulation seconds, with a retry cooldown. It is not unrestricted reversing.

These behaviors create a visible preference for discovering space. They do not establish semantic understanding of objects, guaranteed complete coverage, destination navigation to named objects or a learned emotional system.

## 10. Control, safety thresholds and responsiveness

The control timer runs every 50 ms using a steady clock. Sensor callbacks, depth inference, visual tracking, frontier planning, GUI updates and disk export have separate responsibilities:

```text
Gazebo → ROS sensor callbacks → latest camera/attitude/range buffers
                                  ├─ depth process → relative-depth safety cues
                                  └─ SLAM worker → live pose/map snapshots
                                                      ├─ planning worker → remembered frontier route
                                                      └─ map correction process → loop graph + scan replay

50 ms control loop → freshness checks → map-guided action / obstacle override → /cmd_vel
GUI timer → immutable live snapshots → visual views and labeled tables
Export worker → reports, images and maps without blocking Start/Stop
```

A dedicated rate-limited YOLO worker uses its own latest-frame slot and yields inference opportunities to the depth worker. Single-slot buffers discard old frames rather than building unbounded inference queues. Frontier graph search runs separately from tracking so a slow plan does not freeze map updates. Live snapshot revisions and update ages distinguish a stationary map, tracking loss and a stalled worker. Planning results have their own freshness limit.

| Control check | Default threshold / behavior |
|---|---|
| Camera receipt age | At most 0.65 s |
| Depth age | At most 0.45 s, checking receipt and acquisition-time age |
| Simulation-clock receipt age | At most 1 s |
| Range receipt / acquisition age | At most 0.8 s / 0.65 s, with small timestamp tolerance |
| IMU / compass receipt age | Approximately 0.5 s / 0.7 s |
| Live SLAM readiness | Fresh trusted tracking, approximately 0.45 s |
| Planned-route age | At most approximately 2 s |
| Front emergency | Below 0.18 m |
| Side emergency | Below 0.09 m |
| Front caution / obstacle turn | Below 0.32 m, or high center-band relative proximity/coverage |
| New pivot clearance | Selected side above 0.20 m; rear above 0.12 m |
| Held pivot clearance | Selected side at least approximately 0.17 m |
| Basic avoidance forward speed | Up to 0.08 m/s; exploration further caps this |
| Escape turn | Approximately ±0.65 rad/s; curiosity bounds prolonged turns with recovery |
| ROS command expiry | 0.5 s wall time and simulation time |
| ESP32 motor-firmware expiry | 2 s without received commands; a separate hardware path |

Start is an explicit persistent request. It can be pressed while the camera/model initializes. Adapter handoff is asynchronous, idempotent and retried when unavailable; Stop cancels a pending handshake so a late reply cannot re-arm movement. A handshake timeout prevents an indefinite invisible arming state. Temporary sensor delays hold motion without silently losing the request. A stalled control loop fails closed.

Arming disables the adapter's older sonar-only autonomy so two autonomous controllers do not drive simultaneously. Gazebo W/A/S/D, arrow keys and stop keys take manual control and disarm exploration. Explicit simulator pause/resume also disarms movement; Start exploring requests it again.

## 11. Dashboard and user interaction

The main interface is **JERRY · SLAM exploration and depth**. Its layout provides:

| Area | Information / control |
|---|---|
| Compact toolbar | Start exploring, Stop, Pause Gazebo, Resume Gazebo, Save report + map |
| State banner | Current control state and the reason for movement or waiting |
| Live SLAM map tab | Antialiased vector map, 0.5 m grid and scale bar, hatched unknown space, obstacle outlines, clustered gold frontier markers, labeled target/route, estimated heading, freshness/revision, zoom/pan and Fit map |
| Camera + depth tab | Source-aligned camera view and relative-depth heatmap |
| Live objects · YOLO tab | Actual source frame with class/confidence boxes; stale results removed |
| Objects table | Current labels, confidence percentages and image-space bounding boxes |
| Status table | Connection, movement request, tracking, current interest, target, pose, observed/free area, visited places, keyframes, timing and requested velocities |
| Sensors table | Individually labeled front/left/rear/right distance, receipt age and validity |
| IMU/compass table | Roll/pitch/yaw, compass heading, acceleration XYZ, angular velocity XYZ and magnetic-field XYZ with units |

The Start control is not enabled/disabled based on every brief prediction-age fluctuation. A registered request is shown as pending/armed, and Stop remains available. Export uses a background worker so disk I/O does not block control interaction. The map continues to report live frame revisions even when translation is held because tracking is unreliable.

Keyboard controls: **A** starts exploration, **Space** stops, **Esc** closes. Map wheel zoom and mouse drag affect only visualization.

## 12. Speech, visual reasoning and long-term conversational memory

AURA is a separate host-side companion-robot application. Its principal pipeline is:

```text
Microphone → VAD → faster-whisper STT → intent router → allowed tools
                                   → camera / Moondream / memory evidence
                                   → response composition → Kokoro TTS → speaker
```

| Component | Repository configuration / behavior |
|---|---|
| Speech recognition | `faster-whisper`, default `small.en`, automatic device, INT8 compute, 16 kHz audio |
| Voice activity detection | Enabled by default; threshold 0.5, approximately 1.2 s silence timeout; utterance bounds 0.35–15 s |
| Vision-language model | Local Moondream-2 by default, cached image encoding, structured frame identity and bounded request queue |
| Alternative visual providers | Configurable compatible remote vision backend; data flow depends on selected provider |
| Language routing/response | Configurable Groq, compatible chat APIs or native provider adapters; deterministic routing fallback |
| Router modes | LLM, deterministic, automatic and comparison modes |
| Response modes | Direct evidence-based templates or configured model composition |
| Speech synthesis | Local Kokoro-82M, default `af_heart`, speed 1.0, 24 kHz mono output |
| Camera defaults | Device 0, MJPG 640 × 480 at target 30 FPS, latest-frame buffering and reconnect |
| AURA depth service | Same Depth Anything V2 model family, independently configured background service |
| Orchestration | Validated states from listening/transcription through perception, generation, speech, errors and shutdown |

The repository's local Moondream configuration uses approximately 1.93 billion parameters and BF16 on suitable CUDA hardware. Its standalone visual-question-answering app supports live snapshots and frozen-frame follow-ups. A VLM answer describes a selected image; it is not continuous geometric SLAM or a collision controller.

AURA memory includes SQLite with WAL/migrations, declarative facts and corrections, an entity graph, conversation/episodic event logs, FTS5 retrieval, vector/hybrid search and bounded context assembly. Object memory uses cached CLIP embeddings when available or a deterministic visual-feature fallback. Face enrollment/recognition is configurable and consent-gated; recognition defaults to disabled. Raw-frame/audio persistence and model-download policies are separately configurable.

Optional Gemini text-memory support selects relevant stored context and extracts evidence-backed durable facts. The configured default string is `gemini-3.8-flash`; model access must be configured for the user's provider project. This describes a source-code setting, not a claim of API availability. Offline operation disables cloud-dependent calls as configured. Provider keys, private conversation content and wireless credentials do not belong in this document.

**Conversational memory and exploration memory are separate stores and processes.** AURA fact/face/object memory does not automatically become a navigable occupancy map. The present exploration controller does not use an LLM to choose velocity or identify semantically interesting objects.

## 13. Host software and file organization

The recorded development environment is Python 3.12 on Ubuntu with ROS 2 Jazzy, an NVIDIA RTX 4060 Laptop GPU with approximately 8 GB VRAM, and CUDA-capable PyTorch. Available hardware and drivers must be checked on another machine.

The depth app's requirements record PyTorch 2.13.0, Transformers 5.14.1, OpenCV 5.0.0, NumPy 2+, SciPy 1.11+, PyYAML 6.0.1, safetensors, Pillow, PySide6 6.6+ and pytest. These are repository settings/recorded versions, not a recommendation to replace a working CUDA installation. ROS message packages and Gazebo/ROS bridges are system dependencies.

| Location | Responsibility |
|---|---|
| `cad/vbmd_robot/` | CAD sources, measurements, BOM, exports and validation |
| `simulation/ros2_ws/src/vbmd_description/` | URDF/SDF, meshes, worlds, bridge configuration, launch files and adapter |
| `simulation/run_robot.sh`, `simulation/robot_session.py` | Integrated lifecycle, sensor connection and controller launch |
| `realtime_depth_app/robot_live.py` | ROS sensors, timestamp alignment, arming, control and reporting |
| `realtime_depth_app/robot_window.py` | Live visual views, table-based dashboard and asynchronous export |
| `realtime_depth_app/robot_map_view.py` | Antialiased metric map, obstacle contours, frontier clustering, scale bar, zoom/pan and cached layer updates |
| `realtime_depth_app/src/robot_slam.py` | Pose estimation, keyframes, mapping, independent planning and curiosity |
| `realtime_depth_app/src/robot_motion.py` | Acceleration/deceleration and yaw-rate smoothing, immediate safety stop |
| `realtime_depth_app/src/robot_detection.py` | Independent pretrained YOLO inference and exact-frame annotations |
| `realtime_depth_app/src/robot_attitude.py` | Gyro/accelerometer/compass attitude filter |
| `realtime_depth_app/src/robot_avoidance.py` | Relative-depth/range safety policy; retained legacy policy code is not exposed as an exploration mode |
| `realtime_depth_app/src/depth_engine.py` | Checkpoint loading and depth inference |
| `realtime_depth_app/src/calibration.py`, `geometry.py`, `anchor.py` | Optional empirical distance and camera/marker geometry tools |
| `esp32-*` | Separate firmware projects and host helpers |
| `src/robot/driver.py` | Hardware UDP/manual/text-to-action driver |
| `AURA/` | Dialogue, multimodal perception and persistent conversational memory |
| `moondream-live-vqa/` | Standalone visual-question-answering application |
| `simulation/tests/`, `realtime_depth_app/tests/` | Unit, dashboard and live simulator validation |

## 14. Operation and artifacts

From the repository root:

```bash
# Integrated simulator, live perception, map and memory-driven exploration:
simulation/run_robot.sh

# Open disarmed; press Start exploring when desired:
simulation/run_robot.sh --monitor

# Reuse an already-running Gazebo instance:
simulation/run_robot.sh --connect-only

# Headless automated run:
simulation/run_robot.sh --server-headless --headless --duration 120

# Separate launch workflow:
simulation/run_gazebo.sh
# In another terminal:
simulation/run_depth_view.sh --mode explore --auto
```

The wrapper starts exploration mode by default; the standalone robot application starts disarmed unless `--auto` is supplied. The CLI retains `--mode avoid` for local avoidance diagnostics; it does not expose random-direction roaming. Camera/model warm-up may take several seconds and never authorizes movement with missing inputs.

Hardware and dialogue commands remain separate:

```bash
python3 src/robot/driver.py --manual
python3 esp32-cam-streamer/stream.py stream --ip <board-ip>
cd AURA
python3 main.py --gui
```

Saved robot outputs include the occupancy PGM/YAML, SLAM/visit/trajectory report, RGB and relative-depth images, rendered map, clear vector-view export (`map_view.png`), dashboard image and control history. Default location: `realtime_depth_app/robot_outputs/`. Files contain estimates and observations; they are not a complete replayable simulator or a calibrated physical-room survey.

## 15. Validation evidence and remaining limits

Recorded evidence includes:

- Latest optimized SLAM check: **5.665 m autonomous path over 240 seconds**, **three confirmed destinations**, **no destination change before arrival**, and **618 live YOLO frames**. Visual processing was **50.7 ms at the 95th percentile / 159.9 ms maximum**; depth source age was **0.345 s at the 95th percentile**. Observed area grew from **1.420 to 10.768 m²**. Destination #2 had four STOP samples, rather than a prolonged recovery lock. The longest stationary command interval across the run was **4.95 seconds**, including initialization and observation holds. Brief recovery and depth-freshness stops still occurred; this does not claim uninterrupted tracking. The report is [floor_tracking.json](simulation/artifacts/curiosity/floor_tracking.json). Its assertions required at least two confirmed arrivals, tracking p95 below 150 ms, valid live sensor/model output, map growth, goal retention and bounded normal command acceleration. Loop closures were not triggered in this run; asynchronous correction is covered by the regression test that preserves newly accumulated observations.

- Earlier combined committed-destination/YOLO Gazebo check: **3.711 m autonomous path**, destination #1 confirmed complete before #2 was selected, **no premature destination changes**, and **304 live YOLO result frames**. Normal linear acceleration/deceleration stayed within 0.12 m/s² and yaw acceleration within 0.60 rad/s². The report is [yolo_committed.json](simulation/artifacts/curiosity/yolo_committed.json).

- CAD geometry/sketch/envelope/export checks in [validation.json](cad/vbmd_robot/validation.json).
- A prior exclusive autonomous Gazebo run in [check.json](simulation/artifacts/curiosity/check.json): approximately **1.707 m actual path**, **1.561 m exploration extent**, and observed map area growing from **1.348 m² to 8.130 m²**. Evaluation scene poses were not navigation inputs.
- Pause/resume service checks in [world_controls.json](simulation/artifacts/curiosity/world_controls.json), including a stopped clock while paused and advancing clock after resume.
- GUI/motion reports that include manual takeover must not be presented as exclusive autonomous-exploration evidence.
- Automated tests for metric floor projection, optical-flow recovery, keyframe graph correction, occupancy semantics, known-free routing, goal persistence/visit habituation, bounded recovery, pending-start cancellation, live map revisions, responsive Stop during export and command watchdogs.

**Current regression result:** 74 checks passed after the responsive dashboard and memory-driven planning changes, including queued Start/Stop, spawned-depth source-frame identity and failure handling, asynchronous loop correction with newly accumulated scans, obstacle-standoff destinations, slow-planner isolation, live map revision updates, background export, visit serialization, destination retention through delays/map loss/blocked routes, confirmed-arrival sequencing, bounded acceleration and steering, continuous pursuit, live YOLO buffer/overlay tests, collision-checked lookahead, vector-map hole/thin-obstacle preservation and frontier display clustering. This count describes that specific test run.

Relevant checks can be run with:

```bash
cd realtime_depth_app
QT_QPA_PLATFORM=offscreen python3 -m pytest -q \
  tests/test_robot_slam.py tests/test_robot_navigation.py \
  tests/test_robot_avoidance.py tests/test_robot_dashboard.py \
  tests/test_robot_motion.py tests/test_robot_detection.py tests/test_robot_depth_worker.py \
  ../simulation/tests/test_adapter_watchdog.py
```

The physical robot has not been shown by this integration to navigate autonomously with the same estimator. Camera intrinsics/extrinsics, motor response, range interfaces, timestamps, compass calibration and electrical wiring require physical verification. Repeated floor patterns, motion blur, weak texture, nonplanar floors, magnetic interference and poor lighting can cause tracking errors. Four range beams leave blind spots. Local relocalization and loop constraints do not guarantee globally correct maps or recovery from arbitrary kidnapping.

The map has a fixed extent and finite keyframe budget. Automatic cross-session localization, semantic object-directed exploration, complete Nav2 bring-up, guaranteed full-area coverage, learned personalities, fully onboard compute and hardware validation of the complete software stack are not established capabilities.

## 16. Source references

This description is based primarily on executable code and structured project artifacts; older subsystem READMEs may describe earlier stages.

- [Project overview](README.md)
- [CAD design and provenance](cad/vbmd_robot/README.md)
- [Gazebo and ROS package](simulation/README.md)
- [Curious exploration guide](realtime_depth_app/docs/curious_exploration.md)
- [Robot/depth integration](realtime_depth_app/docs/robot_integration.md)
- [Depth application](realtime_depth_app/README.md)
- [Camera firmware and receiver](esp32-cam-streamer/README.md)
- [AURA architecture](AURA/docs/ARCHITECTURE.md)
- [AURA Gemini memory behavior](AURA/docs/GEMINI_MEMORY.md)
- [Standalone Moondream application](moondream-live-vqa/README.md)
