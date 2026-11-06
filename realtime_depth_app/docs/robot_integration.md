# Robot camera + depth + ultrasonic avoidance

This integration uses the existing `src/depth_engine.py` and the locally cached
`depth-anything/Depth-Anything-V2-Small-hf` checkpoint. It receives **the Gazebo robot's RGB camera** through ROS 2 and runs the actual model on the host computer. It does not use Gazebo's true depth, scene poses, or odometry as controller inputs.

## Run

**Recommended integrated launch:** `simulation/run_robot.sh` starts or reuses
Gazebo, connects its camera/sensors to this app, and opens SLAM + curiosity
exploration mode. Use `--monitor` to start disarmed. See
[curious exploration](curious_exploration.md) for the map, behavior, controls,
saved artifacts and current limitations. The original two-terminal workflow
below remains available.

Keep Gazebo running. From the project root:

```bash
simulation/run_gazebo.sh          # if it is not already running
# In another terminal:
simulation/run_depth_view.sh
```

The second command opens a separate **VBMD / ROBOT VISION** window. It loads the model offline and defaults to monitor mode. The first model warm-up may take several seconds; no autonomous movement is commanded while depth or other required inputs are missing.

The window provides:

- Camera frame paired with the exact frame used for the depth prediction.
- Relative depth/proximity heatmap: warm colors are nearer, cool colors farther.
- Left/center/right obstacle-band overlays, relative proximity scores and near-pixel fractions.
- Individually labeled **FRONT, LEFT, REAR, RIGHT ULTRASONIC** cards, with measured metres and receipt ages. A missing echo is labeled **NO ECHO**, not displayed as a measured 4 m distance.
- Controller state/reason, requested linear/yaw speed, inference latency and prediction age.
- **Start exploring**, **Stop**, and **Save report + map** buttons.

**Space** stops/disarms, **A** arms, **Esc** closes. Gazebo W/S/A/D or arrow-key input also disarms this controller and returns control to Gazebo's manual keyboard path. Closing the window sends a stop command when active. Saved views and status reports go to `realtime_depth_app/robot_outputs/`.

Arming first requests that `/vbmd_gazebo_adapter` disable its existing sonar-only autonomy, so the two controllers do not drive simultaneously. An unavailable/rejecting adapter prevents arming. Avoid other `/cmd_vel` publishers during autonomous operation. Manual ROS publishers do not automatically disarm this controller; use STOP first.

Optional commands:

```bash
simulation/run_depth_view.sh --auto
simulation/run_depth_view.sh --headless --duration 30 --output realtime_depth_app/robot_outputs/smoke
simulation/run_depth_view.sh --input-size 518
simulation/run_depth_view.sh --device cpu
```

The live adapter defaults to 392-pixel model input, autocast FP16 on CUDA (FP32 fallback on CPU), latest-frame buffering and no temporal smoothing in the control path. It reads the existing app configuration without modifying it. The original standalone webcam app remains available through `realtime_depth_app/main.py`. This machine already has the checkpoint and dependencies; `simulation/run_depth_view.sh` sets ROS domain 86 unless overridden and disables online model downloads.

## Relative depth is not a distance sensor

This checkpoint predicts **affine-invariant relative inverse depth**, with larger raw values nearer. The display normalizes the current prediction's 2nd–98th percentile into a **0–1 relative proximity score**. These values are not metres, collision probabilities, or stable absolute thresholds across scenes. Metres in the window come exclusively from the ultrasonic readings.

The control region excludes the top 20% and bottom 30% of the image to reduce sky/near-floor effects. Image thirds provide left, center and right rankings. This is a local depth-region avoidance heuristic, not semantic object recognition, SLAM or goal navigation. The relative model can fail on glass, reflections, blur, low texture and unfamiliar scenes; no collision-free guarantee is established. The low camera height and broad acoustic beams also leave blind spots. The integration has been exercised in the supplied Gazebo world; physical-robot operation requires separate tuning and validation.

## Controller behavior

`src/robot_avoidance.py` contains the engine-independent decision logic:

- Stop for missing, stale, invalid or nearly uniform depth; stale camera/clock; missing/stale/invalid range messages; or inference failure.
- Require the latest depth to be no older than 0.45 s, the camera no older than 0.65 s, and a running simulation clock updated within 1 s. Both receipt age and ROS acquisition-time age are checked for depth/ranges.
- Require all four ranges: receipt age at most 0.8 s and measurement age at most 0.65 s. Positive infinity means no echo within the configured beam limit; negative infinity means too close; NaN is invalid.
- Emergency stop when front clearance is below **0.18 m** or either side below **0.09 m**.
- Begin a pivot when front range is below **0.32 m**, or the center depth band has more than **22%** near pixels with an 80th-percentile proximity above **0.78**.
- Choose the direction using image-region proximity and side ultrasonic clearance; require at least **0.20 m** on the chosen side and **0.12 m** behind before initiating a pivot. Hold the turn direction to avoid oscillation. Stop if pivot clearance deteriorates.
- Resume only after front clearance exceeds **0.44 m**, center near-pixel coverage falls below **16%**, and at least **0.7 s** has elapsed in the turn.
- Advance at up to **0.08 m/s**, reduce speed near obstacles, and steer away from the nearer image sector. Close side ranges add slower corrective steering. The algorithm does not autonomously reverse.

These are conservative starting values for this simulated robot, not measured hardware limits. The Gazebo adapter retains its independent 0.5-second command-expiry watchdog. A paused clock, lost publisher or closed dashboard cannot authorize continued autonomous driving.

## ROS interfaces

Inputs:

- `/camera/image_raw`: RGB/BGR `sensor_msgs/Image`.
- `/ultrasonic/front`, `/left`, `/rear`, `/right`: `sensor_msgs/Range`.
- `/clock`: Gazebo simulation clock.
- `/sim/keyboard`: manual takeover events.

Outputs:

- `/cmd_vel`: `geometry_msgs/Twist`, published only while avoidance is active, plus a stop on disarm.
- `/perception/depth_relative`: `sensor_msgs/Image`, encoding `32FC1`, raw relative inverse depth, carrying the source camera frame and acquisition timestamp. This is **inferred relative depth**, not a calibrated depth-camera image.
- `/perception/avoidance_status`: JSON in `std_msgs/String`, including active state, reason, suggested action, proximity scores, range values, data ages and inference timing.
- `/perception/set_avoidance`: `std_srvs/SetBool`; true requests arming and false disarms. Its arming response acknowledges the request; inspect status to confirm active mode.

Example stop from a sourced terminal:

```bash
export ROS_DOMAIN_ID=86
ros2 service call /perception/set_avoidance std_srvs/srv/SetBool '{data: false}'
```

## Validation

From `realtime_depth_app` (avoids the repository's separate top-level `src` package):

```bash
python3 -m pytest -q tests/test_robot_avoidance.py tests/test_depth_engine.py
```

The checks cover relative-depth affine invariance and orientation, prediction shape/validity, stale inputs, invalid/infinite range semantics, emergency stopping, depth-directed escape selection, rear/side clearance and persistent turn direction. Live Gazebo runs verify offline CUDA inference, window rendering, sensor delivery, arming/control handoff and actual avoidance command output. Reports and a dashboard image are saved under `robot_outputs/`; the normal window stays in monitor mode until you arm it.
