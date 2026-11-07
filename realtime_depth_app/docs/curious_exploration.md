# Curious robot: Gazebo, SLAM and exploration

From the project root, run:

```bash
simulation/run_robot.sh
```

This starts the Gazebo robot and vision dashboard together, or connects to the
existing Gazebo bridge. It selects **Curious creature · SLAM exploration** and
arms movement automatically. The robot waits for fresh camera, depth, compass,
IMU, ultrasonic and SLAM data before moving. Use `--monitor` to start with motion
disarmed. Space or **Stop** stops it. Closing the dashboard stops
its controller; a simulator started by this launcher also closes. An existing
Gazebo session is preserved.

Close any older robot vision/controller window first: the session launcher
rejects a second controller rather than competing for `/cmd_vel`. Paused Gazebo,
stale sensors and invalid depth are displayed as stopped states. The launcher
reuses an existing bridge even if it is paused; it does not start a second world
with a conflicting `/clock`.

```bash
# Connect to the simulator already open on screen:
simulation/run_robot.sh --connect-only
# Observe without moving:
simulation/run_robot.sh --monitor
# Automated check with server rendering, if no simulator is running:
simulation/run_robot.sh --server-headless --headless --duration 90
# Use the existing two-terminal workflow:
simulation/run_depth_view.sh --mode explore --auto
```

## Behavior

The robot has an explicit intrinsic reward for discovering space. Candidate
frontier goals receive rewards for nearby unknown cells and low visit counts,
with penalties for distance and turning. A small persistence reward prevents
switching goals on every frame. The robot:

- Scans unfamiliar surroundings when it cannot yet plan a body-width route.
- Plans through **observed free space**, with obstacle inflation and clearance
  from unknown space. It does not drive straight toward an unknown map pixel.
- Prefers unfamiliar places and loses interest in repeatedly visited areas.
- Confirms arrival within 0.12 m for 0.5 s, then observes before selecting the next destination. Travel has no periodic observation interruptions.
- Keeps its destination while rerouting or observing a blocked corridor. If it
  gets no closer by at least 5 cm for 60 wall-clock seconds during active
  exploration, it abandons that destination and chooses another reachable
  frontier. Turning in place and small pose-estimate jitter do not reset the
  timer. Monitor mode, pauses and tracking outages do not consume it. The
  abandoned destination and its 35 cm neighborhood are avoided for five minutes;
  if no alternative is reachable, the robot holds or performs a bounded scan.
- Defers to depth/ultrasonic avoidance. At a front emergency stop, or after a six-second escape turn, it may first
  pause, then reverse at 0.035 m/s if rear clearance exceeds 0.35 m and both side
  clearances exceed 0.12 m. This ends after 0.10 m of visually measured movement
  or four simulation seconds, whichever occurs first. Stale data stops recovery.

These rules produce exploratory behavior; they do not give the robot emotions,
semantic object understanding or a guarantee of visiting every room.

## SLAM and map

The **SLAM map + exploration** tab shows free space, possible obstacles, unknown
space, candidate frontiers, the planned path and estimated robot pose. The status
shows the current interest, visual tracking inliers, keyframes and loop closures.

The experimental frontend tracks floor features using bidirectional optical
flow and a robust metric translation estimate. Metric scale comes from ROS camera
intrinsics and the supplied CAD camera mount: `(0.0757, 0, 0.072)` metres. Gyro,
accelerometer and compass supply attitude. ORB keyframes support local
relocalization and geometrically checked loop observations; a translation pose
graph corrects keyframes and replays stored ultrasonic observations into the map.
The compass constrains graph rotations. This is a constrained flat-floor SLAM
prototype, not a general 3D visual-inertial SLAM system.

The 12 × 12 m occupancy grid has 0.05 m cells. Only fresh, timestamp-associated
ultrasonic measurements update it. Finite hits add possible occupied endpoints;
no echo clears a capped 15-degree beam without inventing an obstacle. The
simulated minimum range bounds all nine beam rays; free updates stop 6 cm before
a finite hit. Broad-beam ambiguity is retained by spreading possible hit endpoints. Unknown remains -1,
free is 0, and occupied is 100. Relative neural depth remains a local collision
cue and is never converted into metric map distances.

Navigation does not read simulator poses, wheel joint positions, privileged wheel
odometry, true depth, or world layout files. Curiosity uses the SLAM map and
tracking state, with the transform `map → base_footprint`.

The dashboard's **Pause Gazebo** and **Resume Gazebo** buttons use the bridged
Gazebo world-control service. Either action disarms movement; press **Start
movement** again to resume autonomous control. The adapter expires commands
against both wall time and simulation time. Temporary sensor delays stop motion
without silently disarming a healthy control loop; a stalled control loop does
disarm. Closing Gazebo now shuts down its bridges instead of leaving a stale
connection behind.

| ROS output | Meaning |
|---|---|
| `/map` | `nav_msgs/OccupancyGrid`, SLAM occupancy map |
| `/slam/pose` | `geometry_msgs/PoseStamped`, estimated base pose in `map` |
| `/exploration/path` | `nav_msgs/Path`, frontier route |
| `/perception/avoidance_status` | JSON including interest, pose, tracking, map area and movement reason |

**Save map + trajectory** writes `map.pgm`, `map.yaml` and `slam_report.json` to
`realtime_depth_app/robot_outputs/`. Map YAML/PGM can be viewed by ROS map tools;
loading a saved map for continued localization is not implemented.

## Limits and checks

The floor must be flat, static, textured and visible; camera height/extrinsics
must match the robot. Repeated checker tiles can cause ambiguous matches. Loop
matching checks descriptor uniqueness, translation consensus, metric scale and
compass-compatible rotation, but false matches remain possible. The local
relocalization search is limited to 0.5 m around the estimate. Persistent tracking loss holds translation and checks recognized views; use manual
control to return to a known view if recovery is impossible. Tilt and compass calibration,
uneven floors and moving obstacles need further validation on hardware.

```bash
cd realtime_depth_app
python3 -m pytest -q tests/test_robot_slam.py tests/test_robot_navigation.py tests/test_robot_avoidance.py
```

Tests cover metric image projection and rendered floor motion, tracking loss,
loop-constraint graph correction, unknown/occupied semantics, map export,
inflated free-space routing, curiosity pauses, visit habituation and bounded
recovery. Live reports are kept separately under `robot_outputs/`.

Implementation references: [OpenCV planar camera geometry](https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html)
and [ROS OccupancyGrid definition](https://github.com/ros2/common_interfaces/blob/rolling/nav_msgs/msg/OccupancyGrid.msg).

## Responsive map and controls

Random-direction roaming is removed from the dashboard and command-line modes.
Frontier goals are chosen from SLAM occupancy and visit memory, and remain committed
until confirmed arrival, including through temporary obstruction or stalls. Live visual tracking and route search run on
separate workers; snapshots expose their individual ages and revisions.

The compact toolbar has Start exploring, Stop, simulator pause/resume and export.
The main tabs show the live map and camera/depth. Status and Sensors tabs contain
properly labeled tables with units, ages and validity. Start remains available
during initialization and queues an explicit request. Stop cancels pending arming.
Report export runs in the background.

Tracking loss holds translation; the displayed map continues to show live source
revisions and the loss state. Recovery checks trusted optical views and cycles
through remembered nearby headings. No position or map updates are fabricated.

The comprehensive hardware/software description is [JERRY_DESCRIPTION.md](../../JERRY_DESCRIPTION.md).

## Committed destinations, smooth motion and YOLO

Each destination has a stable ID. Frontier-score changes do not replace it. The existing safe route is retained until arrival or 60 seconds without measured travel progress. A stalled goal is skipped without counting an arrival. The status table shows the ID, remaining distance, stall timer and completed destinations. Saved maps survive application and simulator restarts; movement waits for visual relocalization.

Continuous lookahead pursuit and heading hysteresis replace rotate/drive threshold chatter. Commands ramp at bounded acceleration and yaw acceleration; explicit/data-loss/collision stops remain immediate.

The Live objects · YOLO tab shows pretrained YOLO11n boxes, labels and confidence on the exact source frame. The Objects table lists current detections. This runs separately at a capped rate with depth priority; stale boxes are cleared. Run `simulation/setup_vision.sh` once; wrappers then use `.venv-jerry`. Use `--no-detection` to disable, or `--detector-device cpu` to change its device. See [the full technical description](../../JERRY_DESCRIPTION.md#61-live-pretrained-yolo-object-detection) for interfaces and limits.

## SLAM latency and smooth route following

Depth inference and loop-closure graph/map rebuilding each use an isolated process.
Historical scan replay cannot block the live optical-flow frontend. Completed
corrections are applied at frame boundaries, including observations collected
while the rebuild ran; reset simulations discard the old correction job.
Depth queues retain the latest frame and preserve its timestamp and identity.

Floor motion uses the dominant translation cluster, with at least eight inliers
and 25% support; competing similarly supported motions are rejected. Recovery
checks at most three flow references or four descriptor references per attempt,
with descriptor searches spaced 0.4 seconds apart while lost and two seconds
apart while healthy. Two good frames confirm recovery. A fresh, straight, safe
corridor permits only a 0.20-second bridge at 0.020 m/s; otherwise translation
is held. Active remembered-view turning waits for 0.80 seconds of actual pose
loss. Stale camera, depth, heading or range data still stop movement.

Paths discard grid zigzags with collision-checked straight links up to 0.8 m.
Continuous pursuit looks ahead up to 0.40 m. Small heading errors are corrected
while driving; stationary alignment requires a large target error and a large
route-tangent error. The committed destination and safe remaining route persist.

New destinations require 0.35 m mapped-obstacle clearance. Route search penalizes
low-clearance cells so planned approaches agree with sonar caution distances.
Occupancy integration batches beam geometry in NumPy to reduce scan replay cost.

Near-floor matches within 0.70 m take priority over distant wall features. Four
spatially spread floor matches can constrain compass-assisted translation when
60% agree within 8 mm and both planar covariance eigenvalues exceed 0.00015 m².
The latest trusted image remains a recovery anchor after prolonged dropouts;
elapsed time alone no longer disables it. Heading and displacement checks remain.

The final 240-second live check completed three destinations over 5.665 m, with
no early destination changes. Tracking p95 was 50.7 ms, maximum 159.9 ms; longest
stationary command interval was 4.95 seconds. Brief safety/recovery holds remain.
See `simulation/artifacts/curiosity/floor_tracking.json` for measurements,
including per-destination state counts and the separate YOLO output checks.

## Persistent map memory

The desktop dashboard automatically saves the occupancy map, visual localization
references, scan history and exploration visit counts every 30 seconds and on
normal shutdown. Its stable checkpoint is
`simulation/artifacts/jerry/slam-memory.npz`, independent of the report output
directory. On reopening, the saved area appears immediately; movement waits until
the live camera recognizes a saved place. An unrecognized view holds position and
does not erase or overwrite the map with a new origin.

Use `--map-memory /path/to/room.npz` for separate rooms, or `--fresh-map` to
explicitly start over. The integrated MCP runtime uses its own
`simulation/artifacts/jerry/integrated-slam-memory.npz` checkpoint.

In a tight corner, mapping scans select an open turning side rather than requiring
both sides to be open. A reachable waypoint in already observed space may be used
to leave a corner before selecting a new frontier; obstacle inflation and unknown
space clearance still apply.

A corner recovery may bridge an unmapped gap immediately beside the robot using
fresh local depth and all four ultrasonic readings. This exception is restricted
to 3 cm steps at at most 0.02 m/s, with a total limit of 0.10 m or four simulation
seconds. It requires forward clearance above 0.40 m, both sides above 0.12 m,
rear clearance above 0.10 m, and an obstacle-inflated check against every mapped
obstacle. Normal destination routes still require observed free space. Recovery
and stall timers wait for live perception to finish warming up.
