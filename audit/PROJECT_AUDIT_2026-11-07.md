# PROJECT JERRY audit — 2026-11-07

## Assessment

The repository has working perception, dialogue, mapping, exploration, and firmware components. It does not yet provide a single autonomous control system for the physical robot. Several existing stop, cancellation, freshness, and shutdown guarantees fail under conditions reproduced below.

This was a repository-wide inventory and syntax check, followed by a focused review of executable control paths: physical UDP driving and ESP32 firmware; simulator commands and watchdogs; depth processing; visual/compass localization, occupancy mapping and frontier following; AURA orchestration, memory scheduling, and tool execution; and live VQA. CAD, ROS descriptions, launch scripts, configuration, documentation, and test coverage were also inspected for integration assumptions. This is not a claim that every line or every possible defect has been verified.

No application or firmware implementation was changed. The additions are this report and a hardware-free reproduction script. No robot was driven, no firmware was flashed, and no cloud model was called. CAD geometry, physical stopping distances, embedded compilation with the real ESP32 SDK, and full Gazebo missions remain unverified in this audit.

## Evidence and test results

| Check | Result |
|---|---|
| Parse tracked Python sources | 235 files; no syntax errors |
| AURA existing suite | 401 passed |
| Depth application existing suite | 164 passed, 2 failed |
| Focused rerun of the two depth failures | Shutdown passed; frame-skip test failed again |
| Moondream live VQA existing suite | 80 passed, 8 skipped |
| ESP32 camera receiver existing suite | 4 passed |
| Simulator adapter watchdog unit tests | 2 passed |
| MuJoCo physics suite using simulation venv and EGL | 5 passed |
| New audit probes | See `reproduce_findings.py`; observations below |

Tests were run from their application directories to keep the separate `src` packages isolated. Running the camera receiver tests from the repository root initially failed to import `stream`; the directory-local invocation passed. System Python lacks MuJoCo; its tests passed with `.venv-vbmd-sim/bin/python`.

Reproduce the audit evidence from the repository root:

```bash
python3 audit/reproduce_findings.py
```

The script uses installed Python dependencies and `g++`. It asserts the **observed faulty behavior**, so a successful run means it reproduced the defects, not that the software is correct. The motor probe compiles the unchanged firmware against inert Arduino and UDP stubs. The reactive adapter probe executes its actual class from the parsed source without starting ROS. Other probes use the existing Python implementations and hardware-free fakes.

Priority: **P1** = fix before relying on autonomous operation; **P2** = correctness, availability, or reproducibility defect.

## Confirmed defects

### B01 — P1: Invalid UDP packets refresh the motor watchdog

Location: [motor firmware](../esp32-motor-driver/src/main.cpp#L142), particularly line 149.

`lastCommandMs` is updated before validating the packet. Unknown commands are ignored by `execute`, leaving the previous motion active. An `F` followed by `?` once per second still has `lastAction == 'F'` after ten seconds, despite a configured two-second timeout. The compiled firmware probe reproduces this and then verifies that true silence does stop it.

Refresh the motion deadline only after a fully validated command from the intended controller. Parse the complete packet, including speed values, before changing state. Add protocol tests for malformed traffic during movement.

### B02 — P1: Simulator reactive autonomy moves on invalid readings and pivots without clearance

Location: [Gazebo adapter](../simulation/ros2_ws/src/vbmd_description/scripts/gazebo_adapter.py#L59).

`beam_minimum` correctly preserves `NaN` and negative infinity, but `reactive` does not validate them. A fresh front reading of `NaN` falls through the comparisons and commands **0.08 m/s forward**. With front clearance 0.2 m and both sides and rear reporting negative infinity, it commands a **0.9 rad/s pivot**. A held turn also continues without checking side or rear clearance.

These defects affect `autonomy:=true`, the adapter's simpler reactive controller. The dashboard's `AvoidancePolicy` has stronger validation and is a separate path.

Validate all four measurements, require initialized values, and gate every pivot on the robot's swept footprint and current clearances. Share the validated safety policy between controllers.

### B03 — P1: Stationary navigation states retain forward motion

Location: [command smoother](../realtime_depth_app/src/robot_motion.py#L21).

`ALIGN`, `RESCAN`, and `LOOK_AROUND` request zero translation, but only `TURN` and `RELOCALIZE` force translation to zero. Starting at 0.065 m/s, each stationary state produces **0.059 m/s** after the first 50 ms update. The robot therefore moves while the planner expects a stationary observation or pivot. The checked route is not a validation of this residual translating turn.

Give stationary states an explicit zero-translation constraint while retaining appropriate angular smoothing. Test transitions from forward travel into every stationary state.

### B04 — P1: Cancelling an in-flight AURA request does not stop its later stages

Location: [agent execution](../AURA/src/orchestrator/agent.py#L914), [scheduler cancellation](../AURA/src/orchestrator/scheduler.py#L139).

The scheduler sets `Job.cancel`, but `_execute_job` does not pass that cancellation state into `handle_text`, and the pipeline never checks it between stages. The probe pauses routing, cancels the active job, then releases routing: the cancelled request still generates a response and queues speech.

Pass a request cancellation token through routing, tools, generation, memory commits, and playback. Check it before starting each stage and before publishing any result or side effect. This must be fixed before dialogue can participate in movement commands.

### B05 — P1: Tool timeout returns failure while the handler keeps running

Location: [tool execution](../AURA/src/orchestrator/tool_registry.py#L163).

The timeout only bounds `done.wait`. Its daemon thread continues after `execute` returns failure. The probe times out a handler, then releases it and observes its later side effect. Existing registered tools include persistent memory operations, so callers cannot interpret timeout as proof that nothing was changed. Repeated timeouts can also accumulate active handlers outside the bounded request queue.

Use cooperative cancellation and guarded commits for side-effecting tools, or terminate isolated workers where appropriate. Preserve per-request context and prevent a timed-out request from publishing or committing after its turn ends. No current motor tool is registered in AURA; the concern for future actuation is architectural.

### B06 — P2: Memory scheduler shutdown can raise and damage its priority queue

Location: [memory scheduler stop](../AURA/src/memory/resource_scheduler.py#L84).

`PriorityQueue` contains ordered `MemoryJob` objects. `stop` inserts `None`. With pending jobs, heap insertion raises **`TypeError: '<' not supported between instances of 'NoneType' and 'MemoryJob'`**. The exception is not caught, and heap insertion may already have appended the incompatible object. Existing tests do not exercise this pending-queue shutdown case.

Use a consistently ordered sentinel or an event-driven drain/cancel policy. Resolve pending jobs explicitly and join the worker before closing its database resources.

### B07 — P2: Live VQA accepts indefinitely stale camera frames after disconnection

Location: [camera frame access](../moondream-live-vqa/src/camera/camera.py#L342), [question submission](../moondream-live-vqa/src/pipeline/vqa_pipeline.py#L267).

The camera retains its last frame on disconnection. Live-mode `ask` checks neither streaming status nor frame age. A disconnected camera with a **3,600-second-old frame** still accepts “What do you see now?” and completes inference. Timestamp metadata is present, but submission does not enforce freshness.

Reject stale live snapshots using a configured maximum age. Keep intentionally frozen-image questions available with their distinct mode and timestamp.

### B08 — P2: Enabling undistortion does nothing in the depth application

Location: [depth application initialization](../realtime_depth_app/main.py#L109).

`geometry.undistort` exists in configuration, and `Undistorter` is imported, but the application only assigns `self.undistorter = None`. Even with valid, nonzero distortion coefficients and the flag enabled, no undistorter is constructed. Both preparation methods consequently return uncorrected data.

Construct it when enabled, scale intrinsics to the source dimensions, and apply the same transformation to RGB and depth. Verify the setting changes the output on a distorted synthetic image.

### B09 — P2: Direct BF16 mode actually runs FP16

Location: [model conversion](../realtime_depth_app/src/depth_engine.py#L104), [input conversion](../realtime_depth_app/src/depth_engine.py#L140).

`bf16` resolves to `torch.bfloat16`, but both direct weight and input conversion use `.half()`, which means FP16. The probe requests a BF16 conversion and receives `torch.float16`. Metadata still labels the execution BF16. This changes numerical behavior and makes precision comparisons unreliable. Autocast BF16 uses a different path and is not the defect described here.

Convert weights and inputs with the selected dtype, and verify the actual parameter and tensor dtypes for every supported precision.

### B10 — P2: Calibration `--check` crashes before loading the profile

Location: [calibration checker](../realtime_depth_app/tools/calibrate.py#L114).

`run_check` calls `load_profile`, but the function was never imported. The documented check path raises **`NameError: name 'load_profile' is not defined`** before opening a camera. Additionally, the prompt offers `s` to skip a distance but ignores the entered value.

Import the loader and honor the skip input. Add a CLI check using a saved profile and fake capture source.

### B11 — P2: Headless inference errors bypass pipeline cleanup

Location: [headless loop](../realtime_depth_app/main.py#L392).

An inference error returns `2` inside the `try`; its `finally` is `pass`. The later `pipeline.stop()` is skipped. The probe forces this branch and confirms stop was never called. Interruptions and other exceptions have the same cleanup problem.

Move pipeline shutdown into `finally`, including error and interruption exits.

### B12 — P2: Pipeline shutdown can report stopped with inference still running

Location: [pipeline stop](../realtime_depth_app/src/pipeline.py#L117).

The two-second joins are not followed by an alive check or a failure result. `_started` becomes false even if model inference continues. The existing shutdown test failed in the full run and passed in the focused rerun. A separate deterministic probe blocks a fake inference call beyond the join deadline and reproduces the live worker after `stop` returns.

Make shutdown state truthful. Await completion or expose a shutdown failure; isolate inference in a process if a bounded, enforceable stop is required. Ensure a replacement pipeline cannot concurrently reuse a model still owned by the old worker.

### B13 — P2: AURA's documented installation manifest is missing

Location: [AURA quick start](../AURA/README.md#L17), [root quick start](../README.md#L94), [setup guide](../AURA/docs/SETUP.md#L14).

All instruct users to install `AURA/requirements.txt`, but that file does not exist and is not tracked. There is no equivalent AURA packaging manifest in the inspected tree. A fresh checkout cannot follow the documented dependency installation, even though this machine's existing packages let tests run.

Provide a maintained dependency manifest with optional hardware, GUI, model, and test dependencies separated where useful. Validate installation in a clean environment.

## Algorithm and architecture gaps for the autonomous robot

These are distinct from the reproduced implementation defects above.

1. **The physical driving path is open loop.** `src/robot/driver.py::drive_scenario` sends a language-model action and holds it for a duration. It receives no range, pose, camera, or progress feedback. `esp32-hcsr04` prints one sensor's text over serial, while the navigation frontend consumes four ROS Range topics plus camera, IMU, compass, and camera calibration. No physical sensor/actuator bridge connects those interfaces in the current tree. A real obstacle does not interrupt the LLM action through this path. Build a hardware adapter and place deterministic safety checks below every motion source.

2. **Command ownership is not exclusive.** Dashboard arming disables the adapter's reactive autonomy through a parameter; it does not obtain a command lease. Two dashboard instances or another `/cmd_vel` publisher can still send competing commands. The physical UDP protocol likewise has no session, sequence, acknowledgement, or expiry carried in each packet. Implement one command arbiter with manual/emergency priority, ownership, expiring commands, and stale-packet rejection. Validate concurrent-controller and reordered-command behavior.

3. **Loop corrections do not version or rebase planner state.** `FloorSlam.apply_correction` adjusts pose, keyframes, and map but does not update planner goals, paths, or visit locations. The asynchronous planner's epoch changes on a clock reset, not a loop correction; `SlamWorker.snapshot` accepts a recent same-epoch plan after a correction. This can retain a destination in old estimated coordinates or temporarily reuse a route from before the correction. The follower rechecks local map clearance, which mitigates collision risk but does not preserve destination identity. Add map revisions and a defined destination anchoring/rebasing rule; test a loop closure while following a committed goal. This consequence is based on code inspection, not an executed loop-closure mission in this audit.

4. **Localization has restrictive observability assumptions.** Metric translation comes from image rays intersecting a level floor at a fixed configured camera height. The frontend chooses image regions and geometric consensus, not a verified floor surface; walls, clutter, moving objects, slopes, poor texture, and repeated patterns can violate those assumptions. Yaw depends on compass calibration and a world magnetic reference. These are limitations of the current algorithm, not proof of a particular localization failure in this run. Validate them with measured trajectories and expose uncertainty before using the map as a general navigation authority.

5. **Dialogue, semantic memory, and navigation remain separate.** AURA stores people, objects, and conversations; the navigation stack follows frontier goals. There is no shared grounded task representation connecting an instruction such as “go to the red chair” to an object location, navigation goal, success condition, or recovery policy. YOLO publishes image boxes rather than persistent world coordinates. This is an unfinished integration capability consistent with the roadmap, not a defect to disguise with more prompting.

6. **Long-duration operation needs explicit limits.** The occupancy map is a fixed 12 m square, keyframe insertion stops at 300, and scan history grows without a bound. Map replay and storage therefore have different long-run scaling limits. Define behavior at map and keyframe limits, bound or archive scan history, and validate multi-hour operation. The audit did not run a multi-hour mission.

## Test weakness exposed during this review

`test_frame_skip_limits_processing` failed both in the full depth suite and in the focused rerun because it requires `skipped > 0`. With a latest-frame slot, inference can already advance more than three capture IDs between reads, so no explicit frame-skip branch needs to run. This assertion depends on inference speed and is not, by itself, evidence that frame skipping is broken. Test it with controlled frame delivery and a fake inference engine; separately test that intermediate camera frames are discarded under load.

Passing component tests do not establish safe integrated autonomy. The newly reproduced invalid-input, state-transition, timeout, cancellation, and pending-shutdown cases should become regression tests when the corresponding fixes are implemented.

## Recommended repair order

1. Fix B01–B03 and consolidate motion validation under one safety authority.
2. Enforce command ownership and implement the physical sensor/actuator bridge before autonomous physical trials.
3. Fix B04–B06 so cancellation, timeouts, and shutdown have reliable outcomes.
4. Fix B07–B13 and replace timing-sensitive tests with deterministic checks.
5. Version map corrections, validate localization against measured trajectories, and connect semantic goals to navigation with explicit success and recovery states.
