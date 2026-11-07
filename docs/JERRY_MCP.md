# Control Jerry from Codex

Jerry now exposes a local MCP server. Codex is the planner; the robot runtime does not call a separate LLM. ROS navigation, the executive, motion supervisor and adapter watchdog remain independent of Codex response time.

## Installed components

The existing `.venv-jerry` provides the local depth and AURA model dependencies. `requirements-jerry.txt` pins the added MCP SDK. Build the typed interfaces after a checkout or interface change:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths simulation/ros2_ws/src \
  --build-base simulation/ros2_ws/build \
  --install-base simulation/ros2_ws/install \
  --packages-select jerry_interfaces vbmd_description --symlink-install
```

Register the server with Codex:

```bash
codex mcp add jerry_robot -- /bin/bash "/home/prashanna/Documents/PROJECT VBMD/simulation/run_robot_mcp.sh"
```

The equivalent TOML, including a 120-second audio/model tool timeout, is in [codex-mcp.toml](../config/jerry/codex-mcp.toml). Codex supports local stdio MCP servers through `mcp_servers` configuration; see the [official Codex MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli). Reload the MCP connection or start a new Codex session after registration.

## Use it

Ask Codex: “Use Jerry's MCP to start the simulation, show me what the robot sees, explore for 30 seconds, then stop and recall what happened.”

Codex should:

1. Call `robot_status`, then `start_simulation` if needed. `headless=false` opens Gazebo's 3D window; the default runs rendering without that window.
2. Poll status until navigation and sensors are ready. Startup is disarmed.
3. Inspect `robot_camera` and `robot_map` for current evidence.
4. Call `start_exploration` with a unique request ID and finite budgets. Admission is separate from completion.
5. Query `get_task_status` and `robot_status` during the mission.
6. Call `stop_robot` for an explicit stop, then check terminal task state.

MCP does not itself schedule an always-on Codex loop. An active Codex session calls tools in response to your request. Navigation executes an admitted mission without asking Codex for every control tick.

## Tools

| Tool | Behavior |
|---|---|
| `start_simulation` | Starts an owned local Gazebo session disarmed; reports starting rather than ready. |
| `robot_status` | Reports service health, navigation readiness, mission state and actual motion authority, with receipt ages. |
| `robot_camera` | Returns a fresh image block, capture time, session/epoch and evidence ID. No invented metric depth. |
| `robot_map` | Returns the estimated map image and geometry metadata for inspection. |
| `start_exploration` | Submits an exploration mission capped at 120 active seconds, 300 wall seconds and 10 measured metres. |
| `stop_robot` | Independently requests supervisor stop, task cancellation and audio cancellation. Reports each acknowledgement. |
| `get_task_status` | Returns a live or retained terminal task result. |
| `wait_for_task` | Waits up to 15 seconds for a terminal result or measured travel progress; Stop stays independent. |
| `query_memory` | Retrieves bounded persistent task, fact and observation records. |
| `remember_fact` | Saves an explicitly requested statement with a caller-provided idempotency key. |
| `remember_observation` | Saves Codex's interpretation linked to a camera evidence ID and persisted frame manifest. |
| `speak` | Queues a bounded sentence through local Kokoro; queuing does not prove playback. |
| `audio_status` | Reports disabled, synthesizing, playing, idle or failed audio state. |
| `listen_once` | Records one host microphone utterance and transcribes with local Whisper. Returns text to Codex without admitting movement; recognized Stop also requests local revocation immediately. |

The current tool surface intentionally advertises implemented capabilities. Object-relative metric navigation, physical hardware, hands-free background operation and general-purpose terrain support are not enabled. Camera/map imagery and memories are observations; they cannot authorize motion.

## Simulation and storage

Default ROS domain: **87**. Gazebo partition: **jerry_integrated**. These separate the new runtime from the legacy dashboard's domain 86. Set matching `ROS_DOMAIN_ID` and `GZ_PARTITION` overrides on both processes when needed.

To run the simulator separately:

```bash
simulation/run_jerry.sh --no-console
# Or without the 3D window:
simulation/run_jerry.sh --headless --no-console
```

Leave it running while Codex connects to the MCP. The server stops an owned simulation on disconnect. When attaching to a separately launched runtime, disconnect requests stop but leaves that simulator running.

Persistent journal: `simulation/artifacts/jerry/memory.sqlite3`. Run logs and map exports: timestamped directories below `simulation/artifacts/jerry/`. Observation evidence: `simulation/artifacts/jerry/evidence/`, with a 256 MiB admission quota. The server retains at most 16 uncommitted camera frames. It refuses new evidence at quota rather than deleting images referenced by older memories.

Speech uses the laptop's speaker and microphone. `/listen` is push-to-talk-style, one-shot capture. Whisper and Kokoro run in isolated CPU workers; camera inspection can use Codex's MCP image input directly. Optional local Moondream remains available through the diagnostic interaction service. The integrated runtime makes no Groq/OpenAI/Anthropic provider calls.

## Validation

```bash
.venv-jerry/bin/python -m pytest tests/jerry -q
simulation/check_robot_mcp.sh
PYTHONPATH=. .venv-jerry/bin/python simulation/tests/check_speech_roundtrip.py
```

The live check uses the MCP Python client and actual stdio protocol to start Gazebo, wait for readiness, capture a real camera image, persist/retrieve an observation, admit exploration, observe authorized motion, stop, and check no restart. Results are saved to `simulation/artifacts/jerry/mcp-acceptance.json`. This checks the MCP/ROS/Gazebo path; it is not physical robot certification or an endurance benchmark.

## Extended build validation

`simulation/validate_jerry.sh --exploration-wall 90 --faults` measures real Gazebo movement against the planned route, confirmed frontier arrivals, localization error, mission expiry, old-proposal rejection and process-freeze stopping. Ground-truth poses are used only by its evaluator and never enter navigation. Telemetry and reports are saved in timestamped validation directories.

`simulation/validate_codex.sh` runs the installed Codex client against the registered MCP, captures its actual MCP calls and independently observes actuator output. It disables the unrelated FreeCAD/KiCad servers for that test and makes no model override. An MCP Python-client smoke test alone is not treated as proof that Codex operated the robot.

The local adapter combines a receiver-issued token with a command deadline in the same Linux boot's monotonic clock. A boot UUID identifies that clock domain. This shared-clock profile is for the current laptop simulation only; it must not be reused for an MCU or another host without a validated clock/token protocol.
