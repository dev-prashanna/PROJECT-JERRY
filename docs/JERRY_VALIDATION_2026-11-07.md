# Jerry simulation build validation — 7 November 2026

**Build successful for the current Gazebo/MCP simulation profile.** This result combines an actual Gazebo trajectory, fault injection, an installed Codex client using the registered MCP, local speech models and software regression suites.

| Check | Result |
|---|---|
| Software suites | 614 passed, 1 expected environment-specific skip, zero failures |
| Actual autonomous travel | 2.50 m in the measured route run |
| Confirmed frontier arrivals | 1 |
| Planned-route error, p95 | 7.5 cm |
| Localization RMSE | 1.9 cm; no trajectory alignment used |
| Observed map growth | 8.03 m² |
| Premature destination changes | None |
| Stop and stale proposal replay | Passed; no restart after Stop |
| Wall-time mission limit | Passed |
| Navigation / supervisor / executive freeze | All independently removed actuator output within the tested expiry window |
| Actual Codex client | Passed; used Jerry MCP tools, with no shell commands or file changes |
| Codex-driven measured travel | 0.51 m; forward commands independently observed |
| Codex camera / map / memory / cancellation | Actual wire results verified |
| Kokoro → Whisper | Passed offline: “Stop the robot.” |

The skipped depth test is the CPU-only half-precision guard, which is inapplicable when CUDA is available. Speech round-tripping used generated audio; microphone and speaker hardware were not opened by that test.

## Evidence

- [Machine-readable summary](../simulation/artifacts/jerry/build-validation/summary.json), including source hashes.
- [Route and fault-injection report](../simulation/artifacts/jerry/validation-20261107T034022Z/report.json).
- [Actual Codex execution report](../simulation/artifacts/jerry/codex-validation-20261107T034451Z/report.json); its directory also contains the prompt, JSONL MCP trace, returned camera/map images, and wire checks.
- [Speech acceptance](../simulation/artifacts/jerry/speech-acceptance.json).
- JUnit reports: [core](../simulation/artifacts/jerry/software-validation/core.xml), [AURA](../simulation/artifacts/jerry/software-validation/aura.xml), [depth/navigation](../simulation/artifacts/jerry/software-validation/depth.xml).

![Planned and actual trajectory](../simulation/artifacts/jerry/build-validation/trajectory.png)

Ground-truth Gazebo poses were used only by the evaluator; navigation used camera, four ranges, IMU and compass. The independent position check also verified a reported frontier arrival against actual Gazebo pose: 11.3 cm from its goal.

## Fixes uncovered by the longer tests

- Preserve upstream command expiry at the adapter using a boot-identified local monotonic deadline, eliminating gaps between independently phased token windows.
- Retry lost read-only startup status requests within a deadline; revoke/reconcile uncertain authority responses.
- Isolate YOLO in a child process so inference cannot hold navigation's interpreter.
- Count measured goal-distance and heading improvement as mission progress, including travel through already observed space.
- Align in place before requiring a translational lookahead; unsafe translation remains held.
- Publish current admission generations after repeated Stop, ignore obsolete stop feedback, and clear previous-session terminal authority on reset.
- Reduce raw pose-noise accumulation in distance budgets and report motion only when an authorized command is actually forwarded.
- Add bounded task waiting so Codex can observe measured progress through MCP without shell timing or rapid polling.

The evaluator's goal-transition bookkeeping was separately regression-tested so a confirmed arrival followed by a new goal is accepted, while replacing an unfinished goal is detected. Earlier failed runs remain in the artifact directory.

## Reproduce

```bash
simulation/validate_jerry.sh --exploration-wall 90 --faults
simulation/validate_codex.sh
PYTHONPATH=. .venv-jerry/bin/python simulation/tests/check_speech_roundtrip.py
```

Codex is the high-level task client; deterministic navigation and motion supervision execute the route. This validation covers the current flat-floor lab profile. Object-relative metric navigation, physical hardware and multi-world endurance qualification remain outside this build's validated scope.
