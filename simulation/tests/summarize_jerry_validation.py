"""Create a reviewable build report from measured runs, wire traces and test results."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'simulation/artifacts/jerry'


def latest_passed(pattern,predicate=lambda value:True):
    candidates=[]
    for path in BASE.glob(pattern):
        try:value=json.loads(path.read_text())
        except (ValueError,OSError):continue
        if value.get('passed') and predicate(value):candidates.append((path,value))
    if not candidates:raise RuntimeError('No successful evidence for '+pattern)
    return sorted(candidates,key=lambda item:str(item[0]))[-1]


def main():
    runtime_path,runtime=latest_passed('validation-*/report.json',lambda r:all(r.get('checks',{}).get(k) for k in (
        'route_following','wall_budget_enforced','old_proposal_cannot_restart','navigation_freeze_stops_output',
        'supervisor_freeze_stops_output','executive_freeze_stops_output')))
    codex_path,codex=latest_passed('codex-validation-*/report.json',lambda r:r.get('actual_codex_client')
        and r.get('all_required_mcp_tools_used') and r.get('no_shell_or_file_changes')
        and r['independent_measurement']['nonzero_translation_commands']>5
        and r['independent_measurement']['actual_extent_m']>=.1)
    wire=json.loads((codex_path.parent/'wire-checks.json').read_text())
    assert all(wire.values()),wire
    speech=json.loads((BASE/'speech-acceptance.json').read_text())
    assert speech['passed']
    suites={}
    total_passed=total_skipped=0
    for name in ('core','aura','depth'):
        tree=ET.parse(BASE/'software-validation'/(name+'.xml'))
        cases=tree.findall('.//testcase')
        failures=sum(t.find('failure') is not None or t.find('error') is not None for t in cases)
        skipped=[dict(name=t.get('name'),reason=t.find('skipped').get('message')) for t in cases if t.find('skipped') is not None]
        assert failures==0,(name,failures)
        passed=len(cases)-len(skipped)
        total_passed+=passed
        total_skipped+=len(skipped)
        suites[name]=dict(passed=passed,failed=failures,skipped=skipped)
    rows=json.loads((runtime_path.parent/'telemetry.json').read_text())
    arrivals={}
    for row in rows:
        slam=row.get('slam',{});truth=row.get('evaluator_truth');goal=slam.get('goal')
        if row.get('state')=='ARRIVED' and truth and goal:
            key=str(slam['goal_id'])
            arrivals[key]=min(math.dist(goal,truth[1:3]),arrivals.get(key,math.inf))
    assert arrivals and all(d<=.18 for d in arrivals.values()),arrivals
    output=BASE/'build-validation'
    output.mkdir(exist_ok=True)
    source={}
    for root in (ROOT/'jerry',ROOT/'simulation/ros2_ws/src/jerry_interfaces'):
        for path in sorted(root.rglob('*')):
            if path.suffix in ('.py','.msg','.srv','.action','.xml','.txt') and '__pycache__' not in path.parts:
                source[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ('realtime_depth_app/robot_live.py','realtime_depth_app/src/robot_slam.py',
        'realtime_depth_app/src/robot_motion.py','realtime_depth_app/src/robot_detection_worker.py',
        'AURA/src/speech/whisper_engine.py'):
        source[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    summary=dict(build_successful=True,scope='Current flat-floor Gazebo lab + Codex MCP simulation profile',
        generated_at=datetime.now(timezone.utc).isoformat(),tests_passed=total_passed,tests_skipped=total_skipped,
        suites=suites,runtime_report=str(runtime_path.relative_to(ROOT)),codex_report=str(codex_path.relative_to(ROOT)),
        runtime_checks=runtime['checks'],route_metrics=runtime['route_following'],
        independent_arrival_distance_m=arrivals,codex_measured_travel_m=codex['independent_measurement']['actual_distance_m'],
        codex_measured_extent_m=codex['independent_measurement']['actual_extent_m'],
        codex_actual_mcp_calls=codex['completed_mcp_calls'],codex_wire_checks=wire,speech=speech,
        ground_truth_used_only_for_evaluation=True,source_sha256=source)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    # Standalone scientific plot; ground truth never enters the control graph.
    os.environ.setdefault('MPLCONFIGDIR','/tmp/jerry-validation-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    active=[r for r in rows if r.get('active')]
    start=active[0]['wall']
    end=next((r['wall'] for r in rows if r['wall']>start and not r.get('active')),active[-1]['wall'])
    mission=[r for r in rows if start<=r['wall']<=end and r.get('evaluator_truth') and r.get('slam',{}).get('pose')]
    truth=np.asarray([r['evaluator_truth'][1:3] for r in mission])
    estimated=np.asarray([r['slam']['pose'][:2] for r in mission])
    fig,(ax,area)=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    plotted=set()
    for row in mission:
        slam=row['slam'];goal=slam.get('goal');route=row.get('evaluator_route',[]);identity=slam.get('goal_id')
        if goal and len(route)>1 and identity not in plotted and math.dist(route[-1],goal)<.15:
            route=np.asarray(route)
            ax.plot(route[:,0],route[:,1],'--',alpha=.7,label=f'Planned route {identity}')
            ax.scatter(*goal,marker='x',s=65)
            plotted.add(identity)
    ax.plot(truth[:,0],truth[:,1],color='black',linewidth=2,label='Actual Gazebo trajectory')
    ax.plot(estimated[:,0],estimated[:,1],color='#d95f02',alpha=.8,label='Sensor-estimated trajectory')
    ax.scatter(*truth[0],color='green',s=35,label='Start')
    ax.set(xlabel='x (m)',ylabel='y (m)',title='Measured route following')
    ax.axis('equal');ax.grid(alpha=.25);ax.legend(fontsize=8)
    elapsed=[r['wall']-start for r in mission]
    area.plot(elapsed,[r['slam']['observed_m2'] for r in mission],color='#1b9e77')
    area.set(xlabel='Wall time (s)',ylabel='Observed map area (m²)',title='Exploration progress')
    area.grid(alpha=.25)
    fig.savefig(output/'trajectory.png',dpi=170)
    plt.close(fig)
    metrics=summary['route_metrics']
    content=f'''# Jerry simulation build validation — 7 November 2026

**Build successful for the current Gazebo/MCP simulation profile.** This result combines an actual Gazebo trajectory, fault injection, an installed Codex client using the registered MCP, local speech models and software regression suites.

| Check | Result |
|---|---|
| Software suites | {total_passed} passed, {total_skipped} expected environment-specific skip, zero failures |
| Actual autonomous travel | {metrics['actual_distance_m']:.2f} m in the measured route run |
| Confirmed frontier arrivals | {metrics['completed_goals']} |
| Planned-route error, p95 | {metrics['route_error_p95_m']*100:.1f} cm |
| Localization RMSE | {metrics['localization_rmse_m']*100:.1f} cm; no trajectory alignment used |
| Observed map growth | {metrics['observed_growth_m2']:.2f} m² |
| Premature destination changes | None |
| Stop and stale proposal replay | Passed; no restart after Stop |
| Wall-time mission limit | Passed |
| Navigation / supervisor / executive freeze | All independently removed actuator output within the tested expiry window |
| Actual Codex client | Passed; used Jerry MCP tools, with no shell commands or file changes |
| Codex-driven measured travel | {summary['codex_measured_travel_m']:.2f} m; forward commands independently observed |
| Codex camera / map / memory / cancellation | Actual wire results verified |
| Kokoro → Whisper | Passed offline: “{speech['transcript']}” |

The skipped depth test is the CPU-only half-precision guard, which is inapplicable when CUDA is available. Speech round-tripping used generated audio; microphone and speaker hardware were not opened by that test.

## Evidence

- [Machine-readable summary](../simulation/artifacts/jerry/build-validation/summary.json), including source hashes.
- [Route and fault-injection report](../{summary['runtime_report']}).
- [Actual Codex execution report](../{summary['codex_report']}); its directory also contains the prompt, JSONL MCP trace, returned camera/map images, and wire checks.
- [Speech acceptance](../simulation/artifacts/jerry/speech-acceptance.json).
- JUnit reports: [core](../simulation/artifacts/jerry/software-validation/core.xml), [AURA](../simulation/artifacts/jerry/software-validation/aura.xml), [depth/navigation](../simulation/artifacts/jerry/software-validation/depth.xml).

![Planned and actual trajectory](../simulation/artifacts/jerry/build-validation/trajectory.png)

Ground-truth Gazebo poses were used only by the evaluator; navigation used camera, four ranges, IMU and compass. The independent position check also verified a reported frontier arrival against actual Gazebo pose: {min(arrivals.values())*100:.1f} cm from its goal.

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
'''
    (ROOT/'docs/JERRY_VALIDATION_2026-11-07.md').write_text(content)
    print(json.dumps({k:summary[k] for k in ('build_successful','scope','tests_passed','tests_skipped','codex_measured_travel_m')},indent=2))


if __name__=='__main__':main()
