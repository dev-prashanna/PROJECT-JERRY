"""Hardware-free evidence for the November 2026 project audit.

Run: python3 audit/reproduce_findings.py
These probes assert the observed defects, not the desired fixed behavior.
No camera, microphone, network service, model weights, or actuator is used.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def navigation():
    import math
    import time
    import numpy as np
    from src.robot_avoidance import Decision
    from src.robot_motion import CommandSmoother

    # Execute the actual adapter class without importing/starting ROS.
    source = ROOT / 'simulation/ros2_ws/src/vbmd_description/scripts/gazebo_adapter.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GazeboAdapter')
    namespace = dict(Node=object, math=math, time=time,
                     SIDES=('front', 'left', 'rear', 'right'))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    adapter = namespace['GazeboAdapter']
    now = time.monotonic()
    fake = NS(seconds=lambda: 10., cache={s: NS(header=NS(stamp=NS(sec=10, nanosec=0)))
              for s in namespace['SIDES']}, last_scan_wall={s: now for s in namespace['SIDES']},
              ranges=dict(front=math.nan, left=1., rear=1., right=1.),
              avoid_direction=0., avoid_start=0.)
    invalid_command = adapter.reactive(fake)
    assert invalid_command == (.08, 0.)
    fake.ranges = dict(front=.2, left=-math.inf, rear=-math.inf, right=-math.inf)
    pivot_command = adapter.reactive(fake)
    assert pivot_command[1] != 0.

    drift = {}
    for state in ['ALIGN', 'RESCAN', 'LOOK_AROUND']:
        smoother = CommandSmoother()
        smoother.linear = .065
        smoother.last_time = 1.
        result = smoother.apply(Decision(0., .2, state, 'stationary pivot'), 1.05)
        drift[state] = result.linear
        assert result.linear > .05

    # Exercise the actual precision conversion without loading a model.
    import torch
    from src.depth_engine import DepthEngine
    engine = DepthEngine.__new__(DepthEngine)
    engine.device = torch.device('cpu')
    engine._dtype = torch.bfloat16
    engine._use_autocast = False
    actual_dtype = str(engine._to_device(torch.ones(1)).dtype)
    assert actual_dtype == 'torch.float16'

    from src.config import load_config
    app = load_file('audit_depth_main', ROOT / 'realtime_depth_app/main.py')
    from src.geometry import CameraIntrinsics
    cfg = load_config()
    with tempfile.TemporaryDirectory() as directory:
        intrinsics = Path(directory) / 'intrinsics.json'
        intrinsics.write_text(json.dumps(CameraIntrinsics(100.,100.,32.,24.,64,48,
                                                          dist=[.2,0.,0.,0.,0.]).to_dict()))
        cfg.geometry.intrinsics = str(intrinsics)
        cfg.geometry.undistort = True
        app.DepthEngine = lambda _: NS(device='cpu', estimation_type='relative')
        app.Pipeline = lambda *args, **kwargs: NS()
        instance = app.DepthApp(cfg)
        assert instance.intrinsics is not None and instance.undistorter is None

    calibration = load_file('audit_calibrate', ROOT / 'realtime_depth_app/tools/calibrate.py')
    try:
        calibration.run_check(NS(resolve=lambda _: Path('/tmp/unused-audit-profile.json')),
                              NS(out='unused'))
    except NameError as exc:
        calibration_error = str(exc)
        assert 'load_profile' in calibration_error
    else:
        raise AssertionError('expected missing load_profile import')

    failed_app = app.DepthApp.__new__(app.DepthApp)
    stopped = []
    failed_app.pipeline = NS(start=lambda: None,
        depth_slot=NS(wait_new=lambda *args, **kwargs: (0, None)),
        inference=NS(error='audit: forced inference failure'),
        stop=lambda: stopped.append(True))
    assert failed_app.run_headless(1, timeout_s=.1) == 2
    assert not stopped

    from src.camera import Frame, LatestSlot
    from src.pipeline import InferenceWorker, Pipeline
    from src.temporal_filter import DepthSmoother
    entered_infer = threading.Event()
    finish_infer = threading.Event()

    def blocked_infer(*args, **kwargs):
        entered_infer.set()
        finish_infer.wait(5.)
        return NS(depth=np.ones((2,2)))

    frames, depths = LatestSlot(), LatestSlot()
    inference = InferenceWorker(NS(infer=blocked_infer), frames, depths, DepthSmoother(False,.5,1))
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.capture = NS(stop=lambda: None, is_alive=lambda: False)
    pipeline.inference = inference
    pipeline.source = NS(release=lambda: None)
    pipeline._started = True
    inference.start()
    try:
        frames.put(Frame(1,np.zeros((2,2,3),np.uint8),time.time()))
        assert entered_infer.wait(2.)
        pipeline.stop()
        returned_with_live_inference = inference.is_alive()
        assert returned_with_live_inference and not pipeline.started
    finally:
        finish_infer.set()
        inference.join(timeout=2.)

    print(json.dumps(dict(invalid_front_command=invalid_command,
        blocked_pivot_command=pivot_command, stationary_states_linear_m_s=drift,
        requested_precision='bf16', actual_tensor_dtype=actual_dtype,
        enabled_undistorter=instance.undistorter, calibration_check_error=calibration_error,
        headless_error_called_pipeline_stop=bool(stopped),
        pipeline_stop_returned_with_live_inference=returned_with_live_inference)))


def aura():
    os.environ['MEMORY_ENABLED'] = '0'
    os.environ['GROQ_API_KEY'] = 'audit-unused-placeholder'
    os.environ['GEMINI_API_KEY'] = ''
    from src.memory.resource_scheduler import MemoryJobScheduler
    scheduler = MemoryJobScheduler(lambda job: None)
    scheduler.submit('pending-a')
    scheduler.submit('pending-b')
    try:
        scheduler.stop()
    except TypeError as exc:
        stop_error = str(exc)
    else:
        raise AssertionError('expected PriorityQueue sentinel comparison error')

    from src.orchestrator.tool_registry import Tool, ToolRegistry, EmptyArgs
    from src.orchestrator.schemas import ToolResult
    registry = ToolRegistry()
    release = threading.Event()
    committed = threading.Event()

    def handler():
        release.wait(2.)
        committed.set()
        return ToolResult(tool='late-write', text='committed')

    registry.register(Tool('late-write', 'audit', EmptyArgs, handler, timeout_s=.02))
    result = registry.execute('late-write')
    assert not result.ok and not committed.is_set()
    release.set()
    assert committed.wait(2.)

    # Reuse the project's existing hardware-free test doubles.
    fixtures = load_file('audit_aura_fixtures', ROOT / 'AURA/tests/conftest.py')
    agent = fixtures.make_agent()
    entered = threading.Event()
    continue_route = threading.Event()
    original_route = agent._route

    def blocked_route(*args):
        entered.set()
        continue_route.wait(3.)
        return original_route(*args)

    agent._route = blocked_route
    agent.start(warm_models=False)
    try:
        job = agent.submit_text('hello')
        assert entered.wait(2.)
        assert agent.scheduler.cancel_current()
        continue_route.set()
        assert job.done.wait(3.)
        assert job.cancelled and job.result.response and agent.player.played
        spoken_after_cancel = len(agent.player.played)
    finally:
        continue_route.set()
        agent.shutdown()

    print(json.dumps(dict(memory_stop_error=stop_error,
        timed_out_tool_committed_later=committed.is_set(),
        cancelled_request_spoken_chunks=spoken_after_cancel)))


def firmware():
    # Compile the unchanged motor firmware with inert Arduino/UDP stubs.
    # Only the protocol and watchdog logic run; no device is connected.
    header = r'''
#pragma once
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#define OUTPUT 1
#define HIGH 1
#define LOW 0
inline unsigned long audit_now=0;
inline unsigned long millis() { return audit_now; }
inline void analogWrite(int,int) {}
inline void analogWriteResolution(int) {}
inline void analogWriteFrequency(int) {}
inline void digitalWrite(int,int) {}
inline void pinMode(int,int) {}
inline int constrain(int x,int lo,int hi) { return std::clamp(x,lo,hi); }
struct AuditSerial {
  void begin(int) {}
  template<class T> void println(T) {}
  template<class T> void print(T) {}
  template<class... T> void printf(const char*,T...) {}
};
inline AuditSerial Serial;
struct AuditWiFi {
  void softAP(const char*,const char*) {}
  int softAPIP() { return 0; }
};
inline AuditWiFi WiFi;
'''
    udp_header = r'''
#pragma once
#include "WiFi.h"
struct WiFiUDP {
  std::string incoming;
  void begin(unsigned int) {}
  int parsePacket() { return incoming.size(); }
  int read(char* buffer,unsigned int capacity) {
    int size=std::min(capacity,static_cast<unsigned int>(incoming.size()));
    std::memcpy(buffer,incoming.data(),size); incoming.clear(); return size;
  }
};
'''
    source = ROOT / 'esp32-motor-driver/src/main.cpp'
    harness = '#include <cassert>\n#include <iostream>\n#include ' + json.dumps(str(source)) + r'''
int main() {
  setup(); audit_now=1; udp.incoming="F"; loop();
  for(int i=0;i<10;i++) { audit_now+=1000; udp.incoming="?"; loop(); }
  assert(lastAction=='F');
  std::cout << "{\"last_action_after_10_seconds_of_invalid_packets\":\"" << lastAction << "\"}";
  audit_now+=2001; loop(); assert(lastAction=='S');
}
'''
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        (folder/'WiFi.h').write_text(header)
        (folder/'WiFiUdp.h').write_text(udp_header)
        (folder/'probe.cpp').write_text(harness)
        subprocess.run(['g++', '-std=c++17', '-I', directory, str(folder/'probe.cpp'),
                        '-o', str(folder/'probe')], check=True, capture_output=True, text=True)
        result = subprocess.run([str(folder/'probe')], check=True, capture_output=True, text=True)
        print(result.stdout)


def vqa():
    import time
    import numpy as np
    from src.camera.camera import CameraSource, CameraStatus
    from src.pipeline.vqa_pipeline import VQAPipeline
    camera = CameraSource()
    camera.buffer.update(np.zeros((24,32,3), np.uint8), timestamp=time.time()-3600.)
    camera._status = CameraStatus.DISCONNECTED
    completed = threading.Event()
    engine = NS(is_loaded=True, ask=lambda *args, **kwargs: {'answer': 'old image'},
                clear_image_cache=lambda: None)
    pipeline = VQAPipeline(camera=camera, engine=engine,
        on_event=lambda kind, payload: completed.set() if kind == 'inference_result' else None)
    pipeline._model_state = 'loaded'
    try:
        accepted, reason = pipeline.ask('What do you see now?')
        assert accepted and completed.wait(2.)
    finally:
        pipeline.stop(timeout=1.)
    print(json.dumps(dict(disconnected_camera_frame_age_s=3600,
                          live_question_accepted=accepted, inference_completed=completed.is_set())))


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        folder = {'aura': 'AURA', 'vqa': 'moondream-live-vqa'}.get(mode, 'realtime_depth_app')
        sys.path.insert(0, str(ROOT / folder))
        return {'aura': aura, 'navigation': navigation, 'firmware': firmware, 'vqa': vqa}[mode]()
    for mode in ['navigation', 'aura', 'firmware', 'vqa']:
        run = subprocess.run([sys.executable, str(Path(__file__).resolve()), mode],
                             cwd=ROOT, capture_output=True, text=True, timeout=45.)
        print(f'{mode}: {run.stdout.strip()}')
        if run.returncode:
            print(run.stderr, file=sys.stderr)
            raise SystemExit(run.returncode)


if __name__ == '__main__':
    main()
