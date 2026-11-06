"""Unit check: pausing simulation time cannot keep an old command alive."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace


def test_command_expires_in_wall_time_even_if_simulation_clock_is_frozen(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'ros2_ws/src/vbmd_description/scripts/gazebo_adapter.py'
    spec=importlib.util.spec_from_file_location('vbmd_adapter_watchdog_test',path)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    command=adapter.Twist();command.linear.x=.1
    sent=[]
    fake=SimpleNamespace(get_parameter=lambda _:SimpleNamespace(value=False),
        seconds=lambda:10.,last_command=10.,last_command_wall=20.,command=command,
        output=SimpleNamespace(publish=sent.append))
    monkeypatch.setattr(adapter.time,'monotonic',lambda:20.1)
    adapter.GazeboAdapter.drive(fake)
    assert sent[-1].linear.x==.1
    monkeypatch.setattr(adapter.time,'monotonic',lambda:20.6)
    adapter.GazeboAdapter.drive(fake)
    assert sent[-1].linear.x==sent[-1].angular.z==0.


def test_beam_does_not_turn_too_close_or_invalid_returns_into_no_echo():
    path=Path(__file__).resolve().parents[1]/'ros2_ws/src/vbmd_description/scripts/gazebo_adapter.py'
    spec=importlib.util.spec_from_file_location('vbmd_beam_test',path)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    measure=adapter.beam_minimum
    assert measure([math.inf]*9,.02,4.)==math.inf
    assert measure([.5,.3,math.inf],.02,4.)==.3
    assert measure([math.inf,-math.inf],.02,4.)==-math.inf
    assert measure([.01,.4],.02,4.)==-math.inf
    assert math.isnan(measure([math.nan,math.inf],.02,4.))
    assert math.isnan(measure([],.02,4.))
