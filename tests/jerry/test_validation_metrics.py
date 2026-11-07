from collections import deque
from types import SimpleNamespace
import threading
import pytest
pytest.importorskip('rclpy')
pytest.importorskip('gz.transport13')
from simulation.tests.validate_jerry_runtime import Evaluator


def row(goal_id,completed):
    return dict(wall=1.,state='EXPLORE',reason='',evaluator_route=[],evaluator_truth=None,
                slam=dict(goal=[float(goal_id),0.],goal_id=goal_id,completed_goals=completed,loop_closures=0))


def test_confirmed_arrival_then_goal_change_is_not_premature():
    fake=SimpleNamespace(lock=threading.RLock(),commands=deque(),rows=[row(1,0),row(1,1),row(2,1)])
    result=Evaluator.metrics(fake,0.)
    assert result['premature_goal_changes']==[]


def test_goal_change_without_arrival_is_detected():
    fake=SimpleNamespace(lock=threading.RLock(),commands=deque(),rows=[row(1,0),row(1,0),row(2,0)])
    assert Evaluator.metrics(fake,0.)['premature_goal_changes']
