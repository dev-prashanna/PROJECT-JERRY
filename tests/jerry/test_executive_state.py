from collections import deque
import threading
from types import SimpleNamespace
import pytest
pytest.importorskip('jerry_interfaces.msg')
from jerry_interfaces.msg import TaskState
from jerry.executive import Executive
from jerry.authority import Binding


def test_repeated_stop_publishes_current_admission_generation():
    published=[]
    terminal=TaskState(session_id='s',generation=1,task_id='t',status='CANCELLED')
    fake=SimpleNamespace(lock=threading.RLock(),journal_future=None,journal_pending=deque(),
        outage_since=None,task=None,completed={'t':terminal},generation=3,pub=SimpleNamespace(publish=published.append))
    Executive.tick(fake)
    assert published[-1].generation==3
    assert terminal.generation==1  # historical terminal record stays immutable


def test_old_stop_state_cannot_cancel_newly_acquired_task():
    finished=[]
    fake=SimpleNamespace(lock=threading.RLock(),session='s',epoch=0,generation=1,task={'id':'new'},
        binding=Binding('s',0,'navigation','new',7,0),finish=lambda *args:finished.append(args),completed={})
    Executive.motion_state(fake,TaskState(session_id='s',epoch_id=0,generation=6,status='STOPPED'))
    assert not finished
    Executive.motion_state(fake,TaskState(session_id='s',epoch_id=0,generation=8,status='STOPPED'))
    assert finished and finished[-1][0]=='CANCELLED'


def test_reset_does_not_republish_previous_sessions_terminal_authority():
    fake=SimpleNamespace(lock=threading.RLock(),session='old',epoch=0,generation=1,task=None,
        finish=lambda *args:None,completed={'old':TaskState(session_id='old',status='CANCELLED')})
    Executive.motion_state(fake,TaskState(session_id='new',epoch_id=1,generation=0,status='STOPPED'))
    assert fake.completed=={}
    assert fake.session=='new' and fake.epoch==1
