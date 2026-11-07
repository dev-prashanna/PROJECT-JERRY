from types import SimpleNamespace
import pytest
pytest.importorskip('rclpy')
pytest.importorskip('jerry_interfaces.msg')
from rclpy.task import Future
from jerry_interfaces.srv import Authority
from jerry.supervisor import Supervisor
from jerry.authority import AuthorityGate


def test_lost_startup_status_request_is_retried_with_a_bound(monkeypatch):
    monkeypatch.setattr('jerry.supervisor.time.monotonic',lambda:1.)
    old=Future()
    new=Future()
    sent=[]
    fake=SimpleNamespace(pending=old,pending_started=0.,pending_operation='STATUS',session='',epoch=0,
        gate=AuthorityGate(clock=lambda:1.),backend_client=SimpleNamespace(service_is_ready=lambda:True,
        call_async=lambda request:(sent.append(request) or new)))
    Supervisor.backend_tick(fake)
    assert old.cancelled()
    assert sent[-1].operation=='STATUS'
    assert fake.pending is new
    new.set_result(Authority.Response(accepted=True,session_id='fresh-session',epoch_id=2))
    assert fake.session=='fresh-session' and fake.epoch==2


def test_uncertain_actuator_commit_is_revoked_instead_of_retried(monkeypatch):
    monkeypatch.setattr('jerry.supervisor.time.monotonic',lambda:1.)
    old=Future()
    stopped=[]
    fake=SimpleNamespace(pending=old,pending_started=0.,pending_operation='CLAIM',stop=stopped.append)
    Supervisor.backend_tick(fake)
    assert old.cancelled() and fake.pending is None
    assert stopped and 'reconciliation' in stopped[0]
