from dataclasses import asdict
from types import SimpleNamespace
import pytest
pytest.importorskip('rclpy')
pytest.importorskip('jerry_interfaces.msg')
from jerry.adapter import ProtectedActuator
from jerry.authority import AuthorityGate


def test_wrong_boot_identity_and_delayed_first_packet_never_actuate():
    now=[1.]
    gate=AuthorityGate(clock=lambda:now[0])
    binding=gate.acquire('s',0,'supervisor','t')
    token,_=gate.issue(binding)
    command=SimpleNamespace(**asdict(binding),sequence=1,token=token,mode='TRACK',linear_m_s=.05,
        angular_rad_s=0.,clock_id='other-boot',not_after_monotonic_s=1.05)
    fake=SimpleNamespace(gate=gate,clock_id='local-boot',command=(0.,0.,0.))
    ProtectedActuator.receive(fake,command)
    assert fake.command==(0.,0.,0.)
    command.clock_id='local-boot'
    now[0]=1.06
    ProtectedActuator.receive(fake,command)
    assert fake.command==(0.,0.,0.)
