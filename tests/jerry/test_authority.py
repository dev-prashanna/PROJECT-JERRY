from dataclasses import asdict
from types import SimpleNamespace
import math
import pytest
from jerry.authority import AuthorityGate,collision_gate


def fixture():
    now = [1.]
    gate = AuthorityGate(clock=lambda:now[0])
    binding = gate.acquire('session',0,'navigation','task')
    token,_ = gate.issue(binding)
    command = SimpleNamespace(**asdict(binding),token=token,sequence=1,linear_m_s=.05,angular_rad_s=0.,mode='TRACK')
    return gate,now,binding,command


def test_first_delivery_expires_even_with_a_live_renewed_lease():
    gate,now,binding,command = fixture()
    now[0] += .16
    gate.renew(binding)
    with pytest.raises(ValueError,match='token'):
        gate.accept(command)


def test_old_tokens_and_commands_cannot_restart_a_stopped_task():
    gate,now,binding,command = fixture()
    gate.revoke()
    gate.acquire('session',0,'navigation','task')
    with pytest.raises(ValueError):gate.accept(command)


def test_duplicate_and_wrong_geometry_are_rejected():
    gate,now,binding,command = fixture()
    assert gate.accept(command)[:2] == (.05,0.)
    with pytest.raises(ValueError):gate.accept(command)
    command.sequence += 1
    command.map_revision += 1
    with pytest.raises(ValueError):gate.accept(command)


@pytest.mark.parametrize('v,w,mode',[(math.nan,0.,'TRACK'),(0.,math.inf,'PIVOT'),(.05,.2,'PIVOT'),(.05,0.,'HOLD'),(-.02,0.,'TRACK'),(.09,0.,'TRACK')])
def test_bad_or_unconstrained_commands_rejected(v,w,mode):
    gate,_,_,command = fixture()
    command.linear_m_s,command.angular_rad_s,command.mode = v,w,mode
    with pytest.raises(ValueError):gate.accept(command)


def test_token_history_is_bounded_and_receipt_does_not_extend_it():
    gate,now,binding,command = fixture()
    for _ in range(100):gate.issue(binding)
    assert len(gate.tokens)<=8
    command.token,_ = gate.issue(binding)
    _,_,deadline = gate.accept(command)
    now[0] += .1
    command.sequence += 1
    assert gate.accept(command)[2] == deadline


@pytest.mark.parametrize('side,value',[('front',math.nan),('rear',-math.inf),('left',.01),('right',5.)])
def test_invalid_range_blocks_all_motion(side,value):
    ranges={s:(1.,1.,10.,4.) for s in ('front','left','rear','right')}
    ranges[side]=(value,1.,10.,4.)
    assert collision_gate(.05,.1,ranges,1.1,10.1)[:2] == (0.,0.)


def test_no_echo_does_not_override_a_blocked_pivot_corner():
    ranges={s:(math.inf,1.,10.,4.) for s in ('front','left','rear','right')}
    ranges['rear']=(.12,1.,10.,4.)
    assert collision_gate(0.,.2,ranges,1.1,10.1)[:2] == (0.,0.)


def test_downstream_token_uses_remaining_upstream_budget():
    gate,now,binding,command=fixture()
    command.token,duration=gate.issue(binding,maximum_validity_s=.06)
    assert duration==pytest.approx(.06)
    assert gate.accept(command)[2]==pytest.approx(1.06)
    now[0]=1.061
    gate.renew(binding)
    with pytest.raises(ValueError,match='token'):gate.accept(command)


@pytest.mark.parametrize('duration',[0.,-.1,math.nan,math.inf])
def test_invalid_downstream_budget_never_issues_authority(duration):
    gate,_,binding,_=fixture()
    with pytest.raises(ValueError):gate.issue(binding,maximum_validity_s=duration)


def test_same_host_deadline_never_restarts_validity_at_adapter_receipt():
    gate,now,binding,command=fixture()
    _,_,expiry=gate.accept(command,maximum_deadline=1.04)
    assert expiry==pytest.approx(1.04)
    now[0]=1.05
    command.sequence+=1
    with pytest.raises(ValueError):gate.accept(command,maximum_deadline=1.04)


def test_forwarded_deadline_cannot_extend_receiver_token():
    gate,now,binding,command=fixture()
    assert gate.accept(command,maximum_deadline=1000.)[2]==pytest.approx(1.15)
