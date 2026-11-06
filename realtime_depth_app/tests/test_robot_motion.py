import pytest
from src.robot_avoidance import Decision
from src.robot_motion import CommandSmoother


def test_command_acceleration_and_steering_changes_are_bounded():
    smoother=CommandSmoother();previous=(0.,0.)
    for i in range(1,101):
        target=Decision(.065,.35 if i<50 else -.35,'EXPLORE','fixed destination')
        d=smoother.apply(target,i*.05)
        assert abs(d.linear-previous[0])<=.12*.05+1e-9
        assert abs(d.angular-previous[1])<=.60*.05+1e-9
        previous=(d.linear,d.angular)
    assert abs(previous[0]-.065)<1e-6


@pytest.mark.parametrize('state',['STOP','WAITING','PLANNING','ARRIVING','ARRIVED','RECOVERING'])
def test_safety_and_arrival_stops_are_immediate(state):
    smoother=CommandSmoother()
    for i in range(30):smoother.apply(Decision(.065,.3,'EXPLORE','travel'),i*.05)
    d=smoother.apply(Decision(0.,0.,state,'hold'),1.5)
    assert d.linear==d.angular==0.


def test_reversing_brakes_through_zero_and_reset_never_reuses_motion():
    smoother=CommandSmoother()
    for i in range(30):smoother.apply(Decision(.065,0.,'EXPLORE','travel'),i*.05)
    d=smoother.apply(Decision(-.035,0.,'BACK_AWAY','recovery'),1.5)
    assert 0.<d.linear<.065
    smoother.reset()
    d=smoother.apply(Decision(.065,0.,'EXPLORE','resume'),2.)
    assert 0.<d.linear<=.00301 and d.angular==0.


def test_visual_dropout_bridge_cannot_exceed_its_speed_cap():
    smoother=CommandSmoother()
    for i in range(40):smoother.apply(Decision(.065,0.,'EXPLORE','travel'),i*.05)
    d=smoother.apply(Decision(.02,0.,'TRACKING_BRIDGE','brief dropout'),2.)
    assert 0<d.linear<=.02
