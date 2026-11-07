import math
from jerry.progress import MissionProgress


def snapshot(x=0.,yaw=0.,ready=True):
    return dict(goal=[1.,0.],pose=[x,0.,yaw],goal_id=1,map_revision=0,
                ready=ready,observed_m2=2.,completed_goals=0)


def test_valid_travel_through_known_space_keeps_mission_alive():
    progress=MissionProgress(0.)
    for second in range(0,61):
        assert progress.update(snapshot(x=second*.01),second)


def test_heading_alignment_is_measured_progress():
    progress=MissionProgress(0.)
    for second in range(0,31):
        assert progress.update(snapshot(yaw=math.pi-second*.05),second)


def test_stationary_and_oscillating_estimates_do_not_reset_progress_forever():
    progress=MissionProgress(0.)
    assert progress.update(snapshot(),0.)
    for second in range(1,21):
        assert progress.update(snapshot(x=.005 if second%2 else 0.),second)
    assert not progress.update(snapshot(),21.)


def test_untrusted_pose_cannot_supply_progress():
    progress=MissionProgress(0.)
    assert progress.update(snapshot(),0.)
    assert not progress.update(snapshot(x=.5,ready=False),21.)
