import math
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage

from realtime_depth_app.src.robot_slam import (Calibration, OccupancyMap, FrontierPlanner, CuriosityPolicy,
                            FloorSlam, SlamWorker,consensus_translation, rotation,route_target,PathFollower,localization_health,simplify_route,segment_clear)
from realtime_depth_app.src.robot_avoidance import Decision, RangeReading, SIDES


def test_idle_sensor_repeats_leave_capacity_for_a_later_mapping_scan():
    slam=FloorSlam()
    for tick in range(21000):
        stamp=tick*.1
        slam.integrate_ranges(stamp,{'front':(1.,4.,stamp)})
    assert len(slam.scans)==16
    assert not slam.snapshot()['capacity_reached']
    observed=slam.map.observed.sum()
    slam.pose[2]=math.pi/2
    slam.integrate_ranges(2101.,{'front':(1.,4.,2101.)})
    assert len(slam.scans)==17
    assert slam.map.observed.sum()>observed
    # A moved obstacle must also update the map while the robot stays still.
    slam.integrate_ranges(2102.,{'front':(.4,4.,2102.)})
    assert len(slam.scans)==18
    slam.pose[0]+=.03
    slam.integrate_ranges(2103.,{'front':(.4,4.,2103.)})
    assert len(slam.scans)==19


def test_exploration_holds_when_map_has_reached_capacity():
    snapshot={'capacity_reached':True,'pose':[0.,0.,0.],
              'goal':None,'path':[],'planner_state':'NO_REACHABLE_FRONTIER'}
    policy=CuriosityPolicy()
    readings={s:RangeReading(1.,0.,0.) for s in SIDES}
    for now in [0.,1.,30.]:
        command=policy.decide(snapshot,readings,Decision(.08,0.,'ADVANCE','clear'),True,now)
        assert command.state=='STOP' and command.linear==command.angular==0.
        assert 'capacity' in command.reason


def test_mapping_scan_keeps_escape_direction_and_stops_if_it_cannot_find_a_goal():
    snapshot={'pose':[0.,0.,0.],'goal':None,'path':[],
              'planner_state':'NO_REACHABLE_FRONTIER'}
    policy=CuriosityPolicy();readings={s:RangeReading(1.,0.,0.) for s in SIDES}
    clear=Decision(.08,0.,'ADVANCE','clear')
    assert policy.decide(snapshot,readings,clear,True,0.).angular>0
    escape=Decision(0.,-.65,'TURN','Avoiding obstacle')
    assert policy.decide(snapshot,readings,escape,True,1.)==escape
    snapshot['pose'][2]=-.3
    assert policy.decide(snapshot,readings,clear,True,2.).angular<0
    held=policy.decide(snapshot,readings,clear,True,46.)
    assert held.state=='STOP' and held.linear==held.angular==0.


def test_corner_scan_turns_into_open_side_instead_of_waiting_for_both_sides():
    snapshot={'pose':[0.,0.,2.45],'goal':None,'path':[],
              'planner_state':'NO_REACHABLE_FRONTIER'}
    readings={side:RangeReading(distance,0.,0.) for side,distance in
              [('front',.52),('left',.14),('right',3.2),('rear',.17)]}
    policy=CuriosityPolicy()
    command=policy.decide(snapshot,readings,Decision(.03,-.35,'ADVANCE','Side caution'),True,0.)
    assert command.state=='LOOK_AROUND' and command.linear==0. and command.angular<0.
    readings['right']=RangeReading(.13,0.,0.)
    held=policy.decide(snapshot,readings,Decision(.03,0.,'ADVANCE','clear'),True,1.)
    assert held.state=='STOP' and held.linear==held.angular==0.


def test_no_goal_corner_can_gain_pivot_clearance_using_measured_short_corridor():
    with np.load(Path(__file__).parent/'fixtures/corner_escape.npz') as captured:
        snapshot={key:captured[key].copy() for key in ('grid','origin','pose')}
        snapshot.update(resolution=float(captured['resolution']),goal=None,path=[],
                        planner_state='NO_REACHABLE_FRONTIER')
    readings={side:RangeReading(distance,0.,0.) for side,distance in
              [('front',.62),('left',.185),('right',1.12),('rear',.115)]}
    policy=CuriosityPolicy();safe=Decision(.04,0.,'ADVANCE','clear')
    command=policy.decide(snapshot,readings,safe,True,0.)
    assert command.state=='GAIN_CLEARANCE' and 0<command.linear<=.025 and command.angular==0.
    assert policy.decide(snapshot,readings,safe,True,4.1).linear==0.
    snapshot['grid'][:]=100
    blocked=CuriosityPolicy().decide(snapshot,readings,safe,True,0.)
    assert blocked.state=='STOP' and blocked.linear==blocked.angular==0.
    snapshot['grid'][:]=-1
    local=CuriosityPolicy().decide(snapshot,readings,safe,True,0.)
    assert local.state=='GAIN_CLEARANCE' and 0<local.linear<=.02
    readings['front']=RangeReading(.62,2.,2.)
    stale=CuriosityPolicy().decide(snapshot,readings,safe,True,0.)
    assert stale.state=='STOP' and stale.linear==stale.angular==0.


def test_small_corner_steps_update_mapping_before_the_recovery_corridor_runs_out():
    with np.load(Path(__file__).parent/'fixtures/corner_escape.npz') as captured:
        engine=FloorSlam();engine.pose=captured['pose'].copy()
        engine.map=OccupancyMap(captured['grid'].shape[0],float(captured['resolution']))
        engine.map.origin=captured['origin'].copy()
        engine.map.observed=captured['grid']>=0
        engine.map.logodds=np.where(captured['grid']==100,3.,-3.)
    initial=engine.pose.copy();policy=CuriosityPolicy()
    for tick in range(100):
        distance=math.dist(engine.pose[:2],initial[:2])
        # Stationary front/rear surfaces as the robot moves straight between them.
        values={'front':.62-distance,'rear':.115+distance,'left':.185,'right':1.12}
        readings={side:RangeReading(value,0.,0.) for side,value in values.items()}
        engine.integrate_ranges(tick*.05,{side:(value,4.,tick*.05) for side,value in values.items()})
        snapshot=engine.snapshot();snapshot.update(goal=None,path=[],planner_state='NO_REACHABLE_FRONTIER')
        command=policy.decide(snapshot,readings,Decision(.04,0.,'ADVANCE','clear'),True,tick*.05)
        engine.pose[:2]+=.05*command.linear*np.array([math.cos(engine.pose[2]),math.sin(engine.pose[2])])
        if command.state=='LOOK_AROUND':break
    assert math.dist(engine.pose[:2],initial[:2])>.02
    assert command.state=='LOOK_AROUND' and command.angular<0.


def test_forward_caution_does_not_reverse_a_clear_in_place_route_alignment():
    snapshot={'pose':[0.,0.,.8],'goal':[-.3,.2],
              'path':[[0.,0.],[-.3,.2]],'planner_state':'EXPLORING'}
    readings={s:RangeReading(.5,0.,0.) for s in SIDES}
    readings['front']=RangeReading(.3,0.,0.)
    caution=Decision(0.,-.65,'TURN','FRONT ultrasound caution')
    command=CuriosityPolicy().decide(snapshot,readings,caution,True,0.)
    assert command.state=='ALIGN' and command.linear==0. and command.angular>0.
    # Emergency clearance and insufficient pivot room still hold the robot.
    emergency=Decision(0.,0.,'STOP','FRONT emergency clearance < 0.18 m')
    held=CuriosityPolicy().decide(snapshot,readings,emergency,True,0.)
    assert held.state in ['STOP','STARTLED'] and held.linear==held.angular==0.
    readings['rear']=RangeReading(.15,0.,0.)
    blocked=CuriosityPolicy().decide(snapshot,readings,caution,True,0.)
    assert blocked.state!='ALIGN' and blocked.linear==0.


def test_rear_alignment_recovery_requires_known_free_space_and_is_bounded():
    occupancy=OccupancyMap(size=60)
    occupancy.observed[:]=True;occupancy.logodds[:]=-3.
    snapshot={'pose':[0.,0.,0.],'goal':[-1.,0.],
              'path':[[0.,0.],[-1.,0.]],'planner_state':'EXPLORING',
              'grid':occupancy.grid(),'origin':occupancy.origin,'resolution':occupancy.resolution}
    readings={s:RangeReading(.6,0.,0.) for s in SIDES}
    readings['rear']=RangeReading(.17,0.,0.)
    safe=Decision(.065,0.,'ADVANCE','clear')
    policy=CuriosityPolicy()
    command=policy.decide(snapshot,readings,safe,True,0.)
    assert command.state=='GAIN_CLEARANCE' and 0<command.linear<=.025 and command.angular==0.
    assert policy.decide(snapshot,readings,safe,True,4.1).linear==0.
    snapshot['pose'][0]=.11
    assert policy.decide(snapshot,readings,safe,True,1.).linear==0.
    snapshot['pose'][0]=0.
    snapshot['grid'][:]=-1
    assert CuriosityPolicy().decide(snapshot,readings,safe,True,0.).linear==0.
    snapshot['grid']=occupancy.grid()
    readings['front']=RangeReading(.3,0.,0.)
    assert CuriosityPolicy().decide(snapshot,readings,safe,True,0.).linear==0.


def test_measured_corner_stall_recovers_without_replacing_destination():
    # Captured ROS map, estimated pose and retained route from the live stall.
    with np.load(Path(__file__).parent/'fixtures/planning_corner.npz') as captured:
        pose=captured['pose'].copy()
        initial=pose.copy()
        goal=captured['goal'].tolist()
        occupancy=OccupancyMap(captured['grid'].shape[0],float(captured['resolution']))
        occupancy.origin=captured['origin'].copy()
        occupancy.observed=captured['grid']>=0
        occupancy.logodds=np.where(captured['grid']==100,3.,-3.)
        navigation={'grid':captured['grid'].copy(),'origin':occupancy.origin,
                    'resolution':occupancy.resolution}
        planner=FrontierPlanner()
        planner.goal=goal.copy();planner.goal_id=3
        planner.path=captured['path'].tolist()
    readings={s:RangeReading(1.,0.,0.) for s in SIDES}
    safety=Decision(.065,0.,'ADVANCE','clear')
    first=planner.navigator.command(pose,planner.path,goal,safety,navigation)
    assert first.state=='EXPLORE' and 0<first.linear<.03
    grid=navigation['grid'];radius=math.ceil(.13/occupancy.resolution)
    yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
    clear=(grid==0)&~ndimage.binary_dilation(grid==100,structure=xx*xx+yy*yy<=radius*radius)
    clear &= ndimage.distance_transform_edt(grid==0)*occupancy.resolution>=.11
    for tick in range(800):
        if tick%10==0:planner.plan(occupancy,pose,tick*.05)
        planner.navigation=navigation
        decision=planner.command(pose,readings,safety,True)
        previous=pose.copy()
        pose[:2]+=.05*decision.linear*np.array([math.cos(pose[2]),math.sin(pose[2])])
        pose[2]+=.05*decision.angular
        assert segment_clear(previous[:2],pose[:2],clear,occupancy.origin,occupancy.resolution)
        assert planner.goal==goal and planner.goal_id==3
        if planner.completed_goals:break
    assert planner.route_revision>0
    assert planner.completed_goals==1
    assert math.dist(initial[:2],pose[:2])>1.
    assert math.dist(pose[:2],goal)<=.12


def test_short_corner_recovery_still_rejects_unknown_space():
    with np.load(Path(__file__).parent/'fixtures/planning_corner.npz') as captured:
        navigation={'grid':np.full_like(captured['grid'],-1),
                    'origin':captured['origin'],'resolution':float(captured['resolution'])}
        decision=PathFollower().command(captured['pose'],captured['path'],captured['goal'],
                                       Decision(.065,0.,'ADVANCE','clear'),navigation)
    assert decision.state=='PLANNING'
    assert decision.linear==decision.angular==0.


def test_floor_projection_recovers_metric_points_with_tilt():
    k=np.array([[460.,0,320],[0,460.,240],[0,0,1]])
    c=Calibration(k)
    pixels=np.array([[180,360],[450,420],[320,100]])
    points,valid=c.ground(pixels)
    assert valid.tolist()==[True,True,False]
    assert np.allclose(points[0],[.0757+.072*460/120,.072*140/120])


def test_translation_rejects_outliers_and_recovers_compass_corrected_motion():
    rng=np.random.default_rng(9)
    old=rng.uniform([.15,-.2],[.7,.2],(80,2))
    shift=np.array([.013,-.005]);new=old-shift
    new[:15]+=rng.normal(0,.3,(15,2))
    delta,n=consensus_translation(old,new)
    assert np.allclose(delta,shift) and n==65
    assert consensus_translation(old[:5],new[:5])[0] is None


def test_near_floor_motion_survives_many_distant_wall_features():
    from realtime_depth_app.src.robot_slam import floor_translation
    floor=np.array([[.28,-.15],[.32,.16],[.56,-.13],[.60,.14]])
    wall=np.column_stack((np.linspace(.9,1.3,40),np.linspace(-.4,.4,40)))
    delta=np.array([.008,-.003])
    old=np.vstack((floor,wall));new=np.vstack((floor-delta,wall-[.05,.02]))
    measured,count=floor_translation(old,new)
    assert np.allclose(measured,delta) and count==4
    # A single floor edge cannot justify the smaller support requirement.
    edge=np.column_stack((np.linspace(.2,.6,4),np.zeros(4)))
    assert floor_translation(edge,edge-delta)[0] is None


def test_map_keeps_unknown_and_infinite_echo_never_creates_wall():
    m=OccupancyMap()
    m.update([0,0,0],{'front':(math.inf,4.)})
    g=m.grid()
    assert np.sum(g==100)==0 and np.sum(g==-1)>50000 and np.sum(g==0)>10
    m.update([0,0,0],{'left':(.8,4.)})
    assert np.sum(m.grid()==100)>0


def test_map_hit_leaves_space_behind_obstacle_unknown():
    m=OccupancyMap();m.update([0,0,0],{'front':(.6,4.)})
    endpoint=m.cell([.6846,0]);behind=m.cell([.9,0])
    assert m.grid()[endpoint[1],endpoint[0]]==100
    assert m.grid()[behind[1],behind[0]]==-1


def test_frontier_paths_stay_in_observed_inflated_free_space():
    m=OccupancyMap(size=100)
    m.observed[40:60,35:70]=True;m.logodds[40:60,35:70]=-3
    m.observed[45:55,52:54]=True;m.logodds[45:55,52:54]=3
    p=FrontierPlanner();pose=[-.5,0.,0.];p.plan(m,pose)
    assert p.goal is not None and p.path
    for point in p.path:
        x,y=m.cell(point)
        assert m.grid()[y,x]==0
        assert np.linalg.norm(np.array([x,y])-np.array([52,50]))>=3


def test_disconnected_frontier_cannot_produce_path_through_unknown():
    m=OccupancyMap(size=100)
    m.observed[48:53,48:53]=True;m.logodds[48:53,48:53]=-3
    m.observed[25:40,25:40]=True;m.logodds[25:40,25:40]=-3
    p=FrontierPlanner();p.plan(m,[0,0,0])
    assert not p.path


def test_new_destination_has_room_to_arrive_before_sonar_caution():
    m=OccupancyMap(size=100)
    m.observed[30:70,30:70]=True;m.logodds[30:70,30:70]=-3.
    m.logodds[30:70,69]=3.
    p=FrontierPlanner();p.plan(m,[0.,0.,0.])
    assert p.goal is not None
    occupied=np.argwhere(m.grid()==100)[:,::-1]
    walls=m.origin+(occupied+.5)*m.resolution
    assert np.linalg.norm(walls-p.goal,axis=1).min()>=.35


def test_curiosity_stops_for_tracking_loss_and_obstacle_emergency():
    p=CuriosityPolicy();rs={s:RangeReading(1.,0.,0.) for s in SIDES}
    safe=Decision(.08,0.,'ADVANCE','clear')
    s={'pose':[0,0,0],'goal':None,'path':[],'planner_state':'NEED_SCAN'}
    assert p.decide(s,rs,safe,False,0).state=='RECOVERING'
    scan=p.decide(s,rs,safe,True,0)
    assert scan.linear==0 and scan.angular!=0
    stop=Decision(0.,0.,'STOP','front emergency')
    assert p.decide(s,rs,stop,True,1)==stop


def test_travel_is_not_interrupted_by_periodic_curiosity_pauses():
    p=CuriosityPolicy();rs={s:RangeReading(1.,0.,0.) for s in SIDES}
    safe=Decision(.08,0.,'ADVANCE','clear')
    s={'pose':[0,0,0],'goal':[1,0],'path':[[0,0],[.5,0],[1,0]],'planner_state':'EXPLORING'}
    assert p.decide(s,rs,safe,True,0).linear>0
    assert p.decide(s,rs,safe,True,19).state=='EXPLORE'
    assert p.decide(s,rs,safe,True,20.5).state=='EXPLORE'
    assert p.decide(s,rs,safe,True,23).linear>0


def test_front_emergency_recovery_is_bounded_and_needs_rear_clearance():
    p=CuriosityPolicy();rs={s:RangeReading(1.,0.,0.) for s in SIDES}
    rs['front']=RangeReading(.12,0.,0.)
    s={'pose':[0,0,0],'goal':None,'path':[],'planner_state':'NEED_SCAN'}
    stop=Decision(0.,0.,'STOP','FRONT emergency clearance < 0.18 m')
    assert p.decide(s,rs,stop,True,0).state=='STARTLED'
    assert p.decide(s,rs,stop,True,.5).linear<0
    s['pose']=[-.11,0,0]
    assert p.decide(s,rs,stop,True,1).linear==0
    rs['rear']=RangeReading(.2,0.,0.)
    assert p.decide(s,rs,stop,True,6).linear==0
    rs['rear']=RangeReading(1.,2.,0.)
    assert p.decide(s,rs,stop,True,7).linear==0
    stale=Decision(0.,0.,'STOP','Camera stream stale or missing')
    assert p.decide(s,rs,stale,True,8)==stale


def test_frontier_target_habituates_and_unproductive_target_expires():
    m=OccupancyMap(size=100)
    m.observed[35:65,30:70]=True;m.logodds[35:65,30:70]=-3
    p=FrontierPlanner();p.plan(m,[0.,0.,0.],0.)
    first=np.asarray(p.goal);assert p.path
    key=tuple(np.floor(first/.25).astype(int))
    another=FrontierPlanner();another.visits[key]=100
    another.plan(m,[0.,0.,0.],1.)
    assert math.dist(first,another.goal)>.2


def test_monitor_mode_preserves_destination_even_after_long_delays():
    m=OccupancyMap(size=100)
    m.observed[35:65,30:70]=True;m.logodds[35:65,30:70]=-3
    p=FrontierPlanner();p.plan(m,[0.,0.,0.],0.)
    goal=list(p.goal)
    p.plan(m,[.02,.01,0.],1.)
    assert p.goal==goal
    p.plan(m,[.02,.01,0.],12.)
    assert p.goal==goal
    p.plan(m,[.02,.01,0.],300.)
    assert p.goal==goal and p.completed_goals==0


def test_stationary_robot_skips_goal_after_one_wall_minute_but_not_before():
    m=OccupancyMap(size=100)
    m.observed[35:65,30:70]=True;m.logodds[35:65,30:70]=-3
    p=FrontierPlanner();pose=[0.,0.,0.]
    p.plan(m,pose,0.,100.,True)
    first=list(p.goal);identity=p.goal_id
    # Heading changes and centimetre-scale estimate noise are not travel.
    for second in [10.,30.,59.9]:
        p.plan(m,[.005,-.005,second*.1],second*10,100.+second,True)
        assert p.goal==first and p.goal_id==identity
    p.plan(m,pose,600.,160.,True)
    assert p.goal is not None and math.dist(p.goal,first)>.35
    assert p.goal_id==identity+1 and p.stalled_goals==1 and p.completed_goals==0
    assert p.path and '60 s' in p.last_stall_reason


def test_real_progress_resets_stall_timer_and_monitor_time_does_not_count():
    m=OccupancyMap(size=100)
    m.observed[35:65,30:70]=True;m.logodds[35:65,30:70]=-3
    p=FrontierPlanner();pose=np.zeros(3)
    p.plan(m,pose,0.,0.,True);goal=list(p.goal)
    direction=np.array(goal)/math.dist([0.,0.],goal)
    pose[:2]=direction*.06
    p.plan(m,pose,59.,59.,True)
    p.plan(m,pose,100.,100.,True)
    assert p.goal==goal and p.stall_seconds==41.
    p.plan(m,pose,500.,500.,False)
    p.plan(m,pose,1000.,1000.,True)
    p.plan(m,pose,1059.,1059.,True)
    assert p.goal==goal and p.stalled_goals==0 and p.stall_seconds==59.
    p.plan(m,pose,1060.,1060.,True)
    assert p.goal!=goal and p.stalled_goals==1


def test_stalled_goal_is_abandoned_even_when_local_map_is_blocked():
    m=OccupancyMap(size=100)
    m.observed[35:65,30:70]=True;m.logodds[35:65,30:70]=-3
    p=FrontierPlanner();pose=[0.,0.,0.]
    p.plan(m,pose,0.,0.,True);goal=list(p.goal)
    m.observed[48:53,48:53]=False;m.logodds[48:53,48:53]=0.
    p.plan(m,pose,59.,59.,True)
    assert p.goal==goal and p.state=='GOAL_HELD'
    p.plan(m,pose,60.,60.,True)
    assert p.goal is None and not p.path and p.stalled_goals==1
    m.observed[48:53,48:53]=True;m.logodds[48:53,48:53]=-3.
    p.plan(m,pose,61.,61.,True)
    assert p.goal is not None and math.dist(p.goal,goal)>.35


def test_temporary_local_map_loss_and_blocked_goal_do_not_replace_destination():
    m=OccupancyMap(size=100)
    m.observed[35:65,30:70]=True;m.logodds[35:65,30:70]=-3
    p=FrontierPlanner();p.plan(m,[0.,0.,0.],0.)
    goal=list(p.goal);identity=p.goal_id
    m.observed[48:53,48:53]=False;m.logodds[48:53,48:53]=0.
    p.plan(m,[0.,0.,0.],1.)
    assert p.goal==goal and p.goal_id==identity and p.state=='GOAL_HELD'
    m.observed[:]=True;m.logodds[:]=-3.
    x,y=m.cell(goal);m.logodds[y,x]=3.
    p.plan(m,[0.,0.,0.],50.)
    assert p.goal==goal and not p.path and p.state=='GOAL_BLOCKED'
    m.logodds[y,x]=-3.
    p.plan(m,[0.,0.,0.],100.)
    # A goal remains valid even after it ceases to be a frontier.
    assert p.goal==goal and p.path[-1]==goal and p.goal_id==identity


def test_only_confirmed_arrival_permits_a_new_destination():
    m=OccupancyMap(size=100)
    m.observed[35:65,30:70]=True;m.logodds[35:65,30:70]=-3
    p=FrontierPlanner();p.plan(m,[0.,0.,0.],0.)
    goal=list(p.goal);identity=p.goal_id
    near=[goal[0]-.10,goal[1],0.]
    p.plan(m,near,30.)
    assert p.goal==goal and p.state=='ARRIVING' and p.completed_goals==0
    p.plan(m,near,30.6)
    assert p.goal==goal and p.state=='ARRIVED' and p.completed_goals==1
    p.plan(m,near,31.)
    assert p.goal==goal and p.goal_id==identity
    p.plan(m,near,32.2)
    assert p.goal!=goal and p.goal_id==identity+1 and p.completed_goals==1


def test_unchanged_safe_route_is_not_recomputed_on_each_map_update():
    m=OccupancyMap(size=100)
    m.observed[35:65,30:70]=True;m.logodds[35:65,30:70]=-3
    p=FrontierPlanner();p.plan(m,[0.,0.,0.],0.)
    route=list(p.path);revision=p.route_revision
    for t in [1.,5.,20.,100.]:p.plan(m,[.01,.01,0.],t)
    assert p.path==route and p.route_revision==revision


def test_path_follower_uses_turn_hysteresis_and_continuous_steering():
    f=PathFollower();path=[[0.,0.],[1.,0.]];safe=Decision(.08,0.,'ADVANCE','clear')
    assert f.command([0.,0.,-1.2],path,[1.,0.],safe).state=='ALIGN'
    assert f.command([0.,0.,-.7],path,[1.,0.],safe).state=='ALIGN'
    assert f.command([0.,0.,-.35],path,[1.,0.],safe).state=='EXPLORE'
    outputs=[f.command([0.,0.,-error],path,[1.,0.],safe) for error in [.29,.31,.28,.32]]
    assert all(d.state=='EXPLORE' and d.linear>0 for d in outputs)
    assert max(d.angular for d in outputs)-min(d.angular for d in outputs)<.05


def test_small_drift_does_not_latch_stationary_alignment():
    f=PathFollower();path=[[0.,0.],[2.,0.]];safe=Decision(.08,0.,'ADVANCE','clear')
    assert f.command([0.,0.,math.pi],path,[2.,0.],safe).state=='ALIGN'
    # After the coarse turn, a 6 cm lateral error and 12-degree heading error
    # are corrected while traveling, rather than waiting for exact alignment.
    d=f.command([0.,.06,math.radians(12)],path,[2.,0.],safe)
    assert d.state=='EXPLORE' and d.linear>.05 and d.angular<0
    for yaw in np.radians([-15,10,-8,18,-12]):
        d=f.command([.2,.03,float(yaw)],path,[2.,0.],safe)
        assert d.state=='EXPLORE' and d.linear>.05


def test_closed_loop_small_drift_is_corrected_without_stop_and_turn_cycles():
    from realtime_depth_app.src.robot_motion import CommandSmoother
    follower=PathFollower();smoother=CommandSmoother()
    pose=np.array([0.,.06,math.radians(12)])
    safe=Decision(.08,0.,'ADVANCE','clear');speeds=[]
    for i in range(400):
        if i in [80,160]:
            pose[1]+=.015;pose[2]+=math.radians(5)
        decision=follower.command(pose,[[0.,0.],[3.,0.]],[3.,0.],safe)
        assert decision.state=='EXPLORE' and decision.linear>0
        command=smoother.apply(decision,i*.05);speeds.append(command.linear)
        pose[0]+=.05*command.linear*math.cos(pose[2])
        pose[1]+=.05*command.linear*math.sin(pose[2])
        pose[2]+=.05*command.angular
    assert pose[0]>1.2 and abs(pose[1])<.015
    assert np.mean(speeds)>.06


def test_consensus_can_find_a_supported_floor_motion_among_foreground_outliers():
    rng=np.random.default_rng(291)
    new=rng.uniform([.15,-.3],[1.,.3],(120,2))
    delta=rng.uniform(-.15,.15,(120,2))
    delta[:40]=[.012,-.004]
    measured,count=consensus_translation(new+delta,new)
    assert measured is not None and np.allclose(measured,[.012,-.004],atol=.001)
    assert count>=40


def test_competing_visual_motions_remain_rejected():
    rng=np.random.default_rng(21)
    new=rng.uniform([.2,-.2],[.8,.2],(80,2))
    shifts=np.array([[.01,0.]]*40+[[-.04,.01]]*40)
    assert consensus_translation(new+shifts,new)[0] is None


def test_brief_dropouts_have_bounded_motion_and_relocalization_has_a_delay():
    fresh={'last_good_stamp':10.,'age_s':.03,'plan_ready':True}
    assert localization_health(fresh,10.1,True,.02,.02)['can_bridge_pose']
    assert not localization_health(fresh,10.3,True,.02,.02)['can_bridge_pose']
    assert not localization_health(fresh,10.3,True,.02,.02)['can_relocalize']
    assert localization_health(fresh,11.,True,.02,.02)['can_relocalize']
    assert not localization_health(fresh,10.1,True,1.,.02)['can_bridge_pose']
    assert not localization_health(fresh,11.,False,.02,.02)['can_relocalize']
    p=CuriosityPolicy();readings={s:RangeReading(1.,0.,0.) for s in SIDES}
    snapshot={'pose':[0,0,0],'goal':[1,0],'path':[[0,0],[1,0]],'planner_state':'EXPLORING',
              'can_bridge_pose':True,'can_relocalize':False}
    d=p.decide(snapshot,readings,Decision(.08,0.,'ADVANCE','clear'),False,0.)
    assert d.state=='TRACKING_BRIDGE' and 0<d.linear<=.02
    snapshot['can_bridge_pose']=False
    d=p.decide(snapshot,readings,Decision(.08,0.,'ADVANCE','clear'),False,.3)
    assert d.state=='RECOVERING' and d.linear==d.angular==0
    snapshot['can_bridge_pose']=True;readings['front']=RangeReading(.35,0.,0.)
    d=p.decide(snapshot,readings,Decision(.03,0.,'ADVANCE','caution'),False,.4)
    assert d.state=='RECOVERING' and d.linear==0


def test_route_lookahead_cannot_cut_an_obstructed_corner():
    m=OccupancyMap(size=60)
    m.observed[:]=True;m.logodds[:]=-3.
    x,y=m.cell([-.25,.25]);m.logodds[y,x]=3.
    path=[[-.5,0.],[-.5,.25],[-.5,.5],[-.25,.5],[0.,.5]]
    target=route_target([-.5,0.,0.],path,m.grid(),m.origin,m.resolution)
    assert target==[-.5,.5]


def test_grid_staircases_are_shortened_without_cutting_obstacles_or_unknown_space():
    safe=np.ones((60,60),bool);origin=np.array([-1.5,-1.5]);resolution=.05
    points=[[i*.05,i*.05] for i in range(11)]
    simplified=simplify_route(points,safe,origin,resolution)
    assert simplified==[points[0],points[-1]]
    safe[33:38,33:38]=False
    detour=[[0.,0.],[.05,0.],[.10,0.],[.15,0.],[.20,0.],[.25,0.],[.30,0.],[.35,0.],[.40,0.],
            [.45,0.],[.5,0.],[.5,.05],[.5,.10],[.5,.15],[.5,.2],[.5,.25],[.5,.30],[.5,.35],[.5,.4],[.5,.45],[.5,.5]]
    simplified=simplify_route(detour,safe,origin,resolution)
    assert simplified[0]==detour[0] and simplified[-1]==detour[-1]
    assert len(simplified)>2
    assert all(segment_clear(a,b,safe,origin,resolution) for a,b in zip(simplified[:-1],simplified[1:]))


def test_reaching_frontier_does_not_command_a_turn_backwards():
    p=FrontierPlanner();p.path=[[0.,0.],[.05,0.]];p.goal=[.05,0.]
    rs={s:RangeReading(1.,0.,0.) for s in SIDES}
    d=p.command([.05,0.,math.pi],rs,Decision(.05,0.,'ADVANCE','clear'),True)
    assert d.state=='ARRIVING' and d.linear==d.angular==0.


def test_long_escape_turn_uses_bounded_backup_instead_of_spinning_forever():
    p=CuriosityPolicy();rs={s:RangeReading(1.,0.,0.) for s in SIDES}
    s={'pose':[0,0,0],'goal':None,'path':[],'planner_state':'NEED_SCAN'}
    turn=Decision(0.,.65,'TURN','Holding chosen escape direction')
    assert p.decide(s,rs,turn,True,0)==turn
    assert p.decide(s,rs,turn,True,7).state=='STARTLED'
    assert p.decide(s,rs,turn,True,7.5).state=='BACK_AWAY'
    assert p.decide(s,rs,turn,True,11.5)==turn


def test_dashboard_temporary_sensor_delay_stops_without_disarming_healthy_controller():
    import threading,time
    from types import SimpleNamespace
    from robot_live import RobotNode
    now=time.monotonic();sent=[]
    fake=SimpleNamespace(lock=threading.RLock(),frame=SimpleNamespace(received=now-1),
        prediction=SimpleNamespace(packet=SimpleNamespace(received=now-1)),clock_received=now,
        last_control_wall=now,readings=lambda _: {},decision=Decision(.05,.1,'EXPLORE','moving'),
        active=True,command_pub=SimpleNamespace(publish=sent.append))
    _,_,_,decision,active=RobotNode.snapshot(fake)
    assert active and decision.linear==decision.angular==0 and len(sent)==0


def test_pose_graph_loop_constraint_corrects_accumulated_translation():
    s=FloorSlam()
    s.keyframes=[{'pose':np.array([x,0.,0.])} for x in [0.,1.1,2.2]]
    s.edges=[(0,1,np.array([1.1,0]),1.),(1,2,np.array([1.1,0]),1.),(0,2,np.array([2.,0]),5.)]
    s.pose=np.array([2.2,0.,0.]);s.optimize()
    assert abs(s.pose[0]-2.)<.01
    assert abs(s.keyframes[0]['pose'][0])<1e-5


def test_background_loop_correction_preserves_new_tracking_and_scans():
    import time
    from realtime_depth_app.src.robot_slam import AsyncMapOptimizer
    s=FloorSlam()
    s.keyframes=[{'pose':np.array([x,0.,0.])} for x in [0.,1.1,2.2]]
    s.edges=[(0,1,np.array([1.1,0]),1.),(1,2,np.array([1.1,0]),1.),(0,2,np.array([2.,0]),5.)]
    s.pose=np.array([2.2,0.,0.]);s.optimizer=AsyncMapOptimizer();s.optimizer.start()
    try:
        s.optimize()
        assert s.optimizer.pending and s.pose[0]==2.2
        # Tracking and newly measured scans continue in the original frame.
        s.keyframes.append({'pose':np.array([2.3,0.,0.])});s.pose[0]=2.3
        s.scans.append({'pose':s.pose.copy(),'ranges':{'front':(.5,2.5)},'keyframe':3})
        until=time.monotonic()+8.
        while s.optimizer.pending and time.monotonic()<until:
            s.optimizer.collect(s);time.sleep(.01)
        assert not s.optimizer.pending,s.optimizer.error
        assert abs(s.pose[0]-2.1)<.01
        assert abs(s.keyframes[-1]['pose'][0]-s.pose[0])<1e-6
        assert abs(s.scans[-1]['pose'][0]-s.pose[0])<1e-6
        assert s.map.observed.sum()>20
    finally:s.optimizer.stop()


def test_map_export_matches_ros_y_axis_and_metadata(tmp_path):
    import yaml
    m=OccupancyMap(size=20)
    m.observed[2,3]=True;m.logodds[2,3]=3;m.save(tmp_path)
    image=cv2.imread(str(tmp_path/'map.pgm'),cv2.IMREAD_GRAYSCALE)
    assert image[-3,3]==0 and image[2,3]==205
    config=yaml.safe_load((tmp_path/'map.yaml').read_text())
    assert config['resolution']==.05 and config['origin']==[-.5,-.5,0.]


def test_visual_frontend_tracks_rendered_metric_motion_and_stops_on_black_frame():
    rng=np.random.default_rng(31)
    texture=rng.integers(20,235,(1200,1200),np.uint8)
    texture=cv2.GaussianBlur(texture,(5,5),0)
    k=np.array([[460.,0,320],[0,460.,240],[0,0,1.]])
    calibration=Calibration(k)
    def frame(x,y,yaw):
        c,s=math.cos(yaw),math.sin(yaw)
        # Texture pixels -> world floor [-1.2,1.2], then optical camera.
        homography=k@np.array([[.002*s,-.002*c,-1.2*s+1.2*c-x*s+y*c],
                               [0.,0.,.072],
                               [.002*c,.002*s,-1.2*c-1.2*s-x*c-y*s-.0757]])
        gray=cv2.warpPerspective(texture,homography,(640,480))
        return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    engine=FloorSlam()
    for i in range(8):
        engine.process(frame(i*.003,0.,i*.005),i*.1,calibration,(0.,0.,i*.005,True),{})
    assert engine.tracking and engine.inliers>=12
    assert np.linalg.norm(engine.pose[:2]-[.021,0.])<.008
    engine.process(np.zeros((480,640,3),np.uint8),.8,calibration,(0.,0.,.035,True),{})
    assert not engine.tracking
    engine.process(frame(.021,0.,.035),.9,calibration,(0.,0.,.035,True),{})
    assert engine.tracking
    engine.process(np.zeros((480,640,3),np.uint8),1.,calibration,(0.,0.,.035,True),{})
    engine.flow_keyframes=[]
    engine.match_place=lambda *args:None
    engine.match_trusted_view=lambda *args:None
    # The latest accepted image stays usable even after older place matching
    # fails for much longer than the old two-second optical-flow cutoff.
    engine.process(frame(.021,0.,.035),10.,calibration,(0.,0.,.035,True),{})
    assert engine.tracking and np.linalg.norm(engine.pose[:2]-[.021,0.])<.008


def test_slam_export_serializes_numpy_visit_bin_coordinates(tmp_path):
    import json,time
    from realtime_depth_app.src.camera import LatestSlot
    worker=SlamWorker(LatestSlot())
    worker.planning.latest={'epoch':0,'plan_finished':time.monotonic(),
        'visits':{(np.int64(2),np.int64(-3)):np.int64(4)}}
    worker.save(tmp_path)
    report=json.loads((tmp_path/'slam_report.json').read_text())
    assert report['visits']==[{'cell':[2,-3],'count':4}]


def test_slow_frontier_search_cannot_freeze_live_slam_revisions():
    import threading,time
    from realtime_depth_app.src.camera import LatestSlot
    started=threading.Event();release=threading.Event()
    frames=LatestSlot();worker=SlamWorker(frames)
    class LiveEngine(FloorSlam):
        def process(self,*item,plan=True):
            self.last_stamp=item[1];self.tracking=True;self.pose[0]=item[1]
    worker.engine=LiveEngine();worker.planning.epoch=0
    original=worker.planning.planner.plan
    def slow_plan(*args):
        started.set();release.wait(2.);original(*args)
    worker.planning.planner.plan=slow_plan
    worker.start()
    try:
        frames.put((None,1.,None,None,{}));assert started.wait(1.)
        frames.put((None,2.,None,None,{}))
        until=time.monotonic()+1.
        while worker.snapshot().get('revision',0)<2 and time.monotonic()<until:time.sleep(.01)
        assert worker.snapshot()['pose'][0]==2. and not release.is_set()
    finally:
        release.set();worker.stop();worker.join(timeout=3.)


def test_relocalization_changes_remembered_view_if_first_view_does_not_recover():
    p=CuriosityPolicy();rs={s:RangeReading(1.,0.,0.) for s in SIDES}
    s={'pose':[0,0,0],'goal':None,'path':[],'planner_state':'NEED_SCAN',
       'can_relocalize':True,'trajectory':[[0,0,0],[0,0,.25],[0,0,-.25]]}
    safe=Decision(.05,0.,'ADVANCE','clear')
    d=p.decide(s,rs,safe,False,0)
    assert d.state=='RELOCALIZE' and d.angular==0
    d=p.decide(s,rs,safe,False,4.)
    assert d.state=='RELOCALIZE' and abs(d.angular)>0


def test_alignment_precedes_translational_lookahead_but_never_authorizes_unsafe_translation():
    from realtime_depth_app.src.robot_avoidance import Decision
    safety=Decision(.065,0.,'ADVANCE','sensor band clear')
    navigation=dict(grid=np.full((40,40),-1,np.int8),origin=[-1.,-1.],resolution=.05)
    follower=PathFollower()
    # Translation is unknown, but turning toward the route is an independent
    # proposal still subject to the caller's sonar and supervisor sweep gates.
    decision=follower.command([0.,0.,math.pi],[[0.,0.],[.5,0.]],[.5,0.],safety,navigation)
    assert decision.state=='ALIGN'
    assert decision.linear==0. and abs(decision.angular)<=.4
    decision=follower.command([0.,0.,0.],[[0.,0.],[.5,0.]],[.5,0.],safety,navigation)
    assert decision.state=='PLANNING' and decision.linear==decision.angular==0.
