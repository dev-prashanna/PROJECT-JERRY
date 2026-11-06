"""Robot policy invariants: depth convention, conservative gates and turn selection."""
import math
import numpy as np
from src.robot_avoidance import analyze_depth,AvoidancePolicy,RangeReading,SIDES

def ranges(**changes):
 out={s:RangeReading(1.,.05,.05) for s in SIDES}
 for side,value in changes.items():out[side]=RangeReading(value,.05,.05)
 return out

def scene(center=.25,left=.15,right=.2):
 d=np.zeros((100,100),np.float32);d[80:]=1.
 d[20:70,6:34]=left;d[20:70,34:66]=center;d[20:70,66:94]=right
 return d

def decide(policy,view,rs=None,**kw):
 args=dict(now=5.,depth_age=.05,camera_age=.05,clock_age=.05);args.update(kw)
 return policy.decide(view,rs or ranges(),**args)

def test_relative_affine_scale_does_not_change_proximity():
 d=scene(.9);a=analyze_depth(d);b=analyze_depth(d*7+23)
 assert np.allclose(a.proximity,b.proximity,atol=1e-5)
 assert np.allclose(a.fractions,b.fractions)

def test_clear_band_advances_at_bounded_speed():
 d=decide(AvoidancePolicy(),analyze_depth(scene()))
 assert 0<d.linear<=.08 and abs(d.angular)<=.35

def test_near_center_depth_turns_toward_clearer_left_sector():
 d=decide(AvoidancePolicy(),analyze_depth(scene(.95,.1,.6)))
 assert d.linear==0 and d.angular>0 and d.state=='TURN'

def test_ultrasound_overrides_visual_clearance_and_blocks_escape_side():
 view=analyze_depth(scene())
 stop=decide(AvoidancePolicy(),view,ranges(front=.12))
 assert stop.linear==stop.angular==0
 turn=decide(AvoidancePolicy(),view,ranges(front=.29,left=.15))
 assert turn.linear==0 and turn.angular<0

def test_stale_sources_and_invalid_depth_always_stop():
 for kw in [dict(depth_age=1.),dict(camera_age=1.),dict(clock_age=2.)]:
  d=decide(AvoidancePolicy(),analyze_depth(scene()),**kw);assert d.linear==d.angular==0
 for depth in [np.full((100,100),np.nan),np.zeros((100,100))]:
  d=decide(AvoidancePolicy(),analyze_depth(depth));assert d.linear==d.angular==0
 rs=ranges();rs['rear']=RangeReading(1.,2.,.1)
 d=decide(AvoidancePolicy(),analyze_depth(scene()),rs);assert d.linear==d.angular==0

def test_sonar_infinity_semantics_and_nan_fail_closed():
 view=analyze_depth(scene())
 for value in [math.nan,-math.inf]:
  d=decide(AvoidancePolicy(),view,ranges(front=value));assert d.linear==d.angular==0
 d=decide(AvoidancePolicy(),view,ranges(front=math.inf));assert d.linear>0

def test_pivot_direction_is_latched_and_rear_clearance_is_required():
 policy=AvoidancePolicy();a=decide(policy,analyze_depth(scene(.95,.1,.7)))
 b=decide(policy,analyze_depth(scene(.95,.8,.1)),now=5.1)
 assert a.angular==b.angular>0
 c=decide(policy,analyze_depth(scene(.95)),ranges(rear=.08),now=5.2)
 assert c.linear==c.angular==0
