"""Conservative camera-relative-depth + four-range local avoidance.
Relative depth provides ranking, never metres. No simulator pose is consumed.
"""
from dataclasses import dataclass
import math
import numpy as np

SIDES=('front','left','rear','right')
@dataclass(frozen=True)
class RangeReading:
    value: float
    age_wall: float
    age_sensor: float
    minimum: float=.02
    maximum: float=4.
    def usable(self):
        return (self.age_wall<=.8 and -.1<=self.age_sensor<=.65 and not math.isnan(self.value)
                and (not math.isfinite(self.value) or self.minimum<=self.value<=self.maximum))
    def clearance(self):
        if self.value==math.inf:return self.maximum
        if self.value==-math.inf:return 0.
        return self.value
@dataclass(frozen=True)
class DepthView:
    proximity: np.ndarray
    scores: tuple[float,float,float]
    fractions: tuple[float,float,float]
    valid_fraction: float
    usable: bool

@dataclass(frozen=True)
class Decision:
    linear: float
    angular: float
    state: str
    reason: str

def analyze_depth(depth: np.ndarray, convention: str='inverse')->DepthView:
    if depth.ndim!=2 or depth.size==0:
        return DepthView(np.zeros((1,1),np.float32),(1.,)*3,(1.,)*3,0.,False)
    finite=np.isfinite(depth);valid=float(finite.mean());p=np.zeros_like(depth,dtype=np.float32)
    if valid<.95:return DepthView(p,(1.,)*3,(1.,)*3,valid,False)
    lo,hi=np.percentile(depth[finite],[2,98]);span=hi-lo
    if span<=max(1e-6,abs(float(hi))*1e-6):return DepthView(p,(1.,)*3,(1.,)*3,valid,False)
    p[finite]=np.clip((depth[finite]-lo)/span,0,1)
    if convention=='forward':p[finite]=1.-p[finite]
    p[~finite]=1.
    h,w=p.shape;scores=[];fractions=[]
    # Exclude sky and near floor: these ROIs inspect the forward obstacle band.
    for a,b in [(.06,.34),(.34,.66),(.66,.94)]:
        roi=p[int(.20*h):max(int(.20*h)+1,int(.70*h)),int(a*w):max(int(a*w)+1,int(b*w))]
        scores.append(float(np.percentile(roi,80)));fractions.append(float(np.mean(roi>.72)))
    return DepthView(p,tuple(scores),tuple(fractions),valid,True)

class AvoidancePolicy:
    def __init__(self):self.turn=0.;self.turn_start=0.
    def reset(self):self.turn=0.
    def decide(self,view:DepthView|None,readings:dict[str,RangeReading],now:float,
               depth_age:float,camera_age:float,clock_age:float)->Decision:
        def stop(reason):
            self.reset();return Decision(0.,0.,'STOP',reason)
        if clock_age>1.:return stop('Simulation clock paused or missing')
        if camera_age>.65:return stop('Camera stream stale or missing')
        if view is None or not view.usable:return stop('Depth unavailable or unreliable')
        if depth_age>.45:return stop('Depth prediction stale')
        for s in SIDES:
            if s not in readings or not readings[s].usable():return stop(s.upper()+' ultrasound stale or invalid')
        f,l,b,r=[readings[s].clearance() for s in SIDES]
        if f<.18:return stop('FRONT emergency clearance < 0.18 m')
        if min(l,r)<.09:return stop('SIDE emergency clearance < 0.09 m')
        ls,cs,rs=view.scores;lf,cf,rf=view.fractions
        blocked=f<.32 or (cf>.22 and cs>.78)
        # A persistent turn direction avoids left/right oscillation as predictions vary.
        if self.turn:
            side=l if self.turn>0 else r
            if side<.17 or b<.12:return stop('Insufficient clearance for pivot')
            if not blocked and f>.44 and cf<.16 and now-self.turn_start>.7:self.reset()
            else:return Decision(0.,self.turn,'TURN','Holding chosen escape direction')
        if blocked:
            if b<.12:return stop('Rear too close for pivot')
            choices=[]
            if l>.20:choices.append((ls+.7*lf+.07/max(l,.05),.65))
            if r>.20:choices.append((rs+.7*rf+.07/max(r,.05),-.65))
            if not choices:return stop('Both escape sides blocked')
            self.turn=min(choices,key=lambda v:v[0])[1];self.turn_start=now
            why='FRONT ultrasound caution' if f<.32 else 'Near relative-depth region ahead'
            return Decision(0.,self.turn,'TURN',why)
        speed=.08*min(1.,max(.25,(f-.18)/.45))
        # Steer away from the relatively nearer image sector, reinforced by side ranges.
        yaw=float(np.clip((rs-ls)*.35,-.22,.22))
        if l<.18:speed=min(speed,.03);yaw=-.35
        if r<.18:speed=min(speed,.03);yaw=.35
        return Decision(speed,yaw,'ADVANCE','Forward band clear; depth-guided steering')

class RoamingPolicy:
    """Random heading, then ten simulation seconds of travel; safety can interrupt anytime."""
    def __init__(self,segment_seconds=10.,seed=None):
        import random
        self.rng=random.Random(seed);self.segment_seconds=segment_seconds;self.safety=AvoidancePolicy()
        self.segments=0;self.last_offset=None;self.preferred_headings=[];self.reset()
    def reset(self):
        self.phase='SELECT';self.target=None;self.deadline=None;self.phase_started=0.;self.brake_until=0.;self.safety.reset()
    @staticmethod
    def angle(x):return (x+math.pi)%(2*math.pi)-math.pi
    def status(self,now):
        return {'phase':self.phase,'target_yaw_deg':math.degrees(self.target) if self.target is not None else None,'seconds_remaining':max(0.,self.deadline-now) if self.deadline is not None else self.segment_seconds,'segments':self.segments,'segment_seconds':self.segment_seconds}
    def choose(self,view,readings,yaw,now,escape=False):
        f,l,b,r=[readings[s].clearance() for s in SIDES]
        ls,cs,rs=view.scores;lf,cf,rf=view.fractions
        candidates=[]
        if not escape and f>.48 and cf<.16:candidates.append((0.,1./(.3+cs)))
        if l>.26 and min(f,b,r)>.14:
            for angle in ([45] if escape else [25,50]):candidates.append((math.radians(angle),1./(.3+ls+lf+.08/l)))
        if r>.26 and min(f,b,l)>.14:
            for angle in ([-45] if escape else [-25,-50]):candidates.append((math.radians(angle),1./(.3+rs+rf+.08/r)))
        if not escape and self.preferred_headings:
            preferred=[]
            for heading in self.preferred_headings:
                offset=self.angle(heading-yaw)
                if abs(offset)<.12 and f>.48 and cf<.16:preferred.append((offset,8./(.3+cs)))
                elif .12<=offset<1.75 and l>.26 and min(f,b,r)>.14:preferred.append((offset,8./(.3+ls+lf)))
                elif -1.75<offset<=-.12 and r>.26 and min(f,b,l)>.14:preferred.append((offset,8./(.3+rs+rf)))
            if preferred:candidates=preferred
        if not candidates:return False
        if not escape and self.segments>0 and len(candidates)>1:candidates=[c for c in candidates if abs(c[0])>.15] or candidates
        if escape:
            offset=max(candidates,key=lambda x:x[1])[0]
        else:
            distinct=[x for x in candidates if self.last_offset is None or abs(x[0]-self.last_offset)>.1]
            pool=distinct or candidates;offset=self.rng.choices([x[0] for x in pool],weights=[x[1] for x in pool],k=1)[0]
        self.last_offset=offset;self.target=self.angle(yaw+offset);self.phase='ESCAPE' if escape else 'ALIGN';self.phase_started=now;self.deadline=None
        return True
    def decide(self,view,readings,now,depth_age,camera_age,clock_age,yaw=None,heading_ready=False):
        self.safety.reset()
        base=self.safety.decide(view,readings,now,depth_age,camera_age,clock_age)
        if base.state=='STOP':self.reset();return base
        if not heading_ready or yaw is None or not math.isfinite(yaw):
            self.reset();return Decision(0.,0.,'STOP','IMU/compass heading unavailable or stale')
        f,l,b,r=[readings[s].clearance() for s in SIDES];cf=view.fractions[1]
        hazard=f<.42 or min(l,r)<.19 or (cf>.22 and view.scores[1]>.78)
        if self.phase=='SELECT':
            if hazard:self.phase='BRAKE';self.brake_until=now+.35
            elif not self.choose(view,readings,yaw,now):return Decision(0.,0.,'STOP','No clear random travel direction')
        if hazard and self.phase in ['TRAVEL','ALIGN']:
            self.phase='BRAKE';self.deadline=None;self.brake_until=now+.35
        if self.phase=='BRAKE':
            if now<self.brake_until:return Decision(0.,0.,'BRAKE','Obstacle override: stop before selecting clearer heading')
            if not self.choose(view,readings,yaw,now,escape=True):return Decision(0.,0.,'STOP','No safe escape direction; waiting')
        if self.phase in ['ALIGN','ESCAPE']:
            error=self.angle(self.target-yaw)
            side=l if error>0 else r
            if side<.18 or b<.14:return Decision(0.,0.,'STOP','Pivot clearance too small')
            if abs(error)<math.radians(7):
                if hazard:
                    self.phase='BRAKE';self.brake_until=now+.35;return Decision(0.,0.,'BRAKE','Heading reached but path still obstructed')
                self.phase='TRAVEL';self.deadline=now+self.segment_seconds;self.segments+=1
            else:
                if now-self.phase_started>8.:
                    self.phase='BRAKE';self.brake_until=now+.35;return Decision(0.,0.,'BRAKE','Heading alignment timeout; recheck escape')
                return Decision(0.,float(np.clip(error*1.5,-.55,.55)),'REORIENT','Turning to the selected clear heading using IMU/compass yaw')
        if now>=self.deadline:
            self.phase='SELECT';self.deadline=None
            return Decision(0.,0.,'SELECT','Ten-second travel segment complete; choosing next direction')
        error=self.angle(self.target-yaw)
        yaw_command=float(np.clip(1.1*error+(view.scores[2]-view.scores[0])*.16,-.30,.30))
        speed=min(.07,base.linear)
        return Decision(speed,yaw_command,'ROAM','Holding selected heading; obstacle checks override the timer')
