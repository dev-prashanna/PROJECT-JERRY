"""Measured mission progress; exploring known corridors must not look stalled."""
import math


class MissionProgress:
    def __init__(self,now,timeout=20.):
        self.last_progress=now
        self.timeout=timeout
        self.area=0.
        self.completed=0
        self.goal_identity=None
        self.best_distance=math.inf
        self.best_heading=math.inf

    def update(self,snapshot,now):
        changed=False
        area=snapshot.get('observed_m2',0.)
        completed=snapshot.get('completed_goals',0)
        if area>=self.area+.02:
            self.area=area
            changed=True
        if completed>self.completed:
            self.completed=completed
            changed=True
        goal=snapshot.get('goal')
        pose=snapshot.get('pose')
        if goal is not None and pose is not None and snapshot.get('ready',False):
            identity=(snapshot.get('map_revision',0),snapshot.get('goal_id'),tuple(goal))
            if identity!=self.goal_identity:
                self.goal_identity=identity
                self.best_distance=self.best_heading=math.inf
            distance=math.dist(pose[:2],goal)
            angle=math.atan2(goal[1]-pose[1],goal[0]-pose[0])-pose[2]
            heading=abs((angle+math.pi)%(2*math.pi)-math.pi)
            if distance<=self.best_distance-.02:
                self.best_distance=distance
                changed=True
            if heading<=self.best_heading-.05:
                self.best_heading=heading
                changed=True
        if changed:self.last_progress=now
        return now-self.last_progress<=self.timeout
