"""Acceleration-limited autonomous commands; safety stops always take priority."""
from .robot_avoidance import Decision


class CommandSmoother:
    def __init__(self, acceleration=.06, deceleration=.12, yaw_acceleration=.60):
        self.acceleration=acceleration
        self.deceleration=deceleration
        self.yaw_acceleration=yaw_acceleration
        self.reset()

    def reset(self):
        self.linear=0.
        self.angular=0.
        self.last_time=None

    @staticmethod
    def approach(current,target,limit):
        return max(current-limit,min(current+limit,target))

    def apply(self,decision,now):
        # Data loss, explicit stops, route waits and arrival hold are immediate.
        if decision.state in ['STOP','MONITOR','WAITING','PLANNING','NOTICE','STARTLED','ARRIVING','ARRIVED','RECOVERING']:
            self.reset();self.last_time=now
            return Decision(0.,0.,decision.state,decision.reason)
        dt=.05 if self.last_time is None else max(0.,min(.10,now-self.last_time))
        self.last_time=now
        target=decision.linear
        if target*self.linear<0 and abs(self.linear)>.002:
            target=0.
        rate=self.acceleration if abs(target)>abs(self.linear) else self.deceleration
        self.linear=self.approach(self.linear,target,rate*dt)
        if decision.state=='TRACKING_BRIDGE':self.linear=min(self.linear,.020)
        if decision.state in ['TURN','RELOCALIZE']:
            # A local obstacle override or localization recovery never drifts forward.
            self.linear=0.
        self.angular=self.approach(self.angular,decision.angular,self.yaw_acceleration*dt)
        return Decision(self.linear,self.angular,decision.state,decision.reason)
