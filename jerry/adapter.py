"""Protected actuator endpoint attached to the existing Gazebo sensor adapter."""
import math
import time
from uuid import uuid4
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from jerry_interfaces.msg import MotionCommand
from jerry_interfaces.srv import Authority
from .authority import AuthorityGate,host_clock_id
from .ros_utils import wall_timer


class ProtectedActuator:
    def __init__(self, node):
        self.node = node
        self.gate = AuthorityGate()
        self.clock_id = host_clock_id()
        self.session = uuid4().hex
        self.epoch = 0
        self.sim = None
        self.progress = -math.inf
        self.command = (0., 0., 0.)
        node.create_service(Authority, '/jerry/adapter/authority', self.authority)
        node.create_subscription(MotionCommand, '/jerry/actuation/command', self.receive, 1)
        node.create_subscription(Clock, '/clock', self.clock, qos_profile_sensor_data)

    def clock(self, msg):
        stamp = msg.clock.sec+msg.clock.nanosec*1e-9
        if self.sim is not None and stamp < self.sim-.01:
            self.epoch += 1
            self.gate.revoke('Simulation reset')
            self.command = (0.,0.,0.)
        if self.sim is None or stamp != self.sim:
            self.progress = time.monotonic()
        self.sim = stamp

    def authority(self, req, res):
        res.session_id, res.epoch_id = self.session, self.epoch
        try:
            if req.operation == 'STATUS':
                pass
            elif req.operation == 'REVOKE':
                if req.session_id != self.session:
                    raise ValueError('Wrong adapter session')
                self.gate.revoke('Supervisor revocation')
                self.command = (0.,0.,0.)
                self.node.output.publish(Twist())
            elif req.operation == 'CLAIM':
                if req.session_id != self.session or req.epoch_id != self.epoch:
                    raise ValueError('Wrong adapter session/epoch')
                self.gate.acquire(self.session, self.epoch, req.owner_id, req.task_id, req.map_revision)
            elif req.operation == 'TOKEN':
                self.gate.renew(req)
                res.token, res.valid_for_s = self.gate.issue(req,req.maximum_validity_s or None)
            else:
                raise ValueError('Unknown adapter operation')
            res.accepted = True
            res.generation = self.gate.generation
            res.reason = self.gate.reason
        except ValueError as exc:
            res.reason = str(exc)
        return res

    def receive(self, msg):
        try:
            if msg.clock_id != self.clock_id:raise ValueError('Wrong host clock domain')
            self.command = self.gate.accept(msg,maximum_deadline=msg.not_after_monotonic_s)
        except ValueError:
            # A rejected packet never extends the last valid command.
            pass

    def output(self):
        v,w,end = self.command
        now = time.monotonic()
        if now-self.progress > .3 or now >= end or not self.gate.live:
            return Twist()
        msg = Twist()
        msg.linear.x, msg.angular.z = v,w
        return msg
