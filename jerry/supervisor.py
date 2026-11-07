"""Independent final motion authority. No Torch, model SDK or GUI imports."""
from dataclasses import replace
import math
import time
from uuid import uuid4
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Trigger
from sensor_msgs.msg import Range
from rosgraph_msgs.msg import Clock
from jerry_interfaces.msg import MotionCommand, NavigationState, TaskState
from jerry_interfaces.srv import Authority
from .authority import AuthorityGate, Binding, collision_gate, host_clock_id
from .ros_utils import authority_request, wall_timer


class Supervisor(Node):
    def __init__(self):
        super().__init__('jerry_motion_supervisor')
        self.owner = 'supervisor-'+uuid4().hex
        self.gate = AuthorityGate()
        self.clock_id = host_clock_id()
        self.session = ''
        self.epoch = 0
        self.ranges = {}
        self.nav = None
        self.nav_received = -math.inf
        self.sim = None
        self.progress = -math.inf
        self.command = (0.,0.,0.)
        self.mode = 'HOLD'
        self.backend = None
        self.backend_token = ''
        self.backend_end = 0.
        self.pending = None
        self.pending_started = 0.
        self.pending_operation = ''
        self.sequence = 0
        self.reason = 'Waiting for protected adapter'
        self.resource_hold = False
        self.barrier_since = None
        self.barrier = False
        self.barrier_revision = None
        self.pub = self.create_publisher(MotionCommand, '/jerry/actuation/command', 1)
        self.state_pub = self.create_publisher(TaskState, '/jerry/motion/state', 1)
        self.backend_client = self.create_client(Authority, '/jerry/adapter/authority')
        self.create_service(Authority, '/jerry/motion/authority', self.authority)
        self.create_service(Trigger, '/jerry/motion/stop', self.stop_service)
        self.create_subscription(MotionCommand, '/jerry/motion/proposal', self.proposal, 1)
        self.create_subscription(NavigationState, '/jerry/navigation/state', self.navigation, 1)
        self.create_subscription(Clock, '/clock', self.clock, qos_profile_sensor_data)
        for side in ('front','left','rear','right'):
            self.create_subscription(Range, '/ultrasonic/'+side,
                lambda m,s=side:self.ranges.__setitem__(s,(m.range,time.monotonic(),m.header.stamp.sec+m.header.stamp.nanosec*1e-9,m.max_range)), qos_profile_sensor_data)
        wall_timer(self,.02,self.tick)
        wall_timer(self,.05,self.backend_tick)

    def navigation(self, msg):
        self.nav, self.nav_received = msg, time.monotonic()

    def clock(self, msg):
        stamp = msg.clock.sec+msg.clock.nanosec*1e-9
        if self.sim is not None and stamp < self.sim-.01:
            self.stop('Simulation reset')
            self.session = ''
        if self.sim is None or stamp != self.sim:
            self.progress = time.monotonic()
        self.sim = stamp

    def stop(self, reason):
        self.gate.revoke(reason)
        self.command = (0.,0.,0.)
        self.reason = reason
        self.barrier = False
        self.backend_end = 0.
        if self.session and self.backend_client.service_is_ready():
            self.backend_client.call_async(authority_request('REVOKE', session_id=self.session))
        self.backend = None

    def stop_service(self, req, res):
        self.stop('Operator Stop')
        res.success, res.message = True, 'Local stop latched; adapter revocation requested; expiry remains enforced'
        return res

    def authority(self, req, res):
        res.session_id, res.epoch_id = self.session, self.epoch
        try:
            if req.operation == 'STATUS':
                pass
            elif req.operation == 'STOP':
                self.stop('Executive stop')
            elif req.operation == 'ACQUIRE':
                if not self.session or req.session_id != self.session or req.epoch_id != self.epoch:
                    raise ValueError('Adapter not ready or wrong session/epoch')
                if time.monotonic()-self.progress > .3:
                    raise ValueError('Simulation paused')
                self.gate.acquire(self.session,self.epoch,req.owner_id,req.task_id,req.map_revision)
                self.barrier = False
                self.resource_hold = False
                self.reason = 'Mission admitted; waiting for valid proposals'
            elif req.operation in ('HOLD','RESUME'):
                if not self.gate.matches(req):raise ValueError('No matching hold authority')
                self.resource_hold = req.operation == 'HOLD'
                self.command = (0.,0.,0.)
            elif req.operation == 'RENEW':
                if self.barrier and req.map_revision != self.barrier_revision:
                    if self.barrier_revision is None or req.map_revision != self.barrier_revision+1:
                        raise ValueError('Correction barrier not ready')
                    if not self.nav or self.nav.map_revision != req.map_revision or not self.nav.plan_ready:
                        raise ValueError('Corrected route not ready')
                    if self.backend is not None:
                        raise ValueError('Adapter revocation pending')
                    self.gate.binding = replace(self.gate.binding,map_revision=req.map_revision)
                    self.barrier = False
                self.gate.renew(req)
            elif req.operation == 'TOKEN':
                if self.barrier:
                    raise ValueError('Correction barrier active')
                res.token, res.valid_for_s = self.gate.issue(req)
            elif req.operation == 'BARRIER':
                if not self.gate.matches(req):
                    raise ValueError('Barrier has no matching task')
                if not self.barrier:self.barrier_since = None
                self.barrier = True
                self.barrier_revision = req.map_revision
                self.gate.tokens.clear()
                self.command = (0.,0.,0.)
                self.backend_end = 0.
                if self.backend is not None:
                    if self.pending is not None:
                        raise ValueError('Retry barrier after pending adapter operation')
                    future = self.backend_client.call_async(authority_request('REVOKE',session_id=self.session))
                    self.pending = future
                    def acknowledged(f):
                        self.pending = None
                        if f.result().accepted:
                            self.backend = None
                    future.add_done_callback(acknowledged)
                    raise ValueError('Awaiting adapter zero/revocation acknowledgement')
                # Token expiry plus this stationary interval establishes simulator quiescence.
                if not hasattr(self,'barrier_since') or self.barrier_since is None:
                    self.barrier_since = time.monotonic()
                if time.monotonic()-self.barrier_since < .5:
                    raise ValueError('Awaiting conservative simulator settling interval')
            else:
                raise ValueError('Unknown authority operation')
            res.accepted = True
            res.reason = self.reason
        except (ValueError,TypeError) as exc:
            res.reason = str(exc)
        res.generation = self.gate.generation
        return res

    def proposal(self, msg):
        try:
            self.command = self.gate.accept(msg)
            self.mode = msg.mode
        except ValueError as exc:
            self.reason = str(exc)

    def backend_tick(self):
        if self.pending is not None:
            if time.monotonic()-self.pending_started <= .3:
                return
            expired=self.pending
            self.pending=None
            expired.cancel()  # local bookkeeping only; never proof of a server-side undo
            if self.pending_operation != 'STATUS':
                self.stop('Adapter authority response deadline; revocation/reconciliation required')
                return
        if not self.backend_client.service_is_ready():
            return
        if not self.session:
            req = authority_request('STATUS')
        elif self.barrier or not self.gate.live:
            return
        elif self.backend is None:
            req = authority_request('CLAIM',session_id=self.session,epoch_id=self.epoch,
                owner_id=self.owner,task_id=self.gate.binding.task_id,map_revision=self.gate.binding.map_revision)
        else:
            req = authority_request('TOKEN', self.backend)
        upstream_binding = self.gate.binding
        started = time.monotonic()
        future = self.backend_client.call_async(req)
        self.pending = future
        self.pending_started = started
        self.pending_operation = req.operation
        def done(f):
            if self.pending is not future:return
            self.pending = None
            try:response = f.result()
            except Exception:
                self.stop('Adapter authority response failed')
                return
            if req.operation == 'STATUS':
                self.session,self.epoch = response.session_id,response.epoch_id
                return
            # Responses from an old task cannot restore a revoked backend.
            if (not self.gate.live or self.barrier or self.gate.binding != upstream_binding
                    or req.task_id != self.gate.binding.task_id):
                return
            if not response.accepted:
                self.stop('Adapter rejected authority: '+response.reason)
                return
            if req.operation == 'CLAIM':
                self.backend = Binding(self.session,self.epoch,self.owner,req.task_id,response.generation,req.map_revision)
            elif req.operation == 'TOKEN':
                self.backend_token = response.token
                self.backend_end = started+response.valid_for_s
        future.add_done_callback(done)

    def tick(self):
        now = time.monotonic()
        v,w,end = self.command
        b = self.gate.binding
        if self.gate.binding is not None and not self.gate.live:
            self.stop('Motion lease expired')
        if self.gate.live and now-self.progress > .3:
            self.stop('Simulation clock stopped; explicit continuation required')
        if (not self.gate.live or self.barrier or self.resource_hold or now >= end or not self.nav
                or now-self.nav_received > .3 or not self.nav.ready or not self.nav.plan_ready
                or self.nav.capacity_reached or self.nav.session_id != self.session
                or self.nav.epoch_id != self.epoch or not b or self.nav.map_revision != b.map_revision):
            v=w=0.
            self.reason = self.gate.reason if not self.gate.live else 'Holding: navigation/freshness/revision gate'
        elif self.sim is not None:
            pose_age = self.sim-self.nav.sensor_stamp.sec-self.nav.sensor_stamp.nanosec*1e-9
            depth_age = self.sim-self.nav.depth_stamp.sec-self.nav.depth_stamp.nanosec*1e-9
            if not -.1 <= pose_age <= .30 or not -.1 <= depth_age <= .45:
                v=w=0.
                self.reason = 'Pose/depth acquisition stale'
            else:
                v,w,self.reason = collision_gate(v,w,self.ranges,now,self.sim)
        sent_motion=False
        if self.backend and now < self.backend_end:
            self.sequence += 1
            msg = MotionCommand(**self.backend.__dict__,sequence=self.sequence,token=self.backend_token,
                linear_m_s=v,angular_rad_s=w,mode=('HOLD' if v==w==0 else self.mode),
                clock_id=self.clock_id,not_after_monotonic_s=min(self.gate.expiry,end))
            self.pub.publish(msg)
            sent_motion=bool(v or w)
        state = TaskState(session_id=self.session,epoch_id=self.epoch,generation=self.gate.generation,
                          task_id=b.task_id if b else '',status='ARMED' if self.gate.live else 'STOPPED',reason=self.reason,
                          motion_enabled=sent_motion)
        self.state_pub.publish(state)
