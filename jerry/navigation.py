"""ROS mission adapter around the measured floor/compass exploration baseline."""
from dataclasses import replace
import math
import threading
import time
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from jerry_interfaces.action import Mission
from jerry_interfaces.msg import MotionCommand, NavigationState, TaskState
from jerry_interfaces.srv import Authority
from builtin_interfaces.msg import Time
from .authority import Binding
from .progress import MissionProgress
from .ros_utils import authority_request, wall_timer
from realtime_depth_app.robot_live import RobotNode
from realtime_depth_app.src.robot_avoidance import Decision


class ProposalPublisher:
    def __init__(self,node):
        self.node = node
        self.pub = node.create_publisher(MotionCommand,'/jerry/motion/proposal',1)
        self.sequence = 0

    def publish(self,twist):
        node = self.node
        if not node.binding or time.monotonic() >= node.token_end:
            return
        self.sequence += 1
        v,w = twist.linear.x,twist.angular.z
        # Apply kinematic constraints after smoothing; supervisor independently checks again.
        if node.decision.state in ('ALIGN','RESCAN','LOOK_AROUND','TURN','RELOCALIZE','REORIENT'):
            v = 0.
        if node.decision.state == 'TRACKING_BRIDGE':
            v=w=0.
        v = max(-.035,min(.08,v))
        w = max(-.4,min(.4,w))
        mode = 'HOLD' if v==w==0 else ('REVERSE' if v<0 else ('PIVOT' if v==0 else 'TRACK'))
        self.pub.publish(MotionCommand(**node.binding.__dict__,sequence=self.sequence,token=node.token,
            linear_m_s=v,angular_rad_s=w,mode=mode))


class Navigation(RobotNode):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,auto=False,mode='explore',**kwargs)
        self.destroy_publisher(self.command_pub)
        self.command_pub = ProposalPublisher(self)
        self.group = ReentrantCallbackGroup()
        self.authority = self.create_client(Authority,'/jerry/motion/authority',callback_group=self.group)
        self.binding = None
        self.session = ''
        self.epoch_id = 0
        self.token = ''
        self.token_end = 0.
        self.token_future = None
        self.barrier_future = None
        self.correction_ready = threading.Event()
        self.allowed_correction_revision = None
        self.slam.engine.correction_guard = self.consume_correction
        self.current_goal = None
        self.goal_ids = set()
        self.create_subscription(TaskState,'/jerry/motion/state',self.motion_state,1)
        self.nav_pub = self.create_publisher(NavigationState,'/jerry/navigation/state',1)
        self.action = ActionServer(self,Mission,'/jerry/navigation/mission',execute_callback=self.execute,
            goal_callback=self.accept_goal,cancel_callback=lambda _:CancelResponse.ACCEPT,callback_group=self.group)
        wall_timer(self,.05,self.authorize)
        wall_timer(self,.1,self.publish_state)

    def consume_correction(self):
        if (self.correction_ready.is_set() and self.allowed_correction_revision == self.slam.engine.map_revision):
            self.correction_ready.clear()
            self.allowed_correction_revision = None
            return True
        return False

    def motion_state(self,msg):
        with self.lock:
            if self.session and (msg.session_id != self.session or msg.epoch_id != self.epoch_id):
                self.active = False
                self.binding = None
                # The SLAM worker also resets from acquisition timestamps; never publish old ready state meanwhile.
            self.session,self.epoch_id = msg.session_id,msg.epoch_id
            if self.binding and (msg.task_id != self.binding.task_id or msg.generation != self.binding.generation or msg.status!='ARMED'):
                self.active = False
                self.binding = None

    def arm(self,enabled,why='Use the integrated task executive'):
        # Legacy dashboard/service cannot acquire an integrated mission lease.
        if not enabled:
            self.active = False
            self.start_requested = False
            self.monitor_reason = why
        return

    def accept_goal(self,goal):
        if (self.current_goal or goal.task_id in self.goal_ids or goal.kind!='explore'
                or goal.session_id != self.session or goal.epoch_id != self.epoch_id):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def execute(self,handle):
        goal = handle.request
        with self.lock:
            self.current_goal = handle
            self.goal_ids.add(goal.task_id)
            if len(self.goal_ids)>1024:
                handle.abort()
                self.current_goal = None
                return Mission.Result(status='FAILED',reason='Task deduplication capacity reached; restart disarmed')
            self.binding = Binding(goal.session_id,goal.epoch_id,'navigation',goal.task_id,goal.generation,goal.map_revision)
            self.active = True
            self.curiosity.reset()
        started = time.monotonic()
        progress = MissionProgress(started,timeout=65.)
        result = Mission.Result(status='FAILED',reason='Navigation interrupted')
        try:
            while time.monotonic()-started < min(goal.maximum_wall_s,300.):
                if handle.is_cancel_requested:
                    handle.canceled()
                    return Mission.Result(status='CANCELLED',reason='Motion task cancelled')
                with self.lock:
                    if not self.binding or not self.active:
                        handle.abort()
                        return result
                    snapshot = self.slam.snapshot()
                    if snapshot.get('capacity_reached'):
                        handle.abort()
                        return Mission.Result(status='FAILED',reason='MAP_CAPACITY_REACHED')
                    if not progress.update(snapshot,time.monotonic()):
                        handle.abort()
                        return Mission.Result(status='FAILED',reason='NO_PROGRESS: bounded recovery exhausted')
                    if (time.monotonic()-started>5 and snapshot['planner_state']=='NO_REACHABLE_FRONTIER'
                            and snapshot['completed_goals']>0):
                        handle.succeed()
                        return Mission.Result(status='SUCCEEDED',reason='NO_REACHABLE_FRONTIERS; whole-world coverage not implied')
                    handle.publish_feedback(Mission.Feedback(phase=self.decision.state,reason=self.decision.reason))
                time.sleep(.2)
            handle.abort()
            return Mission.Result(status='EXPIRED',reason='Navigation wall budget reached')
        finally:
            with self.lock:
                self.active = False
                self.binding = None
                self.current_goal = None
                self.motion.reset()

    def authorize(self):
        with self.lock:
            if not self.binding or not self.authority.service_is_ready():
                return
            snapshot = self.slam.snapshot()
            if snapshot.get('correction_pending'):
                self.correction_ready.clear()
                if self.barrier_future is None or self.barrier_future.done():
                    if self.barrier_future and self.barrier_future.result().accepted:
                        self.allowed_correction_revision = self.binding.map_revision
                        self.correction_ready.set()
                        return
                    self.barrier_future = self.authority.call_async(authority_request('BARRIER',self.binding))
                return
            if snapshot.get('map_revision',0) != self.binding.map_revision:
                self.binding = replace(self.binding,map_revision=snapshot['map_revision'])
                self.correction_ready.clear()
                self.barrier_future = None
            if self.token_future is None or self.token_future.done():
                started = time.monotonic()
                binding = self.binding
                future = self.authority.call_async(authority_request('TOKEN',binding))
                self.token_future = future
                def done(f):
                    with self.lock:
                        response = f.result()
                        if self.binding == binding and response.accepted:
                            self.token,self.token_end = response.token,started+response.valid_for_s
                future.add_done_callback(done)

    def publish_state(self):
        with self.lock:
            s = self.slam.snapshot()
            a,_,_ = self.navigation_snapshot()
            ready = bool(s['ready'] and a['ready'] and self.prediction and not self.worker.error
                         and not s.get('correction_pending') and time.monotonic()-self.clock_received<.3)
            stamp = s.get('stamp') or 0.
            sensor = Time(sec=int(stamp),nanosec=int((stamp-int(stamp))*1e9))
            depth = self.prediction.packet.stamp if self.prediction else Time()
            self.nav_pub.publish(NavigationState(session_id=self.session,epoch_id=self.epoch_id,
                map_revision=s.get('map_revision',0),occupancy_revision=s.get('revision',0),sensor_stamp=sensor,depth_stamp=depth,
                ready=ready,plan_ready=bool(s.get('plan_ready')),capacity_reached=s.get('capacity_reached',False),
                x=float(s['pose'][0]),y=float(s['pose'][1]),yaw=float(s['pose'][2]),observed_m2=s['observed_m2'],
                completed_goals=s['completed_goals'],phase=self.decision.state,reason=self.decision.reason))
