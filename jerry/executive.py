"""Deterministic mission budgets, admission and cancellation, independent of cognition."""
from collections import deque
import json
import math
import threading
import time
from uuid import uuid4
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from std_srvs.srv import Trigger, SetBool
from jerry_interfaces.msg import TaskState, NavigationState
from jerry_interfaces.srv import SubmitTask, Authority, Journal
from jerry_interfaces.action import Mission
from .authority import Binding
from .ros_utils import authority_request, wait_future, wall_timer


class Executive(Node):
    def __init__(self):
        super().__init__('jerry_executive')
        self.group = ReentrantCallbackGroup()
        self.motion = self.create_client(Authority,'/jerry/motion/authority',callback_group=self.group)
        self.stop_client = self.create_client(Trigger,'/jerry/motion/stop',callback_group=self.group)
        self.memory = self.create_client(Journal,'/jerry/memory/journal',callback_group=self.group)
        self.navigation = ActionClient(self,Mission,'/jerry/navigation/mission',callback_group=self.group)
        self.create_service(SubmitTask,'/jerry/tasks/submit',self.submit,callback_group=self.group)
        self.create_service(Trigger,'/jerry/tasks/cancel',self.cancel_service,callback_group=self.group)
        self.create_service(SetBool,'/jerry/tasks/hold',self.hold_service,callback_group=self.group)
        self.pub = self.create_publisher(TaskState,'/jerry/tasks/state',10)
        self.create_subscription(NavigationState,'/jerry/navigation/state',self.nav_state,1)
        self.create_subscription(TaskState,'/jerry/motion/state',self.motion_state,1)
        self.lock = threading.RLock()
        self.admission = threading.Lock()
        self.session = ''
        self.epoch = self.generation = 0
        self.nav = None
        self.task = None
        self.binding = None
        self.goal = None
        self.renewal = None
        self.journal_pending = deque(maxlen=32)
        self.journal_future = None
        self.outage_since = None
        self.completed = {}
        wall_timer(self,.1,self.tick)

    def nav_state(self,msg):
        self.nav = msg

    def motion_state(self,msg):
        with self.lock:
            if self.session and (msg.session_id != self.session or msg.epoch_id != self.epoch):
                self.finish('FAILED','Simulation session/epoch changed')
                self.generation += 1
                self.completed.clear()
            self.session,self.epoch = msg.session_id,msg.epoch_id
            if (self.task and msg.status == 'STOPPED' and self.binding
                    and msg.generation >= self.binding.generation):
                self.generation += 1
                self.finish('CANCELLED','Motion authority revoked: '+msg.reason)

    def record(self,operation,**kw):
        if len(self.journal_pending) >= self.journal_pending.maxlen:
            self.stop_client.call_async(Trigger.Request())
            raise RuntimeError('Pending journal capacity reached; motion stopped')
        self.journal_pending.append(Journal.Request(operation=operation,**kw))

    def finish(self,status,reason):
        if not self.task:
            return
        task = self.task
        self.task = None
        self.binding = None
        self.renewal = None
        if self.goal:
            self.goal.cancel_goal_async()
        self.goal = None
        self.stop_client.call_async(Trigger.Request())
        self.record('TRANSITION',task_id=task['id'],kind=status,text=reason)
        self.record('APPEND',request_id=uuid4().hex,task_id=task['id'],kind='task_result',text=status+': '+reason,
                    session_id=self.session,epoch_id=self.epoch,map_revision=task['map_revision'])
        state = TaskState(session_id=self.session,epoch_id=self.epoch,generation=self.generation,
            task_id=task['id'],kind=task['kind'],status=status,reason=reason,
            active_seconds=task['active'],travelled_m=task['distance'])
        self.pub.publish(state)
        self.completed[task['id']] = state
        while len(self.completed)>32:
            self.completed.pop(next(iter(self.completed)))

    def cancel_service(self,req,res):
        with self.lock:
            self.generation += 1
            self.stop_client.call_async(Trigger.Request())
            self.finish('CANCELLED','Operator cancelled task')
            res.success,res.message = True,'Motion revocation requested; task cancelled'
        return res

    def hold_service(self,req,res):
        with self.lock:
            binding = self.binding
        if binding is None:
            res.success,res.message = True,'No motion task'
            return res
        try:
            response = wait_future(self.motion.call_async(authority_request('HOLD' if req.data else 'RESUME',binding)),1.)
            res.success,res.message = response.accepted,response.reason
            if req.data and response.accepted:
                time.sleep(.3)  # Allow the independent actuator token window to expire before model admission.
        except Exception as exc:
            res.message = str(exc)
        return res

    def submit(self,req,res):
        if not self.admission.acquire(blocking=False):
            res.reason = 'Another admission is in progress'
            return res
        try:
            with self.lock:
                if req.session_id != self.session or req.expected_generation != self.generation or not self.session:
                    raise ValueError('Stale request session/generation; request fresh status')
                if req.kind not in ('explore','navigate','find_object','look_around'):
                    raise ValueError('Unsupported mission kind')
                if req.kind == 'navigate':
                    raise ValueError('Metric target navigation is not enabled until grounded target/route validation passes')
                if req.kind in ('find_object','look_around'):
                    raise ValueError('Semantic search and observation pivots are not yet enabled in this integration profile')
                if self.task:
                    raise ValueError('A mission is active; cancel it before starting another')
                if not req.request_id or len(req.request_id)>128 or len(req.description)>500:
                    raise ValueError('Invalid request identity/description')
                limits = (req.maximum_active_s,req.maximum_wall_s,req.maximum_distance_m)
                if not all(math.isfinite(v) and v>0 for v in limits):
                    raise ValueError('Mission budgets must be positive finite values')
                if not self.nav or not self.nav.ready or self.nav.capacity_reached:
                    raise ValueError('Localization/perception not ready')
                if not self.navigation.server_is_ready() or not self.memory.service_is_ready() or self.journal_pending:
                    raise ValueError('Navigation or durable memory not ready')
                generation = self.generation
                payload = dict(session_id=self.session,epoch_id=self.epoch,map_revision=self.nav.map_revision,
                    kind=req.kind,description=req.description,maximum_active_s=min(req.maximum_active_s,120.),
                    maximum_wall_s=min(req.maximum_wall_s,300.),maximum_distance_m=min(req.maximum_distance_m,10.))
            response = wait_future(self.memory.call_async(Journal.Request(operation='ADMIT',request_id=req.request_id,
                session_id=req.session_id,text=json.dumps(payload))),timeout=2.)
            if not response.accepted:
                raise ValueError('Durable admission failed: '+response.reason)
            task_id = response.record_id
            with self.lock:
                if generation != self.generation or self.task or req.session_id != self.session:
                    self.record('TRANSITION',task_id=task_id,kind='CANCELLED',text='Request invalidated during admission')
                    raise ValueError('Request cancelled during durable admission')
                if response.reason != 'NEW':
                    res.task_id,res.status = task_id,response.reason
                    res.reason = 'Existing task reconciled; no redispatch'
                    return res
                # Admission authority is serialized with cancellation; the lease result is checked again below.
            authority = wait_future(self.motion.call_async(authority_request('ACQUIRE',session_id=self.session,
                epoch_id=self.epoch,owner_id='navigation',task_id=task_id,map_revision=payload['map_revision'])),timeout=1.)
            with self.lock:
                if not authority.accepted or generation != self.generation or req.session_id != self.session:
                    self.stop_client.call_async(Trigger.Request())
                    self.record('TRANSITION',task_id=task_id,kind='CANCELLED',text='Authority rejected or request invalidated')
                    raise ValueError(authority.reason or 'Request invalidated')
                self.binding = Binding(self.session,self.epoch,'navigation',task_id,authority.generation,payload['map_revision'])
                self.task = dict(payload,id=task_id,started=time.monotonic(),active=0.,distance=0.,last_stamp=None,last_pose=None,last_motion_pose=None,distance_window=[],distance_stamp=None)
                goal = Mission.Goal(session_id=self.session,epoch_id=self.epoch,generation=authority.generation,
                    task_id=task_id,kind=req.kind,description=req.description,maximum_wall_s=payload['maximum_wall_s'],
                    map_revision=payload['map_revision'])
                future = self.navigation.send_goal_async(goal)
                def admitted(f):
                    with self.lock:
                        handle = f.result()
                        if not self.task or self.task['id'] != task_id:
                            if handle.accepted:handle.cancel_goal_async()
                            return
                        if not handle.accepted:
                            self.finish('FAILED','Navigation rejected mission')
                            return
                        self.goal = handle
                        self.record('TRANSITION',task_id=task_id,kind='RUNNING',text='Navigation accepted')
                        handle.get_result_async().add_done_callback(lambda result:self.result(task_id,result))
                future.add_done_callback(admitted)
                res.accepted,res.task_id,res.status = True,task_id,'ADMITTED'
                res.reason = 'Bounded exploration admitted; no destination arrival claimed'
                res.generation = self.generation
        except Exception as exc:
            res.reason = str(exc)[:500]
        finally:
            self.admission.release()
        return res

    def result(self,task_id,future):
        with self.lock:
            if self.task and self.task['id'] == task_id:
                outcome = future.result().result
                self.finish(outcome.status if outcome.status in ('SUCCEEDED','FAILED','CANCELLED','EXPIRED') else 'FAILED',outcome.reason)

    def tick(self):
        with self.lock:
            now = time.monotonic()
            if self.journal_future and self.journal_future.done():
                response = self.journal_future.result()
                self.journal_future = None
                if response.accepted:
                    self.journal_pending.popleft()
                    self.outage_since = None
                elif self.outage_since is None:
                    self.outage_since = now
            if self.journal_pending and self.journal_future is None and self.memory.service_is_ready():
                self.journal_future = self.memory.call_async(self.journal_pending[0])
            if self.journal_pending and self.outage_since is None:
                self.outage_since = now
            task = self.task
            if task:
                if self.outage_since and now-self.outage_since>10:
                    self.finish('FAILED','Durable storage outage exceeded 10 seconds')
                    return
                n = self.nav
                if n:
                    stamp = n.sensor_stamp.sec+n.sensor_stamp.nanosec*1e-9
                    if task['last_stamp'] is not None:
                        task['active'] += max(0.,min(.3,stamp-task['last_stamp']))
                        if n.ready and n.map_revision == task['map_revision']:
                            if n.phase in ('EXPLORE','BACK_AWAY'):
                                task['distance_window'].append((n.x,n.y))
                                if len(task['distance_window'])>10:task['distance_window'].pop(0)
                                if task['distance_stamp'] is None or stamp-task['distance_stamp']>=.5:
                                    from statistics import median
                                    point=tuple(median(p[i] for p in task['distance_window']) for i in (0,1))
                                    anchor=task['last_motion_pose']
                                    if anchor is not None and math.dist(point,anchor)>=.02:
                                        task['distance']+=math.dist(point,anchor)
                                        task['last_motion_pose']=point
                                    elif anchor is None:task['last_motion_pose']=point
                                    task['distance_stamp']=stamp
                            else:
                                task['last_motion_pose']=None
                                task['distance_window'].clear()
                                task['distance_stamp']=None
                    task['last_stamp'],task['last_pose'] = stamp,(n.x,n.y)
                if (now-task['started']>=task['maximum_wall_s'] or task['active']>=task['maximum_active_s']
                        or task['distance']>=task['maximum_distance_m']):
                    self.finish('EXPIRED','Mission budget reached')
                    return
                if self.renewal is None or self.renewal.done():
                    if self.renewal is not None and not self.renewal.result().accepted:
                        reason = self.renewal.result().reason
                        if 'barrier' not in reason.lower() and 'Corrected route' not in reason:
                            self.finish('FAILED','Lease renewal failed: '+reason)
                            return
                    # Corrections use a new coordinate binding only after a fresh route exists.
                    if n and n.plan_ready and n.map_revision != self.binding.map_revision:
                        self.binding = Binding(**{**self.binding.__dict__,'map_revision':n.map_revision})
                        task['map_revision'] = n.map_revision
                    self.renewal = self.motion.call_async(authority_request('RENEW',self.binding))
                self.pub.publish(TaskState(session_id=self.session,epoch_id=self.epoch,generation=self.generation,
                    task_id=task['id'],kind=task['kind'],status='RUNNING',reason='Bounded mission running',
                    active_seconds=task['active'],travelled_m=task['distance']))
            elif self.completed:
                # Terminal state remains visible to late subscribers.
                from copy import deepcopy
                state=deepcopy(next(reversed(self.completed.values())))
                state.generation=self.generation
                self.pub.publish(state)
            else:
                self.pub.publish(TaskState(session_id=self.session,epoch_id=self.epoch,generation=self.generation,status='IDLE',reason='Awaiting an operator mission'))

    def close(self):
        with self.lock:
            self.finish('CANCELLED','Executive shutdown')
            self.stop_client.call_async(Trigger.Request())
