"""Local speech/vision interaction. Codex is the external planner through MCP."""
import base64
from dataclasses import asdict
import json
import math
import re
import threading
import time
from uuid import uuid4
import cv2
import numpy as np
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, SetBool
from jerry_interfaces.msg import NavigationState, TaskState
from jerry_interfaces.srv import Ask, SubmitTask, Journal, Audio
from .model_client import ModelClient
from .ros_utils import wait_future
from .speech import Speech

STOP = re.compile(r'^(?:jerry[ ,]+)?(?:stop(?: the robot| moving| exploring)?|halt|emergency stop|cancel(?: the)?(?: task| mission)?)[.! ]*$',re.I)
EXPLORE = re.compile(r'^(?:jerry[ ,]+)?(?:please )?(?:explore(?: the)?(?: room| area)?|start explor(?:ing|ation)|continue explor(?:ing|ation)|resume explor(?:ing|ation))[.! ]*$',re.I)


class Cognition(Node):
    def __init__(self,speak=True,offline=False):
        super().__init__('jerry_cognition')
        self.group = ReentrantCallbackGroup()
        self.tasks = self.create_client(SubmitTask,'/jerry/tasks/submit',callback_group=self.group)
        self.cancel = self.create_client(Trigger,'/jerry/tasks/cancel',callback_group=self.group)
        self.stop_motion = self.create_client(Trigger,'/jerry/motion/stop',callback_group=self.group)
        self.hold_motion = self.create_client(SetBool,'/jerry/tasks/hold',callback_group=self.group)
        self.memory = self.create_client(Journal,'/jerry/memory/journal',callback_group=self.group)
        self.create_service(Ask,'/jerry/dialogue/ask',self.ask,callback_group=self.group)
        self.create_service(Audio,'/jerry/audio',self.audio_request,callback_group=self.group)
        self.create_subscription(Image,'/camera/image_raw',self.image,qos_profile_sensor_data)
        self.create_subscription(NavigationState,'/jerry/navigation/state',lambda m:setattr(self,'nav',m),1)
        self.create_subscription(TaskState,'/jerry/tasks/state',self.task_state,10)
        self.nav = self.task = self.frame = None
        self.frame_received = -math.inf
        self.generation = 0
        self.turn = threading.Lock()
        self.context_lock = threading.RLock()
        self.worker = ModelClient()
        self.audio = ModelClient('capture/asr')
        self.speech = Speech() if speak else None
        self.offline = offline
        self.last_terminal = None
        self.last_evidence = []

    def image(self,msg):
        self.frame,self.frame_received = msg,time.monotonic()

    def task_state(self,msg):
        self.task = msg
        if msg.status in ('SUCCEEDED','FAILED','CANCELLED','EXPIRED') and msg.task_id != self.last_terminal:
            self.last_terminal = msg.task_id
            text = 'Mission '+msg.status.lower()+': '+msg.reason
            print('Jerry: '+text,flush=True)
            if self.speech:self.speech.say(text)

    def interrupt(self):
        with self.context_lock:
            self.generation += 1
            if self.speech:self.speech.stop()
        # Both authorities get the request: direct stop remains useful if executive has failed.
        if self.stop_motion.service_is_ready():self.stop_motion.call_async(Trigger.Request())
        if self.cancel.service_is_ready():self.cancel.call_async(Trigger.Request())
        return 'Stop requested directly; task cancellation and adapter acknowledgement are reported separately.'

    def status(self):
        return dict(task=dict(id=self.task.task_id,status=self.task.status,reason=self.task.reason) if self.task else None,
            navigation=dict(ready=self.nav.ready,phase=self.nav.phase,reason=self.nav.reason,observed_m2=self.nav.observed_m2,
                map_revision=self.nav.map_revision) if self.nav else None)

    def journal(self,**kw):
        if not self.memory.service_is_ready():
            raise RuntimeError('Persistent memory unavailable')
        result = wait_future(self.memory.call_async(Journal.Request(**kw)),2.)
        if not result.accepted:
            raise RuntimeError(result.reason)
        return result

    def explore(self,request_id,expected_generation):
        if not self.task or not self.tasks.service_is_ready():
            raise RuntimeError('Executive not ready')
        req = SubmitTask.Request(session_id=self.task.session_id,expected_generation=expected_generation,
            request_id=request_id,kind='explore',description='Explore the observed reachable area',
            maximum_active_s=120.,maximum_wall_s=300.,maximum_distance_m=10.)
        result = wait_future(self.tasks.call_async(req),4.)
        if not result.accepted:
            raise RuntimeError(result.reason)
        return dict(status=result.status,task_id=result.task_id,message=result.reason)

    def observe(self,question,cancelled):
        if not self.frame or time.monotonic()-self.frame_received>.65:
            raise RuntimeError('Live camera unavailable or stale')
        held = False
        try:
            if self.task and self.task.status=='RUNNING':
                if not self.hold_motion.service_is_ready():
                    raise RuntimeError('Cannot establish model-resource motion hold')
                response = wait_future(self.hold_motion.call_async(SetBool.Request(data=True)),2.)
                if not response.success:
                    raise RuntimeError(response.message)
                held = True
            msg = self.frame
            frame_id = f'{self.task.session_id if self.task else "unknown"}:{msg.header.frame_id}:{msg.header.stamp.sec}:{msg.header.stamp.nanosec}'
            channels = 4 if msg.encoding in ('rgba8','bgra8') else 3
            if msg.encoding not in ('rgb8','bgr8','rgba8','bgra8'):
                raise ValueError('Unsupported camera encoding')
            bgr = np.frombuffer(msg.data,np.uint8).reshape(msg.height,msg.step)[:,:msg.width*channels].reshape(msg.height,msg.width,channels)[:,:,:3]
            if msg.encoding.startswith('rgb'):bgr = bgr[:,:,::-1]
            ok,encoded = cv2.imencode('.png',bgr)
            if not ok:raise RuntimeError('Snapshot encoding failed')
            result = self.worker.call('observe',question=question,image=base64.b64encode(encoded).decode(),frame_id=frame_id,
                timeout=90.,cancelled=cancelled)
            if cancelled():raise RuntimeError('Observation cancelled')
            answer = result.get('raw_answer') or result.get('answer') or ''
            if result.get('status') not in ('completed','ok','success') or not answer:
                raise RuntimeError(result.get('error') or 'VLM did not return valid evidence')
            evidence_id = uuid4().hex
            record = self.journal(operation='APPEND',request_id=evidence_id,kind='observation',text=answer,
                evidence_id=frame_id,session_id=self.task.session_id if self.task else '',epoch_id=self.nav.epoch_id if self.nav else 0,
                map_revision=self.nav.map_revision if self.nav else 0)
            return dict(answer=answer,evidence_id=record.record_id,frame_id=frame_id,
                capture_sim_s=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,current_geometry=False)
        finally:
            if held and not cancelled():
                self.hold_motion.call_async(SetBool.Request(data=False))

    def answer(self,text,request_id):
        text = text.strip()
        if not text or len(text)>2000:
            raise ValueError('Request must contain 1–2000 characters')
        if STOP.fullmatch(text):
            return self.interrupt(),[]
        if not self.turn.acquire(blocking=False):
            raise RuntimeError('A dialogue request is already running; Stop remains available')
        with self.context_lock:
            generation = self.generation
            session = self.task.session_id if self.task else ''
            expected = self.task.generation if self.task else 0
        cancelled = lambda:self.generation!=generation or (self.task is not None and self.task.session_id!=session)
        try:
            self.last_evidence = []
            if EXPLORE.fullmatch(text):
                return 'Exploration admitted: '+json.dumps(self.explore(request_id,expected)),[]
            if text.lower().rstrip('?.!') in ('status','how are you','what are you doing'):
                return json.dumps(self.status()),[]
            if text.lower().startswith('remember that '):
                if cancelled():raise RuntimeError('Request cancelled')
                record = self.journal(operation='APPEND',request_id=request_id,kind='explicit_fact',text=text[14:],session_id=session)
                return 'Remembered your explicit statement.',[record.record_id]
            if any(word in text.lower() for word in ('earlier','remember','memory','previously')):
                result = self.journal(operation='QUERY',text='',limit=8)
                return '\n'.join(f'{r.created_at}: {r.text} [evidence {r.evidence_id or r.record_id}]' for r in result.records) or 'No saved observations yet.',[r.record_id for r in result.records]
            if any(phrase in text.lower() for phrase in ('what do you see','what can you see','describe the scene','look at','what is in front')):
                result = self.observe(text,cancelled)
                return f'From camera frame captured at simulation time {result["capture_sim_s"]:.2f}s: {result["answer"]}',[result['evidence_id']]
            return 'Use Codex with the Jerry MCP tools for reasoning and mission planning. Local commands: explore the room, stop, status, what can you see, remember that …, and what did you find earlier.',[]

        finally:
            self.turn.release()

    def ask(self,req,res):
        try:
            text = req.text
            if text == '/listen':
                if self.speech:self.speech.stop()
                generation = self.generation
                cancelled = lambda:self.generation!=generation
                recording = self.audio.call('capture',timeout=20.,cancelled=cancelled)
                text = self.audio.call('transcribe',audio=recording['audio'],timeout=60.,cancelled=cancelled)['text']
                print('Heard: '+text,flush=True)
            answer,evidence = self.answer(text,req.request_id or uuid4().hex)
            res.accepted,res.text,res.evidence_ids = True,answer,evidence
            if self.speech and not STOP.fullmatch(text):self.speech.say(answer)
        except Exception as exc:
            res.text = str(exc)
        return res

    def audio_request(self,req,res):
        try:
            if req.operation == 'STATUS':
                if not self.speech:
                    res.status = 'DISABLED'
                elif self.speech.last_error:
                    res.status,res.text = 'FAILED',self.speech.last_error
                elif self.speech.process and self.speech.process.poll() is None:
                    res.status = 'PLAYING'
                elif self.speech.thread and self.speech.thread.is_alive():
                    res.status = 'SYNTHESIZING'
                else:
                    res.status = 'IDLE'
            elif req.operation == 'STOP':
                self.generation += 1
                if self.speech:self.speech.stop()
                res.status = 'CANCELLED'
            elif req.operation == 'SPEAK':
                if not self.speech:raise RuntimeError('Speech disabled in this runtime')
                if not req.text.strip() or len(req.text)>1000:raise ValueError('Speech must contain 1–1000 characters')
                if not self.speech.say(req.text):raise RuntimeError('Speech worker is cancelling previous synthesis; retry shortly')
                res.status = 'QUEUED'
            elif req.operation == 'LISTEN':
                if self.speech:self.speech.stop()
                generation = self.generation
                cancelled = lambda:self.generation!=generation
                recording = self.audio.call('capture',timeout=20.,cancelled=cancelled)
                res.text = self.audio.call('transcribe',audio=recording['audio'],timeout=60.,cancelled=cancelled)['text']
                # A transcript is returned to Codex; it never implicitly starts a mission here.
                res.status = 'TRANSCRIBED'
                if STOP.fullmatch(res.text.strip()):
                    self.interrupt()
                    res.status = 'TRANSCRIBED_STOP_REQUESTED'
            else:
                raise ValueError('Unknown audio operation')
            res.accepted = True
        except Exception as exc:
            res.status,res.text = 'FAILED',str(exc)
        return res

    def close(self):
        self.generation += 1
        self.worker.close()
        self.audio.close()
        if self.speech:self.speech.close()
