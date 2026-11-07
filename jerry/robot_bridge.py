"""MCP-to-ROS client. All motion still passes through the executive and supervisor."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4


def message_dict(message):
    if hasattr(message,'get_fields_and_field_types'):
        return {key:message_dict(getattr(message,key)) for key in message.get_fields_and_field_types()}
    if isinstance(message,(list,tuple)):
        return [message_dict(item) for item in message]
    return message


class RobotBridge:
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from nav_msgs.msg import OccupancyGrid
        from std_srvs.srv import Trigger,SetBool
        from jerry_interfaces.msg import TaskState,NavigationState
        from jerry_interfaces.srv import SubmitTask,Journal,Ask,Audio
        rclpy.init()
        self.node = Node('jerry_codex_mcp')
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.node)
        self.lock = threading.RLock()
        self.latest = {}
        self.terminals = {}
        self.closed = False
        self.owned_runtime = None
        self.runtime_log = None
        self.evidence = {}
        self.evidence_images = {}
        for topic,kind,key in (('/jerry/tasks/state',TaskState,'task'),('/jerry/navigation/state',NavigationState,'navigation'),
                               ('/jerry/motion/state',TaskState,'motion'),('/camera/image_raw',Image,'camera'),('/map',OccupancyGrid,'map')):
            self.node.create_subscription(kind,topic,lambda msg,k=key:self.receive(k,msg),qos_profile_sensor_data)
        self.clients = {
            'submit':self.node.create_client(SubmitTask,'/jerry/tasks/submit'),
            'stop':self.node.create_client(Trigger,'/jerry/motion/stop'),
            'cancel':self.node.create_client(Trigger,'/jerry/tasks/cancel'),
            'journal':self.node.create_client(Journal,'/jerry/memory/journal'),
            'ask':self.node.create_client(Ask,'/jerry/dialogue/ask'),
            'audio':self.node.create_client(Audio,'/jerry/audio'),
        }
        self.thread = threading.Thread(target=self.executor.spin,daemon=True,name='jerry-mcp-ros')
        self.thread.start()

    def receive(self,key,message):
        with self.lock:
            self.latest[key] = (message,time.monotonic())
            if key=='task' and message.status in ('SUCCEEDED','FAILED','CANCELLED','EXPIRED'):
                self.terminals[message.task_id] = (message,time.monotonic())
                while len(self.terminals)>32:self.terminals.pop(next(iter(self.terminals)))

    def fresh(self,key,max_age=1.):
        with self.lock:
            message,received = self.latest.get(key,(None,-math.inf))
            if message is None or time.monotonic()-received>max_age:
                raise RuntimeError(f'{key} unavailable/stale; start the Jerry simulation or wait for readiness')
            return message

    async def rpc(self,name,request,timeout=3.):
        client = self.clients[name]
        if not client.service_is_ready():
            raise RuntimeError(name+' service unavailable; start the Jerry simulation')
        future = client.call_async(request)
        end = time.monotonic()+timeout
        try:
            while not future.done():
                if time.monotonic()>=end:
                    raise TimeoutError(name+' deadline exceeded; side-effect status UNKNOWN, reconcile before retry')
                await asyncio.sleep(.01)
            return future.result()
        except BaseException:
            # Do not cancel ROS requests as proof that a side effect was undone.
            raise

    def status(self):
        with self.lock:
            result = {'backend':'gazebo','planner':'external_codex_mcp','ros_domain_id':os.environ.get('ROS_DOMAIN_ID','87'),
                      'capabilities':['camera','map','bounded_exploration','stop','task_status','memory','speech','listen_once'],
                      'services':{key:client.service_is_ready() for key,client in self.clients.items()}}
            for key in ('task','navigation','motion'):
                msg,received = self.latest.get(key,(None,-math.inf))
                result[key] = dict(state=message_dict(msg),receipt_age_s=time.monotonic()-received,
                                   fresh=time.monotonic()-received<1.) if msg else None
            return result

    async def explore(self,request_id,maximum_active_s,maximum_wall_s,maximum_distance_m):
        from jerry_interfaces.srv import SubmitTask
        task = self.fresh('task')
        self.fresh('motion')
        nav = self.fresh('navigation')
        if not nav.ready:
            raise RuntimeError('Navigation not ready: '+nav.reason)
        result = await self.rpc('submit',SubmitTask.Request(session_id=task.session_id,
            expected_generation=task.generation,request_id=request_id,kind='explore',description='Codex requested bounded exploration',
            maximum_active_s=maximum_active_s,maximum_wall_s=maximum_wall_s,maximum_distance_m=maximum_distance_m),5.)
        return message_dict(result)

    async def stop(self):
        from std_srvs.srv import Trigger
        from jerry_interfaces.srv import Audio
        # Dispatch both immediately. Model/audio work never sits ahead of motion revocation.
        direct = asyncio.create_task(self.rpc('stop',Trigger.Request(),1.))
        cancel = asyncio.create_task(self.rpc('cancel',Trigger.Request(),1.))
        audio = asyncio.create_task(self.rpc('audio',Audio.Request(operation='STOP'),1.))
        results = await asyncio.gather(direct,cancel,audio,return_exceptions=True)
        return {key:({'acknowledged':False,'error':str(result)} if isinstance(result,BaseException) else message_dict(result))
                for key,result in zip(('motion_stop','task_cancel','audio_stop'),results)}

    def camera(self):
        import cv2
        import numpy as np
        msg = self.fresh('camera',.65)
        nav = self.fresh('navigation')
        motion = self.fresh('motion')
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        pose_stamp = nav.sensor_stamp.sec+nav.sensor_stamp.nanosec*1e-9
        if abs(stamp-pose_stamp)>1.:
            raise RuntimeError('Camera acquisition not aligned with live sensor epoch')
        if msg.encoding not in ('rgb8','bgr8','rgba8','bgra8') or msg.width>1920 or msg.height>1080:
            raise ValueError('Unsupported image geometry/encoding')
        channels = 4 if msg.encoding in ('rgba8','bgra8') else 3
        array = np.frombuffer(msg.data,np.uint8).reshape(msg.height,msg.step)[:,:msg.width*channels].reshape(msg.height,msg.width,channels)[:,:,:3]
        if msg.encoding.startswith('rgb'):array=array[:,:,::-1]
        ok,image=cv2.imencode('.png',array)
        if not ok:raise RuntimeError('Camera encoding failed')
        evidence_id = uuid4().hex
        meta = dict(evidence_id=evidence_id,session_id=motion.session_id,epoch_id=motion.epoch_id,
                    map_revision=nav.map_revision,camera_frame=msg.header.frame_id,capture_sim_s=stamp,
                    geometry_status='IMAGE_ONLY',description='Captured camera evidence; not a metric depth measurement')
        self.evidence[evidence_id] = meta
        self.evidence_images[evidence_id] = image.tobytes()
        while len(self.evidence)>16:
            oldest=next(iter(self.evidence))
            self.evidence.pop(oldest)
            self.evidence_images.pop(oldest,None)
        return meta,image.tobytes()

    def map_image(self):
        import cv2
        import numpy as np
        msg = self.fresh('map',2.)
        values = np.asarray(msg.data).reshape(msg.info.height,msg.info.width)
        image = np.full(values.shape,127,np.uint8)
        image[values==0]=245
        image[values>=65]=20
        ok,encoded=cv2.imencode('.png',np.flipud(image))
        if not ok:raise RuntimeError('Map encoding failed')
        return dict(frame_id=msg.header.frame_id,capture_sim_s=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                    resolution_m=msg.info.resolution,origin=message_dict(msg.info.origin),
                    purpose='Visualization only; control uses authoritative navigation snapshots'),encoded.tobytes()

    def persist_evidence(self,evidence_id):
        meta = self.evidence.get(evidence_id)
        image = self.evidence_images.get(evidence_id)
        if not meta or image is None:
            raise ValueError('Evidence no longer retained; capture another frame')
        import hashlib
        import json
        directory = Path(__file__).resolve().parents[1]/'simulation/artifacts/jerry/evidence'
        directory.mkdir(parents=True,exist_ok=True)
        # Refuse new persistence at quota rather than deleting evidence referenced by old memories.
        used = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
        image_path = directory/(evidence_id+'.png')
        if not image_path.exists() and used+len(image)>256*1024*1024:
            raise RuntimeError('Evidence storage quota reached; no observation committed')
        image_path.write_bytes(image)
        manifest = {**meta,'sha256':hashlib.sha256(image).hexdigest(),'image_path':str(image_path)}
        (directory/(evidence_id+'.json')).write_text(json.dumps(manifest,indent=2))
        return manifest

    async def start_simulation(self,headless=True,speech_enabled=True):
        if self.owned_runtime and self.owned_runtime.poll() is None:
            return {'status':'ALREADY_RUNNING','owned':True}
        if self.status()['services']['submit']:
            return {'status':'ALREADY_RUNNING','owned':False}
        root = Path(__file__).resolve().parents[1]
        output = root/'simulation/artifacts/jerry'
        output.mkdir(parents=True,exist_ok=True)
        if self.runtime_log:self.runtime_log.close()
        self.runtime_log = open(output/'mcp-runtime.log','a')
        command = ['bash',str(root/'simulation/run_jerry.sh'),'--no-console']
        if headless:command.append('--headless')
        if not speech_enabled:command.append('--no-speak')
        self.owned_runtime = subprocess.Popen(command,cwd=root,stdout=self.runtime_log,stderr=subprocess.STDOUT,start_new_session=True)
        # Return admission promptly; readiness is observed through robot_status.
        await asyncio.sleep(.25)
        if self.owned_runtime.poll() is not None:
            raise RuntimeError('Simulation launch failed; inspect simulation/artifacts/jerry/mcp-runtime.log')
        return {'status':'STARTING','pid':self.owned_runtime.pid,'owned':True,'motion':'DISARMED','next':'Poll robot_status until navigation is ready'}

    async def close(self):
        if self.closed:return
        self.closed = True
        try:await self.stop()
        except Exception:pass
        if self.owned_runtime and self.owned_runtime.poll() is None:
            os.killpg(self.owned_runtime.pid,signal.SIGINT)
            try:await asyncio.to_thread(self.owned_runtime.wait,16.)
            except subprocess.TimeoutExpired:
                os.killpg(self.owned_runtime.pid,signal.SIGTERM)
                try:await asyncio.to_thread(self.owned_runtime.wait,3.)
                except subprocess.TimeoutExpired:
                    os.killpg(self.owned_runtime.pid,signal.SIGKILL)
                    await asyncio.to_thread(self.owned_runtime.wait,3.)
        self.executor.shutdown(timeout_sec=2.)
        self.thread.join(timeout=2.)
        self.node.destroy_node()
        import rclpy
        rclpy.try_shutdown()
        if self.runtime_log:self.runtime_log.close()
