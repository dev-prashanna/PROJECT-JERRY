"""Latest-frame depth inference isolated from the GUI/ROS Python interpreter."""
from dataclasses import dataclass
import multiprocessing as mp
import queue
import threading
import time
import traceback


@dataclass
class Prediction:
    packet: object
    depth: object
    view: object
    inference_ms: float
    device: str
    finished: float
    preprocess_ms: float = 0.
    postprocess_ms: float = 0.


def put_latest(channel, message):
    try:
        channel.put_nowait(message)
        return
    except queue.Full:
        pass
    try:channel.get_nowait()
    except queue.Empty:pass
    try:channel.put_nowait(message)
    except queue.Full:pass


def model_process(config, inputs, outputs, busy, stop, engine_factory=None):
    outputs.cancel_join_thread()
    try:
        if engine_factory is None:
            import torch
            import cv2
            torch.set_num_threads(2)
            cv2.setNumThreads(1)
            from .depth_engine import DepthEngine
            engine=DepthEngine(config)
        else:engine=engine_factory(config)
        put_latest(outputs,('ready',str(engine.device)))
        busy.clear()
        while not stop.is_set():
            try:identity,image=inputs.get(timeout=.1)
            except queue.Empty:continue
            busy.set()
            try:result=engine.infer(image,frame_id=identity)
            finally:busy.clear()
            put_latest(outputs,('prediction',identity,result.depth,result.inference_ms,
                                str(engine.device),result.preprocess_ms,result.postprocess_ms,
                                getattr(getattr(result,'meta',None),'convention','inverse')))
    except Exception:
        put_latest(outputs,('error',traceback.format_exc()))
    finally:busy.clear()


class DepthWorker(threading.Thread):
    def __init__(self,frames,results,cfg,engine_factory=None):
        super().__init__(daemon=True,name='robot-depth-bridge')
        self.frames=frames;self.results=results;self.cfg=cfg;self.engine_factory=engine_factory
        self.stop_event=threading.Event();self.status='Loading depth model in isolated process'
        self.error=None;self.processed=0
        self.context=mp.get_context('spawn')
        self.busy=self.context.Event();self.busy.set()
        self.child_stop=self.context.Event()
        self.inputs=self.context.Queue(maxsize=1);self.outputs=self.context.Queue(maxsize=2)
        self.process=None

    def run(self):
        from .robot_avoidance import analyze_depth
        self.process=self.context.Process(target=model_process,
            args=(self.cfg,self.inputs,self.outputs,self.busy,self.child_stop,self.engine_factory),
            name='jerry-depth-model',daemon=True)
        pending={};seq=0
        try:
            self.process.start()
            while not self.stop_event.is_set():
                while True:
                    try:message=self.outputs.get_nowait()
                    except queue.Empty:break
                    if message[0]=='ready':self.status='Model ready on '+message[1]
                    elif message[0]=='error':
                        self.error=message[1];self.status='Depth inference failed';return
                    elif message[0]=='prediction':
                        _,identity,depth,inference_ms,device,preprocess_ms,postprocess_ms,convention=message
                        packet=pending.get(identity)
                        if packet is None:continue
                        self.status='Model ready on '+device
                        view=analyze_depth(depth,convention)
                        self.results.put(Prediction(packet,depth,view,inference_ms,device,time.monotonic(),preprocess_ms,postprocess_ms))
                        self.processed+=1
                seq,packet=self.frames.wait_new(seq,.01)
                if packet is not None:
                    pending[packet.id]=packet
                    pending={key:value for key,value in pending.items() if key>=packet.id-12}
                    put_latest(self.inputs,(packet.id,packet.bgr))
                if self.process.exitcode is not None:
                    if self.error is None:
                        self.error=f'Depth process exited ({self.process.exitcode})'
                        self.status=self.error
                    break
        except Exception:
            self.error=traceback.format_exc();self.status='Depth process bridge failed'
        finally:
            self.child_stop.set();self.busy.clear()
            if self.process.pid is not None:
                self.process.join(timeout=2.)
                if self.process.is_alive():self.process.terminate();self.process.join(timeout=1.)
            for channel in [self.inputs,self.outputs]:
                channel.cancel_join_thread();channel.close()

    def stop(self):
        self.stop_event.set();self.child_stop.set();self.frames.close()
