"""Keep optional YOLO inference outside the navigation interpreter and GIL."""
from __future__ import annotations
import multiprocessing as mp
import queue
import threading
import time
import traceback
from .robot_detection import ObjectPrediction,ObjectBox
from .robot_depth_worker import put_latest


def detection_process(model_path,confidence,inputs,outputs,stop,engine_factory=None):
    outputs.cancel_join_thread()
    try:
        if engine_factory is None:
            import os
            from pathlib import Path
            os.environ.setdefault('OMP_NUM_THREADS','2')
            os.environ.setdefault('MKL_NUM_THREADS','2')
            os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).resolve().parents[2]/'.cache/yolo'))
            import torch
            import cv2
            from ultralytics import YOLO,settings
            from .robot_detection import decode_boxes
            torch.set_num_threads(2)
            cv2.setNumThreads(1)
            settings.update({'sync':False})
            model=YOLO(str(model_path))
        else:model=engine_factory()
        put_latest(outputs,('ready',))
        while not stop.is_set():
            try:identity,image=inputs.get(timeout=.1)
            except queue.Empty:continue
            started=time.monotonic()
            if engine_factory is None:
                results=model.predict(source=image,imgsz=416,conf=confidence,device='cpu',half=False,verbose=False,save=False)
                boxes=decode_boxes(results[0])
                # Ultralytics may select its own CPU thread count on first setup.
                torch.set_num_threads(2)
            else:boxes=model(image)
            put_latest(outputs,('prediction',identity,[(b.label,b.confidence,b.xyxy) for b in boxes],(time.monotonic()-started)*1000))
    except Exception:put_latest(outputs,('error',traceback.format_exc()))


class DetectionProcessWorker(threading.Thread):
    def __init__(self,frames,model_path,fps=2.,confidence=.30,engine_factory=None):
        super().__init__(daemon=True,name='robot-yolo-bridge')
        self.frames,self.model_path,self.fps,self.confidence=frames,model_path,fps,confidence
        self.engine_factory=engine_factory
        self.stop_event=threading.Event()
        self.lock=threading.Lock()
        self.latest=None
        self.processed=0
        self.error=None
        self.status='Loading YOLO in isolated CPU worker'
        context=mp.get_context('spawn')
        self.inputs=context.Queue(maxsize=1)
        self.outputs=context.Queue(maxsize=2)
        self.child_stop=context.Event()
        self.process=context.Process(target=detection_process,args=(model_path,confidence,self.inputs,self.outputs,self.child_stop,engine_factory),daemon=True)

    def run(self):
        seq=0
        pending={}
        ready=False
        next_run=0.
        try:
            self.process.start()
            while not self.stop_event.is_set():
                try:message=self.outputs.get(timeout=.02)
                except queue.Empty:message=None
                if message:
                    if message[0]=='ready':ready=True
                    elif message[0]=='error':raise RuntimeError(message[1])
                    elif message[0]=='prediction':
                        _,identity,boxes,elapsed=message
                        packet=pending.pop(identity,None)
                        if packet is not None:
                            prediction=ObjectPrediction(packet,tuple(ObjectBox(*box) for box in boxes),elapsed,'cpu')
                            with self.lock:
                                self.latest=prediction
                                self.processed+=1
                                self.status='YOLO ready on isolated CPU worker'
                if ready and not pending and time.monotonic()>=next_run:
                    seq,packet=self.frames.wait_new(seq,.02)
                    if packet is not None and time.monotonic()-packet.received<.5:
                        pending[packet.id]=packet
                        put_latest(self.inputs,(packet.id,packet.bgr))
                        next_run=time.monotonic()+1./self.fps
                if self.process.exitcode is not None:raise RuntimeError(f'Detector exited ({self.process.exitcode})')
        except Exception:
            self.error=traceback.format_exc()
            self.status='Object detector unavailable'
        finally:
            self.child_stop.set()
            if self.process.pid is not None:
                self.process.join(timeout=2.)
                if self.process.is_alive():self.process.terminate();self.process.join(timeout=2.)
            for channel in (self.inputs,self.outputs):channel.cancel_join_thread();channel.close()

    def snapshot(self):
        with self.lock:
            result=self.latest
            return {'prediction':result,'status':self.status,'error':self.error,'frames':self.processed,
                    'age_s':time.monotonic()-result.packet.received if result else None}

    def stop(self):
        self.stop_event.set()
        self.child_stop.set()
        self.frames.close()
