"""Bounded live YOLO inference, independent of SLAM and motion control."""
from dataclasses import dataclass
from pathlib import Path
import os
import threading
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class ObjectBox:
    label: str
    confidence: float
    xyxy: tuple


@dataclass(frozen=True)
class ObjectPrediction:
    packet: object
    boxes: tuple
    inference_ms: float
    device: str


def decode_boxes(result):
    if result.boxes is None:
        return ()
    coordinates=result.boxes.xyxy.cpu().numpy()
    confidences=result.boxes.conf.cpu().numpy()
    classes=result.boxes.cls.cpu().numpy()
    objects=[]
    for xyxy,confidence,category in zip(coordinates,confidences,classes):
        if not np.isfinite(xyxy).all() or not np.isfinite(confidence):continue
        objects.append(ObjectBox(str(result.names[int(category)]),float(confidence),tuple(float(v) for v in xyxy)))
    return tuple(objects)


def annotate_objects(prediction):
    image=prediction.packet.bgr.copy()
    h,w=image.shape[:2]
    for box in prediction.boxes:
        x1,y1,x2,y2=box.xyxy
        x1,x2=int(np.clip(x1,0,w-1)),int(np.clip(x2,0,w-1))
        y1,y2=int(np.clip(y1,0,h-1)),int(np.clip(y2,0,h-1))
        color=(90,205,90)
        cv2.rectangle(image,(x1,y1),(x2,y2),color,2,cv2.LINE_AA)
        text=f'{box.label} {box.confidence:.0%}'
        (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.55,1)
        top=max(th+8,y1)
        cv2.rectangle(image,(x1,max(0,top-th-8)),(min(w-1,x1+tw+8),top),color,-1)
        cv2.putText(image,text,(x1+4,top-5),cv2.FONT_HERSHEY_SIMPLEX,.55,(15,30,15),1,cv2.LINE_AA)
    return image


class DetectionWorker(threading.Thread):
    def __init__(self,frames,model_path,fps=4.,confidence=.30,device='auto',can_infer=None,engine_factory=None):
        super().__init__(daemon=True,name='robot-yolo')
        self.frames=frames;self.model_path=Path(model_path);self.fps=fps;self.confidence=confidence
        self.device=device;self.can_infer=can_infer or (lambda:True);self.engine_factory=engine_factory
        self.stop_event=threading.Event();self.lock=threading.Lock()
        self.latest=None;self.processed=0;self.error=None;self.status='Loading pretrained YOLO11n'

    def run(self):
        try:
            if self.engine_factory:
                model=self.engine_factory();device='cpu'
            else:
                if not self.model_path.is_file():
                    raise RuntimeError('YOLO checkpoint missing; run simulation/setup_vision.sh')
                root=Path(__file__).resolve().parents[2]
                os.environ.setdefault('YOLO_CONFIG_DIR',str(root/'.cache/yolo'))
                Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True,exist_ok=True)
                from ultralytics import YOLO,settings
                import torch
                settings.update({'sync':False})
                cv2.setNumThreads(2)
                device=('0' if torch.cuda.is_available() else 'cpu') if self.device=='auto' else self.device
                model=YOLO(str(self.model_path))
            self.status=self.model_path.stem+' ready on '+('CUDA:'+device if device.isdigit() else device)
            seq=0;last_id=None;next_run=0.
            while not self.stop_event.is_set():
                seq,packet=self.frames.wait_new(seq,.2)
                if packet is None:continue
                # Depth owns scheduling priority; detection never holds its buffers.
                while not self.stop_event.is_set() and (time.monotonic()<next_run or not self.can_infer()):
                    self.stop_event.wait(.02)
                if self.stop_event.is_set():break
                packet=self.frames.get();seq=self.frames.seq()
                if packet is None or packet.id==last_id or time.monotonic()-packet.received>.5:continue
                started=time.monotonic()
                results=model.predict(source=packet.bgr,imgsz=416,conf=self.confidence,
                                      device=device,half=device!='cpu',verbose=False,save=False)
                result=ObjectPrediction(packet,decode_boxes(results[0]),(time.monotonic()-started)*1000,device)
                with self.lock:self.latest=result;self.processed+=1
                last_id=packet.id;next_run=time.monotonic()+1./self.fps
        except Exception:
            import traceback
            self.error=traceback.format_exc();self.status='Object detector unavailable; see diagnostics'

    def snapshot(self):
        with self.lock:
            result=self.latest
            return {'prediction':result,'status':self.status,'error':self.error,'frames':self.processed,
                    'age_s':time.monotonic()-result.packet.received if result else None}

    def stop(self):
        self.stop_event.set();self.frames.close()
