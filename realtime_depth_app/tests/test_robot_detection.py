from types import SimpleNamespace
import threading
import time
import numpy as np

from src.camera import LatestSlot
from src.robot_detection import DetectionWorker,ObjectBox,ObjectPrediction,annotate_objects,decode_boxes


class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    def cpu(self):return self
    def numpy(self):return self.value


def result():
    return SimpleNamespace(names={0:'person'},boxes=SimpleNamespace(
        xyxy=Tensor([[5.,6.,40.,45.]]),conf=Tensor([.9]),cls=Tensor([0.])))


def packet(identity,age=0.):
    return SimpleNamespace(id=identity,received=time.monotonic()-age,bgr=np.zeros((64,80,3),np.uint8))


def test_detection_decoding_and_overlay_preserve_the_source_image():
    boxes=decode_boxes(result())
    assert boxes[0].label=='person' and boxes[0].confidence==.9
    source=packet(3)
    prediction=ObjectPrediction(source,boxes,10.,'cpu')
    image=annotate_objects(prediction)
    assert image.sum()>0 and source.bgr.sum()==0 and prediction.packet.id==3


def test_detector_gives_depth_priority_and_uses_latest_frame(tmp_path):
    allow=threading.Event();called=threading.Event();seen=[]
    class Engine:
        def predict(self,**kwargs):seen.append(kwargs);called.set();return [result()]
    frames=LatestSlot()
    worker=DetectionWorker(frames,tmp_path/'model.pt',can_infer=allow.is_set,engine_factory=Engine)
    worker.start()
    try:
        frames.put(packet(1));time.sleep(.05)
        assert not called.is_set()
        frames.put(packet(2));frames.put(packet(3));allow.set()
        assert called.wait(1.)
        until=time.monotonic()+1.
        while worker.snapshot()['prediction'] is None and time.monotonic()<until:time.sleep(.01)
        snapshot=worker.snapshot()
        assert snapshot['prediction'].packet.id==3 and snapshot['prediction'].boxes[0].label=='person'
        assert seen[0]['imgsz']==416 and seen[0]['save'] is False
    finally:worker.stop();worker.join(timeout=2.)


def test_missing_checkpoint_is_reported_without_an_inference_crash(tmp_path):
    worker=DetectionWorker(LatestSlot(),tmp_path/'absent.pt')
    worker.start();worker.join(timeout=2.)
    assert worker.error and 'checkpoint missing' in worker.error and worker.latest is None


def test_old_camera_frames_are_not_inferred(tmp_path):
    called=threading.Event()
    class Engine:
        def predict(self,**kwargs):called.set();return [result()]
    frames=LatestSlot();worker=DetectionWorker(frames,tmp_path/'model.pt',engine_factory=Engine)
    worker.start()
    try:
        frames.put(packet(1,age=2.));time.sleep(.08)
        assert not called.is_set()
        frames.put(packet(2));assert called.wait(1.)
    finally:worker.stop();worker.join(timeout=2.)
