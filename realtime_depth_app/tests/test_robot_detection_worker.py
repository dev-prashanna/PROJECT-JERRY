import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from realtime_depth_app.src.camera import LatestSlot
from realtime_depth_app.src.robot_detection import ObjectBox
from realtime_depth_app.src.robot_detection_worker import DetectionProcessWorker


class FakeDetector:
    def __call__(self,image):
        return (ObjectBox('chair',.9,(1.,2.,float(image[0,0,0]),4.)),)


def test_isolated_detector_preserves_inflight_frame_and_shuts_down():
    frames=LatestSlot()
    worker=DetectionProcessWorker(frames,Path('fake.pt'),fps=20.,engine_factory=FakeDetector)
    worker.start()
    try:
        frames.put(SimpleNamespace(id=7,received=time.monotonic(),bgr=np.full((8,8,3),7,np.uint8)))
        end=time.monotonic()+8.
        while worker.latest is None and not worker.error and time.monotonic()<end:time.sleep(.02)
        assert worker.latest is not None,worker.error
        assert worker.latest.packet.id==7
        assert worker.latest.boxes[0].xyxy[2]==7.
    finally:
        worker.stop();worker.join(timeout=5.)
    assert not worker.is_alive()
    assert not worker.process.is_alive()
