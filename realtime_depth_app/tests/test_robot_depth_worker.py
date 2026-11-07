from types import SimpleNamespace
import time
import numpy as np

from realtime_depth_app.src.camera import LatestSlot
from realtime_depth_app.src.robot_depth_worker import DepthWorker


class FakeDepthEngine:
    device='cpu'
    def __init__(self,config):pass
    def infer(self,image,frame_id):
        return SimpleNamespace(depth=np.tile(np.arange(32,dtype=np.float32),(24,1))+frame_id,
                               inference_ms=1.,preprocess_ms=.2,postprocess_ms=.3,
                               meta=SimpleNamespace(convention='inverse'))


class BrokenDepthEngine:
    def __init__(self,config):raise RuntimeError('test model could not load')


def test_depth_process_preserves_source_frame_identity_and_stops_cleanly():
    frames=LatestSlot();results=LatestSlot();worker=DepthWorker(frames,results,None,engine_factory=FakeDepthEngine)
    worker.start()
    try:
        for identity in range(1,11):
            frames.put(SimpleNamespace(id=identity,received=time.monotonic(),bgr=np.full((24,32,3),identity,np.uint8)))
            time.sleep(.005)
        until=time.monotonic()+8.
        while results.get() is None and not worker.error and time.monotonic()<until:time.sleep(.02)
        prediction=results.get()
        assert prediction is not None,worker.error
        assert prediction.packet.id==int(prediction.depth[0,0])
        assert prediction.packet.id==10 and prediction.device=='cpu'
        assert prediction.preprocess_ms==.2 and worker.process.pid is not None
    finally:
        worker.stop();worker.join(timeout=4.)
    assert not worker.is_alive() and not worker.process.is_alive()


def test_depth_process_failure_is_reported_to_the_controller():
    worker=DepthWorker(LatestSlot(),LatestSlot(),None,engine_factory=BrokenDepthEngine)
    worker.start();worker.join(timeout=8.)
    if worker.is_alive():worker.stop();worker.join(timeout=4.)
    assert worker.error is not None
