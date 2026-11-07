import math
import time

import cv2
import numpy as np

from realtime_depth_app.src.camera import LatestSlot
from realtime_depth_app.src.robot_memory import read_memory,write_memory
from realtime_depth_app.src.robot_slam import Calibration,FloorSlam,SlamWorker


def scene():
    rng=np.random.default_rng(31)
    texture=cv2.GaussianBlur(rng.integers(20,235,(1200,1200),np.uint8),(5,5),0)
    k=np.array([[460.,0,320],[0,460.,240],[0,0,1.]])
    def frame(x,y,yaw):
        c,s=math.cos(yaw),math.sin(yaw)
        homography=k@np.array([[.002*s,-.002*c,-1.2*s+1.2*c-x*s+y*c],
                               [0.,0.,.072],
                               [.002*c,.002*s,-1.2*c-1.2*s-x*c-y*s-.0757]])
        gray=cv2.warpPerspective(texture,homography,(640,480))
        return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    return Calibration(k),frame


def test_saved_map_and_visits_survive_restart_and_require_camera_relocalization(tmp_path):
    calibration,frame=scene();source=FloorSlam()
    for tick in range(8):
        stamp=tick*.1
        source.process(frame(tick*.003,0.,tick*.005),stamp,calibration,
                       (0.,0.,tick*.005,True),{'front':(1.,4.,stamp)})
    assert source.tracking
    source.planner.visits={(1,2):7,(-2,0):3}
    source.planner.completed_goals=2
    source.planner.goal_id=4
    path=tmp_path/'map-memory.npz'
    write_memory(path,source.memory_state(source.planner))
    restored=FloorSlam();restored.restore_memory(read_memory(path))
    assert np.array_equal(restored.map.grid(),source.map.grid())
    assert restored.planner.visits==source.planner.visits
    assert restored.planner.completed_goals==2 and restored.planner.goal_id==4
    assert not restored.tracking and restored.relocalizing
    assert restored.planner.goal is None
    before=restored.map.grid().copy()
    restored.process(np.zeros((480,640,3),np.uint8),0.,calibration,(0.,0.,.035,True),{})
    assert not restored.tracking and restored.relocalizing
    assert np.array_equal(restored.map.grid(),before)
    restored.process(frame(.021,0.,.035),.1,calibration,(0.,0.,.035,True),{})
    assert restored.tracking and not restored.relocalizing
    assert math.dist(restored.pose[:2],source.pose[:2])<.02
    # A reset simulator clock must preserve the area and reacquire its location.
    restored.process(frame(.021,0.,.035),0.,calibration,(0.,0.,.035,True),{})
    assert restored.tracking and np.array_equal(restored.map.grid(),before)


def test_worker_saves_on_close_and_reloads_from_stable_path(tmp_path):
    calibration,frame=scene();path=tmp_path/'persistent-map.npz'
    frames=LatestSlot();worker=SlamWorker(frames,memory_path=path)
    worker.start()
    try:
        frames.put((frame(0.,0.,0.),1.,calibration,(0.,0.,0.,True),{'front':(1.,4.,1.)}))
        end=time.monotonic()+5.
        while not worker.processed and time.monotonic()<end:time.sleep(.01)
        assert worker.processed and worker.error is None
        saved_grid=worker.snapshot()['grid'].copy()
    finally:
        worker.stop();worker.join(timeout=5.)
    assert not worker.is_alive() and path.exists()
    resumed=SlamWorker(LatestSlot(),memory_path=path)
    assert resumed.snapshot()['restored_map']
    assert np.array_equal(resumed.snapshot()['grid'],saved_grid)
    resumed.memory_writer.shutdown(wait=True)
