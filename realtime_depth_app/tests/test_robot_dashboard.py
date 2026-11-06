import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import threading
from concurrent.futures import Future
from types import SimpleNamespace
import numpy as np
import pytest
from PySide6 import QtWidgets

from robot_live import RobotNode
from robot_window import RobotWindow
from src.robot_avoidance import Decision
from src.robot_attitude import AttitudeFilter
from src.robot_slam import FloorSlam
from robot_map_view import occupancy_path,scene_point,frontier_markers
from src.robot_detection import ObjectBox,ObjectPrediction


def arm_fixture(ready=False):
    future=Future()
    fake=SimpleNamespace(lock=threading.RLock(),active=False,arming=False,start_requested=False,
        arm_generation=0,auto_pending=False,policy=SimpleNamespace(reset=lambda:None),
        curiosity=SimpleNamespace(reset=lambda:None),spin_error=None,
        command_pub=SimpleNamespace(publish=lambda _:None),
        parameter_client=SimpleNamespace(services_are_ready=lambda:ready,set_parameters=lambda _:future))
    return fake,future


def test_start_request_waits_for_adapter_instead_of_being_lost():
    node,_=arm_fixture(False)
    RobotNode.arm(node,True)
    assert node.start_requested and node.auto_pending and not node.active
    assert node.decision.state=='WAITING'


def test_stop_cancels_in_flight_start_and_start_is_idempotent():
    node,future=arm_fixture(True)
    RobotNode.arm(node,True)
    generation=node.arm_generation
    RobotNode.arm(node,True)
    assert node.arm_generation==generation
    RobotNode.arm(node,False,'Stopped by user')
    future.set_result(SimpleNamespace(results=[SimpleNamespace(successful=True)]))
    assert not node.active and not node.start_requested and not node.arming


@pytest.fixture
def dashboard(tmp_path):
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    snapshot=FloorSlam().snapshot()
    snapshot.update(age_s=0.,ready=False,revision=1,plan_age_s=0.)
    calls=[]
    node=SimpleNamespace(start_requested=False,arming=False,active=False,clock_received=0.,
        world_control_status='',mode='explore',slam=SimpleNamespace(snapshot=lambda:dict(snapshot)),
        navigation_snapshot=lambda:(AttitudeFilter().snapshot(),{},'explore'),
        snapshot=lambda:(None,None,{},Decision(0.,0.,'MONITOR','Stopped'),False),
        set_mode=lambda mode:calls.append(('mode',mode)),set_sim_paused=lambda paused:None)
    def arm(enabled,*args):
        node.start_requested=enabled;calls.append(('arm',enabled))
    node.arm=arm
    window=RobotWindow(node,SimpleNamespace(status='Warming model'),SimpleNamespace(output=tmp_path,duration=0),lambda:None)
    window.timer.stop()
    yield window,snapshot,calls,app
    window.close();window.exports.shutdown(wait=True)


def test_start_stays_clickable_while_sensors_initialize_and_stop_always_works(dashboard):
    window,_,calls,_=dashboard
    window.refresh()
    assert window.start.isEnabled()
    window.start.click();window.refresh()
    assert ('arm',True) in calls and not window.start.isEnabled()
    window.stop.click();window.refresh()
    assert calls[-1]==('arm',False) and window.start.isEnabled()


def test_map_view_updates_with_live_revision_even_when_tracking_is_lost(dashboard):
    window,snapshot,_,_=dashboard
    window.refresh()
    assert 'update #1' in window.map_status.text()
    snapshot.update(revision=2,pose=[.1,.2,.3],tracking=False,status='TRACKING LOST')
    window.refresh()
    assert 'update #2' in window.map_status.text() and 'TRACKING LOST' in window.map_status.text()
    assert window.status_table.item(7,1).text()=='+0.100 m'


def test_saving_does_not_block_stop_button_or_gui_refresh(dashboard):
    window,_,calls,_=dashboard
    started=threading.Event();release=threading.Event()
    def slow_save():
        started.set();release.wait(2.)
    window.save=slow_save
    window.capture()
    assert started.wait(1.)
    try:
        window.stop.click();window.refresh()
        assert calls[-1]==('arm',False) and not window.export_job.done()
    finally:release.set();window.export_job.result(timeout=2.)


def test_vector_map_preserves_unknown_holes_and_thin_obstacles():
    mask=np.zeros((30,30),bool);mask[3:26,3:26]=True;mask[11:18,11:18]=False
    origin=np.array([0.,0.]);resolution=.05
    shape=occupancy_path(mask,origin,resolution)
    assert shape.contains(scene_point([.3,.3]))
    assert not shape.contains(scene_point([.7,.7]))
    thin=np.zeros((30,30),bool);thin[8,4:20]=True
    shape=occupancy_path(thin,origin,resolution)
    assert shape.contains(scene_point([.5,.425]))


def test_frontier_display_clusters_candidates_without_inventing_positions():
    snapshot=FloorSlam().snapshot()
    snapshot['frontiers']=[[x*.05,0.] for x in range(20)]
    markers=frontier_markers(snapshot)
    assert len(markers)==1 and list(markers[0]) in snapshot['frontiers']


def test_live_yolo_view_labels_and_stale_results_are_not_displayed_as_live(dashboard):
    window,_,_,_=dashboard
    source=SimpleNamespace(id=42,bgr=np.zeros((100,120,3),np.uint8))
    prediction=ObjectPrediction(source,(ObjectBox('person',.91,(10.,10.,60.,90.)),),12.,'cpu')
    result={'prediction':prediction,'age_s':.1,'status':'YOLO ready','error':None}
    window.node.detector=SimpleNamespace(snapshot=lambda:dict(result))
    window.refresh()
    assert window.object_table.rowCount()==1
    assert window.object_table.item(0,0).text()=='person'
    assert window.object_table.item(0,1).text()=='91%'
    assert 'frame #42' in window.objects_caption.text()
    assert window.objects_image.pixmap() is not None
    result['age_s']=2.
    window.refresh()
    assert window.object_table.rowCount()==0 and 'fresh' in window.objects_caption.text()
    assert window.objects_image.pixmap() is None or window.objects_image.pixmap().isNull()
