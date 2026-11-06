"""Exercise dashboard's Gazebo service bridge without commanding robot movement."""
import json
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from ros_gz_interfaces.srv import ControlWorld


def main():
    rclpy.init();node=Node('vbmd_world_controls_check');stamp=[None]
    subscription=node.create_subscription(Clock,'/clock',lambda m:stamp.__setitem__(0,m.clock.sec+m.clock.nanosec*1e-9),10)
    client=node.create_client(ControlWorld,'/world/vbmd_lab/control')
    def spin_for(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.03)
    def control(paused):
        request=ControlWorld.Request();request.world_control.pause=paused
        future=client.call_async(request)
        rclpy.spin_until_future_complete(node,future,timeout_sec=5.)
        assert future.done() and future.result().success
    try:
        assert client.wait_for_service(timeout_sec=20.)
        spin_for(1.);assert stamp[0] is not None
        control(True);spin_for(.4);before=stamp[0];spin_for(.8);paused_delta=stamp[0]-before
        assert paused_delta<.03,paused_delta
        control(False);before=stamp[0];spin_for(1.2);resumed_delta=stamp[0]-before
        assert resumed_delta>.05,resumed_delta
        report={'pause_service_passed':True,'paused_clock_delta_s':paused_delta,'resume_service_passed':True,'resumed_clock_delta_s':resumed_delta}
        output=Path('simulation/artifacts/curiosity/world_controls.json');output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:
        if client.service_is_ready():
            request=ControlWorld.Request();request.world_control.pause=False
            future=client.call_async(request);rclpy.spin_until_future_complete(node,future,timeout_sec=2.)
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
