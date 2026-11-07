"""Live integrated simulation acceptance; only hardware-equivalent sensors enter navigation."""
import json
import os
from pathlib import Path
import time
from uuid import uuid4
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from jerry_interfaces.msg import NavigationState,TaskState
from jerry_interfaces.srv import Ask
from rclpy.qos import qos_profile_sensor_data


def main():
    rclpy.init()
    node=Node('jerry_acceptance')
    seen={}
    commands=[]
    node.create_subscription(NavigationState,'/jerry/navigation/state',lambda m:seen.__setitem__('nav',m),1)
    node.create_subscription(TaskState,'/jerry/tasks/state',lambda m:seen.__setitem__('task',m),10)
    node.create_subscription(Twist,'/sim/cmd_vel_safe',lambda m:commands.append((time.monotonic(),m.linear.x,m.angular.z)),qos_profile_sensor_data)
    ask=node.create_client(Ask,'/jerry/dialogue/ask')
    report={}
    def until(predicate,seconds):
        end=time.monotonic()+seconds
        while not predicate():
            if time.monotonic()>end:raise AssertionError('Timed out: '+str(report))
            rclpy.spin_once(node,timeout_sec=.02)
    def question(text,timeout=8.):
        future=ask.call_async(Ask.Request(request_id=uuid4().hex,text=text))
        until(future.done,timeout)
        result=future.result()
        report[text]=dict(accepted=result.accepted,text=result.text,evidence_ids=list(result.evidence_ids))
        print(text+': '+result.text,flush=True)
        assert result.accepted, result.text
        return result
    try:
        until(lambda:ask.service_is_ready() and 'nav' in seen and 'task' in seen and seen['nav'].ready and seen['task'].session_id,75.)
        assert not any(v or w for _,v,w in commands),'Startup moved without admission'
        report['startup_disarmed']=True
        question('remember that the simulation acceptance test was run')
        question('status')
        question('explore the room')
        until(lambda:any(v or w for _,v,w in commands),20.)
        report['authorized_motion_observed']=True
        started=time.monotonic()
        question('stop')
        until(lambda:commands and commands[-1][1:]==(0.,0.),1.)
        report['stop_observed_wall_s']=time.monotonic()-started
        last=len(commands)
        until(lambda:len(commands)>last+15,3.)
        assert not any(v or w for _,v,w in commands[last:]),'Motion restarted after Stop'
        report['no_restart_after_stop']=True
        question('what did you find earlier?')
        report['navigation']=dict(ready=seen['nav'].ready,phase=seen['nav'].phase,reason=seen['nav'].reason,observed_m2=seen['nav'].observed_m2)
        report['passed']=True
    finally:
        path=Path(os.environ.get('JERRY_RUN_DIR','simulation/artifacts/jerry'))/'acceptance.json'
        path.write_text(json.dumps(report,indent=2))
        node.destroy_node()
        rclpy.try_shutdown()


if __name__=='__main__':main()
