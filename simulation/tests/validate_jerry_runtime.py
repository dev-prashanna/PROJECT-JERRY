"""Measured MCP/Gazebo validation. Ground truth is confined to this evaluator."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter,deque
from datetime import timedelta,datetime,timezone
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
from uuid import uuid4
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path as NavPath
from jerry_interfaces.msg import MotionCommand,TaskState
from gz.transport13 import Node as GazeboNode
from gz.msgs10.pose_v_pb2 import Pose_V
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT=Path(__file__).resolve().parents[2]


class Evaluator:
    def __init__(self):
        rclpy.init()
        self.node=Node('jerry_validation_evaluator')
        self.executor=SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.lock=threading.RLock()
        self.rows=[]
        self.commands=deque(maxlen=30000)
        self.proposals=deque(maxlen=1000)
        self.truth=None
        self.path=[]
        self.motion=None
        self.transport=GazeboNode()
        self.transport.subscribe(Pose_V,'/world/vbmd_lab/dynamic_pose/info',self.pose)
        self.node.create_subscription(String,'/perception/avoidance_status',self.status,10)
        self.node.create_subscription(NavPath,'/exploration/path',lambda m:setattr(self,'path',[[p.pose.position.x,p.pose.position.y] for p in m.poses]),1)
        self.node.create_subscription(Twist,'/sim/cmd_vel_safe',self.command,qos_profile_sensor_data)
        self.node.create_subscription(MotionCommand,'/jerry/motion/proposal',lambda m:self.proposals.append(m),1)
        self.node.create_subscription(TaskState,'/jerry/motion/state',lambda m:setattr(self,'motion',m),1)
        self.replay=self.node.create_publisher(MotionCommand,'/jerry/motion/proposal',1)
        self.thread=threading.Thread(target=self.executor.spin,daemon=True)
        self.thread.start()

    def pose(self,msg):
        for p in msg.pose:
            if p.name=='vbmd':
                q=p.orientation
                yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
                self.truth=(time.monotonic(),p.position.x,p.position.y,yaw)
                break

    def command(self,msg):
        with self.lock:self.commands.append((time.monotonic(),msg.linear.x,msg.angular.z))

    def status(self,msg):
        value=json.loads(msg.data)
        value['wall']=time.monotonic()
        value['evaluator_truth']=self.truth
        value['evaluator_route']=self.path
        value['evaluator_output']=self.commands[-1] if self.commands else None
        self.rows.append(value)

    def metrics(self,start):
        with self.lock:
            rows=[r for r in list(self.rows) if r['wall']>=start]
            commands=[c for c in list(self.commands) if c[0]>=start]
        poses=np.asarray([r['evaluator_truth'][1:3] for r in rows if r.get('evaluator_truth')])
        route_errors=[]
        localization_errors=[]
        goals=[]
        bad_switches=[]
        previous=None
        for row in rows:
            slam=row.get('slam',{})
            pose=slam.get('pose')
            truth=row.get('evaluator_truth')
            if pose and truth and abs(row['wall']-truth[0])<.25:
                localization_errors.append(math.dist(pose[:2],truth[1:3]))
            route=np.asarray(row['evaluator_route'],float)
            if pose and len(route)>1 and row['state']=='EXPLORE':
                a,b=route[:-1],route[1:]
                t=np.clip(np.sum((pose[:2]-a)*(b-a),axis=1)/np.maximum(np.sum((b-a)**2,axis=1),1e-12),0,1)
                route_errors.append(float(np.min(np.linalg.norm(a+t[:,None]*(b-a)-pose[:2],axis=1))))
            goal=slam.get('goal')
            if goal is not None:
                current=(slam.get('goal_id'),tuple(goal),slam.get('completed_goals',0),slam.get('loop_closures',0))
                if previous and current[:2]!=previous[:2] and current[2]<=previous[2] and current[3]==previous[3]:
                    bad_switches.append([previous,current])
                if not previous or current[:2]!=previous[:2]:
                    goals.append(current)
                    # Compare completion with the count at goal selection,
                    # not the immediately preceding ARRIVED feedback row.
                    previous=current
        areas=[r.get('slam',{}).get('observed_m2',0.) for r in rows]
        return dict(samples=len(rows),ground_truth_samples=len(poses),
            actual_distance_m=float(np.linalg.norm(np.diff(poses,axis=0),axis=1).sum()) if len(poses)>1 else 0.,
            actual_extent_m=float(np.linalg.norm(np.ptp(poses,axis=0))) if len(poses)>1 else 0.,
            observed_growth_m2=max(areas)-min(areas) if areas else 0.,
            completed_goals=max((r.get('slam',{}).get('completed_goals',0) for r in rows),default=0),
            route_error_p95_m=float(np.percentile(route_errors,95)) if route_errors else None,
            localization_rmse_m=float(np.sqrt(np.mean(np.square(localization_errors)))) if localization_errors else None,
            phase_counts=dict(Counter(r['state'] for r in rows)),
            stop_reasons=dict(Counter(r.get('reason','') for r in rows if r['state']=='STOP')),
            nonzero_translation_commands=sum(abs(v)>.001 for _,v,w in commands),
            nonzero_rotation_commands=sum(abs(w)>.01 for _,v,w in commands),
            premature_goal_changes=bad_switches,goals=goals,
            last=({k:rows[-1].get(k) for k in ('state','reason','slam')} if rows else None))

    def command_snapshot(self):
        with self.lock:return list(self.commands)

    def close(self):
        self.executor.shutdown(timeout_sec=2.)
        self.thread.join(timeout=2.)
        self.node.destroy_node()
        rclpy.try_shutdown()


async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exploration-wall',type=float,default=70.)
    parser.add_argument('--faults',action='store_true')
    args=parser.parse_args()
    folder=ROOT/'simulation/artifacts/jerry'/('validation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    folder.mkdir(parents=True)
    report={'passed':False,'ground_truth_used_only_for_evaluation':True,'checks':{}}
    evaluator=Evaluator()
    params=StdioServerParameters(command='/bin/bash',args=[str(ROOT/'simulation/run_robot_mcp.sh')],env={
        'ROS_DOMAIN_ID':os.environ['ROS_DOMAIN_ID'],'GZ_PARTITION':os.environ['GZ_PARTITION'],'GZ_IP':'127.0.0.1'})
    try:
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write,read_timeout_seconds=timedelta(seconds=120)) as session:
                await session.initialize()
                async def call(name,arguments=None):
                    response=await session.call_tool(name,arguments or {})
                    if response.isError:raise RuntimeError(name+': '+str(response.content))
                    return response.structuredContent if response.structuredContent is not None else json.loads(next(b.text for b in response.content if b.type=='text'))
                async def ready(timeout=80.):
                    end=time.monotonic()+timeout
                    while time.monotonic()<end:
                        s=await call('robot_status')
                        if s.get('navigation') and s['navigation']['fresh'] and s['navigation']['state']['ready'] and s.get('task') and s['task']['state']['session_id']:
                            return s
                        await asyncio.sleep(.2)
                    raise AssertionError('Sensor/localization readiness timed out')
                async def stopped(after,window=1.):
                    await asyncio.sleep(window)
                    later=[c for c in evaluator.command_snapshot() if c[0]>after+.35]
                    assert later and not any(v or w for _,v,w in later),later[-10:]
                await call('start_simulation',dict(headless=True,speech_enabled=False))
                await ready()
                assert evaluator.commands and not any(v or w for _,v,w in evaluator.command_snapshot()),'Startup moved without admission'
                report['checks']['startup_disarmed']=True
                print('READY: starting measured route-following mission',flush=True)
                start=time.monotonic()
                admitted=await call('start_exploration',dict(request_id=uuid4().hex,maximum_active_s=120.,maximum_wall_s=min(300.,args.exploration_wall+20.),maximum_distance_m=5.))
                assert admitted['accepted'],admitted
                report['exploration_task']=admitted
                last_print=0.
                while time.monotonic()-start<args.exploration_wall:
                    await asyncio.sleep(.25)
                    state=await call('get_task_status',dict(task_id=admitted['task_id']))
                    if time.monotonic()-last_print>10.:
                        last_print=time.monotonic()
                        metrics=evaluator.metrics(start)
                        print(json.dumps({'elapsed_wall_s':round(time.monotonic()-start,1),'task':state['status'],
                            'distance_m':round(metrics['actual_distance_m'],3),'goals':metrics['completed_goals'],'phases':metrics['phase_counts']}),flush=True)
                    if state['status'] in ('SUCCEEDED','FAILED','CANCELLED','EXPIRED'):
                        report['exploration_terminal']=state
                        break
                metrics=evaluator.metrics(start)
                report['route_following']=metrics
                stopped_at=time.monotonic()
                await call('stop_robot')
                await stopped(stopped_at)
                report['checks']['explicit_stop_no_restart']=True
                # Save the measured failure before checking usefulness thresholds.
                (folder/'report.json').write_text(json.dumps(report,indent=2))
                assert metrics['ground_truth_samples']>20,'Missing independent trajectory evidence'
                assert metrics['actual_distance_m']>=.25,'Exploration failed to travel a meaningful distance'
                assert metrics['completed_goals']>=1,'No measured frontier arrival'
                assert metrics['observed_growth_m2']>=.10,'Exploration did not extend observed map'
                assert metrics['route_error_p95_m'] is not None and metrics['route_error_p95_m']<=.15,'Did not follow planned route'
                assert metrics['localization_rmse_m'] is not None and metrics['localization_rmse_m']<=.10,'Localization exceeded supported-lab error target'
                assert not metrics['premature_goal_changes'],'Destination changed without arrival or correction'
                report['checks']['route_following']=True
                if evaluator.proposals:
                    old=evaluator.proposals[-1]
                    replayed=time.monotonic()
                    for _ in range(5):
                        evaluator.replay.publish(old)
                        await asyncio.sleep(.04)
                    await stopped(replayed)
                    report['checks']['old_proposal_cannot_restart']=True
                # A short wall budget must terminate even though more exploration is possible.
                await ready()
                bounded=await call('start_exploration',dict(request_id=uuid4().hex,maximum_active_s=120.,maximum_wall_s=3.,maximum_distance_m=5.))
                assert bounded['accepted'],bounded
                await asyncio.sleep(4.)
                state=await call('get_task_status',dict(task_id=bounded['task_id']))
                assert state['status']=='EXPIRED',state
                await stopped(time.monotonic())
                report['checks']['wall_budget_enforced']=True
                if args.faults:
                    for component in ('navigation','supervisor','executive'):
                        await ready()
                        active=await call('start_exploration',dict(request_id=uuid4().hex,maximum_active_s=120.,maximum_wall_s=30.,maximum_distance_m=5.))
                        assert active['accepted'],active
                        end=time.monotonic()+8.
                        while time.monotonic()<end and not any(c[0]>end-8 and (c[1] or c[2]) for c in evaluator.command_snapshot()):
                            await asyncio.sleep(.1)
                        assert any(c[0]>end-8 and (c[1] or c[2]) for c in evaluator.command_snapshot()),'Fault injection requires actual preceding motion'
                        import subprocess
                        output=subprocess.check_output(['pgrep','-f',r'python.*-m jerry.runtime '+component+r'( |$)'],text=True)
                        pids=[int(p) for p in output.split()]
                        assert len(pids)==1,(component,pids)
                        began=time.monotonic()
                        try:
                            os.kill(pids[0],signal.SIGSTOP)
                            await stopped(began,1.)
                        finally:os.kill(pids[0],signal.SIGCONT)
                        await call('stop_robot')
                        await asyncio.sleep(.4)
                        report['checks'][component+'_freeze_stops_output']=True
                        print('PASS: '+component+' freeze removed actuator output',flush=True)
                report['passed']=True
    except BaseException as exc:
        report['failure']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        evaluator.close()
        (folder/'report.json').write_text(json.dumps(report,indent=2))
        (folder/'telemetry.json').write_text(json.dumps(evaluator.rows,allow_nan=True))
        print('Validation artifacts: '+str(folder),flush=True)
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':asyncio.run(main())
