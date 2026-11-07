"""Exercise the real MCP wire protocol, ROS runtime and Gazebo together."""
import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
import time
from uuid import uuid4
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT=Path(__file__).resolve().parents[2]


async def main():
    report={}
    params=StdioServerParameters(command='/bin/bash',args=[str(ROOT/'simulation/run_robot_mcp.sh')],env={
        'ROS_DOMAIN_ID':'87','GZ_PARTITION':'jerry_integrated','GZ_IP':'127.0.0.1'})
    path=ROOT/'simulation/artifacts/jerry/mcp-acceptance.json'
    try:
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write,read_timeout_seconds=timedelta(seconds=120)) as session:
                await session.initialize()
                listed=await session.list_tools()
                report['tools']=[tool.name for tool in listed.tools]
                async def call(name,args=None):
                    result=await session.call_tool(name,args or {})
                    assert not result.isError, (name,result.content)
                    if result.structuredContent is not None:return result.structuredContent
                    return json.loads(next(block.text for block in result.content if block.type=='text'))
                report['start']=await call('start_simulation',dict(headless=True,speech_enabled=False))
                deadline=time.monotonic()+85.
                while True:
                    status=await call('robot_status')
                    if status.get('navigation') and status['navigation']['fresh'] and status['navigation']['state']['ready'] and status.get('task') and status['task']['state']['session_id']:
                        break
                    assert time.monotonic()<deadline,status
                    await asyncio.sleep(.5)
                report['ready_status']=status
                report['startup_disarmed']=status['motion']['state']['status']=='STOPPED'
                assert report['startup_disarmed']
                camera=await session.call_tool('robot_camera',{})
                assert not camera.isError
                assert any(block.type=='image' for block in camera.content)
                meta=json.loads(next(block.text for block in camera.content if block.type=='text'))
                report['camera_evidence']=meta
                report['observation']=await call('remember_observation',dict(description='Acceptance captured a real Gazebo camera frame; no metric object position claimed.',evidence_id=meta['evidence_id'],request_id=uuid4().hex))
                assert report['observation']['accepted']
                report['memory']=await call('query_memory',dict(query='Acceptance captured',limit=3))
                assert report['memory']['records']
                report['explore']=await call('start_exploration',dict(request_id=uuid4().hex,maximum_active_s=30.,maximum_wall_s=60.,maximum_distance_m=1.))
                assert report['explore']['accepted'],report['explore']
                task_id=report['explore']['task_id']
                motion=False
                deadline=time.monotonic()+15.
                while time.monotonic()<deadline:
                    status=await call('robot_status')
                    if status.get('motion') and status['motion']['state']['motion_enabled']:
                        motion=True
                        break
                    await asyncio.sleep(.05)
                assert motion,status
                report['authorized_motion']=True
                report['task_status']=await call('get_task_status',dict(task_id=task_id))
                started=time.monotonic()
                report['stop']=await call('stop_robot')
                assert report['stop']['motion_stop']['success']
                await asyncio.sleep(.5)
                for _ in range(8):
                    status=await call('robot_status')
                    assert not status['motion']['state']['motion_enabled']
                    await asyncio.sleep(.1)
                report['no_restart_after_stop']=True
                report['terminal']=await call('get_task_status',dict(task_id=task_id))
                assert report['terminal']['status'] in ('CANCELLED','FAILED'),report['terminal']
                report['passed']=True
    finally:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2))
        print(json.dumps({key:value for key,value in report.items() if key not in ('memory','ready_status')},indent=2))


if __name__=='__main__':asyncio.run(main())
