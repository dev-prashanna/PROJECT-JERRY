import asyncio
import pytest
from jerry import mcp_server as server


class FakeBridge:
    def __init__(self):self.calls=[]
    def status(self):return {'backend':'gazebo','planner':'external_codex_mcp'}
    async def explore(self,*args):
        self.calls.append(args)
        return {'accepted':True,'task_id':'task-1','status':'ADMITTED'}
    async def stop(self):
        self.calls.append('stop')
        return {'motion_stop':{'success':True}}


def test_mcp_schema_and_bounded_dispatch(monkeypatch):
    fake=FakeBridge()
    monkeypatch.setattr(server,'bridge',fake)
    async def run():
        tools=await server.mcp.list_tools()
        names={tool.name for tool in tools}
        assert {'robot_status','robot_camera','start_exploration','stop_robot','query_memory','speak','listen_once'}<=names
        assert not {'shell','publish_velocity','drive_motor','llm_chat'} & names
        await server.mcp.call_tool('start_exploration',dict(request_id='request',maximum_active_s=10.))
        assert fake.calls[0]==('request',10.,300.,10.)
        await server.mcp.call_tool('stop_robot',{})
        assert fake.calls[-1]=='stop'
        tool=next(t for t in tools if t.name=='robot_status')
        assert tool.annotations.readOnlyHint
    asyncio.run(run())


@pytest.mark.parametrize('arguments',[{'request_id':'x','maximum_active_s':121.},
    {'request_id':'x','maximum_distance_m':-1.},{'request_id':''},{'request_id':'x','maximum_wall_s':float('nan')}])
def test_invalid_mcp_motion_arguments_never_reach_executive(monkeypatch,arguments):
    fake=FakeBridge()
    monkeypatch.setattr(server,'bridge',fake)
    async def run():
        with pytest.raises(Exception):await server.mcp.call_tool('start_exploration',arguments)
    asyncio.run(run())
    assert not fake.calls


def test_bounded_wait_returns_terminal_result_without_new_motion(monkeypatch):
    monkeypatch.setattr(server,'get_task_status',lambda task_id:dict(task_id=task_id,status='CANCELLED',travelled_m=0.))
    result=asyncio.run(server.wait_for_task('task-1',maximum_wait_s=.1))
    assert result['status']=='CANCELLED'


def test_bounded_wait_reports_real_progress(monkeypatch):
    states=iter([dict(status='RUNNING',travelled_m=.05),dict(status='RUNNING',travelled_m=.20)])
    monkeypatch.setattr(server,'get_task_status',lambda _:next(states))
    result=asyncio.run(server.wait_for_task('task-1',maximum_wait_s=1.,minimum_travelled_m=.15))
    assert result['travelled_m']==.20
