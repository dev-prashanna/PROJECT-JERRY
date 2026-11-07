"""Exercise an actual installed Codex client against Jerry's registered MCP."""
from datetime import datetime,timezone
import json
import base64
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[2]


def extract_wire_checks(folder,events,nonce):
    observed=[]
    for event in events:
        item=event.get('item',{})
        if event.get('type')!='item.completed' or item.get('type')!='mcp_tool_call':continue
        result=item.get('result') or {}
        parsed=result.get('structured_content')
        if parsed is None:
            for block in result.get('content',[]):
                if block.get('type')=='text':
                    try:parsed=json.loads(block['text']);break
                    except ValueError:pass
        observed.append((item.get('tool'),parsed))
        if item.get('tool') in ('robot_camera','robot_map'):
            for block in result.get('content',[]):
                if block.get('type')=='image':
                    (folder/(item['tool']+'.png')).write_bytes(base64.b64decode(block['data']))
    task_ids={d['task_id'] for n,d in observed if n=='start_exploration' and d and d.get('accepted')}
    checks={
        'admission_acknowledged':bool(task_ids),
        'reported_motion':any(n=='robot_status' and d and d.get('motion')
            and d['motion']['state']['task_id'] in task_ids and d['motion']['state']['motion_enabled'] for n,d in observed),
        'measured_progress_returned':any(n=='wait_for_task' and d and d.get('task_id') in task_ids
            and d.get('travelled_m',0)>=.20 for n,d in observed),
        'stop_acknowledged':any(n=='stop_robot' and d and d.get('motion_stop',{}).get('success') for n,d in observed),
        'terminal_cancelled':any(n=='get_task_status' and d and d.get('task_id') in task_ids
            and d.get('status')=='CANCELLED' for n,d in observed),
        'memory_recalled_from_store':any(n=='query_memory' and d and d.get('accepted')
            and any(nonce in r.get('text','') for r in d.get('records',[])) for n,d in observed),
        'actual_camera_image':(folder/'robot_camera.png').exists(),
        'actual_map_image':(folder/'robot_map.png').exists()}
    (folder/'wire-checks.json').write_text(json.dumps(checks,indent=2))
    return checks


def main():
    folder=ROOT/'simulation/artifacts/jerry'/('codex-validation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    folder.mkdir(parents=True)
    nonce=uuid4().hex
    prompt=f'''This is an authorized live acceptance test of the user's Jerry robot in Gazebo. Use the registered jerry_robot MCP server to operate the simulated robot yourself. Do not implement software, read files, run shell commands, or use other MCP servers. Tool orchestration is allowed, but all robot operations must use jerry_robot MCP tools. Do not substitute a Python MCP client. Do not spawn other agents.
Your plan: check robot_status, start_simulation(headless=true, speech_enabled=false) if needed, poll readiness until navigation.ready, inspect robot_camera and robot_map, remember the explicit fact "Codex robot acceptance {nonce}", start_exploration(request_id="codex-{nonce}", maximum_active_s=60, maximum_wall_s=90, maximum_distance_m=1.0), use wait_for_task(task_id,maximum_wait_s=15,minimum_travelled_m=0.20), repeating bounded waits as needed, and inspect robot_status until you observe measured forward progress travelled_m>=0.20 and navigation.phase=EXPLORE or a terminal task; a scan rotation alone is insufficient. Use at most 8 bounded waits, then call stop_robot, query get_task_status until terminal, and query_memory for the saved nonce. Distinguish admission, running, actual motion and completion. Always stop before finishing. If tools fail, report the failure honestly. Do not claim successful movement without a returned motion state. Describe only what the returned camera image actually shows. Output the requested JSON result.'''
    schema={'type':'object','properties':{
        'passed':{'type':'boolean'},'plan':{'type':'array','items':{'type':'string'}},
        'task_id':{'type':'string'},'observed_motion':{'type':'boolean'},'terminal_status':{'type':'string'},
        'camera_description':{'type':'string'},'memory_recalled':{'type':'boolean'},'failure':{'type':'string'}},
        'required':['passed','plan','task_id','observed_motion','terminal_status','camera_description','memory_recalled','failure'],
        'additionalProperties':False}
    (folder/'schema.json').write_text(json.dumps(schema))
    (folder/'prompt.txt').write_text(prompt)
    command=['codex','--no-daemon','exec','--ephemeral','--sandbox','read-only','--json',
        '-c','mcp_servers.freecad.enabled=false','-c','mcp_servers.kicad.enabled=false',
        '-c','mcp_servers.jerry_robot.required=true','-c','mcp_servers.jerry_robot.startup_timeout_sec=30',
        '-c','mcp_servers.jerry_robot.tool_timeout_sec=120',
        '--output-schema',str(folder/'schema.json'),'-o',str(folder/'result.json'),'-']
    from validate_jerry_runtime import Evaluator
    evaluator=Evaluator()
    started=time.monotonic()
    with open(folder/'events.jsonl','w') as output,open(folder/'stderr.log','w') as error:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=output,stderr=error,text=True)
        try:process.communicate(prompt,timeout=360.)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=15.)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise RuntimeError('Actual Codex client exceeded acceptance deadline; see '+str(folder))
    evaluator.close()
    measured=evaluator.metrics(started)
    (folder/'telemetry.json').write_text(json.dumps(evaluator.rows,allow_nan=True))
    report={'exit_code':process.returncode,'elapsed_wall_s':time.monotonic()-started,'nonce':nonce,
            'actual_codex_client':True,'model_override':False,'artifacts':str(folder)}
    if (folder/'result.json').exists():
        report['codex_result']=json.loads((folder/'result.json').read_text())
    events=[]
    for line in (folder/'events.jsonl').read_text().splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:pass
    # Preserve actual wire events; a fluent final answer alone is not the evidence.
    calls=[]
    for event in events:
        item=event.get('item',{})
        if item.get('type')=='mcp_tool_call' and event.get('type')=='item.completed':
            calls.append({'server':item.get('server'),'tool':item.get('tool'),'status':item.get('status')})
    report['wire_checks']=extract_wire_checks(folder,events,nonce)
    report['completed_mcp_calls']=calls
    report['independent_measurement']=measured
    required={'robot_status','robot_camera','robot_map','remember_fact','start_exploration','get_task_status','wait_for_task','stop_robot','query_memory'}
    report['all_required_mcp_tools_used']=required<={call['tool'] for call in calls if call['server']=='jerry_robot'}
    report['no_shell_or_file_changes']=not any(event.get('item',{}).get('type') in ('command_execution','file_change') for event in events)
    report['passed']=bool(process.returncode==0 and report.get('codex_result',{}).get('passed')
        and report['all_required_mcp_tools_used'] and report['no_shell_or_file_changes'] and all(report['wire_checks'].values())
        and measured['nonzero_translation_commands']>5 and measured['actual_extent_m']>=.10)
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
