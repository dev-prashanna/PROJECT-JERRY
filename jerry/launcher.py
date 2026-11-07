"""Own and supervise one isolated Gazebo integration session."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true',help='Run Gazebo without its 3D window')
    parser.add_argument('--no-console',action='store_true')
    parser.add_argument('--validate',action='store_true',help='Run live integration acceptance and exit')
    parser.add_argument('--no-speak',action='store_true')
    parser.add_argument('--offline',action='store_true',help='Compatibility flag; runtime planning is always external through Codex MCP')
    parser.add_argument('--no-detection',action='store_true')
    parser.add_argument('--duration',type=float,default=0.)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--world',type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    domain = os.environ.get('ROS_DOMAIN_ID','87')
    instance = open('/tmp/jerry-session-'+domain+'.lock','a+')
    try:fcntl.flock(instance,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('An integrated Jerry session already owns this ROS domain')
    run = root/'simulation/artifacts/jerry'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run.mkdir(parents=True)
    os.environ['JERRY_RUN_DIR'] = str(run)
    os.environ.setdefault('JERRY_MEMORY_DB',str(root/'simulation/artifacts/jerry/memory.sqlite3'))
    (run/'manifest.json').write_text(json.dumps(dict(started=datetime.now(timezone.utc).isoformat(),domain=domain,
        partition=os.environ.get('GZ_PARTITION'),profile='flat_floor_camera_sonar',arguments=vars(args)),default=str,indent=2))
    children = []
    logs = []
    def start(name,command,console=False):
        stream = None if console else open(run/(name+'.log'),'w')
        if stream:logs.append(stream)
        p = subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT if stream else None,start_new_session=True)
        children.append((name,p))
        return p
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        gazebo = ['bash','simulation/run_gazebo.sh','protected_motion:=true','autonomy:=false']
        if args.headless:gazebo.append('headless:=true')
        if args.world:
            import xml.etree.ElementTree as ET
            os.environ['JERRY_WORLD_NAME'] = ET.parse(args.world).getroot().find('world').get('name')
            gazebo.append('world:='+str(args.world.resolve()))
        start('gazebo',gazebo)
        for component in ('memory','supervisor','executive','navigation','cognition'):
            command = [sys.executable,'-m','jerry.runtime',component]
            if component == 'navigation':
                command += ['--device',args.device]
                if args.no_detection:command.append('--no-detection')
            if component == 'cognition':
                if args.no_speak:command.append('--no-speak')
                if args.offline:command.append('--offline')
            start(component,command)
        console = None
        if args.validate:
            validation=start('acceptance',[sys.executable,'simulation/tests/check_jerry.py'])
        if not args.no_console and not args.validate:console=start('console',[sys.executable,'-m','jerry.console'],True)
        print('Jerry simulation started disarmed. Run artifacts: '+str(run),flush=True)
        print('Motion waits for valid localization, depth, ranges and an explicit mission.',flush=True)
        end = time.monotonic()+args.duration if args.duration>0 else float('inf')
        while time.monotonic()<end:
            for name,child in children:
                if child.poll() is not None:
                    if child == console or (args.validate and name=='acceptance'):return child.returncode
                    raise RuntimeError(f'{name} exited ({child.returncode}); inspect {run/name}.log')
            time.sleep(.2)
    except KeyboardInterrupt:
        pass
    finally:
        # Signal the executive first; then stop remaining children. Only this launch's process groups are owned.
        for name,child in sorted(children,key=lambda item:item[0]!='executive'):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                if name=='executive':time.sleep(.3)
        deadline = time.monotonic()+12.
        for name,child in children:
            if child.poll() is None:
                try:child.wait(timeout=max(.1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=2.)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait(timeout=2.)
        for stream in logs:stream.close()
        instance.close()
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except RuntimeError as exc:print(str(exc),file=sys.stderr);sys.exit(1)
