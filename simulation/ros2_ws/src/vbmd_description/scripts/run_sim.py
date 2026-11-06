#!/usr/bin/env python3
"""MuJoCo standalone/ROS runner. W/S forward/reverse, A/D turn, X stop."""
import argparse,os,time,json,math
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--headless',action='store_true');p.add_argument('--ros',action='store_true');p.add_argument('--ground-truth',action='store_true')
 p.add_argument('--autonomy',action='store_true',help='Sensor-only reactive obstacle avoidance, not SLAM')
 p.add_argument('--duration',type=float,default=0,help='Seconds of simulated time; zero runs until stopped')
 p.add_argument('--output',type=Path);p.add_argument('--fast',action='store_true',help='Disable realtime pacing (offline testing)')
 p.add_argument('--no-camera',action='store_true',help='Skip RGB rendering')
 args=p.parse_args()
 if args.headless:os.environ.setdefault('MUJOCO_GL','egl')
 import mujoco
 from sim_core import RobotSim
 sim=RobotSim();node=None;viewer=None;start=time.monotonic();frames=0
 if args.ros:
  import rclpy
  from ros_bridge import SensorBridge
  rclpy.init();node=SensorBridge(sim,args.ground_truth)
 if not args.headless:
  import mujoco.viewer
  def key(k):
   commands={ord('W'):(.12,0),ord('S'):(-.10,0),ord('A'):(0,1),ord('D'):(0,-1),ord('X'):(0,0)}
   if k in commands:sim.set_twist(*commands[k])
  viewer=mujoco.viewer.launch_passive(sim.model,sim.data,key_callback=key)
  viewer.cam.distance=.55;viewer.cam.azimuth=135;viewer.cam.elevation=-25;viewer.cam.lookat[:]=[0,0,.06]
  viewer.opt.geomgroup[3]=0;viewer.opt.geomgroup[4]=0
 print('VBMD MuJoCo ready. W/S/A/D/X controls; .5 s command timeout. Ctrl-C exits.',flush=True)
 try:
  while not args.duration or sim.data.time<args.duration:
   if viewer and not viewer.is_running():break
   if node:
    if not rclpy.ok():break
    rclpy.spin_once(node,timeout_sec=0)
   if args.autonomy:sim.set_twist(*sim.reactive_command())
   sim.advance(.01)
   if node:node.publish(render=not args.no_camera)
   if viewer:viewer.sync()
   if not args.fast:
    delay=start+sim.data.time-time.monotonic()
    if delay>0:time.sleep(min(delay,.02))
  if args.output:
   args.output.mkdir(parents=True,exist_ok=True)
   if not args.no_camera:
    import cv2
    cv2.imwrite(str(args.output/'esp32_camera.png'),sim.camera()[:,:,::-1]);frames+=1
    render=mujoco.Renderer(sim.model,height=900,width=1200)
    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.distance=.48;cam.azimuth=135;cam.elevation=-25;cam.lookat[:]=sim.data.qpos[:3]+[0,0,.06]
    opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.geomgroup[4]=0
    render.update_scene(sim.data,camera=cam,scene_option=opt);cv2.imwrite(str(args.output/'physics_robot.png'),render.render()[:,:,::-1]);render.close()
   report={'simulated_seconds':sim.data.time,'elapsed_seconds':time.monotonic()-start,'robot_position_debug_m':sim.data.qpos[:3].tolist(),'ultrasonic_m':{k:(v if math.isfinite(v) else None) for k,v in sim.ranges.items()},'range_update_times_s':sim.range_times,'finite_state':bool(np.isfinite(sim.data.qpos).all()),'visual_meshes':sim.model.nmesh,'actuators':sim.model.nu,'mujoco_version':mujoco.__version__,'ground_truth_published':args.ground_truth and args.ros,'mode':'reactive avoidance' if args.autonomy else 'manual/ROS commands'}
   (args.output/'run_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 except KeyboardInterrupt:pass
 finally:
  if viewer:viewer.close()
  sim.close()
  if node:node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
