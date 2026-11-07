#!/usr/bin/env python3
"""Live Gazebo RGB/depth dashboard and conservative local avoidance using the existing DepthEngine."""
from __future__ import annotations
import argparse,os,time,threading,json,sys,importlib,types
from pathlib import Path
from dataclasses import dataclass
from collections import deque
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
from realtime_depth_app.src.config import load_config,validate_config
from realtime_depth_app.src.camera import LatestSlot
from realtime_depth_app.src.robot_avoidance import RangeReading,analyze_depth,AvoidancePolicy,Decision,SIDES
from realtime_depth_app.src.robot_attitude import AttitudeFilter
from realtime_depth_app.src.robot_slam import Calibration,SlamWorker,CuriosityPolicy,localization_health
from realtime_depth_app.src.robot_motion import CommandSmoother
from realtime_depth_app.src.robot_detection import DetectionWorker,annotate_objects
from realtime_depth_app.src.robot_depth_worker import DepthWorker,Prediction
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.clock import Clock as RosClock
from rclpy.clock import ClockType
from rclpy.parameter_client import AsyncParameterClient
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image,Range,Imu,MagneticField,CameraInfo
from geometry_msgs.msg import Twist,PoseStamped,TransformStamped
from nav_msgs.msg import OccupancyGrid,Path as RosPath
from tf2_ros import TransformBroadcaster
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String,Int32
from std_srvs.srv import SetBool
from ros_gz_interfaces.srv import ControlWorld

@dataclass
class Packet:
    id:int
    bgr:np.ndarray
    received:float
    stamp:object
    frame_id:str
class RobotNode(Node):
    def __init__(self,frames,results,worker,auto=False,mode='explore',slam=None,slam_frames=None,detector=None,detector_frames=None):
        super().__init__('vbmd_depth_avoidance');self.frames=frames;self.results=results;self.worker=worker
        self.lock=threading.RLock();self.sonars={};self.frame=None;self.frame_count=0;self.sim_time=None;self.clock_received=-1e9
        self.last_control_wall=time.monotonic()
        self.world_control=self.create_client(ControlWorld,'/world/'+os.environ.get('JERRY_WORLD_NAME','vbmd_lab')+'/control');self.world_control_status=''
        self.active=False;self.arming=False;self.auto_pending=auto;self.spin_error=None;self.arm_generation=0;self.parameter_client=AsyncParameterClient(self,'/vbmd_gazebo_adapter');self.prediction=None;self.policy=AvoidancePolicy();self.decision=Decision(0.,0.,'MONITOR','Avoidance not armed')
        self.monitor_reason='Preview only - press Start movement'
        self.start_requested=auto;self.arm_started=-1e9;self.arm_retry_at=0.
        self.motion=CommandSmoother()
        self.detector=detector;self.detector_frames=detector_frames;self.last_detection_id=-1
        self.objects_pub=self.create_publisher(String,'/perception/objects',10)
        self.mode=mode;self.attitude=AttitudeFilter();self.imu_received=self.mag_received=-1e9
        self.attitude_history=deque(maxlen=400);self.last_slam_frame=0
        self.slam=slam;self.slam_frames=slam_frames;self.mapping_status={};self.calibration=None;self.last_map_publish=-1e9;self.curiosity=CuriosityPolicy()
        self.map_pub=self.create_publisher(OccupancyGrid,'/map',1)
        self.path_pub=self.create_publisher(RosPath,'/exploration/path',1)
        self.pose_pub=self.create_publisher(PoseStamped,'/slam/pose',10)
        self.tf_broadcaster=TransformBroadcaster(self)
        self.history=deque(maxlen=2000);self.last_depth_id=-1;self.started=time.monotonic();self.enabled_commands=0
        self.command_pub=self.create_publisher(Twist,'/cmd_vel',10)
        self.depth_pub=self.create_publisher(Image,'/perception/depth_relative',qos_profile_sensor_data)
        self.status_pub=self.create_publisher(String,'/perception/avoidance_status',10)
        self.create_subscription(Image,'/camera/image_raw',self.image,qos_profile_sensor_data)
        self.create_subscription(CameraInfo,'/camera/camera_info',self.camera_info,qos_profile_sensor_data)
        self.create_subscription(Clock,'/clock',self.clock,10)
        self.create_subscription(Imu,'/imu/data_raw',self.imu,qos_profile_sensor_data)
        self.create_subscription(MagneticField,'/magnetometer/data_raw',self.magnetic,qos_profile_sensor_data)
        for side in SIDES:self.create_subscription(Range,'/ultrasonic/'+side,lambda m,s=side:self.range(s,m),qos_profile_sensor_data)
        self.create_subscription(Int32,'/sim/keyboard',self.keyboard,10)
        self.create_service(SetBool,'/perception/set_avoidance',self.set_avoidance)
        self.create_timer(.05,self.control,clock=RosClock(clock_type=ClockType.STEADY_TIME))
    def clock(self,m):
        with self.lock:
            t=m.clock.sec+m.clock.nanosec*1e-9
            if self.sim_time is not None and t<self.sim_time-.1:
                self.attitude=AttitudeFilter();self.policy.reset();self.imu_received=self.mag_received=-1e9;self.attitude_history.clear();self.last_slam_frame=0
            self.sim_time=t;self.clock_received=time.monotonic()
    def imu(self,m):
        with self.lock:
            g=m.angular_velocity;a=m.linear_acceleration;t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
            self.attitude.update_imu(t,[g.x,g.y,g.z],[a.x,a.y,a.z]);self.imu_received=time.monotonic()
            self.attitude_history.append((t,self.attitude.roll,self.attitude.pitch,self.attitude.yaw,self.attitude.heading_ready))
    def magnetic(self,m):
        with self.lock:
            f=m.magnetic_field;t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
            self.attitude.update_mag(t,[f.x,f.y,f.z]);self.mag_received=time.monotonic()
    def set_mode(self,mode):
        with self.lock:
            if mode not in ['avoid','explore']:return
            self.mode=mode;self.policy.reset();self.curiosity.reset()
            self.motion.reset()
            if self.active:self.command_pub.publish(Twist())
    def navigation_snapshot(self):
        with self.lock:
            a=self.attitude.snapshot();a['imu_age_s']=time.monotonic()-self.imu_received;a['mag_age_s']=time.monotonic()-self.mag_received
            a['ready']=a['ready'] and a['imu_age_s']<.5 and a['mag_age_s']<.7 and self.sim_time is not None and self.attitude.last_imu is not None and self.attitude.last_mag is not None and -.1<=self.sim_time-self.attitude.last_imu<.35 and -.1<=self.sim_time-self.attitude.last_mag<.6
            return a,{},self.mode
    def image(self,m):
        if m.encoding not in ('rgb8','bgr8','rgba8','bgra8'):return
        channels=4 if 'a' in m.encoding else 3
        arr=np.frombuffer(m.data,np.uint8).reshape(m.height,m.step)[:,:m.width*channels].reshape(m.height,m.width,channels)[:,:,:3]
        if m.encoding.startswith('rgb'):arr=arr[:,:,::-1]
        with self.lock:
            self.frame_count+=1;self.frame=Packet(self.frame_count,arr.copy(),time.monotonic(),m.header.stamp,m.header.frame_id);self.frames.put(self.frame)
            if self.detector_frames is not None:self.detector_frames.put(self.frame)
    def camera_info(self,m):
        k=np.asarray(m.k,float).reshape(3,3)
        # This frontend uses rectified pinhole images and the supplied CAD mount.
        if np.isfinite(k).all() and k[0,0]>0 and k[1,1]>0 and not any(abs(d)>1e-8 for d in m.d):
            with self.lock:
                self.calibration=Calibration(k)
    def range(self,side,m):
        with self.lock:
            t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
            self.sonars[side]=(float(m.range),time.monotonic(),t,float(m.min_range),float(m.max_range))
    def keyboard(self,m):
        if m.data in [87,83,65,68,88,32,16777235,16777237,16777234,16777236]:self.arm(False,'Gazebo keyboard took manual control')
    def arm(self,enabled,why='Avoidance not armed'):
        with self.lock:
            if enabled and (self.active or self.arming):return
            self.start_requested=enabled
            if hasattr(self,'motion'):self.motion.reset()
            self.auto_pending=False;self.arm_generation+=1;generation=self.arm_generation;was=self.active;self.active=False;self.arming=False;self.policy.reset();self.curiosity.reset()
            if was:self.command_pub.publish(Twist())
            if not enabled:
                self.monitor_reason=why
                self.decision=Decision(0.,0.,'MONITOR',why);return
            if self.spin_error:
                self.decision=Decision(0.,0.,'STOP','ROS connection failed');return
            if not self.parameter_client.services_are_ready():
                self.auto_pending=True;self.arm_retry_at=time.monotonic()+1.
                self.decision=Decision(0.,0.,'WAITING','Start requested; waiting for the Gazebo command adapter');return
            self.arm_started=time.monotonic()
            self.arming=True;self.decision=Decision(0.,0.,'ARMING','Disabling the adapter’s sonar-only autonomy')
            future=self.parameter_client.set_parameters([Parameter('autonomy',value=False)])
            def done(result):
                with self.lock:
                    if generation!=self.arm_generation:return
                    self.arming=False
                    try:
                        response=result.result()
                        if not all(r.successful for r in response.results):raise RuntimeError('Adapter rejected mode change')
                        self.active=True
                    except Exception as exc:
                        self.start_requested=False;self.monitor_reason='Could not acquire command control: '+str(exc)
                        self.decision=Decision(0.,0.,'STOP',self.monitor_reason)
            future.add_done_callback(done)
    def set_avoidance(self,request,response):
        self.arm(request.data)
        response.success=(not request.data) or self.active or self.arming or self.start_requested
        response.message='Arming requested; check avoidance_status' if request.data and response.success else self.decision.reason
        return response
    def set_sim_paused(self,paused):
        self.arm(False,'Simulation pause/resume requested; press Start movement to rearm')
        if not self.world_control.service_is_ready():
            self.world_control_status='Gazebo control bridge unavailable; restart with simulation/run_robot.sh'
            return
        request=ControlWorld.Request();request.world_control.pause=paused
        self.world_control_status='Requesting Gazebo '+('pause' if paused else 'resume')
        future=self.world_control.call_async(request)
        def done(result):
            try:self.world_control_status=('Gazebo paused' if paused else 'Gazebo running') if result.result().success else 'Gazebo rejected the control request'
            except Exception as exc:self.world_control_status='Gazebo control failed: '+str(exc)
        future.add_done_callback(done)
    def readings(self,now):
        return {s:RangeReading(v,now-recv,(self.sim_time-ts) if self.sim_time is not None else 1e9,minimum,maximum) for s,(v,recv,ts,minimum,maximum) in self.sonars.items()}
    def control(self):
        with self.lock:
            now=time.monotonic();self.last_control_wall=now;new=self.results.get()
            if self.arming and now-self.arm_started>3.:
                self.arm_generation+=1;self.arming=False;self.auto_pending=True;self.arm_retry_at=now+1.
            if self.auto_pending and now>=self.arm_retry_at and self.parameter_client.services_are_ready():self.arm(True)
            if new:self.prediction=new
            p=self.prediction;rs=self.readings(now)
            camera_age=now-self.frame.received if self.frame else 1e9
            depth_age=now-p.packet.received if p else 1e9
            ros_depth_age=(self.sim_time-p.packet.stamp.sec-p.packet.stamp.nanosec*1e-9) if p and self.sim_time is not None else 1e9
            d=self.policy.decide(p.view if p else None,rs,now,max(depth_age,max(0.,ros_depth_age)),camera_age,now-self.clock_received)
            attitude,roam,mode=self.navigation_snapshot()
            if self.slam_frames is not None and self.calibration is not None and self.frame and self.frame.id!=self.last_slam_frame and self.attitude_history:
                stamp=self.frame.stamp.sec+self.frame.stamp.nanosec*1e-9
                sample=min(self.attitude_history,key=lambda a:abs(a[0]-stamp))
                if abs(sample[0]-stamp)<.025:
                    ranges={s:(v,maximum,ts) for s,(v,recv,ts,minimum,maximum) in self.sonars.items() if now-recv<.8}
                    self.slam_frames.put((self.frame.bgr,stamp,self.calibration,(*sample[1:4],attitude['ready'] and sample[4]),ranges))
                    self.last_slam_frame=self.frame.id
            slam=self.slam.snapshot() if self.slam else None
            if mode=='explore':
                ready=bool(slam and slam['ready'] and attitude['ready'] and self.sim_time is not None and slam['stamp'] is not None and -.1<=self.sim_time-slam['stamp']<.45)
                if slam:
                    slam.update(localization_health(slam,self.sim_time,attitude['ready'],camera_age,now-self.clock_received))
                d=self.curiosity.decide(slam,rs,d,ready,self.sim_time or 0.)
                if ready and slam and not slam.get('plan_ready',False):
                    d=Decision(0.,0.,'PLANNING','Waiting for a fresh route through the remembered map')
            if slam:self.mapping_status={'x_m':slam['pose'][0],'y_m':slam['pose'][1],'vo_inliers':slam['inliers'],'observed_m2':slam['observed_m2'],'loop_closures':slam['loop_closures']}
            if slam and slam['stamp'] is not None and now-self.last_map_publish>.5:
                self.last_map_publish=now;self.publish_map(slam)
            if self.worker.error:d=Decision(0.,0.,'STOP','Depth inference failed')
            if self.detector and self.detector.processed==0 and not self.detector.error:
                d=Decision(0.,0.,'STOP','Warming the YOLO detector; movement held until live perception is ready')
            if self.active:
                self.decision=self.motion.apply(d,now);m=Twist();m.linear.x=self.decision.linear;m.angular.z=self.decision.angular;self.command_pub.publish(m);self.enabled_commands+=1
            elif not self.arming:self.decision=Decision(0.,0.,'WAITING' if self.start_requested else 'MONITOR','Start requested; waiting for Gazebo' if self.start_requested else self.monitor_reason)
            if p and p.packet.id!=self.last_depth_id:
                self.last_depth_id=p.packet.id
                out=Image();out.header.stamp=p.packet.stamp;out.header.frame_id=p.packet.frame_id;out.width=p.depth.shape[1];out.height=p.depth.shape[0];out.encoding='32FC1';out.step=out.width*4;out.is_bigendian=0;out.data=np.asarray(p.depth,dtype='<f4').tobytes();self.depth_pub.publish(out)
            detection=self.detector.snapshot() if self.detector else None
            objects=detection['prediction'] if detection else None
            if objects and objects.packet.id!=self.last_detection_id:
                self.last_detection_id=objects.packet.id
                payload={'frame_id':objects.packet.id,'camera_frame':objects.packet.frame_id,
                         'stamp_s':objects.packet.stamp.sec+objects.packet.stamp.nanosec*1e-9,
                         'model':self.detector.model_path.name,'coordinate_units':'image pixels',
                         'inference_ms':objects.inference_ms,
                         'objects':[{'class':box.label,'confidence':box.confidence,'xyxy':box.xyxy} for box in objects.boxes]}
                msg=String();msg.data=json.dumps(payload);self.objects_pub.publish(msg)
            status={'active':self.active,'simulation_time_s':self.sim_time,'mode':self.mode,'attitude':attitude,'mapping':dict(self.mapping_status),'state':self.decision.state,'reason':self.decision.reason,'linear_m_s':self.decision.linear,'yaw_rad_s':self.decision.angular,'depth_unit':'relative inverse depth, not metres','depth_age_wall_s':depth_age if p else None,'depth_age_sensor_s':ros_depth_age if p else None,'inference_ms':p.inference_ms if p else None,'preprocess_ms':p.preprocess_ms if p else None,'postprocess_ms':p.postprocess_ms if p else None,'scores':list(p.view.scores) if p else None,'near_coverage':list(p.view.fractions) if p else None,'ultrasound_m':{s:(r.value if np.isfinite(r.value) else ('NO_ECHO' if r.value==np.inf else 'TOO_CLOSE')) for s,r in rs.items()},'ultrasound_age_s':{s:r.age_wall for s,r in rs.items()},'frame_id':p.packet.id if p else None,'suggested_state':d.state,'suggested_reason':d.reason,'suggested_linear_m_s':d.linear,'suggested_yaw_rad_s':d.angular}
            status['start_requested']=self.start_requested;status['arming']=self.arming
            status['command_time_s']=now
            status['target_linear_m_s']=d.linear;status['target_yaw_rad_s']=d.angular
            if detection:status['objects']={'status':detection['status'],'age_s':detection['age_s'],'frames':detection['frames'],'count':len(objects.boxes) if objects else 0}
            if slam:status['slam']={k:slam.get(k) for k in ['pose','ready','status','inliers','good_streak','trusted_pose_age_s','processing_ms','keyframes','loop_closures','observed_m2','free_m2','goal','goal_id','completed_goals','goal_distance_m','route_revision','planner_state','interest','visited_places','age_s','revision','plan_ready','plan_age_s']}
            msg=String();msg.data=json.dumps(status);self.status_pub.publish(msg);self.history.append(status)
    def publish_map(self,s):
        import math
        stamp=rclpy.time.Time(seconds=s['stamp']).to_msg()
        m=OccupancyGrid();m.header.frame_id='map';m.header.stamp=stamp
        m.info.resolution=s['resolution'];m.info.width=s['grid'].shape[1];m.info.height=s['grid'].shape[0]
        m.info.origin.position.x=s['origin'][0];m.info.origin.position.y=s['origin'][1];m.info.origin.orientation.w=1.
        m.data=s['grid'].ravel().tolist();self.map_pub.publish(m)
        def pose(x,y,yaw=0.):
            p=PoseStamped();p.header=m.header;p.pose.position.x=float(x);p.pose.position.y=float(y)
            p.pose.orientation.z=math.sin(yaw/2);p.pose.orientation.w=math.cos(yaw/2);return p
        p=pose(*s['pose']);self.pose_pub.publish(p)
        transform=TransformStamped();transform.header=m.header;transform.child_frame_id='base_footprint'
        transform.transform.translation.x=p.pose.position.x;transform.transform.translation.y=p.pose.position.y;transform.transform.rotation=p.pose.orientation
        self.tf_broadcaster.sendTransform(transform)
        path=RosPath();path.header=m.header;path.poses=[pose(*xy) for xy in s['path']];self.path_pub.publish(path)
    def snapshot(self):
        with self.lock:
            now=time.monotonic();readings=self.readings(now)
            stale=(not self.frame or now-self.frame.received>.65 or not self.prediction or now-self.prediction.packet.received>.45 or now-self.clock_received>1. or any(s not in readings or not readings[s].usable() for s in SIDES))
            if stale and self.decision.state!='ARMING':
                was_moving=bool(self.decision.linear or self.decision.angular)
                stopped=Decision(0.,0.,'STOP','Waiting for fresh sensors/depth; movement held')
                if self.active and was_moving:
                    if now-getattr(self,'last_control_wall',-1e9)>.25:
                        self.active=False;self.start_requested=False;self.policy.reset();self.decision=stopped
                        try:self.command_pub.publish(Twist())
                        except Exception:pass
                return self.frame,self.prediction,readings,stopped,self.active
            return self.frame,self.prediction,readings,self.decision,self.active


def spin_node(node):
    from rclpy.executors import ExternalShutdownException
    try:rclpy.spin(node)
    except ExternalShutdownException:pass
    except Exception:
        import traceback
        node.spin_error=traceback.format_exc();node.arm(False);print(node.spin_error,file=sys.stderr,flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--auto',action='store_true');ap.add_argument('--mode',choices=['explore','avoid'],default='explore');ap.add_argument('--headless',action='store_true');ap.add_argument('--duration',type=float,default=0);ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'robot_outputs');ap.add_argument('--input-size',type=int,default=392);ap.add_argument('--device',default='auto');ap.add_argument('--no-detection',action='store_true');ap.add_argument('--detector-model',type=Path,default=Path(__file__).resolve().parents[1]/'models/yolo/yolo11n.pt');ap.add_argument('--detector-fps',type=float,default=4.);ap.add_argument('--detector-confidence',type=float,default=.30);ap.add_argument('--detector-device',default='auto');args=ap.parse_args()
    if not np.isfinite(args.detector_fps) or args.detector_fps<=0:ap.error('--detector-fps must be positive')
    if not 0.<args.detector_confidence<1.:ap.error('--detector-confidence must be between 0 and 1')
    # One live dashboard/controller per ROS domain: avoid duplicate command/service owners.
    import fcntl
    domain=int(os.environ.get('ROS_DOMAIN_ID','0'))
    instance_file=open(f'/tmp/vbmd_robot_vision_{domain}.lock','a+')
    try:fcntl.flock(instance_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A robot vision/exploration instance is already running in this ROS domain. Close it before starting another.')
    instance_file.seek(0);instance_file.truncate();instance_file.write(str(os.getpid()));instance_file.flush()
    cfg=load_config();cfg.model.input_size=args.input_size;cfg.model.device=args.device;cfg.model.warmup_iters=1;cfg.model.precision='autocast_fp16' if args.device!='cpu' else 'fp32'
    # CUDA mode is selected only if available; leave CPU fallback in supported precision.
    import torch
    import cv2
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if not torch.cuda.is_available():cfg.model.precision='fp32'
    cfg.depth.smoothing.enabled=False
    validate_config(cfg)
    frames=LatestSlot();results=LatestSlot();worker=DepthWorker(frames,results,cfg)
    slam_frames=LatestSlot();slam=SlamWorker(slam_frames)
    detector_frames=LatestSlot() if not args.no_detection else None
    detector=DetectionWorker(detector_frames,args.detector_model,args.detector_fps,args.detector_confidence,args.detector_device,lambda:not worker.busy.is_set()) if detector_frames is not None else None
    rclpy.init();node=RobotNode(frames,results,worker,args.auto,args.mode,slam,slam_frames,detector,detector_frames);spin=threading.Thread(target=lambda:spin_node(node),daemon=True);spin.start();worker.start();slam.start()
    if detector:detector.start()
    def save():
        args.output.mkdir(parents=True,exist_ok=True);frame,p,rs,d,active=node.snapshot()
        report={'checkpoint':cfg.model.repo_id,'device':p.device if p else None,'depth_predictions':worker.processed,'last_processed_frame_id':node.last_depth_id,'received_frames':node.frame_count,'worker_status':worker.status,'worker_error':worker.error,'ros_error':node.spin_error,'commands_published':node.enabled_commands,'mapping':dict(node.mapping_status),'history':list(node.history)}
        (args.output/'live_report.json').write_text(json.dumps(report,indent=2))
        slam.save(args.output)
        if detector:
            detection=detector.snapshot();result=detection['prediction']
            (args.output/'objects_report.json').write_text(json.dumps({'status':detection['status'],'error':detection['error'],'frames':detection['frames'],'frame_id':result.packet.id if result else None,'objects':[{'class':box.label,'confidence':box.confidence,'xyxy':box.xyxy} for box in result.boxes] if result else []},indent=2))
            if result:
                import cv2
                cv2.imwrite(str(args.output/'objects.png'),annotate_objects(result))
        if p:
            import cv2
            from realtime_depth_app.src.robot_slam import render_map
            cv2.imwrite(str(args.output/'local_map.png'),render_map(slam.snapshot()));cv2.imwrite(str(args.output/'rgb.png'),p.packet.bgr);cv2.imwrite(str(args.output/'relative_depth.png'),cv2.applyColorMap(np.uint8(np.clip(p.view.proximity,0,1)*255),cv2.COLORMAP_TURBO))
    try:
        if args.headless:
            until=time.monotonic()+(args.duration or 15)
            next_status=0.
            while time.monotonic()<until and not worker.error and not node.spin_error:
                if time.monotonic()>=next_status:
                    next_status=time.monotonic()+5.
                    s=slam.snapshot();latest=node.history[-1] if node.history else {}
                    print(json.dumps({'model':worker.status,'state':latest.get('state'),'reason':latest.get('reason'),'slam':s['status'],'pose':s['pose'],'observed_m2':s['observed_m2'],'planner':s['planner_state'],'commands':[latest.get('linear_m_s'),latest.get('yaw_rad_s')]}),flush=True)
                time.sleep(.1)
            save()
            if worker.error or node.spin_error or node.prediction is None or slam.error:raise RuntimeError(slam.error or node.spin_error or worker.status)
            if detector and (detector.error or detector.latest is None):raise RuntimeError(detector.error or 'No live YOLO predictions received')
        else:
            from robot_window import run_window
            run_window(node,worker,args,save)
            save()
    finally:
        if detector:detector.stop()
        node.arm(False);worker.stop();slam.stop();rclpy.shutdown();spin.join(timeout=2);worker.join(timeout=3);slam.join(timeout=3);node.destroy_node()
        if detector:detector.join(timeout=3)
if __name__=='__main__':main()
