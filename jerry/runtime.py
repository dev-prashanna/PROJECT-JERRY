"""Process entry points for the integrated simulation runtime."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('component',choices=['supervisor','executive','memory','navigation','cognition'])
    parser.add_argument('--no-speak',action='store_true')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--no-detection',action='store_true')
    parser.add_argument('--device',default='auto')
    args = parser.parse_args()
    from .ros_utils import run_node
    if args.component == 'supervisor':
        from .supervisor import Supervisor
        run_node(Supervisor)
    elif args.component == 'executive':
        from .executive import Executive
        run_node(Executive)
    elif args.component == 'memory':
        from .memory import Memory
        run_node(Memory)
    elif args.component == 'cognition':
        from .cognition import Cognition
        run_node(lambda:Cognition(speak=not args.no_speak,offline=args.offline))
    else:
        from .navigation import Navigation
        from realtime_depth_app.src.camera import LatestSlot
        from realtime_depth_app.src.config import load_config,validate_config
        from realtime_depth_app.src.robot_depth_worker import DepthWorker
        from realtime_depth_app.src.robot_slam import SlamWorker
        from realtime_depth_app.src.robot_detection_worker import DetectionProcessWorker
        import torch
        import cv2
        torch.set_num_threads(2)
        cv2.setNumThreads(1)
        config = load_config(Path(__file__).resolve().parents[1]/'realtime_depth_app/config.yaml')
        config.model.input_size = 392
        config.model.device = args.device
        config.model.precision = 'autocast_fp16' if args.device!='cpu' and torch.cuda.is_available() else 'fp32'
        config.model.warmup_iters = 1
        config.depth.smoothing.enabled = False
        validate_config(config)
        frames,results,slam_frames = LatestSlot(),LatestSlot(),LatestSlot()
        worker = DepthWorker(frames,results,config)
        slam = SlamWorker(slam_frames)
        detector_frames = LatestSlot() if not args.no_detection else None
        detector = DetectionProcessWorker(detector_frames,Path('models/yolo/yolo11n.pt'),2.,.30) if detector_frames else None
        def factory():
            node = Navigation(frames,results,worker,slam=slam,slam_frames=slam_frames,detector=detector,detector_frames=detector_frames)
            worker.start()
            slam.start()
            if detector:detector.start()
            def close():
                node.active = False
                worker.stop()
                slam.stop()
                if detector:detector.stop()
                worker.join(timeout=4.)
                slam.join(timeout=4.)
                if detector:detector.join(timeout=4.)
                path = Path(os.environ.get('JERRY_RUN_DIR','simulation/artifacts/jerry'))
                slam.save(path/'map')
            node.close = close
            return node
        run_node(factory)


if __name__ == '__main__':
    main()
