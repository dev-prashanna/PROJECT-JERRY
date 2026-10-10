#!/usr/bin/env python3
"""Real-time monocular depth estimation and distance inspection.

Launch from the project root:
    python3 main.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.anchor import AnchorError, ArucoAnchor
from src.calibration import CalibrationError, load_profile
from src.camera import list_cameras
from src.config import AppConfig, ConfigError, load_config
from src.depth_engine import DepthEngine, DepthResult
from src.distance_estimator import DistanceEstimator, PointStore
from src.geometry import GeometryError, Undistorter, load_intrinsics
from src.performance import PerformanceMonitor
from src.pipeline import Pipeline
from src.temporal_filter import ScalarEMA
from src.visualization import Renderer, draw_text_box, map_window_point, perf_lines_for

HELP_TEXT = [
    "keys:  q quit | p pause | c clear points | m center | r reset",
    "       d cmap | g gray | t smoothing | f perf | s screenshot | h help",
    "left click: add point      right click: remove nearest point",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time Depth Anything V2 depth + distance inspection")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--camera", type=int, default=None, help="camera index override")
    p.add_argument("--video", default=None, help="use a video file as source")
    p.add_argument("--image", default=None, help="use a still image as source (repeatable frames)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16", "autocast_fp16", "autocast_bf16"], default=None)
    p.add_argument("--input-size", type=int, default=None, help="model input size (multiple of 14)")
    p.add_argument("--frame-skip", type=int, default=None, help="run depth on every Nth frame")
    p.add_argument("--profile", default=None, help="calibration profile path")
    p.add_argument("--no-smoothing", action="store_true", help="disable temporal smoothing")
    p.add_argument("--headless", type=int, default=0, metavar="N", help="process N frames without GUI and exit")
    p.add_argument("--benchmark", action="store_true", help="run inference benchmarks and exit")
    p.add_argument("--list-cameras", action="store_true", help="list usable camera indices and exit")
    return p.parse_args(argv)


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.camera is not None:
        cfg.source.type = "camera"
        cfg.source.index = args.camera
    if args.video:
        cfg.source.type = "video"
        cfg.source.path = args.video
    if args.image:
        cfg.source.type = "image"
        cfg.source.path = args.image
    if args.device:
        cfg.model.device = args.device
    if args.precision:
        cfg.model.precision = args.precision
    if args.input_size:
        cfg.model.input_size = args.input_size
    if args.frame_skip:
        cfg.runtime.frame_skip = args.frame_skip
    if args.profile:
        cfg.calibration.profile = args.profile
    if args.no_smoothing:
        cfg.depth.smoothing.enabled = False
    if args.headless:
        cfg.runtime.headless_frames = args.headless
    return cfg


class DepthApp:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.monitor = PerformanceMonitor(device=str(cfg.model.device))
        self.engine = DepthEngine(cfg)
        self.monitor.device = str(self.engine.device)

        self.profile = None
        profile_path = cfg.resolve(cfg.calibration.profile)
        if profile_path is not None and profile_path.exists():
            try:
                self.profile = load_profile(profile_path)
            except CalibrationError as exc:
                print(f"[warn] ignoring calibration profile: {exc}", file=sys.stderr)

        self.estimator = DistanceEstimator(cfg, self.profile, self.engine.estimation_type)
        self.points = PointStore(cfg)
        self.center_ema = ScalarEMA(cfg.measurement.point_smoothing) if cfg.measurement.point_smoothing > 0 else None
        self.center_measurement = None
        self.renderer = Renderer(cfg)
        self.pipeline = Pipeline(cfg, self.engine, on_capture=self.monitor.on_capture)

        self.intrinsics = None
        self.undistorter = None
        intr_path = cfg.resolve(cfg.geometry.intrinsics)
        if intr_path is not None and intr_path.exists():
            try:
                self.intrinsics = load_intrinsics(intr_path)
            except GeometryError as exc:
                print(f"[warn] ignoring intrinsics: {exc}", file=sys.stderr)
        self.anchor = None
        if cfg.anchor.enabled and self.intrinsics is not None:
            try:
                self.anchor = ArucoAnchor(cfg.anchor.marker_length_m, cfg.anchor.dictionary)
            except AnchorError as exc:
                print(f"[warn] aruco anchor disabled: {exc}", file=sys.stderr)

        self.paused = False
        self.show_help = False
        self.running = True
        self.last_image: np.ndarray | None = None
        self.last_depth: DepthResult | None = None
        self._last_frame_id = -1
        self.screenshots = 0
        self.window = cfg.display.window_name
        self.engine_last_inference_ms = 0.0
        self.panel_width = 0

    # ------------------------------------------------------------------ helpers

    @property
    def mode_label(self) -> str:
        return self.estimator.mode

    def depth_mode_label(self) -> str:
        if self.engine.meta.estimation_type == "metric":
            return "native metric (m)"
        if self.profile is not None:
            return "calibrated approximation (m)"
        return "relative inverse depth (a.u.)"

    def calibration_label(self) -> str:
        if self.profile is None:
            return "none"
        val = self.profile.val_metrics or {}
        rmse = val.get("rmse")
        n = len(self.profile.samples)
        if rmse:
            return f"{Path(self.cfg.calibration.profile).name} (val RMSE {rmse:.2f} m, n={n})"
        return f"{Path(self.cfg.calibration.profile).name} (n={n})"

    def status_lines(self, frame_shape: tuple[int, int]) -> list[str]:
        try:
            import torch

            device = str(self.engine.device)
            if device.startswith("cuda"):
                device = f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:
            device = str(self.engine.device)
        model_short = self.engine.meta.model_id.split("/")[-1]
        h, w = frame_shape[:2]
        return [
            f"{device} | {model_short}",
            f"mode: {self.depth_mode_label()} | input {self.engine.input_size}px, frame {w}x{h}",
            f"calibration: {self.calibration_label()}",
        ]

    def _prepare_depth(self, depth_np: np.ndarray) -> np.ndarray:
        if self.undistorter is not None:
            return self.undistorter.apply_depth(depth_np)
        return depth_np

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.undistorter is not None:
            return self.undistorter.apply(frame)
        return frame

    def _measurements(self, frame: np.ndarray, depth_map: np.ndarray | None) -> dict:
        out = {"points": {}, "center": None, "anchor": None}
        if depth_map is None:
            return out
        out["points"] = self.points.update(depth_map, self.estimator)
        if self.cfg.measurement.show_center:
            h, w = depth_map.shape[:2]
            self.center_measurement = self.estimator.measure(depth_map, w // 2, h // 2, self.center_ema)
            out["center"] = self.center_measurement
        if self.anchor is not None and self.intrinsics is not None:
            intr = self.undistorter.new_intrinsics if self.undistorter is not None else self.intrinsics
            out["anchor"] = self.anchor.estimate(frame, intr)
        return out

    def _compose(
        self,
        frame: np.ndarray,
        depth_result: DepthResult | None,
        measurements: dict,
        stale: bool,
        stale_text: str,
        depth_map: np.ndarray | None,
    ) -> np.ndarray:
        self.panel_width = frame.shape[1]
        snapshot = self.monitor.snapshot()
        info = {
            "device": str(self.engine.device),
            "model": self.engine.meta.model_id,
            "mode": self.mode_label,
            "depth_mode": self.depth_mode_label(),
            "input_res": f"{self.engine.input_size}px",
            "calibration": self.calibration_label(),
        }
        unit = self.estimator.unit
        image = self.renderer.render(
            frame,
            depth_map,
            points=measurements.get("points", {}),
            center_measurement=measurements.get("center"),
            show_center=self.cfg.measurement.show_center,
            status_lines=self.status_lines(frame.shape[:2]),
            perf_lines=perf_lines_for(snapshot, self.engine_last_inference_ms, self.monitor.depth_updates),
            convention=self.engine.meta.convention,
            unit=unit,
            depth_stale=stale,
            stale_text=stale_text,
        )
        if self.show_help:
            y = frame.shape[0] - 14 - 18 * len(HELP_TEXT)
            for line in HELP_TEXT:
                draw_text_box(image, line, (8, y), scale=0.45, color=(255, 255, 0))
                y += 18
        if self.paused:
            draw_text_box(image, "PAUSED", (8, frame.shape[0] // 2 - 40), scale=0.9, color=(0, 0, 255), thickness=2)
        if measurements.get("anchor") is not None and measurements["anchor"].detected:
            a = measurements["anchor"]
            text = f"ArUco #{a.marker_id}: geometric {a.distance_m:.2f} m (Z {a.z_axial_m:.2f} m, reproj {a.reprojection_err_px:.1f}px)"
            draw_text_box(image, text, (8, frame.shape[0] - 8 - 18 * 5), scale=0.45, color=(255, 200, 0))
        return image

    # ------------------------------------------------------------------ interaction

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        if self.last_image is None or self.panel_width <= 0:
            return
        mapped = map_window_point(x, y, self.panel_width, self.renderer.window_scale)
        if mapped is None:
            return
        px, py = mapped
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.add(px, py)
        else:
            self.points.remove_nearest(px, py)

    def handle_key(self, key: int) -> None:
        k = chr(key & 0xFF).lower() if 32 <= (key & 0xFF) < 127 else ""
        if key in (27,) or k == "q":
            self.running = False
        elif k == "c":
            self.points.clear()
        elif k == "r":
            self.points.clear()
            self.renderer.reset_range()
            self.pipeline.smoother.reset()
            if self.center_ema:
                self.center_ema.reset()
        elif k == "m":
            self.cfg.measurement.show_center = not self.cfg.measurement.show_center
        elif k == "f":
            self.renderer.show_perf = not self.renderer.show_perf
        elif k == "d":
            self.renderer.cycle_colormap()
        elif k == "g":
            self.renderer.toggle_grayscale()
        elif k == "t":
            self.pipeline.smoother.enabled = not self.pipeline.smoother.enabled
            if self.pipeline.smoother.enabled:
                self.pipeline.smoother.reset()
        elif k == "p":
            self.paused = not self.paused
            if self.paused:
                self.pipeline.smoother.reset()
        elif k == "h":
            self.show_help = not self.show_help
        elif k == "s":
            self.save_screenshot()
        elif k == "b":
            print(json.dumps(self.monitor.snapshot(), indent=2))

    def save_screenshot(self, image: np.ndarray | None = None) -> Path | None:
        img = image if image is not None else self.last_image
        if img is None:
            return None
        out_dir = self.cfg.resolve(self.cfg.display.screenshot_dir) or Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("screenshot_%Y%m%d_%H%M%S") + f"_{self.screenshots}.png"
        self.screenshots += 1
        path = out_dir / name
        cv2.imwrite(str(path), img)
        return path

    # ------------------------------------------------------------------ loops

    def run_gui(self) -> int:
        self.pipeline.start()
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self._window_sized = False
        cv2.setMouseCallback(self.window, self.on_mouse)
        try:
            while self.running:
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self.handle_key(key)
                if self.paused:
                    if self.last_image is not None:
                        cv2.imshow(self.window, self.last_image)
                    continue
                frame, depth_result, depth_map, stale, stale_text = self._step()
                if frame is None:
                    waiting = np.zeros((480, 640, 3), dtype=np.uint8)
                    draw_text_box(
                        waiting,
                        "waiting for camera frames... (check camera index / connection)",
                        (40, 240),
                        scale=0.7,
                        color=(0, 0, 255),
                        thickness=2,
                    )
                    cv2.imshow(self.window, waiting)
                    self.last_image = waiting
                    continue
                t0 = time.time()
                measurements = self._measurements(frame, depth_map)
                image = self._compose(frame, depth_result, measurements, stale, stale_text, depth_map)
                self.monitor.on_render((time.time() - t0) * 1000.0)
                self.last_image = image
                self.monitor.on_display(depth_result.t_capture if depth_result else None)
                if not self._window_sized:
                    cv2.resizeWindow(self.window, image.shape[1], image.shape[0])
                    self._window_sized = True
                cv2.imshow(self.window, image)
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
        summary = self.finish()
        print(json.dumps(summary, indent=2))
        return 0

    def _step(self):
        frame_obj = self.pipeline.latest_frame()
        depth_result = self.pipeline.latest_depth()
        if frame_obj is None:
            return None, None, None, True, "no camera frame"
        frame = self._prepare_frame(frame_obj.image)
        depth_map = None
        stale = True
        stale_text = ""
        if depth_result is not None:
            self.engine_last_inference_ms = depth_result.inference_ms
            self.monitor.on_depth(depth_result)
            depth_map = self._prepare_depth(depth_result.depth)
            age_ms = (time.time() - depth_result.t_capture) * 1000.0
            self.monitor.add("depth_age_ms", age_ms)
            mismatch = depth_result.frame_id != frame_obj.id
            if mismatch:
                stale = True
                stale_text = f"DEPTH FROM FRAME {depth_result.frame_id} (now {frame_obj.id}) - {age_ms:.0f} ms old"
            elif age_ms > self.cfg.display.depth_age_warning_ms:
                stale = True
                stale_text = f"depth latency {age_ms:.0f} ms"
            else:
                stale = False
        else:
            stale_text = "waiting for first depth prediction"
        return frame, depth_result, depth_map, stale, stale_text

    def run_headless(self, n_frames: int, timeout_s: float = 120.0) -> int:
        self.pipeline.start()
        seq = 0
        got = 0
        deadline = time.time() + timeout_s
        last_depth: DepthResult | None = None
        last_frame = None
        try:
            while got < n_frames and time.time() < deadline:
                seq, result = self.pipeline.depth_slot.wait_new(seq, timeout=0.5)
                if result is None:
                    if self.pipeline.inference.error:
                        print(self.pipeline.inference.error, file=sys.stderr)
                        return 2
                    continue
                got += 1
                last_depth = result
                self.engine_last_inference_ms = result.inference_ms
                self.monitor.on_depth(result)
                last_frame = self.pipeline.latest_frame()
                if got % max(1, n_frames // 5) == 0:
                    print(f"[headless] {got}/{n_frames} depth frames", flush=True)
        finally:
            pass
        time.sleep(0.2)
        self.pipeline.stop()
        if last_depth is None or last_frame is None:
            print("[headless] no frames/depth captured - is the camera connected?", file=sys.stderr)
            return 1
        frame = self._prepare_frame(last_frame.image)
        depth_map = self._prepare_depth(last_depth.depth)
        measurements = self._measurements(frame, depth_map)
        image = self._compose(frame, last_depth, measurements, False, "", depth_map)
        out_dir = self.cfg.resolve(self.cfg.display.screenshot_dir) or Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / "headless_last.png"
        cv2.imwrite(str(image_path), image)
        metrics = {
            "frames": got,
            "pipeline": self.pipeline.snapshot(),
            "performance": self.monitor.snapshot(),
            "engine": {
                "model": self.engine.meta.model_id,
                "estimation_type": self.engine.meta.estimation_type,
                "convention": self.engine.meta.convention,
                "device": str(self.engine.device),
                "precision": self.engine.meta.precision,
                "input_size": self.engine.input_size,
            },
            "calibration": self.profile.summary() if self.profile else None,
            "image": str(image_path),
        }
        metrics_path = out_dir / "headless_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))
        return 0

    def finish(self) -> dict:
        snapshot = self.monitor.snapshot()
        snapshot["pipeline"] = self.pipeline.snapshot()
        return snapshot


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_cameras:
        cams = list_cameras()
        print("usable cameras:", cams if cams else "none found")
        return 0
    try:
        cfg = load_config(args.config)
        cfg = apply_overrides(cfg, args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.benchmark:
            from src.benchmark import run_benchmarks

            run_benchmarks(cfg)
            return 0
        app = DepthApp(cfg)
        if cfg.runtime.headless_frames > 0:
            return app.run_headless(cfg.runtime.headless_frames)
        print(f"model: {app.engine.meta.model_id} | depth type: {app.engine.meta.estimation_type} | device: {app.engine.device}")
        print("controls: q quit, h help, left click add point, right click remove point")
        return app.run_gui()
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except CalibrationError as exc:
        print(f"Calibration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
