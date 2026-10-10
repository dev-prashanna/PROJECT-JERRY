#!/usr/bin/env python3
"""Collect reference distances and fit a relative-depth -> meters calibration profile.

Example:
    python3 tools/calibrate.py --distances 0.5,1.0,1.5,2.0,3.0,4.0 --samples 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from src.calibration import (  # noqa: E402
    CalibrationError,
    Sample,
    fit_profile,
    load_samples,
    save_profile,
    save_samples,
)
from src.camera import CameraSource  # noqa: E402
from src.config import ConfigError, load_config  # noqa: E402
from src.depth_engine import DepthEngine  # noqa: E402
from src.distance_estimator import sample_region  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit a metric calibration for the relative depth model")
    p.add_argument("--config", default=None)
    p.add_argument("--distances", default="0.5,1.0,1.5,2.0,3.0,4.0", help="comma separated meters")
    p.add_argument("--samples", type=int, default=5, help="frames per distance")
    p.add_argument("--interval", type=float, default=0.4, help="seconds between samples")
    p.add_argument("--region", type=int, default=15, help="odd region size for the median")
    p.add_argument("--scene", default="default", help="scene tag stored with each sample")
    p.add_argument("--note", default="")
    p.add_argument("--out", default=None, help="profile output path (default: config calibration.profile)")
    p.add_argument("--samples-file", default="calibration_data/samples.json")
    p.add_argument("--append", action="store_true", help="append to an existing samples file")
    p.add_argument("--fit-only", action="store_true", help="fit from saved samples without capturing")
    p.add_argument("--check", action="store_true",
                   help="measure the given distances against an existing profile and report errors")
    return p.parse_args()


def capture_samples(engine: DepthEngine, source: CameraSource, distances: list[float],
                    n_samples: int, interval: float, region: int, scene: str) -> list[Sample]:
    samples: list[Sample] = []
    print("\nProtocol: place a flat, textured target so it fills the measurement spot in the")
    print("center of the frame, measure the distance from the CAMERA LENS to the target with")
    print("a tape measure, then press ENTER. Press CTRL-C to stop early.\n")
    try:
        for distance in distances:
            answer = input(f"> place target at {distance:.2f} m and press ENTER (or 's' to skip): ").strip().lower()
            if answer == "s":
                print("   skipped")
                continue
            values: list[float] = []
            for i in range(n_samples):
                frame = None
                for _ in range(30):
                    frame = source.read()
                    if frame is not None:
                        break
                    time.sleep(0.05)
                if frame is None:
                    print("   camera read failed, aborting capture")
                    break
                result = engine.infer(frame.image, frame_id=frame.id, t_capture=frame.t_capture)
                h, w = result.depth.shape[:2]
                s = sample_region(result.depth, w // 2, h // 2, region)
                if not s["ok"]:
                    print(f"   invalid sample ({s['reason']})")
                    continue
                values.append(s["median"])
                print(f"   [{i + 1}/{n_samples}] R = {s['median']:.4f}")
                time.sleep(interval)
            if values:
                for v in values:
                    samples.append(Sample(r=float(v), d_true=float(distance), scene=scene))
                print(f"   -> {len(values)} samples at {distance:.2f} m "
                      f"(median R = {float(np.median(values)):.4f})")
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted - fitting with the samples collected so far")
    return samples


def report(profile) -> None:
    print("\n=== calibration report ===")
    print(f"model       : {profile.model}(R) with params {tuple(round(p, 5) for p in profile.params)}")
    print(f"samples     : {len(profile.samples)} over distances {sorted({s.d_true for s in profile.samples})}")
    print(f"fitted range: R in [{profile.r_min:.4f}, {profile.r_max:.4f}]")
    for name, entry in profile.candidate_models.items():
        if "error" in entry:
            print(f"  {name:<11}: failed ({entry['error']})")
            continue
        tr, va = entry["train"], entry.get("val") or {}
        print(f"  {name:<11}: train RMSE {tr['rmse']:.3f} m | "
              + (f"held-out RMSE {va['rmse']:.3f} m (n={va['n']})" if va else "no held-out split"))
    if profile.val_metrics and profile.val_metrics.get("n"):
        v = profile.val_metrics
        print(f"selected    : {profile.model} | held-out MAE {v['mae']:.3f} m, RMSE {v['rmse']:.3f} m, "
              f"bias {v['bias']:+.3f} m, max |err| {v['max_err']:.3f} m")
    print("\nWARNING: this curve is scene/camera dependent (relative depth has per-image scale+shift).")
    print("Values outside the fitted R range are flagged as extrapolated in the app.")


def run_check(cfg, args) -> int:
    """Measure reference distances with the fitted profile and report absolute errors."""
    from src.geometry import CameraIntrinsics  # noqa: F401  (import guard for future use)

    profile_path = cfg.resolve(args.out or cfg.calibration.profile)
    try:
        profile = load_profile(profile_path)
    except CalibrationError as exc:
        print(f"cannot load profile: {exc}", file=sys.stderr)
        return 2
    distances = [float(x) for x in args.distances.split(",") if x.strip()]
    engine = DepthEngine(cfg)
    source = CameraSource(cfg)
    print(f"checking profile {profile_path} ({profile.model}) at {distances} m\n")
    rows = []
    try:
        for distance in distances:
            input(f"> place target at {distance:.2f} m, ENTER to measure (s skip): ").strip()
            values = []
            for _ in range(args.samples):
                frame = None
                for _ in range(30):
                    frame = source.read()
                    if frame is not None:
                        break
                    time.sleep(0.05)
                if frame is None:
                    break
                result = engine.infer(frame.image, frame_id=frame.id, t_capture=frame.t_capture)
                h, w = result.depth.shape[:2]
                s = sample_region(result.depth, w // 2, h // 2, args.region)
                if s["ok"]:
                    pred, extrapolated = profile.predict(s["median"])
                    values.append((pred, extrapolated))
                    time.sleep(args.interval)
            if not values:
                print("   no valid readings")
                continue
            preds = [v[0] for v in values]
            err = np.array(preds) - distance
            rows.append(dict(distance=distance, n=len(preds), pred_mean=float(np.mean(preds)),
                             pred_std=float(np.std(preds)), mae=float(np.mean(np.abs(err))),
                             bias=float(np.mean(err)), extrapolated=any(v[1] for v in values)))
            print(f"   predicted {np.mean(preds):.2f} +/- {np.std(preds):.2f} m | true {distance:.2f} m "
                  f"| error {np.mean(err):+.2f} m | range flag: "
                  f"{'EXTRAPOLATED' if rows[-1]['extrapolated'] else 'in range'}")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        source.release()
    if not rows:
        return 1
    errs = np.array([r["pred_mean"] - r["distance"] for r in rows])
    print("\n=== check report ===")
    print(f"MAE  {float(np.mean(np.abs(errs))):.3f} m")
    print(f"RMSE {float(np.sqrt(np.mean(errs ** 2))):.3f} m")
    print(f"bias {float(np.mean(errs)):+.3f} m | std {float(np.std(errs)):.3f} m | n_distances {len(rows)}")
    print("Distances marked EXTRAPOLATED are outside the fitted R range and must not be trusted.")
    return 0


def main() -> int:
    args = parse_args()
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    samples_path = cfg.resolve(args.samples_file)
    samples: list[Sample] = []
    if (args.append or args.fit_only) and samples_path and samples_path.exists():
        samples = load_samples(samples_path)
        print(f"loaded {len(samples)} existing samples from {samples_path}")

    if args.check:
        return run_check(cfg, args)

    if not args.fit_only:
        distances = [float(x) for x in args.distances.split(",") if x.strip()]
        if len(distances) < 3:
            print("need at least 3 distances for a defensible fit", file=sys.stderr)
            return 2
        try:
            engine = DepthEngine(cfg)
            source = CameraSource(cfg)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"model: {engine.meta.model_id} | type: {engine.meta.estimation_type} | device: {engine.device}")
        if engine.meta.estimation_type == "metric":
            print("note: this checkpoint already predicts metric depth; calibration is unnecessary")
        captured = capture_samples(engine, source, distances, args.samples, args.interval,
                                   args.region, args.scene)
        source.release()
        samples.extend(captured)
        if samples_path:
            save_samples(samples, samples_path)
            print(f"saved {len(samples)} samples -> {samples_path}")

    if len(samples) < 4:
        print(f"not enough samples to fit ({len(samples)})", file=sys.stderr)
        return 2
    try:
        profile = fit_profile(samples, note=args.note)
    except CalibrationError as exc:
        print(f"fit failed: {exc}", file=sys.stderr)
        return 2
    report(profile)
    out = cfg.resolve(args.out or cfg.calibration.profile)
    save_profile(profile, out)
    print(f"\nsaved profile -> {out}")
    print("start the app with: python3 main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
