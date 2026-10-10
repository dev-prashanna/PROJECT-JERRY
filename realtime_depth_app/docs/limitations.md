# Limitations

An honest list of what this system does and does not do.

## 1. Raw output is NOT metric

`Depth-Anything-V2-Small-hf` is a **relative** checkpoint. Its output is affine-invariant
inverse depth: arbitrary scale, arbitrary shift, potentially different on every single frame,
larger = closer.

* Un-calibrated display: `R = 3.42 a.u.` - no unit, no meaning beyond "relatively near".
* Metric display requires a calibration profile fitted to **your** camera and **your** scene.
* The app never labels uncalibrated values as meters, and marks extrapolated predictions
  `~extrapolated`.

## 2. Calibration does not transfer

The fitted curve depends on camera model, lens, focus, FOV, exposure, scene statistics,
lighting, model checkpoint, input size and precision. Change any of them and the profile is
invalid. Practical consequences:

* a profile from scene A is approximate in scene B (expect error growth - measure it with
  `tools/calibrate.py --check`)
* the recommended working range is 0.5 - 4 m, the range a tape measure can reliably cover
  indoors; beyond the fitted range the app extrapolates and says so
* **no real tape-measure calibration has been run in this repository yet** - the fixture
  numbers shown in `docs/depth_calibration.md` are format examples, not measured results

## 3. Absolute accuracy has not been established

What *was* measured in this repo:

* synthetic / fixture accuracy of the fitting code (tests)
* pose accuracy of ArUco validation on a rendered marker (error < 0.1 % on synthetic data)
* latency, fps, VRAM (see [performance.md](performance.md))

What has **not** been measured: end-to-end distance error against real tape measurements in
multiple scenes. That experiment needs physical measurements and is described in
`docs/depth_calibration.md` (collect → fit → `--check` on held-out distances → repeat in a
second scene).

## 4. Inverse-depth edge effects

* `RuntimeWarning` on the first frame: initial division by zero in the inverse-depth
  normalisation is caught and replaced by the valid range - harmless but visible in logs.
* Depth edges are **not** aligned to RGB edges. The DPT head predicts at 1/14 resolution and
  the result is bicubically upsampled, so silhouettes bleed ~2-8 px into the background.
  Clicking exactly on an edge therefore samples a mixed neighborhood - the quality heuristic
  marks this `poor`.
* Reflective / transparent surfaces (glass, mirrors, screens, polished floors), textureless
  walls, and repetitive patterns are known weak points of monocular depth models.

## 5. Scale ambiguity inherent to monocular vision

A single camera cannot observe absolute scale without a reference. The model's "scale" comes
from learned priors about object sizes in training data, not from measurement. Two scenes with
identical images but different real sizes are indistinguishable. Only an external reference
(tape-measured calibration, or the known-size ArUco marker) breaks this.

## 6. Temporal stability

* smoothing (`alpha`, `median_window`) reduces flicker but adds lag; at `alpha = 0.35` a step
  change takes ~6-8 frames to settle
* per-frame affine variation can still produce breathing of the color map when `R` range is
  auto-normalized; the legend range is EMA-smoothed for this reason
* abrupt camera motion degrades quality further (rolling shutter, motion blur)

## 7. Latency is not zero

End-to-end latency ≈ capture → one inference slot → render. Measured ~45-80 ms with bf16 at
518, worse with fp32 (~100 ms). Fast-moving objects will be displayed with their depth from a
slightly older frame; the UI shows the depth's age (`DEPTH FROM FRAME n`) rather than hiding
it.

## 8. Calibration regions and sampling

* measurement takes a **median** of an `region_size` (default 7) square of pixels; sub-pixel
  accuracy is not attempted
* if the clicked point falls on a hole (invalid pixel), the estimator searches a widening
  radius - the reported point may be a few pixels away from the click
* `quality` (`good`/`fair`/`poor`) is a heuristic from local variance, valid fraction and
  calibration range. It is not a calibrated confidence interval.

## 9. Camera constraints observed on this machine

* `/dev/video0` delivers 15-30 fps depending on light; exposure control needs root
* only the built-in camera was tested; external USB cameras may expose different formats
* `cv2.CAP_PROP_FRAME_WIDTH/HEIGHT` requests are best-effort - the driver may ignore them

## 10. Scope

This is a perception prototype, not a safety system. It must not be used for collision
avoidance, medical, automotive, or any application where an incorrect distance has
consequences. No guarantees of any kind are made about the output.
