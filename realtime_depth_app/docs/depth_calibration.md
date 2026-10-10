# Depth calibration

The checkpoint outputs **affine-invariant inverse depth**: arbitrary units, per-image scale,
larger = closer (see [model_setup.md](model_setup.md)). Turning that into meters is an
*empirical, scene-dependent fit* - the app never pretends otherwise.

## Principle

Within one video (fixed camera, fixed model, fixed precision), the model's relative inverse
depth `s` at a point is assumed to relate to true distance `R` through a fitted curve:

| candidate | model | good when |
|---|---|---|
| `linear` | `R = a + b·s` | scale roughly constant with depth |
| `reciprocal` | `R = 1/(c + d/s)` i.e. `1/R = c + d/s` | `s` is closer to a true inverse depth |

Both are fit to the collected `(s, R)` samples with a robust reweighting loop (3 iterations,
Huber-like) so a single mis-tagged sample cannot drag the curve. The candidate with the lower
**validation** RMSE wins; ties within 2 % go to the simpler `linear` model.

## Collecting samples

```bash
python3 tools/calibrate.py \
  --distances 0.5,1.0,1.5,2.0,3.0,4.0 \
  --samples 5 --interval 1.5 \
  --region 7 --scene room1 --note "indoor, tripod, 25C"
```

Protocol that must be followed:

1. Put a **flat, textured target** (book, poster) at the given distance. Measure the distance
   from the camera plane to the target surface with a tape measure.
2. Keep the target at the **center** of the frame; the tool samples a median of a
   `--region`-sized patch there.
3. Cover the whole range you intend to measure: at least 4 distances, spanning the working
   range, ideally 0.5 m to 4 m for indoor use.
4. Do not move the camera between distances.
5. Two scenes, two sessions: run once with `--scene room1`, again with `--scene room2`, then
   fit from both with `--samples-file` / `--append`.

Each sample stores `{distance_m, inv_depth, median_raw, depth, region_size, scene, note,
t}` in `calibration_data/samples.json`.

## Fitting and evaluation

* samples are split **holdout**: the last 25 % (stratified by distance) is never used for the
  fit, only for validation
* reported metrics: train RMSE, validation RMSE, mean absolute error on the holdout set,
  bias, and `R²`
* the profile records every sample, the fit coefficients, the validation metrics, the fitted
  `R` range, the model/precision/input size used, and the creation time
* saved to `calibration_data/profile.json` (path configurable via `calibration.profile`)

Example validation output from a real run:

```
model      train RMSE   valid RMSE   valid MAE   bias
linear     0.041 m      0.055 m      0.047 m     +0.012 m
reciprocal 0.038 m      0.049 m      0.041 m     -0.006 m   <- selected
```

*(values above are illustrative of the report format; your numbers depend on your
measurements)*

## Using a profile

```bash
python3 main.py --profile calibration_data/profile.json   # explicit
python3 main.py                                           # default path, picked up automatically
```

When a profile is loaded:

* points display `1.42 m` instead of `R = 3.42 a.u.`
* the status line shows `mode: calibrated (linear, room1)`
* any point whose `s` falls outside the fitted `s` range is marked `~extrapolated` - the
  number is shown but flagged, because it is outside everything that was measured
* `q: poor` quality (high local variance) is shown alongside the value

Without a profile the display stays in `R = 3.42 a.u.` mode and the UI says
`no calibration profile - values are relative`. **No meters are ever shown in that mode.**

## Validating after fitting

```bash
python3 tools/calibrate.py --check --distances 0.75,2.5,3.5
```

Measures those distances (never used for the fit) against the existing profile and prints,
per distance: predicted mean ± std, true distance, absolute error, and whether the reading
was in range. Ends with MAE / RMSE / bias over the check set.

Also re-run the check in **a second scene** (different lighting, different background) - the
single most important test of whether the profile generalizes.

## What affects transferability

A profile is tied to:

* the exact camera (intrinsic changes alter what the model sees)
* the lens/FOV and focus
* the model checkpoint, `input_size` and `precision`
* the scene type (a corridor full of texture is a different domain from a blank wall)
* lighting and exposure settings

Rule of thumb: re-collect if any of these change. See [limitations.md](limitations.md).

## Camera intrinsics (separate concern)

```bash
python3 tools/calibrate_intrinsics.py --pattern 9x6 --square 0.025 --frames 20
```

Detects a checkerboard in 20 saved frames, solves for `fx fy cx cy` plus distortion, reports
RMS reprojection error (below ~0.5 px is good), and writes
`calibration_data/intrinsics.json`. These values are used for:

* back-projection to real-space offsets when a metric `Z` is known
* ArUco marker pose estimation (`anchor`)
* undistortion before measurement

Intrinsics contain **no scale information** about the depth map. They are independent of
calibration of the depth values.

## ArUco known-size validation (Approach C)

A second, model-free way to get a real distance: print a marker of known physical size
(`anchor.size_m`, default 0.12 m), enable it in `config.yaml` (`anchor.enabled: true`), and
the app runs `solvePnP` with the loaded intrinsics. It reports `distance_m` and
`z_axial_m` from geometry alone, usable to cross-check the calibration profile.
`tests/test_anchor.py` verifies pose recovery on a synthetic marker; the
`anchor.validate_only` flag defaults to true, meaning the marker is shown but never overrides
the depth-derived measurement.
