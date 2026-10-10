# Model setup

## What is installed

| item | value |
|---|---|
| repo id | `depth-anything/Depth-Anything-V2-Small-hf` |
| cache location | `~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small-hf` |
| size | 95 MB total, `model.safetensors` = 99,173,660 bytes |
| parameters | 24.8 M |
| architecture | `DepthAnythingForDepthEstimation` (DPT head on a DINOv2 ViT-S/14 backbone) |
| config | `depth_estimation_type: relative`, `max_depth: 1` (ignored for relative heads) |
| preprocessor | `DPTImageProcessor`, 518 px, keep-aspect, `ensure_multiple_of: 14`, ImageNet normalization |

A second checkpoint, `LiheYoung/depth-anything-small-hf` (Depth Anything V1 small), is also
cached. It is not used by default; switch with `model.repo_id` if you want to compare.

Nothing is downloaded at runtime: `model.local_files_only: true` makes both loads offline.
To use a different local copy, set `model.local_dir: /path/to/checkpoint`.

## Relative vs metric - how it was determined

Three independent lines of evidence:

1. **Config**: `depth_estimation_type = "relative"`. In `transformers`, a `metric` head uses a
   sigmoid scaled by `max_depth`; a `relative` head uses a ReLU and returns unbounded values.
2. **Paper**: Depth Anything V2, section 7.2 - *"Since our model predicts affine-invariant
   inverse depth, for fairness we compare with Depth Anything V1 and MiDaS..."*.
3. **Measurement**: on `assets/sample_frame.jpeg` the near foreground (robot legs, bottom of
   frame) scores a median of ~3.90 while the far background (top of frame) scores ~0.39 -
   i.e. **larger value = closer**.

A test (`tests/test_depth_engine.py::test_inverse_depth_convention_on_sample`) locks this in so
a checkpoint swap that changes the convention fails the suite.

## Output conventions

| checkpoint type | output | unit | "near" |
|---|---|---|---|
| relative (this one) | affine-invariant inverse depth | arbitrary units | larger value |
| metric (Hypersim / VKITTI 2 variants) | axial depth | meters | smaller value |

`DepthEngine.meta` carries `estimation_type`, `convention` (`inverse`/`forward`) and `unit`,
and the UI renders legend text accordingly (`WARM/RED = NEAR` for inverse).

## Input expectations

* images are converted BGR→RGB, resized with aspect ratio preserved so both sides are
  multiples of 14, padded per the processor config, normalized with ImageNet mean/std
* `model.input_size` (default 518, must be a multiple of 14) sets the long side
* the network returns a coarse map (`patch_h*14 × patch_w*14`), which is bicubically
  upsampled to the exact source frame size so pixel coordinates match the RGB panel 1:1

## Precision

`model.precision` supports `fp32`, `autocast_fp16`, `autocast_bf16`, `fp16`, `bf16`.

Measured difference fp32 vs autocast_bf16 on the sample image:

* Pearson r = 0.99993
* mean relative difference = 1.4 %, 99th percentile = 7.4 %
* both outputs finite, same dynamic range

Recommendation: use `fp32` while collecting a calibration profile, then keep the same
precision for measurement - a profile fitted under one precision is about 1 % off under the
other, which is inside the calibration error anyway but is avoidable inconsistency.

## CPU fallback

`model.device: auto` selects CUDA when available. On CPU the engine runs fp32; half precision
configurations raise a clear `ConfigError` instead of silently degrading.

## Verifying the setup

```bash
python3 main.py --headless 10          # prints model, device, timings, writes outputs/
python3 -m pytest tests/test_depth_engine.py -q
```
