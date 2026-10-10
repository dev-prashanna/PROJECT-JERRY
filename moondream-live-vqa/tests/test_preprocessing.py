"""Tests for frame preprocessing (BGR -> RGB -> PIL, validation, resizing)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.vlm.preprocessing import (
    InvalidFrameError,
    bgr_to_rgb,
    describe_frame,
    frame_to_pil,
    resize_max_side,
    validate_frame,
)


class TestValidateFrame:
    def test_none_rejected(self):
        with pytest.raises(InvalidFrameError):
            validate_frame(None)

    def test_wrong_type_rejected(self):
        with pytest.raises(InvalidFrameError):
            validate_frame("not a frame")

    def test_grayscale_rejected(self):
        with pytest.raises(InvalidFrameError):
            validate_frame(np.zeros((10, 10), dtype=np.uint8))

    def test_wrong_channel_count_rejected(self):
        with pytest.raises(InvalidFrameError):
            validate_frame(np.zeros((10, 10, 4), dtype=np.uint8))

    def test_degenerate_shape_rejected(self):
        with pytest.raises(InvalidFrameError):
            validate_frame(np.zeros((1, 64, 3), dtype=np.uint8))

    def test_empty_array_rejected(self):
        with pytest.raises(InvalidFrameError):
            validate_frame(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_valid_frame_passes_through(self, bgr_frame):
        out = validate_frame(bgr_frame)
        assert out is bgr_frame

    def test_float01_converted_to_uint8(self):
        source = np.full((8, 8, 3), 0.5, dtype=np.float32)
        out = validate_frame(source)
        assert out.dtype == np.uint8
        assert out.max() == 127 or out.max() == 128

    def test_non_contiguous_made_contiguous(self):
        source = np.zeros((8, 8, 6), dtype=np.uint8)[:, :, :3]
        assert not source.flags["C_CONTIGUOUS"]
        out = validate_frame(source)
        assert out.flags["C_CONTIGUOUS"]


class TestBgrToRgb:
    def test_channel_order_swapped(self, bgr_frame):
        rgb = bgr_to_rgb(bgr_frame)
        # BGR (10, 200, 30) -> RGB (30, 200, 10)
        assert tuple(rgb[20, 30]) == (30, 200, 10)

    def test_blue_patch_becomes_red_in_rgb(self, bgr_frame):
        rgb = bgr_to_rgb(bgr_frame)
        # BGR (255, 0, 0) is blue; in RGB it must read as (0, 0, 255).
        assert tuple(rgb[5, 5]) == (0, 0, 255)

    def test_source_untouched(self, bgr_frame):
        before = bgr_frame.copy()
        bgr_to_rgb(bgr_frame)
        assert np.array_equal(before, bgr_frame)


class TestResize:
    def test_no_resize_when_disabled(self):
        image = Image.new("RGB", (100, 60), (1, 2, 3))
        assert resize_max_side(image, None).size == (100, 60)
        assert resize_max_side(image, 0).size == (100, 60)

    def test_downscale_caps_longest_side(self):
        image = Image.new("RGB", (1600, 900))
        out = resize_max_side(image, 800)
        assert max(out.size) == 800
        assert out.size == (800, 450)

    def test_no_upscale(self):
        image = Image.new("RGB", (320, 240))
        assert resize_max_side(image, 1024).size == (320, 240)

    def test_already_small_unchanged(self):
        image = Image.new("RGB", (640, 480))
        assert resize_max_side(image, 640).size == (640, 480)


class TestFrameToPil:
    def test_returns_rgb_pil_image(self, bgr_frame):
        image = frame_to_pil(bgr_frame)
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (64, 48)

    def test_pixels_match_rgb_order(self, bgr_frame):
        image = frame_to_pil(bgr_frame)
        assert image.getpixel((5, 5)) == (0, 0, 255)
        assert image.getpixel((30, 20)) == (30, 200, 10)

    def test_copy_is_independent_of_source(self, bgr_frame):
        image = frame_to_pil(bgr_frame, copy=True)
        bgr_frame[:] = 0
        assert image.getpixel((5, 5)) == (0, 0, 255)

    def test_resize_option_applied(self, bgr_frame):
        image = frame_to_pil(bgr_frame, max_side=32)
        assert max(image.size) == 32

    def test_invalid_frame_raises(self):
        with pytest.raises(InvalidFrameError):
            frame_to_pil(None)


def test_describe_frame_handles_bad_input():
    assert describe_frame(None) == "None"
    assert "shape" in describe_frame(np.zeros((4, 4, 3), dtype=np.uint8))
