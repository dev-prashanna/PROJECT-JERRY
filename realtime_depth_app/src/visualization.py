"""Rendering: colorized depth, markers, legends and the information overlay."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import AppConfig
from .distance_estimator import Measurement
from .temporal_filter import RangeEMA

COLOR_MAPS: dict[str, int] = {
    "TURBO": cv2.COLORMAP_TURBO,
    "JET": cv2.COLORMAP_JET,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "CIVIDIS": cv2.COLORMAP_CIVIDIS,
    "BONE": cv2.COLORMAP_BONE,
    "PARULA": cv2.COLORMAP_PARULA,
}
CMAP_ORDER = list(COLOR_MAPS)

PANEL_GAP = 6


def colormap_index(name: str) -> int:
    return CMAP_ORDER.index(name.upper()) % len(CMAP_ORDER)


def compute_auto_range(depth: np.ndarray) -> tuple[float, float]:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [2.0, 98.0])
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def normalize_depth(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    safe = np.nan_to_num(depth, nan=lo, posinf=hi, neginf=lo)
    scaled = (safe - lo) / max(1e-9, hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def colorize(depth: np.ndarray, cmap_name: str, lo: float, hi: float, grayscale: bool = False) -> np.ndarray:
    idx = normalize_depth(depth, lo, hi)
    if grayscale:
        return cv2.cvtColor(idx, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(idx, COLOR_MAPS[cmap_name.upper()])


def draw_text_box(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.45,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    box: bool = True,
    box_alpha: float = 0.45,
) -> None:
    x, y = org
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if box:
        x0, y0 = max(0, x - 3), max(0, y - th - 4)
        x1, y1 = min(image.shape[1], x + tw + 3), min(image.shape[0], y + baseline + 3)
        roi = image[y0:y1, x0:x1]
        if roi.size:
            overlay = roi.copy()
            cv2.rectangle(overlay, (0, 0), (roi.shape[1] - 1, roi.shape[0] - 1), (0, 0, 0), -1)
            cv2.addWeighted(overlay, box_alpha, roi, 1 - box_alpha, 0, roi)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_legend(
    image: np.ndarray,
    lo: float,
    hi: float,
    cmap_name: str,
    unit: str,
    convention: str,
    origin: tuple[int, int],
    size: tuple[int, int] = (16, 140),
) -> None:
    """Vertical legend. Top of the bar = high model value."""
    x, y = origin
    bw, bh = size
    h, w = image.shape[:2]
    if x + bw + 90 > w or y + bh > h:
        return
    ramp = np.linspace(255, 0, bh).astype(np.uint8)
    bar = np.repeat(ramp.reshape(-1, 1), bw, axis=1)
    color = cv2.applyColorMap(bar, COLOR_MAPS[cmap_name.upper()])
    image[y : y + bh, x : x + bw] = color
    cv2.rectangle(image, (x, y), (x + bw - 1, y + bh - 1), (255, 255, 255), 1)

    near_word = "NEAR" if convention == "inverse" else "FAR"
    far_word = "FAR" if convention == "inverse" else "NEAR"
    draw_text_box(image, f"{hi:.2f} {unit}", (x + bw + 4, y + 12), scale=0.4, box=False)
    draw_text_box(image, near_word, (x + bw + 4, y + 28), scale=0.4, color=(0, 255, 255), box=False)
    draw_text_box(image, f"{lo:.2f} {unit}", (x + bw + 4, y + bh - 6), scale=0.4, box=False)
    draw_text_box(image, far_word, (x + bw + 4, y + bh - 22), scale=0.4, color=(255, 255, 0), box=False)


def map_window_point(x: float, y: float, panel_width: int, scale: float = 1.0) -> tuple[int, int] | None:
    """Map a click in window coordinates to source-frame pixel coordinates.

    Returns None for the divider / out-of-range clicks. Both panels show the same
    frame geometry, so either panel can be used to select a point.
    """
    if panel_width <= 0:
        return None
    ux = x / scale if scale else x
    uy = y / scale if scale else y
    w = panel_width
    if ux <= w:
        px = int(ux)
    elif ux <= w + PANEL_GAP:
        return None
    elif ux <= 2 * w + PANEL_GAP:
        px = int(ux - w - PANEL_GAP)
    else:
        return None
    return px, int(uy)


def draw_marker(image: np.ndarray, x: int, y: int, color: tuple[int, int, int], label: str = "") -> None:
    cv2.circle(image, (x, y), 7, color, 2, cv2.LINE_AA)
    cv2.circle(image, (x, y), 2, color, -1, cv2.LINE_AA)
    if label:
        draw_text_box(image, label, (x + 10, y - 8), scale=0.45, color=color)


def draw_crosshair(image: np.ndarray, color: tuple[int, int, int] = (0, 255, 255), arm: int = 14) -> None:
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(image, (cx - arm, cy), (cx + arm, cy), color, 1, cv2.LINE_AA)
    cv2.line(image, (cx, cy - arm), (cx, cy + arm), color, 1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), 4, color, 1, cv2.LINE_AA)


@dataclass
class PanelState:
    lo: float = 0.0
    hi: float = 1.0
    initialized: bool = False


class Renderer:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        v = cfg.depth.visualization
        self.colormap = v.colormap.upper()
        self.grayscale = v.grayscale
        self.range_mode = v.range_mode
        self.range_fixed = tuple(v.range_fixed)
        self._range_ema = RangeEMA(v.range_smooth if v.range_mode == "auto" else 1.0)
        self.show_perf = cfg.display.show_perf
        self.show_legend = cfg.display.show_legend
        self.window_scale = cfg.display.window_scale

    def cycle_colormap(self) -> str:
        self.colormap = CMAP_ORDER[(colormap_index(self.colormap) + 1) % len(CMAP_ORDER)]
        return self.colormap

    def toggle_grayscale(self) -> bool:
        self.grayscale = not self.grayscale
        return self.grayscale

    def reset_range(self) -> None:
        self._range_ema.reset()

    def resolve_range(self, depth: np.ndarray) -> tuple[float, float]:
        if self.range_mode == "fixed":
            return float(self.range_fixed[0]), float(self.range_fixed[1])
        lo, hi = compute_auto_range(depth)
        return self._range_ema.apply(lo, hi)

    def render(
        self,
        frame: np.ndarray,
        depth: np.ndarray | None,
        *,
        range_hint: tuple[float, float] | None = None,
        points: dict[int, Measurement] | None = None,
        center_measurement: Measurement | None = None,
        show_center: bool = False,
        status_lines: list[str] | None = None,
        perf_lines: list[str] | None = None,
        convention: str = "inverse",
        unit: str = "a.u.",
        depth_stale: bool = False,
        stale_text: str = "",
    ) -> np.ndarray:
        rgb = frame.copy()
        h, w = rgb.shape[:2]

        if depth is not None and depth.shape[:2] == (h, w):
            lo, hi = range_hint if range_hint else self.resolve_range(depth)
            depth_vis = colorize(depth, self.colormap, lo, hi, self.grayscale)
        else:
            lo, hi = 0.0, 1.0
            depth_vis = np.zeros_like(rgb)
            cv2.putText(depth_vis, "waiting for depth...", (w // 2 - 110, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        point_items = sorted((points or {}).items())
        for pid, m in point_items:
            if not m.valid:
                continue
            value = m.raw if unit.startswith("a.u.") else m.display_value
            label = f"{pid}: {value:.2f} {unit.split(' ')[0]} [{m.quality}]"
            color = (0, 255, 0) if (m.valid and not m.extrapolated) else (0, 165, 255)
            draw_marker(rgb, m.x, m.y, color, label)
            draw_marker(depth_vis, m.x, m.y, color, label)

        if show_center:
            draw_crosshair(rgb)
            draw_crosshair(depth_vis)
            if center_measurement is not None and center_measurement.valid:
                text = f"center: {center_measurement.raw:.2f}"
                if center_measurement.metric is not None:
                    text = f"center: {center_measurement.metric:.2f} m" + (" ~" if center_measurement.extrapolated else "")
                draw_text_box(rgb, text, (8, h - 12), scale=0.5, color=(0, 255, 255))

        if self.show_legend and depth is not None:
            draw_legend(depth_vis, lo, hi, self.colormap, unit, convention, origin=(10, 44))
            caption = "WARM/RED = NEAR" if convention == "inverse" else "WARM/RED = FAR"
            draw_text_box(depth_vis, caption, (10, 22), scale=0.5, color=(255, 255, 255))
            draw_text_box(depth_vis, "heuristic quality only - not confidence", (10, h - 10),
                          scale=0.4, color=(200, 200, 200))

        if status_lines:
            y = 20
            for line in status_lines:
                draw_text_box(rgb, line, (8, y), scale=0.45)
                y += 18

        if perf_lines and self.show_perf:
            y = h - 12 - 18 * len(perf_lines)
            for line in perf_lines:
                draw_text_box(rgb, line, (8, y), scale=0.45, color=(180, 255, 180))
                y += 18

        if depth_stale and stale_text:
            draw_text_box(rgb, stale_text, (8, h // 2), scale=0.55, color=(0, 0, 255), thickness=2)
            draw_text_box(depth_vis, stale_text, (8, h // 2), scale=0.55, color=(0, 0, 255), thickness=2)

        gap = np.full((h, PANEL_GAP, 3), 255, dtype=np.uint8)
        combined = np.hstack([rgb, gap, depth_vis])
        if self.window_scale != 1.0:
            combined = cv2.resize(combined, None, fx=self.window_scale, fy=self.window_scale, interpolation=cv2.INTER_AREA)
        return combined

    @staticmethod
    def compose(rgb: np.ndarray, depth_vis: np.ndarray) -> np.ndarray:
        gap = np.full((rgb.shape[0], PANEL_GAP, 3), 255, dtype=np.uint8)
        return np.hstack([rgb, gap, depth_vis])


def status_lines_for(info: dict) -> list[str]:
    lines = [
        f"device: {info.get('device', '?')}   model: {info.get('model', '?')}   mode: {info.get('mode', '?')}",
        f"depth: {info.get('depth_mode', '?')}   input: {info.get('input_res', '?')}   calibration: {info.get('calibration', 'none')}",
    ]
    return lines


def perf_lines_for(snapshot: dict, inference_ms: float, depth_updates: int) -> list[str]:
    return [
        f"cam {snapshot.get('camera_fps', 0):.1f} fps | depth {snapshot.get('depth_fps', 0):.1f} fps | infer {inference_ms:.1f} ms",
        f"e2e latency {snapshot.get('capture_to_display_ms_mean', 0):.0f} ms | render {snapshot.get('render_ms_mean', 0):.1f} ms | depth updates {depth_updates}",
    ]


def now_ms() -> float:
    return time.time() * 1000.0
