"""Left panel: live camera preview and camera controls."""

from __future__ import annotations

from typing import List, Optional

import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.camera.camera import CameraInfo, CameraSource

__all__ = ["CameraWidget"]


class CameraWidget(QWidget):
    """Live preview + camera selector/start/stop/freeze controls.

    The widget never touches the capture thread directly: a ``QTimer`` polls
    the thread-safe latest-frame buffer and repaints, which keeps all GUI work
    on the Qt main thread.
    """

    start_requested = Signal()
    stop_requested = Signal()
    restart_requested = Signal()
    refresh_requested = Signal()
    camera_selected = Signal(int)
    freeze_toggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._camera: Optional[CameraSource] = None
        self._pixmap: Optional[QPixmap] = None
        self._last_frame_id = -1
        self._fps_shown = 0.0
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 Hz repaint budget
        self._timer.timeout.connect(self._poll_frame)
        self._timer.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 8, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Camera")
        title.setObjectName("panelTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.mode_badge = QLabel("LIVE")
        self.mode_badge.setObjectName("modeBadgeLive")
        header.addWidget(self.mode_badge)
        root.addLayout(header)

        # --- preview ------------------------------------------------------
        self.preview = QLabel("Starting camera…")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(420, 315)
        self.preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.preview.setWordWrap(True)
        root.addWidget(self.preview, stretch=1)

        # --- controls -----------------------------------------------------
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(180)
        self.camera_combo.currentIndexChanged.connect(self._on_combo_changed)
        row1.addWidget(self.camera_combo, stretch=1)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Rescan available cameras")
        self.refresh_btn.clicked.connect(self.refresh_requested.emit)
        row1.addWidget(self.refresh_btn)

        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_requested.emit)
        row1.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        row1.addWidget(self.stop_btn)
        root.addLayout(row1)

        # --- freeze / mode -------------------------------------------------
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        self.freeze_btn = QPushButton("Freeze Frame")
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.setToolTip(
            "Freeze the current frame so several questions can be asked about "
            "the exact same image (the vision encoding is then reused)."
        )
        self.freeze_btn.toggled.connect(self._on_freeze_toggled)
        row2.addWidget(self.freeze_btn)
        row2.addStretch(1)
        root.addLayout(row2)

        # --- status --------------------------------------------------------
        status = QFrame()
        status.setObjectName("cameraStatusFrame")
        status_layout = QHBoxLayout(status)
        status_layout.setContentsMargins(8, 6, 8, 6)
        status_layout.setSpacing(14)

        self.status_label = QLabel("Camera: idle")
        self.status_label.setObjectName("statusDot")
        self.res_label = QLabel("--x--")
        self.fps_label = QLabel("-- fps")
        self.device_label = QLabel("device --")
        for label in (self.status_label, self.res_label, self.fps_label, self.device_label):
            status_layout.addWidget(label)
        status_layout.addStretch(1)
        root.addWidget(status)

    # ------------------------------------------------------------------
    # public API (called by the main window)
    # ------------------------------------------------------------------
    def attach_camera(self, camera: CameraSource) -> None:
        self._camera = camera

    def set_camera_list(self, cameras: List[CameraInfo]) -> None:
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        if not cameras:
            self.camera_combo.addItem("No camera found", -1)
        else:
            for info in cameras:
                self.camera_combo.addItem(info.label(), info.index)
        self.camera_combo.blockSignals(False)

    def set_selected_index(self, index: int) -> None:
        for pos in range(self.camera_combo.count()):
            if self.camera_combo.itemData(pos) == index:
                self.camera_combo.blockSignals(True)
                self.camera_combo.setCurrentIndex(pos)
                self.camera_combo.blockSignals(False)
                break

    def handle_camera_event(self, payload: dict) -> None:
        status = str(payload.get("status", ""))
        message = str(payload.get("message", ""))
        width = int(payload.get("width", 0) or 0)
        height = int(payload.get("height", 0) or 0)
        fps = float(payload.get("fps", 0.0) or 0.0)

        pretty = {
            "idle": "idle",
            "connecting": "connecting…",
            "streaming": "connected",
            "reconnecting": "reconnecting…",
            "disconnected": "disconnected",
            "error": "error",
            "stopped": "stopped",
        }.get(status, status)
        self.status_label.setText(f"Camera: {pretty}")
        self.status_label.setProperty("state", status)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        if message:
            self.status_label.setToolTip(message)
        if width and height:
            self.res_label.setText(f"{width}x{height}")
        if fps:
            self.fps_label.setText(f"{fps:.0f} fps")
        if self._camera is not None:
            backend = self._camera.backend_name or "--"
            self.device_label.setText(
                f"device {self._camera.active_index} · {backend}"
            )
            self.device_label.setToolTip(self._camera.device_name)
        self.start_btn.setEnabled(status not in {"streaming", "connecting"})
        self.stop_btn.setEnabled(status in {"streaming", "connecting", "reconnecting"})
        self._redraw_no_frame_placeholder(message if status != "streaming" else "")

    def set_mode(self, payload: dict) -> None:
        mode = str(payload.get("mode", "live"))
        frozen = payload.get("frozen")
        if mode == "frozen":
            self.mode_badge.setText("FROZEN")
            self.mode_badge.setObjectName("modeBadgeFrozen")
            self.freeze_btn.blockSignals(True)
            self.freeze_btn.setChecked(True)
            self.freeze_btn.setText("Resume Live")
            self.freeze_btn.blockSignals(False)
            if frozen:
                self.mode_badge.setToolTip(
                    f"Frozen frame #{frozen.get('frame_id')} "
                    f"({frozen.get('width')}x{frozen.get('height')})"
                )
        else:
            self.mode_badge.setText("LIVE")
            self.mode_badge.setObjectName("modeBadgeLive")
            self.mode_badge.setToolTip("Every question captures a new frame")
            self.freeze_btn.blockSignals(True)
            self.freeze_btn.setChecked(False)
            self.freeze_btn.setText("Freeze Frame")
            self.freeze_btn.blockSignals(False)
        self.mode_badge.style().unpolish(self.mode_badge)
        self.mode_badge.style().polish(self.mode_badge)

    def show_error(self, message: str) -> None:
        self.status_label.setText("Camera: error")
        self.status_label.setProperty("state", "error")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.status_label.setToolTip(message)

    def set_frame_hint(self, text: str) -> None:
        """Show a short overlay message (used while the model loads)."""
        self.preview.setText(text)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _on_combo_changed(self, index: int) -> None:
        if index < 0:
            return
        value = self.camera_combo.itemData(index)
        if value is None or int(value) < 0:
            return
        self.camera_selected.emit(int(value))

    def _on_freeze_toggled(self, checked: bool) -> None:
        self.freeze_toggled.emit(checked)

    def _poll_frame(self) -> None:
        camera = self._camera
        if camera is None:
            return
        # Live preview always shows the newest frame, even while frozen, so
        # the user can see that the camera keeps running.
        snap = camera.get_latest_frame(copy=True)
        if snap is None:
            self._redraw_no_frame_placeholder()
            return
        self._last_frame_id = snap.frame_id
        fps = snap.source_fps or camera.actual_fps
        if abs(fps - self._fps_shown) >= 0.5:
            self._fps_shown = fps
            self.fps_label.setText(f"{fps:.0f} fps")
        width, height = camera.resolution
        if width and height:
            self.res_label.setText(f"{width}x{height}")
        self._render(snap.frame)

    def _render(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self._paint()

    def _paint(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        target = self.preview.size()
        if target.width() < 4 or target.height() < 4:
            return
        scaled = self._pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(scaled)

    def _redraw_no_frame_placeholder(self, message: str = "") -> None:
        if self._pixmap is not None:
            return  # keep showing the last good frame
        text = message or "Waiting for camera frames…"
        self.preview.setText(text)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._paint()

    def shutdown(self) -> None:
        self._timer.stop()
