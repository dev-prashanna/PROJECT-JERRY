"""Live camera preview panel.

Polls the agent's frame buffer on a Qt timer (never blocks the pipeline) and
flags the known "privacy shutter / black frames" failure mode loudly.  When
depth estimation is enabled the colourized Depth-Anything map is shown
beside the RGB preview, and the latest depth map is handed to the VLM.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from src.gui.bridge import frame_to_pixmap

_BLANK_WARN_AFTER = 15


class CameraPanel(QFrame):
    def __init__(self, agent: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._agent = agent
        self._blank_streak = 0
        self._warned = False
        self._latest_pixmap = None
        self._depth_pixmap = None
        self._depth = None

        self.setObjectName("CameraPanel")
        self.setStyleSheet(
            "#CameraPanel { border: 1px solid #2c3140; border-radius: 8px; }"
        )

        title = QLabel("Camera")
        title.setStyleSheet("font-weight: 600; color: #d5d9e0;")

        self.view = QLabel("starting camera…")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(200, 150)
        self.view.setStyleSheet("background: #10131a; color: #7f8798; border-radius: 6px;")

        self.depth_view = QLabel("depth off")
        self.depth_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_view.setMinimumSize(200, 150)
        self.depth_view.setStyleSheet("background: #10131a; color: #7f8798; border-radius: 6px;")
        self.depth_view.setVisible(False)

        self.depth_title = QLabel("Depth")
        self.depth_title.setStyleSheet("font-weight: 600; color: #d5d9e0;")
        self.depth_title.setVisible(False)

        self.meta = QLabel("—")
        self.meta.setStyleSheet("color: #7f8798; font-size: 11px;")

        self.depth_meta = QLabel("")
        self.depth_meta.setStyleSheet("color: #7f8798; font-size: 11px;")

        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color: #f2b544; font-size: 11px;")
        self.warn.setVisible(False)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self.view)
        self._splitter.addWidget(self.depth_view)
        self._splitter.setChildrenCollapsible(False)

        head = QHBoxLayout()
        head.addWidget(title)
        head.addWidget(self.depth_title)
        head.addStretch(1)
        head.addWidget(self.meta)
        head.addWidget(self.depth_meta)

        layout = QVBoxLayout(self)
        layout.addLayout(head)
        layout.addWidget(self._splitter, 1)
        layout.addWidget(self.warn)

        self._depth_service = None
        self._depth_loading = False
        self._init_depth()

        self._timer = QTimer(self)
        self._timer.setInterval(66)  # ~15 fps preview
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

    # ------------------------------------------------------------------
    def _init_depth(self) -> None:
        """Start the background depth worker if enabled."""
        cam = getattr(self._agent, "camera", None)
        if cam is None:
            return
        try:
            from config.settings import get_settings

            if not get_settings().depth.enabled:
                return
            from src.perception.depth_service import get_depth_service

            self._depth_service = get_depth_service(cam)
            self._depth_service.start()
            self.depth_view.setVisible(True)
            self.depth_title.setVisible(True)
        except Exception as exc:  # noqa: BLE001 - depth is optional
            print(f"[CameraPanel] depth disabled: {exc}")
            self._depth_service = None
            self.depth_view.setVisible(False)
            self.depth_title.setVisible(False)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        try:
            self._refresh_safe()
        except Exception as exc:  # noqa: BLE001 - one bad frame must not spam
            self._timer.stop()
            self.view.setText("camera preview stopped (see terminal)")
            print(f"[CameraPanel] preview halted: {exc}")

    def _refresh_safe(self) -> None:
        cam = getattr(self._agent, "camera", None)
        if cam is None:
            self.view.setText("camera disabled")
            self.view.setPixmap(None)  # type: ignore[arg-type]
            return
        try:
            frame = cam.buffer.get(copy=True)
        except Exception:
            return
        if frame is None:
            self.view.setText("waiting for frames…")
            return

        self._latest_pixmap = frame_to_pixmap(frame.image, self.view.width() or 320, self.view.height() or 240)
        if not self._latest_pixmap.isNull():
            self.view.setPixmap(self._latest_pixmap)

        self.meta.setText(
            f"{frame.width}×{frame.height} · {frame.fps:.0f} fps · "
            f"age {frame.age_s:.2f}s · {frame.tag}"
        )

        self._refresh_depth(frame)

        # blank-frame heuristic: a solid-black image is almost always the
        # hardware privacy shutter being engaged, not a dark room.
        if frame.image.size and not frame.image.max():
            self._blank_streak += 1
            if self._blank_streak >= _BLANK_WARN_AFTER and not self._warned:
                self.warn.setText(
                    "Frames are uniformly black. The laptop camera privacy "
                    "shutter is likely engaged — toggle the physical switch."
                )
                self.warn.setVisible(True)
                self._warned = True
        else:
            self._blank_streak = 0
            self._warned = False
            self.warn.setVisible(False)

    def _refresh_depth(self, frame: Any) -> None:
        """Pull the newest colourized depth map and render it beside RGB."""
        if self._depth_service is None:
            return
        st = self._depth_service.status()
        if not st.get("loaded") and not self._depth_loading:
            self._depth_loading = True
            self.depth_view.setText("loading depth model…")
        depth = self._depth_service.latest()
        if depth is None:
            if st.get("last_error"):
                self.depth_view.setText(f"depth error:\n{st['last_error'][:60]}")
            return
        self._depth = depth
        self._depth_pixmap = frame_to_pixmap(
            depth.viz, self.depth_view.width() or 320, self.depth_view.height() or 240
        )
        if not self._depth_pixmap.isNull():
            self.depth_view.setPixmap(self._depth_pixmap)
        self.depth_meta.setText(
            f"depth {depth.inference_ms:.0f}ms · {depth.convention} · "
            f"age {depth.age_s:.2f}s"
        )

    # ------------------------------------------------------------------
    @property
    def latest_depth(self) -> Any:
        """Newest :class:`DepthFrame` (or None) — fed to the VLM."""
        return self._depth

    def shutdown(self) -> None:
        self._timer.stop()
        if self._depth_service is not None:
            try:
                self._depth_service.stop()
            except Exception:  # noqa: BLE001
                pass
