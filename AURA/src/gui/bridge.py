"""Thread-safe bridge between agent worker threads and the Qt main loop.

Agent callbacks fire on pipeline threads; Qt widgets are not thread-safe.
:class:`AgentBridge` is a QObject living in the GUI thread whose signals are
emitted from the worker threads — Qt's auto-connection queues them onto the
main thread, so slots always run on the right thread.

Also hosts small numpy/QImage helpers shared by every panel.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QImage, QPixmap


class AgentBridge(QObject):
    state = Signal(object)        # StateTransition
    transcript = Signal(str)      # final text
    route = Signal(object)        # RouteDecision
    response = Signal(object)     # RequestRecord
    error = Signal(str)           # friendly message
    log = Signal(str)             # free-form status line


def bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Convert an OpenCV-style BGR uint8 array to a QImage (no copy issues)."""
    h, w = frame_bgr.shape[:2]
    if frame_bgr.ndim == 2:  # grayscale fallback
        return QImage(frame_bgr.data, w, h, w, QImage.Format_Grayscale8).copy()
    rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def frame_to_pixmap(frame_bgr: Optional[np.ndarray], max_w: int = 640, max_h: int = 480) -> QPixmap:
    if frame_bgr is None:
        return QPixmap()
    img = bgr_to_qimage(frame_bgr)
    return QPixmap.fromImage(img).scaled(
        max(1, int(max_w)),
        max(1, int(max_h)),
        aspectMode=Qt.AspectRatioMode.KeepAspectRatio,
        mode=Qt.TransformationMode.SmoothTransformation,
    )


def wire_agent(agent: Any, bridge: AgentBridge) -> None:
    """Subscribe bridge emissions to agent events (idempotent enough for one GUI)."""
    bridge_state = bridge.state.emit
    bridge_transcript = bridge.transcript.emit
    bridge_route = bridge.route.emit
    bridge_response = bridge.response.emit
    bridge_error = bridge.error.emit
    agent.on("state", bridge_state)
    agent.on("transcript", lambda text, _res=None: bridge_transcript(text))
    agent.on("route", bridge_route)
    agent.on("response", bridge_response)
    agent.on("error", bridge_error)
