#!/usr/bin/env python3
"""ESP32-CAM MJPEG reader built for OpenCV / neural-network pipelines.

The board (src/main.cpp) joins your WiFi as a station and serves MJPEG on
http://esp32cam.local:81/stream (mDNS) or http://<ip>:81/stream.
/capture and /status live on port 80 — the stream owns port 81 so it can
never block them.

Streaming defaults to HD (1280x720). The firmware auto-tunes JPEG quality toward
20 fps on the wire (actual speed depends on light and WiFi), resolution
stays at HD so NN/VLM inputs keep their detail. Override any time with
--size svga|vga or ?size= / ?quality= on the URL.

Feed frames straight into a model:

    from stream import ESP32Cam

    with ESP32Cam(rgb=True) as cam:          # RGB for torch/onnx models
        for frame in cam.frames():            # numpy uint8, 720x1280x3
            result = model(frame)

Or grab the newest frame on demand:

    cam = ESP32Cam()
    cam.start()
    frame = cam.wait_for_frame()            # blocks until one arrives
    cam.stop()

CLI:
    python3 stream.py stream                # OpenCV preview window
    python3 stream.py stream --no-display   # headless, just stats
    python3 stream.py stream --size svga    # 800x600 if WiFi is congested
    python3 stream.py capture               # save one JPEG still
    python3 stream.py flash                 # build + upload firmware
"""

import argparse
import http.client
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import cv2
import numpy as np

DEFAULT_HOST = "esp32cam.local"
DEFAULT_SIZE = "hd"  # 1280x720 — firmware auto-tunes quality to keep fps smooth
DEFAULT_URL = f"http://{DEFAULT_HOST}:81/stream?size={DEFAULT_SIZE}"
CAPTURE_URL = f"http://{DEFAULT_HOST}:80/capture"

ENV = "esp32cam"
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 10.0
MAX_FRAME = 1048576  # HD frames at high quality can pass 128KB
RECV_SIZE = 65536

__all__ = ["ESP32Cam", "DEFAULT_URL", "CAPTURE_URL", "DEFAULT_SIZE"]


class ESP32Cam:
    """Reads the board's MJPEG stream and yields decoded OpenCV frames.

    Frames are plain ``numpy.ndarray`` (dtype uint8, shape HxWx3, BGR order),
    i.e. exactly what ``cv2`` and most model runtimes expect.
    """

    def __init__(self, url=DEFAULT_URL, rgb=False):
        parsed = urlparse(url if "://" in url else "http://" + url)
        if parsed.scheme != "http":
            raise ValueError(
                f"only http:// streams are supported, got {parsed.scheme}://"
            )
        if not parsed.hostname:
            raise ValueError(f"could not parse host from {url!r}")

        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.path = parsed.path or "/stream"
        if parsed.query:
            self.path += "?" + parsed.query
        self.rgb = rgb  # torch/onnx want RGB; OpenCV default is BGR

        self._sock = None
        self._response = None
        self._stop_event = threading.Event()
        self._buf = b""
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._running = False
        self._thread = None
        self.fps = 0.0
        self._frame_count = 0

    # -------------------------------------------------
    # lifecycle
    # -------------------------------------------------
    @property
    def url(self):
        return f"http://{self.host}:{self.port}{self.path}"

    @property
    def running(self):
        return self._running

    def start(self):
        if self._running:
            return self
        if self._thread and self._thread.is_alive():
            raise RuntimeError("previous reader is still stopping")
        self._stop_event.clear()
        with self._cond:
            self._frame = None
        self.fps = 0.0
        self._frame_count = 0
        self._connect()
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        self._close_socket()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._thread and not self._thread.is_alive():
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    # -------------------------------------------------
    # frame access
    # -------------------------------------------------
    def latest(self):
        """Newest frame, or None if nothing has arrived yet."""
        with self._cond:
            return self._frame

    def wait_for_frame(self, timeout=10.0, after=None):
        """Block until a frame arrives.

        ``after=None`` waits for the first frame; pass a previously returned
        frame to wait for the *next* one. Returns None on timeout.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None
        with self._cond:
            while self._running:
                frame = self._frame
                if frame is not None and frame is not after:
                    return frame
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining if remaining is not None else 1.0)
            return None

    def frames(self, timeout=READ_TIMEOUT):
        """Yield every new frame as it is decoded (numpy BGR, uint8)."""
        self.start()
        last = None
        while self._running:
            frame = self.wait_for_frame(timeout=timeout, after=last)
            if frame is None:
                continue
            last = frame
            yield frame

    # -------------------------------------------------
    # internals
    # -------------------------------------------------
    def _close_socket(self):
        sock = self._sock
        self._sock = None
        response = self._response
        self._response = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if response is not None:
            response.close()

    def _connect(self):
        self._close_socket()
        self._buf = bytearray()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock = sock
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(CONNECT_TIMEOUT)
            sock.connect((self.host, self.port))
            request = (
                f"GET {self.path} HTTP/1.1\r\n"
                f"Host: {self.host}:{self.port}\r\n"
                f"Accept: multipart/x-mixed-replace\r\n"
                f"Connection: keep-alive\r\n\r\n"
            )
            sock.sendall(request.encode())
            # HTTPResponse removes transfer chunk lengths even when a JPEG
            # spans several HTTP chunks. read1() returns available bytes
            # without waiting to fill a 64 KiB read with multiple frames.
            response = http.client.HTTPResponse(sock)
            self._response = response
            response.begin()
            if response.status != 200:
                raise ConnectionError(f"unexpected response: {response.status}")
            if "multipart/x-mixed-replace" not in response.getheader("Content-Type", "").lower():
                raise ConnectionError("response is not an MJPEG stream")
            if self._stop_event.is_set():
                raise ConnectionError("reader stopped during connection")
            sock.settimeout(READ_TIMEOUT)
        except Exception:
            self._close_socket()
            raise

    def _read_jpeg(self):
        """Return the *newest* complete JPEG available, discarding any older
        ones still sitting in the buffer. If the consumer ever lags behind the
        network, stale frames are dropped instead of queued — that is what
        keeps latency flat instead of growing over time."""
        buf = self._buf
        while self._running:
            jpeg = self._newest_jpeg(buf)
            if jpeg is not None:
                return jpeg

            if len(buf) > MAX_FRAME:
                # no EOI in a buffer this large: corrupted frame, resync
                start = buf.rfind(b"\xff\xd8")
                if start > 0:
                    del buf[:start]
                else:
                    del buf[:]

            response = self._response
            if response is None:
                raise ConnectionError("stream is disconnected")
            chunk = response.read1(RECV_SIZE)
            if not chunk:
                raise ConnectionError("connection closed")
            buf += chunk
        return None

    @staticmethod
    def _newest_jpeg(buf):
        last = None
        pos = 0
        while True:
            start = buf.find(b"\xff\xd8", pos)
            if start == -1:
                break
            end = buf.find(b"\xff\xd9", start + 2)
            if end == -1:
                break
            last = (start, end + 2)
            pos = end + 2
        if last is None:
            return None
        start, stop = last
        jpeg = bytes(buf[start:stop])
        del buf[:stop]  # drops every older frame along with it
        return jpeg

    def _publish(self, frame):
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    def _reader_loop(self):
        window_start = time.monotonic()
        while self._running:
            try:
                if self._response is None:
                    self._connect()
                    self._frame_count = 0
                    window_start = time.monotonic()
                jpeg = self._read_jpeg()
                if jpeg is None:
                    continue

                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if self.rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                self._publish(frame)
                self._frame_count += 1

                elapsed = time.monotonic() - window_start
                if elapsed >= 1.0:
                    self.fps = self._frame_count / elapsed
                    self._frame_count = 0
                    window_start = time.monotonic()
            except (ConnectionError, OSError, http.client.HTTPException, ValueError):
                self._close_socket()
                self.fps = 0.0
                if not self._running:
                    break
                self._stop_event.wait(1.0)

    def __repr__(self):
        return f"<ESP32Cam {self.url} running={self._running} fps={self.fps:.1f}>"


# backwards-compatible alias
MJPEGReader = ESP32Cam


# =====================================================
# CLI commands
# =====================================================
def flash_firmware(port=None):
    """Build and upload the esp32cam firmware (all flash offsets handled by PIO)."""
    cmd = ["pio", "run", "-e", ENV, "-t", "upload"]
    if port:
        cmd += ["--upload-port", port]

    print(f"[+] Building and flashing env '{ENV}'" + (f" on {port}" if port else ""))
    if subprocess.run(cmd, cwd=str(Path(__file__).parent)).returncode != 0:
        sys.exit("[!] Flash failed")

    print("[+] Flash complete. Board is rebooting...")
    time.sleep(3)


def stream_video(url=DEFAULT_URL, save=False, display=True, max_frames=None, rgb=False):
    """Open the stream; optionally display it and/or write an MP4.

    Returns the number of frames read, so callers can use it as a source:
        n = stream_video(display=False, max_frames=30)
    """
    cam = ESP32Cam(url, rgb=rgb)
    print(f"[+] Connecting to {cam.url} ...")
    try:
        cam.start()
    except (ConnectionError, OSError) as e:
        sys.exit(
            f"[!] Could not reach {cam.host}: {e}\n"
            f"    - check the board is powered and joined to WiFi (serial @ 115200)\n"
            f"    - if mDNS is unavailable, pass --ip <board-ip>"
        )

    print(f"[+] Streaming from {cam.host} ('q' quits, 's' saves a frame)")
    window = "ESP32-CAM Stream"
    if display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    writer = None
    count = 0
    try:
        for frame in cam.frames():
            count += 1

            if save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        "esp32cam_recording.mp4",
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        30,
                        (w, h),
                    )
                writer.write(frame)

            if not display:
                if count % 30 == 0:
                    print(
                        f"    frame {count}  shape={frame.shape}  "
                        f"dtype={frame.dtype}  fps={cam.fps:.1f}",
                        flush=True,
                    )
            else:
                frame = frame.copy()  # keep overlays out of frames shared with models
                cv2.putText(
                    frame,
                    f"{cam.host}  FPS: {cam.fps:.0f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    name = f"capture_{int(time.time())}.jpg"
                    cv2.imwrite(name, frame)
                    print(f"[+] Saved {name}")

            if max_frames and count >= max_frames:
                break
    finally:
        cam.stop()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    print(f"[+] Read {count} frames")
    return count


def capture_frame(url=None, save=True):
    """Grab one JPEG from /capture and return it as an OpenCV image."""
    target = url or CAPTURE_URL
    try:
        with urllib.request.urlopen(target, timeout=5) as resp:
            data = resp.read()
    except OSError as e:
        print(f"[!] Capture failed: {e}")
        return None

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print("[!] Capture returned data that is not a decodable JPEG")
        return None

    if save:
        name = f"capture_{int(time.time())}.jpg"
        cv2.imwrite(name, frame)
        print(f"[+] Captured {name} {frame.shape}")
    return frame


def main():
    parser = argparse.ArgumentParser(description="ESP32-CAM streamer (OpenCV)")
    parser.add_argument(
        "command",
        choices=["stream", "flash", "capture"],
        help="stream: MJPEG -> OpenCV frames | flash: upload firmware | capture: one JPEG",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"stream URL (default {DEFAULT_URL})")
    parser.add_argument("--ip", help="board IP/hostname, overrides --url host")
    parser.add_argument("--port", help="serial port for flashing (e.g. /dev/ttyUSB0)")
    parser.add_argument("--save", action="store_true", help="write esp32cam_recording.mp4")
    parser.add_argument("--no-display", action="store_true", help="headless: no OpenCV window")
    parser.add_argument("--frames", type=int, help="stop after N frames")
    parser.add_argument(
        "--size",
        choices=["qqvga", "qvga", "cif", "hvga", "vga", "svga", "hd"],
        default=None,
        help=f"camera resolution (default {DEFAULT_SIZE})",
    )
    parser.add_argument(
        "--quality", type=int, help="JPEG quality 4..63 (pins quality, disables auto-tune)"
    )
    parser.add_argument(
        "--rgb", action="store_true", help="emit RGB frames (torch/onnx) instead of BGR"
    )
    args = parser.parse_args()

    url = args.url
    if args.quality is not None and not 4 <= args.quality <= 63:
        parser.error("--quality must be between 4 and 63")
    if args.ip:
        parsed = urlparse(url)
        url = urlunparse(parsed._replace(netloc=f"{args.ip}:{parsed.port or 80}"))

    if args.size or args.quality:
        p = urlparse(url)
        qs = dict(parse_qsl(p.query))
        if args.size:
            qs["size"] = args.size
        if args.quality:
            qs["quality"] = str(args.quality)
        url = urlunparse(p._replace(query=urlencode(qs)))

    if args.command == "flash":
        flash_firmware(args.port)
    elif args.command == "stream":
        stream_video(
            url=url,
            save=args.save,
            display=not args.no_display,
            max_frames=args.frames,
            rgb=args.rgb,
        )
    elif args.command == "capture":
        u = urlparse(url)
        capture_frame(url=f"http://{u.hostname}:80/capture" + (f"?{u.query}" if u.query else ""))


if __name__ == "__main__":
    main()
