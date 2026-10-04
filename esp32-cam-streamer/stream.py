#!/usr/bin/env python3
"""ESP32-CAM MJPEG reader built for OpenCV / neural-network pipelines.

The board (src/main.cpp) joins your WiFi as a station and serves MJPEG on
http://esp32cam.local/stream (mDNS) or http://<ip>/stream.

Streaming defaults to QVGA (320x240 @ ~24 fps) — measured on this hardware:
the sensor cannot hold a usable frame rate at VGA (6-8 fps), and low latency
matters more than pixel count for a control loop. Still captures stay at VGA.
Override any time with --size hvga|vga or ?size= on the URL.

Feed frames straight into a model:

    from stream import ESP32Cam

    with ESP32Cam() as cam:
        for frame in cam.frames():          # numpy BGR, uint8, 240x320x3
            result = model(frame)

Or grab the newest frame on demand:

    cam = ESP32Cam()
    cam.start()
    frame = cam.wait_for_frame()            # blocks until one arrives
    cam.stop()

CLI:
    python3 stream.py stream                # OpenCV preview window
    python3 stream.py stream --no-display   # headless, just stats
    python3 stream.py stream --size hvga    # 480x320 @ ~13fps
    python3 stream.py capture               # save one VGA JPEG
    python3 stream.py flash                 # build + upload firmware
"""

import argparse
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
DEFAULT_SIZE = "qvga"  # 320x240 @ ~24fps — best latency/fps for robotics
DEFAULT_URL = f"http://{DEFAULT_HOST}/stream?size={DEFAULT_SIZE}"
CAPTURE_URL = f"http://{DEFAULT_HOST}/capture"

ENV = "esp32cam"
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 10.0
MAX_FRAME = 131072
RECV_SIZE = 65536

__all__ = ["ESP32Cam", "DEFAULT_URL", "CAPTURE_URL", "DEFAULT_SIZE"]


class ESP32Cam:
    """Reads the board's MJPEG stream and yields decoded OpenCV frames.

    Frames are plain ``numpy.ndarray`` (dtype uint8, shape HxWx3, BGR order),
    i.e. exactly what ``cv2`` and most model runtimes expect.
    """

    def __init__(self, url=DEFAULT_URL):
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

        self._sock = None
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
        self._connect()
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        self._close_socket()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
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
        deadline = time.time() + timeout if timeout else None
        with self._cond:
            while self._running:
                frame = self._frame
                if frame is not None and frame is not after:
                    return frame
                remaining = None if deadline is None else deadline - time.time()
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
        if self._sock is None:
            return
        try:
            self._sock.close()
        except OSError:
            pass
        self._sock = None

    def _connect(self):
        self._close_socket()
        self._buf = bytearray()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(CONNECT_TIMEOUT)
        sock.connect((self.host, self.port))

        request = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}\r\n"
            f"Accept: multipart/x-mixed-replace\r\n"
            f"Connection: keep-alive\r\n"
            f"\r\n"
        )
        sock.sendall(request.encode())

        # Read the header block, keeping any body bytes that arrived with it.
        header = b""
        while b"\r\n\r\n" not in header:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("connection closed during headers")
            header += chunk

        head, _, body = header.partition(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        if " 200" not in status:
            raise ConnectionError(f"unexpected response: {status}")

        sock.settimeout(READ_TIMEOUT)
        self._sock = sock
        self._buf = bytearray(body)

    def _read_jpeg(self):
        buf = self._buf
        while self._running:
            start = buf.find(b"\xff\xd8")
            if start != -1:
                end = buf.find(b"\xff\xd9", start)
                if end != -1:
                    jpeg = bytes(buf[start : end + 2])
                    del buf[: end + 2]
                    return jpeg
                if len(buf) > MAX_FRAME:
                    del buf[:]
            elif len(buf) > 4096:
                # keep the tail in case a marker straddles two reads
                del buf[:-2]

            chunk = self._sock.recv(RECV_SIZE)
            if not chunk:
                raise ConnectionError("connection closed")
            buf += chunk
        return None

    def _publish(self, frame):
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    def _reader_loop(self):
        window_start = time.time()
        while self._running:
            try:
                jpeg = self._read_jpeg()
                if jpeg is None:
                    continue

                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                self._publish(frame)
                self._frame_count += 1

                elapsed = time.time() - window_start
                if elapsed >= 1.0:
                    self.fps = self._frame_count / elapsed
                    self._frame_count = 0
                    window_start = time.time()
            except (ConnectionError, socket.timeout, OSError):
                if not self._running:
                    break
                time.sleep(0.5)
                try:
                    self._connect()
                except (ConnectionError, OSError):
                    time.sleep(1.5)

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


def stream_video(url=DEFAULT_URL, save=False, display=True, max_frames=None):
    """Open the stream; optionally display it and/or write an MP4.

    Returns the number of frames read, so callers can use it as a source:
        n = stream_video(display=False, max_frames=30)
    """
    cam = ESP32Cam(url)
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
        help="camera resolution (applied live on the board)",
    )
    parser.add_argument(
        "--quality", type=int, help="JPEG quality 4..63 (lower = bigger/better image)"
    )
    args = parser.parse_args()

    url = args.url
    if args.ip:
        parsed = urlparse(url)
        url = f"http://{args.ip}:{parsed.port or 80}{parsed.path or '/stream'}"

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
        )
    elif args.command == "capture":
        u = urlparse(url)
        capture_frame(url=f"http://{u.hostname}:{u.port or 80}/capture")


if __name__ == "__main__":
    main()
