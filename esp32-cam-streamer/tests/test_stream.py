"""Decode real HTTP response bytes without requiring network or camera access."""
import io
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stream import ESP32Cam


class MemorySocket:
    def __init__(self, response):
        self.response = io.BytesIO(response)
        self.closed = False

    def setsockopt(self, *args):
        pass

    def settimeout(self, *args):
        pass

    def connect(self, *args):
        pass

    def sendall(self, *args):
        pass

    def makefile(self, *args):
        return self.response

    def shutdown(self, *args):
        pass

    def close(self):
        self.closed = True


class StreamTests(unittest.TestCase):
    def setUp(self):
        image = np.full((48, 64, 3), (23, 91, 177), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        self.jpeg = encoded.tobytes()

    def response(self):
        body = (b"--test\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(self.jpeg)).encode() + b"\r\n\r\n" + self.jpeg + b"\r\n")
        wire = (b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: multipart/x-mixed-replace; boundary=test\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n")
        # Split JPEG markers and compressed data across HTTP chunks.
        for offset in range(0, len(body), 7):
            chunk = body[offset:offset + 7]
            wire += f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n"
        return wire + b"0\r\n\r\n"

    def test_jpeg_split_across_http_chunks_decodes_exactly(self):
        sock = MemorySocket(self.response())
        with patch("stream.socket.socket", return_value=sock):
            camera = ESP32Cam()
            camera._connect()
            camera._running = True
            try:
                jpeg = camera._read_jpeg()
                self.assertEqual(jpeg, self.jpeg)
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(frame.shape, (48, 64, 3))
            finally:
                camera.stop()
        self.assertTrue(sock.closed)

    def test_failed_initial_connection_releases_socket(self):
        sock = MemorySocket(b"HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\n\r\n")
        with patch("stream.socket.socket", return_value=sock):
            camera = ESP32Cam()
            with self.assertRaises(ConnectionError):
                camera.start()
            self.assertIsNone(camera._sock)
            self.assertIsNone(camera._response)
            self.assertFalse(camera.running)
            self.assertTrue(sock.closed)

    def test_reconnect_survives_a_failed_retry(self):
        sockets = [MemorySocket(self.response()),
                   MemorySocket(b"HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\n\r\n"),
                   MemorySocket(self.response())]
        with patch("stream.socket.socket", side_effect=sockets) as factory:
            with ESP32Cam() as camera:
                first = camera.wait_for_frame(timeout=1)
                self.assertIsNotNone(first)
                second = camera.wait_for_frame(timeout=4, after=first)
                self.assertIsNotNone(second)
                self.assertEqual(factory.call_count, 3)

    def test_stop_wakes_frame_waiters(self):
        camera = ESP32Cam()
        camera._running = True
        result = []
        waiter = threading.Thread(target=lambda: result.append(
            camera.wait_for_frame(timeout=60)))
        waiter.start()
        camera.stop()
        waiter.join(timeout=1)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [None])


if __name__ == "__main__":
    unittest.main()
