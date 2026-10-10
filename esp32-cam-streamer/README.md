# ESP32-CAM streamer

Firmware targets the AI-Thinker ESP32-CAM with an OV2640 camera. WiFi credentials
are read from `include/secrets.h`. The live MJPEG stream runs on port 81;
`/capture` and `/status` run on port 80.

HD (1280x720), JPEG quality 10, two PSRAM frame buffers, and WiFi power saving
disabled are the defaults. JPEG quality adapts between 10 and 25 toward a
20 FPS target. This target is not a measured or guaranteed HD frame rate.
Actual performance depends on sensor exposure, scene complexity and WiFi.
Without PSRAM, the firmware uses QVGA and one internal-RAM frame buffer.

## Review and fixes

- The Python reader now decodes HTTP transfer chunks before extracting JPEGs.
  Previously chunk framing inside an image could corrupt it.
- Accepted server sockets now use `TCP_NODELAY`. Combining a multipart frame
  into one buffer does not eliminate the separate HTTP chunk framing writes.
- Camera buffers are allocated at UXGA before selecting HD, and resolution
  requests are limited to OV2640-supported sizes and available memory.
- Settings, frame acquisition and FPS/tuner state share a mutex across the
  two HTTP tasks. Frames are copied and returned before network sends.
- Invalid size/quality values return HTTP 400. `quality=auto` resumes tuning
  after a manual quality override. Lower numeric JPEG quality means a larger,
  higher-quality image.
- Stream staging buffers grow in blocks in PSRAM instead of repeatedly
  allocating exact sizes and falling back into WiFi's internal RAM.
- FPS resets between streaming sessions, mDNS restarts after disconnection,
  and a failed Python reconnect continues retrying with socket cleanup.
- Python shutdown wakes frame waiters; timing uses a monotonic clock; preview
  overlays use a copy; CLI IP overrides preserve URL query settings.

## Flash from a local terminal

The review session compiled the firmware, but could not flash it: Linux's sysfs
and udev identify an FTDI FT232R on `ttyUSB0`, while `/dev/ttyUSB0` is absent in
the session's device namespace. The upload failed before any flash write.

For an FTDI connection without automatic boot control, connect GPIO0 to GND
and reset the ESP32-CAM before uploading. After uploading, disconnect GPIO0
from GND and reset again to run the firmware. See Espressif's
[boot mode instructions](https://docs.espressif.com/projects/esptool/en/latest/esp32/advanced-topics/boot-mode-selection.html).

Run these commands from this directory in your computer's terminal:

```bash
pio run -e esp32cam -t upload --upload-port /dev/ttyUSB0
pio device monitor --port /dev/ttyUSB0 --baud 115200
```

The boot log prints the camera initialization result, WiFi IP and RSSI.
The built application is `.pio/build/esp32cam/firmware.bin`; use PlatformIO's
upload target so the bootloader and partition table are also placed correctly.

## Check stream performance

```bash
python3 stream.py stream --ip <board-ip> --no-display --frames 300
curl http://<board-ip>/status
```

`/status` reports the active resolution, quality, automatic/manual quality
mode, streamed FPS, RSSI, free internal RAM and free PSRAM. The Python FPS
counts decoded frames, while firmware FPS counts frames handed to the socket.
Only one active stream is served by the synchronous streaming HTTP task.

If HD remains slow, compare VGA or SVGA at the same location and lighting:

```bash
python3 stream.py stream --ip <board-ip> --size svga --no-display --frames 300
python3 stream.py stream --ip <board-ip> --size vga --no-display --frames 300
```

Live quality control examples:

```text
http://<board-ip>:81/stream?size=hd&quality=10
http://<board-ip>:81/stream?size=hd&quality=auto
```

## Verification

```bash
pio run -e esp32cam
python3 -m unittest discover -s tests -v
```

The four regression tests use actual HTTP response bytes with an in-memory
socket substitute: split-chunk JPEG decoding, rejected-connection cleanup,
retry after a failed reconnect, and waking waiters on shutdown. Camera capture,
WiFi throughput and boot behavior require verification on the physical board.
