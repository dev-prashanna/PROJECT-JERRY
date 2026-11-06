# ESP32-CAM streamer

Firmware targets the AI-Thinker ESP32-CAM with an OV2640 camera. WiFi credentials
are read from `include/secrets.h`. The live MJPEG stream runs on port 81;
`/capture` and `/status` run on port 80.

HD (1280x720), JPEG quality 10, two PSRAM frame buffers, and WiFi power saving
disabled are the defaults. JPEG quality adapts between 10 and 25 toward a
20 FPS target. This target is not a measured or guaranteed HD frame rate.
Actual performance depends on sensor exposure, scene complexity and WiFi.
Without PSRAM, the firmware uses QVGA and one internal-RAM frame buffer.

## WiFi and stream recovery

WiFi association starts without blocking startup for 30 seconds. The hostname
is configured before starting WiFi, and HTTP servers start after a nonzero
DHCP address is available. The serial monitor uses **115200 baud**.

The previous firmware printed its IP only inside the initial connection
attempt. A connection after that timeout, or a later reconnection, silently
acquired an IP. The revised firmware prints the IP, RSSI and available endpoint
URLs on connection and every 30 seconds, so a monitor opened late can still
discover the address. Association, DHCP loss and disconnect reason codes are
logged separately. Automatic WiFi reconnect remains enabled, with a fallback
retry every 30 seconds while no IP is available.

The old HTTP startup results were discarded: a failed server stayed down until
reset. Servers now retry every five seconds and restart after an IP change or
network interruption. A disconnect invalidates an active stream even when
WiFi reconnects quickly, releasing its staging buffer and old connection.
Sensor startup errors and malformed JPEG frames are reported, and FPS tuning
windows reset when resolution or quality changes.

These changes improve diagnostics and recovery; they do not establish a higher
measured frame rate. The serial corruption observed before this revision also
affected ROM boot output, which runs before this application. Its cause is
still unconfirmed and cannot be attributed to the IP-reporting bug.

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

The revised firmware has been compiled locally but has not been uploaded as
part of this revision. The FTDI FT232R adapter is available on the host at
`/dev/ttyUSB0`; access from a sandboxed session may require host permissions.

For an FTDI connection without automatic boot control, connect GPIO0 to GND
and reset the ESP32-CAM before uploading. After uploading, disconnect GPIO0
from GND and reset again to run the firmware. See Espressif's
[boot mode instructions](https://docs.espressif.com/projects/esptool/en/latest/esp32/advanced-topics/boot-mode-selection.html).

Run these commands from this directory in your computer's terminal:

```bash
pio run -e esp32cam -t upload --upload-port /dev/ttyUSB0
pio device monitor --port /dev/ttyUSB0 --baud 115200
```

The boot log prints its build date, reset reason and camera initialization
result. WiFi IP, RSSI and endpoint URLs are also printed every 30 seconds.
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
After flashing, also verify that the IP appears when the monitor is opened
late, and that `/status` and a new stream work after disconnecting and restoring
the access point. Existing streams must reconnect after a network interruption.
