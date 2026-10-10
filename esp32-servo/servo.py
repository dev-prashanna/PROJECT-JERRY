#!/usr/bin/env python3
"""Drive the ESP32 servo on GPIO14 over USB serial (/dev/ttyUSB0).

The firmware (src/main.cpp) takes newline-terminated commands:

    A<deg> or <deg>   set angle, 0-180     e.g. "A90"
    W<us>             raw pulse width      e.g. "W1500"
    S                 toggle sweep
    C                 center (90)
    D                 detach (stop pulses)
    ?                 help

CLI:
    python3 servo.py 90               move to 90 degrees
    python3 servo.py loop             90 <-> 0 forever (Ctrl+C stops)
    python3 servo.py loop 90 --to 0 --cycles 5
    python3 servo.py width 1500       raw 1500 us pulse
    python3 servo.py sweep            start/stop sweep
    python3 servo.py center
    python3 servo.py detach
    python3 servo.py flash            build + upload firmware

    # one-liner from another script:
    from servo import Servo
    with Servo() as s:
        s.angle(45)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import serial

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200
RESET_SETTLE_S = 2.0  # auto-reset on open; give the board time to boot

ENV = "esp32dev"


class Servo:
    """Newline command channel to the ESP32 servo firmware."""

    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD, reset=True):
        self.ser = serial.Serial(port, baud, timeout=1)
        if reset:
            time.sleep(RESET_SETTLE_S)
            self.ser.reset_input_buffer()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.ser.is_open:
            self.ser.close()

    def command(self, text):
        """Send one command and return any reply lines the board prints."""
        self.ser.write((text.strip() + "\n").encode())
        self.ser.flush()
        time.sleep(0.1)
        return [
            line.decode(errors="replace").strip()
            for line in self.ser.readlines()
            if line.strip()
        ]

    def angle(self, degrees):
        if not 0 <= degrees <= 180:
            raise ValueError("angle must be 0-180")
        return self.command(str(int(degrees)))

    def width(self, microseconds):
        if not 500 <= microseconds <= 2500:
            raise ValueError("width must be 500-2500 us")
        return self.command(f"W{int(microseconds)}")

    def sweep(self):
        return self.command("S")

    def center(self):
        return self.command("C")

    def detach(self):
        return self.command("D")

    def loop(self, start=90, end=0, step_delay=0.02, cycles=None):
        """Oscillate start <-> end forever (or for `cycles` round trips).

        Steps one degree at a time so the motion is smooth; Ctrl+C stops it
        and leaves the servo parked at the start angle.
        """
        step = 1 if end >= start else -1
        path = list(range(start, end + step, step)) + list(
            range(end, start + step, step)
        )[1:]  # start -> end -> start, no duplicate endpoint
        try:
            n = 0
            while cycles is None or n < cycles:
                for deg in path:
                    self.ser.write(f"{deg}\n".encode())
                    self.ser.reset_input_buffer()  # drop per-step ACKs so the
                    time.sleep(step_delay)  # board's TX never blocks the motion
                n += 1
        except KeyboardInterrupt:
            pass
        finally:
            self.ser.flush()
            self.angle(start)


def flash_firmware(port=None):
    """Build and upload the servo firmware via PlatformIO."""
    cmd = ["pio", "run", "-e", ENV, "-t", "upload"]
    if port:
        cmd += ["--upload-port", port]

    print(f"[+] Building and flashing env '{ENV}'" + (f" on {port}" if port else ""))
    if subprocess.run(cmd, cwd=str(Path(__file__).parent)).returncode != 0:
        sys.exit("[!] Flash failed")

    print("[+] Flash complete. Board is rebooting...")
    time.sleep(RESET_SETTLE_S)


def print_reply(lines):
    for line in lines:
        print(f"    {line}")


def main():
    parser = argparse.ArgumentParser(description="ESP32 servo control (GPIO14)")
    parser.add_argument(
        "command",
        nargs="?",
        help="angle 0-180, or: loop [deg] | width <us> | sweep | center | detach | flash | demo",
    )
    parser.add_argument(
        "value",
        nargs="?",
        type=int,
        help="start angle for 'loop' (default 90), or pulse width in us for 'width'",
    )
    parser.add_argument("--to", type=int, default=0, help="loop end angle (default 0)")
    parser.add_argument("--cycles", type=int, help="loop round trips before stopping (default forever)")
    parser.add_argument("--delay", type=float, default=0.02, help="loop delay per degree in seconds")
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"serial port (default {DEFAULT_PORT})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="baud rate")
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "flash":
        flash_firmware(args.port)
        return

    try:
        with Servo(args.port, args.baud) as servo:
            if args.command == "width":
                if args.value is None:
                    sys.exit("[!] width needs a value in us, e.g. width 1500")
                print_reply(servo.width(args.value))
            elif args.command == "sweep":
                print_reply(servo.sweep())
            elif args.command == "center":
                print_reply(servo.center())
            elif args.command == "detach":
                print_reply(servo.detach())
            elif args.command == "loop":
                start = args.value if args.value is not None else 90
                for deg in (start, args.to):
                    if not 0 <= deg <= 180:
                        sys.exit("[!] angles must be 0-180")
                print(
                    f"[+] looping {start} <-> {args.to} "
                    f"({'forever, Ctrl+C to stop' if args.cycles is None else f'{args.cycles} cycles'})"
                )
                servo.loop(start, args.to, step_delay=args.delay, cycles=args.cycles)
                print("[+] loop done, parked at", start)
            elif args.command == "demo":
                print("[+] demo: sweep 0 -> 180 -> 0, three times")
                print_reply(servo.sweep())
                time.sleep(3 * 2 * 180 * 0.015 + 1.0)
                print_reply(servo.sweep())  # stop
                print_reply(servo.center())
            else:
                try:
                    print_reply(servo.angle(int(args.command)))
                except ValueError:
                    sys.exit(f"[!] unknown command: {args.command}")
    except serial.SerialException as e:
        sys.exit(f"[!] serial error: {e}")


if __name__ == "__main__":
    main()
