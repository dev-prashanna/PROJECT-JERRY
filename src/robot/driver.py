#!/usr/bin/env python3
"""UDP driver + Gemini driving agent for the ESP32 robot car.

Protocol — must stay in sync with esp32-motor-driver/src/main.cpp:

    "V<0-255>"                    set motor PWM speed
    "F" | "B" | "L" | "R" | "S"   forward / backward / left / right / stop

The board runs a soft-AP at 192.168.4.1 and listens on UDP 8888 (the PC must
be joined to that AP). Its firmware has a 2 s command failsafe: silence means
stop. So while a motion command is active this driver re-sends it as a
keep-alive, and re-sends the speed with it so state survives a board reboot.

Usage:
    python3 src/robot/driver.py                       # interactive: LLM scenarios + F/B/L/R/S/V keys
    python3 src/robot/driver.py "drive to the wall"   # one-shot agent decision
    python3 src/robot/driver.py --manual              # keyboard: w/a/s/d, space=stop, +/-=speed
"""

import argparse
import os
import socket
import sys
import time
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

ESP32_IP = "192.168.4.1"
ESP32_PORT = 8888
DEFAULT_SPEED = 200
KEEPALIVE_S = 0.5  # must stay well under the firmware's 2000 ms failsafe
VALID_ACTIONS = ("F", "B", "L", "R", "S")


def require_api_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")


class CarAction(BaseModel):
    action: Literal["F", "B", "L", "R", "S"] = Field(
        ..., description="Action to perform: F=Forward, B=Backward, L=Left, R=Right, S=Stop"
    )
    speed: int = Field(default=DEFAULT_SPEED, ge=0, le=255, description="Speed of the car (0-255)")
    reasoning: str = Field(description="Reasoning behind the action")


class Driver:
    """UDP link to the car. Keeps speed/action state so keep-alives can replay it."""

    def __init__(self, ip: str = ESP32_IP, port: int = ESP32_PORT):
        self.ip = ip
        self.port = port
        self.speed = DEFAULT_SPEED
        self.action = "S"
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # -------------------------------------------------------------
    # low level
    # -------------------------------------------------------------
    def send(self, command: str) -> None:
        try:
            self._sock.sendto(command.encode("utf-8"), (self.ip, self.port))
        except OSError as e:
            print(f"Error sending command: {e}")

    def set_speed(self, speed: int) -> None:
        self.speed = max(0, min(int(speed), 255))
        self.send(f"V{self.speed}")

    def drive(self, action: str) -> None:
        action = action.upper()
        if action not in VALID_ACTIONS:
            raise ValueError(f"unknown action {action!r}, expected one of {VALID_ACTIONS}")
        self.action = action
        # speed first so the board never runs a decision at a stale/default PWM
        if action != "S":
            self.send(f"V{self.speed}")
        self.send(action)

    def stop(self) -> None:
        self.drive("S")

    def keepalive(self) -> None:
        """Replay the active command; call at >= KEEPALIVE_S cadence while moving."""
        if self.action == "S":
            return
        self.send(f"V{self.speed}")
        self.send(self.action)

    def hold(self, seconds: float) -> None:
        """Keep the current command alive for ``seconds``; caller is responsible for stopping."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(min(KEEPALIVE_S, max(0.0, deadline - time.monotonic())))
            self.keepalive()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self._sock.close()

    def __enter__(self) -> "Driver":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


# -------------------------------------------------------------
# agent
# -------------------------------------------------------------
def build_chain():
    """prompt_template | structured CarAction (requires GOOGLE_API_KEY)."""
    require_api_key()

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an autonomous AI driving agent controlling an ESP32 robot car. "
                "Analyze the user prompt or scene input and decide the immediate control action. "
                "You must output valid driving decisions according to the required schema.",
            ),
            ("human", "{scenario}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    return prompt_template | llm.as_structured_output(CarAction)


def drive_scenario(car: Driver, chain, scenario: str, hold_s: float) -> CarAction:
    """Ask the agent for one decision, apply it, hold it, then stop."""
    result: CarAction = chain.invoke({"scenario": scenario})
    print(f"[agent] {result.action} speed={result.speed} — {result.reasoning}")
    car.set_speed(result.speed)
    car.drive(result.action)
    if result.action != "S":
        try:
            car.hold(hold_s)
        finally:
            car.stop()
    return result


# -------------------------------------------------------------
# manual control
# -------------------------------------------------------------
# Raw keyboard (one keypress at a time): WASD driving, space/x stop.
KEY_WASD = {"w": "F", "a": "L", "s": "B", "d": "R", " ": "S", "x": "S"}


def manual_loop(car: Driver) -> None:
    """w/a/s/d drive, space or x stops, +/- adjust speed, q quits."""
    print("[manual] w/a/s/d=drive  space/x=stop  +/-=speed  q=quit")
    if not sys.stdin.isatty():
        while True:
            try:
                line = input("cmd> ").strip()
            except EOFError:
                break
            if not step_line(car, line):
                break
        return

    # terminal handling is POSIX-only; import here so the module stays importable elsewhere
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            ready, _, _ = select.select([fd], [], [], KEEPALIVE_S)
            if ready:
                if not step_key(car, sys.stdin.read(1)):
                    break
            car.keepalive()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def step_key(car: Driver, key: str) -> bool:
    """Handle one raw keypress. Returns False when the loop should exit."""
    low = key.lower()
    if low in ("q", "quit", "exit"):
        return False
    if low in ("+", "="):
        car.set_speed(car.speed + 25)
        print(f"[speed] {car.speed}")
    elif low in ("-", "_"):
        car.set_speed(car.speed - 25)
        print(f"[speed] {car.speed}")
    elif key != low and key.upper() in VALID_ACTIONS:
        # shifted letter wins over WASD: capital S = stop
        car.drive(key.upper())
        print(f"[cmd] {key.upper()}")
    elif low in KEY_WASD:
        car.drive(KEY_WASD[low])
        print(f"[cmd] {KEY_WASD[low]}")
    elif key.upper() in VALID_ACTIONS:  # f/b/l/r
        car.drive(key.upper())
        print(f"[cmd] {key.upper()}")
    return True


def step_line(car: Driver, text: str) -> bool:
    """Handle one line of input: F/B/L/R/S, V<n>, or +/- speed. False = quit."""
    text = text.strip()
    if not text:
        return True
    low = text.lower()
    if low in ("q", "quit", "exit"):
        return False
    if text.upper() in VALID_ACTIONS:
        car.drive(text.upper())
        print(f"[cmd] {text.upper()}")
    elif low.startswith("v") and low[1:].lstrip("-").isdigit():
        car.set_speed(int(low[1:]))
        print(f"[speed] {car.speed}")
    elif low in ("+", "="):
        car.set_speed(car.speed + 25)
        print(f"[speed] {car.speed}")
    elif low in ("-", "_"):
        car.set_speed(car.speed - 25)
        print(f"[speed] {car.speed}")
    else:
        print(f"[?] unknown command {text!r}")
    return True


def is_command(text: str) -> bool:
    """True for direct commands (F/B/L/R/S or V<n>) as opposed to agent scenarios."""
    text = text.strip().upper()
    if text in VALID_ACTIONS:
        return True
    return text.startswith("V") and text[1:].lstrip("-").isdigit()


def interactive_loop(car: Driver, chain) -> None:
    """Free-form scenarios for the agent; single letters F/B/L/R/S/V<n> drive directly."""
    print("[interactive] type a scenario, or F/B/L/R/S / V200 to drive directly (q quits)")
    while True:
        try:
            text = input("scenario> ").strip()
        except EOFError:
            break
        if not text:
            continue
        if text.lower() in ("q", "quit", "exit"):
            break
        if is_command(text):
            step_line(car, text)
            car.hold(1.0)
            car.stop()
            continue
        try:
            drive_scenario(car, chain, text, hold_s=2.0)
        except KeyboardInterrupt:
            break
        except Exception as e:  # API quota/network/schema hiccup — keep the session alive
            print(f"[agent] error: {e}")
            car.stop()


def pwm_speed(value: str) -> int:
    """argparse type: integer PWM duty in 0..255."""
    try:
        speed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if not 0 <= speed <= 255:
        raise argparse.ArgumentTypeError(f"{speed} out of range 0-255")
    return speed


def main() -> None:
    parser = argparse.ArgumentParser(description="ESP32 robot car UDP driver + Gemini agent")
    parser.add_argument("scenario", nargs="*", help="one-shot agent prompt (omit for interactive)")
    parser.add_argument("--manual", action="store_true", help="keyboard control, no LLM")
    parser.add_argument("--ip", default=ESP32_IP, help=f"board IP (default {ESP32_IP})")
    parser.add_argument("--speed", type=pwm_speed, default=DEFAULT_SPEED, help="initial PWM 0-255")
    parser.add_argument("--duration", type=float, default=3.0, help="seconds to hold each agent action")
    args = parser.parse_args()

    with Driver(args.ip) as car:
        car.set_speed(args.speed)
        print(f"[link] UDP {args.ip}:{ESP32_PORT}  speed={car.speed}")

        try:
            if args.manual:
                manual_loop(car)
                return
            try:
                chain = build_chain()
            except (ValueError, ImportError) as e:
                sys.exit(
                    f"[!] agent unavailable: {e}\n"
                    "    install deps: pip install langchain-google-genai langchain-core"
                )
            if args.scenario:
                drive_scenario(car, chain, " ".join(args.scenario), args.duration)
            else:
                interactive_loop(car, chain)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            sys.exit(f"[agent] {e}")
        finally:
            car.stop()
            print("[link] stopped")


if __name__ == "__main__":
    main()
