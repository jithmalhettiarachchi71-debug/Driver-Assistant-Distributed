"""
ZeroMQ video client for distributed Driver Assistant architecture.

Raspberry Pi role:
- Capture camera frames.
- Resize/compress and send to laptop over Ethernet.
- Receive detection JSON and trigger GPIO outputs.
- Draw bounding boxes and show HDMI preview.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import zmq

try:
    import RPi.GPIO as GPIO  # type: ignore
except Exception:  # Not running on Raspberry Pi or GPIO unavailable.
    GPIO = None


LOGGER = logging.getLogger("pi_client")
PROTOCOL_VERSION = "1.0"
OFFLINE_TIMEOUT_SECONDS = 2.0


class GPIOController:
    """Best-effort GPIO abstraction with graceful fallback."""

    def __init__(self, buzzer_pin: int = 18, led_pin: int = 27) -> None:
        self.buzzer_pin = buzzer_pin
        self.led_pin = led_pin
        self.enabled = False

        try:
            if GPIO is None:
                raise RuntimeError("RPi.GPIO module unavailable")
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.buzzer_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.led_pin, GPIO.OUT, initial=GPIO.LOW)
            self.enabled = True
            LOGGER.info("GPIO enabled (buzzer=%s, led=%s)", buzzer_pin, led_pin)
        except Exception as exc:
            LOGGER.warning("GPIO disabled: %s", exc)
            self.enabled = False

    def set_alert(self, is_danger: bool) -> None:
        if not self.enabled:
            return
        try:
            level = GPIO.HIGH if is_danger else GPIO.LOW
            GPIO.output(self.buzzer_pin, level)
            GPIO.output(self.led_pin, level)
        except Exception as exc:
            LOGGER.error("GPIO write failed: %s", exc)

    def set_offline_alarm(self, enabled: bool, pulse_on: bool = True) -> None:
        """Drive fast-pulse buzzer pattern while system link is offline."""
        if not self.enabled:
            return
        try:
            if enabled:
                GPIO.output(self.buzzer_pin, GPIO.HIGH if pulse_on else GPIO.LOW)
                GPIO.output(self.led_pin, GPIO.HIGH)
            else:
                GPIO.output(self.buzzer_pin, GPIO.LOW)
                GPIO.output(self.led_pin, GPIO.LOW)
        except Exception as exc:
            LOGGER.error("GPIO offline alarm write failed: %s", exc)

    def cleanup(self) -> None:
        if not self.enabled:
            return
        try:
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            GPIO.output(self.led_pin, GPIO.LOW)
            GPIO.cleanup()
        except Exception as exc:
            LOGGER.warning("GPIO cleanup issue: %s", exc)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_socket(context: zmq.Context, server_addr: str, timeout_ms: int) -> zmq.Socket:
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(server_addr)
    return socket


def encode_frame(frame) -> Optional[str]:
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        return None
    return base64.b64encode(jpg.tobytes()).decode("ascii")


def draw_boxes(frame, boxes: List[List[Any]]) -> None:
    for item in boxes:
        if len(item) < 5:
            continue
        x1, y1, x2, y2, label = item[:5]
        try:
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
        except (TypeError, ValueError):
            continue

        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            frame,
            str(label),
            (p1[0], max(20, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def draw_center_warning(frame, message: str, color: Tuple[int, int, int]) -> None:
    """Draw large center-screen warning banner."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    pad = 24
    text_scale = max(0.8, min(w, h) / 700.0)
    thickness = max(2, int(text_scale * 3))
    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
    x1 = max(10, (w - tw) // 2 - pad)
    y1 = max(10, (h - th) // 2 - pad)
    x2 = min(w - 10, x1 + tw + 2 * pad)
    y2 = min(h - 10, y1 + th + 2 * pad)

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0.0, frame)
    text_x = x1 + pad
    text_y = y1 + pad + th
    cv2.putText(
        frame,
        message,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def parse_response(payload: Dict[str, Any]) -> Tuple[str, List[List[Any]], bool]:
    alert = payload.get("alert", "SAFE")
    version = str(payload.get("version", ""))
    boxes = payload.get("boxes", [])
    version_ok = version == PROTOCOL_VERSION
    if not isinstance(boxes, list):
        boxes = []
    if not version_ok:
        boxes = []
    return str(alert), boxes, version_ok


def run_client(
    server_addr: str,
    camera_index: int,
    timeout_ms: int,
    frame_size: Tuple[int, int],
    window_name: str,
) -> None:
    width, height = frame_size
    context = zmq.Context.instance()
    socket = build_socket(context, server_addr, timeout_ms=timeout_ms)
    gpio = GPIOController(buzzer_pin=18, led_pin=27)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}")

    LOGGER.info("Connected to laptop server at %s", server_addr)
    last_success_time = time.monotonic()
    pulse_on = False
    last_pulse_toggle = 0.0
    pulse_interval_s = 0.12

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                LOGGER.warning("Camera frame read failed; retrying")
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            encoded = encode_frame(frame)
            if encoded is None:
                LOGGER.warning("JPEG encoding failed for current frame")
                continue

            reply = None
            try:
                socket.send_json({"frame": encoded})
                reply = socket.recv_json()
            except zmq.error.Again:
                LOGGER.warning("ZMQ timeout: laptop unreachable")
                # REQ sockets must preserve send/recv order, so rebuild on timeout.
                socket.close(0)
                socket = build_socket(context, server_addr, timeout_ms=timeout_ms)
            except Exception as exc:
                LOGGER.error("ZMQ communication error: %s", exc)
                socket.close(0)
                socket = build_socket(context, server_addr, timeout_ms=timeout_ms)
                reply = None

            now = time.monotonic()
            offline = (now - last_success_time) > OFFLINE_TIMEOUT_SECONDS
            show_version_mismatch_warning = False

            if isinstance(reply, dict):
                alert, boxes, version_ok = parse_response(reply)
                if version_ok:
                    last_success_time = now
                    offline = False
                    gpio.set_offline_alarm(False)
                    is_danger = alert.upper() == "DANGER"
                    gpio.set_alert(is_danger)
                    draw_boxes(frame, boxes)
                    cv2.putText(
                        frame,
                        f"ALERT: {alert}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255) if is_danger else (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    gpio.set_alert(False)
                    show_version_mismatch_warning = True
                    draw_center_warning(frame, "VERSION MISMATCH ERROR", (0, 0, 255))
            else:
                gpio.set_alert(False)

            if offline:
                # Fast pulsing alarm pattern while keeping camera/render path alive.
                if (now - last_pulse_toggle) >= pulse_interval_s:
                    pulse_on = not pulse_on
                    last_pulse_toggle = now
                gpio.set_offline_alarm(True, pulse_on=pulse_on)
                if not show_version_mismatch_warning:
                    draw_center_warning(frame, "SYSTEM OFFLINE - CHECK CONNECTION", (0, 0, 255))

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        gpio.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        socket.close(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed Raspberry Pi video client")
    parser.add_argument("--server-ip", default="192.168.1.1", help="Laptop server IP")
    parser.add_argument("--port", type=int, default=5555, help="Laptop server port")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--timeout-ms", type=int, default=1200, help="ZMQ send/recv timeout")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--window-name", default="Pi Eyes", help="Display window name")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    server_addr = f"tcp://{args.server_ip}:{args.port}"
    run_client(
        server_addr=server_addr,
        camera_index=args.camera_index,
        timeout_ms=args.timeout_ms,
        frame_size=(args.width, args.height),
        window_name=args.window_name,
    )


if __name__ == "__main__":
    main()
