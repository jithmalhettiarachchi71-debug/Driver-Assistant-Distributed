"""
ZeroMQ detection server for distributed Driver Assistant architecture.

Laptop role:
- Receive base64-encoded JPEG frames from Raspberry Pi over Ethernet.
- Run YOLO inference using existing src.detection.YOLODetector.
- Return compact JSON detections with alert state.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import zmq

# Ensure project root is importable when this script is launched directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.detection import YOLODetector
from src.detection.result import DetectionLabel


LOGGER = logging.getLogger("laptop_server")
PROTOCOL_VERSION = "1.0"


def _to_int(value: Any) -> int:
    """Safely convert numpy/python numeric values to plain Python int."""
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


def decode_frame_from_base64(payload: str) -> np.ndarray:
    """Decode base64 JPEG payload into an OpenCV BGR frame."""
    raw_bytes = base64.b64decode(payload, validate=True)
    np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG payload into frame.")
    return frame


def detections_to_payload(detector_result: Any) -> Dict[str, Any]:
    """Convert detection result to JSON-serializable API payload."""
    boxes: List[List[Any]] = []
    alert = "SAFE"

    for det in detector_result.detections:
        x1, y1, x2, y2 = det.bbox
        label = det.label.value

        boxes.append([
            _to_int(round(x1)),
            _to_int(round(y1)),
            _to_int(round(x2)),
            _to_int(round(y2)),
            label,
        ])

        if det.label in (DetectionLabel.VEHICLE, DetectionLabel.PEDESTRIAN):
            alert = "DANGER"

    return {"version": PROTOCOL_VERSION, "alert": alert, "boxes": boxes}


def keep_alive_payload(error: str | None = None) -> Dict[str, Any]:
    """Return minimal keep-alive payload so client knows server is alive."""
    payload: Dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "alert": "SAFE",
        "boxes": [],
    }
    if error:
        payload["error"] = error
    return payload


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_server(bind_addr: str, model_path: str) -> None:
    """Main server loop. Never exits on frame-level failures."""
    LOGGER.info("Loading detector model: %s", model_path)
    detector = YOLODetector(model_path=model_path, frame_skip=1)

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(bind_addr)
    LOGGER.info("Laptop detection server listening on %s", bind_addr)

    while True:
        response: Dict[str, Any]

        try:
            request_text = socket.recv_string()
            request = json.loads(request_text)
            frame_b64 = request["frame"]

            frame = decode_frame_from_base64(frame_b64)
            result = detector.detect(frame)
            response = detections_to_payload(result)

        except (KeyError, ValueError, json.JSONDecodeError, binascii.Error) as exc:
            LOGGER.warning("Bad frame payload: %s", exc)
            response = keep_alive_payload("bad_frame_payload")
        except cv2.error as exc:
            LOGGER.warning("OpenCV frame processing error: %s", exc)
            response = keep_alive_payload("opencv_error")
        except RuntimeError as exc:
            # Runtime errors from model/inference should not break bridge availability.
            LOGGER.warning("Model runtime error: %s", exc)
            response = keep_alive_payload("model_runtime_error")
        except Exception as exc:  # Defensive boundary: keep server alive.
            LOGGER.exception("Unhandled server error: %s", exc)
            response = keep_alive_payload("server_error")

        try:
            socket.send_json(response)
        except Exception as exc:
            # If reply fails, keep server loop alive for next request.
            LOGGER.error("Failed sending response: %s", exc)


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(description="Distributed laptop YOLO server")
    parser.add_argument("--host", default="*", help="Bind host (default: *)")
    parser.add_argument("--port", type=int, default=5555, help="Bind port")
    parser.add_argument(
        "--model-path",
        default=config.yolo.model_path,
        help="Path to ONNX model",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    bind_addr = f"tcp://{args.host}:{args.port}"
    run_server(bind_addr=bind_addr, model_path=args.model_path)


if __name__ == "__main__":
    main()
