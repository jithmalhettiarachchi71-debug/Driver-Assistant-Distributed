"""
YOLO Object Detection Module.

Provides YOLOv11s-based detection for:
- Traffic lights
- Pedestrians  
- Vehicles
- Stop signs
- Animals (road hazards)
"""

from src.detection.result import Detection, DetectionResult, DetectionLabel
from src.detection.detector import YOLODetector

__all__ = [
    "Detection",
    "DetectionResult", 
    "DetectionLabel",
    "YOLODetector",
]
