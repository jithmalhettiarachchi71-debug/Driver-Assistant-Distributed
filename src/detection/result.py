"""
Detection result data structures.

Defines the output format for YOLO object detection.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class DetectionLabel(Enum):
    """Valid detection labels for the safety system."""
    # Traffic lights by color
    TRAFFIC_LIGHT_RED = "traffic_light_red"
    TRAFFIC_LIGHT_YELLOW = "traffic_light_yellow"
    TRAFFIC_LIGHT_GREEN = "traffic_light_green"
    TRAFFIC_LIGHT = "traffic_light"  # Generic (no color info)
    
    # Other signals
    STOP_SIGN = "stop_sign"
    
    # Obstacles
    PEDESTRIAN = "pedestrian"
    VEHICLE = "vehicle"
    BIKER = "biker"  # Cyclists/motorcyclists
    
    @classmethod
    def from_string(cls, label: str) -> Optional["DetectionLabel"]:
        """Convert string label to enum."""
        label_map = {
            "traffic_light_red": cls.TRAFFIC_LIGHT_RED,
            "traffic_light_yellow": cls.TRAFFIC_LIGHT_YELLOW,
            "traffic_light_green": cls.TRAFFIC_LIGHT_GREEN,
            "traffic_light": cls.TRAFFIC_LIGHT,
            "trafficlight-red": cls.TRAFFIC_LIGHT_RED,
            "trafficlight-yellow": cls.TRAFFIC_LIGHT_YELLOW,
            "trafficlight-green": cls.TRAFFIC_LIGHT_GREEN,
            "trafficlight": cls.TRAFFIC_LIGHT,
            "stop_sign": cls.STOP_SIGN,
            "pedestrian": cls.PEDESTRIAN,
            "vehicle": cls.VEHICLE,
            "biker": cls.BIKER,
            "car": cls.VEHICLE,
            "truck": cls.VEHICLE,
        }
        return label_map.get(label.lower().replace(" ", ""))
    
    def is_traffic_signal(self) -> bool:
        """Check if this detection is a traffic signal/sign."""
        return self in (
            DetectionLabel.TRAFFIC_LIGHT,
            DetectionLabel.TRAFFIC_LIGHT_RED,
            DetectionLabel.TRAFFIC_LIGHT_YELLOW,
            DetectionLabel.TRAFFIC_LIGHT_GREEN,
            DetectionLabel.STOP_SIGN,
        )
    
    def is_traffic_light(self) -> bool:
        """Check if this is any type of traffic light."""
        return self in (
            DetectionLabel.TRAFFIC_LIGHT,
            DetectionLabel.TRAFFIC_LIGHT_RED,
            DetectionLabel.TRAFFIC_LIGHT_YELLOW,
            DetectionLabel.TRAFFIC_LIGHT_GREEN,
        )
    
    def is_obstacle(self) -> bool:
        """Check if this detection is a collision obstacle."""
        return self in (
            DetectionLabel.PEDESTRIAN,
            DetectionLabel.VEHICLE,
            DetectionLabel.BIKER,
        )


@dataclass
class Detection:
    """
    Represents a single object detection.
    
    Attributes:
        label: Detection class label
        confidence: Detection confidence [0.0, 1.0]
        bbox: Bounding box as (x_min, y_min, x_max, y_max)
        class_name: Original COCO class name (for debugging)
        timestamp: Monotonic timestamp of detection
    """
    label: DetectionLabel
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    timestamp: float
    class_name: str = ""  # Original COCO class name
    
    @property
    def x_min(self) -> float:
        return self.bbox[0]
    
    @property
    def y_min(self) -> float:
        return self.bbox[1]
    
    @property
    def x_max(self) -> float:
        return self.bbox[2]
    
    @property
    def y_max(self) -> float:
        return self.bbox[3]
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def center(self) -> Tuple[float, float]:
        """Return center point of bounding box."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
        )
    
    @property
    def area(self) -> float:
        """Return area of bounding box in pixels."""
        return self.width * self.height


@dataclass
class DetectionResult:
    """
    Container for detection results from a single frame.
    
    Attributes:
        detections: List of Detection objects
        timestamp: Monotonic timestamp of inference
        latency_ms: Inference latency in milliseconds
        from_cache: Whether result was retrieved from cache
        stale: Whether cached result exceeds TTL
    """
    detections: List[Detection]
    timestamp: float
    latency_ms: float = 0.0
    from_cache: bool = False
    stale: bool = False
    
    @classmethod
    def empty(cls, timestamp: float, latency_ms: float = 0.0) -> "DetectionResult":
        """Create an empty result."""
        return cls(
            detections=[],
            timestamp=timestamp,
            latency_ms=latency_ms,
            from_cache=False,
            stale=False,
        )
    
    @classmethod
    def from_cache(
        cls,
        cached: "DetectionResult",
        current_time: float,
        ttl_ms: float,
    ) -> "DetectionResult":
        """Create a result from cached data, checking staleness."""
        age_ms = (current_time - cached.timestamp) * 1000
        return cls(
            detections=cached.detections,
            timestamp=cached.timestamp,
            latency_ms=0.0,
            from_cache=True,
            stale=age_ms > ttl_ms,
        )
    
    def get_by_label(self, label: DetectionLabel) -> List[Detection]:
        """Get all detections of a specific label."""
        return [d for d in self.detections if d.label == label]
    
    def get_obstacles(self) -> List[Detection]:
        """Get all obstacle detections (pedestrians, vehicles, animals)."""
        return [d for d in self.detections if d.label.is_obstacle()]
    
    def get_traffic_signals(self) -> List[Detection]:
        """Get all traffic signal detections."""
        return [d for d in self.detections if d.label.is_traffic_signal()]
    
    @property
    def count(self) -> int:
        """Total number of detections."""
        return len(self.detections)
    
    @property
    def has_obstacles(self) -> bool:
        """Check if any obstacles were detected."""
        return any(d.label.is_obstacle() for d in self.detections)
    
    @property
    def has_traffic_light(self) -> bool:
        """Check if traffic light was detected."""
        return any(d.label == DetectionLabel.TRAFFIC_LIGHT for d in self.detections)
