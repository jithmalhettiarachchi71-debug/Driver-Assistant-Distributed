"""Utility modules for the Vehicle Safety Alert System."""

from src.utils.platform import is_raspberry_pi, get_platform_name
from src.utils.timing import Timer, FrameRateEnforcer
from src.utils.geometry import (
    polygon_intersection,
    bbox_to_polygon,
    point_in_polygon,
    line_intersection,
)

__all__ = [
    "is_raspberry_pi",
    "get_platform_name", 
    "Timer",
    "FrameRateEnforcer",
    "polygon_intersection",
    "bbox_to_polygon",
    "point_in_polygon",
    "line_intersection",
]
