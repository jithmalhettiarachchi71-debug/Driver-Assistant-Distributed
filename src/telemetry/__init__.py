"""
Telemetry and logging module for the Vehicle Safety Alert System.

Provides JSON Lines logging for performance metrics and system state.
"""

from .logger import TelemetryLogger, TelemetryRecord
from .metrics import FrameMetrics, SystemMetrics

__all__ = [
    "TelemetryLogger",
    "TelemetryRecord",
    "FrameMetrics",
    "SystemMetrics",
]
