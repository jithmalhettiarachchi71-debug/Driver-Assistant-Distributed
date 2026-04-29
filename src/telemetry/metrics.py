"""
Performance metrics data structures.

Defines metrics collected for each frame and system-level telemetry.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import platform


@dataclass
class FrameMetrics:
    """
    Metrics collected for a single frame processing cycle.
    
    All latency values are in milliseconds.
    """
    frame_seq: int = 0
    timestamp: float = 0.0
    
    # Capture metrics
    capture_latency_ms: float = 0.0
    
    # Lane detection metrics
    lane_latency_ms: float = 0.0
    lane_valid: bool = False
    lane_partial: bool = False
    
    # YOLO detection metrics
    yolo_latency_ms: Optional[float] = None
    yolo_skipped: bool = False
    detections_count: int = 0
    
    # Decision metrics
    decision_latency_ms: float = 0.0
    collision_risks: int = 0
    
    # Alert metrics
    alert_type: Optional[str] = None
    alert_latency_ms: Optional[float] = None
    
    # Frame drop tracking
    dropped_frames: int = 0
    
    # IP camera specific metrics
    ip_acquisition_latency_ms: Optional[float] = None
    ip_reconnect_count: Optional[int] = None
    ip_downtime_ms: Optional[float] = None
    
    # HIGH FIX: LiDAR sensor metrics for telemetry analysis
    lidar_distance_cm: Optional[float] = None
    lidar_strength: Optional[int] = None
    lidar_valid: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "frame_seq": self.frame_seq,
            "timestamp": self.timestamp,
            "capture_latency_ms": round(self.capture_latency_ms, 2),
            "lane_latency_ms": round(self.lane_latency_ms, 2),
            "lane_valid": self.lane_valid,
            "lane_partial": self.lane_partial,
            "yolo_latency_ms": round(self.yolo_latency_ms, 2) if self.yolo_latency_ms else None,
            "yolo_skipped": self.yolo_skipped,
            "detections_count": self.detections_count,
            "decision_latency_ms": round(self.decision_latency_ms, 2),
            "collision_risks": self.collision_risks,
            "alert_type": self.alert_type,
            "alert_latency_ms": round(self.alert_latency_ms, 2) if self.alert_latency_ms else None,
            "dropped_frames": self.dropped_frames,
        }
        
        # Include IP camera metrics only if present
        if self.ip_acquisition_latency_ms is not None:
            result["ip_acquisition_latency_ms"] = round(self.ip_acquisition_latency_ms, 2)
        if self.ip_reconnect_count is not None:
            result["ip_reconnect_count"] = self.ip_reconnect_count
        if self.ip_downtime_ms is not None:
            result["ip_downtime_ms"] = round(self.ip_downtime_ms, 2)
        
        # Include LiDAR metrics only if present
        if self.lidar_distance_cm is not None:
            result["lidar_distance_cm"] = round(self.lidar_distance_cm, 1)
        if self.lidar_strength is not None:
            result["lidar_strength"] = self.lidar_strength
        if self.lidar_valid is not None:
            result["lidar_valid"] = self.lidar_valid
        
        return result


@dataclass
class SystemMetrics:
    """
    System-level metrics (CPU, memory, temperature).
    
    Updated periodically, not per-frame.
    """
    cpu_temperature_c: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    memory_used_mb: Optional[float] = None
    
    _last_update: float = field(default=0.0, repr=False)
    _update_interval: float = field(default=5.0, repr=False)  # Update every 5 seconds
    
    def update_if_needed(self) -> None:
        """Update system metrics if enough time has passed."""
        current_time = time.monotonic()
        if current_time - self._last_update < self._update_interval:
            return
        
        self._last_update = current_time
        self._update_cpu_temp()
        self._update_cpu_usage()
        self._update_memory()
    
    def _update_cpu_temp(self) -> None:
        """Read CPU temperature (Raspberry Pi only)."""
        if platform.system() != "Linux":
            return
        
        try:
            # Raspberry Pi thermal zone
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_millideg = int(f.read().strip())
                self.cpu_temperature_c = temp_millideg / 1000.0
        except (FileNotFoundError, IOError, ValueError):
            self.cpu_temperature_c = None
    
    def _update_cpu_usage(self) -> None:
        """Read CPU usage percentage."""
        try:
            import psutil
            self.cpu_usage_percent = psutil.cpu_percent(interval=None)
        except ImportError:
            self.cpu_usage_percent = None
    
    def _update_memory(self) -> None:
        """Read memory usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            self.memory_used_mb = mem.used / (1024 * 1024)
        except ImportError:
            self.memory_used_mb = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "cpu_temperature_c": round(self.cpu_temperature_c, 1) if self.cpu_temperature_c else None,
            "cpu_usage_percent": round(self.cpu_usage_percent, 1) if self.cpu_usage_percent else None,
            "memory_used_mb": round(self.memory_used_mb, 1) if self.memory_used_mb else None,
        }


class FPSCounter:
    """
    Calculates frames per second with rolling window.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            window_size: Number of frames to average over
        """
        self._window_size = window_size
        self._timestamps: list = []
    
    def tick(self) -> float:
        """
        Record a frame and return current FPS.
        
        Returns:
            Current FPS based on rolling window
        """
        current_time = time.monotonic()
        self._timestamps.append(current_time)
        
        # Keep only recent timestamps
        if len(self._timestamps) > self._window_size:
            self._timestamps = self._timestamps[-self._window_size:]
        
        return self.fps
    
    @property
    def fps(self) -> float:
        """Calculate current FPS."""
        if len(self._timestamps) < 2:
            return 0.0
        
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        
        return (len(self._timestamps) - 1) / elapsed
    
    def reset(self) -> None:
        """Reset the FPS counter."""
        self._timestamps.clear()


class LatencyTracker:
    """
    Tracks latency statistics for a single operation.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize latency tracker.
        
        Args:
            window_size: Number of samples to keep for statistics
        """
        self._window_size = window_size
        self._samples: list = []
    
    def record(self, latency_ms: float) -> None:
        """Record a latency sample."""
        self._samples.append(latency_ms)
        if len(self._samples) > self._window_size:
            self._samples = self._samples[-self._window_size:]
    
    @property
    def mean(self) -> float:
        """Get mean latency."""
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)
    
    @property
    def max(self) -> float:
        """Get maximum latency."""
        if not self._samples:
            return 0.0
        return max(self._samples)
    
    @property
    def min(self) -> float:
        """Get minimum latency."""
        if not self._samples:
            return 0.0
        return min(self._samples)
    
    @property
    def p95(self) -> float:
        """Get 95th percentile latency."""
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        index = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(index, len(sorted_samples) - 1)]
    
    def reset(self) -> None:
        """Reset the tracker."""
        self._samples.clear()
    
    def to_dict(self) -> Dict[str, float]:
        """Get statistics as dictionary."""
        return {
            "mean_ms": round(self.mean, 2),
            "max_ms": round(self.max, 2),
            "min_ms": round(self.min, 2),
            "p95_ms": round(self.p95, 2),
        }
