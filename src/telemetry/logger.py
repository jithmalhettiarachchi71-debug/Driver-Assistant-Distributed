"""
JSON Lines telemetry logger.

Provides append-only logging of telemetry records for offline analysis.
"""

import json
import logging
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from queue import Queue, Full, Empty

from .metrics import FrameMetrics, SystemMetrics

logger = logging.getLogger(__name__)


@dataclass
class TelemetryRecord:
    """
    Complete telemetry record for a single frame.
    
    Combines frame metrics and system metrics into one record.
    """
    # ISO 8601 timestamp
    timestamp: str
    
    # Frame identification
    frame_seq: int
    
    # Performance metrics
    capture_fps: float
    capture_latency_ms: float
    lane_latency_ms: float
    yolo_latency_ms: Optional[float]
    yolo_skipped: bool
    decision_latency_ms: float
    
    # Alert metrics
    alert_type: Optional[str]
    alert_latency_ms: Optional[float]
    
    # System metrics
    cpu_temperature_c: Optional[float]
    
    # Status
    dropped_frames: int
    lane_valid: bool
    detections_count: int
    collision_risks: int
    
    # IP camera specific metrics (optional)
    ip_acquisition_latency_ms: Optional[float] = None
    ip_reconnect_count: Optional[int] = None
    ip_downtime_ms: Optional[float] = None
    
    # HIGH FIX: LiDAR sensor metrics (optional)
    lidar_distance_cm: Optional[float] = None
    lidar_strength: Optional[int] = None
    lidar_valid: Optional[bool] = None
    
    def to_json(self) -> str:
        """Serialize to JSON string, excluding None optional fields for sensor metrics."""
        data = asdict(self)
        # Remove None IP camera fields to avoid cluttering logs for non-IP sources
        for key in ["ip_acquisition_latency_ms", "ip_reconnect_count", "ip_downtime_ms"]:
            if data.get(key) is None:
                del data[key]
        # Remove None LiDAR fields to avoid cluttering logs when LiDAR not active
        for key in ["lidar_distance_cm", "lidar_strength", "lidar_valid"]:
            if data.get(key) is None:
                del data[key]
        return json.dumps(data, separators=(',', ':'))
    
    @classmethod
    def from_metrics(
        cls,
        frame_metrics: FrameMetrics,
        system_metrics: SystemMetrics,
        capture_fps: float,
    ) -> "TelemetryRecord":
        """Create record from metrics objects."""
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            frame_seq=frame_metrics.frame_seq,
            capture_fps=round(capture_fps, 1),
            capture_latency_ms=round(frame_metrics.capture_latency_ms, 1),
            lane_latency_ms=round(frame_metrics.lane_latency_ms, 1),
            yolo_latency_ms=round(frame_metrics.yolo_latency_ms, 1) if frame_metrics.yolo_latency_ms else None,
            yolo_skipped=frame_metrics.yolo_skipped,
            decision_latency_ms=round(frame_metrics.decision_latency_ms, 1),
            alert_type=frame_metrics.alert_type,
            alert_latency_ms=round(frame_metrics.alert_latency_ms, 1) if frame_metrics.alert_latency_ms else None,
            cpu_temperature_c=round(system_metrics.cpu_temperature_c, 1) if system_metrics.cpu_temperature_c else None,
            dropped_frames=frame_metrics.dropped_frames,
            lane_valid=frame_metrics.lane_valid,
            detections_count=frame_metrics.detections_count,
            collision_risks=frame_metrics.collision_risks,
            ip_acquisition_latency_ms=round(frame_metrics.ip_acquisition_latency_ms, 1) if frame_metrics.ip_acquisition_latency_ms else None,
            ip_reconnect_count=frame_metrics.ip_reconnect_count,
            ip_downtime_ms=round(frame_metrics.ip_downtime_ms, 1) if frame_metrics.ip_downtime_ms else None,
            lidar_distance_cm=round(frame_metrics.lidar_distance_cm, 1) if frame_metrics.lidar_distance_cm else None,
            lidar_strength=frame_metrics.lidar_strength,
            lidar_valid=frame_metrics.lidar_valid,
        )


class TelemetryLogger:
    """
    Append-only JSON Lines logger for telemetry data.
    
    Features:
    - Non-blocking async writes via background thread
    - Configurable flush interval
    - Memory buffering when write fails
    - Log rotation at configurable size
    
    Usage:
        logger = TelemetryLogger("telemetry.jsonl")
        logger.start()
        
        # In frame loop:
        logger.log(telemetry_record)
        
        # On shutdown:
        logger.stop()
    """
    
    DEFAULT_FLUSH_INTERVAL = 1.0  # seconds
    DEFAULT_MAX_BUFFER = 1000  # records
    DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    
    def __init__(
        self,
        log_file: str,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_buffer: int = DEFAULT_MAX_BUFFER,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    ):
        """
        Initialize telemetry logger.
        
        Args:
            log_file: Path to output .jsonl file
            flush_interval: Seconds between flushes
            max_buffer: Maximum records to buffer in memory
            max_file_size: Maximum file size before rotation
        """
        self._log_file = Path(log_file)
        self._flush_interval = flush_interval
        self._max_buffer = max_buffer
        self._max_file_size = max_file_size
        
        # Thread-safe queue for async writes
        self._queue: Queue[TelemetryRecord] = Queue(maxsize=max_buffer)
        
        # Background writer thread
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # File handle
        self._file_handle = None
        
        # Statistics
        self._records_written = 0
        self._records_dropped = 0
    
    def start(self) -> None:
        """Start the background writer thread."""
        if self._writer_thread is not None and self._writer_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="TelemetryWriter",
            daemon=True,
        )
        self._writer_thread.start()
        logger.info(f"Telemetry logger started: {self._log_file}")
    
    def stop(self) -> None:
        """Stop the background writer and flush remaining records."""
        self._stop_event.set()
        
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
            self._writer_thread = None
        
        # Flush any remaining records
        self._flush_remaining()
        
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        
        logger.info(
            f"Telemetry logger stopped. "
            f"Written: {self._records_written}, Dropped: {self._records_dropped}"
        )
    
    def log(self, record: TelemetryRecord) -> bool:
        """
        Log a telemetry record (non-blocking).
        
        Args:
            record: Telemetry record to log
            
        Returns:
            True if record was queued, False if dropped
        """
        try:
            self._queue.put_nowait(record)
            return True
        except Full:
            self._records_dropped += 1
            return False
    
    def log_frame(
        self,
        frame_metrics: FrameMetrics,
        system_metrics: SystemMetrics,
        capture_fps: float,
    ) -> bool:
        """
        Convenience method to log frame metrics.
        
        Args:
            frame_metrics: Frame-level metrics
            system_metrics: System-level metrics
            capture_fps: Current capture FPS
            
        Returns:
            True if record was queued
        """
        record = TelemetryRecord.from_metrics(
            frame_metrics, system_metrics, capture_fps
        )
        return self.log(record)
    
    def _writer_loop(self) -> None:
        """Background thread loop for writing records."""
        buffer: List[str] = []
        last_flush = time.monotonic()
        
        while not self._stop_event.is_set():
            try:
                # Wait for record with timeout
                record = self._queue.get(timeout=0.1)
                buffer.append(record.to_json())
                
                # Check if we should flush
                current_time = time.monotonic()
                should_flush = (
                    current_time - last_flush >= self._flush_interval or
                    len(buffer) >= 100  # Flush every 100 records
                )
                
                if should_flush and buffer:
                    self._write_buffer(buffer)
                    buffer.clear()
                    last_flush = current_time
                    
            except Empty:
                # Timeout - check if we should flush
                if buffer:
                    current_time = time.monotonic()
                    if current_time - last_flush >= self._flush_interval:
                        self._write_buffer(buffer)
                        buffer.clear()
                        last_flush = current_time
        
        # Final flush
        if buffer:
            self._write_buffer(buffer)
    
    def _write_buffer(self, buffer: List[str]) -> None:
        """Write buffered records to file."""
        try:
            # Check file rotation
            self._check_rotation()
            
            # Open file if needed
            if self._file_handle is None:
                self._file_handle = open(self._log_file, "a", encoding="utf-8")
            
            # Write records
            for line in buffer:
                self._file_handle.write(line + "\n")
            
            self._file_handle.flush()
            self._records_written += len(buffer)
            
        except IOError as e:
            logger.error(f"Telemetry write error: {e}")
            self._records_dropped += len(buffer)
    
    def _check_rotation(self) -> None:
        """Check if log file needs rotation."""
        if not self._log_file.exists():
            return
        
        if self._log_file.stat().st_size < self._max_file_size:
            return
        
        # Close current file
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        
        # Rotate: rename current file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = self._log_file.with_suffix(f".{timestamp}.jsonl")
        
        try:
            self._log_file.rename(rotated_name)
            logger.info(f"Rotated telemetry log to: {rotated_name}")
        except OSError as e:
            logger.error(f"Log rotation failed: {e}")
    
    def _flush_remaining(self) -> None:
        """Flush any remaining records in queue."""
        buffer: List[str] = []
        
        while True:
            try:
                record = self._queue.get_nowait()
                buffer.append(record.to_json())
            except Empty:
                break
        
        if buffer:
            self._write_buffer(buffer)
    
    @property
    def records_written(self) -> int:
        """Get total records written."""
        return self._records_written
    
    @property
    def records_dropped(self) -> int:
        """Get total records dropped."""
        return self._records_dropped
    
    def __enter__(self) -> "TelemetryLogger":
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
