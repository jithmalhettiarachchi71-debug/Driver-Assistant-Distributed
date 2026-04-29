"""
IP camera adapter for network stream capture.

Supports MJPEG and RTSP streams via OpenCV VideoCapture with
auto-reconnection, latency minimization via threaded grab, and telemetry.
"""

import time
import logging
import threading
from typing import Optional, Tuple
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.capture.adapter import CameraAdapter, CaptureConfig
from src.capture.frame import Frame, FrameSource

logger = logging.getLogger(__name__)


@dataclass
class IPCameraMetrics:
    """
    Telemetry metrics specific to IP camera operation.
    
    Thread-safe access via lock.
    """
    acquisition_latency_ms: float = 0.0
    reconnect_count: int = 0
    total_downtime_ms: float = 0.0
    last_frame_time: float = 0.0
    connection_lost_time: Optional[float] = None
    
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def record_frame(self, latency_ms: float) -> None:
        """Record successful frame acquisition."""
        with self._lock:
            self.acquisition_latency_ms = latency_ms
            self.last_frame_time = time.monotonic()
            
            # If we were disconnected, add downtime
            if self.connection_lost_time is not None:
                downtime = (time.monotonic() - self.connection_lost_time) * 1000
                self.total_downtime_ms += downtime
                self.connection_lost_time = None
    
    def record_disconnect(self) -> None:
        """Record connection loss."""
        with self._lock:
            if self.connection_lost_time is None:
                self.connection_lost_time = time.monotonic()
    
    def record_reconnect(self) -> None:
        """Record reconnection attempt."""
        with self._lock:
            self.reconnect_count += 1
    
    def get_metrics(self) -> Tuple[float, int, float]:
        """
        Get current metrics (thread-safe).
        
        Returns:
            Tuple of (acquisition_latency_ms, reconnect_count, total_downtime_ms)
        """
        with self._lock:
            return (
                self.acquisition_latency_ms,
                self.reconnect_count,
                self.total_downtime_ms,
            )
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.acquisition_latency_ms = 0.0
            self.reconnect_count = 0
            self.total_downtime_ms = 0.0
            self.last_frame_time = 0.0
            self.connection_lost_time = None


class IPCameraAdapter(CameraAdapter):
    """
    Camera adapter for IP camera streams (MJPEG, RTSP).
    
    Features:
    - Automatic reconnection on stream failure
    - Threaded frame grabbing for minimal latency
    - IP-specific telemetry metrics
    - Configurable timeout and retry behavior
    
    Supported URL formats:
    - MJPEG: http://192.168.1.100:8080/video
    - RTSP: rtsp://user:pass@192.168.1.100:554/stream
    """
    
    # Reconnection settings
    MAX_RECONNECT_ATTEMPTS = 5
    RECONNECT_DELAY_MS = 1000
    HEALTH_CHECK_TIMEOUT_MS = 5000
    
    def __init__(self, config: CaptureConfig):
        """
        Initialize the IP camera adapter.
        
        Args:
            config: Capture configuration with ip_url set
        """
        super().__init__(config)
        self._cap: Optional[cv2.VideoCapture] = None
        self._url: str = config.ip_url or ""
        self._metrics = IPCameraMetrics()
        self._consecutive_failures = 0
        self._last_successful_read = 0.0
        
        # Threaded frame grabbing for low latency
        self._grab_thread: Optional[threading.Thread] = None
        self._grab_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_time: float = 0.0
        self._stop_grab_thread = threading.Event()
    
    @property
    def metrics(self) -> IPCameraMetrics:
        """Get IP camera metrics for telemetry."""
        return self._metrics
    
    def initialize(self) -> bool:
        """
        Initialize connection to IP camera stream.
        
        Returns:
            True if connection successful
        """
        if not self._url:
            logger.error("IP camera URL not configured")
            return False
        
        try:
            logger.info(f"Connecting to IP camera: {self._mask_url(self._url)}")
            
            # Create VideoCapture with URL
            self._cap = cv2.VideoCapture(self._url)
            
            if not self._cap.isOpened():
                logger.error(f"Failed to open IP camera stream: {self._mask_url(self._url)}")
                return False
            
            # Set buffer size to 1 to minimize latency
            # This drops old frames and always gets the latest
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set timeout for read operations
            timeout_sec = self._config.timeout_ms / 1000.0
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._config.timeout_ms)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._config.timeout_ms)
            
            # Try to read a test frame to verify connection
            ret, test_frame = self._cap.read()
            if not ret or test_frame is None:
                logger.error("IP camera connected but no frames received")
                self._cap.release()
                self._cap = None
                return False
            
            # Log actual stream properties
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(
                f"IP camera connected: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
            )
            
            # Store initial frame and start grab thread
            with self._grab_lock:
                self._latest_frame = test_frame
                self._latest_frame_time = time.monotonic()
            
            # Start background thread for continuous frame grabbing
            self._stop_grab_thread.clear()
            self._grab_thread = threading.Thread(
                target=self._frame_grab_loop,
                name="IPCameraGrab",
                daemon=True,
            )
            self._grab_thread.start()
            logger.info("IP camera grab thread started for low-latency capture")
            
            self._is_initialized = True
            self._consecutive_failures = 0
            self._last_successful_read = time.monotonic()
            self.reset_frame_count()
            self._metrics.reset()
            
            return True
            
        except Exception as e:
            logger.error(f"IP camera initialization error: {e}")
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            return False
    
    def _frame_grab_loop(self) -> None:
        """
        Background thread that continuously grabs frames.
        
        This prevents OpenCV's internal buffer from filling up,
        ensuring we always have the most recent frame available.
        Detects stream failures and marks adapter for reconnection.
        """
        logger.debug("Frame grab loop started")
        consecutive_failures = 0
        max_failures = 30  # ~300ms of failures before marking dead
        
        while not self._stop_grab_thread.is_set():
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.01)
                continue
            
            try:
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    with self._grab_lock:
                        self._latest_frame = frame
                        self._latest_frame_time = time.monotonic()
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        # Stream is dead - mark for reconnection
                        logger.warning("IP camera stream appears dead, marking for reconnection")
                        self._is_initialized = False
                        self._metrics.record_disconnect()
                        consecutive_failures = 0
                        time.sleep(0.5)  # Wait before next attempt
                    else:
                        time.sleep(0.01)
            except Exception as e:
                logger.debug(f"Grab thread read error: {e}")
                consecutive_failures += 1
                time.sleep(0.01)
        
        logger.debug("Frame grab loop stopped")
    
    def capture(self) -> Optional[Frame]:
        """
        Capture the latest frame from the IP camera stream.
        
        Uses the threaded grab for minimal latency - always returns
        the most recently captured frame. Attempts reconnection if
        the stream has been down.
        
        HIGH FIX: Reconnection is now non-blocking. If stream is down,
        we return None immediately and let the main loop continue processing.
        Reconnection attempts are rate-limited to avoid blocking.
        
        Returns:
            Frame object if successful, None if capture failed
        """
        if not self._is_initialized or self._cap is None:
            # HIGH FIX: Non-blocking reconnection - only attempt if enough time has passed
            now = time.monotonic()
            if not hasattr(self, '_last_reconnect_attempt'):
                self._last_reconnect_attempt = 0.0
            
            # Rate limit reconnection attempts to once per second
            if now - self._last_reconnect_attempt < 1.0:
                return None  # Return immediately, don't block
            
            self._last_reconnect_attempt = now
            logger.warning("IP camera not initialized, attempting reconnection...")
            self._metrics.record_reconnect()
            
            # Attempt quick reconnection (single attempt, non-blocking)
            if not self._attempt_quick_reconnect():
                return None
            logger.info("IP camera reconnected successfully")
        
        try:
            capture_start = time.monotonic()
            
            # Get latest frame from grab thread
            with self._grab_lock:
                frame_data = self._latest_frame
                frame_time = self._latest_frame_time
            
            # Check if we have a frame and it's recent enough
            if frame_data is None:
                return self._handle_capture_failure()
            
            frame_age_ms = (capture_start - frame_time) * 1000
            if frame_age_ms > self._config.timeout_ms:
                logger.warning(f"IP camera frame too old: {frame_age_ms:.0f}ms")
                return self._handle_capture_failure()
            
            timestamp = time.monotonic()
            
            # Calculate acquisition latency (how old the frame is)
            latency_ms = frame_age_ms
            self._metrics.record_frame(latency_ms)
            
            # Reset failure counter on success
            self._consecutive_failures = 0
            self._last_successful_read = timestamp
            
            # Make a copy to avoid thread conflicts
            frame_data = frame_data.copy()
            
            # Ensure correct format (BGR, uint8)
            if frame_data.dtype != np.uint8:
                frame_data = frame_data.astype(np.uint8)
            
            # Resize if needed
            target_h, target_w = self._config.resolution[1], self._config.resolution[0]
            if frame_data.shape[0] != target_h or frame_data.shape[1] != target_w:
                frame_data = cv2.resize(frame_data, (target_w, target_h))
            
            frame = Frame(
                data=frame_data,
                timestamp=timestamp,
                sequence=self._increment_frame_count(),
                source=FrameSource.IP_CAMERA,
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"IP camera capture error: {e}")
            return self._handle_capture_failure()
    
    def _handle_capture_failure(self) -> Optional[Frame]:
        """
        Handle a frame capture failure.
        
        Attempts reconnection if failure threshold exceeded.
        
        Returns:
            None (no frame available on failure)
        """
        self._consecutive_failures += 1
        self._metrics.record_disconnect()
        
        logger.warning(
            f"IP camera frame capture failed "
            f"(consecutive failures: {self._consecutive_failures})"
        )
        
        # Attempt reconnection after threshold
        if self._consecutive_failures >= 3:
            logger.info("Attempting IP camera reconnection...")
            self._metrics.record_reconnect()
            
            if self._attempt_reconnect():
                logger.info("IP camera reconnected successfully")
                self._consecutive_failures = 0
            else:
                logger.error("IP camera reconnection failed")
        
        return None
    
    def _attempt_quick_reconnect(self) -> bool:
        """
        Attempt a single quick reconnection to the IP camera.
        
        HIGH FIX: This is a non-blocking single-attempt reconnection
        used by the capture() method to avoid blocking the main loop.
        
        Returns:
            True if reconnection successful
        """
        # Stop grab thread first (quick timeout)
        if self._grab_thread is not None:
            self._stop_grab_thread.set()
            self._grab_thread.join(timeout=0.5)  # Short timeout
            if self._grab_thread.is_alive():
                logger.warning("Grab thread still alive during quick reconnect")
            self._grab_thread = None
        
        # Release existing connection
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        
        self._is_initialized = False
        
        # Single quick attempt (no retries, no delays)
        try:
            self._cap = cv2.VideoCapture(self._url)
            
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                ret, test_frame = self._cap.read()
                if ret and test_frame is not None:
                    with self._grab_lock:
                        self._latest_frame = test_frame
                        self._latest_frame_time = time.monotonic()
                    
                    self._stop_grab_thread.clear()
                    self._grab_thread = threading.Thread(
                        target=self._frame_grab_loop,
                        name="IPCameraGrab",
                        daemon=True,
                    )
                    self._grab_thread.start()
                    
                    self._is_initialized = True
                    return True
                
                self._cap.release()
                self._cap = None
                
        except Exception as e:
            logger.debug(f"Quick reconnection failed: {e}")
        
        return False
    
    def _attempt_reconnect(self) -> bool:
        """
        Attempt to reconnect to the IP camera with retries.
        
        This is the full reconnection method with multiple attempts.
        Used by explicit reconnect() calls, not by capture().
        
        Returns:
            True if reconnection successful
        """
        # Stop grab thread first
        if self._grab_thread is not None:
            self._stop_grab_thread.set()
            self._grab_thread.join(timeout=2.0)
            if self._grab_thread.is_alive():
                logger.warning("Grab thread still alive after join timeout")
            self._grab_thread = None
        
        # Release existing connection
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        
        self._is_initialized = False
        
        # Wait before reconnecting
        time.sleep(self.RECONNECT_DELAY_MS / 1000.0)
        
        # Try to reinitialize
        for attempt in range(self.MAX_RECONNECT_ATTEMPTS):
            logger.info(f"Reconnection attempt {attempt + 1}/{self.MAX_RECONNECT_ATTEMPTS}")
            
            try:
                self._cap = cv2.VideoCapture(self._url)
                
                if self._cap.isOpened():
                    # Set buffer size for low latency
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test read
                    ret, test_frame = self._cap.read()
                    if ret and test_frame is not None:
                        # Restart grab thread
                        with self._grab_lock:
                            self._latest_frame = test_frame
                            self._latest_frame_time = time.monotonic()
                        
                        self._stop_grab_thread.clear()
                        self._grab_thread = threading.Thread(
                            target=self._frame_grab_loop,
                            name="IPCameraGrab",
                            daemon=True,
                        )
                        self._grab_thread.start()
                        
                        self._is_initialized = True
                        return True
                    
                    self._cap.release()
                    self._cap = None
                    
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
            
            if attempt < self.MAX_RECONNECT_ATTEMPTS - 1:
                time.sleep(self.RECONNECT_DELAY_MS / 1000.0)
        
        return False
    
    def release(self) -> None:
        """Release IP camera connection and stop grab thread."""
        # Stop the grab thread first
        if self._grab_thread is not None:
            self._stop_grab_thread.set()
            self._grab_thread.join(timeout=2.0)
            # HIGH FIX: Check if thread is still alive after join timeout
            if self._grab_thread.is_alive():
                logger.warning("IP camera grab thread did not stop within timeout")
            self._grab_thread = None
            logger.debug("IP camera grab thread stopped")
        
        # Release VideoCapture
        if self._cap is not None:
            try:
                self._cap.release()
                logger.info("IP camera connection released")
            except Exception as e:
                logger.warning(f"Error releasing IP camera: {e}")
            finally:
                self._cap = None
        
        # Clear latest frame
        with self._grab_lock:
            self._latest_frame = None
            self._latest_frame_time = 0.0
        
        self._is_initialized = False
    
    def is_healthy(self) -> bool:
        """
        Check if IP camera connection is healthy.
        
        Returns:
            True if connection is active and receiving frames
        """
        if not self._is_initialized or self._cap is None:
            return False
        
        if not self._cap.isOpened():
            return False
        
        # Check if we've received frames recently
        time_since_last = (time.monotonic() - self._last_successful_read) * 1000
        if time_since_last > self.HEALTH_CHECK_TIMEOUT_MS:
            logger.warning(
                f"IP camera unhealthy: no frames for {time_since_last:.0f}ms"
            )
            return False
        
        # Check consecutive failure count
        if self._consecutive_failures >= 3:
            return False
        
        return True
    
    def reconnect(self) -> bool:
        """
        Force reconnection to IP camera.
        
        Returns:
            True if reconnection successful
        """
        logger.info("Forced IP camera reconnection requested")
        self._metrics.record_reconnect()
        self.release()
        return self.initialize()
    
    @staticmethod
    def _mask_url(url: str) -> str:
        """
        Mask credentials in URL for logging.
        
        Args:
            url: Full URL potentially containing credentials
            
        Returns:
            URL with password masked
        """
        if "@" in url and "://" in url:
            # URL format: scheme://user:pass@host:port/path
            try:
                scheme_end = url.index("://") + 3
                at_pos = url.index("@")
                
                # Find colon between user and pass
                cred_part = url[scheme_end:at_pos]
                if ":" in cred_part:
                    colon_pos = cred_part.index(":")
                    user = cred_part[:colon_pos]
                    masked = url[:scheme_end] + user + ":***" + url[at_pos:]
                    return masked
            except (ValueError, IndexError):
                pass
        
        return url
    
    def get_ip_metrics(self) -> Tuple[float, int, float]:
        """
        Get IP-specific metrics for telemetry integration.
        
        Returns:
            Tuple of (acquisition_latency_ms, reconnect_count, total_downtime_ms)
        """
        return self._metrics.get_metrics()
