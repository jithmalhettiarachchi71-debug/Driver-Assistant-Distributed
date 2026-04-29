"""
Abstract camera adapter interface.

Defines the contract that all camera adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
from dataclasses import dataclass

from src.capture.frame import Frame, FrameSource


@dataclass
class CaptureConfig:
    """Configuration for frame capture."""
    resolution: Tuple[int, int] = (640, 480)  # (width, height)
    target_fps: int = 15
    timeout_ms: int = 100
    source: FrameSource = FrameSource.WEBCAM
    video_path: Optional[str] = None
    camera_index: int = 0
    ip_url: Optional[str] = None  # IP camera stream URL (MJPEG/RTSP)


class CameraAdapter(ABC):
    """
    Abstract base class for camera adapters.
    
    All platform-specific camera implementations must inherit from this class
    and implement the abstract methods.
    """
    
    def __init__(self, config: CaptureConfig):
        """
        Initialize the camera adapter.
        
        Args:
            config: Capture configuration
        """
        self._config = config
        self._frame_count = 0
        self._is_initialized = False
    
    @property
    def config(self) -> CaptureConfig:
        """Get the capture configuration."""
        return self._config
    
    @property
    def frame_count(self) -> int:
        """Get the number of frames captured."""
        return self._frame_count
    
    @property
    def is_initialized(self) -> bool:
        """Check if the adapter is initialized."""
        return self._is_initialized
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the camera/video source.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def capture(self) -> Optional[Frame]:
        """
        Capture a single frame.
        
        Returns:
            Frame object if successful, None if capture failed
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Check if the camera is functioning properly.
        
        Returns:
            True if camera is healthy, False otherwise
        """
        pass
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the camera.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        self.release()
        return self.initialize()
    
    def _increment_frame_count(self) -> int:
        """Increment and return the frame count."""
        count = self._frame_count
        self._frame_count += 1
        return count
    
    def reset_frame_count(self) -> None:
        """Reset the frame counter to zero."""
        self._frame_count = 0
