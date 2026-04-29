"""
Frame data structure for the Vehicle Safety Alert System.

Defines the standard frame format used throughout the pipeline.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class FrameSource(Enum):
    """Frame source type."""
    CSI = "csi"           # Raspberry Pi CSI camera
    WEBCAM = "webcam"     # USB webcam / integrated camera
    VIDEO_FILE = "video"  # Video file playback
    IP_CAMERA = "ip"      # IP camera stream (MJPEG/RTSP)


@dataclass
class Frame:
    """
    Represents a captured video frame with metadata.
    
    Attributes:
        data: BGR uint8 numpy array, shape (height, width, 3)
        timestamp: Monotonic time of capture in seconds
        sequence: Frame counter (0-indexed)
        source: The type of source this frame came from
    """
    data: np.ndarray
    timestamp: float
    sequence: int
    source: FrameSource
    
    @property
    def height(self) -> int:
        """Get frame height in pixels."""
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        """Get frame width in pixels."""
        return self.data.shape[1]
    
    @property
    def shape(self) -> tuple:
        """Get frame shape (height, width, channels)."""
        return self.data.shape
    
    def validate(self) -> bool:
        """
        Validate frame data integrity.
        
        Returns:
            True if frame is valid, False otherwise
        """
        # Check data type
        if self.data.dtype != np.uint8:
            return False
        
        # Check dimensions (must be 3-channel color image)
        if len(self.data.shape) != 3 or self.data.shape[2] != 3:
            return False
        
        # Check minimum size
        if self.data.shape[0] < 1 or self.data.shape[1] < 1:
            return False
        
        # Check timestamp is positive
        if self.timestamp <= 0:
            return False
        
        return True
    
    def copy(self) -> "Frame":
        """
        Create a deep copy of the frame.
        
        Returns:
            New Frame with copied data
        """
        return Frame(
            data=self.data.copy(),
            timestamp=self.timestamp,
            sequence=self.sequence,
            source=self.source,
        )
    
    def __repr__(self) -> str:
        return (
            f"Frame(shape={self.shape}, seq={self.sequence}, "
            f"source={self.source.value}, ts={self.timestamp:.3f})"
        )
