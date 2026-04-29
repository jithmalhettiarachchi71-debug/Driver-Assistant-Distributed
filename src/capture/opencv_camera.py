"""
OpenCV-based camera adapter for webcam capture.

Works on both Windows and Linux platforms using OpenCV's VideoCapture.
"""

import time
import logging
from typing import Optional

import cv2
import numpy as np

from src.capture.adapter import CameraAdapter, CaptureConfig
from src.capture.frame import Frame, FrameSource

logger = logging.getLogger(__name__)


class OpenCVCameraAdapter(CameraAdapter):
    """
    Camera adapter using OpenCV VideoCapture for webcam input.
    
    Suitable for development on Windows and as fallback on Linux.
    """
    
    def __init__(self, config: CaptureConfig):
        """
        Initialize the OpenCV camera adapter.
        
        Args:
            config: Capture configuration
        """
        super().__init__(config)
        self._cap: Optional[cv2.VideoCapture] = None
    
    def initialize(self) -> bool:
        """
        Initialize the webcam using OpenCV.
        
        Returns:
            True if initialization successful
        """
        try:
            camera_index = self._config.camera_index
            
            # Try DirectShow backend on Windows for better performance
            import platform
            if platform.system() == "Windows":
                self._cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self._cap = cv2.VideoCapture(camera_index)
            
            if not self._cap.isOpened():
                logger.error(f"Failed to open camera index {camera_index}")
                return False
            
            # Set resolution
            width, height = self._config.resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Set FPS
            self._cap.set(cv2.CAP_PROP_FPS, self._config.target_fps)
            
            # Set buffer size to 1 to minimize latency
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify settings
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(
                f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
            )
            
            # Warm up - discard first few frames
            for _ in range(5):
                self._cap.read()
            
            self._is_initialized = True
            self.reset_frame_count()
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return False
    
    def capture(self) -> Optional[Frame]:
        """
        Capture a frame from the webcam.
        
        Returns:
            Frame object if successful, None if capture failed
        """
        if not self._is_initialized or self._cap is None:
            logger.warning("Camera not initialized")
            return None
        
        try:
            # Capture with timeout consideration
            ret, frame_data = self._cap.read()
            timestamp = time.monotonic()
            
            if not ret or frame_data is None:
                logger.warning("Frame capture failed")
                return None
            
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
                source=FrameSource.WEBCAM,
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None
    
    def release(self) -> None:
        """Release the webcam."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_initialized = False
        logger.info("Camera released")
    
    def is_healthy(self) -> bool:
        """
        Check if the camera is functioning properly.
        
        Returns:
            True if camera is healthy
        """
        if not self._is_initialized or self._cap is None:
            return False
        
        return self._cap.isOpened()
    
    def get_actual_resolution(self) -> Optional[tuple]:
        """
        Get the actual resolution being used.
        
        Returns:
            (width, height) tuple or None if not initialized
        """
        if self._cap is None:
            return None
        
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
