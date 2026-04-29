"""
CSI Camera adapter for Raspberry Pi.

Uses picamera2 library for CSI camera interface.
"""

import time
import logging
from typing import Optional

import numpy as np

from src.capture.adapter import CameraAdapter, CaptureConfig
from src.capture.frame import Frame, FrameSource
from src.utils.platform import is_raspberry_pi

logger = logging.getLogger(__name__)


class CSICameraAdapter(CameraAdapter):
    """
    Camera adapter for Raspberry Pi CSI camera using picamera2.
    
    Only available on Raspberry Pi platforms.
    """
    
    def __init__(self, config: CaptureConfig):
        """
        Initialize the CSI camera adapter.
        
        Args:
            config: Capture configuration
        """
        super().__init__(config)
        self._camera = None
        self._picamera2_class = None
    
    def initialize(self) -> bool:
        """
        Initialize the CSI camera using picamera2.
        
        Returns:
            True if initialization successful
        """
        if not is_raspberry_pi():
            logger.error("CSI camera only available on Raspberry Pi")
            return False
        
        # First, ensure any previous instance is fully released
        self.release()
        
        try:
            from picamera2 import Picamera2
            self._picamera2_class = Picamera2
            
            # Check available cameras first
            cameras = Picamera2.global_camera_info()
            if not cameras:
                logger.error("No cameras detected by libcamera")
                return False
            
            logger.info(f"Detected cameras: {cameras}")
            
            # Create camera instance
            self._camera = Picamera2(camera_num=0)
            
            width, height = self._config.resolution
            
            # Configure for video capture with proper alignment
            # Use align_configuration to ensure proper memory alignment
            config = self._camera.create_video_configuration(
                main={"size": (width, height), "format": "BGR888"},
                buffer_count=4,  # Increased buffer count for stability
                queue=True,  # Enable frame queue
            )
            
            # Align configuration to hardware requirements
            self._camera.align_configuration(config)
            
            logger.info(f"Camera config: {config}")
            
            self._camera.configure(config)
            
            # Start the camera
            self._camera.start()
            
            # Allow camera to warm up and stabilize
            time.sleep(1.0)
            
            # Verify camera is actually capturing by taking a test frame
            test_frame = self._camera.capture_array()
            if test_frame is None:
                logger.error("Camera started but capture_array returned None")
                self.release()
                return False
            
            logger.info(f"CSI camera initialized: {width}x{height}, test frame shape: {test_frame.shape}")
            
            self._is_initialized = True
            self.reset_frame_count()
            return True
            
        except ImportError as e:
            logger.error(f"picamera2 library not installed: {e}")
            return False
        except RuntimeError as e:
            # Common error when camera is busy
            logger.error(f"Camera runtime error (may be in use): {e}")
            self.release()
            return False
        except Exception as e:
            logger.error(f"CSI camera initialization error: {type(e).__name__}: {e}")
            self.release()
            return False
    
    def capture(self) -> Optional[Frame]:
        """
        Capture a frame from the CSI camera.
        
        Returns:
            Frame object if successful, None if capture failed
        """
        if not self._is_initialized or self._camera is None:
            logger.warning("CSI camera not initialized")
            return None
        
        try:
            # Capture frame as numpy array
            frame_data = self._camera.capture_array("main")
            timestamp = time.monotonic()
            
            if frame_data is None:
                logger.warning("CSI frame capture failed - got None")
                return None
            
            # picamera2 returns BGR888 as configured
            # Ensure uint8 dtype
            if frame_data.dtype != np.uint8:
                frame_data = frame_data.astype(np.uint8)
            
            # Verify shape - should be (height, width, 3)
            if len(frame_data.shape) != 3 or frame_data.shape[2] != 3:
                logger.error(f"Unexpected frame shape: {frame_data.shape}")
                return None
            
            frame = Frame(
                data=frame_data,
                timestamp=timestamp,
                sequence=self._increment_frame_count(),
                source=FrameSource.CSI,
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"CSI capture error: {type(e).__name__}: {e}")
            return None
    
    def release(self) -> None:
        """Release the CSI camera resources properly."""
        if self._camera is not None:
            try:
                # Check if camera is running before stopping
                if hasattr(self._camera, 'started') and self._camera.started:
                    self._camera.stop()
                    time.sleep(0.1)  # Small delay after stop
            except Exception as e:
                logger.warning(f"Error stopping CSI camera: {e}")
            
            try:
                self._camera.close()
            except Exception as e:
                logger.warning(f"Error closing CSI camera: {e}")
            
            self._camera = None
            time.sleep(0.2)  # Allow resources to be freed
        
        self._is_initialized = False
        logger.info("CSI camera released")
    
    def is_healthy(self) -> bool:
        """
        Check if the CSI camera is functioning properly.
        
        Returns:
            True if camera is healthy
        """
        if not self._is_initialized or self._camera is None:
            return False
        
        try:
            # Check if camera is started and responsive
            if hasattr(self._camera, 'started'):
                return self._camera.started
            # Fallback: check is_open
            if hasattr(self._camera, 'is_open'):
                return self._camera.is_open
            return True
        except Exception:
            return False
    
    def __del__(self):
        """Destructor to ensure camera is released."""
        self.release()
