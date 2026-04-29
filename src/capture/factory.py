"""
Camera adapter factory.

Creates the appropriate camera adapter based on platform and configuration.
"""

import logging
from typing import Optional

from src.capture.adapter import CameraAdapter, CaptureConfig
from src.capture.frame import FrameSource
from src.capture.opencv_camera import OpenCVCameraAdapter
from src.capture.video_file import VideoFileAdapter
from src.capture.ip_camera import IPCameraAdapter
from src.utils.platform import is_raspberry_pi

logger = logging.getLogger(__name__)


def create_camera_adapter(
    config: CaptureConfig,
    video_loop: bool = True
) -> CameraAdapter:
    """
    Create the appropriate camera adapter based on configuration and platform.
    
    Args:
        config: Capture configuration
        video_loop: Whether to loop video file playback
        
    Returns:
        Appropriate CameraAdapter instance
        
    Raises:
        ValueError: If configuration is invalid
        ImportError: If required library is not available
    """
    source = config.source
    
    if source == FrameSource.VIDEO_FILE:
        if config.video_path is None:
            raise ValueError("video_path required for VIDEO_FILE source")
        
        logger.info(f"Creating video file adapter: {config.video_path}")
        return VideoFileAdapter(config, loop=video_loop)
    
    elif source == FrameSource.CSI:
        if not is_raspberry_pi():
            raise ValueError("CSI camera only available on Raspberry Pi")
        
        # Import here to avoid ImportError on non-Pi platforms
        from src.capture.csi_camera import CSICameraAdapter
        logger.info("Creating CSI camera adapter")
        return CSICameraAdapter(config)
    
    elif source == FrameSource.WEBCAM:
        logger.info(f"Creating OpenCV camera adapter (index {config.camera_index})")
        return OpenCVCameraAdapter(config)
    
    elif source == FrameSource.IP_CAMERA:
        if config.ip_url is None:
            raise ValueError("ip_url required for IP_CAMERA source")
        
        logger.info(f"Creating IP camera adapter")
        return IPCameraAdapter(config)
    
    else:
        raise ValueError(f"Unknown frame source: {source}")


def auto_detect_source() -> FrameSource:
    """
    Automatically detect the best available frame source.
    
    Returns:
        Best available FrameSource for current platform
    """
    if is_raspberry_pi():
        try:
            from picamera2 import Picamera2
            return FrameSource.CSI
        except ImportError:
            pass
    
    # Default to webcam
    return FrameSource.WEBCAM
