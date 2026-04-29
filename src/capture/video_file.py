"""
Video file adapter for playback of recorded videos.

Useful for development, testing, and debugging with pre-recorded footage.
"""

import time
import logging
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from src.capture.adapter import CameraAdapter, CaptureConfig
from src.capture.frame import Frame, FrameSource

logger = logging.getLogger(__name__)


class VideoFileAdapter(CameraAdapter):
    """
    Camera adapter for video file playback.
    
    Supports common video formats via OpenCV (MP4, AVI, MKV, etc.).
    Can optionally loop the video for continuous testing.
    """
    
    def __init__(self, config: CaptureConfig, loop: bool = True):
        """
        Initialize the video file adapter.
        
        Args:
            config: Capture configuration (must include video_path)
            loop: Whether to loop the video when it ends
        """
        super().__init__(config)
        self._cap: Optional[cv2.VideoCapture] = None
        self._loop = loop
        self._video_fps: float = 30.0
        self._total_frames: int = 0
        self._last_frame_time: float = 0.0
    
    def initialize(self) -> bool:
        """
        Initialize video file playback.
        
        Returns:
            True if initialization successful
        """
        video_path = self._config.video_path
        
        if video_path is None:
            logger.error("Video path not specified in config")
            return False
        
        path = Path(video_path)
        if not path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False
        
        try:
            self._cap = cv2.VideoCapture(str(path))
            
            if not self._cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                return False
            
            # Get video properties
            self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(
                f"Video opened: {path.name} ({width}x{height} @ {self._video_fps:.1f} FPS, "
                f"{self._total_frames} frames)"
            )
            
            self._is_initialized = True
            self._last_frame_time = time.monotonic()
            self.reset_frame_count()
            return True
            
        except Exception as e:
            logger.error(f"Video initialization error: {e}")
            return False
    
    def capture(self) -> Optional[Frame]:
        """
        Capture the next frame from the video.
        
        Returns:
            Frame object if successful, None if capture failed
        """
        if not self._is_initialized or self._cap is None:
            logger.warning("Video not initialized")
            return None
        
        try:
            # Enforce timing to match target FPS
            current_time = time.monotonic()
            target_interval = 1.0 / self._config.target_fps
            elapsed = current_time - self._last_frame_time
            
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            
            ret, frame_data = self._cap.read()
            timestamp = time.monotonic()
            self._last_frame_time = timestamp
            
            if not ret or frame_data is None:
                if self._loop:
                    # Restart from beginning
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame_data = self._cap.read()
                    if not ret or frame_data is None:
                        logger.error("Failed to loop video")
                        return None
                    logger.debug("Video looped to beginning")
                else:
                    logger.info("Video playback complete")
                    return None
            
            # Ensure correct format
            if frame_data.dtype != np.uint8:
                frame_data = frame_data.astype(np.uint8)
            
            # Resize to target resolution if needed
            target_w, target_h = self._config.resolution
            if frame_data.shape[1] != target_w or frame_data.shape[0] != target_h:
                frame_data = cv2.resize(frame_data, (target_w, target_h))
            
            frame = Frame(
                data=frame_data,
                timestamp=timestamp,
                sequence=self._increment_frame_count(),
                source=FrameSource.VIDEO_FILE,
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"Video capture error: {e}")
            return None
    
    def release(self) -> None:
        """Release the video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_initialized = False
        logger.info("Video file released")
    
    def is_healthy(self) -> bool:
        """
        Check if video playback is functioning.
        
        Returns:
            True if video is healthy
        """
        if not self._is_initialized or self._cap is None:
            return False
        
        return self._cap.isOpened()
    
    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame number.
        
        Args:
            frame_number: Target frame number (0-indexed)
            
        Returns:
            True if seek successful
        """
        if self._cap is None:
            return False
        
        frame_number = max(0, min(frame_number, self._total_frames - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return True
    
    def get_progress(self) -> float:
        """
        Get current playback progress.
        
        Returns:
            Progress as ratio (0.0 to 1.0)
        """
        if self._cap is None or self._total_frames == 0:
            return 0.0
        
        current_frame = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        return current_frame / self._total_frames
    
    @property
    def video_fps(self) -> float:
        """Get the native FPS of the video file."""
        return self._video_fps
    
    @property
    def total_frames(self) -> int:
        """Get total number of frames in the video."""
        return self._total_frames
