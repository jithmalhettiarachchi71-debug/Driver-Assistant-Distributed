"""Frame capture module for the Vehicle Safety Alert System."""

from src.capture.adapter import CameraAdapter, CaptureConfig
from src.capture.frame import Frame, FrameSource
from src.capture.opencv_camera import OpenCVCameraAdapter
from src.capture.video_file import VideoFileAdapter
from src.capture.ip_camera import IPCameraAdapter
from src.capture.factory import create_camera_adapter

__all__ = [
    "CameraAdapter",
    "CaptureConfig",
    "Frame",
    "FrameSource",
    "OpenCVCameraAdapter",
    "VideoFileAdapter",
    "IPCameraAdapter",
    "create_camera_adapter",
]
