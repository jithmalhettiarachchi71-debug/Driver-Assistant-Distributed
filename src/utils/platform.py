"""
Platform detection utilities.

Provides functions to detect the current platform and available hardware.
"""

import platform
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def is_raspberry_pi() -> bool:
    """
    Check if running on a Raspberry Pi.
    
    Returns:
        True if running on Raspberry Pi, False otherwise
    """
    if platform.system() != "Linux":
        return False
    
    # Check device tree model
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read()
            return "Raspberry Pi" in model
    except (FileNotFoundError, PermissionError):
        pass
    
    # Fallback: check cpuinfo
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
            return "BCM" in cpuinfo or "Raspberry" in cpuinfo
    except (FileNotFoundError, PermissionError):
        pass
    
    return False


@lru_cache(maxsize=1)
def get_platform_name() -> str:
    """
    Get a human-readable platform name.
    
    Returns:
        Platform name string
    """
    if is_raspberry_pi():
        return "Raspberry Pi"
    
    system = platform.system()
    if system == "Windows":
        return f"Windows {platform.release()}"
    elif system == "Linux":
        return f"Linux ({platform.release()})"
    elif system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    
    return system


def is_gpio_available() -> bool:
    """
    Check if GPIO is available (Raspberry Pi only).
    
    Returns:
        True if GPIO can be used
    """
    if not is_raspberry_pi():
        return False
    
    try:
        import RPi.GPIO
        return True
    except ImportError:
        return False


def is_csi_camera_available() -> bool:
    """
    Check if CSI camera interface is available.
    
    Returns:
        True if CSI camera can be used
    """
    if not is_raspberry_pi():
        return False
    
    try:
        from picamera2 import Picamera2
        return True
    except ImportError:
        return False


def get_cpu_temperature() -> float | None:
    """
    Get CPU temperature in Celsius.
    
    Returns:
        Temperature in Celsius, or None if unavailable
    """
    if is_raspberry_pi():
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = int(f.read().strip()) / 1000.0
                return temp
        except (FileNotFoundError, ValueError, PermissionError):
            pass
    
    return None
