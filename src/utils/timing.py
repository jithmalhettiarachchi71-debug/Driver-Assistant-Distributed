"""
Timing utilities for deterministic frame processing.

Provides timing measurement and frame rate enforcement.
"""

import time
from typing import Optional
from contextlib import contextmanager


class Timer:
    """
    High-precision timer for measuring operation latency.
    
    Uses monotonic clock for reliable measurements.
    """
    
    def __init__(self):
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
    
    def start(self) -> "Timer":
        """Start the timer."""
        self._start_time = time.monotonic()
        self._end_time = None
        return self
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        self._end_time = time.monotonic()
        return self.elapsed_ms
    
    @property
    def elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds, or 0 if timer not started
        """
        if self._start_time is None:
            return 0.0
        
        end = self._end_time if self._end_time is not None else time.monotonic()
        return (end - self._start_time) * 1000.0
    
    @property
    def elapsed_s(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        return self.elapsed_ms / 1000.0
    
    def reset(self) -> "Timer":
        """Reset the timer."""
        self._start_time = None
        self._end_time = None
        return self


@contextmanager
def measure_time():
    """
    Context manager for measuring execution time.
    
    Yields:
        Timer object that can be queried for elapsed time
        
    Example:
        with measure_time() as timer:
            # do work
        print(f"Elapsed: {timer.elapsed_ms:.2f}ms")
    """
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


class FrameRateEnforcer:
    """
    Enforces a target frame rate by sleeping when processing is faster than needed.
    
    This ensures deterministic timing by maintaining consistent frame intervals.
    """
    
    def __init__(self, target_fps: float):
        """
        Initialize frame rate enforcer.
        
        Args:
            target_fps: Target frames per second (must be > 0)
        """
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        
        self._target_fps = target_fps
        self._frame_interval_s = 1.0 / target_fps
        self._last_frame_time: Optional[float] = None
        self._frame_count = 0
        self._total_sleep_time = 0.0
        self._fps_window: list[float] = []
        self._fps_window_size = 30
    
    @property
    def target_fps(self) -> float:
        """Get target FPS."""
        return self._target_fps
    
    @property
    def frame_interval_ms(self) -> float:
        """Get target frame interval in milliseconds."""
        return self._frame_interval_s * 1000.0
    
    def start_frame(self) -> float:
        """
        Mark the start of a new frame.
        
        Returns:
            Monotonic timestamp of frame start
        """
        now = time.monotonic()
        
        # Calculate actual FPS based on inter-frame time
        if self._last_frame_time is not None:
            delta = now - self._last_frame_time
            if delta > 0:
                actual_fps = 1.0 / delta
                self._fps_window.append(actual_fps)
                if len(self._fps_window) > self._fps_window_size:
                    self._fps_window.pop(0)
        
        return now
    
    def end_frame(self, frame_start: float) -> float:
        """
        Mark the end of frame processing and sleep if needed.
        
        Args:
            frame_start: Timestamp from start_frame()
            
        Returns:
            Actual sleep time in milliseconds (0 if no sleep needed)
        """
        now = time.monotonic()
        processing_time = now - frame_start
        remaining_time = self._frame_interval_s - processing_time
        
        sleep_time_ms = 0.0
        if remaining_time > 0:
            time.sleep(remaining_time)
            sleep_time_ms = remaining_time * 1000.0
            self._total_sleep_time += remaining_time
        
        self._last_frame_time = time.monotonic()
        self._frame_count += 1
        
        return sleep_time_ms
    
    def get_current_fps(self) -> float:
        """
        Get the current measured FPS (smoothed average).
        
        Returns:
            Current FPS estimate, or target FPS if not enough samples
        """
        if len(self._fps_window) < 2:
            return self._target_fps
        
        return sum(self._fps_window) / len(self._fps_window)
    
    def reset(self) -> None:
        """Reset frame rate statistics."""
        self._last_frame_time = None
        self._frame_count = 0
        self._total_sleep_time = 0.0
        self._fps_window.clear()


def get_monotonic_timestamp() -> float:
    """
    Get current monotonic timestamp.
    
    Returns:
        Monotonic time in seconds
    """
    return time.monotonic()


def sleep_ms(milliseconds: float) -> None:
    """
    Sleep for specified milliseconds.
    
    Args:
        milliseconds: Time to sleep in milliseconds
    """
    if milliseconds > 0:
        time.sleep(milliseconds / 1000.0)
