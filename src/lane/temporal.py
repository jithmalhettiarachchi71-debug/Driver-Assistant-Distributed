"""
Temporal stabilization for lane detection.

Applies exponential moving average (EMA) filtering to smooth lane
detections over time and handle brief detection failures.

Also handles erratic detection (e.g. zigzag areas near crosswalks) by
freezing to a stable straight line when the lane position jumps too much.
"""

import time
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

from src.lane.result import LanePolynomial, LaneResult


@dataclass
class StabilizedLane:
    """A temporally stabilized lane with history."""
    polynomial: Optional[LanePolynomial]
    last_valid_time: float
    invalid_frame_count: int
    smoothed_coeffs: Optional[np.ndarray]
    # For erratic detection handling
    last_x_bottom: Optional[float] = None
    erratic_frame_count: int = 0
    frozen_polynomial: Optional[LanePolynomial] = None


class TemporalStabilizer:
    """
    Applies temporal filtering to lane detections.
    
    Features:
    - EMA smoothing of polynomial coefficients
    - Reuse of last valid detection during brief failures
    - Graceful degradation after extended failures
    - Erratic detection handling (freezes to straight line)
    """
    
    def __init__(
        self,
        ema_alpha: float = 0.3,
        max_invalid_frames: int = 5,
        max_invalid_time_ms: float = 500.0,
        max_jump_pixels: int = 30,
        erratic_freeze_frames: int = 10,
    ):
        """
        Initialize temporal stabilizer.
        
        Args:
            ema_alpha: EMA smoothing factor (0-1, higher = more responsive)
            max_invalid_frames: Max frames to reuse last valid detection
            max_invalid_time_ms: Max time to reuse last valid detection
            max_jump_pixels: Max x-position change before considered erratic
            erratic_freeze_frames: Frames to freeze after erratic detection
        """
        self._alpha = ema_alpha
        self._max_invalid_frames = max_invalid_frames
        self._max_invalid_time_s = max_invalid_time_ms / 1000.0
        self._max_jump_pixels = max_jump_pixels
        self._erratic_freeze_frames = erratic_freeze_frames
        
        # State for left and right lanes
        self._left: Optional[StabilizedLane] = None
        self._right: Optional[StabilizedLane] = None
    
    def update(
        self,
        left_lane: Optional[LanePolynomial],
        right_lane: Optional[LanePolynomial],
        timestamp: float,
    ) -> Tuple[Optional[LanePolynomial], Optional[LanePolynomial]]:
        """
        Update stabilizer with new detections.
        
        Args:
            left_lane: Detected left lane (or None)
            right_lane: Detected right lane (or None)
            timestamp: Current timestamp
            
        Returns:
            (stabilized_left, stabilized_right) tuple
        """
        stabilized_left = self._update_lane(
            self._left, left_lane, timestamp, is_left=True
        )
        stabilized_right = self._update_lane(
            self._right, right_lane, timestamp, is_left=False
        )
        
        return stabilized_left, stabilized_right
    
    def _update_lane(
        self,
        state: Optional[StabilizedLane],
        new_detection: Optional[LanePolynomial],
        timestamp: float,
        is_left: bool,
    ) -> Optional[LanePolynomial]:
        """
        Update a single lane's stabilized state.
        
        Args:
            state: Current stabilized state
            new_detection: New detection (or None)
            timestamp: Current timestamp
            is_left: True for left lane, False for right
            
        Returns:
            Stabilized lane polynomial
        """
        # Initialize state if needed
        if state is None:
            state = StabilizedLane(
                polynomial=None,
                last_valid_time=0.0,
                invalid_frame_count=0,
                smoothed_coeffs=None,
                last_x_bottom=None,
                erratic_frame_count=0,
                frozen_polynomial=None,
            )
            if is_left:
                self._left = state
            else:
                self._right = state
        
        if new_detection is not None:
            # Check for erratic jump (indicates zigzag/crosswalk area)
            new_coeffs = np.array(new_detection.coefficients)
            y_bottom = new_detection.y_range[1]
            new_x_bottom = new_coeffs[0] * y_bottom**2 + new_coeffs[1] * y_bottom + new_coeffs[2]
            
            is_erratic = False
            if state.last_x_bottom is not None:
                jump = abs(new_x_bottom - state.last_x_bottom)
                if jump > self._max_jump_pixels:
                    is_erratic = True
                    state.erratic_frame_count = self._erratic_freeze_frames
            
            # If we're in erratic mode, hold the frozen line
            if state.erratic_frame_count > 0:
                state.erratic_frame_count -= 1
                state.last_x_bottom = new_x_bottom  # Still track position
                
                if state.frozen_polynomial is not None:
                    # Return the frozen straight line
                    return state.frozen_polynomial
                # If no frozen line, fall through to normal processing
            
            # Normal processing - create/save frozen line before smoothing
            if state.smoothed_coeffs is not None and state.erratic_frame_count == 0:
                # Save current stable line as potential freeze target
                state.frozen_polynomial = self._create_straight_line(
                    state.smoothed_coeffs, new_detection.y_range
                )
            
            # Apply EMA smoothing
            if state.smoothed_coeffs is not None:
                smoothed = (
                    self._alpha * new_coeffs +
                    (1 - self._alpha) * state.smoothed_coeffs
                )
            else:
                smoothed = new_coeffs
            
            state.smoothed_coeffs = smoothed
            state.last_valid_time = timestamp
            state.invalid_frame_count = 0
            state.last_x_bottom = new_x_bottom
            
            # Create smoothed polynomial
            state.polynomial = LanePolynomial(
                coefficients=tuple(smoothed),
                y_range=new_detection.y_range,
                confidence=new_detection.confidence,
                point_count=new_detection.point_count,
            )
            
            return state.polynomial
        
        else:
            # No detection - check if we can reuse last valid
            state.invalid_frame_count += 1
            
            time_since_valid = timestamp - state.last_valid_time
            
            can_reuse = (
                state.polynomial is not None and
                state.invalid_frame_count <= self._max_invalid_frames and
                time_since_valid <= self._max_invalid_time_s
            )
            
            if can_reuse:
                # Return last valid with reduced confidence
                poly = state.polynomial
                decay_factor = 1.0 - (state.invalid_frame_count / (self._max_invalid_frames + 1))
                
                return LanePolynomial(
                    coefficients=poly.coefficients,
                    y_range=poly.y_range,
                    confidence=poly.confidence * decay_factor,
                    point_count=poly.point_count,
                )
            else:
                # Too long without valid detection - clear state
                state.polynomial = None
                state.smoothed_coeffs = None
                state.frozen_polynomial = None
                state.erratic_frame_count = 0
                return None
    
    def _create_straight_line(
        self,
        coeffs: np.ndarray,
        y_range: Tuple[int, int],
    ) -> LanePolynomial:
        """
        Create a straight line (linear) from current coefficients.
        
        This is used as a frozen placeholder during erratic detection.
        Removes the quadratic term to prevent curved lines in zigzag areas.
        
        Args:
            coeffs: Current polynomial coefficients (a, b, c)
            y_range: Y-coordinate range
            
        Returns:
            Straight line polynomial with a=0
        """
        y_start, y_end = y_range
        
        # Evaluate current polynomial at endpoints
        a, b, c = coeffs
        x_start = a * y_start**2 + b * y_start + c
        x_end = a * y_end**2 + b * y_end + c
        
        # Create linear coefficients: x = 0*y² + b'*y + c'
        if y_end != y_start:
            new_b = (x_end - x_start) / (y_end - y_start)
            new_c = x_start - new_b * y_start
        else:
            new_b = 0.0
            new_c = x_start
        
        return LanePolynomial(
            coefficients=(0.0, new_b, new_c),
            y_range=y_range,
            confidence=0.7,  # Reduced confidence for frozen line
            point_count=0,
        )
    
    def reset(self) -> None:
        """Reset all stabilization state."""
        self._left = None
        self._right = None
    
    @property
    def is_left_stable(self) -> bool:
        """Check if left lane has stable detection."""
        return (
            self._left is not None and
            self._left.polynomial is not None and
            self._left.invalid_frame_count == 0
        )
    
    @property
    def is_right_stable(self) -> bool:
        """Check if right lane has stable detection."""
        return (
            self._right is not None and
            self._right.polynomial is not None and
            self._right.invalid_frame_count == 0
        )
    
    @property
    def left_invalid_frames(self) -> int:
        """Get number of consecutive invalid frames for left lane."""
        return self._left.invalid_frame_count if self._left else 0
    
    @property
    def right_invalid_frames(self) -> int:
        """Get number of consecutive invalid frames for right lane."""
        return self._right.invalid_frame_count if self._right else 0
