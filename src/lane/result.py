"""
Lane detection result data structures.

Defines the output format for lane detection.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np


def find_lane_intersection(
    left: 'LanePolynomial',
    right: 'LanePolynomial',
    min_separation: int = 50,
) -> Optional[int]:
    """
    Find the y-coordinate where two lane polynomials intersect or get too close.
    
    Args:
        left: Left lane polynomial
        right: Right lane polynomial
        min_separation: Minimum pixel separation before considering it a "crossing"
        
    Returns:
        Y-coordinate just below the problem area, or None if lanes are fine
    """
    y_start = max(left.y_range[0], right.y_range[0])
    y_end = min(left.y_range[1], right.y_range[1])
    
    # Sample points to find where lanes cross or get too close
    y_values = np.linspace(y_start, y_end, 100)
    
    left_x = left.evaluate_array(y_values)
    right_x = right.evaluate_array(y_values)
    
    # Find where right_x - left_x (separation) becomes too small or negative
    separation = right_x - left_x
    
    for i in range(len(separation)):
        # Actual crossing (lanes swapped)
        if separation[i] <= 0:
            return int(y_values[i]) + 15
        # Lanes too close (visually looks like crossing)
        if separation[i] < min_separation:
            return int(y_values[i]) + 15
    
    return None


@dataclass
class LanePolynomial:
    """
    Represents a lane line as a second-order polynomial.
    
    The polynomial is defined as: x = ay² + by + c
    where y is the vertical coordinate (increasing downward).
    
    Attributes:
        coefficients: (a, b, c) polynomial coefficients
        y_range: (y_start, y_end) valid y-coordinate range
        confidence: Confidence score [0.0, 1.0]
        point_count: Number of points used in the fit
    """
    coefficients: Tuple[float, float, float]
    y_range: Tuple[int, int]
    confidence: float
    point_count: int
    
    def evaluate(self, y: float) -> float:
        """
        Evaluate the polynomial at a given y coordinate.
        
        Args:
            y: Y coordinate (vertical position)
            
        Returns:
            X coordinate of the lane at that y position
        """
        a, b, c = self.coefficients
        return a * y * y + b * y + c
    
    def evaluate_array(self, y_values: np.ndarray) -> np.ndarray:
        """
        Evaluate the polynomial at multiple y coordinates.
        
        Args:
            y_values: Array of y coordinates
            
        Returns:
            Array of corresponding x coordinates
        """
        a, b, c = self.coefficients
        return a * y_values ** 2 + b * y_values + c
    
    def get_points(self, num_points: int = 50) -> List[Tuple[int, int]]:
        """
        Generate points along the lane line.
        
        Args:
            num_points: Number of points to generate
            
        Returns:
            List of (x, y) integer coordinate tuples
        """
        y_start, y_end = self.y_range
        y_values = np.linspace(y_start, y_end, num_points)
        x_values = self.evaluate_array(y_values)
        
        return [(int(x), int(y)) for x, y in zip(x_values, y_values)]
    
    def is_valid_at(self, y: float, frame_width: int) -> bool:
        """
        Check if the lane position is valid at a given y coordinate.
        
        Args:
            y: Y coordinate to check
            frame_width: Width of the frame for boundary check
            
        Returns:
            True if the x position is within frame bounds
        """
        x = self.evaluate(y)
        return 0 <= x <= frame_width
    
    def with_truncated_range(self, new_y_start: int) -> 'LanePolynomial':
        """
        Create a new LanePolynomial with truncated y_range.
        
        Args:
            new_y_start: New starting y coordinate (cuts off upper part)
            
        Returns:
            New LanePolynomial with updated y_range
        """
        y_start, y_end = self.y_range
        # Ensure new_y_start is within valid bounds
        new_y_start = max(new_y_start, y_start)
        new_y_start = min(new_y_start, y_end - 10)  # Keep at least 10 pixels
        
        return LanePolynomial(
            coefficients=self.coefficients,
            y_range=(new_y_start, y_end),
            confidence=self.confidence,
            point_count=self.point_count,
        )


@dataclass
class LaneResult:
    """
    Result of lane detection for a single frame.
    
    Attributes:
        left_lane: Left lane polynomial, or None if not detected
        right_lane: Right lane polynomial, or None if not detected
        valid: True if both lanes detected and stable
        partial: True if only one lane detected
        timestamp: Monotonic timestamp of detection
        latency_ms: Processing latency in milliseconds
    """
    left_lane: Optional[LanePolynomial]
    right_lane: Optional[LanePolynomial]
    valid: bool
    partial: bool
    timestamp: float
    latency_ms: float
    
    def get_lane_center(self, y: int) -> Optional[float]:
        """
        Calculate the lane center at a given y coordinate.
        
        Args:
            y: Y coordinate (vertical position)
            
        Returns:
            X coordinate of lane center, or None if not calculable
        """
        if not self.valid or self.left_lane is None or self.right_lane is None:
            return None
        
        left_x = self.left_lane.evaluate(y)
        right_x = self.right_lane.evaluate(y)
        
        return (left_x + right_x) / 2
    
    def get_lane_width(self, y: int) -> Optional[float]:
        """
        Calculate the lane width at a given y coordinate.
        
        Args:
            y: Y coordinate
            
        Returns:
            Lane width in pixels, or None if not calculable
        """
        if not self.valid or self.left_lane is None or self.right_lane is None:
            return None
        
        left_x = self.left_lane.evaluate(y)
        right_x = self.right_lane.evaluate(y)
        
        return abs(right_x - left_x)
    
    def is_point_in_lane(self, x: float, y: float, margin: float = 0) -> Optional[bool]:
        """
        Check if a point is within the lane boundaries.
        
        Args:
            x: X coordinate of point
            y: Y coordinate of point
            margin: Additional margin (positive = wider lane)
            
        Returns:
            True if in lane, False if outside, None if can't determine
        """
        if not self.valid or self.left_lane is None or self.right_lane is None:
            return None
        
        left_x = self.left_lane.evaluate(y) - margin
        right_x = self.right_lane.evaluate(y) + margin
        
        return left_x <= x <= right_x
    
    @staticmethod
    def create_invalid(timestamp: float, latency_ms: float) -> "LaneResult":
        """Create an invalid lane result."""
        return LaneResult(
            left_lane=None,
            right_lane=None,
            valid=False,
            partial=False,
            timestamp=timestamp,
            latency_ms=latency_ms,
        )
