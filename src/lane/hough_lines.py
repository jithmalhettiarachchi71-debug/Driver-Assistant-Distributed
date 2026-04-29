"""
Hough transform line extraction for lane detection.

Extracts line candidates from edge images using probabilistic Hough transform.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LineSegment:
    """
    Represents a line segment detected by Hough transform.
    
    Attributes:
        x1, y1: Start point coordinates
        x2, y2: End point coordinates
        slope: Line slope (dy/dx), None if vertical
        length: Line length in pixels
    """
    x1: int
    y1: int
    x2: int
    y2: int
    slope: Optional[float]
    length: float
    
    @staticmethod
    def from_points(x1: int, y1: int, x2: int, y2: int) -> "LineSegment":
        """Create a LineSegment from two points."""
        dx = x2 - x1
        dy = y2 - y1
        
        slope = dy / dx if dx != 0 else None
        length = np.sqrt(dx ** 2 + dy ** 2)
        
        return LineSegment(x1, y1, x2, y2, slope, length)
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        """Get the midpoint of the line segment."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def angle_degrees(self) -> float:
        """Get the angle of the line in degrees from horizontal."""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return np.degrees(np.arctan2(dy, dx))
    
    def get_points(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get start and end points as tuples."""
        return ((self.x1, self.y1), (self.x2, self.y2))


class HoughLineExtractor:
    """
    Extracts line segments from edge images using probabilistic Hough transform.
    """
    
    def __init__(
        self,
        rho: float = 1,
        theta_degrees: float = 1,
        threshold: int = 25,
        min_line_length: int = 10,
        max_line_gap: int = 200,
    ):
        """
        Initialize Hough line extractor.
        
        Args:
            rho: Distance resolution in pixels
            theta_degrees: Angle resolution in degrees
            threshold: Minimum votes for a line
            min_line_length: Minimum line length in pixels (lower = detect shorter broken lines)
            max_line_gap: Maximum gap between line segments to merge (higher = connect broken lines)
        """
        self._rho = rho
        self._theta = np.radians(theta_degrees)
        self._threshold = threshold
        self._min_line_length = min_line_length
        self._max_line_gap = max_line_gap
    
    def extract(self, edge_image: np.ndarray) -> List[LineSegment]:
        """
        Extract line segments from an edge image.
        
        OPT-6: Vectorized length and slope calculation.
        
        Args:
            edge_image: Binary edge image from Canny or similar
            
        Returns:
            List of LineSegment objects
        """
        lines = cv2.HoughLinesP(
            edge_image,
            self._rho,
            self._theta,
            self._threshold,
            minLineLength=self._min_line_length,
            maxLineGap=self._max_line_gap,
        )
        
        if lines is None:
            return []
        
        # OPT-6: Vectorized computation instead of per-line loop
        lines_array = lines.reshape(-1, 4)  # (N, 4): x1, y1, x2, y2
        n_lines = len(lines_array)
        
        if n_lines == 0:
            return []
        
        # Vectorized dx, dy calculation
        dx = lines_array[:, 2] - lines_array[:, 0]
        dy = lines_array[:, 3] - lines_array[:, 1]
        
        # Vectorized length calculation (single sqrt call for all lines)
        lengths = np.sqrt(dx * dx + dy * dy)
        
        # Vectorized slope (handle division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            slopes = np.where(dx != 0, dy / dx, np.nan)
        
        # Build LineSegment objects from vectorized results
        segments = []
        for i in range(n_lines):
            x1, y1, x2, y2 = lines_array[i]
            slope = float(slopes[i]) if not np.isnan(slopes[i]) else None
            segments.append(LineSegment(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                slope=slope,
                length=float(lengths[i])
            ))
        
        return segments
    
    def extract_raw(self, edge_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract raw Hough lines (for debugging).
        
        Args:
            edge_image: Binary edge image
            
        Returns:
            Raw lines array from OpenCV, or None
        """
        return cv2.HoughLinesP(
            edge_image,
            self._rho,
            self._theta,
            self._threshold,
            minLineLength=self._min_line_length,
            maxLineGap=self._max_line_gap,
        )
