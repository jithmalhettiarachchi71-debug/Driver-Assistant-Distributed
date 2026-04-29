"""
Polynomial fitting for lane lines.

Fits second-order polynomials to lane line points.
"""

import warnings
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.lane.hough_lines import LineSegment
from src.lane.result import LanePolynomial

# Suppress numpy polyfit warnings (poorly conditioned fits are handled gracefully)
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')


class PolynomialFitter:
    """
    Fits polynomial curves to lane line points.
    
    Uses least-squares fitting with outlier rejection.
    """
    
    def __init__(
        self,
        degree: int = 2,
        min_points: int = 4,
        ransac_iterations: int = 50,  # Enable RANSAC for outlier rejection
        ransac_threshold: float = 25.0,
    ):
        """
        Initialize polynomial fitter.
        
        Args:
            degree: Polynomial degree (2 for quadratic)
            min_points: Minimum points required for fitting
            ransac_iterations: Number of RANSAC iterations (0 to disable)
            ransac_threshold: RANSAC inlier distance threshold
        """
        self._degree = degree
        self._min_points = min_points
        self._ransac_iterations = ransac_iterations
        self._ransac_threshold = ransac_threshold
    
    def fit_from_lines(
        self,
        lines: List[LineSegment],
        y_range: Tuple[int, int],
    ) -> Optional[LanePolynomial]:
        """
        Fit a polynomial to a set of line segments.
        
        Args:
            lines: List of line segments
            y_range: (y_start, y_end) range for the fitted polynomial
            
        Returns:
            LanePolynomial if successful, None otherwise
        """
        if not lines:
            return None
        
        # Extract points from line segments with length-based weighting
        points_x = []
        points_y = []
        
        for line in lines:
            # Sample more points along longer lines for better weighting
            num_samples = max(2, int(line.length / 15))
            
            for i in range(num_samples + 1):
                t = i / num_samples
                x = line.x1 + t * (line.x2 - line.x1)
                y = line.y1 + t * (line.y2 - line.y1)
                points_x.append(x)
                points_y.append(y)
        
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        
        return self.fit_from_points(points_x, points_y, y_range)
    
    def fit_from_points(
        self,
        points_x: np.ndarray,
        points_y: np.ndarray,
        y_range: Tuple[int, int],
    ) -> Optional[LanePolynomial]:
        """
        Fit a polynomial to a set of points.
        
        Args:
            points_x: X coordinates
            points_y: Y coordinates
            y_range: (y_start, y_end) range
            
        Returns:
            LanePolynomial if successful, None otherwise
        """
        if len(points_x) < self._min_points:
            return None
        
        try:
            if self._ransac_iterations > 0:
                coeffs, inlier_count = self._fit_ransac(points_x, points_y)
                if coeffs is None:
                    return None
                point_count = inlier_count
            else:
                # Standard least-squares fit
                # Fit x = f(y) since lanes are more vertical than horizontal
                coeffs = np.polyfit(points_y, points_x, self._degree)
                point_count = len(points_x)
            
            # Compute confidence based on fit quality
            confidence = self._compute_confidence(
                coeffs, points_x, points_y, point_count
            )
            
            return LanePolynomial(
                coefficients=tuple(coeffs),
                y_range=y_range,
                confidence=confidence,
                point_count=point_count,
            )
            
        except (np.linalg.LinAlgError, ValueError):
            return None
    
    def _fit_ransac(
        self,
        points_x: np.ndarray,
        points_y: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Fit polynomial using RANSAC for outlier robustness.
        
        Uses a fixed random seed for determinism.
        
        Returns:
            (coefficients, inlier_count) or (None, 0) if failed
        """
        n_points = len(points_x)
        if n_points < self._min_points:
            return None, 0
        
        # Use fixed seed for determinism
        rng = np.random.RandomState(42)
        
        best_coeffs = None
        best_inliers = 0
        
        min_samples = self._degree + 1
        
        for _ in range(self._ransac_iterations):
            # Random sample
            indices = rng.choice(n_points, min_samples, replace=False)
            sample_x = points_x[indices]
            sample_y = points_y[indices]
            
            try:
                # Fit to sample
                coeffs = np.polyfit(sample_y, sample_x, self._degree)
                
                # Count inliers
                predicted_x = np.polyval(coeffs, points_y)
                errors = np.abs(points_x - predicted_x)
                inliers = np.sum(errors < self._ransac_threshold)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_coeffs = coeffs
                    
            except (np.linalg.LinAlgError, ValueError):
                continue
        
        if best_coeffs is not None and best_inliers >= self._min_points:
            # Refit using all inliers
            predicted_x = np.polyval(best_coeffs, points_y)
            errors = np.abs(points_x - predicted_x)
            inlier_mask = errors < self._ransac_threshold
            
            try:
                best_coeffs = np.polyfit(
                    points_y[inlier_mask],
                    points_x[inlier_mask],
                    self._degree
                )
            except (np.linalg.LinAlgError, ValueError):
                pass
        
        return best_coeffs, best_inliers
    
    def _compute_confidence(
        self,
        coeffs: np.ndarray,
        points_x: np.ndarray,
        points_y: np.ndarray,
        point_count: int,
    ) -> float:
        """
        Compute confidence score for the fit.
        
        Based on:
        - Number of points used
        - Fit residual error
        """
        # Compute residuals
        predicted_x = np.polyval(coeffs, points_y)
        residuals = np.abs(points_x - predicted_x)
        mean_error = np.mean(residuals)
        
        # Confidence decreases with error
        error_confidence = max(0.0, 1.0 - mean_error / 50.0)
        
        # Confidence increases with point count (up to a limit)
        point_confidence = min(1.0, point_count / 50.0)
        
        # Combined confidence
        confidence = 0.7 * error_confidence + 0.3 * point_confidence
        
        return min(1.0, max(0.0, confidence))
