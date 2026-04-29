"""
Complete lane detection pipeline.

Combines all lane detection stages into a single processing pipeline.
"""

import time
import logging
from typing import Optional, Tuple
import numpy as np
import cv2

from src.config import LaneDetectionConfig
from src.capture.frame import Frame
from src.lane.result import LaneResult, LanePolynomial, find_lane_intersection
from src.lane.color_filter import ColorFilter
from src.lane.edge_detection import EdgeDetector
from src.lane.hough_lines import HoughLineExtractor
from src.lane.geometric_filter import GeometricFilter
from src.lane.polynomial_fit import PolynomialFitter
from src.lane.temporal import TemporalStabilizer

logger = logging.getLogger(__name__)


class LaneDetectionPipeline:
    """
    Complete lane detection pipeline.
    
    Processing stages:
    1. ROI masking (trapezoid covering lower half)
    2. HSV color segmentation (white + yellow)
    3. Gaussian blur + Canny edge detection
    4. Probabilistic Hough transform
    5. Geometric filtering
    6. Polynomial fitting
    7. Temporal stabilization (EMA)
    """
    
    def __init__(self, config: LaneDetectionConfig):
        """
        Initialize lane detection pipeline.
        
        Args:
            config: Lane detection configuration
        """
        self._config = config
        
        # Initialize pipeline stages
        self._color_filter = ColorFilter(
            white_lower=(
                config.hsv_white.h_min,
                config.hsv_white.s_min,
                config.hsv_white.v_min,
            ),
            white_upper=(
                config.hsv_white.h_max,
                config.hsv_white.s_max,
                config.hsv_white.v_max,
            ),
            yellow_lower=(
                config.hsv_yellow.h_min,
                config.hsv_yellow.s_min,
                config.hsv_yellow.v_min,
            ),
            yellow_upper=(
                config.hsv_yellow.h_max,
                config.hsv_yellow.s_max,
                config.hsv_yellow.v_max,
            ),
        )
        
        self._edge_detector = EdgeDetector(
            gaussian_kernel=config.gaussian_kernel,
            canny_low=config.canny_low,
            canny_high=config.canny_high,
        )
        
        self._hough_extractor = HoughLineExtractor(
            rho=config.hough_rho,
            theta_degrees=config.hough_theta_deg,
            threshold=config.hough_threshold,
            min_line_length=config.hough_min_length,
            max_line_gap=config.hough_max_gap,
        )
        
        self._geometric_filter = GeometricFilter(
            slope_min=config.slope_min,
            slope_max=config.slope_max,
            min_length=config.min_line_length,
            angle_tolerance=25.0,
        )
        
        self._poly_fitter = PolynomialFitter(
            degree=2,
            min_points=4,
            ransac_iterations=50,
            ransac_threshold=25.0,
        )
        
        self._stabilizer = TemporalStabilizer(
            ema_alpha=config.ema_alpha,
            max_invalid_frames=config.max_invalid_frames,
        )
        
        # ROI will be computed on first frame
        self._roi_mask: Optional[np.ndarray] = None
        self._roi_vertices: Optional[np.ndarray] = None
        self._frame_shape: Optional[Tuple[int, int]] = None
    
    def process(self, frame: Frame) -> LaneResult:
        """
        Process a frame through the complete lane detection pipeline.
        
        Args:
            frame: Input frame
            
        Returns:
            LaneResult with detected lanes
        """
        start_time = time.monotonic()
        
        try:
            image = frame.data
            height, width = image.shape[:2]
            
            # Initialize ROI mask if needed
            if self._roi_mask is None or self._frame_shape != (height, width):
                self._init_roi_mask(width, height)
            
            # Stage 1: Apply ROI mask
            roi_image = cv2.bitwise_and(image, image, mask=self._roi_mask)
            
            # OPT-3: Fused color filtering + edge detection (single pass)
            # Replaces separate Stage 2 (color) and Stage 3 (edge) with combined operation
            edges = self._color_filter.filter_with_edges(
                roi_image,
                gaussian_kernel=self._edge_detector._gaussian_kernel,
                canny_low=self._edge_detector._canny_low,
                canny_high=self._edge_detector._canny_high,
            )
            
            # Apply ROI to edges
            edges = cv2.bitwise_and(edges, self._roi_mask)
            
            # Stage 4: Hough line extraction
            lines = self._hough_extractor.extract(edges)
            
            # Stage 5: Geometric filtering
            roi_y_start = int(height * self._config.roi_top_ratio)
            filtered = self._geometric_filter.filter(lines, width, roi_y_start)
            
            # Additional filtering to reject dense clusters (zebra crossings)
            left_lines = self._geometric_filter.filter_by_cluster_density(
                filtered.left_lines
            )
            right_lines = self._geometric_filter.filter_by_cluster_density(
                filtered.right_lines
            )
            
            # Merge parallel lines (handles double lane markings)
            left_lines = self._geometric_filter.merge_parallel_lines(left_lines)
            right_lines = self._geometric_filter.merge_parallel_lines(right_lines)
            
            # Stage 6: Polynomial fitting
            y_range = (roi_y_start, height)
            
            left_poly = self._poly_fitter.fit_from_lines(left_lines, y_range)
            right_poly = self._poly_fitter.fit_from_lines(right_lines, y_range)
            
            # Stage 7: Temporal stabilization
            left_stable, right_stable = self._stabilizer.update(
                left_poly, right_poly, frame.timestamp
            )
            
            # Stage 8: Fix intersecting lanes (remove upper crossing part)
            left_stable, right_stable = self._fix_lane_intersection(
                left_stable, right_stable, width
            )
            
            # Determine result validity
            valid = left_stable is not None and right_stable is not None
            partial = (left_stable is not None) != (right_stable is not None)
            
            latency_ms = (time.monotonic() - start_time) * 1000
            
            return LaneResult(
                left_lane=left_stable,
                right_lane=right_stable,
                valid=valid,
                partial=partial,
                timestamp=frame.timestamp,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            logger.error(f"Lane detection error: {e}")
            latency_ms = (time.monotonic() - start_time) * 1000
            return LaneResult.create_invalid(frame.timestamp, latency_ms)
    
    def _init_roi_mask(self, width: int, height: int) -> None:
        """
        Initialize the ROI mask for the given frame dimensions.
        
        Creates a trapezoidal ROI covering the lower portion of the frame.
        """
        self._frame_shape = (height, width)
        
        roi_top = int(height * self._config.roi_top_ratio)
        
        # Define trapezoid vertices
        # Narrower horizontally to focus on lane markings, not road edges
        top_width_ratio = 0.35  # Narrower at top
        bottom_width_ratio = 0.85  # Not full width at bottom - excludes road edges
        
        top_left_x = int(width * (0.5 - top_width_ratio / 2))
        top_right_x = int(width * (0.5 + top_width_ratio / 2))
        bottom_left_x = int(width * (0.5 - bottom_width_ratio / 2))
        bottom_right_x = int(width * (0.5 + bottom_width_ratio / 2))
        
        self._roi_vertices = np.array([
            [bottom_left_x, height],
            [top_left_x, roi_top],
            [top_right_x, roi_top],
            [bottom_right_x, height],
        ], dtype=np.int32)
        
        # Create mask
        self._roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(self._roi_mask, [self._roi_vertices], 255)
        
        logger.debug(f"ROI mask initialized: {width}x{height}, top={roi_top}")
    
    def _fix_lane_intersection(
        self,
        left: Optional[LanePolynomial],
        right: Optional[LanePolynomial],
        frame_width: int,
    ) -> Tuple[Optional[LanePolynomial], Optional[LanePolynomial]]:
        """
        Fix lanes that intersect by truncating the upper portion.
        
        If lanes cross (form an X shape), this removes the upper part
        above the intersection point so lanes don't look crossed.
        
        Args:
            left: Left lane polynomial
            right: Right lane polynomial
            frame_width: Width of frame for validation
            
        Returns:
            (fixed_left, fixed_right) tuple
        """
        if left is None or right is None:
            return left, right
        
        # Check if lanes are on correct sides at bottom of frame
        y_bottom = min(left.y_range[1], right.y_range[1])
        left_x_bottom = left.evaluate(y_bottom)
        right_x_bottom = right.evaluate(y_bottom)
        
        # If lanes are swapped at bottom, something is very wrong - invalidate weaker one
        if left_x_bottom > right_x_bottom:
            if left.confidence > right.confidence:
                return left, None
            else:
                return None, right
        
        # Find intersection point (where lanes actually cross)
        intersection_y = find_lane_intersection(left, right)
        
        if intersection_y is not None:
            # Truncate both lanes to start below the intersection
            left = left.with_truncated_range(intersection_y)
            right = right.with_truncated_range(intersection_y)
        
        return left, right
    
    def get_roi_vertices(self) -> Optional[np.ndarray]:
        """Get the ROI trapezoid vertices for visualization."""
        return self._roi_vertices
    
    def reset(self) -> None:
        """Reset the pipeline state (clears temporal history)."""
        self._stabilizer.reset()
        logger.debug("Lane detection pipeline reset")
    
    @property
    def is_stable(self) -> bool:
        """Check if both lanes have stable detection."""
        return self._stabilizer.is_left_stable and self._stabilizer.is_right_stable
