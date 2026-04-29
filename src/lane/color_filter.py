"""
Color filtering for lane detection.

Isolates white and yellow lane markings using HSV color space.
"""

import cv2
import numpy as np
from typing import Tuple


class ColorFilter:
    """
    Filters lane markings based on color in HSV space.
    
    Isolates white and yellow lane markings while rejecting other colors.
    """
    
    def __init__(
        self,
        white_lower: Tuple[int, int, int] = (0, 0, 180),
        white_upper: Tuple[int, int, int] = (180, 40, 255),
        yellow_lower: Tuple[int, int, int] = (10, 60, 120),
        yellow_upper: Tuple[int, int, int] = (40, 255, 255),
    ):
        """
        Initialize color filter with HSV ranges.
        
        Args:
            white_lower: Lower HSV bound for white
            white_upper: Upper HSV bound for white
            yellow_lower: Lower HSV bound for yellow
            yellow_upper: Upper HSV bound for yellow
        """
        self._white_lower = np.array(white_lower, dtype=np.uint8)
        self._white_upper = np.array(white_upper, dtype=np.uint8)
        self._yellow_lower = np.array(yellow_lower, dtype=np.uint8)
        self._yellow_upper = np.array(yellow_upper, dtype=np.uint8)
    
    def filter(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Apply color filtering to isolate lane markings.
        
        Args:
            bgr_image: Input BGR image
            
        Returns:
            Binary mask (uint8, 0 or 255) of potential lane pixels
        """
        # Convert to HSV
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # Create masks for white and yellow
        white_mask = cv2.inRange(hsv, self._white_lower, self._white_upper)
        yellow_mask = cv2.inRange(hsv, self._yellow_lower, self._yellow_upper)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        return combined_mask
    
    def filter_white(self, bgr_image: np.ndarray) -> np.ndarray:
        """Filter only white pixels."""
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self._white_lower, self._white_upper)
    
    def filter_yellow(self, bgr_image: np.ndarray) -> np.ndarray:
        """Filter only yellow pixels."""
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self._yellow_lower, self._yellow_upper)
    
    def filter_with_edges(
        self,
        bgr_image: np.ndarray,
        gaussian_kernel: Tuple[int, int] = (5, 5),
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> np.ndarray:
        """
        OPT-3: Combined color filtering and edge detection in single pass.
        
        Optimization: Single HSV conversion with fused color+edge pipeline.
        Eliminates redundant image conversions.
        
        Args:
            bgr_image: Input BGR image
            gaussian_kernel: Gaussian blur kernel size
            canny_low: Canny low threshold
            canny_high: Canny high threshold
            
        Returns:
            Binary edge image of lane candidates
        """
        # Single HSV conversion (avoids double conversion)
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # Parallel color masks
        white_mask = cv2.inRange(hsv, self._white_lower, self._white_upper)
        yellow_mask = cv2.inRange(hsv, self._yellow_lower, self._yellow_upper)
        
        # Combine with bitwise OR
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Apply blur directly to mask (skips separate edge detector preprocessing)
        blurred = cv2.GaussianBlur(combined_mask, gaussian_kernel, 0)
        
        # Edge detection on color-filtered result
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        return edges


def apply_morphology(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply morphological operations to clean up the mask.
    
    Args:
        mask: Binary input mask
        kernel_size: Size of morphological kernel
        
    Returns:
        Cleaned binary mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Dilate to connect broken line segments
    dilated = cv2.dilate(mask, kernel, iterations=2)
    
    # Use a vertical-biased kernel for closing (lanes are mostly vertical)
    close_kernel = np.ones((kernel_size + 4, kernel_size), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel)
    
    # Remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    return opened
