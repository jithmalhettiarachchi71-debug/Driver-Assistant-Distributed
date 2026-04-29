"""
Edge detection for lane detection pipeline.

Applies Canny edge detection with preprocessing.
"""

import cv2
import numpy as np
from typing import Tuple


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: float = 0
) -> np.ndarray:
    """
    Apply Gaussian blur for noise reduction.
    
    Args:
        image: Input image (grayscale or color)
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Gaussian sigma (0 = auto-compute from kernel size)
        
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)


def apply_canny(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150
) -> np.ndarray:
    """
    Apply Canny edge detection.
    
    Args:
        image: Input image (grayscale)
        low_threshold: Lower threshold for hysteresis
        high_threshold: Upper threshold for hysteresis
        
    Returns:
        Binary edge image
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def apply_sobel(
    image: np.ndarray,
    ksize: int = 3,
    direction: str = "both"
) -> np.ndarray:
    """
    Apply Sobel edge detection.
    
    Args:
        image: Input grayscale image
        ksize: Sobel kernel size (1, 3, 5, or 7)
        direction: "x", "y", or "both"
        
    Returns:
        Edge magnitude image (uint8)
    """
    if direction == "x":
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    elif direction == "y":
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    else:
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    
    # Normalize to 0-255
    sobel = np.abs(sobel)
    sobel = (sobel / sobel.max() * 255).astype(np.uint8) if sobel.max() > 0 else sobel.astype(np.uint8)
    
    return sobel


class EdgeDetector:
    """
    Edge detection pipeline combining blur and Canny.
    """
    
    def __init__(
        self,
        gaussian_kernel: Tuple[int, int] = (5, 5),
        canny_low: int = 50,
        canny_high: int = 150,
    ):
        """
        Initialize edge detector.
        
        Args:
            gaussian_kernel: Gaussian blur kernel size
            canny_low: Canny low threshold
            canny_high: Canny high threshold
        """
        self._gaussian_kernel = gaussian_kernel
        self._canny_low = canny_low
        self._canny_high = canny_high
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the image.
        
        Args:
            image: Input image (grayscale or binary mask)
            
        Returns:
            Binary edge image
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = apply_gaussian_blur(gray, self._gaussian_kernel)
        
        # Apply Canny edge detection
        edges = apply_canny(blurred, self._canny_low, self._canny_high)
        
        return edges
    
    def detect_on_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Detect edges on a binary mask (skip grayscale conversion).
        
        Args:
            mask: Binary mask (uint8)
            
        Returns:
            Binary edge image
        """
        # Apply Gaussian blur
        blurred = apply_gaussian_blur(mask, self._gaussian_kernel)
        
        # Apply Canny
        edges = apply_canny(blurred, self._canny_low, self._canny_high)
        
        return edges
