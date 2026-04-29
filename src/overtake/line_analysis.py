"""Lane marking analysis for broken/solid line detection."""

from typing import Optional, Tuple
import numpy as np
from src.lane.result import LanePolynomial


def is_broken_line_from_image(
    frame: np.ndarray,
    lane: LanePolynomial,
    sample_width: int = 15,
    intensity_threshold: int = 180,
    min_gap_ratio: float = 0.15,
    min_transitions: int = 2,
) -> Tuple[bool, float]:
    """
    Detect if a lane marking is broken by analyzing actual pixel intensities.
    
    Samples pixels along the lane polynomial and looks for gaps (dark regions)
    between bright lane markings.
    
    Args:
        frame: BGR image frame
        lane: Lane polynomial to analyze
        sample_width: Width of sampling region perpendicular to lane (pixels)
        intensity_threshold: Minimum intensity to consider as "lane marking present"
        min_gap_ratio: Minimum ratio of gaps to total length to be considered broken
        min_transitions: Minimum number of mark->gap transitions for broken line
        
    Returns:
        Tuple of (is_broken: bool, confidence: float)
    """
    if frame is None or lane is None:
        return False, 0.0
    
    h, w = frame.shape[:2]
    y_start, y_end = lane.y_range
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = np.mean(frame, axis=2).astype(np.uint8)
    else:
        gray = frame
    
    # Sample points along the lane
    num_samples = min(100, y_end - y_start)
    if num_samples < 10:
        return False, 0.0
    
    y_samples = np.linspace(y_start, y_end, num_samples).astype(int)
    
    # Track intensity at each sample point
    intensities = []
    
    for y in y_samples:
        x = int(lane.evaluate(y))
        
        # Skip if out of bounds
        if x < sample_width or x >= w - sample_width or y < 0 or y >= h:
            continue
        
        # Sample a small region around the lane position
        region = gray[y, max(0, x - sample_width//2):min(w, x + sample_width//2 + 1)]
        if len(region) > 0:
            # Use max intensity in the region (lane marking is bright)
            intensities.append(np.max(region))
    
    if len(intensities) < 10:
        return False, 0.0
    
    intensities = np.array(intensities)
    
    # Classify each sample as "marking" or "gap"
    is_marking = intensities >= intensity_threshold
    
    # Count transitions (marking -> gap or gap -> marking)
    transitions = np.sum(np.abs(np.diff(is_marking.astype(int))))
    
    # Calculate gap ratio
    gap_ratio = 1.0 - (np.sum(is_marking) / len(is_marking))
    
    # Determine if broken line:
    # - Must have multiple transitions (marking -> gap -> marking pattern)
    # - Must have significant gap ratio
    is_broken = (transitions >= min_transitions) and (gap_ratio >= min_gap_ratio)
    
    # Confidence based on how clear the pattern is
    if is_broken:
        # More transitions = more confident it's broken
        confidence = min(1.0, transitions / 6.0)
    else:
        # Few transitions = confident it's solid
        confidence = min(1.0, 1.0 - (transitions / 10.0))
    
    return is_broken, max(0.0, confidence)


def is_broken_line(
    lane: LanePolynomial,
    density_threshold: float = 0.5,
) -> bool:
    """
    Legacy heuristic - use is_broken_line_from_image instead for accuracy.
    
    This uses point density which is unreliable.
    """
    y_start, y_end = lane.y_range
    y_range = y_end - y_start
    
    if y_range <= 0:
        return False
    
    point_density = lane.point_count / y_range
    return point_density < density_threshold


def estimate_line_confidence(lane: LanePolynomial) -> float:
    """
    Estimate confidence in the lane marking detection.
    
    Combines the polynomial's inherent confidence with point density.
    
    Args:
        lane: Lane polynomial
        
    Returns:
        Confidence score from 0.0 to 1.0
    """
    # Base confidence from polynomial fit
    base_confidence = lane.confidence
    
    # Adjust based on point count (more points = more confident)
    y_range = lane.y_range[1] - lane.y_range[0]
    if y_range > 0:
        density = lane.point_count / y_range
        # Normalize density to 0-1 range (assuming max useful density ~2.0)
        density_factor = min(1.0, density / 2.0)
    else:
        density_factor = 0.0
    
    # Combine factors (weighted average)
    return 0.7 * base_confidence + 0.3 * density_factor
