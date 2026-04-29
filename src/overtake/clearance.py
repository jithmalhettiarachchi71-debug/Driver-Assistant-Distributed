"""Clearance zone geometry calculation for Overtake Assistant."""

import logging
from typing import List, Tuple, Optional
from src.lane.result import LanePolynomial

logger = logging.getLogger(__name__)


def calculate_clearance_zone(
    left_lane: Optional[LanePolynomial],
    right_lane: Optional[LanePolynomial],
    frame_width: int,
    frame_height: int,
    zone_y_top_ratio: float = 0.65,
    zone_width_ratio: float = 1.0,
    overtake_side: str = "right",
    default_lane_width: int = 150,
) -> List[Tuple[int, int]]:
    """
    Calculate the clearance zone polygon for overtake evaluation.
    
    For left-hand traffic (drive on left, overtake on right):
        The clearance zone is to the RIGHT of the right lane.
    For right-hand traffic (drive on right, overtake on left):
        The clearance zone is to the LEFT of the left lane.
    
    The zone width matches our lane width multiplied by zone_width_ratio.
    Can operate with only the critical lane detected (uses default width).
    
    Args:
        left_lane: Left lane polynomial (can be None for left-hand traffic)
        right_lane: Right lane polynomial (can be None for right-hand traffic)
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        zone_y_top_ratio: Top of zone as ratio of frame height
        zone_width_ratio: Zone width as ratio of detected lane width
        overtake_side: "right" for left-hand traffic, "left" for right-hand traffic
        default_lane_width: Default lane width in pixels when one lane is missing
        
    Returns:
        List of 4 (x, y) tuples forming a quadrilateral polygon:
        [top_left, top_right, bottom_right, bottom_left]
    """
    # Define y positions for the zone
    y_top = int(frame_height * zone_y_top_ratio)
    y_bottom = frame_height - 1
    
    # Use a reasonable minimum zone width (10% of frame width)
    min_zone_width = max(50, frame_width // 10)
    
    if overtake_side == "right":
        # Left-hand traffic: clearance zone is to the RIGHT of right lane
        # Right lane is critical, left lane is optional
        if right_lane is None:
            # Cannot proceed without the critical lane
            return []
        
        right_x_top = right_lane.evaluate(y_top)
        right_x_bottom = right_lane.evaluate(y_bottom)
        
        # Calculate lane width - use left lane if available, else default
        if left_lane is not None:
            left_x_top = left_lane.evaluate(y_top)
            left_x_bottom = left_lane.evaluate(y_bottom)
            # Only use detected width if lanes are properly ordered (right > left)
            if right_x_top > left_x_top and right_x_bottom > left_x_bottom:
                lane_width_top = right_x_top - left_x_top
                lane_width_bottom = right_x_bottom - left_x_bottom
            else:
                # Lanes appear swapped or crossed - use default
                logger.debug(f"Lanes appear crossed/swapped at y={y_top}: left_x={left_x_top:.0f}, right_x={right_x_top:.0f}")
                lane_width_top = default_lane_width
                lane_width_bottom = default_lane_width
        else:
            # Use default lane width when left lane is missing
            lane_width_top = default_lane_width
            lane_width_bottom = default_lane_width
        
        zone_width_top = max(min_zone_width, lane_width_top * zone_width_ratio)
        zone_width_bottom = max(min_zone_width, lane_width_bottom * zone_width_ratio)
        
        zone_left_top = int(right_x_top)
        zone_left_bottom = int(right_x_bottom)
        zone_right_top = min(frame_width - 1, int(right_x_top + zone_width_top))
        zone_right_bottom = min(frame_width - 1, int(right_x_bottom + zone_width_bottom))
    else:
        # Right-hand traffic: clearance zone is to the LEFT of left lane
        # Left lane is critical, right lane is optional
        if left_lane is None:
            # Cannot proceed without the critical lane
            return []
        
        left_x_top = left_lane.evaluate(y_top)
        left_x_bottom = left_lane.evaluate(y_bottom)
        
        # Calculate lane width - use right lane if available, else default
        if right_lane is not None:
            right_x_top = right_lane.evaluate(y_top)
            right_x_bottom = right_lane.evaluate(y_bottom)
            # Only use detected width if lanes are properly ordered (right > left)
            if right_x_top > left_x_top and right_x_bottom > left_x_bottom:
                lane_width_top = right_x_top - left_x_top
                lane_width_bottom = right_x_bottom - left_x_bottom
            else:
                # Lanes appear swapped or crossed - use default
                logger.debug(f"Lanes appear crossed/swapped at y={y_top}: left_x={left_x_top:.0f}, right_x={right_x_top:.0f}")
                lane_width_top = default_lane_width
                lane_width_bottom = default_lane_width
        else:
            # Use default lane width when right lane is missing
            lane_width_top = default_lane_width
            lane_width_bottom = default_lane_width
        
        zone_width_top = max(min_zone_width, lane_width_top * zone_width_ratio)
        zone_width_bottom = max(min_zone_width, lane_width_bottom * zone_width_ratio)
        
        zone_left_top = max(0, int(left_x_top - zone_width_top))
        zone_left_bottom = max(0, int(left_x_bottom - zone_width_bottom))
        zone_right_top = int(left_x_top)
        zone_right_bottom = int(left_x_bottom)
    
    return [
        (zone_left_top, y_top),           # Top left
        (zone_right_top, y_top),          # Top right
        (zone_right_bottom, y_bottom),    # Bottom right
        (zone_left_bottom, y_bottom),     # Bottom left
    ]


def is_zone_valid(
    zone: List[Tuple[int, int]],
    frame_width: int,
    min_width_px: int = 10,
) -> bool:
    """
    Validate that the clearance zone is geometrically reasonable.
    
    Args:
        zone: Polygon coordinates
        frame_width: Frame width in pixels
        min_width_px: Minimum zone width to be considered valid (lowered to 10px)
        
    Returns:
        True if zone is valid for evaluation
    """
    if len(zone) == 0:
        logger.debug("Zone invalid: empty zone")
        return False
    
    if len(zone) != 4:
        logger.debug(f"Zone invalid: expected 4 points, got {len(zone)}")
        return False
    
    top_left, top_right, bottom_right, bottom_left = zone
    
    # Check minimum widths
    top_width = top_right[0] - top_left[0]
    bottom_width = bottom_right[0] - bottom_left[0]
    
    if top_width < min_width_px or bottom_width < min_width_px:
        logger.debug(f"Zone invalid: width too small (top={top_width}, bottom={bottom_width}, min={min_width_px})")
        return False
    
    # Check that zone is at least partially within frame
    if top_right[0] < 0 and bottom_right[0] < 0:
        logger.debug("Zone invalid: entirely off-screen left")
        return False
    
    if top_left[0] > frame_width and bottom_left[0] > frame_width:
        logger.debug("Zone invalid: entirely off-screen right")
        return False
    
    return True


def point_in_clearance_zone(
    x: float,
    y: float,
    zone: List[Tuple[int, int]],
) -> bool:
    """
    Check if a point is inside the clearance zone polygon.
    
    Uses ray casting algorithm.
    
    Args:
        x: X coordinate of point
        y: Y coordinate of point
        zone: Polygon coordinates
        
    Returns:
        True if point is inside the polygon
    """
    n = len(zone)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = zone[i]
        xj, yj = zone[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def bbox_intersects_zone(
    bbox: Tuple[float, float, float, float],
    zone: List[Tuple[int, int]],
) -> bool:
    """
    Check if a bounding box intersects the clearance zone.
    
    Uses center point for simplicity (conservative approach).
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        zone: Polygon coordinates
        
    Returns:
        True if the bbox center is inside the zone
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    return point_in_clearance_zone(center_x, center_y, zone)
