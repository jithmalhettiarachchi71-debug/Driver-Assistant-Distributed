"""
Geometry utilities for polygon operations and collision detection.

Provides functions for working with polygons, bounding boxes, and intersections.
"""

from typing import List, Tuple, Optional
import numpy as np

# Type aliases
Point = Tuple[float, float]
Polygon = List[Point]
BBox = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)


def bbox_to_polygon(bbox: BBox) -> Polygon:
    """
    Convert a bounding box to a polygon.
    
    Args:
        bbox: Bounding box as (x_min, y_min, x_max, y_max)
        
    Returns:
        List of 4 corner points (clockwise from top-left)
    """
    x_min, y_min, x_max, y_max = bbox
    return [
        (x_min, y_min),  # top-left
        (x_max, y_min),  # top-right
        (x_max, y_max),  # bottom-right
        (x_min, y_max),  # bottom-left
    ]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: (x, y) coordinates
        polygon: List of (x, y) vertices
        
    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        
        j = i
    
    return inside


def polygon_intersection(poly1: Polygon, poly2: Polygon) -> bool:
    """
    Check if two polygons intersect (including if one contains the other).
    
    Uses Separating Axis Theorem (SAT) for convex polygons.
    
    Args:
        poly1: First polygon vertices
        poly2: Second polygon vertices
        
    Returns:
        True if polygons intersect
    """
    # Check if any vertex of poly1 is inside poly2
    for point in poly1:
        if point_in_polygon(point, poly2):
            return True
    
    # Check if any vertex of poly2 is inside poly1
    for point in poly2:
        if point_in_polygon(point, poly1):
            return True
    
    # Check for edge intersections
    for i in range(len(poly1)):
        edge1 = (poly1[i], poly1[(i + 1) % len(poly1)])
        for j in range(len(poly2)):
            edge2 = (poly2[j], poly2[(j + 1) % len(poly2)])
            if _edges_intersect(edge1, edge2):
                return True
    
    return False


def _edges_intersect(edge1: Tuple[Point, Point], edge2: Tuple[Point, Point]) -> bool:
    """
    Check if two line segments intersect.
    
    Args:
        edge1: First line segment as (point1, point2)
        edge2: Second line segment as (point1, point2)
        
    Returns:
        True if segments intersect
    """
    (x1, y1), (x2, y2) = edge1
    (x3, y3), (x4, y4) = edge2
    
    # Calculate direction vectors
    d1x, d1y = x2 - x1, y2 - y1
    d2x, d2y = x4 - x3, y4 - y3
    
    # Calculate cross product
    cross = d1x * d2y - d1y * d2x
    
    if abs(cross) < 1e-10:
        # Lines are parallel
        return False
    
    # Calculate intersection parameters
    t = ((x3 - x1) * d2y - (y3 - y1) * d2x) / cross
    u = ((x3 - x1) * d1y - (y3 - y1) * d1x) / cross
    
    # Check if intersection is within both segments
    return 0 <= t <= 1 and 0 <= u <= 1


def line_intersection(
    line1: Tuple[Point, Point], 
    line2: Tuple[Point, Point]
) -> Optional[Point]:
    """
    Find intersection point of two lines (extended infinitely).
    
    Args:
        line1: First line as (point1, point2)
        line2: Second line as (point1, point2)
        
    Returns:
        Intersection point, or None if lines are parallel
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None  # Lines are parallel
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (x, y)


def bbox_intersection(bbox1: BBox, bbox2: BBox) -> bool:
    """
    Check if two bounding boxes intersect.
    
    Args:
        bbox1: First bounding box (x_min, y_min, x_max, y_max)
        bbox2: Second bounding box (x_min, y_min, x_max, y_max)
        
    Returns:
        True if bounding boxes overlap
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    return (x1_min <= x2_max and x1_max >= x2_min and
            y1_min <= y2_max and y1_max >= y2_min)


def bbox_area(bbox: BBox) -> float:
    """
    Calculate area of a bounding box.
    
    Args:
        bbox: Bounding box (x_min, y_min, x_max, y_max)
        
    Returns:
        Area in pixels squared
    """
    x_min, y_min, x_max, y_max = bbox
    return max(0, x_max - x_min) * max(0, y_max - y_min)


def bbox_iou(bbox1: BBox, bbox2: BBox) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def polygon_centroid(polygon: Polygon) -> Point:
    """
    Calculate centroid of a polygon.
    
    Args:
        polygon: List of (x, y) vertices
        
    Returns:
        Centroid point (x, y)
    """
    if not polygon:
        return (0.0, 0.0)
    
    x_sum = sum(p[0] for p in polygon)
    y_sum = sum(p[1] for p in polygon)
    n = len(polygon)
    
    return (x_sum / n, y_sum / n)


def create_trapezoid_roi(
    width: int,
    height: int,
    top_width_ratio: float = 0.25,
    bottom_width_ratio: float = 0.75,
    top_y_ratio: float = 0.5,
) -> Polygon:
    """
    Create a trapezoidal region of interest.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        top_width_ratio: Width of top edge as ratio of frame width
        bottom_width_ratio: Width of bottom edge as ratio of frame width
        top_y_ratio: Y position of top edge as ratio of frame height
        
    Returns:
        Trapezoid polygon vertices
    """
    center_x = width / 2
    top_y = int(height * top_y_ratio)
    bottom_y = height
    
    top_half_width = (width * top_width_ratio) / 2
    bottom_half_width = (width * bottom_width_ratio) / 2
    
    return [
        (center_x - top_half_width, top_y),      # top-left
        (center_x + top_half_width, top_y),      # top-right
        (center_x + bottom_half_width, bottom_y), # bottom-right
        (center_x - bottom_half_width, bottom_y), # bottom-left
    ]


def create_roi_mask(
    shape: Tuple[int, int],
    polygon: Polygon
) -> np.ndarray:
    """
    Create a binary mask for a polygonal region of interest.
    
    Args:
        shape: (height, width) of the mask
        polygon: Polygon vertices
        
    Returns:
        Binary mask array (uint8, 0 or 255)
    """
    import cv2
    
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    
    return mask
