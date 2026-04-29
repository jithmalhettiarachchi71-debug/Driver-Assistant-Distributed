"""
Geometric filtering for lane line candidates.

Filters line segments based on slope, length, and position to reject
non-lane markings like zebra crossings, arrows, and stop lines.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.lane.hough_lines import LineSegment


@dataclass
class FilteredLines:
    """Result of geometric filtering."""
    left_lines: List[LineSegment]
    right_lines: List[LineSegment]
    rejected_count: int


class GeometricFilter:
    """
    Filters line segments based on geometric properties.
    
    Rejects lines that are:
    - Near-horizontal (zebra crossings, stop lines)
    - Too short
    - Have incorrect slope sign for their position
    - Inconsistent with expected lane geometry
    """
    
    def __init__(
        self,
        slope_min: float = 0.3,
        slope_max: float = 3.0,
        min_length: int = 20,
        horizontal_threshold: float = 0.2,
        angle_tolerance: float = 25.0,
    ):
        """
        Initialize geometric filter.
        
        Args:
            slope_min: Minimum absolute slope (rejects near-horizontal)
            slope_max: Maximum absolute slope (rejects near-vertical)
            min_length: Minimum line length in pixels
            horizontal_threshold: Slope below this is considered horizontal
            angle_tolerance: Max angle deviation within a lane group (degrees)
        """
        self._slope_min = slope_min
        self._slope_max = slope_max
        self._min_length = min_length
        self._horizontal_threshold = horizontal_threshold
        self._angle_tolerance = angle_tolerance
    
    def filter(
        self,
        lines: List[LineSegment],
        frame_width: int,
        roi_y_start: int,
    ) -> FilteredLines:
        """
        Filter lines into left and right lane candidates.
        
        Args:
            lines: List of detected line segments
            frame_width: Width of the frame in pixels
            roi_y_start: Y coordinate where ROI starts
            
        Returns:
            FilteredLines with separated left/right candidates
        """
        left_candidates = []
        right_candidates = []
        rejected = 0
        
        center_x = frame_width / 2
        # Widen the classification zones - allow more overlap
        # This helps when camera is not perfectly centered
        left_boundary = center_x + frame_width * 0.25  # Left lines can extend past center
        right_boundary = center_x - frame_width * 0.25  # Right lines can extend past center
        
        for line in lines:
            # Reject if too short
            if line.length < self._min_length:
                rejected += 1
                continue
            
            # Reject if no slope (vertical)
            if line.slope is None:
                rejected += 1
                continue
            
            abs_slope = abs(line.slope)
            
            # Reject horizontal lines (zebra crossings, stop lines)
            if abs_slope < self._horizontal_threshold:
                rejected += 1
                continue
            
            # Reject lines with slope outside expected range
            if abs_slope < self._slope_min or abs_slope > self._slope_max:
                rejected += 1
                continue
            
            # Get position info
            mid_x, mid_y = line.midpoint
            # Use bottom point for better spatial classification (more reliable)
            bottom_x = line.x1 if line.y1 > line.y2 else line.x2
            bottom_y = max(line.y1, line.y2)
            
            # Primary classification by SLOPE direction
            # In image coordinates (y increases downward):
            # - Left lane: lines go from bottom-left to top-right -> NEGATIVE slope
            # - Right lane: lines go from bottom-right to top-left -> POSITIVE slope
            
            if line.slope < 0:
                # Negative slope = Left lane candidate
                # Verify it's in left half of frame (with tolerance)
                if mid_x < left_boundary:
                    left_candidates.append(line)
                else:
                    rejected += 1
            else:
                # Positive slope = Right lane candidate
                # Verify it's in right half of frame (with tolerance)
                if mid_x > right_boundary:
                    right_candidates.append(line)
                else:
                    rejected += 1
        
        # Filter out inconsistent lines within each group
        left_lines = self._filter_inconsistent_lines(left_candidates)
        right_lines = self._filter_inconsistent_lines(right_candidates)
        
        return FilteredLines(left_lines, right_lines, rejected)
    
    def _filter_inconsistent_lines(
        self,
        lines: List[LineSegment],
    ) -> List[LineSegment]:
        """
        Remove lines that are inconsistent with the majority in a group.
        
        This prevents crossing lanes by ensuring all lines in a group
        have similar angles.
        """
        if len(lines) < 3:
            return lines
        
        # Get angles of all lines
        angles = [line.angle_degrees for line in lines]
        
        # Find median angle
        median_angle = np.median(angles)
        
        # Keep lines within tolerance of median
        consistent = []
        for line, angle in zip(lines, angles):
            if abs(angle - median_angle) <= self._angle_tolerance:
                consistent.append(line)
        
        return consistent if consistent else lines[:3]  # Fallback to first few lines
    
    def filter_by_cluster_density(
        self,
        lines: List[LineSegment],
        max_cluster_size: int = 5,
        position_threshold: float = 30,
    ) -> List[LineSegment]:
        """
        Filter out dense clusters of lines (likely zebra crossings or arrows).
        
        Lines that cluster too densely in similar positions are likely
        non-lane markings.
        
        Args:
            lines: List of line segments
            max_cluster_size: Maximum lines in a cluster before rejection
            position_threshold: Distance threshold for clustering
            
        Returns:
            Filtered list of lines
        """
        if len(lines) <= max_cluster_size:
            return lines
        
        # Group lines by similar midpoint positions
        clusters: List[List[LineSegment]] = []
        
        for line in lines:
            mid_x, mid_y = line.midpoint
            
            # Find if line belongs to existing cluster
            added = False
            for cluster in clusters:
                ref_x, ref_y = cluster[0].midpoint
                distance = np.sqrt((mid_x - ref_x) ** 2 + (mid_y - ref_y) ** 2)
                
                if distance < position_threshold:
                    cluster.append(line)
                    added = True
                    break
            
            if not added:
                clusters.append([line])
        
        # Keep lines from small clusters only
        result = []
        for cluster in clusters:
            if len(cluster) <= max_cluster_size:
                result.extend(cluster)
        
        return result
    
    def merge_parallel_lines(
        self,
        lines: List[LineSegment],
        angle_threshold: float = 10.0,
        distance_threshold: float = 40.0,
    ) -> List[LineSegment]:
        """
        Merge parallel lines that are close together (double lane markings).
        
        When there are 2 lane lines close together on one side, merge them
        into a single representative line.
        
        Args:
            lines: List of line segments
            angle_threshold: Max angle difference to consider parallel (degrees)
            distance_threshold: Max perpendicular distance to merge
            
        Returns:
            List with parallel lines merged
        """
        if len(lines) < 2:
            return lines
        
        # Sort by length (longest first - more reliable)
        sorted_lines = sorted(lines, key=lambda l: l.length, reverse=True)
        
        merged = []
        used = set()
        
        for i, line1 in enumerate(sorted_lines):
            if i in used:
                continue
            
            # Find lines parallel and close to this one
            parallel_group = [line1]
            
            for j, line2 in enumerate(sorted_lines[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Check if parallel (similar angle)
                angle_diff = abs(line1.angle_degrees - line2.angle_degrees)
                if angle_diff > angle_threshold:
                    continue
                
                # Check perpendicular distance between lines
                mid1_x, mid1_y = line1.midpoint
                mid2_x, mid2_y = line2.midpoint
                
                # Simple distance check (perpendicular would be more accurate but this works)
                dist = abs(mid1_x - mid2_x)  # Horizontal distance for mostly vertical lines
                
                if dist < distance_threshold:
                    parallel_group.append(line2)
                    used.add(j)
            
            # Merge the group - take weighted average position
            if len(parallel_group) > 1:
                # Create merged line using weighted average
                total_length = sum(l.length for l in parallel_group)
                avg_x1 = sum(l.x1 * l.length for l in parallel_group) / total_length
                avg_y1 = sum(l.y1 * l.length for l in parallel_group) / total_length
                avg_x2 = sum(l.x2 * l.length for l in parallel_group) / total_length
                avg_y2 = sum(l.y2 * l.length for l in parallel_group) / total_length
                
                merged_line = LineSegment.from_points(
                    int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2)
                )
                merged.append(merged_line)
            else:
                merged.append(line1)
            
            used.add(i)
        
        return merged


def compute_line_average(lines: List[LineSegment]) -> Optional[Tuple[float, float]]:
    """
    Compute average slope and intercept of a group of lines.
    
    Args:
        lines: List of line segments
        
    Returns:
        (average_slope, average_intercept) or None if empty
    """
    if not lines:
        return None
    
    slopes = []
    intercepts = []
    
    for line in lines:
        if line.slope is not None:
            slopes.append(line.slope)
            # Compute y-intercept: y = mx + b => b = y - mx
            intercepts.append(line.y1 - line.slope * line.x1)
    
    if not slopes:
        return None
    
    return (np.mean(slopes), np.mean(intercepts))


def weighted_line_average(lines: List[LineSegment]) -> Optional[Tuple[float, float]]:
    """
    Compute weighted average of lines, weighting by length.
    
    Longer lines have more influence on the average.
    
    Args:
        lines: List of line segments
        
    Returns:
        (weighted_slope, weighted_intercept) or None if empty
    """
    if not lines:
        return None
    
    total_weight = 0
    weighted_slope = 0
    weighted_intercept = 0
    
    for line in lines:
        if line.slope is not None:
            weight = line.length
            weighted_slope += line.slope * weight
            intercept = line.y1 - line.slope * line.x1
            weighted_intercept += intercept * weight
            total_weight += weight
    
    if total_weight == 0:
        return None
    
    return (weighted_slope / total_weight, weighted_intercept / total_weight)
