"""
Overtake Assistant - Advisory module for overtake safety evaluation.

IMPORTANT DISCLAIMER:
This module is advisory-only and must NEVER be relied upon for actual
driving decisions. The driver is always responsible for safe driving.

This system has significant limitations including:
- Camera blind spots
- Processing delays
- Detection errors
- Cannot see around obstacles
- Does not account for vehicle speeds

NEVER overtake based solely on this system's indication.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.lane.result import LaneResult, LanePolynomial
from src.detection.result import Detection, DetectionLabel

from .types import OvertakeStatus, OvertakeAdvisory
from .state import StateTracker
from .clearance import (
    calculate_clearance_zone,
    is_zone_valid,
    bbox_intersects_zone,
)
from .line_analysis import is_broken_line, is_broken_line_from_image, estimate_line_confidence


logger = logging.getLogger(__name__)


@dataclass
class OvertakeConfig:
    """Configuration for Overtake Assistant."""
    
    # Master enable
    enabled: bool = True
    
    # Traffic side: "left" = drive on left (overtake on right), "right" = drive on right (overtake on left)
    traffic_side: str = "left"  # Default to left-hand traffic
    
    # Enable conditions
    min_lane_confidence: float = 0.3  # Lowered for better detection
    stability_frames: int = 3
    
    # Clearance zone geometry
    zone_width_ratio: float = 1.0
    zone_y_top_ratio: float = 0.65
    
    # Safe determination
    safe_frames_required: int = 5
    
    # Lane marking detection
    # Mode: "auto" = try to detect, "assume_broken" = always allow, "assume_solid" = never allow
    line_detection_mode: str = "auto"  # Changed to auto for actual detection
    broken_line_density_threshold: float = 0.5  # Only used if mode is "auto" and no frame
    
    # Image-based line detection parameters (used when mode is "auto")
    intensity_threshold: int = 180  # Min pixel intensity to consider as lane marking
    min_gap_ratio: float = 0.15     # Min ratio of gaps to be considered broken
    min_transitions: int = 2        # Min mark->gap transitions for broken line
    
    # Display colors (BGR format)
    zone_color_safe: tuple = (0, 255, 0)       # Green
    zone_color_unsafe: tuple = (0, 165, 255)   # Orange
    zone_color_disabled: tuple = (128, 128, 128)  # Gray
    
    # Indicator position
    indicator_position: tuple = (20, 100)
    
    @property
    def overtake_side(self) -> str:
        """Return which side overtaking happens on."""
        return "right" if self.traffic_side == "left" else "left"


class OvertakeAssistant:
    """
    Advisory module for overtake safety evaluation.
    
    This is NOT a safety system. It provides advisory information only
    and should never be relied upon for actual driving decisions.
    
    The module evaluates:
    1. Lane stability and confidence
    2. Clearance zone (area to the left of our lane)
    3. Vehicles in the clearance zone
    4. Lane marking type (broken vs solid)
    
    And provides one of three statuses:
    - DISABLED: Cannot evaluate (conditions not met)
    - UNSAFE: Do not overtake
    - SAFE: Overtake may be possible (advisory only)
    """
    
    def __init__(self, config: OvertakeConfig):
        """
        Initialize Overtake Assistant.
        
        Args:
            config: Configuration parameters
        """
        self._config = config
        self._state_tracker = StateTracker(
            required_safe_frames=config.safe_frames_required,
            required_stable_frames=config.stability_frames,
        )
        self._enabled = config.enabled
        
        logger.info(
            f"OvertakeAssistant initialized: enabled={config.enabled}, "
            f"min_confidence={config.min_lane_confidence}, "
            f"safe_frames={config.safe_frames_required}"
        )
    
    @property
    def config(self) -> OvertakeConfig:
        """Return current configuration."""
        return self._config
    
    def reset(self) -> None:
        """Reset internal state."""
        self._state_tracker.reset()
    
    def evaluate(
        self,
        lane_result: LaneResult,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
        frame: Optional[np.ndarray] = None,
    ) -> OvertakeAdvisory:
        """
        Evaluate current conditions and return advisory.
        
        Args:
            lane_result: Current frame lane detection result
            detections: Current detections (may be cached from YOLO)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            frame: Optional BGR frame for image-based line detection
            
        Returns:
            OvertakeAdvisory with status and visualization data
        """
        # Check if module is enabled
        if not self._enabled:
            return self._create_disabled_advisory(
                reason="Overtake assistant disabled in config"
            )
        
        # Check enable conditions
        enable_check = self._check_enable_conditions(lane_result)
        if not enable_check[0]:
            self._state_tracker.update(
                lanes_valid=False,
                zone_clear=True,
                broken_line=True,
            )
            return self._create_disabled_advisory(reason=enable_check[1])
        
        # Calculate clearance zone
        clearance_zone = calculate_clearance_zone(
            left_lane=lane_result.left_lane,
            right_lane=lane_result.right_lane,
            frame_width=frame_width,
            frame_height=frame_height,
            zone_y_top_ratio=self._config.zone_y_top_ratio,
            zone_width_ratio=self._config.zone_width_ratio,
            overtake_side=self._config.overtake_side,
        )
        
        # Debug: log zone calculation
        if clearance_zone:
            logger.debug(f"Clearance zone: {clearance_zone}")
        else:
            logger.debug("Clearance zone: empty (calculation failed)")
        
        # Validate zone geometry
        if not is_zone_valid(clearance_zone, frame_width):
            logger.debug(f"Zone validation failed for frame {frame_width}x{frame_height}")
            self._state_tracker.update(
                lanes_valid=False,
                zone_clear=True,
                broken_line=True,
            )
            return self._create_disabled_advisory(
                reason="Clearance zone geometry invalid"
            )
        
        # Count vehicles in clearance zone
        vehicles_in_zone = self._count_vehicles_in_zone(detections, clearance_zone)
        zone_clear = (vehicles_in_zone == 0)
        
        # Check lane marking type based on config mode
        if self._config.line_detection_mode == "assume_broken":
            # Always assume broken line (most permissive for advisory)
            broken_line = True
            line_confidence = 1.0
        elif self._config.line_detection_mode == "assume_solid":
            # Always assume solid line (most conservative)
            broken_line = False
            line_confidence = 1.0
        else:
            # Auto mode: use image-based detection if frame available
            lane_to_check = (
                lane_result.right_lane 
                if self._config.overtake_side == "right" 
                else lane_result.left_lane
            )
            
            if frame is not None:
                # Use actual image analysis for better accuracy
                broken_line, line_confidence = is_broken_line_from_image(
                    frame=frame,
                    lane=lane_to_check,
                    intensity_threshold=self._config.intensity_threshold,
                    min_gap_ratio=self._config.min_gap_ratio,
                    min_transitions=self._config.min_transitions,
                )
            else:
                # Fallback to legacy heuristic
                broken_line = is_broken_line(
                    lane_to_check,
                    self._config.broken_line_density_threshold,
                )
                line_confidence = 0.5
        
        # Update state machine
        status = self._state_tracker.update(
            lanes_valid=True,
            zone_clear=zone_clear,
            broken_line=broken_line,
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(lane_result)
        
        # Generate reason string
        reason = self._generate_reason(
            status=status,
            zone_clear=zone_clear,
            broken_line=broken_line,
            vehicles_in_zone=vehicles_in_zone,
        )
        
        return OvertakeAdvisory(
            status=status,
            reason=reason,
            clearance_zone=clearance_zone,
            confidence=confidence,
            vehicles_in_zone=vehicles_in_zone,
        )
    
    def _check_enable_conditions(
        self,
        lane_result: LaneResult,
    ) -> tuple:
        """
        Check if enable conditions are met.
        
        Prioritizes the overtake-side lane (right lane for left-hand traffic).
        Will enable with reduced confidence if only the critical lane is detected.
        
        Returns:
            Tuple of (conditions_met: bool, reason: str)
        """
        # Determine which lane is critical based on traffic side
        # For left-hand traffic (overtake on right), right lane is critical
        # For right-hand traffic (overtake on left), left lane is critical
        if self._config.overtake_side == "right":
            critical_lane = lane_result.right_lane
            other_lane = lane_result.left_lane
            critical_name = "Right"
        else:
            critical_lane = lane_result.left_lane
            other_lane = lane_result.right_lane
            critical_name = "Left"
        
        # Critical lane (overtake side) must be detected
        if critical_lane is None:
            return (False, f"{critical_name} lane not detected")
        
        # Check critical lane confidence
        if critical_lane.confidence < self._config.min_lane_confidence:
            return (False, f"{critical_name} lane confidence too low ({critical_lane.confidence:.2f})")
        
        # Other lane is preferred but not strictly required
        # If missing, we can still proceed with reduced functionality
        if other_lane is None:
            # Allow operation but with a warning logged
            logger.debug(f"Operating with only {critical_name.lower()} lane detected")
        
        return (True, "")
    
    def _count_vehicles_in_zone(
        self,
        detections: List[Detection],
        clearance_zone: List[tuple],
    ) -> int:
        """Count vehicles detected within the clearance zone."""
        count = 0
        for det in detections:
            # Count vehicles and bikers as obstacles
            if det.label not in (DetectionLabel.VEHICLE, DetectionLabel.BIKER):
                continue
            
            if bbox_intersects_zone(det.bbox, clearance_zone):
                count += 1
        
        return count
    
    def _calculate_confidence(self, lane_result: LaneResult) -> float:
        """Calculate overall confidence score for the advisory."""
        if lane_result.left_lane is None or lane_result.right_lane is None:
            return 0.0
        
        left_conf = estimate_line_confidence(lane_result.left_lane)
        right_conf = estimate_line_confidence(lane_result.right_lane)
        
        # Average of both lane confidences
        return (left_conf + right_conf) / 2
    
    def _generate_reason(
        self,
        status: OvertakeStatus,
        zone_clear: bool,
        broken_line: bool,
        vehicles_in_zone: int,
    ) -> str:
        """Generate human-readable reason for the status."""
        if status == OvertakeStatus.DISABLED:
            return "Cannot evaluate - conditions not met"
        
        if status == OvertakeStatus.SAFE:
            return "Zone clear, broken line detected - overtake may be possible"
        
        # UNSAFE - explain why
        reasons = []
        if not zone_clear:
            reasons.append(f"{vehicles_in_zone} vehicle(s) in zone")
        if not broken_line:
            reasons.append("solid line detected")
        
        frames_remaining = self._state_tracker.frames_until_safe
        if frames_remaining > 0 and zone_clear and broken_line:
            reasons.append(f"waiting {frames_remaining} more frames")
        
        if reasons:
            return "Unsafe: " + ", ".join(reasons)
        return "Unsafe - conditions not stable"
    
    def _create_disabled_advisory(self, reason: str) -> OvertakeAdvisory:
        """Create a DISABLED advisory with the given reason."""
        return OvertakeAdvisory(
            status=OvertakeStatus.DISABLED,
            reason=reason,
            clearance_zone=None,
            confidence=0.0,
            vehicles_in_zone=0,
        )


def create_overtake_assistant(config_dict: dict) -> OvertakeAssistant:
    """
    Factory function to create OvertakeAssistant from config dictionary.
    
    Args:
        config_dict: Configuration dictionary (from YAML)
        
    Returns:
        Configured OvertakeAssistant instance
    """
    overtake_config = config_dict.get("overtake_assistant", {})
    
    config = OvertakeConfig(
        enabled=overtake_config.get("enabled", True),
        traffic_side=overtake_config.get("traffic_side", "left"),  # Default: left-hand traffic
        min_lane_confidence=overtake_config.get("min_lane_confidence", 0.6),
        stability_frames=overtake_config.get("stability_frames", 3),
        zone_width_ratio=overtake_config.get("zone_width_ratio", 1.0),
        zone_y_top_ratio=overtake_config.get("zone_y_top_ratio", 0.65),
        safe_frames_required=overtake_config.get("safe_frames_required", 5),
        line_detection_mode=overtake_config.get("line_detection_mode", "auto"),
        broken_line_density_threshold=overtake_config.get(
            "broken_line_density_threshold", 0.5
        ),
        intensity_threshold=overtake_config.get("intensity_threshold", 180),
        min_gap_ratio=overtake_config.get("min_gap_ratio", 0.15),
        min_transitions=overtake_config.get("min_transitions", 2),
        zone_color_safe=tuple(overtake_config.get("zone_color_safe", [0, 255, 0])),
        zone_color_unsafe=tuple(overtake_config.get("zone_color_unsafe", [0, 165, 255])),
        zone_color_disabled=tuple(overtake_config.get("zone_color_disabled", [128, 128, 128])),
        indicator_position=tuple(overtake_config.get("indicator_position", [20, 100])),
    )
    
    return OvertakeAssistant(config)
