"""
Display renderer for visual overlays.

Renders detection bounding boxes, lane overlays, danger zone,
and alert banners on video frames.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import cv2

from src.detection.result import Detection, DetectionLabel
from src.lane.result import LaneResult, LanePolynomial
from src.alerts.types import AlertType, AlertEvent

# Import OvertakeAdvisory for rendering (optional dependency)
try:
    from src.overtake.types import OvertakeAdvisory, OvertakeStatus
    _OVERTAKE_AVAILABLE = True
except ImportError:
    _OVERTAKE_AVAILABLE = False
    OvertakeAdvisory = None
    OvertakeStatus = None

logger = logging.getLogger(__name__)


@dataclass
class OverlayConfig:
    """Configuration for overlay rendering."""
    # Window settings
    window_name: str = "Driver Assistant"
    
    # Colors (BGR format)
    lane_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    danger_zone_color: Tuple[int, int, int] = (0, 0, 255)  # Red
    danger_zone_alpha: float = 0.2
    
    # Detection colors by label
    detection_colors: dict = None
    
    # Drawing parameters
    bbox_thickness: int = 2
    lane_thickness: int = 3
    font_scale: float = 0.6
    font_thickness: int = 1
    show_bbox_labels: bool = True  # Show text labels on bounding boxes
    
    # Alert banner
    banner_height: int = 50
    
    def __post_init__(self):
        if self.detection_colors is None:
            self.detection_colors = {
                DetectionLabel.TRAFFIC_LIGHT_RED: (0, 0, 255),    # Red
                DetectionLabel.TRAFFIC_LIGHT_YELLOW: (0, 255, 255),  # Yellow
                DetectionLabel.TRAFFIC_LIGHT_GREEN: (0, 255, 0),  # Green
                DetectionLabel.TRAFFIC_LIGHT: (0, 200, 200),      # Generic light
                DetectionLabel.STOP_SIGN: (0, 0, 200),            # Dark Red
                DetectionLabel.PEDESTRIAN: (255, 0, 0),           # Blue
                DetectionLabel.VEHICLE: (0, 200, 0),              # Green
                DetectionLabel.BIKER: (255, 165, 0),              # Orange
            }


# Alert colors for banners
ALERT_COLORS = {
    AlertType.COLLISION_IMMINENT: (0, 0, 255),      # Red
    AlertType.LANE_DEPARTURE_LEFT: (0, 165, 255),   # Orange
    AlertType.LANE_DEPARTURE_RIGHT: (0, 165, 255),  # Orange
    AlertType.TRAFFIC_LIGHT_RED: (0, 0, 255),       # Red
    AlertType.TRAFFIC_LIGHT_YELLOW: (0, 255, 255),  # Yellow
    AlertType.TRAFFIC_LIGHT_GREEN: (0, 255, 0),     # Green
    AlertType.STOP_SIGN: (0, 0, 200),               # Dark red
    AlertType.SYSTEM_WARNING: (128, 128, 128),      # Gray
}


class DisplayRenderer:
    """
    Renders visual overlays on video frames.
    
    Features:
    - Detection bounding boxes with labels
    - Lane line overlays
    - Trapezoidal danger zone visualization
    - Alert banner with priority coloring
    - Info panel with FPS and stats
    - Non-blocking rendering in optional separate thread
    
    Usage:
        renderer = DisplayRenderer()
        renderer.initialize()
        
        # In frame loop:
        output = renderer.render(
            frame, 
            detections=detections,
            lane_result=lane_result,
            danger_zone=danger_zone_polygon,
            alert=current_alert,
            info={"fps": 15.0, "frame": 100}
        )
        
        # Show frame
        renderer.show(output)
        
        # Check for quit key
        if renderer.should_quit():
            break
        
        # On shutdown:
        renderer.cleanup()
    """
    
    def __init__(self, config: Optional[OverlayConfig] = None):
        """
        Initialize display renderer.
        
        Args:
            config: Overlay configuration (uses defaults if None)
        """
        self._config = config or OverlayConfig()
        self._window_created = False
        self._last_key = -1
        
        # Double buffer for non-blocking display
        self._display_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
    
    def initialize(self) -> bool:
        """
        Initialize the display window.
        
        Returns:
            True if initialization successful
        """
        try:
            cv2.namedWindow(self._config.window_name, cv2.WINDOW_NORMAL)
            self._window_created = True
            logger.info(f"Display window created: {self._config.window_name}")
            return True
        except Exception as e:
            logger.error(f"Display initialization failed: {e}")
            return False
    
    def render(
        self,
        frame: np.ndarray,
        detections: Optional[List[Detection]] = None,
        lane_result: Optional[LaneResult] = None,
        danger_zone: Optional[List[Tuple[int, int]]] = None,
        alert: Optional[AlertEvent] = None,
        info: Optional[dict] = None,
        collision_risks: Optional[List[Detection]] = None,
        overtake_advisory: Optional['OvertakeAdvisory'] = None,
    ) -> np.ndarray:
        """
        Render all overlays on frame.
        
        OPT-4: Optimized rendering with reduced frame copies.
        
        Args:
            frame: Input BGR frame
            detections: List of YOLO detections
            lane_result: Lane detection result
            danger_zone: Danger zone polygon vertices
            alert: Current alert event
            info: Info panel data (fps, frame count, etc.)
            collision_risks: Detections in danger zone
            overtake_advisory: Overtake assistant advisory (optional)
            
        Returns:
            Frame with overlays rendered
        """
        # OPT-4: Single copy at start, then in-place modifications where possible
        output = frame.copy()
        
        # Pre-compute collision set once (avoid repeated set creation)
        collision_set = {id(d) for d in collision_risks} if collision_risks else set()
        
        # Check if danger zone is dynamic (follows lanes)
        is_dynamic = info.get("danger_zone_dynamic", False) if info else False
        
        # Layer 1: Danger zone (requires blending)
        if danger_zone:
            output = self._draw_danger_zone(output, danger_zone, is_dynamic)
        
        # Layer 1.5: Overtake clearance zone (advisory visualization)
        if overtake_advisory and _OVERTAKE_AVAILABLE:
            output = self._draw_overtake_advisory(output, overtake_advisory)
        
        # Layer 2: Lane lines (in-place)
        if lane_result:
            self._draw_lanes_inplace(output, lane_result)
        
        # Layer 3: Detection bounding boxes (in-place)
        if detections:
            self._draw_detections_inplace(output, detections, collision_set)
        
        # Layer 4: Alert banner (top)
        if alert:
            output = self._draw_alert_banner(output, alert)
        
        # Layer 5: Info panel (corner)
        if info:
            output = self._draw_info_panel(output, info)
        
        return output
    
    def _draw_lanes_inplace(self, frame: np.ndarray, lane_result: LaneResult) -> None:
        """
        OPT-4: Draw lane lines directly on frame (no copy).
        """
        # Draw left lane
        if lane_result.left_lane:
            self._draw_lane_polynomial_inplace(
                frame,
                lane_result.left_lane,
                self._config.lane_color,
                self._config.lane_thickness,
            )
        
        # Draw right lane
        if lane_result.right_lane:
            self._draw_lane_polynomial_inplace(
                frame,
                lane_result.right_lane,
                self._config.lane_color,
                self._config.lane_thickness,
            )
        
        # Draw filled lane area if both lanes valid
        if lane_result.valid and lane_result.left_lane and lane_result.right_lane:
            self._draw_lane_fill_inplace(
                frame,
                lane_result.left_lane,
                lane_result.right_lane,
            )
    
    def _draw_lane_polynomial_inplace(
        self,
        frame: np.ndarray,
        lane: LanePolynomial,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw lane polynomial directly on frame."""
        points = lane.get_points(num_points=50)
        if len(points) < 2:
            return
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [pts], False, color, thickness, cv2.LINE_AA)
    
    def _draw_lane_fill_inplace(
        self,
        frame: np.ndarray,
        left_lane: LanePolynomial,
        right_lane: LanePolynomial,
    ) -> None:
        """Draw filled area between lanes with blending."""
        left_points = left_lane.get_points(num_points=30)
        right_points = right_lane.get_points(num_points=30)
        
        if len(left_points) < 2 or len(right_points) < 2:
            return
        
        polygon_pts = np.array(
            left_points + right_points[::-1],
            dtype=np.int32
        )
        
        # Create overlay for blending
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon_pts], (0, 100, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, dst=frame)
    
    def _draw_detections_inplace(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        collision_set: set,
    ) -> None:
        """
        OPT-4: Draw detection boxes directly on frame (no copy).
        """
        # Cache font parameters to avoid repeated attribute access
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self._config.font_scale
        font_thickness = self._config.font_thickness
        bbox_thickness = self._config.bbox_thickness
        show_labels = self._config.show_bbox_labels
        
        for det in detections:
            color = self._config.detection_colors.get(det.label, (255, 255, 255))
            x1, y1, x2, y2 = map(int, det.bbox)
            
            in_danger = id(det) in collision_set
            thickness = bbox_thickness + 2 if in_danger else bbox_thickness
            
            if in_danger:
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), thickness + 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            if show_labels:
                label = f"{det.class_name}: {det.confidence:.2f}"
                if in_danger:
                    label = "! " + label
                
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), font_thickness)

    def _draw_danger_zone(
        self,
        frame: np.ndarray,
        polygon: List[Tuple[int, int]],
        is_dynamic: bool = False,
    ) -> np.ndarray:
        """Draw semi-transparent danger zone overlay."""
        overlay = frame.copy()
        
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        
        # Use different color when zone is dynamic (lane-aligned)
        # Dynamic = cyan/teal, Fixed = default red
        if is_dynamic:
            border_color = (255, 200, 0)  # Cyan/teal for lane-aligned
            fill_color = (80, 60, 0)  # Darker cyan fill
        else:
            border_color = self._config.danger_zone_color
            fill_color = tuple(c // 3 for c in self._config.danger_zone_color)
        
        cv2.fillPoly(overlay, [pts], fill_color)
        
        # Blend
        alpha = self._config.danger_zone_alpha
        output = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw border
        cv2.polylines(output, [pts], True, border_color, 2)
        
        return output
    
    def _draw_overtake_advisory(
        self,
        frame: np.ndarray,
        advisory: 'OvertakeAdvisory',
    ) -> np.ndarray:
        """
        Draw overtake advisory visualization.
        
        Includes:
        - Clearance zone polygon (if enabled)
        - Status indicator text with reason
        
        Args:
            frame: Input frame
            advisory: Overtake advisory result
            
        Returns:
            Frame with overtake visualization
        """
        output = frame.copy()
        
        if not _OVERTAKE_AVAILABLE:
            return output
        
        # Define colors based on status
        if advisory.status == OvertakeStatus.SAFE:
            zone_color = (0, 255, 0)       # Green
            text_color = (0, 255, 0)
            status_text = "OVERTAKE: SAFE"
        elif advisory.status == OvertakeStatus.UNSAFE:
            zone_color = (0, 165, 255)     # Orange (BGR)
            text_color = (0, 165, 255)
            status_text = "OVERTAKE: UNSAFE"
        else:  # DISABLED
            zone_color = (128, 128, 128)   # Gray
            text_color = (200, 200, 200)   # Lighter gray for visibility
            status_text = "OVERTAKE: DISABLED"
        
        # Draw clearance zone if available (even when DISABLED for debugging)
        if advisory.clearance_zone:
            overlay = output.copy()
            pts = np.array(advisory.clearance_zone, np.int32).reshape((-1, 1, 2))
            
            # Fill with semi-transparent color
            fill_color = tuple(c // 4 for c in zone_color)
            cv2.fillPoly(overlay, [pts], fill_color)
            
            # Blend
            alpha = 0.3
            output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
            
            # Draw border
            cv2.polylines(output, [pts], True, zone_color, 2)
        
        # Draw status indicator box (bottom-left corner) - larger and more visible
        h, w = frame.shape[:2]
        box_x, box_y = 10, h - 80
        box_width = 280
        box_height = 70
        
        # Semi-transparent background box
        overlay_box = output.copy()
        cv2.rectangle(
            overlay_box,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            (0, 0, 0),
            -1
        )
        output = cv2.addWeighted(overlay_box, 0.7, output, 0.3, 0)
        
        # Draw border around box
        cv2.rectangle(
            output,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            text_color,
            2
        )
        
        # Draw status text (larger)
        cv2.putText(
            output,
            status_text,
            (box_x + 10, box_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2,
            cv2.LINE_AA,
        )
        
        # Draw reason (smaller, below status)
        reason_text = advisory.reason[:35] + "..." if len(advisory.reason) > 35 else advisory.reason
        cv2.putText(
            output,
            reason_text,
            (box_x + 10, box_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        
        # Draw vehicles count if any
        if advisory.vehicles_in_zone > 0:
            vehicle_text = f"Vehicles: {advisory.vehicles_in_zone}"
            cv2.putText(
                output,
                vehicle_text,
                (box_x + 180, box_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 100, 255),
                1,
                cv2.LINE_AA,
            )
        
        return output
    
    def _draw_lanes(self, frame: np.ndarray, lane_result: LaneResult) -> np.ndarray:
        """Draw lane lines."""
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw left lane
        if lane_result.left_lane:
            self._draw_lane_polynomial(
                output,
                lane_result.left_lane,
                self._config.lane_color,
                self._config.lane_thickness,
            )
        
        # Draw right lane
        if lane_result.right_lane:
            self._draw_lane_polynomial(
                output,
                lane_result.right_lane,
                self._config.lane_color,
                self._config.lane_thickness,
            )
        
        # Draw filled lane area if both lanes valid
        if lane_result.valid and lane_result.left_lane and lane_result.right_lane:
            output = self._draw_lane_fill(
                output,
                lane_result.left_lane,
                lane_result.right_lane,
            )
        
        return output
    
    def _draw_lane_polynomial(
        self,
        frame: np.ndarray,
        lane: LanePolynomial,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw a lane polynomial as a polyline."""
        points = lane.get_points(num_points=50)
        
        if len(points) < 2:
            return
        
        # Convert to int coordinates
        pts = np.array(points, dtype=np.int32)
        
        # Draw polyline
        cv2.polylines(frame, [pts], False, color, thickness, cv2.LINE_AA)
    
    def _draw_lane_fill(
        self,
        frame: np.ndarray,
        left_lane: LanePolynomial,
        right_lane: LanePolynomial,
    ) -> np.ndarray:
        """Draw filled area between lanes."""
        output = frame.copy()
        overlay = frame.copy()
        
        # Get points for both lanes
        left_points = left_lane.get_points(num_points=30)
        right_points = right_lane.get_points(num_points=30)
        
        if len(left_points) < 2 or len(right_points) < 2:
            return output
        
        # Create polygon from lane points
        # Right lane points need to be reversed to form closed polygon
        polygon_pts = np.array(
            left_points + right_points[::-1],
            dtype=np.int32
        )
        
        # Fill with green
        cv2.fillPoly(overlay, [polygon_pts], (0, 100, 0))
        
        # Blend
        return cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
    
    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        collision_set: set,
    ) -> np.ndarray:
        """Draw detection bounding boxes."""
        output = frame.copy()
        
        for det in detections:
            color = self._config.detection_colors.get(det.label, (255, 255, 255))
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Highlight if in danger zone
            in_danger = id(det) in collision_set
            thickness = self._config.bbox_thickness + 2 if in_danger else self._config.bbox_thickness
            
            if in_danger:
                # Draw red highlight behind
                cv2.rectangle(output, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), thickness + 2)
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label (if enabled)
            if self._config.show_bbox_labels:
                label = f"{det.class_name}: {det.confidence:.2f}"
                if in_danger:
                    label = "! " + label
                
                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self._config.font_scale,
                    self._config.font_thickness,
                )
                
                # Label background
                cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                
                # Label text
                cv2.putText(
                    output,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self._config.font_scale,
                    (0, 0, 0),
                    self._config.font_thickness,
                )
        
        return output
    
    def _draw_alert_banner(
        self,
        frame: np.ndarray,
        alert: AlertEvent,
    ) -> np.ndarray:
        """Draw alert banner at top of frame."""
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Get alert color and text
        color = ALERT_COLORS.get(alert.alert_type, (255, 255, 255))
        text = alert.alert_type.display_name
        
        # Draw banner background
        banner_height = self._config.banner_height
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), color, -1)
        output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)
        
        # Draw text centered
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        tx = (w - tw) // 2
        ty = (banner_height + th) // 2
        
        # Black outline for readability
        cv2.putText(output, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(output, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Flashing border for collision alerts
        if alert.priority == 1:
            if int(time.time() * 4) % 2 == 0:
                cv2.rectangle(output, (0, 0), (w-1, h-1), (0, 0, 255), 8)
        
        return output
    
    def _draw_info_panel(
        self,
        frame: np.ndarray,
        info: dict,
    ) -> np.ndarray:
        """Draw info panel in top-left corner."""
        output = frame.copy()
        
        # Build info lines
        lines = []
        
        if "fps" in info:
            lines.append(f"FPS: {info['fps']:.1f}")
        
        if "frame" in info:
            total = info.get("total", "?")
            lines.append(f"Frame: {info['frame']}/{total}")
        
        if "detections" in info:
            lines.append(f"Detections: {info['detections']}")
        
        if "latency" in info:
            lines.append(f"Latency: {info['latency']:.1f}ms")
        
        if "lane_valid" in info:
            status = "OK" if info["lane_valid"] else "INVALID"
            lines.append(f"Lanes: {status}")
        
        if "danger_zone_dynamic" in info:
            dz_status = "DYNAMIC" if info["danger_zone_dynamic"] else "FIXED"
            lines.append(f"DZ: {dz_status}")
        
        # LiDAR distance display
        if "lidar_distance_cm" in info:
            dist = info["lidar_distance_cm"]
            if dist is not None:
                lines.append(f"LiDAR: {dist:.0f}cm")
            else:
                lines.append("LiDAR: ---")
        
        if "lidar_status" in info:
            lines.append(f"LiDAR: {info['lidar_status']}")
        
        # Draw background
        padding = 10
        line_height = 20
        panel_width = 150
        panel_height = len(lines) * line_height + padding * 2
        
        overlay = output.copy()
        cv2.rectangle(overlay, (5, 5), (5 + panel_width, 5 + panel_height), (0, 0, 0), -1)
        output = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)
        
        # Draw text
        y = 5 + padding + 12
        for line in lines:
            cv2.putText(
                output,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y += line_height
        
        return output
    
    def show(self, frame: np.ndarray) -> None:
        """
        Display frame in window.
        
        Args:
            frame: Frame to display
        """
        if not self._window_created:
            return
        
        try:
            cv2.imshow(self._config.window_name, frame)
            self._last_key = cv2.waitKey(1) & 0xFF
        except Exception as e:
            logger.error(f"Display error: {e}")
    
    def should_quit(self) -> bool:
        """
        Check if quit key was pressed.
        
        Returns:
            True if 'q' or ESC was pressed
        """
        return self._last_key in (ord('q'), ord('Q'), 27)  # q, Q, or ESC
    
    def poll_key(self) -> None:
        """
        Poll for key press without displaying a frame.
        
        Use this when frames are dropping but you still need to
        check for quit commands.
        """
        if not self._window_created:
            return
        
        try:
            self._last_key = cv2.waitKey(1) & 0xFF
        except Exception:
            pass
    
    def cleanup(self) -> None:
        """Clean up display resources."""
        if self._window_created:
            try:
                cv2.destroyWindow(self._config.window_name)
            except Exception:
                # Window may already be destroyed or OpenCV not available
                pass
            self._window_created = False
        
        logger.info("Display renderer cleaned up")
    
    @property
    def is_active(self) -> bool:
        """Check if display is active."""
        return self._window_created
