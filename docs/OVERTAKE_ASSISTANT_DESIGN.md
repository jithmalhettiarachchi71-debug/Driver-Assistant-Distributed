# Overtake Assistant Module - Design Document

**Status: IMPLEMENTED** ✅

## 1. Overview

The Overtake Assistant is an **advisory-only** module that provides visual feedback to the driver indicating whether it's potentially safe to initiate an overtake maneuver. This module is **NOT safety-critical** and should never be relied upon for actual driving decisions.

### Key Principles
- Advisory only - never interferes with safety alerts
- Deterministic logic - no ML models
- Conservative defaults - DISABLED when uncertain
- Non-intrusive - display indication only, no audio alerts

### Implementation Files
- `src/overtake/types.py` - OvertakeStatus enum and OvertakeAdvisory dataclass
- `src/overtake/state.py` - StateTracker temporal state machine
- `src/overtake/clearance.py` - Clearance zone geometry calculations
- `src/overtake/line_analysis.py` - Broken/solid line detection heuristics
- `src/overtake/assistant.py` - Main OvertakeAssistant class
- `test_overtake_assistant.py` - Unit tests

---

## 2. Integration Architecture

### 2.1 Pipeline Position
The module inserts as **Stage 6.5** between Lane Departure Detection and Alert Decision:

```
Stage 5: Danger Zone Evaluation
Stage 6: Lane Departure Detection
Stage 6.5: Overtake Advisory (NEW)
Stage 7: Alert Decision
Stage 8: Alert Dispatch
...
```

### 2.2 Data Flow

```
┌─────────────────┐     ┌─────────────────┐
│   LaneResult    │────▶│                 │
│  (left_lane,    │     │                 │
│   right_lane)   │     │   Overtake      │──────▶ OvertakeAdvisory
├─────────────────┤     │   Assistant     │        (SAFE/UNSAFE/DISABLED)
│   Detections    │────▶│                 │
│  (vehicles)     │     │                 │
└─────────────────┘     └─────────────────┘
```

### 2.3 Module Independence
- Does NOT modify AlertDecisionEngine
- Does NOT generate AlertEvents
- Only produces OvertakeAdvisory for display
- AlertDecisionEngine continues to function identically

---

## 3. Output Data Contract

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple

class OvertakeStatus(Enum):
    """Advisory status for overtake maneuver."""
    DISABLED = "disabled"   # Cannot evaluate (conditions not met)
    UNSAFE = "unsafe"       # Do not overtake
    SAFE = "safe"           # Overtake may be possible

@dataclass
class OvertakeAdvisory:
    """
    Result of overtake assistant evaluation.
    
    This is advisory information only - NOT a safety system.
    """
    status: OvertakeStatus
    reason: str                          # Human-readable explanation
    clearance_zone: Optional[List[Tuple[int, int]]]  # Polygon for visualization
    confidence: float                    # 0.0 to 1.0
    vehicles_in_zone: int               # Count of vehicles detected in clearance zone
```

---

## 4. Enable Conditions (DISABLED if not met)

All of the following must be true for the module to evaluate:

| Condition | Rationale | Config Key |
|-----------|-----------|------------|
| Both lanes detected | Need lane reference for clearance zone | - |
| Lane confidence ≥ threshold | Avoid unreliable geometry | `min_lane_confidence` |
| Lanes temporally stable (N frames) | Avoid flickering | `stability_frames` |
| Oncoming lane visible | Must see clearance zone | - |

If ANY condition fails → `OvertakeStatus.DISABLED`

---

## 5. Clearance Zone Geometry

The clearance zone is the area to the LEFT of our lane (oncoming traffic area) that must be clear for a safe overtake.

### 5.1 Zone Construction

```
                    Frame Width
    ←─────────────────────────────────────→
    
    ┌─────────────────────────────────────┐
    │     ·····(top)·····                 │  y_top (horizon)
    │    ·       ┆        ·               │
    │   ·  CLEAR │         ·              │
    │  · ZONE    │our       ·             │
    │ ·          │ lane      ·            │
    │·───────────┼───────────·────────────│  y_bottom (frame bottom)
    │            │                        │
    └─────────────────────────────────────┘
         │       │
   left edge   left_lane
   (frame or    boundary
    left_lane 
    mirrored)
```

### 5.2 Zone Coordinates Calculation

```python
def calculate_clearance_zone(
    left_lane: LanePolynomial,
    frame_width: int,
    frame_height: int,
    zone_width_ratio: float = 1.0,  # Ratio of our lane width
) -> List[Tuple[int, int]]:
    """
    Calculate the clearance zone polygon.
    
    The zone extends from the left lane boundary outward by
    the same width as our current lane (symmetric mirror).
    """
    y_top = int(frame_height * 0.65)  # Match danger zone top
    y_bottom = frame_height - 1
    
    # Get left lane x positions
    left_x_top = left_lane.evaluate(y_top)
    left_x_bottom = left_lane.evaluate(y_bottom)
    
    # Calculate lane width at each y (using right lane if available)
    # Otherwise estimate based on typical lane width ratio
    lane_width_top = estimate_lane_width(y_top, ...)
    lane_width_bottom = estimate_lane_width(y_bottom, ...)
    
    # Zone extends leftward from left_lane by zone_width_ratio * lane_width
    zone_left_top = max(0, left_x_top - lane_width_top * zone_width_ratio)
    zone_left_bottom = max(0, left_x_bottom - lane_width_bottom * zone_width_ratio)
    
    return [
        (int(zone_left_top), y_top),          # Top left
        (int(left_x_top), y_top),             # Top right (at left lane)
        (int(left_x_bottom), y_bottom),       # Bottom right
        (int(zone_left_bottom), y_bottom),    # Bottom left
    ]
```

---

## 6. State Machine

### 6.1 States

```
                     ┌───────────────┐
        (enable      │               │
         conditions  │   DISABLED    │
         not met)    │               │
                     └───────┬───────┘
                             │
                             │ (conditions met)
                             ▼
                     ┌───────────────┐
        (vehicle in  │               │  (vehicle detected
         zone OR     │    UNSAFE     │◀──in clearance zone)
         broken line │               │
         not detect) └───────┬───────┘
                             │
                             │ (broken line detected AND
                             │  no vehicles in zone)
                             ▼
                     ┌───────────────┐
        (conditions  │   COUNTING    │  (internal state)
         still met   │  (N frames)   │
         & clear)    └───────┬───────┘
                             │
                             │ (N consecutive safe frames)
                             ▼
                     ┌───────────────┐
                     │               │
                     │     SAFE      │
                     │               │
                     └───────────────┘
```

### 6.2 Transition Rules

| From | To | Condition |
|------|----|-----------|
| Any | DISABLED | Enable conditions not met |
| DISABLED | UNSAFE | Enable conditions met |
| UNSAFE | COUNTING | Broken line detected AND zone clear |
| COUNTING | SAFE | N consecutive frames in COUNTING |
| COUNTING | UNSAFE | Vehicle enters zone OR solid line |
| SAFE | UNSAFE | Vehicle enters zone OR solid line |
| SAFE | DISABLED | Enable conditions fail |

### 6.3 Temporal Stability

```python
@dataclass
class StateTracker:
    """Tracks temporal stability for overtake decisions."""
    
    current_state: OvertakeStatus = OvertakeStatus.DISABLED
    safe_frame_count: int = 0
    lane_stable_count: int = 0
    required_safe_frames: int = 5  # Configurable
    required_stable_frames: int = 3  # Configurable
    
    def update(
        self,
        lanes_valid: bool,
        zone_clear: bool,
        broken_line: bool,
    ) -> OvertakeStatus:
        """Update state based on current frame conditions."""
        
        # Track lane stability
        if lanes_valid:
            self.lane_stable_count = min(
                self.lane_stable_count + 1,
                self.required_stable_frames + 1
            )
        else:
            self.lane_stable_count = 0
            self.safe_frame_count = 0
            self.current_state = OvertakeStatus.DISABLED
            return self.current_state
        
        # Check if we have stable lanes
        if self.lane_stable_count < self.required_stable_frames:
            self.current_state = OvertakeStatus.DISABLED
            return self.current_state
        
        # Evaluate safe conditions
        safe_conditions = zone_clear and broken_line
        
        if not safe_conditions:
            self.safe_frame_count = 0
            self.current_state = OvertakeStatus.UNSAFE
            return self.current_state
        
        # Count consecutive safe frames
        self.safe_frame_count += 1
        
        if self.safe_frame_count >= self.required_safe_frames:
            self.current_state = OvertakeStatus.SAFE
        else:
            # Still UNSAFE until we reach threshold
            self.current_state = OvertakeStatus.UNSAFE
        
        return self.current_state
```

---

## 7. Lane Marking Detection (Broken vs Solid)

### 7.1 Approach

Analyze the raw Hough lines before polynomial fitting to detect gaps:

```python
def is_broken_line(
    lines: List[Tuple[int, int, int, int]],  # Raw line segments
    lane_polynomial: LanePolynomial,
    tolerance_px: int = 30,
    min_gap_ratio: float = 0.2,  # Minimum gap as ratio of total length
) -> bool:
    """
    Determine if the lane marking is broken (dashed).
    
    Args:
        lines: Raw Hough line segments (x1, y1, x2, y2)
        lane_polynomial: Fitted polynomial for this lane
        tolerance_px: How close a line must be to the polynomial
        min_gap_ratio: Minimum gap ratio to consider broken
        
    Returns:
        True if line appears to be broken/dashed
    """
    # Filter lines that belong to this lane
    lane_lines = filter_lines_near_polynomial(lines, lane_polynomial, tolerance_px)
    
    if len(lane_lines) < 2:
        return False  # Can't determine, assume solid
    
    # Sort lines by y position
    sorted_lines = sort_lines_by_y(lane_lines)
    
    # Calculate total coverage vs gaps
    y_start, y_end = lane_polynomial.y_range
    total_range = y_end - y_start
    
    covered = calculate_covered_range(sorted_lines)
    gap_ratio = 1.0 - (covered / total_range)
    
    return gap_ratio >= min_gap_ratio
```

### 7.2 Simplified Alternative (Phase 1)

For initial implementation, use a simpler heuristic:

```python
def is_likely_broken_line(lane: LanePolynomial) -> bool:
    """
    Simple heuristic: broken lines have fewer detected points
    relative to their y-range.
    """
    y_range = lane.y_range[1] - lane.y_range[0]
    point_density = lane.point_count / y_range
    
    # Broken lines have lower point density due to gaps
    return point_density < BROKEN_LINE_DENSITY_THRESHOLD
```

---

## 8. Vehicle Detection in Clearance Zone

```python
def count_vehicles_in_zone(
    detections: List[Detection],
    clearance_zone: List[Tuple[int, int]],
) -> int:
    """
    Count vehicles detected within the clearance zone.
    
    Uses center point of vehicle bounding box for detection.
    """
    count = 0
    for det in detections:
        if det.label != DetectionLabel.VEHICLE:
            continue
        
        center_x, center_y = det.center
        if point_in_polygon(center_x, center_y, clearance_zone):
            count += 1
    
    return count
```

---

## 9. Configuration Parameters

```yaml
overtake_assistant:
  enabled: true
  
  # Enable conditions
  min_lane_confidence: 0.6      # Minimum lane polynomial confidence
  stability_frames: 3           # Frames of stable lanes before evaluating
  
  # Clearance zone
  zone_width_ratio: 1.0         # Zone width as ratio of own lane width
  zone_y_top_ratio: 0.65        # Top of zone (same as danger zone)
  
  # Safe determination
  safe_frames_required: 5       # Consecutive safe frames for SAFE status
  
  # Lane marking detection
  broken_line_density_threshold: 0.5  # Points per pixel threshold
  
  # Display
  zone_color_safe: [0, 255, 0]      # Green when SAFE
  zone_color_unsafe: [255, 165, 0]  # Orange when UNSAFE  
  zone_color_disabled: [128, 128, 128]  # Gray when DISABLED
  indicator_position: [20, 100]     # Screen position for text
```

---

## 10. Module Implementation Structure

```
src/
  overtake/
    __init__.py
    assistant.py      # Main OvertakeAssistant class
    clearance.py      # ClearanceZone geometry calculation
    state.py          # StateTracker state machine
    types.py          # OvertakeStatus, OvertakeAdvisory
    line_analysis.py  # Broken/solid line detection
```

### 10.1 Main Class

```python
class OvertakeAssistant:
    """
    Advisory module for overtake safety evaluation.
    
    This is NOT a safety system. It provides advisory information
    only and should never be relied upon for actual driving decisions.
    """
    
    def __init__(self, config: OvertakeConfig):
        self._config = config
        self._state_tracker = StateTracker(
            required_safe_frames=config.safe_frames_required,
            required_stable_frames=config.stability_frames,
        )
        self._enabled = config.enabled
    
    def evaluate(
        self,
        lane_result: LaneResult,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
    ) -> OvertakeAdvisory:
        """
        Evaluate current conditions and return advisory.
        
        Args:
            lane_result: Current frame lane detection
            detections: Current detections (may be cached from YOLO)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            OvertakeAdvisory with status and visualization data
        """
        if not self._enabled:
            return OvertakeAdvisory(
                status=OvertakeStatus.DISABLED,
                reason="Overtake assistant disabled in config",
                clearance_zone=None,
                confidence=0.0,
                vehicles_in_zone=0,
            )
        
        # Check enable conditions
        if not self._check_enable_conditions(lane_result):
            return self._disabled_advisory(
                reason="Lane detection not stable or confident enough"
            )
        
        # Calculate clearance zone
        clearance_zone = self._calculate_clearance_zone(
            lane_result, frame_width, frame_height
        )
        
        # Check for vehicles in zone
        vehicles_in_zone = self._count_vehicles_in_zone(
            detections, clearance_zone
        )
        zone_clear = (vehicles_in_zone == 0)
        
        # Check lane marking type (broken vs solid)
        broken_line = self._is_broken_line(lane_result.left_lane)
        
        # Update state machine
        status = self._state_tracker.update(
            lanes_valid=lane_result.valid,
            zone_clear=zone_clear,
            broken_line=broken_line,
        )
        
        # Generate advisory
        reason = self._generate_reason(status, zone_clear, broken_line, vehicles_in_zone)
        
        return OvertakeAdvisory(
            status=status,
            reason=reason,
            clearance_zone=clearance_zone,
            confidence=self._calculate_confidence(lane_result),
            vehicles_in_zone=vehicles_in_zone,
        )
```

---

## 11. Display Integration

### 11.1 Visual Elements

1. **Clearance Zone Polygon**: Semi-transparent overlay
   - Green: SAFE
   - Orange: UNSAFE
   - Gray: DISABLED

2. **Status Indicator**: Text in corner
   - "OVERTAKE: SAFE" (green)
   - "OVERTAKE: UNSAFE" (orange)  
   - "OVERTAKE: ---" (gray, when DISABLED)

3. **Vehicle Markers**: Highlight vehicles in clearance zone with distinct color

### 11.2 Renderer Update

```python
def _render_overtake_advisory(
    self,
    frame: np.ndarray,
    advisory: OvertakeAdvisory,
) -> None:
    """Render overtake advisory visualization."""
    if advisory.clearance_zone:
        color = self._get_zone_color(advisory.status)
        self._draw_polygon_overlay(
            frame, 
            advisory.clearance_zone,
            color,
            alpha=0.2,
        )
    
    # Draw status text
    text = f"OVERTAKE: {advisory.status.value.upper()}"
    color = self._get_text_color(advisory.status)
    cv2.putText(
        frame,
        text,
        self._config.indicator_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
```

---

## 12. Testing Strategy

### 12.1 Unit Tests

```python
class TestStateTracker:
    def test_starts_disabled(self):
        tracker = StateTracker()
        assert tracker.current_state == OvertakeStatus.DISABLED
    
    def test_requires_stable_lanes(self):
        tracker = StateTracker(required_stable_frames=3)
        # First 2 frames shouldn't enable
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert tracker.current_state == OvertakeStatus.DISABLED
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert tracker.current_state == OvertakeStatus.DISABLED
        # Third frame should transition to UNSAFE (not yet SAFE)
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert tracker.current_state == OvertakeStatus.UNSAFE
    
    def test_requires_consecutive_safe_frames(self):
        tracker = StateTracker(required_stable_frames=1, required_safe_frames=3)
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert tracker.current_state == OvertakeStatus.UNSAFE  # Still counting
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert tracker.current_state == OvertakeStatus.SAFE  # Now safe
    
    def test_vehicle_resets_safe_count(self):
        tracker = StateTracker(required_stable_frames=1, required_safe_frames=3)
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        # Vehicle enters zone
        tracker.update(lanes_valid=True, zone_clear=False, broken_line=True)
        assert tracker.current_state == OvertakeStatus.UNSAFE
        assert tracker.safe_frame_count == 0
```

### 12.2 Integration Tests

- Test full pipeline with simulated lane detection
- Test display rendering with mock advisory
- Test config loading and parameter application

---

## 13. Future Enhancements (Not in Initial Implementation)

1. **Oncoming vehicle speed estimation**: Track vehicle movement across frames
2. **Distance estimation**: Use object size for rough distance
3. **Road type awareness**: Different thresholds for different road types
4. **Speed-based adjustment**: Require more clearance at higher speeds
5. **Turn signal integration**: Only enable when turn signal is activated

---

## 14. Safety Disclaimer

**IMPORTANT**: This module is advisory-only and must never be relied upon for actual driving decisions. The driver is always responsible for:
- Checking mirrors
- Looking over shoulder
- Assessing actual road conditions
- Making safe driving decisions

This system has significant limitations:
- Camera blind spots
- Processing delays
- Detection errors
- Cannot see around obstacles
- Does not account for vehicle speeds

**Never overtake based solely on this system's indication.**
