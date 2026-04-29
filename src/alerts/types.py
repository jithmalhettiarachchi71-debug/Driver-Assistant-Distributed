"""Alert types and event definitions."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class AlertType(Enum):
    """Alert types ordered by priority (lower number = higher priority)."""
    COLLISION_IMMINENT = "collision_imminent"      # Priority 1 - Highest
    LANE_DEPARTURE_LEFT = "lane_departure_left"    # Priority 2
    LANE_DEPARTURE_RIGHT = "lane_departure_right"  # Priority 2
    TRAFFIC_LIGHT_RED = "traffic_light_red"        # Priority 2 - Red light warning
    TRAFFIC_LIGHT_YELLOW = "traffic_light_yellow"  # Priority 3 - Yellow light caution
    TRAFFIC_LIGHT_GREEN = "traffic_light_green"    # Priority 4 - Green light info
    STOP_SIGN = "stop_sign"                        # Priority 3
    SYSTEM_WARNING = "system_warning"              # Priority 5 - Lowest
    
    @property
    def priority(self) -> int:
        """Get priority level (1=highest, 5=lowest)."""
        priority_map = {
            AlertType.COLLISION_IMMINENT: 1,
            AlertType.LANE_DEPARTURE_LEFT: 2,
            AlertType.LANE_DEPARTURE_RIGHT: 2,
            AlertType.TRAFFIC_LIGHT_RED: 2,
            AlertType.TRAFFIC_LIGHT_YELLOW: 3,
            AlertType.STOP_SIGN: 3,
            AlertType.TRAFFIC_LIGHT_GREEN: 4,
            AlertType.SYSTEM_WARNING: 5,
        }
        return priority_map.get(self, 5)
    
    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        names = {
            AlertType.COLLISION_IMMINENT: "COLLISION WARNING",
            AlertType.LANE_DEPARTURE_LEFT: "LANE DEPARTURE LEFT",
            AlertType.LANE_DEPARTURE_RIGHT: "LANE DEPARTURE RIGHT",
            AlertType.TRAFFIC_LIGHT_RED: "RED LIGHT",
            AlertType.TRAFFIC_LIGHT_YELLOW: "YELLOW LIGHT",
            AlertType.TRAFFIC_LIGHT_GREEN: "GREEN LIGHT",
            AlertType.STOP_SIGN: "STOP SIGN",
            AlertType.SYSTEM_WARNING: "SYSTEM WARNING",
        }
        return names.get(self, self.value)


@dataclass
class AlertEvent:
    """Represents an alert event."""
    alert_type: AlertType
    timestamp: float
    confidence: float = 1.0
    trigger_source: str = ""
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def priority(self) -> int:
        """Get alert priority."""
        return self.alert_type.priority
    
    def __lt__(self, other: "AlertEvent") -> bool:
        """Compare by priority for sorting (lower priority number = higher urgency)."""
        return self.priority < other.priority
