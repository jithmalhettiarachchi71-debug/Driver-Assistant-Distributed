"""Type definitions for Overtake Assistant module."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple


class OvertakeStatus(Enum):
    """
    Advisory status for overtake maneuver.
    
    IMPORTANT: This is advisory information only, not a safety system.
    """
    DISABLED = "disabled"   # Cannot evaluate (conditions not met)
    UNSAFE = "unsafe"       # Do not overtake
    SAFE = "safe"           # Overtake may be possible (advisory only)


@dataclass
class OvertakeAdvisory:
    """
    Result of overtake assistant evaluation.
    
    DISCLAIMER: This is advisory information only - NOT a safety system.
    Never rely on this for actual driving decisions.
    
    Attributes:
        status: Current overtake advisory status
        reason: Human-readable explanation of the status
        clearance_zone: Polygon coordinates for visualization (or None if disabled)
        confidence: Confidence score from 0.0 to 1.0
        vehicles_in_zone: Count of vehicles detected in the clearance zone
    """
    status: OvertakeStatus
    reason: str
    clearance_zone: Optional[List[Tuple[int, int]]]
    confidence: float
    vehicles_in_zone: int
    
    def __post_init__(self):
        """Validate advisory data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.vehicles_in_zone < 0:
            raise ValueError(f"vehicles_in_zone must be non-negative, got {self.vehicles_in_zone}")
