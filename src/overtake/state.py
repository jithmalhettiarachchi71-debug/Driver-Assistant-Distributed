"""State tracking for Overtake Assistant temporal stability."""

from dataclasses import dataclass, field
from .types import OvertakeStatus


@dataclass
class StateTracker:
    """
    Tracks temporal stability for overtake decisions.
    
    The state machine ensures that:
    1. Lanes must be stable for N frames before evaluation begins
    2. Safe conditions must persist for M frames before SAFE status
    3. Any unsafe condition immediately resets the safe counter
    
    State transitions:
        DISABLED → UNSAFE: When lanes become stable
        UNSAFE → SAFE: After N consecutive safe frames
        SAFE → UNSAFE: When any unsafe condition detected
        Any → DISABLED: When lanes become unstable
    """
    
    # Configuration
    required_safe_frames: int = 5
    required_stable_frames: int = 3
    
    # Internal state
    current_state: OvertakeStatus = field(default=OvertakeStatus.DISABLED)
    safe_frame_count: int = field(default=0)
    lane_stable_count: int = field(default=0)
    
    def reset(self) -> None:
        """Reset all state to initial values."""
        self.current_state = OvertakeStatus.DISABLED
        self.safe_frame_count = 0
        self.lane_stable_count = 0
    
    def update(
        self,
        lanes_valid: bool,
        zone_clear: bool,
        broken_line: bool,
    ) -> OvertakeStatus:
        """
        Update state based on current frame conditions.
        
        Args:
            lanes_valid: True if both lanes detected with sufficient confidence
            zone_clear: True if no vehicles detected in clearance zone
            broken_line: True if left lane marking appears to be broken/dashed
            
        Returns:
            Updated OvertakeStatus
        """
        # Track lane stability
        if lanes_valid:
            self.lane_stable_count = min(
                self.lane_stable_count + 1,
                self.required_stable_frames + 10  # Cap to prevent overflow
            )
        else:
            # Lanes lost - reset everything
            self.lane_stable_count = 0
            self.safe_frame_count = 0
            self.current_state = OvertakeStatus.DISABLED
            return self.current_state
        
        # Check if we have stable lanes (enable condition)
        if self.lane_stable_count < self.required_stable_frames:
            self.current_state = OvertakeStatus.DISABLED
            return self.current_state
        
        # Evaluate safe conditions
        # Both conditions must be true:
        # 1. Zone is clear of vehicles
        # 2. Lane marking is broken (dashed), indicating overtaking is allowed
        safe_conditions = zone_clear and broken_line
        
        if not safe_conditions:
            # Unsafe condition detected - reset safe counter
            self.safe_frame_count = 0
            self.current_state = OvertakeStatus.UNSAFE
            return self.current_state
        
        # Count consecutive safe frames
        self.safe_frame_count += 1
        
        if self.safe_frame_count >= self.required_safe_frames:
            self.current_state = OvertakeStatus.SAFE
        else:
            # Still counting - remain UNSAFE until threshold reached
            self.current_state = OvertakeStatus.UNSAFE
        
        return self.current_state
    
    @property
    def frames_until_safe(self) -> int:
        """Return frames remaining until SAFE status (0 if already SAFE or DISABLED)."""
        if self.current_state == OvertakeStatus.DISABLED:
            return 0
        remaining = self.required_safe_frames - self.safe_frame_count
        return max(0, remaining)
    
    @property
    def is_lanes_stable(self) -> bool:
        """Return True if lanes have been stable for required frames."""
        return self.lane_stable_count >= self.required_stable_frames
