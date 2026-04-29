"""
GPIO Buzzer Controller for Raspberry Pi.

Provides hardware buzzer control with pattern-based alerts.
"""

import logging
import threading
import time
from typing import Optional, List, Tuple
from enum import Enum

from .types import AlertType

logger = logging.getLogger(__name__)


class BuzzerPattern(Enum):
    """Predefined buzzer patterns."""
    CONTINUOUS = "continuous"
    SHORT_BEEPS = "short_beeps"
    LONG_BEEPS = "long_beeps"
    ALTERNATING = "alternating"


# Pattern definitions: List of (on_ms, off_ms) tuples
# 0 off_ms means continuous until stopped
PATTERNS = {
    BuzzerPattern.CONTINUOUS: [(500, 0)],  # Continuous for 500ms
    BuzzerPattern.SHORT_BEEPS: [(100, 100), (100, 100), (100, 100)],  # 3 short beeps
    BuzzerPattern.LONG_BEEPS: [(300, 200), (300, 200)],  # 2 long beeps
    BuzzerPattern.ALTERNATING: [(150, 100), (150, 100), (150, 100)],  # Rapid beeps
}

# Alert type to pattern mapping
ALERT_PATTERNS = {
    AlertType.COLLISION_IMMINENT: BuzzerPattern.CONTINUOUS,
    AlertType.LANE_DEPARTURE_LEFT: BuzzerPattern.SHORT_BEEPS,
    AlertType.LANE_DEPARTURE_RIGHT: BuzzerPattern.SHORT_BEEPS,
    AlertType.TRAFFIC_LIGHT_RED: BuzzerPattern.LONG_BEEPS,
    AlertType.TRAFFIC_LIGHT_YELLOW: BuzzerPattern.SHORT_BEEPS,
    AlertType.STOP_SIGN: BuzzerPattern.LONG_BEEPS,
    AlertType.SYSTEM_WARNING: BuzzerPattern.LONG_BEEPS,
}


class GPIOBuzzerController:
    """
    Controls a GPIO-connected buzzer on Raspberry Pi.
    
    Features:
    - Non-blocking pattern playback in background thread
    - Priority-based preemption
    - Graceful fallback when GPIO unavailable
    
    Usage:
        buzzer = GPIOBuzzerController(pin=18)
        if buzzer.initialize():
            buzzer.play_alert(AlertType.COLLISION_IMMINENT)
            # ... later
            buzzer.stop()
            buzzer.cleanup()
    """
    
    def __init__(self, pin: int = 18, enabled: bool = True):
        """
        Initialize GPIO buzzer controller.
        
        Args:
            pin: BCM GPIO pin number for buzzer
            enabled: Whether buzzer is enabled
        """
        self._pin = pin
        self._enabled = enabled
        self._gpio = None
        self._initialized = False
        
        # Playback state
        self._playing = False
        self._current_priority = 999
        self._stop_event = threading.Event()
        self._play_thread: Optional[threading.Thread] = None
    
    def initialize(self) -> bool:
        """
        Initialize GPIO for buzzer control.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not self._enabled:
            logger.info("GPIO buzzer disabled by configuration")
            return False
        
        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            
            # Use BCM pin numbering
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup pin as output
            GPIO.setup(self._pin, GPIO.OUT)
            GPIO.output(self._pin, GPIO.LOW)
            
            self._initialized = True
            logger.info(f"GPIO buzzer initialized on pin {self._pin}")
            return True
            
        except ImportError:
            logger.warning("RPi.GPIO not available - buzzer disabled")
            return False
        except Exception as e:
            logger.error(f"GPIO initialization failed: {e}")
            return False
    
    def play_alert(self, alert_type: AlertType) -> bool:
        """
        Play buzzer pattern for alert type.
        
        Args:
            alert_type: Type of alert to play
            
        Returns:
            True if playback started, False if skipped
        """
        if not self._initialized:
            return False
        
        # Get pattern for alert
        pattern_type = ALERT_PATTERNS.get(alert_type, BuzzerPattern.SHORT_BEEPS)
        pattern = PATTERNS.get(pattern_type, [(200, 100)])
        
        # Check priority
        priority = alert_type.priority
        if self._playing and priority >= self._current_priority:
            return False
        
        # Stop current playback
        self.stop()
        
        # Start new playback
        self._current_priority = priority
        self._stop_event.clear()
        self._play_thread = threading.Thread(
            target=self._play_pattern,
            args=(pattern,),
            daemon=True,
        )
        self._playing = True
        self._play_thread.start()
        
        return True
    
    def _play_pattern(self, pattern: List[Tuple[int, int]]) -> None:
        """Play a buzzer pattern in background thread."""
        try:
            for on_ms, off_ms in pattern:
                if self._stop_event.is_set():
                    break
                
                # Turn buzzer on
                self._gpio.output(self._pin, self._gpio.HIGH)
                
                # Wait for on duration
                if on_ms > 0:
                    # Check stop event periodically during long on periods
                    remaining = on_ms
                    while remaining > 0 and not self._stop_event.is_set():
                        sleep_time = min(remaining, 50) / 1000.0
                        time.sleep(sleep_time)
                        remaining -= 50
                
                # Turn buzzer off
                self._gpio.output(self._pin, self._gpio.LOW)
                
                # Wait for off duration (0 means continuous - no off)
                if off_ms > 0:
                    remaining = off_ms
                    while remaining > 0 and not self._stop_event.is_set():
                        sleep_time = min(remaining, 50) / 1000.0
                        time.sleep(sleep_time)
                        remaining -= 50
                        
        except Exception as e:
            logger.error(f"Buzzer playback error: {e}")
        finally:
            # Ensure buzzer is off
            try:
                self._gpio.output(self._pin, self._gpio.LOW)
            except:
                pass
            self._playing = False
            self._current_priority = 999
    
    def stop(self) -> None:
        """Stop current playback."""
        self._stop_event.set()
        
        if self._play_thread is not None and self._play_thread.is_alive():
            self._play_thread.join(timeout=0.5)
            self._play_thread = None
        
        # Ensure buzzer is off
        if self._initialized and self._gpio is not None:
            try:
                self._gpio.output(self._pin, self._gpio.LOW)
            except:
                pass
        
        self._playing = False
        self._current_priority = 999
    
    def cleanup(self) -> None:
        """Clean up GPIO resources."""
        self.stop()
        
        if self._initialized and self._gpio is not None:
            try:
                self._gpio.output(self._pin, self._gpio.LOW)
                self._gpio.cleanup(self._pin)
            except:
                pass
        
        self._initialized = False
        logger.info("GPIO buzzer cleaned up")
    
    @property
    def is_available(self) -> bool:
        """Check if buzzer is available and initialized."""
        return self._initialized
    
    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._playing
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()


class StubBuzzerController:
    """
    Stub buzzer controller for non-Raspberry Pi platforms.
    
    Logs actions instead of controlling hardware.
    """
    
    def __init__(self, pin: int = 18, enabled: bool = True):
        self._pin = pin
        self._enabled = enabled
        self._playing = False
    
    def initialize(self) -> bool:
        """Initialize stub (always succeeds if enabled)."""
        if self._enabled:
            logger.info(f"Stub buzzer initialized (pin {self._pin})")
        return self._enabled
    
    def play_alert(self, alert_type: AlertType) -> bool:
        """Log alert instead of playing."""
        if not self._enabled:
            return False
        
        logger.debug(f"Stub buzzer: {alert_type.value}")
        self._playing = True
        return True
    
    def stop(self) -> None:
        """Stop stub playback."""
        self._playing = False
    
    def cleanup(self) -> None:
        """Cleanup stub (no-op)."""
        self._playing = False
    
    @property
    def is_available(self) -> bool:
        return self._enabled
    
    @property
    def is_playing(self) -> bool:
        return self._playing


def create_buzzer_controller(pin: int = 18, enabled: bool = True) -> "GPIOBuzzerController":
    """
    Factory function to create appropriate buzzer controller.
    
    Returns GPIOBuzzerController on Raspberry Pi, StubBuzzerController otherwise.
    """
    import platform
    
    if platform.system() == "Linux":
        try:
            # Check if we're on Raspberry Pi
            with open("/proc/device-tree/model", "r") as f:
                if "Raspberry Pi" in f.read():
                    return GPIOBuzzerController(pin=pin, enabled=enabled)
        except:
            pass
    
    return StubBuzzerController(pin=pin, enabled=enabled)
