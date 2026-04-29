"""
IR Distance Sensor module for proximity detection.

Provides optional ultrasonic/IR proximity sensing as supplementary input.
This is Phase 4 of development per PRD Section 14.

IMPORTANT: Vision-based collision alerts always take priority over IR.
IR sensor is advisory only and can be fully disabled.
"""

import logging
import threading
import time
from typing import Optional, List
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class IRReading:
    """Single IR sensor reading."""
    distance_cm: float
    timestamp: float
    valid: bool = True
    
    def is_in_range(self, threshold_cm: float) -> bool:
        """Check if reading indicates object within threshold."""
        return self.valid and self.distance_cm <= threshold_cm


class IRDistanceSensor:
    """
    IR/Ultrasonic distance sensor using GPIO on Raspberry Pi.
    
    Uses HC-SR04 compatible ultrasonic sensor with trigger/echo pins.
    
    Features:
    - Non-blocking background polling
    - Median filtering for noise reduction (3-sample)
    - Configurable threshold
    - Graceful failure handling
    
    Integration Rules (per PRD Section 14):
    1. IR sensor is advisory only - never overrides vision-based alerts
    2. Vision collision alerts always take priority
    3. IR may supplement with earlier warning if vision detects no risk
    4. Fully disableable via --disable-ir CLI flag
    
    Usage:
        sensor = IRDistanceSensor(trigger_pin=23, echo_pin=24)
        if sensor.initialize():
            sensor.start()
            
            # In main loop:
            reading = sensor.get_reading()
            if reading and reading.is_in_range(50):
                # Object within 50cm
                pass
            
            # On shutdown:
            sensor.stop()
            sensor.cleanup()
    """
    
    # Speed of sound in cm/us (at 20°C)
    SPEED_OF_SOUND = 0.0343
    
    # Sensor limits
    MIN_DISTANCE_CM = 2
    MAX_DISTANCE_CM = 400
    TIMEOUT_S = 0.05  # 50ms timeout for echo
    
    def __init__(
        self,
        trigger_pin: int = 23,
        echo_pin: int = 24,
        poll_interval_ms: int = 100,
        threshold_cm: float = 50.0,
        enabled: bool = True,
    ):
        """
        Initialize IR distance sensor.
        
        Args:
            trigger_pin: BCM GPIO pin for trigger
            echo_pin: BCM GPIO pin for echo
            poll_interval_ms: Polling interval in milliseconds
            threshold_cm: Distance threshold for proximity warning
            enabled: Whether sensor is enabled
        """
        self._trigger_pin = trigger_pin
        self._echo_pin = echo_pin
        self._poll_interval = poll_interval_ms / 1000.0
        self._threshold_cm = threshold_cm
        self._enabled = enabled
        
        self._gpio = None
        self._initialized = False
        
        # Polling thread
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Reading buffer (for median filtering)
        self._readings: deque = deque(maxlen=3)
        self._reading_lock = threading.Lock()
        
        # Current filtered reading
        self._current_reading: Optional[IRReading] = None
    
    def initialize(self) -> bool:
        """
        Initialize GPIO for IR sensor.
        
        Returns:
            True if initialization successful
        """
        if not self._enabled:
            logger.info("IR sensor disabled by configuration")
            return False
        
        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            
            # Use BCM pin numbering
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup pins
            GPIO.setup(self._trigger_pin, GPIO.OUT)
            GPIO.setup(self._echo_pin, GPIO.IN)
            
            # Ensure trigger is low
            GPIO.output(self._trigger_pin, GPIO.LOW)
            time.sleep(0.1)  # Let sensor settle
            
            self._initialized = True
            logger.info(
                f"IR sensor initialized (trigger={self._trigger_pin}, "
                f"echo={self._echo_pin})"
            )
            return True
            
        except ImportError:
            logger.warning("RPi.GPIO not available - IR sensor disabled")
            return False
        except Exception as e:
            logger.error(f"IR sensor initialization failed: {e}")
            return False
    
    def start(self) -> None:
        """Start background polling thread."""
        if not self._initialized:
            return
        
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="IRSensorPoll",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info("IR sensor polling started")
    
    def stop(self) -> None:
        """Stop background polling."""
        self._stop_event.set()
        
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
            self._poll_thread = None
        
        logger.info("IR sensor polling stopped")
    
    def _poll_loop(self) -> None:
        """Background polling loop."""
        while not self._stop_event.is_set():
            try:
                # Take measurement
                distance = self._measure_distance()
                timestamp = time.monotonic()
                
                if distance is not None:
                    reading = IRReading(
                        distance_cm=distance,
                        timestamp=timestamp,
                        valid=True,
                    )
                else:
                    reading = IRReading(
                        distance_cm=0.0,
                        timestamp=timestamp,
                        valid=False,
                    )
                
                with self._reading_lock:
                    self._readings.append(reading)
                    self._update_filtered_reading()
                
            except Exception as e:
                logger.debug(f"IR sensor read error: {e}")
            
            # Wait for next poll
            self._stop_event.wait(self._poll_interval)
    
    def _measure_distance(self) -> Optional[float]:
        """
        Take a single distance measurement.
        
        Returns:
            Distance in cm, or None if measurement failed
        """
        if self._gpio is None:
            return None
        
        try:
            # Send trigger pulse (10us)
            self._gpio.output(self._trigger_pin, self._gpio.HIGH)
            time.sleep(0.00001)  # 10us
            self._gpio.output(self._trigger_pin, self._gpio.LOW)
            
            # Wait for echo to start
            pulse_start = time.time()
            timeout = pulse_start + self.TIMEOUT_S
            
            while self._gpio.input(self._echo_pin) == 0:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return None
            
            # Wait for echo to end
            pulse_end = time.time()
            timeout = pulse_end + self.TIMEOUT_S
            
            while self._gpio.input(self._echo_pin) == 1:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return None
            
            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            distance = (pulse_duration * 34300) / 2  # cm
            
            # Validate range
            if distance < self.MIN_DISTANCE_CM or distance > self.MAX_DISTANCE_CM:
                return None
            
            return distance
            
        except Exception:
            return None
    
    def _update_filtered_reading(self) -> None:
        """Update filtered reading using median filter."""
        valid_readings = [r for r in self._readings if r.valid]
        
        if not valid_readings:
            self._current_reading = None
            return
        
        # Median filter
        distances = sorted(r.distance_cm for r in valid_readings)
        median_idx = len(distances) // 2
        median_distance = distances[median_idx]
        
        self._current_reading = IRReading(
            distance_cm=median_distance,
            timestamp=time.monotonic(),
            valid=True,
        )
    
    def get_reading(self) -> Optional[IRReading]:
        """
        Get current filtered distance reading.
        
        Returns:
            Current reading, or None if unavailable
        """
        with self._reading_lock:
            return self._current_reading
    
    def is_object_close(self) -> bool:
        """
        Check if an object is within threshold distance.
        
        Returns:
            True if object detected within threshold
        """
        reading = self.get_reading()
        if reading is None:
            return False
        return reading.is_in_range(self._threshold_cm)
    
    def cleanup(self) -> None:
        """Clean up GPIO resources."""
        self.stop()
        
        if self._initialized and self._gpio is not None:
            try:
                self._gpio.cleanup([self._trigger_pin, self._echo_pin])
            except:
                pass
        
        self._initialized = False
        logger.info("IR sensor cleaned up")
    
    @property
    def is_available(self) -> bool:
        """Check if sensor is available."""
        return self._initialized
    
    @property
    def threshold_cm(self) -> float:
        """Get distance threshold."""
        return self._threshold_cm
    
    @threshold_cm.setter
    def threshold_cm(self, value: float) -> None:
        """Set distance threshold."""
        self._threshold_cm = max(self.MIN_DISTANCE_CM, min(value, self.MAX_DISTANCE_CM))


class StubIRSensor:
    """
    Stub IR sensor for non-Raspberry Pi platforms.
    
    Always reports no objects in range.
    """
    
    def __init__(
        self,
        trigger_pin: int = 23,
        echo_pin: int = 24,
        poll_interval_ms: int = 100,
        threshold_cm: float = 50.0,
        enabled: bool = True,
    ):
        self._threshold_cm = threshold_cm
        self._enabled = enabled
    
    def initialize(self) -> bool:
        """Initialize stub (always returns False - no sensor available)."""
        if self._enabled:
            logger.info("Stub IR sensor - no hardware available")
        return False
    
    def start(self) -> None:
        """Start stub (no-op)."""
        pass
    
    def stop(self) -> None:
        """Stop stub (no-op)."""
        pass
    
    def get_reading(self) -> Optional[IRReading]:
        """Get reading (always None for stub)."""
        return None
    
    def is_object_close(self) -> bool:
        """Check proximity (always False for stub)."""
        return False
    
    def cleanup(self) -> None:
        """Cleanup stub (no-op)."""
        pass
    
    @property
    def is_available(self) -> bool:
        return False
    
    @property
    def threshold_cm(self) -> float:
        return self._threshold_cm


def create_ir_sensor(
    trigger_pin: int = 23,
    echo_pin: int = 24,
    poll_interval_ms: int = 100,
    threshold_cm: float = 50.0,
    enabled: bool = True,
) -> "IRDistanceSensor":
    """
    Factory function to create appropriate IR sensor.
    
    Returns IRDistanceSensor on Raspberry Pi, StubIRSensor otherwise.
    """
    import platform
    
    if not enabled:
        return StubIRSensor(
            trigger_pin=trigger_pin,
            echo_pin=echo_pin,
            poll_interval_ms=poll_interval_ms,
            threshold_cm=threshold_cm,
            enabled=False,
        )
    
    if platform.system() == "Linux":
        try:
            with open("/proc/device-tree/model", "r") as f:
                if "Raspberry Pi" in f.read():
                    return IRDistanceSensor(
                        trigger_pin=trigger_pin,
                        echo_pin=echo_pin,
                        poll_interval_ms=poll_interval_ms,
                        threshold_cm=threshold_cm,
                        enabled=True,
                    )
        except:
            pass
    
    return StubIRSensor(
        trigger_pin=trigger_pin,
        echo_pin=echo_pin,
        poll_interval_ms=poll_interval_ms,
        threshold_cm=threshold_cm,
        enabled=enabled,
    )
