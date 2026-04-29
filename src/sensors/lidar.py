"""
TF-Luna LiDAR UART Interface Module.

Provides distance measurement from Benewake TF-Luna single-point LiDAR sensor.
Communicates via UART at 115200 baud.

Hardware Connection (Raspberry Pi):
- TF-Luna TX → GPIO 15 (RXD)
- TF-Luna RX → GPIO 14 (TXD)
- TF-Luna 5V → 5V
- TF-Luna GND → GND

UART must be enabled on the Pi:
- Add 'enable_uart=1' to /boot/config.txt
- Disable serial console in raspi-config
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class LiDARStatus(Enum):
    """Status of the LiDAR sensor."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class LiDARReading:
    """
    A single LiDAR distance reading.
    
    Attributes:
        distance_cm: Distance in centimeters (0-1200 typical range)
        strength: Signal strength (higher = more reliable)
        temperature: Sensor temperature in Celsius
        timestamp: Monotonic timestamp of reading
        valid: Whether the reading is considered valid
    """
    distance_cm: int
    strength: int
    temperature: float
    timestamp: float
    valid: bool = True
    
    @property
    def distance_m(self) -> float:
        """Distance in meters."""
        return self.distance_cm / 100.0


class TFLunaLiDAR:
    """
    UART driver for Benewake TF-Luna LiDAR sensor.
    
    Features:
    - Continuous distance parsing in background thread
    - Exponential moving average filtering
    - Sensor connection validation
    - Automatic reconnection on disconnect
    - Fail-safe behavior with status reporting
    
    TF-Luna Data Frame Format (9 bytes):
    - Byte 0: 0x59 (frame header)
    - Byte 1: 0x59 (frame header)
    - Byte 2: Dist_L (distance low byte)
    - Byte 3: Dist_H (distance high byte)
    - Byte 4: Strength_L (signal strength low byte)
    - Byte 5: Strength_H (signal strength high byte)
    - Byte 6: Temp_L (temperature low byte)
    - Byte 7: Temp_H (temperature high byte)
    - Byte 8: Checksum (sum of bytes 0-7, low 8 bits)
    
    Usage:
        lidar = TFLunaLiDAR(port="/dev/ttyAMA0")
        if lidar.connect():
            lidar.start()
            while True:
                reading = lidar.get_reading()
                if reading and reading.valid:
                    print(f"Distance: {reading.distance_cm} cm")
            lidar.stop()
    """
    
    # TF-Luna constants
    FRAME_HEADER = 0x59
    FRAME_SIZE = 9
    BAUD_RATE = 115200
    
    # Default filtering parameters
    DEFAULT_EMA_ALPHA = 0.3
    DEFAULT_MIN_STRENGTH = 100
    DEFAULT_MAX_DISTANCE_CM = 800  # 8 meters max useful range
    DEFAULT_MIN_DISTANCE_CM = 10   # 10 cm minimum
    
    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baud_rate: int = BAUD_RATE,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        min_strength: int = DEFAULT_MIN_STRENGTH,
        max_distance_cm: int = DEFAULT_MAX_DISTANCE_CM,
        min_distance_cm: int = DEFAULT_MIN_DISTANCE_CM,
        read_timeout_s: float = 0.1,
        reconnect_interval_s: float = 2.0,
        history_size: int = 10,
    ):
        """
        Initialize TF-Luna LiDAR driver.
        
        Args:
            port: UART port (e.g., /dev/ttyAMA0 or /dev/serial0)
            baud_rate: UART baud rate (default 115200)
            ema_alpha: Exponential moving average smoothing factor (0-1)
            min_strength: Minimum signal strength for valid reading
            max_distance_cm: Maximum valid distance in cm
            min_distance_cm: Minimum valid distance in cm
            read_timeout_s: UART read timeout in seconds
            reconnect_interval_s: Time between reconnection attempts
            history_size: Number of readings to keep in history
        """
        self.port = port
        self.baud_rate = baud_rate
        self.ema_alpha = ema_alpha
        self.min_strength = min_strength
        self.max_distance_cm = max_distance_cm
        self.min_distance_cm = min_distance_cm
        self.read_timeout_s = read_timeout_s
        self.reconnect_interval_s = reconnect_interval_s
        self.history_size = history_size
        
        # State
        self._serial = None
        self._status = LiDARStatus.DISCONNECTED
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Reading data
        self._current_reading: Optional[LiDARReading] = None
        self._filtered_distance: Optional[float] = None
        self._history: deque = deque(maxlen=history_size)
        self._last_valid_time: float = 0
        self._read_count: int = 0
        self._error_count: int = 0
        
        # Callbacks
        self._on_reading: Optional[Callable[[LiDARReading], None]] = None
        self._on_status_change: Optional[Callable[[LiDARStatus], None]] = None
    
    @property
    def status(self) -> LiDARStatus:
        """Get current sensor status."""
        with self._lock:
            return self._status
    
    @property
    def is_connected(self) -> bool:
        """Check if sensor is connected and running."""
        return self._status == LiDARStatus.CONNECTED
    
    @property
    def filtered_distance_cm(self) -> Optional[float]:
        """Get EMA-filtered distance in centimeters."""
        with self._lock:
            return self._filtered_distance
    
    def set_callbacks(
        self,
        on_reading: Optional[Callable[[LiDARReading], None]] = None,
        on_status_change: Optional[Callable[[LiDARStatus], None]] = None,
    ) -> None:
        """Set callback functions for readings and status changes."""
        self._on_reading = on_reading
        self._on_status_change = on_status_change
    
    def connect(self) -> bool:
        """
        Attempt to connect to the LiDAR sensor.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            import serial
        except ImportError:
            logger.error("pyserial not installed. Run: pip install pyserial")
            self._set_status(LiDARStatus.ERROR)
            return False
        
        self._set_status(LiDARStatus.CONNECTING)
        
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.read_timeout_s,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            
            # Clear any stale data
            self._serial.reset_input_buffer()
            
            # Wait briefly for sensor to start sending data
            time.sleep(0.1)
            
            # Try to read a valid frame to confirm connection
            if self._read_and_parse_frame():
                self._set_status(LiDARStatus.CONNECTED)
                logger.info(f"TF-Luna connected on {self.port}")
                return True
            else:
                logger.warning(f"TF-Luna connected but no valid data on {self.port}")
                self._set_status(LiDARStatus.CONNECTED)  # Still connected
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to TF-Luna on {self.port}: {e}")
            self._set_status(LiDARStatus.ERROR)
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the LiDAR sensor."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        self._set_status(LiDARStatus.DISCONNECTED)
    
    def start(self) -> bool:
        """
        Start continuous reading in background thread.
        
        Returns:
            True if started successfully
        """
        if self._running:
            return True
        
        if self._status not in (LiDARStatus.CONNECTED, LiDARStatus.CONNECTING):
            if not self.connect():
                return False
        
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info("TF-Luna reading thread started")
        return True
    
    def stop(self) -> None:
        """Stop continuous reading and clean up."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.disconnect()
        logger.info("TF-Luna stopped")
    
    def get_reading(self) -> Optional[LiDARReading]:
        """
        Get the most recent LiDAR reading.
        
        Returns:
            Most recent LiDARReading or None if no valid reading
        """
        with self._lock:
            return self._current_reading
    
    def get_distance_cm(self) -> Optional[int]:
        """
        Get the most recent raw distance in centimeters.
        
        Returns:
            Distance in cm or None if no valid reading
        """
        reading = self.get_reading()
        if reading and reading.valid:
            return reading.distance_cm
        return None
    
    def get_filtered_distance_cm(self) -> Optional[float]:
        """
        Get the EMA-filtered distance in centimeters.
        
        Returns:
            Filtered distance in cm or None if no valid reading
        """
        with self._lock:
            return self._filtered_distance
    
    def get_statistics(self) -> dict:
        """Get diagnostic statistics."""
        with self._lock:
            return {
                "status": self._status.value,
                "read_count": self._read_count,
                "error_count": self._error_count,
                "last_valid_time": self._last_valid_time,
                "history_size": len(self._history),
                "filtered_distance_cm": self._filtered_distance,
            }
    
    def _set_status(self, status: LiDARStatus) -> None:
        """Update status and notify callback."""
        old_status = self._status
        with self._lock:
            self._status = status
        
        if status != old_status and self._on_status_change:
            try:
                self._on_status_change(status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    def _read_loop(self) -> None:
        """Background thread for continuous reading."""
        last_reconnect_attempt = 0
        
        while self._running:
            try:
                if self._serial is None or not self._serial.is_open:
                    # Attempt reconnection with rate limiting
                    now = time.monotonic()
                    if now - last_reconnect_attempt >= self.reconnect_interval_s:
                        last_reconnect_attempt = now
                        logger.info("Attempting TF-Luna reconnection...")
                        self.connect()
                    else:
                        time.sleep(0.1)
                    continue
                
                # Read and parse a frame
                if self._read_and_parse_frame():
                    with self._lock:
                        self._read_count += 1
                else:
                    with self._lock:
                        self._error_count += 1
                        
            except Exception as e:
                logger.error(f"TF-Luna read error: {e}")
                with self._lock:
                    self._error_count += 1
                self._set_status(LiDARStatus.ERROR)
                time.sleep(0.1)
    
    def _read_and_parse_frame(self) -> bool:
        """
        Read and parse a single TF-Luna data frame.
        
        Returns:
            True if valid frame parsed, False otherwise
        """
        if self._serial is None:
            return False
        
        try:
            # Sync to frame header (0x59 0x59)
            header_found = False
            attempts = 0
            max_attempts = 100  # Prevent infinite loop
            
            while not header_found and attempts < max_attempts:
                byte1 = self._serial.read(1)
                if len(byte1) == 0:
                    return False  # Timeout
                
                if byte1[0] == self.FRAME_HEADER:
                    byte2 = self._serial.read(1)
                    if len(byte2) == 0:
                        return False
                    if byte2[0] == self.FRAME_HEADER:
                        header_found = True
                
                attempts += 1
            
            if not header_found:
                return False
            
            # Read remaining 7 bytes
            remaining = self._serial.read(7)
            if len(remaining) != 7:
                return False
            
            # Reconstruct full frame
            frame = bytes([self.FRAME_HEADER, self.FRAME_HEADER]) + remaining
            
            # Validate checksum
            checksum = sum(frame[:8]) & 0xFF
            if checksum != frame[8]:
                logger.debug(f"Checksum mismatch: calculated {checksum}, received {frame[8]}")
                return False
            
            # Parse data
            distance_cm = frame[2] | (frame[3] << 8)
            strength = frame[4] | (frame[5] << 8)
            temp_raw = frame[6] | (frame[7] << 8)
            temperature = temp_raw / 8.0 - 256.0  # Convert to Celsius
            
            timestamp = time.monotonic()
            
            # Validate reading
            valid = (
                strength >= self.min_strength and
                self.min_distance_cm <= distance_cm <= self.max_distance_cm
            )
            
            reading = LiDARReading(
                distance_cm=distance_cm,
                strength=strength,
                temperature=temperature,
                timestamp=timestamp,
                valid=valid,
            )
            
            # Update state
            with self._lock:
                self._current_reading = reading
                
                if valid:
                    self._last_valid_time = timestamp
                    self._history.append(reading)
                    
                    # Update EMA filter
                    if self._filtered_distance is None:
                        self._filtered_distance = float(distance_cm)
                    else:
                        self._filtered_distance = (
                            self.ema_alpha * distance_cm +
                            (1 - self.ema_alpha) * self._filtered_distance
                        )
            
            # Notify callback
            if self._on_reading:
                try:
                    self._on_reading(reading)
                except Exception as e:
                    logger.error(f"Reading callback error: {e}")
            
            return True
            
        except Exception as e:
            logger.debug(f"Frame parse error: {e}")
            return False


class StubLiDAR:
    """
    Stub LiDAR for testing without hardware.
    
    Returns configurable mock distance values.
    """
    
    def __init__(
        self,
        mock_distance_cm: int = 200,
        mock_strength: int = 500,
    ):
        self.mock_distance_cm = mock_distance_cm
        self.mock_strength = mock_strength
        self._status = LiDARStatus.DISABLED
        self._running = False
    
    @property
    def status(self) -> LiDARStatus:
        return self._status
    
    @property
    def is_connected(self) -> bool:
        return self._running
    
    @property
    def filtered_distance_cm(self) -> Optional[float]:
        return float(self.mock_distance_cm) if self._running else None
    
    def set_callbacks(self, on_reading=None, on_status_change=None) -> None:
        pass
    
    def connect(self) -> bool:
        self._status = LiDARStatus.CONNECTED
        return True
    
    def disconnect(self) -> None:
        self._status = LiDARStatus.DISCONNECTED
        self._running = False
    
    def start(self) -> bool:
        self._status = LiDARStatus.CONNECTED
        self._running = True
        return True
    
    def stop(self) -> None:
        self._running = False
        self._status = LiDARStatus.DISABLED
    
    def get_reading(self) -> Optional[LiDARReading]:
        if not self._running:
            return None
        return LiDARReading(
            distance_cm=self.mock_distance_cm,
            strength=self.mock_strength,
            temperature=25.0,
            timestamp=time.monotonic(),
            valid=True,
        )
    
    def get_distance_cm(self) -> Optional[int]:
        return self.mock_distance_cm if self._running else None
    
    def get_filtered_distance_cm(self) -> Optional[float]:
        return float(self.mock_distance_cm) if self._running else None
    
    def get_statistics(self) -> dict:
        return {
            "status": self._status.value,
            "read_count": 0,
            "error_count": 0,
            "last_valid_time": time.monotonic(),
            "history_size": 0,
            "filtered_distance_cm": float(self.mock_distance_cm),
        }
    
    def set_mock_distance(self, distance_cm: int) -> None:
        """Set mock distance for testing."""
        self.mock_distance_cm = distance_cm


def create_lidar(
    enabled: bool = True,
    port: str = "/dev/ttyAMA0",
    **kwargs,
) -> TFLunaLiDAR | StubLiDAR:
    """
    Factory function to create LiDAR instance.
    
    Args:
        enabled: If False, returns StubLiDAR
        port: UART port for TF-Luna
        **kwargs: Additional arguments for TFLunaLiDAR
        
    Returns:
        TFLunaLiDAR if enabled and available, StubLiDAR otherwise
    """
    if not enabled:
        logger.info("LiDAR disabled, using stub")
        return StubLiDAR()
    
    # Check if we can import serial
    try:
        import serial
    except ImportError:
        logger.warning("pyserial not available, using stub LiDAR")
        return StubLiDAR()
    
    # Check if port exists (Linux only)
    import os
    if os.name == "posix" and not os.path.exists(port):
        logger.warning(f"LiDAR port {port} not found, using stub")
        return StubLiDAR()
    
    return TFLunaLiDAR(port=port, **kwargs)
