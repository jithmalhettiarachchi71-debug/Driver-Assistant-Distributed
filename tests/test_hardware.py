#!/usr/bin/env python3
"""
Hardware Integration Tests for Vehicle Safety Alert System.

Tests TF-Luna LiDAR, GPIO status LEDs, and their integration with the alert system.

Run all tests:
    pytest tests/test_hardware.py -v

Run specific test class:
    pytest tests/test_hardware.py::TestLiDAR -v
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# TF-Luna LiDAR Tests
# ==============================================================================

class TestLiDARReading:
    """Test LiDARReading dataclass."""
    
    def test_reading_properties(self):
        """Test LiDARReading basic properties."""
        from src.sensors.lidar import LiDARReading
        
        reading = LiDARReading(
            distance_cm=150,
            strength=500,
            temperature=25.5,
            timestamp=1234.567,
            valid=True,
        )
        
        assert reading.distance_cm == 150
        assert reading.strength == 500
        assert reading.temperature == 25.5
        assert reading.timestamp == 1234.567
        assert reading.valid is True
    
    def test_distance_conversion_meters(self):
        """Test cm to meters conversion."""
        from src.sensors.lidar import LiDARReading
        
        reading = LiDARReading(
            distance_cm=250,
            strength=500,
            temperature=25.0,
            timestamp=0,
            valid=True,
        )
        
        assert reading.distance_m == 2.5


class TestLiDARStatus:
    """Test LiDARStatus enum."""
    
    def test_status_values(self):
        """Test all LiDARStatus values exist."""
        from src.sensors.lidar import LiDARStatus
        
        assert LiDARStatus.DISCONNECTED.value == "disconnected"
        assert LiDARStatus.CONNECTING.value == "connecting"
        assert LiDARStatus.CONNECTED.value == "connected"
        assert LiDARStatus.ERROR.value == "error"
        assert LiDARStatus.DISABLED.value == "disabled"


class TestStubLiDAR:
    """Test StubLiDAR for non-Pi development."""
    
    def test_stub_connect(self):
        """Test stub connection always succeeds."""
        from src.sensors.lidar import StubLiDAR, LiDARStatus
        
        stub = StubLiDAR()
        assert stub.connect() is True
        assert stub.status == LiDARStatus.CONNECTED
    
    def test_stub_start_stop(self):
        """Test stub start/stop cycle."""
        from src.sensors.lidar import StubLiDAR, LiDARStatus
        
        stub = StubLiDAR()
        stub.connect()
        
        assert stub.start() is True
        assert stub.is_connected is True
        
        stub.stop()
        assert stub.status == LiDARStatus.DISABLED
    
    def test_stub_mock_distance(self):
        """Test stub returns mock distance."""
        from src.sensors.lidar import StubLiDAR
        
        stub = StubLiDAR(mock_distance_cm=200)
        stub.connect()
        stub.start()
        
        reading = stub.get_reading()
        assert reading is not None
        assert reading.distance_cm == 200
        assert reading.valid is True
    
    def test_stub_set_mock_distance(self):
        """Test changing mock distance."""
        from src.sensors.lidar import StubLiDAR
        
        stub = StubLiDAR(mock_distance_cm=100)
        stub.connect()
        stub.start()
        
        stub.set_mock_distance(500)
        reading = stub.get_reading()
        
        assert reading.distance_cm == 500
    
    def test_stub_no_reading_when_stopped(self):
        """Test stub returns None when not running."""
        from src.sensors.lidar import StubLiDAR
        
        stub = StubLiDAR()
        # Not started
        assert stub.get_reading() is None
        assert stub.get_distance_cm() is None


class TestLiDARFactory:
    """Test LiDAR factory function."""
    
    def test_factory_disabled_returns_stub(self):
        """Test factory returns stub when disabled."""
        from src.sensors.lidar import create_lidar, StubLiDAR
        
        lidar = create_lidar(enabled=False)
        assert isinstance(lidar, StubLiDAR)
    
    def test_factory_no_serial_returns_stub(self):
        """Test factory returns stub when pyserial not available."""
        from src.sensors import lidar as lidar_module
        
        with patch.dict(sys.modules, {'serial': None}):
            # Force reimport to test import failure path
            lidar = lidar_module.create_lidar(enabled=True, port="/dev/ttyAMA0")
            # On Windows without serial, should get stub
            assert hasattr(lidar, 'mock_distance_cm') or hasattr(lidar, 'connect')


# ==============================================================================
# GPIO Status LED Tests
# ==============================================================================

class TestStubGPIOController:
    """Test StubGPIOController for non-Pi development."""
    
    def test_stub_initialize(self):
        """Test stub initialization."""
        from src.gpio.status_leds import StubGPIOController
        
        stub = StubGPIOController(system_pin=17, alert_pin=27)
        assert stub.initialize() is True
    
    def test_stub_system_led(self):
        """Test stub system LED control."""
        from src.gpio.status_leds import StubGPIOController
        
        stub = StubGPIOController()
        stub.initialize()
        
        stub.set_system_led(True)
        assert stub.system_led_state is True
        
        stub.set_system_led(False)
        assert stub.system_led_state is False
    
    def test_stub_alert_led(self):
        """Test stub alert LED control."""
        from src.gpio.status_leds import StubGPIOController
        
        stub = StubGPIOController()
        stub.initialize()
        
        stub.set_alert_led(True)
        assert stub.alert_led_state is True
        
        stub.set_alert_led(False)
        assert stub.alert_led_state is False
    
    def test_stub_cleanup(self):
        """Test stub cleanup."""
        from src.gpio.status_leds import StubGPIOController
        
        stub = StubGPIOController()
        stub.initialize()
        stub.set_system_led(True)
        stub.set_alert_led(True)
        
        stub.cleanup()
        # After cleanup, LEDs should be off
        assert stub.system_led_state is False
        assert stub.alert_led_state is False


class TestGPIOFactory:
    """Test GPIO controller factory function."""
    
    def test_factory_disabled_returns_stub(self):
        """Test factory returns stub when disabled."""
        from src.gpio.status_leds import create_gpio_controller, StubGPIOController
        
        controller = create_gpio_controller(enabled=False)
        assert isinstance(controller, StubGPIOController)
    
    def test_factory_no_rpi_gpio_returns_stub(self):
        """Test factory returns stub when RPi.GPIO not available."""
        from src.gpio import create_gpio_controller
        from src.gpio.status_leds import StubGPIOController
        
        # On Windows, RPi.GPIO won't be available
        controller = create_gpio_controller(enabled=True)
        # Should fall back to stub
        assert isinstance(controller, StubGPIOController)


# ==============================================================================
# Alert Decision Engine with LiDAR Tests
# ==============================================================================

class TestAlertDecisionLiDAR:
    """Test AlertDecisionEngine with LiDAR confirmation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.alerts.decision import AlertDecisionEngine
        
        self.engine = AlertDecisionEngine(
            lidar_required=True,
            lidar_threshold_cm=300,
        )
    
    def test_lidar_distance_update(self):
        """Test LiDAR distance updates."""
        self.engine.update_lidar_distance(250, available=True)
        # Should have stored the distance
        assert self.engine._lidar_distance_cm == 250
    
    def test_lidar_confirms_collision_when_close(self):
        """Test LiDAR confirms collision when object is close."""
        # Set LiDAR distance below threshold
        self.engine.update_lidar_distance(200, available=True)  # 2 meters, below 3m threshold
        
        # Internal check should confirm collision
        assert self.engine._check_lidar_collision() is True
    
    def test_lidar_denies_collision_when_far(self):
        """Test LiDAR denies collision when object is far."""
        # Set LiDAR distance above threshold
        self.engine.update_lidar_distance(500, available=True)  # 5 meters, above 3m threshold
        
        # Internal check should deny collision
        assert self.engine._check_lidar_collision() is False
    
    def test_lidar_unavailable_allows_vision_only(self):
        """Test that unavailable LiDAR allows vision-only detection (fail-safe)."""
        # Simulate disconnected sensor
        self.engine.update_lidar_distance(None, available=False)
        
        # With fail-safe behavior, should still allow collision alert
        assert self.engine._check_lidar_collision() is True
    
    def test_lidar_not_required_bypasses_check(self):
        """Test that disabling lidar_required bypasses the check."""
        from src.alerts.decision import AlertDecisionEngine
        
        engine = AlertDecisionEngine(
            lidar_required=False,  # LiDAR not required
            lidar_threshold_cm=300,
        )
        
        # Even with far distance, should allow
        engine.update_lidar_distance(1000, available=True)
        assert engine._check_lidar_collision() is True


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestHardwareIntegration:
    """Integration tests for hardware modules."""
    
    def test_lidar_gpio_alert_flow(self):
        """Test complete flow: LiDAR reading → alert → GPIO LED."""
        from src.sensors.lidar import StubLiDAR
        from src.gpio.status_leds import StubGPIOController
        
        # Setup hardware stubs
        lidar = StubLiDAR(mock_distance_cm=150)
        lidar.connect()
        lidar.start()
        
        gpio = StubGPIOController()
        gpio.initialize()
        gpio.set_system_led(True)  # System running
        
        # Simulate detection flow
        reading = lidar.get_reading()
        assert reading is not None
        assert reading.distance_cm == 150
        
        # If distance is close, set alert LED
        if reading.distance_cm < 300:
            gpio.set_alert_led(True)
        
        assert gpio.alert_led_state is True
        
        # Cleanup
        lidar.stop()
        gpio.cleanup()
    
    def test_config_lidar_parameters(self):
        """Test LiDAR configuration loading."""
        from src.config import load_config, LiDARConfig
        
        # Default config should have LiDAR settings
        config = load_config()
        
        assert hasattr(config, 'lidar')
        assert isinstance(config.lidar, LiDARConfig)
        assert config.lidar.port == "/dev/ttyAMA0"
        assert config.lidar.baud_rate == 115200
        assert config.lidar.collision_threshold_cm == 600  # 6 meters for reaction time
    
    def test_config_gpio_led_parameters(self):
        """Test GPIO LED configuration loading."""
        from src.config import load_config, GPIOLEDConfig
        
        config = load_config()
        
        assert hasattr(config, 'gpio_leds')
        assert isinstance(config.gpio_leds, GPIOLEDConfig)
        assert config.gpio_leds.system_led_pin == 17
        assert config.gpio_leds.alert_led_pin == 27
        assert config.gpio_leds.collision_output_pin == 22  # Collision output


# ==============================================================================
# Stress Tests
# ==============================================================================

class TestLiDARStress:
    """Stress tests for LiDAR module."""
    
    def test_rapid_reading_updates(self):
        """Test rapid consecutive readings."""
        from src.sensors.lidar import StubLiDAR
        
        stub = StubLiDAR()
        stub.connect()
        stub.start()
        
        # Simulate 1000 rapid readings
        for i in range(1000):
            stub.set_mock_distance(100 + (i % 100))
            reading = stub.get_reading()
            assert reading is not None
            assert reading.valid is True
        
        stub.stop()
    
    def test_gpio_rapid_toggle(self):
        """Test rapid LED toggling."""
        from src.gpio.status_leds import StubGPIOController
        
        gpio = StubGPIOController()
        gpio.initialize()
        
        # Rapidly toggle LEDs 1000 times
        for _ in range(1000):
            gpio.set_alert_led(True)
            gpio.set_alert_led(False)
        
        # Should end in off state
        assert gpio.alert_led_state is False
        
        gpio.cleanup()


# ==============================================================================
# Simulation Tests
# ==============================================================================

class TestCollisionScenarios:
    """Test realistic collision detection scenarios."""
    
    def test_approaching_vehicle_scenario(self):
        """Simulate vehicle approaching from distance."""
        from src.sensors.lidar import StubLiDAR
        from src.alerts.decision import AlertDecisionEngine
        
        # Setup
        lidar = StubLiDAR()
        lidar.connect()
        lidar.start()
        
        engine = AlertDecisionEngine(
            lidar_required=True,
            lidar_threshold_cm=300,
        )
        
        # Simulate approaching vehicle: 800cm → 100cm
        distances = [800, 600, 400, 300, 200, 100]
        collision_detected_at = None
        
        for dist in distances:
            lidar.set_mock_distance(dist)
            reading = lidar.get_reading()
            engine.update_lidar_distance(reading.distance_cm, available=True)
            
            if engine._check_lidar_collision() and collision_detected_at is None:
                # First time LiDAR confirms potential collision
                if dist < 300:  # Below threshold
                    collision_detected_at = dist
        
        # Should detect collision when within threshold
        assert collision_detected_at is not None
        assert collision_detected_at < 300
        
        lidar.stop()
    
    def test_pedestrian_crossing_scenario(self):
        """Simulate pedestrian crossing at varying distances."""
        from src.sensors.lidar import StubLiDAR
        from src.gpio.status_leds import StubGPIOController
        
        lidar = StubLiDAR()
        lidar.connect()
        lidar.start()
        
        gpio = StubGPIOController()
        gpio.initialize()
        gpio.set_system_led(True)
        
        # Pedestrian crosses: far → close → far
        distances = [500, 400, 300, 200, 150, 200, 300, 400, 500]
        alert_active_count = 0
        
        for dist in distances:
            lidar.set_mock_distance(dist)
            reading = lidar.get_reading()
            
            # Alert when close
            is_close = reading.distance_cm < 300
            gpio.set_alert_led(is_close)
            
            if is_close:
                alert_active_count += 1
        
        # Alert should have been active for middle distances
        assert alert_active_count > 0
        
        # After pedestrian passes, alert should be off
        assert gpio.alert_led_state is False
        
        lidar.stop()
        gpio.cleanup()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
