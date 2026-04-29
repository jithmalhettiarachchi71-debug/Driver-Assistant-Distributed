#!/usr/bin/env python3
"""
Vehicle Safety Alert System - Main Entry Point

Real-time traffic light, pedestrian, vehicle, lane departure & proximity alerts.

System Class: Safety-Critical Driver Assistance (Proof-of-Concept)
Primary Platform: Raspberry Pi 4 (8GB RAM)
Execution Mode: Headless or Graphical (configurable)
Inference Mode: CPU-only, fully on-device

Usage:
    # Raspberry Pi (headless, production)
    python -m src.main --source csi --headless
    
    # Raspberry Pi (with display, testing)
    python -m src.main --source csi --display
    
    # Windows (webcam, development)
    python -m src.main --source webcam --display
    
    # Windows (video file, debugging)
    python -m src.main --source video --video-path test.mp4 --display
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional, List

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, load_config
from src.capture import create_camera_adapter, Frame, FrameSource, CaptureConfig
from src.detection import YOLODetector, Detection, DetectionResult
from src.lane import LaneDetectionPipeline, LaneResult
from src.alerts import (
    AlertDecisionEngine, 
    AlertEvent, 
    AlertType,
    AudioAlertManager,
)
from src.alerts.gpio_buzzer import create_buzzer_controller
from src.display import DisplayRenderer, OverlayConfig
from src.telemetry import TelemetryLogger, FrameMetrics, SystemMetrics
from src.telemetry.metrics import FPSCounter
from src.sensors import create_ir_sensor, create_lidar
from src.gpio import create_gpio_controller
from src.utils.platform import is_raspberry_pi
from src.overtake import OvertakeAssistant, OvertakeAdvisory, OvertakeStatus
from src.overtake.assistant import create_overtake_assistant

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DriverAssistant:
    """
    Main application class for the Vehicle Safety Alert System.
    
    Coordinates all modules in the single-threaded main processing loop:
    1. Frame Acquisition
    2. YOLO Object Detection (scheduled, frame-skipped)
    3. Lane Detection (every frame)
    4. Trapezoidal Danger Zone Evaluation
    5. Alert Decision Logic (priority-based)
    6. Audio + Optional Visual Alert Dispatch
    """
    
    def __init__(
        self,
        config: Config,
        source: FrameSource,
        video_path: Optional[str] = None,
        ip_url: Optional[str] = None,
        display_enabled: bool = False,
        disable_ir: bool = True,
    ):
        """
        Initialize Driver Assistant.
        
        Args:
            config: System configuration
            source: Frame source type
            video_path: Path to video file (if source is VIDEO_FILE)
            ip_url: IP camera stream URL (if source is IP_CAMERA)
            display_enabled: Whether to enable graphical display
            disable_ir: Whether to disable IR sensor
        """
        self._config = config
        self._source = source
        self._video_path = video_path
        self._ip_url = ip_url
        self._display_enabled = display_enabled
        self._disable_ir = disable_ir
        
        # Running state
        self._running = False
        self._frame_count = 0
        self._dropped_frames = 0
        
        # Module instances (initialized in setup)
        self._camera = None
        self._detector = None
        self._lane_pipeline = None
        self._decision_engine = None
        self._audio_manager = None
        self._buzzer = None
        self._display = None
        self._telemetry = None
        self._ir_sensor = None
        self._overtake_assistant = None  # Advisory-only module
        self._lidar = None               # TF-Luna LiDAR sensor
        self._gpio_leds = None           # Status LED controller
        
        # Alert persistence for display (survives across skipped frames)
        self._last_display_alert = None
        self._alert_hold_counter = 0
        self._traffic_light_alert_until = 0.0  # Time-based hold for traffic light alerts
        
        # Traffic light visual persistence (separate from alerts)
        self._last_traffic_light_detection = None  # Detection object
        self._traffic_light_display_until = 0.0    # monotonic time when to stop showing
        
        # Metrics
        self._fps_counter = FPSCounter(window_size=30)
        self._system_metrics = SystemMetrics()
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def setup(self) -> bool:
        """
        Initialize all system modules.
        
        Returns:
            True if all critical modules initialized successfully
        """
        logger.info("=" * 60)
        logger.info("Vehicle Safety Alert System - Initializing")
        logger.info("=" * 60)
        
        # 1. Initialize camera
        if not self._setup_camera():
            return False
        
        # 2. Initialize YOLO detector (optional - can be disabled in config)
        if self._config.yolo.enabled:
            if not self._setup_detector():
                return False
        else:
            logger.info("YOLO detection disabled in config")
            self._detector = None
        
        # 3. Initialize lane detection
        self._setup_lane_detection()
        
        # 4. Initialize alert decision engine
        self._setup_decision_engine()
        
        # 5. Initialize audio alert system
        self._setup_audio()
        
        # 6. Initialize GPIO buzzer (optional)
        self._setup_buzzer()
        
        # 7. Initialize display (optional)
        if self._display_enabled:
            self._setup_display()
        
        # 8. Initialize telemetry
        self._setup_telemetry()
        
        # 9. Initialize IR sensor (optional, Phase 4)
        if not self._disable_ir:
            self._setup_ir_sensor()
        
        # 10. Initialize Overtake Assistant (advisory-only module)
        self._setup_overtake_assistant()
        
        # 11. Initialize LiDAR sensor (TF-Luna)
        self._setup_lidar()
        
        # 12. Initialize GPIO status LEDs
        self._setup_gpio_leds()
        
        logger.info("=" * 60)
        logger.info("System initialization complete")
        logger.info("=" * 60)
        
        # Turn on system LED to indicate successful initialization
        if self._gpio_leds and self._gpio_leds.is_available:
            self._gpio_leds.set_system_led(True)
        
        return True
    
    def _setup_camera(self) -> bool:
        """Initialize camera adapter."""
        try:
            capture_config = CaptureConfig(
                resolution=self._config.capture.resolution,
                target_fps=self._config.capture.target_fps,
                timeout_ms=self._config.capture.timeout_ms,
                source=self._source,
                video_path=self._video_path,
                ip_url=self._ip_url,
            )
            
            self._camera = create_camera_adapter(capture_config)
            
            if not self._camera.initialize():
                logger.error("Camera initialization failed")
                return False
            
            logger.info(f"Camera initialized: {self._source.value}")
            return True
            
        except Exception as e:
            logger.error(f"Camera setup failed: {e}")
            return False
    
    def _setup_detector(self) -> bool:
        """Initialize YOLO detector. CRITICAL - aborts on failure."""
        try:
            model_path = Path(PROJECT_ROOT) / self._config.yolo.model_path
            
            if not model_path.exists():
                logger.critical(f"YOLO model not found: {model_path}")
                logger.critical("System cannot start without detection model")
                return False
            
            self._detector = YOLODetector(
                model_path=str(model_path),
                conf_threshold=self._config.yolo.confidence_threshold,
                iou_threshold=self._config.yolo.iou_threshold,
                input_size=self._config.yolo.input_width,
                frame_skip=self._config.yolo.frame_skip,
            )
            
            # Warmup inference
            logger.info("Running YOLO warmup inference...")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self._detector.detect(dummy_frame)
            
            logger.info(f"YOLO detector initialized: {self._detector.model_info}")
            return True
            
        except Exception as e:
            logger.critical(f"YOLO detector initialization failed: {e}")
            return False
    
    def _setup_lane_detection(self) -> None:
        """Initialize lane detection pipeline."""
        self._lane_pipeline = LaneDetectionPipeline(self._config.lane_detection)
        logger.info("Lane detection pipeline initialized")
    
    def _setup_decision_engine(self) -> None:
        """Initialize alert decision engine."""
        width, height = self._config.capture.resolution
        
        self._decision_engine = AlertDecisionEngine(
            cooldown_ms=self._config.alerts.cooldown_ms,
            traffic_light_cooldown_ms=self._config.alerts.traffic_light_cooldown_ms,
            confidence_threshold=self._config.yolo.confidence_threshold,
            frame_width=width,
            frame_height=height,
            danger_zone_config=self._config.danger_zone,
        )
        logger.info("Alert decision engine initialized")
    
    def _setup_audio(self) -> None:
        """Initialize audio alert manager."""
        self._audio_manager = AudioAlertManager(enabled=True)
        logger.info("Audio alert manager initialized")
    
    def _setup_buzzer(self) -> None:
        """Initialize GPIO buzzer controller."""
        self._buzzer = create_buzzer_controller(
            pin=self._config.gpio.buzzer_pin,
            enabled=self._config.gpio.enabled,
        )
        if self._buzzer.initialize():
            logger.info("GPIO buzzer initialized")
        else:
            logger.info("GPIO buzzer not available (Windows or no GPIO)")
    
    def _setup_display(self) -> None:
        """Initialize display renderer."""
        overlay_config = OverlayConfig(
            window_name=self._config.display.window_name,
            lane_color=self._config.display.lane_color,
            danger_zone_color=self._config.display.danger_zone_color,
            bbox_thickness=self._config.display.bbox_thickness,
            font_scale=self._config.display.font_scale,
            show_bbox_labels=self._config.display.show_bbox_labels,
        )
        
        self._display = DisplayRenderer(overlay_config)
        if self._display.initialize():
            logger.info("Display renderer initialized")
        else:
            logger.warning("Display initialization failed - running headless")
            self._display = None
    
    def _setup_telemetry(self) -> None:
        """Initialize telemetry logger."""
        log_file = Path(PROJECT_ROOT) / self._config.system.log_file
        
        self._telemetry = TelemetryLogger(
            log_file=str(log_file),
            flush_interval=self._config.system.telemetry_flush_interval_s,
        )
        self._telemetry.start()
        logger.info(f"Telemetry logging to: {log_file}")
    
    def _setup_ir_sensor(self) -> None:
        """Initialize IR distance sensor."""
        self._ir_sensor = create_ir_sensor(
            trigger_pin=self._config.ir_sensor.gpio_trigger,
            echo_pin=self._config.ir_sensor.gpio_echo,
            poll_interval_ms=self._config.ir_sensor.poll_interval_ms,
            threshold_cm=self._config.ir_sensor.threshold_cm,
            enabled=self._config.ir_sensor.enabled,
        )
        
        if self._ir_sensor.initialize():
            self._ir_sensor.start()
            logger.info("IR distance sensor initialized")
        else:
            logger.info("IR sensor not available")
    
    def _setup_overtake_assistant(self) -> None:
        """Initialize Overtake Assistant (advisory-only module)."""
        try:
            # Load overtake config from raw config dict
            self._overtake_assistant = create_overtake_assistant(
                self._config._raw_config if hasattr(self._config, '_raw_config') else {}
            )
            
            if self._overtake_assistant.config.enabled:
                logger.info("Overtake Assistant initialized (ADVISORY ONLY)")
            else:
                logger.info("Overtake Assistant disabled in config")
                
        except Exception as e:
            logger.warning(f"Overtake Assistant setup failed: {e}")
            # Create a disabled instance
            from src.overtake.assistant import OvertakeConfig
            self._overtake_assistant = OvertakeAssistant(OvertakeConfig(enabled=False))
    
    def _setup_lidar(self) -> None:
        """Initialize TF-Luna LiDAR sensor for distance measurement."""
        self._lidar = create_lidar(
            port=self._config.lidar.port,
            baud_rate=self._config.lidar.baud_rate,
            max_distance_cm=self._config.lidar.max_distance_cm,
            min_distance_cm=self._config.lidar.min_distance_cm,
            ema_alpha=self._config.lidar.ema_alpha,
            min_strength=self._config.lidar.min_strength,
            enabled=self._config.lidar.enabled,
        )
        
        if self._lidar.connect():
            self._lidar.start()
            logger.info("TF-Luna LiDAR initialized")
        else:
            logger.info("LiDAR not available")
    
    def _setup_gpio_leds(self) -> None:
        """Initialize GPIO status LEDs."""
        self._gpio_leds = create_gpio_controller(
            system_pin=self._config.gpio_leds.system_led_pin,
            alert_pin=self._config.gpio_leds.alert_led_pin,
            collision_output_pin=self._config.gpio_leds.collision_output_pin,
            braking_output_pin=self._config.gpio_leds.braking_output_pin,
            enabled=self._config.gpio_leds.enabled,
        )
        
        if self._gpio_leds.initialize():
            logger.info("GPIO status LEDs initialized")
        else:
            logger.info("GPIO LEDs not available")
    
    def run(self) -> None:
        """
        Run the main processing loop.
        
        This is the deterministic single-threaded main loop that processes
        frames through all pipeline stages.
        """
        logger.info("Starting main processing loop...")
        self._running = True
        target_frame_time = 1.0 / self._config.capture.target_fps
        
        try:
            while self._running:
                loop_start = time.monotonic()
                
                # Process one frame
                self._process_frame()
                
                # Enforce frame timing
                elapsed = time.monotonic() - loop_start
                sleep_time = target_frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Check for quit (display mode)
                if self._display and self._display.should_quit():
                    logger.info("Quit requested via display")
                    self._running = False
                    
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            raise
        finally:
            self.cleanup()
    
    def _process_frame(self) -> None:
        """Process a single frame through the complete pipeline."""
        frame_start = time.monotonic()
        metrics = FrameMetrics(frame_seq=self._frame_count)
        
        # =====================================================================
        # Stage 1: Frame Acquisition
        # =====================================================================
        capture_start = time.monotonic()
        frame = self._camera.capture()
        metrics.capture_latency_ms = (time.monotonic() - capture_start) * 1000
        
        # Collect IP camera specific metrics if applicable
        if self._source == FrameSource.IP_CAMERA:
            from src.capture.ip_camera import IPCameraAdapter
            if isinstance(self._camera, IPCameraAdapter):
                ip_latency, ip_reconnects, ip_downtime = self._camera.get_ip_metrics()
                metrics.ip_acquisition_latency_ms = ip_latency
                metrics.ip_reconnect_count = ip_reconnects
                metrics.ip_downtime_ms = ip_downtime
        
        if frame is None:
            self._dropped_frames += 1
            metrics.dropped_frames = self._dropped_frames
            logger.warning(f"Frame dropped (total: {self._dropped_frames})")
            
            # Still check for quit key even when frames are dropping
            if self._display and self._display.is_active:
                self._display.poll_key()
            
            return
        
        metrics.timestamp = frame.timestamp
        
        # =====================================================================
        # Stage 2: Lane Detection (every frame)
        # =====================================================================
        lane_start = time.monotonic()
        lane_result = self._lane_pipeline.process(frame)
        metrics.lane_latency_ms = (time.monotonic() - lane_start) * 1000
        metrics.lane_valid = lane_result.valid
        metrics.lane_partial = lane_result.partial
        
        # =====================================================================
        # Stage 3: YOLO Object Detection (scheduled with frame skipping)
        # =====================================================================
        if self._detector is not None:
            yolo_start = time.monotonic()
            detection_result = self._detector.detect(frame.data)
            
            if not detection_result.from_cache:
                metrics.yolo_latency_ms = (time.monotonic() - yolo_start) * 1000
                metrics.yolo_skipped = False
            else:
                metrics.yolo_skipped = True
            
            metrics.detections_count = len(detection_result.detections)
            detections = detection_result.detections
        else:
            # YOLO disabled - empty detections
            metrics.yolo_skipped = True
            detections = []
            metrics.detections_count = 0
        
        # =====================================================================
        # Stage 4: Update Dynamic Danger Zone (from lane detection)
        # =====================================================================
        # Must update danger zone BEFORE evaluating collision risks
        if lane_result.valid:
            self._decision_engine.danger_zone.update_from_lanes(
                left_lane=lane_result.left_lane,
                right_lane=lane_result.right_lane,
                frame_width=frame.width,
                frame_height=frame.height,
            )
        
        # =====================================================================
        # Stage 5: Danger Zone Evaluation (collision risks)
        # =====================================================================
        collision_risks = self._evaluate_collision_risks(detections)
        metrics.collision_risks = len(collision_risks)
        
        # =====================================================================
        # Stage 5.5: Collision Output GPIO
        # =====================================================================
        # Set GPIO pin HIGH when object/collision is detected in danger zone
        if self._gpio_leds is not None:
            self._gpio_leds.set_collision_output(len(collision_risks) > 0)
        
        # =====================================================================
        # Stage 6: Lane Departure Detection
        # =====================================================================
        lane_departure = self._check_lane_departure(lane_result, frame.width)
        
        # =====================================================================
        # Stage 6.5: Overtake Advisory (advisory-only, does NOT affect alerts)
        # =====================================================================
        overtake_advisory = self._evaluate_overtake_advisory(
            lane_result, detections, frame.width, frame.height, frame.data
        )
        
        # =====================================================================
        # Stage 6.6: LiDAR Distance Update
        # =====================================================================
        # Update decision engine with current LiDAR distance for collision confirmation
        if self._lidar is not None:
            lidar_reading = self._lidar.get_reading()
            # HIGH FIX: Always update LiDAR status, not just when reading is valid
            # This ensures _lidar_available is correctly set to True
            if lidar_reading and lidar_reading.valid:
                self._decision_engine.update_lidar_distance(
                    lidar_reading.distance_cm, available=True
                )
                # HIGH FIX: Add LiDAR data to frame metrics for telemetry
                metrics.lidar_distance_cm = lidar_reading.distance_cm
                metrics.lidar_strength = lidar_reading.strength
                metrics.lidar_valid = True
            else:
                # LiDAR connected but no valid reading - still mark as available
                # so the fail-safe logic in _check_lidar_collision works correctly
                self._decision_engine.update_lidar_distance(None, available=True)
                metrics.lidar_valid = False
        
        # =====================================================================
        # Stage 7: Alert Decision
        # =====================================================================
        decision_start = time.monotonic()
        alert = self._decision_engine.evaluate(
            detections=detections,
            lane_departure=lane_departure,
            lane_result=lane_result,  # For dynamic danger zone
        )
        metrics.decision_latency_ms = (time.monotonic() - decision_start) * 1000
        
        # =====================================================================
        # Stage 7.5: Autonomous Braking Output (GPIO5)
        # =====================================================================
        # Trigger braking output when collision is imminent
        # WARNING: This is an EXAMPLE output. Real autonomous braking requires
        # safety-critical hardware and redundant fail-safes.
        if self._gpio_leds is not None:
            collision_imminent = (
                alert is not None and 
                alert.alert_type == AlertType.COLLISION_IMMINENT
            )
            self._gpio_leds.set_braking_output(collision_imminent)
        
        # =====================================================================
        # Stage 8: Alert Dispatch (non-blocking)
        # =====================================================================
        if alert:
            alert_start = time.monotonic()
            self._dispatch_alert(alert)
            metrics.alert_latency_ms = (time.monotonic() - alert_start) * 1000
            metrics.alert_type = alert.alert_type.value
        
        # =====================================================================
        # Stage 8.5: GPIO Alert LED Update
        # =====================================================================
        # Turn on alert LED when alert is active, off when no alert
        if self._gpio_leds is not None:
            self._gpio_leds.set_alert_led(alert is not None)
        
        # =====================================================================
        # Stage 9: Alert Persistence for Display
        # =====================================================================
        # Determine which alert to show on display (new alert or held alert)
        display_alert = self._get_display_alert(alert)
        
        # =====================================================================
        # Stage 9.5: Traffic Light Visual Persistence
        # =====================================================================
        # Keep traffic light detection visible on screen for configured duration
        display_detections = self._get_display_detections(detections)
        
        # =====================================================================
        # Stage 10: Display Rendering (optional, non-blocking)
        # =====================================================================
        if self._display and self._display.is_active:
            self._render_display(
                frame,
                display_detections,
                lane_result,
                display_alert,
                collision_risks,
                overtake_advisory,
            )
        
        # =====================================================================
        # Stage 11: Telemetry
        # =====================================================================
        fps = self._fps_counter.tick()
        self._system_metrics.update_if_needed()
        self._telemetry.log_frame(metrics, self._system_metrics, fps)
        
        # Update frame counter
        self._frame_count += 1
        
        # Periodic logging
        if self._frame_count % 100 == 0:
            logger.info(
                f"Frame {self._frame_count}: FPS={fps:.1f}, "
                f"Lane={'OK' if lane_result.valid else 'INVALID'}, "
                f"Detections={metrics.detections_count}, "
                f"Risks={metrics.collision_risks}"
            )
    
    def _evaluate_collision_risks(
        self,
        detections: List[Detection],
    ) -> List[Detection]:
        """Evaluate which detections are collision risks."""
        collision_risks = []
        width, height = self._config.capture.resolution
        danger_zone = self._decision_engine.danger_zone
        
        for det in detections:
            # Only check pedestrians, vehicles, animals
            if not det.label.is_obstacle():
                continue
            
            if danger_zone.intersects_bbox(det.bbox, width, height):
                collision_risks.append(det)
        
        return collision_risks
    
    def _check_lane_departure(
        self,
        lane_result: LaneResult,
        frame_width: int,
    ) -> Optional[str]:
        """
        Check for lane departure.
        
        Uses the center of the frame bottom as the "vehicle position".
        
        Args:
            lane_result: Lane detection result
            frame_width: Width of frame
            
        Returns:
            "left", "right", or None
        """
        if not lane_result.valid:
            return None
        
        # Vehicle position is assumed to be center-bottom of frame
        vehicle_x = frame_width / 2
        
        # Get lane boundaries at the bottom of the lane detection area
        y_check = lane_result.left_lane.y_range[1] - 20  # Near bottom
        
        left_boundary = lane_result.left_lane.evaluate(y_check)
        right_boundary = lane_result.right_lane.evaluate(y_check)
        
        # Check if vehicle center is outside lanes
        margin = 20  # pixels of tolerance
        
        if vehicle_x < left_boundary + margin:
            return "left"
        elif vehicle_x > right_boundary - margin:
            return "right"
        
        return None
    
    def _get_display_alert(self, current_alert: Optional[AlertEvent]) -> Optional[AlertEvent]:
        """
        Get the alert to display, implementing alert persistence across frames.
        
        - Traffic light alerts use time-based hold (`traffic_light_display_hold_s`)
        - Other alerts use frame-based hold (`alert_hold_frames`)
        
        This ensures traffic light alerts remain visible for the configured duration.
        
        Args:
            current_alert: The alert from current frame's evaluation (may be None)
            
        Returns:
            Alert to display (either new alert or held previous alert)
        """
        now = time.monotonic()
        
        if current_alert is not None:
            # New alert - store it and set appropriate hold
            self._last_display_alert = current_alert
            
            # Check if this is a traffic light alert
            is_traffic_light = current_alert.alert_type in (
                AlertType.TRAFFIC_LIGHT_RED,
                AlertType.TRAFFIC_LIGHT_YELLOW,
                AlertType.TRAFFIC_LIGHT_GREEN,
            )
            
            if is_traffic_light:
                # Use time-based hold for traffic lights
                hold_duration = self._config.alerts.traffic_light_display_hold_s
                self._traffic_light_alert_until = now + hold_duration
                self._alert_hold_counter = 0  # Don't use frame-based for traffic lights
            else:
                # Use frame-based hold for other alerts
                self._alert_hold_counter = self._config.alerts.alert_hold_frames
                self._traffic_light_alert_until = 0.0
            
            return current_alert
        
        # No new alert - check if we should keep showing the previous one
        if self._last_display_alert is not None:
            # Check if it's a traffic light with time-based hold
            is_traffic_light = self._last_display_alert.alert_type in (
                AlertType.TRAFFIC_LIGHT_RED,
                AlertType.TRAFFIC_LIGHT_YELLOW,
                AlertType.TRAFFIC_LIGHT_GREEN,
            )
            
            if is_traffic_light and now < self._traffic_light_alert_until:
                # Still within time-based hold period
                return self._last_display_alert
            elif not is_traffic_light and self._alert_hold_counter > 0:
                # Still within frame-based hold period
                self._alert_hold_counter -= 1
                return self._last_display_alert
        
        # Hold period expired, clear the stored alert
        self._last_display_alert = None
        return None
    
    def _get_display_detections(self, current_detections: List[Detection]) -> List[Detection]:
        """
        Get detections to display, with traffic light visual persistence.
        
        Traffic light detections are held on screen for `traffic_light_display_hold_s`
        seconds, even if YOLO doesn't detect them in subsequent frames.
        This is visual only - does not affect audio alerts.
        
        Args:
            current_detections: Detections from current YOLO inference
            
        Returns:
            Detections list including persisted traffic lights
        """
        from src.detection import DetectionLabel
        
        now = time.monotonic()
        hold_duration = self._config.alerts.traffic_light_display_hold_s
        
        # Check for new traffic light detections
        current_traffic_lights = [
            d for d in current_detections 
            if d.label.is_traffic_light()
        ]
        
        if current_traffic_lights:
            # Found traffic light - update persistence
            # Pick the one with highest confidence
            best_light = max(current_traffic_lights, key=lambda d: d.confidence)
            self._last_traffic_light_detection = best_light
            self._traffic_light_display_until = now + hold_duration
        
        # Build display list
        result = list(current_detections)
        
        # Add persisted traffic light if still within hold period and not already in list
        if (self._last_traffic_light_detection is not None and 
            now < self._traffic_light_display_until and
            not current_traffic_lights):
            result.append(self._last_traffic_light_detection)
        
        # Clear expired persistence
        if now >= self._traffic_light_display_until:
            self._last_traffic_light_detection = None
        
        return result
    
    def _dispatch_alert(self, alert: AlertEvent) -> None:
        """Dispatch alert to audio and buzzer outputs."""
        # Audio alert
        self._audio_manager.play_alert(alert)
        
        # GPIO buzzer (if available)
        if self._buzzer and self._buzzer.is_available:
            self._buzzer.play_alert(alert.alert_type)
    
    def _evaluate_overtake_advisory(
        self,
        lane_result: LaneResult,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
        frame_data: np.ndarray,
    ) -> OvertakeAdvisory:
        """
        Evaluate overtake advisory (Stage 6.5).
        
        This is an advisory-only module that provides visual feedback
        about potential overtake safety. It does NOT affect alerts.
        
        Args:
            lane_result: Current lane detection result
            detections: Current detections
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            frame_data: Raw frame data for line analysis
            
        Returns:
            OvertakeAdvisory with status and visualization data
        """
        if self._overtake_assistant is None:
            return OvertakeAdvisory(
                status=OvertakeStatus.DISABLED,
                reason="Overtake assistant not initialized",
                clearance_zone=None,
                confidence=0.0,
                vehicles_in_zone=0,
            )
        
        return self._overtake_assistant.evaluate(
            lane_result=lane_result,
            detections=detections,
            frame_width=frame_width,
            frame_height=frame_height,
            frame=frame_data,
        )
    
    def _render_display(
        self,
        frame: Frame,
        detections: List[Detection],
        lane_result: LaneResult,
        alert: Optional[AlertEvent],
        collision_risks: List[Detection],
        overtake_advisory: Optional[OvertakeAdvisory] = None,
    ) -> None:
        """Render overlays and display frame."""
        # Get danger zone polygon
        width, height = self._config.capture.resolution
        danger_zone_polygon = self._decision_engine.get_danger_zone_polygon()
        
        # Prepare info panel data
        info = {
            "fps": self._fps_counter.fps,
            "frame": self._frame_count,
            "detections": len(detections),
            "lane_valid": lane_result.valid,
            "danger_zone_dynamic": self._decision_engine.danger_zone.is_dynamic,
        }
        
        # Add LiDAR distance to info if available
        if self._lidar is not None:
            lidar_reading = self._lidar.get_reading()
            if lidar_reading and lidar_reading.valid:
                info["lidar_distance_cm"] = lidar_reading.distance_cm
        
        # Render frame with overlays
        output = self._display.render(
            frame=frame.data,
            detections=detections,
            lane_result=lane_result,
            danger_zone=danger_zone_polygon,
            alert=alert,
            info=info,
            collision_risks=collision_risks,
            overtake_advisory=overtake_advisory,
        )
        
        # Show frame
        self._display.show(output)
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Cleaning up...")
        
        self._running = False
        
        # Turn off system LED when shutting down
        if self._gpio_leds:
            self._gpio_leds.set_system_led(False)
            self._gpio_leds.set_alert_led(False)
            self._gpio_leds.cleanup()
        
        if self._lidar:
            self._lidar.stop()
            self._lidar.disconnect()
        
        if self._ir_sensor:
            self._ir_sensor.cleanup()
        
        if self._telemetry:
            self._telemetry.stop()
        
        if self._display:
            self._display.cleanup()
        
        if self._buzzer:
            self._buzzer.cleanup()
        
        if self._audio_manager:
            self._audio_manager.stop()
        
        if self._camera:
            self._camera.release()
        
        logger.info("Cleanup complete")
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Vehicle Safety Alert System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Raspberry Pi (headless, production)
  python -m src.main --source csi --headless

  # Raspberry Pi (with display, testing)
  python -m src.main --source csi --display

  # Windows (webcam, development)
  python -m src.main --source webcam --display

  # Windows (video file, debugging)
  python -m src.main --source video --video-path test.mp4 --display

  # IP camera stream (MJPEG or RTSP)
  python -m src.main --source ip --ip-url http://192.168.1.100:8080/video --display
        """,
    )
    
    # Input source
    source_group = parser.add_argument_group("Input Source")
    source_group.add_argument(
        "--source",
        type=str,
        choices=["csi", "webcam", "video", "ip"],
        default="webcam" if not is_raspberry_pi() else "csi",
        help="Frame source (default: csi on Pi, webcam on Windows)",
    )
    source_group.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to video file (required if source=video)",
    )
    source_group.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for webcam source (default: 0)",
    )
    source_group.add_argument(
        "--ip-url",
        type=str,
        default=None,
        help="IP camera stream URL (required if source=ip). Supports MJPEG and RTSP.",
    )
    
    # Display
    display_group = parser.add_argument_group("Display")
    display_mutex = display_group.add_mutually_exclusive_group()
    display_mutex.add_argument(
        "--display",
        action="store_true",
        help="Enable graphical overlay display",
    )
    display_mutex.add_argument(
        "--headless",
        action="store_true",
        help="Disable display (audio-only mode)",
    )
    
    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLOv11s ONNX model (overrides config)",
    )
    
    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    config_group.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to telemetry log file (default: telemetry.jsonl)",
    )
    config_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    
    # Inference
    inference_group = parser.add_argument_group("Inference")
    inference_group.add_argument(
        "--yolo-skip",
        type=int,
        default=None,
        help="YOLO inference frame skip interval (default: from config)",
    )
    inference_group.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Detection confidence threshold (default: from config)",
    )
    
    # Sensors
    sensor_group = parser.add_argument_group("Sensors")
    sensor_group.add_argument(
        "--disable-ir",
        action="store_true",
        default=True,
        help="Disable IR distance sensor (default: disabled)",
    )
    sensor_group.add_argument(
        "--enable-ir",
        action="store_true",
        help="Enable IR distance sensor",
    )
    
    # Resolution
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Frame resolution as WxH (default: 640x480)",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.model:
        config.yolo.model_path = args.model
    
    if args.log_file:
        config.system.log_file = args.log_file
    
    if args.yolo_skip:
        config.yolo.frame_skip = args.yolo_skip
    
    if args.confidence:
        config.yolo.confidence_threshold = args.confidence
    
    if args.resolution:
        try:
            w, h = args.resolution.lower().split("x")
            config.capture.resolution = (int(w), int(h))
        except ValueError:
            logger.error(f"Invalid resolution format: {args.resolution}")
            return 1
    
    # Determine frame source
    source_map = {
        "csi": FrameSource.CSI,
        "webcam": FrameSource.WEBCAM,
        "video": FrameSource.VIDEO_FILE,
        "ip": FrameSource.IP_CAMERA,
    }
    source = source_map[args.source]
    
    # Validate video path
    if source == FrameSource.VIDEO_FILE and not args.video_path:
        logger.error("--video-path required when source=video")
        return 1
    
    # Validate IP camera URL
    if source == FrameSource.IP_CAMERA and not args.ip_url:
        logger.error("--ip-url required when source=ip")
        return 1
    
    # Determine display mode
    display_enabled = args.display or (not args.headless and not is_raspberry_pi())
    
    # Determine IR sensor state
    disable_ir = args.disable_ir and not args.enable_ir
    
    # Create and run application
    app = DriverAssistant(
        config=config,
        source=source,
        video_path=args.video_path,
        ip_url=args.ip_url,
        display_enabled=display_enabled,
        disable_ir=disable_ir,
    )
    
    try:
        if not app.setup():
            logger.critical("System setup failed - aborting")
            return 1
        
        app.run()
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1
    finally:
        app.cleanup()


if __name__ == "__main__":
    sys.exit(main())
