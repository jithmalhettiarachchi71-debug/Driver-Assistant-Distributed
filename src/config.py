"""
Configuration management for the Vehicle Safety Alert System.

Handles loading, validation, and access to system configuration.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
from pathlib import Path


@dataclass
class CaptureConfig:
    """Frame capture configuration."""
    resolution: Tuple[int, int] = (640, 480)
    target_fps: int = 15
    timeout_ms: int = 100
    reconnect_attempts: int = 3
    reconnect_interval_ms: int = 500


@dataclass
class HSVRange:
    """HSV color range for filtering."""
    h_min: int = 0
    h_max: int = 180
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255
    
    def as_tuple(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Return as (lower, upper) bound tuples."""
        return (
            (self.h_min, self.s_min, self.v_min),
            (self.h_max, self.s_max, self.v_max)
        )


@dataclass
class LaneDetectionConfig:
    """Lane detection pipeline configuration."""
    roi_top_ratio: float = 0.5
    hsv_white: HSVRange = field(default_factory=lambda: HSVRange(
        h_min=0, h_max=180, s_min=0, s_max=30, v_min=200, v_max=255
    ))
    hsv_yellow: HSVRange = field(default_factory=lambda: HSVRange(
        h_min=15, h_max=35, s_min=80, s_max=255, v_min=150, v_max=255
    ))
    gaussian_kernel: Tuple[int, int] = (5, 5)
    canny_low: int = 50
    canny_high: int = 150
    hough_rho: int = 2
    hough_theta_deg: int = 1
    hough_threshold: int = 50
    hough_min_length: int = 40
    hough_max_gap: int = 100
    slope_min: float = 0.5
    slope_max: float = 2.0
    min_line_length: int = 40
    ema_alpha: float = 0.3
    max_invalid_frames: int = 5


@dataclass
class YOLOConfig:
    """YOLO object detection configuration."""
    enabled: bool = True  # Set to False to disable YOLO detection
    model_path: str = "models/object.onnx"  # YOLO11s COCO pre-trained
    input_width: int = 640
    input_height: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    frame_skip: int = 2  # Run inference every N frames
    cache_ttl_ms: float = 400.0
    classes: List[str] = field(default_factory=lambda: [
        "traffic_light",
        "stop_sign",
        "pedestrian",
        "vehicle",
        "animal"
    ])


@dataclass
class DangerZoneConfig:
    """
    Trapezoidal danger zone configuration.
    
    The trapezoid represents the forward driving corridor:
    - Narrower at top (farther objects, center of road)
    - Wider at bottom (closer objects, full lane width)
    - Starts at ~65% down the frame to focus on closer hazards
    """
    # Top edge (narrow, farther distance)
    top_left_x: float = 0.42
    top_left_y: float = 0.65
    top_right_x: float = 0.58
    top_right_y: float = 0.65
    # Bottom edge (wider, closer distance)
    bottom_left_x: float = 0.2
    bottom_left_y: float = 1.0
    bottom_right_x: float = 0.8
    bottom_right_y: float = 1.0
    
    def get_polygon(self, width: int, height: int) -> List[Tuple[int, int]]:
        """Get polygon coordinates for given frame dimensions."""
        return [
            (int(self.top_left_x * width), int(self.top_left_y * height)),
            (int(self.top_right_x * width), int(self.top_right_y * height)),
            (int(self.bottom_right_x * width), int(self.bottom_right_y * height)),
            (int(self.bottom_left_x * width), int(self.bottom_left_y * height)),
        ]


@dataclass
class AlertSoundsConfig:
    """Alert sound file paths."""
    collision: str = "sounds/collision.wav"
    lane_left: str = "sounds/lane_left.wav"
    lane_right: str = "sounds/lane_right.wav"
    traffic_light: str = "sounds/traffic_light.wav"
    stop_sign: str = "sounds/stop_sign.wav"
    animal: str = "sounds/animal.wav"
    system_warning: str = "sounds/warning.wav"


@dataclass
class AlertConfig:
    """Alert system configuration."""
    cooldown_ms: int = 300
    traffic_light_cooldown_ms: int = 5000  # Separate cooldown for traffic lights
    alert_hold_frames: int = 5  # How many frames to keep showing an alert after it triggers
    traffic_light_display_hold_s: float = 2.0  # How long to show traffic light visual (seconds)
    sounds: AlertSoundsConfig = field(default_factory=AlertSoundsConfig)


@dataclass
class GPIOConfig:
    """GPIO configuration for Raspberry Pi."""
    enabled: bool = True
    buzzer_pin: int = 18


@dataclass
class IRSensorConfig:
    """IR distance sensor configuration."""
    enabled: bool = False
    gpio_trigger: int = 23
    gpio_echo: int = 24
    poll_interval_ms: int = 100
    threshold_cm: int = 50


@dataclass
class GPIOLEDConfig:
    """GPIO status LED configuration."""
    enabled: bool = True
    system_led_pin: int = 17  # System running indicator
    alert_led_pin: int = 27   # Alert active indicator
    collision_output_pin: int = 22  # HIGH when collision/object detected
    braking_output_pin: int = 5  # HIGH when autonomous braking triggered


@dataclass
class LiDARConfig:
    """TF-Luna LiDAR sensor configuration."""
    enabled: bool = True
    port: str = "/dev/ttyAMA0"
    baud_rate: int = 115200
    ema_alpha: float = 0.3
    min_strength: int = 100
    max_distance_cm: int = 800
    min_distance_cm: int = 10
    required_for_collision: bool = True  # Require LiDAR confirmation for collision alerts
    collision_threshold_cm: int = 300    # Distance threshold (3 meters)


@dataclass
class DisplayConfig:
    """Display overlay configuration."""
    enabled: bool = True
    window_name: str = "Driver Assistant"
    overlay_alpha: float = 0.6
    bbox_thickness: int = 2
    lane_color: Tuple[int, int, int] = (0, 255, 0)
    danger_zone_color: Tuple[int, int, int] = (0, 0, 255)
    font_scale: float = 0.6
    show_bbox_labels: bool = True  # Show text labels on bounding boxes


@dataclass 
class SystemConfig:
    """Top-level system configuration."""
    log_level: str = "INFO"
    log_file: str = "telemetry.jsonl"
    telemetry_flush_interval_s: float = 1.0


@dataclass
class Config:
    """Complete system configuration."""
    system: SystemConfig = field(default_factory=SystemConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    lane_detection: LaneDetectionConfig = field(default_factory=LaneDetectionConfig)
    danger_zone: DangerZoneConfig = field(default_factory=DangerZoneConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    gpio: GPIOConfig = field(default_factory=GPIOConfig)
    gpio_leds: GPIOLEDConfig = field(default_factory=GPIOLEDConfig)
    lidar: LiDARConfig = field(default_factory=LiDARConfig)
    ir_sensor: IRSensorConfig = field(default_factory=IRSensorConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    # Store raw config dict for modules that parse their own config
    _raw_config: Dict[str, Any] = field(default_factory=dict)


def _parse_hsv_range(data: Dict[str, Any]) -> HSVRange:
    """Parse HSV range from config dict."""
    return HSVRange(
        h_min=data.get("h_min", 0),
        h_max=data.get("h_max", 180),
        s_min=data.get("s_min", 0),
        s_max=data.get("s_max", 255),
        v_min=data.get("v_min", 0),
        v_max=data.get("v_max", 255),
    )


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Populated Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    if config_path is None:
        # Look for config.yaml in project root
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        # Return default configuration
        return Config()
    
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    
    config = Config()
    
    # Parse system config
    if "system" in data:
        sys_data = data["system"]
        config.system = SystemConfig(
            log_level=sys_data.get("log_level", "INFO"),
            log_file=sys_data.get("log_file", "telemetry.jsonl"),
            telemetry_flush_interval_s=sys_data.get("telemetry_flush_interval_s", 1.0),
        )
    
    # Parse capture config
    if "capture" in data:
        cap_data = data["capture"]
        resolution = cap_data.get("resolution", [640, 480])
        config.capture = CaptureConfig(
            resolution=tuple(resolution),
            target_fps=cap_data.get("target_fps", 15),
            timeout_ms=cap_data.get("timeout_ms", 100),
            reconnect_attempts=cap_data.get("reconnect_attempts", 3),
            reconnect_interval_ms=cap_data.get("reconnect_interval_ms", 500),
        )
    
    # Parse YOLO config
    if "yolo" in data:
        yolo_data = data["yolo"]
        # Handle both old input_size and new input_width/input_height formats
        input_size = yolo_data.get("input_size", [640, 640])
        input_width = yolo_data.get("input_width", input_size[0] if isinstance(input_size, list) else 640)
        input_height = yolo_data.get("input_height", input_size[1] if isinstance(input_size, list) else 640)
        config.yolo = YOLOConfig(
            enabled=yolo_data.get("enabled", True),
            model_path=yolo_data.get("model_path", "models/object.onnx"),
            input_width=input_width,
            input_height=input_height,
            confidence_threshold=yolo_data.get("confidence_threshold", 0.25),
            iou_threshold=yolo_data.get("iou_threshold", 0.45),
            frame_skip=yolo_data.get("frame_skip", yolo_data.get("skip_interval", 2)),
            cache_ttl_ms=yolo_data.get("cache_ttl_ms", 400.0),
            classes=yolo_data.get("classes", [
                "traffic_light", "pedestrian", "vehicle", "stop_sign", "animal"
            ]),
        )
    
    # Parse lane detection config
    if "lane_detection" in data:
        lane_data = data["lane_detection"]
        gaussian = lane_data.get("gaussian_kernel", [5, 5])
        config.lane_detection = LaneDetectionConfig(
            roi_top_ratio=lane_data.get("roi_top_ratio", 0.5),
            hsv_white=_parse_hsv_range(lane_data.get("hsv_white", {})),
            hsv_yellow=_parse_hsv_range(lane_data.get("hsv_yellow", {})),
            gaussian_kernel=tuple(gaussian),
            canny_low=lane_data.get("canny_low", 50),
            canny_high=lane_data.get("canny_high", 150),
            hough_rho=lane_data.get("hough_rho", 2),
            hough_theta_deg=lane_data.get("hough_theta_deg", 1),
            hough_threshold=lane_data.get("hough_threshold", 50),
            hough_min_length=lane_data.get("hough_min_length", 40),
            hough_max_gap=lane_data.get("hough_max_gap", 100),
            slope_min=lane_data.get("slope_min", 0.5),
            slope_max=lane_data.get("slope_max", 2.0),
            min_line_length=lane_data.get("min_line_length", 40),
            ema_alpha=lane_data.get("ema_alpha", 0.3),
            max_invalid_frames=lane_data.get("max_invalid_frames", 5),
        )
    
    # Parse danger zone config
    if "danger_zone" in data:
        dz_data = data["danger_zone"]
        config.danger_zone = DangerZoneConfig(
            top_left_x=dz_data.get("top_left_x", 0.375),
            top_left_y=dz_data.get("top_left_y", 0.5),
            top_right_x=dz_data.get("top_right_x", 0.625),
            top_right_y=dz_data.get("top_right_y", 0.5),
            bottom_left_x=dz_data.get("bottom_left_x", 0.125),
            bottom_left_y=dz_data.get("bottom_left_y", 1.0),
            bottom_right_x=dz_data.get("bottom_right_x", 0.875),
            bottom_right_y=dz_data.get("bottom_right_y", 1.0),
        )
    
    # Parse alerts config
    if "alerts" in data:
        alert_data = data["alerts"]
        sounds_data = alert_data.get("sounds", {})
        config.alerts = AlertConfig(
            cooldown_ms=alert_data.get("cooldown_ms", 300),
            traffic_light_cooldown_ms=alert_data.get("traffic_light_cooldown_ms", 5000),
            alert_hold_frames=alert_data.get("alert_hold_frames", 5),
            traffic_light_display_hold_s=alert_data.get("traffic_light_display_hold_s", 2.0),
            sounds=AlertSoundsConfig(
                collision=sounds_data.get("collision", "sounds/collision.wav"),
                lane_left=sounds_data.get("lane_left", "sounds/lane_left.wav"),
                lane_right=sounds_data.get("lane_right", "sounds/lane_right.wav"),
                traffic_light=sounds_data.get("traffic_light", "sounds/traffic_light.wav"),
                stop_sign=sounds_data.get("stop_sign", "sounds/stop_sign.wav"),
                animal=sounds_data.get("animal", "sounds/animal.wav"),
                system_warning=sounds_data.get("system_warning", "sounds/warning.wav"),
            ),
        )
    
    # Parse GPIO config
    if "gpio" in data:
        gpio_data = data["gpio"]
        config.gpio = GPIOConfig(
            enabled=gpio_data.get("enabled", True),
            buzzer_pin=gpio_data.get("buzzer_pin", 18),
        )
    
    # Parse GPIO LED config
    if "gpio_leds" in data:
        led_data = data["gpio_leds"]
        config.gpio_leds = GPIOLEDConfig(
            enabled=led_data.get("enabled", True),
            system_led_pin=led_data.get("system_led_pin", 17),
            alert_led_pin=led_data.get("alert_led_pin", 27),
            collision_output_pin=led_data.get("collision_output_pin", 22),
            braking_output_pin=led_data.get("braking_output_pin", 5),
        )
    
    # Parse LiDAR config
    if "lidar" in data:
        lidar_data = data["lidar"]
        config.lidar = LiDARConfig(
            enabled=lidar_data.get("enabled", True),
            port=lidar_data.get("port", "/dev/ttyAMA0"),
            baud_rate=lidar_data.get("baud_rate", 115200),
            ema_alpha=lidar_data.get("ema_alpha", 0.3),
            min_strength=lidar_data.get("min_strength", 100),
            max_distance_cm=lidar_data.get("max_distance_cm", 800),
            min_distance_cm=lidar_data.get("min_distance_cm", 10),
            required_for_collision=lidar_data.get("required_for_collision", True),
            collision_threshold_cm=lidar_data.get("collision_threshold_cm", 300),
        )
    
    # Parse IR sensor config
    if "ir_sensor" in data:
        ir_data = data["ir_sensor"]
        config.ir_sensor = IRSensorConfig(
            enabled=ir_data.get("enabled", False),
            gpio_trigger=ir_data.get("gpio_trigger", 23),
            gpio_echo=ir_data.get("gpio_echo", 24),
            poll_interval_ms=ir_data.get("poll_interval_ms", 100),
            threshold_cm=ir_data.get("threshold_cm", 50),
        )
    
    # Parse display config
    if "display" in data:
        disp_data = data["display"]
        lane_color = disp_data.get("lane_color", [0, 255, 0])
        dz_color = disp_data.get("danger_zone_color", [0, 0, 255])
        config.display = DisplayConfig(
            enabled=disp_data.get("enabled", True),
            window_name=disp_data.get("window_name", "Driver Assistant"),
            overlay_alpha=disp_data.get("overlay_alpha", 0.6),
            bbox_thickness=disp_data.get("bbox_thickness", 2),
            lane_color=tuple(lane_color),
            danger_zone_color=tuple(dz_color),
            font_scale=disp_data.get("font_scale", 0.6),
            show_bbox_labels=disp_data.get("show_bbox_labels", True),
        )
    
    # Store raw config for modules that parse their own config
    config._raw_config = data
    
    return config
