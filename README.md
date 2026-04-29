# 🚗 Driver Assistant - Vehicle Safety Alert System

A real-time driver assistance system that detects road hazards, lane departures, and traffic signals using computer vision and deep learning. Designed to run on both Windows (for development/testing) and Raspberry Pi 4 (for in-vehicle deployment).

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Raspberry%20Pi-lightgrey.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technologies Used](#-technologies-used)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
  - [Windows Setup](#windows-setup)
  - [Raspberry Pi Setup](#raspberry-pi-setup)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)

---

## ✨ Features

- **Object Detection**: Detects pedestrians, vehicles, animals, traffic lights, and stop signs using YOLOv11s
- **Lane Detection**: Classical computer vision pipeline with polynomial curve fitting
- **Dynamic Danger Zone**: Trapezoidal collision detection zone that aligns with detected lanes
 - **Programmatic Beeps**: Tones generated in software (no external audio files required by default)
 - **Deployment helpers**: systemd service and `scripts/setup-pi.sh` for Raspberry Pi auto-start
- **Priority-Based Alerts**: Collision warnings take precedence over other alerts
- **Audio & Haptic Feedback**: Beep patterns via speakers and GPIO buzzer
- **Cross-Platform**: Runs on Windows (webcam/video) and Raspberry Pi (CSI camera)
- **Configurable**: All parameters tunable via `config.yaml`
- **Telemetry Logging**: JSON Lines format for performance analysis

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DRIVER ASSISTANT SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐  │
│  │   CAPTURE    │    │              PROCESSING PIPELINE                 │  │
│  │              │    │                                                  │  │
│  │ • CSI Camera │──▶│  Frame ──▶ Lane Detection ──▶ YOLO Detection    │  │
│  │ • Webcam     │    │              │                      │            │  │
│  │ • Video File │    │              ▼                      ▼            │  │
│  └──────────────┘    │      Lane Polynomials      Bounding Boxes        │  │
│                      │              │                      │            │  │
│                      │              └──────────┬───────────┘            │  │
│                      │                         ▼                        │  │
│                      │              ┌─────────────────────┐             │  │
│                      │              │  DANGER ZONE CHECK  │             │  │
│                      │              │  (Dynamic/Fixed)    │             │  │
│                      │              └──────────┬──────────┘             │  │
│                      │                         ▼                        │  │
│                      │              ┌─────────────────────┐             │  │
│                      │              │  ALERT DECISION     │             │  │
│                      │              │  ENGINE             │             │  │
│                      │              │  • Priority Queue   │             │  │
│                      │              │  • Cooldown Logic   │             │  │
│                      │              └──────────┬──────────┘             │  │
│                      └───────────────────────────────────────────────────┘ │
│                                                │                            │
│                      ┌─────────────────────────┼─────────────────────────┐  │
│                      │                         ▼                         │  │
│                      │  ┌─────────┐  ┌──────────────┐  ┌─────────────┐   │  │
│                      │  │ DISPLAY │  │ AUDIO ALERTS │  │ GPIO BUZZER │   │  │
│                      │  │ Overlay │  │   (Beeps)    │  │  (Pi Only)  │   │  │
│                      │  └─────────┘  └──────────────┘  └─────────────┘   │  │
│                      │                    OUTPUT LAYER                   │  │
│                      └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Technologies Used

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Object Detection** | YOLOv11s + ONNX Runtime | Detect traffic objects (CPU inference) |
| **Lane Detection** | OpenCV (Classical CV) | HSV filtering, Canny edges, Hough transform |
| **Configuration** | PyYAML | Load settings from `config.yaml` |
| **Audio** | Pygame / winsound | Play alert beep patterns |
| **Camera (Pi)** | picamera2 | CSI camera interface |
| **GPIO (Pi)** | RPi.GPIO | Buzzer control |
| **Visualization** | OpenCV | Real-time overlay rendering |

---

## 📦 Prerequisites

### Common Requirements
- Python 3.9 or higher
- Git

### Windows
- Webcam (optional, can use video files)
- 4GB+ RAM recommended

### Raspberry Pi
- Raspberry Pi 4 (4GB+ RAM recommended)
- Raspberry Pi Camera Module v2 or v3
- MicroSD card (32GB+ recommended)
- Raspberry Pi OS (64-bit recommended)
- Passive buzzer (optional, connects to GPIO 18)

---

## 🚀 Installation

### Windows Setup

1. **Clone the repository**
   ```powershell
   git clone https://github.com/InulaC/Driver-Assistant.git
   cd Driver-Assistant
   ```

2. **Create virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```powershell
   python -c "import cv2; import onnxruntime; print('Ready!')"
   ```

### Raspberry Pi Setup

#### Option A: Automatic Setup (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/InulaC/Driver-Assistant.git
   cd Driver-Assistant
   ```

2. **Run the setup script**
   ```bash
   chmod +x scripts/setup-pi.sh
   ./scripts/setup-pi.sh
   ```

3. **Reboot** (required for group permissions)
   ```bash
   sudo reboot
   ```

#### Option B: Manual Setup

1. **Update system**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install system dependencies**
   ```bash
   sudo apt install -y python3-pip python3-venv python3-opencv
   sudo apt install -y libatlas-base-dev libhdf5-dev pulseaudio
   ```

3. **Enable camera**
   ```bash
   sudo raspi-config
   # Navigate to: Interface Options → Camera → Enable
   # Reboot when prompted
   ```

4. **Clone and setup**
   ```bash
   git clone https://github.com/InulaC/Driver-Assistant.git
   cd Driver-Assistant
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements-pi.txt
   ```

6. **Add user to required groups**
   ```bash
   sudo usermod -aG video,gpio,audio $USER
   # Logout and login again for changes to take effect
   ```

7. **Wire the buzzer (optional)**
   ```
   Buzzer (+) ──▶ GPIO 18 (Pin 12)
   Buzzer (-) ──▶ GND (Pin 14)
   ```

---

## 🔄 Auto-Start on Boot (Raspberry Pi)

The system can be configured to start automatically when the Raspberry Pi boots.

### Enable Auto-Start

1. **Copy the service file** (if not done by setup script)
   ```bash
   sudo cp scripts/driver-assistant.service /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

2. **Enable the service**
   ```bash
   sudo systemctl enable driver-assistant
   ```

3. **Start the service**
   ```bash
   sudo systemctl start driver-assistant
   ```

### Service Management Commands

| Command | Description |
|---------|-------------|
| `sudo systemctl start driver-assistant` | Start the service |
| `sudo systemctl stop driver-assistant` | Stop the service |
| `sudo systemctl restart driver-assistant` | Restart the service |
| `sudo systemctl status driver-assistant` | Check service status |
| `sudo systemctl enable driver-assistant` | Enable auto-start on boot |
| `sudo systemctl disable driver-assistant` | Disable auto-start |

### View Logs

```bash
# View service logs (real-time)
journalctl -u driver-assistant -f

# View application logs
tail -f ~/Driver-Assistant/logs/service.log

# View telemetry data
tail -f ~/Driver-Assistant/telemetry.jsonl
```

### Troubleshooting Auto-Start

If the service fails to start:

1. **Check status for errors**
   ```bash
   sudo systemctl status driver-assistant
   ```

2. **Check detailed logs**
   ```bash
   journalctl -u driver-assistant -n 50 --no-pager
   ```

3. **Common issues:**
   - Camera not enabled: Run `sudo raspi-config` and enable camera
   - Permission denied: Ensure user is in `video`, `gpio`, `audio` groups
   - Python not found: Check the path in the service file matches your setup

4. **Edit service file if needed**
   ```bash
   sudo nano /etc/systemd/system/driver-assistant.service
   # After editing:
   sudo systemctl daemon-reload
   sudo systemctl restart driver-assistant
   ```

---

## 🎮 Usage

### Run with Video File (Windows/Pi)
```bash
python driver_assistant.py --source video --video-path videos/test.mp4 --display
```

### Run with Webcam (Windows)
```bash
python driver_assistant.py --source webcam --camera-index 0 --display
```

### Run with CSI Camera (Raspberry Pi)
```bash
python driver_assistant.py --source csi --display
```

### Run Headless (No Display - Pi Production)
```bash
python driver_assistant.py --source csi
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--source` | Input source: `csi`, `webcam`, `video` | `csi` |
| `--video-path` | Path to video file (if source=video) | None |
| `--camera-index` | Webcam index (if source=webcam) | `0` |
| `--display` | Show visualization window | Disabled |
| `--config` | Path to config file | `config.yaml` |
| `--no-ir` | Disable IR sensor | Enabled |

---

## ⚙ Configuration

All parameters are in `config.yaml`:

### Key Settings

```yaml
# Frame processing
capture:
  resolution: [640, 480]    # Lower = faster, higher = more accurate
  target_fps: 15            # Target frame rate

# YOLO Detection
yolo:
  confidence_threshold: 0.25  # Min confidence to detect
  frame_skip: 4               # Run YOLO every N frames (performance)

# Danger Zone (collision detection area)
danger_zone:
  top_left_y: 0.70      # Higher = shorter zone (less false alerts)
  top_right_y: 0.70     # Must match top_left_y

# Alerts
alerts:
  cooldown_ms: 300                  # Min time between alerts
  traffic_light_cooldown_ms: 25000  # Traffic light specific cooldown
  alert_hold_frames: 5              # Keep alert visible for N frames
```

### Tuning Tips

| Problem | Solution |
|---------|----------|
| Too many false collision alerts | Increase `danger_zone.top_left_y` (e.g., 0.75-0.85) |
| Missing distant hazards | Decrease `danger_zone.top_left_y` (e.g., 0.60-0.65) |
| Alerts too frequent | Increase `alerts.cooldown_ms` |
| Low FPS on Pi | Increase `yolo.frame_skip` to 5-6 |
| Lane detection unstable | Increase `lane_detection.ema_alpha` (0.4-0.5) |
| Traffic light alerts repeat too often | Increase `alerts.traffic_light_cooldown_ms` |
| Alerts disappear on skipped frames | Increase `alerts.alert_hold_frames` to persist display |

---

## 📁 Project Structure

```
Driver-Assistant/
├── driver_assistant.py     # Entry point
├── config.yaml             # Configuration file
├── requirements.txt        # Windows dependencies
├── requirements-pi.txt     # Raspberry Pi dependencies
├── models/
│   └── object.onnx         # YOLOv11s ONNX model
├── sounds/                 # Alert audio files (optional)
├── videos/                 # Test videos
├── src/
│   ├── main.py             # Main application class
│   ├── config.py           # Configuration loader
│   ├── alerts/
│   │   ├── types.py        # AlertType enum, AlertEvent
│   │   ├── decision.py     # Alert decision engine, DangerZone
│   │   ├── audio.py        # Audio alert manager
│   │   └── gpio_buzzer.py  # Raspberry Pi buzzer control
│   ├── capture/
│   │   ├── factory.py      # Camera factory
│   │   ├── csi_camera.py   # Raspberry Pi CSI camera
│   │   ├── opencv_camera.py# Webcam capture
│   │   └── video_file.py   # Video file capture
│   ├── detection/
│   │   ├── detector.py     # YOLO detector with frame skipping
│   │   ├── preprocessing.py# Image preprocessing
│   │   ├── postprocessing.py# NMS and box extraction
│   │   └── result.py       # Detection classes
│   ├── lane/
│   │   ├── pipeline.py     # Lane detection pipeline
│   │   ├── color_filter.py # HSV color filtering
│   │   ├── edge_detection.py# Canny edge detection
│   │   ├── hough_lines.py  # Hough line detection
│   │   ├── polynomial_fit.py# Curve fitting
│   │   └── temporal.py     # EMA smoothing
│   ├── display/
│   │   └── renderer.py     # Overlay rendering
│   ├── telemetry/
│   │   ├── logger.py       # JSON Lines logger
│   │   └── metrics.py      # Performance metrics
│   └── sensors/
│       └── ir_distance.py  # IR sensor (optional)
└── telemetry.jsonl         # Runtime logs
```

---

## 🔬 How It Works

### 1. Frame Capture
- Captures frames from CSI camera, webcam, or video file
- Maintains target FPS with adaptive timing

### 2. Lane Detection (Every Frame)
```
Frame → ROI Crop → HSV Filter (white/yellow) → Gaussian Blur 
→ Canny Edges → Hough Lines → Polynomial Fit → EMA Smoothing
```

### 3. Object Detection (With Frame Skipping)
```
Frame → Resize to 640x640 → ONNX Inference → NMS 
→ Filter by confidence → Classify (pedestrian/vehicle/etc.)
```
- Runs every N frames (configurable) to maintain performance
- Caches results for skipped frames

### 4. Dynamic Danger Zone
- When **both lanes detected**: Trapezoid aligns with lane boundaries
- When **lanes not detected**: Falls back to fixed trapezoid from config
- Objects inside the zone trigger collision alerts

### 5. Alert Priority System

| Priority | Alert Type | Trigger |
|----------|------------|---------|
| 1 (Highest) | Collision Warning | Object in danger zone |
| 2 | Animal Warning | Animal detected anywhere |
| 2 | Lane Departure | Vehicle drifting out of lane |
| 3 | Traffic Light | Traffic light detected |
| 3 | Stop Sign | Stop sign detected |

### 6. Output
- **Display**: Overlays showing lanes, danger zone, detections, alerts
- **Audio**: Distinct beep patterns for each alert type
- **GPIO Buzzer**: Physical buzzer feedback (Raspberry Pi only)
- **Telemetry**: JSON logs for analysis

---



## 📄 License

This project is for educational purposes. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO
- [ONNX Runtime](https://onnxruntime.ai/) for efficient inference
- [OpenCV](https://opencv.org/) for computer vision

---

**⚠️ Disclaimer**: This is a prototype system for educational purposes. Do not rely on it as your sole safety system while driving. Always pay attention to the road and follow traffic laws.
