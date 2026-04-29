# Windows Usage Guide

## Quick Start

### 1. Activate Virtual Environment
```powershell
cd C:\Users\cinul\Desktop\Driver-Assistant
.\venv\Scripts\activate
```

### 2. Run with Video File
```powershell
python -m src.main --source video --video-path videos/laneTest2.mp4 --display
```

### 3. Run with Webcam
```powershell
python -m src.main --source webcam --display
```

### 4. Run with IP Camera (Phone/Network Camera)
```powershell
# Using IP Webcam app on Android or similar
python -m src.main --source ip --ip-url http://192.168.1.100:8080/video --display
```
> **Tip:** Replace `192.168.1.100` with your phone/camera's IP address. 
> Find it in your IP Webcam app settings or check your router's connected devices.

---

## Sample Commands

```powershell
# Run with a test video
python -m src.main --source video --video-path videos/laneTest2.mp4 --display

# Run with webcam (camera index 0)
python -m src.main --source webcam --display

# Run with IP camera (phone as webcam via IP Webcam app)
python -m src.main --source ip --ip-url http://192.168.1.100:8080/video --display

# Run headless (no display window)
python -m src.main --source video --video-path videos/test.mp4 --headless

# Run with custom config
python -m src.main --source webcam --display --config my_config.yaml
```

---

## Command Options

| Option | Description |
|--------|-------------|
| `--source` | `video`, `webcam`, `ip`, or `csi` (Pi only) |
| `--video-path` | Path to video file (required for video source) |
| `--ip-url` | IP camera stream URL (required for ip source) |
| `--display` | Show visual output window |
| `--headless` | Run without display |
| `--config` | Custom config file (default: config.yaml) |

---

## Run Tests
```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Controls
- Press **Q** or **ESC** to quit when display is active

---

## Notes
- GPIO/LiDAR features use stubs on Windows (no hardware)
- Audio alerts work via Windows beep
- Full hardware features require Raspberry Pi
