# Raspberry Pi 4 Deployment Guide

Complete guide for deploying the Vehicle Safety Alert System on Raspberry Pi 4B (8GB).

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Hardware Connections](#hardware-connections)
3. [Raspberry Pi OS Setup](#raspberry-pi-os-setup)
4. [UART Configuration for LiDAR](#uart-configuration-for-lidar)
5. [Software Installation](#software-installation)
6. [Service Configuration](#service-configuration)
7. [Testing & Validation](#testing--validation)
8. [Production Deployment](#production-deployment)

---

## Hardware Requirements

### Core Components

| Component | Model | Purpose |
|-----------|-------|---------|
| Single Board Computer | Raspberry Pi 4B (8GB RAM) | Main processing unit |
| Camera | Pi NoIR Camera V2 | 1080p60 low-light capable |
| Display | 5" 800x480 HDMI TFT LCD | Real-time visualization |
| LiDAR Sensor | Benewake TF-Luna | Distance measurement (0.1-8m) |
| System LED | Green LED + 330Ω resistor | System running indicator |
| Alert LED | Red LED + 330Ω resistor | Alert active indicator |
| Power Supply | 5V 3A USB-C | Adequate for all peripherals |

### Optional Components

| Component | Model | Purpose |
|-----------|-------|---------|
| Active Buzzer | 3.3V PWM buzzer | Audio collision alerts |
| Case | Official Pi 4 case with fan | Thermal management |
| MicroSD | 32GB+ Class 10 | OS and application storage |

---

## Hardware Connections

### Pin Mapping (BCM Numbering)

```
Raspberry Pi 4B GPIO Header (40 pins)
┌─────────────────────────────────────────────────────────────┐
│  3.3V  (1) (2)  5V         ← TF-Luna 5V                     │
│ GPIO2  (3) (4)  5V                                          │
│ GPIO3  (5) (6)  GND        ← TF-Luna GND, LED GND           │
│ GPIO4  (7) (8)  GPIO14/TXD ← TF-Luna RX                     │
│   GND  (9) (10) GPIO15/RXD ← TF-Luna TX                     │
│GPIO17 (11) (12) GPIO18     ← System LED (green)             │
│GPIO27 (13) (14) GND        ← Alert LED (red)                │
│GPIO22 (15) (16) GPIO23                                      │
│  3.3V (17) (18) GPIO24                                      │
│GPIO10 (19) (20) GND                                         │
│ GPIO9 (21) (22) GPIO25                                      │
│GPIO11 (23) (24) GPIO8                                       │
│   GND (25) (26) GPIO7                                       │
│...                                                          │
└─────────────────────────────────────────────────────────────┘
```

### TF-Luna LiDAR Connection

| TF-Luna Pin | Raspberry Pi Pin | Description |
|-------------|------------------|-------------|
| 5V (Red) | Pin 2 or 4 (5V) | Power supply |
| GND (Black) | Pin 6 (GND) | Ground |
| TX (Green) | Pin 10 (GPIO15/RXD) | LiDAR transmit → Pi receive |
| RX (White) | Pin 8 (GPIO14/TXD) | LiDAR receive ← Pi transmit |

**Important:** The TF-Luna TX connects to Pi RXD (cross-over connection).

### Status LED Connections

#### System LED (Green) - GPIO 17

```
GPIO 17 ──────┬──── 330Ω ──────┬──── (+) LED (─) ──── GND
              │                │
         Pin 11           Anode      Cathode
```

#### Alert LED (Red) - GPIO 27

```
GPIO 27 ──────┬──── 330Ω ──────┬──── (+) LED (─) ──── GND
              │                │
         Pin 13           Anode      Cathode
```

### Camera Connection (CSI)

1. Locate the CSI camera port between the HDMI ports and 3.5mm jack
2. Gently lift the plastic clip
3. Insert ribbon cable with blue side facing the USB ports
4. Press down the clip to secure

### Display Connection (HDMI)

Connect the 5" display to HDMI 0 (closest to USB-C power).

---

## Raspberry Pi OS Setup

### 1. Flash OS Image

```bash
# Download Raspberry Pi Imager from https://www.raspberrypi.com/software/
# Select: Raspberry Pi OS (64-bit) - Lite or Desktop
# Configure: SSH, WiFi, hostname, username
```

### 2. Initial Boot Configuration

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Enable required interfaces
sudo raspi-config
# → Interface Options → Camera → Enable
# → Interface Options → Serial Port → 
#   - Login shell over serial? → No
#   - Serial port hardware enabled? → Yes
```

### 3. Reboot

```bash
sudo reboot
```

---

## UART Configuration for LiDAR

### 1. Disable Bluetooth UART (Required)

The Raspberry Pi 4 has two UARTs:
- `/dev/ttyAMA0` - Full UART (PL011) - **Use this for LiDAR**
- `/dev/ttyS0` - Mini UART - Used for Bluetooth by default

To use the full UART for LiDAR, disable Bluetooth or swap UARTs.

#### Option A: Disable Bluetooth (Recommended)

Edit `/boot/config.txt`:

```bash
sudo nano /boot/config.txt
```

Add at the end:

```ini
# Disable Bluetooth to free full UART for LiDAR
dtoverlay=disable-bt

# Enable UART
enable_uart=1
```

Disable Bluetooth modem service:

```bash
sudo systemctl disable hciuart
```

#### Option B: Swap UARTs (Keep Bluetooth)

Edit `/boot/config.txt`:

```ini
# Swap UARTs - give full UART to GPIO, mini UART to Bluetooth
dtoverlay=miniuart-bt

# Enable UART
enable_uart=1
```

### 2. Verify UART Configuration

After reboot:

```bash
# Check UART device exists
ls -la /dev/ttyAMA0
# Should show: crw-rw---- 1 root dialout ...

# Add user to dialout group
sudo usermod -aG dialout $USER

# Logout and login again for group change
```

### 3. Test LiDAR Communication

```bash
# Install minicom for serial testing
sudo apt install minicom -y

# Test connection (Ctrl+A, then X to exit)
minicom -D /dev/ttyAMA0 -b 115200
```

You should see binary data streaming from the TF-Luna.

---

## Software Installation

### 1. Install System Dependencies

```bash
# Install Python and development tools
sudo apt install -y python3-pip python3-venv python3-dev

# Install OpenCV dependencies
sudo apt install -y libopencv-dev python3-opencv

# Install audio dependencies (for alerts)
sudo apt install -y libportaudio2 portaudio19-dev

# Install GPIO library
sudo apt install -y python3-rpi.gpio
```

### 2. Clone and Setup Project

```bash
# Clone repository
cd ~
git clone https://github.com/your-repo/Driver-Assistant.git
cd Driver-Assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-pi.txt
```

### 3. Install PySerial (for LiDAR)

```bash
pip install pyserial
```

### 4. Verify Installation

```bash
# Test imports
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')"
python -c "import serial; print('PySerial: OK')"
python -c "import RPi.GPIO; print('RPi.GPIO: OK')"
```

---

## Service Configuration

### 1. Create Systemd Service

```bash
sudo nano /etc/systemd/system/driver-assistant.service
```

```ini
[Unit]
Description=Vehicle Safety Alert System
After=network.target
Wants=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Driver-Assistant
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/pi/Driver-Assistant/venv/bin/python -m src.main --source csi --headless
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 2. Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start
sudo systemctl enable driver-assistant

# Start service
sudo systemctl start driver-assistant

# Check status
sudo systemctl status driver-assistant
```

### 3. View Logs

```bash
# View live logs
journalctl -u driver-assistant -f

# View recent logs
journalctl -u driver-assistant -n 100
```

---

## Testing & Validation

### 1. Hardware Self-Test

```bash
cd ~/Driver-Assistant
source venv/bin/activate

# Run hardware tests
python -m pytest tests/test_hardware.py -v
```

### 2. LiDAR Connectivity Test

```python
# Save as test_lidar_connection.py
from src.sensors import create_lidar

lidar = create_lidar(enabled=True, port="/dev/ttyAMA0")
if lidar.connect():
    lidar.start()
    import time
    for _ in range(10):
        reading = lidar.get_reading()
        if reading and reading.valid:
            print(f"Distance: {reading.distance_cm} cm, Strength: {reading.strength}")
        time.sleep(0.1)
    lidar.stop()
else:
    print("Failed to connect to LiDAR")
```

```bash
python test_lidar_connection.py
```

### 3. GPIO LED Test

```python
# Save as test_gpio_leds.py
from src.gpio import create_gpio_controller
import time

gpio = create_gpio_controller(enabled=True, system_pin=17, alert_pin=27)
if gpio.initialize():
    print("Turning on system LED...")
    gpio.set_system_led(True)
    time.sleep(2)
    
    print("Blinking alert LED...")
    for _ in range(5):
        gpio.set_alert_led(True)
        time.sleep(0.3)
        gpio.set_alert_led(False)
        time.sleep(0.3)
    
    gpio.cleanup()
    print("Test complete")
else:
    print("GPIO initialization failed")
```

```bash
python test_gpio_leds.py
```

### 4. Full System Test

```bash
# Run with display (requires HDMI connected)
python -m src.main --source csi --display

# Run headless (production mode)
python -m src.main --source csi --headless
```

---

## Production Deployment

### 1. Performance Optimization

Edit `/boot/config.txt`:

```ini
# GPU memory allocation
gpu_mem=128

# Disable unused features
dtparam=audio=off  # If not using audio
dtparam=i2c_arm=off
dtparam=spi=off

# CPU governor for performance
# Add to /etc/rc.local before 'exit 0':
# echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 2. Reduce Boot Time

```bash
# Disable unused services
sudo systemctl disable bluetooth
sudo systemctl disable avahi-daemon
sudo systemctl disable triggerhappy
```

### 3. Read-Only Filesystem (Optional)

For reliability, configure read-only root filesystem:

```bash
# Use overlayroot for read-only operation
sudo apt install overlayroot
# Follow Raspberry Pi documentation for full setup
```

### 4. Watchdog Configuration

Enable hardware watchdog:

```bash
sudo nano /etc/systemd/system.conf
```

Add:

```ini
RuntimeWatchdogSec=10
ShutdownWatchdogSec=10min
```

### 5. Final Checklist

- [ ] Camera positioned correctly
- [ ] LiDAR aligned with camera view
- [ ] Status LEDs visible to driver
- [ ] All connections secure
- [ ] Service auto-starts on boot
- [ ] Logs show no errors
- [ ] Collision alerts trigger correctly

---

## LED Status Meanings

| System LED (Green) | Alert LED (Red) | System State |
|-------------------|-----------------|--------------|
| Off | Off | System not running |
| On (solid) | Off | System running, no alerts |
| On (solid) | On (solid) | Active collision alert |
| On (solid) | Blinking | Active lane departure alert |
| Blinking | Off | System initializing |
| Off | On (solid) | Error state (needs attention) |

---

## Quick Reference Commands

```bash
# Start service
sudo systemctl start driver-assistant

# Stop service
sudo systemctl stop driver-assistant

# Restart service
sudo systemctl restart driver-assistant

# View live logs
journalctl -u driver-assistant -f

# Check service status
sudo systemctl status driver-assistant

# Manual run (for debugging)
cd ~/Driver-Assistant
source venv/bin/activate
python -m src.main --source csi --display --verbose

# Test LiDAR
python -c "from src.sensors import create_lidar; l=create_lidar(); l.connect(); l.start(); import time; time.sleep(1); print(l.get_reading()); l.stop()"
```

---

## Support

For issues, check the [Troubleshooting Guide](TROUBLESHOOTING.md).
