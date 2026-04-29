# Troubleshooting Guide

Quick solutions for common issues with the Vehicle Safety Alert System.

## Table of Contents

1. [LED Status Indicators](#led-status-indicators)
2. [LiDAR Issues](#lidar-issues)
3. [Camera Issues](#camera-issues)
4. [GPIO Issues](#gpio-issues)
5. [Service Issues](#service-issues)
6. [Performance Issues](#performance-issues)
7. [Diagnostic Commands](#diagnostic-commands)

---

## LED Status Indicators

### Understanding LED States

| System LED (Green, GPIO 17) | Alert LED (Red, GPIO 27) | Meaning |
|----------------------------|-------------------------|---------|
| **Off** | Off | System not running or crashed |
| **On (solid)** | Off | Normal operation, no alerts |
| **On (solid)** | On (solid) | Collision alert active |
| **On (solid)** | Blinking (fast) | Lane departure alert |
| **On (solid)** | Blinking (slow) | Traffic light alert |
| **Blinking (slow)** | Off | System initializing |
| **Off** | On (solid) | Error state - immediate attention needed |

### LED Not Working

1. **Check wiring:**
   ```
   GPIO pin → 330Ω resistor → LED anode (+, longer leg) → LED cathode (-) → GND
   ```

2. **Test GPIO manually:**
   ```bash
   # Export GPIO
   echo 17 > /sys/class/gpio/export
   echo out > /sys/class/gpio/gpio17/direction
   
   # Turn on
   echo 1 > /sys/class/gpio/gpio17/value
   
   # Turn off
   echo 0 > /sys/class/gpio/gpio17/value
   
   # Cleanup
   echo 17 > /sys/class/gpio/unexport
   ```

3. **Check config.yaml:**
   ```yaml
   gpio_leds:
     enabled: true
     system_led_pin: 17
     alert_led_pin: 27
   ```

---

## LiDAR Issues

### Issue: "LiDAR not available" in logs

**Cause 1: UART not configured**

Check if UART is enabled:
```bash
# Should show "enable_uart=1"
grep enable_uart /boot/config.txt

# Should show device file
ls -la /dev/ttyAMA0
```

Fix:
```bash
sudo nano /boot/config.txt
# Add:
enable_uart=1
dtoverlay=disable-bt

sudo reboot
```

**Cause 2: User not in dialout group**

```bash
# Check groups
groups $USER

# Add to dialout
sudo usermod -aG dialout $USER

# Logout and login again
```

**Cause 3: Serial console using UART**

```bash
sudo raspi-config
# → Interface Options → Serial Port
# → Login shell? NO
# → Serial hardware? YES
```

**Cause 4: Bluetooth using UART**

```bash
# Disable Bluetooth
sudo systemctl disable hciuart
sudo nano /boot/config.txt
# Add: dtoverlay=disable-bt
sudo reboot
```

### Issue: LiDAR readings erratic or invalid

**Cause 1: Electrical noise**

- Use shielded cables for LiDAR
- Keep LiDAR wires away from motor/power lines
- Add 0.1µF capacitor between LiDAR 5V and GND

**Cause 2: Signal strength too low**

Check readings:
```python
from src.sensors import create_lidar
lidar = create_lidar()
lidar.connect()
lidar.start()
import time
for _ in range(10):
    r = lidar.get_reading()
    if r:
        print(f"Dist: {r.distance_cm}cm, Strength: {r.strength}")
    time.sleep(0.1)
lidar.stop()
```

If strength < 100 consistently:
- Clean LiDAR lens
- Ensure surface is not too reflective (glass) or absorptive (black cloth)
- Adjust `min_strength` in config.yaml

**Cause 3: Baud rate mismatch**

TF-Luna default is 115200. If changed:
```yaml
lidar:
  baud_rate: 115200  # Match sensor setting
```

### Issue: Collision alerts not triggering with LiDAR

**Check 1: Is LiDAR data reaching decision engine?**

```bash
# Run with verbose logging
python -m src.main --source csi --verbose
# Look for: "LiDAR distance: XXX cm"
```

**Check 2: Is threshold correct?**

```yaml
lidar:
  collision_threshold_cm: 300  # 3 meters
  required_for_collision: true
```

With `required_for_collision: true`, BOTH conditions must be met:
- Vision detects object in danger zone
- LiDAR distance < collision_threshold_cm

**Check 3: Test without LiDAR requirement**

Temporarily disable:
```yaml
lidar:
  required_for_collision: false
```

---

## Camera Issues

### Issue: "Failed to open camera"

**CSI Camera:**
```bash
# Check if camera is detected
vcgencmd get_camera
# Should show: supported=1 detected=1

# Test with raspistill
raspistill -o test.jpg

# If not detected:
sudo raspi-config
# → Interface Options → Camera → Enable
sudo reboot
```

**USB Webcam:**
```bash
# List video devices
ls /dev/video*

# Test with v4l2
v4l2-ctl --list-devices
```

### Issue: Camera lag or stuttering

1. **Reduce resolution:**
   ```yaml
   capture:
     resolution: [640, 480]  # Lower from 1280x720
   ```

2. **Reduce FPS:**
   ```yaml
   capture:
     target_fps: 15  # Lower from 30
   ```

3. **Check GPU memory:**
   ```bash
   vcgencmd get_mem gpu
   # Should be at least 128MB
   
   # Increase in /boot/config.txt:
   gpu_mem=256
   ```

---

## GPIO Issues

### Issue: "RPi.GPIO not available"

```bash
# Install RPi.GPIO
pip install RPi.GPIO

# If permission error:
sudo pip install RPi.GPIO
# Or better, use venv with proper permissions
```

### Issue: "Cannot determine SOC peripheral base address"

Running outside Raspberry Pi. GPIO will use stub automatically.

### Issue: GPIO pins not responding

1. **Check pin mode:**
   ```bash
   gpio readall  # Shows all pin states
   ```

2. **Check for conflicts:**
   ```bash
   # See what's using GPIOs
   sudo cat /sys/kernel/debug/gpio
   ```

3. **Reset GPIO:**
   ```python
   import RPi.GPIO as GPIO
   GPIO.setmode(GPIO.BCM)
   GPIO.cleanup()
   ```

---

## Service Issues

### Issue: Service won't start

```bash
# Check status
sudo systemctl status driver-assistant

# View detailed logs
journalctl -u driver-assistant -n 100 --no-pager

# Common fixes:
# 1. Check working directory exists
ls -la /home/pi/Driver-Assistant

# 2. Check venv exists
ls -la /home/pi/Driver-Assistant/venv/bin/python

# 3. Check permissions
sudo chown -R pi:pi /home/pi/Driver-Assistant
```

### Issue: Service keeps restarting

Check for crash loops:
```bash
journalctl -u driver-assistant -f
```

Common causes:
- Camera not available
- Missing dependencies
- Config file syntax error

Test manually:
```bash
cd /home/pi/Driver-Assistant
source venv/bin/activate
python -m src.main --source csi --headless
```

### Issue: Service starts but no alerts

1. Check logs for errors
2. Verify audio output:
   ```bash
   aplay -l  # List audio devices
   speaker-test -t sine -f 1000  # Test speaker
   ```

---

## Performance Issues

### Issue: Low FPS (< 10 FPS)

1. **Check CPU temperature:**
   ```bash
   vcgencmd measure_temp
   # If > 80°C, add cooling
   ```

2. **Check CPU usage:**
   ```bash
   top -d 1
   ```

3. **Reduce YOLO skip frames:**
   ```yaml
   yolo:
     skip_frames: 3  # Increase to skip more frames
   ```

4. **Disable unused features:**
   ```yaml
   lane_detection:
     enabled: false  # If not needed
   ```

### Issue: Memory usage too high

```bash
# Check memory
free -h

# If swap is heavily used:
# 1. Reduce image resolution
# 2. Disable display if headless
# 3. Increase swap (not recommended for SD cards)
```

### Issue: SD card corruption

Symptoms: Random crashes, filesystem errors

Prevention:
1. Use high-quality SD card (Class 10+)
2. Enable read-only filesystem
3. Use journaling filesystem
4. Proper shutdown: `sudo shutdown -h now`

---

## Diagnostic Commands

### System Information

```bash
# Pi model
cat /proc/device-tree/model

# OS version
cat /etc/os-release

# Kernel version
uname -a

# Memory
free -h

# Disk space
df -h

# Temperature
vcgencmd measure_temp

# CPU frequency
vcgencmd measure_clock arm
```

### Hardware Check

```bash
# Camera
vcgencmd get_camera
raspistill -o test.jpg

# UART
ls -la /dev/ttyAMA0
cat /boot/config.txt | grep uart

# GPIO
gpio readall
cat /sys/kernel/debug/gpio

# I2C (if used)
i2cdetect -y 1
```

### Service Diagnostics

```bash
# Service status
sudo systemctl status driver-assistant

# Recent logs
journalctl -u driver-assistant -n 50

# Live logs
journalctl -u driver-assistant -f

# Boot logs
journalctl -b -u driver-assistant
```

### Python/Application Check

```bash
cd /home/pi/Driver-Assistant
source venv/bin/activate

# Check dependencies
pip list

# Test imports
python -c "import cv2; print(cv2.__version__)"
python -c "import onnxruntime; print(onnxruntime.__version__)"
python -c "from src.sensors import create_lidar; print('LiDAR OK')"
python -c "from src.gpio import create_gpio_controller; print('GPIO OK')"

# Run tests
pytest tests/test_hardware.py -v
```

### LiDAR Diagnostics

```bash
# Test serial port
sudo minicom -D /dev/ttyAMA0 -b 115200

# Python test
python -c "
from src.sensors import create_lidar
lidar = create_lidar()
print(f'Type: {type(lidar).__name__}')
if lidar.connect():
    print('Connected')
    lidar.start()
    import time
    time.sleep(1)
    r = lidar.get_reading()
    print(f'Reading: {r}')
    lidar.stop()
else:
    print('Connection failed')
"
```

### GPIO Diagnostics

```bash
# Python test
python -c "
from src.gpio import create_gpio_controller
gpio = create_gpio_controller()
print(f'Type: {type(gpio).__name__}')
if gpio.initialize():
    print('Initialized')
    gpio.set_system_led(True)
    print('System LED ON')
    import time
    time.sleep(2)
    gpio.cleanup()
else:
    print('Init failed')
"
```

---

## Quick Fixes Checklist

- [ ] Reboot the Pi: `sudo reboot`
- [ ] Restart the service: `sudo systemctl restart driver-assistant`
- [ ] Check logs: `journalctl -u driver-assistant -n 50`
- [ ] Verify config.yaml syntax
- [ ] Check all cable connections
- [ ] Update software: `cd ~/Driver-Assistant && git pull && pip install -r requirements-pi.txt`
- [ ] Check temperature: `vcgencmd measure_temp`
- [ ] Free up disk space: `sudo apt autoremove && sudo apt clean`

---

## Getting Help

If issues persist:

1. Collect diagnostic info:
   ```bash
   journalctl -u driver-assistant -n 200 > diagnostic.log
   vcgencmd get_camera >> diagnostic.log
   cat /boot/config.txt >> diagnostic.log
   free -h >> diagnostic.log
   ```

2. Open an issue on GitHub with:
   - diagnostic.log contents
   - Hardware setup description
   - Steps to reproduce
