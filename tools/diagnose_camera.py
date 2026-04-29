#!/usr/bin/env python3
"""
CSI Camera Diagnostic Tool for Raspberry Pi.

Run this script to diagnose camera initialization issues.

Usage:
    python3 tools/diagnose_camera.py
"""

import sys
import time
import subprocess


def run_command(cmd: str) -> tuple:
    """Run a shell command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def check_step(name: str, passed: bool, details: str = ""):
    """Print a diagnostic step result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {name}")
    if details and not passed:
        for line in details.strip().split('\n')[:5]:  # Limit output
            print(f"       {line}")
    return passed


def main():
    print("=" * 60)
    print("CSI Camera Diagnostic Tool")
    print("=" * 60)
    
    all_passed = True
    
    # 1. Check if running on Raspberry Pi
    print("\n📋 System Checks:")
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip()
            is_pi = "Raspberry Pi" in model
            all_passed &= check_step("Raspberry Pi detected", is_pi, model)
    except Exception as e:
        all_passed &= check_step("Raspberry Pi detected", False, str(e))
    
    # 2. Check libcamera is installed
    success, output = run_command("libcamera-hello --version")
    all_passed &= check_step("libcamera installed", success, output)
    
    # 3. List available cameras
    print("\n📷 Camera Detection:")
    success, output = run_command("libcamera-hello --list-cameras 2>&1")
    has_camera = success and ("Available cameras" in output or ":" in output)
    all_passed &= check_step("Camera detected by libcamera", has_camera, output)
    
    # 4. Check for processes using camera
    print("\n🔒 Resource Checks:")
    success, output = run_command("lsof /dev/video* 2>/dev/null")
    no_conflicts = not success or output.strip() == ""
    if not no_conflicts:
        print("       Processes using camera:")
        print(f"       {output[:200]}")
    all_passed &= check_step("No processes blocking camera", no_conflicts, output)
    
    # 5. Check video devices exist
    success, output = run_command("ls -la /dev/video* 2>&1")
    all_passed &= check_step("Video devices exist", success and "/dev/video" in output, output)
    
    # 6. Check user permissions
    success, output = run_command("groups")
    in_video_group = "video" in output
    all_passed &= check_step("User in 'video' group", in_video_group, output)
    
    # 7. Check picamera2 import
    print("\n📦 Python Checks:")
    try:
        from picamera2 import Picamera2
        all_passed &= check_step("picamera2 importable", True)
        
        # Check available cameras via picamera2
        try:
            cameras = Picamera2.global_camera_info()
            has_cameras = len(cameras) > 0
            all_passed &= check_step(
                "Cameras detected by picamera2", 
                has_cameras, 
                f"Found: {cameras}" if cameras else "No cameras found"
            )
        except Exception as e:
            all_passed &= check_step("Cameras detected by picamera2", False, str(e))
            
    except ImportError as e:
        all_passed &= check_step("picamera2 importable", False, str(e))
    
    # 8. Try to initialize camera
    print("\n🚀 Camera Initialization Test:")
    try:
        from picamera2 import Picamera2
        
        print("   Creating Picamera2 instance...")
        cam = Picamera2(camera_num=0)
        
        print("   Configuring camera...")
        config = cam.create_video_configuration(
            main={"size": (640, 480), "format": "BGR888"},
            buffer_count=4,
        )
        cam.align_configuration(config)
        cam.configure(config)
        
        print("   Starting camera...")
        cam.start()
        time.sleep(1.0)
        
        print("   Capturing test frame...")
        frame = cam.capture_array("main")
        
        if frame is not None:
            all_passed &= check_step(
                "Camera capture successful", 
                True, 
                f"Frame shape: {frame.shape}, dtype: {frame.dtype}"
            )
            print(f"       Frame shape: {frame.shape}")
            print(f"       Frame dtype: {frame.dtype}")
        else:
            all_passed &= check_step("Camera capture successful", False, "Got None frame")
        
        print("   Stopping camera...")
        cam.stop()
        
        print("   Closing camera...")
        cam.close()
        time.sleep(0.2)
        
        all_passed &= check_step("Camera cleanup successful", True)
        
    except RuntimeError as e:
        error_msg = str(e)
        all_passed &= check_step("Camera initialization", False, error_msg)
        
        if "Device or resource busy" in error_msg:
            print("\n⚠️  SOLUTION: Camera is being used by another process!")
            print("   Run these commands:")
            print("   $ sudo pkill -f libcamera")
            print("   $ sudo pkill -f python")
            print("   $ sudo fuser -k /dev/video0")
            
    except Exception as e:
        all_passed &= check_step("Camera initialization", False, f"{type(e).__name__}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All checks passed! Camera should work correctly.")
    else:
        print("❌ Some checks failed. See above for details.")
        print("\nCommon fixes:")
        print("  1. Kill blocking processes: sudo pkill -f python")
        print("  2. Add user to video group: sudo usermod -aG video $USER")
        print("  3. Enable camera: sudo raspi-config → Interface → Camera")
        print("  4. Reboot: sudo reboot")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
