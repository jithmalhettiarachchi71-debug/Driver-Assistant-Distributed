#!/bin/bash
# Driver Assistant - Raspberry Pi Setup Script
# Run this script after cloning the repository

set -e

echo "=========================================="
echo "Driver Assistant - Raspberry Pi Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Raspberry Pi${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo -e "${GREEN}Project directory: $PROJECT_DIR${NC}"

# 1. Update system
echo ""
echo "Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y

# 2. Install system dependencies
echo ""
echo "Step 2: Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libcamera-dev \
    libcap-dev \
    pulseaudio \
    alsa-utils

# 3. Create virtual environment
echo ""
echo "Step 3: Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# 4. Install Python dependencies
echo ""
echo "Step 4: Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements-pi.txt

# 5. Create logs directory
echo ""
echo "Step 5: Creating logs directory..."
mkdir -p logs

# 6. Enable camera (if not already enabled)
echo ""
echo "Step 6: Checking camera configuration..."
if ! grep -q "^start_x=1" /boot/config.txt 2>/dev/null; then
    echo -e "${YELLOW}Camera may not be enabled. Run 'sudo raspi-config' to enable it.${NC}"
fi

# 7. Add user to required groups
echo ""
echo "Step 7: Adding user to video, gpio, and audio groups..."
sudo usermod -aG video,gpio,audio $USER

# 8. Install systemd service
echo ""
echo "Step 8: Installing systemd service..."
# Update service file with correct username if not 'pi'
if [ "$USER" != "pi" ]; then
    sed -i "s/User=pi/User=$USER/g" scripts/driver-assistant.service
    sed -i "s/Group=pi/Group=$USER/g" scripts/driver-assistant.service
    sed -i "s|/home/pi|/home/$USER|g" scripts/driver-assistant.service
fi

sudo cp scripts/driver-assistant.service /etc/systemd/system/
sudo systemctl daemon-reload

echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "To test the system manually:"
echo "  source venv/bin/activate"
echo "  python driver_assistant.py --source csi --display"
echo ""
echo "To enable auto-start on boot:"
echo "  sudo systemctl enable driver-assistant"
echo "  sudo systemctl start driver-assistant"
echo ""
echo "To check service status:"
echo "  sudo systemctl status driver-assistant"
echo ""
echo "To view logs:"
echo "  tail -f logs/service.log"
echo "  journalctl -u driver-assistant -f"
echo ""
echo -e "${YELLOW}NOTE: Please reboot for group changes to take effect:${NC}"
echo "  sudo reboot"
