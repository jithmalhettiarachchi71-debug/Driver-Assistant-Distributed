#!/bin/bash
# install-service.sh - Install systemd service for Vehicle Safety Alert System
# 
# Usage:
#   sudo ./systemd/install-service.sh
#
# This script:
# 1. Copies the service file to /etc/systemd/system/
# 2. Reloads systemd daemon
# 3. Enables the service for auto-start
# 4. Starts the service
#
# Prerequisites:
# - Run as root (sudo)
# - Python virtual environment set up at /home/pi/Driver-Assistant/venv
# - config.yaml configured for your hardware

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="driver-assistant"
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo_error "This script must be run as root (use sudo)"
    exit 1
fi

# Check if service file exists
if [[ ! -f "$SERVICE_FILE" ]]; then
    echo_error "Service file not found: $SERVICE_FILE"
    exit 1
fi

# Check if virtual environment exists
if [[ ! -f "$PROJECT_DIR/venv/bin/python" ]]; then
    echo_error "Virtual environment not found at $PROJECT_DIR/venv"
    echo_info "Create it with: python3 -m venv venv && source venv/bin/activate && pip install -r requirements-pi.txt"
    exit 1
fi

echo_info "Installing $SERVICE_NAME service..."

# Stop existing service if running
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo_info "Stopping existing service..."
    systemctl stop "$SERVICE_NAME"
fi

# Copy service file
echo_info "Copying service file to /etc/systemd/system/"
cp "$SERVICE_FILE" /etc/systemd/system/

# Reload systemd
echo_info "Reloading systemd daemon..."
systemctl daemon-reload

# Enable service
echo_info "Enabling service for auto-start..."
systemctl enable "$SERVICE_NAME"

# Start service
echo_info "Starting service..."
systemctl start "$SERVICE_NAME"

# Wait a moment for startup
sleep 2

# Check status
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo_info "Service started successfully!"
    echo ""
    echo "Service Status:"
    systemctl status "$SERVICE_NAME" --no-pager -l
else
    echo_error "Service failed to start. Check logs with:"
    echo "  journalctl -u $SERVICE_NAME -n 50"
fi

echo ""
echo "Useful commands:"
echo "  sudo systemctl status $SERVICE_NAME   # Check status"
echo "  sudo systemctl stop $SERVICE_NAME     # Stop service"
echo "  sudo systemctl start $SERVICE_NAME    # Start service"
echo "  sudo systemctl restart $SERVICE_NAME  # Restart service"
echo "  journalctl -u $SERVICE_NAME -f        # View live logs"
