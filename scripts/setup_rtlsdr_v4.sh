#!/usr/bin/env bash
# =============================================================================
# setup_rtlsdr_v4.sh - Install RTL-SDR v4 drivers and tools on Linux
# =============================================================================
set -euo pipefail

LOG_PREFIX="[rtl-sdr-setup]"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_PREFIX} $*"
}

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
log "Step 1/7: Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  libusb-1.0-0-dev \
  pkg-config \
  libatlas-base-dev

log "Dependencies installed."

# ---------------------------------------------------------------------------
# 2. Blacklist conflicting drivers
# ---------------------------------------------------------------------------
log "Step 2/7: Blacklisting conflicting kernel drivers..."

CONFLICTING_DRIVERS=(
  "dvb_usb_rtl28xxu"
  "rtl2832"
  "rtl2830"
)

for driver in "${CONFLICTING_DRIVERS[@]}"; do
  BLACKLIST_FILE="/etc/modprobe.d/blacklist-${driver}.conf"
  if [ -f "$BLACKLIST_FILE" ]; then
    log "  Blacklist for '${driver}' already exists, skipping."
  else
    echo "blacklist ${driver}" | sudo tee "$BLACKLIST_FILE" > /dev/null
    log "  Blacklisted '${driver}'."
  fi
done

# Unload conflicting modules if currently loaded
for driver in "${CONFLICTING_DRIVERS[@]}"; do
  if lsmod | grep -q "^${driver}"; then
    log "  Unloading '${driver}'..."
    sudo rmmod "$driver" 2>/dev/null || true
  fi
done

log "Conflicting drivers blacklisted."

# ---------------------------------------------------------------------------
# 3. Clone and build rtl-sdr from source
# ---------------------------------------------------------------------------
log "Step 3/7: Building rtl-sdr from source..."

BUILD_DIR="/tmp/rtl-sdr-build"
INSTALL_PREFIX="/usr/local"

rm -rf "$BUILD_DIR"
git clone https://github.com/rtlsdrblog/rtl-sdr-blog "$BUILD_DIR"
cd "$BUILD_DIR"

mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
make -j"$(nproc)"
sudo make install
sudo ldconfig

cd /
rm -rf "$BUILD_DIR"

log "rtl-sdr built and installed to ${INSTALL_PREFIX}."

# ---------------------------------------------------------------------------
# 4. Install udev rules
# ---------------------------------------------------------------------------
log "Step 4/7: Installing udev rules..."

UDEV_RULE="/etc/udev/rules.d/20-rtlsdr.rules"
if [ -f "$UDEV_RULE" ]; then
  log "  udev rules already installed, skipping."
else
  sudo tee "$UDEV_RULE" > /dev/null <<'EOF'
# RTL-SDR
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2832", MODE:="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", MODE:="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2830", MODE:="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2831", MODE:="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2836", MODE:="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2837", MODE:="0666"
EOF
  log "  udev rules installed."
fi

sudo udevadm control --reload-rules
sudo udevadm trigger

log "udev rules applied."

# ---------------------------------------------------------------------------
# 5. Add user to plugdev group
# ---------------------------------------------------------------------------
log "Step 5/7: Adding user '${USER}' to plugdev group..."

if getent group plugdev > /dev/null 2>&1; then
  sudo usermod -aG plugdev "${USER}" || true
  log "  Added '${USER}' to plugdev group."
else
  log "  plugdev group not found, creating..."
  sudo groupadd plugdev
  sudo usermod -aG plugdev "${USER}"
  log "  Created plugdev group and added '${USER}'."
fi

log "Note: You may need to log out and back in for group changes to take effect."

# ---------------------------------------------------------------------------
# 6. Verify installation
# ---------------------------------------------------------------------------
log "Step 6/7: Verifying installation..."

if command -v rtl_test > /dev/null 2>&1; then
  log "  rtl_test found, running device check..."
  rtl_test -t 2>&1 || log "  WARNING: rtl_test exited with non-zero code. Device may not be connected."
else
  log "  ERROR: rtl_test not found after installation!"
  exit 1
fi

log "Installation verified."

# ---------------------------------------------------------------------------
# 7. Install Python dependencies
# ---------------------------------------------------------------------------
log "Step 7/7: Installing Python dependencies..."

pip3 install pyrtlsdr numpy scipy

log "Python dependencies installed."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "============================================"
log "RTL-SDR v4 setup complete!"
log ""
log "Next steps:"
log "  1. Reconnect your RTL-SDR dongle"
log "  2. Verify with: rtl_test -t"
log "  3. Calibrate PPM with: rtl_test -p"
log "  4. Log out and back in for group changes"
log "============================================"
