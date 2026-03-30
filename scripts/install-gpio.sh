#!/usr/bin/env bash
# Install and enable the GPIO pinmux service (runs every boot).
# Usage: sudo ./scripts/install-gpio.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

ln -sf "$REPO_DIR/kian-gpio.service" /etc/systemd/system/kian-gpio.service
systemctl daemon-reload
systemctl enable kian-gpio.service
echo "Installed and enabled kian-gpio.service (runs every boot)"
echo ""
echo "Commands:"
echo "  sudo systemctl status kian-gpio   # check status"
echo "  sudo systemctl start kian-gpio    # run now without rebooting"
