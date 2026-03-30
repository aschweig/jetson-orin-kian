#!/usr/bin/env bash
# Install (or update) the Kian systemd user service.
# Usage: ./scripts/install-service.sh [--enable]
#   --enable  also enable start-on-boot (via lingering user session)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p ~/.config/systemd/user
ln -sf "$REPO_DIR/kian.service" ~/.config/systemd/user/kian.service
systemctl --user daemon-reload
echo "Installed kian.service (user service)"

if [[ "${1:-}" == "--enable" ]]; then
    sudo loginctl enable-linger "$USER"
    systemctl --user enable kian.service
    echo "Enabled kian on boot"
fi

echo ""
echo "Commands:"
echo "  systemctl --user start kian      # start now"
echo "  systemctl --user stop kian       # stop"
echo "  systemctl --user status kian     # check status"
echo "  journalctl --user -u kian -f     # tail logs"
echo "  systemctl --user enable kian     # enable on boot"
echo "  systemctl --user disable kian    # disable on boot"
