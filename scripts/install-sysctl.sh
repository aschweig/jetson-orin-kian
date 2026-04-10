#!/usr/bin/env bash
# Install Jetson memory tuning for Kian and apply immediately.
# Usage: sudo scripts/install-sysctl.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="80-kian-gpu-memory.conf"
SRC="$SCRIPT_DIR/$CONF"
DEST="/etc/sysctl.d/$CONF"

if [ "$(id -u)" -ne 0 ]; then
    echo "Error: must run as root (sudo $0)" >&2
    exit 1
fi

cp "$SRC" "$DEST"
echo "Installed $DEST"

sysctl --system
echo "Done. Current values:"
sysctl vm.min_free_kbytes vm.vfs_cache_pressure vm.watermark_scale_factor
