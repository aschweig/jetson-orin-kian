#!/usr/bin/env bash
# Remove Kian's Jetson memory tuning and restore defaults.
# Usage: sudo scripts/uninstall-sysctl.sh
set -euo pipefail

CONF="/etc/sysctl.d/80-kian-gpu-memory.conf"

if [ "$(id -u)" -ne 0 ]; then
    echo "Error: must run as root (sudo $0)" >&2
    exit 1
fi

if [ ! -f "$CONF" ]; then
    echo "Nothing to remove: $CONF not found."
    exit 0
fi

rm "$CONF"
echo "Removed $CONF"

sysctl --system
echo "Done. Current values:"
sysctl vm.min_free_kbytes vm.vfs_cache_pressure vm.watermark_scale_factor
