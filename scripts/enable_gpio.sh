#!/usr/bin/env bash
# Set up Jetson GPIO access.
#
# Two modes:
#   sudo ./enable_gpio.sh --setup   One-time: create gpio group, add user, install udev rules
#   sudo ./enable_gpio.sh           Boot-time: configure pinmux (resets every reboot)
set -euo pipefail

if [[ "${1:-}" == "--setup" ]]; then
    # --- One-time setup (run manually once) ---
    RULES_SRC=".venv/lib/python3.12/site-packages/Jetson/GPIO/99-gpio.rules"

    # Create gpio group and add current user
    groupadd -f gpio
    usermod -aG gpio "${SUDO_USER:-$USER}"

    # Install udev rules
    if [ -f "$RULES_SRC" ]; then
        cp "$RULES_SRC" /etc/udev/rules.d/99-gpio.rules
    else
        echo "SUBSYSTEM==\"gpio\", KERNEL==\"gpiochip*\", ACTION==\"add\", PROGRAM=\"/bin/sh -c 'chown root:gpio /sys/class/gpio/export /sys/class/gpio/unexport; chmod 220 /sys/class/gpio/export /sys/class/gpio/unexport'\"
SUBSYSTEM==\"gpio\", KERNEL==\"gpio*\", ACTION==\"add\", PROGRAM=\"/bin/sh -c 'chown root:gpio /sys%p/active_low /sys%p/direction /sys%p/edge /sys%p/value; chmod 660 /sys%p/active_low /sys%p/direction /sys%p/edge /sys%p/value'\"" \
            > /etc/udev/rules.d/99-gpio.rules
    fi

    # Reload udev
    udevadm control --reload-rules
    udevadm trigger

    echo "One-time GPIO setup done. Log out and back in for group changes to take effect."
fi

# --- Pinmux config (needed every boot) ---
busybox devmem 0x2430068 w 0x8   # pin 29 (red)
busybox devmem 0x2430070 w 0x8   # pin 31 (yellow)
busybox devmem 0x2434080 w 0x5   # pin 32 (green)
echo "Pinmux configured for GPIO pins 29, 31, 32."
