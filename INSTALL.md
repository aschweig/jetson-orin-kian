# Installing Kian as a Service

Kian uses two independent systemd services:

| Service | Scope | Purpose |
|---|---|---|
| `kian-gpio.service` | system | Configures GPIO pinmux at boot for LED access. Always enabled — pinmux resets every reboot. |
| `kian.service` | user | Runs `setup-audio.sh` to configure PulseAudio, then starts Kian. Optionally enabled. |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed at `~/.local/bin/uv`
- Models downloaded (`./scripts/download-models.sh`)
- Wiki database built (`uv run python scripts/build-wiki-db.py`) — downloads Simple Wikipedia (~350 MB) and builds an FTS5 index (~170 MB)
- USB audio device(s) connected

## GPIO Setup

### One-time setup

Create the gpio group, add your user, and install udev rules (only needed once):

```bash
sudo ./scripts/enable_gpio.sh --setup
```

Log out and back in for group changes to take effect.

### Install GPIO service

Pinmux resets every reboot, so this service should always be installed and enabled:

```bash
sudo ./scripts/install-gpio.sh
```

To apply immediately without rebooting:

```bash
sudo systemctl start kian-gpio
```

## Install Kian Service

```bash
# Install only (manual start/stop):
./scripts/install-service.sh

# Install and enable start-on-boot:
./scripts/install-service.sh --enable
```

The `--enable` flag turns on login lingering so the user service starts at boot without requiring a login session.

## Configuration

Edit `kian.env` in the repo root to configure audio devices and Kian arguments:

```bash
# Audio device match strings (passed to setup-audio.sh)
AUDIO_SPEAKER=eneric_USB2.0
AUDIO_MIC=AB13X

# Extra args for kian (e.g. --backend mlc)
KIAN_ARGS=
```

To find available audio devices, run `./scripts/setup-audio.sh` with no arguments.

After editing `kian.env`, restart the service for changes to take effect:

```bash
systemctl --user restart kian
```

## Headless vs Desktop Mode

Kian loads Whisper, an LLM, and Piper TTS into memory. On a Jetson, running these alongside a desktop session can cause memory pressure. Switch between modes depending on how you're using the device:

```bash
# Switch to headless (Kian starts on boot, no desktop)
sudo systemctl set-default multi-user.target
systemctl --user enable kian

# Switch to desktop (desktop starts on boot, Kian disabled)
sudo systemctl set-default graphical.target
systemctl --user disable kian
```

Reboot after switching for a clean start.

## Common Commands

```bash
# Kian
systemctl --user start kian
systemctl --user stop kian
systemctl --user status kian
systemctl --user enable kian        # enable on boot
systemctl --user disable kian       # disable on boot
journalctl --user -u kian -f        # tail logs

# GPIO
sudo systemctl status kian-gpio
sudo systemctl start kian-gpio      # re-run pinmux setup
```

## Updating

Since the install scripts symlink the service files, pulling changes to the repo is enough. Reload and restart after pulling:

```bash
git pull
sudo systemctl daemon-reload
systemctl --user daemon-reload
systemctl --user restart kian
```

## Uninstalling

```bash
# Kian
systemctl --user stop kian
systemctl --user disable kian
rm ~/.config/systemd/user/kian.service
systemctl --user daemon-reload

# GPIO (optional — harmless to leave enabled)
sudo systemctl stop kian-gpio
sudo systemctl disable kian-gpio
sudo rm /etc/systemd/system/kian-gpio.service
sudo systemctl daemon-reload
```
