#!/usr/bin/env bash
set -euo pipefail

# Start PulseAudio if not already running
if ! pulseaudio --check 2>/dev/null; then
    echo "Starting PulseAudio..."
    pulseaudio --start --exit-idle-time=-1
    sleep 1
else
    echo "PulseAudio already running."
fi

SPEAKER_MATCH="${1:-}"
MIC_MATCH="${2:-$SPEAKER_MATCH}"

# If no arguments, list devices and exit
if [ -z "$SPEAKER_MATCH" ]; then
    echo ""
    echo "=== Speakers (sinks) ==="
    pactl list short sinks | while read -r idx name driver format state; do
        desc=$(pactl list sinks | grep -A1 "Name: $name" | grep "Description:" | sed 's/.*Description: //')
        echo "  $name"
        echo "    $desc"
    done

    echo ""
    echo "=== Microphones (sources) ==="
    pactl list short sources | grep -v '\.monitor\b' | while read -r idx name driver format state; do
        desc=$(pactl list sources | grep -A1 "Name: $name" | sed -n 's/.*Description: //p')
        echo "  $name"
        echo "    $desc"
    done

    echo ""
    echo "Usage: $0 <speaker_match> [mic_match]"
    echo "  If only one argument is given, it is used for both speaker and mic."
    exit 0
fi

# Find and set default sink (speaker)
SINK=$(pactl list short sinks | grep -i "$SPEAKER_MATCH" | head -1 | cut -f2)
if [ -z "$SINK" ]; then
    echo "ERROR: No sink matching '$SPEAKER_MATCH'"
    echo "Available sinks:"
    pactl list short sinks
    exit 1
fi
pactl set-default-sink "$SINK"
echo "Default speaker: $SINK"

# Find and set default source (mic)
SOURCE=$(pactl list short sources | grep -v '\.monitor\b' | grep -i "$MIC_MATCH" | head -1 | cut -f2)
if [ -z "$SOURCE" ]; then
    echo "ERROR: No source matching '$MIC_MATCH'"
    echo "Available sources:"
    pactl list short sources | grep -v '\.monitor$'
    exit 1
fi
pactl set-default-source "$SOURCE"
echo "Default mic: $SOURCE"

echo ""
echo "Audio configured. Ready to run kian."
