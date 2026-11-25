#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# Pair the Moonlight flatpak with `g2g-desktop host --protocol gamestream` and list its apps.
# Moonlight runs on its own Xvfb display, so nothing opens on the desktop.
#
# Prerequisites: Xvfb, and Moonlight from Flathub:
#   flatpak install --user flathub com.moonlight_stream.Moonlight
#
# Usage: tools/moonlight-interop.sh <g2g-desktop binary built with --features gamestream>
# Logs go to $OUT, default ~/tmp/moonlight-interop.
set -euo pipefail

BIN=$1
OUT=${OUT:-$HOME/tmp/moonlight-interop}
PIN=4721
# each run starts unpaired: fresh config directories for Moonlight and the host
rm -rf "$OUT"
mkdir -p "$OUT/moonlight-config" "$OUT/host-config"
# flatpak overrides XDG_CONFIG_HOME from outside, so a shell inside the sandbox sets it
moonlight() {
    flatpak run --nosocket=wayland --socket=x11 --env=QT_QPA_PLATFORM=xcb \
        --filesystem="$OUT" --command=sh com.moonlight_stream.Moonlight \
        -c 'XDG_CONFIG_HOME="$0" exec moonlight "$@"' "$OUT/moonlight-config" "$@"
}

exec 4< <(Xvfb -displayfd 3 -screen 0 1280x720x24 -nolisten tcp 3>&1 >/dev/null 2>&1 & echo "$!")
read -r XVFB <&4
read -r DISPLAY_NUMBER <&4
export DISPLAY=":$DISPLAY_NUMBER"
unset WAYLAND_DISPLAY

XDG_CONFIG_HOME="$OUT/host-config" "$BIN" host --protocol gamestream >"$OUT/host.log" 2>&1 &
HOST=$!
trap 'kill $HOST $XVFB 2>/dev/null || true' EXIT
for _ in $(seq 240); do
    grep -q 'Moonlight host' "$OUT/host.log" && break
    sleep 0.5
done

moonlight pair 127.0.0.1 --pin "$PIN" >"$OUT/pair.log" 2>&1 &
PAIR=$!
for _ in $(seq 120); do
    grep -q 'asks to pair' "$OUT/host.log" && break
    sleep 0.25
done
grep -q 'asks to pair' "$OUT/host.log"
"$BIN" pin "$PIN"
# moonlight's pair command waits on a "Pairing completed" dialog, so it is closed once the host has paired it
for _ in $(seq 120); do
    grep -q 'phrase=pairchallenge' "$OUT/pair.log" && break
    sleep 0.25
done
sleep 1
kill "$PAIR"
grep -q 'paired a Moonlight client' "$OUT/host.log"

moonlight list 127.0.0.1 >"$OUT/list.log" 2>&1
grep -qx 'Desktop' "$OUT/list.log"
echo "moonlight-interop: paired and listed the Desktop app"
