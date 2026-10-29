#!/usr/bin/env bash
# Install the Kotlin example APK on the connected device, run it, and check that
# SMPTE bars reach the screen through surfacesink and every frame is consumed.
# The device must be unlocked.
#
# Usage: tools/android-kotlin-smoke.sh [apk]   (default: build one with android-kotlin-apk.sh)
set -euo pipefail

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APK="${1:-}"
if [ -z "$APK" ]; then
    APK="$("$TOOLS/android-kotlin-apk.sh" | tail -1)"
fi
PACKAGE=dev.g2g.example
FRAME_COUNT=300

if [ -z "$(adb devices | awk 'NR>1 && $2=="device"{print $1}')" ]; then
    echo "no authorised device; 'adb devices' shows:" >&2
    adb devices >&2
    exit 1
fi
adb shell input keyevent KEYCODE_WAKEUP
adb shell wm dismiss-keyguard 2>/dev/null || true
if adb shell dumpsys window 2>/dev/null | grep -q "isKeyguardShowing=true"; then
    echo "device is locked; unlock it (a PIN keyguard cannot be dismissed over adb)" >&2
    exit 1
fi

adb install -r "$APK" >/dev/null
adb logcat -c
adb shell am start -n "$PACKAGE/.MainActivity" >/dev/null
sleep 4
FOREGROUND="$(adb shell dumpsys activity activities | grep -c "topResumedActivity.*$PACKAGE" || true)"

# screencap without -p is a 16-byte header (width, height, format, colorspace) then RGBA rows.
COLORED="$(adb exec-out screencap | python3 -c '
import struct, sys
data = sys.stdin.buffer.read()
width, height = struct.unpack_from("<II", data)
pixels = memoryview(data)[16:16 + width * height * 4]
colored = sum(1 for i in range(0, len(pixels), 4 * 97)
              if max(pixels[i:i + 3]) - min(pixels[i:i + 3]) > 96)
print(colored * 100 // (len(pixels) // (4 * 97)))
')"
echo ">> colored pixels on screen: $COLORED%"

for _ in $(seq 30); do
    adb logcat -d -s 'g2g-kotlin:*' | grep -q "consumed\|failed" && break
    sleep 1
done
LOG="$(adb logcat -d -s 'g2g-kotlin:*' 'AndroidRuntime:E')"
echo "$LOG"
adb shell am force-stop "$PACKAGE"

echo "$LOG" | grep -q "consumed $FRAME_COUNT of $FRAME_COUNT frames" \
    || { echo ">> not every frame was consumed" >&2; exit 1; }
[ "$FOREGROUND" -ge 1 ] || { echo ">> the example was not in the foreground" >&2; exit 1; }
[ "$COLORED" -ge 30 ] || { echo ">> the bars did not reach the screen" >&2; exit 1; }
echo ">> Kotlin surfacesink: OK"
