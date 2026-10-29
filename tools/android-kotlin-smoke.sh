#!/usr/bin/env bash
# Install the Kotlin example APK on the connected device, run each of its modes,
# and check that the frames reach the screen through surfacesink and every frame
# is consumed. The device must be unlocked, and the back camera must face
# something lit.
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
# A picture starts at each IDR or non-IDR slice (NAL types 5 and 1) whose first_mb_in_slice is 0, a leading 1 bit.
CLIP_FRAME_COUNT="$(unzip -p "$APK" 'assets/*.h264' | python3 -c '
import re, sys
data = sys.stdin.buffer.read()
print(sum(1 for start in re.finditer(b"\x00\x00\x01", data)
          if data[start.end()] & 0x1f in (1, 5) and data[start.end() + 1] & 0x80))
')"

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
adb shell pm grant "$PACKAGE" android.permission.CAMERA

# Prints the share of strongly coloured pixels and the spread of brightness, both in percent,
# over the middle half of the screen so the status and navigation bars do not count.
# screencap without -p is a 16-byte header (width, height, format, colorspace) then RGBA rows.
measure_screen() {
    adb exec-out screencap | python3 -c '
import statistics, struct, sys
data = sys.stdin.buffer.read()
width, height = struct.unpack_from("<II", data)
row_bytes = width * 4
pixels = memoryview(data)[16 + height // 4 * row_bytes:16 + height * 3 // 4 * row_bytes]
samples = [pixels[i:i + 3] for i in range(0, len(pixels), 4 * 97)]
colored = sum(1 for rgb in samples if max(rgb) - min(rgb) > 96)
brightness = [sum(rgb) / 3 for rgb in samples]
print(colored * 100 // len(samples), int(statistics.pstdev(brightness) * 100 // 255))
'
}

# Prints how many rows in the middle half of the screen differ between two screenshots half a second apart.
count_changed_rows() {
    python3 -c '
import subprocess, struct, time
def grab():
    return subprocess.run(["adb", "exec-out", "screencap"], check=True, capture_output=True).stdout
first = grab()
time.sleep(0.5)
second = grab()
width, height = struct.unpack_from("<II", first)
row_bytes = width * 4
rows = range(height // 4, height * 3 // 4)
print(sum(1 for row in rows
          if first[16 + row * row_bytes:16 + (row + 1) * row_bytes]
          != second[16 + row * row_bytes:16 + (row + 1) * row_bytes]))
'
}

# Runs one mode of the example and checks all of its frames were consumed with the app in the foreground.
run_mode() {
    local mode="$1" frame_count="$2"
    adb logcat -c
    adb shell am start -n "$PACKAGE/.MainActivity" --es mode "$mode" >/dev/null
    sleep 2
    CHANGED_ROWS="$(count_changed_rows)"
    local foreground
    foreground="$(adb shell dumpsys activity activities | grep -c "topResumedActivity.*$PACKAGE" || true)"
    read -r COLORED SPREAD < <(measure_screen)
    echo ">> $mode: coloured pixels $COLORED%, brightness spread $SPREAD%, rows changed in 0.5 s $CHANGED_ROWS"
    for _ in $(seq 30); do
        adb logcat -d -s 'g2g-kotlin:*' | grep -q "consumed\|failed" && break
        sleep 1
    done
    local log
    log="$(adb logcat -d -s 'g2g-kotlin:*' 'AndroidRuntime:E')"
    echo "$log"
    adb shell am force-stop "$PACKAGE"
    echo "$log" | grep -q "$mode consumed $frame_count of $frame_count frames" \
        || { echo ">> $mode: not every frame was consumed" >&2; exit 1; }
    [ "$foreground" -ge 1 ] || { echo ">> $mode: the example was not in the foreground" >&2; exit 1; }
}

run_mode bars "$FRAME_COUNT"
[ "$COLORED" -ge 30 ] || { echo ">> the bars did not reach the screen" >&2; exit 1; }
[ "$CHANGED_ROWS" -ge 1 ] || { echo ">> the frame number on the bars did not change on screen" >&2; exit 1; }
run_mode camera "$FRAME_COUNT"
[ "$SPREAD" -ge 5 ] || { echo ">> the screen shows a flat colour, not the camera image" >&2; exit 1; }
run_mode decode "$CLIP_FRAME_COUNT"
[ "$COLORED" -ge 30 ] || { echo ">> the decoded clip did not reach the screen" >&2; exit 1; }
echo ">> Kotlin surfacesink: OK"
