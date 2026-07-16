#!/usr/bin/env bash
# FreeRTOS executor proof (M633): build the g2g-noalloc staticlib (the exact
# pipeline the no-heap / panic-free symbol proofs cover), link it into a
# FreeRTOS application (static allocation only, no FreeRTOS heap either) with
# the arm-none-eabi C toolchain, and boot it on QEMU's MPS2-AN386 Cortex-M4.
# A FreeRTOS task calls g2g_noalloc_run() and verifies the wire checksum
# on-target: the C-shop integration path, proven end to end.
#
# Usage: tools/freertos-check.sh
# Requires: rustup target thumbv7em-none-eabihf; arm-none-eabi-gcc + newlib
# (override with $FREERTOS_CC); qemu-system-arm (override with
# $QEMU_SYSTEM_ARM); network on first run (clones the pinned FreeRTOS kernel).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/examples/g2g-freertos"
TARGET="thumbv7em-none-eabihf"
LIB="$ROOT/examples/g2g-noalloc/target/$TARGET/release/libg2g_noalloc.a"
OUT="$SRC/target"
CC="${FREERTOS_CC:-arm-none-eabi-gcc}"
QEMU="${QEMU_SYSTEM_ARM:-qemu-system-arm}"

# The FreeRTOS kernel, pinned. Cached under the example's (gitignored) target
# dir so repeat runs are offline.
KERNEL_TAG="V11.2.0"
KDIR="$OUT/FreeRTOS-Kernel"

echo "== building g2g-noalloc staticlib for $TARGET =="
rustup target add "$TARGET" >/dev/null 2>&1 || true
cargo build --manifest-path "$ROOT/examples/g2g-noalloc/Cargo.toml" --release --target "$TARGET"

command -v "${CC%% *}" >/dev/null 2>&1 \
  || { echo "FAIL: $CC not found (install arm-none-eabi-gcc or set \$FREERTOS_CC)"; exit 1; }

mkdir -p "$OUT"
if [ ! -d "$KDIR" ]; then
  echo "== fetching FreeRTOS-Kernel $KERNEL_TAG =="
  git clone --depth 1 --branch "$KERNEL_TAG" -q \
    https://github.com/FreeRTOS/FreeRTOS-Kernel.git "$KDIR"
fi

echo "== cross-compiling the FreeRTOS application =="
CFLAGS="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O2 \
  -ffunction-sections -fdata-sections -Wall -Werror \
  -I$SRC -I$KDIR/include -I$KDIR/portable/GCC/ARM_CM4F"
ELF="$OUT/g2g-freertos.elf"
$CC $CFLAGS \
  "$SRC/main.c" "$SRC/startup.c" \
  "$KDIR/tasks.c" "$KDIR/list.c" "$KDIR/portable/GCC/ARM_CM4F/port.c" \
  "$LIB" \
  -T "$SRC/mps2_an386.ld" -nostartfiles -Wl,--gc-sections \
  -specs=nano.specs -specs=nosys.specs \
  -o "$ELF"

echo "== running on QEMU MPS2-AN386 (Cortex-M4) =="
# 2>&1: raw semihosting SYS_WRITE0 goes to qemu's *stderr* console (unlike the
# Rust examples' SYS_OPEN(":tt")+SYS_WRITE path, which lands on stdout).
OUT_TEXT="$(timeout 120 $QEMU -machine mps2-an386 -cpu cortex-m4 -nographic \
  -semihosting-config enable=on,target=native -kernel "$ELF" 2>&1)"
echo "$OUT_TEXT"

grep -q "video + flagship audio ran under FreeRTOS on Cortex-M4, checksums OK" <<<"$OUT_TEXT" \
  || { echo "FAIL: expected banner missing from the FreeRTOS QEMU run"; exit 1; }

echo "PASS: the g2g-noalloc pipeline executed under a FreeRTOS task on an"
echo "      emulated Cortex-M4 (static allocation only, wire checksum verified"
echo "      on-target, semihosting exit 0)."
