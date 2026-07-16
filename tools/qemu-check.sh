#!/usr/bin/env bash
# Emulated Cortex-M execution proofs: build `examples/g2g-qemu` (M628, a
# bare-metal binary wrapping the exact g2g-noalloc pipeline the no-heap /
# panic-free symbol proofs run on) and `examples/g2g-embassy` (M632, the same
# pipeline future awaited by a real Embassy executor task, the production MCU
# shape) and execute both on QEMU's MPS2-AN386 (Cortex-M4). This upgrades
# "links for thumbv7em" to "executes on the Cortex-M ISA": real Thumb-2 code,
# real 32-bit pointers, no host stand-in. Each binary asserts its own checksum
# via the semihosting exit code; the script additionally asserts the banners,
# so a silent no-op run cannot pass.
#
# Usage: tools/qemu-check.sh
# Requires: rustup target thumbv7em-none-eabihf; qemu-system-arm (override the
# binary with $QEMU_SYSTEM_ARM, e.g. a podman wrapper on hosts without it).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="thumbv7em-none-eabihf"
ELF="$ROOT/examples/g2g-qemu/target/$TARGET/release/g2g-qemu"
QEMU="${QEMU_SYSTEM_ARM:-qemu-system-arm}"

echo "== building g2g-qemu for $TARGET =="
rustup target add "$TARGET" >/dev/null 2>&1 || true
# Build from inside the crate: its .cargo/config.toml carries the -Tlink.x
# rustflags (the vector table), and cargo only reads that config cwd-relative
# (a --manifest-path build would silently drop it and produce an unbootable
# ELF).
(cd "$ROOT/examples/g2g-qemu" && cargo build --release --target "$TARGET")

command -v "${QEMU%% *}" >/dev/null 2>&1 \
  || { echo "FAIL: $QEMU not found (install qemu-system-arm or set \$QEMU_SYSTEM_ARM)"; exit 1; }

echo "== running on QEMU MPS2-AN386 (Cortex-M4) =="
# The pipeline is a proven-panic-free straight-line run; the timeout guards the
# harness itself (a hung QEMU or a future panic looping in the dead handler).
OUT="$(timeout 120 $QEMU -machine mps2-an386 -cpu cortex-m4 -nographic \
  -semihosting-config enable=on,target=native -kernel "$ELF")"
echo "$OUT"

grep -q "video + flagship audio ran on emulated Cortex-M4, checksums OK" <<<"$OUT" \
  || { echo "FAIL: expected banner missing from the QEMU run"; exit 1; }

EMBASSY_ELF="$ROOT/examples/g2g-embassy/target/$TARGET/release/g2g-embassy"

echo "== building g2g-embassy for $TARGET =="
(cd "$ROOT/examples/g2g-embassy" && cargo build --release --target "$TARGET")

echo "== running the Embassy executor variant on QEMU MPS2-AN386 =="
OUT2="$(timeout 120 $QEMU -machine mps2-an386 -cpu cortex-m4 -nographic \
  -semihosting-config enable=on,target=native -kernel "$EMBASSY_ELF")"
echo "$OUT2"

grep -q "video + flagship audio ran under Embassy on Cortex-M4, checksums OK" <<<"$OUT2" \
  || { echo "FAIL: expected banner missing from the Embassy QEMU run"; exit 1; }

# ISR-driven capture proof (M651): a SysTick interrupt handler is the producer,
# running in real interrupt context; the main-context pipeline drains the shared
# SPSC ring through `SpscCaptureSrc -> G.711 -> checksum`, sleeping on `wfi`
# between frames. Equal checksum to synchronous delivery means every
# interrupt-produced frame reached the pipeline in order, uncorrupted. This is
# the interrupt/DMA concurrency model on the Cortex-M ISA, built by the same
# `cargo build` above (a second bin in the crate).
ISR_ELF="$ROOT/examples/g2g-qemu/target/$TARGET/release/isr_capture"

echo "== running the ISR-driven capture proof on QEMU MPS2-AN386 =="
OUT3="$(timeout 120 $QEMU -machine mps2-an386 -cpu cortex-m4 -nographic \
  -semihosting-config enable=on,target=native -kernel "$ISR_ELF")"
echo "$OUT3"

grep -q "g2g-isr: captured=64 overruns=0 OK" <<<"$OUT3" \
  || { echo "FAIL: expected banner missing from the ISR-capture QEMU run"; exit 1; }

# Runtime fault-recovery proof (M652): the supervisor drives a capture -> G.711
# -> checksum pipeline through a mid-stream latched peripheral fault (retry, then
# reset via the FrameGrabber reset seam, then continue: all 64 frames delivered,
# wire checksum equal to a clean reference, watchdog fed once per frame) and then
# through a dead peripheral (escalates within its bounded retry/reset ladder, the
# watchdog is never fed). Bounded fault handling on the Cortex-M ISA. Built by the
# same `cargo build` above (a third bin in the crate).
SUP_ELF="$ROOT/examples/g2g-qemu/target/$TARGET/release/supervised"

echo "== running the runtime fault-recovery proof on QEMU MPS2-AN386 =="
OUT4="$(timeout 120 $QEMU -machine mps2-an386 -cpu cortex-m4 -nographic \
  -semihosting-config enable=on,target=native -kernel "$SUP_ELF")"
echo "$OUT4"

grep -q "g2g-supervise: delivered=64 resets=1 wd=64 escalated=4 OK" <<<"$OUT4" \
  || { echo "FAIL: expected banner missing from the supervisor QEMU run"; exit 1; }

# Receive-direction proof (M653): a mock network receiver hands the pipeline a
# reordered RTP/PCMU stream; RtpSrc -> JitterBuffer -> G.711 decode must
# reconstruct the ordered PCM, proved by an order-sensitive rolling hash equal to
# an independent in-order decode (a plain sum would not catch a reorder bug). The
# RX chain on the Cortex-M ISA. Built by the same `cargo build` above.
RX_ELF="$ROOT/examples/g2g-qemu/target/$TARGET/release/rx"

echo "== running the receive-direction (jitter buffer) proof on QEMU MPS2-AN386 =="
OUT5="$(timeout 120 $QEMU -machine mps2-an386 -cpu cortex-m4 -nographic \
  -semihosting-config enable=on,target=native -kernel "$RX_ELF")"
echo "$OUT5"

grep -q "g2g-rx: played=14 reordered=3 lost=0 OK" <<<"$OUT5" \
  || { echo "FAIL: expected banner missing from the RX QEMU run"; exit 1; }

# Peripheral-breadth proof (M654): a mock SHT3x on the I2C bus returns a
# datasheet response (raw words + CRC-8); the Sht3xSrc driver validates the CRCs
# and converts per the datasheet transfer functions; a UartSink streams each
# reading out a mock UART. The bytes reaching the UART must equal the datasheet
# conversion, so I2C read + CRC + conversion + UART egress all run on the
# Cortex-M ISA. Built by the same `cargo build` above.
SENSOR_ELF="$ROOT/examples/g2g-qemu/target/$TARGET/release/sensor"

echo "== running the I2C-sensor -> UART peripheral proof on QEMU MPS2-AN386 =="
OUT6="$(timeout 120 $QEMU -machine mps2-an386 -cpu cortex-m4 -nographic \
  -semihosting-config enable=on,target=native -kernel "$SENSOR_ELF")"
echo "$OUT6"

grep -q "g2g-sensor: uart-bytes=32 OK" <<<"$OUT6" \
  || { echo "FAIL: expected banner missing from the sensor QEMU run"; exit 1; }

echo "PASS: the g2g-noalloc pipeline executed on an emulated Cortex-M4, both"
echo "      on the bare single-poll executor and under a real Embassy task"
echo "      (64 frames through the camera seam + SPI display element, wire"
echo "      checksum verified on-target, semihosting exit 0), a SysTick"
echo "      interrupt fed a g2g pipeline lossless and in order across the ISR"
echo "      boundary (M651: the interrupt/DMA capture concurrency model), and"
echo "      the supervisor recovered a latched capture fault and escalated a"
echo "      dead peripheral within bounds (M652: runtime fault recovery), and"
echo "      the RX chain reconstructed an ordered PCM stream from a reordered"
echo "      RTP wire through the jitter buffer (M653: receive direction), and"
echo "      an I2C SHT3x sensor read + CRC + conversion streamed out a UART"
echo "      (M654: I2C sensor + UART peripheral breadth)."
