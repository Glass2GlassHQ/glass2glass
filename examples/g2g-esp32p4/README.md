# g2g on ESP32-P4 (RISC-V) — Tier 1 board harness

This is the on-device Tier-1 harness for the ESP32-P4-EYE board: it drives the
heap-free g2g display pipeline (`camera -> transform -> SpiDisplaySink`, the
same graph every `no_std` proof covers) onto the board's ST7789 panel over
esp-hal's SPI + GPIO, on real RISC-V silicon.

## What is already proven (no board needed)

The **g2g side of Tier 1 compiles for the ESP32-P4 ISA today** and is
machine-checked in CI:

- `g2g-core` (no-alloc subset) + `g2g-mcu` + the `noalloc-pipeline` display
  graph build unchanged for `riscv32imafc-unknown-none-elf` (M656); `tools/noalloc-check.sh`
  asserts zero allocator and zero panic symbols on that archive, and
  `tools/footprint.py --isa riscv` budgets its footprint.
- The pipeline runner is **board-agnostic**: `noalloc_pipeline::run_display_with`
  is generic over the `embedded-hal` 1.0 `SpiDevice` / `OutputPin` / `DelayNs`
  seams, so a real HAL's peripherals slot in where the proof stub bus stood.
  A host test (`noalloc-pipeline`'s `display_pipeline_puts_the_expected_bytes_on_the_panel`)
  drives that generic entry and checks the wire is bit-identical to the pinned
  display protocol.

So this crate is only the **board binding**: esp-hal init, the panel pin map,
and the call into `run_display_with`.

## Why this crate is not in CI (yet)

esp-hal's **`esp32p4` chip support is on the esp-hal git `main` branch only** —
no published crate exposes an `esp32p4` feature (checked against esp-hal
1.0.0 / 1.1.0 / 1.1.1; esp-backtrace 0.17 and esp-println 0.15 also lack it).
Because cargo validates a dependency's features eagerly, an `esp-hal`
+ `esp32p4` dependency cannot live in the normal build without breaking it, so
this crate is `exclude`d from the workspace and not compiled by CI. `src/main.rs`
is therefore a **faithful draft against esp-hal 1.x** — the GPIO numbers and a
few driver calls are marked `(VERIFY)` and must be checked against the
ESP32-P4-EYE schematic and the exact esp-hal revision before flashing.

When esp-hal ships a released `esp32p4`, change the git dependency in
`Cargo.toml` to that version and this becomes a normal buildable example.

## Build + flash (once you have the board)

```bash
rustup target add riscv32imafc-unknown-none-elf
cargo install espflash          # one-time
# pin a known-good esp-hal rev in Cargo.toml for reproducibility, then:
cargo run --release              # espflash flash --monitor (see .cargo/config.toml)
```

First light is a 4x4 window whose top-left pixel advances each frame (the proof
pipeline uses a tiny 4x4 test frame so the `StaticLendRing` stays MCU-small).
Full-panel 240x240 streaming (tiled RGB565 line buffers rather than a
full-frame ring) is a Tier-1.5 follow-up.

## What comes after (Tier 2)

Real MIPI-CSI capture + hardware H.264 + WiFi/RTP egress, bridging the ESP-IDF C
drivers through the M650 C-seams (`CFrameGrabber` / `CPacketSender`). See the
`DESIGN_TODO.md` "ESP32-P4X board bring-up" entry for the full plan and the
unknowns to verify first (esp-hal's pure-Rust CSI/H.264 coverage; whether bare
`no_std` can reach the ESP32-C6 network stack without `esp-idf`).
