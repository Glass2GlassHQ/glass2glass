# g2g on STM32H743 (Cortex-M7) — on-device harness

Runs the heap-free **flagship audio graph** (`capture → convert → resample →
mix → encode → RTP`) on a NUCLEO-H743ZI2 and egresses the RTP over the H743's
on-chip **Ethernet**, through a **pure-Rust** smoltcp / embassy-net stack — no C
in the network path.

## Why this board / this path

- **Cortex-M7 = `thumbv7em-none-eabihf`**, the exact ISA the no-alloc symbol
  proofs, footprint budgets, and QEMU runs already target — so a run on metal is
  the deferred on-device `Hardware` conformance row, not a new port.
- **Wired Ethernet + pure-Rust stack.** The H743 has no radio; its network is
  on-chip Ethernet, driven here by `embassy-stm32`'s `eth` + `embassy-net`
  (smoltcp). That makes RTP egress **pure Rust end to end** — a stronger
  portability story than the ESP32-P4 (whose WiFi means bridging ESP-IDF C).
- It's also the home of the M640 hardware-JPEG codec and the M655 functional
  safety narrative (see the repo's `DESIGN_TODO.md` on-device rows).

## The one piece that matters

The whole g2g-to-network bridge is `EmbassyNetSender` in `src/main.rs`: our
`PacketSender` seam (`send(header, payload)`) maps one-to-one onto an
embassy-net `UdpSocket::send_to`. `run_audio_with(sender)` then drives the full
static pipeline and calls it once per 10 ms frame. That's the entire integration
— the rest is standard embassy board bring-up.

## Status: compiles (verified), runtime config pending the board

This crate **compiles and links** for `thumbv7em-none-eabihf` with the pinned
versions in `Cargo.toml` (`embassy-stm32` 0.2 + `embassy-net` 0.7); it is
`exclude`d from the workspace and not built in CI only for embassy's build
weight, like other heavy examples. Build it yourself with:

```bash
cargo build --release   # in this directory; targets thumbv7em (see .cargo/config.toml)
```

What still needs the board is **runtime** config, not compilation — every such
spot is marked `VERIFY` in `src/main.rs`:

1. The RCC/clock `Config`: `Default` compiles but will not clock the Ethernet
   MAC; copy the exact `Config` from embassy's `stm32h7` ethernet example.
2. Confirm the RMII pins against the **NUCLEO-H743ZI2** schematic.
3. Set the RTP destination IP:port to your receiver.

## Build + flash

```bash
rustup target add thumbv7em-none-eabihf
cargo install probe-rs-tools      # one-time
cargo run --release               # probe-rs run --chip STM32H743ZITx (see .cargo/config.toml)
```

Then, on the receiver, e.g.:

```bash
ffplay -protocol_whitelist file,udp,rtp -i stream.sdp   # PCMU/8000, the graph's RTP
```

## Relation to the other harness

`examples/g2g-esp32p4` is the RISC-V / wireless-media counterpart (MIPI-CSI
camera + HW H.264 + C6 WiFi, once esp-hal ships P4). This H743 harness is the
ARM / wired / cert-track first board — and its network path compiles with
published crates today, whereas the P4's does not.
