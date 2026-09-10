# Embedded and heap-free targets

The `alloc`-optional core, the static element model, the MCU peripheral and
codec elements, the RTOS executors, and the proofs that hold all of it to a
measured footprint. Part of the design in [DESIGN.md](DESIGN.md).

## The heap-free (`alloc`-optional) core

For the safety and no-heap MCU market that forbids a heap outright, `alloc` is
an optional cargo feature. `g2g-core` built `--no-default-features` links no
allocator and carries only the data-plane subset: `Frame`, the `Caps` enum with
`intersect` / `fixate` including `Caps::Tensor` (`TensorShape` is a fixed-rank
inline array of at most `MAX_TENSOR_RANK` dims, so the ML caps kind is heap-free
and `Copy` like the media kinds), `MemoryDomain::System` lending a
`StaticLendRing` slot zero-copy, and the pure clock, time, error, and state
modules. The dynamic layer, the negotiation solver, `parse_launch`, the `dyn`
element traits, and the tooling sit behind `alloc`, which `std` / `runtime` /
`metadata` imply, so host consumers are unaffected.

A heap-free pipeline is a compile-time-static graph of concrete elements using
the static element model (`g2g_core::staticelem`: `StaticSource` /
`StaticTransform` / `StaticSink` with `async fn` in trait, so each stage's future
is unboxed, plus const-arity runners and a `Chain` combinator). It is the generic
twin of the object-safe `AsyncElement`, which boxes a future per frame.
Application code on this surface needs no `unsafe`: `StaticLendRing::new` is
`const`, so the ring lives in a `static` and `GrabberSrc`'s safe constructor
makes the zero-copy lend sound by construction, and the single-poll executor is
the safe `drive_ready`. A test builds a full pipeline under
`#![forbid(unsafe_code)]`.

## Heap-free proofs and footprint

The guarantee is machine-checked. `examples/g2g-noalloc` links a full
source -> transform -> sink pipeline for `thumbv7em-none-eabihf` with no
`#[global_allocator]` and no `alloc` crate dependency, so any heap use fails the
build, `tools/noalloc-check.sh` asserts the archive references zero allocator
symbols, and a counting-allocator test (`m616_no_steady_state_alloc`) confirms the
runner allocates nothing over 100k frames at runtime.

The same archive is panic-free: every reachable path avoids unwrap, slice-index,
and overflow panics, and the single-poll executor lets the optimizer discharge
the compiler's resumed-after-completion guard, so the archive contains zero
`core::panicking` symbols and the mandatory `#[panic_handler]` is provably dead
code. The check script asserts that, then runs the pipeline on the host through a
C harness, so the symbol proofs describe code that executes.

Footprint is reported and budget-enforced at build time
(`tools/footprint-report.sh` + `footprint.py`), computed from the disassembly
call graph rather than estimated. The gc-sectioned ELF of the
source-transform-sink pipeline measures about 4 KB ROM, 0 bytes static RAM, and
about 1.25 KB worst-case stack, dominated by the entry frame holding the capture
ring plus the monomorphized pipeline state machine. The same pipeline, shared as
the `noalloc-pipeline` rlib, executes on the Cortex-M ISA: `examples/g2g-qemu`
boots it on QEMU's MPS2-AN386 Cortex-M4 and verifies the checksum on-target
(`tools/qemu-check.sh`, in CI), with the conformance row marked as emulated.

## MCU peripheral elements

`g2g-mcu` (`no_std`, no `alloc`) holds heap-free `staticelem` elements written
against portable trait seams rather than chip registers, so the driver logic is
host-tested against the datasheet with mock peripherals and a board port is only
the vendor HAL's trait impls.

- `SpiDisplaySink` (ST7789 / ILI9341 over `embedded-hal` `SpiDevice` plus a D/C
  pin): DCS command sequences, window addressing, streaming RGBA to RGB565
  through a fixed stack chunk.
- The `FrameGrabber` camera seam and `GrabberSrc`, the DCMI/CSI shape: capture
  into a lent `StaticLendRing` slot, published downstream zero-copy with
  sequence and PTS. Safe over a `'static` ring, `unsafe` over a borrowed one.
- The `PcmWriter` audio seam and `PcmSink`, the I2S/SAI shape: S16LE interleaved
  decode through a fixed chunk.
- `Sht3xSrc` reads a Sensirion SHT3x temperature and humidity sensor over the
  `embedded-hal` `I2c` seam: the datasheet single-shot command, both CRC-8 check
  bytes validated (polynomial `0x31`, the datasheet `0xBEEF -> 0x92` vector as a
  test), and the datasheet transfer functions `i64`-widened so the fixed-point
  multiply cannot overflow. A CRC mismatch is a bus-integrity fault, not data.
- `g2g-mcu::uart` adds local `SerialTx` / `SerialRx` seams, embedded-hal 1.0
  keeping blocking serial in `embedded-io`, plus `UartSink` (frame payload as a
  byte-stream egress) and `UartSrc` (fixed-size frame ingress).

The camera and display elements are the proof pipeline's source and sink, so
every guarantee above covers real peripheral elements: 4286 B ROM, 0 B static
RAM, 1508 B worst-case stack. Its transform link negotiates `Caps::Tensor` and
validates each frame against it, so the tensor caps kind is covered by the same
proofs. A `Sht3xSrc -> UartSink` pipeline streaming a datasheet reading out a
mock UART is proven on the Cortex-M ISA (`examples/g2g-qemu`'s `sensor` bin,
`g2g-sensor: uart-bytes=32 OK`).

## MCU codecs and the reference audio chain

The MCU-fit codecs are G.711 (mu-law / A-law), pure-integer `const fn`
conversions validated bit-exact against ffmpeg over the entire domain (every
encoder input, every decoder code), and IMA ADPCM (the WAV / DVI4 block layout,
validated bit-exact against ffmpeg in encode, decode, and cross-decode). Both
carry persisted `Oracle` evidence and are wrapped as `G711Enc` / `G711Dec` /
`AdpcmEnc` / `AdpcmDec`, payload-producing static transforms that lend output
frames from a `StaticLendRing` through one shared helper.

The reference audio chain resamples with a fixed-point polyphase resampler over
the {8, 16, 48} kHz set: generated Q14 Blackman-sinc tables with exactly-unity
phase sums so DC gain is exact, and streaming state that makes chunking
byte-invisible. Validated analytically at about 86 dB tone SNR and about 120 dB
alias rejection.

The mix stage uses the const-arity multi-input surface, the `StaticFanIn2` trait
plus the `run_sources_fanin_sink` runner (lockstep pull, EOS when either source
ends, monomorphized like the linear runners). `g2g-mcu`'s `Mixer` implements it
with saturating Q15 gains per input via `const fn mix_q15`, an i64 accumulator
because two full-scale-negative products overflow i32, unequal payloads rejected
rather than truncated, and input `a` as the timing master.

Egress uses the RFC 3550 fixed header defined once for the whole workspace
(`g2g_core::rtp::RtpHeader`, a heap-free `const fn` shared across `rtppay` and
the ST 2110 cores). `g2g-mcu`'s `RtpSink` emits one RTP packet per frame through
the `PacketSender` seam, a header plus payload scatter-gather datagram in the
lwIP / Zephyr-sendmsg shape, mapping PTS to timestamp via `MediaClock` and
rejecting over-MTU payloads rather than fragmenting. It is validated against
ffmpeg as the receiving RTP peer byte-for-byte in the CI conformance job.

## The flagship graph and `g2g-mcugen`

The flagship demo graph is `capture -> convert -> resample -> mix -> encode ->
RTP` as one static pipeline (`noalloc-pipeline::audio`). `SourceChain` /
`SinkChain` fuse transforms into the fan-in runner's source and sink slots, the
static bin analog, and `PcmConvert` narrows left-justified 24-in-32 I2S capture
slots to S16. It is host-validated against an independent float reference,
checksum-pinned, and re-verified bit-exactly on QEMU Cortex-M4 and Cortex-M3
under all four executors (bare, Embassy, FreeRTOS, Zephyr), with a footprint
budget row of 10572 B ROM, 0 B static RAM, 6504 B worst-case stack.

The host graph compiler `g2g-mcugen` emits the same graph from a declarative
YAML/JSON document, monomorphized, with every ring sized from the graph's frame
geometry plus a ring-memory budget report. The generated flagship graph
reproduces the hand-written reference's RTP wire byte-for-byte
(`examples/mcugen-graphs`, checked in CI against `AUDIO_EXPECTED_CHECKSUM`).

The compiler is not audio-specific. Frame geometry is a sum of audio (rate,
width, channels) and raster (pixels, bpp), the sink seam varies per sink kind (an
RTP `PacketSender`, or an SPI bus plus D/C pin plus delay bound on
`embedded-hal`), and a second catalog compiles a `camera -> SPI display` graph
(`g2g-mcugen/examples/display.yaml`) whose generated pipeline reproduces the
hand-written display reference's panel wire byte-for-byte (`EXPECTED_CHECKSUM`,
the reference's byte no-op transform making camera-to-display equivalent). A
timing and jitter row is measured under QEMU icount, deterministic virtual time
where two boots must report identically: steady-state worst case about 764 us of
a 10 ms frame with about 360 ns jitter, budget-enforced in CI like the memory
numbers.

## Hardware codec peripherals

`JpegDecoder` is the hardware-decoder seam, the STM32H7-shaped whole-bitstream
contract, and `HwJpegDec` validates JFIF framing before the peripheral,
cross-checks the emitted byte count against the header-derived MCU tiling with
checked math, and surfaces a self-contradicting peripheral as a fault.
`g2g-mcu::hwh264` is the encode twin: the `H264Encoder` contract (one raw I420
frame in, one Annex-B access unit out, byte count plus keyframe flag reported)
and `HwH264Enc`, which validates 4:2:0 geometry with checked sizing,
cross-checks the reported byte count, and surfaces a faulting peripheral. Both
are datasheet-tested on mocks. The `CH264Encoder` C bridge lets the vendor's
hardware encoder driver be the peripheral, alongside `CFrameGrabber` /
`CPacketSender`, host-tested through a mock and a real `extern "C"` callback,
byte-identical.

A DCMI/DVP camera emits packed YUYV 4:2:2 while `HwH264Enc` wants planar I420,
so `g2g-mcu::videoconvert::YuyvToI420` is the heap-free `StaticTransform` for
that conversion, the MCU twin of the `alloc`-based host `VideoConvert`,
converting in place through a ring slot with checked geometry. An integration
test drives `camera -> YuyvToI420 -> HwH264Enc -> RtpSink` as one static pipeline
and traces a camera-stamped byte to the RTP payload.

## RTOS executors and C integration

The same pipeline runs under a real Embassy task (`examples/g2g-embassy`, the
future awaited directly), under a FreeRTOS task (`examples/g2g-freertos`, the
C-ABI staticlib linked into a static-allocation-only FreeRTOS image) on the
emulated Cortex-M, and as a Zephyr application (`examples/g2g-zephyr`, the same
staticlib built for the `qemu_cortex_m3` board's soft-float thumbv7m, booted on
QEMU's lm3s6965evb). Zephyr consumes it through a reusable module
(`examples/g2g-zephyr-module`: `module.yml` plus CMake that import the archive
and expose `include/g2g.h`, so the app does `#include <g2g.h>` and links nothing
g2g itself), which is the packaging a Zephyr shop lists in its west manifest. The
static element model needs no adaptation layer for an RTOS executor, from either
the Rust or the C side.

`g2g-mcu::cffi` carries peripheral callbacks the other way: `CFrameGrabber` /
`CPacketSender` implement the `FrameGrabber` / `PacketSender` seams over C
function pointers (`CaptureFn` / `SendFn` plus an opaque `ctx`), so a board
registers its existing C capture routine and C network stack and g2g calls them
back. `g2g_core::step_source_sink`, with the `Step` enum, is the frame-at-a-time
runner that hands control back after one frame, so a C superloop owns the loop.
Compose a tail with `SinkChain` to step any linear graph.

`examples/g2g-cffi` proves it: a `no_std` staticlib exposing
`g2g_audio_egress_init` / `_step` / `_reset` (a `capture -> G.711 -> RTP`
pipeline over the C seams) plus `include/g2g_cffi.h`, linked for `thumbv7em` with
zero allocator and zero data-panic symbols. The one-frame-step future leaves only
a benign, runtime-unreachable async re-poll guard that the run-to-EOS runners
discharge, and `tools/cffi-check.sh` permits that alone. A real C caller
(`harness.c`) drives it and its wire matches the pipeline's Rust reference
byte-for-byte.

## Interrupt and DMA capture

A DMA-completion or timer ISR produces frames in interrupt context while the
pipeline consumes them in the main or task context, so the two hand frames across
the ISR boundary through `g2g_core::SpscFrameRing<N, BYTES>`, a fixed-capacity
single-producer, single-consumer FIFO. The producer's `produce`, called from the
ISR, fills the next free slot and publishes it. The consumer's `borrow` /
`release` drains it in capture order, zero-copy, the frame borrowing the ring slot
and releasing it after the frame drops. It uses only atomic load and store, no
compare-and-swap, so it builds on Cortex-M targets without atomic CAS
(`thumbv6m`). Back-pressure is explicit and non-blocking because an interrupt
cannot wait: a full ring drops the frame and bumps an overrun counter the
consumer reads.

`g2g_core::SpscCaptureSrc` is the consumer-side `StaticSource`, the concurrent
twin of the synchronous `GrabberSrc`. It drains the ring and, while empty, calls
a caller-supplied idle hook (`cortex_m::asm::wfi` on hardware, so the consumer
sleeps until the capture interrupt) and retries. Proven on the Cortex-M ISA
(`examples/g2g-qemu`'s `isr_capture` bin, in `tools/qemu-check.sh`): a SysTick
interrupt is the producer, the main-context pipeline drains it through
`SpscCaptureSrc -> G.711 -> checksum` sleeping on `wfi`, and the wire equals
synchronous delivery frame-for-frame (`captured=64 overruns=0 OK`). Host thread
tests cover lossless-when-paced and drop-and-count-under-back-pressure.

## Fault recovery and supervision

The static runners propagate a returned fault straight out, so the first glitch
ends the pipeline. `g2g_core::supervise`, in the no-alloc subset, supplies the
opposite default: bounded, deterministic recovery. A `FaultPolicy` maps each
fault to a `Recovery` action, `Retry` (re-drive a transient fault), `Skip` (drop
the frame and keep cadence, a degraded mode), `Reset` (re-initialize the stages),
or `Escalate`. The supplied `RetryThenReset` and `SkipBounded` cover
recover-in-place and degrade-and-continue, and both escalate a persistent fault
in finite steps.

`Recover` is the per-stage re-init seam, default no-op. `GrabberSrc` re-arms via
`FrameGrabber::reset`, `RtpSink` re-opens via `PacketSender::reset`, and
`SpscCaptureSrc` flushes stale buffered frames so real-time capture resumes from
live data, so a supervised pipeline declares each stage's recovery behavior.
`SupervisorReport` accounts the faults, retries, resets, skips, and escalation. A
`Watchdog` is petted only on real forward progress, so a wedged or escalated
pipeline stops petting and a hardware watchdog resets the chip
(`g2g-mcu::watchdog` supplies the `WatchdogTimer` HAL seam, embedded-hal 1.0
having dropped its watchdog trait, plus the `SupervisorWatchdog` adapter).
`step_supervised`, where a C superloop or RTOS task owns the loop, and
`run_supervised` drive it, bounded by a hard `MAX_ATTEMPTS` cap so even a buggy
never-escalating policy cannot hang.

Proven on the Cortex-M ISA (`examples/g2g-qemu`'s `supervised` bin): a
`capture -> G.711 -> checksum` pipeline recovers a mid-stream latched capture
fault (retry, then reset via the `FrameGrabber::reset` seam, then continue, all
64 frames delivered, wire checksum equal to a clean reference, watchdog fed once
per frame) and then escalates a dead peripheral within its bounded ladder without
hanging, watchdog never fed (`delivered=64 resets=1 wd=64 escalated=4 OK`).

## RTP receive on MCU

`g2g_core::rtp::RtpHeader::parse` is the wire-tolerant inverse of `to_bytes`,
covering the CSRC list, extension header, and padding, every offset checked and
bounds-guarded so a malformed datagram returns `None`. The std H.264
depayloader shares it. `g2g-mcu::rtprecv` adds the `PacketReceiver` ingress seam
and `RtpSrc`, the heap-free `StaticSource` that receives a datagram, parses it,
and lends the payload downstream with `Frame::sequence` set to the RTP sequence
number.

`g2g-mcu::jitter::JitterBuffer<N, BYTES>` is the reorder element: a fixed
`N`-slot reorder window that absorbs arrival jitter, emits the next-in-sequence
packet after a prime depth, and handles reorder, duplicate, late, and loss
explicitly and countably. A packet more than `depth` ahead marks the missing head
lost and advances, so one loss never stalls the stream. Its output frame borrows
the buffer's own slot zero-copy under the single-frame-in-flight discipline.

These compose into the RX flagship `RtpSrc -> JitterBuffer -> G.711 decode`,
validated on mocks (a reordered, duplicated, lossy wire reconstructs the ordered
decoded PCM byte-for-byte) and on the Cortex-M ISA (`examples/g2g-qemu`'s `rx`
bin, proved by an order-sensitive rolling hash equal to an independent in-order
decode, `played=14 reordered=3 lost=0 OK`). Both RX elements are
`Recover`-capable for the supervisor: the source re-opens its socket, the buffer
flushes and re-syncs.

## Safety process artifacts

`docs/safety/REQUIREMENTS.md` is a requirements traceability matrix, 15
requirements across memory, timing, faults, concurrency, input validation, and
data integrity, each linked to the proof script, test, or CI job that verifies
it. `tools/traceability-check.sh` fails if any cited evidence is missing or if a
cited proof script is not wired into CI, so the matrix cannot drift, and it runs
in CI alongside the proofs it indexes. `docs/safety/SAFETY_MANUAL.md` documents
the conditions of use, per-property assumptions, the localized `unsafe`
inventory, and integrator responsibilities, and `tools/qualification-kit.sh` runs
the whole proof set and emits a consolidated requirement-to-evidence-to-result
report. This is a down-payment on a product safety case, emulated rather than
silicon and pre-1.0, not a substitute for one.

## RISC-V and board ports

None of this is ARM-specific. The static element model is ISA-agnostic pure Rust,
only the QEMU harness bins carrying ARM startup, so the `g2g-core` no-alloc
subset and `g2g-mcu` build unchanged for `riscv32imafc-unknown-none-elf`, the
ESP32-P4 class. `tools/noalloc-check.sh` asserts the zero-allocator and
zero-panic guarantees on both `thumbv7em` and RISC-V archives, and
`tools/footprint.py` with an `--isa riscv` stack-frame model budgets the RISC-V
video pipeline at 3718 B ROM, 0 B static RAM, 1328 B stack.

The RISC-V model also budgets the flagship audio graph. rustc encodes that frame
as a constant too large for `addi`'s 12-bit immediate, so it materializes the
size into a register and does `sub sp, sp, <reg>`. The stack model resolves that
register to its compile-time constant, following the `lui` / `addi` / `slli`
materialization chain and failing rather than under-reporting if it is ever not a
known constant, so the RISC-V audio graph is budgeted exactly like the others at
10852 B ROM, 0 B static RAM, 6432 B stack, within the ARM audio budgets.

The RISC-V board path adds two capabilities. `SpiDisplaySink::with_stripe`
streams a panel too large to ring-buffer whole (240x240 RGBA is 230 KB) in
horizontal bands: each frame is one `width x rows` band written to the next
vertical sub-window, so the pipeline ring holds a single 15 KB band.
`noalloc_pipeline::run_display_banded_with` is the board-agnostic full-panel
runner, and the whole-frame path is byte-identical, so the same proofs cover it.

On the ARM side, `examples/g2g-stm32h743` targets a NUCLEO-H743ZI2. It runs the
flagship audio graph and egresses RTP over the H743's on-chip Ethernet through a
pure-Rust `embassy-net`/smoltcp stack, the whole g2g-to-network bridge being one
`EmbassyNetSender: PacketSender` that maps the RTP egress seam onto an
embassy-net `UdpSocket`, with no C in the network path unlike the P4's WiFi. It
compiles for `thumbv7em` and stays outside CI because of embassy's build weight.

