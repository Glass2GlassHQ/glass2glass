# Technical Specification: `glass2glass` (g2g)
**A Next-Generation, Hardware-First, Sans-IO, Asynchronous Multimedia Framework in Rust**

---

## 1. Executive Summary & Design Philosophy
`glass2glass` (`g2g`) is an open-source, ultra-low-latency multimedia graph framework written in 100% pure Rust. It is built around one idea: **a pure-Rust core so the same typed pipeline runs, unchanged, across the whole hardware spectrum: MCU, RTOS, CPU, GPU, and WASM.** A `no_std + alloc`, sans-IO core means the graph, the element traits, the caps negotiation, and the runner are identical on a bare-metal microcontroller, a real-time (Embassy) target, a CPU server, a GPU-resident zero-copy pipeline, and the web browser; only the deployment shell (which executor, which hardware elements) changes.

The project prioritizes minimizing **glass-to-glass latency** — the exact time elapsed between physical photon/audio capture and hardware presentation.

### The Four Pillars of `g2g`:
1. **Asynchronous Execution:** Every element is a cooperative async task (`Future`). No internal OS thread management; the framework is runtime-agnostic.
2. **Hardware-First & Zero-Copy:** Data remains in VRAM or unified memory domains via hardware handles (`DMABUF`, Vulkan Textures). CPU memory copies are treated as system faults.
3. **Modular Predictability (`no_std + alloc` + Sans-IO):** A `no_std + alloc` core allowing the exact same pipelines to execute on bare-metal microcontrollers, heavy multi-threaded servers, or WebAssembly (Wasm) targets. Network and protocol parsers use a pure **Sans-IO** design pattern, stripping I/O operations entirely out of the logic layer.
4. **First-Class Machine Learning Integration:** Tensor allocation, reshaping, and pipeline batching are built directly into the graph orchestration layer, executing in-flight on GPU memory.

### Architecture at a Glance

A `g2g` pipeline is a graph of typed **elements** joined by bounded async
channels. A source produces packets, transforms rewrite them, and sinks consume
them:

```
  Source ─────▶ Transform ─────▶ … ─────▶ Sink
 (RtspSrc,     (H264Parse,               (WaylandSink,
  V4l2Src,      decoder,                  WgpuSink,
  Mp4Src)       ML preprocess)            UdpSink)

  on each link:  CapsChanged · DataFrame(Frame) · Segment · Flush · Eos
```

Before any frame flows, the runner runs **one caps-negotiation pass** over the
whole graph (§4.13 / [DESIGN-caps.md](DESIGN-caps.md)): every link is assigned a
concrete `Caps`, every element allocates its buffers, and the memory domain each
link carries (System / DMABUF / CUDA / Vulkan / WebGPU texture) is settled so a
zero-copy link stays zero-copy. Each element then runs as its own cooperative
async task, paced by channel backpressure rather than an internal thread.

The handful of types you meet everywhere (all in `g2g-core`):

| Type | Role |
| :--- | :--- |
| `Frame` | one media buffer: a `MemoryDomain` payload + `FrameTiming` + a sequence number + optional metadata. Caps live on the *link*, not the frame (§3.1). |
| `PipelinePacket` | what actually crosses a link: `CapsChanged` / `DataFrame` / `Segment` / `Flush` / `Eos`, plus the arm-local `Tick` (§3.1). |
| `Caps` | the typed capability algebra (`RawVideo` / `CompressedVideo` / `Audio` / `Tensor` / `Text` / `ByteStream`), negotiated per link (§4.1). |
| `AsyncElement` / `SourceLoop` | the two element traits (transform-or-sink vs source). Pads are implicit in the trait shape, not a runtime object (§4.3, §4.7). |
| `MemoryDomain` | where a frame's bytes live: System, DMABUF, CUDA, Vulkan / WebGPU texture, … the basis for zero-copy (§3.2). |
| the runner (`run_graph`) | drives negotiation and then one async task per node over the channels (§4.13.3). |

### Reading Guide

The framework is built along a few interlocking **tracks**; the detail sections
map onto them. Skim this table, then read the tracks you care about:

| Track | Where | What |
| :--- | :--- | :--- |
| Data & memory | §3 | `Frame`, memory domains, zero-allocation buffer pools. |
| Orchestration core | §4.1–§4.10 | caps lifecycle, element traits, clock/timing, backpressure, dynamic reconfiguration. |
| Caps negotiation | **[DESIGN-caps.md](DESIGN-caps.md)** (§4.13) | the CSP solver, allocation cascade, auto-plug / `decodebin` / `playbin`, bins. |
| Receive & decode | §4.11, §4.12a/b, §4.19 | RTSP / RTP / WebRTC / capture sources, hardware decoders, Vulkan Video. |
| Display & egress | §4.11.5, §4.12, §4.19 | GPU-resident presentation sinks, RTP / WHIP egress. |
| Lifecycle & control | §4.14–§4.16 | state machine, seek, bus / observability, the `gst-launch` DSL. |
| Containers & text | §4.17, §4.18 | mux / demux, HLS / DASH, subtitles & closed captions. |
| ML | §5 | inline GPU tensor preprocess + inference, batching, detection metadata. |
| Deployment | §6, §7 | server / embedded / browser profiles, the GStreamer bridge. |

Open work lives in [DESIGN_TODO.md](DESIGN_TODO.md); shipped milestones are logged
in [CHANGELOG.md](CHANGELOG.md).

---

## 2. Core Workspace Structure & Licensing
The project is structured as a Cargo Workspace to enforce clean boundaries between interfaces, standard elements, ML backends, and platform bindings.

| Crate Name | Purpose | Target Profile | Licensing |
| :--- | :--- | :--- | :--- |
| `g2g-core` | Core traits, `Frame` definitions, buffer pool allocators, clock model. | `no_std + alloc` | MPL-2.0 |
| `g2g-plugin` | SDK for dynamically loadable plugins (the `declare_plugin!` macro + ABI tag, §4.16). | `no_std + alloc` | MPL-2.0 |
| `g2g-plugins` | Standard collection of source/sink/transform elements (`rtsp`, `wgpu`, `v4l2`). | `no_std + alloc` / `std` mixed | MPL-2.0 |
| `g2g-ml` | ML inference elements built on `burn` (Wasm/embedded) and `ort` (server), plus the multi-stream tensor batcher. | `std` | MPL-2.0 |
| `g2g-bridge` | C-FFI dynamic library to embed `g2g` sub-graphs inside GStreamer pipelines. | `std` (`cdylib`) | MPL-2.0 |
| `g2g-python` | Hosts gst-python-ml elements as first-class `g2g` elements (embedded CPython via pyo3). | `std` | MPL-2.0 |
| `g2g-capi` | C ABI (cdylib/staticlib + `g2g.h`) to drive pipelines from any language: `parse_launch` + run + bus + appsrc/appsink. | `std` (`cdylib`) | MPL-2.0 |
| `g2g-pyapi` | Python (pyo3) bindings to drive pipelines: `parse_launch` + run + bus + appsrc/appsink (the inverse of `g2g-python`). | `std` | MPL-2.0 |

The `no_std + alloc` baseline is deliberate: it admits cooperative async executors (which need a heap for futures) and `Arc` reference counting, while still excluding the OS-dependent surface of `std`. Targets requiring strict no-heap allocation use the static `BufferPool` (§3.3) and avoid the `dyn`-safe element wrappers (§4.3).

**Heap-free (`alloc`-optional) core.** For the safety / no-heap MCU market that forbids a heap outright, `alloc` itself is an optional cargo feature: `g2g-core` built `--no-default-features` links no allocator and carries only the data-plane subset (`Frame`, the `Caps` enum + `intersect` / `fixate` including `Caps::Tensor` (M636: `TensorShape` is a fixed-rank inline array, at most `MAX_TENSOR_RANK` dims, so the ML caps kind is heap-free and `Copy` like the media kinds), `MemoryDomain::System` lending a `StaticLendRing` slot zero-copy, and the pure clock / time / error / state modules). The dynamic layer, negotiation solver, `parse_launch`, the `dyn` element traits, and the tooling live behind `alloc`; `std` / `runtime` / `metadata` imply it, so host consumers are unaffected. In the heap-free build a pipeline is a compile-time-static graph of concrete elements using the static element model (`g2g_core::staticelem`: `StaticSource` / `StaticTransform` / `StaticSink` with `async fn` in trait, so each stage's future is unboxed, plus const-arity runners and a `Chain` combinator), the generic twin of the object-safe `AsyncElement` (which boxes a future per frame). The guarantee is machine-checked, not asserted: `examples/g2g-noalloc` links a full source -> transform -> sink pipeline for `thumbv7em-none-eabihf` with no `#[global_allocator]` and no `alloc` crate dependency (so any heap use fails the build), and `tools/noalloc-check.sh` asserts the archive references zero allocator symbols; a counting-allocator test (`m624`) confirms the runner allocates nothing over 100k frames at runtime. The same archive is also panic-free: every reachable path avoids unwrap / slice-index / overflow panics, and the single-poll executor lets the optimizer discharge the compiler's resumed-after-completion guard, so the archive contains zero `core::panicking` symbols (the mandatory `#[panic_handler]` is provably dead code); the check script asserts that too, then runs the pipeline on the host through a C harness so the symbol proofs describe code that demonstrably executes. Finally the footprint is reported and budget-enforced at build time (`tools/footprint-report.sh` + `footprint.py`): the pipeline linked as a gc-sectioned ELF measures ~4 KB ROM, 0 bytes static RAM, and a worst-case stack (~1.25 KB, dominated by the entry frame holding the capture ring + the monomorphized pipeline state machine) computed from the disassembly call graph, so an MCU integrator gets hard RAM / stack / ROM numbers, not estimates. The same pipeline (shared as the `noalloc-pipeline` rlib) also *executes* on the Cortex-M ISA: `examples/g2g-qemu` boots it on QEMU's MPS2-AN386 Cortex-M4 and verifies the checksum on-target (`tools/qemu-check.sh`, in CI), emulation deliberately distinct from a future on-device `Hardware` conformance row. MCU peripheral elements live in `g2g-mcu` (`no_std`, no `alloc`): heap-free `staticelem` elements written against portable trait seams rather than chip registers, so the driver logic is host-tested against the datasheet with mock peripherals, and a board port is only the vendor HAL's trait impls. Landed: `SpiDisplaySink` (ST7789 / ILI9341 over `embedded-hal` `SpiDevice` + D/C pin: DCS command sequences, window addressing, streaming RGBA -> RGB565 through a fixed stack chunk), the `FrameGrabber` camera seam + `GrabberSrc` (the DCMI/CSI shape: capture into a lent `StaticLendRing` slot, published downstream zero-copy with sequence/PTS; safe over a `'static` ring, `unsafe` over a borrowed one), and the `PcmWriter` audio seam + `PcmSink` (I2S/SAI shape, S16LE interleaved decode through a fixed chunk). MCU-fit codecs live there too: the G.711 (mu-law / A-law) fixed-point codec (M638), pure-integer `const fn` conversions validated bit-exact against ffmpeg over the entire domain (every encoder input, every decoder code), and the IMA ADPCM codec (M639, the WAV / DVI4 block layout, validated bit-exact against ffmpeg in encode, decode, and cross-decode), both with persisted `Oracle` evidence, wrapped as `G711Enc` / `G711Dec` / `AdpcmEnc` / `AdpcmDec`, the first payload-producing static transforms (they lend output frames from a `StaticLendRing`, the capture source's zero-copy model, through one shared helper). The reference audio chain's resampler is in too (M641): a fixed-point polyphase resampler over the {8, 16, 48} kHz set (generated Q14 Blackman-sinc tables with exactly-unity phase sums, so DC gain is exact; streaming state makes chunking byte-invisible; validated analytically, ~86 dB tone SNR / ~120 dB alias rejection), and so is its mix stage (M642): `staticelem` gained its first const-arity multi-input surface, the `StaticFanIn2` trait plus the `run_sources_fanin_sink` runner (lockstep pull, EOS when either source ends, monomorphized like the linear runners), and `g2g-mcu`'s `Mixer` implements it (saturating Q15 gains per input via the `const fn mix_q15`, i64 accumulator because two full-scale-negative products overflow i32, unequal payloads rejected rather than truncated, input `a` the timing master). The chain's egress is in too (M643): the RFC 3550 fixed header is defined once for the whole workspace (`g2g_core::rtp::RtpHeader`, a heap-free `const fn`, replacing five hand-rolled writers across `rtppay` and the ST 2110 cores), and `g2g-mcu`'s `RtpSink` emits one RTP packet per frame through the `PacketSender` seam (a header + payload scatter-gather datagram, the lwIP / Zephyr-sendmsg shape; PTS -> timestamp via `MediaClock`, over-MTU payloads rejected rather than fragmented), validated against ffmpeg as the receiving RTP peer byte-for-byte in the CI conformance job. These compose into the flagship demo graph (M644): `capture -> convert -> resample -> mix -> encode -> RTP` as one static pipeline (`noalloc-pipeline::audio`; `SourceChain` / `SinkChain` fuse transforms into the fan-in runner's source and sink slots, the static bin analog, and `PcmConvert` narrows left-justified 24-in-32 I2S capture slots to S16), host-validated against an independent float reference and checksum-pinned, then re-verified bit-exactly on QEMU Cortex-M4 and Cortex-M3 under all four executors (bare, Embassy, FreeRTOS, Zephyr) with its own footprint budget row (10572 B ROM, 0 B static RAM, 6504 B worst-case stack). The same graph is also emitted by the host graph compiler (M646, `g2g-mcugen`): a declarative YAML/JSON document compiles to the monomorphized static pipeline with every ring sized from the graph's frame geometry (plus a ring-memory budget report), and the generated flagship graph reproduces the hand-written reference's RTP wire byte-for-byte (`examples/mcugen-graphs`, checked in CI against `AUDIO_EXPECTED_CHECKSUM`), which is the develop-on-host-compile-to-MCU story made concrete. The compiler is not audio-specific (M648): frame geometry is a sum of audio (rate / width / channels) and raster (pixels / bpp), the sink seam varies per sink kind (an RTP `PacketSender`, or an SPI bus + D/C pin + delay bound on `embedded-hal`), and a second catalog compiles a `camera -> SPI display` graph (`g2g-mcugen/examples/display.yaml`) whose generated pipeline reproduces the hand-written display reference's panel wire byte-for-byte (`EXPECTED_CHECKSUM`, the reference's byte no-op transform makes camera->display equivalent), so "one declarative graph compiles to a bounded static build" is proven for video / display as well as audio and a timing / jitter row measured under QEMU icount (M645: deterministic virtual time, two boots must report identically; steady-state worst case ~764 us of a 10 ms frame with ~360 ns jitter, budget-enforced in CI like the memory numbers): the deterministic-audio wedge's one-graph-everywhere claim, machine-checked in space and time. The hardware-codec-peripheral seam is in too (M640): `JpegDecoder`, the STM32H7-shaped whole-bitstream contract, and `HwJpegDec`, which validates JFIF framing before the peripheral, cross-checks the emitted byte count against the header-derived MCU tiling with checked math, and surfaces a self-contradicting peripheral as a fault; datasheet-tested on mocks, with the on-device `Hardware` row deferred to real silicon. The camera and display elements are the proof pipeline's source and sink, so every guarantee above covers real peripheral elements (whole pipeline: 4286 B ROM, 0 B static RAM, 1508 B worst-case stack; the transform link negotiates `Caps::Tensor` and validates each frame against it, so the tensor caps kind is covered by the same proofs, M636), and the same pipeline runs under a real Embassy task (`examples/g2g-embassy`, the future awaited directly) and under a FreeRTOS task (`examples/g2g-freertos`, the C-ABI staticlib linked into a static-allocation-only FreeRTOS image) on the emulated Cortex-M, and as a Zephyr application (`examples/g2g-zephyr`: the same staticlib, built for the `qemu_cortex_m3` board's soft-float thumbv7m, and since M647 consumed through a reusable Zephyr *module* (`examples/g2g-zephyr-module`: `module.yml` + CMake that import the archive and expose `include/g2g.h`, so the app `#include <g2g.h>` and links nothing g2g itself, the drop-in packaging a Zephyr shop lists in its west manifest), booted on QEMU's lm3s6965evb, so the *build-system* integration a Zephyr shop uses is proven too, not just the C call): the static element model needs no adaptation layer for an RTOS executor, from either the Rust or the C side. Real capture is interrupt/DMA-driven, not synchronous, and M651 adds that concurrency model. A DMA-completion (or timer) ISR produces frames in interrupt context while the pipeline consumes them in the main/task context, so the two hand frames across the ISR boundary through `g2g_core::SpscFrameRing<N, BYTES>`, a fixed-capacity single-producer / single-consumer FIFO. The producer's `produce` (called from the ISR) fills the next free slot and publishes it; the consumer's `borrow` / `release` drains it in capture order, zero-copy (the frame borrows the ring slot, released after it is dropped). It uses only atomic load/store, no compare-and-swap, so it builds on Cortex-M targets without atomic CAS (`thumbv6m`), and back-pressure is explicit and non-blocking because an interrupt cannot wait: a full ring drops the frame and bumps an overrun counter the consumer reads. `g2g_core::SpscCaptureSrc` is the consumer-side `StaticSource` (the concurrent twin of the synchronous `GrabberSrc`): it drains the ring and, while empty, calls a caller-supplied idle hook (`cortex_m::asm::wfi` on hardware, so the consumer sleeps until the capture interrupt) and retries. Proven on the Cortex-M ISA (`examples/g2g-qemu`'s `isr_capture` bin, in `tools/qemu-check.sh`): a SysTick interrupt is the producer, the main-context pipeline drains it through `SpscCaptureSrc -> G.711 -> checksum` sleeping on `wfi`, and the wire equals synchronous delivery frame-for-frame (`captured=64 overruns=0 OK`), with host thread tests covering lossless-when-paced and drop-and-count-under-back-pressure. The C integration also runs the *other* direction (M650, the zero-Rust driver path a C shop with existing drivers needs): where the FreeRTOS/Zephyr apps link the pipeline and call into it, `g2g-mcu::cffi` lets C code *be* the peripheral, `CFrameGrabber` / `CPacketSender` implement the `FrameGrabber` / `PacketSender` seams over C function pointers (`CaptureFn` / `SendFn` + an opaque `ctx`), so a board registers its existing C capture routine and C network stack and g2g calls them back. `g2g_core::step_source_sink` (with the `Step` enum) is the frame-at-a-time runner that hands control back to the caller after one frame, so a C superloop owns the loop (compose a tail with `SinkChain` to step any linear graph). `examples/g2g-cffi` proves it: a `no_std` staticlib exposing `g2g_audio_egress_init` / `_step` / `_reset` (a `capture -> G.711 -> RTP` pipeline over the C seams) plus `include/g2g_cffi.h`, linked for `thumbv7em` with zero allocator and zero data-panic symbols (the one-frame-step future leaves only a benign, runtime-unreachable async re-poll guard the run-to-EOS runners discharge; `tools/cffi-check.sh` permits that alone), then driven from a real C caller (`harness.c`) whose wire matches the pipeline's Rust reference byte-for-byte, so the C seams are proven byte-transparent. Application code on this surface also needs no `unsafe`: `StaticLendRing::new` is `const` (the ring lives in a `static`, making the zero-copy lend sound by construction via `GrabberSrc`'s safe constructor) and the single-poll executor is the safe `drive_ready`; `m634_forbid_unsafe` proves it by building a full pipeline under `#![forbid(unsafe_code)]`. On top of all this sits the runtime-fault-recovery supervisor the safety / cert market requires (M652, `g2g_core::supervise`, in the no-alloc subset): the static runners propagate a returned fault straight out (the first glitch ends the pipeline), so the supervisor supplies the opposite default, bounded and deterministic recovery. A `FaultPolicy` maps each fault to a `Recovery` action, `Retry` (re-drive a transient fault), `Skip` (drop the frame, keep cadence, degraded mode), `Reset` (re-initialize the stages), or `Escalate`; the supplied `RetryThenReset` and `SkipBounded` cover recover-in-place and degrade-and-continue and both escalate a persistent fault in finite steps. `Recover` is the per-stage re-init seam (default no-op; `GrabberSrc` re-arms via a new `FrameGrabber::reset`, `RtpSink` re-opens via a new `PacketSender::reset`, `SpscCaptureSrc` flushes stale buffered frames so real-time capture resumes from live data), so a supervised pipeline declares each stage's recovery behavior, the traceability a safety case wants, and `SupervisorReport` accounts the faults / retries / resets / skips / escalation. A `Watchdog` is petted only on real forward progress, so a wedged or escalated pipeline stops petting and a hardware watchdog resets the chip (`g2g-mcu::watchdog` supplies the `WatchdogTimer` HAL seam, embedded-hal 1.0 having dropped its watchdog trait, plus the `SupervisorWatchdog` adapter). `step_supervised` (a C superloop / RTOS task owns the loop) and `run_supervised` drive it, bounded by a hard `MAX_ATTEMPTS` cap so even a buggy never-escalating policy cannot hang. Proven on the Cortex-M ISA (`examples/g2g-qemu`'s `supervised` bin, in `tools/qemu-check.sh`): a `capture -> G.711 -> checksum` pipeline recovers a mid-stream latched capture fault (retry, then reset via the `FrameGrabber::reset` seam, then continue, all 64 frames delivered, wire checksum equal to a clean reference, watchdog fed once per frame) and then escalates a dead peripheral within its bounded ladder without hanging, watchdog never fed (`delivered=64 resets=1 wd=64 escalated=4 OK`). The receive direction is the inverse of the capture-to-egress flagship (M653): `g2g_core::rtp::RtpHeader::parse` is the wire-tolerant inverse of `to_bytes` (CSRC list, extension header, and padding, every offset checked and bounds-guarded so a malformed datagram returns `None`, the demuxer discipline; the std H.264 depayloader shares it now), `g2g-mcu::rtprecv` adds the `PacketReceiver` ingress seam and `RtpSrc`, the heap-free `StaticSource` that receives a datagram, parses it, and lends the payload downstream with `Frame::sequence` set to the RTP sequence number, and `g2g-mcu::jitter::JitterBuffer<N, BYTES>` is the reorder element: a fixed `N`-slot reorder window that absorbs arrival jitter, emits the next-in-sequence packet after a prime depth, and handles reorder / duplicate / late / loss explicitly and countably (a packet more than `depth` ahead marks the missing head lost and advances, so one loss never stalls the stream), its output frame borrowing the buffer's own slot zero-copy under the single-frame-in-flight discipline. These compose into the RX flagship `RtpSrc -> JitterBuffer -> G.711 decode`, validated on mocks (a reordered / duplicated / lossy wire reconstructs the ordered decoded PCM byte-for-byte) and on the Cortex-M ISA (`examples/g2g-qemu`'s `rx` bin, proved by an order-sensitive rolling hash equal to an independent in-order decode, `played=14 reordered=3 lost=0 OK`). Both RX elements are `Recover`-capable for the supervisor (the source re-opens its socket, the buffer flushes and re-syncs). The catalog also reaches beyond the media pipeline into the I2C sensor and UART transport a real product needs, with real datasheet-anchored driver logic (M654): `g2g-mcu::sht3x::Sht3xSrc` reads a Sensirion SHT3x temperature/humidity sensor over the `embedded-hal` `I2c` seam, issuing the datasheet single-shot command, validating the two CRC-8 check bytes (polynomial `0x31`, the datasheet `0xBEEF -> 0x92` vector is a test), and converting per the datasheet transfer functions (`i64`-widened so the fixed-point multiply cannot overflow); a CRC mismatch is rejected as a bus-integrity fault rather than trusted. `g2g-mcu::uart` adds local `SerialTx` / `SerialRx` seams (embedded-hal 1.0 keeps blocking serial in `embedded-io`, so a local seam like the packet transports), `UartSink` (frame payload as a byte-stream egress) and `UartSrc` (fixed-size frame ingress), round-tripping over a link. Proven on the Cortex-M ISA (`examples/g2g-qemu`'s `sensor` bin): a mock SHT3x returns a datasheet response and the `Sht3xSrc -> UartSink` pipeline streams each converted reading out a mock UART, the bytes asserted equal to the datasheet conversion (`g2g-sensor: uart-bytes=32 OK`). Finally, the safety / cert market's process artifacts are assembled and, characteristically, made checkable (M655): `docs/safety/REQUIREMENTS.md` is a requirements traceability matrix (15 requirements across memory, timing, faults, concurrency, input validation, and data integrity, each linked to the proof script, test, or CI job that verifies it), and `tools/traceability-check.sh` fails if any cited evidence is missing or if a cited proof script is not wired into CI, so the matrix is a checked claim rather than a document that can drift, run in CI alongside the proofs it indexes; `docs/safety/SAFETY_MANUAL.md` documents the conditions of use, per-property assumptions, the localized `unsafe` inventory, and integrator responsibilities; and `tools/qualification-kit.sh` runs the whole proof set and emits a consolidated requirement-to-evidence-to-result report. This is a down-payment on a product safety case (emulated not silicon, pre-1.0, not a certificate), not a substitute for one. All of the above is also proven to be **not ARM-specific** (M656): because the static element model is ISA-agnostic pure Rust (only the QEMU harness bins carry ARM startup), the `g2g-core` no-alloc subset and `g2g-mcu` build unchanged for `riscv32imafc-unknown-none-elf` (the ESP32-P4 class), and `tools/noalloc-check.sh` asserts the zero-allocator / zero-panic guarantees on both `thumbv7em` and RISC-V archives while `tools/footprint.py` (with an `--isa riscv` stack-frame model) budgets the RISC-V video-pipeline footprint (3718 B ROM / 0 B static RAM / 1328 B stack). The RISC-V footprint model is completed for the flagship audio graph too (M657): rustc encodes that frame as a constant too large for `addi`'s 12-bit immediate, so it materializes the size into a register and does `sub sp, sp, <reg>`; the stack model now resolves that register to its compile-time constant (following the `lui` / `addi` / `slli` materialization chain, and failing rather than under-reporting if it is ever not a known constant), so the RISC-V audio graph is budgeted exactly like the others (10852 B ROM / 0 B static RAM / 6432 B stack, within the ARM audio budgets). The portability claim, a pure-Rust media core across ARM and RISC-V, is machine-checked, not asserted. Targeting a real RISC-V board (the ESP32-P4-EYE) drives two further capabilities. `SpiDisplaySink::with_stripe` (M659) streams a panel too large to ring-buffer whole (240x240 RGBA is 230 KB) in horizontal bands: each frame is one `width x rows` band written to the next vertical sub-window, so the pipeline ring holds a single 15 KB band, and `noalloc_pipeline::run_display_banded_with` is the board-agnostic full-panel runner (the whole-frame path stays byte-identical, so the existing proofs are unchanged). `g2g-mcu::hwh264` (M660) adds the hardware-H.264-encoder seam, the encode twin of `HwJpegDec`: the `H264Encoder` contract (one raw I420 frame in, one Annex-B access unit out, byte count + keyframe flag reported) and `HwH264Enc`, which validates 4:2:0 geometry with checked sizing, cross-checks the reported byte count, and surfaces a faulting peripheral, plus the `CH264Encoder` C bridge so the vendor's hardware encoder driver *is* the peripheral (alongside `CFrameGrabber` / `CPacketSender`); host-tested through a mock and a real `extern "C"` callback (byte-identical, proving the C seam transparent) including a `camera -> encode` pipeline. The board bring-up itself (`examples/g2g-esp32p4`, an esp-hal harness driving the banded panel) is drafted but excluded from CI, since esp-hal's `esp32p4` support is not yet in a published release; on-device execution and the MIPI-CSI / C6-WiFi C drivers are the silicon-gated remainder. The `camera -> encode` path also needs a color convert, since a DCMI/DVP camera emits packed YUYV 4:2:2 but `HwH264Enc` wants planar I420: `g2g-mcu::videoconvert::YuyvToI420` (M661) is the heap-free `StaticTransform` for exactly that (the MCU twin of the `alloc`-based host `VideoConvert`), converting in place through a ring slot with checked geometry, host-tested including a `camera -> convert` pipeline whose output is exactly `HwH264Enc`'s expected I420 size. The convert, encoder, and RTP elements compose: an integration test drives `camera -> YuyvToI420 -> HwH264Enc -> RtpSink` as one static pipeline and traces a camera-stamped byte to the RTP payload. On the ARM side, `examples/g2g-stm32h743` (M661) targets a NUCLEO-H743ZI2: it runs the flagship audio graph and egresses RTP over the H743's on-chip Ethernet through a pure-Rust `embassy-net`/smoltcp stack, the whole g2g-to-network bridge being one `EmbassyNetSender: PacketSender` that maps the RTP egress seam onto an embassy-net `UdpSocket` (no C in the network path, unlike the P4's WiFi). It compiles for `thumbv7em` (verified; excluded from CI only for embassy's build weight), with only runtime config (clock/pins/destination) left for the board.

---

## 3. Data Representation & Memory Subsystem

### 3.1 The Universal `Frame` Carrier
To avoid heavy C-style object allocation, media components flow through lock-free async channels as structured variants representing data packets, lifecycle signals, or negotiation hooks.

```rust
pub enum PipelinePacket {
    CapsChanged(Caps),
    DataFrame(Frame),
    Eos,
    /// Seek flush: discard in-flight and buffered data and reset position
    /// state. Unlike `Eos`, the stream resumes after a flush.
    Flush,
    /// Deadline tick: a fan-in element declaring `tick_interval_ns` gets one
    /// per period even while its inputs stall, so it can emit on its own
    /// cadence (the compositors' zero-order-hold). May fire spuriously, and
    /// never crosses a link: the runner's arm originates and consumes it.
    /// Both runners derive the ticker from the pipeline clock (`as_ticker`
    /// cooperative, `shared_ticker` thread-per-arm; a clock with interior
    /// state reaches the arms via `run_graph_threaded_ticked`).
    Tick,
}

pub struct Frame {
    pub domain: MemoryDomain,
    pub timing: FrameTiming,
    /// Monotonically increasing per-source sequence number assigned at
    /// capture time and preserved unchanged across the pipeline. Used
    /// for drop detection and tracing, never for AV sync.
    pub sequence: u64,
    /// Reserved per-frame attachable metadata (the GstMeta /
    /// GstAnalyticsRelationMeta analog). Empty on construction.
    pub meta: FrameMetaSet,
}
```

**Per-frame metadata (`FrameMetaSet`).** `Frame` carries a reserved `meta`
side-channel for typed blobs that travel with the buffer (ML detection /
classification / tracking results, region-of-interest, reference timestamps).
It is gated behind the `metadata` cargo feature, **off by default**: when off it
is a zero-sized unit, so the `no_std` / RTOS baseline pays nothing per frame;
when on it is a `Vec<Box<dyn FrameMeta>>` where `FrameMeta` is a
`Debug + Send + Sync` trait. The field exists unconditionally so the metadata
system can be filled in without a breaking change to the `Frame` API. The
attach / iterate / propagate contract (GstMeta's `transform_func` / `copy_func`,
plus the `AnalyticsMeta` relation-graph layer) lands with the first
metadata-producing element; until then every frame's set is empty. Construct
frames via `Frame::new(domain, timing, sequence)` so future field additions do
not break call sites. The tee fan-out path gives each clone a fresh empty set
(deep COW propagation is deferred to the full build).

**Caps live on the link, not on the frame.** A `Frame` does not carry its
`Caps`. The current caps of a link are established by the most recent
`PipelinePacket::CapsChanged(Caps)` packet to arrive; every subsequent
`DataFrame` on that link is implicitly under those caps until the next
`CapsChanged` arrives. The runner guarantees `CapsChanged` is **ordered**
in the stream — it sits between the last old-caps `DataFrame` and the
first new-caps `DataFrame`, which is the load-bearing correctness
property for mid-stream format changes (§4.13.4).

See §4.4 for the definition of `FrameTiming` and the pipeline clock model.

### 3.2 Memory Domains
`g2g` treats system RAM as a fallback. Buffers track hardware descriptors to allow cross-process and cross-hardware zero-copy manipulation. Every hardware handle is reference-counted (an `Arc`-held keep-alive owner, or an `Arc`-shared fd for DMABUF): the underlying file descriptor or GPU allocation is released on the *last* drop. `MemoryDomain::share()` produces a second handle for a fan-out branch, a zero-copy refcount bump for the GPU domains and the shared-CPU `SystemView`, a deep copy only for owned-CPU `System` bytes. So a tee broadcasts a GPU-resident frame to several consumers (decode-on-GPU -> {inference, display}) with no device-to-host copy; branches treat the shared memory as read-only (a mutating branch copies first, as the per-frame metadata does copy-on-write).

**Copy / allocation plan.** Because negotiation resolves the memory domain of every link before a frame flows, "is this pipeline zero-copy?" is answerable at construction time, not only measurable after. `copyplan` (pure, like `dot`) turns the negotiated per-edge domains + fixated caps into a `CopyPlan`: the sequence of memory hops (the domain a frame occupies on each edge) and the transfers between differing domains. A transfer is recorded at any node whose output domain differs from the domain it consumed; `classify` sorts it into `None` / `Interop` (dma-buf import/export or a device-to-device bridge) / `DeviceHost` (a GPU download/upload over the bus) / `CrossDevice`, and it counts as a real *frame copy* only when a raw heavy buffer (`Caps::is_raw_media`: raw video, PCM audio, or a tensor) crosses on both sides, so a decode (`CompressedVideo` -> `RawVideo`) or an off-GPU encode is shown in the trace but not miscounted. `CopyPlan::check(CopyPolicy)` (`Allow` / `AtMost(n)` / `DenyAll`) enforces a copy budget as a graph-level contract: a pipeline meant to stay resident on the GPU fails the check the moment an accidental host round-trip appears, rather than silently paying for it at runtime. `g2g-launch --copy-plan` prints the report; `runtime::copy_plan(vg, caps, memory)` builds it from a negotiated graph. The runner enforces it directly: `run_graph_with_copy_policy` runs the plan after negotiation and, *before any frame flows*, refuses to start a graph that exceeds the budget (`G2gError::CopyBudget`), so the guarantee is checked at construction, not measured after. This is what GStreamer cannot state: not "zero-copy is possible" but "this graph is proven zero-copy, or it will not start." The check is scoped precisely, to memory-domain transfers of a raw frame (a device<->host or cross-device copy): an intra-domain algorithmic copy (a `videoconvert` allocating a new System buffer) stays within one domain and is not a domain transfer, and the plan trusts each element's declared `output_memory` / `input_domains`. So "zero-copy" here means "no raw frame crosses a memory-domain boundary," the property that governs GPU-resident and DMA pipelines.

```rust
pub enum MemoryDomain {
    System(SystemSlice),
    DmaBuf(OwnedDmaBuf),
    VulkanTexture(OwnedVulkanTexture),
    WebGPUBuffer(OwnedWebGPUBuffer), // For Wasm targets
}

/// RAII wrapper that closes the underlying DMABUF on drop.
/// On `no_std` targets without libc, the owning `BufferPool` registers
/// a custom close hook via `BufferPool::with_close_fn`.
pub struct OwnedDmaBuf {
    fd: i32,
    pub stride: u32,
    pub offset: u32,
}

impl OwnedDmaBuf {
    /// # Safety
    /// `fd` must be a valid DMABUF descriptor with no other owner.
    pub unsafe fn from_raw(fd: i32, stride: u32, offset: u32) -> Self { /* … */ }
    pub fn as_raw(&self) -> i32 { self.fd }
}
```

Vulkan and WebGPU handles follow the same RAII pattern, parameterised over a backend-specific allocator handle so the spec doesn't bake in a single binding crate.

### 3.3 Zero-Alloc Buffer Pools
Inside real-time or `no_std` loops, dynamic allocation during steady-state streaming is prohibited. Elements acquire pre-allocated slots from a bounded `BufferPool` and dropping the resulting handle automatically returns the buffer.

```rust
let pool = BufferPool::new_byte_pool(count, bytes);
let buf = pool.acquire().await;  // awaits if exhausted; backpressure-friendly
let mut frame = SystemSlice::from_pool(buf, frame_len);  // valid payload length
```

- **`no_std + alloc` environments (and `std`):** `BufferPool<T>` wraps `Arc<Mutex<Vec<T>>>` plus a `VecDeque<Waker>` of acquire waiters. `acquire().await` resolves the moment a `PooledBuffer` elsewhere is dropped. `try_acquire()` is the sync fast path for non-blocking contexts.
- **Strict `no_std` (no heap) environments:** two pure-`core` pools sized at construction, no `alloc`. `StaticBufferPool::<[u8; N], 8>` is the *move-out* pool: `acquire` takes an owned buffer out and the RAII handle returns it on drop, the no-heap analog of `BufferPool`. `StaticLendRing::<N, BYTES>` is the *zero-copy lend* sibling for the capture path (a DMA ring): `N` inline slots, the producer fills the next free slot and `publish`es it as a `SystemSlice` that *borrows* the slot, and a per-slot lease (an `AtomicBool`, plain store, no CAS so it builds on `thumbv6m`) is cleared when the lent frame drops, so the slot is reused only after the consumer is done, the genuine ring back-pressure (the producer stalls when every slot is in flight). The borrow is runtime-guarded, not a Rust lifetime: a `PipelinePacket` crosses the `OutputSink` / stack channel by value (`'static`), so the lend reuses the `'static` foreign-buffer carrier (`SystemSlice::from_foreign`) with the lease standing in for the borrow. This keeps `Frame` / `MemoryDomain` lifetime-free (every element signature stays clean) while still proving a heap-free capture-to-consumer path end to end (validated under `block_on` over the embassy stack channel; a real capture wires a DMA-completion ISR / HAL into the same ring). The heap-free claim is *measured*, not asserted: a counting `#[global_allocator]` test (`m616_no_steady_state_alloc`) runs the `StaticLendRing` capture -> frame -> drop hot path for 100k frames and confirms zero heap allocations across the loop. The control plane carries the same contract since M1000: `OutputSink` is poll-based (its required method is `poll_push`; `push` wraps it in a stack `PushFuture`), so a push through `&mut dyn OutputSink` costs no heap either, and a sibling counting test (`m616_dyn_push_allocates`) pins that at zero. The remaining opt-in is the element's own `ProcessFuture`: an element that declares a boxed one pays one box per `process` call; one that declares a concrete future type runs heap-free through the whole dyn runner (`m1000_dyn_graph_noalloc` proves a 3-stage `run_graph` steady state at zero allocations).

The `SystemSlice` carrier transparently supports these ownership models: `SystemSlice::from_boxed(Box<[u8]>)` for one-off frames, `SystemSlice::from_pool(PooledBuffer<Box<[u8]>>, len)` for recycled frames (the buffer may exceed the frame, so the valid length is carried), and `SystemSlice::from_foreign(ptr, len, free, user)` for a zero-copy lend of borrowed bytes (a `StaticLendRing` slot, or an application buffer through the C ABI). Downstream elements treat them identically.

---

## 4. Graph Orchestration & Capability Negotiation

### 4.1 Compile-Time and Runtime Caps
Traditional architectures rely on runtime string lookups for stream capabilities (e.g. `"video/x-raw, format=NV12"`). `g2g` enforces strongly typed structures.

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum Caps {
    Video {
        format: VideoFormat,
        width: Dim,
        height: Dim,
        framerate: Rate,
    },
    Audio {
        format: AudioFormat,
        channels: u8,
        sample_rate: u32,
    },
    Tensor {
        dtype: TensorDType,
        shape: TensorShape,
        layout: TensorLayout,
    },
}

/// `Fixed` after Phase 2; `Range`/`Any` only legal during Phase 1.
pub enum Dim { Any, Range { min: u32, max: u32 }, Fixed(u32) }
pub enum Rate { Any, Range { min_q16: u32, max_q16: u32 }, Fixed(u32) }
```

The `Tensor` variant is first-class because ML elements (§5) negotiate caps the same way video elements do — they don't sit outside the graph model. (The sketch above is conceptual; the real enum splits video into `CompressedVideo` / `RawVideo` per the codec-vs-raw distinction, and adds `ByteStream` for not-yet-demuxed container links.)

`Text` is likewise a first-class media kind (`Caps::Text { format: TextFormat }`), not a bolted-on subtitle path. It generalizes "subtitles": a `Text` link carries any text payload — a subtitle cue, a caption, a transcription, an OCR result, an overlay string — with `TextFormat` naming the syntax (`Utf8`, `PangoMarkup`, and the structured `Srt` / `WebVtt` / `Ssa` / `Ttml`). "Subtitle" is not a separate variant: it is just *timed* `Text`, the cue's on-screen window carried as the frame's PTS + duration, so one caps kind serves overlay rendering, captioning, and text analytics. A subtitle parser (`SubParse`) is the text-domain analog of a codec decoder, taking a structured format on its sink pad and emitting plain `Utf8` cues via the same `DerivedOutput` negotiation a decoder uses for compressed -> raw, so subtitle text flows through the graph as a typed stream rather than being loaded out-of-band.

### 4.2 The Capability Negotiation Lifecycle
Because `g2g` enforces a Sans-IO and asynchronous execution model, capability negotiation happens in a clear, deterministic handshake before any data frame processing begins. This replaces GStreamer's complex query/event system with a simple, state-machine-driven future matrix.

```
                   Phase 1: Downstream Query (Caps Filter)
           Element A ───────────────────────────────────► Element B
                     "Here is what I can produce.
                      What can you handle?"

                   Phase 2: Upstream Selection (Fixate)
           Element A ◄─────────────────────────────────── Element B
                     "I choose NV12 at 1080p.
                      Allocate your buffers."

                   Phase 3 (rare): Re-fixation
           Element A ◄─────────────────────────────────── Element B
                     "Allocation failed at 1080p;
                      counter-propose 720p."
```

**Phase 1 — Downstream Query (Intersection):** The runner invokes `intercept_caps()` on the source, passing initial configuration or upstream hardware constraints. Each element returns a `Caps` value containing ranges or `Any` where parameters are flexible. The downstream peer intersects against its own internal capabilities and returns a narrowed set.

**Phase 2 — Upstream Selection (Fixation):** Once an intersection is found, the final caps are fixated (all `Dim`/`Rate` values become `Fixed`). The fixated `Caps` travel back upstream via `configure_pipeline()`. Each element allocates exact byte arrays or VRAM texture sizes, ensuring zero dynamic allocations during steady-state streaming.

**Phase 3 — Re-fixation (rare):** If an element's allocation fails (VRAM budget, driver limit), `configure_pipeline()` returns `ConfigureOutcome::ReFixate(Caps)` with a counter-proposal. The runner restarts Phase 2 from that element. This bounded backtrack avoids the GStreamer pattern of failing the entire pipeline on allocation pressure.

### 4.3 The `AsyncElement` and `SourceLoop` Traits
Transform and sink elements implement `AsyncElement` — packet in, 0..N packets out. Source elements have no input pad and instead implement `SourceLoop`, which is called once and iterates internally until EOS. The two traits share `intercept_caps` / `configure_pipeline` semantics.

```rust
use core::future::Future;

pub trait AsyncElement: ElementBound {
    type ProcessFuture<'a>: Future<Output = Result<(), G2gError>> + 'a
    where Self: 'a;

    /// Phase 1: Intersect proposed caps with internal capabilities.
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError>;

    /// Phase 2/3: Fixate the agreed caps and initialize hardware buffer pools.
    /// Returns `ReFixate(caps)` to trigger Phase 3 with a counter-proposal.
    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError>;

    /// Execution: process one input packet, pushing 0..N outputs into `out`.
    /// Mutable self accommodates stateful codecs, demuxers, and parsers;
    /// the sink accommodates fan-out (demuxers), fan-in (batchers), and
    /// elements that emit nothing until enough input has accumulated.
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a>;
}

pub trait SourceLoop: ElementBound {
    type RunFuture<'a>: Future<Output = Result<u64, G2gError>> + 'a
    where Self: 'a;

    fn intercept_caps(&self) -> Result<Caps, G2gError>;
    fn configure_pipeline(&mut self, absolute_caps: &Caps)
        -> Result<ConfigureOutcome, G2gError>;

    /// Runs until EOS or error. Implementation MUST emit a final
    /// `PipelinePacket::Eos` before returning. Returns the count of
    /// `DataFrame` packets pushed (excluding `Eos`).
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a>;
}

pub enum ConfigureOutcome {
    Accepted,
    ReFixate(Caps),
}

/// Output sink for both transform and source elements. Push is async so
/// elements await downstream capacity rather than failing fast on a full
/// bounded link. Dyn-safe via the poll form: `push` (provided for concrete
/// sinks and on the trait object) wraps `poll_push` in a concrete stack
/// `PushFuture`, so a push through `&mut dyn OutputSink` allocates nothing
/// (M1000).
pub trait OutputSink {
    fn poll_push(
        &mut self,
        cx: &mut Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>>;

    fn begin_push(&mut self) {}
}
```

#### Thread-safety bounds
The `ElementBound` marker is `Send` on multi-threaded targets and empty on single-core ones, gated by the `multi-thread` cargo feature. Embassy and the WebGPU/main-thread Wasm executor do not require `Send`, and many hardware-handle types cannot satisfy it.

```rust
#[cfg(feature = "multi-thread")] pub trait ElementBound: Send {}
#[cfg(feature = "multi-thread")] impl<T: Send> ElementBound for T {}
#[cfg(not(feature = "multi-thread"))] pub trait ElementBound {}
#[cfg(not(feature = "multi-thread"))] impl<T> ElementBound for T {}
```

Note: `Sync` is intentionally not required. `AsyncElement::process` takes `&mut self`, so concurrent calls are statically prevented; cross-task sharing happens through channels, not shared references.

#### Dynamic dispatch
The GAT-based `AsyncElement` is not `dyn`-safe. For plugin registries on `std` targets, `g2g-core` provides a boxed adapter:

```rust
#[cfg(feature = "std")]
pub trait DynAsyncElement: ElementBound {
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError>;
    fn configure_pipeline(&mut self, absolute_caps: &Caps)
        -> Result<ConfigureOutcome, G2gError>;
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> core::pin::Pin<alloc::boxed::Box<
        dyn Future<Output = Result<(), G2gError>> + 'a
    >>;
}

#[cfg(feature = "std")]
impl<T: AsyncElement> DynAsyncElement for T { /* blanket boxed-future impl */ }
```

Since M1000 the graph runner does not pay this box per frame: `DynAsyncElement` carries `drive_transform_arm` / `drive_sink_arm` hooks whose blanket impls monomorphize the arm loop over the concrete element type, so `run_graph` awaits each element's own `ProcessFuture` unboxed (one boxed arm future per run, none per frame). The erased `process` above remains for callers that drive an element directly through the trait object; an element opting into the zero-alloc steady state declares a concrete (non-boxed) `ProcessFuture`. M1009 extends the same treatment to the fan-in / fan-out arms: `DynMultiOutputElement::drive_demux_arm` and `DynMultiInputElement::drive_muxer_arm` / `drive_muxer_arm_owned_tick` / `drive_fanin_sink_arm` monomorphize the demux, muxer (arrival-order and PTS-ordered) and terminal fan-in arms over their element, so a demux or muxer node also runs with no per-packet box. A graph node built from a `&mut dyn` element keeps a boxed per-packet future: the concrete type is already erased, so the arm drives it through a private `AsyncElement` / `MultiInputElement` / `MultiOutputElement` face over the trait object.

`no_std` graphs use concrete element types composed via a typed graph builder (no boxing, no virtual dispatch).

### 4.4 Pipeline Clock & Timing Model
All timestamps in `g2g` are `u64` nanoseconds relative to a single **pipeline reference clock**. Source elements map their hardware capture clock onto the reference clock during `configure_pipeline`; downstream elements treat presentation timestamps as monotonic.

```rust
pub struct FrameTiming {
    /// Presentation timestamp, ns relative to the pipeline reference clock.
    pub pts_ns: u64,
    /// Decode timestamp. Differs from PTS for B-frames; equals PTS otherwise.
    pub dts_ns: u64,
    /// Nominal frame duration. 0 means "until next frame arrives".
    pub duration_ns: u64,
    /// Hardware capture timestamp in the source's native clock, preserved
    /// unchanged across the pipeline for end-to-end latency measurement.
    pub capture_ns: u64,
}

pub trait PipelineClock {
    fn now_ns(&self) -> u64;
}

/// Pipeline clock with async sleep. Sync sinks, paced sources, and jitter
/// buffers take `AsyncClock` rather than `PipelineClock` so they can both
/// observe and schedule against time. `sleep_until_ns(d)` resolves
/// immediately if `d <= now_ns()`.
pub trait AsyncClock: PipelineClock {
    type SleepFuture<'a>: Future<Output = ()> + 'a where Self: 'a;
    fn sleep_until_ns<'a>(&'a self, deadline_ns: u64) -> Self::SleepFuture<'a>;
}
```

A `pts_ns` of `FrameTiming::PTS_NONE` (`u64::MAX`, the value GStreamer spells `GST_CLOCK_TIME_NONE`) marks a frame with no presentation time: `FrameTiming::pts()` reads it as `None`, and `PresentationPacer` answers `Pace::Now` for it without latching its anchor, so a sink presents the frame as it arrives rather than holding it to a deadline or counting it a late drop.

Sink elements compare `pts_ns` against `now_ns()` to schedule presentation, and `capture_ns` against `now_ns()` to report true glass-to-glass latency without ambiguity about which clock domain a timestamp lives in. Backends provide concrete implementations: a `WallClock` (`std::time::Instant` + `tokio::time::sleep`) for std targets, `embassy-time` for RTOS, performance.now() for Wasm.

A free-running source feeding a sync sink is paced automatically by upstream backpressure (§4.5): the sink only consumes after `sleep_until_ns(pts)` resolves, which throttles the channel, which throttles the source. No explicit source-side pacing is required for sync playback.

#### Clock distribution to sinks

A pipeline runs against one elected clock (`elect_clock` over `ClockPriority`: a PTP grandmaster-disciplined clock (`PtpGrandmaster`) outranks a live source's hardware clock (`LiveSource`), which outranks an audio sink's DAC clock (`AudioProvider`), which outranks a plain monotonic provider such as a video display sink (`Provider`), which outranks the system fallback). The runner samples the elected clock's `now_ns()` once at startup as the **base time** (the clock reading at running-time zero) and hands both to each sink via `set_clock_sync(ClockSync { clock, base_time_ns })`, called once after election. Both the linear runners and the DAG runner `run_graph` deliver it (the latter walks its sink nodes after election), so a display sink PTS-paces in any topology. A sink that synchronises presents a frame when the elected clock reaches `base_time_ns + running_time`, where running time is the frame's `pts_ns` mapped through the active `Segment`; a sink that ignores the hook presents as fast as backpressure allows.

**Clock loss and re-election.** An elected clock can lose the reference it is disciplined to (a PTP servo going free-running when its grandmaster disappears), which `PipelineClock::healthy` reports: true by default, since a clock reading a monotonic counter or a DAC has nothing to lose, and overridden by `PtpClock` (healthy in `Locked` and `Holdover`, not in `FreeRunning`). When a bus is attached and the runner has a timer to sleep on, `run_graph` (cooperative and thread-per-arm alike) runs a health monitor alongside the arms: it reads the elected clock once a second and, on a loss, posts `BusMessage::ClockLost`, elects again over the candidates that are still healthy, and retargets every sink. The retarget works because in that mode the sinks' `ClockSync` points at an `ElectedClock`, a shared handle over a swappable target, rather than at the clock itself: the elements are already inside their arms (on other threads under the thread-per-arm runner), so there is no second `set_clock_sync`. A sink re-anchors on its next frame, as it does for any epoch change. `ElectedClock` answers `shared_ticker` (owned) but not `as_ticker` (a borrow into a target that can be replaced), which is why the indirection is installed only when the monitor runs. With no healthy candidate left the pipeline keeps the clock it has: it still tells time, it is just no longer disciplined, and a later re-lock is picked up by the same check.

**Pacing mid-graph (`clocksync`).** Presentation is not the only place a stream needs to run at real time: a publisher muxing to a live transport (the MoQ Transport demo's `videotestsrc ! x264enc ! mp4mux ! moqtsink`) has no sync sink at all, so nothing stops it producing minutes of media per minute of wall clock. `ClockSyncTransform` (`clocksync`) is the sink's pacing as a pass-through transform: it holds each buffer until its PTS, anchored on the first one, is due on the clock, and forwards everything else unchanged. It shares the display sinks' `PresentationPacer`, so the anchor, the segment mapping and the seek re-anchor behave identically, and differs in two ways. It never drops: a late or segment-clipped buffer is forwarded immediately, because a hole in a transform's output is one downstream cannot recover. And it supplies its own monotonic clock when none was handed to it, which is both GStreamer's fallback to the pipeline system clock and a necessity here, since the runners deliver `ClockSync` to sink nodes only, and a `clocksync` sits mid-graph. `sync=false` reduces it to an identity; `ts-offset` shifts the whole schedule.

**Audio as the sync master.** For playback the audio sink should drive timing, because samples leave the DAC at the hardware's real rate, which drifts from wall time by tens to hundreds of ppm. `DriftClock` (`g2g-core`) turns that into a usable pipeline clock: it is fed `(local_ns, master_ns)` observations (`local_ns` from a monotonic reference, `master_ns` the true playout position) and fits `master ≈ slope·local + offset` by least squares over a sliding window, so `now_ns()` projects the current reference time through the fit, both estimating the playout rate and smoothing the coarse, jittery per-observation readings. `AlsaSink`'s worker samples `frames_written − snd_pcm_delay()` after each blocking `writei` and feeds the clock, offering it to election at the `AudioProvider` tier (gated by a `provide-clock` property). A video sink then slaves to it: because the elected clock is the disciplined audio timeline rather than raw wall time, video presentation follows audio, giving true A/V sync. A `LiveSource` capture clock still wins when present, so a live pipeline paces to capture.

**Networked sync (PTP).** For facility-wide sync (Pro AV / SMPTE ST 2110), the shared reference is a PTP grandmaster, and every device slaves to it, so a `PtpGrandmaster` clock outranks all of the above. `PtpServo` (`g2g-core::ptp`) is the servo: fed the four timestamps of each PTP delay request-response, it computes the standard `offset` / `mean_path_delay` and folds `(local, master)` into the same `DriftClock` machinery, disciplining the local monotonic reference to the grandmaster's TAI timeline with lock / holdover / outlier-rejection state. `PtpClock` wraps it (interior-mutable, so one worker drives it while sinks read `now_ns` through a shared `Arc`) and offers itself to election only once locked. Because the elected timeline is grandmaster-derived, two machines locked to the same grandmaster read the same clock, so the A/V pacing above holds *across* devices, not just within one process. Two sources feed the servo: raw PTP message timestamps (`sync_exchange`), or a direct absolute-time observation (`observe_master`). Two backends supply them: `PtpSystemClock` (`g2g-plugins`, Linux) delegates to an OS PTP-disciplined `CLOCK_TAI` (from `linuxptp` / `phc2sys`), sampled on a worker; `PtpClient` (`g2g-plugins`) is a from-scratch software PTP SLAVE that speaks PTP over UDP itself (the `ptp::wire` message parser + the `ptp::slave` delay-request-response state machine + a UDP transport), so an endpoint with no OS PTP daemon can still lock. The wire parser and slave state machine are `no_std` and CI-tested end to end (parse -> slave -> servo) without sockets. Both backends coexist with a host `ptp4l`: `PtpClient`'s sockets take `SO_REUSEADDR` + `SO_REUSEPORT` so the daemon keeps receiving its own copy of each multicast message, and `PtpSystemClock` polls the daemon over its management socket (`ptp::management` builds the same GET `pmc` sends; `g2g-plugins::ptp4l` carries it over the Unix datagram socket) so `grandmaster_locked` reports the port state behind `CLOCK_TAI` rather than trusting a clock that is readable either way.

**ST 2110 media transport** rides on this shared clock. Distinct time newtypes guard the seam where three "just an integer" times meet: `TaiNs` (PTP/TAI nanoseconds, absolute), `RtpTs` (the 32-bit wrapping RTP media-clock timestamp on the wire), and `RefNs` (the pipeline's monotonic reference nanoseconds, a relative timeline with an arbitrary epoch). `MediaClock` takes a `TaiNs` and returns an `RtpTs`, so the compiler rejects handing it the wrong clock (the confusion the PTP servo work hit: a monotonic reference minus a TAI master is meaningless); the PTP servo's own seam is typed the same way, `PtpServo` / `PtpClock` `sync_exchange` take `(TaiNs, RefNs, RefNs, TaiNs)` and `observe_master` takes `(RefNs, TaiNs)`, so master and reference can no longer be swapped. Durations stay a plain `u64`. `MediaClock` (`g2g-core`, ST 2110-10) maps a PTP/TAI time to a 32-bit wrapping RTP timestamp and back (a media clock counting at 90 kHz for video / the sample rate for audio from the PTP epoch), so two receivers on the same grandmaster compute the same timestamp for the same sampling instant. `st2110audio` (`g2g-plugins`, ST 2110-30) is the sans-IO PCM payloader/depayloader (L16 / L24 big-endian in the RTP payload, timestamps off the media clock), and `st2110anc` (ST 2110-40 / RFC 8331) carries SMPTE ST 291 ancillary data (closed captions, timecode) as bit-packed 10-bit words with parity + checksum validation, so the caption stack can ride 2110. The sans-IO cores get network element wrappers: `st2110audiortp` (`St2110AudioSink` / `St2110AudioSrc`, behind the `st2110` feature) puts -30 audio on the wire over UDP, the sink mapping each frame's PTS through the elected (PTP) clock to the media-clock timestamp and the source reconstructing PTS from it, so a receiver on the same grandmaster stays in sync. `st2110ancrtp` does the same for -40 captions: `St2110AncSink` taps a compressed H.264 / H.265 stream (a teed branch leaf like `CcExtract`), mines each access unit's caption triples, wraps them in a Caption Distribution Packet (CDP, CEA-708 / SMPTE ST 334-2) carried in a DID 0x61 ANC packet, and sends the RFC 8331 RTP timestamped at the frame's PTP time; `St2110AncSrc` depacketizes -40 back into triples and, through the shared `CaptionDecoder` (the decode core factored out of `CcExtract`, driving the same CEA-608/708 state machines from triples mined from SEI or carried in a CDP), emits timed `Caps::Text{Utf8}` cues. So captions travel end to end over 2110 and stay frame-aligned on a common grandmaster. `st2110video` (ST 2110-20 / RFC 4175) carries uncompressed active video: the packetizer slices a packed frame into Sample Row Data (SRD) line runs (an Extended Sequence Number then per-run headers giving scan line, pixel offset, octet length) sized to the MTU, and the depacketizer writes each run back into the frame, completing it on the RTP marker bit; `st2110videortp` (`St2110VideoSink` / `St2110VideoSrc`) puts it on UDP with the 90 kHz media-clock timestamp shared by every packet of a frame. Each sampling is a `Layout` reading / writing one pgroup at a time, so the packetizer / depacketizer stay layout-agnostic across three mappings: RGBA 8-bit (packed, byte-identical), YCbCr-4:2:2 8-bit (packed `Yuyv`, luma / chroma bytes swapped to the wire), and YCbCr-4:2:2 10-bit (the broadcast norm, from the planar `I422p10` buffer: the four 10-bit samples Cb0 Y0 Cr0 Y1 are MSB-first bit-packed into a 5-octet pgroup, crossing both a planar-to-packed and a byte-to-bit boundary). The source's geometry comes from properties, or from the stream's SDP: `st2110sdp` (RFC 4566 + SMPTE ST 2110-10/-20/-30/-40) is the sans-IO generator / parser for the out-of-band description a receiver configures from, carrying the essence (video sampling / size / rate, audio depth / rate / channels / ptime, or ancillary), the payload type, the multicast group and port, and the `a=ts-refclk` PTP grandmaster all the streams share. every sink has an `sdp()` that publishes its stream and every source an `apply_sdp()` that auto-configures from a parsed one, so a stream self-describes end to end across video, audio, and ancillary. On the audio side `PcmS16Le` rides as L16 and `PcmF32Le` as L24 (float scaled to the 24-bit wire). `st2110jxs` (ST 2110-22 / RFC 9134) carries the compressed mezzanine essence, JPEG XS: the packetizer slices an opaque codestream into codestream-mode packets (the 4-octet RFC 9134 payload header carrying transmode / packetmode / last-packet / frame counter / packet counter), the marker bit ending the frame, every packet on the same 90 kHz media clock; `st2110jxsrtp` (`St2110JxsSink` / `St2110JxsSrc`) puts it on UDP, taking / emitting `Caps::CompressedVideo{JpegXs}` frames. The JPEG XS codec itself is `SvtJpegXsEnc` / `SvtJpegXsDec` (`jpegxs` feature): hand-rolled FFI to Intel SVT-JPEG-XS (ISO/IEC 21122, no libavcodec), planar 4:2:0 / 4:2:2 8-bit and 4:2:2 10-bit, the encoder targeting a bits-per-pixel budget and the decoder discovering geometry from the first codestream. So a plant can move visually lossless video at a fraction of -20's bandwidth with sub-frame latency, end to end (raw -> encode -> -22 -> decode). SDP covers all essences, including -22 (`jxsv`), and `St2110Session` bundles video + audio + ancillary into one multi-section session document (each media tagged with `a=mid`, a shared `a=ts-refclk`), so a whole program self-describes. `AudioFormat::PcmS24Le` (integer 24-bit) rides the -30 L24 wire directly, alongside the float path. `st2110dup` implements ST 2110-7 seamless protection: a receive-side sequence-number merge of two identical redundant streams (first arrival wins, so a loss on one path is filled by the other), with `a=group:DUP` in the session SDP. Its `SeamlessDedup` is the sans-IO core; `RedundantRtpReceiver` (behind the `st2110` feature) is the socket-bound sibling that binds several receive paths, polls them round-robin so two in-order streams merge back into sequence order, and yields deduplicated packets. `St2110VideoSrc` adopts it behind a `redundant` property (a second "blue" path); being essence-agnostic it can serve the other essences the same way. `st2110pacing` implements ST 2110-21 sender pacing: a schedule spreading a frame's packets across the frame period (linear or gapped), which both `St2110VideoSink` and the -22 `St2110JxsSink` realize over the tokio timer (through a shared `pace_send`) so the network sees a smooth flow instead of a burst. `VrxValidator` is the full per-format -21 compliance check: the leaky-bucket virtual-receive-buffer model (a receiver draining one packet every `TRS` after a `TR_OFFSET` head start) that, over a run of actual emission offsets, reports the peak buffer occupancy, whether a packet arrived late (starving the receiver), and whether it stays within the profile's `Cmax`. What is built (from the RFCs, loopback-tested, not yet interop-validated against reference gear) now spans -10/-20/-21/-22/-30/-40/-7 plus SDP; multicast interop remains.

`WaylandSink` is the first display sink to use it: it holds each frame until its running-time deadline, tracking the `Segment` (clipping pre-target frames after an accurate seek) and re-anchoring on `Flush`. It also does **QoS late-drop** (matching `SyncSink`): a frame already past its deadline by more than a configurable `max_lateness` bound is dropped instead of presented late, so the sink catches up instead of accumulating lag, posting a `BusMessage::Qos` (running time, jitter, cumulative processed/dropped) per drop.

`SyncSink` is the same sink without a display, so clock slaving is CI-testable with no hardware: it paces through the shared `PresentationPacer` and adopts the elected `ClockSync`, so in an A/V graph its deadlines land on the audio master's `DriftClock` timeline. Its own clock stays the timer (the only thing that can sleep in `no_std`), and the pacer's wait is relative so the two can be different timelines. Until a clock is elected it paces on its own clock with the anchor pinned at zero (`PresentationPacer::set_anchor_ns`), which makes a frame's deadline its running time and the recorded drift a real end-to-end latency reading; adopting an elected clock drops the pin, since that clock's epoch is its own.

**Playing-transition anchoring.** The startup base time is sampled before the data plane and before the application presses play. For a non-live, prerolled pipeline that sits in `Paused` for a while, that is the wrong epoch: the preroll frame is consumed during `Paused`, so a sink that anchored on the startup base (or on that first frame) then rushes/drops once `Playing` finally arrives. So when a `StateController` drives the run, the runner arms a `PlayAnchor` (a shared cell) on the elected clock and hands each sink `ClockSync::with_play_anchor`; `set_state(Playing)` stamps the anchor with `clock.now_ns()` at the exact play edge (and a transition down to `Ready`/`Null` clears it, so a replay re-bases). `ClockSync::base_time()` then resolves to the play-edge stamp once armed, else the eager startup base time. `WaylandSink` reads it per frame: it first-frame-anchors a preroll frame consumed during `Paused` (presented immediately), then re-bases onto the play edge once `Playing` stamps it; a seek `Flush` forces a first-frame re-anchor so the seek target presents immediately rather than against the stale play base. The non-stateful runners keep the eager base time (no `StateController`, no play edge to anchor to).

**Upstream QoS** carries that lateness back to the producer so it sheds load too, not just the sink. It rides the same per-link reverse channel as `Reconfigure`: a sink returns a `QosMessage` from `AsyncElement::take_qos`, the runner stores it into the incoming link's reverse `QosSlot`, and the producer observes it as `PushOutcome::Qos` on its next push (reconfigure wins when both are pending; QoS is advisory and never holds the packet back). `SyncSink` originates it on a late-drop and `VideoTestSrc` reacts by skipping ~`jitter / frame_period` frames (advancing PTS without generating them). **Relay through a transform** carries the report the rest of the way to the source in a multi-element pipeline. A transform observes a downstream QoS as a `PushOutcome::Qos` inside `process`, but that outcome is discarded by a generic transform, and the runner (not the element) owns the reverse slots, so the relay is runner-mediated: the runner wires the transform's *output* `SenderSink` with a relay handle to its *input* link's `QosSlot` (`relay_qos_to`). When the output adapter then sees a downstream QoS it stores it onto the input link instead of surfacing it, so the upstream neighbour observes it on its next push, and across N transforms the report walks one hop at a time back to the source. The element's `process` is unaffected. **Acting on the report** is the other half: an element that returns `true` from `AsyncElement::handles_qos` is not relayed past, so it observes the report as `PushOutcome::Qos` from its own `push` and sheds work itself, the same opt-out `handles_keyframe_requests` / `handles_bitrate_requests` give an encoder. `FfmpegVideoDec` does that under its `qos` property: a report arms a skip budget of `jitter / frame_period` pictures (capped by `max-skip-frames`), during which the codec context runs with `AVDISCARD_NONREF`, so libavcodec stops decoding the pictures nothing references. Decode cost drops without touching a reference chain, so every frame still emitted is bit-identical to a full decode, and the budget counting down to zero is the recovery. This is the same shape as the reverse `Reconfigure` path. Wired in the bespoke `run_source_transform_sink` runner and in the DAG runner (`run_graph` / `run_linear_chain`, which the `WaylandSink` demo uses), so the sink's own load-shed reaches the source through interior transforms (overlay, convert).

### 4.5 Backpressure & Scheduling
Every link between elements has an explicit `LinkPolicy`, configured at graph construction time. The choice is per-link because a single pipeline may have lossy preview branches and lossless recording branches sharing an upstream source.

```rust
pub enum LinkPolicy {
    /// Block the upstream future until the channel has capacity.
    /// Lossless; raises latency under load.
    Block,
    /// Drop the oldest queued frame on downstream stall.
    /// Default for live camera sources.
    DropOldest,
    /// Drop the newest (incoming) frame on downstream stall.
    /// Use when temporal coherence matters more than freshness
    /// (e.g. driver-assistance ML where stale-but-coherent beats torn).
    DropNewest,
}
```

The leaky variants are implemented in the per-edge data-plane sink: under a full channel, `DropNewest` discards the incoming frame and `DropOldest` evicts the oldest queued frame to make room. Only `DataFrame`s are ever dropped, control packets (`CapsChanged` / `Segment` / `Flush` / `Eos`) always block, so a leaky link never corrupts the stream; if a full queue holds only control packets, `DropOldest` falls back to blocking. Drops are pipeline-observable, never silent: `RunStats::frames_dropped` reports the total, and `run_graph` applies each edge's policy set via `graph.link_with`. This per-edge policy replaces GStreamer's explicit `queue` element, every link is already a bounded channel and every node already its own scheduling arm.

### 4.6 The `G2gError` Type
Errors are a single closed enum so element authors handle the full set exhaustively. Hardware-specific failures carry a backend-tagged payload rather than collapsing to a `String`.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum G2gError {
    /// Phase 1 failure: no non-empty intersection between proposed
    /// upstream caps and this element's supported caps.
    CapsMismatch,
    /// Element received a DataFrame before configure_pipeline succeeded.
    NotConfigured,
    /// Phase 2 failure: caller should retry Phase 1 with the proposal
    /// returned in `ConfigureOutcome::ReFixate`.
    FixationFailed,
    /// Buffer pool exhausted; transient, retry after upstream drain.
    PoolExhausted,
    /// Memory domain handed to an element that cannot consume it
    /// (e.g. a CPU-only filter receiving a VulkanTexture).
    UnsupportedDomain,
    /// Backend-specific hardware/driver failure.
    Hardware(HardwareError),
    /// Pipeline is shutting down; element should drain and propagate Eos.
    Shutdown,
}
```

### 4.7 Pad Model: Implicit by Trait Shape
Pads are not a first-class type. An element's input and output endpoints are encoded by which trait it implements and by the `&mut dyn OutputSink` parameter shape; there is no `pub struct Pad`, no per-pad metadata, no runtime introspection.

| Topology | Trait | Input pad | Output pad |
| :--- | :--- | :--- | :--- |
| Source (0→1) | `SourceLoop` | — | `&mut dyn OutputSink` arg to `run()` |
| Transform / sink (1→0..N) | `AsyncElement` | `PipelinePacket` arg to `process()` | `&mut dyn OutputSink` arg to `process()` |
| Terminal sink | `AsyncElement` whose `process()` ignores `out` | as above | `NullSink` sentinel |

This is deliberate. GStreamer's `GstPad` is a runtime object because GStreamer composes graphs from string-keyed plugin factories loaded at runtime; `g2g` composes typed graphs at compile time, so pad metadata lives in the trait signatures. The cost is that fan-out (tee), fan-in (muxer), and demuxer-style dynamic pads require additional trait variants rather than runtime pad-list mutation — see §4.10.

### 4.8 Dynamic Graph Reconfiguration

#### 4.8.1 Two-Layer Graph API
`g2g` exposes two graph APIs sharing the same element traits, the same negotiation lifecycle, the same `PipelinePacket` variants, and the same runner primitives. Only graph construction and slot mutation differ.

- **Static typed graph** — compile-time topology via tuple types; no `dyn`; zero-cost. Right for embedded / RTOS / static cloud pipelines.
- **Type-erased dynamic graph** — boxed elements (`Box<dyn DynAsyncElement>`) held in `ElementSlot`s and `BranchSlot`s, swappable at runtime. Right for cloud ingestion, desktop applications, and anything that needs runtime topology evolution.

#### 4.8.2 `ElementSlot` — Lock-Free Single-Element Swap
The dynamic graph holds elements in `arc_swap::ArcSwap<Box<dyn DynAsyncElement>>` cells:

```rust
let new_element = SomeTransform::new();
new_element.configure_pipeline(&caps)?;
slot.handle.store(Arc::new(Box::new(new_element)));
```

Frames mid-`process()` against the old element complete naturally; the next push observes the new element. Cost: one atomic store plus the new element's `configure_pipeline()` work. No drain, no pipeline stall.

This is the primary response to a Phase 3 `ReFixate` or a mid-stream `Reconfigure` signal: replace the affected slot's contents, do not rebuild the graph. The swap is validated live under load: an `ElementSlot` sits as a transform in `source -> slot -> sink` driven by `run_graph`, and a `SwapHandle::swap` mid-stream reroutes the remaining frames to the replacement element while every frame still reaches the sink, no drain or rebuild.

#### 4.8.3 `BranchSlot` — Multi-Element Sub-Graph Swap
A branch with one logical input and one logical output is structurally an element. `BranchSlot` is the multi-element analog of `ElementSlot`, with the swap trade-off made explicit at the type level:

```rust
pub struct BranchHandle<I, O> {
    input_tx: LinkSender<I>,
    output_rx: LinkReceiver<O>,
    tasks: Vec<JoinHandle<()>>,
}

pub struct BranchSlot<I, O> {
    handle: arc_swap::ArcSwap<BranchHandle<I, O>>,
    policy: SwapPolicy,
}

pub enum SwapPolicy {
    /// Flip input routing; in-flight frames inside the old branch's
    /// internal channels are discarded. Zero latency; bounded frame loss.
    /// Right for stateless filters (color grade, debug overlay).
    Immediate,

    /// Flip input routing; wait for old branch to drain its in-flight
    /// frames before exposing the new branch's output to the consumer.
    /// Zero loss; pays the old branch's pipeline depth in latency.
    DrainOld,

    /// Both branches consume in parallel for a brief overlap window;
    /// the merger cuts over at the named signal (next IDR, next segment
    /// boundary, etc.). Zero loss, zero per-frame stall; brief duplicated
    /// compute during the overlap.
    ShadowWarm { cutover: CutoverSignal },
}
```

Static-graph users at the embedded layer never instantiate `BranchSlot` and don't pay for any of this machinery.

#### 4.8.4 Router, Gate, Merger Primitives
A `Router` is a 1-to-N transform that reads an atomic discriminator per frame and pushes the frame to exactly one of its outputs. A `Gate` is a 1-to-1 transform that reads an atomic boolean and either forwards or discards each frame. A `Merger` is an N-to-1 transform that reads from one of its inputs, switching on a discriminator. Together they cover branch enable/disable, A/B switching, and the routing + cutover halves of `ShadowWarm`. These primitives form the foundation of the dynamic-graph layer.

#### 4.8.5 Runtime Request Pads
The request-pad analog is a pair of handles over a running graph. `DynamicFanoutHandle::add_branch` (M310) attaches an output branch mid-run, round-robin or broadcast (`FanOutMode`, M319; broadcast duplicates via `Frame::share`, replaying sticky caps to the late branch). `DynamicFaninHandle::add_input` (M320) attaches a source as a new input of a running aggregator/muxer. An input add is a negotiation, not a grant (M975): the runner reserves the pad, validates the source's caps against the pad constraint, then asks the element via `MultiInputElement::accepts_runtime_input` — its veto for what pad count and caps cannot express (no spare pad of that media kind, a container that cannot carry a second track). The caller holds a `PendingInput` whose `accepted()` resolves to the verdict; a refused input fails alone (`InputRefused`, logged under the `fanin` category) and the run continues on the inputs it has.

### 4.9 GStreamer Dynamic-Feature Mapping
`g2g`'s dynamic surface is intended to be a superset of GStreamer's dynamic capabilities, achieved through a different set of primitives.

| GStreamer feature | `g2g` mechanism |
| :--- | :--- |
| Element hot-swap | `ElementSlot::swap` (ArcSwap) |
| Branch insertion / removal | `BranchSlot::swap` with `SwapPolicy::Immediate` |
| Branch enable / disable, A/B switching | `Router` + `Gate` |
| Bin nesting | `BranchSlot` is structurally a bin |
| Mid-stream caps change | `PipelinePacket::CapsChanged` + runner cascade |
| Allocation pressure backtrack | Phase 3 `ConfigureOutcome::ReFixate` |
| Bitrate switching | `BranchSlot` + `ShadowWarm { cutover: NextSegment }` |
| Codec change at keyframe | `BranchSlot` + `ShadowWarm { cutover: NextKeyframe }` |
| Demuxer dynamic-pad (bounded N) | Pre-allocated dark slots, populated on discovery |
| Live source push from app code | Direct `LinkSender::send` from external task |
| Multi-pipeline isolation | One pipeline per task tree; no shared mutable state |
| Async messages (bus) | Pipeline-level mpmc message channel |
| Latency aggregation query | Upstream-traveling query primitive |
| Allocation query | Downstream-proposed allocator handoff |
| Probes (`pad_block`, `pad_idle`) | `LinkInterceptor` trait registered on a slot |
| Seek with FLUSH | `PipelinePacket::Flush` + runner drain handling |
| Live clock distribution | `AsyncClock` provider election |
| EOS aggregation across N inputs | Fan-in / muxer |

#### 4.9.1 Differences Forced by Rust Ownership
GStreamer relies on parent ↔ child reference cycles via GObject reference counting plus signal callbacks. Rust's strict ownership doesn't allow that shape. Equivalent functionality lives in **message channels** instead of direct back-references: a child element that needs to notify its parent posts a bus message; the parent reads it. Functionally identical; structurally cleaner; no `unref` ordering hazards. Similarly, GStreamer's `gst_pad_link()` performs runtime pointer manipulation; the `g2g` equivalent — moving the receive end of a channel — requires explicit ownership transfer under a brief gate hold. Same outcome, more honest about what's happening.

#### 4.9.2 Capabilities That Fall Out For Free
- **No silent caps mismatch at runtime**: exhaustive typed `Caps` enum, `match` checked at compile time. GStreamer's string-keyed caps regularly fail at runtime with `not-negotiated`.
- **Deterministic shutdown**: Rust drop order is a topological walk; no leaked refs holding pipelines alive forever.
- **No GIL / no global state**: independent pipelines spawn on the same async runtime with zero coordination cost.
- **Memory safety across hot-swap**: ArcSwap guarantees no use-after-free when an element is replaced while a frame is in flight. GStreamer's `pad_block` / `pad_unlink` choreography is famously bug-prone here.

#### 4.9.3 The Single Architectural Trade-Off
Pre-allocated "dark slots" handle the common dynamic-pad case (a demuxer with at-most-N tracks). If an application genuinely needs runtime-growable pad count without an upper bound — e.g., a session router that accepts new RTP streams indefinitely — the dynamic layer uses a `Slab<Slot>` instead of a fixed array. Per-push slot lookup becomes one extra indirection. Since this only matters inside the already-type-erased dynamic layer, the cost is in the noise.

The bounded-N realization is `StreamDemux` (`g2g-plugins`), a `MultiOutputElement` with N typed output ports, driven by `run_source_fanout`. Each port carries its own declared caps and is fed by a caller-supplied classifier (`Fn(&Frame) -> usize`); the first frame routed to a port emits that port's `CapsChanged` so the branch retypes from the demuxer's byte-stream input caps to the elementary stream's, the same announce a single-output demuxer does. The N branch links the runner pre-allocates *are* the dark slots: a port no stream ever routes to simply stays silent and takes the merged EOS at end. This is the multi-output demuxer (one element, several typed downstream branches); the prior fan-out elements (`Router`, `Gate`) only broadcast or A-B-switch a single caps. Container parsers (MPEG-TS multi-PID) wire onto it by keying the classifier on parsed stream identity.

The demux is also a first-class DAG node, the symmetric counterpart to the muxer fan-in. Rather than a new `NodeKind`, a demux reuses `NodeKind::Tee(n)` for the structural/solver view (it negotiates exactly like a tee at startup, per the dark-slot retyping above) and carries a `GraphNodeRef::Demux` payload that the runner dispatches to `demux_arm` (the transpose of `muxer_arm`) instead of the broadcast `tee_arm`. So the solver is unchanged and only the runtime behavior differs. `Graph::add_demux` builds the node; `DynMultiOutputElement` is the dyn-safe mirror of `MultiOutputElement`. In `gst-launch`, a name registered via `register_demux` with several outputs builds a demux (`src ! d.  d. ! …  d. ! …  <demux> name=d`) instead of erroring `FanOutWithoutTee`, the transpose of the muxer's link-degree rule. There is no content-agnostic default demux in the registry: routing is inherently stream-specific (as the muxer side ships specific muxers), so `register_demux` is the surface.

### 4.10 Architectural Tracks

The framework is built along five interlocking tracks. The spec sections that
follow describe each track's current architecture.

| Track | Section | Summary |
| :--- | :--- | :--- |
| Receive | §4.11, §4.12a/b, §4.19 | Network + capture sources and hardware decoders (RTSP, raw RTP ingest with jitter buffer + RTCP/NACK, WebRTC WHEP/sendrecv, V4L2 capture, file, fMP4, software/VAAPI/MF/NVDEC decoders). |
| Display & egress | §4.11.5, §4.12, §4.19 | GPU-resident presentation sinks and outbound RTP packetizers; WebRTC WHIP / sendrecv egress. |
| Negotiation | [DESIGN-caps.md](DESIGN-caps.md) (§4.13) | Distributed CSP caps solver with per-link assignment and structured failure. |
| ML | §5 | Inline GPU tensor preprocess and inference (Burn / ORT). |
| Deployment | §6 | Cloud / embedded / browser orchestration over a single core. |

Open work (planned tracks, deferred items, follow-ups) lives in
[DESIGN_TODO.md](DESIGN_TODO.md).

### 4.11 Hardware Decoder Elements

The layers `RtspSrc → H264Parse` cover encoded-bitstream processing
(mux, re-stream, record). Decoded-pixel output — required for ML inference,
display, and colour-space conversion — uses a decoder `AsyncElement` that
accepts `Caps::CompressedVideo { codec: H264 | H265, .. }` and emits
`Caps::RawVideo { format: Nv12 | I420, .. }` backed by `MemoryDomain::System`,
`MemoryDomain::DmaBuf`, `MemoryDomain::Cuda`, or `MemoryDomain::D3D11Texture`
depending on backend.

#### 4.11.1 cros-codecs (Linux VAAPI)

`VaapiDec<C>` (`g2g-plugins/src/vaapidec.rs`, feature `vaapi`, `cfg(target_os = "linux")`) is built on `cros-codecs` (`vaapi` backend); the `VaapiCodec` binding picks the stateless decoder and NAL splitter, giving the two elements `VaapiH264Dec` and `VaapiH265Dec` (M1036). The crate is maintained by the ChromeOS team and exposes a stateless decoder framework that parses the bitstream and manages the DPB; the actual decode runs on the GPU through libva.

**Status (2026-08): Intel-only candidate, not the AMD path.** cros-codecs allocates output surfaces through ChromeOS GBM extensions (`GBM_BO_USE_HW_VIDEO_DECODER`, contiguous NV12) that Mesa `radeonsi` does not provide, so the element cannot start on AMD desktop GPUs, and the path is not being revived. The Linux hardware-decode ranking is `VulkanVideoDec` (vendor-neutral, picked by the domain-aware search for GPU-domain consumers) with ffmpeg's VAAPI hwaccel (`Backend::Vaapi`, 4.11.3) as the hardware route into system memory.

- **Input caps:** `Caps::CompressedVideo { codec: C::CODEC, .. }` — `intercept_caps` intersects with the element's codec and rejects everything else.
- **Output caps:** `Caps::RawVideo { format: Nv12, .. }` backed by `MemoryDomain::System` (CPU copy out of the GBM-allocated surface).
- **Frame allocation:** `GbmDevice::open("/dev/dri/renderD128")` (configurable via `VaapiH264Dec::with_render_node`) allocates `GenericDmaVideoFrame` surfaces; the decoder's allocator callback returns one per output picture.
- **Format negotiation:** the first `decode()` call surfaces `DecodeError::CheckEvents`; the element drains events, picks up the SPS-derived `StreamInfo` on `FormatChanged`, and re-feeds the same NAL.
- **Flush:** forwards `decoder.flush()` and propagates `PipelinePacket::Flush` downstream.
- **EOS:** flushes the decoder, drains the DPB, emits `Eos`.
- **Thread safety:** `libva::Display` is `Rc<Display>` and therefore `!Send`; `unsafe impl Send` is justified by the runner's ownership model (move-not-share).

```text
H.264 Annex-B  (MemoryDomain::System)
       │
       ▼
┌───────────────────────────────┐
│  VaapiH264Dec                 │
│   cros-codecs StatelessDecoder│
│   <H264, VaapiBackend<...>>   │
│   DPB + B-frame reorder       │
└───────────┬───────────────────┘
            │  NV12 row-copied out of GBM surface
            ▼
    downstream AsyncElement
```

#### 4.11.2 Windows Media Foundation Transform (MFT)

`MfDecode` (`g2g-plugins/src/mfdecode.rs`, feature `mf-decode`, `cfg(target_os = "windows")`) wraps `CLSID_MSH264DecoderMFT` via `windows-rs` using an MTA COM apartment.

- **Input caps:** `Caps::CompressedVideo { codec: H264, .. }` — rejects anything else at `intercept_caps`.
- **Output caps:** `Caps::RawVideo { format: Nv12, .. }` backed by `MemoryDomain::System` (CPU copy out of the MFT output buffer).
- **Flush:** forwards `MFT_MESSAGE_COMMAND_FLUSH` and propagates `PipelinePacket::Flush` downstream.
- **EOS:** sends `MFT_MESSAGE_COMMAND_DRAIN` to flush the B-frame reorder buffer before emitting `Eos`.
- **Thread safety:** `!Send` by default (COM); `unsafe impl Send` justified by MTA free-threading — the MS H.264 decoder MFT is callable from any MTA thread without marshaling.

A sibling `MfEncode` (feature `mf-encode`) wraps `CLSID_MSH264EncoderMFT` with `MF_LOW_LATENCY` set (no B-frames) and converts `Caps::RawVideo { format: Nv12 }` to `Caps::CompressedVideo { codec: H264 }`, Annex-B framed. `MfAacEncode` / `MfAacDecode` (feature `mf-aac`) cover the AAC audio path.

#### 4.11.3 ffmpeg / libavcodec

`FfmpegH264Dec` (`g2g-plugins/src/ffmpegdec.rs`, feature `ffmpeg`, `cfg(target_os = "linux")`) wraps system libavcodec via `ffmpeg-next`. Selectable backend:

| `Backend` variant | Codec opened | Output domain | Notes |
| :--- | :--- | :--- | :--- |
| `Software` | `h264` | `System` | Software decode; broadest hardware coverage. |
| `NvdecCuvid` | `h264_cuvid` | `System` | GPU decode, host copy. Pairs with CPU sinks. |
| `NvdecCuda` | `h264` + `AV_HWDEVICE_TYPE_CUDA` | `Cuda` | Zero-copy device-memory output; see §4.11.5. |
| `Vaapi` | `h264` + `AV_HWDEVICE_TYPE_VAAPI` | `System` | GPU decode, surface downloaded to system memory (`av_hwframe_transfer_data`). The Linux AMD / Intel hardware path; works on Mesa `radeonsi` where cros-codecs `VaapiH264Dec` cannot. Pin the render node with `with_vaapi_device` (or the `device` property; launch name `ffmpegvaapidec`). |

- **Input caps:** `Caps::CompressedVideo { codec: H264, .. }`.
- **Output caps:** `Caps::RawVideo`, layout chosen by `with_output_format` / the `output-format` property (all also pad-template alternatives, so a downstream that pins one auto-plugs a decoder built for it). `I420` (the default, libavcodec's native 8-bit 4:2:0) and `Nv12` (a U/V interleave, no swscale); `I422` / `I444` preserve a High 4:2:2 / 4:4:4 source's chroma, and a 4:4:4 source feeding an `I420` / `Nv12` request is box-averaged down. 10-/12-bit sources (High 10 / Main10) keep their depth: `I420p10` .. `I444p12` are a lossless 2-byte-per-sample plane copy, and `P010` is the semi-planar 10-bit layout (NV12's shape, value in each 16-bit word's top bits) that 10-bit samplers and overlay planes take, packed from a planar 10-bit software decode or verbatim from a `P010LE` hardware frame. `OutputFormat::Auto` emits the source's own chroma and depth, advertising the whole set at negotiation and fixing the concrete format per frame via `CapsChanged`. `YUVJ*P` is accepted with the same plane layout as its studio-range sibling. Mismatches that would need a real conversion (chroma upsampling, an 8-bit request from a 10-bit source) are rejected with `CapsMismatch`, not silently converted: put a `videoconvert` downstream, which takes the planar 10-/12-bit family. Validated bit-exact against ffmpeg's own raw decode of a High 10 clip for both `I420p10` and `P010`.
- **Feed loop:** one access unit per `Packet::copy`; PTS is forwarded verbatim (libavcodec echoes it back on the decoded frame); `send_packet()` then `receive_frame()` drained until `EAGAIN`.
- **Flush / EOS:** `decoder.flush()` on `PipelinePacket::Flush`; `send_eof()` + final drain before forwarding `Eos`.
- **Thread safety:** `ffmpeg::decoder::Video` wraps a raw `*mut AVCodecContext` and is `!Send` by default; `unsafe impl Send` is justified by the same ownership-transfer argument as `MfDecode` and `VaapiH264Dec`.

`FfmpegH264Enc` (`g2g-plugins/src/ffmpegenc.rs`, feature `ffmpeg`, `cfg(target_os = "linux")`) is the encode-side mirror: `Caps::RawVideo { format: I420, .. }` in, `Caps::CompressedVideo { codec: H264, .. }` Annex-B out, via `ffmpeg-next`. It gives the Linux production path a hardware H.264 encoder, the codec `WebRtcSink` / `RtpH264Packetizer` / the RTSP server require (the other Linux encoders are AV1 / VP8/9 / MJPEG, none of which those H.264-only sinks accept). Selectable backend:

| `Backend` variant | Encoder opened | Notes |
| :--- | :--- | :--- |
| `Nvenc` (default) | `h264_nvenc` | NVIDIA NVENC; hardware, realtime. The server-side render-and-stream path wants this. Fails loud at configure if absent (no driver / libavcodec built without it). |
| `Software` | `libx264` | Portable CPU fallback (CI / no-GPU hosts), present only if libavcodec was built `--enable-libx264`. |

- **Low latency:** `max_b_frames = 0` (output in presentation order, no reorder hold), in-band SPS/PPS (the `GLOBAL_HEADER` flag is *not* set, so parameter sets ride each IDR, the Annex-B stream a network sink expects), and a per-backend low-latency preset/tune (`p4`/`ll`/CBR/`delay=0` for NVENC, `veryfast`/`zerolatency` for libx264). A downstream PLI (`Reconfigure::ForceKeyframe`) forces an IDR on the next frame via `pict_type`.
- **PTS:** the input frame's nanosecond PTS is mapped through the encoder's frame-index PTS (`time_base = 1/fps`) and recovered on the output packet, surviving any reorder.
- **Validation:** a round-trip test on the RTX 3060 encodes I420 through `Nvenc` (and `Software`) and decodes the result back through `FfmpegVideoDec`, asserting Annex-B framing and that the stream decodes to I420 at the original geometry. Like the decoder, the `ffmpeg` feature is CI-excluded (libav version-sensitivity), so this is validated on libav hosts.

`NvEnc` (`g2g-plugins/src/nvenc.rs`, feature `nvenc` which implies `cuda`, `cfg(target_os = "linux")`) is the **zero-copy, device-resident** H.264 encoder: the device-resident version of the ffmpeg `Nvenc` backend. The ffmpeg encoder takes *system-memory* I420 and copies it into libavcodec; `NvEnc` ingests an NVDEC/CUDA NV12 surface (`MemoryDomain::Cuda`) **in place** and drives the NVIDIA Video Codec SDK (`nvEncodeAPI`) directly, so the pixels never leave the GPU. It closes the native `FfmpegH264Dec(NvdecCuda) -> NvEnc` loop with no PCIe download, the encode-side mirror of the §5.1 `CudaToWgpu` import bridge, and is the egress half of the server-side render-and-stream path fed by the wgpu->CUDA hand-off.

- **Caps:** `Caps::RawVideo { format: Nv12 | Rgba8 | Bgra8, .. }` in, `Caps::CompressedVideo { codec: H264, .. }` Annex-B out (a native `DerivedOutput`, same dims / framerate). Caps do not encode the memory domain, so negotiation is identical to a system encoder; at runtime the frame must be `MemoryDomain::Cuda` (`UnsupportedDomain` otherwise, the symmetric contract `FfmpegH264Enc` upholds for `System`). NV12 input (the NVDEC hwframe domain) must be a contiguous surface (chroma at `luma_ptr + luma_pitch * height`, one base pointer + pitch); RGBA input (the GPU-render domain, e.g. via `WgpuToCuda`) is a single packed plane at `luma_ptr` with `luma_pitch = width * 4`, registered as NVENC `ABGR` (wgpu `Rgba8` byte order) / `ARGB` with NVENC doing the colour conversion to H.264 internally.
- **Bindings: hand-rolled FFI.** Like the `cuda` module (`g2g-plugins/src/cuda.rs`), `cudarc` is not used; the element links `libnvidia-encode` + `libcuda` directly. The SDK's giant version-tagged structs are transcribed `#[repr(C)]` with **compile-time size assertions** (`const _: () = assert!(size_of::<T>() == N)`) checked against the installed `nvEncodeAPI.h` (SDK 13.0; field offsets verified with `offsetof`), so a mismatched SDK fails the build rather than corrupting the wire layout. The one field-heavy codec-config union is left opaque (a correctly-sized `[u32; N]`): the driver fills it via `nvEncGetEncodePresetConfigEx`, and we overwrite only rate control / GOP.
- **Lifecycle:** the encode session opens lazily on the first frame, on that frame's `CUcontext` (the NVDEC source's context). Per frame: `nvEncRegisterResource` (`CUDADEVICEPTR`, NV12) -> `nvEncMapInputResource` -> `nvEncEncodePicture` -> `nvEncLockBitstream` (copy out Annex-B) -> unlock / unmap / unregister.
- **Low latency:** preset P4 + the LOW_LATENCY tuning info, CBR, no B-frames (`frameIntervalP = 1`), and an *infinite GOP* (`NVENC_INFINITE_GOPLENGTH`) so IDRs are emitted only on demand: the first frame, and on a downstream PLI (`Reconfigure::ForceKeyframe`). Each forced IDR sets `OUTPUT_SPSPPS` so in-band parameter sets ride it (the Annex-B a network sink expects). The NV12 nanosecond PTS round-trips through NVENC's `inputTimeStamp`.
- **Validation:** an on-hardware round-trip on the RTX 3060 synthesizes a CUDA-resident NV12 surface (CUDA driver alloc + upload), encodes through `NvEnc`, and decodes the Annex-B back through `FfmpegVideoDec` to the original geometry; it skips cleanly with no NVIDIA GPU. The `nvenc` feature is CI-excluded (no NVENC runtime in CI). **HEVC (H.265)** is supported alongside H.264: `with_codec(VideoCodec::H265)` / the `codec` property switches the encode GUID to `NV_ENC_CODEC_HEVC_GUID` and the output caps to `CompressedVideo{H265}`, the path otherwise identical (the round-trip test covers both). `NvEnc` declares `input_domains = {Cuda}`, so a CPU-side NV12 source feeding it gets a `CudaUpload` spliced in automatically by the converter auto-plug (§4.13.5); the encoder itself stays Cuda-only. The output-bitstream-buffer pool and runtime bitrate retarget are in place. **10-bit** encode: P010 input maps to the 10-bit buffer format and the HEVC Main10 profile (P010 with `codec=h264` is rejected, NVENC has no 10-bit H.264). `gop-size` (-1 = infinite, the low-latency default) and `repeat-sequence-header` write `gopLength` / `idrPeriod` / `repeatSPSPPS`, re-applied live through `nvEncReconfigureEncoder`. The matching native `NvDec` is the other half of the gst-`nvcodec`-style pair.
- **Thread safety:** the session is a raw NVENC handle + CUDA context driven through `&mut self` only; `unsafe impl Send` rests on the same ownership-transfer argument as `FfmpegH264Enc`.

`NvDec` (`g2g-plugins/src/nvdec.rs`, feature `nvdec` which implies `cuda`, `cfg(target_os = "linux")`) is the **decode half of the gst-`nvcodec`-style pair**, the mirror of `NvEnc`. It promotes NVIDIA hardware decode from the `FfmpegH264Dec` `Backend::NvdecCuda` flag (which reaches NVDEC *through* libavcodec's cuvid hwaccel) to a first-class element driving the NVCUVID parser+decoder API directly. With `NvDec -> ... -> NvEnc` both native, the whole H.264 transcode loop stays on the GPU and out of libavcodec.

- **Caps:** `Caps::CompressedVideo { codec: H264, .. }` Annex-B in, `Caps::RawVideo { format: Nv12, .. }` out (a native `DerivedOutput`). The runtime `CapsChanged` carries the actual cropped display geometry the bitstream declares.
- **Multi-domain output.** `NvDec` advertises `output_domains = {Cuda, System}` and, in `configure_allocation`, reconciles the negotiated proposal against that capability (`resolve_for_producer`, §4.13.5): a CUDA-capable consumer keeps each surface device-resident (zero-copy, the default `MemoryDomain::Cuda`); a System-only consumer makes the decoder download (reusing `cuda::download_nv12`) before emitting. The same decoder stays on the GPU or downloads, chosen by downstream demand alone, validated on the RTX 3060.
- **Callback model:** NVCUVID is callback-driven. A parser (`cuvidCreateVideoParser`) is fed the elementary stream and synchronously invokes three callbacks from inside `cuvidParseVideoData`: a *sequence* callback (creates the `CUvideodecoder` once the SPS geometry is known), a *decode* callback (`cuvidDecodePicture`), and a *display* callback (a frame is ready in display order). The display callback cannot `await`, so it maps the surface (`cuvidMapVideoFrame64`) and pushes a ready frame onto a queue that `process` drains and emits after the parse returns. The callbacks reach element state through a `*mut DecoderState` passed as the parser user-data; that pointer targets a heap `Box` so it survives the runner moving the element between worker threads.
- **Bindings: hand-rolled FFI.** Links `libnvcuvid` + `libcuda` directly (no `cudarc`). NVCUVID exports real symbols (no `CreateInstance` dispatch table, unlike NVENC), so the calls are plain `extern "C"`; the structs are transcribed `#[repr(C)]` with compile-time size assertions against the installed `cuviddec.h` / `nvcuvid.h`, and the per-picture `CUVIDPICPARAMS` is opaque (the parser fills it, we pass the pointer straight to `cuvidDecodePicture`).
- **Frame lifetime:** each output frame carries a `CudaKeepAlive` that `cuvidUnmapVideoFrame64`s on drop plus an `Arc` to the decoder, so the decoder and its CUDA context outlive any frame still in flight; the decoder, context lock, and context are destroyed (in that order) only once the last frame is released. The element owns its own CUDA context (created at configure).
- **Validation:** an on-hardware test on the RTX 3060 runs the full native loop, a synthesized CUDA NV12 surface encoded by `NvEnc` to Annex-B and decoded by `NvDec` back to CUDA NV12, asserting geometry and (via a small device->host copy) that the decoded luma holds real content; it skips with no NVIDIA GPU. The `nvdec` feature is CI-excluded. **HEVC (H.265) and AV1** are supported alongside H.264: the input caps accept `CompressedVideo{H264|H265|Av1}`, the codec is inferred and mapped to the `cudaVideoCodec` the NVCUVID parser + decoder are created for. A 10-bit stream decodes to a `P016` surface announced as `RawVideoFormat::P010`; a mid-stream resolution change reconfigures the live decoder in place (`cuvidReconfigureDecoder`) when the new size fits, else rebuilds it (the CUDA context rides a separate `Arc` so in-flight frames survive the rebuild). The display delay defaults to a low-latency 1, settable via `max-display-delay` (0..=16).

#### 4.11.4 End-to-End RTSP Pipeline

The complete glass-to-glass receive pipeline is:

```
RtspSrc ──► H264Parse ──► [decoder] ──► [ML / display / encode]
(System / H264)            (System / DmaBuf / Cuda / D3D11Texture; NV12)
```

| Platform | Decoder element | Feature | Output |
| :--- | :--- | :--- | :--- |
| Linux software | `FfmpegH264Dec` (`Software`) | `ffmpeg` | `System` / I420 |
| Linux + NVIDIA | `FfmpegH264Dec` (`NvdecCuvid` / `NvdecCuda`) | `ffmpeg` + `cuda` | `System` / `Cuda` / NV12 |
| Linux + VAAPI | `VaapiH264Dec` / `VaapiH265Dec` | `vaapi` | `System` / NV12 |
| Windows | `MfDecode` | `mf-decode` | `System` / NV12 |

`RtspSrc` connects via `retina` using standard RTSP/RTP over TCP, negotiates H.264 with `FrameFormat::SIMPLE` (Annex-B) or accepts AVCC framing detected per buffer. The first SPS the parser sees provides geometry; framerate is recovered from the VUI `timing_info` (`time_scale / (2 * num_units_in_tick)`) when present, or left as `Rate::Any` when the VUI is absent. `RtspSrc::with_credentials` supplies the DESCRIBE/SETUP account (threaded into retina's `SessionOptions`).

`OnvifSrc` (`onvif` feature) is the ONVIF *control plane* in front of `RtspSrc`. An ONVIF camera does not stream over ONVIF; its SOAP services tell you the RTSP URL. `discover` sends one WS-Discovery `Probe` to the `239.255.255.250:3702` multicast group and collects each camera's device-service URL from the `ProbeMatch` `XAddrs`; `resolve_stream_uri` then runs `GetCapabilities` → `GetProfiles` → `GetStreamUri`, authenticated with a WS-Security `UsernameToken` digest (`Base64(SHA1(nonce ++ created ++ password))`). The element resolves the RTSP URI lazily during negotiation (`intercept_caps`), builds an inner `RtspSrc` once (forwarding the same credentials, since cameras gate the media stream behind the device account), and delegates the rest of the `SourceLoop` to it. The SOAP layer is hand-rolled (fixed request templates + `roxmltree` response reads) to avoid the git-only `onvif`/`schema` crate tree; the footprint is reqwest + roxmltree + sha1 + base64 + getrandom. Scope is discovery + stream-URI resolution; PTZ and event subscriptions are not implemented.

#### 4.11.5 Zero-copy NVDEC → CUDA → GPU display

`Backend::NvdecCuvid` decodes on the GPU but copies NV12 back to system memory;
the glass-to-glass floor is then dominated by the PCIe round-trip plus the
sink's CPU NV12→XRGB convert. The CUDA-resident path keeps decoded NV12 in
device memory end-to-end so a GPU consumer (display) takes the handoff without
a host round-trip.

**Memory domain.** `MemoryDomain::Cuda(OwnedCudaBuffer)` lives in `g2g-core`,
platform-agnostic. `OwnedCudaBuffer` carries the two NV12 plane device
pointers (luma Y, interleaved chroma UV), row pitches, dims, the `CUcontext`,
and a boxed `CudaKeepAlive` owner. Core never links CUDA: the producing
element supplies the owner as a trait object, and dropping the buffer releases
the backing allocation. `AllocationParams::cuda(...)` makes
`MemoryDomainKind::Cuda` a cross-element pool domain in the allocation
negotiation (§4.13).

**Decoder.** `Backend::NvdecCuda` opens the generic `h264` codec with an
`AV_HWDEVICE_TYPE_CUDA` device and a `get_format` hook selecting
`AV_PIX_FMT_CUDA`; the resulting `AVFrame` is the keep-alive that owns the
device pointers wrapped into `OwnedCudaBuffer`.

**Consumer: CUDA↔GL interop, not dma-buf.** CUDA can only export VMM-allocated
memory (`cuMemCreate` / `cuMemMap`) to a dma-buf fd, and NVDEC decoder frames
come from libavcodec's CUDA hwframe pool (not VMM); the NVIDIA proprietary
driver also doesn't import foreign dma-bufs reliably through `nvidia-drm`.
Presentation therefore uses CUDA↔GL interop — the path GStreamer's `nvcodec`
+ `glimagesink` and NVIDIA's `FramePresenterGL` sample take:

1. Create an EGL context on the display surface.
2. Register a GL texture with `cuGraphicsGLRegisterImage` once.
3. Per frame: `cuGraphicsMapResources`, `cudaMemcpy2D` (device→device,
   honouring source pitch) the NV12 planes into the GL resource,
   `cuGraphicsUnmapResources`.
4. Sample Y + interleaved UV in a fragment shader (BT.601/709 limited range),
   present via `eglSwapBuffers`.

This is not strictly zero-copy (one device→device copy into the GL texture)
but it removes the PCIe round-trip and the CPU colour convert.

**Elements.**
- `CudaDownload` (`cuda` feature) is an `Identity(NV12)` transform that
  copies a `MemoryDomain::Cuda` frame to `MemoryDomain::System` via
  device→host `cuMemcpy2D`. Negates the latency win but lets a `NvdecCuda`
  stream reach the existing CPU sinks for correctness and bring-up.
- `CudaGlSink` (`cuda-gl` feature, Linux + NVIDIA) holds an EGL context on a
  Wayland surface (`wl_egl_window` from SCTK), a `glow` GL ES 3 program with
  the two NV12 textures, and the per-frame map/copy/unmap render loop via
  the CUDA-GL interop entry points. Validated on an RTX 3060:
  ~10.7x lower present latency than `NvdecCuvid -> WaylandSink` at 1080p.
- `CudaKmsSink` (`cuda-kms` feature, Linux + NVIDIA) is the tty /
  no-compositor counterpart: the same CUDA-GL interop + NV12->RGB shader (shared
  via the `glnv12` module), but EGL renders into a GBM surface scanned out via
  DRM page-flips instead of a Wayland surface. Needs DRM master (a bare VT or a
  DRM lease). The shared render half is the validated `CudaGlSink` path.

**CUDA bindings: hand-rolled FFI.** `cudarc` has no CUDA-GL interop wrappers
(`cuGraphicsGLRegisterImage` and friends), and its safe API assumes it owns
the `CudaContext`, whereas the `CUcontext` is created and owned by ffmpeg's
hwdevice and carried on `OwnedCudaBuffer`. The needed surface is small:
`cuCtxPushCurrent_v2` / `_PopCurrent_v2`, `cuMemcpy2D_v2`, and the GL-interop
quartet `cuGraphicsGLRegisterImage` / `cuGraphicsMapResources` /
`cuGraphicsSubResourceGetMappedArray` / `cuGraphicsUnmapResources`. The
plugin links `libcuda` directly.

#### 4.11.6 Vulkan Video (vendor-neutral GPU-resident decode)

The NVDEC→CUDA→wgpu path of §4.11.5 is fast but **vendor-locked**: CUDA has no
AMD or Intel analog, so a wgpu-based consumer (a game engine, a visualization
viewer) that wants hardware decode straight into its own render device gets it
only on NVIDIA. `VulkanVideoDec`
closes that gap by decoding with `VK_KHR_video_queue` + `VK_KHR_video_decode_*`
on the **same Vulkan device wgpu already runs**, so the decoded `VkImage` is
imported as a `wgpu::Texture` with no download and no second interop bridge. One
element then covers AMD (RADV), NVIDIA and Intel (ANV), each validated as
hardware is available.

The **capability probe** (`vulkanvideo::probe_decode_caps`, validated on the RTX
3060) reaches the adapter's raw
`ash` handles via `as_hal::<Vulkan>()`, finds a decode-capable queue family, and
queries `vkGetPhysicalDeviceVideoCapabilitiesKHR` for H.264/H.265/AV1, returning
the coded-extent range, DPB slot / active-reference budget, and the
`DPB_AND_OUTPUT_COINCIDE` flag that `intercept_caps` and DPB sizing negotiate
against (on the 3060: H.264 to 4096², H.265/AV1 to 8192², output coincides with
the DPB). It settled the load-bearing driver wrinkle: the query returns a
generic `ERROR_INITIALIZATION_FAILED` unless the codec-specific output caps
struct (`VkVideoDecodeH264/H265/AV1CapabilitiesKHR`) is chained alongside
`VkVideoDecodeCapabilitiesKHR`, with a `VkVideoDecodeUsageInfoKHR` on the profile.

The element is deliberately mostly reuse: the `VkImage`→`wgpu::Texture` import
(`cudawgpu.rs` / `dmabufwgpu.rs` `texture_from_raw` + `TextureMemory::External`,
§5.1), custom Vulkan device creation with extra extensions (the `cuda-wgpu`
device path), the multiplanar NV12→RGBA `VkSamplerYcbcrConversion` compute pass
(shared with the Android `mediacodec-wgpu` decoder), the Annex-B + SPS/PPS
front-end (`h264parse` / h265parse), and the allocation-domain auto-plug
(§4.13.5) all already exist. The new surface is the decode session itself: a
`VkDevice` with a `VK_QUEUE_VIDEO_DECODE_BIT_KHR` queue adopted into wgpu via
`create_from_hal` (wgpu will not request a decode queue on its own, the
load-bearing integration point), a `vkGetPhysicalDeviceVideoCapabilitiesKHR`
probe feeding `intercept_caps`, a `VkVideoSessionKHR` /
`VkVideoSessionParametersKHR` whose `Std*` parameter structs are populated from
the parsed SPS/PPS/VPS (the correctness-critical part, one mapping module per
codec, re-emitted on mid-stream change via `CapsChanged`), DPB reference-slot
management, and the `vkCmdDecodeVideoKHR` recording, output pipelined through the
YCbCr pass with an in-flight ring. A session's `maxCodedExtent` is the device's
maximum, not the stream's geometry (M1027): it is only an upper bound, each
picture resource carries its real extent, and sizing the session to the picture
made the NVIDIA driver refuse whole small geometries with
`ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR`. The session + DPB rebuild mid-stream on *any*
in-band parameter-set change, keyed by a byte fingerprint of the AU's parameter
sets (M519 geometry / M764 same-geometry content, e.g. a profile or entropy-mode
switch; byte-identical keyframe re-sends keep the session), flushing the outgoing
decoder's pipelined tail first so no frame is lost. Output caps are
`Caps::RawVideo { format: Rgba8, .. }` in `MemoryDomain::WgpuTexture`
(optionally `VulkanTexture` / multiplanar NV12); negotiation and the frame
keep-alive follow the `NvDec` multi-domain pattern (§4.11.3).

**H.264 (validated on the RTX 3060).** The decode path is complete: session +
`Std*` SPS/PPS mapping, IDR then full-DPB P-frame decode bit-exact vs the ffmpeg
software decoder, the zero-copy `VkSamplerYcbcrConversion` NV12->RGBA import into
a `wgpu::Texture`, the `VulkanVideoDec` streaming element and its `WgpuSink`
present, and `produces(WgpuTexture)` auto-plug. Two consumption models sit on the
same `H264DpbDecoder`: the **streaming** `VulkanVideoDec` (push, `AsyncElement`)
for pipelines, and `VulkanVideoPlayer`, a **random-access "pull"** frame server
(`frame_at(pts)` / `frame_at_index`) that indexes GOPs / POC (`index_pictures`),
`reset`s and decodes forward from the enclosing random-access point on a seek
(`decode_range_to_texture`), and caches decoded textures keyed by decoding index.
The player drives H.264 **and** H.265 (the codec is sniffed from the stream,
`sniff_annexb_codec`, behind a `PlayerDecoder` enum), and its seek point is the
nearest IRAP (an IDR for H.264; an IDR / CRA / BLA for H.265, `PictureMeta::
is_random_access`), so scrubbing into a late open-GOP GOP tunes in at that GOP's
CRA (M587 discards its RASL) rather than decoding from the leading IDR. A leading
picture (a CRA's RASL / RADL, whose POC precedes its CRA) instead seeks from the
random-access point before that CRA and decodes continuously through it, so its
references exist (`m588`).
The pull model is the timeline-scrubber a wgpu visualization viewer (whose native
decode is typically CPU software plus a GPU upload copy) needs; the
`vulkan_video_scrubber` example drives it interactively. The player forward-
continues (a forward seek within reach keeps decoding rather than re-decoding
from the keyframe, so linear playback is O(n) coded pictures, not O(n^2)) and
LRU-bounds its decoded-frame cache; display order is by (GOP, POC) since POC
resets at each IDR. The cache is bounded by both a frame count and a byte budget
(the bound that matters at 4K/8K where a count alone pins gigabytes), and can
optionally cache every traversed picture on a decode range so a backward scrub
within a GOP is free (`set_cache_traversed`). Decoded frames are consumable by an
application-owned wgpu render pipeline, not just `WgpuSink`: the imported RGBA
texture carries `TEXTURE_BINDING`, so a foreign pipeline on the shared decode
device samples it zero-copy (`m500_vulkan_video_embed` + the
`vulkan_video_engine_embed` example), the integration primitive a Bevy
viewer-renderer consumer builds on.

**H.265.** The HEVC parameter-set parse + `Std*` mapping and the
decode session are in place (the H.264 analog). `parse_h265_vps/sps/pps`
+ `extract_h265_parameter_sets` read the RBSP (two-byte NAL header, so RBSP at
`nal[2..]`), including the full `profile_tier_level` and the short-term
reference-picture sets, parsed to canonical explicit form (an inter-RPS-predicted
set is derived per H.265 7.4.8). `to_std_h265_params` maps them onto the
`StdVideoH265*` layout, returning a `StdH265Params` bundle that owns the pointee
blocks the SPS/VPS reference by pointer (profile-tier-level, DPB manager,
short-term RPS list). `create_h265_session` (via `open_h265_decode_device`) builds
the `VkVideoSession` + parameters, driver-validated on the RTX 3060 (`m502`).
`H265DpbDecoder` then decodes pixels: per-picture slice-segment-header
parse (`parse_h265_slice_header`), picture-order-count (8.3.1),
reference-picture-set DPB management (every IRAP a clean reset), and the
reference-slot lists (`RefPicSetStCurrBefore/After`, POC-keyed reference info)
handed to `vkCmdDecodeVideoKHR`, reusing the H.264 DPB machinery. The whole
fixture (IDR + CRA GOPs) decodes bit-exact vs the ffmpeg software decoder on the
3060 (`m503`), also straight to GPU-resident RGBA `wgpu::Texture`s. A hardware
gotcha: NVIDIA's Vulkan HEVC slice-header parser needs a 3-byte start code (`00
00 01`); a 4-byte one breaks every non-IDR slice (the IDR tolerates it via the
picture info), so the H.265 path frames slices with 3 bytes while H.264 keeps 4.

**Long-term reference pictures (M743).** The SPS long-term table rides
`pLongTermRefPicsSps`; the slice header's long-term entries (SPS-indexed and
slice-coded, with the accumulated `DeltaPocMsbCycleLt` per 7.4.7.1) resolve
against the DPB by full POC (MSB cycle present) or POC lsb alone, the RPS prune
keeps long-term-listed pictures, `RefPicSetLtCurr` carries the used-by-current
slots, and each reference's `Std*` info flags its short/long-term marking (which
changes the driver's MV scaling, so a wrong marking corrupts prediction silently
rather than erroring). Landing this exposed two latent slice-RPS bugs: an inline
`st_ref_pic_set` coded in a slice header *does* carry `delta_idx_minus1` when
inter-RPS-predicted (the SPS-context parse never read it, desyncing every later
field), and `NumDeltaPocsOfRefRpsIdx` must be the referenced set's delta count,
not 0, for the driver's own slice-header re-parse. All 500 frames of the JCT-VC
`LTRPSPS_A_Qualcomm_1` conformance vector decode bit-exact vs ffmpeg on the 3060
(`m743`); the GPU-texture path shares the same DPB machinery.

**Reference marking.** Which decoded pictures stay available as references is the
stream's decision, not the decoder's: `dec_ref_pic_marking()` in each slice header
either leaves it to the default sliding window (evict the smallest `FrameNumWrap`)
or names the pictures to retire, which is what x264 does for its B-pyramid.
Reading the marking means walking past the reference-list modification and the
prediction weight table first, so `poc::parse_h264_slice_marking` continues the
shared slice parse through them and returns the operations
(`H264RefPicMarking`, fixed-capacity so the header stays `Copy`); the DPB applies
the short-term ones and refuses a long-term operation rather than keep feeding the
driver a reference the stream has retired. Running the sliding window regardless,
as the decoder did before, diverges from the reference set the driver builds its
L0 / L1 lists against.

**B-frames and display order.** The hardware decode handles B-frames directly: the
driver builds the L0 / L1 reference lists from the DPB and the per-picture POC the
decoder supplies (H.264 supplies every DPB slot's FrameNum/POC; H.265 the
`RefPicSetStCurrBefore/After` split by POC sign), so a bidirectionally-predicted
frame reconstructs bit-exact. What B-frames change is *order*: a frame is coded
after pictures that precede it on screen, so coding order differs from display
(presentation) order. The whole-stream `decode_all` / `decode_all_to_textures`
index the stream's POCs (`index_pictures`) and reorder the coding-order output into
display order via `reorder_to_display_order`, keyed by (coded-video-sequence, POC)
so POC resets at each keyframe group correctly; for an I/P stream this is the
identity. The low-level streaming `decode_push` stays in coding order (a low-latency
consumer such as the streaming (`streamdec`) adapter reorders by PTS itself), but the g2g-native
`VulkanVideoDec` element does reorder its system (NV12) path: `decode_push_meta`
returns one `PictureMeta` per submitted picture (the POC the decode already
computed, no second pass), and the element feeds retired frames through a small
`ReorderBuffer` keyed by the same (coded-video-sequence, POC). The GPU-texture
path streams the same way (M744): `decode_push_to_textures` decodes each AU's
pictures to textures with the DPB/POC state intact across calls (the whole-stream
`decode_all_to_textures` indexing pass resets it, so it cannot stream), and the
element reorders them through a texture `ReorderBuffer`. AV1 needs neither
buffer: its display order is the bitstream's op order, so the element op-walks
each temporal unit (`decode_display` / `decode_display_to_textures`, the DPB
persisting across calls), which also makes `show_existing_frame` re-displays and
per-frame film-grain synthesis work when streamed (the old pipelined path
silently skipped both). M744 also fixed a latent AV1 use-after-free: NVIDIA's
driver retains the `pStdSequenceHeader` / `pColorConfig` pointers handed to
`vkCreateVideoSessionParametersKHR` and dereferences them per decode, so
`Av1DecodeSession` now owns a stable boxed copy for its lifetime (dropping the
Std block after creation yielded small, nondeterministic pixel corruption). It releases the whole
previous coded video sequence at each keyframe (where POC resets) and bumps the
lowest-POC held frame once a sequence exceeds the stream's own declared reorder
depth (H.264 VUI `bitstream_restriction` `max_num_reorder_frames` / H.265
`sps_max_num_reorder_pics`, M764; the DPB slot count is the fallback bound when
the stream declares none), so an I/P stream emits without hold and a long GOP
does not buffer unbounded; `Eos` and a reconfig
boundary drain it in display order, a `Flush` (seek) discards it (M586, H.264 /
H.265). AV1 stays in coding order there (its display order comes from
`show_existing_frame` / `order_hint`, handled whole-stream by `decode_all`). Verified
bit-exact vs the software decoder's display-order output for H.264 and closed-GOP
H.265 B-frame clips on the 3060 (`m569`), and the element's AU-by-AU streaming
output matches that display-order oracle byte for byte (`m586`). Full-stream
H.265 open-GOP (CRA anchors with RASL leading pictures that reference pre-CRA
frames) also decodes bit-exact (`m577`): the DPB is flushed only at an IRAP with
`NoRaslOutputFlag == 1` (every IDR / BLA, and a CRA only as the first picture),
so a mid-stream CRA keeps the references its RASL followers use. Mid-stream
random-access tune-in at a CRA is handled too (`m587`): after a `reset` (a seek)
the CRA is the first picture, so `NoRaslOutputFlag == 1` and its RASL leading
pictures (which reference now-absent pre-CRA frames) are discarded rather than
decoded against a flushed DPB - `h265_is_rasl` + a `skip_rasl` flag set from each
IRAP's `NoRaslOutputFlag`, checked before POC derivation so a dropped RASL leaves
no trace. The CRA's trailing pictures and the following GOPs decode bit-exact vs a
full decode. The same flag is 0 in continuous decoding, so full-stream open-GOP is
unchanged. Long-term references are handled too (M743, above).

**Colour space.** Decoded YUV is converted to RGB with the stream's actual colour
space, not a fixed matrix. A `VideoColorSpace` (colour matrix + quantization range)
is resolved at decoder build time from the H.264 / H.265 VUI colour description
(`parse_vui_color`, one helper since the VUI colour prefix is identical in both
codecs) or the AV1 `color_config`, keyed by the CICP `matrix_coefficients`
codepoint (unspecified falls back by resolution, the ffmpeg heuristic). Both the CPU
`nv12_to_rgba` (general Kr/Kb luma-weight formula, studio / full range) and the GPU
`YcbcrConverter` (its `VkSamplerYcbcrConversion` built with the matching
`YCBCR_601/709/2020` model + `ITU_NARROW/FULL` range) apply it, so BT.709 HD and
BT.2020 content get the right matrix instead of BT.601.

**10-bit decode.** HDR is 10-bit, so the decoder is not fixed to 8-bit NV12: the
session derives its bit depth from the SPS and, for a 10-bit HEVC stream, selects
the Main 10 profile and the `G10X6` two-plane 4:2:0 output format (16-bit samples,
value in the top 10 bits); the shared `DpbCore` scales its readback sizing to 2
bytes per sample, and `Nv12Frame::bit_depth` marks the layout. HEVC Main 10 (`m571`)
and AV1 Main 10-bit (`m572`, `av1_profile(bit_depth)` from `color_config.BitDepth`)
both decode bit-exact vs the software decoder on the 3060. The GPU-texture path
carries 10-bit too: the `YcbcrConverter` picks its formats from the decode bit
depth, so a `G10X6` frame samples through a 10-bit `VkSamplerYcbcrConversion` and
stores into an `R16G16B16A16_SFLOAT` image (the `rgba16f` compute shader),
imported as a `Rgba16Float` `wgpu::Texture` (`m573`, matching the CPU reference
under the stream's matrix). The float target preserves the full 10-bit precision
and is where the transfer stage operates.

**HDR transfer (tone mapping).** The fixed-function ycbcr hardware does the matrix
+ range but NOT the transfer function, so an HDR (PQ / HLG) stream reaches the
compute pass as its raw transfer-encoded R'G'B'. `VideoColorSpace` now carries a
`TransferFunction` (PQ = CICP 16, HLG = 18, else SDR) resolved from the stream, and
the `create_*_dpb_decoder_gpu_tonemap` constructors turn on a transfer stage in the
`rgba16f` shader (selected by a push constant): EOTF (PQ ST 2084 / HLG B67) ->
BT.2390 EETF display mapping (maxRGB, source 1000 -> target 100 nits) -> BT.2020 ->
BT.709 gamut -> BT.709 OETF, yielding display-ready SDR (`m574`, GPU output matches
a CPU port of the same pipeline, and the transfer math is unit-pinned to spec
anchors). It is opt-in: the default GPU path stays passthrough (matrix + range
only, the stream's PQ / HLG encoding preserved in the float target for the HDR
swapchain).

**HDR swapchain present** (`vulkanhdrsink`, `hdr-present`). wgpu 29's surface
config has no colour-space knob, so `VulkanHdrSink` owns a raw `VK_KHR_swapchain`
on the decode device's `VkInstance` (the present extensions - `VK_KHR_swapchain`,
and `VK_EXT_hdr_metadata` when advertised - are enabled in `open_decode_device`,
conditionally, so a decode-only GPU is unaffected). It negotiates the best colour
space the surface offers (`HDR10_ST2084` PQ, else `EXTENDED_SRGB_LINEAR` scRGB,
else SDR), presents the passthrough PQ `Rgba16Float` texture by a raw
`vkCmdBlitImage` into the acquired swapchain image (the acquire -> blit -> present
chain is ordered by GPU semaphores - `image_available` + a per-image
`render_finished` - with one in-flight fence waited at the top of the next frame,
so nothing stalls mid-present), and attaches BT.2020 mastering
metadata via `vkSetHdrMetadataEXT` when available. The surface-format / colour-space
selection and metadata construction are unit-tested; the on-screen present is
validated live via `examples/vulkan_video_hdr_on_screen.rs` (HDR is display +
compositor dependent). This completes the HDR track: colour matrix -> 10-bit decode
-> 10-bit GPU texture -> PQ/HLG tone-map -> HDR10 present.

**AV1.** The AV1 parse half is in place, the H.264 / H.265 analog. AV1 is not
NAL / Annex-B framed: `av1_obus` walks the low-overhead OBU stream by its LEB128
size fields (bounds-checked). `parse_av1_sequence_header` reads the sequence
header OBU (operating points, optional timing / decoder-model info, order-hint
config, and the full `color_config`) into an `Av1SequenceHeader`, which
`to_std_av1_seq_header` maps onto `StdVideoAV1SequenceHeader` plus an owned
`StdVideoAV1ColorConfig` block (the Std AV1 color enums are numeric-equal to the
AV1 spec codepoints, so they cast directly). `av1_frame_infos` classifies each
coded frame from its frame-header lead. GPU-free unit tests cover a real libaom
640x480 fixture. The decode session is in place too (validated on the 3060):
`open_av1_decode_device` + `av1_profile` + `create_av1_session`
(`VkVideoDecodeAV1SessionParametersCreateInfoKHR` carrying the Std sequence header,
the H.264 / H.265 analog), whose parameter creation makes the driver validate the
mapping. The full uncompressed frame header parses too
(`parse_av1_frame_header` + all sub-parses, validated field-by-field vs ffmpeg
`trace_headers`), and `Av1DpbDecoder` decodes on the 3060: it maps the
header onto `StdVideoDecodeAV1PictureInfo` + sub-structs (`to_std_av1_picture_info`)
and manages AV1's reference model (a pool of `NUM_REF_FRAMES + 1` physical DPB
images, `ref_frame_idx` -> slot mapping, `refresh_frame_flags` remap, per-tile
offsets, `vkCmdDecodeVideoKHR`). The whole fixture (1 key + 9 inter frames,
including the compound / temporal-MV inter frames) decodes **bit-exact** vs the
ffmpeg software decoder on the 3060 (SAD/px 0 on every frame). Reaching that
needed one non-obvious default: the loop-filter reference deltas from
`setup_past_independence` are `[INTRA=1, LAST/LAST2/LAST3=0, GOLDEN=-1, BWDREF=0,
ALTREF2=-1, ALTREF=-1]`; the ALTREF2 / ALTREF entries are -1, not 0. Defaulting
them to 0 left in-loop deblocking mis-configured for compound blocks referencing
the alt frames, a tiny residual on inter frames past the first (found by diffing
the picture-info sub-structs the driver receives from ffmpeg's Vulkan hwaccel
against ours). Multi-tile frames decode too (`av1_tile_layout` parses the
`OBU_FRAME` tile-group header + the per-tile `TileSizeBytes` size prefixes into
the driver's per-tile offset / size lists; a 2x2 and a 4x4 libaom clip decode
bit-exact on the 3060). Alt-ref (invisible) frames + `show_existing_frame` decode
too (M565): a stream where decode order != display order takes a synchronous
reorder-aware path (`scan_ops` builds the op list; non-shown frames decode into
the DPB without emitting; each `show_existing_frame` emits the referenced stored
slot at its display position), bit-exact on the 3060. Film grain is synthesized on
the decoded NV12 (M566): the 3060 exposes only `DPB_AND_OUTPUT_COINCIDE` for AV1,
so the driver cannot apply grain (that needs a distinct output image), and
`apply_film_grain_nv12` runs the full AV1 grain synthesis (spec 7.18.3, ported from
the re_rav1d scalar reference) on the grain-free hardware reconstruction, bit-exact
vs dav1d (luma + chroma). The GPU-texture path applies the same grain (M568): since
the ycbcr compute pass produces the grain-free reconstruction, `grained_slot_to_texture`
reads the displayed slot back to NV12 (the GPU DPB images carry `TRANSFER_SRC`),
runs `apply_film_grain_nv12`, and uploads the result to the RGBA texture, bit-exact
vs dav1d; grain is output-only, so the read-back leaves the DPB reference untouched
(a grain-free displayed frame stays on the zero-copy GPU convert). Loop restoration
(Wiener / SGR) decodes correctly (M567): `StdVideoAV1LoopRestoration::LoopRestorationSize`
is the `1 + lr_unit_shift` encoding, not the pixel unit size, matching ffmpeg's
Vulkan hwaccel (getting it wrong desynced the whole frame).

**Shared machinery and the system-path decode ring.** All three `*DpbDecoder`s
fold their GPU plumbing onto one codec-agnostic `DpbCore` (device / session / DPB
image pool / readback buffer / command pool, and the `record_decode` barrier +
begin/decode/end recording), so the codec-specific decoders carry only the
`Std*` mapping and reference bookkeeping. `DpbCore` runs two submit paths off that
one recorder. The **texture path** (`decode_all_to_textures`, the player) converts
each decoded slot to an RGBA `wgpu::Texture` through a persistent `YcbcrConverter`
(the ycbcr conversion / sampler / descriptor-set layout / compute pipeline built
once, not rebuilt per picture; its formats are chosen from the decode bit depth,
8-bit `G8_B8R8` -> `Rgba8Unorm` or 10-bit `G10X6` -> `Rgba16Float`) and chains the
decode to its conversion with a
`sem_dc` semaphore: the decode is submitted on the decode queue signalling `sem_dc`
with no fence, and the compute pass on the compute queue waits `sem_dc`, so the
per-picture CPU prep (RGBA image + memory allocation + views + descriptor set)
overlaps the decode's GPU execution and there is no mid-picture fence wait (~1.9x
over the naive per-picture rebuild + double fence wait, ~690 fps at 640x480 on the
3060). It is *not* pipelined across pictures: because the conversion transitions
the DPB slot in place and the next decode references that slot, a decode must wait
the previous slot's conversion restore, so the required cross-queue semaphore is
exactly what forbids cross-frame overlap (an intermediate NV12 copy would decouple
them, at the cost of the copy). The **system NV12 path**
(`decode_all`) is pipelined through a fixed-depth in-flight ring
(`DECODE_RING_DEPTH`, a second `RESET_COMMAND_BUFFER` command pool with a
persistent command buffer + fence per slot, and one readback buffer sized
`DECODE_RING_DEPTH` frames so each slot copies to its own region): each picture is
recorded + submitted without waiting, the oldest slot is retired (fence waited,
its NV12 read back, its bitstream freed) only when the ring wraps onto it, and a
final drain collects the tail. In-order execution on the single decode queue
preserves DPB reference correctness (references are CPU-side bookkeeping), so only
the readback buffer needs per-slot isolation; `reset` (seek) and `Drop` drain the
ring first. This keeps the CPU record + fence-wait latency hidden behind GPU
decode work instead of stalling after every picture, ~16% higher batch decode
throughput on the 3060 (measured H.264, and the m492 / m503 / m506 bit-exact-vs-
ffmpeg guards go through this path unchanged). The streaming `VulkanVideoDec`
element decodes one access unit per `process` call, so it drains per AU by design;
the ring win is on the batch `decode_all` used by the player and tests.

### 4.12 Live Egress

The receive path (§4.11) has an inverse: encoded video out over RTP. The
protocol logic is Sans-IO (§1): a pure packetizer produces the RTP packets
and a thin sink does the UDP I/O.

- `RtpH264Packetizer` (`rtppay.rs`) implements RFC 3550 + RFC 6184. An H.264
  access unit becomes a single-NAL RTP packet if the NAL fits the MTU, else
  FU-A fragments. The marker bit lands on the access unit's last packet;
  sequence numbers increment across packets and calls; one RTP timestamp per
  access unit. Pure `no_std` logic, host-testable.
- `UdpSink` (`udpsink.rs`, `udp-egress` feature) is an `AsyncElement` sink
  that drives the packetizer over each Annex-B access unit and sends the RTP
  packets to a destination on a tokio `UdpSocket`. The RTP timestamp is the
  90 kHz image of `FrameTiming::pts_ns`; sequence numbers and the per-AU
  marker bit come from the packetizer. `with_rtp(pt, ssrc)` and
  `with_max_payload(mtu)` configure the flow. It also keeps a bounded history of
  recently sent packets and honors receive-side RTCP NACK by retransmitting them
  (`with_retransmit(enabled, capacity)`); see the receive-side feedback loop in
  §4.12b.

### 4.12a Live Capture (V4L2, libcamera)

`V4l2Src` (`v4l2src.rs`, `v4l2` feature, Linux-only) is the first real capture
source: it streams frames off a `/dev/videoN` device via V4L2 mmap streaming
I/O, wrapping the pure-Rust `v4l` crate (no libv4l C dependency). Packed
**YUYV** (4:2:2, the near-universal UVC output) is the preferred format, and
`VideoConvert` unpacks it to a planar / RGB target (§3.1 raw formats), so the
canonical chain is `V4l2Src -> VideoConvert(Yuyv -> Nv12) -> sink`.

Two design points carry the element:

- **Blocking ioctls off the async path.** V4L2 dequeue is a blocking ioctl, so
  capture runs on a dedicated `std::thread` that owns the device and the mmap
  stream (which borrows the device) and copies each frame's payload into a
  bounded channel. The `SourceLoop::run` future drains that channel and pushes
  `DataFrame`s. The channel bound (`BUFFER_COUNT`) applies backpressure: the
  capture thread blocks rather than growing memory when the pipeline falls
  behind. The source reports a live `LatencyReport` of one frame period.
- **Up-front format negotiation, re-open for capture.** The probe opens the
  device, enumerates its pixel formats, and for each one it can carry sets that
  format at the requested geometry and reads back what the driver actually chose
  (it may snap to a supported mode); the probe device is then dropped. The
  capture thread re-opens the device under the negotiated format. Keeping no
  device handle in the struct between negotiation and `run` sidesteps `Send` /
  borrow entanglement with the stream. Errors surface as
  `G2gError::Hardware(HardwareError::V4l2(errno))`.
- **The device offers, negotiation decides (M954).** Every confirmed format
  becomes one alternative of the source's `CapsConstraint::Produces` set, in a
  fixed preference order: YUYV, NV12, I420, then MJPEG last (it needs a
  decoder). A chain that constrains nothing therefore takes YUYV, exactly as
  before; a downstream `MjpegDec` (or a pinned `image/jpeg` link) drops the raw
  alternatives during arc consistency and the camera runs in its MJPEG mode
  instead, which is what fits 1080p over USB. `configure_pipeline` reads the
  solved caps back to learn which mode the capture thread runs, and MJPEG's
  per-frame length comes from the buffer's `bytesused` rather than the format's
  `sizeimage`. What a pixel format means on a link (its `Caps`, its frame size)
  lives in `capturepixelformat.rs`, shared with `LibCameraSrc`, which sits on a
  different fourcc registry but agrees on the meaning.
- **DMABUF output (M956).** The `io-mode` property selects how a buffer leaves
  the element: `auto` / `mmap` copy as above, `dmabuf` exports each MMAP buffer
  once (`VIDIOC_EXPBUF`, at stream start) and emits frames in
  `MemoryDomain::DmaBuf` carrying a share of the fd their buffer was filled into,
  so a GPU consumer imports the camera buffer with no copy. The buffer *is* the
  frame there, so the invariant is that a buffer goes back to the driver only
  once every share of its fd has dropped (the element keeps one share per buffer
  for the whole stream and re-queues on the count falling back to it); the
  in-flight bound stays below `BUFFER_COUNT` so the driver always has a buffer to
  fill. An exported fd carries no payload length, so dmabuf mode advertises only
  the raw formats and MJPEG stays on the copy path.

`LibCameraSrc` (`libcamerasrc.rs`, `libcamera` feature, Linux-only) is the
second capture source and the modern Linux camera path: it captures through the
**libcamera** stack (linking the system libcamera via the `libcamera` crate),
which drives UVC webcams through its `uvcvideo` pipeline handler (the same
devices as `V4l2Src`) plus CSI/ISP cameras that need an ISP pipeline V4L2 alone
cannot. It follows the same two design points as `V4l2Src` (blocking work off
the async path; up-front negotiation, re-configure for capture), but differs in
two ways: it asks libcamera for **NV12** and falls back to **YUYV** only when
the camera does not offer NV12 (mapping whatever survives `validate()` to
`Caps::RawVideo`), so a camera that produces planar frames needs no
`VideoConvert`; or, with `with_mjpeg(true)` / `format=mjpeg`, it negotiates
**MJPEG** and emits `CompressedVideo{Mjpeg}` for `MjpegDec` downstream (the
on-camera-compression path for resolutions / frame rates uncompressed YUYV
cannot sustain over USB). Because libcamera is callback-driven and thread-affine, the
capture thread owns the whole libcamera object graph (manager, camera, a
request-buffer ring, and the completion callback) rather than a single device
handle. Each completed request's planes are packed contiguously (Y then
interleaved UV for NV12) before being forwarded over the bounded channel. The
requested frame rate is bounded on the camera with a `FrameDurationLimits` start
control (the minimum frame duration caps the fastest rate; the maximum is left
generous so an unachievable request degrades to the camera's own ceiling instead
of collapsing). Manual exposure / gain (`with_exposure` / `with_gain`, which turn
auto-exposure off) ride the same start-control path and are the real frame-rate
lever in low light: with auto-exposure on the camera lengthens exposure until the
rate collapses (~9 fps on a dim webcam, the same rate in every format and
resolution), while a fixed short exposure restores a high rate (measured 8.8 ->
24.9 fps on the developer's webcam). `Brightness` (and `Contrast` / `Saturation`)
are post-capture image adjustments that do not touch the exposure time, so they
brighten a dim short-exposure frame without giving back the frame rate (measured
mean luma 16 -> 117 at a fixed exposure). The camera can also be selected by an
id substring (`with_camera_id`) rather than enumeration index, stable across
reboots. Start controls are applied through a support
check against the camera's `ControlInfoMap`, because libcamera aborts the process
(a C++ exception across the FFI boundary) if a control list carries an id the
pipeline handler does not advertise (a UVC webcam may expose `ExposureTime` but
not `AnalogueGain`). The `libcamera` crate requires libcamera
`>= 0.4`, newer than some distro packages, so the feature is host-validated (like
the NVIDIA stack) rather than built in CI. The camera also feeds the GPU/ML path:
the g2g-ml `libcamera-wgpu` feature chains `LibCameraSrc -> VideoConvert(NV12) ->
WgpuPreprocess` to turn live frames into a normalized f32 NCHW tensor on the GPU
(validated camera-to-tensor on an RTX 3060). A zero-copy dma-buf import of
libcamera buffers into wgpu (the Linux analog of the CUDA / AHardwareBuffer
interop) was investigated under the `libcamera-dmabuf` feature: libcamera does
export a real dma-buf fd, but on a USB camera + discrete NVIDIA GPU the driver
advertises the buffer as importable (`vkGetMemoryFdPropertiesKHR`) yet the actual
`vkAllocateMemory` import fails to bind, because the buffer is CPU/vmalloc-backed
and the dGPU cannot map it. So the CPU-upload path is correct for that
configuration; zero-copy is expected to work on an integrated GPU (shared memory)
or a CSI/ISP camera (GPU-visible buffers), and the full import-to-texture element
is gated behind the on-hardware probe rather than shipped blind.

Two more capture sources follow the same blocking-work-off-the-async-path shape:
`PipeWireSrc` (`pipewire` feature, Linux) captures interleaved PCM off the
PipeWire graph (the modern Linux media layer) by running a `pw::stream` input on
a dedicated main-loop worker thread feeding the `run` loop over a channel; it
requests a fixed PCM format the PipeWire adapter converts to, so the produced
caps are deterministic. Its video sibling `PipeWireVideoSrc` captures raw frames
from any PipeWire video node (camera, another client, a portal-opened screen-cast
node named through `target-object`) and its `io-mode` property picks the buffer
path: `mmap` copies each frame out of the mapped block into `System` memory, while
`dmabuf` negotiates a `Buffers` param accepting `SPA_DATA_DmaBuf` alone and hands
the descriptor on as `MemoryDomain::DmaBuf`, holding each buffer until every share
of its frame is released (the domain is fixed by negotiation, hence a property
rather than GStreamer's per-caps feature). Screen capture on a Wayland desktop
goes through `portal=true` (`portal` feature) instead of `target-object`, which
only reaches the session's own PipeWire remote: the element runs the
xdg-desktop-portal `ScreenCast` handshake (`CreateSession` / `SelectSources` /
`Start`, each answered on an `org.freedesktop.portal.Request` object, then
`OpenPipeWireRemote`) on the capture worker thread over a blocking zbus
connection, and connects the stream to the granted node id on the private remote
fd the portal returns. Every step is bounded by `portal-timeout`, so an
unattended consent dialog fails the capture instead of hanging, and
`portal-restore-token` re-opens an earlier grant without asking. `MfVideoSrc`
(`mf-video-src`, Windows) is the camera sibling of `WasapiSrc`: it enumerates
video capture devices and drains NV12 / YUY2 frames via an `IMFSourceReader` on a
COM/MTA worker thread.

#### Linux audio output

The audible-output end of the audio path on Linux mirrors the Windows-only
`WasapiSink` across the three Linux audio stacks, each a `std`-gated element with
a dedicated render worker thread: `AlsaSink` (`alsa-sink`, libasound, lowest
level), `PulseSink` (`pulse-sink`, the blocking libpulse "simple" API), and
`PipeWireSink` (`pipewire`). ALSA / Pulse backpressure naturally through the
blocking write; PipeWire's `process` callback pulls on its own clock and cannot
backpressure, so that sink's hand-off queue is leaky (bounded to ~1 s, dropping
the oldest bytes, the `LinkPolicy::DropOldest` analog for an external clock). All
accept interleaved `PcmS16Le` / `PcmF32Le` and reject compressed audio
structurally. Errors surface as `HardwareError::{Alsa,PulseAudio,PipeWire}`.

### 4.12aa Device Discovery

The `GstDeviceProvider` / `GstDeviceMonitor` analog (M938/M939,
`g2g-core/src/runtime/device.rs` + `g2g-plugins/src/devicemon.rs`). A
`DeviceProvider` probes one backend for the devices it can see; a
`DeviceMonitor` aggregates providers behind class + caps filters
(`gst_device_has_classes` semantics: `Video/Source` requires both parts,
`Source` matches any source) and, once started, watches for hotplug. Events
arrive on the monitor's own channel, not the pipeline bus: a monitor is
application-side, not part of a running graph.

A `Device` does not own an element factory. It carries the launch **name** of
the element that drives it plus the textual `key=value` properties that select
it, so construction rides the same `Registry` + `PropertySpec::parse_value`
path as `parse_launch`: `Device::create` builds and configures the element,
`Device::launch_fragment` prints the `v4l2src device=/dev/video0` fragment a
text pipeline would use. `persistent_id` is the monitor's hotplug diff key and
is chosen per backend for cross-reboot stability (USB/PCI bus info for v4l2,
the direction-prefixed hint name for ALSA, `node.name` for PipeWire, which
survives daemon restarts where `object.serial` does not).

Hotplug has two paths. A provider with a native event source implements
`watch()`: PipeWire registers a registry listener on a dedicated loop thread,
relies on the daemon replaying existing globals for the initial `Added` set,
and posts through a filter-applying `DeviceSink` (a try-send retry loop, so N
watcher threads never depend on the channel's single send waker; shutdown
closes the receiver first, so a watcher blocked on a full queue exits instead
of deadlocking the join). Providers without events (v4l2, ALSA, GPU) are
covered by the monitor's poll-and-diff fallback thread keyed on
`persistent_id`.

Standard providers (`default_device_monitor`, mirroring `default_registry`'s
per-feature gating): **v4l2** (capture nodes with YUYV modes probed into real
caps alternatives; other fourccs listed in `detail`), **ALSA** (PCM hints in
both directions, formats probed via `HwParams`, busy devices still listed with
empty caps), **PipeWire** (media-class nodes mapped to
`pipewiresrc`/`pipewiresink`/`pipewirevideosrc`, selected via their
`target-object` property), **GPU** (`Compute/GPU`, a g2g extension beyond
GStreamer's capture/render model: wgpu adapters, CUDA ordinals, VAAPI render
nodes; only the render nodes name a driving element, the rest are
informational), **MF** and **WASAPI** on Windows, and **AVF** and **CoreAudio**
on macOS (M943, below). The `g2g-device-monitor` binary is the CLI over all of
this (`gst-device-monitor-1.0` analog): one-shot listing, class filter,
`--json`, and `--follow` for live hotplug.

**Windows / macOS (M943).** `mfdevice` lists the `MFEnumDeviceSources` video
capture devices with the NV12 / YUY2 native modes `mfvideosrc` can deliver
(reading them activates the source, so a camera another application holds open
lists with empty caps); `wasapidevice` lists the active `IMMDeviceEnumerator`
render / capture endpoints with their shared-mode mix format as caps.
`avfdevice` lists cameras from an `AVCaptureDeviceDiscoverySession`, and
`coreaudiodevice` the HAL's `kAudioHardwarePropertyDevices` entries, one device
record per direction a duplex device carries. WASAPI is the one non-Linux
backend with a native watch (an `IMMNotificationClient` whose callbacks wake a
re-probe on the watch thread, since the callback carries only an id and must
not block); the other three are polled, because MF has no hotplug callback
short of a `WM_DEVICECHANGE` window and the AVFoundation / CoreAudio listeners
need a run loop a library has no business owning.

On these platforms the selection property **is** the persistent id, so no
separate `device-id` is needed: `mfvideosrc device-path=` takes the MF symbolic
link, `wasapisrc` / `wasapisink device=` the endpoint id, `avfvideosrc` /
`avfaudiosrc device=` the `AVCaptureDevice` unique id, and `coreaudiosrc` /
`coreaudiosink device=` the Core Audio device UID (the `AudioDeviceID` is
reassigned every boot, the UID is not). Each element gained that selector as
part of M943, sharing one open-by-id helper with its provider (`wasapipcm` on
Windows) so the id a listing reports and the id `device=` accepts cannot drift.

**`v4l2src device-id` (M944).** V4L2 is the exception: its selection handle is
the node path, which the kernel renumbers across a replug, so `v4l2src` takes a
separate `device-id` carrying the provider's `bus_info:card:path` id and
resolves it against a fresh probe at negotiation. The exact id wins; failing
that, the hardware half (bus + card) matches, which is what survives a replug
into the same port, and the lowest-numbered node of a multi-node camera is
chosen (the capture node on every UVC device). An id nothing carries fails the
negotiation with `HardwareError::V4l2(ENODEV)` rather than silently falling
back to `device`.

**V4L2 camera controls (M944, M1047).** `v4l2src` exposes the user and camera
control classes as runtime properties under the names `v4l2-ctl` uses:
exposure, focus and white balance (`exposure-auto`, `exposure-absolute`,
`focus-auto`, `focus-absolute`, `white-balance-temperature-auto`,
`white-balance-temperature`), the picture controls GStreamer's `v4l2src` also
names (`brightness`, `contrast`, `saturation`, `hue`, `gamma`, `gain`,
`sharpness`, `backlight-compensation`, `power-line-frequency`), and pan / tilt
/ zoom (`pan-absolute`, `tilt-absolute`, `zoom-absolute`). One table drives
both the property specs and the `VIDIOC_S_EXT_CTRLS` ids, and its order is the
apply order: an auto switch precedes the manual value it gates, because a
driver rejects a manual exposure while auto exposure is on. Each is applied
with its own ioctl, since a batch may not span the user and camera control
classes. Only a control that was set is touched, and one the camera does not
implement fails the negotiation instead of being quietly ignored. A control's
property kind follows its range: a switch is `Bool`; the four picture controls
GStreamer also gives a signed property to, plus pan / tilt, are `Int`; the rest,
whose range starts at zero on every device that has them, are `Uint`.

Anything past that table is reachable through `extra-controls`, GStreamer's own
spelling, as a comma-separated `name=value` list. The names are the driver's
own, kebab-cased, which is how `g2g-device-monitor` lists them: the provider
walks `VIDIOC_QUERY_EXT_CTRL` and records every numeric control as a
`control.<name>` detail entry with the range and default the driver reports, so
a listing tells the caller exactly what an `extra-controls` entry may say. The
walk is g2g's own rather than the `v4l` crate's `query_controls`, which panics
on a control type its enum predates (uvcvideo's region-of-interest rectangle).
A malformed list fails the property; a name this device does not offer, or a
value outside its reported range, fails the negotiation and logs the names the
device does carry.

### 4.12b Live Ingress (UDP / RTP)

`UdpSrc` (`udpsrc.rs`, `udp-ingress` feature) is the receive-side inverse of
`UdpSink` (§4.12): it receives RTP on a tokio `UdpSocket` and depayloads H.264
into Annex-B access units pushed downstream as `CompressedVideo` H.264, so the
canonical chain is `UdpSrc -> FfmpegH264Dec -> sink`. The I/O is async, so
unlike `V4l2Src` it needs no capture thread.

The protocol logic is Sans-IO (§1), mirroring the egress split: `rtpdepay.rs`'s
`RtpH264Depayloader` is a pure, `no_std`, host-testable function that inverts
`RtpH264Packetizer`. Single-NAL and STAP-A payloads pass through; FU-A fragments
reassemble (the original NAL header is rebuilt from the FU indicator's F|NRI and
the FU header's type); the RTP marker bit closes an access unit. A sequence-
number gap drops the in-flight reassembly so loss or reorder never welds two
access units together.

**Receive-side resilience (jitter buffer + RTCP + NACK).** Between the socket and
the depayloader sits a Sans-IO jitter buffer (`rtpjitter.rs`,
`RtpJitterBuffer`): it orders packets by an *extended* sequence number (the
16-bit RTP sequence unrolled to a monotonic counter, so wraparound is handled),
releases them in order, holds a gap only until its predecessors fill or a
bounded deadline elapses (then declares loss), and drops duplicates / too-late
packets. RTCP (`rtcp.rs`, Sans-IO RFC 3550 SR/RR/BYE + RFC 4585 Generic NACK,
plus `ReceptionStats` for loss fraction / cumulative loss / interarrival jitter)
runs RTP/RTCP-muxed on the one socket (RFC 5761): `UdpSrc` sends periodic
receiver reports and emits a NACK for each detected gap, and `UdpSink` honors
those NACKs by retransmitting from its send history (§4.12). A retransmit
arriving inside the jitter hold window heals the gap before it is declared lost,
so the loop recovers packet loss end to end. **RFC 4588 RTX** (`rtx.rs`)
wraps a NACK resend in a distinct payload type / SSRC with the original sequence
prepended (`UdpSink::with_rtx` / `UdpSrc::with_rtx`), unambiguous under heavy
loss. **ULPFEC** (`ulpfec.rs`, RFC 5109) adds *feedback-free* recovery: the
sender XORs each group of media packets into a repair packet (`with_fec`), and
the receiver reconstructs a single per-group loss from the repair plus the
survivors and injects it into the jitter buffer, with no round trip, the better
fit for one-way or high-RTT paths. NACK, RTX, and FEC compose.

This is **raw RTP** with no RTSP/SDP, so there is no out-of-band stream
description: the output geometry is a declared hint (`with_video_size` /
`with_framerate`), and since H.264 carries its real dimensions in the SPS a
downstream decoder re-derives and corrects them. `RtspSrc` (via `retina`) already
covers the RTSP case with its own jitter buffer (§4.11.4).

**RTMP ingest.** `RtmpSrc` (`rtmpsrc.rs`, `rtmp` feature) accepts one RTMP
publisher (ffmpeg / OBS pushing `rtmp://host/app/key`) over TCP and streams the
result downstream as `Caps::ByteStream{Flv}`, so the chain is
`RtmpSrc -> flvdemux -> h264parse -> ...`. The protocol is Sans-IO (`rtmp.rs`,
`RtmpSession`): the simple (non-digest) handshake publishers fall back to, the
chunk-stream reassembly (per-chunk-stream header inheritance + `Set Chunk Size`),
and the AMF0 `connect` / `createStream` / `publish` command flow (the session
emits the Window-Ack / Set-Peer-Bandwidth / `_result` / `onStatus` replies). An
RTMP audio/video message payload is exactly an FLV tag *body*, so the session
reframes the messages into an FLV byte stream that the existing `flvdemux` (§4.17)
recovers the H.264 / AAC access units from. Scope is one publisher / one stream,
H.264 + AAC, AMF0.

**RTMP egress.** `RtmpSink` (`rtmpsink.rs`, `rtmp` feature) is the inverse:
it connects out to an RTMP server and *publishes* an incoming FLV byte stream, so
the chain is `... -> flvmux -> RtmpSink location=rtmp://host/app/key`. The
protocol is Sans-IO (`rtmp.rs`, `RtmpPublisher`), the mirror of `RtmpSession`: it
sends C0/C1, drives the `connect` / `createStream` / `publish` command ladder off
the server's `_result` / `onStatus` replies, then splits the FLV stream back into
tags and reframes each as an RTMP audio/video/data message (the tag body is the
message payload). Both directions share one `ChunkReader` (the chunk-stream
reassembly) and one fragmenting `write_message` writer, so the publisher and the
session are true inverses rather than parallel re-implementations. The element
opens the socket lazily on the first buffer (after `flvmux`'s header) and drives
the publish ladder before sending media. Validated sans-IO by pitting the
publisher against the server session (an access unit survives the RTMP round
trip); live publish to a real endpoint is operator-validated.

**RTSP server.** `RtspServerSink` (`rtspserversink.rs`, `rtsp-server` feature)
hosts the server side of RTSP: a player connects over TCP, runs OPTIONS /
DESCRIBE / SETUP / PLAY, and the sink streams the pipeline's H.264 to the
player's negotiated UDP port as RTP, reusing the `RtpH264Packetizer`. The
protocol is Sans-IO (`rtspserver.rs`, `RtspResponder` + `RtspRequest::parse` +
`sdp_h264`): a per-session state machine answering each method and returning an
`RtspEvent` (`Setup{client_rtp_port}` / `Play` / `Record` / `Teardown`) that the
element acts on. It also speaks the publisher path (ANNOUNCE / RECORD), served by
the receive-side `RtspServerSrc`. The sink is multi-client (each player gets its
own RTP session, broadcast per frame) on either transport: unicast UDP or
TCP-interleaved (`$`-framed on the control connection, validated against
`ffmpeg -rtsp_transport tcp`). During PLAY the sink runs RTCP and keepalive:
periodic RFC 3550 sender reports per player (UDP from the socket adjacent to the
RTP one, so the advertised `server_port` pair is real, or `$`-framed on the RTCP
channel), a BYE at EOS, and a session timeout advertised as
`Session: id;timeout=N` at SETUP, with a player reaped when it is silent past
the timeout on both the control channel (GET_PARAMETER / OPTIONS) and RTCP
(receiver reports, which arrive `$`-framed mid-stream on an interleaved
control connection and are consumed there). Validated end-to-end over loopback
(handshake, RTP recovery, SR delivery on both transports, RR-extended lifetime,
silent-client reap).

**SRT (Secure Reliable Transport).** `SrtSink` (caller, egress) and `SrtSrc`
(listener, ingress, `srt` feature) carry an MPEG-TS byte stream over UDP with
SRT's reliable-but-low-latency ARQ — the contribution-link transport. The
protocol is Sans-IO (`srt.rs`): the 16-byte packet header + data/control wire
codec (HSv5 HANDSHAKE with the HSREQ-latency and Stream-ID extensions,
ACK / NAK loss-report / ACKACK / KEEPALIVE / SHUTDOWN), the caller/listener
handshake driver (`SrtHandshake`, induction → conclusion with a listener cookie
challenge), and the ARQ pair `SrtSender` / `SrtReceiver` (the sender buffers and
resends on NAK with the retransmit flag; the receiver reorders by wrap-aware
sequence, NAKs gaps, and delivers in order) — the same shape as the RTP
jitter/NACK path. Validated g2g↔g2g end to end over a lossy loopback (handshake +
data + a dropped packet recovered via NAK). The wire format follows the SRT
draft so real-peer interop is the design target. AES-256 encryption
(`with_aes256`), mid-stream key rotation (`with_key_rotation`), the TSBPD timing
model, and live-mode congestion control / pacing (`with_max_bandwidth`) are in
place.

### 4.13 CSP Caps Negotiation

The handshake sketched in §4.2 is the *interface* contract. Underneath it,
capability negotiation is a **distributed constraint-satisfaction problem
(CSP)**: each element declares a constraint over its `(input, output)` caps, and
a solver finds a per-link caps assignment over the whole graph (or an affected
subgraph on a mid-stream change) that satisfies every constraint, ranked by
preference. This subsumes GStreamer's pad-by-pad negotiation: the solve runs
once, returns structured failure when no assignment exists, and trades query
round-trips for direct calls. The same machinery also settles the allocation
cascade (buffer pools, strides, memory domains), auto-plugs decoders
(`decodebin` / `playbin`) and memory-domain converters, and re-solves mid-stream
on a `CapsChanged`.

This is the largest and most intricate part of the design, so its full treatment
lives in a dedicated document: **[DESIGN-caps.md](DESIGN-caps.md)**. It covers the
`CapsSet` algebra and constraint enum (§4.13.1), the arc-consistency solver
(§4.13.2), the DAG runner and opt-in threaded runner (§4.13.3), mid-stream
re-solve (§4.13.4), the allocation cascade (§4.13.5), fan-out / fan-in (§4.13.6),
bins and ghost pads (§4.13.6a), pad templates (§4.13.7), `ACCEPT_CAPS` /
`CapsFilter` (§4.13.8), auto-plug / registry / playbin (§4.13.9), and the solver's
current limits (§4.13.10). The `§4.13.x` references elsewhere in `DESIGN.md`
resolve there.

### 4.14 Pipeline Lifecycle: State Machine, Preroll, and Seek

The lifecycle spine sits on top of the DAG runner: it turns "build, run to EOS,
drop" into a controllable `NULL → READY → PAUSED → PLAYING` machine that can
preroll, pause, scrub, and resume.

**State machine + preroll.** `PipelineState` (`NULL`/`READY`/`PAUSED`/`PLAYING`)
and `StateChangeReturn` are ungated core types. A `StateController` (runtime
feature) carries the target state and a sink-side **flow gate**: below `PLAYING`
a sink parks at the gate, stops draining its edge, and backpressure stalls the
DAG upstream, the state machine reuses the existing channel backpressure rather
than a separate pause mechanism. Preroll: a non-live `PAUSED` transition admits
exactly one buffer per sink and then holds; the runner calls
`expect_prerolls(n)` and each sink's `notify_prerolled` aggregates so the async
`PAUSED` completes with a single `AsyncDone` once *all* sinks have prerolled.
Live pipelines (`set_live(true)`) take the `NoPreroll` path (no frame is held).
The lifecycle is opt-in via `run_simple_pipeline_stateful` and
`run_graph_stateful`; the plain runners are unchanged.

**Seek + SEGMENT + running time.** `g2g-core::segment` is a pure-core (ungated)
model: `Seek` / `SeekType` / `SeekFlags` describe the request, and `Segment`
carries the rate/direction-aware running-time ↔ stream-time ↔ base-time math
(`GstSegment`-equivalent), with `clip`, `for_flush_seek` (which resets `base`
so running time restarts after a flush), and `accumulate_seek` (the
non-flushing seek: `base` advances to the running time playback has already
reached, so the running-time line stays monotonic across the seek, the gapless /
segment-seek / loop case). `PipelinePacket::Segment` is the
carrier: the runner emits an opening SEGMENT and every element forwards it
(transforms/decoders forward, sinks consume), the same way `Flush` already
flows. A `SeekController` (runtime) is a cloneable handle the application holds;
a seek-aware source's run loop polls `take_pending()` between frames and, on a
flushing seek, emits `Flush`, repositions, emits the post-flush `Segment`, and
resumes, so a seek reaches the source GStreamer-style (upstream) without a
back-reference. `Mp4Src` is the first real repositioning source (flushing
seek, keyframe `SNAP_BEFORE`, re-prepended parameter sets), and `SyncSink` maps PTS
to running time through the `Segment` and clips pre-target frames so accurate seek
presents the exact requested frame. A non-flushing seek emits only
the accumulating `Segment` (no `Flush`), so the source keeps producing on a
continuous running-time line. Reverse playback (`Seek::reverse`,
`rate < 0`) needs no sink-specific code: the source emits frames newest-PTS-first
over `[start, stop]`, and `SyncSink` schedules each by `Segment::to_running_time`
(which measures reverse from `stop`) and clips via `contains`, so descending PTS
maps to ascending running time and presents in the correct visual order, the
`Segment` abstraction generalizing the sink to negative rate transparently.
The producing half is GOP-batched, since a decoder only runs forward: on a
`rate < 0` seek `Mp4Src` walks the container's sync samples backward from the
segment `stop` (the `stss` index `parse_progressive` recovers, or the `trun`
keyframe flags of a fragmented file) and emits one whole GOP at a time in decode
order, newest GOP first, including the samples above `stop` that later frames
reference (the sink clips them). `GopReverse` (`gopreverse`) closes the loop
after the decoder: it buffers a decoded GOP, detects its end by the PTS jumping
backward into the next (earlier) GOP, and re-emits each batch in descending PTS,
so the sink receives reverse presentation order. A forward segment passes
straight through it, so it can sit in any graph that may seek backward.
**Trick-mode KEY_UNIT** frame selection (present only keyframes for fast scrub)
is done: `FrameTiming::keyframe` carries a per-frame flag (set by
`h264parse` from `h264_au_is_keyframe`, and by `mp4src` / `fmp4demux` from the
container sync-sample / `trun` keyframe flag), a `TRICKMODE` seek sets
`Segment::key_units_only` in `from_seek`, and `SyncSink` drops non-keyframe frames
under such a segment before scheduling them (counted by `trick_dropped()`).
**Segment playback / gapless looping** (the `GstSeekFlags::SEGMENT` analog)
is consumed through the `SeekController`, not a new packet: g2g has no
`SEGMENT_DONE` `PipelinePacket` (it would force a new control variant through
every element's exhaustive match), so the controller carries it on the same
app<->source channel a seek already uses. A `SEGMENT`-flagged seek runs the
source to `stop`; instead of `Eos` the source calls `notify_segment_done(stop)`
and parks (polling) for the app's next move. The app observes
`segment_done_count()` / `take_segment_done()` and re-arms a *non-flushing*
`SEGMENT` seek to loop (so `accumulate_seek` advances `base` by one span per
iteration, gapless, no `Flush` downstream) or calls `shutdown()` to end the loop,
at which point the idle source emits `Eos`. The idle park is **wakeful**
(`SeekController::wait_event`): the source `await`s a future that resolves
when `seek` / `shutdown` wakes the registered waker, so a looping source between
loops costs nothing (no busy-poll), the poll-free analog of GStreamer pausing the
source task. `Mp4Src` is the first real source to loop on `SEGMENT`: it
clips playback to the segment `stop`, reports segment-done at the boundary, and
parks on `wait_event` for the app's loop seek (non-flushing, snapping to the
keyframe at or before the target so a decoder resumes cleanly) or `shutdown`. It
also now honours non-flushing repositioning seeks (accumulating `Segment`, no
`Flush`), not just flushing ones. **Re-preroll when paused.** A paused,
prerolled pipeline backpressures its source, so a flushing seek issued now would
never take effect (the held sink never drains). `StateController::request_repreroll`
(called by the app alongside the seek) bumps a preroll generation; `flow_gate`
takes the arm's generation and reopens for a stale one, so each sink arm
re-prerolls. The arm drains the stale pre-seek frames (discarding, not presenting)
until the `Flush`, then prerolls the post-flush target and re-fires `AsyncDone`,
so scrubbing a paused pipeline updates the shown frame. **Byte-source seek
.** `FileSrc` is BYTES-format seekable (`with_seek`): a flushing seek
repositions the file read to a byte offset and emits `Flush`. **Demuxer seek
.** A byte-stream demuxer (a transform with no random access) becomes
seekable by driving that upstream byte source. A shared `DemuxSeek` helper turns
an app time seek into an upstream byte-seek to offset 0, drops in-flight pre-seek
input until the returned `Flush`, resets the demuxer's parser, then discards
decoded units until the keyframe at/after the target and emits a resume
`Segment` (correct for any container without an index; a re-scan, with an
index-derived offset a later optimization). All five carry it
(`fmp4demux` / `tsdemux` / `mkvdemux` / `flvdemux` / `oggdemux`), each using its
own keyframe signal (the container flag, or `annexb::au_is_keyframe` for TS whose
units have none; every audio packet is a resync point, and `oggdemux` now
accumulates an Opus PTS from the TOC byte). Where the container has no index at
all, `oggdemux` guesses the landing byte offset by interpolating through observed
`(byte offset, stream time)` anchors, clamped a page-max below the byte length
the source publishes on the seek controller (`SeekController::set_stream_len`,
`FileSrc` from the file size and `DownloadBuffer` once the spill is complete), so
a guess through a front-dense file cannot land at EOF. **Adaptive segment seek.** The
adaptive sources `HlsSrc` / `DashSrc` are TIME-seekable (`with_seek`): unlike the
BYTES-format `FileSrc`, an app time seek resolves to the media segment containing
the target (HLS walks cumulative `#EXTINF` durations; DASH maps the target onto
the `SegmentRef` `$Time$` line), then the source emits `Flush`, jumps to that
segment, re-emits the fMP4 init segment (the downstream demuxer reset on the flush
needs its `moov` again), emits the post-flush `Segment` at the segment start, and
resumes there. This is the CMAF / DASH segment-transition case (clamped to the
last segment; a target past the end lands there).

### 4.15 Bus and Observability

The pipeline `Bus` (§4.9.1) is a many-producer / single-consumer channel for
out-of-band events, so an element notifies the application without a
back-reference. `BusMessage` covers the lifecycle and quality signals an
application reacts to:

- `StreamStart`, `Eos`, `Error`, `Warning`, `Info(String)` — stream lifecycle,
  faults, and non-fatal status. `StreamStart` is posted by the source arm before
  a source produces (one per source), bracketing each stream with its `Eos`
  (`GST_MESSAGE_STREAM_START`); `Info` is the third severity below `Warning`,
  element- / app-posted for status that is not a problem (`GST_MESSAGE_INFO`).
- `DurationChanged { duration_ns }` — the total stream duration became known
  (§4.15's query handle is the pull side; this is the push notification), posted
  by the source arm from `SourceLoop::query_duration` (`GST_MESSAGE_DURATION_CHANGED`).
- `Tag { tags, program }` — container / stream metadata, posted out of band
  (`GST_MESSAGE_TAG`). `program` scopes the tags to one MPEG-TS `program_number`
  (an SDT service entry, so a multi-program multiplex reports each service
  separately) and is `None` for a container with a single metadata scope.
- `StreamTag { stream_id, tags }` — the same, scoped to one elementary stream
  (a Matroska `Tag` whose `Targets` names a `TagTrackUID`). `stream_id` is the
  id that stream has in the posted `StreamCollection`.
- `NegotiationFailed(NegotiationFailure)` — structured caps conflict naming the
  responsible element pair (§4.13), posted by the coordinator on a startup or
  mid-stream negotiation failure.
- `StateChanged { old, new }` and `AsyncDone` — every effective lifecycle
  transition, and the completion of an async `PAUSED` once preroll aggregates
  (§4.14).
- `Qos { running_time_ns, jitter_ns, processed, dropped }` — a synchronizing
  sink (`SyncSink`, `WaylandSink`) that has fallen behind the clock drops a late
  frame and reports it, the `GST_MESSAGE_QOS` analog. The drop decision, count,
  and post live in a shared `QosTracker` (`g2g-core::qos`), which also posts the
  running stats periodically (`with_qos_interval_ns`, pipeline-clock cadence),
  so an app sees sink health without waiting for a drop.
- `Buffering { percent, element }` — a link's fill (0 = underrun, 100 = full),
  posted on a quartile crossing via `run_graph_with_bus` by the sink *and*
  transform arms, tagged with the instance name of the element the link feeds
  (self-posting prebuffer sources leave it `None`). Since g2g has no `queue`
  element, this reports the bounded link channel's own occupancy
  (`fill_percent`), the `GST_MESSAGE_BUFFERING` analog.
- `SegmentDone { position_ns }` — a `SeekFlags::SEGMENT` seek reached its `stop`
  (`GST_MESSAGE_SEGMENT_DONE`), posted by `SeekController::notify_segment_done`
  when the app attached a bus to the controller (`set_bus`). The take-once
  back-channel (§4.14) is unchanged; this is the push side of the same event, so
  a looping app drives the next loop seek from the bus instead of polling.
- `StreamStatus { entered, thread_id }` — a streaming thread started or finished
  (`GST_MESSAGE_STREAM_STATUS` enter / leave), posted only by the thread-per-arm
  runner, one pair per spawned arm thread (the coordinator's included), so an app
  sees the graph's real thread fan-out. `thread_id` hashes the OS `ThreadId`
  (which has no stable numeric form), so only equality is meaningful.
- `ClockLost` — the elected clock lost the reference it disciplines to
  (`GST_MESSAGE_CLOCK_LOST`); see the clock health monitor in §4.4.

Posting is non-blocking (`try_post`): a control message never stalls the data
path; a full bus drops the report rather than applying backpressure.

**Element-granular logging (`g2g-core::log`)** is the complementary
diagnostic channel, the `GST_DEBUG` analog, for developer tracing rather than
application-facing events. A record carries a `category` (the element *type*,
e.g. `"VideoFlip"`, the filtering key), an optional `instance` name (the
element *instance*, e.g. `"VideoFlip0"`), an optional `timestamp_ns` (from a
host-installed `set_time_source`; core reads no clock), and typed structured
`fields` a sink renders or ships without parsing the message
(`g2g_log_fields!`). An element may override its category per instance
(`set_log_category` / `LogSource::log_category_override`, or `log-category=` on
a launch line, the second launch keyword beside `name=`), and the override is
what the filter matches; the auto instance name stays type-based, so a filter
knob never renumbers probes or `t.` handles. `LogLevel` runs `Error` (most severe)
through `Trace`, matching GStreamer's numeric levels; a per-category threshold
table (a default plus overrides) decides what is emitted, mirrored into an atomic
so a disabled `g2g_trace!` in a hot loop costs one atomic load. The macros
(`g2g_error!` .. `g2g_trace!`) take a `LogSource` (an element via `self`, or a
`Target` for logging about a named element) then a `format_args!` message,
checked against the threshold before formatting. Records route to an installed
`LogSink`; the `std` feature provides a stderr sink and `init_from_env`, which
reads `G2G_DEBUG` (a `GST_DEBUG`-style `*:warning,VideoFlip:trace` spec; category
names take `*` / `?` globs, e.g. `*sink*:5`, with an exact override winning). The
runners (DAG, bespoke linear, fan-in) assign each element, including muxer /
demux / fan-out payloads, an instance name before
negotiation through a shared `InstanceNamer`: an explicit `gst-launch` `name=`
(carried on the graph node, duplicates rejected at parse) or else `<category>N`
(the `videotestsrc0` convention) via `set_instance_name`, logging each element's
addition; the name also keys the element's latency probe. An element that logs
about itself (it implements `LogSource` with a stored name) carries that name in
its lines. Pulls no external logging crate, so
it holds on the `no_std` baseline; the sink is the RTOS plug-in point (UART /
RTT), and the built-in `RingSink` (bounded, overwrites oldest, drain/snapshot)
is the flight-recorder variant for postmortem dumps there. The `tracing` feature adds a `LogSink` that forwards records to the
`tracing` crate (the `g2g` target, `category` / `instance` as fields), so a host
on `tracing-subscriber` / OTLP / tokio-console receives g2g's logs in its
existing pipeline; `log::init_tracing()` installs it and defers filtering to the
subscriber.

**Application queries: position and duration.** A media-player UI needs to
poll *where* playback is and *how long* the stream is, GStreamer's `POSITION` /
`DURATION` queries. GStreamer sends a query object upstream along the pads; g2g
pushes forward and composes paths statically (as with the latency fold, §4.13's
`LatencyReport`), so instead the runner *publishes* into a shared
`runtime::PipelineProgress` handle the application holds and polls
(`position()` / `duration()`, ns). This inverts the `SeekController` idiom: there
the app writes a pending seek and the source reads it; here the runner writes and
the app reads. **Position** is published by the DAG runner's sink arm, mapping
each consumed buffer's PTS through the active segment to stream time (the sink is
the position authority, exactly as a GStreamer sink answers from its segment plus
last buffer), so it needs no element cooperation. **Duration** is the source's
answer: `SourceLoop::query_duration() -> Option<u64>` (default `None`, so a live
source stays "unknown"), polled by the source arm before producing; `Mp4Src`
reports it from the `mdhd` box. A first duration also posts
`BusMessage::DurationChanged` as a push notification. `run_graph_with_progress`
wires the handle in; the handle is plain atomics behind an `Arc`, so reading it
from the app thread while the pipeline runs needs no lock.

### 4.16 Properties, Introspection, and the `gst-launch` DSL

The typed `with_*` builders are the zero-cost construction path and the only one
the `no_std` / RTOS baseline needs, but tooling (a text-pipeline parser, an
inspector, a future GUI) needs a *runtime* face: set a property by string name,
read it back, enumerate what an element exposes. Three layers, each building on
the last:

- **The property bag (`g2g-core::property`, `no_std + alloc`).** `PropValue`
  (`Bool` / `Int` / `Uint` / `Double` / `Fraction` / `Str`), `PropKind`, a static
  `PropertySpec` (name + kind + blurb), and `PropError`, plus
  `PropValue::parse(kind, "text")` for the `key=value` syntax. `AsyncElement` and
  `SourceLoop` (and their dyn mirrors) gain `properties()` / `set_property()` /
  `get_property()`, all defaulting to "no properties" the same zero-cost way
  `latency()` defaults to zero, so the baseline pays nothing and an element opts in
  only by overriding them. The GObject-property analog; the builders stay the
  type-checked path, this is the string-keyed one.
- **By-name construction + introspection (`Registry`, std).** `LaunchFactory`
  registers a transform / sink under a name with a parameterless constructor and
  its pad templates (sources reuse the parameterless `SourceFactory`).
  `make_source` / `make_element` build by name; `inspect(name)` dumps an element's
  role, properties, and pad templates, the `gst-inspect` analog. A factory can
  declare `with_experimental()` when its runtime is host- or device-validated
  rather than a CI promise; the dump then includes `Stability   experimental`
  and the listing suffixes `[experimental]`. The dump is
  GStreamer-shaped: a "Factory Details" header from the element type's
  `metadata()` (`ElementMetadata { long_name, klass, description, author }`, the
  `gst_element_class_set_static_metadata` analog, a zero-cost opt-in like
  `properties()`), then pad templates, then an "Element Properties" section where
  each `PropertySpec` carries its `default`, numeric `range`, enum `values`, and
  read/write `flags` alongside the blurb. `element_listing()` is the no-arg index,
  `name: Long-name` per element.
- **The text parser (`runtime::parse_launch`, std).** Turns
  `"videotestsrc num-buffers=3 ! videoflip method=rotate-180 ! fakesink"` into a
  runnable `Graph`: each `!`-separated stage is `element-name key=value ...`;
  the element is built by name, each value parsed for its property's `PropKind`
  and applied, and the stages linked source -> transforms -> sink. The result
  drops straight onto `run_graph`, so a pipeline is expressible as text without
  hand-written Rust, the `gst-launch` analog. A bare `media/type,field=value,...`
  stage is the inline caps-filter shorthand: `parse_launch` rewrites it to
  a `capsfilter` whose `caps` property is parsed by `capsfilter::parse_caps` (the
  `Caps` text grammar), so `videotestsrc ! video/x-raw,format=nv12,width=320 !
  ...` pins a format / geometry as text. Branching makes this a chain
  parser: `name=t` names an element and a `t.` reference opens a branch, with
  `tee` the structural fan-out node (its width derived from the branch count)
  broadcasting to every branch; roles follow connectivity. Text muxer fan-in is
  the remaining `gst-launch` gap. The tokenizer is quote-aware: a double-quoted
  value is one token, so whitespace and `!` inside it are literal
  (`gstwrap element="x264enc bitrate=4000"`, `filesrc location="/my file.ts"`);
  the surrounding quotes are stripped from the value.

- **ML elements by name (`g2g_ml::register`, `launch` feature, M820).** The
  stock registry is assembled in `g2g-plugins`, which does not depend on
  `g2g-ml`, so an app that wants the ML elements in a launch line calls
  `g2g_ml::register(&mut reg)` on the registry it built. That adds `ortinfer`
  (`ort`), `wgpupreprocess` (`wgpu`), and `detectionpostprocess` (`analytics`),
  each only when its feature builds the element, so `... ! ortinfer
  model=yolov8n.onnx tensor-input=true ! detectionpostprocess
  conf-threshold=0.3 ! ...` parses. `OrtInference` is constructible without a
  model for this: the `model` property loads the session through the same v1
  contract check, `tensor-input` survives the load either side of it, and until
  a model is loaded negotiation and `process` fail with `NotConfigured`.
  `WgpuInference` stays out: it is built from weight tensors and shapes, which a
  text line cannot express.
- **Declarative graph documents (`g2g_plugins::declarative`, `declarative` /
  `declarative-yaml` features, M578).** A launch string is the ergonomic
  one-liner; a JSON / YAML document is the version-controllable, tool-generated,
  comment-carrying form. A document is `nodes` (each `{ id, element, props }`,
  or a `{ id, caps }` capsfilter shorthand) + `edges` (each `{ from, to }` with an
  optional backpressure `policy` / `capacity`). It reaches the graph through
  exactly the launch parser's machinery: roles follow link degree (no inbound =
  source, several inbound = a `MuxerFactory` muxer, a fan-out node gets the M473
  auto-tee spliced in), and every property value is typed by the target element's
  `PropertySpec` and parsed with the same `PropValue::parse`, so a
  `num-buffers: 30` in JSON means exactly what `num-buffers=30` does in a launch
  string. A top-level `pipeline:` string is an escape hatch that defers to
  `parse_launch`. Both formats deserialize into one shared `GraphSpec` (a
  format-agnostic serde model), and `build_spec` turns that into the runnable
  `Graph`. `g2g-launch --graph <file>` runs one.
- **Rhai graph-building scripts (`g2g_plugins::script`, `script-rhai` feature,
  M579).** Where a document describes a *fixed* graph, a script *computes* one:
  the shape can depend on a loop, a parameter, or the environment (fan N cameras
  into a compositor, gate a branch on a flag). The script drives a small builder
  API (`add` / `caps` / `set` / `link` / `link_leaky`) that accumulates into the
  same `GraphSpec`, so a script and a document reach the graph through one builder
  and one set of role / caps / policy rules. Rhai is pure Rust (no C toolchain),
  so scripting reaches the browser (`wasm32`, CI-guarded) and every other `std`
  target without compromising the portability story; the `sync` feature makes its
  values `Send`. It is a `std`-tier capability, though: `script-rhai` implies
  `std` (Rhai's `std` feature, `std::fs` for `location=`), so the bare-metal
  `no_std` / RTOS baseline does not get scripting, by design (an MCU builds a
  fixed graph in Rust). `g2g-launch --script <file>` runs one. These are
  construction-time scripts (run once to emit a graph); the per-frame
  `scriptelement` (§4.16, below) is the runtime complement.
- **Animated properties (`g2g-core::controller`, `runtime` feature, M882).** The
  layers above set a property once, at build time; a controller makes it a
  function of stream time, the `gst-controller` analog. A `ControlSource` is a
  keyframed curve over `(pts_ns, value)` pairs, either `Step` (hold each keyframe)
  or `Linear` (interpolate), clamped to its end values outside the keyframe range;
  a `ControlProgram` binds curves to one node's property names and attaches with
  `Graph::set_node_control(node, program)` (so a `parse_launch` line's `name=` node
  can be animated, via `Graph::node_by_name`). When the run starts, before
  negotiation and before any frame flows, each program is resolved against its
  element's own `PropertySpec` table: an unknown name, a kind with no number to
  animate (`Fraction` / `Str` / `Flags`), an empty curve, or a node whose arm has
  no per-frame hook (a source drives itself; a tee carries no element) fails the
  run with `G2gError::ControlBinding` rather than animating nothing. At runtime the
  arm that owns the element samples every binding at each `DataFrame`'s PTS and
  sets it before handing that frame over, so a frame is always processed under the
  values its own timestamp calls for; the sample is rounded and clamped into the
  property's kind (a negative sample cannot wrap a `Uint`), and a value the element
  refuses fails the run loud. Transform, sink, and fan-in nodes carry controllers,
  under both the cooperative and the thread-per-arm runner (a resolved controller
  is owned data, so it rides the arm's builder closure onto its thread). Two
  deliberate limits: a zero-order-hold `Tick` frame samples nothing (the held
  frame's advanced timestamp lives inside the element, not in the runner), and
  samples use the raw PTS, not segment-mapped running time.

**Dynamic plugin loading.** Beyond build-time registration (a crate that
calls `Registry::register_*`, the primary extension path), a third party can ship
a native element as a dynamically loaded `.so`, the analog of GStreamer's scanned
plugin path. They build a `cdylib` against the published `g2g-core` plus the
`g2g-plugin` SDK and use its `declare_plugin! { elements: [ (name, Type, build) ] }`
macro, which emits two C-ABI entry points: `g2g_plugin_abi` (returns the ABI tag)
and `g2g_plugin_register(&mut Registry)` (registers the elements, body in
`catch_unwind` because unwinding across `extern "C"` is UB). A host built with the
`plugin-loader` feature (`g2g_plugins::plugin_loader`, over `libloading`)
`dlopen`s the object, reads its tag, and registers it only on an exact match;
`g2g-launch` / `g2g-inspect` expose this via `--plugin <path>` and
`$G2G_PLUGIN_PATH`.

The hard constraint is that Rust has no stable ABI, so a plugin and host must
share the same `g2g-core` version, the same `rustc`, and the same
layout-affecting features. Two features change in-memory layout across the
boundary: `metadata` resizes `Frame` (the `FrameMetaSet` side-channel) and
`multi-thread` changes the `Send` bound on the boxed element trait objects.
`g2g_core::ABI_VERSION` (a `build.rs`-computed string folding version + `rustc` +
those features) is embedded in each plugin and checked by the loader, which
refuses a mismatch with a clear `AbiMismatch` error rather than risk passing a
differently-laid-out `Frame` or trait object across the boundary (undefined
behavior). Each loaded `libloading::Library` is held for the life of the process:
the registered factories are `fn` pointers into its mapped code, so dropping it
would be a use-after-free with no back-pointer to catch it. The whole path is
exercised out-of-tree by `g2g-plugins/tests/fixtures/example-plugin` +
`tests/plugin_loader_dlopen.rs`.

**Plugin ABI v2: the cross-toolchain tier (M1010).** The version lock above is
the price of passing Rust types across `dlopen`. v2 is the other trade: a frozen
`repr(C)` boundary (`g2g-plugin::abi`, header `g2g-plugin/include/g2g_plugin_v2.h`)
that carries a smaller surface but loads into a host built by a different
compiler, and can be written in C. The model is GStreamer's `gst_plugin_desc`: a
versioned descriptor plus vtables, hand-rolled rather than taken from
`abi_stable` (dormant) or `stabby` (a leaked heap vtable registry on stable
Rust). `async-ffi` supplies the one thing a hand-rolled C ABI cannot express, an
FFI-safe `Future` (`FfiPoll` / `FfiContext` / a three-pointer future struct), so
`process` stays backpressure-aware across the boundary.

- **The descriptor is data, not code.** A v2 plugin exports one *data* symbol,
  `g2g_plugin_v2_descriptor`, holding a magic, an ABI generation, and the list of
  element names and kinds it will register. The host reads and validates it with
  `dlsym` before calling any plugin function, which is what makes the capability
  gate meaningful: `load_plugin_with_policy` hands the declaration to a
  caller-supplied policy *before* the plugin gets control, and the default policy
  refuses a declaration carrying a capability kind this host does not understand.
  The declaration is then binding. The registrar stages elements rather than
  writing them into the `Registry`, checks each against the declaration, and
  commits only if every one matched: a plugin that registers three declared
  elements and one undeclared one contributes nothing.
- **What crosses.** `configure_pipeline`, `configure_output`, `process`,
  `set_property`, `get_property`, `destroy`, plus a `create` on the registration.
  Caps cross as a `repr(C)` tagged union over a frozen numeric code table (the
  host's caps enums are `#[non_exhaustive]`, so their discriminants can never be
  an ABI); property values likewise. Frames cross as pointer + length + an
  owner-side `free`, which maps exactly onto `SystemSlice::from_foreign`, so a
  frame moves in either direction without a copy.
- **What does not.** v2 elements are **System memory only**: the wrapper narrows
  `input_domains` to `System`, so a GPU-resident producer upstream gets a domain
  converter spliced in rather than a frame the plugin cannot read. GPU domains,
  and the ~50 exotic `AsyncElement` hooks (clock election, QoS, metadata
  propagation, the allocation cascade, the reverse-channel signals), stay v1 and
  host-native: the host-side wrapper element answers them with the trait
  defaults. The flag-set property kind and the tensor / KLV / closed-caption /
  sub-picture caps kinds do not cross either, and a registration that names one
  is refused rather than approximated.
- **Growing it.** Two mechanisms. `abi_version` gates the whole surface: a
  semantic change to an existing field bumps it. Inside one generation, every
  versioned struct carries its own `struct_size` and the host reads
  `min(plugin, host)` bytes into a zeroed local, so an older plugin's shorter
  vtable simply leaves the host's newer entries absent and the host uses its
  defaults; and trailing reserved fn-pointer slots let a future entry appear
  without the size changing, which an older host ignores.
- **Where v1 stays.** The loader probes the v2 symbol first and falls back to the
  v1 pair, so existing v1 plugins load unchanged. v1 remains the path for a
  plugin that needs the whole trait surface or GPU memory and ships alongside the
  host build it was compiled against.
- **The `fn()` slot table.** `LaunchFactory` builds an element from a
  context-free `fn()` pointer, and a v2 element's constructor needs to know
  *which* plugin vtable it belongs to. The host therefore keeps a fixed table of
  64 const-generic trampolines (`MAX_V2_ELEMENT_SLOTS`); past that a load is
  refused rather than silently dropping an element. Slots are never freed,
  matching the loaded-forever library.

**Security posture of the loader.** It defends against a *malformed* plugin, not
a *malicious* one, and the difference is worth stating plainly. `dlopen` runs the
library's initialisers before the loader reads a single field, and a loaded
plugin shares the host's address space with no boundary at all: it can make any
syscall the host can, read the host's memory, and ignore every rule in the ABI.
The capability gate decides whether to load a file and what it may register; it
cannot constrain what loaded code does. It is policy, not sandboxing. Anything
stronger (a separate process, seccomp, signature verification) is out of scope
and deliberately has no half-built stubs. What the loader *does* do is treat
every byte reachable from the descriptor as untrusted input, on the same rules as
a bitstream parser: bound every count before using it as a length, null-check
before dereferencing, UTF-8 check before a byte range becomes a `str`, restrict
element and property names to a `gst-launch`-safe character set, and refuse any
unknown discriminant instead of reinterpreting it. Two things it cannot check and
takes on the plugin's contract: that a pointer+length pair really addresses that
many readable bytes, and that a `struct_size` really matches what the plugin
wrote. The wrapper also asserts `Send` for a plugin instance under a documented
contract (the runner owns an element exclusively but may move it between
threads), so a thread-affine plugin is outside the ABI.

Exercised by `g2g-plugins/tests/plugin_loader_v2.rs` (a Rust plugin built with a
deliberately mismatched `g2g-core` feature set, which v1 refuses and v2 does not
care about) and `tests/plugin_c_abi.rs` (a plugin written in C, compiled against
the hand-written header, including a `sizeof` comparison of every ABI struct
against its Rust type so the two cannot drift).

**Hosted Python elements (`pyelement` / `pysrc` / `pyaggregator`, `g2g-python`).**
A gst-python-ml element shell runs as a first-class g2g element: `g2g-python`
embeds CPython (pyo3, `auto-initialize`), exposes a native `g2g` module the
`backend/g2g` package imports, and negotiates as a same-format passthrough. Each
hosted instance owns a dedicated GIL-holding OS thread; the element hands it the
frame and awaits the reply over a Waker channel, so the cooperative executor keeps
polling other arms while Python runs. A frame reaches Python without a copy on
either of two paths:

- **System memory.** `g2g_process(buf, width, height, fmt, meta)` gets a writable
  buffer-protocol object over the frame's own bytes, so `memoryview` / numpy read
  and overwrite pixels in place. The host counts outstanding buffer exports and
  fails the frame if the element kept a view past return (its pointer would dangle
  once the frame is freed downstream). `g2g_process_batch` and `g2g_produce` are
  the aggregator / source shapes of the same contract.
- **Payloads with no picture shape.** Audio into a transcriber, text into speech:
  the frame reaches `g2g_process_payload(buffers, caps, meta)`, and the element
  hands back buffers of its own through
  `meta.emit(payload, duration_ns=None, pts_ns=None)` instead of overwriting the
  one it read. Each emitted buffer inherits the anchor's timing unless it says
  otherwise: a streaming element gives every chunk its own `pts_ns` (usually the
  previous chunk's pts plus its duration) so the chunks play one after another,
  while outputs that run in parallel (the separation family's stems) leave it
  unset and share the anchor's. `g2g.PTS_NONE` (`FrameTiming::PTS_NONE`, §4.4)
  emits a buffer with no presentation time, which a sink presents on arrival.
- **CUDA device memory (M984).** A `MemoryDomain::Cuda` frame has no CPU bytes, so
  its two semi-planar planes are described to
  `g2g_process_cuda(luma, chroma, width, height, meta)` as `g2g.CudaPlane` objects
  exposing `__cuda_array_interface__` v3: luma `(height, width)` and interleaved
  chroma `(height/2, width/2, 2)`, byte strides carrying the producer's row pitch
  (so pitch != width is described, not repacked), `|u1` for NV12 and `<u2` for
  P010, `stream: None` (the CUDA domain carries no stream; a producer hands the
  frame over once the decode into it completed). `cupy.asarray(luma)` then aliases
  the decoder's surface with no PCIe round-trip. The `data` flag is read-only: the
  device memory belongs to the producer and a teed frame shares it under a
  read-only guarantee, with no copy-on-write to fall back on as the System path
  has. CAI carries no CUDA context, so the pointers are valid only in the context
  the producer decoded into, exposed as the plane's `cuda_context` property for a
  consumer that must push it (cupy and torch use the device's primary context).
  Plane lifetime is the call, enforced by a refcount check after it (a retained
  plane, including one a cupy array holds as its base or a consumed DLPack tensor
  holds as its manager context, fails the frame). An element that defines no hook
  for its shape gets `UnsupportedDomain` for a GPU frame rather than a silent
  readback: `g2g-python` links no CUDA, so a CPU-only element needs an explicit
  `cudadownload` upstream.
- **The batch and produce shapes (M986).** A GPU batch reaches a hosted aggregator
  as `g2g_process_cuda_batch(planes, width, height, meta)`, one `(luma, chroma)`
  pair per contributing input, so a batched detector reads every stream's decoded
  surface in place; the anchor flows on device-resident. A hosted *source* runs the
  handoff backwards: `g2g_produce_cuda(width, height, meta)` returns the two planes
  as any CAI-exporting objects (a cupy or torch allocation) or `None` for end of
  stream, because this crate links no CUDA and cannot allocate device memory
  itself. The returned planes are validated against the negotiated caps (shape,
  sample type, packed within each row, only the row pitch free, non-null pointer)
  before they become a frame, and the frame's keep-alive holds the Python objects
  so the memory outlives it; the source stamps timing, and reports its
  `cuda_context` through an optional attribute for a downstream consumer that must
  push it.
- **DLPack (M986).** The same plane also answers `__dlpack__` /
  `__dlpack_device__`, for the frameworks that prefer it (`torch.from_dlpack`,
  `cupy.from_dlpack`). It carries a device and stream contract CAI does not: the
  device is `(kDLCUDA, 0)`, since the CUDA domain carries the producing context but
  no device ordinal. A consumer asking for 1.0 or newer through `max_version` gets
  a `DLManagedTensorVersioned` capsule with the read-only flag set, one asking for
  nothing gets the pre-1.0 `DLManagedTensor`; `copy=True` or another `dl_device` is
  refused rather than silently ignored, and `stream` is ignored because the domain
  carries no stream. DLPack strides count elements rather than bytes, so a row
  pitch that is not a whole number of samples is refused instead of rounded. The
  capsule's destructor frees the tensor only while the capsule still carries the
  unconsumed name, since a consumer that takes ownership renames it and calls the
  deleter itself.

The frame is read where it lies and forwarded untouched, so a hosted transform
carries one memory domain on both pads (M985): System, or CUDA under
`cuda-frames=true`, the property that says the hosted class reads device memory.
Declaring the same domain on input and output keeps the relation honest, since the
domain a frame leaves in is the one it arrived in. Two things follow. The
domain-converter auto-plug splices a download / upload on the edge *into* the
element when upstream cannot deliver what the hosted code reads, and never after
it (previously a hosted element always claimed System output, so a GPU frame
passing through it drew a needless upload before a GPU consumer). And
`propose_allocation` names that domain upstream, so a multi-domain producer (an
NVDEC that can keep frames on the device or download them) settles on it and no
converter node is needed at all; the proposal constrains only the domain and the
frame size, since the element allocates nothing itself. `format` is a property
too, so a launch-built `pyelement` can accept the NV12 a decoder emits rather than
only its RGBA default.

One worker thread per element is the free-threading unit, and that is measured
rather than assumed (M988, the ignored `m988_gil_offload` test, whose module docs
carry the invocation for each interpreter). Four hosted elements each running one
compute-bound pure-Python callback recover 3.6x of the ideal 4x on free-threaded
CPython 3.14 (`sys._is_gil_enabled()` false in-process) and 0.9x on stock 3.14,
with no code change between the two: pyo3 picks the interpreter up at build time
(`PYO3_PYTHON`), free-threaded rules out `abi3`, and the whole crate's test suite
passes on both. The native `g2g` module declares `gil_used = false`, which is
load-bearing rather than decorative: CPython re-enables the GIL process-wide when
it imports a module that has not declared it, so without the declaration the
`import g2g` inside a hosted element drops the same measurement back to 0.9x. The
sizing consequence on a stock interpreter is that N hosted elements do not overlap,
so a chain's Python cost is the sum of their per-frame times rather than the slowest
one, and `link_capacity` on those links has to absorb the wait while the other
elements hold the GIL (see the `g2g-python` `host` module docs, next to the worker
design it follows from).

**Runtime scripting (`scriptelement`, `script-rhai` feature, M580).** The
construction scripts above run once to emit a graph; `scriptelement` is the
per-frame complement: a raw-video transform whose `process(frame)` is a Rhai
function, the pure-Rust cousin of the `pyelement` CPython host (§4.x). It
negotiates as a same-format passthrough (`DerivedOutput` constraint, like
`pyelement`), and on each `System`-memory frame hands the script a **zero-copy**
handle (`FrameBuf`, M581): the script indexes the live buffer in place
(`frame[i] = 255 - frame[i]`) and reads `frame.width` / `.height` / `.format` /
`.pts` / `.sequence` / `.len`, no bulk copy in or out. The copy-free path is a
custom-type receiver rather than a byte blob because Rhai clones a *value*
argument on entry (so a blob argument is copied regardless), while a custom type
is passed by reference; the handle reaches the buffer through an atomic guard
(pointer + length) armed for the call and nulled the instant it returns, so it is
`Send`/`Sync` with no `unsafe impl` and a handle kept past the call reads/writes
nothing (a clean error) instead of dereferencing freed memory. Per-pixel `frame[i]`
is interpreted (fine for logic / metadata / small regions), so whole-frame work
goes through native bulk methods (`invert` / `fill` / `apply_lut`, M582) the script
calls once and Rust loops at native speed, the control-plane / data-plane split.
Rhai is synchronous pure Rust, so the call runs inline on the pipeline thread (no
GIL, hence no worker-thread isolation the Python host needs);
the compiled `Engine` / `AST` / `Scope` are held on the element and are `Send`
under rhai's `sync` feature, so it runs under the multi-thread runner too. It is
registered by name, so `scriptelement script=... ! ...` parses in a launch line or
a declarative document. A GPU-resident frame yields `UnsupportedDomain` (a script
cannot touch device memory).

`scriptrouter` (M583) is the fan-out sibling: a Rhai-scripted routing demux (a
`MultiOutputElement` registered via `register_demux`, so `scriptrouter name=r
r.0 ! …  r.1 ! …` builds a 1-to-N node). Its `route(frame)` returns the output
port each `DataFrame` goes to: a single index (negative = drop), or an **array**
of indices to *multicast* one frame to several ports at once (a shared duplicate
per port via `Frame::share`, the same fan-out primitive a broadcast tee uses:
the buffer refcounts where the memory domain allows and deep-copies owned CPU
bytes, so the cost is honest). Control packets broadcast to every branch and the
runner broadcasts EOS, exactly like the built-in `Router` (which it is the
scripted analog of). It is the "route buffers into my own pipeline" seam: an
`appsink channel=…` on each output pad turns each route into a separate consumer
the app `pull()`s live while the pipeline runs (control plane in the script,
buffers moved natively, no interpreter on the data path; see the
`scriptrouter_appsink_egress` example). The `route` handle is read-only and
media-agnostic (routes audio / video / byte streams by `pts` / `sequence` /
`keyframe` / `len`, with a `frame[i]` byte peek for content routing), reusing the
`scriptelement` `FrameGuard`. Rhai is a sandboxed interpreter with no I/O or FFI,
so buffer *egress to an external system* stays the host's job (`appsink` + a
binding, or a native callback); the script decides routing, it does not perform
the handoff.

### 4.17 Containers and Byte Streams

A container demuxer splits one stored / transported byte stream into the typed
elementary streams it carries. The link feeding a demuxer is
[`Caps::ByteStream { encoding }`](crate caps), the first byte-stream caps variant:
an opaque container stream not yet demuxed, tagged with a `ByteStreamEncoding`
(e.g. `MpegTs`) so a demuxer accepts only the format it parses, the
byte-stream-level analog of the codec/raw video split. A byte source declares it
(`FileSrc::new(path, Caps::ByteStream{MpegTs})`), and the demuxer's transform
constraint maps it to the elementary stream type.

The MPEG-TS demuxer is the first: `g2g-plugins::mpegts::TsDemuxer` is a
pure `no_std + alloc` parser (sync 188-byte packets, PAT -> PMT -> elementary
streams, reassemble PES per PID into access units with PTS), and the `TsDemux`
element wraps it. The parser reassembles every elementary stream the PMT names;
the element has one output pad, so a `TsStream` selection (`H264` / `H265`
video as `CompressedVideo`, `Aac` audio as `Audio`, default H.264) picks which to
emit, and a second `tsdemux` selecting another stream demuxes the rest of the
multiplex. The selection is by codec, not a runtime-discovered "first video",
because the output pad's media type is fixed at negotiation before any packet is
parsed (H.264 and H.265 are distinct downstream decoders, not a refinement). Video
geometry is unknown until the bitstream parser reads the SPS, so the demuxer
advertises a fixatable placeholder `Range` refined downstream via `CapsChanged`
(the `RtspSrc` pattern, §4.13); AAC advertises the sentinel channels/rate that
`aacparse` refines from the ADTS header. The decode-side container precedent is
`Mp4Src` / `Mp4Sink`. The TS muxer (`g2g-plugins::mpegts::TsMuxer`) is the
inverse path, wrapping access units back into PES + 188-byte packets with
a real PSI CRC. It is multi-stream: `with_streams` builds one program
carrying N elementary streams, each on its own PID and named in one PMT, and
multi-program: `with_programs` takes a `(program_number, stream_type)` per
stream, so the PAT names each program, each program gets its own PMT (naming
only its own streams) and its own PCR on its first stream's PID. The fan-in
element exposes that layout as `prog-map`, one program number per input pad in
pad order (`prog-map=1,1,2`); GStreamer's `mpegtsmux` takes a pad-name structure
there, g2g a comma list because its properties are scalar. The
single-input `tsmux::TsMux` element wraps a one-stream muxer (`! mpegtsmux !`);
the multi-input `tsmuxn::TsMux` (a `MultiInputElement`) muxes A+V, interleaving
access units across inputs by PTS via the `take_earliest_by` merge so the
multiplex is decode-ordered. The `mpegtsmux` name is registered both as the
single-input launch element and as a fan-in muxer, so the text parser
picks `tsmux::TsMux` for one input and `tsmuxn::TsMux` for several by link degree
(`v.! m.  a.! m.  mpegtsmux name=m`), mirroring gst's request sink pads.

Tags ride TS through its standard carriers (M872): the muxers' `with_tags` /
`with_track_tags` write the SDT `service_descriptor` (service name from
`Tag::Title`, provider from the ffprobe-spelled `service_provider` key) and a
per-stream ISO-639 language descriptor in the PMT; the demuxers CRC-check and
parse both and post `BusMessage::Tag` / `StreamTag` on the `mpegts-pid-{pid}`
ids, ffmpeg-validated both directions. Nothing else rides TS: it has no
free-form tag element.

Service text is per program (M878): `with_program_tags(program, tags)` on the
fan-in muxer gives a `prog-map` program its own SDT entry, `with_tags` names
whichever programs do not, and a program's `Tag::Language` is the default for its
streams (global, then program, then track). The SDT describes the whole
multiplex, so a demuxer posts one `BusMessage::Tag` per service it names, each
carrying that service's `program_number`, whichever program the element routes.

AV1 rides TS on the same private PES (stream_type 0x06) the KLV carriage uses,
told apart by its `registration_descriptor` (M1049): AV1 has no `stream_type` of
its own, so a 0x06 stream is AV1 only when its PMT entry names one. The mux writes
the 'AV1G' format_identifier and the demux accepts that and the AOM spec's
'AV01', because only 'AV1G' has a reader: GStreamer's `mpegtsmux` / `tsdemux`
predate the spec and still call the mapping custom
(`enable-custom-mappings=true`), while ffmpeg's muxer writes AV1 with no
descriptor at all, a bare 0x06 not even its own demuxer identifies (it reports
`bin_data`). Each PES payload is one temporal unit in the low-overhead OBU format,
which `av1parse` and the AV1 decoders read unchanged, and `TsStream::Av1`
(`tsdemux stream=av1`) selects it; the seek resume point reads the AV1 frame
header rather than Annex-B start codes. Both directions are validated against
GStreamer: a `svtav1enc ! mpegtsmux` stream demuxes to units ffmpeg's `obu`
demuxer decodes at full size, and `tsdemux ! av1parse ! dav1ddec` decodes the g2g
mux's output.

The DVB EIT (PID 0x12) adds what a service is showing (M1049): the demuxers parse
the present/following table (`table_id` 0x4E, sections 0 and 1) and post each
service's `short_event_descriptor` name and text as a `BusMessage::Tag` scoped to
its `program_number`, under the `event_name` / `event_text` and
`next_event_name` / `next_event_text` keys (`Tag::Title` on that program is
already the SDT service name). Unlike the PAT / PMT / SDT this table changes
during the stream, so a section is read when its `version_number` differs from the
one last accepted for the same `(service_id, section_number)`, and a table
repeating itself costs nothing; unlike them a section routinely outgrows one
packet, so sections reassemble across packets behind a `table_id` filter that
keeps the far larger schedule tables sharing the PID out of the buffer. Event text
goes through the same annex A decoder as the SDT names, which now also decodes the
UTF-8 character table.

The TS stack also carries KLV metadata (STANAG 4609, the airborne-ISR profile of
MPEG-TS): `Caps::Klv` is the metadata elementary-stream caps (GStreamer
`meta/x-klv`), each frame one SMPTE ST 336 key-length-value packet. On the mux
side a `Caps::Klv` input becomes a private PES (stream_type 0x06, PES
`private_stream_1`) whose PMT entry carries the `KLVA` registration descriptor,
the MISB ST 1402 asynchronous carriage ffmpeg keys on; the demux side accepts
both that and metadata-in-PES (stream_type 0x15, the synchronous carriage) via
`TsStream::Klv` (`tsdemux stream=klv`), filtering generic 0x06 PIDs by the
registration the way Opus / DVB AC-3 selection does. `klv-sync` on the mux
elements selects the strict synchronous form instead: stream_type 0x15 on PES
`stream_id` 0xFC, each local set wrapped in one ISO 13818-1 metadata AU cell
(the 5-byte header ffmpeg's demuxer skips per ST 1402; layout cross-checked
against mediacommon's table 2-97 implementation), and a `metadata_descriptor`
(tag 0x26, 'KLVA') in the PMT entry, which measurement showed ffmpeg requires
to identify a 0x15 stream at all. The demux unwraps AU cells behind a
validation gate (cells must tile the payload exactly and each must open with
the ST 336 prefix) and forwards anything else raw, so both the strict and the
bare-payload sync forms decode. Above carriage,
`g2g-plugins::klv` is a pure `no_std` MISB ST 0601 UAS Datalink Local Set codec:
`UasDatalink` decodes / encodes the core telemetry tags (precision timestamp,
platform attitude, sensor position / FOV / relative angles, frame center) with
the standard's fixed-point scalings, BER lengths and BER-OID tags
bounds-checked, and the 16-bit sum checksum (tag 1) required on parse, so a
corrupted set is rejected whole. The `klvdecode` element turns each set into a
timed `Text{Utf8}` `key=value` line (`tsdemux stream=klv ! klvdecode !
textoverlay` overlays live telemetry); the encode direction is the
`UasDatalink::encode` API through an app source. Interop is validated against
ffmpeg both ways: ffprobe identifies the g2g mux's stream as `klv` and extracts
its bytes bit-exact, and a TS re-authored by ffmpeg's muxer demuxes back
bit-exact.

The tag table covers the practical ST 0601 core: telemetry angles and
positions, the identity strings (mission id, platform designation, image
source sensor, coordinate system), slant range / target width, the four offset
corner points, target location, and the nested MISB ST 0102 security local set
(tag 48) as a typed `SecurityLocalSet` (classification enum preserved even for
unknown codes, country coding methods, classifying / object countries), every
scale factor cross-checked against the independent klvdata implementation and
the whole parser validated against the published MISMMS reference packet.
That packet's declared checksum is provably wrong (0xAA43 declared, 0x3E1E
actual, klvdata's own sum agrees), which is why `parse` (strict, the
`klvdecode` default) is paired with `parse_lenient` and a `verify-checksum`
property: real encoders get checksums wrong, and the caller chooses whether
that drops the set. KLV also rides RTP directly (RFC 6597, `rtpklv`): a
sans-IO `RtpKlvPacketizer` / `RtpKlvDepayloader` pair mirroring the H.264
`rtppay` / `rtpdepay` shape, 90 kHz timestamps, MTU fragmentation with the
marker bit closing each KLVunit, and whole-unit discard on any lost fragment
(a fragment carries no unit header, so resync waits for the next marker).

Around that core sit the rest of the STANAG 4609 pieces. `vmti` is the MISB ST
0903 moving-target set (ST 0601 tag 74, nested with no UL or checksum;
standalone it carries both), decoding the VTarget series with ST 1201 IMAPB
scaling along with each target's nested VMask (pixel polygon / run-length
mask), VObject (ontology class), VTracker (track id, life cycle, velocity and
acceleration packs) and VChip (image chip) sets, and `vmti_from_analytics`
turns a frame's `AnalyticsMeta` detections into VTargets (a tracked detection
carries its `object_id` as the target id), so an in-pipeline detector emits
standards-compliant VMTI. The ST 1204 MIIS core identifier (tag 94)
round-trips exactly, refusing rather than half-reading an identifier it cannot
reproduce, and renders the standard text form (grouped hex UUIDs with the
Appendix B permutation check value). `misptime` puts MISB ST 0604 microsecond
timestamps in an H.264 / H.265 SEI so video frames and KLV correlate after a
remux; extraction emits text cues rather than restamping PTS, since an absolute
epoch time on a frame would read as decades of lateness to every sink.
`cotsink` maps decoded telemetry to Cursor-on-Target XML for TAK / ATAK, one
event per local set with the platform as the point and the frame center as a
`<sensor>` cone; with `spi=true` it also emits the ST 0805.1 Sensor Point of
Interest event (`b-m-p-s-p-i` at the target location or frame center, linked
to the platform track by `<link relation="p-p">`, jmisb's `KlvToCot`
conventions). `st2022fec` is SMPTE 2022-1 (Pro-MPEG COP3) FEC for TS over
RTP: the wire format only, since the 2D row / column XOR algebra and the
receiver bookkeeping now live once in `ulpfec` and serve `flexfec` and this
alike. It derives each repair's protected set from that repair's own
SNBase / offset / NA rather than learning global L / D, so a mid-stream
geometry change still decodes, and it refuses a repair whose type field is not
XOR instead of applying the wrong algorithm to it.

Every wire format here was verified against a primary implementation rather
than prose: jmisb for ST 0903 / ST 1204 / ST 0805, GStreamer's `video-sei`
parser plus a real capture vector for ST 0604, FFmpeg's `prompeg` and
GStreamer's `rtpst2022-1-fecenc` (which agree field for field) for ST 2022-1,
and MITRE's CoT schema with pytak's constants for CoT. Where a field could not be
confirmed, the codec preserves the raw bytes or declines to emit rather than
guessing: the VTarget location pack's accuracy tail is kept verbatim, and the
unconfirmed CoT detail elements are simply not written.

The Matroska / WebM demuxer is the second, the same parser + element split
keyed on `Caps::ByteStream{Matroska}`. `g2g-plugins::matroska::MatroskaDemuxer` is
a pure EBML parser (variable-length element IDs / sizes, descend into the Segment,
read Tracks for the elementary streams and `Info` TimestampScale, parse each
Cluster's SimpleBlock / Block frames with scaled timestamps), and `MkvDemux` wraps
it with the same per-codec `MkvStream` selection (H.264 / H.265 / VP8 / VP9 / AV1
video, AAC / Opus audio, default VP9). A `S_TEXT/UTF8` subtitle track is also
read: it maps to `MkvCodec::Subtitle(Utf8)` and fans out of `MkvDemuxN` as a
`Caps::Text { Utf8 }` port (`MkvStream::Subtitle`), with the cue's display window
carried on the frame, the `BlockGroup`'s `BlockDuration` scaled onto
`MkvFrame.duration_ns` (a `SimpleBlock` leaves it `0`); `S_TEXT/ASS` and
`S_TEXT/WEBVTT` are likewise de-framed to plain `Text{Utf8}` cue text (via the
`CodecPrivate` header), and `mkv_playbin` auto-plugs the subtitle overlay
(§4.18). `S_VOBSUB` is the bitmap case: `MkvCodec::VobSub` ->
`Caps::SubPicture { VobSub }` (`MkvStream::VobSub`), whose blocks are forwarded
verbatim as subpicture units after the track's `.idx` `CodecPrivate` goes out in
band ahead of them (§4.18). Unlike `TsDemux`,
Matroska's Tracks element carries concrete geometry and audio parameters, so the
demuxer refines the output caps itself via `CapsChanged` once Tracks is parsed,
without a downstream bitstream parser. An H.264 / H.265 track's blocks are
converted from the container-native AVCC / HVCC length-prefixed framing
(declared by the `avcC` / `hvcC` `CodecPrivate`, whose `lengthSizeMinusOne`
sets the prefix width) to the Annex-B framing the pipeline assumes, with the
config record's parameter sets prepended on keyframes (M766, ffmpeg's
`h264_mp4toannexb` discipline); the whole-block length walk is validated
exactly, so a nonstandard Annex-B block passes through unchanged instead of
being mis-framed. WebM (the VP8/VP9/AV1 + Opus subset) is the browser-delivery motivator. Block
lacing (Xiph / EBML / fixed) is split, so multi-frame audio blocks demux.
The `Cues` index is parsed into a time -> Cluster-byte-position map
(`cue_seek_offset`), and `MkvDemux` seeks through it in three tiers
(`poll_seek`): with `Cues` parsed it byte-seeks straight to the target Cluster
(`DemuxSeek::poll_request_indexed`), keeping Tracks / TimestampScale across the
mid-segment landing (`reset_keeping_tracks`); with only a `SeekHead` locating an
end-of-file `Cues` it prefetches them first (a byte-seek to `Cues`, parse,
then `begin_indexed_seek` to the target Cluster, the internal prefetch flush
consumed so downstream sees one only on the real seek); with neither it re-scans
from offset 0. (`CueClusterPosition` / `SeekPosition` are relative to the
Segment data start, which the parser tracks.) The MKV muxer (`matroskamux`: `MatroskaMuxer` + the
`MkvMux` element) is the inverse path, writing the EBML header, an
unknown-size Segment, Tracks, and one Cluster per frame, with the `webm` DocType
for the WebM codec subset. Scope is one Segment / one track with definite-size
Clusters (multi-track A/V muxing is the sibling `mkvmuxn`). Both muxers also
have a `seekable` (two-pass) mode (M770): the element buffers the file and
finalizes it at EOS with a front `SeekHead` (fixed-layout entries indexing
Info / Tracks / Chapters / Tags / Cues, the Cues position patched in place once known), so
the file seeks from byte 0 without reading past the Clusters; mutually
exclusive with `streamable`, and the default streaming output is unchanged. The
same finalize fills an `Info` `Duration` reserved beside them (M794): the value
is the highest block end across tracks, each block's timestamp plus its own
duration rounded to a `TimestampScale` tick, which is how ffmpeg arrives at the
number it writes, so a remux reports the length ffmpeg's own file does. Only
this mode can carry a duration at all, since a streaming caller has emitted its
header long before the total is known, and a live stream has no length to
declare.
The Segment `Tags` element carries metadata in both directions, per file and per
track (M787): a `Tag` whose `Targets` names a `TagTrackUID` scopes to that track,
and a nested `SimpleTag` flattens to a `parent/child` key. The muxers take
per-track metadata from `with_track_tags` and write each track's `TrackUID` in
its `TrackEntry`; the demuxers map a parsed UID back to its track and post the
tags as `BusMessage::StreamTag` on that stream's collection id, leaving
untargeted tags on `BusMessage::Tag`. A track's title and language are the
exception: they live in the `TrackEntry` itself as `Name` / `Language`
(`LanguageBCP47` preferred when both are present), which is where ffmpeg writes
them and where a player reads them, so the muxers route `Tag::Title` /
`Tag::Language` there instead of writing a `SimpleTag`, and the demuxers merge
both sources into one `StreamTag` per stream (M788). A missing `Language` stays
absent rather than becoming the spec's implicit `eng`.

Chapters (the table of contents, GStreamer's `GstToc`) travel the same
out-of-band route in both containers (M1046). `g2g_core::Chapter` is the shared
shape: a stream-time start in nanoseconds, an optional end, a title, an optional
language, and nested sub-chapters. A demuxer posts what it parsed as
`BusMessage::Chapters` (once, like the tags), so an application builds a chapter
menu and seeks to a start without touching the data path; a muxer takes the same
list through `with_chapters`. Matroska is the container that holds the whole
shape: the `Chapters` element's `EditionEntry` / `ChapterAtom` tree, whose times
are unscaled nanoseconds rather than `TimestampScale` ticks, with nesting and a
per-chapter `ChapLanguage`. The muxers write one default edition; the demuxer
skips a hidden edition or atom, since it is not meant to reach a menu, and bounds
both the nesting depth and the chapter count because the file supplies them. MP4
carries less: the reader prefers the QuickTime chapter *text* track (a media
`trak` points at it with `tref/chap`, and its samples are the titles timed by its
own sample table, so each chapter gets an end) and falls back to the Nero
`udta/chpl` list, which is a flat array of starts in 100 ns ticks with no ends
and no nesting. The writers emit `chpl` only, in the version-1 shape ffmpeg's
`mov` muxer produces, so a g2g-written MP4 round-trips titles and starts but
reports the chapters open-ended.

The Ogg demuxer is the third, the same parser + element split on
`Caps::ByteStream{Ogg}`. `g2g-plugins::ogg::OggDemuxer` parses RFC 3533 pages
(sync to "OggS", frame packets via the segment-table lacing with cross-page
reassembly, sniff the codec from the first packet's `OpusHead`, skip the setup
headers), and `OggDemux` emits the Opus audio packets as `Caps::Audio{Opus}` with
the channel count refined from `OpusHead`. The container is auto-detectable
(`typefind` "OggS", `filesrc bytestream-format=auto`).

Grouped multi-stream Ogg is handled per serial (M790). A file opens with one
beginning-of-stream page per logical bitstream before any other page (RFC 3533
§4), and the parser keeps an `OggLogicalStream` for each: its own codec mapping,
headers, packets and granule anchors. A serial joins only from a page in that
opening block (or as the first stream seen when the file was joined mid-stream,
which a byte-seek does), so a beginning-of-stream page arriving later, meaning a
**chained** physical stream, is ignored rather than misparsed; the concurrent
stream count is capped, since the serials come from the file. `OggDemux`
forwards the first bitstream whose codec matches its `stream` selection and
drains the rest; `OggDemuxN` is the multi-output form, one port per `OggPort`
naming the bitstream it carries. Routing is positional rather than codec-keyed
because two streams of one codec in a file is ordinary. The per-stream caps,
in-band codec config and packet timing are one shared `StreamEmitter` both
elements drive, and `OggDemuxN` announces the file's `StreamCollection` and
posts each stream's VorbisComment as a `StreamTag` under the same ids.

Opus decode applies the two container trims so the PCM sample count matches
ffmpeg / gstreamer (RFC 7845). Pre-skip is codec config the decoder owns:
`OggDemux` forwards the `OpusHead` in-band (the g2g parameter-set convention) and
`OpusDec` reads its pre-skip (offset 10) and drops that many leading output
samples. End-of-stream padding is container knowledge only the demuxer has:
`OggDemux` tracks the running decoded sample count against the final page's
granule position and marks the closing packet(s) short via `duration_ns` (fully
padded packets are dropped), which `OpusDec` honors as a per-frame keep count.
Both trims are attacker-controlled inputs, folded with saturating math (an
oversized pre-skip trims a frame to nothing, an underflowing granule drops it).
A stream with no `OpusHead` and no per-frame duration (the RTP path) decodes
untrimmed, matching gstreamer's SDP-less default.

Vorbis carries the same two facts without a header field: its playable length is
the final page's granule position, and its head trim is the first audio packet,
which primes the overlap window and decodes to nothing. `OggDemux` clips that
priming block off the front and clamps the tail to the end granule, so a decode
yields exactly the granule's worth of samples. The size of the clip comes from
the first audio page's granule (its shortfall against the natural packet
durations, which also covers a stream joined mid-file), except when that page is
also the last: its granule is then the end of the stream and says nothing about
the head, so the clip is the priming packet's own `blocksize / 2`.

MP4 carries the same two facts in its own spelling, and both directions convert
to the in-band convention (M791). The `dOps` OpusSpecificBox holds an
`OpusHead`'s fields big-endian, so the demuxers rebuild one from it and forward
it ahead of the audio (`opusparse::opus_head_from_dops`, validated field by
field: unknown version, zero channel count or truncated channel-mapping table
leave the track configless rather than failing the file); the end trim is the
final sample's short `stts` / `trun` duration, which arrives as `duration_ns`
like the Ogg granule trim does. `Mp4MuxN` is the inverse: an in-band `OpusHead`
is consumed as config and becomes the `dOps` (`dops_from_opus_head`), so a remux
keeps the source's pre-skip, output gain and channel mapping byte for byte,
while a freshly encoded stream (`OpusEnc` emits raw packets, no header, so its
RTP consumers are unaffected) falls back to libopus' 312-sample lookahead. The
Opus `trak` also carries the `edts`/`elst` the Opus-in-ISOBMFF binding requires,
`media_time` = pre-skip.

Which duration a reader then reports depends on the layout, so `Mp4MuxN` writes
both (M793, the `fragmented` property, default `true`). Fragmented is the
streamable one: `ftyp`+`moov` up front, a `moof`+`mdat` per fragment, empty
sample tables and zero header durations. ffmpeg derives such a file's duration
by summing the `trun` sample durations and applies an edit list only as a
timestamp shift, so an Opus track reports the media span with a negative
`start_time` and `segment_duration` is written `0` (the total is unknown when
the `moov` goes out). Progressive is the two-pass one, the shape `matroskamux`'s
`seekable` mode has: every sample is buffered, then `ftyp` + a single `mdat` +
a `moov` are emitted together at EOS, with real `stts` / `ctts` / `stss` /
`stsc` / `stsz` / `stco` tables (one sample per chunk, ordered by decode
timestamp) and real `mvhd`/`tkhd`/`mdhd` durations. That decode timestamp is the
frame's own: `Mp4DemuxN` reads the source's `ctts` and carries `dts_ns` beside
`pts_ns`, so a reordered (B-frame) stream's composition offsets survive a remux
(M972). A frame with no decode timestamp of its own, or one past its PTS, which
`ctts` version 0 cannot express, decodes when it presents. That is enough for ffmpeg to
apply the edit, so the reported duration is the trimmed presentation length
exactly. The cost is holding the movie in memory, so a live or long capture
wants the fragmented default. GStreamer spells this choice `fragment-duration =
0`, which g2g already spends on "one fragment per access unit", so the layout
gets its own boolean rather than a silently redefined property.

Compressed audio negotiates with the `0/0` "unknown until parsed" caps, so
`Mp4MuxN` adopts the concrete channel count and rate from the runtime
`CapsChanged` a demuxer emits, while the `moov` is still unwritten; without it a
remuxed audio track would declare a zero `mdhd` timescale.

Matroska is the third spelling of the same two facts, and converts to the same
in-band convention (M792). The pre-skip is the `CodecPrivate` `OpusHead` itself,
which the demuxers forward ahead of the audio (validated as a real header first),
plus a `CodecDelay` on the TrackEntry, the ns form of the same count, which
`MkvMuxN` derives from the header it is about to write and pairs with the
mapping's fixed 80 ms `SeekPreRoll`. Block timestamps are not shifted: the first
Opus block sits at 0 and `CodecDelay` tells the decoder what to discard, so a
Matroska file starts at zero where the MP4 edit list makes `start_time` negative.
The end trim, with no granule to carry it, is the final block's `DiscardPadding`,
which is nanoseconds and so survives the millisecond `TimestampScale` grid that
`BlockDuration` alone would round away (a 6.5 ms tail becomes 7); the muxer
writes both, as ffmpeg does, and the demuxer lets the ns element win. Because the
packet's own length is needed to turn a tail discard into a kept duration, the
conversion reads the Opus TOC byte and applies to Opus only.

The Ogg muxer (`oggmux`: `g2g-plugins::ogg::OggPageWriter` + the `OggMux`
element) is the inverse path, on the same three mappings (M789). The writer
laces packets into pages (255-byte segments, continuation pages past the
255-segment limit, the RFC 3533 CRC-32 with polynomial `0x04c11db7` and no
reflection), holding the last packet back so the end-of-stream flag always has a
page to ride. Codec config arrives in-band, the same convention the demuxers
emit: an `OpusHead` / the three Vorbis headers / the native `fLaC` block are held
until the first audio packet, then written as the beginning-of-stream page plus
one following page at granule 0, so audio starts on a fresh page as the mappings
require. Vorbis is remux-only (no encoder), and the Ogg-FLAC first packet is
rebuilt around the source STREAMINFO with the mandatory VorbisComment appended.
Granule positions come from each mapping's own sample count (Opus TOC durations,
FLAC block sizes, the lapped `(prev + cur) / 4` from the `VorbisTiming` mode
tables, the inverse of the demux-side durations), held to the total `duration_ns`
the input declared when every packet is timed. That last bound is what carries a
source's end-of-stream trim through a remux, so ffmpeg decodes a remuxed Opus /
Vorbis / FLAC stream to the source's samples bit for bit.

`oggmuxn` is the fan-in form (M790): one `OggStreamMux` per input pad, each its
own logical bitstream with its own serial, packets interleaved by PTS through
the same `InputAggregator` merge the other multi-track muxers use. Grouping
forces the page order, which is why the per-stream header writing is split in
two: every stream's beginning-of-stream page is written first, in pad order,
then each stream's remaining header pages, then the data pages. That block goes
out when the merge first releases a packet, the first moment every pad's in-band
codec config is known to have arrived. Like `mpegtsmux`, the one name `oggmux`
covers both shapes, the parser picking by link degree.

An audio decoder fixates its output caps even when a demuxer only knows the
channel count once it parses the stream: it advertises `PcmS16Le` at a concrete
rate with the `ANY_CHANNELS` placeholder (fixated to stereo for the negotiated
edge), and the real count arrives via a `CapsChanged` (`OpusDec` (re)builds
libopus for it, since the decoder is per-channel-count). So a decode-to-PCM line
negotiates before the count is known. `AudioConvert` is caps-driven like
`AudioResample`: a bare `audioconvert` takes its output format / channel count
from a downstream capsfilter (a mono `channels=1` pin) and otherwise passes the
input through. Its channel mixing is position-aware for multichannel (either
side > 2): speaker positions come from `g2g_core::ChannelLayout::default_for`,
the per-count layout convention (the ffmpeg default-layout table, which is the
order the decode path interleaves), and the mix matrix applies the ITU
BS.775-style coefficients (center and surrounds fold at 1/sqrt(2), back center
at 0.5 into each front, LFE dropped, normalized against clipping), verified
coefficient-for-coefficient against ffmpeg's default rematrix; upmix places
each input at its own speaker and leaves the rest silent. Counts past the
layout table (> 8) fall back to the layout-agnostic round-robin fold so no
channel is silently dropped.

The FLV demuxer is the fourth, on `Caps::ByteStream{Flv}`.
`g2g-plugins::flv::FlvDemuxer` parses the flat FLV tag stream (the "FLV" header,
then `PreviousTagSize` / tag pairs, each tag's 11-byte header framing its body),
and `FlvDemux` forwards the H.264 (AVC) video and AAC audio media access units
with their millisecond timestamps (PTS from the video tag's signed
composition-time offset, DTS from the tag header), selected per `FlvStream`
(h264 | aac, default h264) like `TsDemux`. The sequence-header tags are the
codec-config side channel (M662): the parser retains the
`AVCDecoderConfigurationRecord` / `AudioSpecificConfig`, and the element uses
them the way the MP4 demuxers do, re-framing the AVCC access units to Annex-B
(honouring the `avcC` NAL length-prefix width) with the SPS/PPS prepended
in-band to the first access unit, ADTS-framing the raw AAC so the audio is
self-describing, and announcing the concrete channel layout / sample rate via
`CapsChanged`, so both extracted elementary streams decode standalone
(ffmpeg-oracle-validated, both directions, in the CI conformance job). The
`onMetaData` script tag posts as bus tags. The container is auto-detectable
(`typefind` "FLV", `filesrc bytestream-format=auto`). The FLV muxer (`flvmux`:
`g2g-plugins::flv::FlvMuxer` + the `FlvMux` element) is the inverse path: like
`FlvMuxN` it captures the decoder config in-band from the first access unit
(parameter sets from the IDR / the first ADTS header) and writes it as the
track's sequence-header tag, re-framing video Annex-B -> AVCC (keyframes
flagged from the IDR NAL) and audio de-ADTS'd, so a single-track `flvmux`
output is a playable FLV (what `RtmpSink` publishes). With MP4
(`Mp4Src`/`Mp4Sink`), MPEG-TS, Matroska/WebM, Ogg, and FLV, the demux/mux
coverage spans the major containers.

The MPEG program stream demuxer (`mpegpsdemux`, `Caps::ByteStream{MpegPs}`) is
the `.mpg` / `.vob` read path: VCD-era MPEG-1 program streams and DVD MPEG-2 ones
through one element (M929). `g2g-plugins::psdemux::PsDemuxer` syncs to pack
headers (`00 00 01 BA`, both the MPEG-2 `01`-marker layout with its stuffing and
the flat 8-byte MPEG-1 one) and reads the PES packets between them; `PsDemux` /
`PsDemuxN` wrap it with a `PsStream` selection (`Mpeg2` video, `Mp2` audio, `Ac3`,
`SubPicture`). Two things make it unlike `TsDemux` rather than a copy of it.

There is no PAT/PMT: a stream is identified by its PES `stream_id`
(0xE0..=0xEF video, 0xC0..=0xDF audio) and, for the `private_stream_1` (0xBD)
that DVD carries AC-3 and subpictures on, by the substream id byte opening its
payload (0x80..=0x87 AC-3 behind a 4-byte DVD substream header, 0x20..=0x3F
subpicture). Streams are therefore *discovered* by observing packets, so the
`playbin` / `decodebin` probe hooks report what the probe window has actually
shown rather than reading a table, and geometry comes from the video's own
sequence header (`00 00 01 B3`), which the demuxer parses to fix the video caps
via `CapsChanged`.

And a PES payload is not an access unit. A program stream cuts its packets on
sector boundaries with no regard for picture boundaries, so one packet can hold
the tail of a picture and the head of the next; feeding those to a decoder
verbatim desynchronizes it. The demuxer therefore reframes the video on its own
start codes (a unit runs from one picture header, with any sequence / GOP header
opening it, to the next); a PES timestamp names the first access unit commencing
in its packet and stamps exactly that unit. That is the job an elementary-stream
parser does for the other codecs, kept here because it is program-stream-specific:
MPEG-TS needs none of it. Audio and AC-3 are self-syncing and are grouped per
timestamped packet instead, matching what `TsDemux` emits.

A DVD stamps roughly one PES packet per GOP, so most pictures arrive with no
timestamp of their own; carrying the last stamp forward gave a dozen frames one
shared PTS, which a pacing sink plays as a burst then a freeze (the M934 bug,
found on a real disc as "stutters every half second"). The demuxer instead
synthesizes each unstamped picture's PTS as `gop_base + temporal_reference *
frame_period`: the picture header's `temporal_reference` is its display index
within the GOP, so the arithmetic stays exact across B-frame reordering, a real
PES PTS re-anchors the base (drift never outlives a GOP), and an unstamped GOP
header advances it by the span of the GOP just closed. Unstamped DTS advances
one frame period per picture in coded order.

Subpicture units span several PES packets and declare their own total size in
their first two bytes, so they are reassembled by size (bounded by the 16-bit
maximum that size field can state) and stamped with the opening packet's PTS and
the unit's own hide time as duration. A program stream carries no palette, so
the pad opens on a synthesized `.idx` holding only the video's `size:` line and
`VobSubDec`'s default palette renders the cues (§4.18). Out of scope: LPCM and
DTS substreams, the program stream map, a PS muxer, and seeking.

`VideoCodec::Mpeg2` covers MPEG-1 and MPEG-2 video as one codec, libavcodec's
`MPEG2VIDEO` decoder playing both; MPEG-TS stream types 0x01 and 0x02 map to it
through `TsStream::Mpeg2`, and `au_is_keyframe` reads its sync points (an
I-picture, or a sequence / GOP header).

Disc content is usually interlaced, and presenting its woven frames as-is combs
on motion. `deinterlace` (M932) is the CPU filter that undoes it: a single-rate
yadif port (an edge-directed spatial interpolation clamped to a temporal window
built from the previous and next frames), plus the cheaper `linear` and `blend`
methods, over I420 / NV12 / RGBA / BGRA at unchanged format and geometry, one
frame out per frame in. It is bit-exact against ffmpeg's `yadif=0` on the same
raw frames, including the 3-column border where ffmpeg drops the directional
search. Field order is assumed top-field-first (ffmpeg's default for a stream
that declares nothing).

Interlacing is signalled in the caps (M935): `Caps::RawVideo` carries an
`Interlace` field (`Any` / `Progressive` / `Interleaved`), where the `Any`
wildcard intersects with anything, survives `fixate`, and reads as "progressive
unless declared", so the field never blocks a solve and nearly every caps site
just states `Any`. `FfmpegVideoDec` reads libavcodec's per-picture interlaced
flag and latches `Interleaved` output caps on the first interlaced picture
(sticky for the stream, so telecine content cannot flap `CapsChanged`), covering
interlaced MPEG-2 over any container and interlaced H.264 alike. The element's
`mode` property acts on that declaration: `interlaced` (the default) always
weaves, the pre-M935 contract for hand-written lines whose upstreams declare
nothing; `auto` weaves only a caps-declared `Interleaved` stream in a format the
kernels handle and otherwise forwards packets untouched, with negotiation kept
transparent (any raw video passes) so inserting it never narrows a branch; and
`disabled` is a pure passthrough. Every `playbin` video branch (mkv / mp4 / TS /
PS / HLS, plain fan-out and the subtitle / closed-caption / DVD-subpicture
overlay variants) inserts `deinterlace mode=auto` after the decoder, GStreamer's
playbin `deinterlace` flag parity: a progressive stream pays only a forwarding
hop, and the M932 container-probe decision (`progressive_sequence` in the MPEG-2
sequence extension) is superseded by the decoder's own per-picture report.

Adaptive streaming sits one layer above these demuxers: an HTTP byte source feeds
a playlist/manifest-driven source that fetches media segments and hands them to
the matching byte-stream demuxer. `g2g-plugins::httpsrc::HttpSrc` (the `http-src`
feature, `reqwest`) GETs a URL and streams the body as `Caps::ByteStream` chunks,
the fetch layer the others share. It owns the network-buffering story
(`prebuffer-bytes` + `with_bus`, the queue2 analog since g2g has no queue
element): when set, `run` fills a bounded byte window before pushing downstream,
posting `BusMessage::Buffering` percent on quartile transitions, streams through
while topping the window up without waiting, and re-enters buffering on a
mid-stream underrun (window empty, network not ready), so an application can
pause until `100` and show a buffering indicator on a stall. The window never
grows past the target, and `0` (the default) streams straight through. The
segment loops (`hlssrc` / `dashsrc`) carry the duration-keyed sibling
(`prebuffer-ms` + `with_bus`, M819): a `segprebuf::SegmentPrebuffer` window the
loop fetches into while below its duration target (summed `#EXTINF` / MPD
segment durations) and emits from otherwise, posting the same quartile
`Buffering` levels during the startup / post-seek fill and staying silent in
steady state; init segments ride the window with duration 0 so an ABR re-init
stays ordered behind queued media, and a flushing seek clears the window and
re-arms the fill. Because a manifest/segment URL is
attacker-controlled, the shared `fetch::get_bytes`/`get_text` never buffer an
unbounded body: each accumulates the response chunk-by-chunk against a cap
(`MAX_MANIFEST_BYTES` 16 MiB for playlists/MPDs/keys, `MAX_SEGMENT_BYTES` 256 MiB
for one media segment), failing loud when an honest `Content-Length` or the
streamed running total exceeds it, so one oversized reply cannot exhaust memory.
`hlssrc::HlsSrc` (`hls`) parses an RFC 8216
`.m3u8` (the pure `no_std` `hls` parser: master variants for bandwidth-capped ABR,
media segments), selects a variant, and streams its segments, MPEG-TS into
`tsdemux` or fMP4/CMAF (signalled by `#EXT-X-MAP`, probed at negotiation) as
`ByteStream{IsoBmff}` into `fmp4demux`. A no-ENDLIST live playlist starts near the
live edge (`live_edge_start`: ~3 target durations from the end per RFC 8216
§6.3.3, so playback follows what is being published rather than replaying the
stale front of the sliding window, clamped to the window start for a short window;
`with_full_replay()` opts back into starting from the window front for a DVR
replay), then reloads on an interval, playing each new segment once by media
sequence. With `with_abr()`
 it is throughput-adaptive: a shared `abr::BandwidthEstimator` keeps an EWMA
of measured download throughput (bytes over elapsed `monotonic_ns`) and yields an
effective bandwidth cap (estimate scaled by a safety factor, bounded by
`max-bandwidth`); the run loop feeds that cap to the existing `MasterPlaylist`
selection, re-picks the best-fitting variant after each segment, and on a change
swaps the active media playlist and re-emits the init, keeping the time-aligned
segment index. Off by default (a fixed up-front variant). Single-file CMAF is
supported through `#EXT-X-BYTERANGE` (and `#EXT-X-MAP`'s `BYTERANGE`): a segment
carrying one fetches only its sub-range with an HTTP `Range` request, the
offset continuing from the previous sub-range of the same resource when the tag
omits an explicit `@offset`; a server that ignores the `Range` (replies `200`)
is handled by slicing the requested window from the full body.
`#EXT-X-KEY:METHOD=AES-128` segments are decrypted in place (AES-128-CBC via
`aes`/`cbc`, key fetched from the key URI and cached, IV explicit or derived from
the media-sequence number). `METHOD=SAMPLE-AES` encrypts only the media samples
inside the container, so it is handled after demux by the
`sampleaesdecrypt::SampleAesDecrypt` transform (`tsdemux ! sampleaesdecrypt !
h264parse`): per the Apple TS sample-encryption format it AES-128-CBC decrypts
H.264 slice NALs (32-byte clear leader, 16-encrypted / 144-clear pattern,
emulation-prevention aware, IV reset per NAL) and AAC ADTS frames (ADTS header +
16 clear bytes, then whole-block CBC). The key/IV reach it either configured
directly or, in the HLS chain, auto-wired: `HlsSrc` fetches the `#EXT-X-KEY`
material and publishes it into a shared key handle the decryptor reads, forwarding
the sample-encrypted segments undecrypted (the demuxer needs the clear framing).
For fMP4/CMAF, SAMPLE-AES maps to the `cbcs` Common Encryption scheme
(ISO/IEC 23001-7), handled inside `fmp4demux`: the init segment's `encv`/`sinf`/
`tenc` give the crypt:skip pattern (1:9 for video) and constant IV, each fragment's
`senc` gives the per-sample clear/protected subsample ranges, and the protected
ranges are AES-128-CBC decrypted (IV reset per subsample, chaining over the
encrypted blocks only) using the same shared key handle `HlsSrc` fills. The
sibling schemes decrypt through the same machinery: `cenc` (whole-range AES-CTR),
`cbc1`, and `cens` (M867: pattern AES-CTR, one IV per sample with the counter
advancing only over encrypted blocks, pattern restarting per protected range).
Sample groups can re-key mid-fragment: a `traf` `sbgp`/`sgpd` (`seig`) overrides
the `tenc` defaults per sample run, and M867 added the movie-level `seig` table
(indices below 0x10001 resolve against the track's `stbl` table, fragment-local
ones above it, per 14496-12, strictly scoped). A subsample map that overruns its
sample is an error, never a partial decrypt. A clear
track stays a normal demux; an encrypted track with no key fails loud.
`hlssink::HlsSink` (`std`) is the publishing side (M896): it cuts the byte stream
a muxer feeds it into media segment files and writes an `.m3u8` media playlist
beside them, rendered by the `hls` parser's `write_media` twin. The muxer stays a
separate element (`... ! tsmux ! hlssink`, `... ! mp4mux ! hlssink`), so one sink
packages either carrier. A segment may only start at a keyframe and closes at the
first one at or past `target-duration` (`0` cuts at every keyframe): for MPEG-TS
one input frame is one access unit, so `FrameTiming::keyframe` marks the
candidates and the frame PTSs give the durations; for fMP4 the stream is walked as
boxes, a `moof` whose first sample is a sync sample opens a fragment and is a
candidate, and the `trun` durations in the track timescale give the exact segment
length, with `ftyp`+`moov` split off once into the `#EXT-X-MAP` init segment.
Nothing is added or dropped, so the init segment plus the media segments
concatenate back to the muxer's own byte stream. `playlist-length` bounds the
listed window (advancing `#EXT-X-MEDIA-SEQUENCE` as segments roll off) and
`max-files` deletes the files that leave it, which is the live case; EOS appends
`#EXT-X-ENDLIST` for VOD.
`dashsrc::DashSrc` (`dash`)
is the MPEG-DASH analog: it parses a static MPD (the `mpd` parser, via
`roxmltree`), selects a Representation, and streams its fMP4 init + media segments
into `fmp4demux`. A Representation addresses its segments by a `SegmentSource`, one of three: a
`SegmentTemplate` (the `@duration` profile or a `SegmentTimeline`, the `<S t d r>`
entries expanded into per-segment times, addressed by `$Number$` or `$Time$`); a
`SegmentList` (an explicit ordered list of `<SegmentURL>`, each a `@media`
URL and/or a `mediaRange` byte range of the `BaseURL` resource, with an
`<Initialization>`); or a `SegmentBase` (one resource whose fragment byte
ranges live in a `sidx` Segment Index box at `indexRange`, fetched and parsed at
run time via `parse_sidx` + `Sidx::subsegments`, the index bytes never pushed
downstream). All three resolve to one `ResolvedSegment { url, byte_range, time }`
list, so a range-carrying entry fetches just its sub-range with an HTTP `Range`
request, the DASH analog of HLS `#EXT-X-BYTERANGE`, letting a single-file CMAF
DASH stream play. A `SegmentTemplate`'s `@presentationTimeOffset` is the media
instant that lines up with the start of the Period, so `$Time$` URLs keep the
media value while every `ResolvedSegment.time` (and with it seek matching and the
Period-boundary `Segment`) is period-relative presentation time. A dynamic (live)
MPD is reloaded on its `minimumUpdatePeriod`,
each new segment played once (tracked by start time), ending when the manifest
turns static, the same shape as the HLS live reload. Its wall-clock window comes
from `availabilityStartTime` + `Period@start` bounded by `timeShiftBufferDepth`,
with `@availabilityTimeOffset` publishing each segment that many seconds before
its nominal completion (clamped to one segment duration, so a chunked packager's
in-progress segment is reachable but nothing beyond it). `with_abr()` makes it
throughput-adaptive on the same shared `abr::BandwidthEstimator` as `HlsSrc`: a
`load_rep` helper resolves any Representation (Template / List / `sidx`-fetched
SegmentBase) into the run loop's segment/timescale/init working set, and the
estimate-derived cap drives both the per-reload pick and a per-segment
re-selection (so a static VOD adapts within one pass), re-emitting the init on a
switch.

`low-latency=true` changes how such a segment is *consumed*, not when it is
fetched: the response body is read as a stream and each complete CMAF chunk
(`styp` / `moof`+`mdat`) is pushed downstream as it arrives, so a segment the
packager is still writing flows at chunk latency instead of segment latency. The
split is `fmp4::CmafChunker`, an incremental box framer over the arriving bytes
(sharing `mp4box::next_box_len` with `fmp4demux`) that cuts after every `mdat` and
bounds both a declared box size and its pending run by the segment cap, so a
hostile length fails the fetch instead of buffering on it. Every byte comes out
exactly once in order, so the demuxer sees the same byte stream a whole-response
fetch delivers. Byte-range segments and a set `prebuffer-ms` (which owns emission
order) stay on the whole-response path.

**Still images** are the smallest case of a byte stream carrying coded frames, and
they take the same shape as a container (M1050). A PNG or WebP file is
`CompressedVideo{Png}` / `CompressedVideo{WebP}`, one access unit per file, so
`pngdec` / `webpdec` are ordinary decoders that `decodebin` auto-plugs and a
still is one frame of a video stream rather than a separate kind of media.
`typefind::sniff_caps` types both by magic (the PNG signature, `RIFF`+`WEBP`),
which is what a `.png` or `.webp` extension resolves through, since filesrc arms
content sniffing on an extension it does not know. JPEG is deliberately not typed
by content: `mjpegdec` takes one whole access unit per buffer, so a `.jpg` would
plug a decoder that fails past the source's read size until a `jpegparse` exists.

The decoders do not assume one file per buffer, because a byte source hands over
read-sized chunks (`filesrc`) or whole files (`multifilesrc`). Both cases go
through `stillimage::ImageAssembler`, which accumulates until the format's own
self-describing length says an image is complete: a PNG's chunk list walked to the
end of `IEND`, a WebP's RIFF size field. A stream that ends mid-image reports it at
EOS rather than decoding a partial file, and the bytes held while waiting are
bounded, so a plausible signature followed by silence cannot grow the buffer for
as long as the stream flows. Geometry is the file's word, so it is checked against
a per-side and a total-byte budget (`stillimage::rgba_byte_size`) before any
buffer is sized: both decoder crates size their output from the header, and a
100000x100000 `IHDR` or a 20000x20000 `VP8X` canvas would otherwise ask for tens
of gigabytes from a file of a few dozen bytes. Output is always 8-bit RGBA
(palette and sub-byte grayscale expanded, 16-bit narrowed to its high byte, alpha
added where the file has none), announced by a `CapsChanged` before the first frame
and on any change, since a sequence of stills can change size mid-stream. `pngenc`
is the inverse: RGBA or RGB in, one lossless PNG per frame, `compression-level` as
zlib's 0..=9. There is no WebP encoder: the only pure-Rust one does VP8L lossless
alone, with none of `webpenc`'s quality / speed / preset knobs.

### 4.18 Subtitle Overlay (`textoverlay`)

`textoverlay::TextOverlay` is the `textoverlay` / `subtitleoverlay`
analog: it renders timed subtitle text onto a raw video frame. The path splits
into two `no_std` pieces feeding one element:

- **`subparse`** parses SRT (SubRip) and WebVTT into a common timed `Cue`
  (`{ start_ns, end_ns, text, settings }`). Both formats are blank-line-separated
  blocks with a `start --> end` timing line and text on the following lines, so
  one block walker covers both: the shared timestamp parser accepts the SRT comma
  and the WebVTT dot fractional separators plus the WebVTT short `MM:SS.mmm` form;
  leading lines (SRT index, WebVTT cue id) before the `-->` line are ignored; the
  `WEBVTT` header and `NOTE` / `STYLE` / `REGION` blocks are skipped; inline
  markup (`<i>`, `<c.class>`, inline cue timestamps) is stripped. BOM and CRLF are
  tolerated. Malformed blocks are skipped rather than failing the parse, the way
  players tolerate dirty files. The WebVTT cue settings after the end timestamp
  are parsed into `CueSettings { position, line, align }` (the placement subset
  the bitmap overlay honours; `size` / `vertical` / `region` are recognised but
  not applied).
- **`bitmapfont`** is an embedded 8x8 bitmap font (MSB = leftmost column) so the
  baseline draws glyphs with no font file or rasterizer. It is an all-caps font
  (A-Z, 0-9, space, common punctuation; lowercase folds to uppercase).

`TextOverlay` is an RGBA8-in / RGBA8-out identity transform on the pixels
(`VideoConvert` upstream for other formats) except for the active cue text. By a
linear scan (subtitle tracks are small) it draws *every* cue covering the frame's
`pts_ns`, not just the first: WebVTT (and SRT) allow overlapping cues to show at
once. Each cue is placed independently from its `CueSettings`: `position` (% of
width) is the horizontal anchor and `align` (start / center / end) decides how
the box extends from it; an explicit `line` (% of height) places the box
vertically, while auto-`line` cues stack upward from the bottom in cue order so
overlapping subtitles don't collide. The WebVTT `vertical:rl` / `lr` writing mode
is parsed into `CueSettings::vertical` and carried end-to-end, but the bitmap
overlay does not yet lay text out in vertical columns (CJK vertical subtitles
render horizontally for now). Each cue draws over its own translucent
backing box, integer-scaled to the frame height. Cues are set programmatically (`from_srt` /
`from_webvtt`) or, on `std`, through the `location=` property loading a `.srt` /
`.vtt` file (format by extension, else content sniff); the element is registered
as `textoverlay` for the `gst-launch` text parser. This mirrors the analytics
overlay's CPU baseline (§5): the no_std bitmap renderer is the portable path.

The `truetype-overlay` feature replaces the bitmap font with a real one:
`fontdue` parses a `.ttf` / `.ttc` and rasterizes each glyph to a coverage
bitmap, alpha-blended onto the frame in the text colour, so CJK, accented Latin,
and mixed-case render, laid out horizontal or vertical (`vertical:rl` / `lr`,
from `CueSettings::vertical`) with the same `position` / `line` / `align`
placement. Because `fontdue` does no font fallback, `TextOverlay` holds a fallback
*chain* (`add_font` appends): each glyph is drawn from the first face whose
`lookup_glyph_index` is non-zero, so a Latin primary plus a CJK fallback covers
mixed text. `fontdue` rasterizes glyf (TrueType) outlines only; CFF / CFF2 (e.g.
variable Noto Sans CJK) yields empty glyphs and is one of the reasons the richer
`cosmic-text` backend (shaping, bidi, CFF, system fallback) is the planned
upgrade. The no_std baseline keeps the bitmap font (no font file or rasterizer).

A WebVTT `STYLE` block reaches the pixels. `parse_cue_styles` resolves `::cue`,
`::cue(#id)` and `::cue(.class)` rules onto each cue's `CueSettings`, and a
span-scoped rule lands as a `SpanStyle` run over the byte range its `<c.class>`
tag covers, so nested spans resolve per property (the innermost run that sets one
wins). The presentational `<b>` / `<i>` / `<u>` tags make the same kind of run
with no stylesheet at all, and a rule matching the same span overrides them.
Beyond `color` the properties honoured are `font-size` (`px`, or a percent
of the size the cue itself draws at), `text-shadow`, `background-color`,
`font-weight`, `font-style`, `text-decoration: underline` and `font-stretch`,
and all three render paths apply them. On the shaped path the sized runs become
cosmic-text `Metrics` overrides on the line's `AttrsList`, so a line mixing sizes
is still one shaped, bidi-reordered run and takes the tallest span's line height;
the `ab_glyph` renderer rasterizes each character at its own size on a shared
baseline. A shadow is one offset copy of the glyphs in the shadow colour, drawn
under every glyph of the cue so a neighbour's shadow never lands on top of one. A
blur radius is applied: the glyph's coverage mask is zero-padded and run through
three separable box passes, sized so the stack matches the gaussian CSS asks for
(standard deviation half the radius), and the grown mask is tinted in the shadow
colour. Vello has no filter that blurs a glyph run, so the GPU backend draws a
blurred shadow as one tinted mask image per glyph, blurred by the same code, and
falls back to a glyph run when the radius is 0. A whole-cue
`background-color` is the backing box; a span-scoped one fills the line box
behind that span's own glyphs, over the box and under the text. Weight, slant
and width are carried as per-span cosmic-text `Attrs`, so they pick a face out
of the font database (a real bold or italic face where the family has one, else
the `wght` variation axis for weight; there is no synthetic oblique, so an
italic run with no italic face installed renders upright) and reach the Vello
backend in the glyph ids and the face each run names. That face selection is
the shaped path only: a `vertical:rl` / `lr` cue on the `ab_glyph` column
renderer keeps the element's own `font-variations=` weight. An underline is a
filled bar in the run's text colour, drawn in the glyph layer so a neighbour's
shadow stays under it, below the baseline horizontally and down the column's
right edge vertically. Sizes and offsets
are clamped at parse time, because a stylesheet is as untrusted as the rest of
the subtitle file and the size becomes a glyph raster.

`vellooverlay::VelloTextOverlay` (`vello-text-overlay`) is the GPU backend for
the same cues, for a pipeline that keeps frames on the GPU: RGBA8 in,
`MemoryDomain::WgpuTexture` out, like `VelloAnalyticsOverlay` beside it. It holds
a `TextOverlay` rather than its own state, so cue selection, `CueSettings`
placement, colours, font chain and shaping are one implementation: the shared
step lays each active cue out into canvas-absolute glyph positions, which the CPU
element blits as swash rasters and this one hands to Vello as glyph runs (drawn
from the very face cosmic-text's per-codepoint fallback resolved, so a mixed
Latin + CJK cue uses the same faces on both backends). Vertical cues are the one
gap: they never reach the shaper, so only the CPU element's column renderer
draws them.

`SubParse` feeds that renderer as a stream rather than from a file: it parses a
structured subtitle document arriving on its sink pad and emits each cue as a
timed `Text{Utf8}` frame (PTS + duration = the cue window). Parsing is
*incremental* for the line-based formats (SRT / WebVTT / SSA): each `process`
call drains only the blocks bounded by a blank-line / newline separator, retains
the partial trailing block, and flushes the remainder at `Eos`, so a cue streams
out as soon as it is complete instead of all cues batching at end-of-stream
(chunk-boundary UTF-8 splits and a leading BOM are handled; TTML is XML with no
blank-line boundary and stays batch). `TextOverlayN` pairs the two as a
`MultiInputElement` (video pad + text-stream pad, video out): it opts into the
runner's `input_pts_ordered` merge so each cue lands just before the first video
frame it covers, and because `SubParse` streams, the merge buffers video only up
to the next cue, not to the subtitle stream's end. Cue placement (`CueSettings`:
`position` / `line` / `align`) cannot ride the plain-`Utf8` payload, so it
travels as a `TextCueMeta` frame-meta (the `metadata` feature) that `SubParse`
attaches and `TextOverlayN` reads, recovering WebVTT / SSA positioning; on the
ZST baseline (no meta) streamed cues draw at the renderer default.

Cue streams also go back into a container. A `Caps::Text{Utf8}` pad on either
Matroska muxer or on `Mp4MuxN` is a subtitle track, taking one cue per frame
with the window on the frame's PTS + duration, and the track's init needs nothing
from the stream, so it is fixed at configure rather than at the first cue (which
may be many seconds in, and every other track waits on the header). The two
containers time a cue differently. Matroska states the window per block, so a text
block is always a `BlockGroup` carrying a `BlockDuration` (a `SimpleBlock` has
nowhere to put one); the `subtitle-format` property picks the storage syntax,
`S_TEXT/UTF8` (the default, `subrip` to ffmpeg) or `S_TEXT/ASS`, where each cue is
framed as the mapping's `ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text`
event with `\N` line breaks, behind a script-header `CodecPrivate`. MP4 has no
per-sample timestamp: a `tx3g` sample (2-byte big-endian text length + UTF-8, what
ffmpeg calls `mov_text`) presents where the durations before it end, so the run
before the first cue and each run between cues is filled with an empty sample,
which is also what "no subtitle on screen" means in the format. Either muxer's
per-track metadata reaches a text track unchanged, so a subtitle track's language
and title ride the Matroska `TrackEntry` the way an audio track's do.

Closed captions (CEA-608 / CEA-708) feed the same renderer, but their bytes ride
*inside* the compressed video bitstream rather than in a container text track, so
the path is a track, not a `SubParse`-style drop-in. The `cea` module (`no_std`)
holds the decoders. `extract_cc_data` mines the `(cc_type, b0, b1)` caption
triples from an access unit's SEI `user_data_registered_itu_t_t35` (ATSC A/53
`GA94` `cc_data`) messages for H.264 (NAL type 6) and H.265 (prefix/suffix SEI),
and from picture `user_data` blocks (`00 00 01 B2`) for MPEG-1 / MPEG-2, the same
ATSC block without the T.35 prefix (M963); every count / length / offset
bounds-checked so a malformed block yields no triples.
`Cea608` decodes the legacy line-21 path (`cc_type` 0/1): a 15x32 character grid
with pop-on / roll-up / paint-on modes, PAC row + indent positioning, the
basic / special / extended-Western-European character sets, and channel selection
(CC1..CC4, the other channel's interleaved codes ignored). `Cea708` decodes the
DTVCC path (`cc_type` 2/3): it reassembles the DTVCC packets from the triples,
splits them into service blocks, and runs the selected service's window command
stream (DefineWindow / the DisplayWindows family / SetPenLocation, G0/G1 text)
against an eight-window model. Both emit the same timed `Cue` `SubParse` produces.

`CcExtract` wraps the decoders as a pipeline element: a compressed
H.264 / H.265 stream in, timed `Text{Utf8}` cue frames out, the same shape
`SubParse` emits, so the existing overlay consumes either. Because the captions
ride in the video, no new caps kind is needed (the in-band case taps
`Caps::CompressedVideo` directly; a `Caps::ClosedCaption` variant would only be
justified for an MP4 `c608` / `c708` *raw-caption track*, deferred). The element
selects one service at construction (`CcSource`; default CEA-608 CC1). In the
`playbin` auto-fan-out it sits on a *tee* of the parsed video: one tee
branch decodes for display, the other reframes to access units (so a TS PES does
not split an SEI NAL) and runs `CcExtract` into the video's `TextOverlayN` text
pad. Captions are not discoverable up front, so they are opt-in through a
`#closed-captions=cc1` (alias `#cc=`, or `service-N` / `708-N`) URI fragment, the
file-container analog of the HLS `#subtitle-lang=` hint. The file hooks (MKV / TS /
MP4) honour it; so does `hls_playbin`, which tees the variant's video the
same way for a muxed-TS variant (`build_hls_ts_cc_overlay`), an fMP4 / CMAF variant
(`build_hls_fmp4_cc_overlay`, tracks from the `#EXT-X-MAP` init), or a variant with
a separate audio rendition (`build_hls_separate_cc_overlay`, the audio merged in as
its own source). In every case the explicit caption request wins over an
auto-selected subtitle track (there is one overlay text pad).

The encode direction is the mirror image, for caption authoring and
broadcast egress. `cea::Cc608Enc` is the inverse of the `Cea608` decoder: fed cues
(text + placement) it builds the pop-on command sequence (RCL, a PAC per row, the
row text, EOC; EDM to erase) and queues the `(cc_data_1, cc_data_2)` byte pairs,
doubling the control codes and setting odd parity. `cea::Cc708Enc` is the 708
counterpart, the inverse of `Cea708`: it builds the window command stream
(DefineWindow a hidden window sized to the text and anchored from the cue's
relative placement, SetPenLocation per row, the G0 text, DisplayWindows to reveal
it atomically; HideWindows to erase), packs the commands into DTVCC service blocks
(never splitting a command across the 31-byte `block_size`), wraps each in a DTVCC
packet, and emits the `cc_type` 3/2 triples. Either drains one caption unit per
video frame (a byte pair / a triple; padding when idle). `CcInsert` is the element
wrapping them, the inverse of `CcExtract`: a compressed H.264 / H.265 access-unit
stream plus a timed cue stream in (a `MultiInputElement` merging the two pads by
PTS), the same video out with a `GA94` caption SEI (`cea::build_cc_sei`, the
inverse of `extract_cc_data`) written before each access unit's first VCL slice; it
encodes CEA-608 by default or CEA-708 via `CcInsert::cea708`. The video provides
the frame clock; a cue is queued on arrival and erased when its window ends, and a
warning fires if cues arrive against an untimed video source (the merge would drop
them). `SubtitleSrc` (a `.srt` / `.vtt` / `.ssa` / `.ttml` file as a `Text` stream)
is the head of the authoring pipeline, so `subtitlesrc -> subparse -> ccinsert ->
tsmux` (the `examples/cc_author.rs` flow) embeds captions from a subtitle file; the
whole `subparse -> ccinsert -> ... -> ccextract -> textoverlay` round trip is pure
in-graph.

**Bitmap subtitles** are the one subtitle family that is not text, so they get
their own coded media kind: `Caps::SubPicture { format: SubPictureFormat }`, a
stream of coded bitmap cues (`VobSub`, the DVD subpicture format, `DvbSub`,
ETSI EN 300 743, and `Pgs`, the Blu-ray HDMV Presentation Graphic Stream). It
sits beside `Caps::Text` rather than inside it,
because nothing downstream of a `Text` link can render a palette-indexed run-length
bitmap, and `Caps::ClosedCaption` is the model: a coded carriage variant whose
decoder produces something the rest of the graph already understands. Here that
is raw pixels. `VobSubDec` (`vobsubdec`, gst's `dvdsubdec`), `DvbSubDec`
(`dvbsubdec`; no gst alias, since gst's `dvbsuboverlay` is a video-overlay
element rather than a bare decoder) and `PgsDec` (`pgsdec`; gst has no PGS
decoder at all) all emit one full-frame transparent
`Caps::RawVideo{Rgba8}` canvas per cue at the subpicture display geometry,
stamped with the cue's PTS and duration, so the consumer is a pixel one:
`subpictureoverlay` or the ordinary `compositor`. A cue ends with a
second, fully transparent canvas at its hide time: either consumer holds an overlay
pad's last frame between output frames, and a zero-alpha source-over is a no-op,
so the clear canvas is exactly what makes a cue disappear on time. One more empty
canvas opens the stream, so the consumer is not waiting on this input for
however long it is until the first cue.

`subpictureoverlay::SubPictureOverlay` is the element that puts those canvases on
the picture. It is a two-pad `MultiInputElement` shaped like `TextOverlayN`, video
on pad 0 and the decoder's canvases on pad 1, opting into the runner's
`input_pts_ordered` merge so a canvas lands just before the first video frame it
covers. It holds the last canvas whose PTS the video has reached and source-over
blends it onto every frame, so a cue stays up between canvases; a canvas with no
drawn pixel is dropped rather than held, which is how the clearing canvas takes the
cue down. Both pads are RGBA8 on the CPU (`videoconvert` on either side for another
format), and a canvas whose geometry differs from the video is resampled onto it by
the `compositor`'s bilinear scaler, so a PAL-sized subpicture composites onto a
scaled picture. `mkv_playbin` auto-plugs it: a Matroska bitmap-subtitle track
decodes to canvases and feeds this overlay where a text track feeds `subparse` and
`TextOverlayN`. In a launch line it is a fan-in muxer built by link degree like
`textoverlay`, with the video and subpicture branches on its `video` and `text`
request pads. The MPEG program-stream (DVD) `playbin` composites its subpicture
track with `compositor` at the video's own geometry instead.

The VobSub bitstream itself (`vobsub.rs`, `no_std`) is one subpicture unit per
cue: a packet size and a control-sequence offset, 2-bits-per-pixel run-length data
in two interlaced fields (even rows then odd, each row byte-aligned), then control
sequences carrying the display rectangle, four palette indices and four alpha
nibbles, the two field offsets, and the show / hide dates in 1024/90000 s units.
The 16-entry RGB palette and the display size are *not* in the bitstream: they ride
the `.idx` text a Matroska `S_VOBSUB` track carries as its `CodecPrivate`, which
`MkvDemux` forwards in band ahead of the first cue the way it forwards the FLAC
and Opus headers, and which the decoder tells apart from a cue by parsing it as
`.idx` first. That same text is also the sidecar carriage: `VobSubSrc`
(`vobsubsrc`) reads a `.idx` / `.sub` pair off disk, emits the `.idx` as that
same in-band config frame, then reads each cue's subpicture unit out of the
`.sub` at the byte offset its `timestamp:` line names, stamped with that
timestamp and with the unit's own hide time as its duration, so
`vobsubsrc location=movie.idx ! vobsubdec ! compositor.` plays a sidecar pair
the way a muxed track plays. A `.sub` is an MPEG-2 program stream, so a unit is
reassembled from the `private_stream_1` PES packets at that offset carrying the
same subpicture substream id, bounded by its own 16-bit packet size; an `.idx`
indexing several languages picks one by `id:` code (`language=`) or by the
file's `langidx:`. An entry pointing outside the `.sub`, or a unit the file ends
in the middle of, drops that cue and not the stream. Every size, offset and
coordinate off the wire is range-checked and the parse returns `None` rather than allocating on a bogus rectangle; the pixel
data is bounded by the control-sequence offset, so a truncated packet fails
instead of decoding the control table as run lengths. A track that declares a
Matroska `ContentCompression` is refused outright, since its blocks are not SPU
packets and nothing here inflates them.

DVB subtitles (`dvbsub.rs`, `no_std`) are the broadcast sibling, and a different
shape: not one packet per cue but a *segment stream*, with decoder state carried
across display sets. Each data field holds segments sharing a `page_id`: a
display definition (the display geometry, 720x576 without one), a page
composition (the `page_time_out`, the page state, and where each region sits), a
region composition (a region's size, 2- / 4- / 8-bit depth, CLUT and background,
and which objects are drawn into it where), CLUT definitions (Y / Cr / Cb plus a
transparency, at full or packed precision, converted through the same BT.601
fixed-point path a reference decoder uses so the rendered colours are identical),
and object data (run-length coded pixels in two interlaced fields, with map
tables lifting a shallower code into the region's depth). A page composition
listing no region is how a cue ends; a page whose timeout expires before the next
display set gets the same clear canvas at its deadline. The composition and
ancillary page ids are the out-of-band part: a Matroska `S_DVBSUB` track's
`CodecPrivate` carries them, and `TsDemux` synthesizes the identical five-byte
blob from the PMT `subtitling_descriptor` (tag 0x59) that marks a private (0x06)
stream as DVB subtitles, so both carriages reach the decoder the same way. The
decoder takes a data field with or without its PES `data_identifier` header,
since a Matroska block carries the bare segments. Every segment length, region
dimension, CLUT entry id and object position is bounds-checked: a display set
whose segment layer does not hold together is dropped whole, and a region past
`MAX_REGION_PIXELS` is never allocated.

Blu-ray PGS (`pgs.rs`, `no_std`) is a segment stream too, but a flatter one: a
display set is a presentation composition (the video geometry, the epoch state,
which palette to read, and up to two objects with their positions), window
definitions, palette definitions and object definitions, terminated by an
end-of-display-set segment. Objects are 8-bit run-length coded and drawn straight
onto the video, with no region layer and no interlaced fields, and a cropped
composition object shows only a sub-rectangle of its bitmap, drawn at the
composition position with the crop offset indexing the object bitmap alone.
Cropping is the one part of the decoder with no reference peer to test against:
ffmpeg parses the crop rectangle and never applies it (`pgssubdec.c` carries a
"TODO: Implement cropping"), which every player built on it inherits. It is
verified instead by anchoring it to the uncropped path that ffmpeg does pin pixel
for pixel: cropping is a pure selection, so a fixture whose every object pixel is
a different colour is presented both whole and cropped, and the cropped canvas
has to equal the window of the whole one, with the crop of the entire object
equal to no crop at all. The placement convention that oracle cannot settle comes
from libbluray's `graphics_controller.c`, which does implement cropping. A crop
rectangle running off the object is clamped to what is there rather than trusted
the way libbluray trusts a disc. Palettes and objects
persist across an epoch, keyed by id, so a later palette segment updates only the
entries it names and an object too big for one segment arrives in fragments whose
total is fixed by the first one's declared length. Nothing rides out of band: the
palette is in the stream and the geometry is in the presentation composition, so
unlike the other two codings there is no config frame ahead of the first cue.
Palette entries are Y / Cr / Cb plus an alpha that passes through unscaled,
converted through the shared limited-range fixed-point path in `paint.rs`, whose
matrix a PGS stream picks by video height (BT.709 above 576 lines, BT.601 at or
below) since the format states no colorimetry. PGS has no end-of-display time
either: a cue stands until a later display set replaces it, and a presentation
composition listing no object is how the stream ends one, so the clear canvas is
the stream's own rather than synthesized from a hide time. Both the `.sup`
per-segment `PG` / PTS / DTS framing and the bare Matroska `S_HDMV/PGS` block
framing are accepted, told apart by the magic since no segment type is 0x50.
Every segment length, object dimension, run length, palette index and composition
count is checked before use: a run overflowing the bitmap or codes that do not
cover it drop the object, an object larger than the video or past
`MAX_OBJECT_PIXELS` is never allocated, and a truncated segment stops the walk
with the display sets that did parse intact.

The write paths mirror those two carriages (M927). A `Caps::SubPicture` input pad
on either Matroska muxer becomes an `S_VOBSUB` or `S_DVBSUB` track, and a
`Caps::SubPicture{DvbSub}` pad on either TS muxer becomes a private (0x06) stream
whose PMT entry carries the `subtitling_descriptor` naming its language, type and
pages (that descriptor replaces the 'KLVA' registration a bare private stream
would otherwise get, the same substitution the teletext descriptor makes). The
out-of-band configuration each format needs is not a property but the in-band
config blob the stream already leads with, so `mkvdemux`, `tsdemux` and
`vobsubsrc` all feed a muxer without translation: a VobSub pad's `.idx` becomes
the `CodecPrivate`, normalized to the `size:` / `palette:` lines a container holds
(the cue index is a sidecar's file offset table), and a DVB pad's five-byte page
ids become the `CodecPrivate` or the descriptor's page fields. A stream that
sends no blob is declared on the `dvbsub-page-id` property's page, defaulting to
1 like ffmpeg. The two carriages frame a display set differently, so the muxers
convert: a Matroska block holds the bare segments, a TS PES payload wraps them in
the EN 300 743 data field (`data_identifier` 0x20 and a subtitle stream id ahead,
the end marker behind), and both directions run through `segment_span`, which
finds the segment run by walking its headers rather than trimming bytes. Subtitle
blocks, bitmap as well as text, are written as a `BlockGroup` so a cue's display
window rides its `BlockDuration`. Both Matroska muxers take these pads, the
fan-in `MkvMuxN` beside its A/V tracks and the single-track `MkvMux` on its one
sink pad (M928), so a sidecar subtitle file muxes over one link
(`vobsubsrc location=movie.idx ! matroskamux ! filesink`) rather than the `name=m`
shape. The mapping they share (the codec each format writes, the config-blob
recognition, the block framing, the `S_TEXT/ASS` script header) lives in
`matroska.rs` beside `MkvTrackSpec`, so the two cannot drift.

**EBU teletext** (`teletext.rs`, `no_std`) is the third TS subtitle carriage, and
unlike the two above it is characters rather than pixels, so it lands on the plain
text pad instead of a canvas: `Caps::Text { Teletext }` in, `Caps::Text { Utf8 }`
cues out of `TeletextDec` (`teletextdec`), which is the same pad a `subparse`d SRT
track produces and therefore the same `TextOverlayN` input. DVB carries teletext in
a private PES (EN 300 472): a `data_identifier` byte then fixed 46-byte data units,
each one broadcast line, holding a framing code, a hamming 8/4 magazine / packet
address and 40 odd-parity bytes. Those address and data bytes are transmitted LSB
first, so each is bit-reversed before any code word means anything, while the two
bytes ahead of them are ordinary MSB-first fields. Packet X/0 is the page header
(page number, the C6 subtitle bit, and the national option subset the G0 set is
read under); X/1..X/23 are the display rows, and the decoder holds the rows of the
addressed page until the next header for it replaces or erases the page, which is
what fixes the cue's duration and puts each cue out one page late. Spacing control
codes and parity failures render as spaces so a row keeps its columns, and a
double-height row's blanked bottom half is dropped so the line appears once.
Enhancement packets (X/26, X/28, M/29) are not read, so the national option comes
from the header bits and the wider seven-bit G0 selection is out of reach. Which
page to follow is out of band, as for DVB subtitles: `TsDemux` synthesizes an
eight-byte selection blob from the PMT `teletext_descriptor` (tag 0x56, or the
identical `VBI_teletext_descriptor` 0x46) and forwards it in band ahead of the
first line, and the `page` property overrides it; with neither, the first subtitle
page the stream carries is adopted. The blob leads with `0xFF`, which cannot begin
a teletext payload, so the decoder tells the two apart on one pad. Every data unit
length, hamming code, parity bit and page address is checked before use, so a
corrupt line or a unit length past the payload drops that line or ends the walk
rather than propagating a corrupt page.

### 4.19 Native WebRTC (`str0m`)

The WebRTC elements are built on **[str0m](https://github.com/algesten/str0m)**, a
**sans-IO** WebRTC stack (ICE / DTLS / SRTP / RTP as a pure state machine): g2g
owns the `UdpSocket` and the timer and drives str0m's `poll_output` /
`handle_input` loop, exactly the contract the `srt` and `rtspserver` modules
already follow. str0m's pure-Rust **`rust-crypto`** backend is selected, so there
is no OpenSSL / libnice system dependency. Everything lives behind the opt-in
`webrtc` feature (off by default, so the no_std baseline is unaffected). This is
the native, server-grade counterpart of the browser-only data-channel
`WebRtcSrc` (§6.3).

**Element family.** One PeerConnection can carry one track per element or N tracks
in a session element; the shape is chosen by which trait the element implements,
and each maps to a terminal runner from the fan-in / fan-out family (§4.13.6):

| Element | Tracks | Direction | Trait | Runner |
| :--- | :--- | :--- | :--- | :--- |
| `WebRtcSink` | 1 | send (WHIP) | `AsyncElement` (sink) | linear |
| `WebRtcWhepSrc` | 1 | recv (WHEP) | `SourceLoop` | linear |
| `WebRtcSessionSink` | N | send (WHIP) | `MultiInputElement` | `run_fanin_session` |
| `WebRtcWhepSessionSrc` | N | recv (WHEP) | `MultiOutputSource` | `run_fanout_session` |
| `WebRtcDuplexSession` | N | sendrecv | `MultiDuplexSession` | `run_duplex_session` |

The one-track sink/source keep the `Rtc` on a spawned task and hand access units
over a bounded channel, so the element itself never touches the `Rtc` and stays
`Send`. The multi-track session sink is a terminal `MultiInputElement` (no
downstream sink — the network is the destination); `run_fanin_session` fans N
sources into it over one tagged `(input, packet)` channel. The session source is
the mirror: a terminal `MultiOutputSource` (0 inputs → N outputs) driven by
`run_fanout_session` into N sinks.

**Simulcast (M710-M723).** Send-side simulcast lives in `webrtc_simulcast.rs`,
shared by `LiveKitSink` and (M723) `WebRtcSessionSink`: `SimulcastPads` (the
video-layer + audio pad model, pad 0 = highest resolution), rid assignment
(`f`/`h`/`q` high-to-low), the one-m-line `a=rid`/`a=simulcast` offer,
per-(mid,rid) `KeyframeRoutes`, and the BWE `LayerAllocator` (whole-layer
on/off with time hysteresis; per-layer targets, M722). The LiveKit path is
browser-validated end to end; the WHIP path is validated against Broadcast Box
(M786), the reference peer for client-simulcast ingest (mediamtx cannot ingest
it, LiveKit's WHIP ingress transcodes one layer): a three-layer publish shows up
server-side as three rids with independently growing packet counts.

**Session sources as DAG nodes (M727).** The receive-side mirror:
`NodeKind::FanoutSrc(n)` via `Graph::add_fanout_src` runs a terminal
`MultiOutputSource` (0 inputs, N outputs it generates itself) as a graph node,
its ports solved from `output_caps` (the demux constraint shape with the input
half inert) and its arm just running the element into per-edge senders (the
element owes every port an `Eos`). `FanoutSrcFactory` + named output pads wire
it into `parse_launch` (`livekitsrc name=s url=...  s. ! ...  s. ! ...`), with
a properties surface added to `MultiOutputSource` for the launch knobs.
Validated live: `LiveKitSrc` as a graph node subscribing on a real server.

**Session sinks as DAG nodes (M713).** A terminal fan-in is also a first-class
graph node, `NodeKind::FaninSink(n)` via `Graph::add_fanin_sink`, so a transform
chain can feed each session pad instead of a bare source: the live encoder fan
graph `src -> tee -> videoscale -> ffmpegenc` per simulcast layer ends on
`LiveKitSink` inside one `run_graph` (cooperative or threaded). The node reuses
the muxer's `GraphNodeRef::Muxer` payload and negotiation shape with the output
half inert (no output edge exists), and its arm is the `run_fanin_session`
discipline over the DAG's per-edge channels: round-robin drain, per-input `Eos`
flush, end on all-`Eos`, and each pad's `reverse_channel` relayed onto its own
in-edge, so a per-`(mid,rid)` PLI reaches exactly the encoder feeding that layer
as a `PushOutcome::Reconfigure(ForceKeyframe)`. `MultiInputElement::is_terminal`
marks a session element as legal to end a graph; in `parse_launch` a fan-in name
with nothing downstream builds this node, while a merging muxer without a
downstream stays a `MuxerWithoutOutput` parse error (its merged output would be
silently discarded).

**The duplex shape.** Bidirectional sendrecv needs an element that is *at once* a
sink (for the tracks it publishes) and a source (for the tracks it receives) over
**one** connection — which neither the fan-in nor the fan-out session runner could
express. `MultiDuplexSession` is that union: N send inputs **and** M recv outputs,
driven by `run_duplex_session` (the union of the two session runners). A single
`run(inbound, out)` owns the connection and `select`s over the inbound send
packets (`DuplexInbound`) and the network, pushing received frames to `out`; the
send and recv halves therefore share `&mut self` directly with **no detached
task**, unlike the send-only session which spawns the `Rtc` onto its own task to
dodge `process` / run-loop aliasing.

**Growing the pad count live (M1014).** `run_duplex_session_dynamic` is the
renegotiating sibling: its arms live under a `dynamic_join`, so pads are not
fixed at build time. A local track enters through `DynamicDuplexHandle::
add_send_track` (index reserved and enqueued under one lock, so the session
learns pads in order); the runner fixates the source alone and announces its
caps on the new index before any frame, which is how a session that never
declared the pad learns it exists, and `DuplexInbound::reverse_channel` hands it
the PLI / BWE route back. A remote track with no free pad is taken by the
session calling `MultiOutputSink::add_port` (default `None`, so fixed runners
refuse growth); the runner mints the port's link and asks an app-supplied sink
factory for the element that drains it, a factory `None` leaving the port
counted as drops. Backpressure on the runner's internal add channels delays the
attach rather than failing the run.

**Signaling.** WHIP (egress) and WHEP (ingress) are the same wire move — an
`application/sdp` POST of the local offer that returns the remote answer (reqwest,
`webrtc_util::post_sdp`); the media server is the relay in the middle, so there is
no peer-to-peer mode for WHIP/WHEP. WHIP/WHEP are unidirectional by spec, so
sendrecv cannot use them: the duplex session instead exchanges SDP **directly**
between two peers over an `SdpChannel` (an in-process offer/answer transport for a
P2P loopback; a real SFU signaller — LiveKit, etc. — plugs into the same seam).
Mid-session renegotiation (M729): a cloneable `DuplexControl` toggles a track,
batching direction changes (SendRecv <-> Inactive) into one re-offer over the
`SdpChannel`; the peer answers it in its loop (typed `offer\n` / `answer\n`
prefixes distinguish the exchange; on glare the answerer role yields).

**Mid-session NEW tracks: spare pads (M784).** The fixed-arity pad model has no
pad to grow into, so a session reserves them up front:
`with_spare_tracks(video, audio)` appends declared-but-inactive pads after the
active ones, and they carry no m-line at the handshake. Each negotiated m-line is
a *binding* holding its `Mid`, kind, and the input / output pads it serves, which
replaces the per-kind mid slots: recv routing, PLI, BWE, and the direction
toggles all resolve through it. A spare binds either when its **send** pad gets
its first frame (the session offers the peer a new sendrecv m-line, one exchange
at a time, the frame itself dropped like any frame before its m-line exists) or
when the peer's re-offer lands and `MediaAdded` fires for an unknown mid, which
claims the first free pad of that kind on both sides. A pad bound mid-session
emits its `CapsChanged` before its first frame; the active pads are announced at
session start. A track whose kind has no free pad left is rejected (it stays
unbound and its media is skipped), since the pad count is fixed at graph build
time. `DuplexControl::remove_track` is the inverse (M785): it `stop_media`s the
m-line (port 0, out of the BUNDLE group), batched into the same re-offer as the
direction toggles. Both peers then drop that binding by walking their media
after each SDP application (a stopped m-line stays in the session with
`Media::stopped()` set, so this also retracts an ADD that lost a glare race),
which frees its pads with no `Eos` on the output pad, since a later track may
claim it and the end of the run EOSes every pad anyway. Reuse always negotiates
a NEW m-line, a stopped one cannot be reactivated, and the freed output pad
re-announces its caps before the new track's first frame.
The two roles discover their m-line `Mid`s differently and this asymmetry is
load-bearing: the **offerer** captures its `Mid`s from `SdpApi::add_media`'s
return, while the **answerer** learns them from `Event::MediaAdded` after
`accept_offer` (str0m does not emit `MediaAdded` for media the local side added).

**LiveKit signaller (M707/M714).** `livekit_signal` is the protocol seam: an
HS256 JWT access-token mint and a hand-rolled protobuf codec for the
`livekit_rtc.proto` subset, over a tokio-tungstenite WebSocket (`ws://` and,
M715, `wss://` via native-tls, the TLS stack the WHIP reqwest client already
links). `LiveKitSink` publishes (client offers, per WHIP habits) and
`LiveKitSrc` subscribes — where the offer direction REVERSES: the SFU offers the
subscriber PeerConnection over the signalling socket and re-offers on every
track-set change, and the element answers each with `accept_offer`, learning its
mids from `MediaAdded` per the answerer rule above. The source is a terminal
`MultiOutputSource` (video + audio ports, `run_fanout_session`), gates video
until the first keyframe and repeats a PLI until it arrives, and takes the first
video / audio m-line offered (one-subscription element). Both validated against
a real LiveKit server, including an in-room sink-to-src A/V loopback and the
same loopback over a TLS-terminated `wss://` proxy. `LiveKitDuplex` (M728) is
the full participant: LiveKit has no sendrecv m-lines, so it runs BOTH PCs (a
publisher it offers, a subscriber the server offers) in one loop over one
socket, routing trickle by `SignalTarget`, exposed as a `MultiDuplexSession`
for the duplex runner; two participants exchanging A/V validated live.

**ICE / NAT traversal.** `webrtc_util::add_ice_candidates` always adds the socket's
host candidate and, when a STUN server is configured, a server-reflexive candidate
discovered by a hand-rolled RFC 5389 Binding on the ICE socket; candidates ride in
the SDP, so a same-host P2P pair connects over localhost with no STUN. For the NAT
cases a reflexive candidate cannot punch through, a hand-rolled TURN client
(`turn.rs`, RFC 5766/8656: Allocate with long-term auth, channel binding,
periodic Refresh) provides a relay. str0m only offers
`Candidate::relayed`; the data plane is the run loop's job — a relayed pair's
transmits all carry `source == relay_addr`, which is the routing signal to wrap
the datagram for the relay (direct host/srflx paths are untouched). The first
transmit to a new peer sends a ChannelBind (M716), which installs the peer
permission and, once its success lands, upgrades that peer from 36-byte Send /
Data indications to 4-byte-header ChannelData frames both ways; a `438 Stale
Nonce` on any authenticated request adopts the error response's nonce and
un-caches the affected state so the lazy paths retry with it. The
client-to-server leg also runs over TCP and TLS (M717, `turn:...?transport=tcp`
/ `turns:` RFC 7065 forms): a local bridge task tunnels the client's datagrams
over one stream connection, re-delimiting messages (STUN self-describing
lengths; ChannelData padded to 4 bytes on the stream), so `TurnClient` and
every element run loop stay transport-agnostic and the allocation still relays
UDP toward peers. The codec is address-family agnostic (M718): XOR addresses
encode/decode IPv6 (cookie + transaction id), and a v6-bound client requests a
v6 relayed address (RFC 6156). Validated against a real coturn on all three
transports and over IPv6 (allocate, bind, ChannelData round-trip both
directions). An element takes a comma-separated server list (M719, each entry
optionally carrying GStreamer-style `turn://user:pass@host` credentials): a
`TurnSet` allocates on every server, contributes one relayed candidate each,
and the data plane routes by which relay a transmit's `source` names. The
duplex session gained the same STUN/TURN surface; its relayed candidates ride
in the offer/answer SDP (no trickle channel).

**RTCP feedback** rides the §4.13 reverse channel. A remote PLI
(`Event::KeyframeRequest`) becomes a `Reconfigure::ForceKeyframe` walked upstream
via `AsyncElement::take_reconfigure` to the encoder (`Av1Enc` forces a rav1e IDR);
ingress originates PLI on a mid-GOP join. str0m's BWE (`Event::EgressBitrateEstimate`,
TWCC/REMB) becomes `PushOutcome::Bitrate` via `take_bitrate`, and the encoder
retargets (rav1e by a hysteresis-gated context rebuild). Both signals hop past
intervening transforms (M720): an element that does not consume them
(`AsyncElement::handles_keyframe_requests` / `handles_bitrate_requests`, false by
default; encoders override) has its output adapter relay the pending
`ForceKeyframe` / bitrate onto its input link, the QoS-relay mechanism
generalized, so `enc ! h264parse ! webrtc-sink` reaches the encoder. `Propose` /
`Renegotiate` never relay (they concern the adjacent element's own caps).
`OpusEnc` retargets live too (M721, `OPUS_SET_BITRATE`, no rebuild), and
`FfmpegH264Enc` by a hysteresis-gated reopen (M722; zerolatency, nothing in
flight). Simulcast splits the aggregate BWE estimate per layer (M722): the
allocator hands each active layer its nominal share of the estimate on that
layer's reverse channel and a shed layer the `Bitrate(0)` idle hint, on which
the encoder skips frames unencoded except a sparse 1-in-32 keep-alive (the
resume signal rides push outcomes, so the cadence must not fully stop; resume
forces an IDR). The allocator is also re-ticked with the last estimate once a
second: BWE only emits deltas, and retargeted encoders settle exactly on the
estimate, which would otherwise freeze the drop/restore hysteresis windows.

**Codec plumbing.** A `Track` enum unifies the per-track facts WebRTC needs to
agree on: codec (H.264 / Opus), m-line `MediaKind`, and the RTP clock (90 kHz /
48 kHz), with `media_time` mapping a nanosecond PTS onto the track's RTP
timestamp. H.264 crosses the boundary as **Annex-B** (the pipeline convention,
§4.11.4): str0m's packetizer splits NAL units and its depayloader emits start-code
framing. A receive-side video element advertises a `Dim::Range` /  `Rate::Range`
placeholder rather than `Dim::Any`, because geometry is only known from the in-band
SPS and `fixate()` (§4.13) rejects `Any` at negotiation; a downstream `H264Parse`
recovers the real dimensions.

**Validation status.** On-network validated against a local mediamtx (single-track
WHIP/WHEP and multi-track A/V) and by in-process P2P loopbacks on localhost (video
and full A/V sendrecv); the structural `webrtcbin` parity — one connection, N
tracks, BUNDLE, sendrecv, PLI, BWE — is in place. What remains is maturity rather
than architecture; `DESIGN_TODO.md`'s "WebRTC" item carries the tiered list.

### 4.20 Developer Tooling: DOT Visualization

`g2g_core::dot` renders a pipeline graph as Graphviz DOT, the
`GST_DEBUG_DUMP_DOT_DIR` analog: `Graph::to_dot` (pre-validation) and
`ValidatedGraph::to_dot` (post-`finish`) emit a `digraph { .. }` a developer
renders with `dot -Tsvg`. It is pure `no_std + alloc` string formatting (no I/O),
so it builds on every target the core does, embedded included.

Because the graph carries an opaque element payload `E`, node display names come
from a caller-supplied `Fn(NodeId) -> Option<String>`; returning `None` falls
back to the node's structural kind, the right answer for a `tee` / `mux` that
carries no element. Nodes are role-coded by shape and fill (source / sink /
transform boxes, a `tee` diamond, a muxer trapezium). Edges are annotated from a
`DotAnnotations { edge_caps, edge_memory }`, both indexed by edge id, the same
index `solve_graph` returns its `Vec<Caps>` solution under and `ValidatedGraph::edge`
uses: an edge shows its negotiated caps (`Caps::to_gst_string`), a non-`System`
memory domain (drawn bold, since a GPU / zero-copy link is the interesting one),
its non-default `LinkPolicy`, and fan-out / fan-in pad indices.

`g2g-launch --dot` is the user-facing entry: it parses a pipeline against the
registry, dumps the DOT to stdout, and exits without running, labelling each node
by its element's `log_category` (the short type name, e.g. `VideoTestSrc`) via
the new `GraphNodeRef::log_category`. To show the *chosen* caps it first calls
`negotiate_graph` (§4.20a's seam: Phase 1 source-caps probe + Phase 2 solve,
without running the pipeline), which returns the per-edge fixated caps and each
edge's memory domain (the producing node's `output_memory`) the dump
renders on the edges, marking GPU / zero-copy links bold; a negotiation failure
falls back to a topology-only dump. It also runs the allocation cascade
(§4.13.5) before reading those domains, since that is what settles a
multi-domain producer on the one its consumer asked for: without it a decoder
feeding a CPU sink still reported its `Cuda` default and the dump called a
downloading link a GPU link. Because negotiation probes sources, a `--dot`
of a live-ingress pipeline does that source's `intercept_caps` (typically a
connect) just as a run would. Memory domain is a per-element declaration
(`AsyncElement::output_memory` / `SourceLoop::output_memory`, default `System`,
overridden by GPU producers like `NvDec`), the runtime peer of the auto-plug
`ElementDesc::output_memory` (§4.13.9); it is not part of `Caps`.

### 4.20a Developer Tooling: Caps-Negotiation Explainer

Caps negotiation is the hardest code in the system (§4.13, with accumulating
workarounds), and a `CapsMismatch` historically gave no hint *why*. The
explainer makes the solver narrate itself. `solve_graph` emits under a reserved
`caps` log category (not an element type, so it filters independently): a setup
dump of each node's constraint, then per edge the surviving `CapsSet` and its
fixated `Caps`. On failure it narrates at ERROR, naming the two conflicting nodes
and dumping the set on every edge incident to them, so the log answers "these two
can't agree, and here is what each wanted"; an edge that survives narrowing but
can't reduce to one `Caps` logs `cannot fixate`.

Node labels come from the caller via `solve_graph_labeled`: the runner passes
each element's `log_category` (so the narration reads `h264parse -> nvdec`),
while `solve_graph` defaults to `n{id}:{kind}`. The narration is gated by the
logging framework (§4.15): all formatting is skipped unless the `caps` category
is enabled, which costs one atomic load when off, so it is free in production.
It is turned on with `G2G_CAPS_TRACE=1` (a boolean shortcut, or a level name /
number to tune verbosity) or the general `G2G_DEBUG=caps:debug`; both install the
stderr sink through `log::init_from_env`, which the launch / inspect binaries
already call at startup.

### 4.20b Developer Tooling: the `xtask` crate

`cargo xtask <command>` (a `.cargo/config.toml` alias onto the `xtask` workspace
member) is the home for the build / test invocations that were otherwise
shell-history knowledge. It is dependency-free, orchestrating only `cargo` and
toolchain tools. `ci` runs locally what the GitHub workflow runs (workspace
check / test / clippy, the Linux feature build, the embassy no-alloc tests, the
wasm core check), `--locked` like CI, so a red CI is reproducible offline.
`test --here` probes the host (`nvidia-smi`, `pkg-config` for the syslib-backed
features, `/dev/video*` and `/dev/dri` device nodes) and runs exactly the
feature-gated tests this machine supports, automating the "validate on this host"
dance; `--dry-run` prints the detected plan only. `size` builds the
`examples/g2g-size` Cortex-M harness and reports the gc-sectioned `.text`
footprint (it locates `rust-lld` in the toolchain sysroot for the final link).
`wasm` builds the wasm32 targets. The cross-compiling commands (`size`, `wasm`)
prepend `~/.cargo/bin` to `PATH` so cargo selects the rustup toolchain over a
distro `rustc` that lacks the target std, and `wasm` passes
`--cfg=web_sys_unstable_apis` for the `web-codecs` build.

`ffi-probe <header> <struct> [--field f]...` automates the hand-rolled-FFI
ritual (§4.11 / the `cuda.rs` / `nvenc.rs` convention): it generates a C program
that includes the header and prints `sizeof` of the struct plus `offsetof` of
each field, compiles and runs it, and emits the `const _: () = assert!(size_of::
<Struct>() == N)` to paste alongside the `#[repr(C)]` transcription. Layout is
locked down before it is trusted, and an SDK version bump that resizes a struct
fails the build rather than the GPU. `bench` runs the criterion benchmarks.
`new-element <name> --kind source|transform|sink` stamps the boilerplate every
new element repeats: the `g2g-plugins` source file with the correct
`AsyncElement` / `SourceLoop` skeleton for the kind (`intercept_caps` /
`configure_pipeline` / `process` or `run`, with TODOs), a scaffold test, and the
`pub mod` wiring inserted into `lib.rs` alongside the unconditional module block;
it prints the `registry.rs` registration line to paste (the registration
function is context-dependent). The generated element compiles as-is.

The criterion benchmarks live in a standalone `g2g-bench` crate, excluded from
the workspace (like `examples/g2g-size`) because criterion pulls plotters / rayon
that a `--all-targets` CI job would otherwise build on every push, and Cargo's
`required-features` does not gate a dev-dependency under `--all-targets`. They
guard the latency moat's hot paths: the caps algebra + linear / DAG solvers
(`benches/caps.rs`), the per-pixel software frame conversion
(`benches/convert.rs`), and the runner loop's bounded per-edge channel
(`benches/runner.rs`, the transport every frame crosses; the full `run_graph`
paces to PTS so it is unsuitable for a microbench). `cargo xtask bench` drives
them by manifest path, passing criterion args through (e.g. `--save-baseline`).

`tools/pushtax-bench.sh` (M870) prices the push model against GStreamer's pull
on batch demux: the same `filesrc ! tsdemux ! h264parse ! fakesink` line through
`g2g-launch` (release) and `gst-launch-1.0` over an ffmpeg-authored 60 s 1080p30
TS, five interleaved timed runs each, results appended per iteration. Measured
on the dev host: 176 vs 1175 MB/s (6.7x). The script prints the g2g per-element
attribution under the ratio because the gap is element CPU, not transport:
`TsDemux` (0.26 ms p50 x 1414 chunks) and `NalParse` (0.13 ms p50 x 1800 AUs)
account for essentially the whole wall clock on the single-thread executor, so
the per-chunk channel / wakeup / boxed-future residual is small and a pull mode
would not close the gap; demux/parse throughput would. `benches/runner.rs`
already prices the bare channel.

A dedicated `bench` workflow (separate from the main CI, so criterion never
slows the check / test / clippy jobs) runs on PRs that touch the benched crates:
it benches the PR head and its base and fails if any benchmark's mean regressed
more than 50% (a loose threshold tuned to shared-runner noise, catching a lost
fast path rather than drift).

`RunStats::report()` formats the end-of-run telemetry the runner already gathers,
frame counts + drop rate, the aggregated *declared* latency window (the
per-element `latency()` fold), the elected clock, and the head allocation, which
`g2g-launch` prints at end alongside the measured wall-clock throughput.

Alongside the declared fold, the runner collects *measured* per-element telemetry
(`RunStats::per_element`, one `ElementLatency` row per interior element in
topological order). Each transform/sink arm holds an `Arc<ElementProbe>`
(`runtime/instrument.rs`): on every `DataFrame` it samples its input link's fill
(`LinkReceiver::fill_percent`) and times the `process()` call wall-clock
(`metrics::monotonic_ns` around the `await`), recording into the lock-free log2
`LatencyHistogram` so the hot-path cost is a handful of relaxed atomics and no
allocation. Once every arm has joined, the runner snapshots each probe into the
report, and `report()` prints a per-element `proc p50 / p99 (n) + in-fill
avg/max` table, the by-hand glass-to-glass analyses (the NVDEC-to-system-memory
floor, `link_capacity` dominance) turned into a number the runner emits. The
graph runner and the two linear runners (`run_simple_pipeline`,
`run_source_transform_sink`) collect it, and the dynamic fan-out / fan-in /
muxer-sink runners do too (M869: `_observed` entry points register each arm's
node and edge on the observer incrementally as it attaches, so a late arm
reports like an initial one); the static session runners leave it empty, like
their declared latency. It is `std`-gated where it
needs a clock: the histogram is `no_std`, but with no `monotonic_ns` the timing
compiles out (the table is then empty) so the `no_std` baseline pays nothing.
Sources have no `process()` and so do not appear, their cost surfaces as the
downstream element's input fill.

A paced display sink also reports what it actually put on screen (M933): the
element overrides `presentation_stats()` (frames presented, frames overwritten
before paint under `DropOldest`, frames shed by QoS late-drop), the graph
runner's sink arm stores it on the probe as the arm ends, and `report()` prints
one `present:` line per presenting sink. `frames_consumed` alone cannot
distinguish a healthy display from one silently shedding or stalling;
`g2g-launch` divides the presented count by wall time into a presented-fps
figure next to the pipeline throughput.

The `process()` timing is the "work" half of a stage's latency; the "wait" half
is queue residency, added as measured per-link transit. When an observer is
attached, the graph runner builds `Block` edges into transform/sink arms with a
per-link transit ring (`link_with_transit`): the producer's `SenderSink` stamps a
monotonic send time as each `DataFrame` is queued, and the consuming arm pops the
stamp when it pulls the frame (`LinkReceiver::pop_transit_ns`), recording the
elapsed queue time into `ElementProbe::transit`. The ring stays aligned with the
data channel because `Block` links never drop (leaky edges are left plain, so
their transit is simply not measured), and it is `Option`-gated so an
uninstrumented run carries no stamp and pays nothing. `RunStats::report()` prints
`wait p50/p99` beside `proc`, and the dashboard stacks the two per stage into a
latency waterfall.

Every link also carries a per-edge content-inspection slot (`LinkSender::probe`,
a `ProbeSlot` the wrapping `SenderSink` shares), so a tool can install a
`LinkInterceptor` to sample the packets crossing any edge without touching the
arms; empty (pass-through, zero cost) unless a subscriber installs one. The
dashboard uses it for edge previews: clicking an edge sends a `subscribe` over
the WebSocket, the server installs a rate-limited `PreviewTap` on that edge's
slot (via `Observer::edge_probe` / `edge_caps`), and streams back a `preview`
message: a downscaled thumbnail for RGBA/BGRA and planar NV12/I420 video (and
MJPEG keyframes under the `mjpeg` feature, reusing `videoconvert` / `mjpegdec`
rather than duplicating the conversion), a codec card for other compressed edges
(codec, resolution, header-parsed frame type, and size, no decode), PCM S16
waveform buckets, or a bounded hexdump (`g2g-plugins::preview`), sampled a few
times a second on a copy, never blocking the data path.

The same probes drive a *live* view, not just the end-of-run table. An
`Observer` (`runtime/observe.rs`) captures the graph topology and holds clones of
the arms' probe `Arc`s; `run_graph_observed` registers them during the prepare
phase, before any frame flows. Because the probes are the same lock-free atomics
the report reads, `Observer::snapshot` mid-run is a handful of relaxed loads and
never stalls an arm. The transport lives in `g2g-plugins::dashboard` (the
`observe` feature): `g2g-launch --observe <port>` serves one TCP port that
answers a plain `GET /` with a self-contained dashboard page
(`tools/dashboard/`) and a WebSocket upgrade with a JSON `telemetry` snapshot
every 250 ms plus one `event` per `BusMessage` (fanned out to all clients via a
broadcast channel drained off the `Bus`). Each telemetry edge carries its
negotiated caps (from the `Observer`'s per-edge solution) and live counters
(packets, CPU-payload bytes, drops, and `blocked_ns`, the time producers spent
awaiting link capacity, from a wait-free `EdgeCounters` block the data-plane
sink writes), which the page labels on the link; the page pans / zooms so a
large graph stays navigable. Beside the aggregate per-stage waterfall the page
assembles a single frame's journey: observed probes keep a bounded ring of
`{sequence, wait, enter, exit}` visits, joined at snapshot time along the
linear prefix on the newest sequence id consistent with one frame moving
downstream (restamping elements fail the join rather than fabricate one; fan
nodes truncate it), shown as stacked wait/work/blocked bars with the end-to-end
total against the `2 * capacity * frame_period` floor. A journey stage's
`work_ns` is compute and `blocked_ns` is downstream backpressure, both drawn
from the same push-wait bank as the aggregate `push_wait` percentiles. It binds loopback by default;
`--observe-host <addr>` (e.g. `0.0.0.0`) exposes it to other hosts, gated behind a
no-auth warning since telemetry + edge previews carry frame content. The JSON is
built in the transport, so `g2g-core` stays serde-free, consistent with the
portability-core principle. The
observer rides the cooperative graph runner and, via `run_graph_threaded_observed`,
the threaded runner; both cover the muxer / demux fan nodes. The standalone
hand-built fan-in / fan-out / session runners (`fanin.rs` / `runner.rs`) have
their own `*_observed` entry points, name and probe their nodes like the graph
runner, and fill `RunStats::per_element` even unobserved; the dynamic runners
(arms attach at runtime) still report no per-element rows.

`g2g-inspect --json [element]` (the `tooling-json` feature) emits the registry as
JSON, the machine-readable sibling of the text dump: per element the identity,
role, output caps or pad templates, and each property's machine type, range,
default, and read/write flags, from the same `ElementDoc` / `PropertyDoc`
introspection the text path uses. Like the dashboard it is serialized in
`g2g-plugins` (serde_json), not `g2g-core`. It feeds two consumers: the visual
pipeline builder (`tools/builder/`), a React Flow app (Vite + pnpm) that loads a
`registry.json` snapshot, offers a typed drag-drop canvas with pan / zoom and
either-direction linking, imports and live-exports a `gst-launch` line (the `!`
form for linear chains, named definitions + `elem.` references for branched
graphs) and declarative JSON and YAML (`declarative.rs` schema), all of which
load back into g2g; and the MCP server. Links are validated live: with `g2g-validate-wasm`
built (g2g's real caps solver wrapping `toolingjson::validate_json`, compiled to
wasm and loaded client-side) each edge shows its negotiated caps and a failing
link is flagged; without the blob (the strict-CSP single-file artifact) it falls
back to a coarse caps-family heuristic when the blob is absent. A Vite plugin
embeds the wasm as base64 and instantiates it from bytes, so the solver runs in
`pnpm dev`, the static bundle, and the self-contained artifact alike (no fetch,
CSP-safe). The builder is the one tool with a JS build step (source under
`tools/builder/`, `node_modules` / `dist` / `src/wasm` gitignored); every other
dev tool is a Rust binary or a zero-build page.

`recordsink` / `replaysrc` (std-gated, in `g2g-plugins::record`) turn a live
stream into a file and back, for deterministic repro. `recordsink` writes the
negotiated caps (from `configure_pipeline`) then every `DataFrame` as
length-prefixed `g2g_core::wire` records; `replaysrc` reads the leading record as
its `intercept_caps` result and re-emits the caps + frames as a source, optionally
paced to the recorded PTS (`sync=true`) or as fast as possible (the default, for
deterministic tests). They are ordinary launch-line elements (`... ! recordsink
location=x` / `replaysrc location=x ! ...`), no convenience flag, and the wire
codec is shared with the distributed-graph transports so there is one packet
serialization. A truncated trailing record (a recording cut off mid-write) is
dropped on replay rather than failing.

`g2g-mcp` (the `tooling-json` feature) is a Model Context Protocol server so an
agent can drive g2g development. It speaks newline-delimited JSON-RPC 2.0 over
stdio with no MCP framework dependency (the envelope is hand-rolled with
serde_json), and exposes five tools: `list_elements`, `inspect(element)`,
`validate(pipeline)` (parse + negotiate, no run), `launch(pipeline,
duration_secs)` (run with a deadline, report `RunStats`), and `run_graph`
(a declarative JSON / YAML document by path or inline, advertised only in
`declarative` builds, same run conventions). Both run tools stream live
telemetry while running when the client supplies a `progressToken`: periodic
`notifications/progress` carrying the dashboard's snapshot shape
(`toolingjson::telemetry_json`, the single serializer both consumers share). The tool bodies live in
`g2g-plugins::toolingjson`, shared with `g2g-inspect --json` so the registry-dump
and run shapes have one definition; the async tools drive a current-thread tokio
runtime via `block_on` while the stdio loop stays synchronous.

The `validate` path returns a structured negotiation report, not just pass/fail.
`negotiate_graph` flattens a solve conflict to an opaque `CapsMismatch`;
`negotiate_graph_explained` (its inner) instead returns `NegotiateError`, which
splits a setup failure (`Setup(G2gError)`, e.g. a source caps-probe I/O error)
from a solve conflict (`Solve(NegotiationFailure)`, the structured detail naming
the offending link). `toolingjson::validate_json` reports, on success, the
negotiated caps per edge with the edge's endpoint node indices, and on a solve
conflict the failure kind (`empty-link`, `unfixable`, ...) plus those indices, so
a caller can highlight the failing link. On an `empty-link` the solver also
captures both candidate sets at the failing intersection (`CapsConflict`,
upstream produce vs downstream accept, `Option`al since some sites hold only
one side), and `validate_json` renders them as gst caps strings
(`upstream_caps` / `downstream_caps`).

`toolingjson::observed_graph_json` (`g2g-launch --run-json`) reports the same
graph after running it instead of before: every link's `SenderSink` records the
last `CapsChanged` that entered it on the edge's `EdgeCounters`, so an
`Observer` snapshot carries both the solved caps and the observed ones
(`EdgeInfo::observed_caps`). The dump prefers the observed reading and tags each
edge `caps_source` `runtime` or `negotiated`, which is what makes a
placeholder-then-refine stream (a demuxed file) comparable against an engine
that only reports post-run caps.

### 4.20c Developer Tooling: Conformance and Derived Maturity

Because g2g grows fast under agent-driven development, "how validated is this
element?" has to be answerable without trusting a hand-written label, which under
fast iteration drifts into an overclaim (a maturity bumped in the same change that
adds the feature). `conformance` (`g2g-core`, pure) makes maturity a *derived*
value: an element's `MaturityRecord` is a bag of `Evidence`, each tagging one
`ConformanceDimension` (`Instantiate`, `Properties`, `RoundTrip`, `LossResilience`,
`ZeroCopy`, `Latency`, `Oracle`, `Hardware`) that a check actually verified, plus
the platform / codec / peer it verified against. `MaturityRecord::level()` derives a
`MaturityLevel` (`Unverified` < `Instantiated` < `UnitTested` < `InteropTested` <
`HardwareValidated`) from that bag with no setter, and with honesty guards:
`Oracle` reaches `InteropTested` only with a named peer, `Hardware` reaches
`HardwareValidated` only with a named platform. So the *absence* of evidence is the
signal, a loopback-only element carries no `Oracle` evidence and stays `UnitTested`,
which is the "not interop-validated against reference gear" caveat expressed as data
rather than prose. The conformance batteries (`g2g-plugins::conformance`) exercise a
*real* element (never a mock) with cheap in-process checks and add evidence only on a
pass, so the level is computed from behavior observed this run, not asserted: a
regression that breaks a round-trip drops the level. They cover the sans-IO cores
several transports share: the ST 2110-20 / -30 packetizer pairs (including the -7
seamless merge through per-path loss), the RFC 6184 H.264 payload core
(`rtph264`: FU-A fragmentation reassembled byte-exact, and a dropped fragment
costing only its own access unit rather than welding two together), and the RTP
jitter buffer (`rtpjitter`: reordered arrival released in sequence order, a hole
reported for NACK then skipped rather than stalling). `g2g-inspect --maturity` runs
the battery live and renders the matrix. `Oracle` / `Hardware` evidence, which the
in-process battery cannot produce (it has no ffmpeg / GPU), comes from the
resource-owning integration tests: they append it to a tab-separated evidence log
(`persist::record_evidence`, path `$G2G_CONFORMANCE_LOG`) when a check passes, and
`full_report` folds that log into the in-process report so `--maturity` shows the
`InteropTested` / `HardwareValidated` rows too. The native-muxer oracles mux an
`Mp4MuxN` fMP4 / a `TsMux` transport stream and have `ffprobe` demux them back,
recording peer-tagged `Oracle` evidence deriving `mp4mux` / `mpegtsmux` as
`InteropTested`; the ffmpeg-interop transports carry this further, `udpsrc` (RTP),
`rtmpsrc`, `srtsrc` / `srtsink` (libsrt, incl. the AES variants), and both RTSP
directions (`rtspserversink` played by ffmpeg, `rtspserversrc` published into by
ffmpeg over UDP and TCP-interleaved) each derive `InteropTested` against a named
reference peer, and the Vulkan Video decode tests
persist GPU-tagged `Hardware` evidence (via `VulkanVideoDevice::device_name`) so
`vulkanvideo` derives `HardwareValidated` across H.264 / H.265 / AV1. The rest of
the GPU stack persists the same tier from the tests that own the device: the
native NVIDIA codecs (`nvenc` encoding a CUDA-resident surface, `nvdec` decoding
into one and downloading for a System-only sink) and the `cudawgpu` bridge tag
their evidence with the CUDA device the driver names
(`persist::cuda_platform_tag`, sourced from the GPU device provider rather than
hardcoded), while the dma-buf export pair (`wgputodmabuf` / `dmabuftowgpu`) tags
the subsystem, since each element opens its own high-performance Vulkan adapter
and so cannot honestly name which card ran it. A CI
`conformance` job runs the deterministic ffprobe oracles plus the (best-effort)
transport interop against a real ffmpeg, aggregating into one `$G2G_CONFORMANCE_LOG`
(the muxer oracles honor an externally-set log so they append rather than truncate)
and publishing `--maturity` to the job summary; the GPU `Hardware` rows come from a
self-hosted GPU runner. Together with the copy
plan (§3.2), this is
the validation-first posture: the framework states hard, checkable properties (this
graph is zero-copy; this element is unit-tested but not interop-validated) rather
than leaving them to prose and trust.

### 4.20d Developer Tooling: Codec Goldens and PSNR

The conformance dimensions above say whether data *survived* an element. For a
codec that is not enough: a decoder that starts producing different pixels after a
dependency bump, or an encoder that quietly stops applying its bitrate, still
round-trips frames of the right size and shape. `ConformanceDimension::Quality` is
the dimension for the pixels and samples themselves, and it counts as behavioral
evidence (a `Quality`-only element derives `UnitTested`, and with a peer-tagged
`Oracle` alongside it, `InteropTested`). Its measurement helpers live next to the
batteries in `g2g-plugins::conformance`, dependency-free like the rest: `fnv1a_64`
(a stable digest for a committed golden), `i420_planes`, and `psnr_db` /
`pooled_psnr_db` (per-plane and sample-count-pooled peak signal-to-noise ratio,
infinite for identical input, `std`-gated only because `no_std` has no `log10`).

Three battery kinds produce that evidence, in `g2g-plugins/tests/m1001_*`. The
**decoder goldens** decode a committed fixture with the in-repo decoder and hash
the raw output against a value recorded in the test: `rav1ddec` over
`av1_640x480.obu`, `mjpegdec` over the two 16x16 JPEGs, `opusdec` and `vorbisdec`
over their Ogg fixtures, and behind the `ffmpeg` feature `ffmpegdec` over
`h264_640x480.h264`. Each of those codecs decodes bit-exactly by its specification,
so a mismatch means g2g changed rather than the reference moved; AAC is the
exception (libavcodec decodes it in float and is not bit-exact across versions), so
its leg checks determinism and frame alignment instead of a digest. The
**encode / decode PSNR** batteries encode a synthetic source generated in-test (a
gradient with a checkerboard and a walking bar, so there is both smooth and
hard-edged content) and decode it back with the matching in-repo decoder, requiring
the pooled PSNR and the worst single plane to clear a per-codec floor set a few dB
under the figure observed when the battery was written: AV1 (`av1enc` / `rav1ddec`),
MJPEG (`mjpegenc` / `mjpegdec`, measured in packed RGBA because through I420 the
pair converts colorspace twice and the score stops tracking encode quality), and
H.264 (`ffmpegenc` / `ffmpegdec` at a bitrate that keeps libx264 off its quality
ceiling). The **reference-decoder oracle** closes the loop the goldens cannot: it
has the ffmpeg CLI decode the same fixture and measures g2g's decode against it, so
a pass is evidence about correctness rather than stability, and it persists a
peer-tagged `Oracle` row next to the `Quality` one. AV1 must agree sample for
sample there; JPEG agrees to within each decoder's own IDCT and colorspace
rounding. It self-skips where ffmpeg is absent, like the muxer oracles.

The batteries are codec-feature-gated, so unlike the always-on ST 2110 / RTP
batteries they cannot run inside `g2g-inspect --maturity`; they persist their rows
to `$G2G_CONFORMANCE_LOG` and `full_report` folds them in, the same path the
`Hardware` rows take. CI runs the goldens and the PSNR floors in the Linux feature
job and the ffmpeg oracle in the conformance job.

### 4.20 Distributed Graphs (`remotesink` / `remotesrc`)

A graph is normally one process, but a pipeline stage is not bound to the
machine that produced its input. The **distributed-graph primitive** lets any
edge be cut and the downstream subgraph run in another process or on another
machine, without rewriting the graph: replace the edge with
`... ! remotesink host=H port=P` on the near side and `remotesrc port=P ! ...`
on the far side. This is the general form of the browser-to-server offload the
web track prototyped (a bespoke RGBA-over-WebSocket shim): the same "move a
stage across a boundary by swapping one element" thesis as the portability
story, now for the *network* axis rather than the target axis.

The foundation is a target-agnostic **wire codec** in `g2g-core` (`wire.rs`,
`no_std + alloc`, no dependency): `encode_packet` / `decode_packet` serialize an
entire [`PipelinePacket`] to a self-contained, versioned, little-endian byte
buffer and back, covering every variant, every `Caps` shape, the frame timing /
sequence, and (with the `metadata` feature) the `AnalyticsMeta` detection graph
and `BlobMeta` side-data in band. Because it is pure computation it compiles on
every target the core does, `wasm32` included, so a browser client and a native
peer speak the identical format. Only CPU memory crosses the wire:
`MemoryDomain::System` bytes verbatim, `SystemView` materialized to dense bytes;
a device-resident domain (CUDA / wgpu / D3D11 / DMABUF) returns
`WireError::UnsupportedDomain`, so a GPU frame must pass an explicit download
element first, exactly as reaching any CPU sink already requires.

`RemoteSink` (the `remote` feature, a `std + tokio` element pair) is the TCP
client: it accepts any caps (`caps_constraint_as_sink` = `AcceptsAny`), connects
in `configure_pipeline`, and forwards each packet length-framed (`u32` length,
then the wire body), emitting the negotiated caps as the first packet so the
receiver learns the media type from the stream. `RemoteSrc` is the TCP listener:
it accepts one connection and *discovers* its output caps from that first
`CapsChanged` (the async caps-discovery pattern `RtspSrc` uses), then re-emits
the leading caps and every subsequent packet downstream, ending on the sender's
`Eos` or a clean close. A `metadata`-off receiver ignores a `metadata`-on
sender's meta payload (it is the last field of a `DataFrame` body) rather than
mis-parsing, so a mixed-feature deployment degrades to no metadata, never to
corruption.

**WebSocket transport (`remote-ws`).** `RemoteWsSink` / `RemoteWsSrc` are the
WebSocket siblings of the TCP pair, carrying the identical wire-codec stream over
a WebSocket connection (via `tokio-tungstenite`). WebSocket is already
message-framed, so one `encode_packet` body is one binary WebSocket message,
with no `u32` length prefix; the protocol is otherwise identical (caps as the
first message, discovered by the server in `intercept_caps`). `RemoteWsSink` is
the client and `RemoteWsSrc` the listening server, matching the TCP roles; the
one behavioural difference is that the WebSocket handshake is async, so the sink
connects on its first `process` rather than in `configure_pipeline`. The point of
the WebSocket variant is reach: a browser peer can speak only WebSocket, so this
is the transport that lets a `g2g-web` graph join the same primitive. On the
browser side, `WsWireSink` (`g2g-plugins`, `web`) is the wasm send half: it wraps
the browser `WebSocket` API around the same `encode_packet`, so a browser graph
`... -> WsWireSink` ships an edge to a native `RemoteWsSrc -> ...`. Because the
wire codec compiles unchanged on `wasm32`, the browser and the native server
literally share the serializer.

**Remote transform (`RemoteWsTransform` / `WsWireTransform`).** A one-way edge cut
runs the *whole* downstream subgraph remotely, but some stages must stay put
around the offloaded one: a browser detection offload can move only inference,
because decode and the overlay + canvas present are browser-bound. That is a
*remote transform*: it ships each input packet to a peer over one WebSocket and
emits the processed packet the peer returns, keeping the graph shape. Caps are
identity (the remote stage may attach `metadata`, e.g. `AnalyticsMeta`
detections, which crosses in band, but does not change the format). The protocol
is strictly FIFO so each per-frame read pairs with its frame: the leading
`CapsChanged` (config, no reply), then one `DataFrame` per frame (one processed
reply each), then `Eos`; `Segment` / `Flush` pass through locally. The native
`RemoteWsTransform` (tokio-tungstenite client) offloads a middle stage to another
machine; the browser `WsWireTransform` is its wasm twin. This is what fully
collapses the bespoke M549 `WebRemoteDetect` shim (a hand-rolled RGBA-up /
boxes-down protocol that knew about detection) onto the primitive: the browser
graph `WebSocketSrc -> WebCodecsDecode -> WsWireTransform -> AnalyticsOverlay ->
CanvasSink` moves inference to a native peer (a wire server running the real
`OrtInference -> DetectionPostprocess` chain, attaching the boxes as
`AnalyticsMeta`) by swapping one generic, detection-agnostic element. The
tradeoff versus the bespoke shim is bandwidth: the transform round-trips the
whole frame both ways (the honest cost of a generic packet-in / packet-out
stage), fine on a LAN; a `metadata`-only return for the pixels-unchanged case is
a future optimization.

**WebTransport transport (`webtransport`).** `RemoteWtSink` / `RemoteWtSrc` /
`RemoteWtTransform` (M901) are the third carrier of the same wire codec, over one
reliable bidirectional WebTransport stream per connection (HTTP/3 CONNECT over
QUIC, via `web-transport-quinn`). A WebTransport stream is a QUIC stream, so it is
a byte stream with no message boundaries: the framing is the TCP pair's `u32`
length prefix, shared with it verbatim, not the WebSocket pair's
one-message-per-packet. The protocol above that is identical (caps as the first
message, discovered by the server in `intercept_caps`; the transform's FIFO
frame-out / processed-frame-back round trip), and reconnection behaves as below.
What it adds over the WebSocket carrier is the QUIC connection under it:
head-of-line blocking is per stream, the handshake is 1-RTT, and a browser peer
reaches it with `new WebTransport(url)` and no TLS-terminating proxy in front.
QUIC is always TLS, so unlike the other two servers this one cannot start without
a `certificate` / `private-key` PEM pair, and a client that will not trust a
system root names the certificate by SHA-256 digest in
`server-certificate-hashes` (the browser API's `serverCertificateHashes`).
Datagram mode (unreliable, MTU-bounded) is a separate carrier and is not used:
this milestone is reliable-stream only.

The three carriers share their machinery rather than repeating it: `RemoteClient`
(send side) and `RemoteSource` (receive side) are generic over a transport, and
`RemoteTransform<T>` (the remote stage) is generic over a `PacketDuplex`
transport, so each carrier file supplies only what is transport-specific (how a
connection is dialed or a listener bound, how one packet is written and read) plus
its element identity and properties.

Reconnection (M558) makes the edge resilient across the transports.
`RemoteSink` / `RemoteWsSink` / `RemoteWtSink` gain `with_reconnect(attempts)` (and a
`reconnect-attempts` property): the initial connect is deferred and retried with a
short backoff, and a mid-stream send failure drops the dead socket, reconnects,
and re-sends the current caps (the far side's required first packet) before
retrying, so a peer that starts late or restarts is transparently tolerated up to
the attempt budget. Symmetrically, `RemoteSrc` / `RemoteWsSrc` / `RemoteWtSrc`
gain `with_reconnect()` (a `keep-listening` property): a client that drops *without* a
clean `Eos` is not the stream's end; the source keeps its listener open, accepts a
replacement client (which re-sends its leading caps, forwarded downstream so it
re-negotiates if changed), and continues. Only an explicit `Eos` (or a frame
limit) ends a keep-listening source. Both directions are validated over loopback
(the sink retries until a late-binding server appears; the source stitches a
stream across a sender that drops and is replaced).

Remaining follow-ups: a native WebSocket server that *pushes* an unsolicited
stream to a browser `WsWireSrc` client (a receive-only browser edge, as opposed to
the transform's request/response), a subgraph-as-a-unit wrapper (remoting a whole
`Bin` rather than a single edge), and a WebTransport datagram carrier for the
drop-tolerant case.

### 4.20a MoQ Transport (`moqt` / `moqtsink` / `moqtsrc`)

The distributed-graph carriers above move g2g's *own* packet stream between g2g
peers. **MoQ Transport** is the other use of the same WebTransport carrier: a
published IETF media protocol, so the peer on the far side is a relay and a
player that know nothing about g2g. `moqt` (M902 / M903) implements it in-tree,
both directions.

**Dialect and version.** The dialect is the IETF draft, not moq-lite: moq-lite
is a single-vendor dialect with its own ALPN and cannot talk to IETF endpoints.
The versions are **draft-16, `0xff000010`** (what Cloudflare's `moq-relay-ietf`
runs in production) and **draft-18** (what moq-dev, imquic, moqxr and Meta's
public moxygen relay speak). Nothing on crates.io implements the IETF draft, so
the wire layer is written here, the way the SRT and ST 2110 stacks were: read
the draft, read the reference implementation (`cloudflare/moq-rs`), and validate
against the reference peer.
From draft-16 the version is *not* negotiated in the SETUP payload; the QUIC
ALPN for WebTransport is always `h3`, so the version rides the HTTP/3 CONNECT
request as the WebTransport subprotocol `moqt-16`, and CLIENT_SETUP /
SERVER_SETUP carry parameters only.

**Version negotiation (M907).** The elements offer every version in their
`versions` property (default `18,16`, preference order) as WebTransport
subprotocols on one CONNECT, and the server's pick selects the codec for the
session; `moq-relay-ietf` echoes `moqt-16` when offered it. A server that
echoes no subprotocol predates multi-version offers and every such server is a
draft-16 peer, so the fallback is draft-16 when it was offered; the SETUP
handshake that follows validates the choice either way.

**Draft-18 (`moqt::v18`, M907).** Between draft-16 and draft-18 the wire was
restructured, so draft-18 is a sibling module rather than a flag on the
draft-16 one: its own `vi64` integer (leading-ones length prefix, 1 to 9 bytes,
a full `u64`, non-minimal encodings legal), a single SETUP message (`0x2F00`)
on a *pair of unidirectional control streams*, one *bidirectional stream per
request* whose response carries no request id (the stream is the correlation),
typed control-message parameters in place of KVP parameters (an unknown type
cannot be skipped and is a session error), bit-table SUBGROUP_HEADER and
OBJECT_DATAGRAM types, PADDING streams and datagrams to discard, and
cancellation by stream reset (UNSUBSCRIBE, FETCH_CANCEL and MAX_REQUEST_ID no
longer exist). What is genuinely version-agnostic is shared, not copied: track
namespaces, Key-Value-Pairs and the object-id delta rule live in the draft-16
coding module with a varint flavour on the shared `Reader`, and the reorder
policy, catalog and the M901 carrier are reused as they are. The publisher
answers each request on its own stream and sends PUBLISH_DONE there at EOS;
the subscriber drains PUBLISH_DONE's stream count before ending a
subscription, because the message races the data streams it is counting.
FETCH is refused (`NOT_SUPPORTED`) on both drafts.

**Layering.** `moqt::coding` (varints, byte strings, track namespaces and names,
the delta-coded Key-Value-Pair sequences), `moqt::message` (the control message
set and its `type / 16-bit length / payload` framing), `moqt::data` (the
subgroup stream header and per-object header), `moqt::datagram` (the datagram
object), `moqt::reassembly` (decoding a subgroup stream, and the ordering policy
below) and `moqt::catalog` (the JSON
track list, written and read in one place so the two cannot drift) are pure
`alloc` with no I/O, so the wire layer is unit-testable on byte vectors; the
layouts are asserted against the byte sequences the reference implementation
asserts for itself, because a round trip alone cannot catch two fields swapped
with each other. `moqt::session` adds the live session over the M901 carrier: it
reuses that carrier's dial (`remotewtio::dial`, which grew a subprotocol
argument rather than a second copy of the certificate-hash handling), opens the
control stream as the session's first bidirectional stream, and runs the control
stream's read half in its own task, so a SUBSCRIBE is decoded as it arrives
instead of when the element next has a frame to push. A subscriber additionally
starts a data reader: one task accepting unidirectional streams, one task per
stream decoding it, all funnelling whole objects into a single channel.
Everything a peer sends is bounded before use: counts and lengths are checked
against the draft's limits, the KVP running key is a checked add, nothing is
preallocated from a peer-supplied count, a single object is capped
(`max-object-size`) so one stream cannot allocate without limit, and a message
that does not consume exactly its declared length is a protocol violation.

**`moqtsink`.** The publisher takes an ISO-BMFF byte stream
(`... ! mp4mux ! moqtsink location=https://relay:4443/ namespace=live/cam`), so
the muxer stays a separate element and the sink carries no second fragmenter; it
walks the boxes with the same helpers the HLS segmenter uses
(`fmp4::trun_first_sample_is_sync` is shared between them). The object mapping:

- `ftyp`+`moov` is one object in group 0 on the init track (`0.mp4`), which is
  what a subscriber fetches first.
- each `moof`+`mdat` pair, with the `styp` / `prft` that open its segment, is one
  object on the media track `{track_id}.m4s`. CMAF requires an object to hold at
  least one whole chunk, and a `moof`+`mdat` pair is exactly that.
- a fragment whose first sample is a sync sample starts a new **group**, so a
  group is a GOP and each group is one subgroup on its own unidirectional
  stream. A subscriber that joins mid-group is served from the next keyframe,
  which is the only point it could start decoding anyway.
- a `.catalog` track carries the JSON track list a player reads to learn the
  track names and codec parameters.

Subgroup streams carry the header type `0x15` (explicit subgroup id plus an
extension-header block) and objects whose id delta is measured per stream (the
first object of a stream takes the delta as its absolute id), which is
byte-identical to what the reference publisher emits when one stream carries the
whole group. `subgroups` spreads a group's objects across that many concurrent
subgroup streams round-robin, so one object's loss no longer holds up the next
inside a GOP. The reference relay cannot carry that: it renumbers each subgroup's
objects from zero, discarding the delta, so the subgroups collide on one set of
ids and all but one are dropped as duplicates, so a publisher aimed at that relay
leaves `subgroups` at 1. `SUBSCRIBE_OK` reuses the request id as the track alias
(§10.1 only asks for uniqueness within the session), and a stream is opened only
after that acknowledgement, so the subscriber can resolve the alias in the
stream header. A subscriber-side request the publisher does not serve (FETCH,
TRACK_STATUS, REQUEST_UPDATE) gets an explicit `REQUEST_ERROR NOT_SUPPORTED`
rather than silence. Draft-16 SUBSCRIBE carries neither a group order nor a
filter field, so `priority` (the publisher-priority byte in every subgroup
header) is the only delivery knob and there is no group-order property to
expose.

The control plane runs without frames (M906). `configure_pipeline` dials the
relay and publishes the namespace in the background (a sync caller without a
runtime falls back to dialling on the first frame, and a failed dial is retried
per frame while the relay comes up), and a pump task owns the control stream's
inbound half, answering each message as it lands; the pump and frame publishing
take turns on one lock, so a subscription never changes hands inside a frame.
A media SUBSCRIBE that arrives before the `moov` names any track is held (the
queue is bounded, since request ids are peer-controlled) and resolved with
`SUBSCRIBE_OK` or `DOES_NOT_EXIST` once the `moov` arrives; init and catalog
subscriptions are served the moment their single object exists.

**Datagram objects.** `datagrams=true` carries each media object in a QUIC
datagram instead of on a subgroup stream: unreliable, bounded by the path MTU,
and free of head-of-line blocking, which is what a live path wants for droppable
media. It is off by default because it changes the delivery guarantee. The
layout (`moqt::datagram`, from `moq-transport/src/data/datagram.rs`) is a type
table like the stream header, saying which of the object id, the extension
block and the object status are present, whether a payload follows, and whether
the object ends its group; the payload has no length prefix, since the datagram
boundary ends it. A datagram is a whole message that will never be continued, so
a short one is a protocol violation rather than something to wait for, and a
datagram that does not decode is dropped rather than failing the session: an
unreliable carriage already loses objects, and killing the session over one bad
datagram would lose the rest.

Three consequences shape the publisher. An object larger than the path MTU
cannot be a datagram, so it falls back to a subgroup stream rather than being
dropped, which the delta coding above already handles (a stream opened mid-group
starts from an absolute object id). The init and catalog tracks always ride
streams: losing either loses the whole broadcast. And a group carried only by
datagrams has no stream whose close says it is finished, so the publisher sends
an end-of-group status datagram at each group boundary; without it the
subscriber would hold the group until a buffering bound moved it on. The
subscriber feeds datagram objects into the same `Reassembler` as stream objects,
so ordering, the bounds and the never-stall policy are one implementation and
not two.

**`moqtsrc`.** The subscriber is the inverse and emits a
`ByteStream{IsoBmff}` a demuxer takes unchanged
(`moqtsrc location=... namespace=live/cam ! fmp4demux ! ...`). It reads the
`.catalog` track to learn the media tracks and their init track (falling back to
the reference defaults, `0.mp4` and `{track_id}.m4s`, when the publisher
publishes none), emits the init object as the first frame, then each media
object as it comes into order. `track-name` picks a track other than the
catalog's first.

**Reassembly** is the substance of the receive side. A subgroup is its own
unidirectional QUIC stream, so a track's streams arrive concurrently and its
objects are ordered by (group id, object id) *across* streams, not by arrival.
Object ids are delta-coded per stream (the first object's delta is its absolute
id, each later one is `previous + delta + 1`). `Reassembler` holds a cursor and
a fixed memory budget:

- playback starts at the first group whose object 0 arrives, so joining
  mid-group skips to the next group rather than emitting a partial one;
- objects are emitted strictly in cursor order, and anything below the cursor (a
  late stream, a duplicate) is dropped and counted;
- a group is done when every stream that carried it has finished, or an
  `EndOfGroup` object closed it; then the cursor moves to the next group id;
- a hole in a group that is already done can never be filled, so the cursor
  jumps to the lowest object still buffered in it;
- **a group that never completes is bounded**, not waited on: past `max-groups`
  or `max-buffer-bytes` the oldest group is dropped whole and the cursor moves
  past it, so buffering never grows and the stream resumes at the next group
  boundary instead of stalling. Objects for a group already left behind are
  refused rather than reordered backwards.

Because a data stream can outrun the control stream that names its track alias,
events for an alias no subscription claims yet are held (under the same byte
budget) and replayed when SUBSCRIBE_OK arrives.

Validation is against the reference peers rather than a loopback, in both
directions: `moqtsink` publishes through a locally spawned `moq-relay-ietf` and
the bytes `moq-sub` writes on the far side are compared to the bytes that went
in; `moq-pub` publishes through the same relay into `moqtsrc`; and the g2g round
trip `mp4mux ! moqtsink` -> relay -> `moqtsrc` is compared byte for byte, over a
run long enough to span a group boundary.

**The browser leg** (M904, `tools/moqt-demo/`) is the independent one: everything
above validates against Cloudflare's Rust or our own, and a JS client is neither.
The page subscribes with [MOQtail](https://github.com/moqtail/moqtail), a
third-party draft-16 implementation, reads the `.catalog`, fetches the init track
and appends the `moof`+`mdat` objects to one MSE `SourceBuffer` unchanged, so the
browser's own demuxer and H.264 decoder consume the exact bytes `moqtsink` wrote.
`headless/run-moqt-play.mjs` drives it in headless Chromium against a locally
spawned relay and asserts on what the decoder produced: frame count, decoded
size, and the seven SMPTE bars sampled off a canvas. `watch-live.mjs` is the same
plumbing with `libcamerasrc` as the source and a real browser window.

Two constraints shape it. A browser accepts a self-signed relay only through
`serverCertificateHashes`, which requires an ECDSA P-256 certificate valid at most
14 days, and a certificate that signs itself cannot be the leaf
(`CaUsedAsEndEntity`), so the harness mints a CA and a leaf under it. And the
catalog and init tracks each hold one object published before any subscriber
exists, so they must be subscribed with an absolute start at group 0: a
latest-object filter delivers nothing.

The datagram and multi-subgroup paths cannot be validated that way, because
`moq-relay-ietf` has no datagram code at all and drops all but one subgroup of a
group. They are validated `moqtsink` -> `moqtsrc` directly over a real QUIC
connection, through a test peer that answers each side's CLIENT_SETUP and then
copies control messages, streams and datagrams byte for byte with no track state
of its own, so what the subscriber decodes is what the publisher encoded; the
datagram byte layouts are asserted against vectors the reference implementation's
own encoder produced. The relay is still exercised on those settings, for what it
proves: that a subscriber keeps playing, in order and a whole fragment at a time,
when a peer delivers only part of every group.

### 4.21 Local Zero-Copy IPC (CUDA)

Everything above ships CPU bytes: the wire codec refuses device memory, so a GPU
producer feeding a GPU consumer in *another process* pays a full
device->host->device round trip to cross. On the same machine that copy is
avoidable, because two processes can map the *same* VRAM. `localipc` (the
`local-ipc` feature, NVIDIA-only via the `cuda` gate) is the CUDA path:
[`ipc_export`] turns a `CUdeviceptr` into a 64-byte `CudaIpcHandle`
(`cuIpcGetMemHandle`) that another process passes to [`ipc_open`]
(`cuIpcOpenMemHandle`) to obtain a pointer to the same allocation, reading the
producer's VRAM with no copy.

The design point that makes this cheap: **a CUDA IPC handle is plain bytes**,
unlike a DMABUF file descriptor (which needs `SCM_RIGHTS` fd-passing over a Unix
socket to be meaningful in another process). So the handle rides *any* byte
transport already in the tree, even the wire codec itself, the only constraints
being that the two ends share a machine and a GPU (a handle from device 0 is
meaningless on device 1), that the exporting allocation stays live until the
importer opens it (the producer frame's keep-alive covers this), and that the
importer closes before the exporter frees. The `cuda_ipc_smoke` example validates
the whole path cross-process on real hardware: a parent fills a device
allocation, exports the handle, spawns a child that maps and reads it back
byte-for-byte, with only the 64 bytes crossing between processes (proven on an
RTX 3060).

On top of the primitive, `LocalCudaSink` / `LocalCudaSrc` (the GPU-resident
analog of `RemoteSink` / `RemoteSrc`) carry a `MemoryDomain::Cuda` NV12 frame
across a Unix socket: the sink exports the frame's allocation and sends a
descriptor (handle + plane offsets / pitches / dims + timing); the source maps
it, and here makes the one pragmatic concession to lifetime. The producer's
allocation must stay valid until the consumer is done, and coupling two
processes' whole pipelines is fragile, so the source takes a single **on-GPU**
device->device copy (`cuMemcpyDtoD`, still no PCIe) into its own buffer and then
acks; the sink holds the source frame only until that ack (one frame in flight),
so the two lifetimes decouple cleanly and the design is independent of the
runner's frame-drop timing. The `local_cuda_transport` example validates the full
element path cross-process on an RTX 3060 (NV12 frames verified pixel-exact in the
receiving process).

`LocalCudaSrc::zero_copy()` removes even that receive-side copy: the source emits
the producer's *mapped* VRAM directly, so the consumer reads the producer's
memory in place (e.g. NVENC-from-mapped, no copy anywhere). The lifetime handshake
copy mode traded away now returns: the emitted frame's keep-alive closes the IPC
mapping *and signals the run loop on drop*, and the source acks the producer only
once that fires (the frame is fully consumed downstream), so the producer holds
the source allocation exactly until the consumer is done. That is real
backpressure (one frame in flight across the boundary, the producer stalls for the
consumer), which is why it is opt-in: the default copy mode decouples the two
pipelines and suits a slow or fan-out consumer, while `zero_copy()` suits a
prompt, single-in-flight consumer where eliminating the copy matters. Both modes
are validated cross-process on the 3060 (the example runs either via
`G2G_ZEROCOPY=1`).

#### Vendor-neutral DMABUF transport

The GPU-agnostic counterpart is `DmaBufSink` / `DmaBufSrc` (the `local-dmabuf`
feature, Linux). A dma-buf is *not* plain bytes: it is a file descriptor, so the
byte-handle model above does not apply. Instead the sink passes the frame's
dma-buf fd to the source as `SCM_RIGHTS` ancillary data of a `sendmsg` over a Unix
socket (hand-rolled `sendmsg` / `recvmsg` FFI in `scmfd`, no crate dep; Linux LP64
only), and the kernel installs a *dup* of the fd in the receiver. This makes the
transport both simpler and safer than the CUDA path: the underlying buffer is
kernel-refcounted across both processes' fds, so once the sink's `sendmsg`
returns the receiver's dup already keeps the buffer alive and the sink may drop
its frame immediately, with **no per-frame ack** (backpressure still comes from
the graph's bounded channel upstream). Every message is a fixed-size record sent
and received with a single `sendmsg` / `recvmsg`, so a frame record's fd is never
separated from its bytes and a plain read never crosses (and thus discards) a
pending fd. The transport is GPU-agnostic, carrying *any* dma-buf (a GPU-exported
texture, a V4L2 / CSI capture buffer, a `dma_heap` / `udmabuf` allocation);
importing the received fd into a wgpu buffer is the separate `dmabuf-wgpu`
([`DmaBufToWgpu`]) element on the receive side. The `local_dmabuf_transport`
example validates the whole path cross-process with a genuine `udmabuf` (a
CPU-mappable dma-buf built from a sealed memfd), so it needs no GPU: each frame's
bytes are mmap-verified in the receiving process.

#### Exporting a GPU frame to a dma-buf

`WgpuToDmaBuf` (M559, the `dmabuf-wgpu` feature) is the GPU producer that pairs
with [`DmaBufToWgpu`] across the boundary: it consumes a GPU-resident
`MemoryDomain::WgpuBuffer` and emits a `MemoryDomain::DmaBuf` referencing the same
pixels, so a rendered / decoded GPU frame leaves the process with no CPU copy
(feed the output to `DmaBufSink`). A wgpu-allocated buffer is not itself
exportable, so the element allocates its own Vulkan buffer backed by
`VkExportMemoryAllocateInfo` (dma-buf handle type), copies the input into it on
the GPU, and exports the memory as a dma-buf fd with `vkGetMemoryFdKHR`. The
exported fd is an *independent* reference to the underlying buffer (dma-buf
refcounting), so the element frees its own Vulkan handles immediately and the fd
keeps the memory alive (and, once `DmaBufSink` sends it, the receiver's
`SCM_RIGHTS` dup does). Because the input and the exportable buffer must share one
`wgpu::Device` for the copy, a producer feeds this element on its device (exposed
via `gpu()` / `wrap_buffer`). By default the element waits for the copy to finish
(`device.poll(Wait)`) before exporting, so a consumer sees complete pixels;
`with_external_semaphore(true)` replaces that stall with an exported timeline
semaphore (see the synchronisation paragraph below). Validated on the RTX
3060: a buffer exported to a dma-buf and re-imported by `DmaBufToWgpu` on a
*separate* wgpu device reads back byte-exact (`m559_wgpu_dmabuf_export`), which
also confirms dma-buf export+import work on this NVIDIA driver. Packed
RGBA/BGRA/YUYV and 8-bit NV12 are supported; the plane-aware frame size (a packed
format is one plane, NV12 / I420 add the half-height chroma region) and the row
stride are `RawVideoFormat::frame_bytes` / `row_stride`, which both the export and
the `DmaBufToWgpu` import use, so they always agree on the buffer size (this also
fixed the importer, which previously imported only the luma plane of a planar
frame).

The whole GPU-egress stack composes end-to-end across a process boundary:
`WgpuToDmaBuf -> DmaBufSink -> [process] -> DmaBufSrc -> DmaBufToWgpu` moves a
GPU-resident frame from one process to another with only a dma-buf fd crossing
(SCM_RIGHTS) and no CPU copy either side; the lifetimes compose too (the export
frees its Vulkan handles at once, the sink's fd send keeps the buffer alive via
the receiver's dup). The `gpu_dmabuf_ipc` example proves this cross-process on the
3060 (frames re-imported GPU-resident in the child, every pixel verified).

Cross-device / cross-process synchronisation has two modes. The default is the
producer-side `device.poll(Wait)` (a small stall, but correct: the copy is fully
flushed before the fd is handed off). The zero-stall mode
(`WgpuToDmaBuf::with_external_semaphore(true)`, M562) moves the wait to the
consumer via an exported `VK_KHR_external_semaphore_fd` *timeline* semaphore: the
producer creates one exportable timeline semaphore per stream, signals the next
value on each frame's copy submit (`wgpu_hal::vulkan::Queue::add_signal_semaphore`,
no `poll(Wait)`), and attaches the semaphore fd + value to the emitted dma-buf
(`OwnedDmaBuf`'s optional `SyncFd` + value slot). `DmaBufSink` ships the semaphore
fd once (a `TAG_SYNC` record, `SCM_RIGHTS`) ahead of the first synced frame and
tags each frame with its timeline value; `DmaBufSrc` re-shares the one semaphore
across the frames it reconstructs; `DmaBufToWgpu` imports it once and, before
reading, waits for each frame's value by polling the timeline counter
(`vkGetSemaphoreCounterValue`) and yielding cooperatively between polls rather than
blocking the runtime on `vkWaitSemaphores` (the common case, copy already done at
arrival, passes on the first poll). A timeline semaphore (not
a per-frame binary one) keeps the fd single and the wait a plain host wait, so no
multi-fd ancillary passing or per-frame semaphore churn is needed; the producer
reclaims its exportable copy buffers lazily once the timeline counter passes their
value (a non-blocking `vkGetSemaphoreCounterValue`), never stalling yet never
freeing a buffer whose copy is still in flight. wgpu-hal 29 exposes signal-semaphore
injection but not wait injection, so the consumer wait is a CPU-side cooperative
poll (counter poll + `yield_now`, off the runtime's hot path) rather than a
GPU-queue wait; that still removes the producer stall and decouples the two
pipelines. Validated cross-device and cross-process on the RTX 3060
(`dmabuf_timeline_probe` for the bare primitive, `m562_dmabuf_semaphore_sync` for
the element handoff, and `gpu_dmabuf_ipc` with `G2G_DMABUF_SEM=1` for the full
cross-process chain).

---

## 5. First-Class Machine Learning Integration
To prevent GPU-to-CPU synchronization stalls, tensor execution happens directly inside the VRAM domain. ML elements are `AsyncElement` implementations like any other — they negotiate `Caps::RawVideo` on the input pad and `Caps::Tensor` on the output pad.

### 5.1 Inline Tensor Pre-processing via WebGPU (wgpu)
The ML element sits in the same memory domain context as the hardware decoder. When a `MemoryDomain::DmaBuf` arrives at the ML element:

1. The memory handle is bound directly as a texture inside a `wgpu` compute pipeline.
2. An inline compute shader converts color spaces (e.g. NV12 → planar RGB) and performs normalization scales directly in graphics memory.
3. The resulting tensor handle is emitted as a `Frame { domain: VulkanTexture(...), caps: Caps::Tensor { .. }, .. }`, submitted straight to the inference backend.

`WgpuPreprocess` (`g2g-ml/src/wgpupreprocess.rs`, `wgpu` feature) is the compute-shader half: an NV12 frame is converted and normalized in a wgpu compute shader to a `Caps::Tensor { F32, [1,3,H,W], Nchw }`, the same contract `OrtInference` builds on the CPU. The default system-memory variant uploads NV12 to a storage buffer and reads the f32 tensor back to `MemoryDomain::System`. **GPU-output mode (`with_gpu_output`)** instead leaves the tensor in a `wgpu::Buffer` and emits `MemoryDomain::WgpuBuffer` (an on-device GPU->GPU copy into a fresh per-frame buffer, no map / read-back in the element), so a downstream GPU consumer reads it on-device; a CPU consumer pays the deferred read-back via the buffer owner. This removes the output-side GPU->CPU copy; `WgpuInference` (§5.2) is the consumer that binds the resulting buffer on-device, so `preprocess -> infer` keeps the tensor on the GPU. **Surface-import input** closes the other end: when the NV12 frame arrives already GPU-resident as a `MemoryDomain::WgpuTexture` (a `WgpuNv12Texture` keep-alive wrapping an R8Uint texture of `width x height*3/2` in standard NV12 byte layout), the element adopts that texture's device and samples it with `textureLoad` straight into the compute pass, with no CPU upload, bit-identical to the storage-buffer path. **DMA-BUF import input** (`dmabuf-wgpu` feature, Linux) is the same idea for a `MemoryDomain::DmaBuf` frame from a capture or decode path: the element opens a device carrying `VK_KHR_external_memory_fd` + `VK_EXT_external_memory_dma_buf` on the first such frame and binds the imported buffer as the compute pass's input, sharing the importer (`g2g_plugins::dmabufwgpu::DmaBufImporter`, which also honours a producer's timeline semaphore) with the `dmabuftowgpu` element rather than repeating the handshake. The frame's row stride and plane offset reach the shader in the dims uniform, so a padded capture buffer is read in place with no repack, and the tensor is bit-identical to the same pixels uploaded from system memory (validated on an RTX 3060, including a padded stride). NV12 and packed YUYV both have a compute shader (they share every line but the fetch of one pixel's Y, Cb, Cr), YUYV because that is what a UVC webcam captures, so a camera reaches the tensor with no `videoconvert` in front. Which GPU the import opens on is a real choice, not a default: a discrete GPU binds only GPU-visible dma-bufs, while a CPU-backed one (udmabuf, a USB webcam's capture buffer) binds on an integrated GPU, whose memory is the same system RAM. `ImportAdapter` (the `import-adapter` property on both `wgpupreprocess` and `dmabuftowgpu`) picks between them, `high-performance` by default because that is what a GPU-exported dma-buf needs; `integrated` searches the enumerated Vulkan adapters by device type. Either way an fd the driver cannot bind reports `UnsupportedDomain` and the caller falls back to the upload path. The live camera case is validated on this host (`m993_camera_dmabuf_preprocess`): `v4l2src io-mode=dmabuf` at 640x480 YUYV into `WgpuPreprocess` on the AMD integrated GPU gives the same tensor as that same captured frame taken through the copy path, and the same frame on the RTX 3060 refuses the fd, which is the choice being real. With both ends GPU-resident, `capture / decode -> WgpuPreprocess -> WgpuInference` runs with the pixels never touching the CPU. **CUDA<->wgpu interop (`CudaToWgpu`, `g2g-plugins/src/cudawgpu.rs`)** joins the NVDEC decode side to this surface-import path: there is no portable "share this CUDA pointer with wgpu" call, so the bridge allocates an exportable Vulkan image (`VK_KHR_external_memory_fd`, wrapped as a `wgpu::Texture` via wgpu-hal), CUDA imports the same memory by FD (`cuImportExternalMemory`) and copies the NVDEC NV12 planes into it device->device, and the wgpu device travels on the frame's keep-alive so `WgpuPreprocess` adopts it (the device-identity pattern). The whole `NVDEC -> CudaToWgpu -> WgpuPreprocess -> WgpuInference` chain is validated on an RTX 3060, matching a CPU reference with no PCIe download. Shared images are recycled from a reuse pool: the Vulkan image, its CUDA import, and the `wgpu::Texture` are allocated once and returned to a free list when the downstream frame is released (a drop guard on the emitted keep-alive), so per frame only the two device->device plane copies and a sync run; a recycled entry is drained (`Device::poll`) before reuse since a wgpu submission may still sample it. The pool cut the bridge step ~2.6x at 1080p (p50 0.38 ms pooled vs 0.98 ms per-frame-allocated). **The reverse direction (`WgpuToCuda`)** closes the *encode* side: a renderer writes a packed-RGBA `wgpu::Texture` on FD-exportable Vulkan memory (`export_rgba_image` / `wrap_rgba_as_texture`, the `R8G8B8A8` mirror), CUDA imports it as a 4-channel array, and `to_cuda_frame` copies it device->device into a linear `CUdeviceptr` emitted as a `MemoryDomain::Cuda` `Rgba8` frame that `NvEnc` registers as `ABGR` (§4.11.3). So a GPU render reaches the H.264 encoder with no device->host read-back, validated on an RTX 3060 (`wgpu_to_cuda` test). This is the zero-copy egress for server-side rendering / cloud-gaming, and the `bevy-g2g` crate's `RemoteRenderPlugins` is the packaged Bevy proof (M796): Bevy renders on the interop device, g2g copies the target through `WgpuToCuda`, and `NvEnc` emits H.264 without a full-frame download, egressing to WHIP/WebRTC or a file; without an NVIDIA GPU the plugin falls back to a GPU->CPU readback + libx264 encode so the same app streams on any adapter. The crate completes the remote-rendering loop with a WebSocket input backchannel (M797: viewer keyboard/mouse injected as ordinary Bevy input messages; a WebRTC data channel cannot reach the publisher through a WHIP/WHEP server, the viewer being a separate peer connection) and a windowed mode (M798: the scene camera renders to the stream texture and the window shows it through a fullscreen UI mirror, so desktop view and stream are the same pixels). The bridge retains its own CUDA primary context (the GPU the interop device selects) and owns the exportable render-target texture.

### 5.2 Unified Pure-Rust Inference Backends
`g2g` avoids bundling heavy, unsafe proprietary C++ engines. The `g2g-ml` crate provides wrapper elements targeting two execution paradigms:

- **`g2g-ml::burn`** (Embedded / Wasm / RTOS): leverages the pure-Rust Burn framework with a `wgpu` backend, compiling ONNX workflows into type-safe, compile-time Rust graphics shaders. `BurnInference` (`g2g-ml/src/burninfer.rs`, `burn` feature) is the wgpu-backend inference element over the `RawVideo` → `Tensor` contract, driving an `input · W + b` linear layer on any Vulkan / Metal / DX12 / WebGPU adapter. **An ONNX topology runs through the same element**, but the import is build-time: `burn-onnx` (what `burn-import` 0.21 forwards to) generates a burn `Module` plus an embedded burnpack weight blob from the `.onnx` at compile time, so there is no runtime graph loader to hand the file to. The seam is the `BurnModule` trait: one forward pass from the `[1, 3, H, W]` NCHW f32 tensor the element normalizes to `[1, N]` logits. The importing crate implements it over its generated `Model<Wgpu>` and passes it to `BurnInference::module`, which then drives it frame by frame exactly like the built-in linear layer (a forward pass whose output is not the declared `num_outputs` fails the frame, so the emitted `Caps::Tensor` cannot lie). Because the codegen crate drags burn's whole dependency tree into any lockfile that resolves it, the worked case is a workspace-excluded standalone crate, `examples/g2g-onnx-import`: a `Conv2d -> BatchNorm -> ReLU -> global average pool -> linear` graph whose logits match the ONNX Runtime reference for the same frame on the RTX 3060. Attention imports through the same seam: the standard-domain ONNX `Attention` op (opset 23, one node for a whole multi-head block) is lowered by `burn-onnx` onto `burn::tensor::module::attention`, so the GPU runs burn's own attention kernel rather than a hand-unrolled matmul / softmax chain, validated on the 3060 by a second fixture in that crate (pixels as a token sequence -> multi-head self-attention -> mean pool -> linear). Because that node is opaque in the graph, the fixture generator folds the attention formula in numpy and asserts ONNX Runtime agrees before emitting the reference logits, so the reference is not ORT agreeing with itself. This is the topology half of the Burn story, the counterpart of the runtime `safetensors` weight import below.
- **`g2g-ml::ort`** (High-Performance Enterprise Server): wraps ONNX Runtime bindings to pass underlying memory domains to hardware-specific execution paths (CUDA / TensorRT / DirectML / Apple CoreML) natively. Each execution provider is a constructor variant on `OrtInference` that registers the EP ahead of the CPU fallback; registration is best-effort, so the session keeps running (on CPU) when the device is absent. Desktop: `from_memory_with_cuda`, `from_memory_with_directml`. **Android edge**: `from_memory_with_nnapi` (the system NeuralNetworks API: NPU / GPU / DSP), `from_memory_with_xnnpack` (ARM-optimized CPU), and `from_memory_for_android`, which registers NNAPI then XNNPACK then the default CPU EP in one call so ORT assigns each node to the first provider that supports it, the MediaPipe delegate-with-fallback shape. The `nnapi` / `xnnpack` features link symbols only the Android ONNX Runtime build carries, so they are Android-target features (a host build / CI never enables them); the EP stack is validated on a device (`tools/android-nnapi-smoke.sh` runs `g2g-ml/tests/android_nnapi_probe.rs` from `/data/local/tmp`, a binder-threadpool shim for the vendor NNAPI HAL, output byte-exact with the CPU reference). **Edge TPU offload is proven**: an int8 QDQ Conv->ReLU fixture run through `from_memory_for_android` is placed on `NnapiExecutionProvider` (read from ORT's profiling JSON), and on a Pixel 10a (Tensor G4) the DarwiNN HAL log confirms the Edge TPU compiled and executed it (`/dev/edgetpu core0` firmware load); the float-typed input-boundary `QuantizeLinear` is the one op the TPU declines, correctly delegated to CPU (`tools/android-nnapi-conv-smoke.sh`, which also greps the `darwinn` logcat to disambiguate the TPU from other NNAPI accelerators). **Full-graph offload**: a uint8-input variant of the model (the boundary `QuantizeLinear` removed, the graph input retyped to uint8) runs *entirely* on the TPU, every node on `NnapiExecutionProvider` with nothing on the CPU, and the DarwiNN log confirms `Ops supported = ..., not supported = 0` / `compilation finished successfully on google-edgetpu`. The f32->uint8 quantization that feeds such a model is `TensorConvert` (`g2g-plugins`), the tensor-domain sibling of `VideoConvert`: it quantizes an f32 tensor to int8 / uint8 (`q = round(x / scale) + zero_point`, clamped) or dequantizes the inverse, shape and layout passing through. So `preprocess -> TensorConvert(quantize) -> inference` keeps the boundary quantize *out* of the model, leaving the whole inference graph accelerator-eligible. `TensorConvert` also transposes NCHW<->NHWC and narrows / widens f32<->F16 in the same pass, so a model that wants `NHWC uint8` (NNAPI / TFLite) is fed straight from an `NCHW f32` source. `OrtInference` itself accepts the integer input: `from_session` reads the model's input element type and `with_tensor_input` on a u8 / i8 model feeds the quantized tensor straight to the session (RGBA mode stays f32-only). **The whole chain is validated live on the device**: `Camera2Src -> TensorConvert(quantize) -> OrtInference(uint8) ` runs a real camera frame onto the Edge TPU (`tools/android-camera-tpu-smoke.sh`; on a Pixel 10a the logcat shows `accelerator name: EDGETPU` and `compilation finished successfully on google-edgetpu`), the g2g answer to "an edge framework that moves inference between CPU and accelerator" demonstrated end to end on real hardware. The same constructor shape extends to the other vendor accelerators: `from_memory_with_qnn` (Qualcomm AI Engine Direct, the Hexagon NPU / Adreno GPU on Snapdragon, the alternative to reaching the Hexagon through NNAPI) and `from_memory_with_coreml` (the Apple Neural Engine / GPU on macOS / iOS), each behind a target-only feature like `nnapi` (a host build never links them); both are validated to compile for their target, with on-device runtime validation pending the hardware (no Snapdragon / Apple device in CI, like the CUDA EP). This is the heterogeneous-device story (a desktop NVIDIA box, a Windows D3D12 GPU, an Android phone NPU, and the Qualcomm / Apple NPUs all run the same element, the EP picked per platform), the architectural answer to MediaPipe's runtime CPU/GPU delegate switch.

`WgpuInference` (`g2g-ml/src/wgpuinfer.rs`, `wgpu` feature) is the GPU-resident counterpart of `BurnInference`: a raw wgpu compute pass that binds the GPU-resident tensor `WgpuPreprocess::with_gpu_output` (§5.1) produced **directly**, rather than taking `RawVideo` / `System` and uploading. It runs one of a small op zoo on that tensor, selected at construction (each its own WGSL shader behind the shared device-adopt / dispatch / read-back machinery): the original `input · W + b` linear matmul (`linear`); a same-padding stride-1 2D convolution (`conv2d`) over the `[1, Cin, H, W]` NCHW tensor with `[Cout, Cin, KH, KW]` weights, leaving a `[1, Cout, H, W]` feature map; the elementwise activations `relu` / `sigmoid`; and `maxpool2d` / `avgpool2d` spatial pooling. The weighted ops (linear, conv2d) bind a 5-entry group (meta, input, weights, bias, out); the weightless ops (activation, pooling) bind a 3-entry group (meta, input, out), the bind-group layout following the active shader. The conv is the keystone that lets the chain run an actual CNN layer, not just a final classifier; the activation is the nonlinearity that keeps stacked convs from collapsing to one linear map, and the pool the spatial downsampler. Chained GPU-resident (`conv2d -> relu -> maxpool`, each in `with_gpu_output` mode so the data never leaves the device between layers), they are a real small-CNN body, validated on the RTX 3060 against a CPU reference folding the same ops (`conv2d_reference` / `relu_reference` / `maxpool2d_reference`) over the exact tensor the GPU preprocess produced. **Trained weights are imported at runtime** from a `safetensors` file via a dependency-free reader (`g2g-ml::safetensors`, a focused parser for the format's `u64` length + JSON-subset header + raw tensor bytes, no `serde` / no `safetensors` crate): `conv2d_from_safetensors` reads the `[Cout, Cin, KH, KW]` weight and `[Cout]` bias by name and infers the kernel dims, so picking a different trained checkpoint is "parse a different file" while the layer topology stays this compiled element. This is the weights half; the architecture stays Rust (truly dynamic *graphs* at runtime are the `ort` backend's job, and `burn-onnx` build-time codegen is the Burn-side topology path, above). It owns no device: because a `wgpu::Buffer` is bindable only on the device that created it, the element adopts the producer's device / queue (carried by the incoming `WgpuBufferOwner`) on the first frame and submits its compute on the producer's queue, which orders it after the producer's work with no fence or read-back. The logits are read back to `MemoryDomain::System` by default or left GPU-resident (`with_gpu_output`) for a downstream GPU consumer. A burn / ort consumer cannot do this zero-copy: their tensor handles are opaque (no foreign-buffer adopt) and run on their own device, so they would force the GPU->CPU->GPU round-trip the GPU-resident preprocess and inference paths exist to delete.

### 5.3 Native Async Batching Engine
`g2g-ml::batcher` provides a lock-free, multi-channel execution sink that groups separate asynchronous video input streams into a single hardware tensor execution array:

```
[ Camera Stream 1 ] ──► Async Channel ──┐
[ Camera Stream 2 ] ──► Async Channel ──┼─► [ Bounded Batcher ] ──► [ GPU Tensor Core ]
[ Camera Stream 3 ] ──► Async Channel ──┘     (Select / Timeout)
```

### 5.4 Per-Frame Metadata & Detection Post-processing

Inference output is only useful once it is structured and travels with the
picture. Two pieces, both `no_std`-friendly:

- **The metadata system (`g2g-core::meta`, `metadata` feature).** The `Frame`
  carries a `FrameMetaSet`: a list of typed [`FrameMeta`] trait objects (the
  GstMeta analog) with attach / typed-get / iterate and a `propagate(Transform)
  -> Propagation` survival contract (a re-encode drops pixel-derived meta; a
  scale / crop / copy keeps it). Off by default, so the RTOS baseline pays
  nothing (`FrameMetaSet` is a ZST); the field was reserved earlier and built out
  here. The standard `AnalyticsMeta` is the `GstAnalyticsRelationMeta` analog: a
  relation graph of `ObjectDetection` / `Classification` / `Tracking` nodes plus
  directed edges, so a detector → tracker → classifier → overlay chain reads
  results by node kind and traversal instead of re-deriving joins through tensor
  offsets. Bounding boxes are normalized `[0,1]`, so they survive a downstream
  resample without a coordinate rewrite.
- **The first producer (`g2g-ml::DetectionPostprocess`, `analytics` feature).**
  Decodes a YOLOv8-style `[1, 4+C, A]` output tensor (confidence threshold +
  per-class NMS) into `ObjectDetection`s, attaches an `AnalyticsMeta`, and
  forwards the frame. A real client shaping the metadata API (rather than
  speculation) is why the system was deferred to this point.
- **The mask producer (`g2g-ml::OrtSegmentation`, `ort` + `analytics`).** Runs a
  YOLO `-seg` export (Ultralytics YOLOv8-seg / YOLO11-seg) and attaches
  `Segmentation` plus `Roi` nodes to the frame it forwards: an identity transform
  that adds metadata, so the picture and its masks reach an overlay together.
  Both of the model's outputs stay inside the element, unlike the detection split
  (`OrtInference -> DetectionPostprocess`): a tensor frame carries one tensor and
  a mask needs both the box-plus-coefficient output `[1, 4+C+M, A]` and the
  prototype planes `[1, M, mh, mw]`. An instance's mask is the
  coefficient-weighted prototype sum through a sigmoid, read over the instance's
  box at prototype resolution, so a consumer places sample `i` of
  `mask.width()` at `bbox.x + (i + 0.5) / mask.width() * bbox.w` and needs
  nothing else; the `Roi` is the mask-tight sub-box, the region an encoder or
  tracker should treat specially, related to its `Segmentation` by `Contains`.
  The decode is pure Rust (`g2g-ml::segmentation`), so an `ort-web` caller in the
  browser that already holds both outputs reuses it without an element.
- **Metadata through fan-out.** `FrameMetaSet` holds each `FrameMeta` as
  an `Arc<dyn FrameMeta>` and is `Clone`, so a tee clone shares the analytics
  graph by refcount rather than dropping it: the graph runner's
  `try_clone_packet` carries `frame.meta.clone()`, landing the same
  `AnalyticsMeta` on both branches of a `decode -> tee -> {detect, video}`
  diamond. Mutation is copy-on-write via `FrameMeta::clone_box` (the GstMeta
  `copy_func` analog): `FrameMetaSet::get_mut` deep-copies a shared entry before
  the mutable borrow, so a branch editing its analytics never aliases the
  sibling. Still a ZST no-op when the `metadata` feature is off.
- **Metadata through a linear transform.** Fan-out shares the same frame, so
  meta rides for free; a transform that emits a *new* frame (videoscale,
  videoconvert, videocrop, a re-encode) would otherwise drop it. An element
  declares `AsyncElement::meta_transform() -> Option<Transform>`; when it returns
  `Some(t)`, the graph runner clones the input frame's `FrameMetaSet`, applies
  `propagate(t)`, and stashes the survivors on the transform arm's output
  adapter, which attaches them to any outgoing `DataFrame` whose own meta is
  empty (element-authored meta is never overwritten, and a Drop verdict that
  empties the set clears the stash so nothing stale leaks). `None` (the default)
  opts out: a pass-through that forwards the same frame already carries its meta,
  and an element that produces none wants nothing added. The stash is recomputed
  per input frame, so association is exact for a 1-in-1-out transform and
  most-recent-input for a pipelined one (an encoder with lookahead). The standard
  elements declare the obvious mapping: videoconvert `Copy`, videoscale `Scale`,
  videocrop `Crop`, the software video encoders (av1enc, vpxenc, ffmpegenc,
  mjpegenc) `Encode`. Still a no-op when the `metadata` feature is off (the
  method and stash are cfg'd out, so the baseline build is byte-identical).
- **Metadata on demand (the pull half).** `meta_transform` moves metadata that
  already exists; `AsyncElement::meta_requests() -> MetaRequests` is how a
  consumer says which metadata it wants to exist in the first place (the
  GStreamer allocation-query `add_meta` analog). `MetaRequests` is a
  fixed-capacity `Copy` set of `(TypeId, RequestPolicy)` entries (four, sorted so
  equality is order-independent), carried as a field of `AllocationParams`, so
  the demand travels on the allocation cascade that already runs sink → source.
  A producer reads `params.meta_requests.wants::<T>()` in `configure_allocation`
  and can then skip work nobody reads. Downstream demand also crosses a fan-in's
  *output* boundary, where its pool parameters deliberately do not: a compositor
  writes the frames the demand describes. An element with no pool requirement of
  its own still forwards the demand (as `AllocationParams::meta_demand`, which
  accepts every memory domain, and which the source-side reconciliation skips so
  a metadata request can never decide a producer's memory domain). A request is a
  hint, never a guarantee, so a consumer still handles a frame arriving without
  the meta. With nothing declared the cascade is byte-identical to before, and
  without the `metadata` feature `MetaRequests` is a ZST empty set.
- **Demand policy: what a request needs of the other consumers.** Every request
  carries a `RequestPolicy`, because two kinds of metadata combine differently
  when several consumers read one producer's frames. `AnyConsumer` (the default,
  `request::<T>()`): attaching the meta costs a consumer that did not ask
  nothing, so one asking consumer is enough and the demands union
  (`AnalyticsMeta`, `CaptionMeta`, `TimecodeMeta`). `EveryConsumer`
  (`request_from_every_consumer::<T>()`): honouring it changes the *buffer*, so a
  consumer that did not ask would misread it, and the demand only stands where
  every consumer asked. Two folds implement this: `join_branches` at a tee (a
  branch that proposed nothing is still a branch that asked for nothing, and
  vetoes) and `carry_upstream` at each hop (the producer's frames pass through
  that element first, so a hop that does not share the request vetoes it exactly
  as a sibling branch does). The strictest policy wins when two elements ask for
  one meta differently. A demand that dies leaves the cascade as it found it: a
  proposal carrying neither pool constraints nor demand collapses back to none.
- **The buffer's own shape (`PlaneLayout`).** The first meta produced on demand,
  and the `GstVideoMeta` analog: per-plane byte offset and row stride (up to four
  planes, every derived offset checked) for a raw frame whose rows are padded.
  Without it a raw frame is assumed tightly packed, so a producer whose rows are
  not (a GPU readback at the API's 256-byte row alignment, a capture driver's
  `bytesperline`) has to repack them row by row. `WgpuCompositor` asks
  `wants::<PlaneLayout>()` when the cascade configures its output: when a
  consumer downstream requested one it hands over the canvas as the GPU wrote it
  and declares the pitch, and the per-frame repack disappears. `VideoConvert` is
  that consumer: it requests the layout and reads a packed RGBA / BGRA input's
  rows where they lie (a padded planar input it packs out first, which is correct
  and costs what the producer skipped). It is the `EveryConsumer` request the
  policy above exists for: `VideoConvert` asks with
  `request_from_every_consumer`, so any consumer or hop that would take the
  padded rows for tightly packed ones vetoes the padding and the producer repacks
  as it always did. The meta is dropped by every `meta_transform`, since an
  element only declares one when it writes a new buffer; a tee's clone shares the
  described buffer and keeps it.
- **The overlay.** The visible end of the detector chain reads the
  `AnalyticsMeta` carried onto the *display* frame (via the fan-out path) and
  draws it, so `decode -> tee -> {detect, video} -> overlay -> display`
  works. Three shapes, in one shared palette: a detection box as a solid outline
  in its class colour, an instance segmentation as a translucent fill of its mask
  (`mask-alpha`), and a region of interest as a dashed rectangle. A mask spans
  exactly its instance's box at the model's own grid resolution, which is the
  whole placement rule either backend needs, and an ROI takes the palette slot of
  the segmentation that `Contains` it, so a mask and its tight box read as one
  instance rather than two findings. Two backends: the CPU
  `g2g-plugins::analyticsoverlay::AnalyticsOverlay` (`analytics` feature) paints
  onto RGBA8 with the compositor's integer source-over blend (the
  `no_std` baseline), and the GPU `vellooverlay::VelloAnalyticsOverlay`
  (`vello-overlay` feature) strokes antialiased boxes and scales each mask on as
  an alpha image fill over a full-frame image
  with the Vello GPU 2D renderer, emitting the result in the new
  `MemoryDomain::WgpuTexture` domain. That domain (an `OwnedWgpuTexture` whose
  `wgpu::Texture` lives in a `WgpuKeepAlive` owner, since `g2g-core` never links
  wgpu) is the render-side analog of the decode-side CUDA / D3D11 texture
  domains: the rendered frame stays on the GPU with no readback, so a GPU sink
  presents it directly.
- **The GPU sink.** `g2g-plugins::wgpusink::WgpuSink` (`wgpu-sink`) is
  that consumer: it presents a `WgpuTexture` frame by sampling it in a small
  fullscreen blit pass onto its target (an owned offscreen texture for
  render-to-texture / screenshots, or a caller-built `wgpu::Surface` for an
  on-screen window), again with no readback. Because a wgpu texture is bound to
  its device, the overlay and the sink share one device through a cloneable
  `gpu::GpuContext` (the overlay's `with_context`, the sink's constructors), and
  the producer's texture is recovered by the sink through the shared
  `gpu::WgpuTextureKeepAlive` type. This closes the analytics path end to end:
  `decode -> tee -> {detect, video} -> overlay -> WgpuSink`, detections rendered
  on the GPU reaching the display with no system-memory round-trip. Window and
  event-loop ownership stay with the application (wgpu surfaces are built from a
  window handle and must drive the app's event loop), so the sink presents to a
  surface the app supplies rather than opening its own window. The app also owns
  the resize event, and forwards it as `WgpuSink::resize(width, height)`, which
  reconfigures the swapchain (or reallocates the offscreen texture) at the new
  size; the frame's negotiated geometry is untouched, the blit just scales it to
  whatever the target now is.

- **Bring-your-own-device.** The same `GpuContext` sharing extends one
  step further out, to an embedding application that *already owns* a
  `wgpu::Device` (a game engine, a Bevy / Tauri app, an editor's renderer):
  `GpuContext::from_wgpu(instance, adapter, device, queue)` wraps the embedder's
  device instead of opening one, so every GPU element produces textures *on that
  device*. A decoded frame's `MemoryDomain::WgpuTexture` is then a first-class
  object in the embedder's own render graph, recovered with `gpu::texture_of` and
  bindable directly (sample it onto a 3D surface, composite it in the UI) with no
  second device, no surface hand-off, and no copy, the opposite of `for_surface`
  (where g2g opens the device). This is the integration path for the
  lightweight-app / engine use case where the application drives rendering and
  g2g is just the pipeline that hands it textures: validated on the RTX 3060 (a
  texture produced through a `from_wgpu` context reads back correctly on the
  embedder's own device handles). The frame still flows to the app through any
  sink, including the `appsink` pull channel, which carries a GPU-domain `Frame`
  unchanged. The `bevy-g2g` crate's `VideoPlayerPlugin` (M741 -> M796) is the
  packaged proof: a stock Bevy app's render device is adopted into `from_wgpu`
  in the plugin's `finish`, a
  `filesrc -> h264parse -> ffmpegdec -> videoconvert -> vello overlay -> appsink`
  pipeline lands each decoded frame in a `wgpu::Texture` on Bevy's device, and
  the plugin registers it as a render-world `GpuImage` and binds it to the
  material of every `VideoScreen`-tagged mesh (through an sRGB view; the
  overlay's texture lists `Rgba8UnormSrgb` in `view_formats` for exactly this).
  The mirror of the crate's streaming side (§4.11.3), which renders in Bevy and
  encodes in g2g.

- **Presenting on the producer's device.** A GPU decoder cannot be handed a
  device: Vulkan Video decode needs queues and extensions wgpu never asks for, so
  `VulkanVideoDec` opens its own, and its textures bind to no other. A launch line
  has no application to pass a `GpuContext` between the two, so the decoder
  publishes its own (`gpu::publish_producer_context`, only when the device's
  swapchain extension is enabled) and a windowed sink builds its surface on that
  instance and presents from that device
  (`gpu::present_on_producer_device`, shared by every wgpu display sink),
  falling back to opening its own device when nothing published or the published
  one cannot drive this display. The decode device is opened once per codec and
  kept across the repeated `configure_pipeline` a launch line does, since a second
  device would leave the sink presenting from one nothing produces on. So
  `filesrc ! decodebin ! wgpusink` decodes on the GPU and presents the frame where
  it already lies, with no application code.

---

## 6. Target Deployment Environments
Because the core processing loop requires only `core` and `alloc`, deployment profiles vary purely based on the top-level orchestration binary.

### 6.1 Enterprise Server Node (Cloud Scaling)
- **Runtime Driver:** Tokio multi-threaded runtime.
- **Inter-Element Channels:** Bounded MPMC async channels (`flume`).
- **Hardware Interop:** `cros-codecs` bitstream parsing feeding Linux kernel VAAPI / V4L2 drivers, producing `OwnedDmaBuf` handles.
- **Cargo features:** `multi-thread`, `std`.

### 6.2 Deep Embedded / Bare-Metal RTOS (Industrial & Robotics)
- **Target Hardware:** RTOS targets such as FreeRTOS, Zephyr, or microkernels.
- **Runtime Driver:** Embassy async executor (single-threaded, cooperative multitasking hardware timer loop).
- **Inter-Element Channels:** Zero-allocation stack channels (`embassy-sync`).
- **Hardware Interop:** Fixed-memory DMA rings mapped to microcontroller video capture peripherals.
- **Cargo features:** none (default `no_std + alloc`), or strict no-heap via `StaticBufferPool<_, N>` only.

#### 6.2.1 Embedded / Embassy Element Surface

The `no_std + alloc` core runs here directly: runner futures are
executor-agnostic and `ElementBound` is empty without `multi-thread` (§4.3).
The embedded surface comprises:

- `StaticBufferPool<T, N>` in `g2g-core` (pure `core`, no feature gate) — a
  compile-time-sized zero-allocation pool yielding bounded mutable references
  checked via compile-time lifetimes. This is the strict no-heap pool the
  `Arc<Mutex<Vec<T>>>` `BufferPool` (§3.3) cannot serve.
- `EmbassyClock` (`embassy` feature) over `embassy-time`, the `no_std` analog
  of `WallClock`. The tick rate is selected at the feature; a HAL provides
  the time driver at link.
- `PacketChannel` + `EmbassySink` (`embassy-link` feature) over
  `embassy-sync`, a zero-allocation inter-task packet link — the §6.2 stack
  channel. `SinglePacketChannel` (`NoopRawMutex`) is the single-executor
  default; `SharedPacketChannel` (`CriticalSectionRawMutex`, hence `Sync`) is
  the variant that can live in a `static`, so spawned tasks reach it by
  `&'static` (an executor's tasks take `'static` arguments).
- Two executor models, both over the same runner / element futures:
  `embassy-futures::block_on` drives a whole pipeline as one joined task (the
  bare-metal `fn main` entry, used by the host tests); a real
  `embassy-executor` runs each element as an independently *spawned* task wired
  by static stack channels, the scheduler interleaving them. The latter is
  host-verified via the std platform's `Executor::run_until` (polls then
  returns on a completion flag, instead of the diverging `run()` an embedded
  app's `fn main() -> !` calls); a three-task source -> transform -> sink
  pipeline runs there with no HAL time driver.

`portable-atomic` backs the `metrics::LatencyHistogram` `AtomicU64` so
`thumbv7em` (Cortex-M) and `riscv32` (which lack 64-bit atomics) compile;
`critical-section` makes the lock-based fallback interrupt-safe.

### 6.3 Browser Sandbox (Web Application Scaling)
- **Runtime Driver:** Web Workers spawned via `wasm-bindgen-futures`.
- **Hardware Interop:** Packets ingested via WebSockets / WebRTC data channels, parsed by browser hardware via the native WebCodecs JS API, and injected into WebGPU textures.
- **Cargo features:** `std` (`wasm32-unknown-unknown` provides a usable `std` shim).

#### 6.3.1 Browser / Wasm Element Surface

The browser target is `cfg(target_arch = "wasm32")` elements in `g2g-plugins`
behind the `web` feature (which implies `std`). The wasm bindings
(`wasm-bindgen` / `js-sys` / `web-sys` / `wasm-bindgen-futures`) are
target-gated so native builds never resolve them. No core change is needed:
the runner future is executor-agnostic, so `wasm_bindgen_futures::spawn_local`
drives it on the browser event loop, and wasm builds without `multi-thread`,
so the `!Send` JS handle types satisfy the empty `ElementBound` (§4.3).

The browser element surface comprises:

- `WasmClock` — `performance.now()` + `setTimeout` sleep, the wasm analog
  of `WallClock`.
- `WebSocketSrc` — ingest over a browser `WebSocket`, parallel to `FileSrc`
  / `RtspSrc`.
- `WebRtcSrc` (`web` feature) — ingest over a provided `RtcDataChannel`.
- `WebCodecsDecode` (`web-codecs` feature) — wraps the browser `VideoDecoder`;
  H.264 or H.265 Annex-B access units in, `VideoFrame` copied to `System` RGBA
  out. The codec comes from the negotiated caps and picks the WebCodecs codec
  string built from the in-band SPS (`avc1.` / `hev1.`, ISO/IEC 14496-15 Annex
  E.3); chunks stay Annex-B, which is what a config without a `description`
  means. Build needs `--cfg=web_sys_unstable_apis`.
- `CanvasSink` — presents decoded RGBA to an HTML canvas via the 2D context.
  `WebGpuCanvasSink` (`web-gpu` feature) is the zero-copy variant: it imports
  the decoded `VideoFrame` as a `GPUExternalTexture` and samples it in a render
  pass, with no readback into wasm memory.

A complete in-browser glass-to-glass pipeline is
`WebSocketSrc → H264Parse → WebCodecsDecode → CanvasSink`. The local gate
for the wasm build is
`cargo check --target wasm32-unknown-unknown -p g2g-plugins --features web`.

**Off the main thread (M1054).** A whole graph can run inside a dedicated module
worker: one wasm instance per worker, the same single-threaded executor, no
SharedArrayBuffer and no cross-origin isolation. The page hands the worker an
`OffscreenCanvas` (`canvas.transferControlToOffscreen()`), which the sinks take
through `CanvasSink::from_offscreen_canvas` /
`WebGpuCanvasSink::from_offscreen_canvas` instead of looking an element id up in
`document`. A worker has neither `window` nor `document`, so `WasmClock` and the
WebGPU sink resolve `performance` / `setTimeout` / `navigator.gpu` off
`js_sys::global()`, cast to `Window` or `WorkerGlobalScope`. A transferred canvas
belongs to the worker for good, so a page switches graphs by reloading.

**In-browser ONNX inference (`WebOrtDetect`).** The chain
`WebSocketSrc → WebCodecsDecode → WebOrtDetect → AnalyticsOverlay → CanvasSink`
runs a real `.onnx` model over each decoded frame in the browser with CPU
tensors. `WebOrtDetect` lives in the `g2g-web` wasm leaf crate (not `g2g-plugins`,
which cannot depend on `g2g-ml`), and splits the work so the pipeline stays one
typed graph: g2g owns preprocess (RGBA → `[1,3,640,640]` NCHW f32, whole-to-whole
resize) and postprocess (the SAME `g2g-ml` `DetectionPostprocess` channel-major
YOLOv8 decode + NMS the native chain uses); a small `ort-shim.js`
(wasm-bindgen `module`, bundled into the pkg) owns only `session.run` over
onnxruntime-web. onnxruntime-web runs single-threaded (`numThreads = 1`,
`executionProviders: ['wasm']`), so it needs no SharedArrayBuffer and the demo
serves from plain static HTTP with no COOP/COEP headers; the `.onnx` is fetched
same-origin, the same model format the native ORT path loads. The chain runs on
the single browser thread via `run_linear_chain`. Validated headless
(`tools/wasm-demo/headless/run-ortdetect.mjs`, a WebCodecs-capable Chromium) against
a committed deterministic fixture (`tools/wasm-demo/fixtures/tiny-detect.onnx`,
generated by `gen-tiny-detect.py`) that plants two detections per frame: the model
loads, each frame yields exactly two decoded detections, and the overlay boxes
render to the canvas. Finite and unbounded sources both run clean end to end. (An
unbounded source used to throw `closure invoked recursively or after being
dropped` once a downstream error crossed the backpressured source loop: the
failed `push` returned early past `WebSocketSrc`'s callback detach, so the
still-open socket kept calling a freed `onmessage` closure. Fixed M967, the
detach now runs on every exit; `tools/wasm-demo/headless/repro-unbounded.mjs` is
the repro harness.)

---

## 7. Ecosystem Coexistence Strategy: GStreamer Bridge
To drive early enterprise adoption without forcing full system redesigns, `g2g` provides the `g2g-bridge` wrapper library, compiled as a compliant C dynamic library (`libgstglass2glass.so`). An isolated `g2g` processing sub-graph executes inside a legacy GStreamer pipeline.

```
┌────────────────────────────────────────────────────────┐
│               Legacy C GStreamer Pipeline              │
├────────────────────────────────────────────────────────┤
│  gst-rtsp-src ──► [ gst-glass2glass-bridge ] ──► qtmux │
│                          │                             │
│                          ▼                             │
│             ┌───────────────────────────┐              │
│             │   g2g Async Safe Core     │              │
│             │  (Wgpu Filter / Burn ML)  │              │
│             └───────────────────────────┘              │
└────────────────────────────────────────────────────────┘
```

The bridge intercepts the GStreamer pipeline's internal `GstBuffer`, extracts the underlying OS hardware file descriptor (`GstDmaBufMemory`), wraps it as a `g2g::OwnedDmaBuf` with a no-op close hook (GStreamer retains ownership of the fd), and forwards execution to the Rust async engine.

**Sync/async impedance:** the bridge runs a dedicated Tokio current-thread runtime on its own OS thread, communicating with the synchronous GStreamer `chain` function via bounded channels. This isolates GStreamer's threading model from the async future matrix without blocking either side.

**Implementation (two layers).** The bridge splits into a transport-agnostic core and a GStreamer-facing FFI shell, so the hard, novel part (the sync/async match and lifecycle) is testable on any host without a GStreamer dependency:

1. **`BridgeGraph` (the impedance core, `g2g-bridge`).** Embeds a g2g sub-graph by wrapping a user launch fragment as `appsrc ! <fragment> ! appsink`, parsing it against the standard registry, and running it on a dedicated OS thread with its own current-thread runtime. It exposes a synchronous API: `push(bytes, pts)` feeds the embedded `appsrc`, `try_pull()`/`pull_blocking()` drain the `appsink`, `end_of_stream()`/`finish()`/`Drop` tear down. The `appsrc`/`appsink` elements (§4.x) *are* the boundary the §7 design needs (synchronous external code feeding/draining a running async graph, with bounded-channel backpressure), so the bridge reuses them rather than reinventing the channel plumbing. Per-instance channel names are made collision-free with an atomic counter (the named-feed registries are process-global). On shutdown the drain handle is released before EOS is signalled, so an un-drained graph cannot deadlock the join. Requires the `multi-thread` feature, since the boxed graph must be `Send` to move to the run thread (as in `g2g-capi`).

2. **The GObject `GstBaseTransform` shell (`libgstglass2glass.so`, the `gstreamer` feature).** A thin C shim (`csrc/gstglass2glass.c`, built by `build.rs` via pkg-config + `cc`) registers `glass2glass` as a real GStreamer element and includes the actual GStreamer headers, so the GObject struct layouts are correct by construction rather than hand-transcribed. It delegates to the C-ABI functions in `src/ffi.rs`, which drive one `BridgeGraph` per instance: `set_caps` builds it from the `fragment` property and the serialized sink/src caps (normalized: the `(type)` annotations and whitespace GStreamer emits are stripped so g2g's caps reader and launch DSL accept them), `stop` destroys it. The element handles both **caps-preserving** and **caps/size-changing** fragments. A preserving fragment (a wgpu effect, `videoflip`, an ML preprocessor keeping the pixel format) runs in place via `transform_ip` (the fast path). A fragment that rescales or reformats declares its result through an `output-caps` property; the shell then advertises it via `transform_caps`, sizes the output buffer via `get_unit_size` (`gst_video_info_from_caps`), and runs the out-of-place `transform` (`inbuf`→`outbuf`). GstBaseTransform dispatches between the two by whether the negotiated caps differ. `BridgeGraph` pins the sub-graph's trailing inline caps filter to the output caps (equal to the input when preserving), which both enforces the contract and gives a caps-driven transform a fixate target.

The **zero-copy DMABUF import** path exists at the ingest side: `appsrc` accepts a `MemoryDomain::DmaBuf` frame (`AppSrcFeed::push_dmabuf`), `BridgeGraph::push_dmabuf` feeds it, and the C-ABI `g2g_bridge_push_dmabuf` `dup`s a GStreamer buffer's dma-buf fd (GStreamer keeps the original; g2g's `OwnedDmaBuf` closes only the dup) so no pixel bytes are copied at the boundary. The dma-buf-**consuming** element exists: `dmabuftowgpu` (`g2g-plugins`, the `dmabuf-wgpu` feature) imports a `MemoryDomain::DmaBuf` frame into a GPU-resident `wgpu::Buffer` via `VK_EXT_external_memory_dma_buf` (Vulkan `from_raw_managed` -> `create_buffer_from_hal`), so a bridge fragment like `dmabuftowgpu ! <wgpu compute>` runs the imported buffer on the GPU with no CPU copy. Validated on an RTX 3060 by exporting GPU memory as a dma-buf fd and re-importing it (a discrete GPU binds a GPU-visible dma-buf; a CPU/vmalloc-backed one, e.g. a USB webcam or udmabuf, it cannot, and the element returns `UnsupportedDomain` rather than a wrong result).

**The shell's dma-buf round-trip is wired on both sides.** The data path is a single `generate_output` override (not `transform`/`transform_ip`, so the output buffer may differ from the input in size *and* memory kind): on input it checks `gst_is_dmabuf_memory` and imports the fd via `g2g_bridge_push_dmabuf` (else maps and copies bytes); on output the pull returns either system bytes or a dma-buf (the FFI `G2gOut` carries a `kind` discriminant), and the shell wraps a dma-buf frame back into a `GstBuffer` via `gst_dmabuf_allocator_alloc` (the fd dup'ed, so the g2g frame keeps its own). A full `dma-buf in -> glass2glass(identity) -> dma-buf out` round-trip is validated with a memfd-backed dma-buf (`tools/gst-bridge-dmabuf-smoke.sh`), and the system-memory path is unchanged (`tools/gst-bridge-smoke.sh`). The one remaining piece for a *GPU-compute* round-trip (`dmabuftowgpu ! <compute>`) is bringing that leg's `WgpuBuffer` output back to the shell: a `WgpuBuffer -> System` download or a `WgpuBuffer -> DmaBuf` export element at the fragment's tail (the shell already hands both system and dma-buf frames back). That download/export element is the remaining GPU-track work.

   The plugin entry points are subtle: rustc exports only its own `#[no_mangle]` symbols from a cdylib and localizes anything pulled from a statically-linked C archive, so a C `GST_PLUGIN_DEFINE` descriptor is invisible to GStreamer's loader. The `GstPluginDesc` and the `gst_plugin_<name>_get_desc`/`_register` entry points the loader resolves (by the `libgst<name>.so` filename) are therefore authored in Rust (`src/ffi.rs`), pointing at the C `plugin_init` that does the actual element registration. This is the same split `gst-plugins-rs` uses. Because the feature links the system GStreamer, the shell is built and smoke-tested locally (`tools/gst-bridge-smoke.sh`), not in CI.

3. **The reverse direction (`gstwrap`, `g2g-plugins`, the `gstreamer` feature).** The two layers above put a g2g stage inside a GStreamer app; `gstwrap` does the opposite, hosting an unported GStreamer element *inside* a g2g graph. This is the incremental-migration path in the g2g-as-top-framework direction: adopt g2g now and keep the stages you have not ported yet running as real GStreamer elements. It is a normal g2g `AsyncElement` whose `element` property is a GStreamer element description (`x264enc bitrate=4000`, `videoflip method=horizontal-flip`); internally it drives `appsrc ! <element> ! appsink` in a real GStreamer pipeline on GStreamer's own streaming threads. `process` copies each `System` input frame into a `GstBuffer` (`gst_app_src_push_buffer`), drains ready output non-blockingly (`gst_app_sink_try_pull_sample`, 0 timeout), and on EOS flushes the element's buffered frames. The C interop mirrors the shell's: a small helper (`csrc/gstwrap_host.c`, built by the crate's `build.rs` via pkg-config + `cc`) over the gstreamer-1.0 / gstreamer-app-1.0 C API, driven from `src/gstwrap.rs` over a C ABI. Caps translate with the existing `Caps::to_gst_string()` (g2g caps → the appsrc's caps) and `parse_caps()` (an `output-caps` property → the caps a reformatting element like an encoder or `videoscale` produces); a caps-preserving element declares nothing and couples input == output. The pipeline handle is `Send` because the appsrc/appsink APIs are MT-safe (the element drives them from one runner task at a time). v1 is system-memory (copy in, copy out, like the shell's non-dma-buf path); dma-buf zero-copy through `gstwrap` is future work. Validated locally (`cargo test -p g2g-plugins --features gstreamer --test gstwrap`, not CI) by hosting a real `videoflip` and asserting the pixels come back flipped, and by running `videotestsrc ! gstwrap element="videoflip method=horizontal-flip" ! fakesink` through `parse_launch`. A multi-word element description reaches `gstwrap` from a `gst-launch` line because the launch tokenizer is quote-aware (it treats a `"..."` region as one token, so spaces and `!` inside a value are literal); see §4.16.
