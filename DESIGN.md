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
| `PipelinePacket` | what actually crosses a link: `CapsChanged` / `DataFrame` / `Segment` / `Flush` / `Eos` (§3.1). |
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
| `g2g-core` | Core traits, `Frame` definitions, buffer pool allocators, clock model. | `no_std + alloc` | LGPL v2.1+ |
| `g2g-plugin` | SDK for dynamically loadable plugins (the `declare_plugin!` macro + ABI tag, §4.16). | `no_std + alloc` | LGPL v2.1+ |
| `g2g-plugins` | Standard collection of source/sink/transform elements (`rtsp`, `wgpu`, `v4l2`). | `no_std + alloc` / `std` mixed | LGPL v2.1+ |
| `g2g-ml` | ML inference elements built on `burn` (Wasm/embedded) and `ort` (server), plus the multi-stream tensor batcher. | `std` | LGPL v2.1+ |
| `g2g-bridge` | C-FFI dynamic library to embed `g2g` sub-graphs inside GStreamer pipelines. | `std` (`cdylib`) | LGPL v2.1+ |
| `g2g-python` | Hosts gst-python-ml elements as first-class `g2g` elements (embedded CPython via pyo3). | `std` | LGPL v2.1+ |
| `g2g-capi` | C ABI (cdylib/staticlib + `g2g.h`) to drive pipelines from any language: `parse_launch` + run + bus + appsrc/appsink. | `std` (`cdylib`) | LGPL v2.1+ |
| `g2g-pyapi` | Python (pyo3) bindings to drive pipelines: `parse_launch` + run + bus + appsrc/appsink (the inverse of `g2g-python`). | `std` | LGPL v2.1+ |

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
- **Strict `no_std` (no heap) environments:** two pure-`core` pools sized at construction, no `alloc`. `StaticBufferPool::<[u8; N], 8>` is the *move-out* pool: `acquire` takes an owned buffer out and the RAII handle returns it on drop, the no-heap analog of `BufferPool`. `StaticLendRing::<N, BYTES>` is the *zero-copy lend* sibling for the capture path (a DMA ring): `N` inline slots, the producer fills the next free slot and `publish`es it as a `SystemSlice` that *borrows* the slot, and a per-slot lease (an `AtomicBool`, plain store, no CAS so it builds on `thumbv6m`) is cleared when the lent frame drops, so the slot is reused only after the consumer is done, the genuine ring back-pressure (the producer stalls when every slot is in flight). The borrow is runtime-guarded, not a Rust lifetime: a `PipelinePacket` crosses the `OutputSink` / stack channel by value (`'static`), so the lend reuses the `'static` foreign-buffer carrier (`SystemSlice::from_foreign`) with the lease standing in for the borrow. This keeps `Frame` / `MemoryDomain` lifetime-free (every element signature stays clean) while still proving a heap-free capture-to-consumer path end to end (validated under `block_on` over the embassy stack channel; a real capture wires a DMA-completion ISR / HAL into the same ring). The heap-free claim is *measured*, not asserted: a counting `#[global_allocator]` test (`m616_no_steady_state_alloc`) runs the `StaticLendRing` capture -> frame -> drop hot path for 100k frames and confirms zero heap allocations across the loop. The one place the async plumbing still allocates is pinned honestly by a sibling test: the object-safe `OutputSink::push` returns `Pin<Box<dyn Future>>`, so a `dyn` sink boxes one future per frame; the zero-alloc contract therefore covers the data path and a concrete (non-`dyn`) link, and that control-plane cost is measured rather than hidden.

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

/// Output sink for both transform and source elements. `push` is async
/// so elements await downstream capacity rather than failing fast on a
/// full bounded link. Dyn-safe via a boxed future.
pub trait OutputSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;
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

Sink elements compare `pts_ns` against `now_ns()` to schedule presentation, and `capture_ns` against `now_ns()` to report true glass-to-glass latency without ambiguity about which clock domain a timestamp lives in. Backends provide concrete implementations: a `WallClock` (`std::time::Instant` + `tokio::time::sleep`) for std targets, `embassy-time` for RTOS, performance.now() for Wasm.

A free-running source feeding a sync sink is paced automatically by upstream backpressure (§4.5): the sink only consumes after `sleep_until_ns(pts)` resolves, which throttles the channel, which throttles the source. No explicit source-side pacing is required for sync playback.

#### Clock distribution to sinks

A pipeline runs against one elected clock (`elect_clock` over `ClockPriority`: a PTP grandmaster-disciplined clock (`PtpGrandmaster`) outranks a live source's hardware clock (`LiveSource`), which outranks an audio sink's DAC clock (`AudioProvider`), which outranks a plain monotonic provider such as a video display sink (`Provider`), which outranks the system fallback). The runner samples the elected clock's `now_ns()` once at startup as the **base time** (the clock reading at running-time zero) and hands both to each sink via `set_clock_sync(ClockSync { clock, base_time_ns })`, called once after election. Both the linear runners and the DAG runner `run_graph` deliver it (the latter walks its sink nodes after election), so a display sink PTS-paces in any topology. A sink that synchronises presents a frame when the elected clock reaches `base_time_ns + running_time`, where running time is the frame's `pts_ns` mapped through the active `Segment`; a sink that ignores the hook presents as fast as backpressure allows.

**Audio as the sync master.** For playback the audio sink should drive timing, because samples leave the DAC at the hardware's real rate, which drifts from wall time by tens to hundreds of ppm. `DriftClock` (`g2g-core`) turns that into a usable pipeline clock: it is fed `(local_ns, master_ns)` observations (`local_ns` from a monotonic reference, `master_ns` the true playout position) and fits `master ≈ slope·local + offset` by least squares over a sliding window, so `now_ns()` projects the current reference time through the fit, both estimating the playout rate and smoothing the coarse, jittery per-observation readings. `AlsaSink`'s worker samples `frames_written − snd_pcm_delay()` after each blocking `writei` and feeds the clock, offering it to election at the `AudioProvider` tier (gated by a `provide-clock` property). A video sink then slaves to it: because the elected clock is the disciplined audio timeline rather than raw wall time, video presentation follows audio, giving true A/V sync. A `LiveSource` capture clock still wins when present, so a live pipeline paces to capture.

**Networked sync (PTP).** For facility-wide sync (Pro AV / SMPTE ST 2110), the shared reference is a PTP grandmaster, and every device slaves to it, so a `PtpGrandmaster` clock outranks all of the above. `PtpServo` (`g2g-core::ptp`) is the servo: fed the four timestamps of each PTP delay request-response, it computes the standard `offset` / `mean_path_delay` and folds `(local, master)` into the same `DriftClock` machinery, disciplining the local monotonic reference to the grandmaster's TAI timeline with lock / holdover / outlier-rejection state. `PtpClock` wraps it (interior-mutable, so one worker drives it while sinks read `now_ns` through a shared `Arc`) and offers itself to election only once locked. Because the elected timeline is grandmaster-derived, two machines locked to the same grandmaster read the same clock, so the A/V pacing above holds *across* devices, not just within one process. Two sources feed the servo: raw PTP message timestamps (`sync_exchange`), or a direct absolute-time observation (`observe_master`). Two backends supply them: `PtpSystemClock` (`g2g-plugins`, Linux) delegates to an OS PTP-disciplined `CLOCK_TAI` (from `linuxptp` / `phc2sys`), sampled on a worker; `PtpClient` (`g2g-plugins`) is a from-scratch software PTP SLAVE that speaks PTP over UDP itself (the `ptp::wire` message parser + the `ptp::slave` delay-request-response state machine + a UDP transport), so an endpoint with no OS PTP daemon can still lock. The wire parser and slave state machine are `no_std` and CI-tested end to end (parse -> slave -> servo) without sockets.

**ST 2110 media transport** rides on this shared clock. Distinct time newtypes guard the seam where three "just an integer" times meet: `TaiNs` (PTP/TAI nanoseconds, absolute), `RtpTs` (the 32-bit wrapping RTP media-clock timestamp on the wire), and `RefNs` (the pipeline's monotonic reference nanoseconds, a relative timeline with an arbitrary epoch). `MediaClock` takes a `TaiNs` and returns an `RtpTs`, so the compiler rejects handing it the wrong clock (the confusion the PTP servo work hit: a monotonic reference minus a TAI master is meaningless); the PTP servo's own seam is typed the same way, `PtpServo` / `PtpClock` `sync_exchange` take `(TaiNs, RefNs, RefNs, TaiNs)` and `observe_master` takes `(RefNs, TaiNs)`, so master and reference can no longer be swapped. Durations stay a plain `u64`. `MediaClock` (`g2g-core`, ST 2110-10) maps a PTP/TAI time to a 32-bit wrapping RTP timestamp and back (a media clock counting at 90 kHz for video / the sample rate for audio from the PTP epoch), so two receivers on the same grandmaster compute the same timestamp for the same sampling instant. `st2110audio` (`g2g-plugins`, ST 2110-30) is the sans-IO PCM payloader/depayloader (L16 / L24 big-endian in the RTP payload, timestamps off the media clock), and `st2110anc` (ST 2110-40 / RFC 8331) carries SMPTE ST 291 ancillary data (closed captions, timecode) as bit-packed 10-bit words with parity + checksum validation, so the caption stack can ride 2110. The sans-IO cores get network element wrappers: `st2110audiortp` (`St2110AudioSink` / `St2110AudioSrc`, behind the `st2110` feature) puts -30 audio on the wire over UDP, the sink mapping each frame's PTS through the elected (PTP) clock to the media-clock timestamp and the source reconstructing PTS from it, so a receiver on the same grandmaster stays in sync. `st2110ancrtp` does the same for -40 captions: `St2110AncSink` taps a compressed H.264 / H.265 stream (a teed branch leaf like `CcExtract`), mines each access unit's caption triples, wraps them in a Caption Distribution Packet (CDP, CEA-708 / SMPTE ST 334-2) carried in a DID 0x61 ANC packet, and sends the RFC 8331 RTP timestamped at the frame's PTP time; `St2110AncSrc` depacketizes -40 back into triples and, through the shared `CaptionDecoder` (the decode core factored out of `CcExtract`, driving the same CEA-608/708 state machines from triples mined from SEI or carried in a CDP), emits timed `Caps::Text{Utf8}` cues. So captions travel end to end over 2110 and stay frame-aligned on a common grandmaster. `st2110video` (ST 2110-20 / RFC 4175) carries uncompressed active video: the packetizer slices a packed frame into Sample Row Data (SRD) line runs (an Extended Sequence Number then per-run headers giving scan line, pixel offset, octet length) sized to the MTU, and the depacketizer writes each run back into the frame, completing it on the RTP marker bit; `st2110videortp` (`St2110VideoSink` / `St2110VideoSrc`) puts it on UDP with the 90 kHz media-clock timestamp shared by every packet of a frame. Each sampling is a `Layout` reading / writing one pgroup at a time, so the packetizer / depacketizer stay layout-agnostic across three mappings: RGBA 8-bit (packed, byte-identical), YCbCr-4:2:2 8-bit (packed `Yuyv`, luma / chroma bytes swapped to the wire), and YCbCr-4:2:2 10-bit (the broadcast norm, from the planar `I422p10` buffer: the four 10-bit samples Cb0 Y0 Cr0 Y1 are MSB-first bit-packed into a 5-octet pgroup, crossing both a planar-to-packed and a byte-to-bit boundary). The source's geometry comes from properties, or from the stream's SDP: `st2110sdp` (RFC 4566 + SMPTE ST 2110-10/-20/-30/-40) is the sans-IO generator / parser for the out-of-band description a receiver configures from, carrying the essence (video sampling / size / rate, audio depth / rate / channels / ptime, or ancillary), the payload type, the multicast group and port, and the `a=ts-refclk` PTP grandmaster all the streams share. every sink has an `sdp()` that publishes its stream and every source an `apply_sdp()` that auto-configures from a parsed one, so a stream self-describes end to end across video, audio, and ancillary. On the audio side `PcmS16Le` rides as L16 and `PcmF32Le` as L24 (float scaled to the 24-bit wire). `st2110jxs` (ST 2110-22 / RFC 9134) carries the compressed mezzanine essence, JPEG XS: the packetizer slices an opaque codestream into codestream-mode packets (the 4-octet RFC 9134 payload header carrying transmode / packetmode / last-packet / frame counter / packet counter), the marker bit ending the frame, every packet on the same 90 kHz media clock; `st2110jxsrtp` (`St2110JxsSink` / `St2110JxsSrc`) puts it on UDP, taking / emitting `Caps::CompressedVideo{JpegXs}` frames. The JPEG XS codec itself is `SvtJpegXsEnc` / `SvtJpegXsDec` (`jpegxs` feature): hand-rolled FFI to Intel SVT-JPEG-XS (ISO/IEC 21122, no libavcodec), planar 4:2:0 / 4:2:2 8-bit and 4:2:2 10-bit, the encoder targeting a bits-per-pixel budget and the decoder discovering geometry from the first codestream. So a plant can move visually lossless video at a fraction of -20's bandwidth with sub-frame latency, end to end (raw -> encode -> -22 -> decode). SDP covers all essences, including -22 (`jxsv`), and `St2110Session` bundles video + audio + ancillary into one multi-section session document (each media tagged with `a=mid`, a shared `a=ts-refclk`), so a whole program self-describes. `AudioFormat::PcmS24Le` (integer 24-bit) rides the -30 L24 wire directly, alongside the float path. `st2110dup` implements ST 2110-7 seamless protection: a receive-side sequence-number merge of two identical redundant streams (first arrival wins, so a loss on one path is filled by the other), with `a=group:DUP` in the session SDP. Its `SeamlessDedup` is the sans-IO core; `RedundantRtpReceiver` (behind the `st2110` feature) is the socket-bound sibling that binds several receive paths, polls them round-robin so two in-order streams merge back into sequence order, and yields deduplicated packets. `St2110VideoSrc` adopts it behind a `redundant` property (a second "blue" path); being essence-agnostic it can serve the other essences the same way. `st2110pacing` implements ST 2110-21 sender pacing: a schedule spreading a frame's packets across the frame period (linear or gapped), which both `St2110VideoSink` and the -22 `St2110JxsSink` realize over the tokio timer (through a shared `pace_send`) so the network sees a smooth flow instead of a burst. `VrxValidator` is the full per-format -21 compliance check: the leaky-bucket virtual-receive-buffer model (a receiver draining one packet every `TRS` after a `TR_OFFSET` head start) that, over a run of actual emission offsets, reports the peak buffer occupancy, whether a packet arrived late (starving the receiver), and whether it stays within the profile's `Cmax`. What is built (from the RFCs, loopback-tested, not yet interop-validated against reference gear) now spans -10/-20/-21/-22/-30/-40/-7 plus SDP; multicast interop remains.

`WaylandSink` is the first display sink to use it: it holds each frame until its running-time deadline, tracking the `Segment` (clipping pre-target frames after an accurate seek) and re-anchoring on `Flush`. It also does **QoS late-drop** (matching `SyncSink`): a frame already past its deadline by more than a configurable `max_lateness` bound is dropped instead of presented late, so the sink catches up instead of accumulating lag, posting a `BusMessage::Qos` (running time, jitter, cumulative processed/dropped) per drop.

**Playing-transition anchoring.** The startup base time is sampled before the data plane and before the application presses play. For a non-live, prerolled pipeline that sits in `Paused` for a while, that is the wrong epoch: the preroll frame is consumed during `Paused`, so a sink that anchored on the startup base (or on that first frame) then rushes/drops once `Playing` finally arrives. So when a `StateController` drives the run, the runner arms a `PlayAnchor` (a shared cell) on the elected clock and hands each sink `ClockSync::with_play_anchor`; `set_state(Playing)` stamps the anchor with `clock.now_ns()` at the exact play edge (and a transition down to `Ready`/`Null` clears it, so a replay re-bases). `ClockSync::base_time()` then resolves to the play-edge stamp once armed, else the eager startup base time. `WaylandSink` reads it per frame: it first-frame-anchors a preroll frame consumed during `Paused` (presented immediately), then re-bases onto the play edge once `Playing` stamps it; a seek `Flush` forces a first-frame re-anchor so the seek target presents immediately rather than against the stale play base. The non-stateful runners keep the eager base time (no `StateController`, no play edge to anchor to).

**Upstream QoS** carries that lateness back to the producer so it sheds load too, not just the sink. It rides the same per-link reverse channel as `Reconfigure`: a sink returns a `QosMessage` from `AsyncElement::take_qos`, the runner stores it into the incoming link's reverse `QosSlot`, and the producer observes it as `PushOutcome::Qos` on its next push (reconfigure wins when both are pending; QoS is advisory and never holds the packet back). `SyncSink` originates it on a late-drop and `VideoTestSrc` reacts by skipping ~`jitter / frame_period` frames (advancing PTS without generating them). **Relay through a transform** carries the report the rest of the way to the source in a multi-element pipeline. A transform observes a downstream QoS as a `PushOutcome::Qos` inside `process`, but that outcome is discarded by a generic transform, and the runner (not the element) owns the reverse slots, so the relay is runner-mediated: the runner wires the transform's *output* `SenderSink` with a relay handle to its *input* link's `QosSlot` (`relay_qos_to`). When the output adapter then sees a downstream QoS it stores it onto the input link instead of surfacing it, so the upstream neighbour observes it on its next push, and across N transforms the report walks one hop at a time back to the source. The element's `process` is unaffected; a QoS-aware transform that wants to act on the report itself is a later refinement. This is the same shape as the reverse `Reconfigure` path. Wired in the bespoke `run_source_transform_sink` runner and in the DAG runner (`run_graph` / `run_linear_chain`, which the `WaylandSink` demo uses), so the sink's own load-shed reaches the source through interior transforms (overlay, convert).

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

`VaapiH264Dec` (`g2g-plugins/src/vaapidec.rs`, feature `vaapi`, `cfg(target_os = "linux")`) is built on `cros-codecs` (`vaapi` backend). The crate is maintained by the ChromeOS team and exposes a stateless decoder framework that parses H.264 bitstreams and manages the DPB; the actual decode runs on the GPU through libva.

- **Input caps:** `Caps::CompressedVideo { codec: H264, .. }` — `intercept_caps` intersects with H.264 and rejects everything else.
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
- **Output caps:** `Caps::RawVideo { format: I420 | Nv12, .. }`. `I420` is the libavcodec native 8-bit 4:2:0 format; `Nv12` is selectable via `FfmpegH264Dec::with_output_format(OutputFormat::Nv12)`, produced by a U/V interleave with no swscale. `YUVJ420P` is accepted with the same plane layout; `YUV444P` / `YUVJ444P` are accepted with the chroma planes box-averaged down to 4:2:0. Other pixel formats are rejected with `CapsMismatch`.
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
- **Validation:** an on-hardware round-trip on the RTX 3060 synthesizes a CUDA-resident NV12 surface (CUDA driver alloc + upload), encodes through `NvEnc`, and decodes the Annex-B back through `FfmpegVideoDec` to the original geometry; it skips cleanly with no NVIDIA GPU. The `nvenc` feature is CI-excluded (no NVENC runtime in CI). **HEVC (H.265)** is supported alongside H.264: `with_codec(VideoCodec::H265)` / the `codec` property switches the encode GUID to `NV_ENC_CODEC_HEVC_GUID` and the output caps to `CompressedVideo{H265}`, the path otherwise identical (the round-trip test covers both). `NvEnc` declares `input_domains = {Cuda}`, so a CPU-side NV12 source feeding it gets a `CudaUpload` spliced in automatically by the converter auto-plug (§4.13.5); the encoder itself stays Cuda-only. The output-bitstream-buffer pool and runtime bitrate retarget are in place. The matching native `NvDec` is the other half of the gst-`nvcodec`-style pair.
- **Thread safety:** the session is a raw NVENC handle + CUDA context driven through `&mut self` only; `unsafe impl Send` rests on the same ownership-transfer argument as `FfmpegH264Enc`.

`NvDec` (`g2g-plugins/src/nvdec.rs`, feature `nvdec` which implies `cuda`, `cfg(target_os = "linux")`) is the **decode half of the gst-`nvcodec`-style pair**, the mirror of `NvEnc`. It promotes NVIDIA hardware decode from the `FfmpegH264Dec` `Backend::NvdecCuda` flag (which reaches NVDEC *through* libavcodec's cuvid hwaccel) to a first-class element driving the NVCUVID parser+decoder API directly. With `NvDec -> ... -> NvEnc` both native, the whole H.264 transcode loop stays on the GPU and out of libavcodec.

- **Caps:** `Caps::CompressedVideo { codec: H264, .. }` Annex-B in, `Caps::RawVideo { format: Nv12, .. }` out (a native `DerivedOutput`). The runtime `CapsChanged` carries the actual cropped display geometry the bitstream declares.
- **Multi-domain output.** `NvDec` advertises `output_domains = {Cuda, System}` and, in `configure_allocation`, reconciles the negotiated proposal against that capability (`resolve_for_producer`, §4.13.5): a CUDA-capable consumer keeps each surface device-resident (zero-copy, the default `MemoryDomain::Cuda`); a System-only consumer makes the decoder download (reusing `cuda::download_nv12`) before emitting. The same decoder stays on the GPU or downloads, chosen by downstream demand alone, validated on the RTX 3060.
- **Callback model:** NVCUVID is callback-driven. A parser (`cuvidCreateVideoParser`) is fed the elementary stream and synchronously invokes three callbacks from inside `cuvidParseVideoData`: a *sequence* callback (creates the `CUvideodecoder` once the SPS geometry is known), a *decode* callback (`cuvidDecodePicture`), and a *display* callback (a frame is ready in display order). The display callback cannot `await`, so it maps the surface (`cuvidMapVideoFrame64`) and pushes a ready frame onto a queue that `process` drains and emits after the parse returns. The callbacks reach element state through a `*mut DecoderState` passed as the parser user-data; that pointer targets a heap `Box` so it survives the runner moving the element between worker threads.
- **Bindings: hand-rolled FFI.** Links `libnvcuvid` + `libcuda` directly (no `cudarc`). NVCUVID exports real symbols (no `CreateInstance` dispatch table, unlike NVENC), so the calls are plain `extern "C"`; the structs are transcribed `#[repr(C)]` with compile-time size assertions against the installed `cuviddec.h` / `nvcuvid.h`, and the per-picture `CUVIDPICPARAMS` is opaque (the parser fills it, we pass the pointer straight to `cuvidDecodePicture`).
- **Frame lifetime:** each output frame carries a `CudaKeepAlive` that `cuvidUnmapVideoFrame64`s on drop plus an `Arc` to the decoder, so the decoder and its CUDA context outlive any frame still in flight; the decoder, context lock, and context are destroyed (in that order) only once the last frame is released. The element owns its own CUDA context (created at configure).
- **Validation:** an on-hardware test on the RTX 3060 runs the full native loop, a synthesized CUDA NV12 surface encoded by `NvEnc` to Annex-B and decoded by `NvDec` back to CUDA NV12, asserting geometry and (via a small device->host copy) that the decoded luma holds real content; it skips with no NVIDIA GPU. The `nvdec` feature is CI-excluded. **HEVC (H.265)** is supported alongside H.264: the input caps accept `CompressedVideo{H264|H265}`, the codec is inferred and mapped to the `cudaVideoCodec` the NVCUVID parser + decoder are created for. The display delay is fixed at a low-latency 1.

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
| Linux + VAAPI | `VaapiH264Dec` | `vaapi` | `System` / NV12 |
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
YCbCr pass with an in-flight ring. Output caps are
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
`re_renderer`-style consumer builds on.

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
consumer such as the re_video adapter reorders by PTS itself), but the g2g-native
`VulkanVideoDec` element does reorder its system (NV12) path: `decode_push_meta`
returns one `PictureMeta` per submitted picture (the POC the decode already
computed, no second pass), and the element feeds retired frames through a small
`ReorderBuffer` keyed by the same (coded-video-sequence, POC). It releases the whole
previous coded video sequence at each keyframe (where POC resets) and bumps the
lowest-POC held frame once a sequence exceeds the DPB depth (a safe reorder-depth
bound, so a long GOP does not buffer unbounded); `Eos` and a resolution-reconfig
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
unchanged. Long-term references are the remaining H.265 gap.

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
source: it streams packed **YUYV** (4:2:2, the near-universal UVC output) off a
`/dev/videoN` device via V4L2 mmap streaming I/O, wrapping the pure-Rust `v4l`
crate (no libv4l C dependency). `VideoConvert` unpacks YUYV to a planar / RGB
target (§3.1 raw formats), so the canonical chain is
`V4l2Src -> VideoConvert(Yuyv -> Nv12) -> sink`.

Two design points carry the element:

- **Blocking ioctls off the async path.** V4L2 dequeue is a blocking ioctl, so
  capture runs on a dedicated `std::thread` that owns the device and the mmap
  stream (which borrows the device) and copies each frame's payload into a
  bounded channel. The `SourceLoop::run` future drains that channel and pushes
  `DataFrame`s. The channel bound (`BUFFER_COUNT`) applies backpressure: the
  capture thread blocks rather than growing memory when the pipeline falls
  behind. The source reports a live `LatencyReport` of one frame period.
- **Up-front format negotiation, re-open for capture.** `intercept_caps` opens
  the device, sets YUYV at the requested geometry, and reads back what the
  driver actually chose (it may snap to a supported mode); the probe device is
  then dropped. The capture thread re-opens the device under that exact format.
  Keeping no device handle in the struct between negotiation and `run`
  sidesteps `Send` / borrow entanglement with the stream. Errors surface as
  `G2gError::Hardware(HardwareError::V4l2(errno))`.

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
caps are deterministic. `MfVideoSrc`
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
element acts on. It also speaks the publisher path (ANNOUNCE / RECORD) for a
future receive-side source. Validated end-to-end over loopback (an in-test player
handshakes and recovers every streamed access unit). Scope is one client / one
session / unicast UDP / the PLAY direction.

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
accumulates an Opus PTS from the TOC byte). **Adaptive segment seek.** The
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
- `Tag(TagList)` — container / stream metadata, posted out of band
  (`GST_MESSAGE_TAG`).
- `NegotiationFailed(NegotiationFailure)` — structured caps conflict naming the
  responsible element pair (§4.13), posted by the coordinator on a startup or
  mid-stream negotiation failure.
- `StateChanged { old, new }` and `AsyncDone` — every effective lifecycle
  transition, and the completion of an async `PAUSED` once preroll aggregates
  (§4.14).
- `Qos { running_time_ns, jitter_ns, processed, dropped }` — a synchronizing
  sink (`SyncSink`) that has fallen behind the clock drops a late frame
  (`with_max_lateness_ns`) and reports it, the `GST_MESSAGE_QOS` analog.
- `Buffering { percent }` — a sink's input link fill (0 = underrun, 100 = full),
  posted by the sink arm on a quartile crossing via `run_graph_with_bus`. Since
  g2g has no `queue` element, this reports the bounded link channel's own
  occupancy (`fill_percent`), the `GST_MESSAGE_BUFFERING` analog.

Posting is non-blocking (`try_post`): a control message never stalls the data
path; a full bus drops the report rather than applying backpressure.

**Element-granular logging (`g2g-core::log`)** is the complementary
diagnostic channel, the `GST_DEBUG` analog, for developer tracing rather than
application-facing events. A record carries a `category` (the element *type*,
e.g. `"VideoFlip"`, the filtering key) and an optional `instance` name (the
element *instance*, e.g. `"VideoFlip0"`). `LogLevel` runs `Error` (most severe)
through `Trace`, matching GStreamer's numeric levels; a per-category threshold
table (a default plus overrides) decides what is emitted, mirrored into an atomic
so a disabled `g2g_trace!` in a hot loop costs one atomic load. The macros
(`g2g_error!` .. `g2g_trace!`) take a `LogSource` (an element via `self`, or a
`Target` for logging about a named element) then a `format_args!` message,
checked against the threshold before formatting. Records route to an installed
`LogSink`; the `std` feature provides a stderr sink and `init_from_env`, which
reads `G2G_DEBUG` (a `GST_DEBUG`-style `*:warning,VideoFlip:trace` spec). The DAG
runner assigns each element a `<category>N` instance name before negotiation (the
`videotestsrc0` convention) via `set_instance_name`, logs each element's
addition, and an element that logs about itself (it implements `LogSource` with a
stored name) carries that name in its lines. Pulls no external logging crate, so
it holds on the `no_std` baseline; the sink is the RTOS plug-in point (UART /
RTT). The `tracing` feature adds a `LogSink` that forwards records to the
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
  role, properties, and pad templates, the `gst-inspect` analog. The dump is
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
would be a use-after-free with no back-pointer to catch it. This version+toolchain
lock is the v1 design; an `abi_stable`/`stabby` facade over the element traits is
the later upgrade for cross-toolchain binary plugins, and a pure C-ABI shim was
rejected (it loses the ergonomic Rust trait). The whole path is exercised
out-of-tree by `g2g-plugins/tests/fixtures/example-plugin` +
`tests/plugin_loader_dlopen.rs`.

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
carrying N elementary streams, each on its own PID and named in one PMT. The
single-input `tsmux::TsMux` element wraps a one-stream muxer (`! mpegtsmux !`);
the multi-input `tsmuxn::TsMux` (a `MultiInputElement`) muxes A+V, interleaving
access units across inputs by PTS via the `take_earliest_by` merge so the
multiplex is decode-ordered. The `mpegtsmux` name is registered both as the
single-input launch element and as a fan-in muxer, so the text parser
picks `tsmux::TsMux` for one input and `tsmuxn::TsMux` for several by link degree
(`v.! m.  a.! m.  mpegtsmux name=m`), mirroring gst's request sink pads.

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
(§4.18). Unlike `TsDemux`,
Matroska's Tracks element carries concrete geometry and audio parameters, so the
demuxer refines the output caps itself via `CapsChanged` once Tracks is parsed,
without a downstream bitstream parser. WebM (the VP8/VP9/AV1 + Opus subset) is the browser-delivery motivator. Block
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
Clusters (multi-track A/V muxing is the sibling `mkvmuxn`).

The Ogg demuxer is the third, the same parser + element split on
`Caps::ByteStream{Ogg}`. `g2g-plugins::ogg::OggDemuxer` parses RFC 3533 pages
(sync to "OggS", frame packets via the segment-table lacing with cross-page
reassembly, sniff the codec from the first packet's `OpusHead`, skip the setup
headers), and `OggDemux` emits the Opus audio packets as `Caps::Audio{Opus}` with
the channel count refined from `OpusHead`. The container is auto-detectable
(`typefind` "OggS", `filesrc bytestream-format=auto`).

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

Adaptive streaming sits one layer above these demuxers: an HTTP byte source feeds
a playlist/manifest-driven source that fetches media segments and hands them to
the matching byte-stream demuxer. `g2g-plugins::httpsrc::HttpSrc` (the `http-src`
feature, `reqwest`) GETs a URL and streams the body as `Caps::ByteStream` chunks,
the fetch layer the others share. Because a manifest/segment URL is
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
encrypted blocks only) using the same shared key handle `HlsSrc` fills. A clear
track stays a normal demux; an encrypted track with no key fails loud.
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
DASH stream play. A dynamic (live) MPD is reloaded on its `minimumUpdatePeriod`,
each new segment played once (tracked by start time), ending when the manifest
turns static, the same shape as the HLS live reload. `with_abr()` makes it
throughput-adaptive on the same shared `abr::BandwidthEstimator` as `HlsSrc`: a
`load_rep` helper resolves any Representation (Template / List / `sidx`-fetched
SegmentBase) into the run loop's segment/timescale/init working set, and the
estimate-derived cap drives both the per-reload pick and a per-segment
re-selection (so a static VOD adapts within one pass), re-emitting the init on a
switch.

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

Closed captions (CEA-608 / CEA-708) feed the same renderer, but their bytes ride
*inside* the compressed video bitstream rather than in a container text track, so
the path is a track, not a `SubParse`-style drop-in. The `cea` module (`no_std`)
holds the decoders. `extract_cc_data` mines the `(cc_type, b0, b1)` caption
triples from an access unit's SEI `user_data_registered_itu_t_t35` (ATSC A/53
`GA94` `cc_data`) messages for H.264 (NAL type 6) and H.265 (prefix/suffix SEI),
every count / length / offset bounds-checked so a malformed SEI yields no triples.
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

### 4.19 Native WebRTC (`str0m`)

The WebRTC elements are built on **[str0m](https://github.com/algesten/str0m)**, a
**sans-IO** WebRTC stack (ICE / DTLS / SRTP / RTP as a pure state machine): g2g
owns the `UdpSocket` and the timer and drives str0m's `poll_output` /
`handle_input` loop, exactly the contract the `srt` and `rtspserver` modules
already follow. str0m's pure-Rust **`rust-crypto`** backend is selected, so there
is no OpenSSL / libnice system dependency. Everything lives behind the opt-in
`webrtc` feature (it raises the effective MSRV above the workspace floor, so it is
off by default and the no_std baseline is unaffected). This is the native,
server-grade counterpart of the browser-only data-channel `WebRtcSrc` (§6.3).

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

**Signaling.** WHIP (egress) and WHEP (ingress) are the same wire move — an
`application/sdp` POST of the local offer that returns the remote answer (reqwest,
`webrtc_util::post_sdp`); the media server is the relay in the middle, so there is
no peer-to-peer mode for WHIP/WHEP. WHIP/WHEP are unidirectional by spec, so
sendrecv cannot use them: the duplex session instead exchanges SDP **directly**
between two peers over an `SdpChannel` (an in-process offer/answer transport for a
P2P loopback; a real SFU signaller — LiveKit, etc. — plugs into the same seam).
The two roles discover their m-line `Mid`s differently and this asymmetry is
load-bearing: the **offerer** captures its `Mid`s from `SdpApi::add_media`'s
return, while the **answerer** learns them from `Event::MediaAdded` after
`accept_offer` (str0m does not emit `MediaAdded` for media the local side added).

**ICE / NAT traversal.** `webrtc_util::add_ice_candidates` always adds the socket's
host candidate and, when a STUN server is configured, a server-reflexive candidate
discovered by a hand-rolled RFC 5389 Binding on the ICE socket; candidates ride in
the SDP, so a same-host P2P pair connects over localhost with no STUN. For the NAT
cases a reflexive candidate cannot punch through, a hand-rolled TURN client
(`turn.rs`, RFC 5766/8656: Allocate with long-term auth, Send/Data indications,
CreatePermission, periodic Refresh) provides a relay. str0m only offers
`Candidate::relayed`; the data plane is the run loop's job — a relayed pair's
transmits all carry `source == relay_addr`, which is the routing signal to wrap
the datagram in a TURN Send indication (direct host/srflx paths are untouched).

**RTCP feedback** rides the §4.13 reverse channel. A remote PLI
(`Event::KeyframeRequest`) becomes a `Reconfigure::ForceKeyframe` walked upstream
via `AsyncElement::take_reconfigure` to the encoder (`Av1Enc` forces a rav1e IDR);
ingress originates PLI on a mid-GOP join. str0m's BWE (`Event::EgressBitrateEstimate`,
TWCC/REMB) becomes `PushOutcome::Bitrate` via `take_bitrate`, and the encoder
retargets (rav1e by a hysteresis-gated context rebuild).

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
falls back to a topology-only dump. Because negotiation probes sources, a `--dot`
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
`run_source_transform_sink`) collect it; fan-in / fan-out / session / muxer
runners leave it empty, like their declared latency. It is `std`-gated where it
needs a clock: the histogram is `no_std`, but with no `monotonic_ns` the timing
compiles out (the table is then empty) so the `no_std` baseline pays nothing.
Sources have no `process()` and so do not appear, their cost surfaces as the
downstream element's input fill. Still open: per-*link* transit (queue-residency)
time, which needs a wall-clock stamp carried with each packet rather than the
element-side timing collected here.

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
regression that breaks a round-trip drops the level. `g2g-inspect --maturity` runs
the battery live and renders the matrix. `Oracle` / `Hardware` evidence, which the
in-process battery cannot produce (it has no ffmpeg / GPU), comes from the
resource-owning integration tests: they append it to a tab-separated evidence log
(`persist::record_evidence`, path `$G2G_CONFORMANCE_LOG`) when a check passes, and
`full_report` folds that log into the in-process report so `--maturity` shows the
`InteropTested` / `HardwareValidated` rows too. The native-muxer oracles mux an
`Mp4MuxN` fMP4 / a `TsMux` transport stream and have `ffprobe` demux them back,
recording peer-tagged `Oracle` evidence deriving `mp4mux` / `mpegtsmux` as
`InteropTested`; the ffmpeg-interop transports carry this further, `udpsrc` (RTP),
`rtmpsrc`, and `srtsrc` / `srtsink` (libsrt, incl. the AES variants) each derive
`InteropTested` against a named reference peer, and the Vulkan Video decode tests
persist GPU-tagged `Hardware` evidence (via `VulkanVideoDevice::device_name`) so
`vulkanvideo` derives `HardwareValidated` across H.264 / H.265 / AV1. A CI
`conformance` job runs the deterministic ffprobe oracles plus the (best-effort)
transport interop against a real ffmpeg, aggregating into one `$G2G_CONFORMANCE_LOG`
(the muxer oracles honor an externally-set log so they append rather than truncate)
and publishing `--maturity` to the job summary; the GPU `Hardware` rows come from a
self-hosted GPU runner. Together with the copy
plan (§3.2), this is
the validation-first posture: the framework states hard, checkable properties (this
graph is zero-copy; this element is unit-tested but not interop-validated) rather
than leaving them to prose and trust.

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

Reconnection (M558) makes the edge resilient across both transports.
`RemoteSink` / `RemoteWsSink` gain `with_reconnect(attempts)` (and a
`reconnect-attempts` property): the initial connect is deferred and retried with a
short backoff, and a mid-stream send failure drops the dead socket, reconnects,
and re-sends the current caps (the far side's required first packet) before
retrying, so a peer that starts late or restarts is transparently tolerated up to
the attempt budget. Symmetrically, `RemoteSrc` / `RemoteWsSrc` gain
`with_reconnect()` (a `keep-listening` property): a client that drops *without* a
clean `Eos` is not the stream's end; the source keeps its listener open, accepts a
replacement client (which re-sends its leading caps, forwarded downstream so it
re-negotiates if changed), and continues. Only an explicit `Eos` (or a frame
limit) ends a keep-listening source. Both directions are validated over loopback
(the sink retries until a late-binding server appears; the source stitches a
stream across a sender that drops and is replaced).

Remaining follow-ups: a native WebSocket server that *pushes* an unsolicited
stream to a browser `WsWireSrc` client (a receive-only browser edge, as opposed to
the transform's request/response), and a subgraph-as-a-unit wrapper (remoting a
whole `Bin` rather than a single edge).

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
also confirms dma-buf export+import work on this NVIDIA driver. Both packed
RGBA/BGRA and 8-bit NV12 are supported; the plane-aware frame size (RGBA is one
plane, NV12 / I420 add the half-height chroma region) is the shared
`dmabuf_frame_bytes` helper that both the export and the `DmaBufToWgpu` import use,
so they always agree on the buffer size (this also fixed the importer, which
previously imported only the luma plane of a planar frame).

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

`WgpuPreprocess` (`g2g-ml/src/wgpupreprocess.rs`, `wgpu` feature) is the compute-shader half: an NV12 frame is converted and normalized in a wgpu compute shader to a `Caps::Tensor { F32, [1,3,H,W], Nchw }`, the same contract `OrtInference` builds on the CPU. The default system-memory variant uploads NV12 to a storage buffer and reads the f32 tensor back to `MemoryDomain::System`. **GPU-output mode (`with_gpu_output`)** instead leaves the tensor in a `wgpu::Buffer` and emits `MemoryDomain::WgpuBuffer` (an on-device GPU->GPU copy into a fresh per-frame buffer, no map / read-back in the element), so a downstream GPU consumer reads it on-device; a CPU consumer pays the deferred read-back via the buffer owner. This removes the output-side GPU->CPU copy; `WgpuInference` (§5.2) is the consumer that binds the resulting buffer on-device, so `preprocess -> infer` keeps the tensor on the GPU. **Surface-import input** closes the other end: when the NV12 frame arrives already GPU-resident as a `MemoryDomain::WgpuTexture` (a `WgpuNv12Texture` keep-alive wrapping an R8Uint texture of `width x height*3/2` in standard NV12 byte layout), the element adopts that texture's device and samples it with `textureLoad` straight into the compute pass, with no CPU upload, bit-identical to the storage-buffer path. With both ends GPU-resident, `surface -> WgpuPreprocess -> WgpuInference` runs with the pixels never touching the CPU. **CUDA<->wgpu interop (`CudaToWgpu`, `g2g-plugins/src/cudawgpu.rs`)** joins the NVDEC decode side to this surface-import path: there is no portable "share this CUDA pointer with wgpu" call, so the bridge allocates an exportable Vulkan image (`VK_KHR_external_memory_fd`, wrapped as a `wgpu::Texture` via wgpu-hal), CUDA imports the same memory by FD (`cuImportExternalMemory`) and copies the NVDEC NV12 planes into it device->device, and the wgpu device travels on the frame's keep-alive so `WgpuPreprocess` adopts it (the device-identity pattern). The whole `NVDEC -> CudaToWgpu -> WgpuPreprocess -> WgpuInference` chain is validated on an RTX 3060, matching a CPU reference with no PCIe download. Shared images are recycled from a reuse pool: the Vulkan image, its CUDA import, and the `wgpu::Texture` are allocated once and returned to a free list when the downstream frame is released (a drop guard on the emitted keep-alive), so per frame only the two device->device plane copies and a sync run; a recycled entry is drained (`Device::poll`) before reuse since a wgpu submission may still sample it. The pool cut the bridge step ~2.6x at 1080p (p50 0.38 ms pooled vs 0.98 ms per-frame-allocated). **The reverse direction (`WgpuToCuda`)** closes the *encode* side: a renderer writes a packed-RGBA `wgpu::Texture` on FD-exportable Vulkan memory (`export_rgba_image` / `wrap_rgba_as_texture`, the `R8G8B8A8` mirror), CUDA imports it as a 4-channel array, and `to_cuda_frame` copies it device->device into a linear `CUdeviceptr` emitted as a `MemoryDomain::Cuda` `Rgba8` frame that `NvEnc` registers as `ABGR` (§4.11.3). So a GPU render reaches the H.264 encoder with no device->host read-back, validated on an RTX 3060 (`wgpu_to_cuda` test). This is the zero-copy egress for server-side rendering / cloud-gaming, and `examples/bevy-g2g-stream` is the runnable Bevy proof: Bevy renders on the interop device, g2g copies the target through `WgpuToCuda`, and `NvEnc` emits H.264 without a full-frame download. The bridge retains its own CUDA primary context (the GPU the interop device selects) and owns the exportable render-target texture.

### 5.2 Unified Pure-Rust Inference Backends
`g2g` avoids bundling heavy, unsafe proprietary C++ engines. The `g2g-ml` crate provides wrapper elements targeting two execution paradigms:

- **`g2g-ml::burn`** (Embedded / Wasm / RTOS): leverages the pure-Rust Burn framework with a `wgpu` backend, compiling ONNX workflows into type-safe, compile-time Rust graphics shaders. `BurnInference` (`g2g-ml/src/burninfer.rs`, `burn` feature) is the wgpu-backend inference element over the `RawVideo` → `Tensor` contract, driving an `input · W + b` linear layer on any Vulkan / Metal / DX12 / WebGPU adapter.
- **`g2g-ml::ort`** (High-Performance Enterprise Server): wraps ONNX Runtime bindings to pass underlying memory domains to hardware-specific execution paths (CUDA / TensorRT / DirectML / Apple CoreML) natively. Each execution provider is a constructor variant on `OrtInference` that registers the EP ahead of the CPU fallback; registration is best-effort, so the session keeps running (on CPU) when the device is absent. Desktop: `from_memory_with_cuda`, `from_memory_with_directml`. **Android edge**: `from_memory_with_nnapi` (the system NeuralNetworks API: NPU / GPU / DSP), `from_memory_with_xnnpack` (ARM-optimized CPU), and `from_memory_for_android`, which registers NNAPI then XNNPACK then the default CPU EP in one call so ORT assigns each node to the first provider that supports it, the MediaPipe delegate-with-fallback shape. The `nnapi` / `xnnpack` features link symbols only the Android ONNX Runtime build carries, so they are Android-target features (a host build / CI never enables them); the EP stack is validated on a device (`tools/android-nnapi-smoke.sh` runs `g2g-ml/tests/android_nnapi_probe.rs` from `/data/local/tmp`, a binder-threadpool shim for the vendor NNAPI HAL, output byte-exact with the CPU reference). **Edge TPU offload is proven**: an int8 QDQ Conv->ReLU fixture run through `from_memory_for_android` is placed on `NnapiExecutionProvider` (read from ORT's profiling JSON), and on a Pixel 10a (Tensor G4) the DarwiNN HAL log confirms the Edge TPU compiled and executed it (`/dev/edgetpu core0` firmware load); the float-typed input-boundary `QuantizeLinear` is the one op the TPU declines, correctly delegated to CPU (`tools/android-nnapi-conv-smoke.sh`, which also greps the `darwinn` logcat to disambiguate the TPU from other NNAPI accelerators). **Full-graph offload**: a uint8-input variant of the model (the boundary `QuantizeLinear` removed, the graph input retyped to uint8) runs *entirely* on the TPU, every node on `NnapiExecutionProvider` with nothing on the CPU, and the DarwiNN log confirms `Ops supported = ..., not supported = 0` / `compilation finished successfully on google-edgetpu`. The f32->uint8 quantization that feeds such a model is `TensorConvert` (`g2g-plugins`), the tensor-domain sibling of `VideoConvert`: it quantizes an f32 tensor to int8 / uint8 (`q = round(x / scale) + zero_point`, clamped) or dequantizes the inverse, shape and layout passing through. So `preprocess -> TensorConvert(quantize) -> inference` keeps the boundary quantize *out* of the model, leaving the whole inference graph accelerator-eligible. `TensorConvert` also transposes NCHW<->NHWC and narrows / widens f32<->F16 in the same pass, so a model that wants `NHWC uint8` (NNAPI / TFLite) is fed straight from an `NCHW f32` source. `OrtInference` itself accepts the integer input: `from_session` reads the model's input element type and `with_tensor_input` on a u8 / i8 model feeds the quantized tensor straight to the session (RGBA mode stays f32-only). **The whole chain is validated live on the device**: `Camera2Src -> TensorConvert(quantize) -> OrtInference(uint8) ` runs a real camera frame onto the Edge TPU (`tools/android-camera-tpu-smoke.sh`; on a Pixel 10a the logcat shows `accelerator name: EDGETPU` and `compilation finished successfully on google-edgetpu`), the g2g answer to "an edge framework that moves inference between CPU and accelerator" demonstrated end to end on real hardware. The same constructor shape extends to the other vendor accelerators: `from_memory_with_qnn` (Qualcomm AI Engine Direct, the Hexagon NPU / Adreno GPU on Snapdragon, the alternative to reaching the Hexagon through NNAPI) and `from_memory_with_coreml` (the Apple Neural Engine / GPU on macOS / iOS), each behind a target-only feature like `nnapi` (a host build never links them); both are validated to compile for their target, with on-device runtime validation pending the hardware (no Snapdragon / Apple device in CI, like the CUDA EP). This is the heterogeneous-device story (a desktop NVIDIA box, a Windows D3D12 GPU, an Android phone NPU, and the Qualcomm / Apple NPUs all run the same element, the EP picked per platform), the architectural answer to MediaPipe's runtime CPU/GPU delegate switch.

`WgpuInference` (`g2g-ml/src/wgpuinfer.rs`, `wgpu` feature) is the GPU-resident counterpart of `BurnInference`: a raw wgpu compute pass that binds the GPU-resident tensor `WgpuPreprocess::with_gpu_output` (§5.1) produced **directly**, rather than taking `RawVideo` / `System` and uploading. It runs one of a small op zoo on that tensor, selected at construction (each its own WGSL shader behind the shared device-adopt / dispatch / read-back machinery): the original `input · W + b` linear matmul (`linear`); a same-padding stride-1 2D convolution (`conv2d`) over the `[1, Cin, H, W]` NCHW tensor with `[Cout, Cin, KH, KW]` weights, leaving a `[1, Cout, H, W]` feature map; the elementwise activations `relu` / `sigmoid`; and `maxpool2d` / `avgpool2d` spatial pooling. The weighted ops (linear, conv2d) bind a 5-entry group (meta, input, weights, bias, out); the weightless ops (activation, pooling) bind a 3-entry group (meta, input, out), the bind-group layout following the active shader. The conv is the keystone that lets the chain run an actual CNN layer, not just a final classifier; the activation is the nonlinearity that keeps stacked convs from collapsing to one linear map, and the pool the spatial downsampler. Chained GPU-resident (`conv2d -> relu -> maxpool`, each in `with_gpu_output` mode so the data never leaves the device between layers), they are a real small-CNN body, validated on the RTX 3060 against a CPU reference folding the same ops (`conv2d_reference` / `relu_reference` / `maxpool2d_reference`) over the exact tensor the GPU preprocess produced. **Trained weights are imported at runtime** from a `safetensors` file via a dependency-free reader (`g2g-ml::safetensors`, a focused parser for the format's `u64` length + JSON-subset header + raw tensor bytes, no `serde` / no `safetensors` crate): `conv2d_from_safetensors` reads the `[Cout, Cin, KH, KW]` weight and `[Cout]` bias by name and infers the kernel dims, so picking a different trained checkpoint is "parse a different file" while the layer topology stays this compiled element. This is the weights half; the architecture stays Rust (truly dynamic *graphs* at runtime are the `ort` backend's job, and `burn-import` build-time codegen is the Burn-side topology path). It owns no device: because a `wgpu::Buffer` is bindable only on the device that created it, the element adopts the producer's device / queue (carried by the incoming `WgpuBufferOwner`) on the first frame and submits its compute on the producer's queue, which orders it after the producer's work with no fence or read-back. The logits are read back to `MemoryDomain::System` by default or left GPU-resident (`with_gpu_output`) for a downstream GPU consumer. A burn / ort consumer cannot do this zero-copy: their tensor handles are opaque (no foreign-buffer adopt) and run on their own device, so they would force the GPU->CPU->GPU round-trip the GPU-resident preprocess and inference paths exist to delete.

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
- **Metadata through fan-out.** `FrameMetaSet` holds each `FrameMeta` as
  an `Arc<dyn FrameMeta>` and is `Clone`, so a tee clone shares the analytics
  graph by refcount rather than dropping it: the graph runner's
  `try_clone_packet` carries `frame.meta.clone()`, landing the same
  `AnalyticsMeta` on both branches of a `decode -> tee -> {detect, video}`
  diamond. Mutation is copy-on-write via `FrameMeta::clone_box` (the GstMeta
  `copy_func` analog): `FrameMetaSet::get_mut` deep-copies a shared entry before
  the mutable borrow, so a branch editing its analytics never aliases the
  sibling. Still a ZST no-op when the `metadata` feature is off.
- **The overlay.** The visible end of the detector chain reads the
  `AnalyticsMeta` carried onto the *display* frame (via the fan-out path) and
  draws each box, so `decode -> tee -> {detect, video} -> overlay -> display`
  works. Two backends with a shared per-class palette: the CPU
  `g2g-plugins::analyticsoverlay::AnalyticsOverlay` (`analytics` feature) paints
  box outlines onto RGBA8 with the compositor's integer source-over blend (the
  `no_std` baseline), and the GPU `vellooverlay::VelloAnalyticsOverlay`
  (`vello-overlay` feature) strokes antialiased boxes over a full-frame image
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
  surface the app supplies rather than opening its own window.

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
  unchanged.

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
  H.264 Annex-B access units in, `VideoFrame` copied to `System` RGBA out.
  Build needs `--cfg=web_sys_unstable_apis`.
- `CanvasSink` — presents decoded RGBA to an HTML canvas via the 2D context.
  A WebGPU-texture zero-copy variant uses `MemoryDomain::WebGPUBuffer` into
  a `GPUTexture` once the async device handshake lands in the keep-alive.

A complete in-browser glass-to-glass pipeline is
`WebSocketSrc → H264Parse → WebCodecsDecode → CanvasSink`. The local gate
for the wasm build is
`cargo check --target wasm32-unknown-unknown -p g2g-plugins --features web`.

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
