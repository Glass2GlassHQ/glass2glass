# glass2glass MCU safety manual

Conditions of use and safety characteristics of the `glass2glass` MCU / safety
path: the heap-free, `no_std`, no-`alloc` static element model in `g2g-core`
(the `staticelem`, `spsc`, `supervise`, `staticpool` modules and the heap-free
data plane) and the `g2g-mcu` peripheral elements. It is the companion to the
machine-checkable [`REQUIREMENTS.md`](REQUIREMENTS.md): the matrix says *what* is
guaranteed and points at the evidence; this manual says *under what assumptions*
the guarantees hold and *what the integrator is responsible for*.

This is a pre-release document for a pre-1.0 framework. It is not a certificate
and does not by itself qualify a product for any standard (IEC 61508, ISO 26262,
DO-178C, IEC 62304). It is a structured, evidence-backed description of the
framework's safety-relevant properties, intended as a starting point and a
down-payment for a product-level safety case, not a substitute for one.

## 1. Scope

In scope: the no-`alloc` build of `g2g-core` (`--no-default-features`) and
`g2g-mcu`, driven by the static runners (`run_source_sink`,
`run_source_transform_sink`, `run_sources_fanin_sink`, `step_source_sink`, and
the `supervise` runners). This is the configuration an MCU integrator ships.

Out of scope: the `alloc` / `std` / `runtime` builds (the dynamic `Graph`,
`parse_launch`, the caps solver, the object-safe `dyn` element traits), which
target hosts and are not part of the MCU claim. Every guarantee below is stated
for the no-`alloc` static configuration only.

## 2. Safety-relevant properties (and their assumptions)

### 2.1 No dynamic memory (REQ-MEM-01, REQ-MEM-04)

The MCU pipeline links **no allocator**: built for a bare-metal target with no
`#[global_allocator]` and no `alloc` crate, it references zero allocator symbols
(asserted at link time by `tools/noalloc-check.sh`). All state is stack-local or
in caller-provided `static` rings (`StaticLendRing`, `SpscFrameRing`). There is
no heap to fragment, exhaust, or leak.

Assumption: the integrator builds the no-`alloc` configuration and does not add a
global allocator. Enabling any `alloc`-implying feature voids this property.

### 2.2 No panics (REQ-MEM-02, REQ-INPUT-01)

The MCU archive contains **no reachable panic machinery**: no bounds-check,
overflow, unwrap, or slice-index panic paths, and the compiler's
resumed-after-completion async guard is discharged by the single-poll executor.
`tools/noalloc-check.sh` asserts zero `core::panicking` symbols, so the mandatory
`#[panic_handler]` is provably dead code. Parsers and elements achieve this by
construction: checked / saturating arithmetic, slice patterns and `.get()` rather
than indexing, and folding attacker-controlled lengths so malformed input returns
an error rather than panicking (REQ-INPUT-01).

Assumption: the integrator's own application code and HAL adapters uphold the
same discipline. The framework cannot prove the absence of panics in code it does
not compile. Application code compiled under `#![forbid(unsafe_code)]` and free of
indexing/unwrap keeps the property end to end (REQ-UNSAFE-01).

### 2.3 Bounded footprint and timing (REQ-MEM-03, REQ-TIME-01)

ROM, static RAM, and worst-case stack are computed from the linked image's call
graph and **budget-enforced** in CI (`tools/footprint-report.sh`); worst-case
execution time and jitter are measured deterministically under instruction-count
timing and budget-enforced (`tools/timing-report.sh`). A regression that grows
any budget fails CI.

Assumption: the reported stack bound covers the framework's call graph; the
integrator must add the stack used by interrupt handlers and their own code, and
size the RTOS task / bare-metal stack accordingly. The timing numbers are
emulated instruction timing, not silicon timing (see §5).

### 2.4 Bounded fault handling (REQ-FAULT-01..03)

Faults are **typed values, never panics**: a peripheral or pool failure returns
`G2gError` (REQ-FAULT-01). The `supervise` layer turns a returned fault into a
*bounded, deterministic* action, retry, degrade (skip), reset, or escalate, with
a hard iteration cap (`MAX_ATTEMPTS`) so even a mis-written policy cannot loop
(REQ-FAULT-02). A `Watchdog` is refreshed only on real forward progress, so a
wedged or escalated pipeline stops petting and a hardware watchdog resets the MCU
(REQ-FAULT-03).

Assumption: the integrator wires a real hardware watchdog to the `Watchdog` seam
and configures its timeout longer than the worst-case per-frame processing time
(§2.3) but short enough for the application's fault-reaction budget. The
framework detects and reacts to faults it is told about (returned errors, absent
progress); it cannot detect a fault the HAL adapter swallows.

### 2.5 Concurrency and the receive path (REQ-CONC-01, REQ-RECV-01)

The ISR-to-pipeline capture hand-off (`SpscFrameRing`) is lock-free (atomic
load/store only, no CAS, so it holds on cores without CAS) with bounded,
non-blocking back-pressure: a full ring drops and counts rather than blocking the
interrupt (REQ-CONC-01). The receive jitter buffer bounds latency to its window
and never stalls the stream on a lost packet (REQ-RECV-01).

Assumption: exactly one producer context (the ISR) and one consumer context (the
pipeline task) use each `SpscFrameRing`, per its single-producer/single-consumer
contract. Two producers or two consumers void its soundness.

### 2.6 Data integrity (REQ-INTEG-01)

Codec math is validated bit-exact against an independent reference (G.711 and
IMA ADPCM vs ffmpeg; RTP vs ffmpeg as a peer), and sensor drivers validate the
device's integrity checks (SHT3x CRC-8) and reject corrupt data rather than
trusting it. Conversions use the datasheet transfer functions with overflow-safe
fixed-point arithmetic.

## 3. The `unsafe` inventory

Application code needs **zero** `unsafe` (REQ-UNSAFE-01, proven by building a full
camera-to-display pipeline under `#![forbid(unsafe_code)]`). The framework's
`unsafe` is localized to a small set of memory-lending primitives, each with a
documented `SAFETY:` justification (the workspace lints
`undocumented_unsafe_blocks` and `unsafe_op_in_unsafe_fn` enforce this):

- `SystemSlice::from_foreign` and the `StaticLendRing` / `SpscFrameRing` /
  `JitterBuffer` zero-copy lend, whose soundness rests on the ring outliving the
  lent frame and the single-frame-in-flight discipline the static runners follow.
- The `drive_ready` no-op waker (the single-poll executor).
- `SpscFrameRing`'s `Sync` impl (the single-producer/single-consumer contract).

An integrator reviewing the framework audits these sites; the count is small and
fixed, and none are in application code.

## 4. Integrator responsibilities (summary)

1. Build the no-`alloc` configuration; do not add a global allocator (§2.1).
2. Keep application/HAL code panic-free and, where possible, `forbid(unsafe_code)`
   (§2.2).
3. Size stacks to include interrupt and application usage on top of the reported
   framework bound (§2.3).
4. Wire a hardware watchdog to the `Watchdog` seam with an appropriate timeout,
   and choose a `FaultPolicy` matching the application's degraded-mode needs
   (§2.4).
5. Honour the single-producer/single-consumer contract on each `SpscFrameRing`
   (§2.5).
6. Provide HAL adapters that report faults (return errors) rather than swallowing
   them, and do not panic (§2.2, §2.4).

## 5. Limitations

- Emulated, not silicon: on-target proofs run on QEMU (Cortex-M3/M4). Emulated
  instruction timing is not silicon timing; the on-device `Hardware` conformance
  rows (real STM32 / i.MX RT) are future work.
- Pre-1.0 and not certified: no standard's qualification has been performed. This
  manual and the traceability matrix are inputs to a product safety case, not a
  certificate.
- The guarantees cover the framework's compiled code and its documented
  contracts, not the integrator's application code, HAL adapters, or hardware.

## 6. Qualification kit

`tools/qualification-kit.sh` runs the full set of safety proofs
(no-heap/panic-free, footprint, timing, on-target execution across executors, the
codec/sensor oracles, and the traceability check) and prints a consolidated
requirement-to-evidence-to-result report, the evidence package an integrator
attaches to a safety case. The individual proofs also run in CI on every change.
