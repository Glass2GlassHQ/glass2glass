# Safety requirements traceability matrix

This is the machine-checkable requirements matrix for the `glass2glass` MCU /
safety path (the heap-free, `no_std`, no-`alloc` element model in `g2g-core` and
`g2g-mcu`). Each requirement states a property a safety integrator needs, its
verification method, and the **evidence** that verifies it, a proof script, a
named test, or a CI job that all exist in this repository.

`tools/traceability-check.sh` parses this table and fails if any cited evidence
is missing (a renamed test, a deleted proof script, a removed CI job) or if any
proof script is not actually wired into a CI workflow. So this matrix is a
*checked* claim, not a document that can drift from the code: it is run in CI
like every other proof. The conditions of use, assumptions, and the fault model
are in [`SAFETY_MANUAL.md`](SAFETY_MANUAL.md).

Evidence tokens: `script:<path>` (a proof script that runs in CI),
`test:<fn>` (a `#[test]` function), `job:<key>` (a CI job).

| ID | Category | Requirement | Verification | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| REQ-MEM-01 | Memory | The MCU pipeline links no dynamic memory allocator (no heap). | Link-time symbol analysis | `script:tools/noalloc-check.sh job:features-linux` |
| REQ-MEM-02 | Memory | The MCU pipeline contains no reachable panic machinery (no bounds / overflow / unwrap / slice panic paths). | Link-time symbol analysis | `script:tools/noalloc-check.sh` |
| REQ-MEM-03 | Memory | The pipeline's ROM, static RAM, and worst-case stack are bounded and budget-enforced against regression. | Footprint call-graph analysis | `script:tools/footprint-report.sh` |
| REQ-MEM-04 | Memory | The data path performs zero heap allocation at steady state. | Runtime allocator counter | `test:static_runner_pipeline_makes_zero_heap_allocations test:fixed_pool_frame_path_makes_zero_heap_allocations` |
| REQ-UNSAFE-01 | Unsafe | Application code on the MCU surface needs no `unsafe`; the `unsafe` contracts live once in the framework. | Compile under `forbid(unsafe_code)` | `test:whole_pipeline_builds_and_runs_without_unsafe` |
| REQ-TIME-01 | Timing | The pipeline's worst-case execution time and jitter are bounded and data-stable, budget-enforced. | Deterministic (icount) timing | `script:tools/timing-report.sh` |
| REQ-EXEC-01 | Execution | The pipeline executes bit-exactly on the target ISA, on the bare, Embassy, FreeRTOS, and Zephyr executors. | On-target emulation | `script:tools/qemu-check.sh script:tools/freertos-check.sh script:tools/zephyr-check.sh` |
| REQ-FAULT-01 | Faults | Peripheral and pool faults are surfaced as typed errors, never as a panic. | Fault-injection tests | `test:capture_failure_propagates test:ring_exhaustion_surfaces_as_pool_exhausted` |
| REQ-FAULT-02 | Faults | Fault recovery is bounded and deterministic (retry, then reset, then escalate, within a hard cap). | Supervisor tests + on-target | `test:permanent_fault_escalates_within_bounds_and_stops_petting test:a_never_escalating_policy_still_stops_at_the_hard_cap script:tools/qemu-check.sh` |
| REQ-FAULT-03 | Faults | A watchdog is refreshed only on real forward progress, so a wedged or escalated pipeline resets the chip. | Supervisor tests + on-target | `test:transient_fault_recovers_via_reset_and_delivers_every_frame script:tools/qemu-check.sh` |
| REQ-INPUT-01 | Input validation | Wire / bitstream parsers reject malformed input without panicking, overflowing, or over-allocating. | Malformed-input tests | `test:parse_rejects_malformed_input_without_panicking test:framing_and_sizing_are_validated` |
| REQ-CONC-01 | Concurrency | The ISR-to-pipeline capture hand-off is lock-free with bounded, non-blocking back-pressure. | SPSC ring tests + on-target | `test:full_ring_drops_and_counts_overruns script:tools/qemu-check.sh` |
| REQ-RECV-01 | Receive path | The jitter buffer bounds latency and never stalls the stream on a lost packet. | Reorder / loss tests + on-target | `test:a_lost_packet_yields_a_gap_and_does_not_stall script:tools/qemu-check.sh` |
| REQ-INTEG-01 | Data integrity | Codec and sensor conversions are validated against an independent reference (ffmpeg / datasheet), and integrity checks reject corrupt data. | Oracle / datasheet tests | `test:ffmpeg_oracle_bit_exact_full_domain test:crc8_matches_the_datasheet_worked_example test:sht3x_source_rejects_a_corrupt_crc` |
| REQ-TRACE-01 | Process | This requirements matrix is itself verified: every cited proof exists and runs in CI. | Traceability check | `script:tools/traceability-check.sh` |
