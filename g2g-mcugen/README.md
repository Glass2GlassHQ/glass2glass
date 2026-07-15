# g2g-mcugen

Host graph compiler: turn a declarative audio graph document into a
monomorphized static MCU pipeline (heap-free Rust) plus its ring-memory budget.

The develop-on-Linux, ship-a-bounded-static-build-to-the-MCU story: the same
`capture -> convert -> resample -> mix -> encode -> RTP` graph the hand-written
`noalloc-pipeline::audio` wires by hand, this compiler wires from a
`graph.yaml`, computing each ring's size from the graph's frame geometry
instead of hard-coding it.

```sh
cargo run -p g2g-mcugen -- g2g-mcugen/examples/flagship.yaml -o out/graph.rs
```

The generated module exposes `run_<name>_with(grab_a, grab_b, sender)`, generic
over the capture-grabber and RTP-sender seams: a board supplies its HAL impls,
a proof harness supplies mocks. The ring-memory budget prints to stderr.

## Element catalog

| element      | role      | props |
| :---         | :---      | :--- |
| `grabbersrc` | source    | `sample-rate` (8000/16000/48000), `width` (2=S16, 4=S32 slot), `channels` |
| `pcmconvert` | transform | (32-bit slots -> S16) |
| `resample`   | transform | `from`, `to` (integer-ratio pairs of the rate set) |
| `mixer`      | fan-in    | `gain-a`, `gain-b` (Q15) |
| `g711enc`    | transform | `law` (mulaw / alaw) |
| `rtpsink`    | sink      | `clock-rate`, `payload-type`, `ssrc`, `sequence` |

Topologies: a linear `source -> transforms -> sink` chain, or two source
branches joined by one fan-in (the mixer) then a linear tail. Anything else is
rejected with a diagnostic.

## Why its own schema (not the dynamic `GraphSpec`)

This document schema carries the frame geometry a static build needs
(`frame_ns`, `frames`, per-source sample format) and drops the dynamic-only
escape hatches (`pipeline:` launch string, tee/demux fan-out, per-edge
backpressure) that cannot monomorphize. It is a different backend for a
different target, not a copy of the runtime graph loader.

## Proof

`examples/mcugen-audio` regenerates the flagship graph, builds it heap-free for
`thumbv7em`, and asserts its RTP wire output matches the hand-written reference
`AUDIO_EXPECTED_CHECKSUM` byte-for-byte (`tools/mcugen-check.sh`, in CI).
