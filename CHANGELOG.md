# Changelog

Pre-release. Work is tracked by milestone (Mn) following the roadmap in `DESIGN.md` §4.10.
Nothing is published yet; all versions are `0.1.0`.

## Unreleased

### M9: Dynamic fan-out / fan-in (in progress)
- `Router` (1->N): routes each frame to an atomic-selected output port, broadcasts `CapsChanged` to all ports.
- `Gate` (1->1): forwards or drops data by an atomic flag; a plain `AsyncElement`, so it needs no new runner.
- `Merger` (N->1): selects one active input, drains the rest, emits one merged `Eos` after every input ends.
- Multi-output / fan-in plumbing: `MultiOutputSink`, `MultiOutputElement`, `DynSourceLoop`, `join_all`, and the `run_source_fanout` / `run_fanin_sink` runners. Each primitive has a cloneable control handle (`RouterHandle` / `GateHandle` / `MergerHandle`).
- Deferred: `BranchSlot` + `SwapPolicy` variants; per-branch caps negotiation (M10).

### M8: Caps negotiation + dynamic element swap
- Caps algebra: `Dim`/`Rate`/`Caps` `intersect`, `is_fixed`, `fixate` (Phase 1 narrowing, Phase 2 fixation per §4.2).
- Runners fixate negotiated caps before configuring elements; bounded `ReFixate` retry; mid-stream `CapsChanged` cascade and upstream `Reconfigure` sideband.
- `ElementSlot` + `SwapHandle`: lock-free atomic hot-swap of one element mid-stream; `DynAsyncElement` blanket impl so any element boxes into a slot.

## M0-M5 (prior)
- M0/M1: Cargo workspace scaffold and minimal `no_std + alloc` pipeline runtime (bounded channel, `Join2`, linear runner).
- M2: async `OutputSink`, identity transform, 3-element runner.
- M3: `AsyncClock` + `WallClock` + `SyncSink` for PTS-paced presentation.
- M4: Arc-recycled `BufferPool` with async `acquire`; DMABUF close-on-drop fix.
- M5: `RtspSrc` element wrapping `retina`.
