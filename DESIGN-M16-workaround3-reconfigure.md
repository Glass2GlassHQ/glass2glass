# M16 Workaround #3 — forward in-band reconfigure

> Implementable spec for the last open M16 piece. Expands
> `DESIGN-M16-caps-nego.md` §6 from concept into a phased plan with the
> design decisions called out for sign-off. No code has landed yet.

## 1. What the workaround is

All three decoders (`FfmpegH264Dec`, `MfDecode`, `VaapiH264Dec`) treat an
incoming `PipelinePacket::CapsChanged` as a no-op:

```rust
PipelinePacket::CapsChanged(_) => {
    // Upstream H.264 caps are swallowed; we emit NV12 CapsChanged from
    // the decoder's first decoded frame and on geometry changes.
}
```

The decoder instead *self-detects* its output geometry from each decoded
frame and emits its own `CapsChanged(NV12, w, h)` downstream, de-duped
against `last_caps`. The runner re-configures the next element on that
forward `CapsChanged` (`runner.rs:218` for the sink, `:569` for a
transform) before any new-caps `DataFrame` reaches it.

So today's behavior is functionally correct for the linear
`rtsp -> decoder -> sink` chain: a mid-stream resolution change works,
just driven by decode-time self-detection rather than by the upstream
caps event.

## 2. Why it is still a workaround

1. **The constraint change bypasses the solver.** The decoder is a
   `DerivedOutput` element; an upstream caps change should re-derive its
   output through that declared constraint and re-solve the downstream
   subgraph. Instead it is recovered heuristically from pixels. A
   downstream allocator that sizes pools from negotiated caps, or a
   second format-changing transform, never participates.
2. **Silent drop.** Swallowing means the decoder cannot reject an
   illegal mid-stream format change (e.g. H.264 -> VP9) loud; it just
   ignores it.
3. **Non-linear topologies.** Fan-out / muxer branches downstream of a
   boundary can't be re-solved off a self-detected event.

## 3. The ordering constraint (why the naive fix is wrong)

The obvious "fix" is: on `CapsChanged(in)`, run the decoder's
`DerivedOutput(in)` and forward `CapsChanged(out)` immediately. **This
corrupts the stream.** Decoders buffer for B-frame reorder, so across a
resolution boundary, old-geometry frames may still drain *after* the
input `CapsChanged` arrives. Forwarding the derived output caps eagerly
reconfigures the sink to the new geometry before the buffered old frames
land, so those frames are presented under the wrong configuration.

The reconfigure event must sit in the forward stream **between the last
old-caps `DataFrame` and the first new-caps one**. That is exactly the
position the decode-time self-detected `CapsChanged` occupies today.
`Frame.caps` already tags every `DataFrame` with the caps it was produced
under, so the ordering information exists; the fix must preserve it.

## 4. Design

**Principle.** The reconfigure is an ordered forward event positioned at
the caps boundary in the data stream. `CapsChanged` already is that
event. The work is to (a) route it through the declared constraint
instead of pixel heuristics, (b) make the runner treat a boundary's
forward `CapsChanged` as a downstream *subgraph re-solve*, not just a
single-element `configure_pipeline`, and (c) stop dropping the input
event silently.

### Decisions for sign-off

- **D1 — carrier.** Reuse `CapsChanged` as the in-band reconfigure
  carrier (it is already ordered and already drives downstream
  `configure_pipeline`), vs. add a dedicated forward `Reconfigure`
  packet variant. *Recommendation: reuse.* Add semantics, not a variant;
  a new variant would duplicate `CapsChanged`'s ordering role.
- **D2 — output derivation.** The boundary's output caps come from its
  `DerivedOutput` closure (single source of truth) rather than ad-hoc
  geometry helpers. For a linear chain the decode-time geometry equals
  `DerivedOutput(input)`, so the two agree; the design keeps emitting at
  the correctly-ordered decode boundary but derives the value from the
  constraint (and asserts equality in debug).
- **D3 — runner re-solve.** On a forward `CapsChanged` crossing a format
  boundary, the runner re-solves the downstream subgraph
  (boundary -> ... -> sink) and re-issues configure to every link whose
  assignment changed, instead of configuring only the immediate next
  element. Required for chains with >1 downstream element or an
  allocator. (Today's `runner.rs:218/569` configure only the adjacent
  element.)
- **D4 — acknowledge, don't swallow.** The decoder consumes the input
  `CapsChanged`: records it as the current input caps and validates it
  (reject an incompatible format change loud), rather than dropping it.

### Phased implementation

- **Phase A (linear, low risk, CI-testable).** Decoders stop silently
  swallowing: on `CapsChanged(in)` they validate the format (reject
  non-H.264 with `CapsMismatch`), record `in` as current input caps, and
  keep emitting the output `CapsChanged` at the decode boundary (ordering
  preserved). Add a debug assertion that the decode-time output caps
  equal `DerivedOutput(recorded_input)`. Behavior is unchanged for valid
  streams; the silent drop becomes validate + record. Testable with a
  fake in-memory decoder, no hardware.
- **Phase B (subgraph re-solve).** Runner, on a forward `CapsChanged` at
  a boundary, re-solves boundary -> sink and re-configures all changed
  links, not just the next element. Unlocks downstream allocators and
  multi-element downstream segments. Testable with a 3+ element fake
  chain.
- **Phase C (non-linear).** Extend the subgraph re-solve to fan-out /
  muxer downstream of a boundary.

## 5. Frame tagging invariant

Already satisfied structurally: every `DataFrame` carries `Frame.caps`.
The spec promotes this to a stated invariant: *a `DataFrame`'s caps are
authoritative for that frame; downstream elements must honor per-frame
caps, not assume the last `CapsChanged` applies to every subsequent
frame.* Phase A should audit the display sinks to confirm they key on the
frame they render, not a cached caps, across a reconfigure.

## 6. Testing

A fake decoder driven through a resolution-change sequence, no hardware:

```
CapsChanged(in@A) -> frames@A ... -> CapsChanged(in@B)
  -> (buffered frames@A drain) -> frames@B
```

Assert downstream observes `CapsChanged(out@A)` before any A-frame, and
`CapsChanged(out@B)` only *after* the last A-frame and before the first
B-frame. This is the ordering property §3 is about and is the regression
guard for any future change to the swallow path.

## 7. Open questions for the architect

- D1 and D3 are the load-bearing choices; confirm before Phase B.
- Is **Phase A alone** enough to declare #3 retired for the linear case
  (current production topology), deferring B/C until an allocator or a
  branched topology actually needs them?
- Interaction with the existing reverse `Reconfigure` path
  (`PushOutcome::Reconfigure`, `SourceLoop::reconfigure`): the forward
  reconfigure and the downstream counter-proposal should not race. A
  forward `CapsChanged` re-solve that a downstream element rejects must
  fall back to the existing `ReFixate` -> `request_reconfigure` flow
  (`runner.rs:233`).

## 8. Out of scope

- The codec-vs-raw `Caps` split (`DESIGN-M16-caps-nego.md` §12) would
  make the decoder's input/output media types distinct types, which
  simplifies D4's validation. Independent; not required here.
