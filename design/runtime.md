# Runtime: dynamic graphs, lifecycle and observability

Changing a graph while it runs, the state machine and seek model, and the bus and
logging channels an application watches. Part of the design in
[README.md](README.md).

## Two graph APIs

g2g exposes two graph APIs sharing the same element traits, the same negotiation
lifecycle, the same `PipelinePacket` variants, and the same runner primitives.
Only graph construction and slot mutation differ.

The static typed graph is a compile-time topology via tuple types, with no `dyn`
and zero cost, right for embedded, RTOS and static cloud pipelines. The
type-erased dynamic graph holds boxed elements (`Box<dyn DynAsyncElement>`) in
`ElementSlot`s, swappable at runtime, right for cloud
ingestion, desktop applications, and anything that needs runtime topology
evolution.

### ElementSlot

The dynamic graph holds elements in `arc_swap::ArcSwap<Box<dyn DynAsyncElement>>`
cells:

```rust
let new_element = SomeTransform::new();
new_element.configure_pipeline(&caps)?;
slot.handle.store(Arc::new(Box::new(new_element)));
```

Frames mid-`process()` against the old element complete naturally, and the next
push observes the new element. The cost is one atomic store plus the new
element's `configure_pipeline()` work, with no drain and no pipeline stall.

This is the primary response to a Phase 3 `ReFixate` or a mid-stream
`Reconfigure` signal: replace the affected slot's contents rather than rebuild
the graph. The swap is validated live under load, with an `ElementSlot` sitting
as a transform in `source -> slot -> sink` driven by `run_graph` and a
`SwapHandle::swap` mid-stream rerouting the remaining frames to the replacement
while every frame still reaches the sink.

### Router, Gate and Merger

A `Router` is a 1-to-N transform that reads an atomic discriminator per frame and
pushes the frame to exactly one of its outputs. A `Gate` is a 1-to-1 transform
that reads an atomic boolean and either forwards or discards each frame. A
`Merger` is an N-to-1 transform that reads from one of its inputs, switching on a
discriminator. Together they cover branch enable and disable and A/B switching.

### Runtime request pads

The request-pad analog is a pair of handles over a running graph.
`DynamicFanoutHandle::add_branch` attaches an output branch mid-run, round-robin
or broadcast under `FanOutMode`, where broadcast duplicates via `Frame::share`
and replays sticky caps to the late branch. `DynamicFaninHandle::add_input`
attaches a source as a new input of a running aggregator or muxer.

An input add is a negotiation, not a grant. The runner reserves the pad,
validates the source's caps against the pad constraint, then asks the element via
`MultiInputElement::accepts_runtime_input`, its veto for what pad count and caps
cannot express: no spare pad of that media kind, or a container that cannot carry
a second track. The caller holds a `PendingInput` whose `accepted()` resolves to
the verdict, and a refused input fails alone (`InputRefused`, logged under the
`fanin` category) while the run continues on the inputs it has.

## Mid-Playing splice: GraphMutator

`ElementSlot` swaps what sits in a position. `GraphMutator` changes the positions
themselves: it splices a transform onto a live edge of a running `run_graph` or
`run_graph_threaded` graph, lifts one back off, and swaps the source or the sink
on either end of it, while frames keep flowing. It also grows a tee by a branch and
a muxer by an input. `RunBuilder::run_mutable` and `run_threaded_mutable` hand the
caller the handle beside the run future, the way `run_source_router_dynamic` does.

`RunBuilder` is the one entry point for a run, whatever it carries: a bus, a state
controller, progress, a copy budget, an observer, a flight recorder, a visit
recorder, a tick timer and the mutator, in any combination. `RunBuilder::new(graph, &clock, capacity)`
takes the required parts and each `with_*` adds one. The `run_graph_*` functions
remain as shorthands for the single-feature cases.

The mechanism is a retargetable producing endpoint per edge. The arm pushing into
an edge does so through a `SenderSink` whose `LinkSender` the mutator can take
away and replace, so the producer is what moves rather than the consumer, and a
consumer never learns that its upstream changed. Between packets the producer
checks one relaxed atomic, and while a mutation is in flight it parks there,
holding the packet it has not sent, and resumes on whichever link the mutator
staged.

Resuming and parking again are separate steps, and a resume clears only the park
request it answers: on the thread-per-arm runner the next operation can ask for a
park while the producer is still on its way back from the last resume, and that
request has to survive, or it would wait on a park that already happened. The
endpoint also carries the edge's sticky caps, updated as each `CapsChanged`
crosses it, because a mid-stream re-solve moves the shape away from the negotiated
solution and a splice has to be configured against what is flowing now.

### Insert needs no drain

The new element's arm is given the producer's original sender as its output, and
the producer is retargeted to a fresh channel feeding the new element's input.
Packets already queued on the original channel therefore stay ahead of everything
the new element emits, in the same FIFO, so ordering is preserved without draining
anything.

Negotiation happens before the producer is touched: the element must accept the
edge's current caps through `intercept_caps` and `configure_pipeline`, and if what
it emits differs, that shape must be in the downstream feasibility set the run
computed at startup. A refusal leaves the graph untouched. When it does change the
caps, the mutator queues the `CapsChanged` on the original channel while the
producer is still parked, behind the packets already in flight and ahead of the new
element's first frame, which is exactly where the existing mid-stream re-solve
expects it.

Consent is required, not merely checked. A caps change with no feasibility set to
check it against is refused (`DownstreamRefused`), never waved through: an element
that turns one down mid-stream fails the whole run, so an unverifiable change is a
worse outcome than a refused operation. That covers the chains the startup backward
sweep cannot express, and the edge above a spliced element, which carries exactly
one known-good shape, the caps that element accepted and was configured for,
because nothing solved what else it would take. A further splice there must
therefore be caps-preserving.

### Remove drains, then flushes

Parking the producer and dropping the link it gave up closes the removed element's
input, so its arm consumes what is queued, forwards the results and ends, leaving
its output link on its own endpoint as it goes. Only then does the producer take
that link over, so every frame that was queued at the removed element passes
through it before the first bypassed frame arrives. The element itself is handed
back to the caller. A caps-changing element's consumer is about to start receiving
the producer's caps instead, so it must accept them, the same feasibility test and
refused otherwise, and is told with a `CapsChanged` ahead of the first frame.

The frames a removed element holds internally come out too, so a reordering
position is as removable as a stateless one. The mutator raises a drain flag on the
element's own endpoint before it closes the input, and when the arm reaches the end
of that input it hands the element an `Eos`, the one signal every g2g element
treats as release what you are holding, where `Flush` means discard. The frames
that produces enter the same downstream link, still ahead of the first bypassed
frame.

The marker itself stops at the element's sink adapter, which swallows an `Eos` for
the length of the flush: the consumer's run continues, and an end of stream
crossing here would end it. The adapter tests one bool before matching on the
packet, so an ordinary push pays a single short-circuiting branch for it. A real
`Eos` arriving from upstream mid-remove takes the arm's normal end-of-stream path
instead, unstripped, and propagates as it should, and the arm skips its post-`Eos`
wait on the allocation coordinator in that case, since the parked producer means no
other arm can end and the wait would never return.

### Replacing the two ends

A splice needs an edge on both sides of it, and the source and sink ends have only
one side each, so they are not splice positions. The element sitting on one is
still swappable.

`replace_sink` negotiates the new sink against the caps flowing on the edge, its
own sink constraint saying what shape it reads them as the way a mid-stream
re-solve does, parks the producer above it, and queues an `Eos` on the link it
takes from that producer. The old sink therefore renders everything still queued
for it and finalizes on its normal end-of-stream path, which is what a sink writing
a container needs before it is handed back. Only once its element has come back
does the replacement's arm start, on a fresh link the producer is then resumed
onto. The two never render the same stretch of stream, and the stall in between is
bounded by what was queued, exactly as for a remove.

`replace_source` is the transpose, with the node being replaced as the edge's
producer. The replacement is asked first for the shape already on the wire, which
needs no consent from anyone, and a source that cannot produce it picks one out of
the downstream feasibility set instead, and is refused (`DownstreamRefused`) when
that intersection is empty or there is no set, under the same consent rule a splice
follows. The old source is parked, its link taken, and then retired: a retired
endpoint resumes its producer onto the dead link it parked with, so the source's
next push fails and it leaves by the same path it takes when its consumer goes
away. Its arm reports no frames rather than that failure, since the swap is not a
run failure. What the old source had already queued stays on the link, which the
replacement inherits, and the one packet it was holding un-pushed is lost. Both
operations name the replacement afresh as `<category>N` and the name the old
element had is never handed out again.

A replacement source stamps from its own zero, and running time may not go
backwards, so the endpoint follows the timeline as well as the caps: the segment in
force and the last `DataFrame` PTS, with its duration, that crossed, updated per
push while a run is mutable and not at all otherwise. On the swap the mutator maps
that PTS through that segment and opens the replacement's stream with a segment
based at the result, so running time continues across the join while each source
keeps its own timestamps. An edge nothing has crossed yet continues from the base
of the segment in force. A caps change is announced ahead of that segment, where a
splice announces its own.

A replacement takes no part in the clock election or the latency fold, both settled
at startup: `provide_clock` is ignored and a slower replacement does not re-fold
the reported latency. It does receive what the election produced, so a replacement
sink is handed the same latency-folded `ClockSync` the startup sweep gave every
negotiated sink and paces presentation like the sink it took over from. A
replacement sink starts directly in the playing state, since there is no preroll for
it to take and no state transition to gate on. Neither takes part in the allocation
cascade, as a spliced element does not, and neither counts into `RunStats`, an arm
added mid-run landing past the per-arm bookkeeping. A wedged old sink, or a producer
that never reaches a packet boundary, defers the operation the way every other one is
deferred, and an end whose stream already carried its terminal `Eos` is
`GraphEnded`.

### Scope and addressing

The mutable position is a transform on a 1:1 edge carrying one stream. The
producing end is a source, a transform, or one output of a tee or demux, and the
consuming end is a transform, a sink, one input pad of a muxer or terminal fan-in,
or the single input of a tee or demux, where the splice feeds every branch at once
through the existing sticky-caps broadcast. `insert_after` on a fan-out node is
refused with `MutationError::NotMutable`, since which branch is ambiguous, as is a
source's upstream side or a sink's downstream side. Those two ends take
`replace_source` and `replace_sink` instead, and a terminal fan-in is neither, its
element not being the runner's to hand back.

Every node of a run carries an instance name, a plain broadcast tee included
(`tee0`, the same `<category>N` convention), which is what makes the edge above it
addressable and what labels it in topology dumps. Auto-generated names count past
every explicit `name=` in the graph, so one can never shadow a user's.

A removed or replaced element's node and edges leave the mutator's tables, and the
next element added takes their slots, as the next spliced arm takes a finished one's
slot in the run's arm set. A run mutated without end therefore keeps a flat heap
(`m1965_mutation_memory`). A name leaves with its node but is never handed out
again: the `<category>N` counter moves past it.

A refused remove leaves nothing behind. The drain flag and the output-link claim
are raised only once the producer's link is in hand: holding that link keeps the
element's input open, so its arm cannot yet have read either flag, and every
refusal before that point exits with the graph untouched. Raised earlier, a refusal
left an arm that swallowed its own end of stream and a claimed link holding the
consumer's channel open forever. The runner's own per-branch `Eos` behind a fan-out
element checks the port's terminal flag first, so an element that forwards `Eos`
itself does not put a second terminal behind it.

A structural edge needs no new gate. A tee, a demux and a muxer's producer all push
through the same `SenderSink`, so giving those edges an endpoint is all it takes,
and what changes is the addressing. The mutator's model is therefore a list of
edges rather than a property of each node, because neither end of an edge is unique
in general: a tee produces several and a muxer consumes several. An operation names
the end that is unique. `insert_after` takes the one edge below a node, which
addresses a producer feeding a muxer pad, and `insert_before` takes the one edge
above a node, which addresses a tee or demux branch by its consumer. A node with
several edges on the side asked for is `NotMutable` there and is addressed from the
other side. An element spliced onto a tee branch also inherits that branch's
reaction to an unsolvable mid-stream caps change, where a branch under
`AllowBranchDrop` drops rather than failing the run.

Every operation completes at the producer's next packet boundary, so a producer
that has gone quiet defers it, unless the handle carries a timeout.

### Chains, branches and inputs

`insert_chain_after` and `insert_chain_before` splice several elements in stream
order as one operation. The producer parks once, each element is negotiated against
what the one before it emits, and only the last one's output has to be accepted
below. Either every element goes in or none does, and the call returns the names in
order. `insert_after` and `insert_before` are the one-element case.

`add_branch` is the request-pad analog on a named tee. The branch is a list of
elements whose last one is the sink. It is negotiated against the caps the tee
broadcasts before the tee is touched, and the tee starts sending to it between two
packets. `remove_branch` names the branch by its sink. The tee stops feeding it at
its next packet boundary, what was queued drains through every element, and the
sink takes an end of stream so it finalizes before the elements come back. A new
branch sink starts playing and receives the elected clock, and like a replacement
sink it takes no part in the clock election, the latency fold or the allocation
cascade.

`add_input` attaches a source to the next free declared pad of a named muxer. The
muxer's arm negotiates the pad between two packets: the source's produce set against
the pad's constraint, then `accepts_runtime_input`, whose refusal comes back as
`Refused`. A muxer with every declared pad in use is `NoFreePad`. A launch line
builds a muxer with one pad per linked input, so only a muxer that declares a fixed
pad count, such as `cccombiner` fed only video, has a free pad there.

### Converters

A handle built `with_converters` splices converters behind an element whose output
the chain below would refuse, instead of refusing the operation. The handle holds a
`ConverterFactory`: given the caps the element produces and the set the edge
accepts, it returns the chain between them, or `None`. The element and its
converters go in as one splice. `Registry::converters` builds a factory from a
registry's autoplug search, at most `CONVERTER_MAX_DEPTH` (4) elements deep.
`g2g_plugins::registry::converters()` is that factory over `videoconvert` and
`audioconvert` alone, since the default registry's autoplug factories are parsers,
demuxers and decoders. `videoscale` is left out because it cannot be built from
search caps whose geometry is open.

A spliced element's output is worked out from its `caps_constraint_as_transform`
through `resolve_forward_output`, the same resolve a mid-stream re-solve uses, and
from `intercept_caps` only when the constraint gives no answer. A converter's
`intercept_caps` checks only its input, so without the constraint a spliced
`videoconvert format=I420` would claim to pass its input through.

### Timeouts

A handle built `with_timeout` gives up after that long waiting for a producer to
reach a packet boundary and answers `TimedOut` with the graph untouched. The bounded
wait is the one before anything moves. Once an old element is being ended the
operation completes regardless, since that step cannot be undone. A timed property
set whose arm has not taken it yet is withdrawn from the mailbox, so it is never
applied after the caller was told it failed. The deadline sleeps on the run's tick
timer, `with_ticker` or the clock's own `as_ticker`, and a run with neither answers
`NoTimer`.

The language bindings and the MCP server hand out `mutator_with_defaults` from
`g2g_plugins::registry`: a 5 s timeout and `converters()`. Their runs are on
`WallClock`, which is its own ticker. A pipeline a host registers with the MCP
server keeps the host's handle unchanged, since its run may have no timer.

### Reporting and sequence numbers

A run with a bus posts `BusMessage::GraphMutated` once per operation, property reads
excluded: the kind, the node it addressed, and the names it created or why it was
refused. The caller gets the same answer back, so the message is for whoever else
watches the topology.

A frame's sequence number keeps rising at the consumer across a splice. Each
endpoint records the last sequence it passed on. After a takeover, the endpoint
shifts the new producer's numbers only when they would otherwise repeat or go back,
so a replacement source that numbers from its own large base passes through as it
is.

### Setting a property while it runs

`set_property` and `get_property` name a node and a property, and the element's own
`set_property` / `get_property` runs inside the arm that owns that element, at that
arm's next packet boundary. The element is therefore only ever touched from the task
driving it, and a value takes effect between two packets rather than inside one. The
queue is one mailbox per node, an atomic flag over a small list, so an arm with
nothing waiting pays one acquire load per packet and allocates nothing.

Only a transform, a sink or a fan-in position carries a mailbox, the rule animated
properties follow: those are the arms that hand an element one packet at a time. A
source drives itself and a tee carries no element, so both are `NotMutable`. A
spliced element and a replacement sink get one of their own, so they are as settable
as a negotiated element. A value the element refuses comes back as
`PropertyRejected` carrying the element's own `PropError`, which is also how an
unknown property name is reported. A get of a name the element does not carry is
`Ok(None)`, again the element's answer rather than the mutator's. An arm that ended
before it reached a boundary is `GraphEnded`.

## GStreamer dynamic-feature mapping

g2g's dynamic surface is intended to be a superset of GStreamer's dynamic
capabilities, achieved through a different set of primitives.

| GStreamer feature | g2g mechanism |
| :--- | :--- |
| Element hot-swap | `ElementSlot::swap` (ArcSwap) |
| Element or chain insertion / removal | `GraphMutator::insert_chain_after` / `remove` |
| Tee request pad | `GraphMutator::add_branch` / `remove_branch` |
| Muxer request pad | `GraphMutator::add_input` |
| Branch enable / disable, A/B switching | `Router` + `Gate` |
| Bin nesting | `Graph::merge` with ghost pads, flattened when built |
| Mid-stream caps change | `PipelinePacket::CapsChanged` + runner cascade |
| Allocation pressure backtrack | Phase 3 `ConfigureOutcome::ReFixate` |
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

### Differences forced by Rust ownership

GStreamer relies on parent-child reference cycles via GObject reference counting
plus signal callbacks. Rust's strict ownership does not allow that shape.
Equivalent functionality lives in message channels instead of direct
back-references: a child element that needs to notify its parent posts a bus
message and the parent reads it. That is functionally identical, structurally
cleaner, and has no `unref` ordering hazards.

Similarly, GStreamer's `gst_pad_link()` performs runtime pointer manipulation,
while the g2g equivalent, moving the receive end of a channel, requires explicit
ownership transfer under a brief gate hold. Same outcome, more honest about what
is happening.

Some things fall out for free. There is no silent caps mismatch at runtime,
because the typed `Caps` enum is exhaustive and `match` is checked at compile
time, where GStreamer's string-keyed caps regularly fail at runtime with
`not-negotiated`. Shutdown is deterministic, since Rust drop order is a
topological walk and no leaked refs hold pipelines alive forever. There is no GIL
and no global state, so independent pipelines spawn on the same async runtime with
zero coordination cost. And memory safety survives a hot-swap, since ArcSwap
guarantees no use-after-free when an element is replaced while a frame is in
flight, where GStreamer's `pad_block` and `pad_unlink` choreography is famously
bug-prone.

### The single architectural trade-off

Pre-allocated dark slots handle the common dynamic-pad case, a demuxer with at
most N tracks. If an application genuinely needs a runtime-growable pad count
without an upper bound, such as a session router that accepts new RTP streams
indefinitely, the dynamic layer uses a `Slab<Slot>` instead of a fixed array, and
per-push slot lookup becomes one extra indirection. Since this only matters inside
the already-type-erased dynamic layer, the cost is in the noise.

The bounded-N realization is `StreamDemux` (`g2g-plugins`), a `MultiOutputElement`
with N typed output ports, driven by `run_source_fanout`. Each port carries its own
declared caps and is fed by a caller-supplied classifier (`Fn(&Frame) -> usize`),
and the first frame routed to a port emits that port's `CapsChanged` so the branch
retypes from the demuxer's byte-stream input caps to the elementary stream's, the
same announce a single-output demuxer does. The N branch links the runner
pre-allocates are the dark slots: a port no stream ever routes to simply stays
silent and takes the merged EOS at end. This is the multi-output demuxer, one
element with several typed downstream branches, where the other fan-out elements,
`Router` and `Gate`, only broadcast or A-B-switch a single caps. Container parsers,
MPEG-TS multi-PID, wire onto it by keying the classifier on parsed stream identity.

The demux is also a first-class DAG node, the symmetric counterpart to the muxer
fan-in. Rather than a new `NodeKind`, a demux reuses `NodeKind::Tee(n)` for the
structural and solver view, since it negotiates exactly like a tee at startup per
the dark-slot retyping above, and carries a `GraphNodeRef::Demux` payload that the
runner dispatches to `demux_arm`, the transpose of `muxer_arm`, instead of the
broadcast `tee_arm`. So the solver is unchanged and only the runtime behaviour
differs. `Graph::add_demux` builds the node and `DynMultiOutputElement` is the
dyn-safe mirror of `MultiOutputElement`. In `gst-launch`, a name registered via
`register_demux` with several outputs builds a demux
(`src ! d.  d. ! ...  d. ! ...  <demux> name=d`) instead of erroring
`FanOutWithoutTee`, the transpose of the muxer's link-degree rule. There is no
content-agnostic default demux in the registry, because routing is inherently
stream-specific as the muxer side ships specific muxers, so `register_demux` is the
surface.

### Batch frames: `streammux` and `streamdemux`

A batch carries N video frames as one packet, so one inference element can see
several streams at once. The model is gst-plugins-rs's `analyticscombiner` and
`analyticssplitter`. The link caps are `Caps::Batch { streams }`, one entry per
mux input in input order, printed as
`multistream/x-analytics-batch,streams=<"video/x-raw,...","video/x-raw,...">`. The
printed stream caps carry only fixed fields, as every printed caps does, so a ranged
stream does not survive the round trip through text. The carrier frame's own buffer
is empty. Its `BatchMeta` lists each contributing stream as a `BatchStream` with the
mux input `index`, the stream `caps` and the `frame`. No pixels are copied.

`streammux` is `BatchMux` in Rust, a `MultiInputElement` built with the input count
from link degree. It takes raw video in system memory, any format and geometry per
input. Frames queue in an `InputAggregator`, and every complete round, one frame from
each input still running, becomes one batch. An ended input drops out once its queue
drains. There is no timeout, so a late input holds the batch back. The batch takes
input 0's timing when input 0 contributes, else the earliest contributing frame's.
Segment and flush pass through from input 0 only.

`streamdemux` is `BatchDemux` in Rust, a `MultiOutputElement`. The launch names
differ from the Rust names because `StreamDemux` is already the content-routing demux.
It takes each frame out of the meta and pushes it to the port of its mux input. A
stream missing from a batch pushes nothing, and a frame some earlier consumer already
took fails with `G2gError::BatchFrameTaken`. Its port count must equal the stream
count, else configure fails with `G2gError::PortCountMismatch`. A registered demux's
branches are wired in link order and a `d.src_N` suffix does not pick the port, so
`d.src_0`, `d.src_1` have to be written in that order.

The startup solve runs before any element is configured, so neither element could
state its caps there. `MultiInputElement::output_batches_inputs` and
`MultiOutputElement::ports_unbatch` give the solver two node shapes instead,
`NodeConstraint::BatchMuxer` and `NodeConstraint::BatchDemux`. The mux output edge is
every batch of its input edges' alternatives, and each input edge narrows back to its
stream of the output. Each demux port edge is its stream of the input edge, and the
input narrows back to the batches whose streams the ports accept. So every branch
negotiates against its own stream, and a branch's demand reaches the source feeding
that stream. Past `MAX_BATCH_ALTERNATIVES` combinations the mux lists one batch
instead, each input's first alternative, which is the format and size its producer
prefers. A `videotestsrc` offers over a hundred, so two of them settle that way, and a
capsfilter on an input keeps the branch's choice. A demux whose input would exceed the
cap fails the solve with `NegotiationFailure::TooManyBatchAlternatives`.

A batch can be teed. `BatchMeta`'s copy-on-write clone shares each frame with
`Frame::share`, the duplication the tee uses. `streammux` makes each frame shareable
as it enters the batch, which moves owned and pooled bytes behind an `Arc` and copies
bytes lent through the C ABI once.

## Lifecycle: state machine, preroll and seek

The lifecycle spine sits on top of the DAG runner. It turns build, run to EOS and
drop into a controllable `NULL -> READY -> PAUSED -> PLAYING` machine that can
preroll, pause, scrub and resume.

`PipelineState` and `StateChangeReturn` are ungated core types. A
`StateController` (`runtime` feature) carries the target state and a sink-side flow
gate: below `PLAYING` a sink parks at the gate, stops draining its edge, and
backpressure stalls the DAG upstream, so the state machine reuses the existing
channel backpressure rather than a separate pause mechanism.

For preroll, a non-live `PAUSED` transition admits exactly one buffer per sink and
then holds. The runner calls `expect_prerolls(n)` and each sink's
`notify_prerolled` aggregates, so the async `PAUSED` completes with a single
`AsyncDone` once all sinks have prerolled. Live pipelines under `set_live(true)`
take the `NoPreroll` path, holding no frame. The lifecycle is opt-in via
`run_simple_pipeline_stateful` and `run_graph_stateful`, and the plain runners are
unchanged. `set_state(Null)` ends a running pipeline: each sink stops at its gate,
the arms upstream end on the closed links, and the run returns its stats without
posting `Eos`.

### Seek, segment and running time

`g2g-core::segment` is a pure-core, ungated model. `Seek`, `SeekType` and
`SeekFlags` describe the request, and `Segment` carries the rate- and
direction-aware running-time, stream-time and base-time math, the `GstSegment`
equivalent, with `clip`, `for_flush_seek` which resets `base` so running time
restarts after a flush, and `accumulate_seek`, the non-flushing seek where `base`
advances to the running time playback has already reached so the running-time line
stays monotonic across the seek, the gapless, segment-seek and loop case.

`PipelinePacket::Segment` is the carrier: the runner emits an opening SEGMENT and
every element forwards it, with transforms and decoders forwarding and sinks
consuming, the same way `Flush` already flows.

A `SeekController` (`runtime`) is a cloneable handle the application holds. A
seek-aware source's run loop polls `take_seek()` between frames and, on a
flushing seek, emits `Flush`, repositions, emits the post-flush `Segment`, and
resumes, so a seek reaches the source GStreamer-style, upstream, without a
back-reference.

`seek()` returns a `SeekTicket` that resolves to `Served`, `Refused` with a
`SeekRefusal`, or `Superseded` when a later seek replaced it before the element
took it. The element answers inside `take_seek`, through the check it passes:
`Ok` serves every seek, and `flushing_only` refuses a non-flushing one, which
`FileSrc`, `DownloadBuffer`, `HlsSrc`, `DashSrc` and the byte-stream demuxers pass.
A refused seek is logged and playback carries on, as a GStreamer seek returning
false does. `wait_for_seek` parks an element until a seek or `shutdown` arrives,
and `eos_then_wait_for_seek` ends a source's stream and parks it when the run
takes seeks after the end. `Mp4Src` is the first real repositioning source, with a flushing
seek, keyframe `SNAP_BEFORE` and re-prepended parameter sets, and `SyncSink` maps
PTS to running time through the `Segment` and clips pre-target frames so an accurate
seek presents the exact requested frame. A non-flushing seek emits only the
accumulating `Segment`, with no `Flush`, so the source keeps producing on a
continuous running-time line.

`HlsSrc` and `DashSrc` are time sources: a seek target is playlist time, or
presentation time across the Periods for DASH (a segment's time plus its Period's
`@start`), which the demuxer after them does not stamp. A DASH seek finds the
Period holding the target, then the segment. A TS segment's PTS carry the encoder's own start, 1.4 s for ffmpeg. So the
source sends its post-flush `Segment` with the first media it fetches after the
seek, once it has read the earliest PTS in it, with the same reader the demuxer uses
(`tsdemux::earliest_pts_ns`, `fmp4::earliest_fragment_pts_ns`). `start` is the
target moved onto those timestamps and `time` is the target itself, so the sink
clips the frames before the target and reports positions in playlist time. A
segment with no readable timestamp keeps the `Segment` on playlist time, with a
warning. At an HLS `#EXT-X-DISCONTINUITY` and a DASH Period boundary the source sends a `Segment` the same way, its
`base` the playlist time played since the first media out or the last seek, so
running time carries on over timestamps that jump. The source's `Segment` rides
the prebuffer window with the bytes it precedes. `tsdemux` keeps a `Segment` that
follows an upstream `Flush` or arrives once its stream is open, instead of opening
its own, while the runner's opening `Segment` it still replaces.
`attach_seek` skips a time source that feeds a buffering element (`qtdemux` behind
`hlssrc`), since that element holds the whole stream and serves the seek.

A seek on a live HLS or DASH stream has to land in the live window, gst
adaptivedemux's seek range (`adaptiveseek::LiveWindow`). For HLS the window runs
from the first listed segment to the end of the segment, or part, a fresh join
starts in. Playlist time 0 is the first segment of the first playlist loaded, and
each reload carries it over by media sequence. For DASH the window runs from now
less `timeShiftBufferDepth` to now less `maxSegmentDuration`, on presentation time
from `availabilityStartTime`, and a live MPD without one refuses every seek. A start
before the window moves onto it unless the seek is accurate, an `End` start counts
back from the window end, and a target still outside is refused with
`SeekRefusal::OutsideLiveWindow`, which `g2g-launch` reports before exiting.
Both sources also take a seek while a segment downloads
(`adaptiveseek::download_unless_seek`): a served seek drops the download, and a
refused one lets it finish, so no segment is skipped or sent twice.

Reverse playback (`Seek::reverse`, `rate < 0`) needs no sink-specific code. The
source emits frames newest-PTS-first over `[start, stop]`, and `SyncSink` schedules
each by `Segment::to_running_time`, which measures reverse from `stop`, and clips
via `contains`, so descending PTS maps to ascending running time and presents in the
correct visual order, the `Segment` abstraction generalizing the sink to negative
rate transparently.

The producing half is GOP-batched, since a decoder only runs forward. On a
`rate < 0` seek `Mp4Src` walks the container's sync samples backward from the
segment `stop`, the `stss` index `parse_progressive` recovers or the `trun` keyframe
flags of a fragmented file, and emits one whole GOP at a time in decode order,
newest GOP first, including the samples above `stop` that later frames reference,
which the sink clips. `GopReverse` (`gopreverse`) closes the loop after the decoder:
it buffers a decoded GOP, detects its end by the PTS jumping backward into the next,
earlier GOP, and re-emits each batch in descending PTS, so the sink receives reverse
presentation order. A forward segment passes straight through it, so it can sit in
any graph that may seek backward.

Trick-mode KEY_UNIT frame selection, presenting only keyframes for a fast scrub,
works through `FrameTiming::keyframe`, a per-frame flag set by `h264parse` from
`annexb::au_is_keyframe` and by `mp4src` and `fmp4demux` from the container sync-sample
or `trun` keyframe flag. A `TRICKMODE` seek sets `Segment::key_units_only` in
`from_seek`, and `SyncSink` drops non-keyframe frames under such a segment before
scheduling them, counted by `trick_dropped()`.

### Segment playback and looping

Segment playback, the `GstSeekFlags::SEGMENT` analog, is consumed through the
`SeekController` rather than a new packet, because g2g has no `SEGMENT_DONE`
`PipelinePacket` and adding one would force a new control variant through every
element's exhaustive match. The controller carries it on the same app-to-source
channel a seek already uses.

A `SEGMENT`-flagged seek runs the source to `stop`. Instead of `Eos` the source
calls `notify_segment_done(stop)` and parks on `wait_event` for the app's next move. The
app observes `segment_done_count()` and `take_segment_done()` and re-arms a
non-flushing `SEGMENT` seek to loop, so `accumulate_seek` advances `base` by one
span per iteration, gapless with no `Flush` downstream, or calls `shutdown()` to end
the loop, at which point the idle source emits `Eos`.

The idle park is wakeful (`SeekController::wait_event`): the source awaits a future
that resolves when `seek` or `shutdown` wakes the registered waker, so a looping
source between loops costs nothing, no busy-poll, the poll-free analog of GStreamer
pausing the source task. `Mp4Src` is the first real source to loop on `SEGMENT`: it
clips playback to the segment `stop`, reports segment-done at the boundary, and
parks on `wait_event` for the app's loop seek, non-flushing and snapping to the
keyframe at or before the target so a decoder resumes cleanly, or for `shutdown`. It
honours non-flushing repositioning seeks, an accumulating `Segment` with no `Flush`,
as well as flushing ones.

An application seeks a pipeline built from a launch string through
`runtime::attach_seek(graph, seek)`. A source reports `seek_format()`, `Time` for
`Mp4Src` and `Bytes` for `FileSrc`. A transform reports `seek_route()`: `Buffered`
when it serves a time seek from the whole stream it holds (`Mp4Demux`),
`UpstreamBytes` when it byte-seeks the source that feeds it (the byte-stream
demuxers, and the audio parsers `flacparse`, `mpegaudioparse`, `aacparse` and `ac3parse`,
which rescan a bare file from its start and restart their sample count there). The
MP3, AAC and AC-3 parsers also send the frames before the target that the decoder needs
first: one for the overlap with the previous frame, and for MP3 as many more as hold
the bit reservoir the frame may borrow from (511 bytes in MPEG-1, 255 in MPEG-2 and
2.5), counted from the sizes of the frames just read. An MP3 target counts from the
first sample the LAME gapless window keeps, as `ffmpeg -ss` does. `aacparse` refuses a
seek on a LOAS or raw AAC stream, which has no ADTS headers to count samples by. A `MultiOutputElement` reports the same, but the runner seeks no demux
node, so only `OnePortElement` reads it, for a demuxer run as its one-output
form. `attach_seek` finds the one node that can take the time seek, gives it a
controller with the seek already queued and a demuxer's byte source a second one,
so the seek lands before the first frame, and returns the seek's ticket. `attach_seek_controller` attaches the same
controllers with nothing queued, for a seek made while the pipeline runs, which
`g2g-launch --seek-at` issues once the sink reports a given position. After a seek every seekable source and
demuxer restarts at the keyframe at or before the target with a segment starting at
the target, so decoders drop the earlier frames and the first frame out is the first
one at or after the target, as `ffmpeg -ss` gives. Every seekable demuxer and source counts
the target from the stream's first timestamp, as ffmpeg's `-ss` counts from the start
time (gst counts from the first PCR in TS). `DemuxSeek` records the pts of the first
unit of any track, `flvdemux` lowers it to the earliest track's first pts as ffmpeg's FLV
start time is, while Matroska keeps the first unit since ffmpeg drops the AAC priming
from its start time there. A seek taken before that unit becomes a pts once it arrives, and an
index lookup (Matroska `Cues`, the Ogg guess) waits for it and rescans from the file
start until then. `tsdemux` holds every PID's units while a seek waits for the stream
start, until each PMT stream has shown a stamped unit (or EOS, or 5 MB, ffmpeg's
probesize), and counts from the earliest of those first stamps, so a file whose audio
starts before its video seeks as ffmpeg does. `Mp4Src` counts from the first sample, `Mp4Demux` from the later of
the edit-list start and the first sample. Positions use the same origin. Until a
seek, a unit presenting before the segment start or the stream start lowers both, as
does another FLV track's earlier first pts, and the demuxer sends the lowered segment again on every port that opened, as
`matroskademux` does. `tsdemux` compares by pts, so its opening at the first DTS stays.
`PsDemuxN` lowers its first-DTS base where gst's `mpegpsdemux` opens at the first SCR.
`Mp4DemuxN` lowers a port's start when a later fragment carries an earlier frame. A
seek's segment and the one kept across a source restart are never lowered. Every segment
a demuxer or `Mp4Src` sends has `time` set to its start minus the stream start, so the
sink arm's `Segment::to_stream_time` publishes a position that a seek takes back to the
same frame, as GStreamer's `matroskademux` does. `tsdemux` opens at the first DTS, so a
stream with B frames reads that much high until its first seek. A seek
past the end gives no frames, which g2g-launch reports as an error. `Mp4Demux`
holds the whole file, so a seek that arrives once its samples have gone out sends
`Flush` and the samples again from the target.

### Seeks after the end of the stream

A run handed the controller (`RunBuilder::with_seek_controller`) takes a flushing seek
after the end of the stream and plays on from the target, as a GStreamer pipeline does
after EOS. Only such a run changes, and only when the pipeline holds no muxer, no
fan-in and no sink that finalizes its output at `Eos` (`finalizes_on_eos`: `hlssink`,
`splitmuxsink`). There the run ends at the end of the stream as before, and a seek that
comes after the source returned is never taken, which g2g-launch reports. A
byte-stream demuxer that took such a seek and gets the source's `Eos` instead of its
`Flush` fails with `SeekRefused`. `filesink` and `wavsink` keep writing after a
`Flush`, so a seek after the end appends to their file.

In a run that takes seeks after the end, the controller's `takes_seeks_after_end()` is
set before any arm starts. `Mp4Src` and `FileSrc` push `Eos` at the end of the file and
park on `wait_event`: a seek pushes `Flush` and plays on, `shutdown` returns. A
byte-stream demuxer receives nothing while its source is parked, so after its `Eos` it
pushes the `Eos` itself and waits on the app controller inside `process`, then takes
the seek, which wakes the parked `FileSrc`. `Mp4Demux` waits the same way after its
drain. Every transform, sink, tee and demux arm from the seekable node down hands its
element the `Eos` and keeps reading, so a later `Flush` resets the chain, and still
ends when its input closes. `shutdown` on the app controller also shuts down the byte
source's controller, so the parked source returns and the chain ends through closed
links as before. An arm in that part of the graph that ends early, on an error or a
`Null` stop, shuts the controllers down too, so the parked source cannot hold the run
open.

The run ends the stream by counting. Every flushing seek taken from the app
controller counts, and the element that took it always pushes a `Flush` downstream,
`Mp4Demux` included for a seek it serves before its first sample. Each arm counts the
`Flush` packets it has received, and an `Eos` that reaches an arm with fewer than the
seeks taken is stale: the source sent it before the `Flush` of a seek already taken,
so the arm drops it, and a decoder never drains frames the `Flush` would discard. Each
sink downstream of the seekable node records its `Flush` count and whether an `Eos`
came after its latest `Flush`. A future the runner drives next to the arms sees every
such sink at a current `Eos` with no seek pending, marks `stream_ended()` on the
controller, and calls `shutdown` unless the application holds the run open
(`keep_run_open`, released by `let_run_end`). `g2g-launch --seek-at` holds it until it
has issued the seek, or until the stream ended short of the position, so a seek near
the end does not race the run's end.

### Re-preroll and byte-source seek

A paused, prerolled pipeline backpressures its source, so a flushing seek issued now
would never take effect, since the held sink never drains.
`StateController::request_repreroll`, called by the app alongside the seek, bumps a
preroll generation, and `flow_gate` takes the arm's generation and reopens for a
stale one, so each sink arm re-prerolls. The arm drains the stale pre-seek frames,
discarding rather than presenting, until the `Flush`, then prerolls the post-flush
target and re-fires `AsyncDone`, so scrubbing a paused pipeline updates the shown
frame.

`FileSrc` is BYTES-format seekable through `with_seek`: a flushing seek repositions
the file read to a byte offset and emits `Flush`.

A byte-stream demuxer, a transform with no random access, becomes seekable by
driving that upstream byte source. A shared `DemuxSeek` helper turns an app time
seek into an upstream byte-seek to offset 0, drops in-flight pre-seek input until
the returned `Flush`, resets the demuxer's parser, then keeps the decoded units
from the last keyframe at or before the target and, when the first unit with pts at
or after the target arrives, hands them back with a `Segment` starting at the
target. The keyframe test follows decode order and the release follows pts, so
reordered B-frames below the target still go out and the decoder's segment check
drops them. A run of more than 1024 units without a keyframe releases early under
the same segment. That is correct
for any container without an index, being a re-scan, with an index-derived offset a
later optimization.

All five demuxers carry it, `fmp4demux`, `tsdemux`, `mkvdemux`, `flvdemux` and
`oggdemux`, each using its own keyframe signal: the container flag, or
`annexb::au_is_keyframe` for TS whose units have none, while every audio packet is a
resync point and `oggdemux` accumulates an Opus PTS from the TOC byte. Where the
container has no index at all, `oggdemux` guesses the landing byte offset by
interpolating through observed `(byte offset, stream time)` anchors, clamped a
page-max below the byte length the source publishes on the seek controller
(`SeekController::set_stream_len`, from `FileSrc` by file size and from
`DownloadBuffer` once the spill is complete), so a guess through a front-dense file
cannot land at EOF.

The adaptive sources `HlsSrc` and `DashSrc` are TIME-seekable through `with_seek`.
Unlike the BYTES-format `FileSrc`, an app time seek resolves to the media segment
containing the target, HLS walking cumulative `#EXTINF` durations and DASH mapping
the target onto the `SegmentRef` `$Time$` line, then the source emits `Flush`, jumps
to that segment, re-emits the fMP4 init segment since the downstream demuxer reset
on the flush needs its `moov` again, emits the post-flush `Segment` at the segment
start, and resumes there. This is the CMAF and DASH segment-transition case, clamped
to the last segment, and a target past the end lands there.

## The bus

The pipeline `Bus` is a many-producer, single-consumer channel for out-of-band
events, so an element notifies the application without a back-reference.
`BusMessage` covers the lifecycle and quality signals an application reacts to.

- `StreamStart`, `Eos`, `Error`, `Warning`, `Info(String)`: stream lifecycle,
  faults, and non-fatal status. `StreamStart` is posted by the source arm before a
  source produces, one per source or fan-out source (`GST_MESSAGE_STREAM_START`), and `Eos` once
  by the runner when every arm has ended without error (`GST_MESSAGE_EOS`).
  `Info` is the third severity below `Warning`,
  element- or app-posted for status that is not a problem (`GST_MESSAGE_INFO`).
- `DurationChanged { duration_ns }`: the total stream duration became known, posted
  by the source arm from `SourceLoop::query_duration`
  (`GST_MESSAGE_DURATION_CHANGED`).
- `Tag { tags, program }`: container or stream metadata, posted out of band
  (`GST_MESSAGE_TAG`). `program` scopes the tags to one MPEG-TS `program_number`, an
  SDT service entry, so a multi-program multiplex reports each service separately,
  and is `None` for a container with a single metadata scope.
- `StreamTag { stream_id, tags }`: the same, scoped to one elementary stream, a
  Matroska `Tag` whose `Targets` names a `TagTrackUID`. `stream_id` is the id that
  stream has in the posted `StreamCollection`.
- `NegotiationFailed(NegotiationFailure)`: a structured caps conflict naming the
  responsible element pair, posted by the coordinator on a startup or mid-stream
  negotiation failure.
- `StateChanged { old, new }` and `AsyncDone`: every effective lifecycle
  transition, and the completion of an async `PAUSED` once preroll aggregates.
- `Qos { running_time_ns, jitter_ns, processed, dropped }`: a synchronizing sink,
  `SyncSink` or `WaylandSink`, that has fallen behind the clock drops a late frame
  and reports it, the `GST_MESSAGE_QOS` analog. The drop decision, count and post
  live in a shared `QosTracker` (`g2g-core::qos`), which also posts the running
  stats periodically (`with_qos_interval_ns`, on pipeline-clock cadence), so an app
  sees sink health without waiting for a drop.
- `Buffering { percent, element }`: a link's fill, 0 for underrun and 100 for full,
  posted on a quartile crossing via `run_graph_with_bus` by the sink and transform
  arms, tagged with the instance name of the element the link feeds, and
  self-posting prebuffer sources leave it `None`. Since g2g has no `queue` element,
  this reports the bounded link channel's own occupancy (`fill_percent`), the
  `GST_MESSAGE_BUFFERING` analog.
- `SegmentDone { position_ns }`: a `SeekFlags::SEGMENT` seek reached its `stop`
  (`GST_MESSAGE_SEGMENT_DONE`), posted by `SeekController::notify_segment_done` when
  the app attached a bus to the controller through `set_bus`. The take-once
  back-channel is unchanged, and this is the push side of the same event, so a
  looping app drives the next loop seek from the bus instead of polling.
- `StreamStatus { entered, thread_id }`: a streaming thread started or finished
  (`GST_MESSAGE_STREAM_STATUS` enter and leave), posted only by the thread-per-arm
  runner, one pair per spawned arm thread including the coordinator's, so an app
  sees the graph's real thread fan-out. `thread_id` hashes the OS `ThreadId`, which
  has no stable numeric form, so only equality is meaningful.
- `ClockLost`: the elected clock lost the reference it disciplines to
  (`GST_MESSAGE_CLOCK_LOST`).
- `SourceRestart`: a `fallbacksrc` life started, is being retried, or stopped
  ([live.md](live.md)).

Posting is non-blocking through `try_post`: a control message never stalls the data
path, and a full bus drops the report rather than applying backpressure.

## Logging

Element-granular logging (`g2g-core::log`) is the complementary diagnostic channel,
the `GST_DEBUG` analog, for developer tracing rather than application-facing events.

A record carries a `category`, the element type such as `"VideoFlip"` and the
filtering key, an optional `instance` name, the element instance such as
`"VideoFlip0"`, an optional `timestamp_ns` from a host-installed `set_time_source`
since core reads no clock, and typed structured `fields` a sink renders or ships
without parsing the message (`g2g_log_fields!`).

An element may override its category per instance, through `set_log_category`,
`LogSource::log_category_override`, or `log-category=` on a launch line, the second
launch keyword beside `name=`, and the override is what the filter matches. The auto
instance name stays type-based, so a filter knob never renumbers probes or `t.`
handles.

`LogLevel` runs `Error`, the most severe, through `Trace`, matching GStreamer's
numeric levels. A per-category threshold table, a default plus overrides, decides
what is emitted, mirrored into an atomic so a disabled `g2g_trace!` in a hot loop
costs one atomic load. The macros, `g2g_error!` through `g2g_trace!`, take a
`LogSource`, an element via `self` or a `Target` for logging about a named element,
then a `format_args!` message, checked against the threshold before formatting.

Records route to an installed `LogSink`. The `std` feature provides a stderr sink
and `init_from_env`, which reads `G2G_DEBUG`, a `GST_DEBUG`-style
`*:warning,VideoFlip:trace` spec where category names take `*` and `?` globs such as
`*sink*:5`, with an exact override winning.

The runners, DAG, bespoke linear and fan-in, assign each element an instance name
before negotiation through a shared `InstanceNamer`, muxer, demux and fan-out
payloads included: an explicit `gst-launch` `name=`, carried on the graph node with
duplicates rejected at parse, or else `<category>N`, the `videotestsrc0` convention,
via `set_instance_name`, logging each element's addition. The name also keys the
element's latency probe, and an element that logs about itself, implementing
`LogSource` with a stored name, carries that name in its lines.

It pulls no external logging crate, so it holds on the `no_std` baseline. The sink
is the RTOS plug-in point over UART or RTT, and the built-in `RingSink`, bounded,
overwriting oldest, with drain and snapshot, is the flight-recorder variant for
postmortem dumps there. The `tracing` feature adds a `LogSink` that forwards records
to the `tracing` crate, on the `g2g` target with `category` and `instance` as
fields, so a host on `tracing-subscriber`, OTLP or tokio-console receives g2g's logs
in its existing pipeline, and `log::init_tracing()` installs it and defers filtering
to the subscriber.

## Application queries

A media-player UI needs to poll where playback is and how long the stream is,
GStreamer's POSITION and DURATION queries. GStreamer sends a query object upstream
along the pads, while g2g pushes forward and composes paths statically, as with the
latency fold, so instead the runner publishes into a shared
`runtime::PipelineProgress` handle the application holds and polls, `position()` and
`duration()` in ns. This inverts the `SeekController` idiom: there the app writes a
pending seek and the source reads it, and here the runner writes and the app reads.

Position is published by the DAG runner's sink arm, mapping each consumed buffer's
PTS through the active segment to stream time, the sink being the position authority
exactly as a GStreamer sink answers from its segment plus last buffer, so it needs
no element cooperation. Duration is the source's answer:
`SourceLoop::query_duration() -> Option<u64>`, defaulting to `None` so a live source
stays unknown, polled by the source arm before producing, and `Mp4Src` reports it
from the `mdhd` box. A first duration also posts `BusMessage::DurationChanged` as a
push notification. `run_graph_with_progress` wires the handle in, and the handle is
plain atomics behind an `Arc`, so reading it from the app thread while the pipeline
runs needs no lock.

## Language bindings

The C ABI (`g2g-capi`), the Python bindings (`g2g-pyapi`) and the Kotlin and Swift
bindings (`g2g-uniffi`) all drive a pipeline through `g2g_plugins::launched::LaunchedPipeline`.
It parses a launch line, runs it on its own thread under a current-thread tokio
runtime, and turns each edit into a parse plus a `GraphMutator` call. The run thread
catches its own panic and stores the outcome itself, so `is_done` and `bus_poll`
answer from any thread while another thread blocks in `wait`. The C# and C++ wrappers
sit on the C ABI.

An Android app shows video through `surfacesink channel=<name>`. The app attaches the
`ANativeWindow` of its `Surface` under that name (`Surfaces.attach` in the Kotlin
bindings, a JNI call next to the UniFFI exports). The sink copies each RGBA frame into
the window and drops frames while nothing is attached. Sinks draw while holding the
lock on the attached windows, so `detach` returns only after the last draw, and an
app can call it from `surfaceDestroyed` before Android releases the window.

With `mediacodec-wgpu` the sink also takes the decoder's RGBA wgpu textures. On the
first one it builds a `wgpu::Surface` over the window from the `GpuContext` the frame
carries, sized to the window, and presents through `WgpuSink::with_surface`. That
presenter is stored in the window's entry ahead of the window, so it drops first. It
lives until `detach`, which keeps the last frame on screen after the run. A texture
from another device replaces it. `ANativeWindow_lock` fails once a wgpu
surface exists, so the first frame fixes the window to one kind, and a frame of the
other kind fails the run with `UnsupportedDomain`.
