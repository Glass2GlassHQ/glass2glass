//! M976: pull-based metadata requests. A consumer declares the meta types it
//! wants ([`AsyncElement::meta_requests`]); the declaration rides the allocation
//! cascade upstream, joined at every hop under each request's policy, so a producer
//! can ask whether anyone downstream reads a given meta before it produces one.
//!
//! The source here records the `AllocationParams` it absorbs, so a green run
//! proves the runner (not the elements) carried the demand across the hops.
//!
//! Needs the graph runner (std/runtime) and the real `MetaRequests` (metadata).
#![cfg(all(feature = "std", feature = "metadata", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::meta::{AnalyticsMeta, BlobMeta, CaptionMeta, MetaRequests, RequestPolicy};
use g2g_core::runtime::{block_on, run_graph, GraphNodeRef, SourceLoop};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph,
    MemoryDomain, MultiInputElement, OutputSink, PipelineClock, PipelinePacket, Rate,
    RawVideoFormat,
};

/// What a producer absorbed from the cascade, `None` when it was never called.
type Absorbed = Arc<Mutex<Option<AllocationParams>>>;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn frame(seq: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 2 * 2 * 4]))),
        FrameTiming::default(),
        seq,
    )
}

/// Source that records the allocation params it is handed, then emits one frame.
struct RecordingSource {
    absorbed: Absorbed,
}

impl SourceLoop for RecordingSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn configure_allocation(&mut self, params: &AllocationParams) {
        *self.absorbed.lock().unwrap() = Some(*params);
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::DataFrame(frame(0))).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// Pass-through transform. `requests` is its own declaration, so a test can also
/// prove an intermediate hop's demand joins what it forwards.
struct PassThrough {
    requests: MetaRequests,
}

impl AsyncElement for PassThrough {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn meta_requests(&self) -> MetaRequests {
        self.requests
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::Eos => {}
                other => out.push(other).await.map(|_| ())?,
            }
            Ok(())
        })
    }
}

/// Sink declaring `requests`, and optionally its own pool proposal so the two
/// can be seen to travel together.
struct RequestingSink {
    requests: MetaRequests,
    proposal: Option<AllocationParams>,
}

impl RequestingSink {
    fn wanting(requests: MetaRequests) -> Self {
        Self {
            requests,
            proposal: None,
        }
    }
}

impl AsyncElement for RequestingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn propose_allocation(&self, _caps: &Caps) -> Option<AllocationParams> {
        self.proposal
    }
    fn meta_requests(&self) -> MetaRequests {
        self.requests
    }
    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// One-input fan-in recording the params handed to its *output* side, standing
/// in for a fan-in producer (the GPU compositor) that decides what to attach to
/// the frames it writes.
struct RecordingFanIn {
    absorbed: Absorbed,
}

impl MultiInputElement for RecordingFanIn {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        1
    }
    fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::LegacySource(caps()))
    }
    fn configure_pipeline(
        &mut self,
        _input: usize,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(caps())
    }
    fn configure_allocation_for_output(&mut self, params: &AllocationParams) {
        *self.absorbed.lock().unwrap() = Some(*params);
    }
    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::Eos => {}
                other => out.push(other).await.map(|_| ())?,
            }
            Ok(())
        })
    }
}

/// source -> transform -> sink, returning what the source absorbed.
fn run_line(transform: MetaRequests, sink: RequestingSink) -> Option<AllocationParams> {
    let absorbed: Absorbed = Arc::new(Mutex::new(None));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(RecordingSource {
        absorbed: absorbed.clone(),
    }));
    let xform = g.add_transform(GraphNodeRef::element(PassThrough {
        requests: transform,
    }));
    let snk = g.add_sink(GraphNodeRef::element(sink));
    g.link(src, xform).unwrap();
    g.link(xform, snk).unwrap();
    block_on(run_graph(g, &ZeroClock, 4)).expect("graph runs");
    let out = *absorbed.lock().unwrap();
    out
}

#[test]
fn a_sinks_request_reaches_the_source_two_hops_upstream() {
    let params = run_line(
        MetaRequests::new(),
        RequestingSink::wanting(MetaRequests::new().request::<AnalyticsMeta>()),
    )
    .expect("the demand alone reaches the source");
    assert!(
        params.meta_requests.wants::<AnalyticsMeta>(),
        "the sink's request crossed the transform"
    );
    assert!(
        !params.meta_requests.wants::<BlobMeta>(),
        "only what was asked for"
    );
}

#[test]
fn no_request_leaves_the_cascade_untouched() {
    // Nothing declared anywhere: the source is never handed params at all, i.e.
    // the mechanism is inert for a graph that opts out (every graph today).
    let params = run_line(
        MetaRequests::new(),
        RequestingSink::wanting(MetaRequests::new()),
    );
    assert_eq!(params, None);
}

#[test]
fn a_request_rides_alongside_a_pool_proposal() {
    // A sink that wants both buffers and metadata: the pool parameters reach the
    // source unchanged, with the demand carried on the same proposal.
    let absorbed: Absorbed = Arc::new(Mutex::new(None));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(RecordingSource {
        absorbed: absorbed.clone(),
    }));
    let snk = g.add_sink(GraphNodeRef::element(RequestingSink {
        requests: MetaRequests::new().request::<CaptionMeta>(),
        proposal: Some(AllocationParams::system(4096, 3)),
    }));
    g.link(src, snk).unwrap();
    block_on(run_graph(g, &ZeroClock, 4)).expect("graph runs");

    let params = absorbed
        .lock()
        .unwrap()
        .expect("proposal reaches the source");
    assert_eq!((params.size_bytes, params.min_buffers), (4096, 3));
    assert!(params.meta_requests.wants::<CaptionMeta>());
}

#[test]
fn an_intermediate_hop_adds_its_own_request() {
    let params = run_line(
        MetaRequests::new().request::<BlobMeta>(),
        RequestingSink::wanting(MetaRequests::new().request::<AnalyticsMeta>()),
    )
    .expect("demand reaches the source");
    assert!(params.meta_requests.wants::<AnalyticsMeta>(), "sink's");
    assert!(params.meta_requests.wants::<BlobMeta>(), "transform's own");
    assert_eq!(params.meta_requests.len(), 2);
}

#[test]
fn two_sinks_additive_requests_union_at_the_tee() {
    // Both branches read the one producer's frames, so it must satisfy both.
    let absorbed: Absorbed = Arc::new(Mutex::new(None));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(RecordingSource {
        absorbed: absorbed.clone(),
    }));
    let tee = g.add_tee(2);
    let a = g.add_sink(GraphNodeRef::element(RequestingSink::wanting(
        MetaRequests::new().request::<AnalyticsMeta>(),
    )));
    let b = g.add_sink(GraphNodeRef::element(RequestingSink::wanting(
        MetaRequests::new().request::<CaptionMeta>(),
    )));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), a).unwrap();
    g.link(tee.out(1), b).unwrap();
    block_on(run_graph(g, &ZeroClock, 4)).expect("graph runs");

    let params = absorbed
        .lock()
        .unwrap()
        .expect("joined demand reaches the source");
    assert!(params.meta_requests.wants::<AnalyticsMeta>());
    assert!(params.meta_requests.wants::<CaptionMeta>());
}

#[test]
fn demand_crosses_a_fan_in_producers_output() {
    // A fan-in writes its own output frames, so downstream demand has to reach
    // it even though its pool parameters do not cross the boundary.
    let absorbed: Absorbed = Arc::new(Mutex::new(None));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(RecordingSource {
        absorbed: Arc::new(Mutex::new(None)),
    }));
    let mux = g.add_muxer(
        GraphNodeRef::muxer(RecordingFanIn {
            absorbed: absorbed.clone(),
        }),
        1,
    );
    let snk = g.add_sink(GraphNodeRef::element(RequestingSink::wanting(
        MetaRequests::new().request::<AnalyticsMeta>(),
    )));
    g.link(src, mux.input(0)).unwrap();
    g.link(mux.output(), snk).unwrap();
    block_on(run_graph(g, &ZeroClock, 4)).expect("graph runs");

    let params = absorbed
        .lock()
        .unwrap()
        .expect("the fan-in is told what downstream wants");
    assert!(params.meta_requests.wants::<AnalyticsMeta>());
}

#[test]
fn a_full_request_set_drops_the_overflow_without_losing_the_rest() {
    // Capacity is a fixed 4; the fifth request is dropped, which costs an
    // optimization, never correctness.
    let base = MetaRequests::new()
        .request::<AnalyticsMeta>()
        .request::<BlobMeta>()
        .request::<CaptionMeta>()
        .request::<g2g_core::meta::HdrStaticMeta>();
    assert_eq!(base.len(), 4);
    let full = base.request::<g2g_core::meta::TimecodeMeta>();
    assert_eq!(full.len(), 4);
    assert!(full.wants::<AnalyticsMeta>());
    assert!(!full.wants::<g2g_core::meta::TimecodeMeta>());
}

#[test]
fn a_request_set_is_order_independent() {
    // The cascade compares params for equality to suppress a re-propose, so two
    // sets holding the same types must be equal whatever order they were built.
    let a = MetaRequests::new()
        .request::<AnalyticsMeta>()
        .request::<CaptionMeta>();
    let b = MetaRequests::new()
        .request::<CaptionMeta>()
        .request::<AnalyticsMeta>();
    assert_eq!(a, b);
    assert_eq!(a.join_branches(b), a, "joining equal sets is idempotent");
}

#[test]
fn the_stricter_policy_wins_when_two_elements_ask_differently() {
    // One element is happy either way and asks under the loose policy; another
    // would misread the meta's effect. The strict one is the one that can be
    // misread, so it has to stand.
    let loose = MetaRequests::new().request::<BlobMeta>();
    let strict = MetaRequests::new().request_from_every_consumer::<BlobMeta>();
    assert_eq!(loose.policy::<BlobMeta>(), Some(RequestPolicy::AnyConsumer));
    assert_eq!(
        strict.carry_upstream(loose).policy::<BlobMeta>(),
        Some(RequestPolicy::EveryConsumer)
    );
    assert_eq!(
        loose.carry_upstream(strict).policy::<BlobMeta>(),
        Some(RequestPolicy::EveryConsumer)
    );
}

/// source -> tee -> {sink asking for `asked`, sink asking for nothing}, returning
/// what the source absorbed.
fn run_tee_with_one_silent_branch(asked: MetaRequests) -> Option<AllocationParams> {
    let absorbed: Absorbed = Arc::new(Mutex::new(None));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(RecordingSource {
        absorbed: absorbed.clone(),
    }));
    let tee = g.add_tee(2);
    let asking = g.add_sink(GraphNodeRef::element(RequestingSink::wanting(asked)));
    let silent = g.add_sink(GraphNodeRef::element(RequestingSink::wanting(
        MetaRequests::new(),
    )));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), asking).unwrap();
    g.link(tee.out(1), silent).unwrap();
    block_on(run_graph(g, &ZeroClock, 4)).expect("graph runs");
    let out = *absorbed.lock().unwrap();
    out
}

#[test]
fn a_silent_branch_vetoes_a_demand_every_consumer_must_share() {
    // Honouring such a request changes the frames themselves, and the silent
    // branch reads those frames too: the producer must never be told to change
    // them. The demand dies at the tee, and with nothing else riding on it the
    // source is left exactly as it was.
    let params = run_tee_with_one_silent_branch(
        MetaRequests::new().request_from_every_consumer::<BlobMeta>(),
    );
    assert_eq!(params, None, "no demand reaches the producer");
}

#[test]
fn a_silent_branch_does_not_veto_an_additive_demand() {
    // Attaching analytics costs the silent branch nothing, so one asking branch
    // is enough.
    let params = run_tee_with_one_silent_branch(MetaRequests::new().request::<AnalyticsMeta>())
        .expect("the demand reaches the producer");
    assert!(params.meta_requests.wants::<AnalyticsMeta>());
}

#[test]
fn both_branches_asking_carries_an_every_consumer_demand_through() {
    let absorbed: Absorbed = Arc::new(Mutex::new(None));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(RecordingSource {
        absorbed: absorbed.clone(),
    }));
    let tee = g.add_tee(2);
    let strict = || MetaRequests::new().request_from_every_consumer::<BlobMeta>();
    let a = g.add_sink(GraphNodeRef::element(RequestingSink::wanting(strict())));
    let b = g.add_sink(GraphNodeRef::element(RequestingSink::wanting(strict())));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), a).unwrap();
    g.link(tee.out(1), b).unwrap();
    block_on(run_graph(g, &ZeroClock, 4)).expect("graph runs");

    let params = absorbed
        .lock()
        .unwrap()
        .expect("both branches asked, so the demand stands");
    assert!(params.meta_requests.wants::<BlobMeta>());
}

#[test]
fn a_hop_that_does_not_share_the_demand_vetoes_it_too() {
    // The producer's frames pass through the transform before they reach the
    // sink, so a transform that does not handle the changed frames kills the
    // demand exactly as a silent tee branch does.
    let strict = || MetaRequests::new().request_from_every_consumer::<BlobMeta>();
    let through = run_line(MetaRequests::new(), RequestingSink::wanting(strict()));
    assert_eq!(through, None, "the demand died at the transform");

    // The same sink behind a transform that does handle them: it travels.
    let carried = run_line(strict(), RequestingSink::wanting(strict()))
        .expect("both hops handle the changed frames");
    assert!(carried.meta_requests.wants::<BlobMeta>());
}
