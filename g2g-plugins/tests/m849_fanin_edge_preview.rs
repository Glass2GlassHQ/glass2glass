//! M849 preview taps on fan-in arms: an edge whose destination is a muxer /
//! session takes a [`LinkInterceptor`] through `Observer::edge_probe` exactly
//! like a linear edge, so the dashboard's edge preview works on a muxer input.
//! Covers the graph runner's muxer arm, the hand-built fan-in sink runner, and
//! the fan-in session runner (whose inputs share one tagged channel, so each
//! carries its own slot).
//!
//! std-gated (graph runner + plugin elements): `cargo test -p g2g-plugins`.
#![cfg(feature = "std")]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{oneshot, Notify};

use g2g_core::runtime::{
    run_fanin_session_observed, run_fanin_sink_observed, run_graph_observed, DynSourceLoop,
    GraphNode, LinkInterceptor, Observer, ProbeAction, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, Merger,
    MultiInputElement, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::inputselector::InputSelector;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// The sampling half of the dashboard's edge preview: records the payload size of
/// every `DataFrame` crossing the edge and passes it through untouched.
#[derive(Default)]
struct SamplingTap {
    frames: AtomicU64,
    last_bytes: AtomicU64,
}

impl SamplingTap {
    fn frames(&self) -> u64 {
        self.frames.load(Ordering::Relaxed)
    }
    fn last_bytes(&self) -> u64 {
        self.last_bytes.load(Ordering::Relaxed)
    }
}

impl LinkInterceptor for SamplingTap {
    fn on_packet(&self, packet: &PipelinePacket) -> ProbeAction {
        if let PipelinePacket::DataFrame(f) = packet {
            if let Some(bytes) = f.domain.as_system_slice() {
                self.frames.fetch_add(1, Ordering::Relaxed);
                self.last_bytes.store(bytes.len() as u64, Ordering::Relaxed);
            }
        }
        ProbeAction::Pass
    }
}

/// Sink that parks on its first frame until released, so a tap installed while it
/// is parked provably sees frames that cross afterwards.
struct GateSink {
    started: Option<oneshot::Sender<()>>,
    release: Arc<Notify>,
    parked: bool,
}

impl AsyncElement for GateSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let park = matches!(packet, PipelinePacket::DataFrame(_)) && !self.parked;
        if park {
            self.parked = true;
            if let Some(tx) = self.started.take() {
                let _ = tx.send(());
            }
        }
        let release = self.release.clone();
        Box::pin(async move {
            if park {
                release.notified().await;
            }
            Ok(())
        })
    }
}

/// Install `tap` on every edge that ends at node `to`, returning the edge ids.
fn tap_edges_into(obs: &Observer, to: usize, tap: &Arc<SamplingTap>) -> Vec<usize> {
    let snap = obs.snapshot();
    let mut ids = Vec::new();
    for (id, edge) in snap.edges.iter().enumerate() {
        if edge.to == to {
            obs.edge_probe(id).expect("edge slot").install(tap.clone());
            ids.push(id);
        }
    }
    ids
}

/// Graph runner: both arms into a muxer node accept a probe, and the tap samples
/// frames on the input the muxer does not even forward.
#[tokio::test]
async fn graph_muxer_input_edges_accept_a_preview_probe() {
    let release = Arc::new(Notify::new());
    let (started_tx, started_rx) = oneshot::channel();

    let mut g: Graph<GraphNode> = Graph::new();
    let a = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 16)));
    let b = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 16)));
    // input-selector forwards input 0 only, so the sink sees one monotonic stream.
    let mux = g.add_muxer(GraphNode::muxer(InputSelector::new(2)), 2);
    let sink = g.add_sink(GraphNode::element(GateSink {
        started: Some(started_tx),
        release: release.clone(),
        parked: false,
    }));
    g.link(a, mux.input(0)).unwrap();
    g.link(b, mux.input(1)).unwrap();
    g.link(mux.output(), sink).unwrap();

    let obs = Observer::new();
    let tap = Arc::new(SamplingTap::default());
    let watcher = {
        let obs = obs.clone();
        let tap = tap.clone();
        let release = release.clone();
        async move {
            if started_rx.await.is_err() {
                return Vec::new();
            }
            // The sink is parked, so the run cannot finish before the tap lands.
            let ids = tap_edges_into(&obs, mux.node().0 as usize, &tap);
            release.notify_one();
            ids
        }
    };

    let (stats, ids) = tokio::join!(run_graph_observed(g, &ZeroClock, 2, &obs, None), watcher);
    stats.expect("observed muxer run");
    assert_eq!(ids.len(), 2, "both fan-in arms exposed a probe slot");
    assert!(
        tap.frames() > 0,
        "the muxer input edges sampled frames through the tap"
    );
    assert_eq!(tap.last_bytes(), 8 * 8 * 4, "an 8x8 RGBA frame was sampled");
}

/// Hand-built fan-in sink runner: its input arms carry the same slot.
#[tokio::test]
async fn fanin_sink_input_edges_accept_a_preview_probe() {
    let release = Arc::new(Notify::new());
    let (started_tx, started_rx) = oneshot::channel();

    let mut a = VideoTestSrc::new(8, 8, 30, 16);
    let mut b = VideoTestSrc::new(8, 8, 30, 16);
    let mut merger = Merger::new(2); // forwards input 0
    let mut snk = GateSink {
        started: Some(started_tx),
        release: release.clone(),
        parked: false,
    };

    let obs = Observer::new();
    let tap = Arc::new(SamplingTap::default());
    let watcher = {
        let obs = obs.clone();
        let tap = tap.clone();
        let release = release.clone();
        async move {
            if started_rx.await.is_err() {
                return Vec::new();
            }
            // Node 2 is the merger (sources 0 and 1, then the sink).
            let ids = tap_edges_into(&obs, 2, &tap);
            release.notify_one();
            ids
        }
    };

    let run = async {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut a, &mut b];
        run_fanin_sink_observed(sources, &mut merger, &mut snk, &ZeroClock, 2, &obs).await
    };
    let (stats, ids) = tokio::join!(run, watcher);
    stats.expect("observed fan-in run");
    assert_eq!(ids, vec![0, 1], "both merger input edges are tappable");
    assert!(
        tap.frames() > 0,
        "frames were sampled on a merger input edge"
    );
    assert_eq!(tap.last_bytes(), 8 * 8 * 4);
}

fn session_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

/// Source pushing `n` 2x2 RGBA frames, then EOS, yielding between frames so a
/// concurrent watcher can install its tap mid-stream.
struct CountedSource {
    n: u64,
}

impl SourceLoop for CountedSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(session_caps()))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(session_caps()))
                .await?;
            for seq in 0..self.n {
                let frame = g2g_core::frame::Frame::new(
                    g2g_core::MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
                        vec![7u8; 2 * 2 * 4].into_boxed_slice(),
                    )),
                    g2g_core::FrameTiming::default(),
                    seq,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.n)
        })
    }
}

/// Terminal multi-input session: counts frames per pad, parks on the first one.
struct GateSession {
    started: Option<oneshot::Sender<()>>,
    release: Arc<Notify>,
    parked: bool,
    frames: Arc<Mutex<Vec<u64>>>,
}

impl MultiInputElement for GateSession {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        2
    }
    fn intercept_caps(&self, _i: usize, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, _i: usize, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(session_caps())
    }
    fn caps_constraint_as_input(&self, _i: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let mut park = false;
        if matches!(packet, PipelinePacket::DataFrame(_)) {
            self.frames.lock().unwrap()[input] += 1;
            if !self.parked {
                self.parked = true;
                park = true;
                if let Some(tx) = self.started.take() {
                    let _ = tx.send(());
                }
            }
        }
        let release = self.release.clone();
        Box::pin(async move {
            if park {
                release.notified().await;
            }
            Ok(())
        })
    }
}

/// Fan-in session runner: its inputs share one tagged channel, so before M849
/// they had no slot at all. Each input now carries its own, and a tap on one
/// samples only that input's frames.
#[tokio::test]
async fn fanin_session_input_edges_accept_a_preview_probe() {
    let release = Arc::new(Notify::new());
    let (started_tx, started_rx) = oneshot::channel();
    let frames = Arc::new(Mutex::new(vec![0u64, 0]));

    let mut a = CountedSource { n: 16 };
    let mut b = CountedSource { n: 16 };
    let mut session = GateSession {
        started: Some(started_tx),
        release: release.clone(),
        parked: false,
        frames: frames.clone(),
    };

    let obs = Observer::new();
    let tap = Arc::new(SamplingTap::default());
    let watcher = {
        let obs = obs.clone();
        let tap = tap.clone();
        let release = release.clone();
        async move {
            if started_rx.await.is_err() {
                return None;
            }
            // Tap input 1 only: the sample count must track that pad alone.
            obs.edge_probe(1)
                .expect("input 1 slot")
                .install(tap.clone());
            release.notify_one();
            obs.edge_caps(1)
        }
    };

    let run = async {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut a, &mut b];
        run_fanin_session_observed(sources, &mut session, &ZeroClock, 2, &obs).await
    };
    let (stats, caps) = tokio::join!(run, watcher);
    let stats = stats.expect("observed fan-in session run");
    assert_eq!(stats.frames_consumed, 32);
    assert_eq!(
        caps,
        Some(session_caps()),
        "the tapped edge reports its caps"
    );

    let sampled = tap.frames();
    assert!(sampled > 0, "frames were sampled on a session input edge");
    assert_eq!(tap.last_bytes(), 2 * 2 * 4);
    let per_pad = frames.lock().unwrap().clone();
    assert!(
        sampled <= per_pad[1],
        "the tap saw only input 1's frames ({sampled} of {})",
        per_pad[1]
    );
}
