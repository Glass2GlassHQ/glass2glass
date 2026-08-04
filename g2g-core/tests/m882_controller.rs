//! M882: animated properties (the `gst-controller` analog).
//!
//! A [`ControlProgram`] attached to a graph node makes its properties functions of
//! stream time: the runner samples every binding at each frame's PTS and sets it
//! before the element processes that frame. The element here records the values it
//! held per frame, so a sample that never reached `set_property` fails the test.
//!
//! The startup cases cover the loud half: a program is checked against the
//! element's own property table before any frame flows, so a misspelled name, a
//! non-animatable kind, an empty curve, or a node whose arm has no per-frame hook
//! ends the run with `ControlBinding` instead of animating nothing.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::property::{PropError, PropKind, PropValue, PropertySpec};
use g2g_core::runtime::{block_on, run_graph, GraphNodeRef, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, ControlProgram, ControlSource, Dim, Frame, FrameTiming,
    G2gError, Graph, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

/// Frame PTS spacing, so a curve keyed in these units lands on frame times.
const STEP_NS: u64 = 50;

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
    }
}

/// Emits `count` frames at PTS `0, STEP_NS, 2*STEP_NS, ...`, then EOS.
struct PacedSrc {
    count: u64,
}

impl SourceLoop for PacedSrc {
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

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for i in 0..self.count {
                out.push(PipelinePacket::DataFrame(Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 16]))),
                    FrameTiming {
                        pts_ns: i * STEP_NS,
                        ..FrameTiming::default()
                    },
                    i,
                )))
                .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.count)
        })
    }
}

/// The property values in force when a frame was processed.
#[derive(Debug, Clone, PartialEq)]
struct Held {
    pts_ns: u64,
    offset: i64,
    level: u64,
    enabled: bool,
    ratio: f64,
}

static PROPS: &[PropertySpec] = &[
    PropertySpec::new("offset", PropKind::Int, "signed, may go negative"),
    PropertySpec::new("level", PropKind::Uint, "unsigned"),
    PropertySpec::new("enabled", PropKind::Bool, "on/off"),
    PropertySpec::new("ratio", PropKind::Double, "no conversion needed"),
    PropertySpec::new("strict", PropKind::Int, "refuses anything above 100"),
    PropertySpec::new("label", PropKind::Str, "not animatable"),
];

/// Records the property values it holds as each frame arrives, so the test reads
/// the animation the runner actually applied. Shared through an `Arc` so the same
/// element works in a borrowing graph and an owning (thread-per-arm) one.
struct ScriptedSink {
    log: Arc<Mutex<Vec<Held>>>,
    offset: i64,
    level: u64,
    enabled: bool,
    ratio: f64,
}

impl ScriptedSink {
    fn new(log: Arc<Mutex<Vec<Held>>>) -> Self {
        Self {
            log,
            offset: 0,
            level: 0,
            enabled: false,
            ratio: 0.0,
        }
    }
}

impl AsyncElement for ScriptedSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "offset" => self.offset = value.as_int().ok_or(PropError::Type)?,
            "level" => self.level = value.as_uint().ok_or(PropError::Type)?,
            "enabled" => self.enabled = value.as_bool().ok_or(PropError::Type)?,
            "ratio" => self.ratio = value.as_double().ok_or(PropError::Type)?,
            // The range check a real element does; the controller must not be
            // able to walk a property past what the element accepts unnoticed.
            "strict" => match value.as_int().ok_or(PropError::Type)? {
                v if v <= 100 => {}
                _ => return Err(PropError::Value),
            },
            "label" => {}
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::DataFrame(frame) = packet {
            self.log.lock().unwrap().push(Held {
                pts_ns: frame.timing.pts_ns,
                offset: self.offset,
                level: self.level,
                enabled: self.enabled,
                ratio: self.ratio,
            });
        }
        Box::pin(core::future::ready(Ok(())))
    }
}

/// Runs `PacedSrc(frames) -> ScriptedSink`, with `program` attached to the sink
/// (or, when `on_source`, to the source, which no arm can sample).
fn run_controlled(
    frames: u64,
    program: ControlProgram,
    on_source: bool,
) -> (Result<(), G2gError>, Vec<Held>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut src = PacedSrc { count: frames };
    let mut sink = ScriptedSink::new(log.clone());

    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    let s = g.add_source(GraphNodeRef::source_ref(&mut src));
    let k = g.add_sink(GraphNodeRef::element_ref(&mut sink));
    g.link(s, k).expect("link source to sink");
    g.set_node_control(if on_source { s } else { k }, program);

    let outcome = block_on(run_graph(g, &ZeroClock, 4)).map(|_| ());
    let held = log.lock().unwrap().clone();
    (outcome, held)
}

/// The program the value tests share: one curve per animatable kind.
fn every_kind() -> ControlProgram {
    ControlProgram::new()
        .bind("offset", ControlSource::linear([(0, 0.0), (100, 10.0)]))
        // Starts below zero, so a `Uint` has to clamp rather than wrap.
        .bind("level", ControlSource::linear([(0, -5.0), (100, 5.0)]))
        .bind("enabled", ControlSource::step([(0, 0.0), (100, 1.0)]))
        .bind("ratio", ControlSource::linear([(0, 0.25), (100, 0.75)]))
}

#[test]
fn each_frame_is_processed_under_the_values_its_pts_calls_for() {
    let (outcome, held) = run_controlled(4, every_kind(), false);
    outcome.expect("controlled run");

    let expected = vec![
        Held {
            pts_ns: 0,
            offset: 0,
            level: 0, // -5.0 clamped
            enabled: false,
            ratio: 0.25,
        },
        Held {
            pts_ns: 50,
            offset: 5,
            level: 0, // 0.0 at the midpoint
            enabled: false,
            ratio: 0.5,
        },
        Held {
            pts_ns: 100,
            offset: 10,
            level: 5,
            enabled: true,
            ratio: 0.75,
        },
        // Past the last keyframe: every curve holds its end value.
        Held {
            pts_ns: 150,
            offset: 10,
            level: 5,
            enabled: true,
            ratio: 0.75,
        },
    ];
    assert_eq!(held, expected);
}

#[test]
fn an_unknown_property_fails_the_run_before_any_frame_flows() {
    let program = ControlProgram::new().bind("nosuchknob", ControlSource::step([(0, 1.0)]));
    let (outcome, held) = run_controlled(4, program, false);
    assert_eq!(outcome, Err(G2gError::ControlBinding));
    assert!(
        held.is_empty(),
        "the run must fail at startup, not after animating some frames"
    );
}

#[test]
fn a_property_with_no_number_to_animate_is_rejected() {
    let program = ControlProgram::new().bind("label", ControlSource::step([(0, 1.0)]));
    let (outcome, _) = run_controlled(4, program, false);
    assert_eq!(
        outcome,
        Err(G2gError::ControlBinding),
        "a string property has no interpolation"
    );
}

#[test]
fn a_curve_with_no_keyframes_is_rejected() {
    let program = ControlProgram::new().bind("offset", ControlSource::linear([]));
    let (outcome, _) = run_controlled(4, program, false);
    assert_eq!(outcome, Err(G2gError::ControlBinding));
}

#[test]
fn a_node_whose_arm_has_no_per_frame_hook_is_rejected() {
    // A source drives itself, so the runner has no point at which to sample its
    // properties; attaching a program there is a mistake, not a no-op.
    let (outcome, held) = run_controlled(4, every_kind(), true);
    assert_eq!(outcome, Err(G2gError::ControlBinding));
    assert!(held.is_empty());
}

#[test]
fn a_sampled_value_the_element_refuses_fails_the_run() {
    // `strict` accepts up to 100; the curve walks past it at pts 100.
    let program =
        ControlProgram::new().bind("strict", ControlSource::linear([(0, 0.0), (100, 500.0)]));
    let (outcome, held) = run_controlled(4, program, false);
    assert_eq!(outcome, Err(G2gError::ControlBinding));
    assert!(
        !held.is_empty() && held.len() < 4,
        "the early frames flowed and the out-of-range sample stopped the run, got {} frames",
        held.len()
    );
}

/// The same animation under the thread-per-arm runner, which owns its elements:
/// the resolved controller is owned data, so it rides the arm's builder closure
/// onto the worker thread.
#[cfg(feature = "multi-thread")]
#[test]
fn the_threaded_runner_animates_the_same_way() {
    use g2g_core::runtime::{run_graph_threaded, GraphNode, ThreadSpawner};

    let log = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let s = g.add_source(GraphNode::source(PacedSrc { count: 3 }));
    let k = g.add_sink(GraphNode::element(ScriptedSink::new(log.clone())));
    g.link(s, k).expect("link source to sink");
    g.set_node_control(k, every_kind());

    block_on(run_graph_threaded(g, &ZeroClock, 4, &ThreadSpawner)).expect("threaded run");

    let held = log.lock().unwrap().clone();
    assert_eq!(
        held.iter().map(|h| h.offset).collect::<Vec<_>>(),
        vec![0, 5, 10],
        "the curve was sampled per frame on the arm's own thread"
    );
    assert_eq!(
        held.iter().map(|h| h.enabled).collect::<Vec<_>>(),
        vec![false, false, true]
    );
}
