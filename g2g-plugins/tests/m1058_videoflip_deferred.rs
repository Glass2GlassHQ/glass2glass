//! M1058: `VideoFlip` defers the rotation when the sink says it will apply the
//! descriptor itself.
//!
//! The sink advertises `Reconfigure::AbsorbOrientation` up its input link before
//! the first frame is pulled. `VideoFlip` answers it by passing the buffer
//! through with an `OrientationMeta` on it and announcing the *input* geometry,
//! so no pixel is ever remapped. A sink that does not advertise gets the old
//! eager behaviour, and a pass-through transform between the two relays the
//! advertisement instead of swallowing it.
//!
//! Needs the real `FrameMetaSet` (`metadata`) and the runners (`std`).
#![cfg(all(feature = "std", feature = "metadata"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::meta::{Orientation, OrientationMeta};
use g2g_core::runtime::{run_graph, run_source_transform_sink, GraphNodeRef};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, HardwareError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videoflip::VideoFlip;
use g2g_plugins::videotestsrc::{Pattern, VideoTestSrc};

const WIDTH: u32 = 8;
const HEIGHT: u32 = 4;
const FRAMES: u64 = 8;
const FRAMERATE: u32 = 30;
/// Two, so the arms really interleave. A link deep enough to hold the whole
/// stream lets the flip finish before the convert forwards anything, and the
/// relayed advertisement then arrives after the last frame.
const LINK_CAPACITY: usize = 2;
/// The one turn every graph here asks for.
const TURN: Orientation = Orientation::Rotate90Cw;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// `shared` hands each frame over as an `Arc`-backed `SystemView`, which is what
/// makes "the sink got the source's own buffer" checkable. `videoconvert` reads
/// contiguous bytes only, so the graph that puts one downstream asks for owned
/// frames instead.
fn source(shared: bool) -> VideoTestSrc {
    let src = VideoTestSrc::new(WIDTH, HEIGHT, FRAMERATE, FRAMES).with_pattern(Pattern::SmpteBars);
    if shared {
        src.with_shared_memory()
    } else {
        src
    }
}

fn rgba_caps(width: u32, height: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(FRAMERATE << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

/// Dense row-major bytes of `frame`, whichever system domain it arrived in.
fn frame_bytes(frame: &Frame) -> Vec<u8> {
    match &frame.domain {
        MemoryDomain::System(slice) => slice.as_slice().to_vec(),
        MemoryDomain::SystemView(view) => view.materialize().into_vec(),
        other => panic!("unexpected domain {other:?}"),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Received {
    bytes: Vec<u8>,
    orientation: Option<Orientation>,
    /// The caps in force when this frame arrived, i.e. the last `CapsChanged`
    /// the sink saw before it.
    caps: Caps,
}

#[derive(Debug, Default, Clone)]
struct Recording {
    frames: Vec<Received>,
    caps_changes: Vec<Caps>,
}

/// A sink that records what really reached it, and, when `absorbs` is set, tells
/// upstream it applies an `OrientationMeta` itself (what `WaylandSink` does with
/// `set_buffer_transform`).
///
/// Strict on purpose: it sizes its buffer from the caps it was configured with
/// and refuses a frame that does not match, so a missing or wrong mid-stream
/// `CapsChanged` fails the run instead of passing silently.
#[derive(Debug)]
struct RecordingSink {
    absorbs: bool,
    log: Arc<Mutex<Recording>>,
    caps: Option<Caps>,
}

impl RecordingSink {
    fn new(absorbs: bool) -> (Self, Arc<Mutex<Recording>>) {
        let log = Arc::new(Mutex::new(Recording::default()));
        (
            RecordingSink {
                absorbs,
                log: log.clone(),
                caps: None,
            },
            log,
        )
    }

    /// Bytes one frame must hold under the caps this sink last accepted.
    fn expected_bytes(&self) -> Result<usize, G2gError> {
        let Some(Caps::RawVideo {
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        }) = self.caps
        else {
            return Err(G2gError::NotConfigured);
        };
        Ok((w * h * 4) as usize)
    }
}

impl AsyncElement for RecordingSink {
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

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn absorbs_orientation(&self) -> bool {
        self.absorbs
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(caps) => {
                    self.caps = Some(caps.clone());
                    self.log.lock().unwrap().caps_changes.push(caps);
                }
                PipelinePacket::DataFrame(frame) => {
                    let bytes = frame_bytes(&frame);
                    if bytes.len() != self.expected_bytes()? {
                        // The announced geometry and the frame disagree.
                        return Err(G2gError::Hardware(HardwareError::Other));
                    }
                    let caps = self.caps.clone().ok_or(G2gError::NotConfigured)?;
                    self.log.lock().unwrap().frames.push(Received {
                        bytes,
                        orientation: frame
                            .meta
                            .get::<OrientationMeta>()
                            .map(|meta| meta.orientation),
                        caps,
                    });
                }
                _ => {}
            }
            Ok(())
        })
    }
}

/// `videotestsrc ! videoflip ! sink`, run on the graph runner.
async fn run_flip_to(absorbs: bool) -> Recording {
    let mut src = source(true);
    let mut flip = VideoFlip::new(TURN);
    let (mut sink, log) = RecordingSink::new(absorbs);

    let mut graph: Graph<GraphNodeRef<'_>> = Graph::new();
    let src_id = graph.add_source(GraphNodeRef::source_ref(&mut src));
    let flip_id = graph.add_transform(GraphNodeRef::element_ref(&mut flip));
    let sink_id = graph.add_sink(GraphNodeRef::element_ref(&mut sink));
    graph.link(src_id, flip_id).expect("src -> flip");
    graph.link(flip_id, sink_id).expect("flip -> sink");
    run_graph(graph, &NullClock, LINK_CAPACITY)
        .await
        .expect("graph runs");

    let recorded = log.lock().expect("recording lock").clone();
    recorded
}

/// `videotestsrc ! sink`: the pixels the source produced, untouched.
async fn run_source_only(shared: bool) -> Recording {
    let mut src = source(shared);
    let (mut sink, log) = RecordingSink::new(false);

    let mut graph: Graph<GraphNodeRef<'_>> = Graph::new();
    let src_id = graph.add_source(GraphNodeRef::source_ref(&mut src));
    let sink_id = graph.add_sink(GraphNodeRef::element_ref(&mut sink));
    graph.link(src_id, sink_id).expect("src -> sink");
    run_graph(graph, &NullClock, LINK_CAPACITY)
        .await
        .expect("graph runs");

    let recorded = log.lock().expect("recording lock").clone();
    recorded
}

#[tokio::test]
async fn an_absorbing_sink_gets_the_descriptor_and_the_untouched_pixels() {
    let source_frames = run_source_only(true).await;
    let deferred = run_flip_to(true).await;

    assert_eq!(
        deferred.frames.len(),
        FRAMES as usize,
        "every frame arrived"
    );
    assert_eq!(source_frames.frames.len(), FRAMES as usize);

    for (index, (got, reference)) in deferred
        .frames
        .iter()
        .zip(source_frames.frames.iter())
        .enumerate()
    {
        assert_eq!(
            got.orientation,
            Some(TURN),
            "frame {index} must carry the turn the flip did not apply"
        );
        assert_eq!(
            got.caps,
            rgba_caps(WIDTH, HEIGHT),
            "frame {index} arrived under the input geometry, not the rotated one"
        );
        assert_eq!(
            got.bytes, reference.bytes,
            "frame {index} reached the sink with the source's pixels, unrotated"
        );
    }

    assert!(
        deferred
            .caps_changes
            .iter()
            .all(|caps| *caps == rgba_caps(WIDTH, HEIGHT)),
        "the sink is never told about the rotated shape: {:?}",
        deferred.caps_changes
    );
}

#[tokio::test]
async fn a_sink_that_does_not_absorb_still_gets_the_rotation() {
    let source_frames = run_source_only(true).await;
    let eager = run_flip_to(false).await;

    assert_eq!(eager.frames.len(), FRAMES as usize);
    for (index, got) in eager.frames.iter().enumerate() {
        assert_eq!(got.orientation, None, "frame {index} carries no descriptor");
        assert_eq!(
            got.caps,
            rgba_caps(HEIGHT, WIDTH),
            "frame {index} arrived under the swapped geometry"
        );
        assert_ne!(
            got.bytes, source_frames.frames[index].bytes,
            "frame {index} was really rotated"
        );
    }
}

/// The eager and the deferred run have to describe the same picture: rotating
/// the deferred run's buffer by its own descriptor reproduces the eager run's
/// bytes. Without this the two paths could each be self-consistent and still
/// disagree about which way the picture turns.
#[tokio::test]
async fn the_descriptor_names_the_turn_the_eager_path_applies() {
    let deferred = run_flip_to(true).await;
    let eager = run_flip_to(false).await;

    for (index, (described, rotated)) in deferred.frames.iter().zip(eager.frames.iter()).enumerate()
    {
        assert_eq!(described.orientation, Some(TURN));
        assert_eq!(
            rotate90cw_rgba(&described.bytes, WIDTH, HEIGHT),
            rotated.bytes,
            "frame {index}"
        );
    }
}

/// A quarter turn clockwise of a packed RGBA image, written out here so the test
/// does not lean on the element it is checking.
fn rotate90cw_rgba(src: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let mut out = vec![0u8; src.len()];
    for out_y in 0..w {
        for out_x in 0..h {
            // The output pixel at (x, y) reads the input at (y, h - 1 - x).
            let src_index = ((h - 1 - out_x) * w + out_y) * 4;
            let dst_index = (out_y * h + out_x) * 4;
            out[dst_index..dst_index + 4].copy_from_slice(&src[src_index..src_index + 4]);
        }
    }
    out
}

#[tokio::test]
async fn the_advertisement_relays_through_a_pass_through_transform() {
    let source_frames = run_source_only(false).await;

    let mut src = source(false);
    let mut flip = VideoFlip::new(TURN);
    let mut convert = VideoConvert::new(RawVideoFormat::Rgba8);
    let (mut sink, log) = RecordingSink::new(true);

    let mut graph: Graph<GraphNodeRef<'_>> = Graph::new();
    let src_id = graph.add_source(GraphNodeRef::source_ref(&mut src));
    let flip_id = graph.add_transform(GraphNodeRef::element_ref(&mut flip));
    let convert_id = graph.add_transform(GraphNodeRef::element_ref(&mut convert));
    let sink_id = graph.add_sink(GraphNodeRef::element_ref(&mut sink));
    graph.link(src_id, flip_id).expect("src -> flip");
    graph.link(flip_id, convert_id).expect("flip -> convert");
    graph.link(convert_id, sink_id).expect("convert -> sink");
    run_graph(graph, &NullClock, LINK_CAPACITY)
        .await
        .expect("graph runs");

    let relayed = log.lock().expect("recording lock").clone();
    assert_eq!(relayed.frames.len(), FRAMES as usize);

    // The advertisement crosses one link per push, so the convert relays it
    // only once it has forwarded the flip's first packet: the frames before
    // that still arrive rotated. What matters is that it arrives at all.
    let deferred: Vec<&Received> = relayed
        .frames
        .iter()
        .filter(|f| f.orientation == Some(TURN))
        .collect();
    assert!(
        !deferred.is_empty(),
        "the advertisement never reached the flip through the convert: {:?}",
        relayed
            .frames
            .iter()
            .map(|f| f.orientation)
            .collect::<Vec<_>>()
    );
    for got in &deferred {
        assert_eq!(
            got.caps,
            rgba_caps(WIDTH, HEIGHT),
            "a deferred frame arrives under the input geometry"
        );
        assert_eq!(
            got.bytes, source_frames.frames[0].bytes,
            "a deferred frame is the source's pixels, unrotated"
        );
    }
    assert!(
        relayed.frames.last().expect("frames").orientation == Some(TURN),
        "once switched, the flip stays in descriptor mode"
    );
}

/// The same deferral on the linear (non-graph) runner, whose sink arm advertises
/// from its own call site.
#[tokio::test]
async fn the_linear_runner_advertises_too() {
    let mut src = source(true);
    let mut flip = VideoFlip::new(TURN);
    let (mut sink, log) = RecordingSink::new(true);

    run_source_transform_sink(&mut src, &mut flip, &mut sink, &NullClock, LINK_CAPACITY)
        .await
        .expect("source -> flip -> sink runs");

    let recorded = log.lock().unwrap();
    assert_eq!(recorded.frames.len(), FRAMES as usize);
    assert!(
        recorded
            .frames
            .iter()
            .all(|f| f.orientation == Some(TURN) && f.caps == rgba_caps(WIDTH, HEIGHT)),
        "every frame deferred: {:?}",
        recorded.frames.iter().map(|f| &f.caps).collect::<Vec<_>>()
    );
}
