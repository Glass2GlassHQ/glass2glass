#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{block_on, run_graph, run_simple_pipeline, GraphNodeRef, SourceLoop};
use g2g_core::{
    graph::Graph, AsyncElement, Caps, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

// the frame the user closes the window on, late enough that the run is under way
const CLOSED_AT: u64 = 3;

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
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    }
}

// a live camera: it never ends its stream by itself
struct EndlessSource;

impl SourceLoop for EndlessSource {
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
            for sequence in 0.. {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 2 * 2 * 4]))),
                    FrameTiming::default(),
                    sequence,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            unreachable!("the source counts frames forever")
        })
    }
}

struct EosWatchingTransform {
    saw_eos: Arc<AtomicBool>,
}

impl AsyncElement for EosWatchingTransform {
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
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if matches!(packet, PipelinePacket::Eos) {
                self.saw_eos.store(true, Ordering::SeqCst);
            }
            out.push(packet).await?;
            Ok(())
        })
    }
}

struct WindowSink;

impl AsyncElement for WindowSink {
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
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) if frame.sequence >= CLOSED_AT => {
                    Err(G2gError::WindowClosed)
                }
                _ => Ok(()),
            }
        })
    }
}

#[test]
fn closing_the_window_drains_a_graph_run_from_its_live_source() {
    let saw_eos = Arc::new(AtomicBool::new(false));
    let mut graph: Graph<GraphNodeRef<'static>> = Graph::new();
    let source = graph.add_source(GraphNodeRef::source(EndlessSource));
    let transform = graph.add_transform(GraphNodeRef::element(EosWatchingTransform {
        saw_eos: saw_eos.clone(),
    }));
    let sink = graph.add_sink(GraphNodeRef::element(WindowSink));
    graph.link(source, transform).unwrap();
    graph.link(transform, sink).unwrap();

    let stats = block_on(run_graph(graph, &ZeroClock, 1)).expect("the run ends cleanly");
    assert!(
        saw_eos.load(Ordering::SeqCst),
        "the source's Eos drained the transform"
    );
    assert!(stats.frames_emitted > CLOSED_AT, "{stats:?}");
}

#[test]
fn closing_the_window_drains_a_linear_run() {
    let mut source = EndlessSource;
    let mut sink = WindowSink;
    let stats = block_on(run_simple_pipeline(&mut source, &mut sink, &ZeroClock, 1))
        .expect("the run ends cleanly");
    assert!(stats.frames_emitted > CLOSED_AT, "{stats:?}");
}

struct SignallingWindowSink {
    closed: Arc<AtomicBool>,
}

impl AsyncElement for SignallingWindowSink {
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
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) if frame.sequence >= CLOSED_AT => {
                    self.closed.store(true, Ordering::SeqCst);
                    Err(G2gError::WindowClosed)
                }
                _ => Ok(()),
            }
        })
    }
}

struct HoldingSink {
    klass: &'static str,
    window_closed: Arc<AtomicBool>,
    frames_rendered: Arc<AtomicU64>,
}

impl AsyncElement for HoldingSink {
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
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new("holding sink", self.klass, "holds its first frame", "test")
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !matches!(packet, PipelinePacket::DataFrame(_)) {
                return Ok(());
            }
            core::future::poll_fn(|cx| {
                if self.window_closed.load(Ordering::SeqCst) {
                    return core::task::Poll::Ready(());
                }
                cx.waker().wake_by_ref();
                core::task::Poll::Pending
            })
            .await;
            self.frames_rendered.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}

fn frames_rendered_after_closing_the_window(klass: &'static str) -> u64 {
    let closed = Arc::new(AtomicBool::new(false));
    let frames_rendered = Arc::new(AtomicU64::new(0));
    let mut graph: Graph<GraphNodeRef<'static>> = Graph::new();
    let window_source = graph.add_source(GraphNodeRef::source(EndlessSource));
    let window_sink = graph.add_sink(GraphNodeRef::element(SignallingWindowSink {
        closed: closed.clone(),
    }));
    graph.link(window_source, window_sink).unwrap();
    let other_source = graph.add_source(GraphNodeRef::source(EndlessSource));
    let other_sink = graph.add_sink(GraphNodeRef::element(HoldingSink {
        klass,
        window_closed: closed,
        frames_rendered: frames_rendered.clone(),
    }));
    graph.link(other_source, other_sink).unwrap();
    block_on(run_graph(graph, &ZeroClock, 4)).expect("the run ends cleanly");
    frames_rendered.load(Ordering::SeqCst)
}

#[test]
fn closing_the_window_drops_the_frames_queued_for_an_audio_sink() {
    assert_eq!(frames_rendered_after_closing_the_window("Sink/Audio"), 1);
}

#[test]
fn closing_the_window_still_drains_a_file_sink() {
    assert!(frames_rendered_after_closing_the_window("Sink/File") > 1);
}
