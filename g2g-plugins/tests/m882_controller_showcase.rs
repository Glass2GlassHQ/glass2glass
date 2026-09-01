//! M882 showcase: animating a real element's property over stream time.
//!
//! A `parse_launch` line builds the compositor fan-in, then a `ControlProgram`
//! attached to the named node animates the overlay pad's `sink1-xpos`. The
//! assertions are on where the red overlay actually lands in each composited
//! frame, so a curve that sampled but never reached the blend fails the test.

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{
    parse_launch, run_graph, LaunchFactory, Registry, SourceFactory, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, ControlProgram, ControlSource, Dim, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::registry::default_registry;

const CANVAS: usize = 32;
const BLUE: [u8; 4] = [0, 0, 255, 255];
const RED: [u8; 4] = [255, 0, 0, 255];

/// Base-frame PTS spacing (1 ms), and the frames the base pad delivers.
const STEP_NS: u64 = 1_000_000;
const FRAMES: u64 = 4;

/// The animation: the overlay pad slides 12 px to the right over the run, so each
/// frame's PTS maps to a distinct on-canvas position (0, 4, 8, 12).
fn xpos_curve() -> ControlSource {
    ControlSource::linear([(0, 0.0), ((FRAMES - 1) * STEP_NS, 12.0)])
}

fn expected_xpos() -> Vec<usize> {
    vec![0, 4, 8, 12]
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn solid(w: u32, h: u32, color: [u8; 4], pts_ns: u64, sequence: u64) -> PipelinePacket {
    let mut buf = vec![0u8; (w * h) as usize * 4];
    for px in buf.as_chunks_mut::<4>().0 {
        px.copy_from_slice(&color);
    }
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        sequence,
    ))
}

/// Set once the overlay pad has delivered its frame. The base pad waits for it so
/// the compositor is primed before its own frames arrive: otherwise the element
/// buffers the first few and composites them in one later batch, under whichever
/// sample was last applied, and "one frame per position" would not hold.
fn overlay_sent() -> &'static AtomicBool {
    static SENT: OnceLock<AtomicBool> = OnceLock::new();
    SENT.get_or_init(AtomicBool::default)
}

/// Yields once so the sibling arms get polled.
struct YieldOnce(bool);

impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<()> {
        if self.0 {
            return std::task::Poll::Ready(());
        }
        self.0 = true;
        cx.waker().wake_by_ref();
        std::task::Poll::Pending
    }
}

/// The compositor's timing pad (pad 0): a 16x16 blue base, one frame per PTS step.
struct BaseSrc;

impl SourceLoop for BaseSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(rgba(16, 16)))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut spins = 0;
            while !overlay_sent().load(Ordering::Acquire) && spins < 10_000 {
                YieldOnce(false).await;
                spins += 1;
            }
            for i in 0..FRAMES {
                out.push(solid(16, 16, BLUE, i * STEP_NS, i)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAMES)
        })
    }
}

/// The overlay pad (pad 1): one 8x8 red frame at PTS 0, then EOS.
struct OverlaySrc;

impl SourceLoop for OverlaySrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(rgba(8, 8)))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(solid(8, 8, RED, 0, 0)).await?;
            overlay_sent().store(true, Ordering::Release);
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// The composited frames, process-global: a launch factory builds its element
/// from a plain `fn`, so the sink cannot carry a per-test handle.
fn composited() -> &'static Mutex<Vec<Box<[u8]>>> {
    static FRAMES: OnceLock<Mutex<Vec<Box<[u8]>>>> = OnceLock::new();
    FRAMES.get_or_init(Mutex::default)
}

struct CaptureSink;

impl AsyncElement for CaptureSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                if let Some(slice) = frame.domain.as_system_slice() {
                    composited().lock().unwrap().push(slice.into());
                }
            }
            Ok(())
        })
    }
}

fn test_registry() -> Registry {
    let mut reg = default_registry();
    reg.register_source(SourceFactory::new("basesrc", rgba(16, 16), || {
        Box::new(BaseSrc)
    }));
    reg.register_source(SourceFactory::new("overlaysrc", rgba(8, 8), || {
        Box::new(OverlaySrc)
    }));
    reg.register_launch(LaunchFactory::new("capturesink", Vec::new(), || {
        Box::new(CaptureSink)
    }));
    reg
}

const LINE: &str = "basesrc ! c.   overlaysrc ! c.   \
                    compositor name=c width=32 height=32 ! capturesink";

fn px(canvas: &[u8], x: usize, y: usize) -> [u8; 4] {
    let i = (y * CANVAS + x) * 4;
    [canvas[i], canvas[i + 1], canvas[i + 2], canvas[i + 3]]
}

// One test, because a launch factory can only reach its recorded frames through
// process-global state: two `#[tokio::test]`s would race over it.
#[tokio::test]
async fn an_animated_pad_property_moves_the_overlay_per_frame() {
    reject_a_property_the_compositor_does_not_have().await;

    let reg = test_registry();
    let mut graph = parse_launch(&reg, LINE).expect("the compositor fan-in parses");
    let compositor = graph
        .node_by_name("c")
        .expect("the launch line named the compositor");
    graph.set_node_control(
        compositor,
        ControlProgram::new().bind("sink1-xpos", xpos_curve()),
    );

    composited().lock().unwrap().clear();
    overlay_sent().store(false, Ordering::Release);
    let stats = run_graph(graph, &ZeroClock, 8)
        .await
        .expect("animated pipeline runs");
    assert_eq!(stats.frames_consumed, FRAMES, "one output per base frame");

    let frames = composited().lock().unwrap().clone();
    assert_eq!(frames.len() as u64, FRAMES);
    for (i, (canvas, x)) in frames.iter().zip(expected_xpos()).enumerate() {
        assert_eq!(canvas.len(), CANVAS * CANVAS * 4, "frame {i} canvas size");
        assert_eq!(
            px(canvas, x, 0),
            RED,
            "frame {i}: the overlay's left edge should have reached x={x}"
        );
        if x > 0 {
            assert_eq!(
                px(canvas, x - 1, 0),
                BLUE,
                "frame {i}: the overlay must have left x={} behind",
                x - 1
            );
        }
        // And its right edge moved with it (8 px wide, unscaled).
        assert_eq!(px(canvas, x + 7, 0), RED, "frame {i}: overlay right edge");
    }
}

/// The startup half: a name the element does not declare fails the run before any
/// frame flows, rather than animating nothing.
async fn reject_a_property_the_compositor_does_not_have() {
    let reg = test_registry();
    let mut graph = parse_launch(&reg, LINE).expect("the compositor fan-in parses");
    let compositor = graph.node_by_name("c").expect("named compositor");
    graph.set_node_control(
        compositor,
        ControlProgram::new().bind("sink1-nope", xpos_curve()),
    );

    composited().lock().unwrap().clear();
    overlay_sent().store(true, Ordering::Release);
    assert_eq!(
        run_graph(graph, &ZeroClock, 8).await,
        Err(G2gError::ControlBinding),
        "an unknown property name fails at startup"
    );
    assert!(
        composited().lock().unwrap().is_empty(),
        "no frame flowed before the failure"
    );
}
