//! M876: the compositor as a `gst-launch` element. A text pipeline builds the
//! fan-in by link degree and configures the whole composite through properties:
//! the canvas geometry and fill, and the per-pad placement gst expresses as
//! request-pad properties (`sink_1::xpos`), flattened here to `sinkN-xpos`.
//! The assertions are on the composited pixels, so a property that parsed but did
//! not reach the blend fails the test.

use core::future::Future;
use core::pin::Pin;
use std::sync::{Mutex, OnceLock};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph, LaunchFactory, ParseError, Registry, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::registry::default_registry;

const CANVAS: usize = 32;
const BLUE: [u8; 4] = [0, 0, 255, 255];
const RED: [u8; 4] = [255, 0, 0, 255];
/// The `background-color=` value below, 0xAARRGGBB opaque green, as pixels.
const GREEN: [u8; 4] = [0, 255, 0, 255];

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
    }
}

/// Emits two solid-colour RGBA frames, then EOS.
struct ColorSrc {
    w: u32,
    h: u32,
    color: [u8; 4],
}

impl SourceLoop for ColorSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(rgba(self.w, self.h)))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for sequence in 0..2u64 {
                let mut buf = vec![0u8; (self.w * self.h) as usize * 4];
                for px in buf.as_chunks_mut::<4>().0 {
                    px.copy_from_slice(&self.color);
                }
                out.push(PipelinePacket::DataFrame(Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
                    Default::default(),
                    sequence,
                )))
                .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(2)
        })
    }
}

/// The composited frames, in a process-global cell: a launch factory builds its
/// element from a plain `fn`, so the sink cannot carry a per-test handle.
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

/// A registry with the two solid-colour sources and the capturing sink this test
/// composites: a 16x16 blue base and an 8x8 red overlay.
fn test_registry() -> Registry {
    let mut reg = default_registry();
    reg.register_source(g2g_core::runtime::SourceFactory::new(
        "bluesrc",
        rgba(16, 16),
        || {
            Box::new(ColorSrc {
                w: 16,
                h: 16,
                color: BLUE,
            })
        },
    ));
    reg.register_source(g2g_core::runtime::SourceFactory::new(
        "redsrc",
        rgba(8, 8),
        || {
            Box::new(ColorSrc {
                w: 8,
                h: 8,
                color: RED,
            })
        },
    ));
    reg.register_launch(LaunchFactory::new("capturesink", Vec::new(), || {
        Box::new(CaptureSink)
    }));
    reg
}

fn px(canvas: &[u8], x: usize, y: usize) -> [u8; 4] {
    let i = (y * CANVAS + x) * 4;
    [canvas[i], canvas[i + 1], canvas[i + 2], canvas[i + 3]]
}

#[tokio::test]
async fn a_launch_line_places_every_pad_on_the_canvas() {
    let reg = test_registry();
    // 32x32 opaque-green canvas; the blue base covers its top-left 16x16, and the
    // 8x8 red overlay is scaled to a 16x16 inset at (4,4) above it.
    let graph = parse_launch(
        &reg,
        "bluesrc ! c.   redsrc ! c.   \
         compositor name=c width=32 height=32 background-color=4278255360 \
         sink1-xpos=4 sink1-ypos=4 sink1-zorder=1 sink1-width=16 sink1-height=16 \
         ! capturesink",
    )
    .expect("the compositor fan-in parses");

    composited().lock().unwrap().clear();
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs");
    assert_eq!(stats.frames_consumed, 2, "one output per base frame");

    let frames = composited().lock().unwrap().clone();
    let canvas = frames.last().expect("a composited frame");
    assert_eq!(
        canvas.len(),
        CANVAS * CANVAS * 4,
        "width= / height= sized the canvas"
    );
    assert_eq!(px(canvas, 0, 0), BLUE, "the base covers its own 16x16");
    assert_eq!(px(canvas, 2, 2), BLUE, "outside the inset, still the base");
    assert_eq!(
        px(canvas, 4, 4),
        RED,
        "sink1-xpos / -ypos placed the overlay"
    );
    assert_eq!(
        px(canvas, 19, 19),
        RED,
        "sink1-width / -height scaled it to 16x16"
    );
    assert_eq!(
        px(canvas, 20, 20),
        GREEN,
        "past the inset and the base: background-color"
    );
    assert_eq!(px(canvas, 24, 24), GREEN, "uncovered canvas");
}

#[test]
fn a_pad_the_compositor_does_not_have_is_rejected() {
    let reg = test_registry();
    // Two branches, so pads 0 and 1 exist. `sink4-xpos` parses (it is a declared
    // name) but cannot apply, and a placement that silently vanished would be
    // worse than a failed pipeline.
    let err = parse_launch(
        &reg,
        "bluesrc ! c.   redsrc ! c.   compositor name=c sink4-xpos=4 ! capturesink",
    )
    .unwrap_err();
    assert_eq!(
        err,
        ParseError::BadValue {
            element: "compositor".into(),
            key: "sink4-xpos".into(),
            value: "4".into(),
        }
    );
}

#[test]
fn the_gpu_compositor_is_registered_under_its_own_name() {
    let reg = test_registry();
    // Registration only: building it opens a device, which the run test above
    // does not need and a CI host may not have.
    let line = "bluesrc ! c.   redsrc ! c.   compositor name=c ! capturesink";
    assert!(parse_launch(&reg, line).is_ok());
    #[cfg(feature = "wgpu-sink")]
    assert!(
        parse_launch(&reg, &line.replace("compositor", "wgpucompositor")).is_ok(),
        "wgpucompositor is a launch element too"
    );
}
