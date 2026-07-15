//! Shows WebVTT subtitles over a `videotestsrc` SMPTE pattern, the Rust form of:
//!
//!   videotestsrc ! textoverlay ! videoconvert ! waylandsink   (live window)
//!   videotestsrc ! textoverlay ! <capture>                    (still image)
//!
//! Two run modes:
//!
//!   # Live: open a Wayland window (Fedora, inside a Wayland session)
//!   cargo run -p g2g-plugins --features wayland-sink --example textoverlay_demo
//!   cargo run -p g2g-plugins --features wayland-sink --example textoverlay_demo -- 600
//!
//!   # Still: write one rendered frame as a PPM (no display needed)
//!   cargo run -p g2g-plugins --features std --example textoverlay_demo -- /tmp/out.ppm
//!
//! The WebVTT track exercises the placement settings (a top banner, a default
//! bottom caption, and a left-aligned note pinned to mid-height); all three cues
//! run the whole clip, so overlapping cues show at once on every frame.

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// Frame geometry from optional `WIDTHxHEIGHT` argument (`arg`), else the
/// default. Lets the live demo run small enough that a debug build outpaces the
/// framerate, so PTS pacing is the throttle rather than CPU. Only the live
/// (`wayland-sink`) path parses geometry; the still path uses the default size.
#[cfg(feature = "wayland-sink")]
fn geometry(arg: Option<String>) -> (u32, u32) {
    arg.and_then(|s| {
        let (w, h) = s.split_once('x')?;
        Some((w.parse().ok()?, h.parse().ok()?))
    })
    .unwrap_or((WIDTH, HEIGHT))
}

/// A WebVTT track using the placement settings. All three cues run the whole
/// clip, so they overlap on every frame.
const SUBS: &str = "\
WEBVTT

00:00:00.000 --> 01:00:00.000 line:6% position:50% align:center
LIVE - CAM 01

00:00:00.000 --> 01:00:00.000
GLASS TO GLASS SUBTITLES

00:00:00.000 --> 01:00:00.000 line:46% position:0% align:start
SPEAKER ONE:
NICE TO MEET YOU";

#[cfg(feature = "wayland-sink")]
fn main() {
    use g2g_core::runtime::{run_graph, GraphNodeRef};
    use g2g_core::{graph::Graph, Bus, BusMessage, RawVideoFormat};
    use g2g_plugins::clock::WallClock;
    use g2g_plugins::textoverlay::TextOverlay;
    use g2g_plugins::videoconvert::VideoConvert;
    use g2g_plugins::videotestsrc::{Pattern, VideoTestSrc};
    use g2g_plugins::waylandsink::WaylandSink;

    let frames: u64 = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(300);
    let (w, h) = geometry(std::env::args().nth(2));
    // Optional QoS bound, in ms: a frame later than this past its deadline is
    // dropped (and a Qos report posted) instead of presented late.
    let max_late_ms: Option<u64> = std::env::args().nth(3).and_then(|a| a.parse().ok());

    // QoS bus: the sink posts a Qos message per late-drop; we drain it after.
    let (bus, handle) = Bus::new(256);
    let mut sink = WaylandSink::new().with_title("g2g textoverlay demo");
    if let Some(ms) = max_late_ms {
        sink = sink.with_max_lateness_ns(ms * 1_000_000).with_bus(handle);
    }

    // videotestsrc(SMPTE) -> textoverlay(WebVTT) -> videoconvert(NV12) -> waylandsink.
    // WaylandSink consumes NV12, so the convert sits between the RGBA8 overlay
    // and the sink.
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(
        VideoTestSrc::new(w, h, 30, frames).with_pattern(Pattern::SmpteBars),
    ));
    let overlay = g.add_transform(GraphNodeRef::element(TextOverlay::from_webvtt(SUBS)));
    let convert = g.add_transform(GraphNodeRef::element(VideoConvert::new(RawVideoFormat::Nv12)));
    let sink = g.add_sink(GraphNodeRef::element(sink));
    g.link(src, overlay).expect("link src->overlay");
    g.link(overlay, convert).expect("link overlay->convert");
    g.link(convert, sink).expect("link convert->sink");

    let clock = WallClock::new();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("tokio rt");
    match max_late_ms {
        Some(ms) => println!("playing {frames} frames of {w}x{h} (QoS: drop if >{ms} ms late)..."),
        None => println!("playing {frames} frames of {w}x{h} with subtitles..."),
    }
    let consumed = match rt.block_on(run_graph(g, &clock, 4)) {
        Ok(stats) => stats.frames_consumed,
        Err(e) => {
            eprintln!("pipeline error: {e:?}");
            return;
        }
    };

    // Drain QoS reports (the sink posted one per late-drop).
    let mut qos_drops = 0u64;
    while let Some(msg) = bus.try_recv() {
        if let BusMessage::Qos { jitter_ns, dropped, .. } = msg {
            qos_drops = dropped;
            if dropped == 1 || dropped % 30 == 0 {
                println!("  QoS: dropped late frame ({:.1} ms behind)", jitter_ns as f64 / 1e6);
            }
        }
    }
    if max_late_ms.is_some() {
        println!(
            "done: {consumed} frames reached the sink, {qos_drops} QoS-dropped, {} presented",
            consumed - qos_drops
        );
    } else {
        println!("done: {consumed} frames presented");
    }
}

#[cfg(all(feature = "std", not(feature = "wayland-sink")))]
fn main() {
    still::run();
}

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("build with --features std (still image) or --features wayland-sink (live window)");
}

/// Still-image path: render one frame to a PPM, no display required. Used to
/// preview the overlay headless.
#[cfg(all(feature = "std", not(feature = "wayland-sink")))]
mod still {
    use core::future::Future;
    use core::pin::Pin;

    use g2g_core::runtime::run_source_transform_sink;
    use g2g_core::{
        AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, MemoryDomain,
        OutputSink, PipelinePacket,
    };
    use g2g_plugins::clock::WallClock;
    use g2g_plugins::textoverlay::TextOverlay;
    use g2g_plugins::videotestsrc::{Pattern, VideoTestSrc};

    /// Sink that keeps the last RGBA8 frame and its geometry.
    #[derive(Default)]
    struct CaptureSink {
        width: u32,
        height: u32,
        last: Option<Vec<u8>>,
    }

    impl AsyncElement for CaptureSink {
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
        fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
            if let Caps::RawVideo { width: Dim::Fixed(w), height: Dim::Fixed(h), .. } = caps {
                self.width = *w;
                self.height = *h;
            }
            Ok(ConfigureOutcome::Accepted)
        }
        fn process<'a>(
            &'a mut self,
            packet: PipelinePacket,
            _out: &'a mut dyn OutputSink,
        ) -> Self::ProcessFuture<'a> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(frame) = packet {
                    if let MemoryDomain::System(slice) = &frame.domain {
                        self.last = Some(slice.as_slice().to_vec());
                    }
                }
                Ok(())
            })
        }
    }

    fn write_ppm(path: &str, rgba: &[u8], w: u32, h: u32) -> std::io::Result<()> {
        use std::io::Write;
        let mut out = Vec::with_capacity((w * h * 3) as usize + 32);
        write!(out, "P6\n{w} {h}\n255\n")?;
        for px in rgba.chunks_exact(4) {
            out.extend_from_slice(&px[..3]);
        }
        std::fs::write(path, out)
    }

    pub(super) fn run() {
        let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/textoverlay_demo.ppm".into());

        let mut src =
            VideoTestSrc::new(super::WIDTH, super::HEIGHT, 30, 3).with_pattern(Pattern::SmpteBars);
        let mut overlay = TextOverlay::from_webvtt(super::SUBS);
        let mut sink = CaptureSink::default();
        let clock = WallClock::new();

        let rt =
            tokio::runtime::Builder::new_current_thread().enable_time().build().expect("tokio rt");
        let stats = rt
            .block_on(run_source_transform_sink(&mut src, &mut overlay, &mut sink, &clock, 4))
            .expect("pipeline runs");

        let frame = sink.last.expect("a frame was captured");
        write_ppm(&path, &frame, sink.width, sink.height).expect("write ppm");
        println!(
            "rendered {} frames ({} cues) -> {path} ({}x{})",
            stats.frames_consumed,
            overlay.cue_count(),
            sink.width,
            sink.height
        );
    }
}
