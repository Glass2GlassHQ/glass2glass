#![cfg(all(feature = "std", target_os = "linux", feature = "ffmpeg"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::Mutex;

use g2g_core::runtime::{parse_launch, run_graph, LatencyProfile, LaunchFactory};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, OutputSink,
    PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

// HEVC conformance stream whose SPS carries no VUI, so no frame rate
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/LTRPSPS_A_Qualcomm_1.bit"
);
const SCALED_WIDTH: u32 = 208;
const SCALED_HEIGHT: u32 = 120;

static LAST_CAPS_CHANGE: Mutex<Option<Caps>> = Mutex::new(None);

struct CapsProbe;

impl AsyncElement for CapsProbe {
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
        if let PipelinePacket::CapsChanged(caps) = packet {
            *LAST_CAPS_CHANGE.lock().unwrap() = Some(caps);
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn ffmpegdec_keeps_an_undeclared_framerate_through_converters() {
    let mut registry = default_registry();
    registry.register_launch(LaunchFactory::new("capsprobe", Vec::new(), || {
        Box::new(CapsProbe)
    }));
    let line = format!(
        "filesrc location={FIXTURE} ! h265parse ! ffmpegdec ! videoconvert ! videoscale \
         ! video/x-raw,format=RGBA,width={SCALED_WIDTH},height={SCALED_HEIGHT} ! capsprobe"
    );
    let graph = parse_launch(&registry, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = run_graph(graph, &WallClock::new(), LatencyProfile::Throughput)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    assert!(stats.frames_consumed > 0, "decoded frames reach the sink");

    let last_caps = LAST_CAPS_CHANGE
        .lock()
        .unwrap()
        .clone()
        .expect("the sink saw the decoded stream's caps");
    let Caps::RawVideo {
        format,
        width,
        height,
        framerate,
        ..
    } = last_caps
    else {
        panic!("raw video at the sink, got {last_caps:?}");
    };
    assert_eq!(format, RawVideoFormat::Rgba8);
    assert_eq!(
        (width, height),
        (Dim::Fixed(SCALED_WIDTH), Dim::Fixed(SCALED_HEIGHT))
    );
    assert_eq!(
        framerate,
        Rate::Any,
        "no frame rate invented for the stream"
    );
}
