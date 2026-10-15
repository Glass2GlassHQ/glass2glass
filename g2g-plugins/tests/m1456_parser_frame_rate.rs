#![cfg(all(feature = "std", any(feature = "dav1d", feature = "rav1d")))]

use core::future::Future;
use core::pin::Pin;
use std::sync::Mutex;

use g2g_core::runtime::{parse_launch, run_graph, LatencyProfile, LaunchFactory};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, OutputSink, PipelinePacket,
    Rate,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

// av1_321x181.obu stream-copied by `ffmpeg -framerate 24`
const FIXTURE_FPS: u32 = 24;
const FIXTURES: [&str; 2] = [
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/av1_321x181_24fps.mp4"
    ),
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/av1_321x181_24fps.mkv"
    ),
];

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
async fn decodebin_keeps_the_container_frame_rate_through_the_parser() {
    let mut registry = default_registry();
    registry.register_launch(LaunchFactory::new("capsprobe", Vec::new(), || {
        Box::new(CapsProbe)
    }));
    for fixture in FIXTURES {
        *LAST_CAPS_CHANGE.lock().unwrap() = None;
        let line = format!("filesrc location={fixture} ! decodebin ! capsprobe");
        let graph =
            parse_launch(&registry, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
        run_graph(graph, &WallClock::new(), LatencyProfile::Throughput)
            .await
            .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
        let last_caps = LAST_CAPS_CHANGE
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| panic!("the sink saw the decoded caps of {fixture}"));
        let Caps::RawVideo { framerate, .. } = last_caps else {
            panic!("raw video at the sink, got {last_caps:?}");
        };
        assert_eq!(framerate, Rate::Fixed(FIXTURE_FPS << 16), "{fixture}");
    }
}
