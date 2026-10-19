#![cfg(all(
    feature = "std",
    feature = "srt",
    feature = "ffmpeg",
    target_os = "linux"
))]

use core::future::Future;
use core::pin::Pin;
use std::net::UdpSocket;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use g2g_core::runtime::{
    is_raw_audio, is_raw_video, parse_launch, run_graph, LatencyProfile, LaunchFactory,
};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, OutputSink, PipelinePacket,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
use ffmpeg_tool::have_ffmpeg;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av_h264_aac44100.ts"
);
// the listener binds before ffmpeg calls in
const LISTENER_START: Duration = Duration::from_millis(500);
const RUN_LIMIT: Duration = Duration::from_secs(30);
const POLL_PERIOD: Duration = Duration::from_millis(100);

static VIDEO_FRAMES: AtomicU64 = AtomicU64::new(0);
static AUDIO_FRAMES: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
struct KindCounter {
    counter: Option<&'static AtomicU64>,
}

impl AsyncElement for KindCounter {
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
        self.counter = if is_raw_video(absolute_caps) {
            Some(&VIDEO_FRAMES)
        } else if is_raw_audio(absolute_caps) {
            Some(&AUDIO_FRAMES)
        } else {
            None
        };
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let (PipelinePacket::DataFrame(_), Some(counter)) = (&packet, self.counter) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
        Box::pin(async { Ok(()) })
    }
}

struct KillOnDrop(Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_udp_port() -> u16 {
    UdpSocket::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

// an ffmpeg SRT caller is the peer, and the branches name only parsers, as a gst line does
#[tokio::test]
async fn tsdemux_behind_srtsrc_routes_each_branch_the_stream_its_parser_takes() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present; skipping the SRT A/V fan-out");
        return;
    }
    let port = free_udp_port();
    let mut registry = default_registry();
    registry.register_launch(LaunchFactory::new("kindcounter", Vec::new(), || {
        Box::new(KindCounter::default())
    }));
    let line = format!(
        "srtsrc address=127.0.0.1 port={port} ! tsdemux name=d \
         d. ! aacparse ! avdec_aac ! kindcounter \
         d. ! h264parse ! avdec_h264 ! kindcounter"
    );
    let graph = parse_launch(&registry, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let clock = WallClock::new();
    let run = run_graph(graph, &clock, LatencyProfile::Throughput);
    tokio::pin!(run);
    tokio::select! {
        result = &mut run => panic!("`{line}` ended before the caller connected: {result:?}"),
        _ = tokio::time::sleep(LISTENER_START) => {}
    }
    let _caller = KillOnDrop(
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-stream_loop",
                "-1",
                "-i",
                FIXTURE,
            ])
            .args(["-c", "copy", "-f", "mpegts"])
            .arg(format!("srt://127.0.0.1:{port}?mode=caller"))
            .stdin(Stdio::null())
            .spawn()
            .expect("spawn ffmpeg"),
    );
    let both_decoded = async {
        while VIDEO_FRAMES.load(Ordering::Relaxed) == 0 || AUDIO_FRAMES.load(Ordering::Relaxed) == 0
        {
            tokio::time::sleep(POLL_PERIOD).await;
        }
    };
    tokio::select! {
        result = &mut run => panic!("`{line}` ended before both branches decoded: {result:?}"),
        decoded = tokio::time::timeout(RUN_LIMIT, both_decoded) => decoded.expect("both branches decode"),
    }
    assert!(
        VIDEO_FRAMES.load(Ordering::Relaxed) > 0,
        "the h264parse branch decodes video"
    );
    assert!(
        AUDIO_FRAMES.load(Ordering::Relaxed) > 0,
        "the aacparse branch decodes audio"
    );
}
