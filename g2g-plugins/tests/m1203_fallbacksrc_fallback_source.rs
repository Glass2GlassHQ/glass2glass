#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use g2g_core::error::HardwareError;
use g2g_core::runtime::{
    parse_launch, run_graph, FallbacksrcSourceFactory, Registry, SourceLoop,
    FALLBACK_FALLBACK_SOURCE_SUFFIX,
};
use g2g_core::{Caps, ConfigureOutcome, G2gError, OutputSink, PipelineClock};
use g2g_plugins::appsink::register_appsink_pull;
use g2g_plugins::fallbacksrc::RETRY_DELAY;
use g2g_plugins::registry::default_registry;
use g2g_plugins::videotestsrc::{Pattern, VideoTestSrc};

mod fallback_fanout_common;

type DynSource = Box<dyn g2g_core::runtime::DynSourceLoop>;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

const MAIN_WIDTH: u32 = 16;
const MAIN_HEIGHT: u32 = 16;

const FALLBACK_WIDTH: u32 = 32;
const FALLBACK_HEIGHT: u32 = 32;

// distinct from the other sizes whether the still decodes to RGB or RGBA
const STILL_WIDTH: u32 = 24;
const STILL_HEIGHT: u32 = 24;

const FRAMERATE: u32 = 30;
const BYTES_PER_PIXEL: usize = 4;

const FRAMES_PER_LIFE: u64 = 3;

const SHORT_TIMEOUT_NS: u64 = 50_000_000;

const SETTLE: Duration = Duration::from_millis(500);

const GENEROUS_RETRY_NS: u64 = 60_000_000_000;

// ENODEV, an unplugged capture device
const DEVICE_GONE_ERRNO: i32 = 19;

fn main_source() -> DynSource {
    Box::new(
        VideoTestSrc::new(MAIN_WIDTH, MAIN_HEIGHT, FRAMERATE, FRAMES_PER_LIFE)
            .with_pattern(Pattern::SmpteBars),
    )
}

fn endless_fallback_source() -> DynSource {
    Box::new(
        VideoTestSrc::new(FALLBACK_WIDTH, FALLBACK_HEIGHT, FRAMERATE, u64::MAX)
            .with_pattern(Pattern::SmpteBars),
    )
}

fn ending_fallback_source() -> DynSource {
    Box::new(
        VideoTestSrc::new(FALLBACK_WIDTH, FALLBACK_HEIGHT, FRAMERATE, FRAMES_PER_LIFE)
            .with_pattern(Pattern::SmpteBars),
    )
}

fn frame_bytes(width: u32, height: u32) -> usize {
    width as usize * height as usize * BYTES_PER_PIXEL
}

fn dummy_frame_bytes() -> usize {
    let dummy = default_registry()
        .make_source("videotestsrc")
        .expect("videotestsrc is a baseline source");
    let dimension = |name| {
        dummy
            .get_property(name)
            .and_then(|v| v.as_uint())
            .unwrap_or_else(|| panic!("videotestsrc reports its {name}")) as u32
    };
    frame_bytes(dimension("width"), dimension("height"))
}

fn immediate_caps(mut source: DynSource) -> Caps {
    let waker = core::task::Waker::noop();
    let mut cx = core::task::Context::from_waker(waker);
    let mut future = source.intercept_caps();
    match future.as_mut().poll(&mut cx) {
        core::task::Poll::Ready(caps) => caps.expect("a test source's caps"),
        core::task::Poll::Pending => panic!("a test source's intercept_caps must be immediate"),
    }
}

fn counting_factory(build: fn() -> DynSource) -> (FallbacksrcSourceFactory, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let caps = immediate_caps(build());
    let factory = FallbacksrcSourceFactory::new(move || {
        seen.fetch_add(1, Ordering::SeqCst);
        (build(), caps.clone())
    });
    (factory, calls)
}

fn registry_with(
    main: fn() -> DynSource,
    fallback: Option<fn() -> DynSource>,
) -> (Registry, Arc<AtomicUsize>) {
    let mut registry = default_registry();
    registry.register_fallbacksrc_main_source(counting_factory(main).0);
    let calls = match fallback {
        Some(build) => {
            let (factory, calls) = counting_factory(build);
            registry.register_fallbacksrc_fallback_source(factory);
            calls
        }
        None => Arc::new(AtomicUsize::new(0)),
    };
    (registry, calls)
}

#[derive(Debug, Default)]
struct Delivered {
    main: u64,
    fallback_source: u64,
    dummy: u64,
    other: u64,
}

// cancelled rather than awaited, since a fallback may never end
async fn run_for(registry: &Registry, line: &str, channel: &str, duration: Duration) -> Delivered {
    let pull = register_appsink_pull(channel);
    let graph = parse_launch(registry, line).unwrap_or_else(|e| panic!("{line}: {e}"));
    let main_bytes = frame_bytes(MAIN_WIDTH, MAIN_HEIGHT);
    let fallback_bytes = frame_bytes(FALLBACK_WIDTH, FALLBACK_HEIGHT);
    let dummy_bytes = dummy_frame_bytes();
    let mut delivered = Delivered::default();
    let drain = async {
        while let Some(frame) = pull.pull().await {
            let bytes = frame
                .domain
                .as_system_slice()
                .expect("the sources here emit system memory")
                .len();
            match bytes {
                b if b == main_bytes => delivered.main += 1,
                b if b == fallback_bytes => delivered.fallback_source += 1,
                b if b == dummy_bytes => delivered.dummy += 1,
                _ => delivered.other += 1,
            }
        }
    };
    let _ = tokio::time::timeout(duration, async {
        tokio::join!(run_graph(graph, &ZeroClock, 4), drain)
    })
    .await;
    delivered
}

#[test]
fn every_source_here_has_its_own_frame_size() {
    let mut sizes = vec![
        frame_bytes(MAIN_WIDTH, MAIN_HEIGHT),
        frame_bytes(FALLBACK_WIDTH, FALLBACK_HEIGHT),
        dummy_frame_bytes(),
    ];
    let count = sizes.len();
    sizes.sort();
    sizes.dedup();
    assert_eq!(sizes.len(), count, "the tests here count frames by size");
}

#[tokio::test]
async fn the_registered_fallback_takes_over_when_the_main_ends() {
    let (registry, calls) = registry_with(main_source, Some(endless_fallback_source));
    let channel = "m1203_takes_over";
    let line = format!("fallbacksrc timeout={SHORT_TIMEOUT_NS} ! appsink channel={channel}");
    let delivered = run_for(&registry, &line, channel, RETRY_DELAY + SETTLE).await;

    assert_eq!(
        delivered.main, FRAMES_PER_LIFE,
        "the main source plays first: {delivered:?}"
    );
    assert!(
        delivered.fallback_source > 0,
        "the registered fallback took over once the main ended: {delivered:?}"
    );
    assert_eq!(
        delivered.dummy, 0,
        "no dummy generator with a registered fallback: {delivered:?}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1, "built once at parse");
}

struct FailingSrc;

impl SourceLoop for FailingSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(immediate_caps(main_source())))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, _out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async { Err(G2gError::Hardware(HardwareError::V4l2(DEVICE_GONE_ERRNO))) })
    }
}

fn failing_source() -> DynSource {
    Box::new(FailingSrc)
}

#[tokio::test]
async fn restart_on_eos_rebuilds_the_registered_fallback() {
    let (registry, calls) = registry_with(failing_source, Some(ending_fallback_source));
    let channel = "m1203_restart_on_eos";
    let line = format!(
        "fallbacksrc timeout={SHORT_TIMEOUT_NS} restart-on-eos=true retry-timeout={GENEROUS_RETRY_NS} ! appsink channel={channel}"
    );
    let delivered = run_for(&registry, &line, channel, RETRY_DELAY * 2 + SETTLE).await;

    let builds = calls.load(Ordering::SeqCst);
    assert!(
        builds >= 3,
        "two retry delays fit two rebuilds after the parse-time build, got {builds} builds"
    );
    assert!(
        delivered.fallback_source > 0,
        "the registered fallback holds the output while the main is down: {delivered:?}"
    );
}

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m1203_{}_{tag}", std::process::id()))
}

async fn still(tag: &str) -> PathBuf {
    let path = temp_path(tag);
    let line = format!(
        "videotestsrc num-buffers=1 width={STILL_WIDTH} height={STILL_HEIGHT} pattern=smpte ! pnmenc ! filesink location={}",
        path.display()
    );
    let graph = parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("{line}: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line}: {e:?}"));
    path
}

#[tokio::test]
async fn a_fallback_uri_wins_over_the_registered_fallback() {
    let still = still("fallback_uri_wins.pnm").await;
    let (registry, calls) = registry_with(failing_source, Some(endless_fallback_source));
    let channel = "m1203_uri_wins";
    let line = format!(
        "fallbacksrc timeout={SHORT_TIMEOUT_NS} fallback-uri=file://{} restart-on-eos=true retry-timeout={GENEROUS_RETRY_NS} ! appsink channel={channel}",
        still.display()
    );
    let delivered = run_for(&registry, &line, channel, RETRY_DELAY * 2 + SETTLE).await;
    std::fs::remove_file(&still).ok();

    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "the fallback factory is never called when the line names a fallback-uri"
    );
    assert!(
        delivered.other > 0,
        "the still's frames reached the sink: {delivered:?}"
    );
    assert_eq!(delivered.fallback_source, 0, "{delivered:?}");
}

#[tokio::test]
async fn with_neither_the_dummy_generator_is_the_fallback() {
    let (registry, _) = registry_with(main_source, None);
    let channel = "m1203_dummy";
    let line = format!("fallbacksrc timeout={SHORT_TIMEOUT_NS} ! appsink channel={channel}");
    let delivered = run_for(&registry, &line, channel, RETRY_DELAY + SETTLE).await;

    assert_eq!(delivered.main, FRAMES_PER_LIFE, "{delivered:?}");
    assert!(
        delivered.dummy > 0,
        "the dummy took over once the main ended: {delivered:?}"
    );
}

#[test]
fn a_fanned_out_fallbacksrc_puts_the_registered_fallback_on_its_kind() {
    use fallback_fanout_common::{
        byte_stream, categories, compressed_audio, compressed_video, dummy_video, kinds,
        node_names, parsed, short_type_name, source_caps, StubContainer,
    };
    use g2g_core::{AudioFormat, ByteStreamEncoding, VideoCodec};

    const AV_FIXTURE: &str = "tests/fixtures/av_h264_aac44100.ts";
    const NAMED: &str = "fb";

    let mut registry = fallback_fanout_common::registry(StubContainer {
        fanout: g2g_plugins::uridecodebin::ts_uri_fanout,
        bytes: byte_stream(ByteStreamEncoding::MpegTs),
        video: compressed_video(VideoCodec::H264),
        audio: compressed_audio(AudioFormat::Aac),
    });
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let caps = source_caps(dummy_video());
    registry.register_fallbacksrc_fallback_source(FallbacksrcSourceFactory::new(move || {
        seen.fetch_add(1, Ordering::SeqCst);
        (Box::new(dummy_video()) as DynSource, caps.clone())
    }));
    let uri = format!("file://{}/{AV_FIXTURE}", env!("CARGO_MANIFEST_DIR"));
    let graph = parsed(&registry, &format!("fallbacksrc name={NAMED} uri={uri}"));
    let names = node_names(&graph);
    let categories = categories(&graph);
    kinds(graph);

    assert_eq!(calls.load(Ordering::SeqCst), 1, "built once at parse");
    let fallback_name = format!("{NAMED}{FALLBACK_FALLBACK_SOURCE_SUFFIX}");
    assert!(
        names.contains(&fallback_name),
        "{fallback_name} missing from {names:?}"
    );
    let pacer = short_type_name::<g2g_plugins::clocksync::ClockSyncTransform>();
    assert_eq!(
        categories.iter().filter(|c| **c == pacer).count(),
        1,
        "only the audio dummy keeps a pacer: {categories:?}"
    );
    assert!(
        categories.contains(&short_type_name::<g2g_plugins::audiotestsrc::AudioTestSrc>()),
        "the audio switch keeps its dummy: {categories:?}"
    );
}
