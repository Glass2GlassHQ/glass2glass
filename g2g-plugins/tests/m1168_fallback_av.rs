//! M1168: a lone `fallbacksrc` carries audio and video at once. When the whole
//! pipeline is a single `fallbacksrc uri=X`, the URI is probed by the same
//! container hooks a lone `playbin` uses, and every stream kind the probe reports
//! gets its own decode chain, its own `fallbackswitch`, and its own automatic
//! sink off one byte source and one demuxer. An inline `fallbacksrc`, and a URI
//! no hook claims, stay single-stream.
//!
//! The structural checks build a registry of stub decoders and counting sinks, so
//! they need no decoder feature; the decode-and-run check does too, and the frames
//! it counts are the container's real access units.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use g2g_core::element::AsyncElement;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::runtime::{
    parse_launch, run_graph, ElementFactory, GraphNode, LaunchFactory, MuxerFactory, Registry,
    SourceFactory, FALLBACK_AUDIO_SUFFIX, FALLBACK_FALLBACK_SOURCE_SUFFIX,
    FALLBACK_MAIN_SOURCE_SUFFIX, FALLBACK_SWITCH_NAME, FALLBACK_VIDEO_SUFFIX,
};
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    G2gError, Graph, MemoryDomain, NodeId, NodeKind, OutputSink, PadTemplate, PadTemplates,
    PipelineClock, PipelinePacket, Rate, VideoCodec,
};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// A `fallbacksrc` graph never ends by itself (the dummy fallback generator runs
/// forever), so every run here is cancelled rather than waited out.
const RUN_BOUND: Duration = Duration::from_secs(4);

/// The A/V transport stream every fan-out check probes: one H.264 stream and one
/// AAC stream in one container.
const AV_FIXTURE: &str = "tests/fixtures/av_h264_aac44100.ts";

/// The `fallbacksrc name=` the named checks use, and the stem the unnamed ones
/// fall back to (the expansion numbers its generated switches from zero).
const NAMED: &str = "fb";
fn unnamed_stem() -> String {
    format!("{FALLBACK_SWITCH_NAME}-0")
}

fn av_uri() -> String {
    format!("file://{}/{AV_FIXTURE}", env!("CARGO_MANIFEST_DIR"))
}

// --- the stub decode chain ---
//
// The dummy fallback generators fix the raw caps both switch inputs must agree
// on, so each stub decoder declares the generator's own caps rather than a
// second spelling of them.

/// The dummy video generator's geometry: small, so a life is cheap.
const DUMMY_WIDTH: u32 = 64;
const DUMMY_HEIGHT: u32 = 48;
const DUMMY_FRAMERATE: u32 = 30;

/// The dummy audio generator's shape.
const DUMMY_SAMPLE_RATE: u32 = 48_000;
const DUMMY_CHANNELS: u8 = 2;
const DUMMY_TONE_HZ: u32 = 440;

/// Byte widths the stub decoder sizes its blank output with.
const RGBA_BYTES_PER_PIXEL: usize = 4;
const S16_BYTES_PER_SAMPLE: usize = 2;

fn dummy_video() -> g2g_plugins::videotestsrc::VideoTestSrc {
    g2g_plugins::videotestsrc::VideoTestSrc::new(
        DUMMY_WIDTH,
        DUMMY_HEIGHT,
        DUMMY_FRAMERATE,
        u64::MAX,
    )
}

fn dummy_audio() -> g2g_plugins::audiotestsrc::AudioTestSrc {
    g2g_plugins::audiotestsrc::AudioTestSrc::new(
        DUMMY_SAMPLE_RATE,
        DUMMY_CHANNELS,
        DUMMY_TONE_HZ,
        u64::MAX,
    )
}

/// A source's declared caps. The test sources answer synchronously, so polling
/// once is enough.
fn source_caps(mut source: impl SourceLoop) -> Caps {
    let waker = core::task::Waker::noop();
    let mut cx = core::task::Context::from_waker(waker);
    let mut future = core::pin::pin!(source.intercept_caps());
    match future.as_mut().poll(&mut cx) {
        core::task::Poll::Ready(caps) => caps.expect("a test source declares its caps"),
        core::task::Poll::Pending => panic!("a test source's intercept_caps must be immediate"),
    }
}

fn ts_bytes() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn aac_any() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 0,
        sample_rate: 0,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// Samples of stereo PCM one stub-decoded audio access unit stands for, an AAC
/// frame's worth.
const STUB_AUDIO_SAMPLES: usize = 1024;

/// A stand-in for a decoder: it re-types its input to the raw caps the branch has
/// to reach and answers each access unit with one blank raw buffer of that shape,
/// so the fan-out runs, and its converters work, without a decoder feature
/// compiled in.
struct StubDecode {
    input: Caps,
    output: Caps,
    output_bytes: usize,
    configured: bool,
}

impl StubDecode {
    /// Stands in for the single-stream `tsdemux` an inline `fallbacksrc` plugs
    /// between the sniffed byte source and the video decoder.
    fn demux() -> Self {
        Self::new(ts_bytes(), h264_any(), 0)
    }
    fn video() -> Self {
        Self::new(
            h264_any(),
            source_caps(dummy_video()),
            (DUMMY_WIDTH * DUMMY_HEIGHT) as usize * RGBA_BYTES_PER_PIXEL,
        )
    }
    fn audio() -> Self {
        Self::new(
            aac_any(),
            source_caps(dummy_audio()),
            STUB_AUDIO_SAMPLES * DUMMY_CHANNELS as usize * S16_BYTES_PER_SAMPLE,
        )
    }
    fn new(input: Caps, output: Caps, output_bytes: usize) -> Self {
        Self {
            input,
            output,
            output_bytes,
            configured: false,
        }
    }
}

impl AsyncElement for StubDecode {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, _upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(self.output.clone())
    }
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Mapping(Vec::from([(
            CapsSet::one(self.input.clone()),
            CapsSet::one(self.output.clone()),
        )]))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::Eos => Ok(()),
                PipelinePacket::DataFrame(frame) => {
                    let bytes = alloc_zeroed(self.output_bytes);
                    out.push(PipelinePacket::DataFrame(Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
                        timing: frame.timing,
                        sequence: frame.sequence,
                        meta: Default::default(),
                    }))
                    .await?;
                    Ok(())
                }
                other => {
                    out.push(other).await?;
                    Ok(())
                }
            }
        })
    }
}

static VIDEO_FRAMES: AtomicUsize = AtomicUsize::new(0);
static AUDIO_FRAMES: AtomicUsize = AtomicUsize::new(0);

/// A sink that tallies the data frames it is handed into the counter it was built
/// with, so a run can say how many frames reached each kind's end of the graph.
/// It declares one concrete format, as a display or audio sink does, which is
/// what the audio tail's converters fixate against.
struct CountingSink {
    accepts: Caps,
    frames: &'static AtomicUsize,
}

impl PadTemplates for CountingSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::new()
    }
}

impl AsyncElement for CountingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, caps: &Caps) -> Result<Caps, G2gError> {
        Ok(caps.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(self.accepts.clone()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if matches!(packet, PipelinePacket::DataFrame(_)) {
            self.frames.fetch_add(1, Ordering::SeqCst);
        }
        Box::pin(async { Ok(()) })
    }
}

/// A registry that can build the whole fanned-out graph without a decoder
/// feature: the MPEG-TS fan-out hook, pass-through stubs standing in for the
/// H.264 and AAC decoders, the dummy generators and their pacer, the audio tail,
/// the switch, and a counting sink under each automatic-sink name.
fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_uri_fanout(g2g_plugins::uridecodebin::ts_uri_fanout);
    reg.register_restart_source(g2g_plugins::fallbacksrc::restart_source);
    reg.register_uri(g2g_plugins::uridecodebin::file_handler());
    reg.register(ElementFactory::new(
        "tsdemuxstub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(ts_bytes())),
            PadTemplate::source(CapsSet::one(h264_any())),
        ]),
        |_| Box::new(StubDecode::demux()),
    ));
    reg.register(ElementFactory::new(
        "h264stub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(h264_any())),
            PadTemplate::source(CapsSet::one(source_caps(dummy_video()))),
        ]),
        |_| Box::new(StubDecode::video()),
    ));
    reg.register(ElementFactory::new(
        "aacstub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(aac_any())),
            PadTemplate::source(CapsSet::one(source_caps(dummy_audio()))),
        ]),
        |_| Box::new(StubDecode::audio()),
    ));
    reg.register_source(SourceFactory::new(
        "videotestsrc",
        source_caps(dummy_video()),
        || Box::new(dummy_video()),
    ));
    reg.register_source(SourceFactory::new(
        "audiotestsrc",
        source_caps(dummy_audio()),
        || Box::new(dummy_audio()),
    ));
    reg.register_launch(LaunchFactory::new("clocksync", Vec::new(), || {
        Box::new(g2g_plugins::clocksync::ClockSyncTransform::new())
    }));
    reg.register_launch(
        LaunchFactory::of::<g2g_plugins::audioconvert::AudioConvert>("audioconvert", || {
            Box::new(g2g_plugins::audioconvert::AudioConvert::auto())
        }),
    );
    reg.register_launch(
        LaunchFactory::of::<g2g_plugins::audioresample::AudioResample>("audioresample", || {
            Box::new(g2g_plugins::audioresample::AudioResample::auto())
        }),
    );
    reg.register_muxer(MuxerFactory::new("fallbackswitch", |inputs| {
        Box::new(g2g_plugins::fallbackswitch::FallbackSwitch::new(inputs))
    }));
    reg.register_launch(LaunchFactory::of::<CountingSink>("autovideosink", || {
        Box::new(CountingSink {
            accepts: source_caps(dummy_video()),
            frames: &VIDEO_FRAMES,
        })
    }));
    reg.register_launch(LaunchFactory::of::<CountingSink>("autoaudiosink", || {
        Box::new(CountingSink {
            accepts: source_caps(dummy_audio()),
            frames: &AUDIO_FRAMES,
        })
    }));
    reg
}

fn node_names(graph: &Graph<GraphNode>) -> Vec<String> {
    (0..graph.node_count())
        .filter_map(|i| graph.node_name(NodeId(i as u32)).map(String::from))
        .collect()
}

fn categories(graph: &Graph<GraphNode>) -> Vec<&'static str> {
    (0..graph.node_count())
        .filter_map(|i| graph.element(NodeId(i as u32)).map(|e| e.log_category()))
        .collect()
}

fn kinds(graph: Graph<GraphNode>) -> Vec<NodeKind> {
    let valid = graph.finish().expect("the built graph is valid");
    valid.topo().iter().map(|&n| valid.kind(n)).collect()
}

fn parsed(reg: &Registry, line: &str) -> Graph<GraphNode> {
    parse_launch(reg, line).unwrap_or_else(|e| panic!("{line}: {e}"))
}

/// A PNM still no container fan-out hook claims, so a lone `fallbacksrc` over it
/// stays single-stream.
async fn still(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("g2g_m1168_{}_{tag}", std::process::id()));
    let reg = default_registry();
    let line = format!(
        "videotestsrc num-buffers=1 pattern=smpte ! pnmenc ! filesink location={}",
        path.display()
    );
    let graph = parsed(&reg, &line);
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line}: {e:?}"));
    path
}

#[test]
fn a_lone_fallbacksrc_gives_each_kind_a_switch_and_a_sink() {
    let reg = registry();
    let line = format!("fallbacksrc name={NAMED} uri={}", av_uri());
    let graph = parsed(&reg, &line);
    let names = node_names(&graph);
    let categories = categories(&graph);
    let kinds = kinds(graph);

    // One switch and one sink per kind, off one byte source and one demuxer.
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Muxer(2)).count(),
        2,
        "one 2-input fallbackswitch per kind: {kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Sink).count(),
        2,
        "one automatic sink per kind: {kinds:?}"
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|k| matches!(k, NodeKind::Tee(_)))
            .count(),
        1,
        "one demuxer feeding both kinds: {kinds:?}"
    );
    // The main byte source plus one dummy generator per kind.
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Source).count(),
        3,
        "the byte source and both dummy generators: {kinds:?}"
    );
    for expected in [
        format!("{NAMED}{FALLBACK_VIDEO_SUFFIX}"),
        format!("{NAMED}{FALLBACK_AUDIO_SUFFIX}"),
    ] {
        assert!(
            names.contains(&expected),
            "{expected} missing from {names:?}"
        );
    }
    // Only the audio branch carries the converters that fix the sink's PCM shape.
    for tail in [
        short_type_name::<g2g_plugins::audioconvert::AudioConvert>(),
        short_type_name::<g2g_plugins::audioresample::AudioResample>(),
    ] {
        assert_eq!(
            categories.iter().filter(|c| **c == tail).count(),
            1,
            "{tail} sits between the audio switch and its sink: {categories:?}"
        );
    }
    // Each kind's dummy fallback is its own generator.
    for dummy in [
        short_type_name::<g2g_plugins::videotestsrc::VideoTestSrc>(),
        short_type_name::<g2g_plugins::audiotestsrc::AudioTestSrc>(),
    ] {
        assert!(
            categories.contains(&dummy),
            "{dummy} missing from {categories:?}"
        );
    }
}

#[test]
fn the_fanned_out_nodes_take_their_generated_names() {
    let reg = registry();
    let line = format!(
        "fallbacksrc name={NAMED} uri={uri} fallback-uri={uri}",
        uri = av_uri()
    );
    let names = node_names(&parsed(&reg, &line));
    for expected in [
        format!("{NAMED}{FALLBACK_MAIN_SOURCE_SUFFIX}"),
        format!("{NAMED}{FALLBACK_FALLBACK_SOURCE_SUFFIX}"),
        format!("{NAMED}{FALLBACK_VIDEO_SUFFIX}"),
        format!("{NAMED}{FALLBACK_AUDIO_SUFFIX}"),
    ] {
        assert!(
            names.contains(&expected),
            "{expected} missing from {names:?}"
        );
    }

    // Unnamed, every generated name hangs off the generated stem instead.
    let line = format!("fallbacksrc uri={}", av_uri());
    let names = node_names(&parsed(&reg, &line));
    let stem = unnamed_stem();
    for expected in [
        format!("{stem}{FALLBACK_MAIN_SOURCE_SUFFIX}"),
        format!("{stem}{FALLBACK_VIDEO_SUFFIX}"),
        format!("{stem}{FALLBACK_AUDIO_SUFFIX}"),
    ] {
        assert!(
            names.contains(&expected),
            "{expected} missing from {names:?}"
        );
    }
}

/// Every name the fan-out generates goes through the same duplicate check the
/// parser applies to a line's own `name=`, so two nodes can never share one.
#[test]
fn the_generated_names_are_all_distinct() {
    let reg = registry();
    let line = format!(
        "fallbacksrc name={NAMED} uri={uri} fallback-uri={uri}",
        uri = av_uri()
    );
    let mut names = node_names(&parsed(&reg, &line));
    let count = names.len();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), count, "{names:?}");
}

/// The fallback URI's own fan-out feeds each switch's second input, so a kind's
/// fallback is that kind's stream rather than the dummy generator.
#[test]
fn a_fallback_uri_that_fans_out_replaces_both_dummies() {
    let reg = registry();
    let line = format!(
        "fallbacksrc name={NAMED} uri={uri} fallback-uri={uri}",
        uri = av_uri()
    );
    let graph = parsed(&reg, &line);
    let categories = categories(&graph);
    for dummy in [
        short_type_name::<g2g_plugins::videotestsrc::VideoTestSrc>(),
        short_type_name::<g2g_plugins::audiotestsrc::AudioTestSrc>(),
    ] {
        assert!(
            !categories.contains(&dummy),
            "{dummy} should not be built when the fallback URI carries that kind: {categories:?}"
        );
    }
    assert_eq!(
        kinds(graph)
            .iter()
            .filter(|k| matches!(k, NodeKind::Tee(_)))
            .count(),
        2,
        "the main and the fallback container each get a demuxer"
    );
}

/// A URI no fan-out hook claims still builds a runnable single-stream pipeline:
/// the expansion appends the kind's automatic sink so the lone keyword is a
/// complete line.
#[tokio::test]
async fn an_unclaimed_uri_stays_single_stream() {
    let path = still("unclaimed.pnm").await;
    let reg = default_registry();
    let line = format!("fallbacksrc uri=file://{}", path.display());
    let kinds = kinds(parsed(&reg, &line));
    std::fs::remove_file(&path).ok();

    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Muxer(2)).count(),
        1,
        "one switch, not one per kind: {kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Sink).count(),
        1,
        "one automatic sink: {kinds:?}"
    );
}

/// An inline `fallbacksrc` heads a chain the line wrote itself, which has no way
/// to name a second output, so it keeps its single-stream shape even over a
/// container the fan-out hooks do claim.
#[test]
fn an_inline_fallbacksrc_stays_single_stream() {
    let reg = registry();
    let line = format!("fallbacksrc uri={} ! autovideosink", av_uri());
    let kinds = kinds(parsed(&reg, &line));
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Muxer(2)).count(),
        1,
        "one switch: {kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Sink).count(),
        1,
        "the line's own sink, and only it: {kinds:?}"
    );
}

/// The fanned-out graph runs: frames off the container's video stream and its
/// audio stream both reach their own sink within one bounded run. The fallback
/// URI is the same container, so no dummy generator is built and every frame
/// counted came out of a decode chain.
#[tokio::test]
async fn both_kinds_deliver_frames_to_their_own_sink() {
    VIDEO_FRAMES.store(0, Ordering::SeqCst);
    AUDIO_FRAMES.store(0, Ordering::SeqCst);
    let reg = registry();
    let line = format!("fallbacksrc uri={uri} fallback-uri={uri}", uri = av_uri());
    let graph = parsed(&reg, &line);
    tokio::time::timeout(RUN_BOUND, run_graph(graph, &ZeroClock, 4))
        .await
        .unwrap_or_else(|_| panic!("{line}: the run did not finish inside {RUN_BOUND:?}"))
        .unwrap_or_else(|e| panic!("{line}: {e:?}"));

    assert!(
        VIDEO_FRAMES.load(Ordering::SeqCst) > 0,
        "the video branch delivered nothing"
    );
    assert!(
        AUDIO_FRAMES.load(Ordering::SeqCst) > 0,
        "the audio branch delivered nothing"
    );
}

fn alloc_zeroed(bytes: usize) -> Box<[u8]> {
    vec![0u8; bytes].into_boxed_slice()
}

fn short_type_name<T>() -> &'static str {
    g2g_core::log::short_type_name::<T>()
}
