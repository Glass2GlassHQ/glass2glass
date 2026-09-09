//! Shared stub machinery for the lone-`fallbacksrc` fan-out checks (M1168 MPEG-TS,
//! M1169 program stream / HLS): a registry that builds a whole fanned-out graph
//! without any decoder feature compiled in, and the inspectors each check reads
//! off the parsed graph.
//!
//! The stubs stand in for the container's decoders, so a check asserts the shape
//! the parser built and the frames that crossed it, not a codec.
#![allow(dead_code)] // no one battery uses every helper here

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use g2g_core::element::{AsyncElement, DynAsyncElement};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::runtime::{
    parse_launch, ElementFactory, GraphNode, LaunchFactory, MuxerFactory, Registry, SourceFactory,
    UriFanoutHook,
};
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    G2gError, Graph, MemoryDomain, NodeId, NodeKind, OutputSink, PadTemplate, PadTemplates,
    PipelineClock, PipelinePacket, Rate, VideoCodec,
};

pub(crate) struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// A `fallbacksrc` graph never ends by itself (the dummy fallback generator runs
/// forever), so every run over one is cancelled rather than waited out.
pub(crate) const RUN_BOUND: Duration = Duration::from_secs(4);

/// The dummy video generator's geometry: small, so a life is cheap.
pub(crate) const DUMMY_WIDTH: u32 = 64;
pub(crate) const DUMMY_HEIGHT: u32 = 48;
pub(crate) const DUMMY_FRAMERATE: u32 = 30;

/// The dummy audio generator's shape.
pub(crate) const DUMMY_SAMPLE_RATE: u32 = 48_000;
pub(crate) const DUMMY_CHANNELS: u8 = 2;
pub(crate) const DUMMY_TONE_HZ: u32 = 440;

/// Byte widths the stub decoder sizes its blank output with.
const RGBA_BYTES_PER_PIXEL: usize = 4;
const S16_BYTES_PER_SAMPLE: usize = 2;

/// Samples of stereo PCM one stub-decoded audio access unit stands for, an AAC
/// frame's worth.
const STUB_AUDIO_SAMPLES: usize = 1024;

pub(crate) fn dummy_video() -> g2g_plugins::videotestsrc::VideoTestSrc {
    g2g_plugins::videotestsrc::VideoTestSrc::new(
        DUMMY_WIDTH,
        DUMMY_HEIGHT,
        DUMMY_FRAMERATE,
        u64::MAX,
    )
}

pub(crate) fn dummy_audio() -> g2g_plugins::audiotestsrc::AudioTestSrc {
    g2g_plugins::audiotestsrc::AudioTestSrc::new(
        DUMMY_SAMPLE_RATE,
        DUMMY_CHANNELS,
        DUMMY_TONE_HZ,
        u64::MAX,
    )
}

/// A source's declared caps. The test sources answer synchronously, so polling
/// once is enough.
pub(crate) fn source_caps(mut source: impl SourceLoop) -> Caps {
    let waker = core::task::Waker::noop();
    let mut cx = core::task::Context::from_waker(waker);
    let mut future = core::pin::pin!(source.intercept_caps());
    match future.as_mut().poll(&mut cx) {
        core::task::Poll::Ready(caps) => caps.expect("a test source declares its caps"),
        core::task::Poll::Pending => panic!("a test source's intercept_caps must be immediate"),
    }
}

pub(crate) fn byte_stream(encoding: ByteStreamEncoding) -> Caps {
    Caps::ByteStream { encoding }
}

/// A compressed video codec at any geometry, which intersects whatever range a
/// demuxer declares on its port.
pub(crate) fn compressed_video(codec: VideoCodec) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// A compressed audio format with the sentinel shape a demuxer port carries
/// before the stream's own header refines it.
pub(crate) fn compressed_audio(format: AudioFormat) -> Caps {
    Caps::Audio {
        format,
        channels: 0,
        sample_rate: 0,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

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
                    let bytes = vec![0u8; self.output_bytes].into_boxed_slice();
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

pub(crate) static VIDEO_FRAMES: AtomicUsize = AtomicUsize::new(0);
pub(crate) static AUDIO_FRAMES: AtomicUsize = AtomicUsize::new(0);

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

/// One container's fan-out, as the stub registry needs to describe it: the hook
/// that probes its URI, the caps its byte source declares, and the elementary
/// caps its video and audio ports carry.
#[derive(Clone)]
pub(crate) struct StubContainer {
    pub(crate) fanout: UriFanoutHook,
    pub(crate) bytes: Caps,
    pub(crate) video: Caps,
    pub(crate) audio: Caps,
}

/// The container the stubs in this test binary stand in for. `ElementFactory`
/// takes a plain `fn` pointer, which cannot capture, so the three build
/// functions below read the caps from here instead. One per binary: a file that
/// parses two containers needs two registries, which this does not give it.
static CONTAINER: OnceLock<StubContainer> = OnceLock::new();

fn stub_container() -> &'static StubContainer {
    CONTAINER
        .get()
        .expect("registry() records the container before anything builds a stub")
}

/// Stands in for the single-stream demuxer an inline `fallbacksrc` plugs between
/// the sniffed byte source and the video decoder.
fn build_demux_stub(_: &Caps) -> Box<dyn DynAsyncElement> {
    let container = stub_container();
    Box::new(StubDecode::new(
        container.bytes.clone(),
        container.video.clone(),
        0,
    ))
}

fn build_video_stub(_: &Caps) -> Box<dyn DynAsyncElement> {
    Box::new(StubDecode::new(
        stub_container().video.clone(),
        source_caps(dummy_video()),
        (DUMMY_WIDTH * DUMMY_HEIGHT) as usize * RGBA_BYTES_PER_PIXEL,
    ))
}

fn build_audio_stub(_: &Caps) -> Box<dyn DynAsyncElement> {
    Box::new(StubDecode::new(
        stub_container().audio.clone(),
        source_caps(dummy_audio()),
        STUB_AUDIO_SAMPLES * DUMMY_CHANNELS as usize * S16_BYTES_PER_SAMPLE,
    ))
}

/// A registry that can build the whole fanned-out graph for `container` without a
/// decoder feature: its fan-out hook, pass-through stubs standing in for its
/// video and audio decoders, the dummy generators and their pacer, the audio
/// tail, the switch, and a counting sink under each automatic-sink name.
pub(crate) fn registry(container: StubContainer) -> Registry {
    let stubs = CONTAINER.get_or_init(|| container.clone());
    assert_eq!(
        (&stubs.bytes, &stubs.video, &stubs.audio),
        (&container.bytes, &container.video, &container.audio),
        "one stub container per test binary: the stubs read theirs from a static"
    );
    let container = stub_container();
    let mut reg = Registry::new();
    reg.register_uri_fanout(container.fanout);
    reg.register_restart_source(g2g_plugins::fallbacksrc::restart_source);
    reg.register_uri(g2g_plugins::uridecodebin::file_handler());
    reg.register(ElementFactory::new(
        "demuxstub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(container.bytes.clone())),
            PadTemplate::source(CapsSet::one(container.video.clone())),
        ]),
        build_demux_stub,
    ));
    reg.register(ElementFactory::new(
        "videostub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(container.video.clone())),
            PadTemplate::source(CapsSet::one(source_caps(dummy_video()))),
        ]),
        build_video_stub,
    ));
    reg.register(ElementFactory::new(
        "audiostub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(container.audio.clone())),
            PadTemplate::source(CapsSet::one(source_caps(dummy_audio()))),
        ]),
        build_audio_stub,
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

pub(crate) fn node_names(graph: &Graph<GraphNode>) -> Vec<String> {
    (0..graph.node_count())
        .filter_map(|i| graph.node_name(NodeId(i as u32)).map(String::from))
        .collect()
}

pub(crate) fn categories(graph: &Graph<GraphNode>) -> Vec<&'static str> {
    (0..graph.node_count())
        .filter_map(|i| graph.element(NodeId(i as u32)).map(|e| e.log_category()))
        .collect()
}

pub(crate) fn kinds(graph: Graph<GraphNode>) -> Vec<NodeKind> {
    let valid = graph.finish().expect("the built graph is valid");
    valid.topo().iter().map(|&n| valid.kind(n)).collect()
}

pub(crate) fn parsed(reg: &Registry, line: &str) -> Graph<GraphNode> {
    parse_launch(reg, line).unwrap_or_else(|e| panic!("{line}: {e}"))
}

pub(crate) fn short_type_name<T>() -> &'static str {
    g2g_core::log::short_type_name::<T>()
}
