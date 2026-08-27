//! M1074 raw AAC and AC-3 playback: `typefind` types an `.aac` (ADTS) and an
//! `.ac3` by content, `aacparse` / `ac3parse` split them into frames with
//! sample-accurate timestamps, and `decodebin` splices the parser ahead of a
//! decoder.
//!
//! The fixtures and the expected counts come from ffmpeg / ffprobe, checked in
//! next to this test so it needs neither at run time:
//!
//! ```text
//! ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=1" -ac 2 \
//!   -c:a aac -b:a 128k -f adts tests/fixtures/aac_stereo_44100_adts.aac
//! ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=1" -ac 2 \
//!   -c:a ac3 -b:a 192k tests/fixtures/ac3_stereo_48000.ac3
//! ffprobe -v error -show_format -show_streams -count_frames -of json <file> > <file>.json
//! ```
#![cfg(feature = "std")]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{
    is_raw_audio, parse_launch, run_graph, ElementFactory, GraphNode, Registry,
};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, Graph, OutputSink, PadTemplate, PadTemplates, PipelineClock, PushOutcome,
    ANY_CHANNELS, ANY_SAMPLE_RATE,
};
use g2g_plugins::aacparse::AacParse;
use g2g_plugins::ac3parse::Ac3Parse;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::registry::default_registry;
use g2g_plugins::typefind::elementary_audio_caps;

mod ffprobe;
use ffprobe::{fixture_path, Probe};

/// A one-second sine in each raw elementary framing.
const AAC_FIXTURE: &str = "aac_stereo_44100_adts.aac";
const AC3_FIXTURE: &str = "ac3_stereo_48000.ac3";

/// Samples per channel one frame of each format decodes to.
const AAC_SAMPLES_PER_FRAME: u64 = 1024;
const AC3_SAMPLES_PER_FRAME: u64 = 1536;

/// Bytes per input buffer when a parser is driven directly: an odd size, so
/// frames straddle buffer boundaries the way `filesrc` chunks make them.
const CHUNK_LEN: usize = 997;

const NS_PER_SECOND: u64 = 1_000_000_000;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    frames: Vec<Frame>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        match packet {
            PipelinePacket::CapsChanged(c) => self.caps.push(c),
            PipelinePacket::DataFrame(f) => self.frames.push(f),
            _ => {}
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn chunk(bytes: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

/// Drive `element` over the whole fixture in [`CHUNK_LEN`] pieces, then end the
/// stream.
async fn feed<E: AsyncElement>(element: &mut E, format: AudioFormat, name: &str) -> CaptureSink {
    let bytes = std::fs::read(fixture_path(name)).expect("the fixture is checked in");
    element
        .configure_pipeline(&elementary_audio_caps(format))
        .expect("the sentinel caps are accepted");
    let mut sink = CaptureSink::default();
    for piece in bytes.chunks(CHUNK_LEN) {
        element
            .process(chunk(piece), &mut sink)
            .await
            .expect("the stream parses");
    }
    element
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("the tail flushes");
    sink
}

/// Run `filesrc location=<fixture> ! typefind ! <parser> ! fakesink` and return
/// the buffers the sink consumed.
async fn launch_frames(name: &str, parser: &str) -> u64 {
    let reg = default_registry();
    let line = format!(
        "filesrc location={} ! typefind ! {parser} ! fakesink",
        fixture_path(name).display()
    );
    let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs")
        .frames_consumed
}

/// Every frame's presentation time is one frame duration past the one before it,
/// and the parsed timeline spans the probed duration.
fn assert_timeline(sink: &CaptureSink, probe: &Probe, samples_per_frame: u64) {
    let step = |frames: u64| samples_per_frame * frames * NS_PER_SECOND / probe.sample_rate as u64;
    let times: Vec<u64> = sink.frames.iter().map(|f| f.timing.pts_ns).collect();
    assert_eq!(times, (0..times.len() as u64).map(step).collect::<Vec<_>>());
    let last = sink.frames.last().expect("frames").timing;
    assert_eq!(last.duration_ns, step(1));
    let stream_ns = last.pts_ns + last.duration_ns;
    let probed_ns = (probe.duration_seconds * 1e9) as u64;
    assert!(
        stream_ns.abs_diff(probed_ns) <= last.duration_ns,
        "the parsed timeline spans the probed duration: {stream_ns} vs {probed_ns}"
    );
}

#[tokio::test]
async fn launch_line_types_adts_aac_and_parses_every_frame() {
    let probe = Probe::read(AAC_FIXTURE);
    assert_eq!(
        launch_frames(AAC_FIXTURE, "aacparse").await,
        probe.frames,
        "one buffer per ADTS access unit"
    );
}

#[tokio::test]
async fn launch_line_types_ac3_and_parses_every_frame() {
    let probe = Probe::read(AC3_FIXTURE);
    assert_eq!(
        launch_frames(AC3_FIXTURE, "ac3parse").await,
        probe.frames,
        "one buffer per AC-3 syncframe"
    );
}

#[tokio::test]
async fn parsed_aac_frames_carry_the_probed_caps_and_timing() {
    let probe = Probe::read(AAC_FIXTURE);
    let mut parser = AacParse::new();
    let sink = feed(&mut parser, AudioFormat::Aac, AAC_FIXTURE).await;

    assert_eq!(sink.frames.len() as u64, probe.frames);
    assert_eq!(
        sink.caps,
        vec![probe.audio_caps(AudioFormat::Aac)],
        "the concrete caps are announced once, before the first frame"
    );
    assert_timeline(&sink, &probe, AAC_SAMPLES_PER_FRAME);
}

#[tokio::test]
async fn parsed_ac3_frames_carry_the_probed_caps_and_timing() {
    let probe = Probe::read(AC3_FIXTURE);
    let mut parser = Ac3Parse::new();
    let sink = feed(&mut parser, AudioFormat::Ac3, AC3_FIXTURE).await;

    assert_eq!(sink.frames.len() as u64, probe.frames);
    assert_eq!(
        sink.caps,
        vec![probe.audio_caps(AudioFormat::Ac3)],
        "the concrete caps are announced once, before the first frame"
    );
    assert_timeline(&sink, &probe, AC3_SAMPLES_PER_FRAME);
}

/// Registry name of the stub decoder below.
const STUB_DECODER: &str = "m1074-stub-audio-decoder";

/// A decoder that turns compressed audio into PCM, for the auto-plug search to
/// find in a build with no real audio decoder compiled in. Never run.
#[derive(Debug, Default)]
struct StubAudioDec;

impl AsyncElement for StubAudioDec {
    type ProcessFuture<'a>
        = core::pin::Pin<Box<dyn core::future::Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, _upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(stub_pcm_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new("stub audio decoder", "Codec/Decoder/Audio", "test", "g2g")
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

fn stub_pcm_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: ANY_CHANNELS,
        sample_rate: ANY_SAMPLE_RATE,
    }
}

impl PadTemplates for StubAudioDec {
    fn pad_templates() -> Vec<PadTemplate> {
        // The sentinel shape a parsed elementary stream is declared with, as
        // `ffmpegaudiodec` declares it.
        let compressed = |format| Caps::Audio {
            format,
            channels: ANY_CHANNELS,
            sample_rate: 0,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(Vec::from([
                compressed(AudioFormat::Aac),
                compressed(AudioFormat::Ac3),
            ]))),
            PadTemplate::source(CapsSet::one(stub_pcm_caps())),
        ])
    }
}

/// `decodebin` over the sniffed caps of `fixture` must expand to the named
/// parser plus a decoder.
fn assert_decodebin_splices(fixture: &str, format: AudioFormat, parser: &str) {
    let sentinel = elementary_audio_caps(format);
    let mut reg: Registry = default_registry();
    assert_eq!(
        reg.parser_name(&sentinel),
        Some(parser),
        "the decode chain's parser for this format"
    );
    reg.register(ElementFactory::of::<StubAudioDec>(STUB_DECODER, |_| {
        Box::new(StubAudioDec)
    }));

    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(FileSrc::new(
        fixture_path(fixture),
        sentinel.clone(),
    )));
    let sink = graph.add_sink(GraphNode::element(FakeSink::new()));
    let inserted = reg
        .decodebin(&mut graph, src, sink, &sentinel, &is_raw_audio, 4)
        .expect("the stream reaches PCM");
    assert_eq!(inserted.len(), 2, "the parser plus a decoder");
    assert_eq!(
        graph
            .element(inserted[0])
            .expect("the spliced parser")
            .log_category(),
        parser_type_name(format),
        "the parser leads the chain"
    );
}

fn parser_type_name(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Aac => g2g_core::log::short_type_name::<AacParse>(),
        _ => g2g_core::log::short_type_name::<Ac3Parse>(),
    }
}

#[test]
fn decodebin_splices_aacparse_ahead_of_the_decoder() {
    assert_decodebin_splices(AAC_FIXTURE, AudioFormat::Aac, "aacparse");
}

#[test]
fn decodebin_splices_ac3parse_ahead_of_the_decoder() {
    assert_decodebin_splices(AC3_FIXTURE, AudioFormat::Ac3, "ac3parse");
}

/// Decoding needs a real decoder, which only the `ffmpeg` build has.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
mod decode {
    use super::*;
    use g2g_plugins::ffmpegaudiodec::FfmpegAudioDec;

    /// Bytes per interleaved `PcmS16Le` sample.
    const BYTES_PER_SAMPLE: usize = 2;

    /// The whole GStreamer-shaped line: content typing, the auto-plugged parser,
    /// and a real decoder.
    async fn decodebin_line_frames(name: &str) -> u64 {
        let reg = default_registry();
        let line = format!(
            "filesrc location={} ! decodebin ! fakesink",
            fixture_path(name).display()
        );
        let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
        run_graph(graph, &ZeroClock, 4)
            .await
            .expect("the pipeline runs")
            .frames_consumed
    }

    /// Decode every parsed frame and return the interleaved sample count.
    async fn decode_samples(sink: CaptureSink, probe: &Probe, format: AudioFormat) -> usize {
        let mut decoder = FfmpegAudioDec::new();
        decoder
            .configure_pipeline(&probe.audio_caps(format))
            .expect("the decoder takes the parsed caps");
        let mut pcm = CaptureSink::default();
        for frame in sink.frames {
            decoder
                .process(PipelinePacket::DataFrame(frame), &mut pcm)
                .await
                .expect("every frame decodes");
        }
        decoder
            .process(PipelinePacket::Eos, &mut pcm)
            .await
            .expect("the decoder drains");
        let decoded_bytes: usize = pcm
            .frames
            .iter()
            .map(|f| f.domain.as_system_slice().map_or(0, <[u8]>::len))
            .sum();
        decoded_bytes / (BYTES_PER_SAMPLE * probe.channels as usize)
    }

    #[tokio::test]
    async fn decodebin_line_plays_an_adts_aac_file() {
        let probe = Probe::read(AAC_FIXTURE);
        assert!(decodebin_line_frames(AAC_FIXTURE).await >= probe.frames);
    }

    #[tokio::test]
    async fn decodebin_line_plays_an_ac3_file() {
        let probe = Probe::read(AC3_FIXTURE);
        assert!(decodebin_line_frames(AC3_FIXTURE).await >= probe.frames);
    }

    #[tokio::test]
    async fn decodes_aac_to_pcm_of_the_probed_length() {
        let probe = Probe::read(AAC_FIXTURE);
        let mut parser = AacParse::new();
        let parsed = feed(&mut parser, AudioFormat::Aac, AAC_FIXTURE).await;
        let samples = decode_samples(parsed, &probe, AudioFormat::Aac).await;
        // The encoder's own delay and padding stretch the coded stream past the
        // nominal duration, by less than a frame at each end.
        let expected = (probe.duration_seconds * probe.sample_rate as f64) as usize;
        assert!(
            samples.abs_diff(expected) <= 2 * AAC_SAMPLES_PER_FRAME as usize,
            "{samples} decoded samples against {expected} probed"
        );
    }

    #[tokio::test]
    async fn decodes_ac3_to_pcm_of_the_probed_length() {
        let probe = Probe::read(AC3_FIXTURE);
        let mut parser = Ac3Parse::new();
        let parsed = feed(&mut parser, AudioFormat::Ac3, AC3_FIXTURE).await;
        let samples = decode_samples(parsed, &probe, AudioFormat::Ac3).await;
        assert_eq!(
            samples,
            probe.frames as usize * AC3_SAMPLES_PER_FRAME as usize,
            "every parsed syncframe decoded"
        );
    }
}
