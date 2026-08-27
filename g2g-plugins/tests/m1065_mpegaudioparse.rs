//! M1065 MPEG audio playback: `typefind` types an `.mp3` by content,
//! `mpegaudioparse` splits it into frames with sample-accurate timestamps and
//! surfaces its ID3 tags, `id3demux` strips the tags for a parser that would
//! rather not see them, and `decodebin` splices the parser ahead of a decoder.
//!
//! The fixtures and the expected counts come from ffmpeg / ffprobe, checked in
//! next to this test so it needs neither at run time:
//!
//! ```text
//! ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=1" -ac 2 \
//!   -c:a libmp3lame -b:a 128k -id3v2_version 3 \
//!   -metadata title="g2g sine" -metadata artist="glass2glass" \
//!   tests/fixtures/mp3_stereo_44100_id3v2.mp3
//! ffmpeg -f lavfi -i "sine=frequency=330:sample_rate=44100:duration=1" -ac 1 \
//!   -c:a libmp3lame -b:a 64k -id3v2_version 4 -write_id3v1 1 \
//!   -metadata title="g2g mono" -metadata artist="glass2glass" \
//!   tests/fixtures/mp3_mono_44100_id3v1.mp3
//! ffprobe -v error -show_format -show_streams -count_frames -of json <file> > <file>.json
//! ```
#![cfg(feature = "std")]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{
    is_raw_audio, parse_launch, run_graph, ElementFactory, GraphNode, Registry,
};
use g2g_core::{
    AsyncElement, AudioFormat, Bus, BusMessage, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, Graph, OutputSink, PadTemplate, PadTemplates, PipelineClock,
    PushOutcome, Tag, TagList, ANY_CHANNELS, ANY_SAMPLE_RATE,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::id3demux::Id3Demux;
use g2g_plugins::mpegaudioparse::MpegAudioParse;
use g2g_plugins::registry::default_registry;

mod ffprobe;
use ffprobe::{fixture_path, Probe};

/// The stereo fixture: an ID3v2.3 tag, a Xing header frame, then the audio.
const STEREO_FIXTURE: &str = "mp3_stereo_44100_id3v2.mp3";
/// The mono fixture: an ID3v2.4 tag and an ID3v1 trailer around the audio.
const MONO_FIXTURE: &str = "mp3_mono_44100_id3v1.mp3";

/// Bytes per input buffer when the parser is driven directly: an odd size, so
/// frames straddle buffer boundaries the way `filesrc` chunks make them.
const CHUNK_LEN: usize = 997;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Caps an MPEG audio stream is declared with before a frame header refines it.
fn mp3_sentinel_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Mp3,
        channels: 0,
        sample_rate: 0,
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
async fn feed<E: AsyncElement>(element: &mut E, bytes: &[u8], sink: &mut CaptureSink) {
    element
        .configure_pipeline(&mp3_sentinel_caps())
        .expect("MPEG audio caps are accepted");
    for piece in bytes.chunks(CHUNK_LEN) {
        element
            .process(chunk(piece), sink)
            .await
            .expect("the stream parses");
    }
    element
        .process(PipelinePacket::Eos, sink)
        .await
        .expect("the tail flushes");
}

/// The tags of every [`BusMessage::Tag`] posted, in order.
fn posted_tags(bus: &Bus) -> Vec<TagList> {
    let mut posted = Vec::new();
    while let Some(message) = bus.try_recv() {
        if let BusMessage::Tag { tags, .. } = message {
            posted.push(tags);
        }
    }
    posted
}

#[tokio::test]
async fn launch_line_types_mp3_and_parses_every_frame() {
    let probe = Probe::read(STEREO_FIXTURE);
    let reg = default_registry();
    let line = format!(
        "filesrc location={} ! typefind ! mpegaudioparse ! fakesink",
        fixture_path(STEREO_FIXTURE).display()
    );
    let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    assert_eq!(
        stats.frames_consumed, probe.frames,
        "one buffer per MPEG audio frame, the Xing header frame dropped"
    );
}

#[tokio::test]
async fn id3demux_ahead_of_the_parser_yields_the_same_frames() {
    let probe = Probe::read(MONO_FIXTURE);
    let reg = default_registry();
    let line = format!(
        "filesrc location={} ! id3demux ! mpegaudioparse ! fakesink",
        fixture_path(MONO_FIXTURE).display()
    );
    let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    assert_eq!(stats.frames_consumed, probe.frames);
}

#[tokio::test]
async fn parsed_frames_carry_the_probed_caps_timing_and_tags() {
    let probe = Probe::read(STEREO_FIXTURE);
    let bytes = std::fs::read(fixture_path(STEREO_FIXTURE)).expect("the fixture is checked in");
    let (bus, handle) = Bus::new(8);
    let mut parser = MpegAudioParse::new().with_bus(handle);
    let mut sink = CaptureSink::default();
    feed(&mut parser, &bytes, &mut sink).await;

    assert_eq!(sink.frames.len() as u64, probe.frames);
    assert_eq!(
        sink.caps,
        vec![probe.audio_caps(AudioFormat::Mp3)],
        "the concrete caps are announced once, before the first frame"
    );

    // Presentation time runs from a sample counter, so every frame is one frame
    // duration past the one before it.
    let mut previous = None;
    for (i, frame) in sink.frames.iter().enumerate() {
        let timing = frame.timing;
        assert!(timing.duration_ns > 0, "frame {i} has a duration");
        if let Some(prior) = previous {
            assert!(timing.pts_ns > prior, "frame {i} pts moves forward");
        }
        previous = Some(timing.pts_ns);
    }
    let last = sink.frames.last().expect("frames").timing;
    let stream_ns = last.pts_ns + last.duration_ns;
    let probed_ns = (probe.duration_seconds * 1e9) as u64;
    assert!(
        stream_ns.abs_diff(probed_ns) <= 2 * last.duration_ns,
        "the parsed timeline spans the probed duration: {stream_ns} vs {probed_ns}"
    );

    let posted = posted_tags(&bus);
    assert_eq!(posted.len(), 1, "the tags are posted once");
    assert_eq!(
        posted[0].tags()[..2],
        [
            Tag::Title(probe.text("title")),
            Tag::Artist(probe.text("artist"))
        ],
        "the ID3v2 text frames reach the bus"
    );
    assert!(
        posted[0]
            .tags()
            .iter()
            .any(|t| matches!(t, Tag::Encoder(_))),
        "the fixture's TSSE frame maps to the typed encoder tag"
    );
}

#[tokio::test]
async fn id3demux_posts_the_tags_and_the_parser_then_finds_none() {
    let probe = Probe::read(MONO_FIXTURE);
    let bytes = std::fs::read(fixture_path(MONO_FIXTURE)).expect("the fixture is checked in");

    let (demux_bus, demux_handle) = Bus::new(8);
    let mut demux = Id3Demux::new().with_bus(demux_handle);
    let mut stripped = CaptureSink::default();
    feed(&mut demux, &bytes, &mut stripped).await;

    let (parser_bus, parser_handle) = Bus::new(8);
    let mut parser = MpegAudioParse::new().with_bus(parser_handle);
    let mut parsed = CaptureSink::default();
    let payload: Vec<u8> = stripped
        .frames
        .iter()
        .flat_map(|f| f.domain.as_system_slice().expect("system bytes").to_vec())
        .collect();
    feed(&mut parser, &payload, &mut parsed).await;

    assert_eq!(parsed.frames.len() as u64, probe.frames);
    assert_eq!(parsed.caps, vec![probe.audio_caps(AudioFormat::Mp3)]);
    let demuxed = posted_tags(&demux_bus);
    assert_eq!(demuxed.len(), 1, "id3demux posts the tags once");
    assert_eq!(demuxed[0].tags()[0], Tag::Title(probe.text("title")));
    assert_eq!(demuxed[0].tags()[1], Tag::Artist(probe.text("artist")));
    assert!(
        posted_tags(&parser_bus).is_empty(),
        "the stripped stream carries no tags, so nothing is posted twice"
    );
}

/// Registry name of the stub decoder below.
const STUB_DECODER: &str = "m1065-stub-audio-decoder";

/// A decoder that turns MPEG audio into PCM, for the auto-plug search to find
/// in a build with no real audio decoder compiled in. Never run.
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
        Vec::from([
            // The sentinel shape a parsed MPEG audio stream is declared with,
            // as `ffmpegaudiodec` declares it.
            PadTemplate::sink(CapsSet::one(Caps::Audio {
                format: AudioFormat::Mp3,
                channels: ANY_CHANNELS,
                sample_rate: 0,
            })),
            PadTemplate::source(CapsSet::one(stub_pcm_caps())),
        ])
    }
}

#[test]
fn decodebin_splices_the_parser_ahead_of_the_decoder() {
    let mut reg: Registry = default_registry();
    assert_eq!(
        reg.parser_name(&mp3_sentinel_caps()),
        Some("mpegaudioparse"),
        "the decode chain's parser for MPEG audio"
    );
    reg.register(ElementFactory::of::<StubAudioDec>(STUB_DECODER, |_| {
        Box::new(StubAudioDec)
    }));

    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(FileSrc::new(
        fixture_path(STEREO_FIXTURE),
        mp3_sentinel_caps(),
    )));
    let sink = graph.add_sink(GraphNode::element(FakeSink::new()));
    let inserted = reg
        .decodebin(
            &mut graph,
            src,
            sink,
            &mp3_sentinel_caps(),
            &is_raw_audio,
            4,
        )
        .expect("MPEG audio reaches PCM");
    assert_eq!(inserted.len(), 2, "the parser plus a decoder");
    assert_eq!(
        graph
            .element(inserted[0])
            .expect("the spliced parser")
            .log_category(),
        g2g_core::log::short_type_name::<MpegAudioParse>(),
        "the parser leads the chain"
    );
}

/// Decoding needs a real decoder, which only the `ffmpeg` build has.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
mod decode {
    use super::*;
    use g2g_plugins::ffmpegaudiodec::FfmpegAudioDec;

    /// Bytes per interleaved `PcmS16Le` sample.
    const BYTES_PER_SAMPLE: usize = 2;
    /// Samples one MPEG-1 Layer III frame decodes to.
    const SAMPLES_PER_FRAME: usize = 1152;

    /// The whole GStreamer-shaped line: content typing, the auto-plugged
    /// parser, and a real decoder.
    #[tokio::test]
    async fn decodebin_line_plays_an_mp3() {
        let probe = Probe::read(STEREO_FIXTURE);
        let reg = default_registry();
        let line = format!(
            "filesrc location={} ! decodebin ! fakesink",
            fixture_path(STEREO_FIXTURE).display()
        );
        let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
        let stats = run_graph(graph, &ZeroClock, 4)
            .await
            .expect("the pipeline runs");
        assert!(
            stats.frames_consumed >= probe.frames,
            "decoded buffers reached the sink, got {}",
            stats.frames_consumed
        );
    }

    #[tokio::test]
    async fn decodes_to_pcm_of_the_probed_length() {
        let probe = Probe::read(STEREO_FIXTURE);
        let bytes = std::fs::read(fixture_path(STEREO_FIXTURE)).expect("the fixture is checked in");
        let mut parser = MpegAudioParse::new();
        let mut parsed = CaptureSink::default();
        feed(&mut parser, &bytes, &mut parsed).await;

        let mut decoder = FfmpegAudioDec::new();
        decoder
            .configure_pipeline(&probe.audio_caps(AudioFormat::Mp3))
            .expect("the decoder takes the parsed caps");
        let mut pcm = CaptureSink::default();
        for frame in parsed.frames {
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
        let samples = decoded_bytes / (BYTES_PER_SAMPLE * probe.channels as usize);
        assert_eq!(
            samples,
            probe.frames as usize * SAMPLES_PER_FRAME,
            "every parsed frame decoded"
        );
        // The encoder's own delay and padding stretch the coded stream past the
        // nominal duration, by less than a frame at each end.
        let expected = (probe.duration_seconds * probe.sample_rate as f64) as usize;
        assert!(
            samples.abs_diff(expected) <= 2 * SAMPLES_PER_FRAME,
            "{samples} decoded samples against {expected} probed"
        );
    }
}
