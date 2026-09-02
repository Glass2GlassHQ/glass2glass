//! M898 - subtitle track muxing: `MkvMuxN` writes a text input as an `S_TEXT/*`
//! Matroska track (cues as `BlockGroup`s carrying a `BlockDuration`) and `Mp4MuxN`
//! writes one as a `tx3g` (`mov_text`) track, gaps between cues filled with the
//! empty samples the format uses for "no subtitle on screen".
//!
//! The cues come from `SubParse` over an SRT snippet, so the whole write path runs
//! for real. Legs: a round trip through our own demuxers (text, PTS and duration
//! exact), ffprobe / ffmpeg over a g2g-muxed file, and our demuxers over an
//! ffmpeg-muxed one. The ffmpeg legs self-skip when the binary is absent.
#![cfg(feature = "std")]

use std::process::Command;

use g2g_core::element::AsyncElement;
use g2g_core::fanout::MultiOutputSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement,
    MultiOutputElement, OutputSink, PropValue, PushOutcome, Rate, Tag, TagList, TextFormat,
    VideoCodec,
};
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::matroska::{MatroskaDemuxer, MkvCodec};
use g2g_plugins::mkvdemux::{MkvDemuxN, MkvStream};
use g2g_plugins::mkvmuxn::MkvMuxN;
use g2g_plugins::mp4demuxn::{Mp4DemuxN, Mp4Port};
use g2g_plugins::mp4muxn::Mp4MuxN;
use g2g_plugins::subparse::SubParse;

/// The cue source for every leg: two cues with a gap between them and a start
/// past zero, so the gap handling each container needs is exercised.
const SRT: &str = "1\n\
00:00:01,000 --> 00:00:03,500\n\
Hello world\n\
\n\
2\n\
00:00:05,000 --> 00:00:06,000\n\
Second cue\n";

/// `(text, pts_ns, duration_ns)` of the SRT above, what every round trip must
/// recover.
fn expected_cues() -> Vec<(String, u64, u64)> {
    vec![
        ("Hello world".into(), 1_000_000_000, 2_500_000_000),
        ("Second cue".into(), 5_000_000_000, 1_000_000_000),
    ]
}

// --- sinks -----------------------------------------------------------------

#[derive(Default)]
struct Collect {
    bytes: Vec<u8>,
}
impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.bytes.extend_from_slice(s);
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Collects each pushed frame's bytes whole (the access units of a parsed
/// elementary stream).
#[derive(Default)]
struct AuSink {
    aus: Vec<Vec<u8>>,
}
impl OutputSink for AuSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.aus.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// One demuxed frame's payload and timing, the shape every assertion compares.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Cue {
    text: String,
    pts_ns: u64,
    duration_ns: u64,
}

#[derive(Default)]
struct CueSink {
    frames: Vec<Cue>,
}
impl OutputSink for CueSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push(Cue {
                        text: String::from_utf8_lossy(s).into_owned(),
                        pts_ns: f.timing.pts_ns,
                        duration_ns: f.timing.duration_ns,
                    });
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// A multi-port tap recording each port's frames with their timing.
struct PortTap {
    ports: Vec<Vec<Cue>>,
    caps: Vec<Option<Caps>>,
}
impl PortTap {
    fn new(ports: usize) -> Self {
        Self {
            ports: vec![Vec::new(); ports],
            caps: vec![None; ports],
        }
    }
}
impl MultiOutputSink for PortTap {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.ports[port].push(Cue {
                            text: String::from_utf8_lossy(s).into_owned(),
                            pts_ns: f.timing.pts_ns,
                            duration_ns: f.timing.duration_ns,
                        });
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps[port] = Some(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
    fn port_count(&self) -> usize {
        self.ports.len()
    }
}

// --- inputs ----------------------------------------------------------------

fn text_caps() -> Caps {
    Caps::Text {
        format: TextFormat::Utf8,
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn aac_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48_000,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

fn frame(data: Vec<u8>, pts_ns: u64, duration_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

/// The video the muxed file carries beside the cues: the access units and the
/// interval between them.
struct Video {
    aus: Vec<Vec<u8>>,
    step_ns: u64,
}

/// A minimal H.264 IDR access unit (SPS + PPS + IDR), enough for both muxers to
/// build a track. Not decodable, which is fine for our own demuxers.
fn idr() -> Vec<u8> {
    annexb(&[
        &[0x67, 0x42, 0x00, 0x1e, 0x88],
        &[0x68, 0xce, 0x3c, 0x80],
        &[0x65, 0x88, 0x84, 0x00],
    ])
}

/// Six one-second stand-in access units, spanning the cues.
fn stub_video() -> Video {
    Video {
        aus: vec![idr(); 6],
        step_ns: 1_000_000_000,
    }
}

/// A minimal ADTS AAC access unit (7-byte header + payload) at 48 kHz stereo.
fn adts_au(payload: &[u8]) -> Vec<u8> {
    let frame_len = payload.len() + 7;
    let mut au = vec![
        0xFF,
        0xF1,
        (1 << 6) | (3 << 2),
        ((2 & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F,
        0xFC,
    ];
    au.extend_from_slice(payload);
    au
}

/// Run the real `SubParse` over the SRT snippet and return its timed cues, which
/// is what a `filesrc ! subparse ! <mux>` graph puts on the muxer's text pad.
async fn parsed_cues() -> Vec<Cue> {
    let mut parse = SubParse::new();
    parse
        .configure_pipeline(&Caps::Text {
            format: TextFormat::Srt,
        })
        .expect("configure subparse");
    let mut sink = CueSink::default();
    parse
        .process(frame(SRT.as_bytes().to_vec(), 0, 0), &mut sink)
        .await
        .unwrap();
    parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.frames
}

// --- muxing ----------------------------------------------------------------

/// Mux one video track and one text track into Matroska. Input 0 is video, input
/// 1 the cues; `tags` (if any) become the text track's `TrackEntry` metadata.
async fn mux_mkv(
    format: TextFormat,
    tags: Option<TagList>,
    cues: &[Cue],
    video: &Video,
) -> Vec<u8> {
    let mut mux = MkvMuxN::new(2).with_subtitle_format(format);
    if let Some(tags) = tags {
        mux = mux.with_track_tags(1, tags);
    }
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &text_caps()).unwrap();
    let mut sink = Collect::default();
    for (i, au) in video.aus.iter().enumerate() {
        mux.process(0, frame(au.clone(), i as u64 * video.step_ns, 0), &mut sink)
            .await
            .unwrap();
    }
    for cue in cues {
        mux.process(
            1,
            frame(cue.text.as_bytes().to_vec(), cue.pts_ns, cue.duration_ns),
            &mut sink,
        )
        .await
        .unwrap();
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    sink.bytes
}

/// Mux one video track and one text track into a progressive MP4 (real sample
/// tables, which is what carries the cue timing a reader reports).
async fn mux_mp4(cues: &[Cue], video: &Video) -> Vec<u8> {
    mux_mp4_layout(cues, video, false).await
}

/// The same two tracks in either layout: progressive (real sample tables) or
/// fragmented (`moof` `trun`s carrying the timing).
async fn mux_mp4_layout(cues: &[Cue], video: &Video, fragmented: bool) -> Vec<u8> {
    let mut mux = Mp4MuxN::new(2).with_fragmented(fragmented);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &text_caps()).unwrap();
    let mut sink = Collect::default();
    for (i, au) in video.aus.iter().enumerate() {
        mux.process(
            0,
            frame(au.clone(), i as u64 * video.step_ns, video.step_ns),
            &mut sink,
        )
        .await
        .unwrap();
    }
    for cue in cues {
        mux.process(
            1,
            frame(cue.text.as_bytes().to_vec(), cue.pts_ns, cue.duration_ns),
            &mut sink,
        )
        .await
        .unwrap();
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    sink.bytes
}

fn data_frame(bytes: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

/// Demux a Matroska file into `ports`, returning each port's frames.
async fn demux_mkv(file: &[u8], ports: Vec<MkvStream>) -> PortTap {
    let n = ports.len();
    let mut demux = MkvDemuxN::new(ports);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure");
    let mut tap = PortTap::new(n);
    demux.process(data_frame(file), &mut tap).await.unwrap();
    demux.process(PipelinePacket::Eos, &mut tap).await.unwrap();
    tap
}

/// Demux an MP4 file into a port per `track_id`, returning each port's frames.
async fn demux_mp4(file: &[u8], ports: Vec<Mp4Port>) -> PortTap {
    let n = ports.len();
    let mut demux = Mp4DemuxN::new(ports);
    let mut tap = PortTap::new(n);
    demux.process(data_frame(file), &mut tap).await.unwrap();
    demux.process(PipelinePacket::Eos, &mut tap).await.unwrap();
    tap
}

/// The cues of a text port with the empty gap samples dropped: what was actually
/// on screen.
fn spoken(frames: &[Cue]) -> Vec<Cue> {
    frames
        .iter()
        .filter(|c| !c.text.is_empty())
        .cloned()
        .collect()
}

// --- our own round trips ---------------------------------------------------

/// The SRT cues survive a Matroska round trip exactly: text, PTS and the display
/// duration the `BlockDuration` carries (a `SimpleBlock` has nowhere to put one,
/// so this fails if the muxer writes the cue as one).
#[tokio::test]
async fn matroska_text_track_round_trips_with_cue_durations() {
    let cues = parsed_cues().await;
    assert_eq!(
        cues.iter()
            .map(|c| (c.text.clone(), c.pts_ns, c.duration_ns))
            .collect::<Vec<_>>(),
        expected_cues(),
        "SubParse produced the cues the muxer is fed"
    );

    let file = mux_mkv(TextFormat::Utf8, None, &cues, &stub_video()).await;

    // The track is announced as timed text with the codec id the demuxer maps.
    let mut parser = MatroskaDemuxer::new();
    parser.push_data(&file);
    let tracks = parser.tracks();
    assert_eq!(tracks.len(), 2, "video + text tracks: {tracks:?}");
    assert_eq!(tracks[1].codec, MkvCodec::Subtitle(TextFormat::Utf8));
    assert!(
        file.windows(11).any(|w| w == b"S_TEXT/UTF8"),
        "the S_TEXT/UTF8 CodecID is written"
    );

    let tap = demux_mkv(
        &file,
        vec![MkvStream::H264, MkvStream::Subtitle(TextFormat::Utf8)],
    )
    .await;
    assert_eq!(tap.caps[1], Some(text_caps()), "the text port retypes");
    assert_eq!(tap.ports[1], cues, "cue text, PTS and duration survive");
    assert_eq!(tap.ports[0].len(), 6, "the video track still demuxes");
}

/// The storage syntax is the muxer's choice, not the pad's: `ass` frames each cue
/// as the `Dialogue` fields the Matroska ASS mapping defines, behind the script
/// header `CodecPrivate` a reader needs to interpret them, and de-frames back to
/// the same plain text.
#[tokio::test]
async fn the_ass_storage_format_round_trips_to_the_same_cues() {
    let cues = parsed_cues().await;

    let ass = mux_mkv(TextFormat::Ssa, None, &cues, &stub_video()).await;
    let mut parser = MatroskaDemuxer::new();
    parser.push_data(&ass);
    assert_eq!(
        parser.tracks()[1].codec,
        MkvCodec::Subtitle(TextFormat::Ssa)
    );
    assert!(
        ass.windows(10).any(|w| w == b"S_TEXT/ASS"),
        "the S_TEXT/ASS CodecID is written"
    );
    let private = parser
        .codec_private(2)
        .expect("the ASS track carries a script header");
    let header = String::from_utf8_lossy(private);
    assert!(
        header.contains("[Events]")
            && header.contains(
                "Format: ReadOrder, Layer, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
            ),
        "the CodecPrivate names the block fields: {header}"
    );
    // The blocks are the ASS event fields, with a rising ReadOrder.
    assert!(
        ass.windows(21).any(|w| w == b"0,0,Default,,0,0,0,,H"),
        "the first cue is framed as ASS event fields"
    );
    assert!(
        ass.windows(21).any(|w| w == b"1,0,Default,,0,0,0,,S"),
        "the second cue's ReadOrder rises"
    );
    let tap = demux_mkv(
        &ass,
        vec![MkvStream::H264, MkvStream::Subtitle(TextFormat::Ssa)],
    )
    .await;
    assert_eq!(tap.ports[1], cues, "ASS blocks de-frame to the same cues");

    // An ASS event is one line, so a cue's own line breaks are written as `\N`.
    let multiline = vec![Cue {
        text: "two\nlines".into(),
        pts_ns: 1_000_000_000,
        duration_ns: 1_000_000_000,
    }];
    let ass = mux_mkv(TextFormat::Ssa, None, &multiline, &stub_video()).await;
    assert!(
        ass.windows(30)
            .any(|w| w == b"0,0,Default,,0,0,0,,two\\Nlines"),
        "the cue's line break is written as the ASS escape"
    );
    let tap = demux_mkv(
        &ass,
        vec![MkvStream::H264, MkvStream::Subtitle(TextFormat::Ssa)],
    )
    .await;
    assert_eq!(tap.ports[1], multiline, "and comes back as a line break");

    // WebVTT is not offered: ffmpeg maps only the WebM `D_WEBVTT/*` ids, whose
    // block leads with the cue identifier and settings, so an `S_TEXT/WEBVTT`
    // track would be a carriage only this codebase reads back.
    let plain = MkvMuxN::new(2).with_subtitle_format(TextFormat::WebVtt);
    assert_eq!(
        plain.get_property("subtitle-format"),
        Some(PropValue::Str("utf8".into())),
        "an unwritable syntax leaves the default"
    );
}

/// `subtitle-format` is settable at runtime, the path `parse_launch` takes.
#[test]
fn subtitle_format_is_a_runtime_property() {
    let mut mux = MkvMuxN::new(2);
    assert_eq!(
        mux.get_property("subtitle-format"),
        Some(PropValue::Str("utf8".into()))
    );
    mux.set_property("subtitle-format", PropValue::Str("ass".into()))
        .expect("ass is a storage syntax");
    assert_eq!(
        mux.get_property("subtitle-format"),
        Some(PropValue::Str("ass".into()))
    );
    assert!(
        mux.set_property("subtitle-format", PropValue::Str("webvtt".into()))
            .is_err(),
        "a syntax the muxer cannot write is refused, not ignored"
    );
    // A document-format text pad carries whole-file bytes, not cues: refused.
    assert!(mux
        .intercept_caps(
            1,
            &Caps::Text {
                format: TextFormat::Srt
            }
        )
        .is_err());
    assert!(mux.intercept_caps(1, &text_caps()).is_ok());
}

/// The MP4 side: cues become `tx3g` samples and come back with their timing. A
/// text sample presents where the durations before it end, so the runs with no
/// cue on screen must be filled with empty samples, or every cue after a gap
/// would show early.
#[tokio::test]
async fn mp4_tx3g_track_round_trips_with_gap_samples() {
    let cues = parsed_cues().await;
    let file = mux_mp4(&cues, &stub_video()).await;

    let subtitles = g2g_plugins::mp4demuxn::subtitle_streams(&file);
    assert_eq!(subtitles.len(), 1, "one text track: {subtitles:?}");
    assert_eq!(subtitles[0].track_id, 2);
    assert_eq!(subtitles[0].caps, text_caps());
    assert!(
        file.windows(4).any(|w| w == b"tx3g"),
        "the tx3g sample entry is written"
    );

    let tap = demux_mp4(
        &file,
        vec![
            Mp4Port {
                track_id: 1,
                caps: h264_caps(),
            },
            Mp4Port {
                track_id: 2,
                caps: text_caps(),
            },
        ],
    )
    .await;
    assert_eq!(
        spoken(&tap.ports[1]),
        cues,
        "cue text, PTS and duration survive"
    );
    assert_eq!(
        tap.ports[1]
            .iter()
            .map(|c| (c.text.as_str(), c.pts_ns, c.duration_ns))
            .collect::<Vec<_>>(),
        vec![
            ("", 0, 1_000_000_000),
            ("Hello world", 1_000_000_000, 2_500_000_000),
            ("", 3_500_000_000, 1_500_000_000),
            ("Second cue", 5_000_000_000, 1_000_000_000),
        ],
        "the lead-in and the inter-cue gap are empty samples"
    );
    assert_eq!(tap.ports[0].len(), 6, "the video track still demuxes");

    // The fragmented layout carries the same cues: there the timing rides each
    // fragment's `tfdt` + `trun`, which the gap samples keep aligned to the cues.
    let fragmented = mux_mp4_layout(&cues, &stub_video(), true).await;
    let tap = demux_mp4(
        &fragmented,
        vec![
            Mp4Port {
                track_id: 1,
                caps: h264_caps(),
            },
            Mp4Port {
                track_id: 2,
                caps: text_caps(),
            },
        ],
    )
    .await;
    assert_eq!(
        spoken(&tap.ports[1]),
        cues,
        "the fragmented layout carries the same cues"
    );
}

/// A text track carries the same per-track metadata as any other: its title and
/// language ride the `TrackEntry` (M788), where a player reads them.
#[tokio::test]
async fn text_track_carries_its_language_and_title() {
    let cues = parsed_cues().await;
    let tags: TagList = [Tag::Title("Captions".into()), Tag::Language("fra".into())]
        .into_iter()
        .collect();
    let file = mux_mkv(TextFormat::Utf8, Some(tags.clone()), &cues, &stub_video()).await;

    let (bus, handle) = Bus::new(64);
    let mut demux = MkvDemuxN::new(vec![MkvStream::H264, MkvStream::Subtitle(TextFormat::Utf8)])
        .with_bus(handle);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure");
    let mut tap = PortTap::new(2);
    demux.process(data_frame(&file), &mut tap).await.unwrap();

    let mut text_tags = Vec::new();
    while let Some(msg) = bus.try_recv() {
        if let BusMessage::StreamTag { stream_id, tags } = msg {
            if stream_id == "matroska-track-2" {
                text_tags.extend(tags.tags().iter().cloned());
            }
        }
    }
    assert_eq!(text_tags, tags.tags(), "the text track's own metadata");
}

/// One file, three tracks: video, audio and text all demux back with the right
/// types and payloads.
#[tokio::test]
async fn video_audio_and_text_share_one_matroska_file() {
    let cues = parsed_cues().await;
    let mut mux = MkvMuxN::new(3);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    mux.configure_pipeline(2, &text_caps()).unwrap();
    let mut sink = Collect::default();
    for i in 0..6u64 {
        mux.process(0, frame(idr(), i * 1_000_000_000, 0), &mut sink)
            .await
            .unwrap();
        mux.process(
            1,
            frame(adts_au(&[0xA1, 0xA2]), i * 1_000_000_000, 0),
            &mut sink,
        )
        .await
        .unwrap();
    }
    for cue in &cues {
        mux.process(
            2,
            frame(cue.text.as_bytes().to_vec(), cue.pts_ns, cue.duration_ns),
            &mut sink,
        )
        .await
        .unwrap();
    }
    for input in 0..3 {
        mux.process(input, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
    }

    let mut parser = MatroskaDemuxer::new();
    parser.push_data(&sink.bytes);
    let codecs: Vec<MkvCodec> = parser.tracks().iter().map(|t| t.codec).collect();
    assert_eq!(
        codecs,
        vec![
            MkvCodec::H264,
            MkvCodec::Aac,
            MkvCodec::Subtitle(TextFormat::Utf8)
        ]
    );

    let tap = demux_mkv(
        &sink.bytes,
        vec![
            MkvStream::H264,
            MkvStream::Aac,
            MkvStream::Subtitle(TextFormat::Utf8),
        ],
    )
    .await;
    assert_eq!(tap.ports[0].len(), 6, "video");
    assert_eq!(tap.ports[1].len(), 6, "audio");
    assert_eq!(tap.ports[2], cues, "text");
}

// --- reference peer --------------------------------------------------------

fn have(tool: &str) -> bool {
    Command::new(tool).arg("-version").output().is_ok()
}

/// Real H.264 access units for the files ffmpeg reads back: an ffmpeg-encoded
/// elementary stream split by the real `H264Parse`. The stand-in access units
/// above are not decodable, and ffmpeg then takes the file's start time from the
/// subtitle track alone, shifting every extracted cue by the first cue's PTS.
async fn reference_video(dir: &std::path::Path) -> Video {
    let path = dir.join("g2g-m898-ref.h264");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10:duration=6",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            "10",
            "-f",
            "h264",
        ])
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg encoded the reference video");
    let bytes = std::fs::read(&path).expect("read the reference video");
    let _ = std::fs::remove_file(&path);

    let mut parse = H264Parse::reframing();
    parse.configure_pipeline(&h264_caps()).expect("configure");
    let mut sink = AuSink::default();
    parse.process(frame(bytes, 0, 0), &mut sink).await.unwrap();
    parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    assert!(
        sink.aus.len() > 10,
        "the reference stream split into access units"
    );
    Video {
        aus: sink.aus,
        step_ns: 100_000_000,
    }
}

/// `ffprobe -show_entries stream=index,codec_name,codec_type` over `path`, one
/// `stream|...` line per stream.
fn probe_streams(path: &std::path::Path) -> Vec<String> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_name,codec_type",
            "-of",
            "compact",
        ])
        .arg(path)
        .output()
        .expect("run ffprobe");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "ffprobe read the file: {text}");
    text.lines()
        .filter(|l| l.starts_with("stream|"))
        .map(str::to_string)
        .collect()
}

/// Extract the subtitle stream of `path` as SRT with ffmpeg, returning the file.
fn extract_srt(path: &std::path::Path, out_path: &std::path::Path) -> String {
    let status = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-map", "0:s:0", "-f", "srt"])
        .arg(out_path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg extracted the subtitle stream");
    std::fs::read_to_string(out_path).expect("read the extracted srt")
}

/// The oracle for both containers: ffprobe must see the subtitle stream with the
/// codec each mapping implies, and ffmpeg must extract cues whose text and timing
/// are the ones muxed in.
#[tokio::test]
async fn ffmpeg_reads_the_cues_back_from_a_g2g_muxed_file() {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("ffmpeg/ffprobe not present; skipping the reference-peer read of a g2g file");
        return;
    }
    let cues = parsed_cues().await;
    let dir = std::env::temp_dir();
    let video = reference_video(&dir).await;

    for (name, codec, bytes) in [
        (
            "g2g-m898-utf8.mkv",
            "subrip",
            mux_mkv(TextFormat::Utf8, None, &cues, &video).await,
        ),
        (
            "g2g-m898-ass.mkv",
            "ass",
            mux_mkv(TextFormat::Ssa, None, &cues, &video).await,
        ),
        ("g2g-m898.mp4", "mov_text", mux_mp4(&cues, &video).await),
    ] {
        let path = dir.join(name);
        std::fs::write(&path, &bytes).expect("write the muxed file");
        let streams = probe_streams(&path);
        assert_eq!(streams.len(), 2, "{name}: video + subtitle: {streams:?}");
        assert!(
            streams[1].contains("codec_type=subtitle")
                && streams[1].contains(&format!("codec_name={codec}")),
            "{name}: the subtitle stream's codec: {streams:?}"
        );

        let srt = extract_srt(&path, &dir.join(format!("{name}.srt")));
        for (text, start, end) in [
            ("Hello world", "00:00:01,000", "00:00:03,500"),
            ("Second cue", "00:00:05,000", "00:00:06,000"),
        ] {
            assert!(
                srt.contains(text),
                "{name}: the cue text survives ffmpeg's read: {srt}"
            );
            assert!(
                srt.contains(&format!("{start} --> {end}")),
                "{name}: the cue window survives ffmpeg's read: {srt}"
            );
        }
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(dir.join(format!("{name}.srt")));
    }
}

/// The other direction: our demuxers read the subtitle track of a file ffmpeg
/// muxed from the same SRT, so the framing is the shared one and not a private
/// convention.
#[tokio::test]
async fn our_demuxers_read_an_ffmpeg_muxed_subtitle_track() {
    if !have("ffmpeg") {
        eprintln!("ffmpeg not present; skipping the reference-peer muxed file");
        return;
    }
    let dir = std::env::temp_dir();
    let srt_path = dir.join("g2g-m898-ref.srt");
    std::fs::write(&srt_path, SRT).expect("write the srt");
    let cues = parsed_cues().await;

    let mux = |name: &str, codec: &str| -> Vec<u8> {
        let path = dir.join(name);
        let status = Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=10:duration=6",
            ])
            .arg("-i")
            .arg(&srt_path)
            .args([
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:s",
                codec,
                "-shortest",
            ])
            .arg(&path)
            .status()
            .expect("run ffmpeg");
        assert!(status.success(), "ffmpeg muxed {name}");
        let bytes = std::fs::read(&path).expect("read the reference file");
        let _ = std::fs::remove_file(&path);
        bytes
    };

    let mkv = mux("g2g-m898-ref.mkv", "srt");
    let tap = demux_mkv(
        &mkv,
        vec![MkvStream::H264, MkvStream::Subtitle(TextFormat::Utf8)],
    )
    .await;
    assert_eq!(
        spoken(&tap.ports[1]),
        cues,
        "the cues of an ffmpeg-muxed Matroska"
    );

    let mp4 = mux("g2g-m898-ref.mp4", "mov_text");
    let subtitles = g2g_plugins::mp4demuxn::subtitle_streams(&mp4);
    assert_eq!(subtitles.len(), 1, "ffmpeg wrote one text track");
    let tap = demux_mp4(
        &mp4,
        vec![Mp4Port {
            track_id: subtitles[0].track_id,
            caps: text_caps(),
        }],
    )
    .await;
    assert_eq!(
        spoken(&tap.ports[0]),
        cues,
        "the cues of an ffmpeg-muxed MP4"
    );

    let _ = std::fs::remove_file(&srt_path);
}
