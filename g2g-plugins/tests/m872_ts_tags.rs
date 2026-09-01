//! M872 - MPEG-TS tag carriage. A transport stream has no free-form tag element,
//! so g2g carries the two things TS standardizes: the SDT `service_descriptor`
//! (service name + provider) for the whole service, and the PMT's
//! `ISO_639_language_descriptor` per elementary stream. `TsMux` writes both from
//! `with_tags` / `with_track_tags`; the demuxers post them as `BusMessage::Tag`
//! and `BusMessage::StreamTag` on the `mpegts-pid-{pid}` ids the
//! `StreamCollection` uses.
//!
//! Three legs: a g2g round trip, ffprobe reading the g2g-muxed stream (the
//! oracle), and g2g demuxing an ffmpeg-authored stream (the direction that pins
//! the key mapping to what real TS ecosystems produce). The ffmpeg legs self-skip
//! when the binary is absent.
#![cfg(feature = "std")]

use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::fanout::MultiOutputSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement,
    MultiOutputElement, OutputSink, PushOutcome, Rate, Tag, TagList, VideoCodec,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::tsdemux::{TsDemuxN, TsStream};
use g2g_plugins::tsmuxn::TsMux;

/// The stream ids the mux's PID layout gives its two elementary streams.
const VIDEO_ID: &str = "mpegts-pid-256";
const AUDIO_ID: &str = "mpegts-pid-257";

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
    }
}

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

/// A `MultiOutputSink` that counts what each port received (this milestone is
/// about the bus; the frames only prove the demux ran).
#[derive(Default)]
struct PortTap {
    frames: Vec<usize>,
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
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                self.frames[port] += 1;
            }
            Ok(PushOutcome::Accepted)
        })
    }
    fn port_count(&self) -> usize {
        self.frames.len()
    }
}

fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
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

fn tags(list: &[Tag]) -> TagList {
    list.iter().cloned().collect()
}

/// The whole-service tags: the name under the typed `Tag::Title`, the provider
/// under ffprobe's own `service_provider` key (`Tag` has no typed variant for it).
fn service_tags() -> TagList {
    tags(&[
        Tag::Title("G2G News".into()),
        Tag::Other {
            key: "service_provider".into(),
            value: "G2G Broadcasting".into(),
        },
    ])
}
fn video_tags() -> TagList {
    tags(&[Tag::Language("deu".into())])
}
fn audio_tags() -> TagList {
    tags(&[Tag::Language("fra".into())])
}

/// Mux an A/V transport stream (H.264 + AAC) carrying the service tags and a
/// distinct language on each elementary stream.
async fn mux_av_with_tags() -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let mut mux = TsMux::new(2)
        .with_tags(service_tags())
        .with_track_tags(0, video_tags())
        .with_track_tags(1, audio_tags());
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = Collect::default();
    for i in 0..4u64 {
        mux.process(
            0,
            frame(annexb(&[&sps, &pps, &idr]), i * 33_000_000),
            &mut sink,
        )
        .await
        .unwrap();
        mux.process(
            1,
            frame(adts_au(&[0xA1, 0xA2, 0xA3]), i * 21_000_000),
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

/// Demux `ts` with a two-port `TsDemuxN` (port 0 = H.264, port 1 = AAC) and return
/// `(collection stream ids, service tags with their program scope, per-stream tags)`.
#[allow(clippy::type_complexity)]
async fn demux_bus_messages(
    ts: &[u8],
) -> (
    Vec<String>,
    Vec<(Option<u16>, TagList)>,
    Vec<(String, TagList)>,
) {
    let (bus, handle) = Bus::new(64);
    let mut demux = TsDemuxN::new(vec![TsStream::H264, TsStream::Aac]).with_bus(handle);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .expect("configure");
    let mut tap = PortTap { frames: vec![0, 0] };
    demux.process(data_frame(ts), &mut tap).await.unwrap();
    demux.process(PipelinePacket::Eos, &mut tap).await.unwrap();
    assert!(
        tap.frames[0] > 0 && tap.frames[1] > 0,
        "both elementary streams demuxed: {:?}",
        tap.frames
    );

    let (mut ids, mut global, mut per_stream) = (Vec::new(), Vec::new(), Vec::new());
    while let Some(msg) = bus.try_recv() {
        match msg {
            BusMessage::StreamCollection(c) => {
                ids = c.streams().iter().map(|s| s.id.to_string()).collect()
            }
            BusMessage::Tag { tags, program } => global.push((program, tags)),
            BusMessage::StreamTag { stream_id, tags } => per_stream.push((stream_id, tags)),
            _ => {}
        }
    }
    (ids, global, per_stream)
}

/// The tags posted for `stream_id`, flattened across messages.
fn tags_of(per_stream: &[(String, TagList)], stream_id: &str) -> Vec<Tag> {
    per_stream
        .iter()
        .filter(|(id, _)| id == stream_id)
        .flat_map(|(_, t)| t.tags().iter().cloned())
        .collect()
}

#[tokio::test]
async fn service_and_language_tags_round_trip_through_the_ts_elements() {
    let ts = mux_av_with_tags().await;
    let (ids, global, per_stream) = demux_bus_messages(&ts).await;

    assert_eq!(ids, vec![VIDEO_ID, AUDIO_ID]);
    assert_eq!(
        global,
        vec![(Some(1), service_tags())],
        "the SDT service text posts once, scoped to the program it names"
    );

    assert_eq!(
        tags_of(&per_stream, VIDEO_ID),
        video_tags().tags(),
        "the video stream's language lands on its own id"
    );
    assert_eq!(
        tags_of(&per_stream, AUDIO_ID),
        audio_tags().tags(),
        "the audio stream's language lands on its own id"
    );
    assert_eq!(
        per_stream.len(),
        2,
        "one StreamTag per language-carrying stream, posted once each"
    );
}

/// A stream with no tags posts none: the demuxer must not invent a service or a
/// language for a transport stream that carries neither table field.
#[tokio::test]
async fn an_untagged_stream_posts_no_tags() {
    let mut mux = TsMux::new(1);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    let mut sink = Collect::default();
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    for i in 0..2u64 {
        mux.process(0, frame(annexb(&[&idr]), i * 33_000_000), &mut sink)
            .await
            .unwrap();
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();

    let (bus, handle) = Bus::new(16);
    let mut demux = TsDemuxN::new(vec![TsStream::H264]).with_bus(handle);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .unwrap();
    let mut tap = PortTap { frames: vec![0] };
    demux
        .process(data_frame(&sink.bytes), &mut tap)
        .await
        .unwrap();
    let mut saw_collection = false;
    while let Some(msg) = bus.try_recv() {
        match msg {
            BusMessage::StreamCollection(_) => saw_collection = true,
            BusMessage::Tag { tags, .. } => panic!("no service to report: {tags:?}"),
            BusMessage::StreamTag { stream_id, tags } => {
                panic!("no language to report on {stream_id}: {tags:?}")
            }
            _ => {}
        }
    }
    assert!(saw_collection, "the collection still posts");
}

/// The ffprobe oracle: the reference implementation must read the g2g-written SDT
/// and language descriptors, reporting the service text on the program and each
/// language on its own stream. Records peer-tagged `Oracle` evidence for
/// `mpegtsmux` on success (the M619 pattern).
#[tokio::test]
async fn ffprobe_reports_the_service_and_stream_languages() {
    if Command::new("ffprobe").arg("-version").output().is_err() {
        eprintln!("ffprobe not present; skipping the MPEG-TS tag oracle");
        return;
    }
    // Dedicated freshly-truncated log standalone; append to a shared CI log when
    // $G2G_CONFORMANCE_LOG is already set.
    let external = std::env::var_os("G2G_CONFORMANCE_LOG");
    let log = match &external {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let l = std::env::temp_dir().join("g2g-conformance-m872.tsv");
            std::env::set_var("G2G_CONFORMANCE_LOG", &l);
            let _ = std::fs::remove_file(&l);
            l
        }
    };
    let ts = mux_av_with_tags().await;
    let path = std::env::temp_dir().join("g2g-m872-tags.ts");
    std::fs::write(&path, &ts).expect("write ts");

    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "program_tags:stream=index:stream_tags",
            "-of",
            "compact",
        ])
        .arg(&path)
        .output()
        .expect("run ffprobe");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "ffprobe read the stream: {text}");
    assert!(
        text.contains("service_name=G2G News"),
        "ffprobe reports the SDT service name on the program: {text}"
    );
    assert!(
        text.contains("service_provider=G2G Broadcasting"),
        "and the provider under the same key g2g reads it back from: {text}"
    );
    // The program section repeats its stream indices, so key on the tag itself.
    let languages: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("tag:language"))
        .collect();
    assert_eq!(languages.len(), 2, "one language per stream: {text}");
    assert!(
        languages[0].contains("index=0") && languages[0].contains("language=deu"),
        "the video stream's language descriptor: {text}"
    );
    assert!(
        languages[1].contains("index=1") && languages[1].contains("language=fra"),
        "the audio stream's language descriptor: {text}"
    );

    persist::record_evidence(
        "mpegtsmux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .detail("ffprobe reports the SDT service text and per-stream ISO 639 languages"),
    )
    .expect("record oracle evidence");
    let report = persist::full_report();
    assert!(
        report.records.iter().any(|r| r.element == "mpegtsmux"),
        "mpegtsmux present after persisting evidence"
    );

    if external.is_none() {
        let _ = std::fs::remove_file(&log);
    }
    let _ = std::fs::remove_file(&path);
}

/// The reference-peer direction: ffmpeg authors the transport stream, g2g demuxes
/// it. This is the leg that pins the mapping to what a real TS carries, including
/// ffmpeg's habit of writing the language descriptor only on the audio stream.
#[tokio::test]
async fn demuxes_service_and_language_tags_from_an_ffmpeg_authored_ts() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not present; skipping the reference-peer TS demux");
        return;
    }
    let path = std::env::temp_dir().join("g2g-m872-ffmpeg.ts");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-metadata",
            "service_name=Peer News",
            "-metadata",
            "service_provider=Peer Broadcasting",
            "-metadata:s:1",
            "language=fra",
            "-f",
            "mpegts",
        ])
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the reference TS");
    let ts = std::fs::read(&path).expect("read reference ts");

    let (ids, global, per_stream) = demux_bus_messages(&ts).await;
    assert_eq!(ids, vec![VIDEO_ID, AUDIO_ID], "ffmpeg's PID layout");
    assert_eq!(
        global,
        vec![(
            Some(1),
            tags(&[
                Tag::Title("Peer News".into()),
                Tag::Other {
                    key: "service_provider".into(),
                    value: "Peer Broadcasting".into(),
                },
            ])
        )],
        "the peer's SDT service text maps to the same two tags g2g writes, on its program"
    );

    assert_eq!(
        tags_of(&per_stream, AUDIO_ID),
        vec![Tag::Language("fra".into())],
        "the audio stream's ISO 639 descriptor"
    );
    assert!(
        tags_of(&per_stream, VIDEO_ID).is_empty(),
        "ffmpeg writes no language descriptor on video, so nothing posts for it"
    );
    let _ = std::fs::remove_file(&path);
}
