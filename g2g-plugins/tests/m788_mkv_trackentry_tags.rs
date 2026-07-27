//! M788 - a Matroska track's title and language live in its `TrackEntry`
//! (`Name` / `Language`), not in the `Targets`-scoped `Tags` element, which is
//! where ffmpeg writes them and where every player reads them. The muxer routes
//! `Tag::Title` / `Tag::Language` there and the demuxer merges them into the same
//! per-stream `BusMessage::StreamTag` view as the M787 `Tags` metadata.
//!
//! Legs: g2g demuxing an ffmpeg-authored file (the assertion M787 could not make),
//! ffprobe reading a g2g-muxed file, and a g2g round trip. The ffmpeg legs
//! self-skip when the binary is absent.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
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
use g2g_plugins::mkvdemux::{MkvDemuxN, MkvStream};
use g2g_plugins::mkvmuxn::MkvMuxN;

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
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
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.bytes.extend_from_slice(s);
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[derive(Default)]
struct PortTap {
    frames: Vec<usize>,
}
impl MultiOutputSink for PortTap {
    fn push_to<'a>(
        &'a mut self,
        port: usize,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
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

fn video_tags() -> TagList {
    [Tag::Title("Camera A".into()), Tag::Language("eng".into())]
        .into_iter()
        .collect()
}
fn audio_tags() -> TagList {
    [Tag::Title("Commentary".into()), Tag::Language("fra".into())]
        .into_iter()
        .collect()
}

/// Mux an A/V Matroska stream whose per-track metadata is title + language only,
/// so every tag goes to a `TrackEntry`. `seekable` exercises the M770 two-pass
/// finalize, which writes the same header.
async fn mux_av(seekable: bool) -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let mut mux = MkvMuxN::new(2)
        .with_seekable(seekable)
        .with_track_tags(0, video_tags())
        .with_track_tags(1, audio_tags());
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = Collect::default();
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2, 0xA3]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(
        0,
        frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000),
        &mut sink,
    )
    .await
    .unwrap();
    mux.process(1, frame(adts_au(&[0xB4, 0xB5]), 21_000_000), &mut sink)
        .await
        .unwrap();
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

/// Demux `file` with a two-port `MkvDemuxN` and return the per-stream tag
/// messages, in the order posted.
async fn demux_stream_tags(file: &[u8]) -> Vec<(String, TagList)> {
    let (bus, handle) = Bus::new(64);
    let mut demux = MkvDemuxN::new(vec![MkvStream::H264, MkvStream::Aac]).with_bus(handle);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure");
    let mut tap = PortTap { frames: vec![0, 0] };
    demux.process(data_frame(file), &mut tap).await.unwrap();
    assert!(
        tap.frames[0] > 0 && tap.frames[1] > 0,
        "both tracks demuxed: {:?}",
        tap.frames
    );
    let mut out = Vec::new();
    while let Some(msg) = bus.try_recv() {
        if let BusMessage::StreamTag { stream_id, tags } = msg {
            out.push((stream_id, tags));
        }
    }
    out
}

fn tags_of(per_stream: &[(String, TagList)], stream_id: &str) -> Vec<Tag> {
    per_stream
        .iter()
        .filter(|(id, _)| id == stream_id)
        .flat_map(|(_, t)| t.tags().iter().cloned())
        .collect()
}

#[tokio::test]
async fn track_entry_tags_round_trip_through_the_matroska_elements() {
    for seekable in [false, true] {
        let file = mux_av(seekable).await;
        let per_stream = demux_stream_tags(&file).await;
        assert_eq!(
            tags_of(&per_stream, "matroska-track-1"),
            video_tags().tags(),
            "video title + language (seekable={seekable})"
        );
        assert_eq!(
            tags_of(&per_stream, "matroska-track-2"),
            audio_tags().tags(),
            "audio title + language (seekable={seekable})"
        );
        assert_eq!(
            per_stream.len(),
            2,
            "one merged StreamTag per stream (seekable={seekable})"
        );
    }
}

/// The reference-peer direction: ffmpeg puts a stream's language and title in the
/// `TrackEntry`, so this is the leg that proves g2g reads them from there.
#[tokio::test]
async fn demuxes_language_and_title_from_an_ffmpeg_authored_matroska() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not present; skipping the reference-peer TrackEntry demux");
        return;
    }
    let path = std::env::temp_dir().join("g2g-m788-ffmpeg.mkv");
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
            "-metadata:s:v:0",
            "language=eng",
            "-metadata:s:v:0",
            "title=Camera A",
            "-metadata:s:a:0",
            "language=fra",
            "-metadata:s:a:0",
            "title=Commentary",
            // Rides the Tags element instead, so this checks the merge of both
            // sources into one per-stream view.
            "-metadata:s:a:0",
            "artist=Ada",
        ])
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the reference mkv");
    let file = std::fs::read(&path).expect("read reference mkv");

    let per_stream = demux_stream_tags(&file).await;
    let video = tags_of(&per_stream, "matroska-track-1");
    let audio = tags_of(&per_stream, "matroska-track-2");
    assert!(
        video.contains(&Tag::Language("eng".into()))
            && video.contains(&Tag::Title("Camera A".into())),
        "the video TrackEntry's language and title: {video:?}"
    );
    assert!(
        audio.contains(&Tag::Language("fra".into()))
            && audio.contains(&Tag::Title("Commentary".into())),
        "the audio TrackEntry's language and title: {audio:?}"
    );
    assert!(
        !video.contains(&Tag::Title("Commentary".into())),
        "no cross-track leak: {video:?}"
    );
    assert!(
        audio.contains(&Tag::Artist("Ada".into())),
        "the Tags-element metadata merges into the same stream view: {audio:?}"
    );
    let _ = std::fs::remove_file(&path);
}

/// The ffprobe oracle: ffmpeg reports a `TrackEntry` `Language` / `Name` as the
/// stream's `language` / `title`, so a g2g-muxed file must land them there.
#[tokio::test]
async fn ffprobe_reports_language_and_title_per_stream() {
    if Command::new("ffprobe").arg("-version").output().is_err() {
        eprintln!("ffprobe not present; skipping the TrackEntry tag oracle");
        return;
    }
    // Dedicated freshly-truncated log standalone; append to a shared CI log when
    // $G2G_CONFORMANCE_LOG is already set.
    let external = std::env::var_os("G2G_CONFORMANCE_LOG");
    let log = match &external {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let l = std::env::temp_dir().join("g2g-conformance-m788.tsv");
            std::env::set_var("G2G_CONFORMANCE_LOG", &l);
            let _ = std::fs::remove_file(&l);
            l
        }
    };
    let path = std::env::temp_dir().join("g2g-m788-track-entry.mkv");
    std::fs::write(&path, mux_av(false).await).expect("write mkv");

    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=index:stream_tags",
            "-of",
            "compact",
        ])
        .arg(&path)
        .output()
        .expect("run ffprobe");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "ffprobe read the file: {text}");
    let lines: Vec<&str> = text.lines().filter(|l| l.starts_with("stream|")).collect();
    assert_eq!(lines.len(), 2, "two streams: {text}");
    // ffprobe names them `language` / `title` only when they come from the
    // TrackEntry; a SimpleTag would surface as `LANGUAGE` / `TITLE`.
    assert!(
        lines[0].contains("tag:language=eng") && lines[0].contains("tag:title=Camera A"),
        "video stream: {text}"
    );
    assert!(
        lines[1].contains("tag:language=fra") && lines[1].contains("tag:title=Commentary"),
        "audio stream: {text}"
    );

    persist::record_evidence(
        "matroskamux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .detail("ffprobe reports the TrackEntry language and title per stream"),
    )
    .expect("record oracle evidence");
    assert!(
        persist::full_report()
            .records
            .iter()
            .any(|r| r.element == "matroskamux"),
        "matroskamux present after persisting evidence"
    );

    if external.is_none() {
        let _ = std::fs::remove_file(&log);
    }
    let _ = std::fs::remove_file(&path);
}
