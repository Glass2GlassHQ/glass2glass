//! M787 - Matroska `Targets`-scoped (per-track) tags, both directions. `MkvMuxN`
//! writes each input pad's metadata as a `Tag` whose `Targets` names that track's
//! `TagTrackUID`; the demuxer maps the UID back to the track and posts it as a
//! `BusMessage::StreamTag` on that stream's id, so an application can label the
//! streams it discovered in the `StreamCollection`.
//!
//! Three legs: a g2g round trip, ffprobe reading the g2g-muxed file (the oracle),
//! and g2g demuxing an ffmpeg-authored file (the direction that catches a wrong
//! UID -> track mapping, which a loopback cannot). The ffmpeg legs self-skip when
//! the binary is absent.
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
/// about the bus, the frames only prove the demux ran).
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

fn global_tags() -> TagList {
    tags(&[Tag::Title("Whole file".into())])
}
fn video_tags() -> TagList {
    tags(&[Tag::Language("eng".into()), Tag::Artist("Camera A".into())])
}
fn audio_tags() -> TagList {
    tags(&[
        Tag::Language("fra".into()),
        Tag::Artist("Commentary".into()),
    ])
}

/// Mux an A/V Matroska stream (H.264 + AAC) carrying one whole-file tag and a
/// distinct tag set on each track.
async fn mux_av_with_track_tags() -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let mut mux = MkvMuxN::new(2)
        .with_tags(global_tags())
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

/// Demux `file` with a two-port `MkvDemuxN` (port 0 = H.264, port 1 = AAC) and
/// return `(stream ids of the collection, whole-file tags, per-stream tags)`.
#[allow(clippy::type_complexity)]
async fn demux_bus_messages(file: &[u8]) -> (Vec<String>, Vec<TagList>, Vec<(String, TagList)>) {
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

    let (mut ids, mut global, mut per_stream) = (Vec::new(), Vec::new(), Vec::new());
    while let Some(msg) = bus.try_recv() {
        match msg {
            BusMessage::StreamCollection(c) => {
                ids = c.streams().iter().map(|s| s.id.to_string()).collect()
            }
            BusMessage::Tag { tags, .. } => global.push(tags),
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
async fn per_track_tags_round_trip_through_the_matroska_elements() {
    let file = mux_av_with_track_tags().await;
    let (ids, global, per_stream) = demux_bus_messages(&file).await;

    assert_eq!(ids, vec!["matroska-track-1", "matroska-track-2"]);
    let flat: Vec<Tag> = global
        .iter()
        .flat_map(|t| t.tags().iter().cloned())
        .collect();
    assert_eq!(
        flat,
        global_tags().tags(),
        "the whole-file tag posts as a plain Tag"
    );

    assert_eq!(
        tags_of(&per_stream, "matroska-track-1"),
        video_tags().tags(),
        "the video track's tags land on the video stream id"
    );
    assert_eq!(
        tags_of(&per_stream, "matroska-track-2"),
        audio_tags().tags(),
        "the audio track's tags land on the audio stream id"
    );
    assert_eq!(
        per_stream.len(),
        2,
        "one StreamTag per tagged track, posted once each"
    );
}

/// The ffprobe oracle: the reference implementation must attach the g2g-written
/// `Targets`-scoped tags to the right stream. Records peer-tagged `Oracle`
/// evidence for `matroskamux` on success (the M619 pattern).
#[tokio::test]
async fn ffprobe_reports_the_per_track_tags_on_each_stream() {
    if Command::new("ffprobe").arg("-version").output().is_err() {
        eprintln!("ffprobe not present; skipping the mkv per-track tag oracle");
        return;
    }
    // Dedicated freshly-truncated log standalone; append to a shared CI log when
    // $G2G_CONFORMANCE_LOG is already set.
    let external = std::env::var_os("G2G_CONFORMANCE_LOG");
    let log = match &external {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let l = std::env::temp_dir().join("g2g-conformance-m787.tsv");
            std::env::set_var("G2G_CONFORMANCE_LOG", &l);
            let _ = std::fs::remove_file(&l);
            l
        }
    };
    let file = mux_av_with_track_tags().await;
    let path = std::env::temp_dir().join("g2g-m787-track-tags.mkv");
    std::fs::write(&path, &file).expect("write mkv");

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
    assert!(
        lines[0].contains("Camera A") && lines[0].contains("eng"),
        "ffprobe puts the video tags on stream 0: {text}"
    );
    assert!(
        !lines[0].contains("Commentary"),
        "the audio tags do not leak onto the video stream: {text}"
    );
    assert!(
        lines[1].contains("Commentary") && lines[1].contains("fra"),
        "ffprobe puts the audio tags on stream 1: {text}"
    );

    persist::record_evidence(
        "matroskamux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .detail("ffprobe reports the Targets-scoped tags on the right stream"),
    )
    .expect("record oracle evidence");
    let report = persist::full_report();
    assert!(
        report.records.iter().any(|r| r.element == "matroskamux"),
        "matroskamux present after persisting evidence"
    );

    if external.is_none() {
        let _ = std::fs::remove_file(&log);
    }
    let _ = std::fs::remove_file(&path);
}

/// The reference-peer direction: ffmpeg authors the file, g2g demuxes it. Only
/// this leg catches a wrong `TagTrackUID` -> track mapping, since ffmpeg picks
/// the UIDs (large random values, unrelated to the track numbers).
#[tokio::test]
async fn demuxes_per_track_tags_from_an_ffmpeg_authored_matroska() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not present; skipping the reference-peer mkv demux");
        return;
    }
    let path = std::env::temp_dir().join("g2g-m787-ffmpeg.mkv");
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
            // `language` / `title` are native TrackEntry elements in Matroska;
            // ARTIST / ALBUM are what ffmpeg writes as Targets-scoped tags.
            "-metadata:s:v:0",
            "artist=Camera A",
            "-metadata:s:a:0",
            "artist=Commentary",
            "-metadata:s:a:0",
            "album=Second Take",
            "-metadata",
            "artist=Whole File",
        ])
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the reference mkv");
    let file = std::fs::read(&path).expect("read reference mkv");

    let (ids, global, per_stream) = demux_bus_messages(&file).await;
    assert_eq!(ids, vec!["matroska-track-1", "matroska-track-2"]);
    let flat: Vec<Tag> = global
        .iter()
        .flat_map(|t| t.tags().iter().cloned())
        .collect();
    assert!(
        flat.contains(&Tag::Artist("Whole File".into())),
        "the file-scoped tag stays whole-stream: {flat:?}"
    );

    let video = tags_of(&per_stream, "matroska-track-1");
    let audio = tags_of(&per_stream, "matroska-track-2");
    assert!(
        video.contains(&Tag::Artist("Camera A".into())),
        "the video track's tag resolved through its TrackUID: {video:?}"
    );
    assert!(
        !video.contains(&Tag::Artist("Commentary".into())),
        "the audio tag did not land on the video stream: {video:?}"
    );
    assert!(
        audio.contains(&Tag::Artist("Commentary".into()))
            && audio.contains(&Tag::Album("Second Take".into())),
        "the audio track's tags resolved through its TrackUID: {audio:?}"
    );
    let _ = std::fs::remove_file(&path);
}
