//! M838 - the MP4 freeform (`----`) and integer (`trkn` / `disk` / `cpil` /
//! `tmpo`) metadata atoms, and the global / per-stream tag merge policy that
//! spans the multi-stream containers (`g2g_core::tag`).
//!
//! `Mp4MuxN` writes the file's own tags as the `moov`'s `udta/meta/ilst` and
//! each input pad's as that `trak`'s, after splitting them: a tag every input
//! repeats identically moves up to the file level, and a tag already set
//! globally is not written again per track. `Mp4DemuxN` posts the two scopes
//! separately (a `BusMessage::Tag` for the file, a `BusMessage::StreamTag` per
//! stream id), leaving the conflict rule (the stream's tag wins on its own pad)
//! to `g2g_core::resolve_tags`.
//!
//! Legs: a g2g round trip, ffprobe reading the g2g-written atoms (the oracle),
//! and g2g reading an ffmpeg-authored file. ffmpeg has no way to *author* a
//! `----` atom (its `use_metadata_tags` writes the QuickTime `keys` form
//! instead), so the freeform atom is validated in the direction ffmpeg does
//! cover: it reads the one g2g writes. The ffmpeg legs self-skip when the binary
//! is absent.
#![cfg(feature = "std")]

use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::fanout::MultiOutputSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    resolve_tags, AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, Dim, G2gError,
    MultiInputElement, MultiOutputElement, OutputSink, PushOutcome, Rate, Tag, TagList, VideoCodec,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::mp4muxn::Mp4MuxN;

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

/// Counts what each port received (this milestone is about the bus; the frames
/// only prove the demux ran).
#[derive(Default)]
struct PortTap {
    frames: Vec<usize>,
}
impl MultiOutputSink for PortTap {
    fn port_count(&self) -> usize {
        self.frames.len()
    }

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
}

fn frame(bytes: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
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

fn number(key: &str, value: u64) -> Tag {
    Tag::Number {
        key: key.into(),
        value,
    }
}

fn mood() -> Tag {
    Tag::Freeform {
        namespace: "com.apple.iTunes".into(),
        key: "MOOD".into(),
        value: "calm".into(),
    }
}

/// The file's own metadata: a text atom, a freeform `----` atom, and the integer
/// atoms in both forms (an index+total pair and a single value).
fn global_tags() -> TagList {
    tags(&[
        Tag::Title("Whole file".into()),
        mood(),
        number(Tag::TRACK_NUMBER, 3),
        number(Tag::TRACK_COUNT, 12),
        number(Tag::DISC_NUMBER, 1),
        number(Tag::DISC_COUNT, 2),
        number(Tag::COMPILATION, 1),
    ])
}

/// Per-track metadata. `Title` conflicts with the file's (the stream wins on its
/// own pad) and `Encoder` is identical on both tracks (it hoists to the file).
fn video_tags() -> TagList {
    tags(&[
        Tag::Title("Video Track".into()),
        Tag::Artist("Camera A".into()),
        Tag::Encoder("g2g".into()),
    ])
}
fn audio_tags() -> TagList {
    tags(&[
        Tag::Title("Audio Track".into()),
        Tag::Artist("Commentary".into()),
        Tag::Encoder("g2g".into()),
    ])
}

/// Mux a progressive (non-fragmented) A/V MP4 carrying the file-level tags and a
/// distinct tag set on each track. Progressive, so the file has a real `moov`
/// with sample tables, which is what both ffprobe and the g2g demuxer read.
async fn mux_av_with_tags() -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let mut mux = Mp4MuxN::new(2)
        .with_fragmented(false)
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

/// Demux `file` with an `Mp4DemuxN` over its forwardable tracks and return
/// `(stream ids, whole-file tags, per-stream tags)`.
#[allow(clippy::type_complexity)]
async fn demux_bus_messages(file: &[u8]) -> (Vec<String>, Vec<TagList>, Vec<(String, TagList)>) {
    let streams = forwardable_streams(file);
    assert!(!streams.is_empty(), "the file has forwardable tracks");
    let ports: Vec<Mp4Port> = streams
        .iter()
        .map(|s| Mp4Port {
            track_id: s.track_id,
            caps: s.caps.clone(),
        })
        .collect();
    let (bus, handle) = Bus::new(64);
    let mut demux = Mp4DemuxN::new(ports).with_bus(handle);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .expect("configure");
    let mut tap = PortTap {
        frames: vec![0; streams.len()],
    };
    demux
        .process(frame(file.to_vec(), 0), &mut tap)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("demux eos");
    assert!(
        tap.frames.iter().all(|n| *n > 0),
        "every track demuxed: {:?}",
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
fn tags_of(per_stream: &[(String, TagList)], stream_id: &str) -> TagList {
    per_stream
        .iter()
        .filter(|(id, _)| id == stream_id)
        .flat_map(|(_, t)| t.tags().iter().cloned())
        .collect()
}

fn flatten(lists: &[TagList]) -> Vec<Tag> {
    lists
        .iter()
        .flat_map(|t| t.tags().iter().cloned())
        .collect()
}

/// The freeform and integer atoms survive `Mp4MuxN` -> `Mp4DemuxN`, and the two
/// tag scopes stay separate: the file's own tags post once as a `Tag`, each
/// track's on its own stream id as a `StreamTag`.
#[tokio::test]
async fn tags_round_trip_through_the_mp4_elements_per_scope() {
    let file = mux_av_with_tags().await;
    let (ids, global, per_stream) = demux_bus_messages(&file).await;

    assert_eq!(ids, vec!["mp4-track-1", "mp4-track-2"]);
    let file_tags = flatten(&global);
    for expected in global_tags().tags() {
        assert!(
            file_tags.contains(expected),
            "{expected:?} posted as a whole-file tag: {file_tags:?}"
        );
    }
    assert!(
        file_tags.contains(&mood()),
        "the freeform ---- atom round-trips with its namespace: {file_tags:?}"
    );

    let video = tags_of(&per_stream, "mp4-track-1");
    let audio = tags_of(&per_stream, "mp4-track-2");
    assert_eq!(
        video.tags(),
        &[
            Tag::Title("Video Track".into()),
            Tag::Artist("Camera A".into())
        ],
        "the video track's own tags land on the video stream id"
    );
    assert_eq!(
        audio.tags(),
        &[
            Tag::Title("Audio Track".into()),
            Tag::Artist("Commentary".into())
        ],
        "the audio track's own tags land on the audio stream id"
    );
    assert_eq!(
        per_stream.len(),
        2,
        "one StreamTag per tagged track, posted once each"
    );
}

/// The mux-side split: a tag both inputs carry identically is written once at
/// the file level, not repeated in either `trak`.
#[tokio::test]
async fn shared_input_tags_hoist_to_the_file_level() {
    let file = mux_av_with_tags().await;
    let (_, global, per_stream) = demux_bus_messages(&file).await;

    assert!(
        flatten(&global).contains(&Tag::Encoder("g2g".into())),
        "the tag both tracks repeat moved up to the file"
    );
    for (id, list) in &per_stream {
        assert!(
            !list.tags().contains(&Tag::Encoder("g2g".into())),
            "{id} does not repeat the hoisted tag: {list:?}"
        );
    }
    // Three `ilst` boxes: the file's own plus one per track that kept tags.
    let ilst = file.windows(4).filter(|w| *w == b"ilst").count();
    assert_eq!(ilst, 3, "one file-level ilst and one per tagged trak");
}

/// The conflict rule: a key set both globally and on a stream resolves to the
/// stream's value on that stream's pad, while the global tags the stream leaves
/// alone still apply to it.
#[tokio::test]
async fn a_stream_tag_wins_over_the_global_tag_for_the_same_key() {
    let file = mux_av_with_tags().await;
    let (_, global, per_stream) = demux_bus_messages(&file).await;
    let file_tags: TagList = flatten(&global).into_iter().collect();

    assert!(
        file_tags.tags().contains(&Tag::Title("Whole file".into())),
        "the file keeps its own title"
    );
    let video = tags_of(&per_stream, "mp4-track-1");
    let effective = resolve_tags(&file_tags, &video);
    assert!(
        effective.tags().contains(&Tag::Title("Video Track".into()))
            && !effective.tags().contains(&Tag::Title("Whole file".into())),
        "the stream's title overrides the file's on its own pad: {effective:?}"
    );
    assert!(
        effective.tags().contains(&mood()),
        "a global tag the stream does not set still applies to it: {effective:?}"
    );
}

/// The ffprobe oracle: the reference implementation reads the g2g-written
/// freeform and integer atoms, and puts a `trak`-scoped text atom on that
/// stream. Records peer-tagged `Oracle` evidence for `mp4mux` on success.
#[tokio::test]
async fn ffprobe_reads_the_freeform_and_integer_atoms() {
    if Command::new("ffprobe").arg("-version").output().is_err() {
        eprintln!("ffprobe not present; skipping the mp4 tag oracle");
        return;
    }
    let external = std::env::var_os("G2G_CONFORMANCE_LOG");
    let log = match &external {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let l = std::env::temp_dir().join("g2g-conformance-m838.tsv");
            std::env::set_var("G2G_CONFORMANCE_LOG", &l);
            let _ = std::fs::remove_file(&l);
            l
        }
    };
    let file = mux_av_with_tags().await;
    let path = std::env::temp_dir().join("g2g-m838-tags.mp4");
    std::fs::write(&path, &file).expect("write mp4");

    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format_tags:stream=index:stream_tags",
            "-of",
            "compact",
        ])
        .arg(&path)
        .output()
        .expect("run ffprobe");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "ffprobe read the file: {text}");

    // ffmpeg's own names for the atoms: trkn -> track, disk -> disc, cpil ->
    // compilation, and a `----` item -> its `name` as the key.
    for expected in [
        "title=Whole file",
        "track=3/12",
        "disc=1/2",
        "compilation=1",
        "MOOD=calm",
    ] {
        assert!(
            text.contains(expected),
            "ffprobe reports {expected}: {text}"
        );
    }
    let streams: Vec<&str> = text.lines().filter(|l| l.starts_with("stream|")).collect();
    assert_eq!(streams.len(), 2, "two streams: {text}");
    // Older ffprobe (CI's ubuntu build) does not read a trak-level ilst into
    // stream tags at all; when it does, the titles must land on the right
    // streams. The per-stream read itself is asserted against g2g's own
    // demuxer in tags_round_trip_through_the_mp4_elements_per_scope.
    if text.contains("Video Track") || text.contains("Audio Track") {
        assert!(
            streams[0].contains("Video Track") && !streams[0].contains("Audio Track"),
            "the trak-scoped title lands on the video stream: {text}"
        );
        assert!(
            streams[1].contains("Audio Track"),
            "the trak-scoped title lands on the audio stream: {text}"
        );
    }

    persist::record_evidence(
        "mp4mux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .detail(
                "ffprobe reads the ---- freeform and trkn/disk/cpil atoms, and the per-trak tags",
            ),
    )
    .expect("record oracle evidence");
    let report = persist::full_report();
    assert!(
        report.records.iter().any(|r| r.element == "mp4mux"),
        "mp4mux present after persisting evidence"
    );

    if external.is_none() {
        let _ = std::fs::remove_file(&log);
    }
    let _ = std::fs::remove_file(&path);
}

/// The reference-peer direction: ffmpeg authors the atoms, g2g reads them. Only
/// this leg proves the reader matches ffmpeg's byte layout for the index+total
/// pair (`trkn` / `disk`) and the type-21 `cpil` flag.
#[tokio::test]
async fn reads_the_integer_atoms_from_an_ffmpeg_authored_mp4() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not present; skipping the reference-peer mp4 read");
        return;
    }
    let path = std::env::temp_dir().join("g2g-m838-ffmpeg.mp4");
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
            "title=Reference File",
            "-metadata",
            "track=3/12",
            "-metadata",
            "disc=1/2",
            "-metadata",
            "compilation=1",
            // MP4 has no per-track ilst in ffmpeg's writer: it puts a stream
            // title in the trak's `name` box, so only the file scope is checked.
            "-metadata:s:v:0",
            "title=Video Track",
        ])
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the reference mp4");
    let file = std::fs::read(&path).expect("read reference mp4");

    let (ids, global, _) = demux_bus_messages(&file).await;
    assert_eq!(ids.len(), 2, "two streams discovered");
    let file_tags = flatten(&global);
    for expected in [
        Tag::Title("Reference File".into()),
        number(Tag::TRACK_NUMBER, 3),
        number(Tag::TRACK_COUNT, 12),
        number(Tag::DISC_NUMBER, 1),
        number(Tag::DISC_COUNT, 2),
        number(Tag::COMPILATION, 1),
    ] {
        assert!(
            file_tags.contains(&expected),
            "{expected:?} read from the ffmpeg-authored file: {file_tags:?}"
        );
    }
    let _ = std::fs::remove_file(&path);
}
