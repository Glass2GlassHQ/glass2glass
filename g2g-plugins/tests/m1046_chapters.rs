//! M1046 - container chapters (the table of contents) for Matroska and MP4,
//! both directions.
//!
//! `MkvMuxN` writes a `Chapters` element (one default `EditionEntry` of
//! `ChapterAtom`s, times in nanoseconds); `Mp4MuxN` writes the `moov`'s Nero
//! `udta/chpl` list. Both demuxers post what they parse as a
//! `BusMessage::Chapters`, out of band like the tags. The MP4 reader prefers a
//! QuickTime chapter *text* track when the file has one, because that shape
//! carries an end time per chapter, which `chpl` cannot express.
//!
//! Legs: a g2g round trip per container, ffprobe reading the g2g-written
//! chapters (the oracle), and g2g reading ffmpeg-authored files (the direction
//! that catches a wrong byte layout, which a loopback cannot). The ffmpeg legs
//! self-skip when the binaries are absent. Plus the untrusted-input bounds: a
//! hidden chapter is dropped, nesting is depth-capped, and a lying `chpl` count
//! or title length stops at what the box really holds instead of panicking.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::fanout::MultiOutputSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, Chapter, Dim, G2gError,
    MultiInputElement, MultiOutputElement, OutputSink, PushOutcome, Rate, Tag, TagList, VideoCodec,
};
use g2g_plugins::matroska::MatroskaDemuxer;
use g2g_plugins::mkvdemux::{MkvDemuxN, MkvStream};
use g2g_plugins::mkvmuxn::MkvMuxN;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::mp4mux::Mp4Mux;
use g2g_plugins::mp4muxn::Mp4MuxN;
use g2g_plugins::mp4src::Mp4Src;

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

/// Counts what each port received: this milestone is about the bus, the frames
/// only prove the demux ran.
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

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1046-{}-{name}", std::process::id()))
}

/// The chapter list every leg writes and expects back. Three bounded chapters
/// on second boundaries, so ffprobe's `start_time` / `end_time` are exact.
fn reference_chapters() -> Vec<Chapter> {
    vec![
        Chapter::new(0, "Opening").with_end_ns(2_000_000_000),
        Chapter::new(2_000_000_000, "Middle Part").with_end_ns(5_000_000_000),
        Chapter::new(5_000_000_000, "Finale").with_end_ns(8_000_000_000),
    ]
}

/// The FFMETADATA input that makes ffmpeg author the same three chapters.
const FFMETADATA_CHAPTERS: &str = "\
;FFMETADATA1

[CHAPTER]
TIMEBASE=1/1000
START=0
END=2000
title=Opening

[CHAPTER]
TIMEBASE=1/1000
START=2000
END=5000
title=Middle Part

[CHAPTER]
TIMEBASE=1/1000
START=5000
END=8000
title=Finale
";

/// Author a real file with ffmpeg carrying the reference chapters, and return
/// its bytes. `suffix` picks the container.
fn ffmpeg_file_with_chapters(suffix: &str) -> Vec<u8> {
    let meta = temp_path("meta.txt");
    std::fs::write(&meta, FFMETADATA_CHAPTERS).expect("write ffmetadata");
    let path = temp_path(&format!("reference.{suffix}"));
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10:duration=8",
        ])
        .arg("-i")
        .arg(&meta)
        .args([
            "-map_metadata",
            "1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the reference {suffix}");
    std::fs::read(&path).expect("read reference file")
}

/// ffprobe's chapter list as `(start_time, end_time, title)` triples.
fn ffprobe_chapters(bytes: &[u8], suffix: &str) -> Vec<(String, String, String)> {
    let path = temp_path(&format!("g2g.{suffix}"));
    std::fs::write(&path, bytes).expect("write g2g file");
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_chapters",
            "-show_entries",
            "chapter=start_time,end_time:chapter_tags=title",
            "-of",
            "compact=nokey=0",
        ])
        .arg(&path)
        .output()
        .expect("run ffprobe");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "ffprobe read the file: {text}");
    text.lines()
        .filter(|l| l.starts_with("chapter|"))
        .map(|line| {
            let field = |key: &str| {
                line.split('|')
                    .find_map(|f| f.strip_prefix(key))
                    .unwrap_or_default()
                    .to_string()
            };
            (
                field("start_time="),
                field("end_time="),
                field("tag:title="),
            )
        })
        .collect()
}

/// Mux an A/V Matroska stream (H.264 + AAC) carrying the reference chapters.
async fn mux_mkv_with_chapters() -> Vec<u8> {
    let mut mux = MkvMuxN::new(2).with_chapters(reference_chapters());
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = Collect::default();
    feed_av(&mut mux, &mut sink).await;
    sink.bytes
}

/// Mux a progressive (non-fragmented) A/V MP4 carrying the reference chapters.
/// Progressive, so the file has a real `moov` with sample tables, which is what
/// both ffprobe and the g2g demuxer read.
async fn mux_mp4_with_chapters() -> Vec<u8> {
    let mut mux = Mp4MuxN::new(2)
        .with_fragmented(false)
        .with_chapters(reference_chapters());
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = Collect::default();
    feed_av(&mut mux, &mut sink).await;
    sink.bytes
}

/// Push two H.264 access units and two AAC ones, then EOS both pads: enough for
/// either muxer to write a complete file.
async fn feed_av<M>(mux: &mut M, sink: &mut Collect)
where
    M: MultiInputElement,
{
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2, 0xA3]), 0), sink)
        .await
        .unwrap();
    mux.process(0, frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000), sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0xB4, 0xB5]), 21_000_000), sink)
        .await
        .unwrap();
    mux.process(0, PipelinePacket::Eos, sink).await.unwrap();
    mux.process(1, PipelinePacket::Eos, sink).await.unwrap();
}

/// The chapters an `MkvDemuxN` posts for `file`.
async fn mkv_bus_chapters(file: &[u8]) -> Vec<Chapter> {
    let (bus, handle) = Bus::new(64);
    let mut demux = MkvDemuxN::new(vec![MkvStream::H264, MkvStream::Aac]).with_bus(handle);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure");
    let mut tap = PortTap { frames: vec![0, 0] };
    demux.process(frame(file.to_vec(), 0), &mut tap).await.ok();
    drain_chapters(&bus)
}

/// The chapters an `Mp4DemuxN` posts for `file`.
async fn mp4_bus_chapters(file: &[u8]) -> Vec<Chapter> {
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
    demux.process(frame(file.to_vec(), 0), &mut tap).await.ok();
    demux.process(PipelinePacket::Eos, &mut tap).await.ok();
    drain_chapters(&bus)
}

fn drain_chapters(bus: &Bus) -> Vec<Chapter> {
    let mut out = Vec::new();
    while let Some(msg) = bus.try_recv() {
        if let BusMessage::Chapters(chapters) = msg {
            out.extend(chapters);
        }
    }
    out
}

fn titles(chapters: &[Chapter]) -> Vec<&str> {
    chapters.iter().map(|c| c.title.as_str()).collect()
}

// --- Matroska --------------------------------------------------------------

/// The g2g round trip: what `MkvMuxN` writes, `MkvDemuxN` posts back on the bus,
/// times and titles intact.
#[tokio::test]
async fn mkv_round_trips_chapters_through_the_bus() {
    let file = mux_mkv_with_chapters().await;
    let chapters = mkv_bus_chapters(&file).await;
    assert_eq!(chapters, reference_chapters(), "chapters survive the mux");
}

/// The ffprobe oracle: the reference implementation reads the `Chapters`
/// element g2g wrote, with the right times and titles.
#[tokio::test]
async fn ffprobe_reads_the_g2g_written_mkv_chapters() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg/ffprobe not present; skipping the mkv chapter oracle");
        return;
    }
    let file = mux_mkv_with_chapters().await;
    let probed = ffprobe_chapters(&file, "mkv");
    assert_eq!(
        probed,
        vec![
            ("0.000000".into(), "2.000000".into(), "Opening".into()),
            ("2.000000".into(), "5.000000".into(), "Middle Part".into()),
            ("5.000000".into(), "8.000000".into(), "Finale".into()),
        ],
        "ffprobe reports the g2g chapters"
    );
}

/// The direction a loopback cannot cover: g2g parses the `Chapters` element an
/// ffmpeg-authored file carries. Catches a wrong element id or a time read as
/// `TimestampScale` ticks instead of the nanoseconds the spec stores.
#[tokio::test]
async fn reads_the_chapters_of_an_ffmpeg_authored_mkv() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present; skipping the reference-peer mkv read");
        return;
    }
    let file = ffmpeg_file_with_chapters("mkv");
    let mut demux = MatroskaDemuxer::new();
    demux.push_data(&file);
    let chapters = demux.chapters();
    assert_eq!(titles(chapters), ["Opening", "Middle Part", "Finale"]);
    assert_eq!(chapters[1].start_ns, 2_000_000_000);
    assert_eq!(chapters[1].end_ns, Some(5_000_000_000));
    assert_eq!(
        chapters[1].language.as_deref(),
        Some("und"),
        "ffmpeg's ChapLanguage is surfaced"
    );
}

/// Nested `ChapterAtom`s round-trip, and a hidden edition or atom never reaches
/// the application: it is not meant to appear in a chapter menu.
#[tokio::test]
async fn mkv_keeps_nesting_and_drops_hidden_chapters() {
    let mut parent = Chapter::new(0, "Part One").with_end_ns(4_000_000_000);
    parent.sub_chapters = vec![
        Chapter::new(0, "Scene A").with_end_ns(2_000_000_000),
        Chapter::new(2_000_000_000, "Scene B").with_end_ns(4_000_000_000),
    ];
    let mut mux = MkvMuxN::new(2).with_chapters(vec![parent.clone()]);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = Collect::default();
    feed_av(&mut mux, &mut sink).await;

    let mut demux = MatroskaDemuxer::new();
    demux.push_data(&sink.bytes);
    assert_eq!(demux.chapters(), [parent], "the nested atoms come back");

    // Set ChapterFlagHidden (0x98) on the parent atom in the written bytes: the
    // demuxer must drop it and everything under it.
    let mut hidden = sink.bytes.clone();
    let atom_at = find_subslice(&hidden, &[0xB6]).expect("a ChapterAtom is written");
    // ChapterUID (0x73C4) opens the atom body; overwrite its 2-byte id with a
    // one-byte ChapterFlagHidden carrying 1, which is the same 4 bytes long.
    let uid_at = find_subslice(&hidden[atom_at..], &[0x73, 0xC4]).expect("a ChapterUID") + atom_at;
    hidden[uid_at..uid_at + 4].copy_from_slice(&[0x98, 0x81, 0x01, 0x80]);
    let mut demux = MatroskaDemuxer::new();
    demux.push_data(&hidden);
    assert!(
        demux.chapters().is_empty(),
        "a hidden atom is not surfaced: {:?}",
        demux.chapters()
    );
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

const ID_SEGMENT: [u8; 4] = [0x18, 0x53, 0x80, 0x67];
const ID_CHAPTERS: [u8; 4] = [0x10, 0x43, 0xA7, 0x70];
const ID_TAGS: [u8; 4] = [0x12, 0x54, 0xC3, 0x67];
const ID_CUES: [u8; 4] = [0x1C, 0x53, 0xBB, 0x6B];

/// In the two-pass (seekable) shape the front `SeekHead` indexes the `Chapters`
/// element too, so a reader jumps to the table of contents without scanning.
/// The entries are fixed-width and positioned by hand, which is exactly what a
/// miscounted entry breaks: this checks the indexed position lands on the
/// element it names, and that the pre-existing `Cues` entry still does.
#[tokio::test]
async fn mkv_seekable_seekhead_indexes_the_chapters_element() {
    // Tags as well as chapters: the Tags entry is positioned after the
    // Chapters element, so its offset is what a miscount silently corrupts.
    let tags: TagList = [Tag::Title("Whole file".into())].into_iter().collect();
    let mut mux = MkvMuxN::new(2)
        .with_seekable(true)
        .with_tags(tags)
        .with_chapters(reference_chapters());
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = Collect::default();
    feed_av(&mut mux, &mut sink).await;
    let file = sink.bytes;

    // Segment data starts after the id and the 8-byte size the muxer reserves.
    let segment_data_at = find_subslice(&file, &ID_SEGMENT).expect("a Segment") + 4 + 8;
    // One SeekHead entry: SeekID (size 4) holding the target id, then
    // SeekPosition (size 8) relative to the Segment data start.
    let position_of = |target: &[u8; 4]| -> usize {
        let mut key = vec![0x53, 0xAB, 0x84];
        key.extend_from_slice(target);
        key.extend_from_slice(&[0x53, 0xAC, 0x88]);
        let at = find_subslice(&file, &key).expect("a SeekHead entry for the target") + key.len();
        let position = u64::from_be_bytes(file[at..at + 8].try_into().unwrap()) as usize;
        segment_data_at + position
    };
    assert_eq!(
        &file[position_of(&ID_CHAPTERS)..][..4],
        &ID_CHAPTERS,
        "the indexed position lands on the Chapters element"
    );
    assert_eq!(
        &file[position_of(&ID_TAGS)..][..4],
        &ID_TAGS,
        "the Tags entry, which sits after the Chapters element, still lands on it"
    );
    assert_eq!(
        &file[position_of(&ID_CUES)..][..4],
        &ID_CUES,
        "the Cues entry still lands on the Cues element"
    );
    assert_eq!(mkv_bus_chapters(&file).await, reference_chapters());
}

// --- MP4 -------------------------------------------------------------------

/// The g2g round trip through the Nero `chpl` list: `Mp4MuxN` writes it,
/// `Mp4DemuxN` posts it back. `chpl` has no end-time field, so the chapters
/// come back open-ended, which is the honest reading of what the box stores.
#[tokio::test]
async fn mp4_round_trips_chapters_through_the_bus() {
    let file = mux_mp4_with_chapters().await;
    let chapters = mp4_bus_chapters(&file).await;
    assert_eq!(titles(&chapters), ["Opening", "Middle Part", "Finale"]);
    let starts: Vec<u64> = chapters.iter().map(|c| c.start_ns).collect();
    assert_eq!(starts, [0, 2_000_000_000, 5_000_000_000]);
    assert!(
        chapters.iter().all(|c| c.end_ns.is_none()),
        "chpl carries no end times: {chapters:?}"
    );
}

/// The ffprobe oracle for MP4: the reference implementation reads the `chpl`
/// list g2g wrote, at the right start times and titles.
#[tokio::test]
async fn ffprobe_reads_the_g2g_written_mp4_chapters() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg/ffprobe not present; skipping the mp4 chapter oracle");
        return;
    }
    let file = mux_mp4_with_chapters().await;
    let probed = ffprobe_chapters(&file, "mp4");
    let starts: Vec<&str> = probed.iter().map(|(s, _, _)| s.as_str()).collect();
    let names: Vec<&str> = probed.iter().map(|(_, _, t)| t.as_str()).collect();
    assert_eq!(starts, ["0.000000", "2.000000", "5.000000"]);
    assert_eq!(names, ["Opening", "Middle Part", "Finale"]);
}

/// The direction a loopback cannot cover: g2g reads the chapters of an
/// ffmpeg-authored MP4. ffmpeg writes both shapes, so this exercises the
/// QuickTime chapter text track (the preferred one, since it carries the end
/// times) against the layout ffmpeg really produces.
#[tokio::test]
async fn reads_the_chapters_of_an_ffmpeg_authored_mp4() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present; skipping the reference-peer mp4 read");
        return;
    }
    let file = ffmpeg_file_with_chapters("mp4");
    let chapters = mp4_bus_chapters(&file).await;
    assert_eq!(titles(&chapters), ["Opening", "Middle Part", "Finale"]);
    assert_eq!(chapters[1].start_ns, 2_000_000_000);
    assert_eq!(
        chapters[1].end_ns,
        Some(5_000_000_000),
        "the chapter text track carries an end per chapter"
    );
}

/// The fragmented single-track path, which builds its `moov` separately from the
/// progressive one: `Mp4Mux` puts the `chpl` in the init segment, and `Mp4Src`
/// posts it when it opens the file.
#[tokio::test]
async fn fragmented_mp4_carries_chapters_to_the_source_bus() {
    let mut mux = Mp4Mux::new().with_chapters(reference_chapters());
    mux.configure_pipeline(&h264_caps()).expect("configure mux");
    let mut written = Collect::default();
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    mux.process(frame(annexb(&[&sps, &pps, &idr]), 0), &mut written)
        .await
        .unwrap();
    mux.process(PipelinePacket::Eos, &mut written)
        .await
        .unwrap();

    let path = temp_path("fragmented.mp4");
    std::fs::write(&path, &written.bytes).expect("write the fragmented file");
    let (bus, handle) = Bus::new(64);
    let mut src = Mp4Src::new(&path).with_bus(handle);
    let caps = src.intercept_caps().await.expect("probe the header");
    src.configure_pipeline(&caps).expect("configure src");
    let mut out = Collect::default();
    src.run(&mut out).await.expect("demux to EOS");
    assert_eq!(
        titles(&drain_chapters(&bus)),
        ["Opening", "Middle Part", "Finale"]
    );
}

/// A `chpl` whose count or title length overruns the box parses to what the box
/// really holds and no further. Both are the file's claim, so a lying one has to
/// end the list rather than read past the box or panic.
#[tokio::test]
async fn mp4_chpl_with_a_lying_count_does_not_overrun() {
    let file = mux_mp4_with_chapters().await;
    let chpl_at = find_subslice(&file, b"chpl").expect("the muxer wrote a chpl");
    // Body layout: 4cc, version+flags, the version-1 reserved word, then the
    // one-byte count, then the first entry's u64 start and its title length.
    let count_at = chpl_at + 4 + 4 + 4;
    let first_title_len_at = count_at + 1 + 8;

    let mut over_count = file.clone();
    over_count[count_at] = 0xFF;
    assert_eq!(
        titles(&mp4_bus_chapters(&over_count).await),
        ["Opening", "Middle Part", "Finale"],
        "a count past the end stops at the last real entry"
    );

    let mut over_title = file.clone();
    over_title[first_title_len_at] = 0xFF;
    assert!(
        mp4_bus_chapters(&over_title).await.is_empty(),
        "a title length past the end of the box ends the list"
    );
}
