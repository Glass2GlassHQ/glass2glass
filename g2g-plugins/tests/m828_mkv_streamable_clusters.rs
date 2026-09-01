//! M828: the two shapes the single-track `matroskamux` writes, and what each one
//! buys.
//!
//! Streaming (the default, and `streamable`): an unknown-size `Segment` (the
//! 8-byte all-ones EBML size) and unknown-size `Cluster`s (the one-byte `0xFF`
//! marker), so no element's length has to be known before its content is emitted
//! and nothing is patched at EOS. A reader can consume the stream while it is
//! still being written. An unknown-size Cluster ends at the next level-1 element
//! or at end of stream, so a truncated prefix is still parseable up to the last
//! block that arrived whole. That is asserted against both the in-repo demuxer (a
//! byte-chunked feed and a mid-Cluster cut) and ffmpeg reading the same stream
//! from a pipe, where the input is not seekable at all. `streamable` differs from
//! the default only in dropping the `Cues` index, the one part of the output that
//! needs the end of the stream.
//!
//! Two-pass (`seekable`): the caller buffers the file, so the Segment and every
//! Cluster reserve an 8-byte size that the finalize fills in beside the front
//! `SeekHead` (M770) and the `Info` `Duration` (M794). The result declares its
//! bounds and tiles exactly, element by element, the way ffmpeg and GStreamer
//! write a file. The wider Cluster header is in place before a `CueClusterPosition`
//! is recorded, so the index still lands on the Clusters.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::parse_launch;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, Dim, G2gError, OutputSink, PropValue, PushOutcome,
    Rate, VideoCodec,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::matroska::MatroskaDemuxer;
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};
use g2g_plugins::mkvmux::MkvMux;
use g2g_plugins::registry::default_registry;

const ID_SEGMENT: [u8; 4] = [0x18, 0x53, 0x80, 0x67];
const ID_CLUSTER: [u8; 4] = [0x1F, 0x43, 0xB6, 0x75];
const ID_CUES: [u8; 4] = [0x1C, 0x53, 0xBB, 0x6B];
const ID_SIMPLE_BLOCK: u8 = 0xA3;

#[derive(Default)]
struct CaptureSink {
    frames: Vec<(Vec<u8>, u64)>,
}

impl CaptureSink {
    fn bytes(&self) -> Vec<u8> {
        self.frames.iter().flat_map(|(b, _)| b.clone()).collect()
    }
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push((s.to_vec(), f.timing.pts_ns));
                }
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
            ..FrameTiming::default()
        },
        0,
    ))
}

fn vp9_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Vp9,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(25 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m828-{}-{name}", std::process::id()))
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

/// Payloads of varying length across a 3 s timeline, so the default 1 s Cluster
/// span opens several Clusters and a byte cut can land inside one.
fn script() -> Vec<(Vec<u8>, u64)> {
    (0..40u64)
        .map(|i| {
            let len = 40 + (i as usize % 7) * 130;
            let payload = (0..len).map(|b| (b as u8) ^ (i as u8)).collect();
            (payload, i * 80_000_000)
        })
        .collect()
}

/// Mux a script through the element, in whichever mode the flags select.
async fn mux(streamable: bool, seekable: bool, aus: &[(Vec<u8>, u64)]) -> Vec<u8> {
    let mut mux = MkvMux::new()
        .with_streamable(streamable)
        .with_seekable(seekable);
    mux.configure_pipeline(&vp9_caps()).expect("configure");
    let mut sink = CaptureSink::default();
    for (data, pts) in aus {
        mux.process(frame(data.clone(), *pts), &mut sink)
            .await
            .expect("mux frame");
    }
    mux.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    sink.bytes()
}

/// Feed a whole (or truncated) byte stream to the demuxer and collect its
/// `(payload, pts)` output. No EOS, so nothing is flushed that the bytes alone
/// did not yield: this is what a live reader has at that instant.
async fn demux(bytes: &[u8]) -> Vec<(Vec<u8>, u64)> {
    let mut demux = MkvDemux::new().with_stream(MkvStream::Vp9);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure demux");
    let mut sink = CaptureSink::default();
    demux
        .process(frame(bytes.to_vec(), 0), &mut sink)
        .await
        .expect("demux");
    sink.frames
}

/// `(size value, header length, is the unknown-size marker)` of the EBML size at
/// `pos`, the element-id length already skipped by the caller.
fn read_size(data: &[u8], pos: usize) -> (u64, usize, bool) {
    let first = data[pos];
    assert_ne!(first, 0, "an EBML size is never all-zero at {pos}");
    let len = (first.leading_zeros() + 1) as usize;
    let mut value = u64::from(first) & ((1u64 << (8 - len)) - 1);
    for b in &data[pos + 1..pos + len] {
        value = (value << 8) | u64::from(*b);
    }
    (value, len, value == (1u64 << (7 * len)) - 1)
}

/// `(value, byte length)` of the EBML element id at `pos`.
fn read_id(data: &[u8], pos: usize) -> (u32, usize) {
    let len = (data[pos].leading_zeros() + 1) as usize;
    assert!(len <= 4, "an EBML id is at most 4 bytes, at {pos}");
    let mut value = 0u32;
    for b in &data[pos..pos + len] {
        value = (value << 8) | u32::from(*b);
    }
    (value, len)
}

/// `(id, header start, end)` of the element at `pos`, by its declared size.
fn element_at(file: &[u8], pos: usize) -> (u32, usize, usize) {
    let (id, id_len) = read_id(file, pos);
    let (size, size_len, unknown) = read_size(file, pos + id_len);
    assert!(!unknown, "the element at {pos} declares a definite size");
    (
        id,
        pos + id_len + size_len,
        pos + id_len + size_len + size as usize,
    )
}

/// Walk a master element's children by their declared sizes and assert they tile
/// it exactly, ending on its last byte. A size that is short or long lands the
/// next read off an element boundary or past the end, so this is what proves each
/// backpatched length is the right one.
fn assert_children_tile(file: &[u8], data_at: usize, end: usize, label: &str) {
    let mut pos = data_at;
    while pos < end {
        let (_, _, child_end) = element_at(file, pos);
        assert!(
            child_end <= end,
            "a child of {label} ends inside it, at {pos}"
        );
        pos = child_end;
    }
    assert_eq!(pos, end, "the children of {label} tile it exactly");
}

/// Byte offsets of every `Cluster` element id in the stream.
fn cluster_offsets(file: &[u8]) -> Vec<usize> {
    (0..file.len().saturating_sub(4))
        .filter(|&i| file[i..i + 4] == ID_CLUSTER)
        .collect()
}

/// End offset (exclusive) of every `SimpleBlock` in the stream, in order. Walks
/// the Cluster children rather than scanning for the id, so a payload byte that
/// happens to equal `0xA3` is not mistaken for a block header.
fn simple_block_ends(file: &[u8]) -> Vec<usize> {
    let mut ends = Vec::new();
    for start in cluster_offsets(file) {
        // Cluster id (4 bytes) + its unknown-size marker (1 byte).
        let mut pos = start + 5;
        while pos < file.len() {
            // A level-1 id (4 bytes, 0x1n...) ends this Cluster.
            if file[pos] & 0xF0 == 0x10 {
                break;
            }
            let id_len = (file[pos].leading_zeros() + 1) as usize;
            let (size, size_len, _) = read_size(file, pos + id_len);
            let end = pos + id_len + size_len + size as usize;
            if file[pos] == ID_SIMPLE_BLOCK {
                ends.push(end);
            }
            pos = end;
        }
    }
    ends
}

/// The `streamable` knob is settable both ways the repo requires: the builder
/// and the runtime property `parse_launch` drives. It stays exclusive with the
/// two-pass `seekable` mode, whose backpatches a live stream cannot carry.
#[test]
fn streamable_property_round_trips() {
    let mux = MkvMux::new().with_streamable(true);
    assert_eq!(mux.get_property("streamable"), Some(PropValue::Bool(true)));

    let mut mux = MkvMux::new();
    assert_eq!(mux.get_property("streamable"), Some(PropValue::Bool(false)));
    assert!(mux
        .properties()
        .iter()
        .any(|p| p.name == "streamable" && p.kind == g2g_core::PropKind::Bool));
    mux.set_property("streamable", PropValue::Bool(true))
        .expect("set streamable");
    assert_eq!(mux.get_property("streamable"), Some(PropValue::Bool(true)));
    assert!(
        mux.set_property("seekable", PropValue::Bool(true)).is_err(),
        "a live stream has nowhere to put the two-pass backpatches"
    );

    // The same name through the launch text, on the single-input matroskamux.
    let reg = default_registry();
    parse_launch(
        &reg,
        "videotestsrc num-buffers=1 ! matroskamux streamable=true ! fakesink",
    )
    .expect("matroskamux takes streamable from a launch line");
}

/// The live shape in the bytes: an unknown-size Segment, an unknown-size Cluster
/// each, and no `Cues`. Nothing in the output depends on a length known only
/// after the fact.
#[tokio::test]
async fn streamable_writes_unknown_size_segment_and_clusters() {
    let file = mux(true, false, &script()).await;

    let seg = file
        .windows(4)
        .position(|w| w == ID_SEGMENT)
        .expect("a Segment element");
    let (_, size_len, unknown) = read_size(&file, seg + 4);
    assert!(unknown, "the Segment carries the unknown-size marker");
    assert_eq!(size_len, 8, "written as the 8-byte all-ones form");

    let clusters = cluster_offsets(&file);
    assert!(
        clusters.len() >= 3,
        "the 3 s script spans several Clusters, got {}",
        clusters.len()
    );
    for start in &clusters {
        let (_, size_len, unknown) = read_size(&file, start + 4);
        assert!(unknown, "the Cluster at {start} is unknown-size");
        assert_eq!(size_len, 1, "the one-byte 0xFF marker");
    }

    assert!(
        !file.windows(4).any(|w| w == ID_CUES),
        "no Cues index: it is the one part that needs the end of the stream"
    );
}

/// The two streaming modes write the same bytes: `streamable` differs from the
/// default only by the `Cues` index the default appends at EOS. The guard that
/// the two-pass work below changed nothing on the streaming path.
#[tokio::test]
async fn both_streaming_modes_write_the_same_bytes() {
    let aus = script();
    let live = mux(true, false, &aus).await;
    let default = mux(false, false, &aus).await;

    assert!(default.len() > live.len(), "the default appends an index");
    assert_eq!(
        &default[..live.len()],
        &live[..],
        "up to that index the two modes are byte for byte the same stream"
    );
    assert_eq!(
        &default[live.len()..live.len() + 4],
        &ID_CUES,
        "and the index is all it appends"
    );

    // The default keeps the streaming shape too, not just `streamable`.
    let seg = default
        .windows(4)
        .position(|w| w == ID_SEGMENT)
        .expect("a Segment element");
    assert!(read_size(&default, seg + 4).2, "unknown-size Segment");
    for start in cluster_offsets(&default) {
        assert_eq!(default[start + 4], 0xFF, "unknown-size Cluster at {start}");
    }
}

/// The two-pass file declares every bound: a definite-size Segment covering the
/// whole file, definite-size Clusters, and a child chain that tiles each master
/// element exactly. A backpatched length that is off by any amount breaks the
/// tiling.
#[tokio::test]
async fn seekable_writes_a_definite_size_segment_and_clusters() {
    let aus = script();
    let file = mux(false, true, &aus).await;

    let seg = file
        .windows(4)
        .position(|w| w == ID_SEGMENT)
        .expect("a Segment element");
    let (size, size_len, unknown) = read_size(&file, seg + 4);
    assert!(!unknown, "the two-pass Segment declares its length");
    assert_eq!(size_len, 8, "written into the reserved 8-byte field");
    let seg_data = seg + 4 + size_len;
    assert_eq!(
        seg_data + size as usize,
        file.len(),
        "and it spans the file to the last byte, the trailing Cues included"
    );

    let mut children = Vec::new();
    let mut pos = seg_data;
    while pos < file.len() {
        let (id, data_at, end) = element_at(&file, pos);
        assert!(
            end <= file.len(),
            "the element at {pos} ends inside the file"
        );
        assert_children_tile(&file, data_at, end, "a Segment child");
        children.push((id, pos));
        pos = end;
    }
    assert_eq!(pos, file.len(), "the Segment children tile it exactly");

    let cluster_id = u32::from_be_bytes(ID_CLUSTER);
    let clusters: Vec<usize> = children
        .iter()
        .filter(|(id, _)| *id == cluster_id)
        .map(|(_, at)| *at)
        .collect();
    assert!(
        clusters.len() >= 3,
        "the 3 s script spans several Clusters, got {}",
        clusters.len()
    );
    for start in &clusters {
        assert_eq!(
            read_size(&file, start + 4).1,
            8,
            "the Cluster at {start} sizes itself in the reserved 8-byte field"
        );
    }
    assert_eq!(
        children.last().map(|(id, _)| *id),
        Some(u32::from_be_bytes(ID_CUES)),
        "the Cues trail the Clusters, inside the Segment"
    );

    assert_eq!(demux(&file).await, aus, "and it demuxes to its input");
}

/// The `Cues` still point at the Clusters: a `CueClusterPosition` is recorded
/// when the Cluster opens, with the wider two-pass header already written, so
/// nothing shifts under the index.
#[tokio::test]
async fn seekable_cue_positions_land_on_the_clusters() {
    let aus = script();
    let file = mux(false, true, &aus).await;

    let mut demux = MatroskaDemuxer::new();
    demux.push_data(&file);
    assert_eq!(demux.take_frames().len(), aus.len(), "all frames recovered");
    let cues = demux.cues().to_vec();
    assert!(
        cues.len() >= 3,
        "one cue per Cluster holding a keyframe, got {}",
        cues.len()
    );
    for cue in &cues {
        let at = demux
            .cue_seek_offset(cue.time_ns)
            .expect("an indexed seek resolves") as usize;
        assert_eq!(
            &file[at..at + 4],
            &ID_CLUSTER,
            "the cue for {} ns lands on a Cluster id",
            cue.time_ns
        );
    }
}

/// The live-read property. Cut the stream inside a Cluster and the demuxer still
/// yields every block that arrived whole, sample-exact, and nothing past the cut.
#[tokio::test]
async fn a_truncated_stream_yields_every_complete_block() {
    let aus = script();
    let file = mux(true, false, &aus).await;
    let ends = simple_block_ends(&file);
    assert_eq!(ends.len(), aus.len(), "one SimpleBlock per input frame");

    // A cut inside the 25th block: past its header, short of its payload end.
    let cut = ends[24] - 5;
    assert!(
        cut > ends[23],
        "the cut lands inside a block, not between two"
    );
    let clusters = cluster_offsets(&file);
    assert!(
        clusters.iter().any(|&c| c < cut) && clusters.iter().any(|&c| c > cut),
        "the cut is mid-stream, inside a Cluster that never closes"
    );

    let got = demux(&file[..cut]).await;
    let want: Vec<(Vec<u8>, u64)> = aus[..24].to_vec();
    assert_eq!(
        got, want,
        "every block that arrived whole comes out, and none of the partial one"
    );
}

/// A live reader consuming the stream as it is written: frames come out as the
/// bytes arrive, well before the stream ends.
#[tokio::test]
async fn frames_come_out_before_the_stream_ends() {
    let aus = script();
    let file = mux(true, false, &aus).await;

    let mut demux = MkvDemux::new().with_stream(MkvStream::Vp9);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure demux");
    let mut sink = CaptureSink::default();
    let mut seen_at_half = 0;
    let chunk = 512;
    for (i, piece) in file.chunks(chunk).enumerate() {
        demux
            .process(frame(piece.to_vec(), 0), &mut sink)
            .await
            .expect("demux chunk");
        if (i + 1) * chunk >= file.len() / 2 && seen_at_half == 0 {
            seen_at_half = sink.frames.len();
        }
    }
    assert!(
        seen_at_half > 0,
        "frames are delivered from the first half of the stream"
    );
    assert_eq!(sink.frames, aus, "and the whole stream is sample-exact");
}

/// The live stream carries the same samples as the two-pass file of the same
/// input: the mode changes what is indexed, not what is muxed.
#[tokio::test]
async fn streamable_and_seekable_carry_the_same_samples() {
    let aus = script();
    let live = demux(&mux(true, false, &aus).await).await;
    let recorded = demux(&mux(false, true, &aus).await).await;
    assert_eq!(live, aus, "the live stream demuxes to its input");
    assert_eq!(
        recorded, live,
        "so does the two-pass file, sample for sample"
    );
}

/// Reference peer: ffmpeg reads the live stream from a pipe (an input it cannot
/// seek), reports the same stream metadata as the two-pass file, and decodes to
/// the same pixels.
#[tokio::test]
async fn ffmpeg_reads_the_live_stream_from_a_pipe() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    // A real VP9 stream, so ffmpeg has something to decode: ffmpeg authors it,
    // the g2g demuxer takes the frames out, and the g2g muxer rewrites them.
    let src = temp_path("src.webm");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg("testsrc=size=320x240:rate=25:duration=2")
        .args(["-c:v", "libvpx-vp9", "-b:v", "200k", "-cpu-used", "8"])
        .arg(&src)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the VP9 source");
    let aus = demux(&std::fs::read(&src).expect("read fixture")).await;
    assert_eq!(aus.len(), 50, "2 s at 25 fps");

    let live_path = temp_path("live.webm");
    let file_path = temp_path("file.webm");
    std::fs::write(&live_path, mux(true, false, &aus).await).expect("write live");
    std::fs::write(&file_path, mux(false, true, &aus).await).expect("write file");

    assert_eq!(
        probe_streams(&live_path),
        probe_streams(&file_path),
        "the live stream probes as the same 50-packet VP9 stream as the file"
    );
    assert_eq!(
        probed_duration(&file_path),
        Some(2.0),
        "the two-pass file declares its length, so ffprobe reads a duration"
    );
    assert_eq!(
        probed_duration(&live_path),
        None,
        "the live stream has none to declare"
    );

    let live_pixels = decode(&["-i", live_path.to_str().expect("path")]);
    let file_pixels = decode(&["-i", file_path.to_str().expect("path")]);
    assert_eq!(
        live_pixels.len(),
        50 * 320 * 240 * 3 / 2,
        "50 decoded yuv420p frames"
    );
    assert_eq!(
        live_pixels, file_pixels,
        "both modes decode to identical pixels"
    );

    // The live read itself: stdin is a pipe, so ffmpeg cannot seek the input.
    let piped = decode_from_pipe(&std::fs::read(&live_path).expect("read live"));
    assert_eq!(
        piped, live_pixels,
        "ffmpeg decodes the same pixels from a non-seekable pipe"
    );

    persist::record_evidence(
        "matroskamux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("vp9")
            .detail("ffmpeg decodes the streamable (unknown-size Cluster) output from a pipe to the same pixels as the seekable file"),
    )
    .expect("record oracle evidence");

    for p in [&src, &live_path, &file_path] {
        let _ = std::fs::remove_file(p);
    }
}

/// ffprobe's view of the file's one stream, plus its packet count.
fn probe_streams(path: &PathBuf) -> String {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_packets",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,nb_read_packets",
            "-of",
            "default=nw=1",
        ])
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(
        out.status.success(),
        "ffprobe read {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// The container duration ffprobe reports, or `None` for `N/A`. Matroska carries
/// it on the Segment, so this is the `format` field, not a stream's.
fn probed_duration(path: &PathBuf) -> Option<f64> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
        ])
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(
        out.status.success(),
        "ffprobe read {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// Decode to raw yuv420p bytes.
fn decode(input: &[&str]) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error"])
        .args(input)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .output()
        .expect("run ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg decoded: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// The same decode with the stream arriving on stdin, which ffmpeg cannot seek.
fn decode_from_pipe(bytes: &[u8]) -> Vec<u8> {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-i", "pipe:0"])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffmpeg");
    let mut stdin = child.stdin.take().expect("stdin");
    let owned = bytes.to_vec();
    let writer = std::thread::spawn(move || stdin.write_all(&owned));
    let out = child.wait_with_output().expect("wait ffmpeg");
    writer.join().expect("writer thread").expect("write stream");
    assert!(
        out.status.success(),
        "ffmpeg read the pipe: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}
