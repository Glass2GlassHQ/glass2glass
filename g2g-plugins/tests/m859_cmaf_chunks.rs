//! M859: CMAF low-latency chunking on the fragmented-MP4 muxers.
//!
//! `chunk-duration` splits each fragment into CMAF chunks, a `moof`+`mdat` pair
//! over the samples buffered so far, emitted the moment the chunk fills instead
//! of at the end of the fragment. Only the chunk that opens a fragment carries
//! the segment's `styp` (which then also declares the CMAF chunk brand `cmfl`)
//! and its `prft`; every chunk carries its own `tfdt` and its own `mfhd`
//! sequence number, so the fragment's first chunk holds the fragment's base
//! decode time and the rest continue from it.
//!
//! `write-prft` adds the ProducerReferenceTimeBox that maps a fragment's first
//! decode time to the producer's wall clock, which is how a low-latency player
//! measures its own end-to-end latency.
//!
//! ffmpeg is the oracle: its own low-latency DASH output has exactly this
//! layout (one `styp` per segment, then `moof`+`mdat` per chunk with advancing
//! `tfdt` and `mfhd`), and it must decode a chunked g2g file to the same
//! pictures as the unchunked mux of the same access units.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement, MultiOutputElement,
    MultiOutputSink, OutputSink, PropError, PropValue, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::mp4mux::Mp4Mux;
use g2g_plugins::mp4muxn::Mp4MuxN;

const WIDTH: usize = 320;
const HEIGHT: usize = 240;
/// One frame at 25 fps, which is a whole number of 90 kHz ticks (3600), so the
/// chunk arithmetic in the expectations below is exact.
const FRAME_25FPS_NS: u64 = 40_000_000;
/// That frame in the muxers' 90 kHz video timescale.
const FRAME_25FPS_TICKS: u64 = 3_600;

// --- box walking ----------------------------------------------------------

/// The top-level boxes of an ISO-BMFF file as `(fourcc, payload)`, in order.
fn top_level(file: &[u8]) -> Vec<([u8; 4], &[u8])> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 8 <= file.len() {
        let size = u32::from_be_bytes(file[i..i + 4].try_into().unwrap()) as usize;
        assert!(
            size >= 8 && i + size <= file.len(),
            "box at {i} declares {size} bytes inside a {} byte file",
            file.len()
        );
        let kind: [u8; 4] = file[i + 4..i + 8].try_into().unwrap();
        out.push((kind, &file[i + 8..i + size]));
        i += size;
    }
    assert_eq!(i, file.len(), "the file is a whole number of boxes");
    out
}

fn child<'a>(payload: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    top_level(payload)
        .into_iter()
        .find(|(k, _)| k == kind)
        .map(|(_, p)| p)
}

fn path<'a>(mut payload: &'a [u8], kinds: &[&[u8; 4]]) -> Option<&'a [u8]> {
    for k in kinds {
        payload = child(payload, k)?;
    }
    Some(payload)
}

fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(b[at..at + 4].try_into().unwrap())
}

fn be64(b: &[u8], at: usize) -> u64 {
    u64::from_be_bytes(b[at..at + 8].try_into().unwrap())
}

/// A `ftyp` / `styp` payload as (major brand, compatible brands).
fn brands(payload: &[u8]) -> (String, Vec<String>) {
    let major = String::from_utf8_lossy(&payload[0..4]).into_owned();
    let compat = payload[8..]
        .chunks_exact(4)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect();
    (major, compat)
}

/// A fourcc string per top-level box, for readable layout assertions.
fn layout(file: &[u8]) -> Vec<String> {
    top_level(file)
        .iter()
        .map(|(k, _)| String::from_utf8_lossy(k).into_owned())
        .collect()
}

/// One `moof`'s (mfhd sequence_number, tfdt base decode time, trun sample count).
/// Both `tfdt` versions are read, since ffmpeg's own output is the other side of
/// these comparisons.
fn moof_fields(moof: &[u8]) -> (u32, u64, u32) {
    let mfhd = child(moof, b"mfhd").expect("mfhd");
    let traf = child(moof, b"traf").expect("traf");
    let tfdt = child(traf, b"tfdt").expect("tfdt");
    let base = match tfdt[0] {
        1 => be64(tfdt, 4),
        _ => be32(tfdt, 4) as u64,
    };
    let trun = child(traf, b"trun").expect("trun");
    (be32(mfhd, 4), base, be32(trun, 4))
}

/// Every `moof` of a file, in order.
fn moofs(file: &[u8]) -> Vec<&[u8]> {
    top_level(file)
        .into_iter()
        .filter(|(k, _)| k == b"moof")
        .map(|(_, p)| p)
        .collect()
}

/// A `prft` payload as (version, reference_track_ID, ntp_timestamp, media_time).
fn prft_fields(prft: &[u8]) -> (u8, u32, u64, u64) {
    assert_eq!(prft[0], 1, "version 1 (64-bit media_time)");
    (prft[0], be32(prft, 4), be64(prft, 8), be64(prft, 16))
}

/// The current time as a 64-bit NTP timestamp, the same mapping the muxer uses.
fn ntp_now() -> u64 {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    (d.as_secs() + 2_208_988_800) << 32
}

// --- sinks ----------------------------------------------------------------

#[derive(Default)]
struct CaptureSink {
    bytes: Vec<u8>,
    /// Byte-stream frames pushed downstream, one per chunk once chunking is on.
    frames: usize,
}

impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.bytes.extend_from_slice(s);
                    self.frames += 1;
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[derive(Default)]
struct PortCapture {
    ports: Vec<Vec<(Vec<u8>, FrameTiming)>>,
}

impl MultiOutputSink for PortCapture {
    fn push_to<'a>(
        &'a mut self,
        port: usize,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if self.ports.len() <= port {
                self.ports.resize(port + 1, Vec::new());
            }
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.ports[port].push((s.to_vec(), f.timing));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }

    fn port_count(&self) -> usize {
        self.ports.len().max(1)
    }
}

fn frame(data: Vec<u8>, timing: FrameTiming) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        timing,
        0,
    ))
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(WIDTH as u32),
        height: Dim::Fixed(HEIGHT as u32),
        framerate: Rate::Any,
    }
}

/// Annex-B framing for a list of NAL bodies.
fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

/// `groups` GOPs of `gop` access units each at 30 fps, every GOP opening on an
/// IDR. The box writer never looks inside a slice, so synthetic NALs suffice.
fn synthetic_aus(groups: usize, gop: usize) -> Vec<Vec<u8>> {
    let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1e, 0x88];
    let pps: &[u8] = &[0x68, 0xce, 0x3c, 0x80];
    let idr: &[u8] = &[0x65, 0x88, 0x84, 0x00];
    let inter: &[u8] = &[0x41, 0x9a, 0x00];
    let mut aus = Vec::new();
    for _ in 0..groups {
        aus.push(annexb(&[sps, pps, idr]));
        for _ in 1..gop {
            aus.push(annexb(&[inter]));
        }
    }
    aus
}

/// `aus` stamped at 25 fps, the form the fan-in muxer takes.
fn timed_aus(aus: &[Vec<u8>]) -> Vec<(Vec<u8>, FrameTiming)> {
    aus.iter()
        .enumerate()
        .map(|(i, au)| {
            (
                au.clone(),
                FrameTiming {
                    pts_ns: i as u64 * FRAME_25FPS_NS,
                    duration_ns: FRAME_25FPS_NS,
                    ..FrameTiming::default()
                },
            )
        })
        .collect()
}

/// Push `aus`, `step_ns` apart, through a single-track `Mp4Mux` and return the
/// byte stream plus the number of downstream frames it was pushed in.
async fn mux_single(aus: &[Vec<u8>], step_ns: u64, m: Mp4Mux) -> (Vec<u8>, usize) {
    let mut m = m;
    m.configure_pipeline(&h264_caps()).expect("configure");
    let mut sink = CaptureSink::default();
    for (i, au) in aus.iter().enumerate() {
        let timing = FrameTiming {
            pts_ns: i as u64 * step_ns,
            duration_ns: step_ns,
            ..FrameTiming::default()
        };
        m.process(frame(au.clone(), timing), &mut sink)
            .await
            .expect("mux");
    }
    m.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    (sink.bytes, sink.frames)
}

/// The same through the multi-track `Mp4MuxN` on one pad, with the timing the
/// samples carry.
async fn mux_n_single(aus: &[(Vec<u8>, FrameTiming)], m: Mp4MuxN) -> (Vec<u8>, usize) {
    let mut m = m;
    m.configure_pipeline(0, &h264_caps()).expect("configure");
    let mut sink = CaptureSink::default();
    for (au, timing) in aus {
        m.process(0, frame(au.clone(), *timing), &mut sink)
            .await
            .expect("mux");
    }
    m.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    (sink.bytes, sink.frames)
}

async fn demux_mp4(file: &[u8]) -> PortCapture {
    let streams = forwardable_streams(file);
    assert!(!streams.is_empty(), "the file has forwardable tracks");
    let ports: Vec<Mp4Port> = streams
        .iter()
        .map(|s| Mp4Port {
            track_id: s.track_id,
            caps: s.caps.clone(),
        })
        .collect();
    let mut d = Mp4DemuxN::new(ports);
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("configure mp4demux");
    let mut tap = PortCapture::default();
    tap.ports.resize(streams.len(), Vec::new());
    d.process(frame(file.to_vec(), FrameTiming::default()), &mut tap)
        .await
        .expect("demux");
    d.process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("demux eos");
    tap
}

// --- ffmpeg helpers -------------------------------------------------------

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m859-{}-{name}", std::process::id()))
}

/// A 2 s H.264 elementary stream in a plain fragmented MP4, 15-frame GOPs.
fn author_source(path: &PathBuf) -> Vec<u8> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg(format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=30:duration=2"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0"])
        .args(["-g", "15", "-movflags", "cmaf", "-f", "mp4"])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the fixture");
    std::fs::read(path).expect("read fixture")
}

/// ffmpeg's raw I420 decode of a file, failing on any decoder complaint.
fn decode_video(path: &PathBuf) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .output()
        .expect("run ffmpeg");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && err.is_empty(),
        "ffmpeg decoded {} cleanly: {err}",
        path.display()
    );
    assert!(!out.stdout.is_empty(), "ffmpeg decoded something");
    out.stdout
}

fn probe(path: &PathBuf) -> Vec<(String, String)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args([
            "-show_entries",
            "stream=codec_name,width,height,nb_read_packets",
            "-show_entries",
            "format=duration",
            "-count_packets",
            "-of",
            "default=nw=1",
        ])
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(
        out.status.success() && out.stderr.is_empty(),
        "ffprobe read {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn field<'a>(probed: &'a [(String, String)], key: &str) -> &'a str {
    probed
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("ffprobe reported {key}, got {probed:?}"))
}

// --- tests ----------------------------------------------------------------

/// Both new properties round-trip on both muxers, are typed, and default to the
/// pre-M859 behaviour.
#[test]
fn chunk_duration_and_prft_properties_round_trip() {
    let mut single = Mp4Mux::new();
    assert_eq!(
        single.get_property("chunk-duration"),
        Some(PropValue::Uint(0)),
        "unchunked by default"
    );
    assert_eq!(
        single.get_property("write-prft"),
        Some(PropValue::Bool(false))
    );
    single
        .set_property("chunk-duration", PropValue::Uint(200))
        .expect("set chunk-duration");
    single
        .set_property("write-prft", PropValue::Bool(true))
        .expect("set write-prft");
    assert_eq!(
        single.get_property("chunk-duration"),
        Some(PropValue::Uint(200))
    );
    assert_eq!(
        single.get_property("write-prft"),
        Some(PropValue::Bool(true))
    );
    assert_eq!(
        single.set_property("chunk-duration", PropValue::Bool(true)),
        Err(PropError::Type),
        "chunk-duration is an unsigned millisecond count"
    );
    for (name, default) in [("chunk-duration", "0"), ("write-prft", "false")] {
        assert!(
            single
                .properties()
                .iter()
                .any(|p| p.name == name && p.default == Some(default)),
            "{name} is declared with its default so parse_launch can set it"
        );
    }

    let mut fanin = Mp4MuxN::new(2);
    assert_eq!(
        fanin.get_property("chunk-duration"),
        Some(PropValue::Uint(0))
    );
    fanin
        .set_property("chunk-duration", PropValue::Uint(100))
        .expect("set chunk-duration");
    fanin
        .set_property("write-prft", PropValue::Bool(true))
        .expect("set write-prft");
    assert_eq!(
        fanin.get_property("chunk-duration"),
        Some(PropValue::Uint(100))
    );
    assert_eq!(
        fanin.get_property("write-prft"),
        Some(PropValue::Bool(true))
    );
    for name in ["chunk-duration", "write-prft"] {
        assert!(fanin.properties().iter().any(|p| p.name == name));
    }
}

/// The chunk layout: one `styp` per fragment (not per chunk), a `moof`+`mdat`
/// per chunk with its own increasing `mfhd` sequence number, the fragment's base
/// decode time on its first chunk and a continuing `tfdt` on the rest, and only
/// the fragment's first sample a sync sample.
#[tokio::test]
async fn chunks_subdivide_a_fragment_with_continuous_decode_times() {
    // Two 12-frame GOPs at 25 fps (480 ms each), chunked at 120 ms: three frames
    // a chunk, so four chunks per fragment.
    let aus = synthetic_aus(2, 12);
    let (file, frames) = mux_single(
        &aus,
        FRAME_25FPS_NS,
        Mp4Mux::new().with_cmaf(true).with_chunk_duration_ms(120),
    )
    .await;

    let kinds = layout(&file);
    assert_eq!(
        &kinds[..2],
        &["ftyp".to_string(), "moov".to_string()],
        "the init segment still leads"
    );
    let segments: Vec<usize> = kinds
        .iter()
        .enumerate()
        .filter(|(_, k)| *k == "styp")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        segments,
        vec![2, 11],
        "one styp per fragment, not per chunk"
    );
    assert!(
        kinds[3..11]
            .chunks(2)
            .all(|c| c == ["moof".to_string(), "mdat".to_string()]),
        "each chunk is a bare moof + mdat: {kinds:?}"
    );
    assert_eq!(kinds.len(), 20, "2 + 2 * (1 styp + 4 * 2 boxes): {kinds:?}");
    assert_eq!(
        frames, 9,
        "the init segment, then one byte-stream frame per chunk"
    );

    // The `cmfl` chunk brand appears once chunking is on, alongside the segment
    // and fragment brands the unchunked mode already declares.
    let (major, compat) = brands(top_level(&file)[2].1);
    assert_eq!(major, "cmfs");
    assert!(
        compat.contains(&"cmfl".to_string()) && compat.contains(&"cmff".to_string()),
        "the segment declares it is delivered as CMAF chunks: {compat:?}"
    );

    // Sequence numbers increase per moof; decode times are contiguous and every
    // sample is accounted for exactly once.
    let mut expect_tfdt = 0u64;
    let mut total_samples = 0u32;
    for (i, moof) in moofs(&file).iter().enumerate() {
        let (sequence, tfdt, samples) = moof_fields(moof);
        assert_eq!(sequence as usize, i + 1, "mfhd counts every chunk");
        assert_eq!(tfdt, expect_tfdt, "chunk {i} continues the decode timeline");
        assert_eq!(samples, 3, "120 ms of 25 fps video");
        expect_tfdt += samples as u64 * FRAME_25FPS_TICKS;
        total_samples += samples;
    }
    assert_eq!(total_samples as usize, aus.len());

    // Only a fragment's first chunk starts at a keyframe; the chunks after it are
    // mid-GOP, which is exactly what makes them cheaper than a fragment.
    let sync_first: Vec<bool> = moofs(&file)
        .iter()
        .map(|m| {
            let trun = path(m, &[b"traf", b"trun"]).expect("trun");
            (be32(trun, 12 + 8) >> 24) & 0x03 == 2
        })
        .collect();
    assert_eq!(
        sync_first,
        vec![true, false, false, false, true, false, false, false],
        "each fragment opens at a stream access point, its chunks do not"
    );
}

/// Chunking off (the default) is byte for byte what the muxer wrote before, in
/// every mode: per access unit, batched, and CMAF.
#[tokio::test]
async fn unchunked_output_is_unchanged() {
    let aus = synthetic_aus(2, 6);
    for (name, mux, reference) in [
        ("per-AU", Mp4Mux::new(), Mp4Mux::new()),
        (
            "batched",
            Mp4Mux::new().with_fragment_duration_ms(100),
            Mp4Mux::new().with_fragment_duration_ms(100),
        ),
        (
            "cmaf",
            Mp4Mux::new().with_cmaf(true),
            Mp4Mux::new().with_cmaf(true).with_chunk_duration_ms(0),
        ),
    ] {
        let (a, _) = mux_single(&aus, FRAME_25FPS_NS, mux).await;
        let (b, _) = mux_single(&aus, FRAME_25FPS_NS, reference).await;
        assert_eq!(a, b, "{name}: chunk-duration 0 changes nothing");
        assert!(
            !a.windows(4).any(|w| w == b"prft"),
            "{name}: no prft unless asked for"
        );
        assert!(
            !a.windows(4).any(|w| w == b"cmfl"),
            "{name}: no chunk brand unless chunking"
        );
    }

    // And the fan-in muxer's per-AU default likewise.
    let timed = timed_aus(&aus);
    let (a, _) = mux_n_single(&timed, Mp4MuxN::new(1)).await;
    let (b, _) = mux_n_single(&timed, Mp4MuxN::new(1).with_chunk_duration_ms(0)).await;
    assert_eq!(a, b);
    assert_eq!(
        moofs(&a).len(),
        aus.len(),
        "still one fragment per access unit"
    );
}

/// The `prft` sits ahead of each fragment's first `moof`, names the track, and
/// maps that fragment's own base decode time to a plausible wall clock. It is
/// written once per fragment, not once per chunk: one wall-clock anchor is all a
/// player needs to time the chunks that follow.
#[tokio::test]
async fn prft_anchors_each_fragment_to_the_producer_wall_clock() {
    let before = ntp_now();
    let aus = synthetic_aus(2, 12);
    let (file, _) = mux_single(
        &aus,
        FRAME_25FPS_NS,
        Mp4Mux::new()
            .with_cmaf(true)
            .with_chunk_duration_ms(120)
            .with_prft(true),
    )
    .await;
    let after = ntp_now();

    let kinds = layout(&file);
    let prfts: Vec<usize> = kinds
        .iter()
        .enumerate()
        .filter(|(_, k)| *k == "prft")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(prfts.len(), 2, "one prft per fragment: {kinds:?}");
    for at in &prfts {
        assert_eq!(kinds[at - 1], "styp", "the prft follows the segment's styp");
        assert_eq!(kinds[at + 1], "moof", "and precedes the fragment it times");
    }

    let boxes = top_level(&file);
    for at in prfts {
        let (version, track_id, ntp, media_time) = prft_fields(boxes[at].1);
        assert_eq!(version, 1);
        assert_eq!(track_id, 1, "the reference track");
        let (_, tfdt, _) = moof_fields(boxes[at + 1].1);
        assert_eq!(
            media_time, tfdt,
            "the producer time is anchored to the fragment's first sample"
        );
        assert!(
            (before..=after).contains(&(ntp & !0xFFFF_FFFF)),
            "the NTP seconds are the wall clock at mux time: {ntp:#x}"
        );
    }

    // The same on the fan-in muxer, where the box names that pad's track.
    let (n_file, _) = mux_n_single(
        &timed_aus(&aus),
        Mp4MuxN::new(1)
            .with_cmaf(true)
            .with_chunk_duration_ms(120)
            .with_prft(true),
    )
    .await;
    let n_kinds = layout(&n_file);
    assert_eq!(
        n_kinds.iter().filter(|k| *k == "prft").count(),
        2,
        "{n_kinds:?}"
    );
    let n_boxes = top_level(&n_file);
    let at = n_kinds.iter().position(|k| k == "prft").expect("prft");
    let (_, track_id, _, media_time) = prft_fields(n_boxes[at].1);
    assert_eq!(track_id, 1, "pad 0 is track_ID 1");
    assert_eq!(media_time, 0, "the first fragment starts the timeline");
}

/// ffmpeg reads a chunked, `prft`-carrying CMAF file of a real encode: the same
/// pictures as the unchunked mux of the identical access units, the expected
/// number of chunks, and g2g demuxes it back to the same access units.
#[tokio::test]
async fn ffmpeg_decodes_a_chunked_cmaf_file_identically_to_the_unchunked_mux() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("src.mp4");
    let source = author_source(&src);
    let aus = demux_mp4(&source).await.ports[0].clone();
    assert_eq!(aus.len(), 60, "2 s at 30 fps");

    // 15-frame GOPs (500 ms) chunked at 100 ms. A 30 fps frame is 33.33 ms, so a
    // chunk takes four of them to reach the target and each GOP splits 4/4/4/3.
    let chunked_mux = Mp4MuxN::new(1)
        .with_cmaf(true)
        .with_chunk_duration_ms(100)
        .with_prft(true);
    let (chunked, frames) = mux_n_single(&aus, chunked_mux).await;
    let (whole, whole_frames) = mux_n_single(&aus, Mp4MuxN::new(1).with_cmaf(true)).await;

    assert_eq!(whole_frames, 4, "one fragment per GOP");
    assert_eq!(frames, 16, "each fragment goes out as four chunks");
    assert_eq!(moofs(&chunked).len(), 16, "one moof per chunk");
    assert_eq!(moofs(&whole).len(), 4, "one moof per fragment");
    let counts: Vec<u32> = moofs(&chunked).iter().map(|m| moof_fields(m).2).collect();
    assert_eq!(
        counts,
        vec![4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 3],
        "a chunk closes on the first sample past the target"
    );
    assert_eq!(
        layout(&chunked).iter().filter(|k| *k == "styp").count(),
        4,
        "still one segment per fragment"
    );
    assert_eq!(
        layout(&chunked).iter().filter(|k| *k == "prft").count(),
        4,
        "one producer reference time per fragment"
    );
    // Every sample survives, exactly once, in decode order.
    let counted: u32 = moofs(&chunked).iter().map(|m| moof_fields(m).2).sum();
    assert_eq!(counted as usize, aus.len());

    let chunked_path = temp_path("chunked.mp4");
    let whole_path = temp_path("whole.mp4");
    std::fs::write(&chunked_path, &chunked).expect("write");
    std::fs::write(&whole_path, &whole).expect("write");

    let probed = probe(&chunked_path);
    println!("ffprobe chunked cmaf: {probed:?}");
    assert_eq!(field(&probed, "codec_name"), "h264");
    assert_eq!(field(&probed, "width"), WIDTH.to_string());
    assert_eq!(field(&probed, "height"), HEIGHT.to_string());
    assert_eq!(
        field(&probed, "nb_read_packets"),
        aus.len().to_string(),
        "the chunks demux to one packet per access unit"
    );
    let duration: f64 = field(&probed, "duration").parse().expect("duration");
    assert!(
        (duration - 2.0).abs() < 0.01,
        "the 2 s source stays 2 s: {duration}"
    );

    assert_eq!(
        decode_video(&chunked_path),
        decode_video(&whole_path),
        "chunking a fragment does not change a single decoded pixel"
    );

    // g2g reads its own chunked output back to the same access units.
    let back = demux_mp4(&chunked).await;
    let before: Vec<&Vec<u8>> = aus.iter().map(|(b, _)| b).collect();
    let after: Vec<&Vec<u8>> = back.ports[0].iter().map(|(b, _)| b).collect();
    assert_eq!(
        after, before,
        "packets survive the chunked remux byte for byte"
    );
    let pts: Vec<u64> = back.ports[0].iter().map(|(_, t)| t.pts_ns).collect();
    assert!(
        pts.windows(2).all(|w| w[1] > w[0]),
        "and keep their timeline across chunk boundaries"
    );

    persist::record_evidence(
        "mp4mux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail("ffmpeg decodes a chunked CMAF (cmfl) file with prft to the unchunked mux's pictures"),
    )
    .expect("record oracle evidence");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&chunked_path);
    let _ = std::fs::remove_file(&whole_path);
}

/// A chunked g2g segment has the same box layout as ffmpeg's own low-latency
/// DASH output: one `styp`, then a `moof`+`mdat` per chunk with an increasing
/// `mfhd` sequence number and an advancing `tfdt`.
#[tokio::test]
async fn chunk_layout_matches_ffmpegs_low_latency_dash_segments() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let dir = temp_path("lldash");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg(format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=30:duration=2"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0"])
        .args(["-g", "15", "-f", "dash", "-ldash", "1", "-streaming", "1"])
        .args(["-use_template", "1", "-seg_duration", "1"])
        .arg(dir.join("out.mpd"))
        .status()
        .expect("run ffmpeg");
    assert!(
        status.success(),
        "ffmpeg authored a low-latency DASH stream"
    );
    let segment = std::fs::read(dir.join("chunk-stream0-00001.m4s")).expect("read segment");

    let kinds = layout(&segment);
    assert_eq!(kinds[0], "styp", "ffmpeg opens the segment with one styp");
    assert!(
        kinds[1..]
            .chunks(2)
            .all(|c| c == ["moof".to_string(), "mdat".to_string()]),
        "and chunks it into bare moof + mdat pairs: {kinds:?}"
    );
    let mut prev = None;
    for moof in moofs(&segment) {
        let (sequence, tfdt, _) = moof_fields(moof);
        if let Some((prev_seq, prev_tfdt)) = prev {
            assert_eq!(sequence, prev_seq + 1, "sequence numbers count chunks");
            assert!(tfdt > prev_tfdt, "each chunk advances the decode time");
        }
        prev = Some((sequence, tfdt));
    }

    // g2g's own chunked segment has that same shape.
    let aus = synthetic_aus(1, 24);
    let (file, _) = mux_single(
        &aus,
        FRAME_25FPS_NS,
        Mp4Mux::new().with_cmaf(true).with_chunk_duration_ms(120),
    )
    .await;
    let g2g_kinds = layout(&file);
    assert_eq!(g2g_kinds[2], "styp");
    assert!(g2g_kinds[3..]
        .chunks(2)
        .all(|c| c == ["moof".to_string(), "mdat".to_string()]));

    let _ = std::fs::remove_dir_all(&dir);
}
