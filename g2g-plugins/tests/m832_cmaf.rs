//! M832: CMAF (ISO/IEC 23000-19) signalling on the fragmented-MP4 path.
//!
//! `cmaf=true` turns the fMP4 muxers into CMAF track-file writers: the `ftyp`
//! carries the `cmfc` structural brand, every fragment opens a CMAF segment with
//! its own `styp`, its `tfhd` states the sample description instead of
//! inheriting it, and a fragment starts only at a sync sample (so a fragment is
//! one GOP rather than one access unit). The layout was already CMAF-shaped
//! everywhere else: one `moof`+`mdat` per fragment with nothing between them, a
//! `tfdt` in every `traf`, `default-base-is-moof`, `mvex`/`trex`.
//!
//! The demux side accepts `styp` and the CMAF brands, and now honours the `tfhd`
//! `default_sample_duration` that a CMAF `trun` omits: without it every sample of
//! a fragment lands on the fragment's own `tfdt`.
//!
//! ffmpeg's `-movflags cmaf` is the oracle in both directions.
#![cfg(feature = "std")]

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

/// Child boxes of a container box's payload.
fn children(payload: &[u8]) -> Vec<([u8; 4], &[u8])> {
    top_level(payload)
}

fn child<'a>(payload: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    children(payload)
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

/// A `ftyp` / `styp` payload as (major brand, compatible brands).
fn brands(payload: &[u8]) -> (String, Vec<String>) {
    let major = String::from_utf8_lossy(&payload[0..4]).into_owned();
    let compat = payload[8..]
        .chunks_exact(4)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect();
    (major, compat)
}

fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(b[at..at + 4].try_into().unwrap())
}

/// A `trun` payload's per-sample flags. The writer sets data-offset + duration +
/// size + flags (0x000701), so each sample entry is 12 bytes after the 4-byte
/// version/flags, 4-byte count and 4-byte data offset.
fn trun_sample_flags(trun: &[u8]) -> Vec<u32> {
    assert_eq!(be32(trun, 0) & 0x00FF_FFFF, 0x0701, "trun field set");
    let count = be32(trun, 4) as usize;
    (0..count).map(|i| be32(trun, 12 + i * 12 + 8)).collect()
}

/// Whether a `trun` sample_flags value marks a sync sample (`sample_depends_on`
/// = 2, "this sample does not depend on others").
fn is_sync(sample_flags: u32) -> bool {
    (sample_flags >> 24) & 0x03 == 2
}

// --- sinks ----------------------------------------------------------------

#[derive(Default)]
struct CaptureSink {
    bytes: Vec<u8>,
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
                    self.bytes.extend_from_slice(s);
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Just the frame payloads a single-output element emits.
#[derive(Default)]
struct FrameCapture {
    frames: Vec<Vec<u8>>,
}

impl OutputSink for FrameCapture {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[derive(Default)]
struct PortCapture {
    ports: Vec<Vec<(Vec<u8>, FrameTiming)>>,
    caps: Vec<Option<Caps>>,
}

impl MultiOutputSink for PortCapture {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if self.ports.len() <= port {
                self.ports.resize(port + 1, Vec::new());
                self.caps.resize(port + 1, None);
            }
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.ports[port].push((s.to_vec(), f.timing));
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps[port] = Some(c),
                _ => {}
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

/// Six synthetic access units at 30 fps: IDR, four non-IDR, IDR. Enough shape for
/// the box writer, which never looks inside a slice.
fn synthetic_aus() -> Vec<Vec<u8>> {
    let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1e, 0x88];
    let pps: &[u8] = &[0x68, 0xce, 0x3c, 0x80];
    let idr: &[u8] = &[0x65, 0x88, 0x84, 0x00];
    let inter: &[u8] = &[0x41, 0x9a, 0x00];
    let key = annexb(&[sps, pps, idr]);
    vec![
        key.clone(),
        annexb(&[inter]),
        annexb(&[inter]),
        annexb(&[inter]),
        annexb(&[inter]),
        key,
    ]
}

/// Push `aus` through a single-track `Mp4Mux` and return the whole byte stream.
async fn mux_single(aus: &[Vec<u8>], cmaf: bool) -> Vec<u8> {
    let mut m = Mp4Mux::new().with_cmaf(cmaf);
    m.configure_pipeline(&h264_caps()).expect("configure");
    let mut sink = CaptureSink::default();
    for (i, au) in aus.iter().enumerate() {
        let timing = FrameTiming {
            pts_ns: i as u64 * 33_333_333,
            ..FrameTiming::default()
        };
        m.process(frame(au.clone(), timing), &mut sink)
            .await
            .expect("mux");
    }
    m.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    sink.bytes
}

/// The same through the multi-track `Mp4MuxN` on one pad.
async fn mux_n_single(aus: &[(Vec<u8>, FrameTiming)], cmaf: bool) -> Vec<u8> {
    let mut m = Mp4MuxN::new(1).with_cmaf(cmaf);
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
    sink.bytes
}

async fn demux_mp4(file: &[u8]) -> (Vec<Caps>, PortCapture) {
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
    tap.caps.resize(streams.len(), None);
    d.process(frame(file.to_vec(), FrameTiming::default()), &mut tap)
        .await
        .expect("demux");
    d.process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("demux eos");
    (streams.into_iter().map(|s| s.caps).collect(), tap)
}

// --- ffmpeg helpers -------------------------------------------------------

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m832-{}-{name}", std::process::id()))
}

/// A 2 s H.264 CMAF track file authored by ffmpeg's own `-movflags cmaf`, with a
/// 15-frame GOP so the file has four fragments.
fn author_cmaf(path: &PathBuf) -> Vec<u8> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg(format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=30:duration=2"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0"])
        .args(["-g", "15", "-movflags", "cmaf", "-f", "mp4"])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the CMAF fixture");
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
            "stream=codec_name,width,height",
            "-show_entries",
            "format=duration",
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

/// The `cmaf` property round-trips on both muxers, and the multi-track muxer
/// refuses the mode: a CMAF track file carries exactly one media stream, and the
/// mode has no meaning in the progressive (unfragmented) layout.
#[test]
fn cmaf_property_round_trips_and_multi_track_is_refused() {
    let mut single = Mp4Mux::new();
    assert_eq!(single.get_property("cmaf"), Some(PropValue::Bool(false)));
    single
        .set_property("cmaf", PropValue::Bool(true))
        .expect("set cmaf");
    assert_eq!(single.get_property("cmaf"), Some(PropValue::Bool(true)));
    assert!(single
        .properties()
        .iter()
        .any(|p| p.name == "cmaf" && p.default == Some("false")));
    assert_eq!(
        single.set_property("cmaf", PropValue::Uint(1)),
        Err(PropError::Type),
        "the property is a boolean"
    );

    // One pad: accepted, both through the property and the builder.
    let mut one = Mp4MuxN::new(1);
    one.set_property("cmaf", PropValue::Bool(true))
        .expect("set cmaf on a single-pad fan-in muxer");
    assert_eq!(one.get_property("cmaf"), Some(PropValue::Bool(true)));
    assert!(one.configure_pipeline(0, &h264_caps()).is_ok());

    // Two pads: the property is refused outright.
    let mut two = Mp4MuxN::new(2);
    assert_eq!(
        two.set_property("cmaf", PropValue::Bool(true)),
        Err(PropError::Value),
        "a CMAF track file holds one media stream"
    );
    assert_eq!(two.get_property("cmaf"), Some(PropValue::Bool(false)));

    // The builder bypasses `set_property`, so configuration is the backstop.
    let mut built = Mp4MuxN::new(2).with_cmaf(true);
    assert!(
        matches!(
            built.configure_pipeline(0, &h264_caps()),
            Err(G2gError::CapsMismatch)
        ),
        "a two-track CMAF file fails loud instead of being written"
    );

    // CMAF is a fragmented format; the progressive layout cannot carry it.
    let mut progressive = Mp4MuxN::new(1).with_cmaf(true).with_fragmented(false);
    assert!(
        matches!(
            progressive.configure_pipeline(0, &h264_caps()),
            Err(G2gError::CapsMismatch)
        ),
        "CMAF has no progressive layout"
    );
}

/// The box-level shape of a CMAF track file: `cmfc` brands, one track, `mvex` /
/// `trex`, a `styp` opening every segment, an `mdat` only ever directly after its
/// `moof`, a `tfdt` in every `traf`, an explicit sample description in the
/// `tfhd`, and a sync sample first in every fragment.
#[tokio::test]
async fn cmaf_track_file_carries_the_brands_styp_and_sync_aligned_fragments() {
    let file = mux_single(&synthetic_aus(), true).await;
    let boxes = top_level(&file);

    let (kind, ftyp) = boxes[0];
    assert_eq!(&kind, b"ftyp", "a track file starts with its ftyp");
    let (major, compat) = brands(ftyp);
    assert_eq!(
        major, "cmfc",
        "the CMAF structural brand is the major brand"
    );
    assert!(
        compat.contains(&"cmfc".to_string()),
        "and is repeated in the compatible brands: {compat:?}"
    );

    let moov = child(&file, b"moov").expect("moov");
    assert_eq!(
        children(moov).iter().filter(|(k, _)| k == b"trak").count(),
        1,
        "a CMAF track file carries exactly one track"
    );
    let mvex = child(moov, b"mvex").expect("mvex announces the fragments");
    assert!(child(mvex, b"trex").is_some(), "mvex carries a trex");

    // Segment structure: styp, moof, mdat, repeating. Nothing else at top level,
    // so no mdat ever sits outside a fragment.
    let after_header: Vec<[u8; 4]> = boxes[2..].iter().map(|(k, _)| *k).collect();
    assert!(
        after_header
            .chunks(3)
            .all(|c| c == [*b"styp", *b"moof", *b"mdat"]),
        "each segment is styp + moof + mdat: {:?}",
        after_header
            .iter()
            .map(|k| String::from_utf8_lossy(k).into_owned())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        after_header.len(),
        6,
        "two keyframes make two GOP-length segments"
    );

    let (styp_major, styp_compat) = brands(boxes[2].1);
    assert_eq!(styp_major, "cmfs", "the segment brand");
    assert!(
        styp_compat.contains(&"cmff".to_string()),
        "the segment is also one CMAF fragment: {styp_compat:?}"
    );

    for (kind, moof) in boxes.iter().filter(|(k, _)| k == b"moof") {
        assert_eq!(kind, b"moof");
        let traf = child(moof, b"traf").expect("traf");
        let tfdt = child(traf, b"tfdt").expect("every traf carries a tfdt");
        assert_eq!(tfdt[0], 1, "64-bit tfdt");
        let tfhd = child(traf, b"tfhd").expect("tfhd");
        assert_eq!(
            be32(tfhd, 0) & 0x00FF_FFFF,
            0x02_0002,
            "default-base-is-moof plus an explicit sample description index"
        );
        assert_eq!(be32(tfhd, 8), 1, "sample_description_index");
        let trun = child(traf, b"trun").expect("trun");
        let flags = trun_sample_flags(trun);
        assert!(
            is_sync(flags[0]),
            "a CMAF fragment starts at a stream access point"
        );
        assert!(
            flags[1..].iter().all(|f| !is_sync(*f)),
            "the rest of the GOP depends on it"
        );
    }

    // The same holds through the fan-in muxer on one pad.
    let timed: Vec<(Vec<u8>, FrameTiming)> = synthetic_aus()
        .into_iter()
        .enumerate()
        .map(|(i, au)| {
            (
                au,
                FrameTiming {
                    pts_ns: i as u64 * 33_333_333,
                    ..FrameTiming::default()
                },
            )
        })
        .collect();
    let n_file = mux_n_single(&timed, true).await;
    let n_boxes = top_level(&n_file);
    assert_eq!(&n_boxes[0].0, b"ftyp");
    assert_eq!(brands(n_boxes[0].1).0, "cmfc");
    let n_after: Vec<[u8; 4]> = n_boxes[2..].iter().map(|(k, _)| *k).collect();
    assert!(
        n_after
            .chunks(3)
            .all(|c| c == [*b"styp", *b"moof", *b"mdat"]),
        "the fan-in muxer writes the same segment shape"
    );
    assert_eq!(n_after.len(), 6);
}

/// The default (non-CMAF) output is untouched: the `iso5` `ftyp` byte for byte,
/// no `styp` anywhere, the bare `default-base-is-moof` `tfhd`, and one fragment
/// per access unit.
#[tokio::test]
async fn non_cmaf_output_keeps_its_exact_default_shape() {
    let aus = synthetic_aus();
    let file = mux_single(&aus, false).await;
    let boxes = top_level(&file);

    assert_eq!(&boxes[0].0, b"ftyp");
    assert_eq!(
        boxes[0].1, b"iso5\x00\x00\x02\x00iso5isom",
        "the default ftyp is unchanged"
    );
    assert!(
        !file.windows(4).any(|w| w == b"styp"),
        "the default layout has no segment boxes"
    );
    assert!(
        !file.windows(4).any(|w| w == b"cmfc"),
        "and claims no CMAF brand"
    );

    let after_header: Vec<[u8; 4]> = boxes[2..].iter().map(|(k, _)| *k).collect();
    assert!(
        after_header.chunks(2).all(|c| c == [*b"moof", *b"mdat"]),
        "moof + mdat with nothing between them"
    );
    assert_eq!(
        after_header.len(),
        aus.len() * 2,
        "the default is one fragment per access unit"
    );

    let tfhd = path(&file, &[b"moof", b"traf", b"tfhd"]).expect("tfhd");
    assert_eq!(tfhd.len(), 8, "version/flags + track_ID only");
    assert_eq!(be32(tfhd, 0) & 0x00FF_FFFF, 0x02_0000);
    assert_eq!(be32(tfhd, 4), 1, "track_ID");
}

/// ffmpeg reads a g2g CMAF track file and decodes it to the same pictures as the
/// source it was remuxed from.
#[tokio::test]
async fn ffmpeg_reads_a_g2g_cmaf_track_file() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("src-cmaf.mp4");
    let source = author_cmaf(&src);
    let (_, tap) = demux_mp4(&source).await;
    let aus: Vec<(Vec<u8>, FrameTiming)> = tap.ports[0].clone();
    assert_eq!(aus.len(), 60, "2 s at 30 fps");

    let out = temp_path("g2g-cmaf.mp4");
    let file = mux_n_single(&aus, true).await;
    std::fs::write(&out, &file).expect("write");

    let probed = probe(&out);
    println!("ffprobe g2g cmaf: {probed:?}");
    assert_eq!(field(&probed, "codec_name"), "h264");
    assert_eq!(field(&probed, "width"), WIDTH.to_string());
    assert_eq!(field(&probed, "height"), HEIGHT.to_string());
    // A fragmented file's duration is the sum of the sample durations ffprobe
    // reads back, which the 90 kHz tick rounding leaves a hair under 2 s.
    let duration: f64 = field(&probed, "duration").parse().expect("duration");
    assert!(
        (duration - 2.0).abs() < 0.01,
        "the 2 s source stays 2 s: {duration}"
    );

    let decoded = decode_video(&out);
    assert_eq!(
        decoded,
        decode_video(&src),
        "the CMAF remux decodes to the source's pictures"
    );
    assert_eq!(
        decoded.len() / (WIDTH * HEIGHT * 3 / 2),
        aus.len(),
        "every access unit comes back"
    );

    // Every fragment of the real stream still starts at a keyframe.
    for (_, moof) in top_level(&file).iter().filter(|(k, _)| k == b"moof") {
        let trun = path(moof, &[b"traf", b"trun"]).expect("trun");
        assert!(is_sync(trun_sample_flags(trun)[0]));
    }

    persist::record_evidence(
        "mp4mux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail("ffmpeg reads and decodes a g2g CMAF (cmfc) track file"),
    )
    .expect("record oracle evidence");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
}

/// g2g demuxes ffmpeg's `-movflags cmaf` output sample-exactly: the same access
/// units, with the same timing, as ffmpeg's progressive remux of the identical
/// encode. The timing half of that is what the `tfhd default_sample_duration`
/// support buys, since a CMAF `trun` carries no per-sample duration.
#[tokio::test]
async fn g2g_demuxes_ffmpeg_cmaf_sample_exact() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("oracle-cmaf.mp4");
    let cmaf = author_cmaf(&src);

    // The CMAF file carries the brand and one styp-free single-file layout;
    // whatever its shape, g2g must read it.
    let (major, compat) = brands(top_level(&cmaf)[0].1);
    assert!(
        major == "cmfc" || compat.contains(&"cmfc".to_string()),
        "ffmpeg marked it CMAF: {major} {compat:?}"
    );

    // The same encode, rewritten progressively, is the reference sample list.
    let prog = temp_path("oracle-progressive.mp4");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-c", "copy"])
        .arg(&prog)
        .status()
        .expect("run ffmpeg");
    assert!(
        status.success(),
        "ffmpeg remuxed the CMAF file progressively"
    );
    let progressive = std::fs::read(&prog).expect("read");

    let (_, from_cmaf) = demux_mp4(&cmaf).await;
    let (_, from_prog) = demux_mp4(&progressive).await;

    let cmaf_aus: Vec<&Vec<u8>> = from_cmaf.ports[0].iter().map(|(b, _)| b).collect();
    let prog_aus: Vec<&Vec<u8>> = from_prog.ports[0].iter().map(|(b, _)| b).collect();
    assert_eq!(cmaf_aus.len(), 60, "2 s at 30 fps");
    assert_eq!(
        cmaf_aus, prog_aus,
        "the CMAF fragments split into the same access units, byte for byte"
    );

    let cmaf_pts: Vec<u64> = from_cmaf.ports[0].iter().map(|(_, t)| t.pts_ns).collect();
    let prog_pts: Vec<u64> = from_prog.ports[0].iter().map(|(_, t)| t.pts_ns).collect();
    assert_eq!(cmaf_pts, prog_pts, "and at the same presentation times");
    assert!(
        cmaf_pts.windows(2).all(|w| w[1] > w[0]),
        "the tfhd default sample duration advances every sample: {:?}",
        &cmaf_pts[..8]
    );
    assert!(
        from_cmaf.ports[0].iter().all(|(_, t)| t.duration_ns > 0),
        "every sample has a duration"
    );

    persist::record_evidence(
        "mp4demux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail("g2g demuxes ffmpeg -movflags cmaf sample-exact, timing included"),
    )
    .expect("record oracle evidence");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&prog);
}

/// The byte-stream demuxer behind the HLS `#EXT-X-MAP` / DASH `SegmentBase`
/// CMAF paths reads a CMAF track file arriving in arbitrary chunks: the `styp`
/// opening each segment is skipped, not choked on, and the access units come out
/// whole.
#[tokio::test]
async fn fmp4demux_reads_a_cmaf_byte_stream_in_pieces() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("bytestream-src.mp4");
    let source = author_cmaf(&src);
    let (_, tap) = demux_mp4(&source).await;
    let expected: Vec<Vec<u8>> = tap.ports[0].iter().map(|(b, _)| b.clone()).collect();

    let cmaf = mux_n_single(&tap.ports[0], true).await;
    assert!(
        cmaf.windows(4).any(|w| w == b"styp"),
        "the stream under test carries segment boxes"
    );

    let mut d = g2g_plugins::fmp4demux::Fmp4Demux::new();
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("configure fmp4demux");
    let mut sink = FrameCapture::default();
    for piece in cmaf.chunks(1021) {
        d.process(frame(piece.to_vec(), FrameTiming::default()), &mut sink)
            .await
            .expect("demux");
    }
    d.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("demux eos");

    assert_eq!(
        sink.frames, expected,
        "the byte-stream demuxer recovers every access unit across the styp boundaries"
    );

    let _ = std::fs::remove_file(&src);
}

/// The full loop: ffmpeg CMAF in, g2g demux, g2g CMAF out, ffmpeg decode. The
/// pictures match the original, and a second g2g demux recovers the same access
/// units.
#[tokio::test]
async fn cmaf_round_trip_through_g2g_is_packet_exact() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("round-src.mp4");
    let source = author_cmaf(&src);
    let (_, first) = demux_mp4(&source).await;

    let remuxed = mux_n_single(&first.ports[0], true).await;
    let out = temp_path("round-out.mp4");
    std::fs::write(&out, &remuxed).expect("write");

    let (_, second) = demux_mp4(&remuxed).await;
    let before: Vec<&Vec<u8>> = first.ports[0].iter().map(|(b, _)| b).collect();
    let after: Vec<&Vec<u8>> = second.ports[0].iter().map(|(b, _)| b).collect();
    assert_eq!(
        after, before,
        "packets survive the CMAF remux byte for byte"
    );

    assert_eq!(
        decode_video(&out),
        decode_video(&src),
        "and ffmpeg decodes both to the same pictures"
    );

    persist::record_evidence(
        "mp4mux",
        &Evidence::new(ConformanceDimension::RoundTrip)
            .codec("h264")
            .detail("mp4demux -> cmaf mp4mux -> mp4demux is packet-exact"),
    )
    .expect("record round-trip evidence");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
}
