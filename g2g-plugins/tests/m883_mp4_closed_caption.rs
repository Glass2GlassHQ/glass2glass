//! M883: MP4 `c608` / `c708` raw closed-caption tracks. The muxer writes a `clcp`
//! track whose samples carry the caption atoms (`cdat` / `cdt2` byte pairs for
//! 608, a `ccdp` CDP for 708), the demuxer surfaces such a track as
//! `Caps::ClosedCaption` with the samples de-framed back to `cc_data` triples, and
//! `CcExtract` decodes that stream to text without a video bitstream in sight.
//!
//! The peer leg here is ffmpeg on the `c608` track: ffprobe must see it as an
//! `eia_608` subtitle stream and ffmpeg must decode it to the same caption text
//! (skipped when ffmpeg is missing). ffmpeg can neither write `c608` nor read
//! `c708`, so the write-side and the 708 peer checks are GStreamer `qtmux` /
//! `qtdemux` runs off CI.
#![cfg(feature = "std")]

use std::process::Command;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    ByteStreamEncoding, Caps, ClosedCaptionFormat, Dim, G2gError, MultiInputElement,
    MultiOutputElement, MultiOutputSink, OutputSink, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::ccextract::CcExtract;
use g2g_plugins::cea::{write_cc_data, Cc608Enc, Cc708Enc, CcTriple};
use g2g_plugins::mp4demuxn::{caption_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::mp4muxn::Mp4MuxN;
use g2g_plugins::subparse::Cue;

/// Nominal 30 fps caption cadence: one caption sample per video frame.
const FRAME_NS: u64 = 33_333_333;

fn cc_caps(format: ClosedCaptionFormat) -> Caps {
    Caps::ClosedCaption { format }
}

fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: FRAME_NS,
            ..FrameTiming::default()
        },
        0,
    ))
}

fn cue(text: &str) -> Cue {
    Cue {
        start_ns: 0,
        end_ns: 10 * FRAME_NS,
        text: text.into(),
        settings: Default::default(),
    }
}

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

#[derive(Default)]
struct TextSink {
    cues: Vec<String>,
}
impl OutputSink for TextSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.cues.push(String::from_utf8_lossy(s).into_owned());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[derive(Default)]
struct PortCapture {
    frames: Vec<(u64, Vec<u8>)>,
    caps: Vec<Caps>,
}
impl MultiOutputSink for PortCapture {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        _port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push((f.timing.pts_ns, s.to_vec()));
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }

    fn port_count(&self) -> usize {
        1
    }
}

/// The per-frame `cc_data` payloads that carry `text` as a CEA-608 pop-on
/// caption: the real encoder's byte pairs, one triple (field 1) per frame, then
/// idle padding so the caption's display command lands before the erase.
fn cea608_stream(text: &str, frames: usize) -> Vec<Vec<u8>> {
    let mut enc = Cc608Enc::new();
    enc.push_cue(&cue(text));
    (0..frames)
        .map(|_| {
            let (b0, b1) = enc.next_pair();
            write_cc_data(&[CcTriple { cc_type: 0, b0, b1 }])
        })
        .collect()
}

/// The per-frame `cc_data` payloads for the same caption as CEA-708 service 1.
fn cea708_stream(text: &str, frames: usize) -> Vec<Vec<u8>> {
    let mut enc = Cc708Enc::new();
    enc.push_cue(&cue(text));
    (0..frames)
        .map(|_| {
            let (cc_type, b0, b1) = enc.next_triple();
            write_cc_data(&[CcTriple { cc_type, b0, b1 }])
        })
        .collect()
}

/// Mux `payloads` as a raw-caption track alongside one H.264 video sample, so the
/// file is the realistic shape (a caption track hanging off a video movie).
/// `fragmented` selects the layout, so both sample-reading paths get covered: the
/// `moof` / `trun` fragments and the progressive `stbl` tables.
async fn mux_caption_mp4(
    format: ClosedCaptionFormat,
    payloads: &[Vec<u8>],
    fragmented: bool,
) -> Vec<u8> {
    let mut mux = Mp4MuxN::new(2).with_fragmented(fragmented);
    mux.configure_pipeline(
        0,
        &Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        },
    )
    .expect("video pad");
    mux.configure_pipeline(1, &cc_caps(format))
        .expect("caption pad");

    let mut sink = CaptureSink::default();
    let sps = [0x67u8, 0x42, 0x00, 0x1E, 0x88];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let mut au = Vec::new();
    for nal in [&sps[..], &pps[..], &idr[..]] {
        au.extend_from_slice(&[0, 0, 0, 1]);
        au.extend_from_slice(nal);
    }
    mux.process(0, frame(au, 0), &mut sink)
        .await
        .expect("video");
    for (i, p) in payloads.iter().enumerate() {
        mux.process(1, frame(p.clone(), i as u64 * FRAME_NS), &mut sink)
            .await
            .expect("caption sample");
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .expect("video eos");
    mux.process(1, PipelinePacket::Eos, &mut sink)
        .await
        .expect("caption eos");
    sink.bytes
}

/// Demux the caption track of `file`: the caps its port announced and the
/// `(pts, cc_data)` frames it forwarded.
async fn demux_captions(file: &[u8]) -> (Vec<Caps>, Vec<(u64, Vec<u8>)>) {
    let streams = caption_streams(file);
    assert_eq!(streams.len(), 1, "one caption track discovered");
    let ports = vec![Mp4Port {
        track_id: streams[0].track_id,
        caps: streams[0].caps.clone(),
    }];
    let mut demux = Mp4DemuxN::new(ports);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .expect("configure");
    let mut tap = PortCapture::default();
    demux
        .process(frame(file.to_vec(), 0), &mut tap)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("eos");
    (tap.caps, tap.frames)
}

/// Decode a demuxed caption stream to text through `CcExtract`, which negotiates
/// the `Caps::ClosedCaption` link and drives the same decoder the SEI path uses.
async fn decode_captions(format: ClosedCaptionFormat, frames: &[(u64, Vec<u8>)]) -> Vec<String> {
    let mut el = match format {
        ClosedCaptionFormat::Cea708 => CcExtract::cea708(1),
        _ => CcExtract::new(),
    };
    el.configure_pipeline(&cc_caps(format))
        .expect("CcExtract takes a raw-caption link");
    let mut sink = TextSink::default();
    for (pts, data) in frames {
        el.process(frame(data.clone(), *pts), &mut sink)
            .await
            .expect("decode");
    }
    el.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink.cues
}

fn contains_box(file: &[u8], fourcc: &[u8; 4]) -> bool {
    file.windows(4).any(|w| w == fourcc)
}

#[tokio::test]
async fn c608_track_round_trips_to_caption_text() {
    let payloads = cea608_stream("HELLO", 24);
    // Progressive: the samples come back through the `stbl` tables.
    let file = mux_caption_mp4(ClosedCaptionFormat::Cea608, &payloads, false).await;
    // The QuickTime shape: a closed-caption handler, the c608 sample entry, and
    // field-1 byte pairs in cdat atoms.
    assert!(
        contains_box(&file, b"clcp"),
        "closed-caption handler written"
    );
    assert!(contains_box(&file, b"c608"), "c608 sample entry written");
    assert!(contains_box(&file, b"cdat"), "field-1 cdat atoms written");

    let (caps, frames) = demux_captions(&file).await;
    assert!(
        caps.contains(&cc_caps(ClosedCaptionFormat::Cea608)),
        "the port announced CEA-608 caption caps, got {caps:?}"
    );
    assert_eq!(frames.len(), payloads.len(), "one frame per caption sample");
    let recovered: Vec<Vec<u8>> = frames.iter().map(|(_, d)| d.clone()).collect();
    assert_eq!(recovered, payloads, "cc_data triples recovered verbatim");

    let cues = decode_captions(ClosedCaptionFormat::Cea608, &frames).await;
    assert!(
        cues.iter().any(|c| c == "HELLO"),
        "the caption decodes to its text, got {cues:?}"
    );
}

#[tokio::test]
async fn c708_track_round_trips_to_caption_text() {
    let payloads = cea708_stream("HELLO", 24);
    // Fragmented: the samples come back through the `moof` / `trun` fragments.
    let file = mux_caption_mp4(ClosedCaptionFormat::Cea708, &payloads, true).await;
    assert!(contains_box(&file, b"c708"), "c708 sample entry written");
    assert!(contains_box(&file, b"ccdp"), "CDP sample atoms written");

    let (caps, frames) = demux_captions(&file).await;
    assert!(
        caps.contains(&cc_caps(ClosedCaptionFormat::Cea708)),
        "the port announced CEA-708 caption caps, got {caps:?}"
    );
    // Every frame carries a DTVCC triple (an idle one is a null continuation), so
    // each becomes its own CDP sample and comes back one for one.
    assert_eq!(frames.len(), payloads.len(), "one frame per caption sample");
    let cues = decode_captions(ClosedCaptionFormat::Cea708, &frames).await;
    assert!(
        cues.iter().any(|c| c.contains("HELLO")),
        "the 708 caption decodes to its text, got {cues:?}"
    );
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

/// ffprobe must recognize the g2g-written caption track as an `eia_608` subtitle
/// stream, and ffmpeg must decode it to the same caption text: the reader half of
/// the peer check (ffmpeg has no `c708` support, so this is the 608 track).
#[tokio::test]
async fn ffmpeg_reads_a_g2g_muxed_c608_track() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg / ffprobe not available");
        return;
    }
    let file = mux_caption_mp4(
        ClosedCaptionFormat::Cea608,
        &cea608_stream("HELLO", 24),
        false,
    )
    .await;
    let dir = std::env::temp_dir();
    let path = dir.join(format!("g2g-m883-{}.mp4", std::process::id()));
    std::fs::write(&path, &file).expect("write fixture");

    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
        ])
        .arg(&path)
        .output()
        .expect("run ffprobe");
    let codecs = String::from_utf8_lossy(&probe.stdout).trim().to_string();
    assert_eq!(codecs, "eia_608", "ffprobe sees the caption track");

    let srt = dir.join(format!("g2g-m883-{}.srt", std::process::id()));
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&path)
        .args(["-map", "0:s:0", "-f", "srt"])
        .arg(&srt)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg decodes the caption track");
    let text = std::fs::read_to_string(&srt).expect("read srt");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&srt);
    assert!(
        text.contains("HELLO"),
        "ffmpeg decodes the same caption text, got {text:?}"
    );
}
