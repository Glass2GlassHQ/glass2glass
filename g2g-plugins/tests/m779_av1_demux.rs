//! M779: AV1 demux. The MP4 demuxers read the `av01` sample entry (config from
//! the `av1C` record; samples are plain low-overhead OBUs, passed through
//! verbatim), completing the M773 mux round trip. Oracled two ways: our own
//! `Mp4MuxN` output demuxes back to byte-identical temporal units, and
//! ffmpeg's own mux of the same stream demuxes to the same units. The mkv
//! side (`V_AV1`, already demuxable) gets the same element-level round trip.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement, OutputSink,
    PropValue, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::fmp4demux::Fmp4Demux;
use g2g_plugins::mkvdemux::MkvDemux;
use g2g_plugins::mkvmuxn::MkvMuxN;
use g2g_plugins::mp4demux::Mp4Demux;
use g2g_plugins::mp4muxn::Mp4MuxN;

fn av1_caps(w: u32, h: u32) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Av1,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Concatenating byte sink (for the muxers).
#[derive(Default)]
struct ByteSink {
    bytes: Vec<u8>,
}
impl OutputSink for ByteSink {
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

/// Frame-capturing sink (for the demuxers).
#[derive(Default)]
struct FrameSink {
    caps: Vec<Caps>,
    frames: Vec<(bool, Vec<u8>)>,
}
impl OutputSink for FrameSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push((f.timing.keyframe, s.to_vec()));
                    }
                }
                _ => {}
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

/// The frames of an IVF file (12-byte "DKIF" header, then per-frame 4-byte LE
/// size + 8-byte pts + payload).
fn ivf_frames(ivf: &[u8]) -> Vec<Vec<u8>> {
    assert_eq!(&ivf[..4], b"DKIF", "an IVF file");
    let header_len = u16::from_le_bytes([ivf[6], ivf[7]]) as usize;
    let mut frames = Vec::new();
    let mut at = header_len;
    while at + 12 <= ivf.len() {
        let size = u32::from_le_bytes(ivf[at..at + 4].try_into().unwrap()) as usize;
        at += 12;
        frames.push(ivf[at..at + size].to_vec());
        at += size;
    }
    frames
}

/// Strip temporal-delimiter OBUs from a temporal unit (the container mappings
/// do not store them), by the low-overhead OBU walk.
fn strip_tds(unit: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < unit.len() {
        let start = pos;
        let header = unit[pos];
        pos += 1;
        let obu_type = (header >> 3) & 0x0F;
        if (header >> 2) & 1 == 1 {
            pos += 1; // extension byte
        }
        let size = if (header >> 1) & 1 == 1 {
            // leb128
            let mut v = 0usize;
            for shift in (0..).step_by(7) {
                let b = unit[pos];
                pos += 1;
                v |= ((b & 0x7F) as usize) << shift;
                if b & 0x80 == 0 {
                    break;
                }
            }
            v
        } else {
            unit.len() - pos
        };
        let end = pos + size;
        if obu_type != 2 {
            out.extend_from_slice(&unit[start..end]);
        }
        pos = end;
    }
    out
}

/// Encode a short AV1 clip (two GOPs) to IVF with ffmpeg's libaom, or `None`
/// when the host cannot.
fn encode_ivf(tag: &str) -> Option<(std::path::PathBuf, Vec<Vec<u8>>)> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return None;
    }
    let path = std::env::temp_dir().join(format!("g2g-m779-{tag}-{}.ivf", std::process::id()));
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=192x108:rate=30",
            "-frames:v",
            "16",
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "8",
            "-g",
            "8",
            "-f",
            "ivf",
        ])
        .arg(&path)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let frames = ivf_frames(&std::fs::read(&path).ok()?);
    (frames.len() == 16).then_some((path, frames))
}

/// Demux an MP4 byte stream's video track, returning caps + frames.
async fn demux_mp4(bytes: &[u8]) -> FrameSink {
    let mut d = Mp4Demux::new();
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Mp4,
    })
    .expect("configure");
    let mut sink = FrameSink::default();
    for piece in bytes.chunks(4096) {
        d.process(frame(piece.to_vec(), 0), &mut sink)
            .await
            .expect("demux");
    }
    d.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink
}

/// Assert the demuxed track is the AV1 stream: refined caps, every temporal
/// unit byte-identical to the TD-stripped source, keyframes at the GOP heads.
fn assert_av1_track(sink: &FrameSink, expected: &[Vec<u8>]) {
    assert_eq!(
        sink.caps.last(),
        Some(&Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Fixed(192),
            height: Dim::Fixed(108),
            framerate: Rate::Any,
        }),
        "moov-refined AV1 caps"
    );
    assert_eq!(sink.frames.len(), expected.len(), "every unit recovered");
    let mut keyframes = Vec::new();
    for (i, ((kf, data), reference)) in sink.frames.iter().zip(expected).enumerate() {
        assert_eq!(data, reference, "unit {i} is byte-identical");
        if *kf {
            keyframes.push(i);
        }
    }
    assert_eq!(keyframes, [0, 8], "sync samples at the GOP heads");
}

/// Our own M773 mux (fragmented MP4) demuxes back to byte-identical temporal
/// units through `Fmp4Demux` (the same `av01` parse the progressive path uses).
#[tokio::test]
async fn av1_mp4_round_trips() {
    let Some((ivf, frames)) = encode_ivf("roundtrip") else {
        eprintln!("skipping: no ffmpeg / libaom");
        return;
    };
    let _ = std::fs::remove_file(&ivf);

    let mut mux = Mp4MuxN::new(1);
    MultiInputElement::configure_pipeline(&mut mux, 0, &av1_caps(192, 108)).unwrap();
    let mut mp4 = ByteSink::default();
    for (i, f) in frames.iter().enumerate() {
        mux.process(0, frame(f.clone(), i as u64 * 33_333_333), &mut mp4)
            .await
            .unwrap();
    }
    mux.process(0, PipelinePacket::Eos, &mut mp4).await.unwrap();

    let mut d = Fmp4Demux::new();
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("configure");
    let mut sink = FrameSink::default();
    for piece in mp4.bytes.chunks(4096) {
        d.process(frame(piece.to_vec(), 0), &mut sink)
            .await
            .expect("demux");
    }
    d.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");

    let expected: Vec<Vec<u8>> = frames.iter().map(|f| strip_tds(f)).collect();
    assert_av1_track(&sink, &expected);
}

/// ffmpeg's own mux of the same stream demuxes to the same temporal units
/// (the reference-peer direction).
#[tokio::test]
async fn av1_ffmpeg_mp4_demuxes() {
    let Some((ivf, frames)) = encode_ivf("ffmpeg") else {
        eprintln!("skipping: no ffmpeg / libaom");
        return;
    };
    let mp4_path = std::env::temp_dir().join(format!("g2g-m779-ref-{}.mp4", std::process::id()));
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&ivf)
        .args(["-c", "copy"])
        .arg(&mp4_path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg remuxes the ivf");
    let mp4 = std::fs::read(&mp4_path).unwrap();
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&mp4_path);

    let expected: Vec<Vec<u8>> = frames.iter().map(|f| strip_tds(f)).collect();
    let sink = demux_mp4(&mp4).await;
    assert_av1_track(&sink, &expected);
}

/// The mkv element (`matroskademux stream=av1`) round-trips the M773 mux at
/// the element level (caps + byte-identical units).
#[tokio::test]
async fn av1_mkv_element_round_trips() {
    let Some((ivf, frames)) = encode_ivf("mkv") else {
        eprintln!("skipping: no ffmpeg / libaom");
        return;
    };
    let _ = std::fs::remove_file(&ivf);

    let mut mux = MkvMuxN::new(1);
    MultiInputElement::configure_pipeline(&mut mux, 0, &av1_caps(192, 108)).unwrap();
    let mut mkv = ByteSink::default();
    for (i, f) in frames.iter().enumerate() {
        mux.process(0, frame(f.clone(), i as u64 * 33_333_333), &mut mkv)
            .await
            .unwrap();
    }
    mux.process(0, PipelinePacket::Eos, &mut mkv).await.unwrap();

    let mut d = MkvDemux::new();
    d.set_property("stream", PropValue::Str("av1".into()))
        .expect("stream property");
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Matroska,
    })
    .expect("configure");
    let mut sink = FrameSink::default();
    for piece in mkv.bytes.chunks(4096) {
        d.process(frame(piece.to_vec(), 0), &mut sink)
            .await
            .expect("demux");
    }
    d.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");

    let expected: Vec<Vec<u8>> = frames.iter().map(|f| strip_tds(f)).collect();
    assert_eq!(sink.frames.len(), expected.len(), "every unit recovered");
    for (i, ((_, data), reference)) in sink.frames.iter().zip(&expected).enumerate() {
        assert_eq!(data, reference, "unit {i} is byte-identical");
    }
}
