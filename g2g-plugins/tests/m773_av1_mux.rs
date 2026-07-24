//! M773: AV1 in the fan-in muxers. `MkvMuxN` writes a `V_AV1` track whose
//! `CodecPrivate` is the `AV1CodecConfigurationRecord` and `Mp4MuxN` an `av01`
//! sample entry with the `av1C` box, both built from the stream's own sequence
//! header; samples store the temporal unit with its temporal-delimiter OBUs
//! stripped (the ISOBMFF / Matroska AV1 mappings), keyframes flagged from the
//! frame headers. The record is byte-compared against the `av1C` ffmpeg's own
//! muxer writes for the same stream, and the Matroska side round-trips through
//! our demuxer.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, Dim, G2gError, MultiInputElement, OutputSink, PushOutcome, Rate, VideoCodec};
use g2g_plugins::matroska::{MatroskaDemuxer, MkvCodec};
use g2g_plugins::mkvmuxn::MkvMuxN;
use g2g_plugins::mp4muxn::Mp4MuxN;

fn av1_caps(w: u32, h: u32) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Av1,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[derive(Default)]
struct CaptureSink {
    bytes: Vec<u8>,
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

/// Encode a short AV1 clip (two GOPs) to IVF with ffmpeg's libaom, or `None`
/// when the host cannot. `tag` keeps the two tests (parallel, same process)
/// off each other's file.
fn encode_ivf(tag: &str) -> Option<(std::path::PathBuf, Vec<Vec<u8>>)> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return None;
    }
    let path = std::env::temp_dir().join(format!("g2g-m773-{tag}-{}.ivf", std::process::id()));
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

/// The payload of the first `name` box in an ISO-BMFF byte stream.
fn find_box_payload(data: &[u8], name: &[u8; 4]) -> Option<Vec<u8>> {
    let at = data.windows(4).position(|w| w == name)?;
    let size = u32::from_be_bytes(data[at - 4..at].try_into().unwrap()) as usize;
    Some(data[at + 4..at - 4 + size].to_vec())
}

#[tokio::test]
async fn av1_mp4_av1c_matches_ffmpeg() {
    let Some((ivf, frames)) = encode_ivf("mp4") else {
        eprintln!("skipping: no ffmpeg / libaom");
        return;
    };

    let mut mux = Mp4MuxN::new(1);
    mux.configure_pipeline(0, &av1_caps(192, 108)).unwrap();
    let mut sink = CaptureSink::default();
    for (i, f) in frames.iter().enumerate() {
        mux.process(0, frame(f.clone(), i as u64 * 33_333_333), &mut sink)
            .await
            .unwrap();
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    let ours = sink.bytes;
    assert!(
        ours.windows(4).any(|w| w == b"av01"),
        "an av01 sample entry is written"
    );
    let our_av1c = find_box_payload(&ours, b"av1C").expect("an av1C box is written");

    // Oracle: ffmpeg's own mux of the same stream writes the same record.
    let ref_mp4 = std::env::temp_dir().join(format!("g2g-m773-{}-ref.mp4", std::process::id()));
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&ivf)
        .args(["-c", "copy"])
        .arg(&ref_mp4)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg remuxes the ivf");
    let reference = std::fs::read(&ref_mp4).unwrap();
    let ref_av1c = find_box_payload(&reference, b"av1C").expect("ffmpeg writes av1C");
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&ref_mp4);

    assert_eq!(
        our_av1c, ref_av1c,
        "our av1C record is byte-identical to ffmpeg's"
    );
}

#[tokio::test]
async fn av1_matroska_track_round_trips() {
    let Some((ivf, frames)) = encode_ivf("mkv") else {
        eprintln!("skipping: no ffmpeg / libaom");
        return;
    };
    let _ = std::fs::remove_file(&ivf);

    let mut mux = MkvMuxN::new(1);
    mux.configure_pipeline(0, &av1_caps(192, 108)).unwrap();
    let mut sink = CaptureSink::default();
    for (i, f) in frames.iter().enumerate() {
        mux.process(0, frame(f.clone(), i as u64 * 33_333_333), &mut sink)
            .await
            .unwrap();
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    let mkv = sink.bytes;

    let mut demux = MatroskaDemuxer::new();
    demux.push_data(&mkv);
    let out = demux.take_frames();
    assert_eq!(out.len(), 16, "every temporal unit recovered");
    let mut keyframes = 0;
    for (i, f) in out.iter().enumerate() {
        assert_eq!(f.codec, MkvCodec::Av1, "frame {i} on the V_AV1 track");
        assert!(!f.data.is_empty(), "frame {i} carries data");
        // Samples store the unit with temporal delimiters stripped: a TD OBU
        // (type 2) never opens a stored sample.
        let first_type = (f.data[0] >> 3) & 0x0F;
        assert_ne!(
            first_type, 2,
            "frame {i} starts past the temporal delimiter"
        );
        if f.keyframe {
            keyframes += 1;
            assert!(i == 0 || i == 8, "keyframes at the GOP heads, got {i}");
        }
    }
    assert_eq!(keyframes, 2, "two GOPs -> two sync samples");
}
