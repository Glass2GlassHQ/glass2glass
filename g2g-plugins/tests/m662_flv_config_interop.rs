//! M662 FLV codec-config side channel, validated against ffmpeg as the
//! reference peer (both directions):
//!
//! - **Demux:** ffmpeg encodes a real H.264 + AAC FLV; `FlvDemux` extracts each
//!   elementary stream. The video must leave as Annex-B with the `avcC`
//!   parameter sets prepended in-band, the audio as self-describing ADTS, so
//!   ffmpeg itself can decode both extracted streams standalone (the proof the
//!   side channel makes them decodable without the container).
//! - **Mux:** `FlvMux` muxes an ffmpeg-encoded H.264 elementary stream into an
//!   FLV, capturing the sequence header from the first IDR; ffprobe must read
//!   the result back as a playable h264 FLV (the M615 oracle discipline).
//!
//! Self-skips where ffmpeg/ffprobe are absent.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, Dim, G2gError, OutputSink, PushOutcome,
    Rate, VideoCodec,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::flvdemux::{FlvDemux, FlvStream};
use g2g_plugins::flvmux::FlvMux;

fn have(bin: &str) -> bool {
    Command::new(bin).arg("-version").output().is_ok()
}

/// Point the persisted-evidence log at a shared temp file unless a CI
/// conformance run already set one ($G2G_CONFORMANCE_LOG aggregates every
/// oracle's rows). Never truncated here: both tests in this file append to it
/// concurrently and search their own element's rows.
fn ensure_conformance_log() {
    if std::env::var_os("G2G_CONFORMANCE_LOG").is_none() {
        std::env::set_var(
            "G2G_CONFORMANCE_LOG",
            std::env::temp_dir().join("g2g-conformance-m662.tsv"),
        );
    }
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

#[derive(Default)]
struct CaptureSink {
    bytes: Vec<u8>,
    caps: Vec<Caps>,
    frames: usize,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let MemoryDomain::System(s) = &f.domain {
                        self.bytes.extend_from_slice(s.as_slice());
                    }
                    self.frames += 1;
                }
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Encode a short synthetic A/V FLV with ffmpeg (H.264 + AAC 44.1 kHz stereo).
fn ffmpeg_encode_flv(path: &PathBuf) {
    let out = Command::new("ffmpeg")
        .args(["-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=25"])
        .args(["-f", "lavfi", "-i", "sine=frequency=440:duration=1"])
        .args(["-c:v", "libx264", "-preset", "ultrafast", "-g", "12"])
        .args(["-c:a", "aac", "-ar", "44100", "-ac", "2"])
        .arg(path)
        .output()
        .expect("ffmpeg runs");
    assert!(out.status.success(), "ffmpeg encode failed: {}", String::from_utf8_lossy(&out.stderr));
}

/// Run the demux element over the whole FLV byte stream.
async fn demux(flv: &[u8], stream: FlvStream) -> CaptureSink {
    let mut d = FlvDemux::new().with_stream(stream);
    d.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::Flv }).unwrap();
    let mut sink = CaptureSink::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(flv.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    d.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();
    d.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink
}

/// ffmpeg decodes an elementary stream file; returns whether it produced frames.
fn ffmpeg_decodes(path: &PathBuf, format: &str) -> bool {
    let out = Command::new("ffmpeg")
        .args(["-y", "-f", format, "-i"])
        .arg(path)
        .args(["-f", "null", "-"])
        .output()
        .expect("ffmpeg runs");
    out.status.success()
}

#[tokio::test]
async fn ffmpeg_flv_demuxes_to_self_describing_elementary_streams() {
    if !have("ffmpeg") {
        eprintln!("ffmpeg not present; skipping the FLV interop test");
        return;
    }
    let flv_path = tmp("g2g_m662_src.flv");
    ffmpeg_encode_flv(&flv_path);
    let flv = std::fs::read(&flv_path).expect("read the encoded FLV");

    // Video: Annex-B out, SPS/PPS in-band at the front (from the avcC).
    let video = demux(&flv, FlvStream::H264).await;
    assert!(video.frames >= 20, "a second of 25 fps video demuxed, got {}", video.frames);
    assert_eq!(&video.bytes[..4], &[0, 0, 0, 1], "Annex-B start code leads");
    assert_eq!(video.bytes[4] & 0x1F, 7, "the first NAL is the prepended SPS");
    let h264_path = tmp("g2g_m662_video.h264");
    std::fs::write(&h264_path, &video.bytes).unwrap();
    assert!(
        ffmpeg_decodes(&h264_path, "h264"),
        "ffmpeg decodes the extracted elementary stream standalone"
    );

    // Audio: ADTS out (self-describing), concrete caps announced from the ASC.
    let audio = demux(&flv, FlvStream::Aac).await;
    assert!(audio.frames >= 40, "a second of AAC frames demuxed, got {}", audio.frames);
    assert_eq!(&audio.bytes[..2], &[0xFF, 0xF1], "ADTS syncword leads");
    assert_eq!(
        audio.caps,
        vec![Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 44_100 }],
        "concrete audio caps from the AudioSpecificConfig"
    );
    let aac_path = tmp("g2g_m662_audio.aac");
    std::fs::write(&aac_path, &audio.bytes).unwrap();
    assert!(ffmpeg_decodes(&aac_path, "aac"), "ffmpeg decodes the extracted ADTS standalone");

    // ffmpeg (the external peer) validated both extracted streams: record the
    // interop Oracle evidence for the demuxer (the M615 mechanism).
    ensure_conformance_log();
    persist::record_evidence(
        "flvdemux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264+aac")
            .detail("ffmpeg decodes both extracted elementary streams standalone"),
    )
    .expect("record oracle evidence");
    assert!(
        persist::full_report().records.iter().any(|r| r.element == "flvdemux"),
        "flvdemux row present after persisting evidence"
    );

    for p in [&flv_path, &h264_path, &aac_path] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn flvmux_output_probes_as_playable_h264_flv() {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("ffmpeg/ffprobe not present; skipping the FLV mux oracle");
        return;
    }
    // A real Annex-B H.264 elementary stream from ffmpeg (SPS/PPS on the IDR).
    let h264_path = tmp("g2g_m662_mux_in.h264");
    let out = Command::new("ffmpeg")
        .args(["-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=25"])
        .args(["-c:v", "libx264", "-preset", "ultrafast", "-g", "12", "-f", "h264"])
        .arg(&h264_path)
        .output()
        .expect("ffmpeg runs");
    assert!(out.status.success(), "ffmpeg encode failed: {}", String::from_utf8_lossy(&out.stderr));
    let es = std::fs::read(&h264_path).unwrap();

    // The whole elementary stream as one access unit is enough here: the muxer
    // captures the sequence header from the first parameter sets it sees, which
    // is what this oracle is proving.
    let caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(25 << 16),
    };
    let mut mux = FlvMux::new();
    mux.configure_pipeline(&caps).unwrap();
    let mut sink = CaptureSink::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(es.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    mux.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();

    let flv_path = tmp("g2g_m662_mux_out.flv");
    std::fs::write(&flv_path, &sink.bytes).unwrap();

    // ffprobe reads the FLV back: container recognized, h264 stream present.
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0"])
        .arg(&flv_path)
        .output()
        .expect("ffprobe runs");
    assert!(probe.status.success(), "ffprobe failed: {}", String::from_utf8_lossy(&probe.stderr));
    let codecs = String::from_utf8_lossy(&probe.stdout);
    assert!(codecs.contains("h264"), "ffprobe sees the h264 stream, got: {codecs}");

    // And ffmpeg can decode the FLV end to end (the sequence header works).
    assert!(ffmpeg_decodes(&flv_path, "flv"), "ffmpeg decodes the muxed FLV");

    // ffprobe + ffmpeg validated the native FLV: record the interop Oracle
    // evidence for the muxer (the M615 mechanism).
    ensure_conformance_log();
    persist::record_evidence(
        "flvmux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail("ffprobe demuxes and ffmpeg decodes the native FLV"),
    )
    .expect("record oracle evidence");
    assert!(
        persist::full_report().records.iter().any(|r| r.element == "flvmux"),
        "flvmux row present after persisting evidence"
    );

    for p in [&h264_path, &flv_path] {
        let _ = std::fs::remove_file(p);
    }
}
