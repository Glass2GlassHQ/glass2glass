//! M619: grow the conformance matrix with a second reference-implementation oracle,
//! the native MPEG-TS muxer (`TsMux`) validated by `ffprobe`. Different container
//! from the M615 MP4 oracle (transport stream vs ISO-BMFF), so it exercises a
//! distinct native muxer and adds a second `InteropTested` row.
//!
//! `TsMux` muxes synthetic H.264 access units into a `Caps::ByteStream{MpegTs}`
//! stream; ffprobe demuxes it back and must report an h264 stream. On success we
//! persist peer-tagged `Oracle` evidence for `mpegtsmux`, which `full_report`
//! derives as `InteropTested`. Self-skips without ffprobe.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence, MaturityLevel};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, OutputSink, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::tsmux::TsMux;

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
                if let MemoryDomain::System(s) = &f.domain {
                    self.bytes.extend_from_slice(s.as_slice());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
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

async fn mux_ts() -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    };
    let mut mux = TsMux::new();
    mux.configure_pipeline(&caps).unwrap();
    let mut sink = CaptureSink::default();
    mux.process(frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink).await.unwrap();
    mux.process(frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000), &mut sink).await.unwrap();
    mux.process(frame(annexb(&[&[0x41u8, 0x9a, 0x01]]), 66_000_000), &mut sink).await.unwrap();
    mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.bytes
}

#[tokio::test]
async fn ffmpeg_validates_the_native_ts_muxer_and_records_interop_evidence() {
    if Command::new("ffprobe").arg("-version").output().is_err() {
        eprintln!("ffprobe not present; skipping the TS interop oracle");
        return;
    }

    // Dedicated freshly-truncated log standalone; append to a shared CI log when
    // $G2G_CONFORMANCE_LOG is already set (assertions search by element name).
    let external = std::env::var_os("G2G_CONFORMANCE_LOG");
    let log = match &external {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let l = std::env::temp_dir().join("g2g-conformance-m619.tsv");
            std::env::set_var("G2G_CONFORMANCE_LOG", &l);
            let _ = std::fs::remove_file(&l);
            l
        }
    };

    let bytes = mux_ts().await;
    assert!(!bytes.is_empty(), "TsMux emitted a transport stream");
    // MPEG-TS packets are 188 bytes starting with the 0x47 sync byte.
    assert_eq!(bytes[0], 0x47, "TS sync byte");
    assert_eq!(bytes.len() % 188, 0, "whole TS packets");

    let ts = std::env::temp_dir().join("g2g-conformance-m619.ts");
    std::fs::write(&ts, &bytes).expect("write ts");
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1"])
        .arg(&ts)
        .output()
        .expect("run ffprobe");
    let codecs = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "ffprobe accepted the native TS: {codecs}");
    assert!(codecs.contains("h264"), "ffprobe demuxed the H.264 elementary stream: {codecs}");

    persist::record_evidence(
        "mpegtsmux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail("ffprobe demuxes the H.264 stream from the native MPEG-TS"),
    )
    .expect("record oracle evidence");

    let report = persist::full_report();
    let ts_mux = report
        .records
        .iter()
        .find(|r| r.element == "mpegtsmux")
        .expect("mpegtsmux present after persisting evidence");
    assert_eq!(ts_mux.peers(), vec!["ffmpeg"]);
    assert_eq!(ts_mux.level(), MaturityLevel::InteropTested);

    if external.is_none() {
        let _ = std::fs::remove_file(&log);
    }
    let _ = std::fs::remove_file(&ts);
}
