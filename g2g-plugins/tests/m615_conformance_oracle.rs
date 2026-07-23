//! M615: a reference-implementation (`ffmpeg`) conformance oracle that produces
//! *persisted* `Oracle` evidence, the tier the in-process battery cannot reach.
//!
//! g2g's native fragmented-MP4 muxer (`Mp4MuxN`) muxes a synthetic H.264 + AAC
//! stream into an ISO-BMFF byte stream; `ffprobe` (a real external demuxer) is asked
//! to read it back. If ffprobe demuxes both tracks, the native muxer is validated
//! against an independent implementation, so we record `Oracle` evidence (peer =
//! ffmpeg) for `mp4mux` in the shared conformance log. `full_report` then folds that
//! log into the in-process batteries and derives `mp4mux` as `InteropTested`, which
//! is exactly what `g2g-inspect --maturity` shows once this test has run. Self-skips
//! where ffprobe is absent (no false failure on a bare CI box).
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence, MaturityLevel};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, Caps, Dim, G2gError, MultiInputElement, OutputSink, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::mp4muxn::Mp4MuxN;

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

fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

fn adts_au(payload: &[u8]) -> Vec<u8> {
    let frame_len = payload.len() + 7;
    let sr_index = 3u8; // 48000
    let channels = 2u8;
    let mut au = vec![
        0xFF,
        0xF1,
        (1 << 6) | (sr_index << 2) | ((channels >> 2) & 1),
        ((channels & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F,
        0xFC,
    ];
    au.extend_from_slice(payload);
    au
}

/// Mux a two-track (H.264 + AAC) fragmented MP4 from synthetic access units, exactly
/// as `m293_mp4mux_av` does, and return the byte stream.
async fn mux_fmp4() -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];

    let h264 = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    };
    let aac = Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48000,
    };

    let mut mux = Mp4MuxN::new(2);
    mux.configure_pipeline(0, &h264).unwrap();
    mux.configure_pipeline(1, &aac).unwrap();
    let mut sink = CaptureSink::default();

    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0x01, 0x02, 0x03]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0x04, 0x05]), 21_000_000), &mut sink)
        .await
        .unwrap();
    mux.process(
        0,
        frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000),
        &mut sink,
    )
    .await
    .unwrap();
    mux.process(1, frame(adts_au(&[0x06, 0x07]), 42_000_000), &mut sink)
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

#[tokio::test]
async fn ffmpeg_validates_the_native_mp4_muxer_and_records_interop_evidence() {
    // Self-skip on a box without ffprobe.
    if Command::new("ffprobe").arg("-version").output().is_err() {
        eprintln!("ffprobe not present; skipping the interop oracle");
        return;
    }

    // Use a dedicated, freshly-truncated log so the assertion is deterministic,
    // unless a log path is already set (a CI conformance run sets
    // $G2G_CONFORMANCE_LOG to aggregate every oracle's evidence): then append to
    // that shared log and leave it in place. The assertions below search by element
    // name, so other elements' rows in the shared log do not disturb them.
    let external = std::env::var_os("G2G_CONFORMANCE_LOG");
    let log = match &external {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let l = std::env::temp_dir().join("g2g-conformance-m615.tsv");
            std::env::set_var("G2G_CONFORMANCE_LOG", &l);
            let _ = std::fs::remove_file(&l);
            l
        }
    };

    let bytes = mux_fmp4().await;
    assert_eq!(&bytes[4..8], b"ftyp", "native muxer emits ISO-BMFF");
    // The in-process structural round-trip is UnitTested-tier evidence.
    persist::record_evidence(
        "mp4mux",
        &Evidence::new(ConformanceDimension::RoundTrip).detail("ISO-BMFF ftyp/moov/trak structure"),
    )
    .expect("record round-trip evidence");

    // Write the muxed stream and ask ffprobe (the external peer) to demux it.
    let mp4 = std::env::temp_dir().join("g2g-conformance-m615.mp4");
    std::fs::write(&mp4, &bytes).expect("write mp4");
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nw=1:nk=1",
        ])
        .arg(&mp4)
        .output()
        .expect("run ffprobe");
    let codecs = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "ffprobe accepted the native fMP4: {codecs}"
    );
    assert!(
        codecs.contains("h264"),
        "ffprobe demuxed the H.264 track: {codecs}"
    );
    assert!(
        codecs.contains("aac"),
        "ffprobe demuxed the AAC track: {codecs}"
    );

    // ffprobe validated it: record the interop Oracle evidence (peer = ffmpeg).
    persist::record_evidence(
        "mp4mux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264+aac")
            .detail("ffprobe demuxes both tracks from the native fMP4"),
    )
    .expect("record oracle evidence");

    // The persisted evidence folds into the report and lifts mp4mux to InteropTested,
    // which is what `g2g-inspect --maturity` will now show for it.
    let report = persist::full_report();
    let mp4mux = report
        .records
        .iter()
        .find(|r| r.element == "mp4mux")
        .expect("mp4mux present after persisting evidence");
    assert!(
        mp4mux.has(ConformanceDimension::Oracle),
        "oracle evidence persisted"
    );
    assert_eq!(mp4mux.peers(), vec!["ffmpeg"]);
    assert_eq!(
        mp4mux.level(),
        MaturityLevel::InteropTested,
        "native muxer validated against ffmpeg reaches interop-tested",
    );

    // Only remove the log if we created a dedicated one; leave a shared CI log for
    // the aggregate maturity report.
    if external.is_none() {
        let _ = std::fs::remove_file(&log);
    }
    let _ = std::fs::remove_file(&mp4);
}
