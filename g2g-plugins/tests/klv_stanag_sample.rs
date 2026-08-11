//! Validation against a real STANAG 4609 UAS capture (M806): the public
//! "Day Flight" sample (samples.ffmpeg.org, MPEG2/mpegts-klv), a 2009 EO
//! sensor flight muxed by a real airborne encoder: 1280x720 H.264 plus an
//! asynchronous ST 0601 KLV stream. Local-only, like the AV1 vector tests:
//! point `G2G_STANAG_SAMPLE` at the file (it is ~100 MB, not committed), and
//! the ffmpeg cross-check additionally needs ffmpeg on PATH.
//!
//! Notable: every packet in the real capture carries a VALID checksum (the
//! strict parse passes), unlike the published MISMMS doc fixture (m801), which
//! is exactly why `klvdecode` keeps strict the default and lenient a knob.

use std::process::Command;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    ByteStreamEncoding, Caps, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome,
};

use g2g_core::element::AsyncElement;
use g2g_plugins::klv::{split_klv_packets, UasDatalink};
use g2g_plugins::tsdemux::{TsDemux, TsStream};

/// First-packet telemetry, cross-derived with klvdata from the same bytes.
const FIRST_TIMESTAMP_US: u64 = 1_245_257_585_099_653; // 2009-06-17 16:53:05.099653Z
const FIRST_LAT: f64 = 54.681_323_284_600_55;
const FIRST_LON: f64 = -110.168_559_770_178_3;
const FIRST_ALT: f64 = 1_532.272_831_311_512_6;

fn sample_path() -> Option<std::path::PathBuf> {
    let p = std::env::var_os("G2G_STANAG_SAMPLE")?;
    Some(std::path::PathBuf::from(p))
}

#[derive(Default)]
struct CaptureSink {
    aus: Vec<Vec<u8>>,
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
                    self.aus.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Demux the selected stream from the whole file, fed in 64 KiB chunks to
/// exercise TS packet realignment across input frame boundaries.
async fn demux_file(path: &std::path::Path, stream: TsStream) -> Vec<Vec<u8>> {
    let bytes = std::fs::read(path).expect("read sample");
    let mut demux = TsDemux::new().with_stream(stream);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .unwrap();
    let mut sink = CaptureSink::default();
    for chunk in bytes.chunks(64 * 1024) {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        demux
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
    }
    demux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.aus
}

/// The real capture's KLV stream demuxes to strictly-valid ST 0601 sets with
/// the known first-packet telemetry, and the video stream demuxes alongside.
#[tokio::test]
async fn real_capture_demuxes_and_parses_strict() {
    let Some(path) = sample_path() else {
        eprintln!("skipping: set G2G_STANAG_SAMPLE to the Day Flight sample");
        return;
    };

    let klv_frames = demux_file(&path, TsStream::Klv).await;
    assert!(!klv_frames.is_empty(), "KLV stream present");
    let packets: Vec<&[u8]> = klv_frames
        .iter()
        .flat_map(|f| split_klv_packets(f))
        .collect();
    assert_eq!(packets.len(), 6, "the sample carries six local sets");

    // A real airborne encoder writes valid checksums: strict parse passes on
    // every packet.
    let sets: Vec<UasDatalink> = packets
        .iter()
        .map(|p| UasDatalink::parse(p).expect("strict parse (valid checksum)"))
        .collect();

    let first = &sets[0];
    assert_eq!(first.timestamp_us, Some(FIRST_TIMESTAMP_US));
    let close = |got: Option<f64>, want: f64, eps: f64| {
        assert!((got.unwrap() - want).abs() <= eps, "{got:?} != {want}");
    };
    close(first.sensor_lat_deg, FIRST_LAT, 1e-6);
    close(first.sensor_lon_deg, FIRST_LON, 1e-6);
    close(first.sensor_alt_m, FIRST_ALT, 0.4);
    assert_eq!(first.version, Some(1));

    // Timestamps advance monotonically across the flight's sets.
    let ts: Vec<u64> = sets.iter().map(|s| s.timestamp_us.unwrap()).collect();
    assert!(ts.windows(2).all(|w| w[0] < w[1]), "monotonic: {ts:?}");

    // The video elementary stream demuxes alongside (real 720p H.264).
    let video = demux_file(&path, TsStream::H264).await;
    assert!(video.len() > 100, "real video AUs: {}", video.len());
}

/// Our demux recovers byte-for-byte what ffmpeg extracts from the same file.
#[tokio::test]
async fn real_capture_matches_ffmpeg_extraction() {
    let Some(path) = sample_path() else {
        eprintln!("skipping: set G2G_STANAG_SAMPLE to the Day Flight sample");
        return;
    };
    let have_ffmpeg = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !have_ffmpeg {
        eprintln!("skipping: no ffmpeg");
        return;
    }

    let out = std::env::temp_dir().join(format!("g2g-dayflight-{}.klv", std::process::id()));
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "fatal", "-i"])
        .arg(&path)
        .args(["-map", "0:d:0", "-c", "copy", "-f", "data"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success(), "ffmpeg extracts the data stream");
    let ffmpeg_bytes = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_file(&out);

    let ours: Vec<u8> = demux_file(&path, TsStream::Klv).await.concat();
    assert_eq!(
        ours, ffmpeg_bytes,
        "bit-exact with ffmpeg on the real capture"
    );
}
