//! M1001 reference-decoder PSNR oracle: decode a committed fixture with the ffmpeg
//! CLI and with the in-repo decoder, then measure one against the other.
//!
//! The golden batteries (`m1001_golden_decode`) pin g2g against its own past output,
//! which catches drift but would happily preserve a decode that was wrong from the
//! start. This one asks an independent implementation what the fixture contains, so a
//! pass is evidence about correctness rather than stability, and it is persisted as
//! peer-tagged `Oracle` evidence alongside the `Quality` row: an element validated
//! here derives `InteropTested`.
//!
//! Self-skips where the ffmpeg CLI is absent, like the other oracles, so a bare box
//! reports nothing rather than failing.
#![cfg(feature = "std")]
#![allow(dead_code)] // the helpers below serve whichever codec features are on

mod m1001_common;

use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use m1001_common::EvidenceLog;

/// The battery's own log name, so a local run does not inherit another's rows.
const LOG: &str = "m1001-oracle";

/// Whether the ffmpeg CLI is on this box.
fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

/// Decode `fixture` with the ffmpeg CLI into raw `pixel_format` frames.
fn ffmpeg_decode(fixture: &str, pixel_format: &str) -> Vec<u8> {
    let out: PathBuf = std::env::temp_dir().join(format!(
        "g2g_m1001_oracle_{}_{fixture}.raw",
        std::process::id()
    ));
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(m1001_common::fixture_path(fixture))
        .args(["-pix_fmt", pixel_format, "-f", "rawvideo", "-y"])
        .arg(&out)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg decoded {fixture}");
    let raw = std::fs::read(&out).expect("ffmpeg wrote raw frames");
    let _ = std::fs::remove_file(&out);
    raw
}

/// Record a passed oracle: the measured `Quality` figure plus the peer-tagged
/// `Oracle` row that lifts the element to `InteropTested`.
fn record_oracle(log: &EvidenceLog, element: &str, codec: &str, db: f64, floor: f64) {
    eprintln!("{element} vs ffmpeg: {db:.2} dB PSNR (floor {floor:.0} dB)");
    let detail = if db.is_infinite() {
        "bit-identical to ffmpeg's own decode of the same fixture".to_string()
    } else {
        format!("within {db:.1} dB PSNR of ffmpeg's own decode of the same fixture")
    };
    log.record(
        element,
        &Evidence::new(ConformanceDimension::Quality)
            .codec(codec)
            .detail(detail.clone()),
    );
    log.record(
        element,
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec(codec)
            .detail(detail),
    );
}

// -- AV1: rav1d against ffmpeg's AV1 decoder ---------------------------------

#[cfg(feature = "rav1d")]
mod av1 {
    use super::*;
    use g2g_core::{AsyncElement, Caps, Dim, PipelinePacket, Rate, VideoCodec};
    use g2g_plugins::conformance::{i420_planes, pooled_psnr_db};
    use g2g_plugins::rav1ddec::Rav1dDec;
    use m1001_common::{data_frame, split_temporal_units, CaptureSink};

    const CLIP: &[u8] = include_bytes!("fixtures/av1_640x480.obu");
    const WIDTH: usize = 640;
    const HEIGHT: usize = 480;

    /// AV1 decode is bit-exact by the specification, so the two decoders must agree
    /// sample for sample and the measured figure is infinite. The floor is finite
    /// only so a near-miss reports a number instead of a bare inequality.
    const FLOOR_DB: f64 = 80.0;

    #[tokio::test]
    async fn rav1d_matches_ffmpegs_decode_of_the_av1_fixture() {
        if !have_ffmpeg() {
            eprintln!("ffmpeg not present; skipping the AV1 decode oracle");
            return;
        }
        let log = EvidenceLog::scoped(LOG);

        let mut decoder = Rav1dDec::new();
        decoder
            .configure_pipeline(&Caps::CompressedVideo {
                codec: VideoCodec::Av1,
                width: Dim::Fixed(WIDTH as u32),
                height: Dim::Fixed(HEIGHT as u32),
                framerate: Rate::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            })
            .expect("rav1d configures");
        let mut sink = CaptureSink::default();
        for (i, unit) in split_temporal_units(CLIP).iter().enumerate() {
            decoder
                .process(
                    data_frame(unit.to_vec(), i as u64 * 33_000_000, i as u64),
                    &mut sink,
                )
                .await
                .expect("decode a temporal unit");
        }
        decoder
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("drain");

        let frame_bytes = WIDTH * HEIGHT * 3 / 2;
        let reference = ffmpeg_decode("av1_640x480.obu", "yuv420p");
        assert_eq!(
            reference.len() / frame_bytes,
            sink.frames.len(),
            "both decoders produced the same frame count"
        );

        let mut planes = Vec::new();
        for (i, measured) in sink.frames.iter().enumerate() {
            let their_frame = &reference[i * frame_bytes..(i + 1) * frame_bytes];
            let theirs = i420_planes(their_frame, WIDTH, HEIGHT).expect("ffmpeg frame is I420");
            let ours = i420_planes(measured, WIDTH, HEIGHT).expect("our frame is I420");
            for plane in 0..3 {
                planes.push((theirs[plane], ours[plane]));
            }
        }
        let db = pooled_psnr_db(&planes).expect("matched planes");
        assert!(
            db >= FLOOR_DB,
            "rav1d and ffmpeg disagree on the AV1 fixture: {db:.2} dB"
        );

        record_oracle(&log, "rav1ddec", "av1", db, FLOOR_DB);
    }
}

// -- JPEG: mjpegdec against ffmpeg's JPEG decoder ----------------------------

#[cfg(feature = "mjpeg")]
mod mjpeg {
    use super::*;
    use g2g_core::{AsyncElement, Caps, Dim, Rate, VideoCodec};
    use g2g_plugins::conformance::pooled_psnr_db;
    use g2g_plugins::mjpegdec::MjpegDec;
    use m1001_common::{data_frame, deinterleave_rgb, CaptureSink};

    const RED16: &[u8] = include_bytes!("data/red16.jpg");
    const SIDE: usize = 16;

    /// Both decoders run the same baseline JPEG through their own integer IDCT and
    /// YCbCr-to-RGB matrix, so they agree to within rounding rather than exactly.
    /// Observed 49.89 dB against ffmpeg n8.1.
    const FLOOR_DB: f64 = 45.0;

    #[tokio::test]
    async fn mjpegdec_matches_ffmpegs_decode_of_the_jpeg_fixture() {
        if !have_ffmpeg() {
            eprintln!("ffmpeg not present; skipping the JPEG decode oracle");
            return;
        }
        let log = EvidenceLog::scoped(LOG);

        let mut decoder = MjpegDec::new();
        decoder
            .configure_pipeline(&Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Fixed(30 << 16),
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            })
            .expect("mjpegdec configures");
        let mut sink = CaptureSink::default();
        decoder
            .process(data_frame(RED16.to_vec(), 0, 0), &mut sink)
            .await
            .expect("decode the JPEG");
        assert_eq!(sink.frames.len(), 1);

        // The fixture lives under tests/data, not tests/fixtures, so ffmpeg is
        // pointed at it directly rather than through the fixture helper.
        let out = std::env::temp_dir().join(format!("g2g_m1001_oracle_{}.rgb", std::process::id()));
        let jpeg = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/red16.jpg");
        let status = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&jpeg)
            .args(["-pix_fmt", "rgb24", "-f", "rawvideo", "-y"])
            .arg(&out)
            .status()
            .expect("run ffmpeg");
        assert!(status.success(), "ffmpeg decoded the JPEG");
        let reference = std::fs::read(&out).expect("ffmpeg wrote RGB");
        let _ = std::fs::remove_file(&out);
        assert_eq!(reference.len(), SIDE * SIDE * 3);

        let ours = deinterleave_rgb(&sink.frames[0]);
        let mut theirs = [Vec::new(), Vec::new(), Vec::new()];
        for pixel in reference.as_chunks::<3>().0 {
            for (channel, value) in theirs.iter_mut().zip(pixel) {
                channel.push(*value);
            }
        }
        let planes: Vec<(&[u8], &[u8])> = (0..3)
            .map(|c| (theirs[c].as_slice(), ours[c].as_slice()))
            .collect();
        let db = pooled_psnr_db(&planes).expect("matched channels");
        assert!(
            db >= FLOOR_DB,
            "mjpegdec and ffmpeg disagree on red16.jpg: {db:.2} dB"
        );

        record_oracle(&log, "mjpegdec", "mjpeg", db, FLOOR_DB);
    }
}
