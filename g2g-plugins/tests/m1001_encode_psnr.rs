//! M1001 encode / decode PSNR: encode a deterministic synthetic source with an
//! in-repo encoder, decode it with the matching in-repo decoder, and require the
//! recovered pixels to stay above a per-codec PSNR floor.
//!
//! This is the check a structural round-trip cannot make: an encoder that silently
//! loses a quality setting, or a decoder that drops a post-filter, still round-trips
//! frames of the right size. The source is generated in-test (a gradient with a
//! checkerboard and a walking bar, so there is both smooth and hard-edged content:
//! a flat image would score arbitrarily high) and never committed, so the batteries
//! carry no fixture. Each floor is set a few dB under the figure observed when the
//! battery was written, recorded next to it, so normal encoder drift does not fail
//! the build but a real quality regression does.
//!
//! A pass persists `Quality` evidence for the encoder and the decoder both, since
//! the measurement covers the pair.
#![cfg(feature = "std")]
#![allow(dead_code)] // the helpers below serve whichever codec features are on

mod m1001_common;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_plugins::conformance::{i420_planes, pooled_psnr_db, psnr_db};
use m1001_common::EvidenceLog;

/// The battery's own log name, so a local run does not inherit another's rows.
const LOG: &str = "m1001-psnr";

/// Frame geometry the batteries encode. Both dimensions are multiples of 16 so the
/// checkerboard lands on macroblock / superblock boundaries and no codec pads.
const WIDTH: usize = 160;
const HEIGHT: usize = 128;
const FRAMES: usize = 8;

/// How far under a codec's floor the worst single frame may sit. Inter-coded frames
/// legitimately score below the average; a larger gap means one frame broke.
const FRAME_MARGIN_DB: f64 = 4.0;

/// Record a measured PSNR as `Quality` evidence for every element in the pair, and
/// echo it so a `--nocapture` run shows how much headroom the floor has.
fn record_psnr(log: &EvidenceLog, elements: &[&str], codec: &str, measured: Psnr, floor: f64) {
    let Psnr { pooled, worst } = measured;
    eprintln!("{codec}: {pooled:.2} dB pooled, {worst:.2} dB worst (floor {floor:.0} dB)");
    for element in elements {
        log.record(
            element,
            &Evidence::new(ConformanceDimension::Quality)
                .codec(codec)
                .detail(format!(
                    "encode/decode of a synthetic {WIDTH}x{HEIGHT} source: {pooled:.1} dB PSNR pooled, \
                     {worst:.1} dB worst (floor {floor:.0} dB)"
                )),
        );
    }
}

/// A sequence's PSNR: pooled over every plane of every frame, and the worst single
/// plane (an average would hide one badly coded frame).
#[derive(Debug, Clone, Copy)]
struct Psnr {
    pooled: f64,
    worst: f64,
}

impl Psnr {
    /// Whether both figures clear `floor`, the worst allowed [`FRAME_MARGIN_DB`] under.
    fn clears(self, floor: f64) -> bool {
        self.pooled >= floor && self.worst >= floor - FRAME_MARGIN_DB
    }
}

/// The PSNR of a decoded I420 sequence against the source that produced it.
fn i420_psnr(source: &[Vec<u8>], decoded: &[Vec<u8>]) -> Psnr {
    let pairs = source.len().min(decoded.len());
    assert!(pairs > 0, "nothing decoded to measure");
    let mut planes = Vec::new();
    for i in 0..pairs {
        let reference = i420_planes(&source[i], WIDTH, HEIGHT).expect("source is I420");
        let measured = i420_planes(&decoded[i], WIDTH, HEIGHT).expect("decoded frame is I420");
        for plane in 0..3 {
            planes.push((reference[plane], measured[plane]));
        }
    }
    plane_psnr(&planes)
}

/// The pooled and worst-plane PSNR of a set of reference / measured plane pairs.
fn plane_psnr(planes: &[(&[u8], &[u8])]) -> Psnr {
    let worst = planes
        .iter()
        .map(|(reference, measured)| psnr_db(reference, measured).expect("matched plane"))
        .fold(f64::INFINITY, f64::min);
    Psnr {
        pooled: pooled_psnr_db(planes).expect("matched planes"),
        worst,
    }
}

/// The synthetic source, one frame per phase.
fn source_frames() -> Vec<Vec<u8>> {
    (0..FRAMES)
        .map(|phase| m1001_common::synthetic_i420(WIDTH, HEIGHT, phase))
        .collect()
}

fn i420_caps() -> g2g_core::Caps {
    g2g_core::Caps::RawVideo {
        format: g2g_core::RawVideoFormat::I420,
        width: g2g_core::Dim::Fixed(WIDTH as u32),
        height: g2g_core::Dim::Fixed(HEIGHT as u32),
        framerate: g2g_core::Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

// -- AV1 (rav1e encode, rav1d decode) ----------------------------------------

#[cfg(all(feature = "av1-encode", feature = "rav1d"))]
mod av1 {
    use super::*;
    use g2g_core::{AsyncElement, PipelinePacket};
    use g2g_plugins::av1enc::Av1Enc;
    use g2g_plugins::rav1ddec::Rav1dDec;
    use m1001_common::{data_frame, CaptureSink};

    /// Observed 47.29 dB pooled, 45.20 dB worst at speed 10, quantizer 80. The floor
    /// sits a few dB under: rav1e's rate control moves between releases, but a
    /// dropped quality setting costs far more than that.
    const FLOOR_DB: f64 = 42.0;

    #[tokio::test]
    async fn av1_encode_decode_holds_its_psnr_floor() {
        let log = EvidenceLog::scoped(LOG);
        let source = source_frames();

        let mut encoder = Av1Enc::new().with_speed(10).with_quantizer(80);
        encoder
            .configure_pipeline(&i420_caps())
            .expect("rav1e configures for I420");
        let mut encoded = CaptureSink::default();
        for (i, frame) in source.iter().enumerate() {
            encoder
                .process(
                    data_frame(frame.clone(), i as u64 * 33_000_000, i as u64),
                    &mut encoded,
                )
                .await
                .expect("encode a frame");
        }
        encoder
            .process(PipelinePacket::Eos, &mut encoded)
            .await
            .expect("flush the encoder");
        assert!(!encoded.frames.is_empty(), "the encoder produced AV1");

        let mut decoder = Rav1dDec::new();
        decoder
            .configure_pipeline(&encoded.caps[0])
            .expect("rav1d configures from the encoder's caps");
        let mut decoded = CaptureSink::default();
        for (i, unit) in encoded.frames.iter().enumerate() {
            decoder
                .process(
                    data_frame(unit.clone(), i as u64 * 33_000_000, i as u64),
                    &mut decoded,
                )
                .await
                .expect("decode a temporal unit");
        }
        decoder
            .process(PipelinePacket::Eos, &mut decoded)
            .await
            .expect("drain");
        assert_eq!(encoded.frames.len(), source.len(), "one packet per frame");
        // The `Eos` drain (M1003) returns the reordering tail, so the count is
        // exact.
        assert_eq!(
            decoded.frames.len(),
            source.len(),
            "every encoded frame decodes back"
        );

        let psnr = i420_psnr(&source, &decoded.frames);
        assert!(
            psnr.clears(FLOOR_DB),
            "AV1 encode/decode PSNR fell: {psnr:?}"
        );
        record_psnr(&log, &["av1enc", "rav1ddec"], "av1", psnr, FLOOR_DB);
    }
}

// -- VP8 / VP9 (libvpx encode, libavcodec decode) ----------------------------

#[cfg(all(feature = "vpx", feature = "ffmpeg"))]
mod vpx {
    use super::*;
    use g2g_core::{AsyncElement, PipelinePacket, VideoCodec};
    use g2g_plugins::ffmpegdec::{FfmpegVideoDec, OutputFormat};
    use g2g_plugins::vpxenc::VpxEnc;
    use m1001_common::{data_frame, CaptureSink};

    /// Same figure as the H.264 battery, for the same reason: high enough to code
    /// this geometry cleanly, low enough that the measurement still tracks the
    /// bitrate setting instead of pinning at the encoder's quality ceiling.
    const BITRATE_KBPS: u32 = 500;

    /// Observed 50.50 dB pooled, 47.27 dB worst (libvpx 1.15). Dropping the target
    /// to 100 kbps costs VP8 8 dB, so the figure still tracks the rate setting.
    const VP8_FLOOR_DB: f64 = 45.0;

    /// Observed 47.46 dB pooled, 45.58 dB worst (libvpx 1.15). VP9 scores under VP8
    /// here because the wrapper drives it at cpu-used 6 with the realtime deadline.
    const VP9_FLOOR_DB: f64 = 42.0;

    /// Encode the synthetic source with libvpx and decode it back with libavcodec:
    /// there is no in-repo VP8 / VP9 decoder by design, so the pair is asymmetric.
    async fn vpx_psnr(codec: VideoCodec) -> Psnr {
        let source = source_frames();

        let mut encoder = VpxEnc::new()
            .with_codec(codec)
            .with_bitrate_kbps(BITRATE_KBPS);
        encoder
            .configure_pipeline(&i420_caps())
            .expect("libvpx configures for I420");
        let mut encoded = CaptureSink::default();
        for (i, frame) in source.iter().enumerate() {
            encoder
                .process(
                    data_frame(frame.clone(), i as u64 * 33_000_000, i as u64),
                    &mut encoded,
                )
                .await
                .expect("encode a frame");
        }
        encoder
            .process(PipelinePacket::Eos, &mut encoded)
            .await
            .expect("flush the encoder");
        assert_eq!(encoded.frames.len(), source.len(), "one packet per frame");

        let mut decoder = FfmpegVideoDec::new().with_output_format(OutputFormat::I420);
        let narrowed = decoder
            .intercept_caps(&encoded.caps[0])
            .expect("vp8 / vp9 supported");
        decoder
            .configure_pipeline(&narrowed)
            .expect("libavcodec opens the decoder");
        let mut decoded = CaptureSink::default();
        for (i, unit) in encoded.frames.iter().enumerate() {
            decoder
                .process(
                    data_frame(unit.clone(), i as u64 * 33_000_000, i as u64),
                    &mut decoded,
                )
                .await
                .expect("decode a frame");
        }
        decoder
            .process(PipelinePacket::Eos, &mut decoded)
            .await
            .expect("drain");
        assert_eq!(
            decoded.frames.len(),
            source.len(),
            "every encoded frame decoded back"
        );

        i420_psnr(&source, &decoded.frames)
    }

    #[tokio::test]
    async fn vp8_encode_decode_holds_its_psnr_floor() {
        let log = EvidenceLog::scoped(LOG);
        let psnr = vpx_psnr(VideoCodec::Vp8).await;
        assert!(
            psnr.clears(VP8_FLOOR_DB),
            "VP8 encode/decode PSNR fell: {psnr:?}"
        );
        record_psnr(&log, &["vpxenc", "ffmpegdec"], "vp8", psnr, VP8_FLOOR_DB);
    }

    #[tokio::test]
    async fn vp9_encode_decode_holds_its_psnr_floor() {
        let log = EvidenceLog::scoped(LOG);
        let psnr = vpx_psnr(VideoCodec::Vp9).await;
        assert!(
            psnr.clears(VP9_FLOOR_DB),
            "VP9 encode/decode PSNR fell: {psnr:?}"
        );
        record_psnr(&log, &["vpxenc", "ffmpegdec"], "vp9", psnr, VP9_FLOOR_DB);
    }
}

// -- Motion JPEG -------------------------------------------------------------

#[cfg(all(feature = "mjpeg", feature = "mjpeg-encode"))]
mod mjpeg {
    use super::*;
    use g2g_core::{AsyncElement, Caps, Dim, Interlace, Rate, RawVideoFormat};
    use g2g_plugins::mjpegdec::MjpegDec;
    use g2g_plugins::mjpegenc::MjpegEnc;
    use m1001_common::{data_frame, deinterleave_rgb, synthetic_rgba8, CaptureSink};

    /// Measured in packed RGBA, both elements' native format. Through I420 the pair
    /// converts RGBA <-> YCbCr twice and the score is dominated by that conversion,
    /// not by JPEG, so an I420 floor would not track encode quality.
    ///
    /// Observed 50.64 dB pooled, 49.11 dB worst at quality 95. JPEG is intra-only
    /// with quantization tables fixed by the quality setting, so the figure is stable
    /// and the floor allows for a library-side rounding change only.
    const FLOOR_DB: f64 = 46.0;

    fn rgba_caps() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(WIDTH as u32),
            height: Dim::Fixed(HEIGHT as u32),
            framerate: Rate::Fixed(30 << 16),
            interlace: Interlace::Any,
        }
    }

    #[tokio::test]
    async fn mjpeg_encode_decode_holds_its_psnr_floor() {
        let log = EvidenceLog::scoped(LOG);
        let source: Vec<Vec<u8>> = (0..FRAMES)
            .map(|phase| synthetic_rgba8(WIDTH, HEIGHT, phase))
            .collect();

        let mut encoder = MjpegEnc::new().with_quality(95);
        encoder
            .configure_pipeline(&rgba_caps())
            .expect("mjpegenc configures for RGBA");
        let mut encoded = CaptureSink::default();
        for (i, frame) in source.iter().enumerate() {
            encoder
                .process(
                    data_frame(frame.clone(), i as u64 * 33_000_000, i as u64),
                    &mut encoded,
                )
                .await
                .expect("encode a frame");
        }
        assert_eq!(encoded.frames.len(), source.len(), "JPEG is intra-only");

        let mut decoder = MjpegDec::new();
        decoder
            .configure_pipeline(&encoded.caps[0])
            .expect("mjpegdec configures from the encoder's caps");
        let mut decoded = CaptureSink::default();
        for (i, jpeg) in encoded.frames.iter().enumerate() {
            decoder
                .process(
                    data_frame(jpeg.clone(), i as u64 * 33_000_000, i as u64),
                    &mut decoded,
                )
                .await
                .expect("decode a JPEG");
        }
        assert_eq!(decoded.frames.len(), source.len());

        // Split every frame into R, G, B first: `plane_psnr` borrows the planes, so
        // they have to outlive the pair list it is handed.
        let mut channels: Vec<[Vec<u8>; 3]> = Vec::new();
        for (reference, measured) in source.iter().zip(&decoded.frames) {
            channels.push(deinterleave_rgb(reference));
            channels.push(deinterleave_rgb(measured));
        }
        let planes: Vec<(&[u8], &[u8])> = channels
            .chunks_exact(2)
            .flat_map(|pair| (0..3).map(|c| (pair[0][c].as_slice(), pair[1][c].as_slice())))
            .collect();
        let psnr = plane_psnr(&planes);
        assert!(
            psnr.clears(FLOOR_DB),
            "MJPEG encode/decode PSNR fell: {psnr:?}"
        );
        record_psnr(&log, &["mjpegenc", "mjpegdec"], "mjpeg", psnr, FLOOR_DB);
    }
}

// -- H.264 (libavcodec software encode and decode) ---------------------------

#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
mod h264 {
    use super::*;
    use g2g_core::{AsyncElement, PipelinePacket};
    use g2g_plugins::ffmpegdec::{FfmpegVideoDec, OutputFormat};
    use g2g_plugins::ffmpegenc::{Backend, FfmpegH264Enc};
    use m1001_common::{data_frame, CaptureSink};

    /// 500 kbit/s at this geometry, which keeps the encoder off its quality ceiling:
    /// at 4 Mbit/s libx264 codes the source at QP 1 and the measurement stops
    /// tracking the bitrate setting at all.
    const BITRATE: usize = 500_000;

    /// Observed 53.69 dB pooled, 50.56 dB worst (ffmpeg n8.1). The floor sits well
    /// under that: libx264's rate control moves between versions, and what this
    /// catches is a bitrate or preset that stopped being applied, not drift.
    const FLOOR_DB: f64 = 47.0;

    #[tokio::test]
    async fn ffmpeg_h264_encode_decode_holds_its_psnr_floor() {
        let log = EvidenceLog::scoped(LOG);
        let source = source_frames();

        let mut encoder = FfmpegH264Enc::new()
            .with_backend(Backend::Software)
            .with_bitrate(BITRATE);
        encoder
            .configure_pipeline(&i420_caps())
            .expect("libavcodec opens the H.264 encoder");
        let mut encoded = CaptureSink::default();
        for (i, frame) in source.iter().enumerate() {
            encoder
                .process(
                    data_frame(frame.clone(), i as u64 * 33_000_000, i as u64),
                    &mut encoded,
                )
                .await
                .expect("encode a frame");
        }
        encoder
            .process(PipelinePacket::Eos, &mut encoded)
            .await
            .expect("flush the encoder");
        assert!(!encoded.frames.is_empty(), "the encoder produced H.264");

        let mut decoder = FfmpegVideoDec::new().with_output_format(OutputFormat::I420);
        let narrowed = decoder
            .intercept_caps(&encoded.caps[0])
            .expect("h264 supported");
        decoder
            .configure_pipeline(&narrowed)
            .expect("libavcodec opens the H.264 decoder");
        let mut decoded = CaptureSink::default();
        for (i, au) in encoded.frames.iter().enumerate() {
            decoder
                .process(
                    data_frame(au.clone(), i as u64 * 33_000_000, i as u64),
                    &mut decoded,
                )
                .await
                .expect("decode an access unit");
        }
        decoder
            .process(PipelinePacket::Eos, &mut decoded)
            .await
            .expect("drain");
        assert_eq!(
            decoded.frames.len(),
            source.len(),
            "every encoded frame decoded back"
        );

        let psnr = i420_psnr(&source, &decoded.frames);
        assert!(
            psnr.clears(FLOOR_DB),
            "H.264 encode/decode PSNR fell: {psnr:?}"
        );
        record_psnr(&log, &["ffmpegenc", "ffmpegdec"], "h264", psnr, FLOOR_DB);
    }
}
