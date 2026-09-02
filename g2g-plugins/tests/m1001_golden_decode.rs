//! M1001 decoder regression goldens: decode a committed fixture and check the raw
//! output against a digest committed here.
//!
//! A round-trip test proves data survived an element; it does not notice a decoder
//! quietly producing *different* pixels after a dependency bump or a refactor. These
//! batteries pin the output: each fixture is decoded by the in-repo decoder and the
//! concatenated raw frames are hashed (FNV-1a 64, `conformance::fnv1a_64`) against a
//! value recorded when the decode was verified correct. Every pass persists `Quality`
//! evidence for the decoder, so `g2g-inspect --maturity` shows which decoders have a
//! measured-output check and which only round-trip.
//!
//! The goldens are per-decoder, not per-codec: they are checked against the pure-Rust
//! / all-platform decoders whose output is bit-exact by the codec spec (AV1, JPEG,
//! Opus, Vorbis, H.264), so a mismatch means g2g changed, not the reference. The
//! ffmpeg-gated AAC leg is deliberately a determinism check instead: libavcodec
//! decodes AAC in float and is not bit-exact across versions, so a committed digest
//! there would fail on an unrelated ffmpeg upgrade.
#![cfg(feature = "std")]
#![allow(dead_code)] // the helpers below serve whichever codec features are on

mod m1001_common;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_plugins::conformance::fnv1a_64;
use m1001_common::EvidenceLog;

/// The battery's own log name, so a local run does not inherit another's rows.
const LOG: &str = "m1001-golden";

/// Check `raw` against its committed digest and, on a pass, record the `Quality`
/// evidence for `element`. `detail` names the fixture and the decode it covers.
fn assert_golden(
    log: &EvidenceLog,
    element: &str,
    codec: &str,
    detail: &str,
    raw: &[u8],
    golden: u64,
) {
    let digest = fnv1a_64(raw);
    assert_eq!(
        digest, golden,
        "{element} output changed (got {digest:#018x}): {detail}"
    );
    log.record(
        element,
        &Evidence::new(ConformanceDimension::Quality)
            .codec(codec)
            .detail(detail),
    );
}

// -- AV1 (pure-Rust rav1d) ---------------------------------------------------

#[cfg(feature = "rav1d")]
mod av1 {
    use super::*;
    use g2g_core::{AsyncElement, Caps, Dim, PipelinePacket, Rate, RawVideoFormat, VideoCodec};
    use g2g_plugins::rav1ddec::Rav1dDec;
    use m1001_common::{data_frame, split_temporal_units, CaptureSink};

    const CLIP: &[u8] = include_bytes!("fixtures/av1_640x480.obu");
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;

    /// FNV-1a 64 of every decoded I420 frame of `av1_640x480.obu` concatenated.
    /// AV1 decode is bit-exact by the specification, so this value is a property of
    /// the fixture, not of the rav1d version: a change means g2g's decode changed.
    const GOLDEN: u64 = 0xfdd5_aadc_eac4_b3ad;

    #[tokio::test]
    async fn rav1d_decode_of_the_av1_fixture_matches_its_golden() {
        let log = EvidenceLog::scoped(LOG);
        let mut decoder = Rav1dDec::new();
        decoder
            .configure_pipeline(&Caps::CompressedVideo {
                codec: VideoCodec::Av1,
                width: Dim::Fixed(WIDTH),
                height: Dim::Fixed(HEIGHT),
                framerate: Rate::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            })
            .expect("rav1d configures for the fixture geometry");

        let mut sink = CaptureSink::default();
        let units = split_temporal_units(CLIP);
        assert!(units.len() > 1, "fixture carries several temporal units");
        for (i, unit) in units.iter().enumerate() {
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

        let expected_frame = (WIDTH * HEIGHT * 3 / 2) as usize;
        assert!(!sink.frames.is_empty(), "the fixture decoded to frames");
        assert!(
            sink.frames.iter().all(|f| f.len() == expected_frame),
            "every frame is 640x480 I420"
        );
        assert!(
            sink.caps.iter().any(|c| matches!(
                c,
                Caps::RawVideo {
                    format: RawVideoFormat::I420,
                    ..
                }
            )),
            "the decoder announced I420: {:?}",
            sink.caps
        );
        assert_golden(
            &log,
            "rav1ddec",
            "av1",
            "golden digest of av1_640x480.obu decoded to I420",
            &sink.concatenated(),
            GOLDEN,
        );
    }
}

// -- Motion JPEG -------------------------------------------------------------

#[cfg(feature = "mjpeg")]
mod mjpeg {
    use super::*;
    use g2g_core::{AsyncElement, Caps, Dim, Rate, RawVideoFormat, VideoCodec};
    use g2g_plugins::mjpegdec::MjpegDec;
    use m1001_common::{data_frame, CaptureSink};

    const RED16: &[u8] = include_bytes!("data/red16.jpg");
    const GRAY16: &[u8] = include_bytes!("data/gray16.jpg");
    const SIDE: usize = 16;

    /// FNV-1a 64 of the two 16x16 fixtures decoded to I420 back to back. JPEG's
    /// integer IDCT is implementation-defined at the LSB, so this pins zune-jpeg's
    /// output specifically, which is the point: a silent backend swap must fail.
    const GOLDEN: u64 = 0x0b8a_8c68_42ec_de48;

    #[tokio::test]
    async fn mjpeg_decode_of_the_committed_jpegs_matches_its_golden() {
        let log = EvidenceLog::scoped(LOG);
        let mut decoder = MjpegDec::new().with_output_format(RawVideoFormat::I420);
        decoder
            .configure_pipeline(&Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Fixed(30 << 16),
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            })
            .expect("mjpegdec configures with recovered geometry");

        let mut sink = CaptureSink::default();
        for (i, jpeg) in [RED16, GRAY16].into_iter().enumerate() {
            decoder
                .process(
                    data_frame(jpeg.to_vec(), i as u64 * 33_000_000, i as u64),
                    &mut sink,
                )
                .await
                .expect("decode a JPEG");
        }

        assert_eq!(sink.frames.len(), 2, "one frame per JPEG");
        assert!(
            sink.frames.iter().all(|f| f.len() == SIDE * SIDE * 3 / 2),
            "16x16 I420"
        );
        assert_golden(
            &log,
            "mjpegdec",
            "mjpeg",
            "golden digest of red16.jpg + gray16.jpg decoded to I420",
            &sink.concatenated(),
            GOLDEN,
        );
    }
}

// -- Ogg audio (Opus, Vorbis) ------------------------------------------------

#[cfg(any(feature = "opus", feature = "vorbis"))]
mod ogg_audio {
    use super::*;
    use g2g_core::runtime::{parse_launch, run_graph};
    use g2g_core::PipelineClock;
    use g2g_plugins::registry::default_registry;
    use m1001_common::fixture_path;

    struct ZeroClock;
    impl PipelineClock for ZeroClock {
        fn now_ns(&self) -> u64 {
            0
        }
    }

    /// Decode `fixture` through the auto-plugged chain to interleaved S16LE and
    /// return the PCM. The launch line is also the repro a reader can paste.
    async fn decode_pcm(fixture: &str, channels: u8, rate: u32) -> Vec<u8> {
        let out =
            std::env::temp_dir().join(format!("g2g_m1001_{}_{fixture}.raw", std::process::id()));
        let line = format!(
            "filesrc location={src} ! decodebin ! audioconvert ! \
             audio/x-raw,format=S16LE,rate={rate},channels={channels} ! filesink location={out}",
            src = fixture_path(fixture).display(),
            out = out.display(),
        );
        let registry = default_registry();
        let graph = parse_launch(&registry, &line).expect("pipeline parses");
        run_graph(graph, &ZeroClock, 4)
            .await
            .expect("pipeline runs");
        let pcm = std::fs::read(&out).expect("decoded PCM written");
        let _ = std::fs::remove_file(&out);
        pcm
    }

    /// FNV-1a 64 of `opus_mono_48k.opus` decoded to S16LE mono. Opus decode is
    /// bit-exact, and this is the same value `m750_opus_trim` pins for the
    /// pre-skip / padding trim, so the two batteries cross-check each other.
    #[cfg(feature = "opus")]
    const OPUS_MONO_GOLDEN: u64 = 0xa989_609d_8af3_d090;

    #[cfg(feature = "opus")]
    #[tokio::test]
    async fn opus_decode_of_the_mono_fixture_matches_its_golden() {
        let log = EvidenceLog::scoped(LOG);
        let pcm = decode_pcm("opus_mono_48k.opus", 1, 48_000).await;
        assert_eq!(pcm.len(), 12_000 * 2, "0.25 s of mono at 48 kHz, trimmed");
        assert_golden(
            &log,
            "opusdec",
            "opus",
            "golden digest of opus_mono_48k.opus decoded to S16LE",
            &pcm,
            OPUS_MONO_GOLDEN,
        );
    }

    /// FNV-1a 64 of `vorbis_stereo_48k.ogg` decoded to interleaved S16LE stereo.
    #[cfg(feature = "vorbis")]
    const VORBIS_STEREO_GOLDEN: u64 = 0x775c_081b_f8b2_3868;

    #[cfg(feature = "vorbis")]
    #[tokio::test]
    async fn vorbis_decode_of_the_stereo_fixture_matches_its_golden() {
        let log = EvidenceLog::scoped(LOG);
        let pcm = decode_pcm("vorbis_stereo_48k.ogg", 2, 48_000).await;
        assert_eq!(pcm.len(), 12_000 * 2 * 2, "0.25 s of stereo at 48 kHz");
        assert_golden(
            &log,
            "vorbisdec",
            "vorbis",
            "golden digest of vorbis_stereo_48k.ogg decoded to S16LE",
            &pcm,
            VORBIS_STEREO_GOLDEN,
        );
    }
}

// -- ffmpeg-gated (H.264 golden, AAC determinism) ----------------------------

#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
mod ffmpeg_codecs {
    use super::*;
    use g2g_core::{AsyncElement, AudioFormat, Caps, Dim, PipelinePacket, Rate, VideoCodec};
    use g2g_plugins::ffmpegaudiodec::FfmpegAudioDec;
    use g2g_plugins::ffmpegdec::{FfmpegVideoDec, OutputFormat};
    use g2g_plugins::h264parse::H264Parse;
    use m1001_common::{data_frame, split_adts_frames, CaptureSink};

    const H264: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
    const ADTS: &[u8] = include_bytes!("fixtures/aac_stereo_44100.adts");
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;

    /// FNV-1a 64 of `h264_640x480.h264` decoded to I420. H.264 decode is bit-exact
    /// by the specification, so this holds across libavcodec versions.
    const H264_GOLDEN: u64 = 0x79ad_c7fb_ccd8_3662;

    /// Re-frame the Annex-B stream to one access unit per `DataFrame`, the shape a
    /// decoder needs, and return the access units.
    async fn access_units() -> Vec<Vec<u8>> {
        let mut parser = H264Parse::reframing();
        parser
            .configure_pipeline(&Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            })
            .expect("h264parse configures");
        let mut sink = CaptureSink::default();
        parser
            .process(data_frame(H264.to_vec(), 0, 0), &mut sink)
            .await
            .expect("reframe");
        parser
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("flush the tail access unit");
        sink.frames
    }

    #[tokio::test]
    async fn ffmpeg_h264_decode_of_the_fixture_matches_its_golden() {
        let log = EvidenceLog::scoped(LOG);
        let mut decoder = FfmpegVideoDec::new().with_output_format(OutputFormat::I420);
        let upstream = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let narrowed = decoder.intercept_caps(&upstream).expect("h264 supported");
        decoder
            .configure_pipeline(&narrowed)
            .expect("libavcodec opens the H.264 decoder");

        let mut sink = CaptureSink::default();
        for (i, au) in access_units().await.into_iter().enumerate() {
            decoder
                .process(data_frame(au, i as u64 * 33_000_000, i as u64), &mut sink)
                .await
                .expect("decode an access unit");
        }
        decoder
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("drain");

        assert!(!sink.frames.is_empty(), "the fixture decoded to frames");
        assert!(
            sink.frames
                .iter()
                .all(|f| f.len() == (WIDTH * HEIGHT * 3 / 2) as usize),
            "every frame is 640x480 I420"
        );
        assert_golden(
            &log,
            "ffmpegdec",
            "h264",
            "golden digest of h264_640x480.h264 decoded to I420",
            &sink.concatenated(),
            H264_GOLDEN,
        );
    }

    /// Decode the ADTS fixture once, returning the interleaved S16LE PCM.
    async fn decode_aac() -> Vec<u8> {
        let mut decoder = FfmpegAudioDec::new();
        decoder
            .configure_pipeline(&Caps::Audio {
                format: AudioFormat::Aac,
                channels: 2,
                sample_rate: 44_100,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            })
            .expect("libavcodec opens the AAC decoder");
        let mut sink = CaptureSink::default();
        for (i, au) in split_adts_frames(ADTS).into_iter().enumerate() {
            decoder
                .process(
                    data_frame(au.to_vec(), i as u64 * 23_219_954, i as u64),
                    &mut sink,
                )
                .await
                .expect("decode an ADTS frame");
        }
        decoder
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("drain");
        sink.concatenated()
    }

    #[tokio::test]
    async fn ffmpeg_aac_decode_is_deterministic_and_frame_exact() {
        // No committed digest here: libavcodec decodes AAC in float, so its output
        // moves between versions. What must hold is that a given build decodes the
        // fixture identically every time and yields the full 1024-sample frames.
        let log = EvidenceLog::scoped(LOG);
        let frames = split_adts_frames(ADTS).len();
        assert!(frames > 1, "the fixture carries several ADTS frames");

        let first = decode_aac().await;
        let second = decode_aac().await;
        assert!(!first.is_empty(), "the fixture decoded to PCM");
        assert!(
            first.len().is_multiple_of(1024 * 2 * 2),
            "whole 1024-sample stereo S16 frames, no partial tail"
        );

        // Not a committed golden: the digest compared here is this run's own second
        // decode, so what it asserts is that the two agree.
        assert_golden(
            &log,
            "ffmpegaudiodec",
            "aac",
            "aac_stereo_44100.adts decodes deterministically to whole S16LE frames",
            &first,
            fnv1a_64(&second),
        );
    }
}
