//! M1137: the AV1 parser and decoders and the JPEG decoder tag their output with
//! how the samples map to colour, instead of reporting `Colorimetry::UNKNOWN` on
//! a tagged stream.
//!
//! `Av1Parse` reads the sequence header's `color_config`, so the oracles are a
//! stream ffmpeg encoded with a known colour request and a round trip through
//! `Av1Enc` (M1136 checked what that writes against ffprobe). For JPEG there is
//! nothing in the file to read: the claim is that the decoder says what JFIF
//! defines and that its pixels match the tag it declares.
#![cfg(any(
    feature = "av1-encode",
    feature = "rav1d",
    feature = "dav1d",
    feature = "mjpeg"
))]

mod common {
    use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::{Caps, Colorimetry, G2gError, OutputSink, PushOutcome};

    pub(crate) const WIDTH: u32 = 320;
    pub(crate) const HEIGHT: u32 = 240;

    pub(crate) fn system_frame(pixels: Vec<u8>, seq: u64) -> Frame {
        Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
            FrameTiming::default(),
            seq,
        )
    }

    #[derive(Default)]
    pub(crate) struct CaptureSink {
        pub(crate) caps: Vec<Caps>,
        pub(crate) frames: Vec<Vec<u8>>,
    }

    impl CaptureSink {
        /// The colorimetry of the last caps announced, or `None` if none were.
        pub(crate) fn announced_colorimetry(&self) -> Option<Colorimetry> {
            self.caps.last().map(|caps| match caps {
                Caps::RawVideo { colorimetry, .. } | Caps::CompressedVideo { colorimetry, .. } => {
                    *colorimetry
                }
                other => panic!("unexpected caps {other:?}"),
            })
        }
    }

    impl OutputSink for CaptureSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            match packet_slot.take().expect("poll_push without a packet") {
                PipelinePacket::CapsChanged(caps) => self.caps.push(caps),
                PipelinePacket::DataFrame(frame) => {
                    if let Some(slice) = frame.domain.as_system_slice() {
                        self.frames.push(slice.to_vec());
                    }
                }
                _ => {}
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }
}

/// `Av1Parse` recovers the colour description from the sequence header, and the
/// AV1 decoders carry it onto their raw output.
#[cfg(any(feature = "av1-encode", feature = "rav1d", feature = "dav1d"))]
mod av1 {
    use super::common::*;

    use g2g_core::frame::PipelinePacket;
    use g2g_core::{AsyncElement, Caps, Colorimetry, Dim, Rate, VideoCodec};
    use g2g_plugins::av1parse::Av1Parse;

    pub(crate) fn av1_caps(colorimetry: Colorimetry) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry,
        }
    }

    /// Run one temporal unit through `Av1Parse` and report the colorimetry it
    /// refined the caps to.
    pub(crate) async fn parse_colorimetry(unit: &[u8]) -> Option<Colorimetry> {
        let mut parser = Av1Parse::new();
        parser
            .configure_pipeline(&av1_caps(Colorimetry::UNKNOWN))
            .expect("configure Av1Parse");
        let mut sink = CaptureSink::default();
        parser
            .process(
                PipelinePacket::DataFrame(system_frame(unit.to_vec(), 0)),
                &mut sink,
            )
            .await
            .expect("parse the temporal unit");
        sink.announced_colorimetry()
    }

    /// An AV1 stream ffmpeg encoded with a known colour request, or `None` where
    /// ffmpeg is not on PATH. ffmpeg's AV1 encoders write the matrix and range
    /// but leave the primaries and transfer unspecified (M1136 saw the same
    /// through ffprobe), so only the matrix is asserted against it.
    pub(crate) fn ffmpeg_av1(colour_args: &[&str]) -> Option<Vec<u8>> {
        use std::process::Command;
        use std::sync::atomic::{AtomicU64, Ordering};

        static FILE_NO: AtomicU64 = AtomicU64::new(0);
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return None;
        }
        let path = std::env::temp_dir().join(format!(
            "g2g-m1137-{}-{}.obu",
            std::process::id(),
            FILE_NO.fetch_add(1, Ordering::Relaxed)
        ));
        let status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=30"),
                "-frames:v",
                "1",
                "-c:v",
                "libaom-av1",
                "-cpu-used",
                "8",
                "-pix_fmt",
                "yuv420p",
            ])
            .args(colour_args)
            .args(["-f", "obu"])
            .arg(&path)
            .status()
            .expect("run ffmpeg");
        assert!(status.success(), "ffmpeg authored the AV1 reference");
        let bytes = std::fs::read(&path).expect("read the reference stream");
        let _ = std::fs::remove_file(&path);
        Some(bytes)
    }

    /// The matrix ffmpeg was asked for comes back out of its own stream, and an
    /// untagged stream comes back untagged: the parser reads the header rather
    /// than reporting a constant.
    #[tokio::test]
    async fn ffmpeg_authored_color_config_parses_back() {
        let Some(tagged) = ffmpeg_av1(&["-colorspace", "bt709"]) else {
            eprintln!("skipping: ffmpeg not on PATH");
            return;
        };
        let untagged = ffmpeg_av1(&[]).expect("ffmpeg is on PATH");
        let parsed_tagged = parse_colorimetry(&tagged).await.expect("refined caps");
        let parsed_untagged = parse_colorimetry(&untagged).await.expect("refined caps");
        assert_ne!(
            parsed_tagged, parsed_untagged,
            "a tagged stream must parse differently from an untagged one"
        );
        assert_eq!(
            parsed_tagged.matrix,
            g2g_core::MatrixCoefficients::Bt709,
            "the matrix ffmpeg was asked for must come back out of the header"
        );
        assert_eq!(
            parsed_untagged.matrix,
            g2g_core::MatrixCoefficients::Unknown,
            "an untagged header must not invent a matrix"
        );
    }

    /// The full description survives encode and parse: what `Av1Enc` writes into
    /// `color_config` is what `Av1Parse` reads back.
    #[cfg(feature = "av1-encode")]
    #[tokio::test]
    async fn av1enc_color_config_round_trips_through_av1parse() {
        // AV1 always codes the range bit, so an untagged encode reads back as the
        // limited range it wrote, never as a fully unknown description.
        let untagged = Colorimetry {
            range: g2g_core::ColorRange::Limited,
            ..Colorimetry::UNKNOWN
        };
        for (encoded_as, expected) in [
            (Colorimetry::BT709, Colorimetry::BT709),
            (Colorimetry::BT2020, Colorimetry::BT2020),
            (Colorimetry::UNKNOWN, untagged),
        ] {
            let encoded = super::encode_av1(encoded_as).await;
            let parsed = parse_colorimetry(&encoded[0])
                .await
                .expect("Av1Parse refined the caps");
            assert_eq!(
                parsed, expected,
                "{encoded_as:?} must survive the sequence header"
            );
        }
    }

    /// And the decoder carries it onto the raw frames, whether it arrives on the
    /// negotiated caps or only in the bitstream.
    #[cfg(all(feature = "av1-encode", feature = "rav1d"))]
    #[tokio::test]
    async fn the_decoder_tags_its_raw_output() {
        use g2g_plugins::rav1ddec::Rav1dDec;

        for negotiated in [Colorimetry::BT2020, Colorimetry::UNKNOWN] {
            let encoded = super::encode_av1(Colorimetry::BT2020).await;
            let mut decoder = Rav1dDec::new();
            decoder
                .configure_pipeline(&av1_caps(negotiated))
                .expect("configure Rav1dDec");
            let mut sink = CaptureSink::default();
            for (seq, unit) in encoded.iter().enumerate() {
                decoder
                    .process(
                        PipelinePacket::DataFrame(system_frame(unit.clone(), seq as u64)),
                        &mut sink,
                    )
                    .await
                    .expect("decode the temporal unit");
            }
            assert_eq!(
                sink.announced_colorimetry(),
                Some(Colorimetry::BT2020),
                "the decoder announces the stream's colour (negotiated {negotiated:?})"
            );
        }
    }
}

/// Encode the test pattern as AV1 under `colorimetry`, one temporal unit per
/// emitted frame.
#[cfg(feature = "av1-encode")]
async fn encode_av1(colorimetry: g2g_core::Colorimetry) -> Vec<Vec<u8>> {
    use common::*;
    use g2g_core::frame::PipelinePacket;
    use g2g_core::{AsyncElement, Caps, Dim, Interlace, Rate, RawVideoFormat};
    use g2g_plugins::av1enc::Av1Enc;

    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut encoder = Av1Enc::new().with_speed(10);
    encoder
        .configure_pipeline(&Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(30 << 16),
            interlace: Interlace::Any,
            colorimetry,
        })
        .expect("configure Av1Enc");
    let mut sink = CaptureSink::default();
    for seq in 0..4u64 {
        let mut pixels = Vec::with_capacity(w * h + 2 * cw * ch);
        for y in 0..h {
            for x in 0..w {
                pixels.push(((x + y + seq as usize * 7) & 0xff) as u8);
            }
        }
        pixels.extend(std::iter::repeat_n(110u8, cw * ch));
        pixels.extend(std::iter::repeat_n(150u8, cw * ch));
        encoder
            .process(
                PipelinePacket::DataFrame(system_frame(pixels, seq)),
                &mut sink,
            )
            .await
            .expect("encode frame");
    }
    encoder
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("flush Av1Enc");
    sink.frames
}

/// A JPEG carries no colour signalling, so `MjpegDec` declares what JFIF defines
/// and its pixels match that declaration.
#[cfg(feature = "mjpeg")]
mod mjpeg {
    use super::common::*;

    use g2g_core::frame::PipelinePacket;
    use g2g_core::{
        AsyncElement, Caps, ColorRange, Colorimetry, Dim, Rate, RawVideoFormat, VideoCodec,
    };
    use g2g_plugins::mjpegdec::MjpegDec;

    /// The JFIF colour at the limited range the planar route scales to.
    const I420_COLORIMETRY: Colorimetry = Colorimetry {
        range: ColorRange::Limited,
        ..Colorimetry::JPEG
    };

    /// A baseline JPEG of a colour ramp, authored by ffmpeg so the input is not
    /// one of our own encoders. `None` where ffmpeg is not on PATH.
    fn ffmpeg_jpeg() -> Option<Vec<u8>> {
        use std::process::Command;
        use std::sync::atomic::{AtomicU64, Ordering};

        // Tests run concurrently and each removes its file, so every call needs
        // its own path.
        static FILE_NO: AtomicU64 = AtomicU64::new(0);
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return None;
        }
        let path = std::env::temp_dir().join(format!(
            "g2g-m1137-{}-{}.jpg",
            std::process::id(),
            FILE_NO.fetch_add(1, Ordering::Relaxed)
        ));
        let status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc2=size={WIDTH}x{HEIGHT}"),
                "-frames:v",
                "1",
                "-pix_fmt",
                "yuvj420p",
            ])
            .arg(&path)
            .status()
            .expect("run ffmpeg");
        assert!(status.success(), "ffmpeg authored the JPEG");
        let bytes = std::fs::read(&path).expect("read the JPEG");
        let _ = std::fs::remove_file(&path);
        Some(bytes)
    }

    async fn decode(jpeg: &[u8], out_format: RawVideoFormat) -> CaptureSink {
        let mut decoder = MjpegDec::new().with_output_format(out_format);
        decoder
            .configure_pipeline(&Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                colorimetry: Colorimetry::UNKNOWN,
            })
            .expect("configure MjpegDec");
        let mut sink = CaptureSink::default();
        decoder
            .process(
                PipelinePacket::DataFrame(system_frame(jpeg.to_vec(), 0)),
                &mut sink,
            )
            .await
            .expect("decode the JPEG");
        sink
    }

    #[tokio::test]
    async fn decoded_output_is_tagged_per_jfif() {
        let Some(jpeg) = ffmpeg_jpeg() else {
            eprintln!("skipping: ffmpeg not on PATH");
            return;
        };
        assert_eq!(
            decode(&jpeg, RawVideoFormat::Rgba8)
                .await
                .announced_colorimetry(),
            Some(Colorimetry::SRGB),
            "the RGBA route ends in sRGB"
        );
        assert_eq!(
            decode(&jpeg, RawVideoFormat::I420)
                .await
                .announced_colorimetry(),
            Some(I420_COLORIMETRY),
            "the planar route declares the range it scaled to"
        );
    }

    /// The tag describes the samples, not just the file: the planar output is
    /// closer to its own RGBA converted under the declared limited range than
    /// under the full range JFIF itself uses.
    #[tokio::test]
    async fn the_planar_pixels_match_the_declared_range() {
        let Some(jpeg) = ffmpeg_jpeg() else {
            eprintln!("skipping: ffmpeg not on PATH");
            return;
        };
        let rgba = decode(&jpeg, RawVideoFormat::Rgba8).await.frames.remove(0);
        let planar = decode(&jpeg, RawVideoFormat::I420).await.frames.remove(0);
        let error_against = |colorimetry| {
            let expected = g2g_plugins::videoconvert::convert(
                &rgba,
                RawVideoFormat::Rgba8,
                RawVideoFormat::I420,
                WIDTH as usize,
                HEIGHT as usize,
                colorimetry,
            );
            let total: u64 = planar
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| a.abs_diff(*b) as u64)
                .sum();
            total as f64 / planar.len() as f64
        };
        let declared = error_against(I420_COLORIMETRY);
        let full = error_against(Colorimetry::JPEG);
        assert!(
            declared < full,
            "the declared range must describe the samples ({declared}) better than \
             JFIF's own full range ({full})"
        );
    }
}
