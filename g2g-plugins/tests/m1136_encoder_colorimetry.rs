//! M1136: the native encoders write the caps colorimetry into their bitstream
//! and carry it on their compressed output caps.
//!
//! The oracle is ffprobe reading the encoded stream, the same one M1124 uses for
//! `FfmpegH264Enc`: what ffmpeg reads back is what any other decoder sees. No
//! colour value is spelled against a probe here, every expected reading comes
//! from probing a stream ffmpeg itself encoded with the same colour request, and
//! the tagged and untagged references have to disagree, so a probe that reported
//! nothing could not pass.
//!
//! JPEG is the exception: it has no colour signalling to write, so the claim is
//! that the encoder says so ([`Colorimetry::JPEG`]) and honours the input tag in
//! the conversion instead of mislabeling the result.
#![cfg(any(
    feature = "av1-encode",
    feature = "mjpeg-encode",
    all(target_os = "linux", feature = "nvenc")
))]

mod common {
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};

    use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::{Caps, G2gError, OutputSink, PushOutcome};

    pub(crate) const WIDTH: u32 = 320;
    pub(crate) const HEIGHT: u32 = 240;
    pub(crate) const FPS: u32 = 30;
    pub(crate) const FRAMES: u64 = 8;
    pub(crate) const FRAME_PERIOD_NS: u64 = 1_000_000_000 / FPS as u64;

    /// A moving luma ramp over flat chroma, so successive frames differ.
    #[cfg(any(feature = "av1-encode", feature = "mjpeg-encode"))]
    pub(crate) fn i420_frame(seq: u64) -> Vec<u8> {
        let (w, h) = (WIDTH as usize, HEIGHT as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut v = Vec::with_capacity(w * h + 2 * cw * ch);
        for y in 0..h {
            for x in 0..w {
                v.push(((x + y + seq as usize * 7) & 0xff) as u8);
            }
        }
        v.extend(std::iter::repeat_n(110u8, cw * ch));
        v.extend(std::iter::repeat_n(150u8, cw * ch));
        v
    }

    pub(crate) fn system_frame(pixels: Vec<u8>, seq: u64) -> Frame {
        Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
            FrameTiming {
                pts_ns: seq * FRAME_PERIOD_NS,
                ..FrameTiming::default()
            },
            seq,
        )
    }

    #[derive(Default)]
    pub(crate) struct CaptureSink {
        pub(crate) caps: Vec<Caps>,
        pub(crate) access_units: Vec<Vec<u8>>,
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
                        self.access_units.push(slice.to_vec());
                    }
                }
                _ => {}
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    /// The colour fields ffprobe reports for a stream, as it spells them.
    #[derive(Debug, PartialEq, Eq)]
    pub(crate) struct ProbedColour {
        pub(crate) range: String,
        pub(crate) space: String,
        pub(crate) transfer: String,
        pub(crate) primaries: String,
    }

    /// Tests run concurrently and reuse tags, so every file gets its own number.
    static FILE_NO: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn temp_path(tag: &str, extension: &str) -> PathBuf {
        let file_no = FILE_NO.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "g2g-m1136-{tag}-{}-{file_no}.{extension}",
            std::process::id()
        ))
    }

    pub(crate) fn have_ffmpeg() -> bool {
        Command::new("ffmpeg").arg("-version").output().is_ok()
            && Command::new("ffprobe").arg("-version").output().is_ok()
    }

    /// What ffprobe reads out of a stream's colour description. `extension` picks
    /// the demuxer (`h264`, `obu`, `jpg`).
    pub(crate) fn probe_colour(bytes: &[u8], tag: &str, extension: &str) -> ProbedColour {
        let path = temp_path(tag, extension);
        std::fs::write(&path, bytes).expect("write the stream to probe");
        let out = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=color_range,color_space,color_transfer,color_primaries",
                "-of",
                "default=noprint_wrappers=1",
            ])
            .arg(&path)
            .output()
            .expect("run ffprobe");
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(out.status.success(), "ffprobe read {tag}: {text}");
        let field = |name: &str| {
            text.lines()
                .find_map(|line| line.strip_prefix(&format!("{name}=")))
                .unwrap_or_else(|| panic!("ffprobe reported {name} for {tag}, got:\n{text}"))
                .to_string()
        };
        let probed = ProbedColour {
            range: field("color_range"),
            space: field("color_space"),
            transfer: field("color_transfer"),
            primaries: field("color_primaries"),
        };
        let _ = std::fs::remove_file(&path);
        probed
    }

    /// Encode the same geometry with the ffmpeg CLI under `colour_args` and probe
    /// the result: the expected reading the element's stream is compared against.
    pub(crate) fn ffmpeg_reference(
        codec_args: &[&str],
        colour_args: &[&str],
        tag: &str,
        extension: &str,
    ) -> ProbedColour {
        let path = temp_path(&format!("ref-{tag}"), extension);
        let status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}"),
                "-frames:v",
                &FRAMES.to_string(),
                "-pix_fmt",
                "yuv420p",
            ])
            .args(codec_args)
            .args(colour_args)
            .arg(&path)
            .status()
            .expect("run ffmpeg");
        assert!(status.success(), "ffmpeg authored the {tag} reference");
        let bytes = std::fs::read(&path).expect("read the reference stream");
        let _ = std::fs::remove_file(&path);
        probe_colour(&bytes, &format!("ref-{tag}"), extension)
    }
}

/// `Av1Enc` writes the caps colorimetry into the sequence header's
/// `color_config`, and leaves an untagged stream untagged.
#[cfg(feature = "av1-encode")]
mod av1 {
    use super::common::*;

    use g2g_core::frame::PipelinePacket;
    use g2g_core::{
        AsyncElement, Caps, Colorimetry, Dim, Interlace, Rate, RawVideoFormat, VideoCodec,
    };
    use g2g_plugins::av1enc::Av1Enc;

    /// The x264 colour request matching `Colorimetry::BT709`. ffmpeg's `-color_*`
    /// output options only tag the container, so the description has to be asked
    /// for through the encoder's own parameters.
    const BT709_ARGS: [&str; 4] = [
        "-x264-params",
        "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        "-color_range",
        "tv",
    ];
    const AOM_ARGS: [&str; 6] = ["-c:v", "libaom-av1", "-cpu-used", "8", "-f", "obu"];
    /// ffmpeg's AV1 encoders take no such colour parameters and write only the
    /// matrix and range into the sequence header, so the fully tagged reference is
    /// an H.264 stream asked for the same description: ffprobe prints the
    /// codepoint names out of the same table whatever the codec.
    const X264_ARGS: [&str; 2] = ["-c:v", "libx264"];

    fn raw_caps(colorimetry: Colorimetry) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(FPS << 16),
            interlace: Interlace::Any,
            colorimetry,
        }
    }

    async fn encode(colorimetry: Colorimetry) -> CaptureSink {
        let mut encoder = Av1Enc::new().with_speed(10);
        encoder
            .configure_pipeline(&raw_caps(colorimetry))
            .expect("configure Av1Enc");
        let mut sink = CaptureSink::default();
        for seq in 0..FRAMES {
            encoder
                .process(
                    PipelinePacket::DataFrame(system_frame(i420_frame(seq), seq)),
                    &mut sink,
                )
                .await
                .expect("encode frame");
        }
        encoder
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("flush Av1Enc");
        sink
    }

    #[tokio::test]
    async fn bt709_caps_reach_the_sequence_header_color_config() {
        let sink = encode(Colorimetry::BT709).await;
        assert_eq!(
            sink.caps,
            vec![Caps::CompressedVideo {
                codec: VideoCodec::Av1,
                width: Dim::Fixed(WIDTH),
                height: Dim::Fixed(HEIGHT),
                framerate: Rate::Fixed(FPS << 16),
                colorimetry: Colorimetry::BT709,
            }],
            "the encoder announces the input colorimetry on its output caps"
        );
        if !have_ffmpeg() {
            eprintln!("skipping the bitstream probe: ffmpeg / ffprobe not on PATH");
            return;
        }
        let tagged = ffmpeg_reference(&X264_ARGS, &BT709_ARGS, "bt709", "h264");
        let untagged = ffmpeg_reference(&X264_ARGS, &[], "untagged", "h264");
        assert_ne!(
            tagged, untagged,
            "the reference encodes have to differ, or the probe sees nothing"
        );
        assert_eq!(
            probe_colour(&sink.access_units.concat(), "bt709", "obu"),
            tagged,
            "bt709 caps must encode to the colour description ffmpeg writes for bt709"
        );
    }

    #[tokio::test]
    async fn untagged_caps_write_no_color_description() {
        let sink = encode(Colorimetry::UNKNOWN).await;
        assert_eq!(
            sink.caps,
            vec![Caps::CompressedVideo {
                codec: VideoCodec::Av1,
                width: Dim::Fixed(WIDTH),
                height: Dim::Fixed(HEIGHT),
                framerate: Rate::Fixed(FPS << 16),
                colorimetry: Colorimetry::UNKNOWN,
            }],
            "an untagged input stays untagged on the output caps"
        );
        if !have_ffmpeg() {
            eprintln!("skipping the bitstream probe: ffmpeg / ffprobe not on PATH");
            return;
        }
        assert_eq!(
            probe_colour(&sink.access_units.concat(), "untagged", "obu"),
            ffmpeg_reference(&AOM_ARGS, &[], "untagged", "obu"),
            "untagged caps must leave the stream as untagged as libaom's own"
        );
    }

    /// The sRGB triple means GBR planes at 4:4:4 full range, the one description
    /// AV1 constrains. A 4:2:0 input claiming it is refused rather than coded.
    #[test]
    fn srgb_tagged_subsampled_input_is_refused() {
        let mut encoder = Av1Enc::new();
        assert!(
            encoder
                .configure_pipeline(&raw_caps(Colorimetry::SRGB))
                .is_err(),
            "an identity matrix over subsampled chroma is not codable"
        );
    }
}

/// JPEG has no colour signalling, so `MjpegEnc` says what a JFIF file is and
/// converts with the input's own matrix rather than mislabeling the output.
#[cfg(feature = "mjpeg-encode")]
mod mjpeg {
    use super::common::*;

    use g2g_core::frame::PipelinePacket;
    use g2g_core::{
        AsyncElement, Caps, Colorimetry, Dim, Interlace, Rate, RawVideoFormat, VideoCodec,
    };
    use g2g_plugins::mjpegenc::MjpegEnc;

    fn raw_caps(colorimetry: Colorimetry) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(FPS << 16),
            interlace: Interlace::Any,
            colorimetry,
        }
    }

    async fn encode(colorimetry: Colorimetry) -> CaptureSink {
        let mut encoder = MjpegEnc::new();
        encoder
            .configure_pipeline(&raw_caps(colorimetry))
            .expect("configure MjpegEnc");
        let mut sink = CaptureSink::default();
        encoder
            .process(
                PipelinePacket::DataFrame(system_frame(i420_frame(0), 0)),
                &mut sink,
            )
            .await
            .expect("encode frame");
        sink
    }

    /// Whatever the input was tagged, the output caps say what the file is.
    #[tokio::test]
    async fn output_caps_say_jfif_whatever_the_input_said() {
        for input in [Colorimetry::UNKNOWN, Colorimetry::BT709, Colorimetry::BT601] {
            let sink = encode(input).await;
            assert_eq!(
                sink.caps,
                vec![Caps::CompressedVideo {
                    codec: VideoCodec::Mjpeg,
                    width: Dim::Fixed(WIDTH),
                    height: Dim::Fixed(HEIGHT),
                    framerate: Rate::Fixed(FPS << 16),
                    colorimetry: Colorimetry::JPEG,
                }],
                "a JPEG is JFIF colorimetry whatever fed it ({input:?})"
            );
        }
    }

    /// And the file really is a plain JFIF: ffprobe reads the same colour out of
    /// it as out of a JPEG ffmpeg wrote itself.
    #[tokio::test]
    async fn the_encoded_file_probes_as_jfif() {
        if !have_ffmpeg() {
            eprintln!("skipping: ffmpeg / ffprobe not on PATH");
            return;
        }
        let sink = encode(Colorimetry::BT709).await;
        assert_eq!(
            probe_colour(&sink.access_units[0], "jfif", "jpg"),
            ffmpeg_reference(&["-frames:v", "1"], &[], "jfif", "jpg"),
            "the encoded JPEG must read back as the same colour ffmpeg's own does"
        );
    }

    /// Nothing is mislabeled by that: the input tag picks the matrix the samples
    /// are converted through, so two differently tagged inputs encode differently.
    #[tokio::test]
    async fn the_input_tag_picks_the_conversion_matrix() {
        let bt709 = encode(Colorimetry::BT709).await;
        let bt601 = encode(Colorimetry::BT601).await;
        assert_ne!(
            bt709.access_units[0], bt601.access_units[0],
            "the same pixels under a different matrix must encode differently"
        );
    }

    /// The decoded result is the one the input's own matrix produces, not the
    /// other one: the encoder applied the tag rather than assuming a matrix.
    #[cfg(feature = "mjpeg")]
    #[tokio::test]
    async fn the_decoded_pixels_follow_the_input_matrix() {
        use g2g_plugins::mjpegdec::MjpegDec;

        async fn decode(jpeg: &[u8]) -> Vec<u8> {
            let mut decoder = MjpegDec::new();
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
                .expect("decode the JPEG back");
            sink.access_units.remove(0)
        }

        let (w, h) = (WIDTH as usize, HEIGHT as usize);
        let reference = |colorimetry| {
            g2g_plugins::videoconvert::convert(
                &i420_frame(0),
                RawVideoFormat::I420,
                RawVideoFormat::Rgba8,
                w,
                h,
                colorimetry,
            )
        };
        let mean_error = |a: &[u8], b: &[u8]| {
            let total: u64 = a
                .iter()
                .zip(b)
                .map(|(x, y)| x.abs_diff(*y) as u64)
                .sum::<u64>();
            total as f64 / a.len() as f64
        };

        for tag in [Colorimetry::BT709, Colorimetry::BT601] {
            let other = if tag == Colorimetry::BT709 {
                Colorimetry::BT601
            } else {
                Colorimetry::BT709
            };
            let encoded = encode(tag).await;
            let decoded = decode(&encoded.access_units[0]).await;
            assert_eq!(decoded.len(), w * h * 4, "RGBA out of the decoder");
            let matching = mean_error(&decoded, &reference(tag));
            let mismatched = mean_error(&decoded, &reference(other));
            assert!(
                matching < mismatched,
                "{tag:?}-tagged input must decode closer to its own matrix \
                 ({matching}) than to {other:?} ({mismatched})"
            );
        }
    }
}

/// `NvEnc` writes the caps colorimetry into the SPS VUI colour description.
/// Needs a real NVIDIA GPU; skips where CUDA / NVENC is unavailable. One session
/// only: two NVENC instances in one process crash the driver.
#[cfg(all(target_os = "linux", feature = "nvenc", feature = "cuda"))]
mod nvenc {
    use super::common::*;

    use g2g_core::frame::PipelinePacket;
    use g2g_core::{
        AsyncElement, Caps, Colorimetry, Dim, G2gError, Interlace, Rate, RawVideoFormat, VideoCodec,
    };
    use g2g_plugins::cuda::CudaUpload;
    use g2g_plugins::nvenc::NvEnc;

    const BT709_ARGS: [&str; 4] = [
        "-x264-params",
        "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        "-color_range",
        "tv",
    ];
    const X264_ARGS: [&str; 2] = ["-c:v", "libx264"];

    /// NV12 with a moving luma ramp over neutral chroma.
    fn nv12_frame(seq: u64) -> Vec<u8> {
        let (w, h) = (WIDTH as usize, HEIGHT as usize);
        let mut buf = vec![128u8; w * h + w * h / 2];
        for (i, b) in buf[..w * h].iter_mut().enumerate() {
            *b = ((i as u64 + seq * 7) & 0xff) as u8;
        }
        buf
    }

    fn nv12_caps(colorimetry: Colorimetry) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(FPS << 16),
            interlace: Interlace::Any,
            colorimetry,
        }
    }

    fn no_gpu(err: &G2gError) -> bool {
        if matches!(err, G2gError::Hardware(_)) {
            eprintln!("skipping m1136 nvenc: CUDA / NVENC unavailable ({err:?})");
            true
        } else {
            false
        }
    }

    #[tokio::test]
    async fn bt709_caps_reach_the_vui_colour_description() {
        let caps = nv12_caps(Colorimetry::BT709);
        let mut upload = CudaUpload::new();
        match upload.configure_pipeline(&caps) {
            Ok(_) => {}
            Err(e) if no_gpu(&e) => return,
            Err(e) => panic!("unexpected CudaUpload configure error: {e:?}"),
        }
        let mut encoder = NvEnc::new();
        match encoder.configure_pipeline(&caps) {
            Ok(_) => {}
            Err(e) if no_gpu(&e) => return,
            Err(e) => panic!("unexpected NvEnc configure error: {e:?}"),
        }

        let mut sink = CaptureSink::default();
        for seq in 0..FRAMES {
            let mut hop = CudaHop::default();
            upload
                .process(
                    PipelinePacket::DataFrame(system_frame(nv12_frame(seq), seq)),
                    &mut hop,
                )
                .await
                .expect("upload frame");
            for frame in hop.frames.drain(..) {
                encoder
                    .process(PipelinePacket::DataFrame(frame), &mut sink)
                    .await
                    .expect("encode frame");
            }
        }
        encoder
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("flush NvEnc");

        assert_eq!(
            sink.caps,
            vec![Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Fixed(WIDTH),
                height: Dim::Fixed(HEIGHT),
                framerate: Rate::Fixed(FPS << 16),
                colorimetry: Colorimetry::BT709,
            }],
            "the encoder announces the input colorimetry on its output caps"
        );
        if !have_ffmpeg() {
            eprintln!("skipping the bitstream probe: ffmpeg / ffprobe not on PATH");
            return;
        }
        let tagged = ffmpeg_reference(&X264_ARGS, &BT709_ARGS, "bt709", "h264");
        let untagged = ffmpeg_reference(&X264_ARGS, &[], "untagged", "h264");
        assert_ne!(
            tagged, untagged,
            "the reference encodes have to differ, or the probe sees nothing"
        );
        assert_eq!(
            probe_colour(&sink.access_units.concat(), "nvenc-bt709", "h264"),
            tagged,
            "NVENC must write the colour description the caps asked for"
        );
    }

    /// Collects the CUDA frames `CudaUpload` emits, which cannot go through
    /// `CaptureSink` (it keeps only system-memory payloads).
    #[derive(Default)]
    struct CudaHop {
        frames: Vec<g2g_core::Frame>,
    }

    impl g2g_core::OutputSink for CudaHop {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
                self.frames.push(frame);
            }
            core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
        }
    }
}
