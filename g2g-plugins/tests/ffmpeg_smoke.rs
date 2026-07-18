//! M13: smoke test for `FfmpegH264Dec` (libavcodec software H.264 decoder).
//!
//! Ignored by default — requires:
//! - Linux with system libavcodec/libavformat/libavutil (Fedora:
//!   `ffmpeg-free-devel`; Debian: `libavcodec-dev libavformat-dev libavutil-dev`).
//! - An H.264 Annex-B fixture file path in `G2G_H264_FIXTURE`.
//!
//! Run with:
//!
//! ```sh
//! G2G_H264_FIXTURE=/path/to/clip.h264 cargo test -p g2g-plugins \
//!     --features ffmpeg --test ffmpeg_smoke -- --ignored --nocapture
//! ```
//!
//! Unlike `vaapi_smoke`, this test asserts decoded frames are produced —
//! ffmpeg's software decoder is portable enough that a green run is a real
//! end-to-end signal.

#![cfg(all(target_os = "linux", feature = "ffmpeg"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, ConfigureOutcome, Dim, G2gError, Rate, VideoCodec, RawVideoFormat};
use g2g_plugins::ffmpegdec::{Backend, FfmpegVideoDec, OutputFormat};

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

#[tokio::test]
#[ignore = "requires libav* and a G2G_H264_FIXTURE path"]
async fn ffmpeg_h264_decodes_fixture() {
    decode_fixture(VideoCodec::H264, "G2G_H264_FIXTURE", OutputFormat::I420).await;
}

#[tokio::test]
#[ignore = "requires libav* and a G2G_H264_FIXTURE path"]
async fn ffmpeg_h264_decodes_fixture_nv12() {
    decode_fixture(VideoCodec::H264, "G2G_H264_FIXTURE", OutputFormat::Nv12).await;
}

// M111: the generalized decoder takes VP9 too. Set G2G_VP9_FIXTURE to a raw VP9
// elementary stream (e.g. extracted with `mkvextract` or `ffmpeg -c:v copy`).
#[tokio::test]
#[ignore = "requires libav* and a G2G_VP9_FIXTURE path"]
async fn ffmpeg_vp9_decodes_fixture() {
    decode_fixture(VideoCodec::Vp9, "G2G_VP9_FIXTURE", OutputFormat::I420).await;
}

// M(vaapi): VAAPI hardware decode through ffmpeg. Same fixture + assertion as
// the software path, but `Backend::Vaapi` pinned to a render node (default
// `/dev/dri/renderD128`, overridable via `G2G_VAAPI_DEVICE`). Validates the
// libavcodec VAAPI hwaccel path on AMD / Intel where cros-codecs `VaapiH264Dec`
// is blocked. `configure_pipeline` fails loud if the libavcodec build lacks the
// VAAPI hwaccel or the render node isn't libva-capable, so a green run is a
// real end-to-end hardware-decode signal.
#[tokio::test]
#[ignore = "requires libav* with VAAPI, a libva render node, and a G2G_H264_FIXTURE path"]
async fn ffmpeg_h264_decodes_fixture_vaapi() {
    let device = std::env::var("G2G_VAAPI_DEVICE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
    let dec = FfmpegVideoDec::new()
        .with_output_format(OutputFormat::Nv12)
        .with_backend(Backend::Vaapi)
        .with_vaapi_device(Some(&device));
    decode_fixture_with(dec, VideoCodec::H264, "G2G_H264_FIXTURE", OutputFormat::Nv12).await;
}

async fn decode_fixture(codec: VideoCodec, env_var: &str, output: OutputFormat) {
    let dec = FfmpegVideoDec::new().with_output_format(output);
    decode_fixture_with(dec, codec, env_var, output).await;
}

async fn decode_fixture_with(
    mut dec: FfmpegVideoDec,
    codec: VideoCodec,
    env_var: &str,
    output: OutputFormat,
) {
    let Some(path) = std::env::var_os(env_var) else {
        eprintln!("skipping: set {env_var}=/path/to/clip to run");
        return;
    };
    let bitstream = std::fs::read(&path).expect("read fixture");
    assert!(!bitstream.is_empty(), "fixture is empty");

    let upstream = Caps::CompressedVideo {
        codec,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    let narrowed = dec.intercept_caps(&upstream).expect("intercept supported codec");
    let outcome = dec
        .configure_pipeline(&narrowed)
        .expect("libavcodec must initialise");
    assert!(matches!(outcome, ConfigureOutcome::Accepted));

    let mut sink = Collect::default();

    // Feed the whole fixture as one packet. `H264Parse` will normally
    // deliver one *access unit* (SPS + PPS + SEI + slices for one picture)
    // per `DataFrame`; libavcodec's bitstream filter accepts that shape
    // happily. Splitting further (one NAL per packet) breaks the SPS/PPS
    // bookkeeping that h264 expects to see alongside the first slice. This
    // smoke test just validates that the path produces a decoded frame; a
    // multi-frame test belongs alongside the real `H264Parse` element.
    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bitstream.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };
    dec.process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("process DataFrame");
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("process Eos");

    let caps_changes: Vec<_> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .collect();
    let data_frames: Vec<_> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();

    eprintln!(
        "decoded {} frame(s); {} CapsChanged emitted",
        data_frames.len(),
        caps_changes.len()
    );
    assert!(!caps_changes.is_empty(), "expected at least one CapsChanged");
    assert!(!data_frames.is_empty(), "expected at least one decoded frame");

    // I420 and NV12 have identical byte length (w*h*3/2 for even dims); only
    // the chroma layout differs. The runner checks length + format tag.
    let expected_format = match output {
        OutputFormat::I420 => RawVideoFormat::I420,
        OutputFormat::Nv12 => RawVideoFormat::Nv12,
        OutputFormat::I422 => RawVideoFormat::I422,
        OutputFormat::I444 => RawVideoFormat::I444,
        // These smoke fixtures are decoded with a fixed output format; Auto is
        // resolved per frame and covered by the unit tests in the module.
        OutputFormat::Auto => unreachable!("smoke fixtures use a fixed output format"),
    };
    let first = caps_changes.first().unwrap();
    match first {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } if *format == expected_format => {
            eprintln!("first {:?} caps: {}x{}", expected_format, w, h);
            let f = data_frames.first().unwrap();
            let cw = (*w).div_ceil(2) as usize;
            let ch = (*h).div_ceil(2) as usize;
            let expected = (*w as usize) * (*h as usize) + 2 * cw * ch;
            match &f.domain {
                MemoryDomain::System(slice) => {
                    assert_eq!(
                        slice.as_slice().len(),
                        expected,
                        "{:?} byte length mismatch",
                        expected_format,
                    );
                }
                _ => panic!("decoder must emit System-domain frames"),
            }
        }
        other => panic!("expected fixed {expected_format:?} caps, got {other:?}"),
    }
}

/// Regression for the playbin -> strict-NV12-sink startup gap: the auto-plug
/// factory must build the ffmpeg decoder with the output layout the search chose,
/// not a fixed I420. A KMS / waylandsink advertises `Accepts(NV12)`, so the search
/// settles the decode hop on NV12 (the first source-pad-template alternative) and
/// the runner forward-prefixes `CapsChanged(NV12)` into the decoder before the
/// first frame. The old `|_| FfmpegH264Dec::new()` factory ignored the chosen caps
/// and built an I420 decoder, whose run loop rejected that NV12 pre-fix with
/// `CapsMismatch` (output layout != built layout), stalling startup negotiation.
///
/// Deterministic and fixture-free: the failure is in negotiation, before any data,
/// so it needs only `libavcodec` present to open the (frameless) decoder.
#[tokio::test]
async fn autoplug_builds_nv12_decoder_for_strict_nv12_sink() {
    // Scoped here so the blanket `impl DynAsyncElement for T: AsyncElement` does
    // not collide with the concrete `AsyncElement` calls in the fixture tests.
    use g2g_core::element::DynAsyncElement;
    use g2g_plugins::registry::default_registry;

    let reg = default_registry();
    let h264 = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    // A strict-NV12 sink target, mirroring `CudaKmsSink` / `waylandsink`'s
    // `Accepts(NV12)` constraint.
    let strict_nv12 = |c: &Caps| matches!(c, Caps::RawVideo { format: RawVideoFormat::Nv12, .. });

    let mut chain = reg.autoplug(&h264, &strict_nv12, 4).expect("a decode chain reaches NV12");
    assert_eq!(chain.len(), 1, "one decoder hop to NV12, got {}", chain.len());
    let dec = chain[0].as_mut();
    // Fully qualified: the boxed element satisfies both `AsyncElement` and
    // `DynAsyncElement`, so disambiguate to the dyn methods.
    DynAsyncElement::configure_pipeline(dec, &h264).expect("libavcodec opens the H.264 decoder");

    // The exact forward-caps pre-fix the runner pushes from the NV12 sink. An
    // I420-built decoder (the bug) returns `CapsMismatch` here; an NV12-built one
    // forwards it.
    let nv12 = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    let mut sink = Collect::default();
    DynAsyncElement::process(dec, PipelinePacket::CapsChanged(nv12), &mut sink)
        .await
        .expect("the auto-plugged decoder must emit NV12 for a strict-NV12 sink");
}

