//! M1113: `VideoConvert` converts with the matrix and range its caps
//! colorimetry names, not a hardcoded BT.601 limited.
//!
//! Every expected value here is derived from the stream's `(Kr, Kb)` and its
//! quantization range in floating point, independently of the element's integer
//! tables: a table that drifts from the definition fails, and so does an element
//! that ignores the caps (bt601 and bt709 have to disagree on a saturated
//! colour).

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, ColorRange, Colorimetry, Dim, Frame, FrameTiming, G2gError,
    LumaCoefficients, MatrixCoefficients, MemoryDomain, OutputSink, PipelinePacket, PushOutcome,
    Rate, RawVideoFormat,
};
use g2g_plugins::videoconvert::VideoConvert;

/// How far a converted channel may sit from the floating-point reference: the
/// element rounds its coefficients into 8-bit fixed point and rounds again per
/// pixel, and 4:2:0 chroma averaging adds nothing here because every fixture is
/// a solid colour.
const CHANNEL_TOLERANCE: i32 = 2;

#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn raw(format: RawVideoFormat, w: u32, h: u32, colorimetry: Colorimetry) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry,
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current-thread runtime")
}

fn frame(bytes: Vec<u8>) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

/// The colorimetry of a matrix and range, with the transfer and primaries left
/// unknown: this milestone converts neither.
fn tagged(matrix: MatrixCoefficients, range: ColorRange) -> Colorimetry {
    Colorimetry {
        matrix,
        range,
        ..Colorimetry::UNKNOWN
    }
}

/// Reference RGB -> YUV in floating point, straight from the definitions: luma
/// is the Kr/Kg/Kb weighted sum, Cb and Cr are the blue- and red-difference
/// signals, each scaled to its range's swing.
fn reference_rgb_to_yuv(rgb: [u8; 3], colorimetry: Colorimetry) -> [f32; 3] {
    let (kr, kb, full) = definition(colorimetry);
    let kg = 1.0 - kr - kb;
    let (r, g, b) = (rgb[0] as f32, rgb[1] as f32, rgb[2] as f32);
    let luma = (kr * r + kg * g + kb * b) / 255.0;
    let cb = (b / 255.0 - luma) / (2.0 * (1.0 - kb));
    let cr = (r / 255.0 - luma) / (2.0 * (1.0 - kr));
    match full {
        true => [luma * 255.0, cb * 255.0 + 128.0, cr * 255.0 + 128.0],
        false => [16.0 + luma * 219.0, cb * 224.0 + 128.0, cr * 224.0 + 128.0],
    }
}

/// The inverse of [`reference_rgb_to_yuv`], again from the definitions.
fn reference_yuv_to_rgb(yuv: [u8; 3], colorimetry: Colorimetry) -> [f32; 3] {
    let (kr, kb, full) = definition(colorimetry);
    let kg = 1.0 - kr - kb;
    let (y, u, v) = (yuv[0] as f32, yuv[1] as f32, yuv[2] as f32);
    let (luma, cb, cr) = match full {
        true => (y / 255.0, (u - 128.0) / 255.0, (v - 128.0) / 255.0),
        false => ((y - 16.0) / 219.0, (u - 128.0) / 224.0, (v - 128.0) / 224.0),
    };
    let r = luma + 2.0 * (1.0 - kr) * cr;
    let b = luma + 2.0 * (1.0 - kb) * cb;
    let g = luma - (2.0 * (1.0 - kb) * kb * cb + 2.0 * (1.0 - kr) * kr * cr) / kg;
    [r * 255.0, g * 255.0, b * 255.0]
}

/// `(Kr, Kb, full_range)` of a colorimetry, read off the core's coefficient
/// table rather than written out here.
fn definition(colorimetry: Colorimetry) -> (f32, f32, bool) {
    let luma = match colorimetry.matrix {
        MatrixCoefficients::Bt709 => LumaCoefficients::BT709,
        MatrixCoefficients::Bt2020Ncl => LumaCoefficients::BT2020_NCL,
        _ => LumaCoefficients::BT601,
    };
    (
        luma.kr,
        luma.kb,
        matches!(colorimetry.range, ColorRange::Full),
    )
}

/// Compare against the reference clamped into 8 bits: a YUV triple can decode
/// outside the RGB cube (nothing constrains the samples to be a real colour),
/// and the element clamps there.
fn assert_close(got: [u8; 3], want: [f32; 3], what: &str) {
    for channel in 0..3 {
        let reference = want[channel].round().clamp(0.0, 255.0) as i32;
        let delta = (got[channel] as i32 - reference).abs();
        assert!(
            delta <= CHANNEL_TOLERANCE,
            "{what}: channel {channel} is {}, reference {reference} ({:.2})",
            got[channel],
            want[channel]
        );
    }
}

/// Run one frame through a convert configured for `input` -> `output` caps and
/// return the emitted bytes.
fn convert_frame(input: Caps, output: Caps, bytes: Vec<u8>) -> Vec<u8> {
    let mut convert = VideoConvert::auto();
    convert.configure_pipeline(&input).expect("input caps");
    convert.configure_output(&output).expect("output caps");
    let mut sink = CollectSink::default();
    runtime()
        .block_on(convert.process(PipelinePacket::DataFrame(frame(bytes)), &mut sink))
        .expect("convert");
    sink.packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(
                f.domain
                    .require_system_slice("test")
                    .expect("system frame")
                    .to_vec(),
            ),
            _ => None,
        })
        .expect("a converted frame")
}

/// A 2x2 solid-colour RGBA frame: uniform, so 4:2:0 chroma averaging is exact
/// and the comparison isolates the colour math.
fn solid_rgba(rgb: [u8; 3]) -> Vec<u8> {
    (0..4).flat_map(|_| [rgb[0], rgb[1], rgb[2], 255]).collect()
}

/// A 2x2 solid-colour NV12 frame at the given YUV sample values.
fn solid_nv12(yuv: [u8; 3]) -> Vec<u8> {
    Vec::from([yuv[0], yuv[0], yuv[0], yuv[0], yuv[1], yuv[2]])
}

/// The four matrix / range combinations this milestone converts, exercised
/// RGBA -> NV12 against the floating-point definition.
#[test]
fn rgb_to_yuv_follows_the_caps_matrix_and_range() {
    let colors = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [30, 200, 90]];
    for matrix in [MatrixCoefficients::Bt601, MatrixCoefficients::Bt709] {
        for range in [ColorRange::Limited, ColorRange::Full] {
            let colorimetry = tagged(matrix, range);
            for rgb in colors {
                let out = convert_frame(
                    raw(RawVideoFormat::Rgba8, 2, 2, Colorimetry::UNKNOWN),
                    raw(RawVideoFormat::Nv12, 2, 2, colorimetry),
                    solid_rgba(rgb),
                );
                assert_close(
                    [out[0], out[4], out[5]],
                    reference_rgb_to_yuv(rgb, colorimetry),
                    &label(matrix, range, rgb),
                );
            }
        }
    }
}

/// The same four combinations the other way, NV12 -> RGBA.
#[test]
fn yuv_to_rgb_follows_the_caps_matrix_and_range() {
    let samples = [
        [81u8, 90, 240],
        [145, 54, 34],
        [41, 240, 110],
        [128, 128, 128],
    ];
    for matrix in [MatrixCoefficients::Bt601, MatrixCoefficients::Bt709] {
        for range in [ColorRange::Limited, ColorRange::Full] {
            let colorimetry = tagged(matrix, range);
            for yuv in samples {
                let out = convert_frame(
                    raw(RawVideoFormat::Nv12, 2, 2, colorimetry),
                    raw(RawVideoFormat::Rgba8, 2, 2, Colorimetry::UNKNOWN),
                    solid_nv12(yuv),
                );
                assert_close(
                    [out[0], out[1], out[2]],
                    reference_yuv_to_rgb(yuv, colorimetry),
                    &label(matrix, range, yuv),
                );
                assert_eq!(out[3], 255, "alpha stays opaque");
            }
        }
    }
}

/// The test that fails if the caps are ignored: a saturated colour must land on
/// different samples under BT.601 than under BT.709.
#[test]
fn bt601_and_bt709_disagree_on_a_saturated_colour() {
    let red = [255u8, 0, 0];
    let out_601 = convert_frame(
        raw(RawVideoFormat::Rgba8, 2, 2, Colorimetry::UNKNOWN),
        raw(RawVideoFormat::Nv12, 2, 2, Colorimetry::BT601),
        solid_rgba(red),
    );
    let out_709 = convert_frame(
        raw(RawVideoFormat::Rgba8, 2, 2, Colorimetry::UNKNOWN),
        raw(RawVideoFormat::Nv12, 2, 2, Colorimetry::BT709),
        solid_rgba(red),
    );
    assert_ne!(
        out_601, out_709,
        "the same red encoded under two matrices must differ"
    );
    // BT.709 weights red lower than BT.601, so its luma is the darker one.
    assert!(out_709[0] < out_601[0], "{} vs {}", out_709[0], out_601[0]);
}

/// An untagged stream keeps converting exactly as it did before caps carried
/// colour: BT.601 limited range.
#[test]
fn an_untagged_stream_converts_bt601_limited() {
    let rgb = [30u8, 200, 90];
    let untagged = convert_frame(
        raw(RawVideoFormat::Rgba8, 2, 2, Colorimetry::UNKNOWN),
        raw(RawVideoFormat::Nv12, 2, 2, Colorimetry::UNKNOWN),
        solid_rgba(rgb),
    );
    let tagged_601 = convert_frame(
        raw(RawVideoFormat::Rgba8, 2, 2, Colorimetry::UNKNOWN),
        raw(RawVideoFormat::Nv12, 2, 2, Colorimetry::BT601),
        solid_rgba(rgb),
    );
    assert_eq!(untagged, tagged_601);
}

/// A mid-stream `CapsChanged` re-derives the table: the same NV12 bytes convert
/// differently once the parser's VUI colour description arrives.
#[test]
fn a_mid_stream_caps_change_switches_the_matrix() {
    let yuv = [81u8, 90, 240];
    let mut convert = VideoConvert::auto();
    let output = raw(RawVideoFormat::Rgba8, 2, 2, Colorimetry::UNKNOWN);
    convert
        .configure_pipeline(&raw(RawVideoFormat::Nv12, 2, 2, Colorimetry::UNKNOWN))
        .unwrap();
    convert.configure_output(&output).unwrap();

    let mut sink = CollectSink::default();
    let rt = runtime();
    rt.block_on(convert.process(PipelinePacket::DataFrame(frame(solid_nv12(yuv))), &mut sink))
        .unwrap();

    // What the runner does on a refinement: reconfigure, then deliver the packet.
    convert
        .configure_pipeline(&raw(RawVideoFormat::Nv12, 2, 2, Colorimetry::BT709))
        .unwrap();
    rt.block_on(convert.process(PipelinePacket::DataFrame(frame(solid_nv12(yuv))), &mut sink))
        .unwrap();

    let frames: Vec<Vec<u8>> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                Some(f.domain.require_system_slice("test").unwrap().to_vec())
            }
            _ => None,
        })
        .collect();
    assert_eq!(frames.len(), 2);
    assert_close(
        [frames[0][0], frames[0][1], frames[0][2]],
        reference_yuv_to_rgb(yuv, Colorimetry::BT601),
        "before the refinement",
    );
    assert_close(
        [frames[1][0], frames[1][1], frames[1][2]],
        reference_yuv_to_rgb(yuv, Colorimetry::BT709),
        "after the refinement",
    );
}

/// A YUV output declares the matrix and range its samples were written with, so
/// the sink downstream converts them back the same way. An RGB output declares
/// neither: RGB samples have no matrix.
#[test]
fn the_output_caps_declare_what_was_written() {
    let mut convert = VideoConvert::auto();
    convert
        .configure_pipeline(&raw(RawVideoFormat::Nv12, 2, 2, Colorimetry::BT709))
        .unwrap();
    convert
        .configure_output(&raw(RawVideoFormat::I420, 2, 2, Colorimetry::UNKNOWN))
        .unwrap();
    let mut sink = CollectSink::default();
    runtime()
        .block_on(convert.process(
            PipelinePacket::DataFrame(frame(solid_nv12([81, 90, 240]))),
            &mut sink,
        ))
        .unwrap();

    let caps = sink
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .expect("caps announced");
    let Caps::RawVideo { colorimetry, .. } = caps else {
        panic!("expected raw-video caps");
    };
    assert_eq!(colorimetry.matrix, MatrixCoefficients::Bt709);
    assert_eq!(colorimetry.range, ColorRange::Limited);
}

fn label(matrix: MatrixCoefficients, range: ColorRange, values: [u8; 3]) -> String {
    format!("{matrix:?} {range:?} {values:?}")
}
