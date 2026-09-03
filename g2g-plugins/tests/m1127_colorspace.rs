//! M1127: `colorspace` converts the colorimetry of raw video and leaves the
//! pixel format alone.
//!
//! The matrix / range half has an oracle already in the crate: decoding to
//! RGBA with the input colorimetry and re-encoding with the output one is what
//! `videoconvert ! videoconvert` does, and both read the same coefficient
//! table, so the expected pixels are computed rather than written down.

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, ColorRange, Colorimetry, Dim, Frame, FrameTiming, G2gError,
    MatrixCoefficients, MemoryDomain, OutputSink, PipelinePacket, PropValue, PushOutcome, Rate,
    RawVideoFormat, TransferCharacteristics,
};
use g2g_plugins::colorspace::Colorspace;
use g2g_plugins::videoconvert::convert;

const WIDTH: usize = 4;
const HEIGHT: usize = 4;

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

fn raw(format: RawVideoFormat, colorimetry: Colorimetry) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(WIDTH as u32),
        height: Dim::Fixed(HEIGHT as u32),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
        colorimetry,
    }
}

fn frame(bytes: Vec<u8>) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: 0,
            dts_ns: 0,
            duration_ns: 0,
            capture_ns: 0,
            arrival_ns: 0,
            keyframe: true,
        },
        sequence: 0,
        meta: Default::default(),
    }
}

/// A 4x4 picture whose every pixel is a different colour, built from RGB so
/// that each YUV triple is one that exists: a hand-written chroma plane can
/// pair a black luma with a saturated chroma, and the clamp that costs would
/// then show up as conversion error.
fn i420_test_frame() -> Vec<u8> {
    let rgba: Vec<u8> = (0..WIDTH * HEIGHT)
        .flat_map(|index| {
            [
                (40 + index * 12) as u8,
                (200 - index * 9) as u8,
                (70 + index * 6) as u8,
                255,
            ]
        })
        .collect();
    convert(
        &rgba,
        RawVideoFormat::Rgba8,
        RawVideoFormat::I420,
        WIDTH,
        HEIGHT,
        Colorimetry::BT709,
    )
    .into_vec()
}

/// The matrix / range conversion the element must reproduce, built from the
/// crate's own converter: `source` YUV -> RGBA, RGBA -> `target` YUV. Both
/// steps read the same colorimetry-derived coefficients the element does.
fn expected_via_rgb(
    src: &[u8],
    format: RawVideoFormat,
    source: Colorimetry,
    target: Colorimetry,
) -> Vec<u8> {
    let rgba = convert(src, format, RawVideoFormat::Rgba8, WIDTH, HEIGHT, source);
    convert(&rgba, RawVideoFormat::Rgba8, format, WIDTH, HEIGHT, target).into_vec()
}

/// Drive one frame through a configured element and return what it pushed.
fn run(element: &mut Colorspace, input: Caps, output: Caps, bytes: Vec<u8>) -> Vec<PipelinePacket> {
    element.configure_pipeline(&input).expect("input accepted");
    element.configure_output(&output).expect("output accepted");
    let mut sink = CollectSink::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime
        .block_on(element.process(PipelinePacket::DataFrame(frame(bytes)), &mut sink))
        .expect("one frame converts");
    sink.packets
}

fn pushed_caps(packets: &[PipelinePacket]) -> Caps {
    packets
        .iter()
        .find_map(|packet| match packet {
            PipelinePacket::CapsChanged(caps) => Some(caps.clone()),
            _ => None,
        })
        .expect("the output colorimetry is announced downstream")
}

fn pushed_bytes(packets: &[PipelinePacket]) -> Vec<u8> {
    packets
        .iter()
        .find_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => {
                let bytes = frame
                    .domain
                    .require_system_bytes("m1127")
                    .expect("system bytes out");
                Some(bytes.to_vec())
            }
            _ => None,
        })
        .expect("one frame out")
}

/// Matrix and range only, so the transfer and primaries stay out of it and the
/// RGB oracle applies exactly.
fn matrix_only(matrix: MatrixCoefficients, range: ColorRange) -> Colorimetry {
    Colorimetry {
        matrix,
        range,
        ..Colorimetry::UNKNOWN
    }
}

#[test]
fn bt709_to_bt601_matches_a_decode_then_reencode() {
    let source = matrix_only(MatrixCoefficients::Bt709, ColorRange::Limited);
    let target = matrix_only(MatrixCoefficients::Bt601, ColorRange::Limited);
    let src = i420_test_frame();

    let mut element = Colorspace::new();
    let packets = run(
        &mut element,
        raw(RawVideoFormat::I420, source),
        raw(RawVideoFormat::I420, target),
        src.clone(),
    );

    let expected = expected_via_rgb(&src, RawVideoFormat::I420, source, target);
    assert_eq!(pushed_bytes(&packets), expected);
    // The two matrices really differ, so the test would pass on a passthrough
    // only if the oracle were also a passthrough.
    assert_ne!(expected, src, "BT.709 and BT.601 luma weights differ");
    assert_eq!(pushed_caps(&packets), raw(RawVideoFormat::I420, target));
}

/// The same conversion on NV12: the chroma layout changes, the colour math does
/// not.
#[test]
fn nv12_converts_with_the_same_math_as_i420() {
    let source = matrix_only(MatrixCoefficients::Bt709, ColorRange::Limited);
    let target = matrix_only(MatrixCoefficients::Bt601, ColorRange::Full);
    let i420 = i420_test_frame();
    let nv12 = convert(
        &i420,
        RawVideoFormat::I420,
        RawVideoFormat::Nv12,
        WIDTH,
        HEIGHT,
        source,
    )
    .into_vec();

    let mut element = Colorspace::new();
    let packets = run(
        &mut element,
        raw(RawVideoFormat::Nv12, source),
        raw(RawVideoFormat::Nv12, target),
        nv12.clone(),
    );
    assert_eq!(
        pushed_bytes(&packets),
        expected_via_rgb(&nv12, RawVideoFormat::Nv12, source, target)
    );
}

/// Equal colorimetry on both sides copies the frame through, byte for byte.
#[test]
fn matching_colorimetry_passes_bytes_through() {
    let src = i420_test_frame();
    let mut element = Colorspace::new();
    let packets = run(
        &mut element,
        raw(RawVideoFormat::I420, Colorimetry::BT709),
        raw(RawVideoFormat::I420, Colorimetry::BT709),
        src.clone(),
    );
    assert_eq!(pushed_bytes(&packets), src);
    assert_eq!(
        pushed_caps(&packets),
        raw(RawVideoFormat::I420, Colorimetry::BT709)
    );
}

/// An unconstrained downstream leaves the stream alone: nothing to convert to.
#[test]
fn an_unconstrained_output_passes_bytes_through() {
    let src = i420_test_frame();
    let mut element = Colorspace::new();
    let packets = run(
        &mut element,
        raw(RawVideoFormat::I420, Colorimetry::BT709),
        raw(RawVideoFormat::I420, Colorimetry::UNKNOWN),
        src.clone(),
    );
    assert_eq!(pushed_bytes(&packets), src);
}

/// M1153: both HDR directions negotiate, and the emitted caps carry the
/// transfer that was asked for. The pixels are m1153's business.
#[test]
fn both_pq_directions_negotiate() {
    let mut element = Colorspace::new();
    element
        .configure_pipeline(&raw(RawVideoFormat::I420, Colorimetry::BT709))
        .expect("a BT.709 input is fine");
    element
        .configure_output(&raw(RawVideoFormat::I420, Colorimetry::BT2100_PQ))
        .expect("SDR -> PQ encodes absolute light");
    assert_eq!(element.output_colorimetry(), Colorimetry::BT2100_PQ);

    let mut from_pq = Colorspace::new();
    from_pq
        .configure_pipeline(&raw(RawVideoFormat::I420, Colorimetry::BT2100_PQ))
        .expect("a PQ input is fine to carry");
    from_pq
        .configure_output(&raw(RawVideoFormat::I420, Colorimetry::BT709))
        .expect("PQ -> SDR tone maps");
    assert_eq!(from_pq.output_colorimetry(), Colorimetry::BT709);
}

/// The property forces the target whatever downstream negotiated, and reads
/// back in the gst colorimetry spelling.
#[test]
fn the_colorimetry_property_forces_the_target() {
    let mut element = Colorspace::new();
    assert!(element
        .properties()
        .iter()
        .any(|spec| spec.name == "colorimetry"));
    element
        .set_property("colorimetry", PropValue::Str("bt601".into()))
        .expect("a preset name is a valid value");
    assert_eq!(
        element.get_property("colorimetry"),
        Some(PropValue::Str("bt601".into()))
    );
    assert_eq!(
        element.set_property("colorimetry", PropValue::Str("nonsense".into())),
        Err(g2g_core::PropError::Value)
    );

    // An unconstrained downstream now gets the forced target, not a
    // passthrough.
    let src = i420_test_frame();
    let packets = run(
        &mut element,
        raw(RawVideoFormat::I420, Colorimetry::BT709),
        raw(RawVideoFormat::I420, Colorimetry::UNKNOWN),
        src.clone(),
    );
    assert_ne!(pushed_bytes(&packets), src, "the property converted");
    assert_eq!(
        pushed_caps(&packets),
        raw(RawVideoFormat::I420, Colorimetry::BT601)
    );
}

/// A downstream pin the property contradicts fails negotiation instead of
/// emitting frames labelled one thing and converted to another.
#[test]
fn a_pinned_output_the_property_contradicts_is_refused() {
    let mut element = Colorspace::to(Colorimetry::BT601);
    element
        .configure_pipeline(&raw(RawVideoFormat::I420, Colorimetry::BT709))
        .unwrap();
    assert_eq!(
        element.configure_output(&raw(RawVideoFormat::I420, Colorimetry::BT2020)),
        Err(G2gError::CapsMismatch)
    );
}

/// The transfer and primaries convert through linear light, and the round trip
/// back recovers the picture within the two 8-bit quantizations it costs.
#[test]
fn a_primaries_change_round_trips() {
    let src = i420_test_frame();
    let mut out = Colorspace::to(Colorimetry::BT2020);
    let wide = pushed_bytes(&run(
        &mut out,
        raw(RawVideoFormat::I420, Colorimetry::BT709),
        raw(RawVideoFormat::I420, Colorimetry::UNKNOWN),
        src.clone(),
    ));
    assert_ne!(wide, src, "the gamut and matrix both moved");

    let mut back = Colorspace::to(Colorimetry::BT709);
    let recovered = pushed_bytes(&run(
        &mut back,
        raw(RawVideoFormat::I420, Colorimetry::BT2020),
        raw(RawVideoFormat::I420, Colorimetry::UNKNOWN),
        wide,
    ));
    let worst = recovered
        .iter()
        .zip(src.iter())
        .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
        .max()
        .unwrap();
    // Each leg quantizes to 8 bits three times (YUV -> RGB, the linear-light
    // encode, RGB -> YUV), so a couple of codes of drift is the cost of the
    // round trip, not a wrong matrix.
    assert!(worst <= 3, "worst byte off by {worst}");
}

/// An RGB layout carries a transfer and primaries but no matrix or range, so a
/// transfer change converts and the output caps say only what the pixels hold.
#[test]
fn an_rgb_stream_converts_its_transfer_only() {
    let source = Colorimetry::SRGB;
    let target = Colorimetry {
        transfer: TransferCharacteristics::Bt709,
        ..Colorimetry::SRGB
    };
    let src: Vec<u8> = (0..WIDTH * HEIGHT)
        .flat_map(|index| [(index * 15) as u8, 40, 200, 255])
        .collect();

    let mut element = Colorspace::new();
    let packets = run(
        &mut element,
        raw(RawVideoFormat::Rgba8, source),
        raw(RawVideoFormat::Rgba8, target),
        src.clone(),
    );
    let out = pushed_bytes(&packets);
    assert_ne!(out, src, "the sRGB and BT.709 curves differ");
    for (pixel, original) in out.as_chunks::<4>().0.iter().zip(src.as_chunks::<4>().0) {
        assert_eq!(pixel[3], original[3], "alpha rides through");
    }
    let Caps::RawVideo { colorimetry, .. } = pushed_caps(&packets) else {
        panic!("raw video out");
    };
    assert_eq!(colorimetry.transfer, TransferCharacteristics::Bt709);
    assert_eq!(colorimetry.matrix, MatrixCoefficients::Unknown);
    assert_eq!(colorimetry.range, ColorRange::Unknown);
}

/// An untagged transfer cannot be linearized, so it is not relabelled: the
/// matrix and range still convert, and the output says so on those two fields
/// alone.
#[test]
fn an_untagged_input_keeps_its_untagged_transfer() {
    let src = i420_test_frame();
    let mut element = Colorspace::to(Colorimetry::BT709);
    let packets = run(
        &mut element,
        raw(RawVideoFormat::I420, Colorimetry::UNKNOWN),
        raw(RawVideoFormat::I420, Colorimetry::UNKNOWN),
        src.clone(),
    );
    let Caps::RawVideo { colorimetry, .. } = pushed_caps(&packets) else {
        panic!("raw video out");
    };
    assert_eq!(
        colorimetry,
        matrix_only(MatrixCoefficients::Bt709, ColorRange::Limited)
    );
    // The BT.601 fallback for an unknown matrix is what it decoded with, so the
    // pixels are the 601 -> 709 conversion.
    assert_eq!(
        pushed_bytes(&packets),
        expected_via_rgb(
            &src,
            RawVideoFormat::I420,
            Colorimetry::UNKNOWN,
            matrix_only(MatrixCoefficients::Bt709, ColorRange::Limited),
        )
    );
}

/// A format with no colorimetry conversion here fails negotiation instead of
/// passing through silently mislabelled.
#[test]
fn an_unsupported_format_is_refused() {
    let mut element = Colorspace::new();
    assert_eq!(
        element
            .configure_pipeline(&raw(RawVideoFormat::I444, Colorimetry::BT709))
            .err(),
        Some(G2gError::CapsMismatch)
    );
}

/// 4:2:0 needs even dims on both axes, or the chroma plane does not divide.
#[test]
fn odd_dims_are_refused_for_420() {
    let mut element = Colorspace::new();
    let odd = Caps::RawVideo {
        format: RawVideoFormat::I420,
        width: Dim::Fixed(3),
        height: Dim::Fixed(4),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
        colorimetry: Colorimetry::BT709,
    };
    assert_eq!(
        element.configure_pipeline(&odd).err(),
        Some(G2gError::CapsMismatch)
    );
}
