//! M1153: `colorspace` converts PQ and HLG to and from the SDR transfers,
//! tone mapping an HDR source down to the 203 cd/m2 reference white that the
//! SDR curves put at code 255.
//!
//! The oracle is a straight f64 implementation of the same chain, written out
//! below: PQ EOTF, the BT.2390 EETF, the BT.709 OETF. The element folds each
//! curve into a 256-entry table and searches it to encode, so the two agree
//! only if the tables and the per-pixel steps are right.

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Colorimetry, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PropValue, PushOutcome, Rate, RawVideoFormat, TransferCharacteristics,
};
use g2g_plugins::colorspace::{Colorspace, DEFAULT_HDR_PEAK_NITS, MINIMUM_HDR_PEAK_NITS};

const WIDTH: usize = 8;
const HEIGHT: usize = 4;
const PIXELS: usize = WIDTH * HEIGHT;
const BYTES_PER_PIXEL: usize = 4;
const OPAQUE: u8 = 255;
const CODE_MAX: i32 = 255;

/// Reference white (ITU-R BT.2408), the light both sides call 1.0.
const REFERENCE_WHITE_NITS: f64 = 203.0;
/// A specular level above reference white, to exercise the roll-off.
const HIGHLIGHT_NITS: f64 = 600.0;
/// BT.2408 puts HDR reference white at 75% of the HLG signal.
const HLG_REFERENCE_WHITE_SIGNAL: f64 = 0.75;
/// Every eighth code, so one 32-pixel frame covers the range.
const RAMP_STEP: usize = 8;

/// Two codes: the element picks the output code nearest in linear light where
/// the reference rounds in the code domain, which differ by one either side,
/// and its f32 tables cost a fraction of another.
const REFERENCE_TOLERANCE: i32 = 2;
/// One code: the PQ and HLG paths arrive at the same light, so only the single
/// rounding of the shared BT.709 encode separates them, plus the 8-bit HLG
/// signal landing on 203.15 rather than 203.0 cd/m2.
const WHITE_AGREEMENT_TOLERANCE: i32 = 1;
/// Five percent: the darker of the two channels carries the ratio, and one
/// output code there is worth about half a percent of its light.
const HUE_RATIO_TOLERANCE: f64 = 0.05;
/// Three codes: a round trip quantizes to 8 bits at the PQ hop and again at the
/// BT.709 encode, and one PQ code is worth about 1.3 BT.709 codes near
/// reference white.
const ROUND_TRIP_TOLERANCE: i32 = 3;

// PQ (SMPTE ST 2084).
const PQ_PEAK_NITS: f64 = 10_000.0;
const PQ_M1: f64 = 2610.0 / 16384.0;
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0;

// HLG (ARIB STD-B67 / BT.2100).
const HLG_A: f64 = 0.178_832_77;
const HLG_B: f64 = 0.284_668_92;
const HLG_C: f64 = 0.559_910_73;
const HLG_DISPLAY_PEAK_NITS: f64 = 1000.0;
const HLG_SYSTEM_GAMMA: f64 = 1.2;

// BT.709's OETF.
const BT709_SLOPE: f64 = 4.5;
const BT709_LINEAR_BREAK: f64 = 0.018;
const BT709_ALPHA: f64 = 1.099;
const BT709_EXPONENT: f64 = 0.45;

/// PQ's EOTF: signal in 0..1 to cd/m2.
fn pq_to_nits(signal: f64) -> f64 {
    let encoded = signal.clamp(0.0, 1.0).powf(1.0 / PQ_M2);
    let numerator = (encoded - PQ_C1).max(0.0);
    (numerator / (PQ_C2 - PQ_C3 * encoded)).powf(1.0 / PQ_M1) * PQ_PEAK_NITS
}

/// PQ's inverse EOTF: cd/m2 to signal in 0..1.
fn nits_to_pq(nits: f64) -> f64 {
    let light = (nits / PQ_PEAK_NITS).clamp(0.0, 1.0).powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * light) / (1.0 + PQ_C3 * light)).powf(PQ_M2)
}

/// The 8-bit PQ code that carries `nits`.
fn pq_code(nits: f64) -> u8 {
    (nits_to_pq(nits) * f64::from(CODE_MAX)).round() as u8
}

/// The display light an 8-bit HLG code carries on the 1000 cd/m2 display its
/// OOTF is defined against, for a grey (all three channels equal, so the scene
/// luminance is the channel value).
fn hlg_grey_to_nits(signal: f64) -> f64 {
    let scene = match signal <= 0.5 {
        true => signal * signal / 3.0,
        false => (((signal - HLG_C) / HLG_A).exp() + HLG_B) / 12.0,
    };
    HLG_DISPLAY_PEAK_NITS * scene.powf(HLG_SYSTEM_GAMMA - 1.0) * scene
}

/// The BT.2390 EETF: cd/m2 in, reference-white-relative light out.
fn tone_map(nits: f64, peak_nits: f64) -> f64 {
    let peak_signal = nits_to_pq(peak_nits);
    let maximum_luminance = nits_to_pq(REFERENCE_WHITE_NITS) / peak_signal;
    let knee_start = 1.5 * maximum_luminance - 0.5;
    if knee_start >= 1.0 {
        return nits / REFERENCE_WHITE_NITS;
    }
    let signal = nits_to_pq(nits) / peak_signal;
    if signal < knee_start {
        return nits / REFERENCE_WHITE_NITS;
    }
    let t = ((signal - knee_start) / (1.0 - knee_start)).min(1.0);
    let (square, cube) = (t * t, t * t * t);
    let rolled = (2.0 * cube - 3.0 * square + 1.0) * knee_start
        + (cube - 2.0 * square + t) * (1.0 - knee_start)
        + (-2.0 * cube + 3.0 * square) * maximum_luminance;
    pq_to_nits(rolled * peak_signal) / REFERENCE_WHITE_NITS
}

/// The light at which the EETF stops being the identity, in reference-white
/// units. Below it a source code survives the tone map untouched.
fn knee_light(peak_nits: f64) -> f64 {
    let peak_signal = nits_to_pq(peak_nits);
    let knee_start = 1.5 * (nits_to_pq(REFERENCE_WHITE_NITS) / peak_signal) - 0.5;
    match knee_start >= 1.0 {
        true => f64::INFINITY,
        false => pq_to_nits(knee_start * peak_signal) / REFERENCE_WHITE_NITS,
    }
}

/// BT.709's OETF, to the 8-bit code the element writes.
fn bt709_code(linear: f64) -> i32 {
    let linear = linear.clamp(0.0, 1.0);
    let signal = match linear <= BT709_LINEAR_BREAK {
        true => BT709_SLOPE * linear,
        false => BT709_ALPHA * linear.powf(BT709_EXPONENT) - (BT709_ALPHA - 1.0),
    };
    (signal * f64::from(CODE_MAX)).round() as i32
}

/// The light an 8-bit BT.709 code carries.
fn bt709_light(code: u8) -> f64 {
    let signal = f64::from(code) / f64::from(CODE_MAX);
    match signal <= BT709_SLOPE * BT709_LINEAR_BREAK {
        true => signal / BT709_SLOPE,
        false => ((signal + BT709_ALPHA - 1.0) / BT709_ALPHA).powf(1.0 / BT709_EXPONENT),
    }
}

/// The whole PQ -> BT.709 chain in floats: what the element must reproduce.
fn expected_pq_to_bt709(code: u8, peak_nits: f64) -> i32 {
    let nits = pq_to_nits(f64::from(code) / f64::from(CODE_MAX));
    bt709_code(tone_map(nits, peak_nits))
}

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

fn raw(colorimetry: Colorimetry) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
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

/// One RGBA frame, `codes[index]` at pixel `index`.
fn rgba(codes: &[[u8; 3]]) -> Vec<u8> {
    assert_eq!(codes.len(), PIXELS, "one code triple per pixel");
    codes
        .iter()
        .flat_map(|[r, g, b]| [*r, *g, *b, OPAQUE])
        .collect()
}

/// A frame of one colour.
fn flat(code: [u8; 3]) -> Vec<u8> {
    rgba(&[code; PIXELS])
}

/// Convert one frame from `source` to `target` and hand back the RGB triple of
/// every pixel.
fn convert(
    source: Colorimetry,
    target: Colorimetry,
    peak_nits: Option<u32>,
    bytes: Vec<u8>,
) -> Vec<[i32; 3]> {
    let mut element = Colorspace::new();
    if let Some(nits) = peak_nits {
        element
            .set_property("hdr-peak-nits", PropValue::Uint(u64::from(nits)))
            .expect("a peak inside the declared range");
    }
    element
        .configure_pipeline(&raw(source))
        .expect("the input is accepted");
    element
        .configure_output(&raw(target))
        .expect("the output is accepted");
    let mut sink = CollectSink::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime
        .block_on(element.process(PipelinePacket::DataFrame(frame(bytes)), &mut sink))
        .expect("one frame converts");
    let out = sink
        .packets
        .iter()
        .find_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(
                frame
                    .domain
                    .require_system_bytes("m1153")
                    .expect("system bytes out")
                    .to_vec(),
            ),
            _ => None,
        })
        .expect("one frame out");
    out.as_chunks::<BYTES_PER_PIXEL>()
        .0
        .iter()
        .map(|pixel| {
            [
                i32::from(pixel[0]),
                i32::from(pixel[1]),
                i32::from(pixel[2]),
            ]
        })
        .collect()
}

/// The grey level of a pixel the conversion kept neutral.
fn grey_out(pixel: [i32; 3]) -> i32 {
    assert_eq!(pixel[0], pixel[1], "a grey stays neutral: {pixel:?}");
    assert_eq!(pixel[1], pixel[2], "a grey stays neutral: {pixel:?}");
    pixel[0]
}

/// PQ grey through to BT.709, against the float reference: reference white,
/// then a ramp over the whole code range.
#[test]
fn pq_grey_matches_the_float_reference() {
    let white = pq_code(REFERENCE_WHITE_NITS);
    let peak = f64::from(DEFAULT_HDR_PEAK_NITS);
    let converted = convert(
        Colorimetry::BT2100_PQ,
        Colorimetry::BT709,
        None,
        flat([white; 3]),
    );
    let expected = expected_pq_to_bt709(white, peak);
    assert!(
        (grey_out(converted[0]) - expected).abs() <= REFERENCE_TOLERANCE,
        "PQ {white} ({REFERENCE_WHITE_NITS} cd/m2) -> {}, reference {expected}",
        grey_out(converted[0])
    );

    let ramp: Vec<[u8; 3]> = (0..PIXELS)
        .map(|index| match index == PIXELS - 1 {
            true => [CODE_MAX as u8; 3],
            false => [(index * RAMP_STEP) as u8; 3],
        })
        .collect();
    let converted = convert(
        Colorimetry::BT2100_PQ,
        Colorimetry::BT709,
        None,
        rgba(&ramp),
    );
    for (pixel, [code, _, _]) in converted.iter().zip(&ramp) {
        let expected = expected_pq_to_bt709(*code, peak);
        assert!(
            (grey_out(*pixel) - expected).abs() <= REFERENCE_TOLERANCE,
            "PQ code {code} -> {}, reference {expected}",
            grey_out(*pixel)
        );
    }
}

/// The output caps say what the pixels became.
#[test]
fn a_tone_mapped_frame_is_labelled_sdr() {
    let mut element = Colorspace::new();
    element
        .configure_pipeline(&raw(Colorimetry::BT2100_PQ))
        .expect("a PQ input converts now");
    element
        .configure_output(&raw(Colorimetry::BT709))
        .expect("PQ -> BT.709 is a tone map, not a refusal");
    assert_eq!(
        element.output_colorimetry().transfer,
        TransferCharacteristics::Bt709
    );
}

/// Light above reference white stays above it on the way out, and telling the
/// tone map the source really peaks there puts it on SDR white.
#[test]
fn a_highlight_rides_the_roll_off() {
    let white = pq_code(REFERENCE_WHITE_NITS);
    let highlight = pq_code(HIGHLIGHT_NITS);
    let at_white = grey_out(
        convert(
            Colorimetry::BT2100_PQ,
            Colorimetry::BT709,
            None,
            flat([white; 3]),
        )[0],
    );
    let at_highlight = grey_out(
        convert(
            Colorimetry::BT2100_PQ,
            Colorimetry::BT709,
            None,
            flat([highlight; 3]),
        )[0],
    );
    assert!(
        at_highlight > at_white,
        "{HIGHLIGHT_NITS} cd/m2 -> {at_highlight} is no brighter than {REFERENCE_WHITE_NITS} cd/m2 -> {at_white}"
    );
    assert!(
        at_highlight <= CODE_MAX,
        "{at_highlight} is over full scale"
    );

    let peaked = grey_out(
        convert(
            Colorimetry::BT2100_PQ,
            Colorimetry::BT709,
            Some(HIGHLIGHT_NITS as u32),
            flat([highlight; 3]),
        )[0],
    );
    assert!(
        (peaked - CODE_MAX).abs() <= REFERENCE_TOLERANCE,
        "the declared peak maps to SDR white, got {peaked}"
    );
}

/// HLG's 75% signal and PQ's 203 cd/m2 are the same reference white, so the two
/// paths, inverse OETF plus OOTF against EOTF, have to land together.
#[test]
fn hlg_and_pq_reference_white_agree() {
    let hlg_white = (HLG_REFERENCE_WHITE_SIGNAL * f64::from(CODE_MAX)).round() as u8;
    // The 8-bit code is not exactly 75%, so it lands within one code of white.
    let carried = hlg_grey_to_nits(f64::from(hlg_white) / f64::from(CODE_MAX));
    let code_step = hlg_grey_to_nits(f64::from(hlg_white + 1) / f64::from(CODE_MAX)) - carried;
    assert!(
        (carried - REFERENCE_WHITE_NITS).abs() < code_step,
        "HLG {hlg_white} carries {carried} cd/m2, a code being {code_step}"
    );
    let from_hlg = grey_out(
        convert(
            Colorimetry::BT2100_HLG,
            Colorimetry::BT709,
            None,
            flat([hlg_white; 3]),
        )[0],
    );
    let from_pq = grey_out(
        convert(
            Colorimetry::BT2100_PQ,
            Colorimetry::BT709,
            None,
            flat([pq_code(REFERENCE_WHITE_NITS); 3]),
        )[0],
    );
    assert!(
        (from_hlg - from_pq).abs() <= WHITE_AGREEMENT_TOLERANCE,
        "HLG white -> {from_hlg}, PQ white -> {from_pq}"
    );
    // Both are the tone-mapped reference white, whose peak for HLG is its own
    // display, not the property.
    let expected = bt709_code(tone_map(REFERENCE_WHITE_NITS, HLG_DISPLAY_PEAK_NITS));
    assert!(
        (from_hlg - expected).abs() <= REFERENCE_TOLERANCE,
        "HLG white -> {from_hlg}, reference {expected}"
    );
}

/// The other direction: SDR white is reference white, so it encodes to the PQ
/// code for 203 cd/m2 and no higher.
#[test]
fn bt709_white_encodes_to_pq_reference_white() {
    let converted = convert(
        Colorimetry::BT709,
        Colorimetry::BT2100_PQ,
        None,
        flat([CODE_MAX as u8; 3]),
    );
    let expected = i32::from(pq_code(REFERENCE_WHITE_NITS));
    let out = grey_out(converted[0]);
    assert!(
        (out - expected).abs() <= WHITE_AGREEMENT_TOLERANCE,
        "BT.709 white -> PQ {out}, expected {expected}"
    );
}

/// A ramp out to PQ and back. Told the source peaks at reference white, the
/// tone map is the identity and the whole ramp survives; at the default peak,
/// everything below the knee still does.
#[test]
fn the_round_trip_through_pq_keeps_the_ramp() {
    let ramp: Vec<[u8; 3]> = (0..PIXELS)
        .map(|index| match index == PIXELS - 1 {
            true => [CODE_MAX as u8; 3],
            false => [(index * RAMP_STEP) as u8; 3],
        })
        .collect();
    let wide = convert(
        Colorimetry::BT709,
        Colorimetry::BT2100_PQ,
        None,
        rgba(&ramp),
    );
    let encoded: Vec<[u8; 3]> = wide
        .iter()
        .map(|pixel| [pixel[0] as u8, pixel[1] as u8, pixel[2] as u8])
        .collect();

    let recovered = convert(
        Colorimetry::BT2100_PQ,
        Colorimetry::BT709,
        Some(MINIMUM_HDR_PEAK_NITS),
        rgba(&encoded),
    );
    for (pixel, [code, _, _]) in recovered.iter().zip(&ramp) {
        let out = grey_out(*pixel);
        assert!(
            (out - i32::from(*code)).abs() <= ROUND_TRIP_TOLERANCE,
            "code {code} came back as {out} at a reference-white peak"
        );
    }

    let recovered = convert(
        Colorimetry::BT2100_PQ,
        Colorimetry::BT709,
        None,
        rgba(&encoded),
    );
    let knee = knee_light(f64::from(DEFAULT_HDR_PEAK_NITS));
    let mut checked = 0;
    for (pixel, [code, _, _]) in recovered.iter().zip(&ramp) {
        if bt709_light(*code) >= knee {
            continue;
        }
        checked += 1;
        let out = grey_out(*pixel);
        assert!(
            (out - i32::from(*code)).abs() <= ROUND_TRIP_TOLERANCE,
            "code {code} is below the knee and came back as {out}"
        );
    }
    assert!(checked > 0, "the knee left nothing to check");
}

/// The tone map scales all three channels by what it did to the brightest, so a
/// saturated colour keeps its hue. The BT.2020 -> BT.709 gamut step then takes
/// the two empty channels negative, which clips at zero.
#[test]
fn a_saturated_pq_red_keeps_its_hue() {
    let red = pq_code(HIGHLIGHT_NITS);
    let converted = convert(
        Colorimetry::BT2100_PQ,
        Colorimetry::BT709,
        None,
        flat([red, 0, 0]),
    );
    let [out_red, out_green, out_blue] = converted[0];
    assert!(out_red > 0, "the red channel survives: {out_red}");
    assert_eq!(out_green, 0, "green stays empty");
    assert_eq!(out_blue, 0, "blue stays empty");

    // With the primaries left alone there is no gamut step to mix the channels,
    // so the ratio between two unequal channels is the tone map's own doing: one
    // gain from the brightest, not a curve per channel. BT.2020's transfer is
    // the BT.709 curve, so the output codes read back with the same OETF.
    let converted = convert(
        Colorimetry::BT2100_PQ,
        Colorimetry::BT2020,
        None,
        flat([red, pq_code(REFERENCE_WHITE_NITS), 0]),
    );
    let [out_red, out_green, _] = converted[0];
    let ratio = bt709_light(out_red as u8) / bt709_light(out_green as u8);
    let expected = HIGHLIGHT_NITS / REFERENCE_WHITE_NITS;
    assert!(
        (ratio - expected).abs() < HUE_RATIO_TOLERANCE * expected,
        "red over green came out {ratio}, expected {expected} ({out_red} and {out_green})"
    );
}

/// HLG as a target: the inverse OOTF plus the OETF table put SDR white on the
/// signal that carries reference white on HLG's own display.
#[test]
fn bt709_white_encodes_to_hlg_reference_white() {
    let expected = (0..=CODE_MAX)
        .min_by(|left, right| {
            let light = |code: &i32| {
                (hlg_grey_to_nits(f64::from(*code) / f64::from(CODE_MAX)) - REFERENCE_WHITE_NITS)
                    .abs()
            };
            light(left).total_cmp(&light(right))
        })
        .expect("the code range is not empty");
    let converted = convert(
        Colorimetry::BT709,
        Colorimetry::BT2100_HLG,
        None,
        flat([CODE_MAX as u8; 3]),
    );
    let out = grey_out(converted[0]);
    assert!(
        (out - expected).abs() <= WHITE_AGREEMENT_TOLERANCE,
        "BT.709 white -> HLG {out}, expected {expected}"
    );
}
