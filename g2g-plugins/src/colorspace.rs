//! Colorimetry converter (M1127): changes what a raw frame's samples *mean*
//! while leaving the pixel format alone, the complement of
//! [`videoconvert`](crate::videoconvert), which changes the format and carries
//! the colorimetry through.
//!
//! Two conversions, either or both per frame:
//!
//! - Matrix and range, through an RGB intermediate: decode with the input
//!   colorimetry's `YuvRgbMatrix`, encode with the output's. This is the
//!   BT.709 -> BT.601 / limited -> full case.
//! - Transfer and primaries, through linear-light RGB: linearize with the input
//!   transfer, apply the source-primaries-to-target-primaries matrix (both
//!   D65, so no chromatic adaptation), re-encode with the output transfer.
//!
//! PQ and HLG (M1153) go through the same linear step: light relative to the
//! 203 cd/m2 HDR reference white of BT.2408, which is what the SDR curves
//! already produce at code 255. An HDR source headed for an SDR transfer is
//! tone mapped on the way, with the BT.2390 EETF from `hdr-peak-nits` down to
//! that white; the other direction encodes the light it has and expands no
//! highlights.
//!
//! The transfer and primaries cannot be converted away from an *untagged*
//! input, since there is no curve to linearize with; the output then stays
//! untagged on those two fields rather than claiming a conversion that did not
//! happen. The matrix and range do convert from an untagged input, because
//! [`Colorimetry::yuv_conversion`](g2g_core::Colorimetry::yuv_conversion)
//! resolves an unknown one to BT.601 limited.
//!
//! CPU-only and `no_std`: the curves run on `mathf`, and each is folded into a
//! 256-entry table at negotiation, so a pixel pays a `powf` only for the two
//! steps whose value is per pixel rather than per code, the HLG OOTF and the
//! tone map.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::mathf::{exp, powf};
use crate::pixel::{carries_yuv, even_dims_required, frame_byte_size, rgba_rb_offsets};
use crate::yuvmatrix::{YuvRgbMatrix, SAMPLE_MAX};
use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, CapsTransform, ColorPrimaries, ColorRange,
    Colorimetry, ConfigureOutcome, Dim, ElementMetadata, G2gError, LumaCoefficients,
    MatrixCoefficients, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, RawVideoShape,
    TransferCharacteristics,
};

/// Formats whose colorimetry this element converts. The 4:2:0 pair carry a YUV
/// matrix and range; the packed RGB three carry only a transfer and primaries.
/// A format outside this set fails negotiation rather than passing through
/// mislabelled.
const FORMATS: [RawVideoFormat; 5] = [
    RawVideoFormat::I420,
    RawVideoFormat::Nv12,
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Rgb8,
];

/// How many 8-bit codes a transfer table has an entry for.
const CODE_COUNT: usize = SAMPLE_MAX as usize + 1;

/// The light linear 1.0 stands for: HDR reference white, 203 cd/m2 (ITU-R
/// BT.2408). An SDR curve puts code 255 here, so the SDR and HDR halves of a
/// conversion meet on one scale.
const REFERENCE_WHITE_NITS: f64 = 203.0;

/// The `hdr-peak-nits` default, the mastering peak of a common HDR10 grade.
pub const DEFAULT_HDR_PEAK_NITS: u32 = 1000;
/// Reference white is the lowest peak worth naming: at it there is nothing
/// above the SDR range and the tone map is the identity.
pub const MINIMUM_HDR_PEAK_NITS: u32 = REFERENCE_WHITE_NITS as u32;
/// PQ's own peak, the brightest a signal can name.
pub const MAXIMUM_HDR_PEAK_NITS: u32 = PQ_PEAK_NITS as u32;

/// # Example
///
/// ```no_run
/// use g2g_core::Colorimetry;
/// use g2g_plugins::colorspace::Colorspace;
///
/// let negotiated = Colorspace::new();
/// let forced = Colorspace::to(Colorimetry::BT601);
/// ```
#[derive(Debug)]
pub struct Colorspace {
    /// The `colorimetry` property: the output colorimetry, whatever downstream
    /// negotiated. `None` takes it from the negotiated output caps.
    forced: Option<Colorimetry>,
    input: Option<InputStream>,
    /// Colorimetry of the negotiated output caps, the target when `forced` is
    /// unset.
    negotiated: Colorimetry,
    /// What the emitted frames carry, and how to produce them. Rebuilt whenever
    /// either side's caps arrive.
    target: Colorimetry,
    /// The `hdr-peak-nits` property: the peak a PQ source is graded to, which
    /// is what the tone map compresses down to reference white.
    hdr_peak_nits: u32,
    plan: Option<Plan>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

/// Format, geometry, framerate and colorimetry of the stream being converted.
/// The framerate rides through unchanged (this does not retime), so downstream
/// sees a fixed rate rather than a `Rate::Any` a fixating peer would reject.
#[derive(Clone, Debug)]
struct InputStream {
    format: RawVideoFormat,
    width: u32,
    height: u32,
    framerate: Rate,
    colorimetry: Colorimetry,
}

impl Colorspace {
    /// Take the output colorimetry from the negotiated caps (a downstream
    /// capsfilter). With no downstream constraint this passes frames through.
    pub fn new() -> Self {
        Self {
            forced: None,
            input: None,
            negotiated: Colorimetry::UNKNOWN,
            target: Colorimetry::UNKNOWN,
            hdr_peak_nits: DEFAULT_HDR_PEAK_NITS,
            plan: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Convert to a fixed output colorimetry, whatever downstream asked for.
    pub fn to(colorimetry: Colorimetry) -> Self {
        Self {
            forced: Some(colorimetry),
            ..Self::new()
        }
    }

    /// The colorimetry the emitted frames carry: the requested one narrowed to
    /// what this element can actually deliver from the input, then stripped of
    /// the fields the output pixel format does not carry.
    pub fn output_colorimetry(&self) -> Colorimetry {
        self.target
    }

    /// Resolve the target and build the conversion. Called from both configure
    /// hooks, so a runner that delivers only the input caps still gets a plan.
    fn rebuild_plan(&mut self) -> Result<(), G2gError> {
        let Some(input) = self.input.clone() else {
            return Ok(());
        };
        let requested = self.forced.unwrap_or(self.negotiated);
        let target = carried(input.format, achievable(input.colorimetry, requested));
        let source = carried(input.format, input.colorimetry);
        self.plan = Some(Plan::build(source, target, f64::from(self.hdr_peak_nits))?);
        self.target = target;
        Ok(())
    }

    /// Validate a raw-video caps as a convertible input. 4:2:0 needs even dims
    /// so the chroma plane divides.
    fn accept_input(caps: &Caps) -> Result<InputStream, G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate,
            colorimetry,
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !FORMATS.contains(format) || *w == 0 || *h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        let (even_width, even_height) = even_dims_required(*format);
        if (even_width && *w % 2 != 0) || (even_height && *h % 2 != 0) {
            return Err(G2gError::CapsMismatch);
        }
        Ok(InputStream {
            format: *format,
            width: *w,
            height: *h,
            framerate: framerate.clone(),
            colorimetry: *colorimetry,
        })
    }
}

impl Default for Colorspace {
    fn default() -> Self {
        Self::new()
    }
}

/// What of a colorimetry the pixels of `format` carry. An RGB layout has no YUV
/// matrix and no studio range, so those two fields name nothing to convert and
/// nothing to declare.
fn carried(format: RawVideoFormat, colorimetry: Colorimetry) -> Colorimetry {
    match carries_yuv(format) {
        true => colorimetry,
        false => Colorimetry {
            range: ColorRange::Unknown,
            matrix: MatrixCoefficients::Unknown,
            ..colorimetry
        },
    }
}

/// The colorimetry this element can deliver from `input` when `requested` is
/// asked for. A field `requested` leaves unknown keeps the input's. The
/// transfer and primaries additionally stay at the input's when *it* is the
/// unknown one: there is no curve to linearize an untagged stream with, so
/// relabelling it would be a claim the pixels do not support.
fn achievable(input: Colorimetry, requested: Colorimetry) -> Colorimetry {
    /// The request when it names something, else what the input carries.
    fn convertible<T: PartialEq>(request: T, source: T, unknown: T) -> T {
        match request == unknown {
            true => source,
            false => request,
        }
    }
    /// [`convertible`] for a field that can only be converted away from a
    /// concrete value.
    fn from_tagged<T: PartialEq + Copy>(request: T, source: T, unknown: T) -> T {
        match source == unknown {
            true => source,
            false => convertible(request, source, unknown),
        }
    }
    Colorimetry {
        range: convertible(requested.range, input.range, ColorRange::Unknown),
        matrix: convertible(requested.matrix, input.matrix, MatrixCoefficients::Unknown),
        transfer: from_tagged(
            requested.transfer,
            input.transfer,
            TransferCharacteristics::Unknown,
        ),
        primaries: from_tagged(
            requested.primaries,
            input.primaries,
            ColorPrimaries::Unknown,
        ),
    }
}

/// What one frame needs: nothing at all, or the colour transform to run over it.
#[derive(Debug)]
enum Plan {
    Passthrough,
    Convert(ColorTransform),
}

impl Plan {
    fn build(
        source: Colorimetry,
        target: Colorimetry,
        pq_peak_nits: f64,
    ) -> Result<Self, G2gError> {
        if source == target {
            return Ok(Plan::Passthrough);
        }
        Ok(Plan::Convert(ColorTransform {
            decode: YuvRgbMatrix::new(source),
            encode: YuvRgbMatrix::new(target),
            light: LightTransform::build(source, target, pq_peak_nits)?,
        }))
    }
}

/// One stream's colour conversion: the YUV matrices of both ends, plus the
/// linear-light step when the transfer or primaries move too.
#[derive(Debug)]
struct ColorTransform {
    decode: YuvRgbMatrix,
    encode: YuvRgbMatrix,
    /// Boxed: the two 256-entry tables dwarf everything else here, and a
    /// matrix-only conversion carries none of it.
    light: Option<Box<LightTransform>>,
}

impl ColorTransform {
    /// Source YUV to target-colorimetry 8-bit RGB.
    fn yuv_to_target_rgb(&self, y: i32, u: i32, v: i32) -> (i32, i32, i32) {
        let (r, g, b) = self.decode.yuv_to_rgb(y, u, v);
        self.rgb_to_target_rgb(r, g, b)
    }

    /// Source 8-bit RGB to target-colorimetry 8-bit RGB.
    fn rgb_to_target_rgb(&self, r: i32, g: i32, b: i32) -> (i32, i32, i32) {
        match &self.light {
            Some(light) => light.apply(r, g, b),
            None => (r, g, b),
        }
    }
}

/// The transfer + primaries half of a conversion, as tables: linear light per
/// source code, the source-to-target primaries matrix in linear light, and the
/// linear light each target code stands for (searched to encode). Built once
/// per negotiated colorimetry pair, so the per-pixel cost is two table reads
/// and a 3x3 multiply.
struct LightTransform {
    source_linear: [f32; CODE_COUNT],
    target_linear: [f32; CODE_COUNT],
    gamut: [[f32; 3]; 3],
    /// The HLG tables hold scene light, so the OOTF has to run on top of them
    /// to reach the display light everything else is in.
    source_is_scene_light: bool,
    target_is_scene_light: bool,
    tone_map: Option<ToneMap>,
}

impl core::fmt::Debug for LightTransform {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LightTransform")
            .field("gamut", &self.gamut)
            .field("tone_map", &self.tone_map)
            .finish_non_exhaustive()
    }
}

impl LightTransform {
    /// `None` when the transfer and primaries both stay put. `Err` when the
    /// conversion needs linear light but one end is untagged, which has no
    /// curve to linearize with.
    fn build(
        source: Colorimetry,
        target: Colorimetry,
        pq_peak_nits: f64,
    ) -> Result<Option<Box<Self>>, G2gError> {
        if source.transfer == target.transfer && source.primaries == target.primaries {
            return Ok(None);
        }
        let (Some(source_curve), Some(target_curve)) = (
            transfer_curve(source.transfer),
            transfer_curve(target.transfer),
        ) else {
            return Err(G2gError::CapsMismatch);
        };
        // HDR down to an SDR transfer is the one direction that compresses: the
        // other way encodes the light it has, expanding no highlights.
        let tone_map = match (
            source_curve.peak_nits(pq_peak_nits),
            target_curve.peak_nits(pq_peak_nits),
        ) {
            (Some(source_peak), None) => ToneMap::new(source_peak),
            _ => None,
        };
        Ok(Some(Box::new(Self {
            source_linear: linear_table(&source_curve),
            target_linear: linear_table(&target_curve),
            gamut: gamut_matrix(source.primaries, target.primaries)?,
            source_is_scene_light: source_curve.is_scene_light(),
            target_is_scene_light: target_curve.is_scene_light(),
            tone_map,
        })))
    }

    /// Source 8-bit RGB to target 8-bit RGB through linear light.
    fn apply(&self, r: i32, g: i32, b: i32) -> (i32, i32, i32) {
        let code = |v: i32| self.source_linear[v.clamp(0, SAMPLE_MAX) as usize];
        let mut linear = [code(r), code(g), code(b)];
        if self.source_is_scene_light {
            let gain = hlg_ootf_gain(linear);
            scale(&mut linear, gain);
        }
        if let Some(tone_map) = &self.tone_map {
            let gain = tone_map.gain(linear);
            scale(&mut linear, gain);
        }
        let mixed = |row: [f32; 3]| row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2];
        let mut out = [
            mixed(self.gamut[0]),
            mixed(self.gamut[1]),
            mixed(self.gamut[2]),
        ];
        if self.target_is_scene_light {
            let gain = hlg_inverse_ootf_gain(out);
            scale(&mut out, gain);
        }
        (
            self.encode(out[0]),
            self.encode(out[1]),
            self.encode(out[2]),
        )
    }

    /// The target code whose linear light is nearest `linear`. The table
    /// ascends, so a partition point plus one comparison with its predecessor
    /// picks it, and an out-of-gamut value lands on an endpoint.
    fn encode(&self, linear: f32) -> i32 {
        let above = self.target_linear.partition_point(|&entry| entry < linear);
        if above == 0 {
            return 0;
        }
        if above >= CODE_COUNT {
            return SAMPLE_MAX;
        }
        let (below, above) = (above - 1, above);
        match linear - self.target_linear[below] <= self.target_linear[above] - linear {
            true => below as i32,
            false => above as i32,
        }
    }
}

/// An SDR opto-electronic transfer function in the shape BT.709 and sRGB share:
/// a linear segment of slope `slope` up to `linear_break`, then
/// `alpha * L^exponent - (alpha - 1)`.
struct TransferCurve {
    slope: f64,
    linear_break: f64,
    alpha: f64,
    exponent: f64,
}

/// BT.709's OETF, which BT.601 and BT.2020 tag as their own: 4.5 L below the
/// break, 1.099 L^0.45 - 0.099 above it.
const BT709_CURVE: TransferCurve = TransferCurve {
    slope: 4.5,
    linear_break: 0.018,
    alpha: 1.099,
    exponent: 0.45,
};

/// sRGB (IEC 61966-2-1): 12.92 L below the break, 1.055 L^(1/2.4) - 0.055.
const SRGB_CURVE: TransferCurve = TransferCurve {
    slope: 12.92,
    linear_break: 0.003_130_8,
    alpha: 1.055,
    exponent: 1.0 / 2.4,
};

impl TransferCurve {
    /// Nonlinear code value in 0..1 to linear light.
    fn to_linear(&self, value: f64) -> f64 {
        let value = value.clamp(0.0, 1.0);
        match value <= self.slope * self.linear_break {
            true => value / self.slope,
            false => powf((value + self.alpha - 1.0) / self.alpha, 1.0 / self.exponent),
        }
    }
}

/// PQ's peak (SMPTE ST 2084), the light its signal 1.0 names.
const PQ_PEAK_NITS: f64 = 10_000.0;
const PQ_M1: f64 = 2610.0 / 16384.0;
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0;

/// PQ's EOTF: signal in 0..1 to display light in cd/m2.
fn pq_to_nits(signal: f64) -> f64 {
    let encoded = powf(signal.clamp(0.0, 1.0), 1.0 / PQ_M2);
    let numerator = (encoded - PQ_C1).max(0.0);
    // The denominator only vanishes past signal 1, which the clamp above cuts.
    powf(numerator / (PQ_C2 - PQ_C3 * encoded), 1.0 / PQ_M1) * PQ_PEAK_NITS
}

/// PQ's inverse EOTF: display light in cd/m2 to signal in 0..1.
fn nits_to_pq(nits: f64) -> f64 {
    let light = powf((nits / PQ_PEAK_NITS).clamp(0.0, 1.0), PQ_M1);
    powf((PQ_C1 + PQ_C2 * light) / (1.0 + PQ_C3 * light), PQ_M2)
}

/// HLG's OETF constants (ARIB STD-B67 / BT.2100).
const HLG_A: f64 = 0.178_832_77;
const HLG_B: f64 = 0.284_668_92;
const HLG_C: f64 = 0.559_910_73;
/// Where HLG's OETF changes from the square-root part to the logarithmic one.
const HLG_OETF_BREAK: f64 = 0.5;
/// The display HLG's OOTF is defined against, and so the peak an HLG source
/// carries.
const HLG_DISPLAY_PEAK_NITS: f64 = 1000.0;
/// BT.2100's system gamma for that display.
const HLG_SYSTEM_GAMMA: f64 = 1.2;

/// HLG's inverse OETF: signal in 0..1 to scene light in 0..1, before the OOTF.
fn hlg_to_scene(signal: f64) -> f64 {
    let signal = signal.clamp(0.0, 1.0);
    match signal <= HLG_OETF_BREAK {
        true => signal * signal / 3.0,
        false => (exp((signal - HLG_C) / HLG_A) + HLG_B) / 12.0,
    }
}

/// The BT.2020 luma weights HLG's OOTF measures scene luminance with, which is
/// why the OOTF is one gain for the pixel and not a per-channel curve.
fn scene_luminance(channels: [f32; 3]) -> f64 {
    let weights = LumaCoefficients::BT2020_NCL;
    f64::from(weights.kr * channels[0] + weights.kg() * channels[1] + weights.kb * channels[2])
}

/// The gain HLG's OOTF puts on all three channels to take the pixel from scene
/// light to display light relative to reference white.
fn hlg_ootf_gain(scene: [f32; 3]) -> f32 {
    let luminance = powf(scene_luminance(scene), HLG_SYSTEM_GAMMA - 1.0);
    (luminance * HLG_DISPLAY_PEAK_NITS / REFERENCE_WHITE_NITS) as f32
}

/// The inverse of [`hlg_ootf_gain`]. The display luminance fixes the scene
/// luminance, since the OOTF raises the latter to `HLG_SYSTEM_GAMMA`.
fn hlg_inverse_ootf_gain(display: [f32; 3]) -> f32 {
    let display_nits = scene_luminance(display) * REFERENCE_WHITE_NITS;
    if display_nits <= 0.0 {
        return 0.0;
    }
    let luminance = powf(display_nits / HLG_DISPLAY_PEAK_NITS, 1.0 / HLG_SYSTEM_GAMMA);
    (REFERENCE_WHITE_NITS / (HLG_DISPLAY_PEAK_NITS * powf(luminance, HLG_SYSTEM_GAMMA - 1.0)))
        as f32
}

/// The BT.2390 EETF that brings an HDR source's light into the SDR range, as
/// its fixed part: the source peak as a PQ signal, the reference white the top
/// of the source range lands on, and where the roll-off starts. Source black is
/// 0, so BT.2390's black-lift term is zero and left out.
#[derive(Debug)]
struct ToneMap {
    peak_signal: f64,
    maximum_luminance: f64,
    knee_start: f64,
}

impl ToneMap {
    /// `None` when the source peaks at reference white or below: the knee then
    /// sits at the top of the range, there is nothing to compress, and the
    /// spline's `1 - knee_start` would divide by zero.
    fn new(peak_nits: f64) -> Option<Self> {
        let peak_signal = nits_to_pq(peak_nits);
        let maximum_luminance = nits_to_pq(REFERENCE_WHITE_NITS) / peak_signal;
        let knee_start = 1.5 * maximum_luminance - 0.5;
        match knee_start >= 1.0 {
            true => None,
            false => Some(Self {
                peak_signal,
                maximum_luminance,
                knee_start,
            }),
        }
    }

    /// The gain that puts the pixel's brightest channel on its tone-mapped
    /// light, for all three channels, so the hue stays where it was.
    fn gain(&self, linear: [f32; 3]) -> f32 {
        let brightest = linear[0].max(linear[1]).max(linear[2]);
        if brightest <= 0.0 {
            return 1.0;
        }
        (self.map(f64::from(brightest)) / f64::from(brightest)) as f32
    }

    /// One reference-white-relative light through the EETF.
    fn map(&self, linear: f64) -> f64 {
        let signal = nits_to_pq(linear * REFERENCE_WHITE_NITS) / self.peak_signal;
        if signal < self.knee_start {
            return linear;
        }
        // Hermite across the roll-off: slope 1 where it leaves the straight
        // part, flat where it reaches the target white. Light past the declared
        // peak lands on that white rather than extrapolating above it.
        let t = ((signal - self.knee_start) / (1.0 - self.knee_start)).min(1.0);
        let (square, cube) = (t * t, t * t * t);
        let rolled = (2.0 * cube - 3.0 * square + 1.0) * self.knee_start
            + (cube - 2.0 * square + t) * (1.0 - self.knee_start)
            + (-2.0 * cube + 3.0 * square) * self.maximum_luminance;
        pq_to_nits(rolled * self.peak_signal) / REFERENCE_WHITE_NITS
    }
}

fn scale(channels: &mut [f32; 3], gain: f32) {
    for channel in channels.iter_mut() {
        *channel *= gain;
    }
}

/// The transfers this converts between: the SDR shape BT.709 and sRGB share,
/// PQ, and HLG.
enum Transfer {
    Sdr(TransferCurve),
    Pq,
    Hlg,
}

impl Transfer {
    /// Nonlinear code value in 0..1 to the light a table holds: display light
    /// relative to reference white, except for HLG, whose table stops at the
    /// scene light the signal carries.
    fn to_linear(&self, value: f64) -> f64 {
        match self {
            Transfer::Sdr(curve) => curve.to_linear(value),
            Transfer::Pq => pq_to_nits(value.clamp(0.0, 1.0)) / REFERENCE_WHITE_NITS,
            Transfer::Hlg => hlg_to_scene(value),
        }
    }

    fn is_scene_light(&self) -> bool {
        matches!(self, Transfer::Hlg)
    }

    /// The peak light this carries, `None` for an SDR curve. PQ's comes from
    /// the property, since the signal reaches 10000 cd/m2 but a grade does not;
    /// HLG's is the display its OOTF is defined against.
    fn peak_nits(&self, pq_peak: f64) -> Option<f64> {
        match self {
            Transfer::Sdr(_) => None,
            Transfer::Pq => Some(pq_peak),
            Transfer::Hlg => Some(HLG_DISPLAY_PEAK_NITS),
        }
    }
}

/// The curve a transfer names, or `None` for an untagged stream, which has none
/// to linearize with.
fn transfer_curve(transfer: TransferCharacteristics) -> Option<Transfer> {
    match transfer {
        TransferCharacteristics::Srgb => Some(Transfer::Sdr(SRGB_CURVE)),
        TransferCharacteristics::Bt601
        | TransferCharacteristics::Bt709
        | TransferCharacteristics::Bt2020 => Some(Transfer::Sdr(BT709_CURVE)),
        TransferCharacteristics::Pq => Some(Transfer::Pq),
        TransferCharacteristics::Hlg => Some(Transfer::Hlg),
        _ => None,
    }
}

/// The linear light of every 8-bit code under `curve`.
fn linear_table(curve: &Transfer) -> [f32; CODE_COUNT] {
    let mut table = [0f32; CODE_COUNT];
    for (code, slot) in table.iter_mut().enumerate() {
        *slot = curve.to_linear(code as f64 / SAMPLE_MAX as f64) as f32;
    }
    table
}

/// CIE xy chromaticities of a primary set and its white point.
struct Chromaticities {
    red: [f64; 2],
    green: [f64; 2],
    blue: [f64; 2],
    white: [f64; 2],
}

/// D65, the white point of every primary set modelled here, which is why no
/// conversion needs a chromatic adaptation.
const D65: [f64; 2] = [0.3127, 0.3290];

fn chromaticities(primaries: ColorPrimaries) -> Option<Chromaticities> {
    let set = match primaries {
        // BT.709, also sRGB's.
        ColorPrimaries::Bt709 => ([0.640, 0.330], [0.300, 0.600], [0.150, 0.060]),
        // BT.470BG, the 625-line EBU set.
        ColorPrimaries::Bt470bg => ([0.640, 0.330], [0.290, 0.600], [0.150, 0.060]),
        // SMPTE 170M, the 525-line SMPTE C set.
        ColorPrimaries::Smpte170m => ([0.630, 0.340], [0.310, 0.595], [0.155, 0.070]),
        ColorPrimaries::Bt2020 => ([0.708, 0.292], [0.170, 0.797], [0.131, 0.046]),
        _ => return None,
    };
    Some(Chromaticities {
        red: set.0,
        green: set.1,
        blue: set.2,
        white: D65,
    })
}

/// Linear RGB -> CIE XYZ for one primary set, the standard construction: the
/// primaries as XYZ columns, scaled so that RGB (1, 1, 1) lands on the white
/// point.
fn rgb_to_xyz(set: &Chromaticities) -> Option<[[f64; 3]; 3]> {
    let xyz = |chromaticity: [f64; 2]| {
        let [x, y] = chromaticity;
        [x / y, 1.0, (1.0 - x - y) / y]
    };
    let (red, green, blue) = (xyz(set.red), xyz(set.green), xyz(set.blue));
    let primaries = [
        [red[0], green[0], blue[0]],
        [red[1], green[1], blue[1]],
        [red[2], green[2], blue[2]],
    ];
    let white = xyz(set.white);
    let scale = multiply_vector(&invert(&primaries)?, white);
    Some([
        [
            primaries[0][0] * scale[0],
            primaries[0][1] * scale[1],
            primaries[0][2] * scale[2],
        ],
        [
            primaries[1][0] * scale[0],
            primaries[1][1] * scale[1],
            primaries[1][2] * scale[2],
        ],
        [
            primaries[2][0] * scale[0],
            primaries[2][1] * scale[1],
            primaries[2][2] * scale[2],
        ],
    ])
}

/// Source linear RGB -> target linear RGB. The identity when the two sets match
/// or either is untagged, since there is then nothing to map between.
fn gamut_matrix(source: ColorPrimaries, target: ColorPrimaries) -> Result<[[f32; 3]; 3], G2gError> {
    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    if source == target {
        return Ok(IDENTITY);
    }
    let (Some(source), Some(target)) = (chromaticities(source), chromaticities(target)) else {
        return Ok(IDENTITY);
    };
    let forward = rgb_to_xyz(&source).ok_or(G2gError::CapsMismatch)?;
    let backward = invert(&rgb_to_xyz(&target).ok_or(G2gError::CapsMismatch)?)
        .ok_or(G2gError::CapsMismatch)?;
    let mut out = [[0f32; 3]; 3];
    for (row, slot) in out.iter_mut().enumerate() {
        for (column, cell) in slot.iter_mut().enumerate() {
            *cell = (0..3)
                .map(|k| backward[row][k] * forward[k][column])
                .sum::<f64>() as f32;
        }
    }
    Ok(out)
}

fn multiply_vector(matrix: &[[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    let row =
        |r: usize| matrix[r][0] * vector[0] + matrix[r][1] * vector[1] + matrix[r][2] * vector[2];
    [row(0), row(1), row(2)]
}

/// 3x3 inverse by the adjugate; `None` for a singular matrix, which a valid
/// primary set never produces.
fn invert(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let cofactor = |r: usize, c: usize| {
        let rows: [usize; 2] = [(r + 1) % 3, (r + 2) % 3];
        let columns: [usize; 2] = [(c + 1) % 3, (c + 2) % 3];
        m[rows[0]][columns[0]] * m[rows[1]][columns[1]]
            - m[rows[0]][columns[1]] * m[rows[1]][columns[0]]
    };
    let determinant =
        m[0][0] * cofactor(0, 0) + m[0][1] * cofactor(0, 1) + m[0][2] * cofactor(0, 2);
    if determinant == 0.0 {
        return None;
    }
    let mut out = [[0f64; 3]; 3];
    for (row, slot) in out.iter_mut().enumerate() {
        for (column, cell) in slot.iter_mut().enumerate() {
            // Transposed: the inverse is the adjugate (cofactor transpose) over
            // the determinant.
            *cell = cofactor(column, row) / determinant;
        }
    }
    Some(out)
}

/// Convert one frame in place of its own format.
fn convert_frame(
    src: &[u8],
    format: RawVideoFormat,
    width: usize,
    height: usize,
    transform: &ColorTransform,
) -> Box<[u8]> {
    match format {
        RawVideoFormat::I420 => convert_yuv420(src, width, height, false, transform),
        RawVideoFormat::Nv12 => convert_yuv420(src, width, height, true, transform),
        RawVideoFormat::Rgb8 => convert_packed_rgb(src, width, height, 3, 0, 2, transform),
        format => {
            let (red, blue) = rgba_rb_offsets(format);
            convert_packed_rgb(src, width, height, 4, red, blue, transform)
        }
    }
}

/// 4:2:0 YUV, one chroma cell at a time: each of the block's four luma samples
/// is converted on its own, and the block's chroma comes from the mean of the
/// four converted RGB values, the same 2x2 box filter `videoconvert` writes its
/// 4:2:0 chroma with. `interleaved` picks NV12 over I420.
fn convert_yuv420(
    src: &[u8],
    width: usize,
    height: usize,
    interleaved: bool,
    transform: &ColorTransform,
) -> Box<[u8]> {
    let luma = width * height;
    let mut dst = vec![0u8; luma + luma / 2];
    let (chroma_width, chroma_height) = (width / 2, height / 2);
    for cy in 0..chroma_height {
        for cx in 0..chroma_width {
            let cell = cy * chroma_width + cx;
            let (u, v) = match interleaved {
                true => (src[luma + 2 * cell] as i32, src[luma + 2 * cell + 1] as i32),
                false => (src[luma + cell] as i32, src[luma + luma / 4 + cell] as i32),
            };
            let (mut sum_r, mut sum_g, mut sum_b) = (0i32, 0i32, 0i32);
            for dy in 0..2 {
                for dx in 0..2 {
                    let index = (cy * 2 + dy) * width + cx * 2 + dx;
                    let (r, g, b) = transform.yuv_to_target_rgb(src[index] as i32, u, v);
                    dst[index] = transform.encode.rgb_to_yuv(r, g, b).0 as u8;
                    sum_r += r;
                    sum_g += g;
                    sum_b += b;
                }
            }
            let (_, u, v) = transform.encode.rgb_to_yuv(sum_r / 4, sum_g / 4, sum_b / 4);
            match interleaved {
                true => {
                    dst[luma + 2 * cell] = u as u8;
                    dst[luma + 2 * cell + 1] = v as u8;
                }
                false => {
                    dst[luma + cell] = u as u8;
                    dst[luma + luma / 4 + cell] = v as u8;
                }
            }
        }
    }
    dst.into_boxed_slice()
}

/// Packed RGB(A), pixel by pixel. Alpha and any other byte of the pixel ride
/// through untouched.
fn convert_packed_rgb(
    src: &[u8],
    width: usize,
    height: usize,
    bytes_per_pixel: usize,
    red: usize,
    blue: usize,
    transform: &ColorTransform,
) -> Box<[u8]> {
    let mut dst = src[..width * height * bytes_per_pixel].to_vec();
    for pixel in dst.chunks_exact_mut(bytes_per_pixel) {
        let (r, g, b) =
            transform.rgb_to_target_rgb(pixel[red] as i32, pixel[1] as i32, pixel[blue] as i32);
        pixel[red] = r as u8;
        pixel[1] = g as u8;
        pixel[blue] = b as u8;
    }
    dst.into_boxed_slice()
}

impl AsyncElement for Colorspace {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// The conversion reads and writes host memory, so it takes system frames
    /// only; the allocation cascade turns that into a download demand on a GPU
    /// producer.
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::SystemView)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        for format in FORMATS {
            let candidate = Caps::RawVideo {
                format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: Colorimetry::UNKNOWN,
            };
            if let Ok(narrowed) = upstream_caps.intersect(&candidate) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    /// The format and geometry pass through, so every field the solver knows
    /// how to couple couples backward. The colorimetry the derivation leaves
    /// unknown is what lets a downstream capsfilter name the target.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedFields(CapsTransform::RawVideo {
            accept: FORMATS.to_vec(),
            produce: FORMATS.to_vec(),
            shapes: vec![RawVideoShape::PASSTHROUGH],
        })
    }

    // A colorimetry-only convert leaves geometry alone, so normalized meta
    // rides through unchanged.
    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<g2g_core::meta::Transform> {
        Some(g2g_core::meta::Transform::Copy)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input = Some(Self::accept_input(absolute_caps)?);
        self.rebuild_plan()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Take the target from the negotiated output caps, and refuse a target the
    /// element cannot deliver: one that contradicts the `colorimetry` property,
    /// or a transfer change away from an untagged input.
    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let Caps::RawVideo {
            format,
            colorimetry,
            ..
        } = output_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !FORMATS.contains(format) {
            return Err(G2gError::CapsMismatch);
        }
        self.negotiated = *colorimetry;
        self.rebuild_plan()?;
        // What this emits has to be something downstream accepts: a pinned
        // colorimetry the property contradicts, or one no conversion reaches,
        // fails here rather than mid-stream.
        self.target
            .intersect(colorimetry)
            .ok_or(G2gError::CapsMismatch)?;
        Ok(())
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let InputStream {
                        format,
                        width,
                        height,
                        framerate,
                        ..
                    } = self.input.clone().ok_or(G2gError::NotConfigured)?;
                    let bytes = frame
                        .domain
                        .require_system_bytes(g2g_core::log::short_type_name::<Self>())?;
                    let src: &[u8] = &bytes;
                    let needed = frame_byte_size(format, width, height);
                    if src.len() < needed {
                        return Err(G2gError::CapsMismatch);
                    }
                    let converted: Box<[u8]> = match self.plan.as_ref() {
                        Some(Plan::Convert(transform)) => {
                            convert_frame(src, format, width as usize, height as usize, transform)
                        }
                        Some(Plan::Passthrough) => src[..needed].into(),
                        None => return Err(G2gError::NotConfigured),
                    };

                    let new_caps = Caps::RawVideo {
                        format,
                        width: Dim::Fixed(width),
                        height: Dim::Fixed(height),
                        framerate,
                        interlace: g2g_core::Interlace::Any,
                        colorimetry: self.target,
                    };
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
                        self.last_caps = Some(new_caps);
                    }
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(converted)),
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    // The runner's transform arm calls `configure_pipeline`
                    // (input) then `configure_output` (output) immediately
                    // before pushing this, whose caps is the arm's pre-fixed
                    // forward *output*, not a new input. Forward it and record
                    // it so the data path does not repeat it. Do not adopt it
                    // as the input: both sides are `Caps::RawVideo`, so only
                    // that ordering tells them apart.
                    out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                    self.last_caps = Some(caps);
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(segment) => {
                    out.push(PipelinePacket::Segment(segment)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        COLORSPACE_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Video colorimetry converter",
            "Filter/Converter/Video",
            "Converts the matrix, range, transfer and primaries of raw video, keeping the pixel format",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "colorimetry" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                self.forced = Some(Colorimetry::from_gst_string(text).ok_or(PropError::Value)?);
                self.rebuild_plan().map_err(|_| PropError::Value)?;
                Ok(())
            }
            "hdr-peak-nits" => {
                let nits = value.as_uint().ok_or(PropError::Type)?;
                let range = u64::from(MINIMUM_HDR_PEAK_NITS)..=u64::from(MAXIMUM_HDR_PEAK_NITS);
                if !range.contains(&nits) {
                    return Err(PropError::Value);
                }
                self.hdr_peak_nits = nits as u32;
                self.rebuild_plan().map_err(|_| PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "hdr-peak-nits" => Some(PropValue::Uint(u64::from(self.hdr_peak_nits))),
            // The effective output colorimetry: the property when set, else
            // what negotiation resolved. `None` before either says anything.
            "colorimetry" => self
                .forced
                .unwrap_or(self.target)
                .to_gst_string()
                .map(PropValue::Str),
            _ => None,
        }
    }
}

/// `Colorspace`'s settable properties: the output colorimetry, and the source
/// peak the tone map works from.
static COLORSPACE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "colorimetry",
        PropKind::Str,
        "output colorimetry: bt709 | bt601 | bt2020 | sRGB | range:matrix:transfer:primaries; unset takes it from the negotiated caps",
    ),
    PropertySpec::new(
        "hdr-peak-nits",
        PropKind::Uint,
        "peak light in cd/m2 the PQ source is graded to, which the tone map compresses down to SDR white; applies to a PQ source only, an HLG one peaks at its 1000 cd/m2 display",
    )
    .with_range("203", "10000")
    .with_default("1000"),
];

impl PadTemplates for Colorspace {
    /// Static superset: any convertible format at any geometry, the same set on
    /// both pads, since the format never changes.
    fn pad_templates() -> Vec<PadTemplate> {
        let any_geometry = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
        };
        let set = CapsSet::from_alternatives(FORMATS.map(any_geometry).to_vec());
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn colorimetry(matrix: MatrixCoefficients, range: ColorRange) -> Colorimetry {
        Colorimetry {
            matrix,
            range,
            ..Colorimetry::UNKNOWN
        }
    }

    /// The light transform of a pair that moves, at the default source peak.
    fn light(source: Colorimetry, target: Colorimetry) -> Box<LightTransform> {
        LightTransform::build(source, target, f64::from(DEFAULT_HDR_PEAK_NITS))
            .expect("the pair converts")
            .expect("the transfer or primaries move")
    }

    /// The primaries construction has to reproduce the published BT.709 ->
    /// XYZ matrix, or every gamut conversion is off.
    #[test]
    fn bt709_rgb_to_xyz_matches_the_published_matrix() {
        let m = rgb_to_xyz(&chromaticities(ColorPrimaries::Bt709).unwrap()).unwrap();
        let expected = [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ];
        for row in 0..3 {
            for column in 0..3 {
                assert!(
                    (m[row][column] - expected[row][column]).abs() < 5e-4,
                    "row {row} column {column}: {} vs {}",
                    m[row][column],
                    expected[row][column]
                );
            }
        }
        // Its middle row is the BT.709 luma weights, which is the same fact the
        // YUV matrix is built from.
        assert!((m[1][0] - f64::from(g2g_core::LumaCoefficients::BT709.kr)).abs() < 5e-4);
        assert!((m[1][2] - f64::from(g2g_core::LumaCoefficients::BT709.kb)).abs() < 5e-4);
    }

    /// A gamut change keeps neutral neutral: both sets are D65, so equal linear
    /// R, G, B stay equal.
    #[test]
    fn a_gamut_change_leaves_white_alone() {
        let gamut = gamut_matrix(ColorPrimaries::Bt709, ColorPrimaries::Bt2020).unwrap();
        for row in gamut {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row sums to {sum}, not 1");
        }
    }

    /// The BT.709 primaries sit inside BT.2020, so a saturated BT.709 red needs
    /// less than full red in the wider set, with the shortfall showing up in
    /// the other two channels.
    #[test]
    fn a_saturated_red_narrows_into_the_wider_gamut() {
        let transform = ColorTransform {
            decode: YuvRgbMatrix::new(Colorimetry::BT709),
            encode: YuvRgbMatrix::new(Colorimetry::BT2020),
            light: Some(light(Colorimetry::BT709, Colorimetry::BT2020)),
        };
        let (r, g, b) = transform.rgb_to_target_rgb(255, 0, 0);
        assert!(r < 255 && r > 200, "red stays dominant but drops: {r}");
        assert!(g > 0 && b > 0, "the 709 red is a mix in 2020: ({g}, {b})");
    }

    /// A transfer-only change preserves linear light: that is the whole point
    /// of going through it.
    #[test]
    fn a_transfer_change_preserves_linear_light() {
        let bt709 = Colorimetry::BT709;
        let srgb = Colorimetry {
            transfer: TransferCharacteristics::Srgb,
            ..Colorimetry::BT709
        };
        let converter = light(bt709, srgb);
        for code in [1i32, 20, 60, 128, 200, 254] {
            let (converted, _, _) = converter.apply(code, code, code);
            let before = BT709_CURVE.to_linear(f64::from(code) / f64::from(SAMPLE_MAX));
            let after = SRGB_CURVE.to_linear(f64::from(converted) / f64::from(SAMPLE_MAX));
            assert!(
                (before - after).abs() < 4e-3,
                "code {code} -> {converted}: linear {before} vs {after}"
            );
        }
    }

    /// Both HDR curves convert, and only the direction that has light above the
    /// SDR range tone maps.
    #[test]
    fn an_hdr_source_tone_maps_and_an_hdr_target_does_not() {
        assert!(light(Colorimetry::BT2100_PQ, Colorimetry::BT709)
            .tone_map
            .is_some());
        assert!(light(Colorimetry::BT2100_HLG, Colorimetry::BT709)
            .tone_map
            .is_some());
        assert!(light(Colorimetry::BT709, Colorimetry::BT2100_PQ)
            .tone_map
            .is_none());
        // A matrix-only change under an unchanged PQ transfer needs no linear
        // light at all.
        let pq_full = Colorimetry {
            range: ColorRange::Full,
            ..Colorimetry::BT2100_PQ
        };
        assert!(matches!(
            LightTransform::build(
                Colorimetry::BT2100_PQ,
                pq_full,
                f64::from(DEFAULT_HDR_PEAK_NITS)
            ),
            Ok(None)
        ));
    }

    /// PQ's EOTF and its inverse are each other's, which is what both the code
    /// tables and the tone map's PQ domain rest on.
    #[test]
    fn the_pq_curve_inverts_itself() {
        for nits in [0.1f64, 1.0, REFERENCE_WHITE_NITS, 600.0, PQ_PEAK_NITS] {
            let recovered = pq_to_nits(nits_to_pq(nits));
            assert!(
                (recovered - nits).abs() < 1e-3 * nits,
                "{nits} cd/m2 came back as {recovered}"
            );
        }
    }

    /// A source that peaks at reference white has nothing above the SDR range,
    /// so there is no tone map to build. One that peaks higher is the identity
    /// below its knee and lands exactly on white at its peak.
    #[test]
    fn the_tone_map_spans_the_source_peak_to_reference_white() {
        assert!(ToneMap::new(f64::from(MINIMUM_HDR_PEAK_NITS)).is_none());
        let tone_map =
            ToneMap::new(f64::from(DEFAULT_HDR_PEAK_NITS)).expect("1000 cd/m2 compresses");
        assert_eq!(tone_map.map(0.0), 0.0);
        let peak = f64::from(DEFAULT_HDR_PEAK_NITS) / REFERENCE_WHITE_NITS;
        assert!(
            (tone_map.map(peak) - 1.0).abs() < 1e-3,
            "{}",
            tone_map.map(peak)
        );
        // Monotone, or a brighter source pixel would come out darker.
        let mut previous = 0.0;
        for step in 0..64 {
            let mapped = tone_map.map(peak * f64::from(step) / 63.0);
            assert!(mapped >= previous, "step {step}: {mapped} after {previous}");
            previous = mapped;
        }
    }

    /// An untagged transfer cannot be linearized, so it is not relabelled: the
    /// matrix and range still convert, because an unknown one of those resolves
    /// to BT.601 limited.
    #[test]
    fn an_untagged_stream_keeps_its_untagged_transfer_and_primaries() {
        let target = achievable(Colorimetry::UNKNOWN, Colorimetry::BT709);
        assert_eq!(
            target,
            colorimetry(MatrixCoefficients::Bt709, ColorRange::Limited)
        );
    }

    /// An RGB layout carries no matrix and no range, so those two never reach
    /// the output caps and a request that only moves them converts nothing.
    #[test]
    fn an_rgb_target_drops_the_matrix_and_range() {
        let dropped = carried(RawVideoFormat::Rgba8, Colorimetry::BT709);
        assert_eq!(dropped.matrix, MatrixCoefficients::Unknown);
        assert_eq!(dropped.range, ColorRange::Unknown);
        assert_eq!(dropped.transfer, Colorimetry::BT709.transfer);
        let source = carried(RawVideoFormat::Rgba8, Colorimetry::BT601);
        assert!(matches!(
            Plan::build(source, dropped, f64::from(DEFAULT_HDR_PEAK_NITS)).unwrap(),
            Plan::Convert(_)
        ));
    }

    #[test]
    fn equal_colorimetry_is_a_passthrough() {
        assert!(matches!(
            Plan::build(
                Colorimetry::BT709,
                Colorimetry::BT709,
                f64::from(DEFAULT_HDR_PEAK_NITS)
            )
            .unwrap(),
            Plan::Passthrough
        ));
    }
}
