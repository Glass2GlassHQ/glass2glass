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
//! PQ and HLG are refused: converting either to or from an SDR curve is tone
//! mapping, which this does not do, so such a target fails negotiation instead
//! of producing a wrongly-labelled frame. The transfer and primaries also
//! cannot be converted away from an *untagged* input, since there is no curve
//! to linearize with; the output then stays untagged on those two fields rather
//! than claiming a conversion that did not happen. The matrix and range do
//! convert from an untagged input, because
//! [`Colorimetry::yuv_conversion`](g2g_core::Colorimetry::yuv_conversion)
//! resolves an unknown one to BT.601 limited.
//!
//! CPU-only and `no_std`: the transfer curves run on `mathf`, and each
//! is folded into a 256-entry table at negotiation so no pixel pays a `powf`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::mathf::powf;
use crate::pixel::{carries_yuv, even_dims_required, frame_byte_size, rgba_rb_offsets};
use crate::yuvmatrix::{YuvRgbMatrix, SAMPLE_MAX};
use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, CapsTransform, ColorPrimaries, ColorRange,
    Colorimetry, ConfigureOutcome, Dim, ElementMetadata, G2gError, MatrixCoefficients,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat, RawVideoShape, TransferCharacteristics,
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
        self.plan = Some(Plan::build(source, target)?);
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
    fn build(source: Colorimetry, target: Colorimetry) -> Result<Self, G2gError> {
        if source == target {
            return Ok(Plan::Passthrough);
        }
        Ok(Plan::Convert(ColorTransform {
            decode: YuvRgbMatrix::new(source),
            encode: YuvRgbMatrix::new(target),
            light: LightTransform::build(source, target)?,
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
}

impl core::fmt::Debug for LightTransform {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LightTransform")
            .field("gamut", &self.gamut)
            .finish_non_exhaustive()
    }
}

impl LightTransform {
    /// `None` when the transfer and primaries both stay put. `Err` when the
    /// conversion needs linear light but one end's curve is PQ or HLG, which is
    /// tone mapping.
    fn build(source: Colorimetry, target: Colorimetry) -> Result<Option<Box<Self>>, G2gError> {
        if source.transfer == target.transfer && source.primaries == target.primaries {
            return Ok(None);
        }
        let (Some(source_curve), Some(target_curve)) = (
            transfer_curve(source.transfer),
            transfer_curve(target.transfer),
        ) else {
            return Err(G2gError::CapsMismatch);
        };
        Ok(Some(Box::new(Self {
            source_linear: linear_table(&source_curve),
            target_linear: linear_table(&target_curve),
            gamut: gamut_matrix(source.primaries, target.primaries)?,
        })))
    }

    /// Source 8-bit RGB to target 8-bit RGB through linear light.
    fn apply(&self, r: i32, g: i32, b: i32) -> (i32, i32, i32) {
        let code = |v: i32| self.source_linear[v.clamp(0, SAMPLE_MAX) as usize];
        let linear = [code(r), code(g), code(b)];
        let mixed = |row: [f32; 3]| row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2];
        (
            self.encode(mixed(self.gamut[0])),
            self.encode(mixed(self.gamut[1])),
            self.encode(mixed(self.gamut[2])),
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

/// The curve a transfer names, or `None` for one this cannot linearize: PQ and
/// HLG (tone mapping) and an untagged stream.
fn transfer_curve(transfer: TransferCharacteristics) -> Option<TransferCurve> {
    match transfer {
        TransferCharacteristics::Srgb => Some(SRGB_CURVE),
        TransferCharacteristics::Bt601
        | TransferCharacteristics::Bt709
        | TransferCharacteristics::Bt2020 => Some(BT709_CURVE),
        _ => None,
    }
}

/// The linear light of every 8-bit code under `curve`.
fn linear_table(curve: &TransferCurve) -> [f32; CODE_COUNT] {
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
    /// element cannot deliver: a PQ / HLG conversion, or one that contradicts
    /// the `colorimetry` property.
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
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
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

/// `Colorspace`'s settable properties: the output colorimetry.
static COLORSPACE_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "colorimetry",
    PropKind::Str,
    "output colorimetry: bt709 | bt601 | bt2020 | sRGB | range:matrix:transfer:primaries; unset takes it from the negotiated caps",
)];

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
            light: Some(
                LightTransform::build(Colorimetry::BT709, Colorimetry::BT2020)
                    .unwrap()
                    .expect("709 and 2020 differ in primaries"),
            ),
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
        let light = LightTransform::build(bt709, srgb)
            .unwrap()
            .expect("the curves differ");
        for code in [1i32, 20, 60, 128, 200, 254] {
            let (converted, _, _) = light.apply(code, code, code);
            let before = BT709_CURVE.to_linear(f64::from(code) / f64::from(SAMPLE_MAX));
            let after = SRGB_CURVE.to_linear(f64::from(converted) / f64::from(SAMPLE_MAX));
            assert!(
                (before - after).abs() < 4e-3,
                "code {code} -> {converted}: linear {before} vs {after}"
            );
        }
    }

    /// Tone mapping is out of scope, so a PQ end fails to build rather than
    /// producing an SDR-looking frame labelled HDR.
    #[test]
    fn a_pq_conversion_is_refused() {
        assert_eq!(
            LightTransform::build(Colorimetry::BT709, Colorimetry::BT2100_PQ).err(),
            Some(G2gError::CapsMismatch)
        );
        assert_eq!(
            LightTransform::build(Colorimetry::BT2100_HLG, Colorimetry::BT709).err(),
            Some(G2gError::CapsMismatch)
        );
        // A matrix-only change under an unchanged PQ transfer needs no linear
        // light, so it is allowed.
        let pq_full = Colorimetry {
            range: ColorRange::Full,
            ..Colorimetry::BT2100_PQ
        };
        assert!(matches!(
            LightTransform::build(Colorimetry::BT2100_PQ, pq_full),
            Ok(None)
        ));
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
            Plan::build(source, dropped).unwrap(),
            Plan::Convert(_)
        ));
    }

    #[test]
    fn equal_colorimetry_is_a_passthrough() {
        assert!(matches!(
            Plan::build(Colorimetry::BT709, Colorimetry::BT709).unwrap(),
            Plan::Passthrough
        ));
    }
}
