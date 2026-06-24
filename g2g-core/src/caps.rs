use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::G2gError;

/// Sentinel sample rate meaning "any / unspecified" in [`Caps::Audio`] (M187).
/// `Caps::Audio.sample_rate` is a bare `u32` (not a ranged [`Dim`]); 0 Hz is
/// never a real rate, so it serves as the wildcard a caps-driven element
/// (`audioresample`) advertises so a downstream capsfilter can pin the rate.
/// `intersect` treats it as a wildcard and `fixate` rejects it (like
/// [`Dim::Any`]).
pub const ANY_SAMPLE_RATE: u32 = 0;

/// Caps describes one fixated (or partially-narrowed) link.
///
/// Video is split into [`Caps::CompressedVideo`] and [`Caps::RawVideo`]
/// because a codec bitstream and a raw pixel buffer are *different
/// kinds* of media, not different values of one "format" slot. A raw
/// sink (waylandsink, kmssink) intercepting a `CompressedVideo` caps is
/// a category error; the type system now expresses that as a variant
/// mismatch rather than a runtime enum compare. (Mirrors GStreamer's
/// `video/x-h264` vs `video/x-raw` distinction; M17 split.)
///
/// Both video variants carry geometry today. That's pragmatic, not
/// honest: GStreamer's `video/x-h264` doesn't have width/height because
/// they live in the SPS. Our solver, the RtspSrc placeholder Range, and
/// our `Range`-as-placeholder convention all hang off geometry on
/// compressed caps. Dropping it is a deeper rework that overlaps
/// workaround #1's redesign; out of scope here.
#[derive(Clone, Debug, PartialEq)]
pub enum Caps {
    /// Compressed video bitstream (codec). Width/height/framerate are
    /// nominal until the bitstream parser confirms them via SPS/equivalent.
    CompressedVideo {
        codec: VideoCodec,
        width: Dim,
        height: Dim,
        framerate: Rate,
    },
    /// Raw pixel buffer in `format`. Geometry is authoritative.
    RawVideo {
        format: RawVideoFormat,
        width: Dim,
        height: Dim,
        framerate: Rate,
    },
    Audio {
        format: AudioFormat,
        channels: u8,
        sample_rate: u32,
    },
    Tensor {
        dtype: TensorDType,
        shape: TensorShape,
        layout: TensorLayout,
    },
    /// An opaque container / elementary byte stream, not yet demuxed or parsed
    /// into a typed media stream. The link type between a byte source (a file or
    /// network source carrying e.g. an MPEG-TS) and a demuxer that splits it into
    /// elementary streams. `encoding` names the wire format so a demuxer only
    /// accepts a stream it understands.
    ByteStream {
        encoding: ByteStreamEncoding,
    },
}

impl Caps {
    /// Phase 1 intersection (DESIGN.md §4.2). Narrow `self` against `other`,
    /// returning the overlap. Both must be the same variant; ranged fields
    /// (`Dim`/`Rate`) intersect field-wise, scalar fields (`codec` /
    /// `format`, `channels`, `sample_rate`, tensor dtype/shape/layout) must
    /// be equal. Any empty field overlap, variant mismatch, or scalar
    /// mismatch yields `CapsMismatch`.
    ///
    /// `CompressedVideo` and `RawVideo` are distinct variants — a raw
    /// sink offered compressed input gets `CapsMismatch` structurally,
    /// not a runtime format compare.
    pub fn intersect(&self, other: &Caps) -> Result<Caps, G2gError> {
        match (self, other) {
            (
                Caps::CompressedVideo { codec: ca, width: wa, height: ha, framerate: ra },
                Caps::CompressedVideo { codec: cb, width: wb, height: hb, framerate: rb },
            ) if ca == cb => Ok(Caps::CompressedVideo {
                codec: *ca,
                width: wa.intersect(wb).ok_or(G2gError::CapsMismatch)?,
                height: ha.intersect(hb).ok_or(G2gError::CapsMismatch)?,
                framerate: ra.intersect(rb).ok_or(G2gError::CapsMismatch)?,
            }),
            (
                Caps::RawVideo { format: fa, width: wa, height: ha, framerate: ra },
                Caps::RawVideo { format: fb, width: wb, height: hb, framerate: rb },
            ) if fa == fb => Ok(Caps::RawVideo {
                format: *fa,
                width: wa.intersect(wb).ok_or(G2gError::CapsMismatch)?,
                height: ha.intersect(hb).ok_or(G2gError::CapsMismatch)?,
                framerate: ra.intersect(rb).ok_or(G2gError::CapsMismatch)?,
            }),
            (
                Caps::Audio { format: fa, channels: ca, sample_rate: sa },
                Caps::Audio { format: fb, channels: cb, sample_rate: sb },
            ) if fa == fb && ca == cb => {
                // The "any rate" wildcard (M187) is a raw-PCM concept: a
                // caps-driven resampler leaves its output rate open. Compressed
                // audio (AAC/Opus) uses `sample_rate: 0` as "unknown until
                // parsed" and keeps strict equality, unchanged.
                let rate = if is_pcm(*fa) {
                    intersect_sample_rate(*sa, *sb)
                } else {
                    (sa == sb).then_some(*sa)
                };
                match rate {
                    Some(sample_rate) => Ok(Caps::Audio { format: *fa, channels: *ca, sample_rate }),
                    None => Err(G2gError::CapsMismatch),
                }
            }
            (
                Caps::Tensor { dtype: da, shape: sha, layout: la },
                Caps::Tensor { dtype: db, shape: shb, layout: lb },
            ) if da == db && sha == shb && la == lb => Ok(self.clone()),
            (Caps::ByteStream { encoding: ea }, Caps::ByteStream { encoding: eb }) if ea == eb => {
                Ok(self.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// True when every ranged field is `Fixed`. Scalar-only variants are
    /// always fixed.
    pub fn is_fixed(&self) -> bool {
        if let Caps::Audio { format, sample_rate, .. } = self {
            // Only raw PCM uses the "any rate" wildcard (M187); compressed audio
            // keeps `0` as a fixed (if nominal) value.
            return !(is_pcm(*format) && *sample_rate == ANY_SAMPLE_RATE);
        }
        match self.dims() {
            Some((width, height, framerate)) => {
                width.is_fixed() && height.is_fixed() && framerate.is_fixed()
            }
            None => true,
        }
    }

    /// Phase 2 fixation (DESIGN.md §4.2): collapse every ranged field to a
    /// single `Fixed` value. `Range` fixates to its **minimum**, reflecting
    /// the latency-first design (less data is lower latency); an element
    /// preferring a different value counter-proposes via
    /// [`ConfigureOutcome::ReFixate`](crate::element::ConfigureOutcome).
    /// `Any` carries no information to fixate against and yields
    /// `CapsMismatch`.
    pub fn fixate(&self) -> Result<Caps, G2gError> {
        match self {
            Caps::CompressedVideo { codec, width, height, framerate } => {
                Ok(Caps::CompressedVideo {
                    codec: *codec,
                    width: width.fixate().ok_or(G2gError::CapsMismatch)?,
                    height: height.fixate().ok_or(G2gError::CapsMismatch)?,
                    framerate: framerate.fixate().ok_or(G2gError::CapsMismatch)?,
                })
            }
            Caps::RawVideo { format, width, height, framerate } => Ok(Caps::RawVideo {
                format: *format,
                width: width.fixate().ok_or(G2gError::CapsMismatch)?,
                height: height.fixate().ok_or(G2gError::CapsMismatch)?,
                framerate: framerate.fixate().ok_or(G2gError::CapsMismatch)?,
            }),
            // A raw-PCM "any" sample rate carries no value to fixate against
            // (M187); compressed audio's nominal `0` fixates as-is.
            Caps::Audio { format, sample_rate, .. }
                if is_pcm(*format) && *sample_rate == ANY_SAMPLE_RATE =>
            {
                Err(G2gError::CapsMismatch)
            }
            Caps::Audio { .. } | Caps::Tensor { .. } | Caps::ByteStream { .. } => Ok(self.clone()),
        }
    }

    /// Borrow the geometry triple if this caps carries one. Both video
    /// variants (compressed + raw) currently do; `Audio` and `Tensor`
    /// return `None`. Used by element code that needs width/height/fps
    /// without caring whether the link is pre- or post-decode.
    pub fn dims(&self) -> Option<(&Dim, &Dim, &Rate)> {
        match self {
            Caps::CompressedVideo { width, height, framerate, .. }
            | Caps::RawVideo { width, height, framerate, .. } => Some((width, height, framerate)),
            Caps::Audio { .. } | Caps::Tensor { .. } | Caps::ByteStream { .. } => None,
        }
    }

    /// Render these caps as a GStreamer caps string, the inverse of the
    /// `capsfilter` parser (`g2g_plugins::capsfilter::parse_caps`). For `-v`
    /// pipeline dumps, logs, and porting diagnostics. The fixed media types
    /// round-trip through the parser; `Tensor` has no GStreamer media type and
    /// is rendered as a g2g-specific `tensor/x-raw` descriptor.
    pub fn to_gst_string(&self) -> String {
        match self {
            Caps::RawVideo { format, width, height, framerate } => {
                let mut s = format!("video/x-raw,format={}", raw_format_gst_name(*format));
                push_dim(&mut s, "width", width);
                push_dim(&mut s, "height", height);
                push_rate(&mut s, framerate);
                s
            }
            Caps::CompressedVideo { codec, width, height, framerate } => {
                let mut s = String::from(codec_gst_media_type(*codec));
                push_dim(&mut s, "width", width);
                push_dim(&mut s, "height", height);
                push_rate(&mut s, framerate);
                s
            }
            Caps::Audio { format, channels, sample_rate } => {
                let (media_type, fmt) = audio_gst_media_type(*format);
                let mut s = String::from(media_type);
                if let Some(f) = fmt {
                    s.push_str(&format!(",format={f}"));
                }
                if *channels != 0 {
                    s.push_str(&format!(",channels={channels}"));
                }
                if *sample_rate != ANY_SAMPLE_RATE && *sample_rate != 0 {
                    s.push_str(&format!(",rate={sample_rate}"));
                }
                s
            }
            // No GStreamer media type for tensors; a g2g-specific descriptor.
            Caps::Tensor { dtype, shape, layout } => {
                format!("tensor/x-raw,dtype={dtype:?},layout={layout:?},shape={shape:?}")
            }
            Caps::ByteStream { encoding } => String::from(bytestream_gst_media_type(*encoding)),
        }
    }
}

/// The GStreamer `format=` name for a raw video format (uppercase, the M182
/// vocabulary the parser also accepts).
fn raw_format_gst_name(f: RawVideoFormat) -> &'static str {
    match f {
        RawVideoFormat::Nv12 => "NV12",
        RawVideoFormat::I420 => "I420",
        RawVideoFormat::Rgba8 => "RGBA",
        RawVideoFormat::Bgra8 => "BGRA",
        RawVideoFormat::Yuyv => "YUY2",
    }
}

/// The GStreamer media type for a compressed video codec.
fn codec_gst_media_type(c: VideoCodec) -> &'static str {
    match c {
        VideoCodec::H264 => "video/x-h264",
        VideoCodec::H265 => "video/x-h265",
        VideoCodec::Av1 => "video/x-av1",
        VideoCodec::Vp8 => "video/x-vp8",
        VideoCodec::Vp9 => "video/x-vp9",
        VideoCodec::Mjpeg => "image/jpeg",
    }
}

/// The GStreamer media type (and raw `format=` name, if raw) for an audio format.
fn audio_gst_media_type(f: AudioFormat) -> (&'static str, Option<&'static str>) {
    match f {
        AudioFormat::Aac => ("audio/mpeg", None),
        AudioFormat::Opus => ("audio/x-opus", None),
        AudioFormat::PcmS16Le => ("audio/x-raw", Some("S16LE")),
        AudioFormat::PcmF32Le => ("audio/x-raw", Some("F32LE")),
    }
}

/// The GStreamer media type for a container byte stream.
fn bytestream_gst_media_type(e: ByteStreamEncoding) -> &'static str {
    match e {
        ByteStreamEncoding::MpegTs => "video/mpegts",
        ByteStreamEncoding::Matroska => "video/x-matroska",
        ByteStreamEncoding::Ogg => "application/ogg",
        ByteStreamEncoding::Flv => "video/x-flv",
        ByteStreamEncoding::IsoBmff => "video/quicktime",
    }
}

/// Append `,key=value` for a fixed dimension; omit `Any` / `Range` (a wildcard
/// is the absence of the field in GStreamer caps).
fn push_dim(s: &mut String, key: &str, d: &Dim) {
    if let Dim::Fixed(v) = d {
        s.push_str(&format!(",{key}={v}"));
    }
}

/// Append `,framerate=N/D` for a fixed rate (Q16 fps). A whole-number fps prints
/// as `fps/1`; otherwise the exact Q16 value prints as `q16/65536`, which the
/// parser reads back to the same Q16.
fn push_rate(s: &mut String, r: &Rate) {
    if let Rate::Fixed(q16) = r {
        if q16 % 65536 == 0 {
            s.push_str(&format!(",framerate={}/1", q16 >> 16));
        } else {
            s.push_str(&format!(",framerate={q16}/65536"));
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    Any,
    Range { min: u32, max: u32 },
    Fixed(u32),
}

impl Dim {
    /// Intersect two dimension constraints. `Any` is the identity; two
    /// `Range`s overlap to their tighter bounds (collapsing to `Fixed` when
    /// the bounds meet); disjoint constraints yield `None`.
    pub fn intersect(&self, other: &Dim) -> Option<Dim> {
        intersect_range(self.bounds(), other.bounds()).map(Dim::from_bounds)
    }

    pub fn is_fixed(&self) -> bool {
        matches!(self, Dim::Fixed(_))
    }

    /// Collapse to a single `Fixed` value: `Range` picks its minimum, `Any`
    /// has nothing to pick and yields `None`. See [`Caps::fixate`].
    pub fn fixate(&self) -> Option<Dim> {
        match self {
            Dim::Fixed(v) => Some(Dim::Fixed(*v)),
            Dim::Range { min, .. } => Some(Dim::Fixed(*min)),
            Dim::Any => None,
        }
    }

    fn bounds(&self) -> (u32, u32) {
        match self {
            Dim::Any => (u32::MIN, u32::MAX),
            Dim::Range { min, max } => (*min, *max),
            Dim::Fixed(v) => (*v, *v),
        }
    }

    fn from_bounds((min, max): (u32, u32)) -> Dim {
        match (min, max) {
            (lo, hi) if lo == hi => Dim::Fixed(lo),
            (u32::MIN, u32::MAX) => Dim::Any, // full span is unconstrained
            (lo, hi) => Dim::Range { min: lo, max: hi },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rate {
    Any,
    /// Min/max framerate in Q16 fixed-point fps.
    Range { min_q16: u32, max_q16: u32 },
    /// Framerate in Q16 fixed-point fps.
    Fixed(u32),
}

impl Rate {
    /// Intersect two framerate constraints over their Q16 values; same
    /// semantics as [`Dim::intersect`].
    pub fn intersect(&self, other: &Rate) -> Option<Rate> {
        intersect_range(self.bounds(), other.bounds()).map(Rate::from_bounds)
    }

    pub fn is_fixed(&self) -> bool {
        matches!(self, Rate::Fixed(_))
    }

    /// Collapse to a single `Fixed` value: `Range` picks its minimum, `Any`
    /// yields `None`. See [`Caps::fixate`].
    pub fn fixate(&self) -> Option<Rate> {
        match self {
            Rate::Fixed(v) => Some(Rate::Fixed(*v)),
            Rate::Range { min_q16, .. } => Some(Rate::Fixed(*min_q16)),
            Rate::Any => None,
        }
    }

    fn bounds(&self) -> (u32, u32) {
        match self {
            Rate::Any => (u32::MIN, u32::MAX),
            Rate::Range { min_q16, max_q16 } => (*min_q16, *max_q16),
            Rate::Fixed(v) => (*v, *v),
        }
    }

    fn from_bounds((min, max): (u32, u32)) -> Rate {
        match (min, max) {
            (lo, hi) if lo == hi => Rate::Fixed(lo),
            (u32::MIN, u32::MAX) => Rate::Any, // full span is unconstrained
            (lo, hi) => Rate::Range { min_q16: lo, max_q16: hi },
        }
    }
}

/// Raw (uncompressed) PCM formats, the only ones the "any rate" wildcard (M187)
/// and the resampler apply to.
fn is_pcm(f: AudioFormat) -> bool {
    matches!(f, AudioFormat::PcmS16Le | AudioFormat::PcmF32Le)
}

/// Intersect two [`Caps::Audio`] sample rates, where [`ANY_SAMPLE_RATE`] (0) is
/// a wildcard (M187): `any ∩ x = x`, equal rates agree, distinct concrete rates
/// are disjoint (`None`).
fn intersect_sample_rate(a: u32, b: u32) -> Option<u32> {
    match (a, b) {
        (ANY_SAMPLE_RATE, x) | (x, ANY_SAMPLE_RATE) => Some(x),
        (x, y) if x == y => Some(x),
        _ => None,
    }
}

/// Which caps fields a transform passes through unchanged (output field ==
/// input field), declared alongside a
/// [`CapsConstraint::DerivedCoupled`](crate::format_element::CapsConstraint)
/// closure. The solver uses the declared passthrough fields to couple input and
/// output *field by field* in both directions, so a downstream pin on a
/// passthrough field narrows the corresponding input field (`Range ∩ Fixed =
/// Fixed`) instead of only dropping whole alternatives. The closure stays the
/// source of truth for the *retargeted* (non-passthrough) fields.
///
/// `format` covers the variant's scalar media identity:
/// [`Caps::RawVideo`]'s `format`, [`Caps::CompressedVideo`]'s `codec`, and
/// [`Caps::Audio`]'s `format`. The geometry / rate / channel flags apply to the
/// matching field where the variant has one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PassthroughFields {
    pub format: bool,
    pub width: bool,
    pub height: bool,
    pub framerate: bool,
    pub channels: bool,
    pub sample_rate: bool,
}

impl PassthroughFields {
    /// No field coupled (everything retargeted). Build with the `with_*`
    /// const setters: `PassthroughFields::NONE.with_format().with_framerate()`.
    pub const NONE: Self = Self {
        format: false,
        width: false,
        height: false,
        framerate: false,
        channels: false,
        sample_rate: false,
    };

    pub const fn with_format(mut self) -> Self {
        self.format = true;
        self
    }
    pub const fn with_width(mut self) -> Self {
        self.width = true;
        self
    }
    pub const fn with_height(mut self) -> Self {
        self.height = true;
        self
    }
    pub const fn with_framerate(mut self) -> Self {
        self.framerate = true;
        self
    }
    pub const fn with_channels(mut self) -> Self {
        self.channels = true;
        self
    }
    pub const fn with_sample_rate(mut self) -> Self {
        self.sample_rate = true;
        self
    }
}

/// Narrow `input` by intersecting each *passthrough* field against the
/// corresponding field of `pin` (the field-level backward coupling: e.g.
/// `Range(1..MAX) ∩ Fixed(160) = Fixed(160)`). Retarget fields (not in `mask`)
/// are left as `input` carries them, since the transform sets them
/// independently of its input. Same media variant required; `None` if a
/// passthrough field has no overlap (the alternative dies) or the variants
/// differ. Used by the solver's `DerivedCoupled` backward sweep.
pub(crate) fn couple_passthrough(input: &Caps, pin: &Caps, mask: PassthroughFields) -> Option<Caps> {
    match (input, pin) {
        (
            Caps::RawVideo { format: fi, width: wi, height: hi, framerate: ri },
            Caps::RawVideo { format: fp, width: wp, height: hp, framerate: rp },
        ) => {
            let format = if mask.format {
                if fi != fp {
                    return None;
                }
                *fi
            } else {
                *fi
            };
            let width = if mask.width { wi.intersect(wp)? } else { wi.clone() };
            let height = if mask.height { hi.intersect(hp)? } else { hi.clone() };
            let framerate = if mask.framerate { ri.intersect(rp)? } else { ri.clone() };
            Some(Caps::RawVideo { format, width, height, framerate })
        }
        (
            Caps::CompressedVideo { codec: ci, width: wi, height: hi, framerate: ri },
            Caps::CompressedVideo { codec: cp, width: wp, height: hp, framerate: rp },
        ) => {
            let codec = if mask.format {
                if ci != cp {
                    return None;
                }
                *ci
            } else {
                *ci
            };
            let width = if mask.width { wi.intersect(wp)? } else { wi.clone() };
            let height = if mask.height { hi.intersect(hp)? } else { hi.clone() };
            let framerate = if mask.framerate { ri.intersect(rp)? } else { ri.clone() };
            Some(Caps::CompressedVideo { codec, width, height, framerate })
        }
        (
            Caps::Audio { format: fi, channels: ci, sample_rate: si },
            Caps::Audio { format: fp, channels: cp, sample_rate: sp },
        ) => {
            let format = if mask.format {
                if fi != fp {
                    return None;
                }
                *fi
            } else {
                *fi
            };
            let channels = if mask.channels {
                if ci != cp {
                    return None;
                }
                *ci
            } else {
                *ci
            };
            let sample_rate =
                if mask.sample_rate { intersect_sample_rate(*si, *sp)? } else { *si };
            Some(Caps::Audio { format, channels, sample_rate })
        }
        _ => None,
    }
}

/// Like [`couple_passthrough`], but tolerates a *variant change* across the
/// transform (a decoder `CompressedVideo -> RawVideo`, an encoder the reverse),
/// for the discovered-passthrough backward coupling of a plain `DerivedOutput`.
/// Same-variant inputs defer to [`couple_passthrough`] (exact field coupling,
/// including `format`/`channels`/`sample_rate`). Across the two video variants
/// only the shared geometry / framerate fields can couple (the `format` slot is a
/// codec vs raw-format boundary, so it is never a passthrough field there); the
/// input keeps its own variant and scalar identity. `None` if a masked shared
/// field has no overlap, or for a cross-variant pair with no shared geometry.
pub(crate) fn couple_passthrough_derived(input: &Caps, pin: &Caps, mask: PassthroughFields) -> Option<Caps> {
    match (input, pin) {
        (Caps::RawVideo { .. }, Caps::RawVideo { .. })
        | (Caps::CompressedVideo { .. }, Caps::CompressedVideo { .. })
        | (Caps::Audio { .. }, Caps::Audio { .. }) => return couple_passthrough(input, pin, mask),
        _ => {}
    }
    // Cross video-variant: couple the geometry / rate both carry, keep `input`'s
    // variant + scalar identity (`format`/`codec` is retargeted across a codec
    // boundary, so `mask.format` is not applied here).
    let (wi, hi, ri) = (geo_width(input)?, geo_height(input)?, geo_rate(input)?);
    let (wp, hp, rp) = (geo_width(pin)?, geo_height(pin)?, geo_rate(pin)?);
    let width = if mask.width { wi.intersect(wp)? } else { wi.clone() };
    let height = if mask.height { hi.intersect(hp)? } else { hi.clone() };
    let framerate = if mask.framerate { ri.intersect(rp)? } else { ri.clone() };
    match input {
        Caps::RawVideo { format, .. } => Some(Caps::RawVideo { format: *format, width, height, framerate }),
        Caps::CompressedVideo { codec, .. } => {
            Some(Caps::CompressedVideo { codec: *codec, width, height, framerate })
        }
        _ => None,
    }
}

/// Project an output-side feasible `out` onto the *input* side of a
/// `DerivedCoupled` transform: keep passthrough fields, widen each retargeted
/// field to "anything the transform can take" (`Dim`/`Rate` -> `Any`,
/// `sample_rate` -> [`ANY_SAMPLE_RATE`]). Returns `None` when a retargeted field
/// is a non-rangeable scalar (`format` / `codec` / `channels`) with no wildcard,
/// i.e. the input feasibility can't be expressed as a single `Caps` (the solver
/// then imposes no upstream feasibility constraint, the status quo). Used by
/// `backward_feasible` for the mid-stream snapshot.
#[cfg(feature = "std")]
pub(crate) fn project_passthrough(out: &Caps, mask: PassthroughFields) -> Option<Caps> {
    match out {
        Caps::RawVideo { format, width, height, framerate } => {
            if !mask.format {
                return None; // retargeted format has no wildcard
            }
            Some(Caps::RawVideo {
                format: *format,
                width: if mask.width { width.clone() } else { Dim::Any },
                height: if mask.height { height.clone() } else { Dim::Any },
                framerate: if mask.framerate { framerate.clone() } else { Rate::Any },
            })
        }
        Caps::CompressedVideo { codec, width, height, framerate } => {
            if !mask.format {
                return None;
            }
            Some(Caps::CompressedVideo {
                codec: *codec,
                width: if mask.width { width.clone() } else { Dim::Any },
                height: if mask.height { height.clone() } else { Dim::Any },
                framerate: if mask.framerate { framerate.clone() } else { Rate::Any },
            })
        }
        Caps::Audio { format, channels, sample_rate } => {
            if !mask.format || !mask.channels {
                return None; // no format / channel wildcard
            }
            Some(Caps::Audio {
                format: *format,
                channels: *channels,
                sample_rate: if mask.sample_rate { *sample_rate } else { ANY_SAMPLE_RATE },
            })
        }
        _ => None,
    }
}

/// Project an output-side feasible `out` onto the *input* side of a plain
/// `DerivedOutput` for the mid-stream snapshot ([`backward_feasible`]). Unlike
/// [`couple_passthrough_derived`] (the full-chain coupling, which keeps the input
/// sample's own value on a non-passthrough field), this *widens* every
/// non-passthrough geometry / rate field to `Any`: the transform re-derives that
/// field from whatever input it receives mid-stream, so the input edge must stay
/// unconstrained on it. Freezing it to the startup sample (the M258 v1 behaviour)
/// made the snapshot reject a legitimately re-derived mid-stream geometry, the
/// Caps-β forward gap.
///
/// Same-variant transforms defer to [`project_passthrough`] (which already widens
/// retargeted fields and rejects a non-rangeable retargeted scalar). Across the
/// decoder / encoder variant change, the passthrough geometry / rate fields take
/// the downstream value from `out` while the non-passthrough fields widen to
/// `Any`; `sample` supplies the input variant and its scalar identity (codec /
/// format), which `out` cannot give.
#[cfg(feature = "std")]
pub(crate) fn project_passthrough_derived(
    sample: &Caps,
    out: &Caps,
    mask: PassthroughFields,
) -> Option<Caps> {
    match (sample, out) {
        (Caps::RawVideo { .. }, Caps::RawVideo { .. })
        | (Caps::CompressedVideo { .. }, Caps::CompressedVideo { .. })
        | (Caps::Audio { .. }, Caps::Audio { .. }) => return project_passthrough(out, mask),
        _ => {}
    }
    // Cross video-variant (decoder / encoder): passthrough geometry / rate take the
    // downstream value, the rest widen to `Any`; keep `sample`'s variant + scalar id.
    let (wp, hp, rp) = (geo_width(out)?, geo_height(out)?, geo_rate(out)?);
    let width = if mask.width { wp.clone() } else { Dim::Any };
    let height = if mask.height { hp.clone() } else { Dim::Any };
    let framerate = if mask.framerate { rp.clone() } else { Rate::Any };
    match sample {
        Caps::RawVideo { format, .. } => Some(Caps::RawVideo { format: *format, width, height, framerate }),
        Caps::CompressedVideo { codec, .. } => {
            Some(Caps::CompressedVideo { codec: *codec, width, height, framerate })
        }
        _ => None,
    }
}

/// The fields [`discover_passthrough`] probes for, one per [`PassthroughFields`]
/// flag.
#[derive(Clone, Copy)]
enum ProbeField {
    Width,
    Height,
    Framerate,
    Format,
    Channels,
    SampleRate,
}

/// Probe a `DerivedOutput`-style closure to discover which caps fields it passes
/// through unchanged (output field tracks input field), so the solver can couple
/// those fields backward the same way a declared
/// [`DerivedCoupled`](crate::format_element::CapsConstraint::DerivedCoupled) mask
/// does, the "invertible fields of a `DerivedOutput`". `f` is not analytically
/// invertible, but it is evaluable, so a field's behaviour is read off two
/// concrete probes.
///
/// Conservative by construction: a field is marked passthrough only when two
/// distinct concrete probes *both* show the closure's single, same-shaped output
/// field equal to the probed input field. A closure that rejects a probe, fixes
/// the field (a retargeted decoder format), or returns multiple/ambiguous outputs
/// yields `false` for that field, so discovery never invents coupling that is not
/// there (a wrong `true` would narrow the input incorrectly). `sample` is a
/// representative input alternative; its geometry is concretised first so a
/// `Range`/`Any` input field does not confuse the equality test.
pub(crate) fn discover_passthrough(f: &dyn Fn(&Caps) -> CapsSet, sample: &Caps) -> PassthroughFields {
    let base = concrete_probe_base(sample);
    // Soundness gate: a field is probed by *varying* it, so a closure that is
    // multi-valued on the sample's own identity (e.g. a converter that offers
    // `{passthrough, retargeted}` for the sample's format but is coincidentally
    // single-valued at the probe values) would be mis-read as passthrough. Per-
    // field equality alone can't see that, so require the closure to be single-
    // valued on the sample's representative input before trusting any field: a
    // genuinely ambiguous transform has no well-defined per-field passthrough.
    if single_out(f, &base).is_none() {
        return PassthroughFields::NONE;
    }
    PassthroughFields {
        width: probe_field(f, &base, ProbeField::Width),
        height: probe_field(f, &base, ProbeField::Height),
        framerate: probe_field(f, &base, ProbeField::Framerate),
        format: probe_field(f, &base, ProbeField::Format),
        channels: probe_field(f, &base, ProbeField::Channels),
        sample_rate: probe_field(f, &base, ProbeField::SampleRate),
    }
}

/// Concretise `sample`'s ranged geometry/rate to fixed sentinels so the closure
/// is probed on concrete inputs (a `Range`/`Any` input field would otherwise
/// make the output-equals-input test ambiguous). Scalar identity (format / codec
/// / channels) is kept from `sample`, since the closure may key on it.
fn concrete_probe_base(sample: &Caps) -> Caps {
    match sample {
        Caps::RawVideo { format, .. } => Caps::RawVideo {
            format: *format,
            width: Dim::Fixed(64),
            height: Dim::Fixed(64),
            framerate: Rate::Fixed(30 << 16),
        },
        Caps::CompressedVideo { codec, .. } => Caps::CompressedVideo {
            codec: *codec,
            width: Dim::Fixed(64),
            height: Dim::Fixed(64),
            framerate: Rate::Fixed(30 << 16),
        },
        Caps::Audio { format, .. } => Caps::Audio { format: *format, channels: 2, sample_rate: 48_000 },
        other => other.clone(),
    }
}

/// True when `f` passes `field` through: two concrete probes that differ only in
/// `field` each produce a single output whose `field` equals the probe's.
fn probe_field(f: &dyn Fn(&Caps) -> CapsSet, base: &Caps, field: ProbeField) -> bool {
    let (Some(p0), Some(p1)) = (set_probe(base, field, false), set_probe(base, field, true)) else {
        return false;
    };
    let (Some(o0), Some(o1)) = (single_out(f, &p0), single_out(f, &p1)) else {
        return false;
    };
    field_eq(&o0, &p0, field) && field_eq(&o1, &p1, field)
}

/// The single output of `f(input)`, or `None` if it produced zero or several
/// alternatives (discovery stays conservative on ambiguous closures).
fn single_out(f: &dyn Fn(&Caps) -> CapsSet, input: &Caps) -> Option<Caps> {
    let set = f(input);
    match set.alternatives() {
        [only] => Some(only.clone()),
        _ => None,
    }
}

/// `base` with `field` set to probe value 0 (`hi = false`) or 1 (`hi = true`),
/// or `None` if `base`'s variant has no such field.
fn set_probe(base: &Caps, field: ProbeField, hi: bool) -> Option<Caps> {
    let mut c = base.clone();
    match (&mut c, field) {
        (Caps::RawVideo { width, .. }, ProbeField::Width)
        | (Caps::CompressedVideo { width, .. }, ProbeField::Width) => {
            *width = Dim::Fixed(if hi { 128 } else { 64 });
        }
        (Caps::RawVideo { height, .. }, ProbeField::Height)
        | (Caps::CompressedVideo { height, .. }, ProbeField::Height) => {
            *height = Dim::Fixed(if hi { 128 } else { 64 });
        }
        (Caps::RawVideo { framerate, .. }, ProbeField::Framerate)
        | (Caps::CompressedVideo { framerate, .. }, ProbeField::Framerate) => {
            *framerate = Rate::Fixed(if hi { 60 << 16 } else { 30 << 16 });
        }
        (Caps::RawVideo { format, .. }, ProbeField::Format) => {
            *format = if hi { RawVideoFormat::I420 } else { RawVideoFormat::Nv12 };
        }
        (Caps::CompressedVideo { codec, .. }, ProbeField::Format) => {
            *codec = if hi { VideoCodec::H265 } else { VideoCodec::H264 };
        }
        (Caps::Audio { format, .. }, ProbeField::Format) => {
            *format = if hi { AudioFormat::PcmF32Le } else { AudioFormat::PcmS16Le };
        }
        (Caps::Audio { channels, .. }, ProbeField::Channels) => {
            *channels = if hi { 1 } else { 2 };
        }
        (Caps::Audio { sample_rate, .. }, ProbeField::SampleRate) => {
            *sample_rate = if hi { 44_100 } else { 48_000 };
        }
        _ => return None,
    }
    Some(c)
}

/// True when `out`'s `field` equals `inp`'s. Geometry/rate compare across
/// variants (both `RawVideo` and `CompressedVideo` carry them); the scalar
/// identity / channels / sample_rate require the same variant.
fn field_eq(out: &Caps, inp: &Caps, field: ProbeField) -> bool {
    match field {
        ProbeField::Width => geo_width(out).zip(geo_width(inp)).is_some_and(|(a, b)| a == b),
        ProbeField::Height => geo_height(out).zip(geo_height(inp)).is_some_and(|(a, b)| a == b),
        ProbeField::Framerate => geo_rate(out).zip(geo_rate(inp)).is_some_and(|(a, b)| a == b),
        ProbeField::Format => match (out, inp) {
            (Caps::RawVideo { format: a, .. }, Caps::RawVideo { format: b, .. }) => a == b,
            (Caps::CompressedVideo { codec: a, .. }, Caps::CompressedVideo { codec: b, .. }) => a == b,
            (Caps::Audio { format: a, .. }, Caps::Audio { format: b, .. }) => a == b,
            _ => false,
        },
        ProbeField::Channels => match (out, inp) {
            (Caps::Audio { channels: a, .. }, Caps::Audio { channels: b, .. }) => a == b,
            _ => false,
        },
        ProbeField::SampleRate => match (out, inp) {
            (Caps::Audio { sample_rate: a, .. }, Caps::Audio { sample_rate: b, .. }) => a == b,
            _ => false,
        },
    }
}

fn geo_width(c: &Caps) -> Option<&Dim> {
    match c {
        Caps::RawVideo { width, .. } | Caps::CompressedVideo { width, .. } => Some(width),
        _ => None,
    }
}

fn geo_height(c: &Caps) -> Option<&Dim> {
    match c {
        Caps::RawVideo { height, .. } | Caps::CompressedVideo { height, .. } => Some(height),
        _ => None,
    }
}

fn geo_rate(c: &Caps) -> Option<&Rate> {
    match c {
        Caps::RawVideo { framerate, .. } | Caps::CompressedVideo { framerate, .. } => Some(framerate),
        _ => None,
    }
}

/// Overlap two inclusive `[min, max]` bounds, returning `None` when disjoint.
/// Shared by [`Dim::intersect`] and [`Rate::intersect`].
fn intersect_range((amin, amax): (u32, u32), (bmin, bmax): (u32, u32)) -> Option<(u32, u32)> {
    let lo = amin.max(bmin);
    let hi = amax.min(bmax);
    (lo <= hi).then_some((lo, hi))
}

/// A preference-ordered set of acceptable `Caps` descriptions.
///
/// `Caps` itself remains the *fixed* description used at runtime
/// (`DataFrame.caps`, `configure_*`). `CapsSet` is the negotiation-time
/// vocabulary: it carries alternatives and preference, neither of which
/// fits in a single `Caps`. See DESIGN.md §4.13.1.
///
/// The first alternative is highest preference; later ones are
/// fallbacks the element will accept if no peer agrees on the first.
#[derive(Clone, Debug, PartialEq)]
pub struct CapsSet {
    alternatives: Vec<Caps>,
}

impl CapsSet {
    /// Build from a single concrete description (equivalent to today's
    /// `Caps` for static call sites that don't express alternatives).
    pub fn one(caps: Caps) -> Self {
        Self { alternatives: alloc::vec![caps] }
    }

    /// Build directly from an ordered list of alternatives. The first
    /// element is highest preference. Empty input is allowed and yields
    /// the empty set (no agreement possible with any peer).
    pub fn from_alternatives(alternatives: Vec<Caps>) -> Self {
        Self { alternatives }
    }

    /// Return the ordered alternatives.
    pub fn alternatives(&self) -> &[Caps] {
        &self.alternatives
    }

    /// True when no alternatives remain. An empty `CapsSet` on a link
    /// means the two peers' constraints do not intersect.
    pub fn is_empty(&self) -> bool {
        self.alternatives.is_empty()
    }

    /// Intersection: the caps both sets agree on, preserving `self`'s
    /// outer preference order, then `other`'s within each row.
    /// Empty result = no assignment exists for a link between elements
    /// with these two sets.
    pub fn intersect(&self, other: &Self) -> Self {
        let mut out = Vec::new();
        for a in &self.alternatives {
            for b in &other.alternatives {
                if let Ok(c) = a.intersect(b) {
                    if !out.contains(&c) {
                        out.push(c);
                    }
                }
            }
        }
        Self { alternatives: out }
    }

    /// Union: every alternative in `self` followed by every alternative
    /// in `other` not already present. Preserves `self`'s preference
    /// order and dedupes structurally-equal entries. Used by the
    /// `Mapping` solver path to combine the surviving (input, output)
    /// pair sides.
    pub fn union(&self, other: &Self) -> Self {
        let mut out = self.alternatives.clone();
        for c in &other.alternatives {
            if !out.contains(c) {
                out.push(c.clone());
            }
        }
        Self { alternatives: out }
    }

    /// Fixate the highest-preference alternative that can collapse to a
    /// single concrete `Caps`. Returns `None` if the set is empty or
    /// every alternative still has `Any` fields after attempting
    /// fixation.
    pub fn fixate(&self) -> Option<Caps> {
        self.alternatives.iter().find_map(|c| c.fixate().ok())
    }

    /// True if any alternative is compatible with `caps` (a non-empty
    /// intersection exists). The ACCEPT_CAPS predicate (DESIGN.md §4.13.1):
    /// "would a link carrying `caps` satisfy this set?" — a pure check,
    /// no negotiation.
    pub fn accepts(&self, caps: &Caps) -> bool {
        self.alternatives.iter().any(|a| a.intersect(caps).is_ok())
    }
}

/// Compressed video codec carried in a [`Caps::CompressedVideo`] link.
/// Split out of the old `VideoFormat` enum so a decoder's "I accept
/// codec, I emit raw" boundary is type-level rather than a runtime
/// format compare. M17 split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    H264,
    H265,
    Av1,
    Vp8,
    Vp9,
    /// Motion JPEG: each frame an independent baseline JPEG. The near-universal
    /// fallback output of cheap UVC webcams, decoded by `MjpegDec`.
    Mjpeg,
}

/// Wire format of a [`Caps::ByteStream`] link, so a demuxer accepts only the
/// container it parses (an MPEG-TS demuxer rejects an arbitrary byte stream
/// structurally, like the codec/raw split does for video).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ByteStreamEncoding {
    /// MPEG-2 Transport Stream (ISO/IEC 13818-1): 188-byte packets, PAT/PMT,
    /// PES. The broadcast / SRT / HLS-segment carrier.
    MpegTs,
    /// Matroska / WebM (EBML): nested variable-length elements; Tracks describe
    /// the elementary streams and Clusters carry the SimpleBlock frames. The
    /// common file container, WebM being the browser-delivery subset (VP8 / VP9 /
    /// AV1 video + Opus / Vorbis audio).
    Matroska,
    /// Ogg (RFC 3533): "OggS" pages with a segment-table lacing that frames the
    /// packets of a logical bitstream. The canonical Opus / Vorbis carrier.
    Ogg,
    /// FLV (Flash Video): an "FLV" header then `PreviousTagSize` / tag pairs, each
    /// tag a codec-tagged audio / video / script payload. The RTMP carrier.
    Flv,
    /// ISO Base Media File Format / fragmented MP4 (CMAF): `ftyp`/`moov` init then
    /// `moof`+`mdat` fragments. The modern HLS/DASH segment container, demuxed by
    /// `fmp4demux`.
    IsoBmff,
}

/// Raw pixel layout carried in a [`Caps::RawVideo`] link. Split out of
/// the old `VideoFormat` enum so a raw sink (waylandsink/kmssink)
/// rejects compressed input structurally rather than via runtime check.
/// M17 split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RawVideoFormat {
    Nv12,
    I420,
    Rgba8,
    Bgra8,
    /// Packed YUV 4:2:2, byte order Y0 U Y1 V (the V4L2 `YUYV` / `YUY2`
    /// fourcc). Two bytes per pixel; the near-universal UVC webcam output.
    /// Packed (not planar), so it needs unpacking before planar consumers.
    Yuyv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AudioFormat {
    Aac,
    Opus,
    PcmS16Le,
    PcmF32Le,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorDType {
    F16,
    F32,
    I8,
    U8,
}

impl TensorDType {
    /// Size in bytes of one element of this dtype. Used by [`crate::tensor`]
    /// to turn element strides into byte strides and size a materialization.
    pub const fn size(self) -> usize {
        match self {
            TensorDType::F16 => 2,
            TensorDType::F32 => 4,
            TensorDType::I8 | TensorDType::U8 => 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorShape(pub Vec<u32>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorLayout {
    Nchw,
    Nhwc,
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn video(width: Dim, height: Dim, framerate: Rate) -> Caps {
        Caps::RawVideo { format: RawVideoFormat::Rgba8, width, height, framerate }
    }

    #[test]
    fn dim_intersect_any_is_identity() {
        assert_eq!(Dim::Any.intersect(&Dim::Fixed(720)), Some(Dim::Fixed(720)));
        assert_eq!(Dim::Fixed(720).intersect(&Dim::Any), Some(Dim::Fixed(720)));
        assert_eq!(Dim::Any.intersect(&Dim::Any), Some(Dim::Any));
    }

    #[test]
    fn dim_intersect_fixed_pairs() {
        assert_eq!(Dim::Fixed(64).intersect(&Dim::Fixed(64)), Some(Dim::Fixed(64)));
        assert_eq!(Dim::Fixed(64).intersect(&Dim::Fixed(65)), None);
    }

    #[test]
    fn dim_intersect_fixed_against_range() {
        let range = Dim::Range { min: 100, max: 200 };
        assert_eq!(Dim::Fixed(150).intersect(&range), Some(Dim::Fixed(150)));
        assert_eq!(Dim::Fixed(100).intersect(&range), Some(Dim::Fixed(100))); // inclusive lo
        assert_eq!(Dim::Fixed(200).intersect(&range), Some(Dim::Fixed(200))); // inclusive hi
        assert_eq!(Dim::Fixed(99).intersect(&range), None);
        assert_eq!(Dim::Fixed(201).intersect(&range), None);
    }

    #[test]
    fn dim_intersect_overlapping_ranges_tighten() {
        let a = Dim::Range { min: 100, max: 300 };
        let b = Dim::Range { min: 200, max: 400 };
        assert_eq!(a.intersect(&b), Some(Dim::Range { min: 200, max: 300 }));
    }

    #[test]
    fn dim_intersect_ranges_meeting_at_a_point_collapse_to_fixed() {
        let a = Dim::Range { min: 100, max: 200 };
        let b = Dim::Range { min: 200, max: 300 };
        assert_eq!(a.intersect(&b), Some(Dim::Fixed(200)));
    }

    #[test]
    fn dim_intersect_disjoint_ranges_none() {
        let a = Dim::Range { min: 100, max: 199 };
        let b = Dim::Range { min: 200, max: 300 };
        assert_eq!(a.intersect(&b), None);
    }

    #[test]
    fn rate_intersect_mirrors_dim() {
        let a = Rate::Range { min_q16: 15 << 16, max_q16: 60 << 16 };
        let b = Rate::Fixed(30 << 16);
        assert_eq!(a.intersect(&b), Some(Rate::Fixed(30 << 16)));
        assert_eq!(Rate::Any.intersect(&b), Some(Rate::Fixed(30 << 16)));
        // 10 fps falls below the [15, 60] range → no overlap.
        assert_eq!(Rate::Fixed(10 << 16).intersect(&a), None);
    }

    #[test]
    fn dim_fixate_picks_range_minimum() {
        assert_eq!(Dim::Range { min: 480, max: 1080 }.fixate(), Some(Dim::Fixed(480)));
        assert_eq!(Dim::Fixed(720).fixate(), Some(Dim::Fixed(720)));
        assert_eq!(Dim::Any.fixate(), None);
    }

    #[test]
    fn caps_intersect_video_fields() {
        let a = video(Dim::Range { min: 640, max: 1920 }, Dim::Any, Rate::Any);
        let b = video(Dim::Fixed(1280), Dim::Fixed(720), Rate::Fixed(30 << 16));
        assert_eq!(
            a.intersect(&b).unwrap(),
            video(Dim::Fixed(1280), Dim::Fixed(720), Rate::Fixed(30 << 16))
        );
    }

    #[test]
    fn caps_intersect_rejects_format_mismatch() {
        let a = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let b = video(Dim::Any, Dim::Any, Rate::Any); // Rgba8
        assert_eq!(a.intersect(&b), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_intersect_rejects_empty_field_overlap() {
        let a = video(Dim::Fixed(640), Dim::Any, Rate::Any);
        let b = video(Dim::Fixed(1280), Dim::Any, Rate::Any);
        assert_eq!(a.intersect(&b), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_intersect_rejects_variant_mismatch() {
        let v = video(Dim::Any, Dim::Any, Rate::Any);
        let a = Caps::Audio { format: AudioFormat::Opus, channels: 2, sample_rate: 48_000 };
        assert_eq!(v.intersect(&a), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_intersect_audio_and_tensor_require_scalar_equality() {
        let a = Caps::Audio { format: AudioFormat::Opus, channels: 2, sample_rate: 48_000 };
        assert_eq!(a.intersect(&a), Ok(a.clone()));
        let b = Caps::Audio { format: AudioFormat::Opus, channels: 1, sample_rate: 48_000 };
        assert_eq!(a.intersect(&b), Err(G2gError::CapsMismatch));

        let t = Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape(vec![1, 3, 224, 224]),
            layout: TensorLayout::Nchw,
        };
        assert_eq!(t.intersect(&t), Ok(t.clone()));
    }

    #[test]
    fn caps_is_fixed() {
        assert!(video(Dim::Fixed(1), Dim::Fixed(1), Rate::Fixed(1)).is_fixed());
        assert!(!video(Dim::Any, Dim::Fixed(1), Rate::Fixed(1)).is_fixed());
        assert!(!video(Dim::Fixed(1), Dim::Range { min: 1, max: 2 }, Rate::Fixed(1)).is_fixed());
        assert!(Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 44_100 }.is_fixed());
    }

    #[test]
    fn capsset_one_wraps_single_caps() {
        let c = video(Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(30 << 16));
        let set = CapsSet::one(c.clone());
        assert_eq!(set.alternatives(), &[c]);
        assert!(!set.is_empty());
    }

    #[test]
    fn capsset_intersect_single_pair() {
        let a = CapsSet::one(video(Dim::Range { min: 640, max: 1920 }, Dim::Any, Rate::Any));
        let b = CapsSet::one(video(Dim::Fixed(1280), Dim::Fixed(720), Rate::Fixed(30 << 16)));
        let i = a.intersect(&b);
        assert_eq!(i.alternatives(), &[video(Dim::Fixed(1280), Dim::Fixed(720), Rate::Fixed(30 << 16))]);
    }

    #[test]
    fn capsset_intersect_empty_when_no_overlap() {
        let a = CapsSet::one(video(Dim::Fixed(640), Dim::Any, Rate::Any));
        let b = CapsSet::one(video(Dim::Fixed(1280), Dim::Any, Rate::Any));
        assert!(a.intersect(&b).is_empty());
    }

    #[test]
    fn capsset_intersect_preserves_self_preference_order() {
        // self: prefers Rgba8 then H264; other: accepts both with any dims.
        let rgba = |w| Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: w,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let h264 = |w| Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: w,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let a = CapsSet::from_alternatives(alloc::vec![rgba(Dim::Any), h264(Dim::Any)]);
        let b = CapsSet::from_alternatives(alloc::vec![h264(Dim::Fixed(1280)), rgba(Dim::Fixed(640))]);
        let i = a.intersect(&b);
        // self's outer order wins: Rgba8 first even though other lists H264 first.
        assert_eq!(i.alternatives(), &[rgba(Dim::Fixed(640)), h264(Dim::Fixed(1280))]);
    }

    #[test]
    fn capsset_intersect_dedupes_equal_results() {
        // Two self-alternatives that both intersect `other` to the same Caps.
        let any = video(Dim::Any, Dim::Any, Rate::Any);
        let fixed = video(Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(30 << 16));
        let a = CapsSet::from_alternatives(alloc::vec![any.clone(), any.clone()]);
        let b = CapsSet::one(fixed.clone());
        let i = a.intersect(&b);
        assert_eq!(i.alternatives(), &[fixed]);
    }

    #[test]
    fn capsset_union_preserves_self_order_and_dedupes() {
        let a = video(Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(30 << 16));
        let b = video(Dim::Fixed(1280), Dim::Fixed(720), Rate::Fixed(30 << 16));
        let c = video(Dim::Fixed(1920), Dim::Fixed(1080), Rate::Fixed(30 << 16));
        let lhs = CapsSet::from_alternatives(alloc::vec![a.clone(), b.clone()]);
        let rhs = CapsSet::from_alternatives(alloc::vec![b.clone(), c.clone()]);
        let u = lhs.union(&rhs);
        assert_eq!(u.alternatives(), &[a, b, c]);
    }

    #[test]
    fn capsset_fixate_picks_first_fixable_alternative() {
        // First alt has framerate Any (not fixable); second is fully fixable.
        let unfixable = video(Dim::Fixed(640), Dim::Fixed(480), Rate::Any);
        let fixable = video(Dim::Range { min: 800, max: 1920 }, Dim::Fixed(720), Rate::Fixed(30 << 16));
        let set = CapsSet::from_alternatives(alloc::vec![unfixable, fixable]);
        assert_eq!(
            set.fixate(),
            Some(video(Dim::Fixed(800), Dim::Fixed(720), Rate::Fixed(30 << 16)))
        );
    }

    #[test]
    fn capsset_fixate_empty_or_all_unfixable_yields_none() {
        assert!(CapsSet::from_alternatives(Vec::new()).fixate().is_none());
        let only_any = video(Dim::Any, Dim::Any, Rate::Any);
        assert!(CapsSet::one(only_any).fixate().is_none());
    }

    #[test]
    fn caps_fixate_collapses_ranges_and_rejects_any() {
        let ranged = video(Dim::Range { min: 640, max: 1920 }, Dim::Fixed(480), Rate::Any);
        assert_eq!(ranged.fixate(), Err(G2gError::CapsMismatch)); // framerate Any

        let fixable = video(Dim::Range { min: 640, max: 1920 }, Dim::Fixed(480), Rate::Fixed(30 << 16));
        let fixed = fixable.fixate().unwrap();
        assert!(fixed.is_fixed());
        assert_eq!(fixed, video(Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(30 << 16)));
    }

    #[test]
    fn discover_passthrough_decoder_geometry_and_framerate() {
        // H264 -> Nv12: geometry + framerate copied through, format retargeted.
        let dec = |input: &Caps| match input {
            Caps::CompressedVideo { width, height, framerate, .. } => CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        };
        let sample = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let pt = discover_passthrough(&dec, &sample);
        assert!(pt.width && pt.height && pt.framerate, "geometry + rate copied through");
        assert!(!pt.format, "codec -> format is retargeted, not passthrough");
    }

    #[test]
    fn discover_passthrough_none_for_fixed_output() {
        // Output ignores the input (fixed dims): nothing invertible to discover.
        let dec = |_: &Caps| {
            CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Fixed(30 << 16),
            })
        };
        let sample = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(discover_passthrough(&dec, &sample), PassthroughFields::NONE);
    }

    #[test]
    fn discover_passthrough_identity_convert_all_fields() {
        // RawVideo -> RawVideo identity: every probed field passes through.
        let id = |input: &Caps| CapsSet::one(input.clone());
        let pt = discover_passthrough(&id, &video(Dim::Any, Dim::Any, Rate::Any));
        assert!(pt.width && pt.height && pt.framerate && pt.format);
    }

    #[test]
    fn discover_passthrough_scaler_retargets_geometry_only() {
        // A scaler fixes output geometry but keeps format + framerate: those two
        // are passthrough, width/height are not.
        let scale = |input: &Caps| match input {
            Caps::RawVideo { format, framerate, .. } => CapsSet::one(Caps::RawVideo {
                format: *format,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: framerate.clone(),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        };
        let pt = discover_passthrough(&scale, &video(Dim::Any, Dim::Any, Rate::Any));
        assert!(pt.format && pt.framerate, "format + rate kept");
        assert!(!pt.width && !pt.height, "geometry is retargeted by the scaler");
    }

    #[test]
    fn discover_passthrough_none_for_multivalued_closure() {
        // A converter that offers {passthrough, retargeted-NV12} for an RGBA input
        // is multi-valued on its own sample, but coincidentally single-valued at
        // the format-probe values (Nv12 / I420, neither in `from`). Per-field
        // probing alone would mis-read `format` as passthrough and then drop the
        // RGBA input when coupling it against an NV12 pin (the M257 startup-failure
        // bug). The single-valued gate on the sample makes discovery bail to NONE.
        let from = [RawVideoFormat::Rgba8];
        let conv = move |input: &Caps| {
            let mut alts = vec![input.clone()];
            if let Caps::RawVideo { format, width, height, framerate } = input {
                if from.contains(format) {
                    alts.push(Caps::RawVideo {
                        format: RawVideoFormat::Nv12,
                        width: width.clone(),
                        height: height.clone(),
                        framerate: framerate.clone(),
                    });
                }
            }
            CapsSet::from_alternatives(alts)
        };
        let sample = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
        };
        assert_eq!(discover_passthrough(&conv, &sample), PassthroughFields::NONE);
    }
}
