use alloc::vec::Vec;

use crate::error::G2gError;

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
            ) if fa == fb && ca == cb && sa == sb => Ok(self.clone()),
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
}
