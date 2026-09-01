//! M837: declarative forward derivation for caps transforms
//! (DESIGN.md §4.13.1).
//!
//! A caps-driven transform (videoscale / videoconvert / videorate /
//! audioconvert / audioresample) derives its output caps field by field from
//! its input: some fields pass through, some are retargeted to a value the
//! element's properties or the downstream pin choose. [`CapsTransform`] states
//! that relation as data instead of a closure, so the solver can read it: the
//! backward-coupling mask ([`CapsTransform::passthrough`]) is the set of fields
//! every output alternative derives with [`FieldTransform::Identity`], which
//! makes a mask that disagrees with the derivation unrepresentable.

use alloc::vec::Vec;

use crate::caps::{AudioFormat, Caps, CapsSet, Dim, PassthroughFields, Rate, RawVideoFormat};

/// How one output caps field is derived from the corresponding input field.
///
/// `Fixed` carries a whole caps value, so it also expresses a *ranged* retarget
/// (`Dim::Range`, `Rate::Any`, `ANY_SAMPLE_RATE`): the alternative a downstream
/// capsfilter then pins. A scale-by-rational variant (`width * num / den`) was
/// considered and left out: no element derives a field that way, so the
/// vocabulary stays at these two until one does.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldTransform<T> {
    /// Output field == input field. The declaration that makes the field a
    /// passthrough, so the solver couples it backward.
    Identity,
    /// Output field is this value, whatever the input carries.
    Fixed(T),
}

impl<T: Clone> FieldTransform<T> {
    /// Derive the output field from `input`.
    pub fn apply(&self, input: &T) -> T {
        match self {
            Self::Identity => input.clone(),
            Self::Fixed(v) => v.clone(),
        }
    }
}

impl<T> FieldTransform<T> {
    pub fn is_identity(&self) -> bool {
        matches!(self, Self::Identity)
    }
}

/// One output alternative of a raw-video [`CapsTransform`].
#[derive(Clone, Debug, PartialEq)]
pub struct RawVideoShape {
    pub format: FieldTransform<RawVideoFormat>,
    pub width: FieldTransform<Dim>,
    pub height: FieldTransform<Dim>,
    pub framerate: FieldTransform<Rate>,
}

impl RawVideoShape {
    /// Every field passed through. Retarget from here with the `with_*`
    /// setters: `RawVideoShape::PASSTHROUGH.with_width(FieldTransform::Fixed(w))`.
    pub const PASSTHROUGH: Self = Self {
        format: FieldTransform::Identity,
        width: FieldTransform::Identity,
        height: FieldTransform::Identity,
        framerate: FieldTransform::Identity,
    };

    pub fn with_format(mut self, t: FieldTransform<RawVideoFormat>) -> Self {
        self.format = t;
        self
    }
    pub fn with_width(mut self, t: FieldTransform<Dim>) -> Self {
        self.width = t;
        self
    }
    pub fn with_height(mut self, t: FieldTransform<Dim>) -> Self {
        self.height = t;
        self
    }
    pub fn with_framerate(mut self, t: FieldTransform<Rate>) -> Self {
        self.framerate = t;
        self
    }
}

/// One output alternative of an audio [`CapsTransform`].
#[derive(Clone, Debug, PartialEq)]
pub struct AudioShape {
    pub format: FieldTransform<AudioFormat>,
    pub channels: FieldTransform<u8>,
    pub sample_rate: FieldTransform<u32>,
}

impl AudioShape {
    /// Every field passed through; retarget with the `with_*` setters.
    pub const PASSTHROUGH: Self = Self {
        format: FieldTransform::Identity,
        channels: FieldTransform::Identity,
        sample_rate: FieldTransform::Identity,
    };

    pub fn with_format(mut self, t: FieldTransform<AudioFormat>) -> Self {
        self.format = t;
        self
    }
    pub fn with_channels(mut self, t: FieldTransform<u8>) -> Self {
        self.channels = t;
        self
    }
    pub fn with_sample_rate(mut self, t: FieldTransform<u32>) -> Self {
        self.sample_rate = t;
        self
    }
}

/// Declarative forward derivation for a transform element, read by the solver
/// through [`CapsConstraint::DerivedFields`](crate::format_element::CapsConstraint).
///
/// `shapes` are the output alternatives in preference order (first preferred);
/// duplicates are dropped, so a shape that coincides with the passthrough on a
/// given input costs nothing. `accept` gates the input format (empty = every
/// format of that media kind) and `produce` gates the derived output format
/// (empty = unrestricted), which is what keeps an input-only format out of the
/// derived set. An input the transform doesn't accept, or no surviving shape,
/// derives the empty set, which the solver reads as an unsatisfiable link.
#[derive(Clone, Debug, PartialEq)]
pub enum CapsTransform {
    RawVideo {
        accept: Vec<RawVideoFormat>,
        produce: Vec<RawVideoFormat>,
        shapes: Vec<RawVideoShape>,
    },
    Audio {
        accept: Vec<AudioFormat>,
        produce: Vec<AudioFormat>,
        shapes: Vec<AudioShape>,
    },
}

impl CapsTransform {
    /// Forward derivation: the ordered output alternatives for `input`.
    pub fn derive(&self, input: &Caps) -> CapsSet {
        let mut alts: Vec<Caps> = Vec::new();
        match (self, input) {
            (
                Self::RawVideo {
                    accept,
                    produce,
                    shapes,
                },
                Caps::RawVideo {
                    format,
                    width,
                    height,
                    framerate,
                    interlace,
                    ..
                },
            ) => {
                if !accept.is_empty() && !accept.contains(format) {
                    return CapsSet::from_alternatives(Vec::new());
                }
                for s in shapes {
                    let f = s.format.apply(format);
                    if !produce.is_empty() && !produce.contains(&f) {
                        continue;
                    }
                    push_unique(
                        &mut alts,
                        Caps::RawVideo {
                            format: f,
                            width: s.width.apply(width),
                            height: s.height.apply(height),
                            framerate: s.framerate.apply(framerate),
                            // A format/geometry reshape leaves scan structure alone.
                            interlace: *interlace,
                            // Unknown, not passthrough: a format reshape (YUV ->
                            // RGB) changes what colorimetry means for the output.
                            colorimetry: crate::Colorimetry::UNKNOWN,
                        },
                    );
                }
            }
            (
                Self::Audio {
                    accept,
                    produce,
                    shapes,
                },
                Caps::Audio {
                    format,
                    channels,
                    sample_rate,
                },
            ) => {
                if !accept.is_empty() && !accept.contains(format) {
                    return CapsSet::from_alternatives(Vec::new());
                }
                for s in shapes {
                    let f = s.format.apply(format);
                    if !produce.is_empty() && !produce.contains(&f) {
                        continue;
                    }
                    push_unique(
                        &mut alts,
                        Caps::Audio {
                            format: f,
                            channels: s.channels.apply(channels),
                            sample_rate: s.sample_rate.apply(sample_rate),
                        },
                    );
                }
            }
            _ => {}
        }
        CapsSet::from_alternatives(alts)
    }

    /// Which fields the solver may couple backward: those every output shape
    /// derives with [`FieldTransform::Identity`]. A field one alternative
    /// retargets is not coupled, whichever alternative the solve picks. No
    /// shapes means nothing derives, so nothing couples.
    pub fn passthrough(&self) -> PassthroughFields {
        match self {
            Self::RawVideo { shapes, .. } if !shapes.is_empty() => PassthroughFields {
                format: shapes.iter().all(|s| s.format.is_identity()),
                width: shapes.iter().all(|s| s.width.is_identity()),
                height: shapes.iter().all(|s| s.height.is_identity()),
                framerate: shapes.iter().all(|s| s.framerate.is_identity()),
                channels: false,
                sample_rate: false,
            },
            Self::Audio { shapes, .. } if !shapes.is_empty() => PassthroughFields {
                format: shapes.iter().all(|s| s.format.is_identity()),
                width: false,
                height: false,
                framerate: false,
                channels: shapes.iter().all(|s| s.channels.is_identity()),
                sample_rate: shapes.iter().all(|s| s.sample_rate.is_identity()),
            },
            _ => PassthroughFields::NONE,
        }
    }
}

fn push_unique(alts: &mut Vec<Caps>, c: Caps) {
    if !alts.contains(&c) {
        alts.push(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::ANY_SAMPLE_RATE;
    use alloc::vec;

    fn raw(format: RawVideoFormat, w: u32, h: u32, fps: u32) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(fps << 16),
            interlace: crate::Interlace::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
        }
    }

    fn pcm(format: AudioFormat, channels: u8, sample_rate: u32) -> Caps {
        Caps::Audio {
            format,
            channels,
            sample_rate,
        }
    }

    fn video(accept: &[RawVideoFormat], shapes: Vec<RawVideoShape>) -> CapsTransform {
        CapsTransform::RawVideo {
            accept: accept.to_vec(),
            produce: Vec::new(),
            shapes,
        }
    }

    #[test]
    fn identity_shape_derives_the_input_and_couples_every_field() {
        let t = video(&[RawVideoFormat::Nv12], vec![RawVideoShape::PASSTHROUGH]);
        let inp = raw(RawVideoFormat::Nv12, 320, 240, 30);
        assert_eq!(t.derive(&inp).alternatives(), core::slice::from_ref(&inp));
        let pt = t.passthrough();
        assert!(pt.format && pt.width && pt.height && pt.framerate);
    }

    #[test]
    fn fixed_shape_retargets_the_field_and_drops_it_from_the_mask() {
        let t = video(
            &[RawVideoFormat::Nv12],
            vec![RawVideoShape::PASSTHROUGH
                .with_width(FieldTransform::Fixed(Dim::Fixed(64)))
                .with_height(FieldTransform::Fixed(Dim::Fixed(32)))],
        );
        assert_eq!(
            t.derive(&raw(RawVideoFormat::Nv12, 320, 240, 30))
                .alternatives(),
            &[raw(RawVideoFormat::Nv12, 64, 32, 30)]
        );
        let pt = t.passthrough();
        assert!(
            pt.format && pt.framerate,
            "format + rate still pass through"
        );
        assert!(
            !pt.width && !pt.height,
            "retargeted geometry is not coupled"
        );
    }

    #[test]
    fn unaccepted_input_derives_nothing() {
        let t = video(&[RawVideoFormat::Nv12], vec![RawVideoShape::PASSTHROUGH]);
        assert!(t
            .derive(&raw(RawVideoFormat::Rgba8, 320, 240, 30))
            .is_empty());
        // Wrong media kind entirely.
        assert!(t.derive(&pcm(AudioFormat::PcmS16Le, 2, 48_000)).is_empty());
        // An empty accept list takes any format of the kind.
        let any = video(&[], vec![RawVideoShape::PASSTHROUGH]);
        assert!(!any
            .derive(&raw(RawVideoFormat::Rgba8, 320, 240, 30))
            .is_empty());
    }

    #[test]
    fn produce_gate_drops_an_input_only_passthrough() {
        // Yuyv in, never out: the passthrough alternative is filtered, so the
        // derived set is the producible format only.
        let t = CapsTransform::RawVideo {
            accept: vec![RawVideoFormat::Yuyv, RawVideoFormat::Nv12],
            produce: vec![RawVideoFormat::Nv12],
            shapes: vec![
                RawVideoShape::PASSTHROUGH,
                RawVideoShape::PASSTHROUGH.with_format(FieldTransform::Fixed(RawVideoFormat::Nv12)),
            ],
        };
        assert_eq!(
            t.derive(&raw(RawVideoFormat::Yuyv, 320, 240, 30))
                .alternatives(),
            &[raw(RawVideoFormat::Nv12, 320, 240, 30)]
        );
        // An Nv12 input hits the passthrough first, and the duplicate retarget
        // collapses into it.
        assert_eq!(
            t.derive(&raw(RawVideoFormat::Nv12, 320, 240, 30))
                .alternatives(),
            &[raw(RawVideoFormat::Nv12, 320, 240, 30)]
        );
    }

    #[test]
    fn audio_shapes_derive_channels_and_rate() {
        let t = CapsTransform::Audio {
            accept: vec![AudioFormat::PcmS16Le],
            produce: Vec::new(),
            shapes: vec![
                AudioShape::PASSTHROUGH,
                AudioShape::PASSTHROUGH.with_sample_rate(FieldTransform::Fixed(ANY_SAMPLE_RATE)),
            ],
        };
        assert_eq!(
            t.derive(&pcm(AudioFormat::PcmS16Le, 2, 44_100))
                .alternatives(),
            &[
                pcm(AudioFormat::PcmS16Le, 2, 44_100),
                pcm(AudioFormat::PcmS16Le, 2, ANY_SAMPLE_RATE)
            ]
        );
        let pt = t.passthrough();
        assert!(pt.format && pt.channels);
        assert!(!pt.sample_rate, "one alternative retargets the rate");
    }

    #[test]
    fn no_shapes_derives_nothing_and_couples_nothing() {
        // How an element declares an invalid configuration (videorate with a
        // non-positive target): the solve fails loud instead of fixating.
        let t = video(&[RawVideoFormat::Nv12], Vec::new());
        assert!(t
            .derive(&raw(RawVideoFormat::Nv12, 320, 240, 30))
            .is_empty());
        assert_eq!(t.passthrough(), PassthroughFields::NONE);
    }
}
