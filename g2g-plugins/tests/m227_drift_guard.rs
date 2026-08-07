//! M227 drift guard: the coupling invariant behind the solver's backward
//! field-narrowing. A caps-driven transform's passthrough mask must never claim
//! a field its forward derivation alters. Since M837 both come from one
//! `CapsTransform` declaration (the mask is the fields every output shape
//! derives with `Identity`), so this drives the real elements' declarations and
//! checks the property end to end.

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, Dim, PassthroughFields, Rate, RawVideoFormat,
};
use g2g_plugins::audioresample::AudioResample;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videoscale::VideoScale;

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn raw(format: RawVideoFormat, w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn pcm(rate: u32) -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 2,
        sample_rate: rate,
    }
}

/// Assert every field the mask declares passthrough is identical in `out` and
/// `inp` (and the media variant is unchanged).
fn assert_passthrough_preserved(inp: &Caps, out: &Caps, mask: PassthroughFields) {
    match (inp, out) {
        (
            Caps::RawVideo {
                format: fi,
                width: wi,
                height: hi,
                framerate: ri,
                interlace: _,
            },
            Caps::RawVideo {
                format: fo,
                width: wo,
                height: ho,
                framerate: ro,
                interlace: _,
            },
        ) => {
            if mask.format {
                assert_eq!(fi, fo, "format declared passthrough but changed");
            }
            if mask.width {
                assert_eq!(wi, wo, "width declared passthrough but changed");
            }
            if mask.height {
                assert_eq!(hi, ho, "height declared passthrough but changed");
            }
            if mask.framerate {
                assert_eq!(ri, ro, "framerate declared passthrough but changed");
            }
        }
        (
            Caps::Audio {
                format: fi,
                channels: ci,
                sample_rate: si,
            },
            Caps::Audio {
                format: fo,
                channels: co,
                sample_rate: so,
            },
        ) => {
            if mask.format {
                assert_eq!(fi, fo, "format declared passthrough but changed");
            }
            if mask.channels {
                assert_eq!(ci, co, "channels declared passthrough but changed");
            }
            if mask.sample_rate {
                assert_eq!(si, so, "sample_rate declared passthrough but changed");
            }
        }
        _ => panic!("media variant changed across the transform"),
    }
}

/// Drive the element's declared transform with `inputs` and check the mask never
/// lies: every derived alternative preserves the passthrough fields. Also
/// asserts each valid input yields at least one alternative.
fn check<E: AsyncElement>(element: &E, inputs: &[Caps]) {
    let CapsConstraint::DerivedFields(transform) = element.caps_constraint_as_transform() else {
        panic!("expected a DerivedFields constraint");
    };
    let passthrough = transform.passthrough();
    for inp in inputs {
        let out = transform.derive(inp);
        assert!(!out.is_empty(), "valid input {inp:?} produced no output");
        for alt in out.alternatives() {
            assert_passthrough_preserved(inp, alt, passthrough);
        }
    }
}

#[test]
fn videoscale_derivation_honors_its_passthrough_mask() {
    let inputs = [rgba(320, 240), raw(RawVideoFormat::Nv12, 320, 240)];
    check(&VideoScale::new(0, 0), &inputs); // auto
    check(&VideoScale::new(64, 32), &inputs); // property-driven target
}

#[test]
fn videoconvert_derivation_honors_its_passthrough_mask() {
    // Includes a Yuyv input (input-only): its outputs are the producible formats
    // at the *same* geometry, so width/height/framerate must stay passthrough.
    let inputs = [rgba(320, 240), raw(RawVideoFormat::Yuyv, 320, 240)];
    check(&VideoConvert::auto(), &inputs);
    check(&VideoConvert::new(RawVideoFormat::Nv12), &inputs);
}

#[test]
fn audioresample_derivation_honors_its_passthrough_mask() {
    let inputs = [pcm(44_100), pcm(48_000)];
    check(&AudioResample::auto(), &inputs);
    check(&AudioResample::new(16_000), &inputs);
}
