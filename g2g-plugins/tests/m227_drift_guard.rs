//! M227 drift guard: the `DerivedCoupled` invariant. A caps-driven transform
//! declares a `PassthroughFields` mask *and* a forward closure; the solver's
//! backward field-coupling trusts the mask, so the closure must never alter a
//! field the mask claims is passed through. This test fails loud if the two ever
//! drift apart (e.g. a future edit retargets a field but forgets to drop it from
//! the mask, or vice versa). It is the realizable robustness benefit of a
//! single-source-of-truth descriptor without the closure-free refactor.

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
    }
}

fn raw(format: RawVideoFormat, w: u32, h: u32) -> Caps {
    Caps::RawVideo { format, width: Dim::Fixed(w), height: Dim::Fixed(h), framerate: Rate::Fixed(30 << 16) }
}

fn pcm(rate: u32) -> Caps {
    Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: rate }
}

/// Assert every field the mask declares passthrough is identical in `out` and
/// `inp` (and the media variant is unchanged).
fn assert_passthrough_preserved(inp: &Caps, out: &Caps, mask: PassthroughFields) {
    match (inp, out) {
        (
            Caps::RawVideo { format: fi, width: wi, height: hi, framerate: ri },
            Caps::RawVideo { format: fo, width: wo, height: ho, framerate: ro },
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
            Caps::Audio { format: fi, channels: ci, sample_rate: si },
            Caps::Audio { format: fo, channels: co, sample_rate: so },
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

/// Drive the element's `DerivedCoupled` closure with `inputs` and check the mask
/// never lies: every produced alternative preserves the declared passthrough
/// fields. Also asserts each valid input yields at least one alternative.
fn check<E: AsyncElement>(element: &E, inputs: &[Caps]) {
    let CapsConstraint::DerivedCoupled { derive, passthrough } =
        element.caps_constraint_as_transform()
    else {
        panic!("expected a DerivedCoupled constraint");
    };
    for inp in inputs {
        let out = derive(inp);
        assert!(!out.is_empty(), "valid input {inp:?} produced no output");
        for alt in out.alternatives() {
            assert_passthrough_preserved(inp, alt, passthrough);
        }
    }
}

#[test]
fn videoscale_closure_honors_its_passthrough_mask() {
    let inputs = [rgba(320, 240), raw(RawVideoFormat::Nv12, 320, 240)];
    check(&VideoScale::new(0, 0), &inputs); // auto
    check(&VideoScale::new(64, 32), &inputs); // property-driven target
}

#[test]
fn videoconvert_closure_honors_its_passthrough_mask() {
    // Includes a Yuyv input (input-only): its outputs are the producible formats
    // at the *same* geometry, so width/height/framerate must stay passthrough.
    let inputs = [rgba(320, 240), raw(RawVideoFormat::Yuyv, 320, 240)];
    check(&VideoConvert::auto(), &inputs);
    check(&VideoConvert::new(RawVideoFormat::Nv12), &inputs);
}

#[test]
fn audioresample_closure_honors_its_passthrough_mask() {
    let inputs = [pcm(44_100), pcm(48_000)];
    check(&AudioResample::auto(), &inputs);
    check(&AudioResample::new(16_000), &inputs);
}
