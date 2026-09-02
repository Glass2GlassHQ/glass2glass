//! M837: closure-free field transforms. A caps-driven transform declares its
//! forward derivation as data (`CapsTransform`: an ordered list of output shapes,
//! each field `Identity` / `Fixed` / `Scale`) instead of a closure plus a
//! hand-written passthrough mask. The solver reads the mask off the declaration,
//! so the two can no longer disagree, and the negotiation the migrated elements
//! produce is unchanged.
//!
//! `solve_linear` and the element registry are `std`-gated, so this file is too.
#![cfg(feature = "std")]

use g2g_core::runtime::solver::solve_linear;
use g2g_core::{
    AsyncElement, AudioFormat, AudioShape, Caps, CapsConstraint, CapsSet, CapsTransform, Dim,
    FieldTransform, PassthroughFields, Rate, RawVideoFormat, RawVideoShape,
};
use g2g_plugins::audioconvert::AudioConvert;
use g2g_plugins::audioresample::AudioResample;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videorate::VideoRate;
use g2g_plugins::videoscale::VideoScale;

const FPS30: Rate = Rate::Fixed(30 << 16);

fn raw(format: RawVideoFormat, w: u32, h: u32, framerate: Rate) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate,
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn pcm(format: AudioFormat, channels: u8, sample_rate: u32) -> Caps {
    Caps::Audio {
        format,
        channels,
        sample_rate,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

fn video_transform(shapes: Vec<RawVideoShape>) -> CapsTransform {
    CapsTransform::RawVideo {
        accept: Vec::new(),
        produce: Vec::new(),
        shapes,
    }
}

/// The declared transform of an element that has migrated to `DerivedFields`.
fn transform_of<E: AsyncElement>(element: &E) -> CapsTransform {
    match element.caps_constraint_as_transform() {
        CapsConstraint::DerivedFields(t) => t,
        other => panic!("expected DerivedFields, got {other:?}"),
    }
}

#[test]
fn each_declarative_shape_derives_its_field() {
    let inp = raw(RawVideoFormat::Nv12, 320, 240, FPS30);

    // Identity: the input field, verbatim.
    let identity = video_transform(vec![RawVideoShape::PASSTHROUGH]);
    assert_eq!(
        identity.derive(&inp).alternatives(),
        std::slice::from_ref(&inp)
    );

    // Fixed: a constant, including a ranged one a downstream pin narrows.
    let fixed = video_transform(vec![RawVideoShape::PASSTHROUGH
        .with_width(FieldTransform::Fixed(Dim::Fixed(64)))
        .with_height(FieldTransform::Fixed(Dim::Fixed(48)))]);
    assert_eq!(
        fixed.derive(&inp).alternatives(),
        &[raw(RawVideoFormat::Nv12, 64, 48, FPS30)]
    );
    let ranged = video_transform(vec![RawVideoShape::PASSTHROUGH.with_framerate(
        FieldTransform::Fixed(Rate::Range {
            min_q16: 1 << 16,
            max_q16: 60 << 16,
        }),
    )]);
    assert_eq!(
        ranged.derive(&inp).alternatives(),
        &[Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Range {
                min_q16: 1 << 16,
                max_q16: 60 << 16,
            },
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN
        }]
    );

    // Audio fields derive the same way.
    let audio = CapsTransform::Audio {
        accept: Vec::new(),
        produce: Vec::new(),
        shapes: vec![AudioShape::PASSTHROUGH
            .with_channels(FieldTransform::Fixed(1))
            .with_sample_rate(FieldTransform::Fixed(24_000))],
    };
    assert_eq!(
        audio
            .derive(&pcm(AudioFormat::PcmS16Le, 2, 48_000))
            .alternatives(),
        &[pcm(AudioFormat::PcmS16Le, 1, 24_000)]
    );
}

#[test]
fn a_retargeted_field_can_never_be_declared_passthrough() {
    // The drift a separate mask allowed: claiming `width` passthrough while the
    // derivation retargets it, which made the solver narrow an input field the
    // transform rewrites. There is no mask to state, so adding the retargeting
    // alternative removes width from the coupling *and* from the derivation in
    // one step.
    let passthrough_only = video_transform(vec![RawVideoShape::PASSTHROUGH]);
    assert!(passthrough_only.passthrough().width);

    let with_retarget = video_transform(vec![
        RawVideoShape::PASSTHROUGH,
        RawVideoShape::PASSTHROUGH
            .with_width(FieldTransform::Fixed(Dim::Range { min: 1, max: 32768 })),
    ]);
    let mask = with_retarget.passthrough();
    assert!(
        !mask.width,
        "one alternative retargets width, so it is not coupled"
    );
    assert!(mask.format && mask.height && mask.framerate);

    // And the mask matches what the derivation actually does: every field the
    // mask claims is identical in every derived alternative.
    let inp = raw(RawVideoFormat::Nv12, 320, 240, FPS30);
    for alt in with_retarget.derive(&inp).alternatives() {
        let (
            Caps::RawVideo {
                height,
                framerate,
                format,
                ..
            },
            Caps::RawVideo {
                height: hi,
                framerate: ri,
                format: fi,
                ..
            },
        ) = (alt, &inp)
        else {
            panic!("raw video in, raw video out");
        };
        assert_eq!((format, height, framerate), (fi, hi, ri));
    }
}

#[test]
fn migrated_elements_couple_exactly_the_fields_their_shapes_pass_through() {
    // Each element's mask is a function of its declared shapes, so this pins the
    // coupling the solver gets for the five migrated elements.
    let cases: Vec<(&str, CapsTransform, PassthroughFields)> = vec![
        (
            "videoscale(auto)",
            transform_of(&VideoScale::new(0, 0)),
            PassthroughFields::NONE.with_format().with_framerate(),
        ),
        (
            "videoscale(160x120)",
            transform_of(&VideoScale::new(160, 120)),
            PassthroughFields::NONE.with_format().with_framerate(),
        ),
        (
            "videoconvert(auto)",
            transform_of(&VideoConvert::auto()),
            PassthroughFields::NONE
                .with_width()
                .with_height()
                .with_framerate(),
        ),
        (
            "videorate(auto)",
            transform_of(&VideoRate::auto()),
            PassthroughFields::NONE
                .with_format()
                .with_width()
                .with_height(),
        ),
        (
            "videorate(10fps)",
            transform_of(&VideoRate::new(10.0)),
            PassthroughFields::NONE
                .with_format()
                .with_width()
                .with_height(),
        ),
        (
            "audioconvert(auto)",
            transform_of(&AudioConvert::auto()),
            PassthroughFields::NONE.with_sample_rate(),
        ),
        (
            "audioresample(48k)",
            transform_of(&AudioResample::new(48_000)),
            PassthroughFields::NONE.with_format().with_channels(),
        ),
    ];
    for (name, transform, expected) in cases {
        assert_eq!(transform.passthrough(), expected, "{name}");
    }
}

#[test]
fn geometry_pin_solves_back_through_scale_and_convert() {
    // The M227 chain on the declarative transforms: an NV12 160x120 pin two hops
    // downstream fixates the scaler's output geometry (coupled backward through
    // videoconvert's passthrough geometry) and videoconvert's output format.
    let scale = VideoScale::new(0, 0);
    let convert = VideoConvert::auto();
    let src = CapsConstraint::Produces(CapsSet::one(raw(RawVideoFormat::Rgba8, 320, 240, FPS30)));
    let pin = CapsConstraint::Identity(CapsSet::one(Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(160),
        height: Dim::Fixed(120),
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }));
    let (sc, cv) = (
        scale.caps_constraint_as_transform(),
        convert.caps_constraint_as_transform(),
    );
    let sink = CapsConstraint::AcceptsAny;
    let solution = solve_linear(&[&src, &sc, &cv, &pin, &sink]).expect("chain solves");
    assert_eq!(
        solution,
        vec![
            raw(RawVideoFormat::Rgba8, 320, 240, FPS30),
            raw(RawVideoFormat::Rgba8, 160, 120, FPS30),
            raw(RawVideoFormat::Nv12, 160, 120, FPS30),
            raw(RawVideoFormat::Nv12, 160, 120, FPS30),
        ]
    );
}

#[test]
fn audio_rate_pin_solves_back_through_resample_and_convert() {
    // The audio mirror: audioresample retargets the rate, audioconvert passes it
    // through and retargets format/channels, so a 48 kHz + F32 pin resolves both.
    let resample = AudioResample::auto();
    let convert = AudioConvert::auto();
    let src = CapsConstraint::Produces(CapsSet::one(pcm(AudioFormat::PcmS16Le, 2, 44_100)));
    let pin = CapsConstraint::Identity(CapsSet::one(pcm(AudioFormat::PcmF32Le, 2, 48_000)));
    let (rs, cv) = (
        resample.caps_constraint_as_transform(),
        convert.caps_constraint_as_transform(),
    );
    let sink = CapsConstraint::AcceptsAny;
    let solution = solve_linear(&[&src, &rs, &cv, &pin, &sink]).expect("chain solves");
    assert_eq!(
        solution,
        vec![
            pcm(AudioFormat::PcmS16Le, 2, 44_100),
            pcm(AudioFormat::PcmS16Le, 2, 48_000),
            pcm(AudioFormat::PcmF32Le, 2, 48_000),
            pcm(AudioFormat::PcmF32Le, 2, 48_000),
        ]
    );
}

#[test]
fn an_impossible_target_declares_no_shape_and_fails_loud() {
    // videoscale with an odd target for a 4:2:0 format derives nothing (the
    // format is not accepted), so the solve fails rather than fixating caps the
    // element cannot produce.
    let scale = VideoScale::new(63, 32);
    let t = transform_of(&scale);
    assert!(t
        .derive(&raw(RawVideoFormat::Nv12, 320, 240, FPS30))
        .is_empty());
    assert!(!t
        .derive(&raw(RawVideoFormat::Rgba8, 320, 240, FPS30))
        .is_empty());

    // videorate with a non-positive target declares no output shape at all.
    let rate = VideoRate::new(0.0);
    let t = transform_of(&rate);
    assert!(t
        .derive(&raw(RawVideoFormat::Nv12, 320, 240, FPS30))
        .is_empty());
    assert_eq!(t.passthrough(), PassthroughFields::NONE);
}
