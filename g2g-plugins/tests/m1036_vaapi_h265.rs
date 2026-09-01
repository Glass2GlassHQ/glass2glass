//! M1036: the H.265 sibling of the cros-codecs VAAPI decoder, and the
//! codec-agnostic negotiation both elements share.
//!
//! Hardware-free: nothing here opens a VA display, so it runs anywhere the
//! `vaapi` feature compiles. The decode path itself is covered by the
//! env-gated `vaapi_smoke` twins.
//!
//! ```sh
//! cargo test -p g2g-plugins --features vaapi --test m1036_vaapi_h265
//! ```

#![cfg(all(target_os = "linux", feature = "vaapi"))]

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, Dim, G2gError, PadTemplates, Rate, RawVideoFormat,
    VideoCodec,
};
use g2g_plugins::vaapidec::{VaapiH264Dec, VaapiH265Dec};

fn compressed(codec: VideoCodec, width: u32, height: u32) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(30 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

#[test]
fn h265_element_accepts_h265_and_rejects_h264() {
    let dec = VaapiH265Dec::new();
    let h265 = compressed(VideoCodec::H265, 1920, 1080);
    assert_eq!(dec.intercept_caps(&h265), Ok(h265));
    assert_eq!(
        dec.intercept_caps(&compressed(VideoCodec::H264, 1920, 1080)),
        Err(G2gError::CapsMismatch)
    );
}

#[test]
fn h264_element_still_rejects_h265() {
    let dec = VaapiH264Dec::new();
    let h264 = compressed(VideoCodec::H264, 1280, 720);
    assert_eq!(dec.intercept_caps(&h264), Ok(h264));
    assert_eq!(
        dec.intercept_caps(&compressed(VideoCodec::H265, 1280, 720)),
        Err(G2gError::CapsMismatch)
    );
}

#[test]
fn h265_derives_nv12_output_at_the_input_geometry() {
    let dec = VaapiH265Dec::new();
    let CapsConstraint::DerivedOutput(derive) = dec.caps_constraint_as_transform() else {
        panic!("expected DerivedOutput");
    };
    assert_eq!(
        derive(&compressed(VideoCodec::H265, 3840, 2160)).alternatives(),
        &[Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(3840),
            height: Dim::Fixed(2160),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN
        }]
    );
    // H.264 into the H.265 element has no solution.
    assert!(derive(&compressed(VideoCodec::H264, 3840, 2160)).is_empty());
}

#[test]
fn h265_pad_templates_are_h265_in_nv12_out() {
    use g2g_core::pad_template::{PadCaps, PadDirection};

    let alternatives = |direction: PadDirection| {
        let template = VaapiH265Dec::pad_templates()
            .into_iter()
            .find(|t| t.direction == direction)
            .expect("template for this direction");
        match template.caps {
            PadCaps::Fixed(set) => set.alternatives().to_vec(),
            other => panic!("expected a fixed caps set, got {other:?}"),
        }
    };

    assert!(alternatives(PadDirection::Sink).iter().any(|c| matches!(
        c,
        Caps::CompressedVideo {
            codec: VideoCodec::H265,
            ..
        }
    )));
    assert!(alternatives(PadDirection::Source).iter().any(|c| matches!(
        c,
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            ..
        }
    )));
}

#[test]
fn both_elements_take_system_memory_only() {
    use g2g_core::memory::{DomainSet, MemoryDomainKind};
    let expected = DomainSet::only(MemoryDomainKind::System);
    assert_eq!(VaapiH264Dec::new().input_domains(), expected);
    assert_eq!(VaapiH265Dec::new().input_domains(), expected);
}

#[test]
fn no_reconfigure_before_a_resolution_change() {
    let mut dec = VaapiH265Dec::new();
    assert_eq!(dec.take_reconfigure(), None);
}

#[test]
fn device_property_round_trips_on_the_h265_element() {
    use g2g_core::PropValue;
    let mut dec = VaapiH265Dec::new();
    dec.set_property("device", PropValue::Str("/dev/dri/renderD129".into()))
        .expect("device is a property");
    assert_eq!(
        dec.get_property("device"),
        Some(PropValue::Str("/dev/dri/renderD129".into()))
    );
}

#[test]
fn h265_decoder_is_registered_for_launch_lines() {
    let reg = g2g_plugins::registry::default_registry();
    assert!(reg.element_names().contains(&"vaapidech265"));
    assert_eq!(
        VaapiH265Dec::new().metadata().long_name,
        "VA-API H.265 decoder"
    );
}
