// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M1915: the display and WebRTC sinks declare the caps they accept as a sink
//! pad template, so `g2g-inspect` lists them.
#![cfg(feature = "std")]

#[cfg(any(all(target_os = "linux", feature = "kms-sink"), feature = "webrtc"))]
fn assert_template_matches_constraint(name: &str) {
    use g2g_core::CapsConstraint;
    let reg = g2g_plugins::registry::default_registry();
    let element = reg.make_element(name).expect("the sink is registered");
    let CapsConstraint::Accepts(accepted) = element.caps_constraint_as_sink() else {
        panic!("{name} negotiates with an accept set");
    };
    assert_eq!(
        reg.declared_input_caps(name),
        Some(accepted),
        "{name}'s sink template is what it negotiates with"
    );
    let dump = reg.inspect(name).expect("the sink is inspectable");
    assert!(dump.contains("SINK:"), "{name} lists a sink pad:\n{dump}");
}

#[cfg(all(target_os = "linux", feature = "wayland-sink"))]
#[test]
fn waylandsink_negotiates_within_its_template() {
    use g2g_core::{Caps, CapsConstraint, RawVideoFormat};
    let reg = g2g_plugins::registry::default_registry();
    let template = reg
        .declared_input_caps("waylandsink")
        .expect("waylandsink declares a sink template");
    assert!(
        template.alternatives().iter().any(|caps| matches!(
            caps,
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                ..
            }
        )),
        "the template lists NV12"
    );
    // waylandsink accepts the formats the running compositor offers, none without one
    let element = reg
        .make_element("waylandsink")
        .expect("the sink is registered");
    let CapsConstraint::Accepts(accepted) = element.caps_constraint_as_sink() else {
        panic!("waylandsink negotiates with an accept set");
    };
    for caps in accepted.alternatives() {
        assert!(
            template.alternatives().contains(caps),
            "{caps:?} is outside the template"
        );
    }
    let dump = reg.inspect("waylandsink").expect("the sink is inspectable");
    assert!(
        dump.contains("SINK:"),
        "waylandsink lists a sink pad:\n{dump}"
    );
}

#[cfg(all(target_os = "linux", feature = "kms-sink"))]
#[test]
fn kmssink_declares_its_input() {
    assert_template_matches_constraint("kmssink");
}

#[cfg(feature = "webrtc")]
#[test]
fn webrtcsink_declares_its_input() {
    assert_template_matches_constraint("webrtcsink");
}
