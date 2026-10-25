//! M1915: the display and WebRTC sinks declare the caps they accept as a sink
//! pad template, so `g2g-inspect` lists them.
#![cfg(feature = "std")]

#[cfg(any(
    all(target_os = "linux", feature = "wayland-sink"),
    all(target_os = "linux", feature = "kms-sink"),
    feature = "webrtc"
))]
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
fn waylandsink_declares_its_input() {
    assert_template_matches_constraint("waylandsink");
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
