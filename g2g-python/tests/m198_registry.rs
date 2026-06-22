//! M198 step 4: `pyelement` as a first-class `gst-launch` / autoplug element.
//!
//! Proves the registry factory + property parsing: a hosted Python element is
//! instantiable by name in a launch line, with `module=` / `class=` /
//! `draw-label=` applied through the property system. Parse-level only (no
//! interpreter), so it runs on the default build.

use g2g_core::runtime::parse_launch;

fn registry() -> g2g_core::runtime::Registry {
    let mut reg = g2g_plugins::registry::default_registry();
    g2g_python::register(&mut reg);
    reg
}

#[test]
fn pyelement_parses_as_a_first_class_element() {
    let reg = registry();
    let line = "videotestsrc ! videoconvert ! \
                pyelement module=echo_element class=EchoTransform draw-label=true ! fakesink";
    let graph = parse_launch(&reg, line);
    assert!(graph.is_ok(), "pyelement should parse in a launch line: {:?}", graph.err());
}

#[test]
fn pyelement_unknown_property_is_rejected() {
    let reg = registry();
    let line = "videotestsrc ! videoconvert ! pyelement module=m class=C bogus=1 ! fakesink";
    // An unknown property name surfaces as a parse error (set_property -> Unknown),
    // proving properties are actually routed to the element, not ignored.
    assert!(parse_launch(&reg, line).is_err(), "unknown property should be rejected");
}

#[test]
fn pyelement_bad_bool_value_is_rejected() {
    let reg = registry();
    let line = "videotestsrc ! videoconvert ! \
                pyelement module=m class=C draw-label=notabool ! fakesink";
    // draw-label is a Bool; a non-bool text fails parsing/validation.
    assert!(parse_launch(&reg, line).is_err(), "a bad draw-label value should be rejected");
}
