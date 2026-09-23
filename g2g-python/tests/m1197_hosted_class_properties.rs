//! M1197: a hosted element whose class answers `g2g_properties` lists that
//! class's properties in its inspection dump, and a launch line naming one the
//! class lacks is refused at parse. Needs libpython (`python` feature).
#![cfg(feature = "python")]

use g2g_core::runtime::{parse_launch, ParseError, Registry};
use pyo3::prelude::*;

const FIXTURE_MODULE: &str = "echo_element";
const DECLARING_CLASS: &str = "DeclaredProps";
const DECLARED_PROPERTIES_HOOK: &str = "g2g_properties";
const UNDECLARED_PROPERTY: &str = "no-such-property";
const PROPERTIES_HEADER: &str = "Element Properties:";

fn use_fixtures() {
    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );
}

/// The fixture class's own answer, as the gst names a launch line spells.
fn fixture_property_names() -> Vec<String> {
    g2g_python::init_host();
    let attributes: Vec<String> = Python::attach(|py| {
        PyModule::import(py, FIXTURE_MODULE)?
            .getattr(DECLARING_CLASS)?
            .call0()?
            .call_method0(DECLARED_PROPERTIES_HOOK)?
            .extract()
    })
    .expect("fixture class declares its properties");
    attributes
        .into_iter()
        .map(|attribute| attribute.replace('_', "-"))
        .collect()
}

fn registry() -> Registry {
    let mut registry = g2g_plugins::registry::default_registry();
    g2g_python::register(&mut registry);
    registry
}

/// What `g2g-launch-py --inspect <element> module=... class=...` prints, for
/// both hosted elements that forward properties to their class.
#[test]
fn launcher_inspection_lists_the_hosted_class_properties() {
    use_fixtures();
    let declared = fixture_property_names();
    assert!(!declared.is_empty(), "the fixture declares properties");
    let registry = registry();
    for element in [g2g_python::PYELEMENT, g2g_python::PYAGGREGATOR] {
        let dump =
            g2g_python::inspect_hosted_class(&registry, element, FIXTURE_MODULE, DECLARING_CLASS)
                .unwrap_or_else(|| panic!("{element} is a hosted element"));
        let plain = registry.inspect(element).expect("registered");
        assert_eq!(
            dump.split_once(PROPERTIES_HEADER).map(|(head, _)| head),
            plain.split_once(PROPERTIES_HEADER).map(|(head, _)| head),
            "{element} keeps the rest of the g2g-inspect dump"
        );
        for name in &declared {
            assert!(
                dump.contains(&format!("  {name}:")),
                "{element} inspection lists {name}:\n{dump}"
            );
        }
    }
}

#[test]
fn launch_refuses_a_property_the_hosted_class_lacks() {
    use_fixtures();
    let registry = registry();
    let line = format!(
        "videotestsrc ! videoconvert ! pyelement module={FIXTURE_MODULE} \
         class={DECLARING_CLASS} {UNDECLARED_PROPERTY}=1 ! fakesink"
    );
    let result = parse_launch(&registry, &line);
    assert!(
        matches!(
            &result,
            Err(ParseError::UnknownProperty { key, .. }) if key == UNDECLARED_PROPERTY
        ),
        "a name the class does not declare is refused at parse: {:?}",
        result.err()
    );
}
