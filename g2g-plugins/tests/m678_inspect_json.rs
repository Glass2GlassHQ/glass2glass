//! M678 machine-readable registry dump: `g2g-inspect --json` emits the element
//! catalog (identity, role, pads, typed properties) the visual pipeline builder
//! and the MCP server consume.
//!
//! Runs the built binary end to end (the JSON shaping lives in the bin). Needs
//! the `tooling-json` feature: `cargo test -p g2g-plugins --features tooling-json
//! --test m678_inspect_json`.
#![cfg(feature = "tooling-json")]

use std::process::Command;

fn inspect(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_g2g-inspect"))
        .args(args)
        .output()
        .expect("run g2g-inspect");
    assert!(out.status.success(), "g2g-inspect {args:?} failed: {:?}", out);
    String::from_utf8(out.stdout).expect("utf8 output")
}

#[test]
fn json_single_element_has_typed_properties() {
    let json = inspect(&["--json", "videoscale"]);
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    let elements = v["elements"].as_array().expect("elements array");
    assert_eq!(elements.len(), 1);
    let e = &elements[0];
    assert_eq!(e["name"], "videoscale");
    assert_eq!(e["role"], "element");
    assert!(e["pads"].as_array().unwrap().len() >= 2, "SINK + SRC pad templates");

    let props = e["properties"].as_array().expect("properties array");
    let names: Vec<&str> = props.iter().map(|p| p["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"width") && names.contains(&"height"));
    let width = props.iter().find(|p| p["name"] == "width").unwrap();
    assert_eq!(width["type"], "Unsigned Integer", "machine type for typed inputs");
    assert_eq!(width["writable"], true);
}

#[test]
fn json_full_catalog_covers_all_roles() {
    let json = inspect(&["--json"]);
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    let elements = v["elements"].as_array().expect("elements array");
    assert!(elements.len() > 20, "the standard registry has many elements");

    let roles: std::collections::HashSet<&str> =
        elements.iter().map(|e| e["role"].as_str().unwrap()).collect();
    assert!(roles.contains("source"), "sources present");
    assert!(roles.contains("element"), "transforms / sinks present");

    // A source advertises output caps; a transform advertises pad templates.
    let src = elements.iter().find(|e| e["name"] == "videotestsrc").unwrap();
    assert!(!src["caps"].is_null(), "source carries output caps");
}

#[test]
fn json_unknown_element_errors() {
    let out = Command::new(env!("CARGO_BIN_EXE_g2g-inspect"))
        .args(["--json", "nope_not_an_element"])
        .output()
        .expect("run g2g-inspect");
    assert!(!out.status.success(), "unknown element is a failure exit");
}
