//! M1204: `g2g-launch-py --inspect`, `--docgen` and `--mcp` list the hosted
//! Python elements, which the g2g-plugins tools cannot see. Drives the built
//! binary: `cargo test -p g2g-python --features launch --test
//! m1204_launcher_tooling`.
#![cfg(feature = "launch")]

use std::io::Write;
use std::process::{Command, Output, Stdio};

use g2g_core::property::PropertySpec;
use g2g_core::{AsyncElement as _, MultiInputElement as _};
use g2g_python::{PyAggregator, PyTransform, PYAGGREGATOR, PYELEMENT};
use serde_json::Value;

/// Each hosted element with the properties it declares before a class loads.
fn hosted_elements() -> [(&'static str, &'static [PropertySpec]); 2] {
    [
        (PYELEMENT, PyTransform::new("", "").properties()),
        (PYAGGREGATOR, PyAggregator::new("", "", 1).properties()),
    ]
}

fn launcher(args: &[&str], stdin: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_g2g-launch-py"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn g2g-launch-py");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(stdin.as_bytes())
        .expect("write stdin");
    let output = child.wait_with_output().expect("wait for g2g-launch-py");
    assert!(
        output.status.success(),
        "g2g-launch-py {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn element_named<'a>(elements: &'a [Value], name: &str) -> &'a Value {
    elements
        .iter()
        .find(|element| element["name"] == name)
        .unwrap_or_else(|| panic!("{name} missing from the listing"))
}

#[test]
fn json_dump_lists_the_hosted_elements_with_their_properties() {
    let output = launcher(&["--inspect", "--json"], "");
    let dump: Value = serde_json::from_slice(&output.stdout).expect("dump is JSON");
    let elements = dump["elements"].as_array().expect("elements array");
    for (name, specs) in hosted_elements() {
        let listed: Vec<&str> = element_named(elements, name)["properties"]
            .as_array()
            .expect("properties array")
            .iter()
            .map(|property| property["name"].as_str().expect("property name"))
            .collect();
        let declared: Vec<&str> = specs.iter().map(|spec| spec.name).collect();
        assert!(!declared.is_empty(), "{name} declares properties");
        assert_eq!(listed, declared, "{name} properties");
    }
}

#[test]
fn element_reference_has_a_card_per_hosted_element() {
    let page = std::env::temp_dir().join(format!("m1204-elements-{}.html", std::process::id()));
    let page_path = page.to_str().expect("utf-8 temp path");
    launcher(&["--docgen", page_path], "");
    let html = std::fs::read_to_string(&page).expect("read generated page");
    std::fs::remove_file(&page).expect("remove generated page");
    for (name, specs) in hosted_elements() {
        let card = html
            .split("<article")
            .find(|card| card.contains(&format!("id=\"el-{name}\"")))
            .unwrap_or_else(|| panic!("no card for {name}"));
        for spec in specs {
            assert!(
                card.contains(&format!("<code class=\"pn\">{}</code>", spec.name)),
                "{name} card lacks property {}",
                spec.name
            );
        }
    }
}

#[test]
fn mcp_element_list_names_the_hosted_elements() {
    let request = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_elements","arguments":{}}}"#;
    let output = launcher(&["--mcp"], &format!("{request}\n"));
    let reply: Value = serde_json::from_slice(&output.stdout).expect("reply is JSON");
    let text = reply["result"]["content"][0]["text"]
        .as_str()
        .expect("tool result text");
    let listing: Value = serde_json::from_str(text).expect("listing is JSON");
    let elements = listing["elements"].as_array().expect("elements array");
    for (name, _) in hosted_elements() {
        element_named(elements, name);
    }
}
