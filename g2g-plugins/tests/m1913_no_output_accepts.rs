//! M1913: when an element derives no output from its input, the error lists
//! what the element accepts and names the converter that would fix the line.
#![cfg(all(feature = "std", unix))]

use std::process::Command;

use g2g_core::runtime::parse_launch;
use g2g_core::NodeId;
use g2g_plugins::registry::default_registry;

const ENCODER: &str = "y4menc";

// NV12 is semi-planar, and a y4m file holds planar YUV only
const LINE: &str = "videotestsrc num-buffers=1 ! video/x-raw,format=NV12 ! y4menc ! fakesink";

const HINT: &str = "hint: put ";

fn launch_stderr(line: &str) -> (bool, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .args(line.split(' '))
        .output()
        .expect("g2g-launch runs");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn label_of(line: &str, index: u32) -> &'static str {
    parse_launch(&default_registry(), line)
        .expect("the line parses")
        .element(NodeId(index))
        .expect("a linear line has no tee")
        .log_category()
}

#[test]
fn the_error_lists_what_the_encoder_accepts() {
    let encoder = label_of(LINE, 2);
    let accepted = default_registry()
        .declared_input_caps(ENCODER)
        .expect("the encoder declares its input");

    let (success, stderr) = launch_stderr(LINE);
    assert!(!success, "{stderr}");
    let accepts_line = stderr
        .lines()
        .find(|l| l.contains(&format!("{encoder} accepts: ")))
        .unwrap_or_else(|| panic!("no accepts line:\n{stderr}"));
    let first = accepted.alternatives()[0].to_gst_string();
    assert!(accepts_line.contains(&first), "{accepts_line}");
}

#[test]
fn the_hinted_converter_makes_the_line_run() {
    let (_, stderr) = launch_stderr(LINE);
    let hint = stderr
        .lines()
        .find_map(|l| l.split_once(HINT).map(|(_, rest)| rest))
        .unwrap_or_else(|| panic!("no hint:\n{stderr}"));
    let converter = hint.split(' ').next().expect("the hint names a converter");

    let fixed = LINE.replace(
        &format!("! {ENCODER}"),
        &format!("! {converter} ! {ENCODER}"),
    );
    let (success, stderr) = launch_stderr(&fixed);
    assert!(success, "`{fixed}` runs:\n{stderr}");
}

#[test]
fn audio_into_a_video_element_gets_no_converter_hint() {
    let (success, stderr) = launch_stderr("audiotestsrc num-buffers=1 ! videoflip ! fakesink");
    assert!(!success, "{stderr}");
    assert!(stderr.contains("produces no output for"), "{stderr}");
    assert!(!stderr.contains(HINT), "{stderr}");
}
