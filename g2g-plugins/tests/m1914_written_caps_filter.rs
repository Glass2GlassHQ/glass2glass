//! M1914: a caps error shows a caps filter's side as the launch line wrote it,
//! and `G2G_CAPS_TRACE=trace` expands it into the alternatives it parses to.
#![cfg(all(feature = "std", unix))]

use std::process::Command;

use g2g_core::runtime::parse_launch;
use g2g_core::{CapsSet, NodeId};
use g2g_plugins::registry::default_registry;

// wider than videotestsrc draws, and no format, so the filter parses to one alternative per format
const WIDTH_FILTER: &str = "video/x-raw,width=100000";

// a y4m file holds planar YUV only, so y4menc derives nothing from NV12
const FORMAT_FILTER: &str = "video/x-raw,format=NV12";

const TRACE_VARIABLE: &str = "G2G_CAPS_TRACE";

fn launch_stderr(line: &str, trace: Option<&str>) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_g2g-launch"));
    command.args(line.split(' ')).env_remove(TRACE_VARIABLE);
    if let Some(level) = trace {
        command.env(TRACE_VARIABLE, level);
    }
    let output = command.output().expect("g2g-launch runs");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(!output.status.success(), "`{line}` fails:\n{stderr}");
    stderr
}

fn label_of(line: &str, index: u32) -> &'static str {
    parse_launch(&default_registry(), line)
        .expect("the line parses")
        .element(NodeId(index))
        .expect("a linear line has no tee")
        .log_category()
}

fn line_containing<'a>(stderr: &'a str, needle: &str) -> &'a str {
    stderr
        .lines()
        .find(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no line with `{needle}`:\n{stderr}"))
}

#[test]
fn the_filter_side_reads_as_written() {
    let line = format!("videotestsrc num-buffers=1 ! {WIDTH_FILTER} ! fakesink");
    let wants = format!("{} wants: ", label_of(&line, 1));
    let stderr = launch_stderr(&line, None);
    let wants_line = line_containing(&stderr, &wants);
    let shown = wants_line.split_once(&wants).expect("the wants line").1;
    assert!(shown.starts_with(WIDTH_FILTER), "{wants_line}");
    for caps in CapsSet::from_gst_string(WIDTH_FILTER)
        .expect("the filter parses")
        .alternatives()
    {
        assert!(!shown.contains(&caps.to_gst_string()), "{wants_line}");
    }
}

#[test]
fn trace_expands_the_filter_side() {
    let line = format!("videotestsrc num-buffers=1 ! {WIDTH_FILTER} ! fakesink");
    let wants = format!("{} wants: ", label_of(&line, 1));
    let stderr = launch_stderr(&line, Some("trace"));
    let wants_line = line_containing(&stderr, &wants);
    for caps in CapsSet::from_gst_string(WIDTH_FILTER)
        .expect("the filter parses")
        .alternatives()
    {
        assert!(wants_line.contains(&caps.to_gst_string()), "{wants_line}");
    }
}

#[test]
fn a_narrowed_filter_output_reads_as_written() {
    let line = format!("videotestsrc num-buffers=1 ! {FORMAT_FILTER} ! y4menc ! fakesink");
    let stderr = launch_stderr(&line, None);
    let header = line_containing(&stderr, "produces no output for ");
    let shown = header
        .split_once("produces no output for ")
        .expect("the header")
        .1;
    assert!(shown.starts_with(FORMAT_FILTER), "{header}");
    assert!(!shown.contains("width="), "{header}");
}
