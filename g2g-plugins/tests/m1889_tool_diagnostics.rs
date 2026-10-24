#![cfg(all(feature = "std", unix))]

use std::io::{BufRead, BufReader};
use std::os::unix::process::ExitStatusExt;
use std::process::{Command, Output, Stdio};

use g2g_core::runtime::parse_launch;
use g2g_core::{Caps, Dim, NodeId, PadCaps, PadTemplates};
use g2g_plugins::registry::default_registry;
use g2g_plugins::videotestsrc::VideoTestSrc;

fn launch(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .args(args)
        .output()
        .expect("g2g-launch runs")
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

// the largest width videotestsrc draws, read off its pad template
fn drawable_width() -> (u32, u32) {
    let templates = VideoTestSrc::pad_templates();
    let PadCaps::Fixed(caps) = &templates[0].caps else {
        panic!("videotestsrc declares a fixed template");
    };
    match &caps.alternatives()[0] {
        Caps::RawVideo {
            width: Dim::Range { min, max },
            ..
        } => (*min, *max),
        other => panic!("videotestsrc declares a bounded width, got {other:?}"),
    }
}

// each element's label in the order the line names them, as the ERROR block and `--dot` print it
fn node_names(line: &str) -> Vec<&'static str> {
    let graph = parse_launch(&default_registry(), line).expect("the line parses");
    (0..graph.node_count())
        .map(|i| {
            graph
                .element(NodeId(i as u32))
                .expect("a linear line has no tee")
                .log_category()
        })
        .collect()
}

#[test]
fn an_out_of_range_caps_filter_names_the_field_and_both_sides() {
    let (min, max) = drawable_width();
    let wanted = max + 1;
    let line = format!("videotestsrc ! video/x-raw,width={wanted},height=100 ! fakesink");
    let names = node_names(&line);
    let (source, filter) = (names[0], names[1]);

    let output = launch(&line.split(' ').collect::<Vec<_>>());
    let stderr = stderr_of(&output);
    assert!(!output.status.success(), "{stderr}");
    assert!(stderr.contains(&format!("{filter} wants: ")), "{stderr}");
    let width_line = stderr
        .lines()
        .find(|l| l.contains(&format!("width: {source} offers")))
        .unwrap_or_else(|| panic!("no width line:\n{stderr}"));
    assert!(
        width_line.contains(&format!("[{min},{max}]")),
        "{width_line}"
    );
    assert!(
        width_line.contains(&format!("{filter} wants {wanted}")),
        "{width_line}"
    );
}

#[test]
fn a_width_property_past_the_drawable_range_fails_the_parse() {
    let (_, max) = drawable_width();
    let output = launch(&[
        "videotestsrc",
        &format!("width={}", max + 1),
        "num-buffers=1",
        "!",
        "fakesink",
    ]);
    let stderr = stderr_of(&output);
    assert!(!output.status.success(), "{stderr}");
    assert!(stderr.contains("property 'width'"), "{stderr}");

    let at_the_limit = launch(&[
        "-q",
        "videotestsrc",
        &format!("width={max}"),
        "height=1",
        "num-buffers=1",
        "!",
        "fakesink",
    ]);
    assert!(
        at_the_limit.status.success(),
        "{}",
        stderr_of(&at_the_limit)
    );
}

#[test]
fn verbose_prints_a_failed_negotiation_once_with_element_names() {
    let line = "audiotestsrc ! videoflip ! fakesink";
    let names = node_names(line);
    let mut args = vec!["-v"];
    args.extend(line.split(' '));
    let output = launch(&args);
    let stderr = stderr_of(&output);
    assert!(!output.status.success(), "{stderr}");
    let headers = stderr.matches("produces no output for").count();
    assert_eq!(headers, 1, "{stderr}");
    for pair in names.windows(2) {
        assert!(
            stderr.contains(&format!("{} -> {}  policy=", pair[0], pair[1])),
            "the topology names {} -> {}:\n{stderr}",
            pair[0],
            pair[1]
        );
    }
}

#[cfg(feature = "wayland-sink")]
#[test]
fn a_pipeline_without_a_display_sink_prints_nothing_about_wayland() {
    let output = Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .args(["-q", "videotestsrc", "num-buffers=1", "!", "fakesink"])
        .env_remove("WAYLAND_DISPLAY")
        .output()
        .expect("g2g-launch runs");
    let stderr = stderr_of(&output);
    assert!(output.status.success(), "{stderr}");
    assert!(!stderr.contains("WAYLAND_DISPLAY"), "{stderr}");
}

#[test]
fn inspect_prints_caps_as_gst_strings_with_the_drawable_range() {
    let (min, max) = drawable_width();
    let output = Command::new(env!("CARGO_BIN_EXE_g2g-inspect"))
        .arg("videotestsrc")
        .output()
        .expect("g2g-inspect runs");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.contains(&format!("width=[{min},{max}]")), "{stdout}");
    assert!(!stdout.contains("RawVideo {"), "{stdout}");
}

#[test]
fn inspect_into_a_closed_pipe_ends_by_sigpipe() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_g2g-inspect"))
        .arg("--all")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("g2g-inspect starts");
    let mut stdout = BufReader::new(child.stdout.take().expect("stdout is piped"));
    let mut first_line = String::new();
    stdout.read_line(&mut first_line).expect("a first line");
    assert!(!first_line.is_empty(), "inspect printed a line");
    drop(stdout);

    let output = child.wait_with_output().expect("g2g-inspect ends");
    assert_eq!(
        output.status.signal(),
        Some(libc::SIGPIPE),
        "status {:?}, stderr: {}",
        output.status,
        stderr_of(&output)
    );
    assert!(output.stderr.is_empty(), "nothing printed on stderr");
}
