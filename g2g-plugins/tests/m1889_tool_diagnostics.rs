// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

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

#[cfg(all(target_os = "linux", feature = "local-dmabuf"))]
#[path = "common/collect_sink.rs"]
mod collect_sink;

#[cfg(all(target_os = "linux", feature = "local-dmabuf"))]
#[tokio::test]
async fn verbose_launch_keeps_its_socket_source_connection() {
    use g2g_core::memory::OwnedDmaBuf;
    use g2g_core::{
        AsyncElement, Colorimetry, Frame, FrameTiming, Interlace, MemoryDomain, PipelinePacket,
        PixelAspect, Rate, RawVideoFormat,
    };
    use g2g_plugins::localdmabuf::DmaBufSink;
    use std::os::fd::IntoRawFd;
    use std::time::{Duration, Instant};

    const FRAMES: u64 = 5;
    const DEADLINE: Duration = Duration::from_secs(20);
    let socket =
        std::env::temp_dir().join(format!("g2g-m1889-verbose-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&socket);
    let location = socket.to_str().expect("a utf-8 temp path");
    let mut receiver = Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .args([
            "-v",
            "dmabufsrc",
            &format!("location={location}"),
            &format!("num-buffers={FRAMES}"),
            "!",
            "fakesink",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("g2g-launch runs");
    let started = Instant::now();
    while !socket.exists() {
        assert!(started.elapsed() < DEADLINE, "dmabufsrc never listened");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let caps = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
        interlace: Interlace::Progressive,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: PixelAspect::Any,
    };
    let mut sink = DmaBufSink::new(location);
    sink.configure_pipeline(&caps).expect("the sink configures");
    let mut unused = collect_sink::CollectSink::default();
    for sequence in 0..FRAMES {
        let fd = std::fs::File::open("/dev/null")
            .expect("open /dev/null")
            .into_raw_fd();
        // SAFETY: the descriptor was just opened and is handed over whole.
        let dmabuf = unsafe { OwnedDmaBuf::from_raw(fd, 16, 0) };
        let frame = Frame::new(
            MemoryDomain::DmaBuf(dmabuf),
            FrameTiming::default(),
            sequence,
        );
        sink.process(PipelinePacket::DataFrame(frame), &mut unused)
            .await
            .expect("the receiver keeps the connection");
    }

    while receiver.try_wait().expect("poll g2g-launch").is_none() {
        if started.elapsed() > DEADLINE {
            let _ = receiver.kill();
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let output = receiver.wait_with_output().expect("g2g-launch exits");
    let _ = std::fs::remove_file(&socket);
    let stderr = stderr_of(&output);
    assert!(output.status.success(), "{stderr}");
    assert!(stderr.contains("negotiated links (1)"), "{stderr}");
}
