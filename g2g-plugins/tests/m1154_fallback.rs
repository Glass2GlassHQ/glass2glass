//! M1154: `fallbacksrc` as a launch keyword. `fallbacksrc uri=X` expands to the
//! URI's decode chain and a second branch, both feeding a `fallbackswitch` whose
//! input 0 is the main stream. The second branch is `fallback-uri`'s decode chain
//! when one is given, else a dummy generator (black frames or silence) paced by
//! `clocksync`, since neither test source is live.
//!
//! The switching rule itself is unit-tested in `g2g-plugins/src/fallbackswitch.rs`
//! against a driven clock; these assert the expansion's wiring and that the dummy
//! really reaches the sink once the main stream stops.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::time::Duration;

use g2g_core::runtime::{parse_launch, run_graph, ParseError};
use g2g_core::{NodeKind, PipelineClock};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// The dummy video fallback's frame, as `videotestsrc pattern=black` draws it.
const BLACK_PIXEL: [u8; 4] = [0, 0, 0, 255];

/// A `timeout` short enough that the test does not sit through gst's one-second
/// default while the main stream's EOS ages into a stall.
const SHORT_TIMEOUT_NS: u64 = 50_000_000;

/// How long a timed run gets before the dummy's frames are declared missing. The
/// dummy is unbounded (a real fallback never ends), so the run is cancelled
/// rather than waited out.
const DUMMY_RUN: Duration = Duration::from_secs(4);

/// Pixels in one dummy video frame, read off `videotestsrc`'s declared geometry
/// (the expansion does not override it), so this cannot drift from the source.
fn dummy_frame_pixels() -> usize {
    let source = default_registry()
        .make_source("videotestsrc")
        .expect("videotestsrc is a baseline source");
    let dimension = |name| {
        source
            .get_property(name)
            .and_then(|v| v.as_uint())
            .unwrap_or_else(|| panic!("videotestsrc reports its {name}")) as usize
    };
    dimension("width") * dimension("height")
}

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m1154_{}_{tag}", std::process::id()))
}

/// Write a fixture by running `line` (which must end in a `filesink`) and return
/// its path, so the expected bytes come from the real encoder rather than a
/// hand-rolled header.
async fn fixture(tag: &str, line: &str) -> PathBuf {
    let path = temp_path(tag);
    let reg = default_registry();
    let line = format!("{line} ! filesink location={}", path.display());
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line}: {e:?}"));
    path
}

/// A single-frame PNM still, the baseline registry's shortest path from a
/// `file://` URI to raw video (`pnmdec` is compiled in unconditionally).
async fn pnm_fixture(tag: &str) -> PathBuf {
    fixture(tag, "videotestsrc num-buffers=1 pattern=smpte ! pnmenc").await
}

/// A `.au` fixture, the baseline registry's shortest path from a `file://` URI to
/// raw audio (`auparse` is compiled in unconditionally).
async fn au_fixture(tag: &str) -> PathBuf {
    fixture(tag, "audiotestsrc num-buffers=3 ! avmux_au").await
}

fn kinds(line: &str) -> Vec<NodeKind> {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line}: {e}"));
    let valid = graph.finish().unwrap_or_else(|e| panic!("{line}: {e:?}"));
    valid.topo().iter().map(|&n| valid.kind(n)).collect()
}

#[tokio::test]
async fn fallbacksrc_wires_the_main_uri_and_a_dummy_into_the_switch() {
    let path = pnm_fixture("wiring.pnm").await;
    let line = format!("fallbacksrc uri=file://{} ! fakesink", path.display());
    let kinds = kinds(&line);
    std::fs::remove_file(&path).ok();

    // Two sources: the URI's file source and the dummy generator.
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Source).count(),
        2,
        "the main URI and the dummy fallback: {kinds:?}"
    );
    // One two-input switch, both branches landing on it.
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Muxer(2)).count(),
        1,
        "a single 2-input fallbackswitch: {kinds:?}"
    );
    // main: filesrc + pnmdec. fallback: videotestsrc + clocksync. Plus the
    // switch and the sink.
    assert_eq!(kinds.len(), 6, "{kinds:?}");
}

#[tokio::test]
async fn fallbacksrc_audio_uri_gets_a_silence_dummy() {
    let path = au_fixture("wiring.au").await;
    let line = format!(
        "fallbacksrc uri=file://{} enable-video=false ! fakesink",
        path.display()
    );
    let kinds = kinds(&line);
    std::fs::remove_file(&path).ok();

    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Source).count(),
        2,
        "the .au source and the silence dummy: {kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Muxer(2)).count(),
        1,
        "{kinds:?}"
    );
}

/// A `name=` on the `fallbacksrc` names the switch, so the line can hang further
/// fallbacks off it by pad reference: input 0 stays the main URI, 1 the dummy, and
/// the referenced pad is the next one down.
#[tokio::test]
async fn fallbacksrc_name_resolves_to_its_switch() {
    let path = pnm_fixture("named.pnm").await;
    let line = format!(
        "fallbacksrc name=f uri=file://{} ! fakesink  videotestsrc ! f.sink_2",
        path.display()
    );
    let kinds = kinds(&line);
    std::fs::remove_file(&path).ok();

    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Muxer(3)).count(),
        1,
        "the extra branch became a third switch input: {kinds:?}"
    );
}

/// The whole point of the dummy: once the main stream stops delivering, black
/// frames keep coming out of the switch. The run is timed out because the dummy
/// never ends, and the frames are read back off a `filesink`.
#[tokio::test]
async fn dummy_black_frames_reach_the_sink_after_the_main_stream_stalls() {
    let source = pnm_fixture("black.pnm").await;
    let out = temp_path("black.raw");
    let line = format!(
        "fallbacksrc uri=file://{} timeout={SHORT_TIMEOUT_NS} ! filesink location={}",
        source.display(),
        out.display()
    );
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));
    // Cancelled, not awaited to completion: an unbounded fallback has no EOS.
    let _ = tokio::time::timeout(DUMMY_RUN, run_graph(graph, &ZeroClock, 4)).await;

    let bytes = std::fs::read(&out).expect("the sink wrote what the switch forwarded");
    std::fs::remove_file(&source).ok();
    std::fs::remove_file(&out).ok();
    let black = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| **px == BLACK_PIXEL)
        .count();
    // More black pixels than a whole dummy frame holds can only have come from
    // the fallback branch.
    assert!(
        black > dummy_frame_pixels(),
        "the fallback's black frames reached the sink, got {black} black pixels in {} bytes",
        bytes.len()
    );
}

#[test]
fn fallbacksrc_without_a_uri_is_an_error() {
    let reg = default_registry();
    assert_eq!(
        parse_launch(&reg, "fallbacksrc ! fakesink").unwrap_err(),
        ParseError::MissingUri("fallbacksrc".to_string())
    );
}

#[test]
fn fallbacksrc_rejects_a_property_it_does_not_have() {
    let reg = default_registry();
    assert_eq!(
        parse_launch(
            &reg,
            "fallbacksrc uri=file:///x.pnm restart-on-eos=true ! fakesink"
        )
        .unwrap_err(),
        ParseError::UnknownProperty {
            element: "fallbacksrc".to_string(),
            key: "restart-on-eos".to_string(),
        }
    );
}

#[test]
fn fallbacksrc_rejects_a_non_boolean_enable_flag() {
    let reg = default_registry();
    assert_eq!(
        parse_launch(
            &reg,
            "fallbacksrc uri=file:///x.pnm enable-audio=maybe ! fakesink"
        )
        .unwrap_err(),
        ParseError::BadValue {
            element: "fallbacksrc".to_string(),
            key: "enable-audio".to_string(),
            value: "maybe".to_string(),
        }
    );
}

#[test]
fn fallbacksrc_must_head_its_chain() {
    let reg = default_registry();
    assert_eq!(
        parse_launch(
            &reg,
            "videotestsrc ! fallbacksrc uri=file:///x.pnm ! fakesink"
        )
        .unwrap_err(),
        ParseError::UriSourceNotAtHead("fallbacksrc".to_string())
    );
}

/// A `fallback-uri` replaces the dummy with that URI's own decode chain. Both
/// branches are files here, so unlike the dummy case the run reaches EOS: the
/// main stream's frame is forwarded and the fallback's is dropped, since the main
/// never stalls.
#[tokio::test]
async fn fallback_uri_branch_runs_to_eos_with_the_main_stream_forwarded() {
    let main = pnm_fixture("main.pnm").await;
    let spare = pnm_fixture("spare.pnm").await;
    let line = format!(
        "fallbacksrc uri=file://{} fallback-uri=file://{} ! fakesink",
        main.display(),
        spare.display()
    );
    let kinds = kinds(&line);
    assert_eq!(
        kinds.iter().filter(|k| **k == NodeKind::Source).count(),
        2,
        "both URIs became sources: {kinds:?}"
    );

    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line}: {e}"));
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line}: {e:?}"));
    std::fs::remove_file(&main).ok();
    std::fs::remove_file(&spare).ok();
    assert_eq!(
        stats.frames_consumed, 1,
        "the main stream's one frame, the fallback's dropped"
    );
}

#[tokio::test]
async fn fallback_uri_with_no_handler_is_an_error() {
    let main = pnm_fixture("nohandler.pnm").await;
    let reg = default_registry();
    let error = parse_launch(
        &reg,
        &format!(
            "fallbacksrc uri=file://{} fallback-uri=nosuchscheme:///y ! fakesink",
            main.display()
        ),
    )
    .unwrap_err();
    std::fs::remove_file(&main).ok();
    assert!(
        matches!(&error, ParseError::Uri(message) if message.contains("nosuchscheme")),
        "the fallback URI is reported, got {error:?}"
    );
}

/// `timeout` and `immediate-fallback` are handed to the generated switch, not
/// swallowed by the macro: a value the switch's property rejects fails the parse
/// under the switch's name.
#[tokio::test]
async fn timeout_reaches_the_switch() {
    let main = pnm_fixture("timeout.pnm").await;
    let reg = default_registry();
    let error = parse_launch(
        &reg,
        &format!(
            "fallbacksrc uri=file://{} timeout=-1 ! fakesink",
            main.display()
        ),
    )
    .unwrap_err();
    std::fs::remove_file(&main).ok();
    assert_eq!(
        error,
        ParseError::BadValue {
            element: "fallbackswitch".to_string(),
            key: "timeout".to_string(),
            value: "-1".to_string(),
        }
    );
}
