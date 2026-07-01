//! M481 - named input / request pads on muxers, the input-side transpose of the
//! M476 output-pad selection. A `gst-launch` line references a muxer's inputs by
//! name (`... ! o.video`, `... ! o.text`, `... ! m.audio_0`) instead of relying on
//! the order the branches are written, resolved via the element's own
//! `input_pad_index` scheme.
//!
//! The correctness-critical case is `textoverlay`: its video pad MUST be input 0
//! (`output_follows_input(0)` + the PTS merge), so named pads must route video to 0
//! regardless of reference order. If they didn't, the swapped-order line would put
//! the text stream on pad 0 and RGBA video negotiation would fail.

#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph, ParseError};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn write_srt(tag: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m481-{}-{}.srt", std::process::id(), tag));
    std::fs::write(&path, "1\n00:00:00,000 --> 00:00:02,000\nHi\n").expect("write srt");
    path
}

async fn run_line(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4).await.unwrap_or_else(|e| panic!("runs `{line}`: {e:?}")).frames_consumed
}

/// Video branch referenced first: `o.video` -> pad 0, `o.text` -> pad 1.
#[tokio::test]
async fn textoverlay_named_pads_video_first() {
    let srt = write_srt("vfirst");
    let line = format!(
        "videotestsrc num-buffers=3 ! o.video   subtitlesrc location={} ! subparse ! o.text   \
         textoverlay name=o ! fakesink",
        srt.display()
    );
    let consumed = run_line(&line).await;
    std::fs::remove_file(&srt).ok();
    assert!(consumed >= 3, "named-pad overlay passed the video frames: {consumed}");
}

/// SWAPPED: text branch referenced FIRST. Named pads must still route video to
/// pad 0 (not the positional 0 the text branch would otherwise take), so the
/// overlay negotiates RGBA video and runs. This is the whole point of M481.
#[tokio::test]
async fn textoverlay_named_pads_text_first_is_order_independent() {
    let srt = write_srt("tfirst");
    let line = format!(
        "subtitlesrc location={} ! subparse ! o.text   videotestsrc num-buffers=3 ! o.video   \
         textoverlay name=o ! fakesink",
        srt.display()
    );
    let consumed = run_line(&line).await;
    std::fs::remove_file(&srt).ok();
    assert!(consumed >= 3, "text-first named pads still route video to pad 0: {consumed}");
}

/// Two references to the same named pad collide: a clear parse error, not a
/// silent mis-wire.
#[test]
fn duplicate_named_input_pad_is_an_error() {
    let reg = default_registry();
    let srt = write_srt("dup");
    let line = format!(
        "videotestsrc num-buffers=1 ! o.video   subtitlesrc location={} ! subparse ! o.video   \
         textoverlay name=o ! fakesink",
        srt.display()
    );
    let err = parse_launch(&reg, &line).expect_err("two o.video refs collide");
    std::fs::remove_file(&srt).ok();
    assert!(matches!(err, ParseError::DuplicateInputPad(_)), "got {err:?}");
}

/// A typed request pad on a muxer with no such pad (a homogeneous `funnel` has no
/// `video` pad) is rejected with `UnknownInputPad`.
#[test]
fn unknown_named_input_pad_is_an_error() {
    let reg = default_registry();
    let line = "videotestsrc num-buffers=1 ! f.video_0   videotestsrc num-buffers=1 ! f.   \
                funnel name=f ! fakesink";
    let err = parse_launch(&reg, line).expect_err("funnel has no video_0 pad");
    assert!(matches!(err, ParseError::UnknownInputPad(_)), "got {err:?}");
}

/// Container muxers accept the gst request-pad names either order: `matroskamux`
/// takes typed `video_%u` / `audio_%u` (its slots are caps-typed, so order is
/// cosmetic), and the homogeneous `funnel` takes the generic `sink_%u`. Parse-only
/// (codecs would be needed to run); the point is the named refs resolve and build.
#[test]
fn container_mux_accepts_named_pads_either_order() {
    let reg = default_registry();
    for line in [
        "videotestsrc num-buffers=1 ! m.video_0   audiotestsrc num-buffers=1 ! m.audio_0   matroskamux name=m ! fakesink",
        "audiotestsrc num-buffers=1 ! m.audio_0   videotestsrc num-buffers=1 ! m.video_0   matroskamux name=m ! fakesink",
        "videotestsrc num-buffers=1 ! f.sink_1   videotestsrc num-buffers=1 ! f.sink_0   funnel name=f ! fakesink",
    ] {
        parse_launch(&reg, line).unwrap_or_else(|e| panic!("named-pad line parses `{line}`: {e}"));
    }
}
