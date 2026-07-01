//! M479 - progressive / whole-file MP4 through a linear chain. A local `.mp4`
//! types as `ByteStream{Mp4}` (M478), and the single-output `Mp4Demux` (registered
//! as `qtdemux` and in the `decodebin` autoplug pool) demuxes it to its video
//! track, so a bare `filesrc location=X.mp4 ! decodebin` / `! qtdemux !` runs
//! without the multi-branch `d.video_0 ! ... d.audio_0 ! ...` fan-out (which needs
//! 2+ branches). The whole-file demuxer is distinct from the streaming `fmp4demux`
//! (which keeps serving the `IsoBmff` that HLS / DASH produce), so the fragmented
//! path is untouched.
//!
//! These assert the parse-time WIRING (no file is read: the demuxer advertises a
//! fixatable `Range` placeholder, refined from the moov at run time). Runtime
//! demux is covered by the `fmp4::parse_progressive` unit tests and live playback.

#![cfg(feature = "std")]

use g2g_core::runtime::parse_launch;
use g2g_core::NodeKind;
use g2g_plugins::registry::default_registry;

/// A single-output `qtdemux` (no fan-out branches) builds a linear graph whose
/// demuxer is a `Transform` (the whole-file `Mp4Demux`), not a `Tee` fan-out
/// (`Mp4DemuxN`): `filesrc(Source) -> Mp4Demux(Transform) -> h264parse -> sink`.
#[test]
fn qtdemux_single_output_is_a_linear_transform() {
    let reg = default_registry();
    let graph = parse_launch(&reg, "filesrc location=/x/movie.mp4 ! qtdemux ! h264parse ! fakesink")
        .expect("progressive-MP4 qtdemux chain parses");
    let vg = graph.finish().expect("valid graph");
    let kinds: Vec<NodeKind> = vg.topo().iter().map(|&n| vg.kind(n)).collect();
    assert_eq!(kinds.len(), 4, "source + demux + parser + sink");
    assert!(!kinds.iter().any(|k| matches!(k, NodeKind::Tee(_))), "single-output demux, not a fan-out");
}

/// The `d.video_0` named-pad form on a single branch also resolves to the
/// single-output demuxer (the video-only file case, #2): the demux-select fan-out
/// needs 2+ branches, so one branch falls to the `qtdemux` launch element.
#[test]
fn qtdemux_named_video_pad_single_branch_parses() {
    let reg = default_registry();
    let graph =
        parse_launch(&reg, "filesrc location=/x/movie.mp4 ! qtdemux name=d d.video_0 ! h264parse ! fakesink")
            .expect("single-branch named video pad parses");
    let vg = graph.finish().expect("valid graph");
    assert!(
        !vg.topo().iter().any(|&n| matches!(vg.kind(n), NodeKind::Tee(_))),
        "one branch resolves to the single-output demux, not a fan-out"
    );
}

/// `filesrc location=X.mp4 ! decodebin` auto-plugs the whole-file demuxer for the
/// `Mp4` byte stream (then a parser + the ffmpeg H.264 decoder): the linear decode
/// path a GStreamer user expects. Needs a decoder in the autoplug pool (ffmpeg).
#[cfg(feature = "ffmpeg")]
#[test]
fn decodebin_autoplugs_the_progressive_mp4_demuxer() {
    let reg = default_registry();
    let graph = parse_launch(&reg, "filesrc location=/x/movie.mp4 ! decodebin ! videoconvert ! fakesink")
        .expect("decodebin auto-plugs a demux + decode chain for a progressive MP4");
    let vg = graph.finish().expect("valid graph");
    let kinds: Vec<NodeKind> = vg.topo().iter().map(|&n| vg.kind(n)).collect();
    // filesrc -> Mp4Demux -> (parser) -> ffmpeg decoder -> videoconvert -> fakesink:
    // a single linear chain, no fan-out.
    assert!(kinds.len() >= 5, "demux + parser + decoder + convert + sink: {}", kinds.len());
    assert!(!kinds.iter().any(|k| matches!(k, NodeKind::Tee(_))), "linear decode path, no fan-out");
}
