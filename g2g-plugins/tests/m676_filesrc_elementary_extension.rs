//! M676 - a bare `filesrc` types raw Annex-B elementary streams by extension
//! (`.h264` / `.264` / `.avc`, `.h265` / `.265` / `.hevc`), found by the
//! calliope differential harness: the extension map and the sniffer both
//! missed them, so the MPEG-TS default made
//! `filesrc location=x.h264 ! h264parse` fail negotiation with CapsMismatch.
//!
//! Follow-up: an unknown extension (e.g. a `.jsv` JVT conformance vector) now
//! content-sniffs the Annex-B start codes instead of falling back to MPEG-TS,
//! also surfaced by calliope (`conformance` decoded empty output for `.jsv`).

#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp(tag: &str, ext: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path =
        std::env::temp_dir().join(format!("g2g-m676-{}-{}.{}", std::process::id(), tag, ext));
    std::fs::write(&path, bytes).expect("write temp");
    path
}

fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

/// Minimal H.264 stream: SPS + PPS + two IDR slices, so the reframing parser
/// emits at least the first access unit before EOS.
fn h264_stream() -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    annexb(&[&sps, &pps, &idr, &idr])
}

/// A bare `filesrc location=X.h264` types as `CompressedVideo{H264}` from the
/// extension, so `h264parse` negotiates and access units flow to the sink.
#[tokio::test]
async fn filesrc_types_h264_by_extension_into_h264parse() {
    for ext in ["h264", "264", "avc"] {
        let path = temp("es", ext, &h264_stream());
        let line = format!("filesrc location={} ! h264parse ! fakesink", path.display());
        let reg = default_registry();
        let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
        let consumed = run_graph(graph, &ZeroClock, 4)
            .await
            .unwrap_or_else(|e| panic!(".{ext} negotiates and runs: {e:?}"))
            .frames_consumed;
        std::fs::remove_file(&path).ok();
        assert!(
            consumed >= 1,
            ".{ext}: an access unit reached the sink: {consumed}"
        );
    }
}

/// The `.h265` family types as `CompressedVideo{H265}` so `h265parse`
/// negotiates (the pre-M676 MPEG-TS default failed with CapsMismatch).
#[tokio::test]
async fn filesrc_types_h265_by_extension_into_h265parse() {
    for ext in ["h265", "265", "hevc"] {
        // Header-only payload: negotiation is what regressed, not parsing.
        let path = temp("es", ext, &annexb(&[&[0x40u8, 0x01]]));
        let line = format!("filesrc location={} ! h265parse ! fakesink", path.display());
        let reg = default_registry();
        let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
        let outcome = run_graph(graph, &ZeroClock, 4).await;
        std::fs::remove_file(&path).ok();
        outcome.unwrap_or_else(|e| panic!(".{ext} negotiates and runs: {e:?}"));
    }
}

/// An unknown extension content-sniffs the Annex-B start codes rather than
/// defaulting to MPEG-TS: `filesrc location=x.jsv` types as `CompressedVideo`
/// so `h264parse` negotiates and access units flow.
#[tokio::test]
async fn filesrc_types_unknown_extension_by_content_sniff() {
    // `.jsv` (JVT H.264) and a bare `.bin` both exercise the sniff fallback.
    for ext in ["jsv", "bin"] {
        let path = temp("sniff", ext, &h264_stream());
        let line = format!("filesrc location={} ! h264parse ! fakesink", path.display());
        let reg = default_registry();
        let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
        let consumed = run_graph(graph, &ZeroClock, 4)
            .await
            .unwrap_or_else(|e| panic!(".{ext} content-sniffs and runs: {e:?}"))
            .frames_consumed;
        std::fs::remove_file(&path).ok();
        assert!(
            consumed >= 1,
            ".{ext}: an access unit reached the sink: {consumed}"
        );
    }
}

/// `decodebin` auto-plugs from the extension-typed caps: an `.h264` source
/// expands to an H.264 decode chain, not the MPEG-TS demux default. Needs a
/// compiled-in H.264 decoder, so gated like the `ffmpeg` registry entries.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
#[test]
fn decodebin_expands_h264_extension_to_decode_chain() {
    let path = temp("db", "h264", &h264_stream());
    let line = format!("filesrc location={} ! decodebin ! fakesink", path.display());
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let node_count = graph.finish().expect("valid graph").topo().len();
    std::fs::remove_file(&path).ok();
    // filesrc ! h264parse ! ffmpegdec ! fakesink: the M421 re-framing parser
    // is spliced ahead of the decoder in the name-based expansion too (it fed
    // whole file chunks to the decoder before, decoding only frame 0).
    assert_eq!(
        node_count, 4,
        "parser + decoder expanded from extension caps"
    );
}

/// `decodebin` auto-plugs a content-sniffed unknown extension into the same
/// decode chain (the `.jsv` conformance-vector case).
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
#[test]
fn decodebin_expands_sniffed_unknown_extension_to_decode_chain() {
    let path = temp("db", "jsv", &h264_stream());
    let line = format!("filesrc location={} ! decodebin ! fakesink", path.display());
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let node_count = graph.finish().expect("valid graph").topo().len();
    std::fs::remove_file(&path).ok();
    assert_eq!(node_count, 4, "parser + decoder expanded from sniffed caps");
}
