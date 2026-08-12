//! M478 - content/extension typefind. A bare `filesrc` (no `bytestream-format`)
//! types itself from the `location` extension, so a GStreamer-style line runs
//! without naming the container / subtitle format:
//!
//! ```text
//! filesrc location=subs.vtt ! subparse ! fakesink          # -> Caps::Text{WebVtt}
//! filesrc location=movie.mkv ! matroskademux name=d ...     # -> ByteStream{Matroska}
//! ```
//!
//! Content sniffing (`bytestream-format=auto`) additionally covers a mis-named or
//! extensionless file, and now recognizes MP4 (`ftyp`) and subtitle documents,
//! not just the M112 container set (covered by the `typefind` unit tests).

#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph, SourceLoop as _};
use g2g_core::{Caps, NodeKind, PipelineClock, PropValue, TextFormat};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::registry::default_registry;
use g2g_plugins::typefind::sniff_caps;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp(tag: &str, ext: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path =
        std::env::temp_dir().join(format!("g2g-m478-{}-{}.{}", std::process::id(), tag, ext));
    std::fs::write(&path, bytes).expect("write temp");
    path
}

/// A bare `filesrc location=X.vtt` types as `Caps::Text{WebVtt}` from the
/// extension (no `bytestream-format`), so `subparse` accepts it and streams cues.
#[tokio::test]
async fn filesrc_types_vtt_by_extension_into_subparse() {
    let vtt = "WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.000\nHello\n\n2\n00:00:02.000 --> 00:00:04.000\nWorld\n";
    let path = temp("subs", "vtt", vtt.as_bytes());
    let line = format!("filesrc location={} ! subparse ! fakesink", path.display());

    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let consumed = run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs: {e:?}"))
        .frames_consumed;
    std::fs::remove_file(&path).ok();
    assert!(
        consumed >= 2,
        "both WebVTT cues flowed through subparse to the sink: {consumed}"
    );
}

/// A bare `filesrc location=X.mkv` types as `ByteStream{Matroska}` from the
/// extension, so the M476 explicit-demux fan-out fires (it only fires for a
/// container source). Header-only MKV: demux-select probes `Tracks`, not clusters.
#[test]
fn filesrc_types_mkv_by_extension_for_demux_fanout() {
    let path = temp("movie", "mkv", &mkv_video_plus_audio());
    let p = path.display();
    // No bytestream-format: the .mkv extension must route filesrc to Matroska for
    // the demux-select hook to build a two-port fan-out.
    let line = format!(
        "filesrc location={p} ! matroskademux name=d  \
         d.video_0 ! h264parse ! fakesink  d.audio_0 ! aacparse ! fakesink"
    );
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let vg = graph.finish().expect("valid graph");
    let demuxes: Vec<NodeKind> = vg
        .topo()
        .iter()
        .map(|&n| vg.kind(n))
        .filter(|k| matches!(k, NodeKind::Tee(_)))
        .collect();
    std::fs::remove_file(&path).ok();
    assert_eq!(
        demuxes,
        [NodeKind::Tee(2)],
        "extension typed filesrc as Matroska, demux fanned out"
    );
}

/// An explicit `bytestream-format` still pins the type regardless of extension (a
/// regression guard on the extension-defaulting): a `.dat` file named as matroska
/// still routes to the demuxer.
#[test]
fn explicit_bytestream_format_overrides_extension() {
    let path = temp("movie", "dat", &mkv_video_plus_audio());
    let p = path.display();
    let line = format!(
        "filesrc location={p} bytestream-format=matroska ! matroskademux name=d  \
         d.video_0 ! h264parse ! fakesink  d.audio_0 ! aacparse ! fakesink"
    );
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let vg = graph.finish().expect("valid graph");
    let has_demux = vg
        .topo()
        .iter()
        .any(|&n| matches!(vg.kind(n), NodeKind::Tee(2)));
    std::fs::remove_file(&path).ok();
    assert!(
        has_demux,
        "explicit bytestream-format=matroska typed the .dat file for the demuxer"
    );
}

/// The extension is the whole text-typing rule: `.txt` / `.text` is plain UTF-8,
/// every subtitle extension keeps its own `TextFormat`, and an extension that
/// names neither is left to content sniffing (`None` here, so `filesrc` arms
/// `auto`) rather than being guessed as text.
#[test]
fn filesrc_types_text_extensions_to_their_own_format() {
    let prose = b"Tell me a joke about a cat.\n";
    let cases: &[(&str, &[u8], Option<TextFormat>)] = &[
        ("txt", prose, Some(TextFormat::Utf8)),
        ("text", prose, Some(TextFormat::Utf8)),
        (
            "vtt",
            b"WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHi\n",
            Some(TextFormat::WebVtt),
        ),
        (
            "srt",
            b"1\n00:00:20,000 --> 00:00:24,400\nHello\n",
            Some(TextFormat::Srt),
        ),
        ("ssa", b"[Script Info]\nTitle: x\n", Some(TextFormat::Ssa)),
        ("ass", b"[Script Info]\nTitle: x\n", Some(TextFormat::Ssa)),
        // Not claimed as text: an unknown extension, and a container one.
        ("dat", prose, None),
        ("mkv", prose, None),
    ];
    for (ext, bytes, expected) in cases {
        let path = temp("typed", ext, bytes);
        let mut src = FileSrc::untyped();
        src.set_property("location", PropValue::Str(path.display().to_string()))
            .expect("location");
        let typed = match src.configured_output_caps() {
            Some(Caps::Text { format }) => Some(format),
            _ => None,
        };
        std::fs::remove_file(&path).ok();
        assert_eq!(typed, *expected, ".{ext} typed as text");
    }
}

/// A `.txt` prose file negotiates and runs into an element that accepts
/// `Caps::Text{Utf8}`, the end-to-end shape of
/// `filesrc location=prompt.txt ! <text element>`.
#[tokio::test]
async fn filesrc_runs_a_plain_text_file_into_a_text_sink() {
    let path = temp("prompt", "txt", b"Tell me a joke about a cat.\n");
    let line = format!(
        "filesrc location={} ! capsfilter caps=text/x-raw,format=utf8 ! fakesink",
        path.display()
    );
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let consumed = run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs: {e:?}"))
        .frames_consumed;
    std::fs::remove_file(&path).ok();
    assert!(consumed >= 1, "the text chunk reached the sink: {consumed}");
}

/// Content sniffing is unchanged by the extension rule: subtitle documents still
/// sniff to their own format, and prose still sniffs to nothing. A "valid UTF-8"
/// content rule would claim almost any small ASCII-ish binary, so there is none.
#[test]
fn content_sniffing_still_refuses_plain_text() {
    assert_eq!(
        sniff_caps(b"WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHi\n"),
        Some(Caps::Text {
            format: TextFormat::WebVtt
        })
    );
    assert_eq!(
        sniff_caps(b"1\n00:00:20,000 --> 00:00:24,400\nHello\n"),
        Some(Caps::Text {
            format: TextFormat::Srt
        })
    );
    assert_eq!(
        sniff_caps(b"[Script Info]\nTitle: x\n"),
        Some(Caps::Text {
            format: TextFormat::Ssa
        })
    );
    assert_eq!(sniff_caps(b"Tell me a joke about a cat.\n"), None);
}

// --- synthetic Matroska builder (Tracks-only header, mirrors m477 / mkvdemux) ---
fn vint(value: u64) -> Vec<u8> {
    let mut len = 1usize;
    while len < 8 && value >= (1u64 << (7 * len)) - 1 {
        len += 1;
    }
    let mut out = vec![0u8; len];
    let mut v = value;
    for i in (0..len).rev() {
        out[i] = (v & 0xFF) as u8;
        v >>= 8;
    }
    out[0] |= 1 << (8 - len);
    out
}
fn elem(id: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&vint(body.len() as u64));
    out.extend_from_slice(body);
    out
}
fn uint_body(v: u64) -> Vec<u8> {
    if v == 0 {
        return vec![0];
    }
    let mut bytes = v.to_be_bytes().to_vec();
    while bytes.len() > 1 && bytes[0] == 0 {
        bytes.remove(0);
    }
    bytes
}
fn video_track(num: u64, codec: &[u8], w: u32, h: u32) -> Vec<u8> {
    let v = [
        elem(&[0xB0], &uint_body(w as u64)),
        elem(&[0xBA], &uint_body(h as u64)),
    ]
    .concat();
    let body = [
        elem(&[0xD7], &uint_body(num)),
        elem(&[0x83], &uint_body(1)),
        elem(&[0x86], codec),
        elem(&[0xE0], &v),
    ]
    .concat();
    elem(&[0xAE], &body)
}
fn audio_track(num: u64, codec: &[u8]) -> Vec<u8> {
    // TrackNumber, TrackType(audio=2), CodecID.
    let body = [
        elem(&[0xD7], &uint_body(num)),
        elem(&[0x83], &uint_body(2)),
        elem(&[0x86], codec),
    ]
    .concat();
    elem(&[0xAE], &body)
}
fn mkv_video_plus_audio() -> Vec<u8> {
    let mut tracks_body = video_track(1, b"V_MPEG4/ISO/AVC", 320, 240);
    tracks_body.extend_from_slice(&audio_track(2, b"A_AAC"));
    let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &tracks_body);
    let segment = elem(&[0x18, 0x53, 0x80, 0x67], &tracks);
    [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
}
