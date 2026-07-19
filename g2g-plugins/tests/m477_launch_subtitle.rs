//! M477 - subtitle handling in a `gst-launch` line: the `subparse` launch element,
//! the `textoverlay` fan-in muxer (video + text stream join, the analog of
//! GStreamer's `textoverlay` text_sink request pad), and `d.text_0` subtitle-stream
//! selection in an explicit demux fan-out.
//!
//! ```text
//! videotestsrc num-buffers=3 ! o.
//! subtitlesrc location=x.srt ! subparse ! o.
//! textoverlay name=o ! fakesink
//! ```
//!
//! and, from a container's embedded subtitle track:
//!
//! ```text
//! filesrc location=x.mkv bytestream-format=matroska ! matroskademux name=d
//!   d.video_0 ! h264parse ! fakesink
//!   d.text_0  ! fakesink
//! ```

#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{NodeKind, PipelineClock};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Write a two-cue SubRip file to a uniquely-named temp path.
fn write_srt(tag: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m477-{}-{}.srt", std::process::id(), tag));
    let srt =
        "1\n00:00:00,000 --> 00:00:02,000\nHello\n\n2\n00:00:02,000 --> 00:00:04,000\nWorld\n";
    std::fs::write(&path, srt).expect("write srt");
    path
}

async fn run_line(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"))
        .frames_consumed
}

/// The `subparse` launch element turns a `subtitlesrc` SubRip file into timed
/// `Text{Utf8}` cues that flow to a sink: proof it is registered (M477) and links
/// downstream of `subtitlesrc`.
#[tokio::test]
async fn subparse_launch_element_streams_cues() {
    let srt = write_srt("subparse");
    let line = format!(
        "subtitlesrc location={} ! subparse ! fakesink",
        srt.display()
    );
    let consumed = run_line(&line).await;
    std::fs::remove_file(&srt).ok();
    assert!(
        consumed >= 2,
        "both SubRip cues flowed through subparse to the sink: {consumed}"
    );
}

/// The `textoverlay` fan-in muxer joins a video pad and a text-stream pad: an RGBA
/// `videotestsrc` on input 0 and the parsed subtitle cues on input 1 merge by PTS,
/// so every video frame reaches the sink painted with its cue. Proves the two-role
/// registration (single-input overlay element + fan-in muxer, picked by link
/// degree) and the video-then-text pad order.
#[tokio::test]
async fn textoverlay_fan_in_joins_video_and_text() {
    let srt = write_srt("overlay");
    let line = format!(
        "videotestsrc num-buffers=3 ! o.   subtitlesrc location={} ! subparse ! o.   \
         textoverlay name=o ! fakesink",
        srt.display()
    );
    let consumed = run_line(&line).await;
    std::fs::remove_file(&srt).ok();
    assert!(
        consumed >= 3,
        "all three video frames reached the sink through the overlay: {consumed}"
    );
}

/// `d.text_0` selects a container's embedded subtitle track in an explicit demux
/// fan-out: the video branch takes the H.264 track (strict `h264parse`), the text
/// branch the `S_TEXT/UTF8` subtitle track. The demux builds a two-output node,
/// proving the subtitle stream was resolved rather than declined. A header-only MKV
/// suffices: stream selection reads `Tracks`, not clusters.
#[test]
fn matroskademux_text_pad_selects_subtitle_stream() {
    let bytes = mkv_video_plus_subtitle();
    let path = std::env::temp_dir().join(format!("g2g-m477-{}-sel.mkv", std::process::id()));
    std::fs::write(&path, &bytes).expect("write mkv");
    let p = path.display();

    let reg = default_registry();
    let line = format!(
        "filesrc location={p} bytestream-format=matroska ! matroskademux name=d  \
         d.video_0 ! h264parse ! fakesink  d.text_0 ! fakesink"
    );
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
        "demux fans out a video port and a text port"
    );
}

// --- synthetic Matroska builder (mirrors the mkvdemux / m415 unit fixtures) ---
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
fn subtitle_track(num: u64, codec: &[u8]) -> Vec<u8> {
    let body = [
        elem(&[0xD7], &uint_body(num)),
        elem(&[0x83], &uint_body(0x11)),
        elem(&[0x86], codec),
    ]
    .concat();
    elem(&[0xAE], &body)
}

/// An MKV whose `Tracks` element carries a V_MPEG4/ISO/AVC (H.264) video track and
/// an `S_TEXT/UTF8` subtitle track. Only the header is needed: `d.text_0` selection
/// probes `Tracks`, not clusters.
fn mkv_video_plus_subtitle() -> Vec<u8> {
    let mut tracks_body = video_track(1, b"V_MPEG4/ISO/AVC", 320, 240);
    tracks_body.extend_from_slice(&subtitle_track(2, b"S_TEXT/UTF8"));
    let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &tracks_body);
    let segment = elem(&[0x18, 0x53, 0x80, 0x67], &tracks);
    [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
}
