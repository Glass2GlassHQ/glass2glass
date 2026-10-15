#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, ParseError};
use g2g_plugins::registry::default_registry;

// an extension filesrc does not map, so it types the file by content
const UNMAPPED_EXTENSION: &str = "bin";

fn write_temp(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "g2g-m1454-{tag}-{}.{UNMAPPED_EXTENSION}",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write temp");
    path
}

#[test]
fn decodebin_refuses_content_no_sniff_recognizes() {
    let path = write_temp("noise", &[0xDE, 0xAD, 0xBE, 0xEF].repeat(64));
    let line = format!("filesrc location={} ! decodebin ! fakesink", path.display());
    let parsed = parse_launch(&default_registry(), &line);
    std::fs::remove_file(&path).ok();
    match parsed {
        Err(error) => assert_eq!(error, ParseError::UnrecognizedContent("filesrc".into())),
        Ok(_) => panic!("`{line}` parsed into a graph"),
    }
}

#[cfg(feature = "ffmpeg")]
#[test]
fn decodebin_still_sniffs_a_transport_stream_with_an_unmapped_extension() {
    let ts = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/av_h264_aac44100.ts"
    ))
    .expect("fixture reads");
    let path = write_temp("transport-stream", &ts);
    let line = format!("filesrc location={} ! decodebin ! fakesink", path.display());
    let parsed = parse_launch(&default_registry(), &line);
    std::fs::remove_file(&path).ok();
    let graph = parsed
        .unwrap_or_else(|e| panic!("parses `{line}`: {e}"))
        .finish()
        .expect("valid graph");
    let names: Vec<String> = graph
        .topo()
        .iter()
        .filter_map(|&node| graph.element(node).map(|e| e.log_category().to_string()))
        .collect();
    assert!(
        names.iter().any(|name| name == "TsDemux"),
        "the sniff types the file as MPEG-TS, got {names:?}"
    );
}
