//! M480 - parse-time content sniff for `decodebin`. `bytestream-format=auto` now
//! sniffs the file header at PARSE time (not just run time), so `decodebin` picks
//! the demuxer from the real content even when the extension lies, the way
//! GStreamer's runtime `typefind` would. `FileSrc::probe_output_caps` is the hook
//! the `decodebin` expansion calls; it reads the header, where the no-I/O
//! `configured_output_caps` returns `None` for an unresolved `auto`.

#![cfg(feature = "std")]

use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, PropValue};
use g2g_plugins::filesrc::FileSrc;

/// A minimal ISO-BMFF header: a `ftyp` box, enough for the content sniff.
fn ftyp_bytes() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&0x18u32.to_be_bytes());
    b.extend_from_slice(b"ftypisom");
    b.extend_from_slice(&[0, 0, 0x02, 0]);
    b.extend_from_slice(b"isomiso2mp41");
    b
}

fn write_temp(tag: &str, ext: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m480-{}-{}.{}", std::process::id(), tag, ext));
    std::fs::write(&path, bytes).expect("write temp");
    path
}

/// A file whose bytes are MP4 but whose extension is `.ts` sniffs to the whole-file
/// `Mp4` type at parse time under `bytestream-format=auto`, so `decodebin` would
/// plug the MP4 demuxer, not the (wrong) `tsdemux` the extension implies.
#[test]
fn auto_sniff_overrides_a_wrong_extension_at_parse_time() {
    let path = write_temp("mislabeled", "ts", &ftyp_bytes());
    let mut fs = FileSrc::untyped();
    fs.set_property("location", PropValue::Str(path.to_str().unwrap().into())).unwrap();
    fs.set_property("bytestream-format", PropValue::Str("auto".into())).unwrap();
    let caps = fs.probe_output_caps();
    std::fs::remove_file(&path).ok();
    assert_eq!(
        caps,
        Some(Caps::ByteStream { encoding: ByteStreamEncoding::Mp4 }),
        "content sniff at parse time detects MP4 despite the .ts extension"
    );
}

/// Without `bytestream-format=auto`, the extension is trusted: the same `.ts` file
/// types as MPEG-TS at parse time (no content sniff), which is why a mislabeled
/// file needs `auto` (or an explicit format) to decode.
#[test]
fn extension_alone_is_trusted_without_auto() {
    let path = write_temp("trusted", "ts", &ftyp_bytes());
    let mut fs = FileSrc::untyped();
    fs.set_property("location", PropValue::Str(path.to_str().unwrap().into())).unwrap();
    let caps = fs.probe_output_caps();
    std::fs::remove_file(&path).ok();
    assert_eq!(
        caps,
        Some(Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }),
        "the .ts extension types as MPEG-TS without a content sniff"
    );
}

/// An unreadable `auto` file leaves the type unresolved (`None`), so `decodebin`
/// falls back to the declared default rather than failing the parse.
#[test]
fn auto_probe_on_a_missing_file_is_none() {
    let mut fs = FileSrc::untyped();
    fs.set_property("location", PropValue::Str("/no/such/file.mp4".into())).unwrap();
    fs.set_property("bytestream-format", PropValue::Str("auto".into())).unwrap();
    assert_eq!(fs.probe_output_caps(), None, "a missing auto file resolves to no caps, not a panic");
}
