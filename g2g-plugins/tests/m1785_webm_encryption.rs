// Shaka Packager encrypted these files with a raw key, in tools/webm-encryption-fixtures.sh,
// and ffmpeg's framemd5 of the clear inputs is what decryption must give back

use std::path::{Path, PathBuf};

use g2g_plugins::matroska::MatroskaDemuxer;
use md5::{Digest, Md5};

const VP9_ENCRYPTED: &str = "webm_vp9_encrypted.webm";
#[cfg(feature = "matroska-decrypt")]
const OPUS_ENCRYPTED: &str = "webm_opus_encrypted.webm";
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

// the key id and key the fixtures were encrypted with
#[cfg(feature = "matroska-decrypt")]
fn content_key() -> ([u8; 16], [u8; 16]) {
    let text =
        std::fs::read_to_string(fixture("webm_encryption.keys")).expect("the keys are committed");
    let field = |name: &str| -> [u8; 16] {
        let hex = text
            .lines()
            .find_map(|line| line.strip_prefix(name)?.strip_prefix('='))
            .expect("the field is there");
        core::array::from_fn(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex"))
    };
    (field("key_id"), field("key"))
}

fn demuxed(mut demux: MatroskaDemuxer, name: &str) -> MatroskaDemuxer {
    demux.push_data(&std::fs::read(fixture(name)).expect("the fixture is committed"));
    demux
}

fn blocks(demux: &mut MatroskaDemuxer) -> Vec<Vec<u8>> {
    demux
        .take_frames()
        .into_iter()
        .map(|frame| frame.data)
        .collect()
}

fn md5_hex(data: &[u8]) -> String {
    Md5::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// the hash column of ffmpeg's framemd5 lines: `stream, dts, pts, duration, size, hash`
fn ffmpeg_packet_md5s(name: &str) -> Vec<String> {
    let text = std::fs::read_to_string(fixture(name)).expect("the framemd5 sidecar is committed");
    text.lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            line.split(',')
                .nth(5)
                .expect("a hash column")
                .trim()
                .to_string()
        })
        .collect()
}

fn block_md5s(blocks: &[Vec<u8>]) -> Vec<String> {
    blocks.iter().map(|block| md5_hex(block)).collect()
}

#[cfg(feature = "matroska-decrypt")]
#[test]
fn partitioned_vp9_decrypts_to_the_clear_packets() {
    let mut demux = demuxed(
        MatroskaDemuxer::new().with_content_keys(&[content_key()]),
        VP9_ENCRYPTED,
    );
    assert!(!demux.tracks()[0].unsupported_encoding);
    let expected = ffmpeg_packet_md5s("webm_vp9_clear.webm.framemd5");
    assert!(!expected.is_empty());
    assert_eq!(block_md5s(&blocks(&mut demux)), expected);
}

#[cfg(feature = "matroska-decrypt")]
#[test]
fn opus_decrypts_to_the_clear_packets() {
    let mut demux = demuxed(
        MatroskaDemuxer::new().with_content_keys(&[content_key()]),
        OPUS_ENCRYPTED,
    );
    assert!(!demux.tracks()[0].unsupported_encoding);
    let expected = ffmpeg_packet_md5s("webm_opus_clear.webm.framemd5");
    assert!(!expected.is_empty());
    assert_eq!(block_md5s(&blocks(&mut demux)), expected);
}

// the stored frames whose signal byte is 0 lose that byte and nothing else
#[cfg(feature = "matroska-decrypt")]
#[test]
fn clear_frames_in_an_encrypted_track_pass_through() {
    let stored = blocks(&mut demuxed(MatroskaDemuxer::new(), OPUS_ENCRYPTED));
    let decrypted = blocks(&mut demuxed(
        MatroskaDemuxer::new().with_content_keys(&[content_key()]),
        OPUS_ENCRYPTED,
    ));
    let clear_lead = stored.iter().take_while(|block| block[0] == 0).count();
    assert!(clear_lead > 0, "the fixture opens with clear frames");
    assert!(clear_lead < stored.len(), "and goes on encrypted");
    for (stored, decrypted) in stored.iter().zip(&decrypted).take(clear_lead) {
        assert_eq!(decrypted[..], stored[1..]);
    }
}

// the demuxer element resets its parser this way for a seek that re-scans from offset 0
#[cfg(feature = "matroska-decrypt")]
#[test]
fn a_reset_parser_keeps_its_content_keys() {
    let mut demux = demuxed(
        MatroskaDemuxer::new().with_content_keys(&[content_key()]),
        VP9_ENCRYPTED,
    );
    demux.reset();
    let mut demux = demuxed(demux, VP9_ENCRYPTED);
    assert!(!demux.tracks()[0].unsupported_encoding);
    assert_eq!(
        block_md5s(&blocks(&mut demux)),
        ffmpeg_packet_md5s("webm_vp9_clear.webm.framemd5")
    );
}

#[cfg(feature = "matroska-decrypt")]
#[test]
fn a_key_for_another_key_id_leaves_the_track_flagged_and_stored() {
    let (key_id, key) = content_key();
    let mut other_key_id = key_id;
    other_key_id[15] ^= 1;
    let mut demux = demuxed(
        MatroskaDemuxer::new().with_content_keys(&[(other_key_id, key)]),
        VP9_ENCRYPTED,
    );
    assert!(demux.tracks()[0].unsupported_encoding);
    assert_eq!(
        block_md5s(&blocks(&mut demux)),
        ffmpeg_packet_md5s("webm_vp9_encrypted.webm.framemd5")
    );
}

#[cfg(not(feature = "matroska-decrypt"))]
#[test]
fn encrypted_tracks_are_flagged_and_stored_without_the_feature() {
    let mut demux = demuxed(MatroskaDemuxer::new(), VP9_ENCRYPTED);
    assert!(demux.tracks()[0].unsupported_encoding);
    assert_eq!(
        block_md5s(&blocks(&mut demux)),
        ffmpeg_packet_md5s("webm_vp9_encrypted.webm.framemd5")
    );
}
