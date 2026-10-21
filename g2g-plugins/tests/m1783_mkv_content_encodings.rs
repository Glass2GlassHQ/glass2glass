// Python compressed these files, in tools/mkv-content-encoding-fixtures.sh, and
// ffmpeg demuxes each single-encoding one like its plain file

use std::path::Path;

use g2g_plugins::matroska::MatroskaDemuxer;

const ASS_PLAIN: &str = "mkv_ass_plain.mks";
const TEXT_PLAIN: &str = "mkv_text_plain.mks";

fn demux(name: &str) -> MatroskaDemuxer {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let mut demux = MatroskaDemuxer::new();
    demux.push_data(&std::fs::read(&path).expect("the fixture is committed"));
    demux
}

fn blocks(demux: &mut MatroskaDemuxer) -> Vec<Vec<u8>> {
    demux
        .take_frames()
        .into_iter()
        .map(|frame| frame.data)
        .collect()
}

// the file's blocks and CodecPrivate as the plain file stores them
fn assert_decodes_like(encoded: &str, plain: &str) {
    let mut expected = demux(plain);
    let mut decoded = demux(encoded);
    assert!(!decoded.tracks()[0].unsupported_encoding, "{encoded}");
    assert_eq!(
        decoded.codec_private(1),
        expected.codec_private(1),
        "{encoded}"
    );
    let expected_blocks = blocks(&mut expected);
    assert!(!expected_blocks.is_empty(), "{plain} has blocks");
    assert_eq!(blocks(&mut decoded), expected_blocks, "{encoded}");
}

#[test]
fn header_strip_under_zlib_decodes_like_the_plain_file() {
    assert_decodes_like("mkv_text_strip_zlib.mks", TEXT_PLAIN);
}

#[test]
fn zlib_codec_private_decodes_like_the_plain_file() {
    assert!(demux(ASS_PLAIN).codec_private(1).is_some());
    assert_decodes_like("mkv_ass_zlib_codec_private.mks", ASS_PLAIN);
}

#[cfg(feature = "matroska-compression")]
#[test]
fn bzip2_and_lzo_blocks_and_codec_private_decode_like_the_plain_file() {
    assert!(demux(ASS_PLAIN).codec_private(1).is_some());
    assert_decodes_like("mkv_ass_bzip2.mks", ASS_PLAIN);
    assert_decodes_like("mkv_ass_lzo.mks", ASS_PLAIN);
}

#[cfg(not(feature = "matroska-compression"))]
#[test]
fn bzip2_and_lzo_tracks_are_flagged_without_the_feature() {
    let plain = blocks(&mut demux(ASS_PLAIN));
    for name in ["mkv_ass_bzip2.mks", "mkv_ass_lzo.mks"] {
        let mut stored = demux(name);
        assert!(stored.tracks()[0].unsupported_encoding, "{name}");
        let stored_blocks = blocks(&mut stored);
        assert_eq!(stored_blocks.len(), plain.len(), "{name}");
        assert_ne!(stored_blocks, plain, "{name} forwards the stored bytes");
    }
}
