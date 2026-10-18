//! M1546: a bare `decodebin` on an FLV selects the codec the file's tags carry,
//! so a VP6 FLV no longer plugs a decoder on the default H.264 port.

#![cfg(all(feature = "std", target_os = "linux", feature = "ffmpeg"))]

mod framemd5_common;
use framemd5_common::{decoded_i420_md5s, ffmpeg_frame_md5s};

const VP6F: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vp6f_112x80.flv"
);

#[tokio::test]
async fn vp6_flv_decodes_like_ffmpeg() {
    let expected = ffmpeg_frame_md5s(VP6F);
    assert_eq!(
        decoded_i420_md5s(VP6F, expected.frame_size).await,
        expected.md5s
    );
}
