//! M1544: `decodebin` splices `mpegvideoparse` ahead of the MPEG-2 decoder, so a
//! transport stream whose PES packets split field pictures decodes bit-exact.

#![cfg(all(feature = "std", target_os = "linux", feature = "ffmpeg"))]

mod framemd5_common;
use framemd5_common::{decoded_i420_md5s, ffmpeg_frame_md5s};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mpeg2_field_pictures_unaligned_pes.ts"
);

#[tokio::test]
async fn field_coded_mpeg2_in_ts_decodes_like_ffmpeg() {
    let expected = ffmpeg_frame_md5s(FIXTURE);
    assert_eq!(
        decoded_i420_md5s(FIXTURE, expected.frame_size).await,
        expected.md5s
    );
}
