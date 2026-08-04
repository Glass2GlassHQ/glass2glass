//! An MP4 whose `vide` track carries a codec the demuxer has no reader for
//! (MJPEG, which ffmpeg boxes as an `mp4v` sample entry with objectTypeIndication
//! 0x6C) demuxes its remaining tracks instead of failing the whole file. A
//! well-formed sample entry we decline skips that track, like an unknown handler
//! type; genuinely malformed track data still fails the parse.
//!
//! Fixture: `ffmpeg -f lavfi -i testsrc=size=160x120:rate=5 -f lavfi -i
//! sine=frequency=440:sample_rate=48000 -t 0.4 -c:v mjpeg -q:v 20 -pix_fmt
//! yuvj420p -c:a aac -b:a 32k -ac 1 mjpeg_aac.mp4`.

#![cfg(feature = "std")]

use g2g_core::{AudioFormat, Caps};
use g2g_plugins::mp4demuxn::forwardable_streams;

const MJPEG_AAC: &[u8] = include_bytes!("fixtures/mjpeg_aac.mp4");

/// Offset of the video sample entry's box header in the fixture (its `mp4v`
/// fourcc is unique in the file), so a test can corrupt it in place.
fn video_entry_fourcc() -> usize {
    let at = MJPEG_AAC
        .windows(4)
        .position(|w| w == b"mp4v")
        .expect("fixture has an mp4v video sample entry");
    assert!(
        MJPEG_AAC[at + 4..].windows(4).all(|w| w != b"mp4v"),
        "the fourcc is unique, so patching it hits the sample entry"
    );
    at
}

/// The AAC track still demuxes: the MJPEG track is skipped, not an error.
#[test]
fn unsupported_video_codec_skips_its_track() {
    let streams = forwardable_streams(MJPEG_AAC);
    assert_eq!(
        streams.len(),
        1,
        "only the audio track is forwardable: {streams:?}"
    );
    assert!(!streams[0].video, "the MJPEG track was skipped");
    assert_eq!(streams[0].track_id, 2, "the fixture's second trak");
    assert!(
        matches!(
            streams[0].caps,
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            }
        ),
        "the AAC track's caps came through: {:?}",
        streams[0].caps
    );
    assert!(
        !streams[0].config.is_empty(),
        "its AudioSpecificConfig parsed out of the esds"
    );
}

/// A sample entry naming a codec we DO read but missing its config record
/// (`avc1` with no `avcC`) is malformed, and still fails the whole parse rather
/// than being skipped as unsupported.
#[test]
fn malformed_video_entry_still_fails_the_parse() {
    let mut broken = MJPEG_AAC.to_vec();
    let at = video_entry_fourcc();
    broken[at..at + 4].copy_from_slice(b"avc1");
    assert!(
        forwardable_streams(&broken).is_empty(),
        "an avc1 entry with no avcC fails the parse, so no track is forwarded"
    );
}

/// Truncated / unparseable sample-entry data (a box size below the 8-byte header)
/// also fails the parse: only a well-formed entry earns a skip.
#[test]
fn unparseable_sample_entry_still_fails_the_parse() {
    let mut broken = MJPEG_AAC.to_vec();
    let size = video_entry_fourcc() - 4;
    broken[size..size + 4].copy_from_slice(&4u32.to_be_bytes());
    assert!(
        forwardable_streams(&broken).is_empty(),
        "a bogus box size fails the parse, so no track is forwarded"
    );
}
