//! M1568: a bare `decodebin` on Ogg offers a VP8 or Theora stream to a video
//! consumer ahead of the audio behind it.

#![cfg(feature = "std")]

use g2g_core::{ByteStreamEncoding, Caps, VideoCodec};
use g2g_plugins::uridecodebin::ogg_primary_stream;

const VORBIS: &[u8] = include_bytes!("fixtures/vorbis_stereo_48k.ogg");
const VIDEO_SERIAL: u32 = 0x7E0A_0001;
const BEGINNING_OF_STREAM: u8 = 0x02;

fn bos_page(serial: u32, packet: &[u8]) -> Vec<u8> {
    let mut out = b"OggS".to_vec();
    out.push(0);
    out.push(BEGINNING_OF_STREAM);
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&serial.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    // the CRC is not checked on read
    out.extend_from_slice(&0u32.to_le_bytes());
    out.push(1);
    out.push(packet.len() as u8);
    out.extend_from_slice(packet);
    out
}

// the VP8 mapping's stream-info header: 320x240 at square pixels and 30 fps
fn vp8_stream_info() -> Vec<u8> {
    const MAPPING_VERSION: [u8; 2] = [1, 0];
    const WIDTH: u16 = 320;
    const HEIGHT: u16 = 240;
    const ONE_24_BIT: [u8; 3] = [0, 0, 1];
    const FRAMERATE_NUMERATOR: u32 = 30;
    const FRAMERATE_DENOMINATOR: u32 = 1;
    [
        b"OVP80\x01".as_slice(),
        &MAPPING_VERSION,
        &WIDTH.to_be_bytes(),
        &HEIGHT.to_be_bytes(),
        &ONE_24_BIT,
        &ONE_24_BIT,
        &FRAMERATE_NUMERATOR.to_be_bytes(),
        &FRAMERATE_DENOMINATOR.to_be_bytes(),
    ]
    .concat()
}

// a video stream's identification page ahead of the Vorbis stream, written to a temp file
fn video_then_vorbis(tag: &str, video_identification: &[u8]) -> std::path::PathBuf {
    let mut ogg = bos_page(VIDEO_SERIAL, video_identification);
    ogg.extend_from_slice(VORBIS);
    let path = std::env::temp_dir().join(format!("g2g-m1568-{tag}-{}.ogg", std::process::id()));
    std::fs::write(&path, &ogg).expect("write ogg");
    path
}

#[test]
fn vp8_is_the_first_stream_decodebin_tries() {
    let path = video_then_vorbis("vp8", &vp8_stream_info());
    let caps = Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    };
    let streams = ogg_primary_stream(path.to_str().expect("utf8 path"), &caps);
    std::fs::remove_file(&path).ok();
    let streams = streams.expect("the ogg file has readable streams").streams;
    let selections: Vec<_> = streams.iter().map(|stream| stream.props.clone()).collect();
    assert_eq!(
        selections,
        vec![
            vec![("stream".to_string(), "vp8".to_string())],
            vec![("stream".to_string(), "vorbis".to_string())],
        ]
    );
    assert!(matches!(
        streams[0].caps,
        Caps::CompressedVideo {
            codec: VideoCodec::Vp8,
            ..
        }
    ));
}

// a Theora identification header: 320x240 frame, 300x200 picture, 25 fps
fn theora_identification() -> Vec<u8> {
    const VERSION: [u8; 3] = [3, 2, 1];
    const FRAME_MACROBLOCKS: [u16; 2] = [20, 15];
    const PICTURE: [u8; 6] = [0, 0x01, 0x2C, 0, 0, 0xC8];
    const PICTURE_OFFSET: [u8; 2] = [0, 0];
    const FRAMERATE: (u32, u32) = (25, 1);
    const TRAILER: [u8; 12] = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0xC0, 0];
    [
        b"\x80theora".as_slice(),
        &VERSION,
        &FRAME_MACROBLOCKS[0].to_be_bytes(),
        &FRAME_MACROBLOCKS[1].to_be_bytes(),
        &PICTURE,
        &PICTURE_OFFSET,
        &FRAMERATE.0.to_be_bytes(),
        &FRAMERATE.1.to_be_bytes(),
        &TRAILER,
    ]
    .concat()
}

#[test]
fn theora_is_the_first_stream_decodebin_tries() {
    let path = video_then_vorbis("theora", &theora_identification());
    let caps = Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    };
    let streams = ogg_primary_stream(path.to_str().expect("utf8 path"), &caps);
    std::fs::remove_file(&path).ok();
    let streams = streams.expect("the ogg file has readable streams").streams;
    let selections: Vec<_> = streams.iter().map(|stream| stream.props.clone()).collect();
    assert_eq!(
        selections,
        vec![
            vec![("stream".to_string(), "theora".to_string())],
            vec![("stream".to_string(), "vorbis".to_string())],
        ]
    );
    assert!(matches!(
        streams[0].caps,
        Caps::CompressedVideo {
            codec: VideoCodec::Theora,
            ..
        }
    ));
}
