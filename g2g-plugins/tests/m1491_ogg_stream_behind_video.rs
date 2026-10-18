//! M1491: a bare `decodebin` on an Ogg file whose first logical bitstream is
//! video selects the audio bitstream behind it, not the Opus default port.

#![cfg(feature = "std")]

use g2g_core::{ByteStreamEncoding, Caps};
use g2g_plugins::uridecodebin::ogg_primary_stream;

const VORBIS: &[u8] = include_bytes!("fixtures/vorbis_stereo_48k.ogg");
const THEORA_IDENTIFICATION: &[u8] = b"\x80theora\x03\x02\x01";
const THEORA_SERIAL: u32 = 0x7E0A_0001;
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

#[test]
fn vorbis_behind_a_theora_stream_is_selected() {
    let mut ogg = bos_page(THEORA_SERIAL, THEORA_IDENTIFICATION);
    ogg.extend_from_slice(VORBIS);
    let path = std::env::temp_dir().join(format!("g2g-m1491-{}.ogg", std::process::id()));
    std::fs::write(&path, &ogg).expect("write ogg");
    let caps = Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    };
    let streams = ogg_primary_stream(path.to_str().expect("utf8 path"), &caps);
    std::fs::remove_file(&path).ok();
    let selections: Vec<_> = streams
        .expect("the ogg file has a readable stream")
        .streams
        .into_iter()
        .map(|stream| stream.props)
        .collect();
    assert_eq!(
        selections,
        vec![vec![("stream".to_string(), "vorbis".to_string())]]
    );
}
