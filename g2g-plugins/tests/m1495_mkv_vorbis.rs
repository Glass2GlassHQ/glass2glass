//! M1495: Matroska carries Vorbis (headers Xiph-laced in the `CodecPrivate`),
//! so a Vorbis WebM decodes through a bare `decodebin` instead of failing on
//! the demuxer's default video port.

#![cfg(all(feature = "std", feature = "vorbis"))]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::ogg::OggDemuxer;
use g2g_plugins::registry::default_registry;

const VORBIS_OGG: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vorbis_stereo_48k.ogg"
);
const CHANNELS: u64 = 2;
const SAMPLE_RATE: u32 = 48_000;
const TRACK_NUMBER: u8 = 1;
const KEYFRAME_FLAG: u8 = 0x80;
const AUDIO_TRACK_TYPE: u64 = 2;

struct ZeroClock;
impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn decode_to_pcm(source: &std::path::Path, pcm: &std::path::Path) -> Vec<u8> {
    let line = format!(
        "filesrc location={} ! decodebin ! audioconvert \
         ! audio/x-raw,format=S16LE,channels={CHANNELS},rate={SAMPLE_RATE} ! filesink location={}",
        source.display(),
        pcm.display()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let bytes = std::fs::read(pcm).expect("decoded pcm");
    std::fs::remove_file(pcm).ok();
    bytes
}

fn ebml(id: &[u8], body: &[u8]) -> Vec<u8> {
    const EIGHT_BYTE_SIZE_MARKER: u8 = 0x01;
    let mut out = id.to_vec();
    out.push(EIGHT_BYTE_SIZE_MARKER);
    out.extend_from_slice(&(body.len() as u64).to_be_bytes()[1..]);
    out.extend_from_slice(body);
    out
}

fn ebml_uint(id: &[u8], value: u64) -> Vec<u8> {
    ebml(id, &value.to_be_bytes())
}

fn xiph_laced(packets: &[&[u8]]) -> Vec<u8> {
    let mut out = vec![(packets.len() - 1) as u8];
    for packet in &packets[..packets.len() - 1] {
        let mut len = packet.len();
        while len >= 255 {
            out.push(255);
            len -= 255;
        }
        out.push(len as u8);
    }
    for packet in packets {
        out.extend_from_slice(packet);
    }
    out
}

// the Ogg fixture's Vorbis packets rewrapped as a WebM, one SimpleBlock each
fn vorbis_webm() -> Vec<u8> {
    let mut ogg = OggDemuxer::new();
    ogg.push_data(&std::fs::read(VORBIS_OGG).expect("fixture"));
    let headers = [
        ogg.head_header().expect("identification header").to_vec(),
        ogg.comment_header().expect("comment header").to_vec(),
        ogg.setup_header().expect("setup header").to_vec(),
    ];
    let packets = ogg.take_packets();
    let header_slices: Vec<&[u8]> = headers.iter().map(Vec::as_slice).collect();
    let audio = ebml(
        &[0xE1],
        &[
            ebml(&[0xB5], &f64::from(SAMPLE_RATE).to_be_bytes()),
            ebml_uint(&[0x9F], CHANNELS),
        ]
        .concat(),
    );
    let entry = [
        ebml_uint(&[0xD7], u64::from(TRACK_NUMBER)),
        ebml_uint(&[0x83], AUDIO_TRACK_TYPE),
        ebml(&[0x86], b"A_VORBIS"),
        ebml(&[0x63, 0xA2], &xiph_laced(&header_slices)),
        audio,
    ]
    .concat();
    let tracks = ebml(&[0x16, 0x54, 0xAE, 0x6B], &ebml(&[0xAE], &entry));
    let mut cluster = ebml_uint(&[0xE7], 0);
    for packet in &packets {
        let block = [&[0x80 | TRACK_NUMBER, 0, 0, KEYFRAME_FLAG][..], packet].concat();
        cluster.extend_from_slice(&ebml(&[0xA3], &block));
    }
    let segment = ebml(
        &[0x18, 0x53, 0x80, 0x67],
        &[tracks, ebml(&[0x1F, 0x43, 0xB6, 0x75], &cluster)].concat(),
    );
    let header = ebml(&[0x1A, 0x45, 0xDF, 0xA3], &ebml(&[0x42, 0x82], b"webm"));
    [header, segment].concat()
}

fn temp(tag: &str, ext: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g-m1495-{tag}-{}.{ext}", std::process::id()))
}

#[tokio::test]
async fn vorbis_in_webm_decodes_like_the_ogg_it_came_from() {
    let webm = temp("vorbis", "webm");
    std::fs::write(&webm, vorbis_webm()).expect("write webm");
    let from_webm = decode_to_pcm(&webm, &temp("webm", "pcm")).await;
    std::fs::remove_file(&webm).ok();
    let from_ogg = decode_to_pcm(std::path::Path::new(VORBIS_OGG), &temp("ogg", "pcm")).await;
    // the Ogg end granule trims the last packet, a Matroska block keeps it all
    assert!(
        from_webm.len() >= from_ogg.len() && !from_ogg.is_empty(),
        "webm {} bytes, ogg {} bytes",
        from_webm.len(),
        from_ogg.len()
    );
    assert_eq!(&from_webm[..from_ogg.len()], &from_ogg[..]);
}
