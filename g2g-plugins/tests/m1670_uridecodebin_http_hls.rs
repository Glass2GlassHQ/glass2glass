#![cfg(feature = "hls")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/dash_live_server.rs"]
mod dash_live_server;

use collect_sink::CollectSink;
use g2g_core::frame::PipelinePacket;
use g2g_core::{ByteStreamEncoding, Caps, StreamType};
use g2g_plugins::registry::default_registry;
use g2g_plugins::uridecodebin::hls_uri_fanout;

const SEGMENT_FIXTURE: &str = "tests/fixtures/av_h264_aac44100.ts";
const SEGMENT_COUNT: usize = 2;

const MASTER: &str = "#EXTM3U\n\
    #EXT-X-STREAM-INF:BANDWIDTH=800000,CODECS=\"avc1.42c01e,mp4a.40.2\"\n\
    media.m3u8\n";

const MEDIA: &str = "#EXTM3U\n\
    #EXT-X-TARGETDURATION:4\n\
    #EXT-X-MEDIA-SEQUENCE:0\n\
    #EXTINF:4.0,\n\
    seg0.ts\n\
    #EXTINF:4.0,\n\
    seg1.ts\n\
    #EXT-X-ENDLIST\n";

fn segment() -> Vec<u8> {
    std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(SEGMENT_FIXTURE)).unwrap()
}

fn serve_hls() -> String {
    let body = segment();
    let manifest_url = dash_live_server::serve(move |path| match path {
        "/master.m3u8" => Some(MASTER.as_bytes().to_vec()),
        "/media.m3u8" => Some(MEDIA.as_bytes().to_vec()),
        "/seg0.ts" | "/seg1.ts" | "/clip.ts" => Some(body.clone()),
        _ => None,
    });
    manifest_url
        .strip_suffix("manifest.mpd")
        .expect("the server names its manifest")
        .to_string()
}

async fn run_uri_source(uri: &str) -> (Caps, Vec<u8>) {
    let reg = default_registry();
    let (mut source, caps) = reg.build_uri_source(uri).expect("http uri builds a source");
    source.intercept_caps().await.unwrap();
    source.configure_pipeline(&caps).unwrap();
    let mut sink = CollectSink::default();
    source.run(&mut sink).await.unwrap();
    let body = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(<[u8]>::to_vec),
            _ => None,
        })
        .flatten()
        .collect();
    (caps, body)
}

#[tokio::test]
async fn http_playlist_uri_streams_the_hls_segments() {
    let origin = serve_hls();
    let (caps, body) = run_uri_source(&format!("{origin}master.m3u8")).await;
    assert_eq!(
        caps,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs
        }
    );
    assert_eq!(
        body,
        segment().repeat(SEGMENT_COUNT),
        "the source reads the media playlist's segments, not the playlist text"
    );
}

#[tokio::test]
async fn http_non_playlist_uri_streams_the_body_under_sniffed_caps() {
    let origin = serve_hls();
    let (caps, body) = run_uri_source(&format!("{origin}clip.ts")).await;
    assert_eq!(
        caps,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs
        }
    );
    assert_eq!(body, segment());
}

#[test]
fn http_playlist_uri_fans_out_the_variant_streams() {
    let origin = serve_hls();
    let reg = default_registry();
    let fanout = hls_uri_fanout(&reg, &format!("{origin}master.m3u8"))
        .unwrap()
        .expect("an http master fans out like an hls:// one");
    assert_eq!(
        fanout
            .ports
            .iter()
            .map(|p| p.stream_type)
            .collect::<Vec<_>>(),
        vec![StreamType::Video, StreamType::Audio]
    );
    assert!(
        hls_uri_fanout(&reg, &format!("{origin}clip.ts"))
            .unwrap()
            .is_none(),
        "a body that is not a playlist is left to the other hooks"
    );
}
