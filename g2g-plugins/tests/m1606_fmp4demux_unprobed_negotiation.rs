#![cfg(all(feature = "dash", feature = "ffmpeg", target_os = "linux"))]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use g2g_core::runtime::{parse_launch, run_graph, LatencyProfile};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

// an ffmpeg-authored LL-DASH stream, the recipe is in m888_dash_cmaf_chunked
const MPD: &[u8] = include_bytes!("fixtures/cmaf_ldash.mpd");
const INIT: &[u8] = include_bytes!("fixtures/cmaf_ldash_init.m4s");
const SEGMENT: &[u8] = include_bytes!("fixtures/cmaf_ldash_chunked.m4s");

fn serve() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { break };
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                request.push(byte[0]);
                if request.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let request = String::from_utf8_lossy(&request);
            let path = request.split(' ').nth(1).unwrap_or_default();
            let body = match path {
                "/manifest.mpd" => MPD,
                "/init-stream0.m4s" => INIT,
                "/chunk-stream0-00001.m4s" => SEGMENT,
                _ => {
                    let _ =
                        stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                    continue;
                }
            };
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body);
        }
    });
    format!("http://127.0.0.1:{port}/manifest.mpd")
}

// a DASH or HLS source cannot be probed before the run, so the demuxer negotiates on placeholder geometry
#[tokio::test]
async fn fmp4demux_behind_dashsrc_negotiates_before_the_moov_arrives() {
    let line = format!(
        "dashsrc location={} ! fmp4demux ! h264parse ! avdec_h264 ! fakesink",
        serve()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = run_graph(graph, &WallClock::new(), LatencyProfile::Throughput)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    assert!(stats.frames_consumed > 0, "decoded frames reach the sink");
}
