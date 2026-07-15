//! WebSocket test source for the glass2glass browser demo.
//!
//! Streams an H.264 Annex-B elementary stream to any connected browser, **one
//! access unit per WebSocket binary message** (the framing `WebCodecsDecode`
//! expects), on a fixed-fps timer, looping forever. This is the sender side of the
//! `tools/wasm-demo` glass-to-glass loop; the browser runs
//! `run_websocket_to_canvas(ws_url, canvas_id)`.
//!
//! Standalone dev tool (not a g2g element): the AU split is a plain start-code
//! scan, not g2g's parser. Usage:
//!   cargo run --release -- [bind_addr=127.0.0.1:8080] [fixture=../../../g2g-plugins/tests/fixtures/h264_640x480.h264] [fps=15]

use std::time::Duration;

use futures_util::SinkExt;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message;

const DEFAULT_ADDR: &str = "127.0.0.1:8080";
const DEFAULT_FIXTURE: &str = "../../../g2g-plugins/tests/fixtures/h264_640x480.h264";
const DEFAULT_FPS: u64 = 15;

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let addr = args.next().unwrap_or_else(|| DEFAULT_ADDR.to_string());
    let fixture = args.next().unwrap_or_else(|| DEFAULT_FIXTURE.to_string());
    let fps: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_FPS);

    let stream = tokio::fs::read(&fixture)
        .await
        .unwrap_or_else(|e| panic!("read fixture {fixture}: {e}"));
    let aus = split_access_units(&stream);
    let owned: Vec<Vec<u8>> = aus.iter().map(|s| s.to_vec()).collect();
    println!(
        "ws-fixture-server: {} access units from {fixture} ({} bytes), {fps} fps, serving ws://{addr}",
        owned.len(),
        stream.len()
    );

    let listener = TcpListener::bind(&addr).await.unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    loop {
        let (tcp, peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                eprintln!("accept error: {e}");
                continue;
            }
        };
        println!("client connected: {peer}");
        let aus = owned.clone();
        tokio::spawn(async move {
            if let Err(e) = serve(tcp, aus, fps).await {
                println!("client {peer} disconnected: {e}");
            }
        });
    }
}

/// Upgrade one TCP connection to WebSocket and stream the access units on a timer,
/// looping until the peer goes away.
async fn serve(
    tcp: TcpStream,
    aus: Vec<Vec<u8>>,
    fps: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut ws = tokio_tungstenite::accept_async(tcp).await?;
    let period = Duration::from_millis(1000 / fps.max(1));
    let mut ticker = tokio::time::interval(period);
    loop {
        for au in &aus {
            ticker.tick().await;
            // A send error means the browser closed the socket; end this task.
            ws.send(Message::Binary(au.clone())).await?;
        }
    }
}

/// Split an Annex-B H.264 stream into one slice per access unit (start codes
/// preserved). An AU boundary opens at the first VCL NAL (type 1/5) after a
/// previous VCL NAL, so a leading SPS/PPS/SEI attaches to the following coded
/// picture. Matches the framing the browser decoder expects.
fn split_access_units(stream: &[u8]) -> Vec<&[u8]> {
    let mut nal_starts = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            nal_starts.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut aus = Vec::new();
    let mut au_start = 0;
    let mut seen_vcl = false;
    for &ns in &nal_starts {
        let Some(&hdr) = stream.get(ns + 3) else { continue };
        let nal_type = hdr & 0x1f;
        let is_vcl = nal_type == 1 || nal_type == 5;
        if is_vcl {
            if seen_vcl {
                let cut = if ns > 0 && stream[ns - 1] == 0 { ns - 1 } else { ns };
                aus.push(&stream[au_start..cut]);
                au_start = cut;
            }
            seen_vcl = true;
        }
    }
    if au_start < stream.len() {
        aus.push(&stream[au_start..]);
    }
    aus
}
