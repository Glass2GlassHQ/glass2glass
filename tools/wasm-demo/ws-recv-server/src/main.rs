//! WebSocket receiver for the glass2glass browser SEND demo.
//!
//! Accepts one browser connection, appends every binary message (an H.264 Annex-B
//! access unit produced by `WebCodecsEncode` in the browser) to an output file, and
//! prints progress. When the browser closes the socket, the file is a complete
//! elementary stream you can play:  `ffplay received.h264`  /  `vlc received.h264`.
//!
//! This is the sink end of the browser glass-to-glass loop
//! (`PatternSrc -> WebCodecsEncode -> WebSocketSink` in wasm -> here). Standalone dev
//! tool. Usage:
//!   cargo run --release -- [bind_addr=127.0.0.1:8081] [out=received.h264]

use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

const DEFAULT_ADDR: &str = "127.0.0.1:8081";
const DEFAULT_OUT: &str = "received.h264";

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let addr = args.next().unwrap_or_else(|| DEFAULT_ADDR.to_string());
    let out = args.next().unwrap_or_else(|| DEFAULT_OUT.to_string());

    let listener = TcpListener::bind(&addr).await.unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    println!("ws-recv-server: listening ws://{addr}, writing to {out}");
    println!("point the browser demo's Send URL at ws://{addr}");

    loop {
        let (tcp, peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                eprintln!("accept error: {e}");
                continue;
            }
        };
        println!("client connected: {peer}");
        // One recording per connection; overwrite the file each time.
        let mut file = match tokio::fs::File::create(&out).await {
            Ok(f) => f,
            Err(e) => {
                eprintln!("create {out}: {e}");
                continue;
            }
        };
        let mut ws = match tokio_tungstenite::accept_async(tcp).await {
            Ok(ws) => ws,
            Err(e) => {
                eprintln!("ws handshake with {peer}: {e}");
                continue;
            }
        };
        let mut units = 0u64;
        let mut bytes = 0u64;
        while let Some(msg) = ws.next().await {
            match msg {
                Ok(Message::Binary(data)) => {
                    if file.write_all(&data).await.is_err() {
                        break;
                    }
                    units += 1;
                    bytes += data.len() as u64;
                }
                Ok(Message::Close(_)) | Err(_) => break,
                _ => {}
            }
        }
        let _ = file.flush().await;
        println!("client {peer} done: {units} access units, {bytes} bytes -> {out}");
        println!("play it:  ffplay {out}   (or)   vlc {out}");
    }
}
