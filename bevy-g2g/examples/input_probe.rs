//! Tiny WebSocket client for the input backchannel: sends a few input
//! messages and exits. Doubles as protocol documentation and as the automated
//! validation peer for `RemoteInputPlugin`.

use futures_util::SinkExt;
use tokio_tungstenite::tungstenite::Message;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ws://127.0.0.1:8877".into());
    let (mut ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("connect");
    for msg in [
        r#"{"type":"key","code":"KeyW","down":true}"#,
        r#"{"type":"mouse_move","dx":2.0,"dy":1.0}"#,
        r#"{"type":"wheel","dx":0.0,"dy":-1.0}"#,
        r#"{"type":"key","code":"KeyW","down":false}"#,
    ] {
        ws.send(Message::Text(msg.into())).await.expect("send");
    }
    // Give the server a beat to drain before the close tears the stream down.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    ws.close(None).await.ok();
    println!("input probe sent 4 messages to {url}");
}
