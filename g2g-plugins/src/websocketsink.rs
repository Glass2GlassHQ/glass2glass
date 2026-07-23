//! WebSocket egress sink (browser/wasm). Opens a `WebSocket` and sends each frame's
//! bytes as a binary message: the send side of the browser pipeline
//! (`PatternSrc -> WebCodecsEncode -> WebSocketSink`), the egress analog of
//! `WebSocketSrc`. The peer (e.g. a native server) receives the H.264 Annex-B
//! elementary stream one access unit per message.
//!
//! The socket open is async (the `onopen` event), so the first frame awaits it via
//! an [`crate::webutil::Inbox`] before sending; later frames send directly.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, HardwareError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec,
};

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Event, WebSocket};

use crate::webutil::Inbox;

pub struct WebSocketSink {
    url: String,
    socket: Option<WebSocket>,
    open_inbox: Option<Inbox<()>>,
    _on_open: Option<Closure<dyn FnMut(Event)>>,
    opened: bool,
    configured: bool,
    sent: u64,
}

impl core::fmt::Debug for WebSocketSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WebSocketSink")
            .field("url", &self.url)
            .field("configured", &self.configured)
            .field("opened", &self.opened)
            .field("sent", &self.sent)
            .finish_non_exhaustive()
    }
}

impl WebSocketSink {
    /// Send frames to `url` (e.g. `ws://host:port`). The socket opens in
    /// `configure_pipeline`; the first send waits for it to connect.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            socket: None,
            open_inbox: None,
            _on_open: None,
            opened: false,
            configured: false,
            sent: 0,
        }
    }

    /// Count of binary messages sent. Useful in tests.
    pub fn sent(&self) -> u64 {
        self.sent
    }

    async fn send_bytes(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        let err = || G2gError::Hardware(HardwareError::Other);
        // Wait for the socket to open before the first send.
        if !self.opened {
            if let Some(inbox) = &self.open_inbox {
                inbox.next().await;
            }
            self.opened = true;
        }
        let socket = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
        socket.send_with_u8_array(bytes).map_err(|_| err())?;
        self.sent += 1;
        Ok(())
    }
}

impl AsyncElement for WebSocketSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&h264_any())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(h264_any()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Idempotent: the socket is caps-independent, and the runner re-calls
        // configure on every mid-stream CapsChanged (e.g. the encoder's output-caps
        // announce). Recreating the socket here would discard an already-open
        // connection (and its buffered `onopen` signal), hanging the first send.
        if self.configured {
            return Ok(ConfigureOutcome::Accepted);
        }
        let err = || G2gError::Hardware(HardwareError::Other);
        let socket = WebSocket::new(&self.url).map_err(|_| err())?;
        let inbox: Inbox<()> = Inbox::new();
        let on_open = {
            let tx = inbox.sender();
            Closure::<dyn FnMut(Event)>::new(move |_e: Event| tx.push(()))
        };
        socket.set_onopen(Some(on_open.as_ref().unchecked_ref()));
        self.socket = Some(socket);
        self.open_inbox = Some(inbox);
        self._on_open = Some(on_open);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // Copy out of the borrow so the send future owns its bytes.
                    let bytes: Vec<u8> = slice.to_vec();
                    self.send_bytes(&bytes).await?;
                }
                PipelinePacket::Eos => {
                    if let Some(socket) = self.socket.as_ref() {
                        let _ = socket.close();
                    }
                }
                PipelinePacket::CapsChanged(_)
                | PipelinePacket::Flush
                | PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for WebSocketSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(h264_any()))])
    }
}

fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}
