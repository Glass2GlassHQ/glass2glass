//! Browser WebSocket sink for the distributed-graph primitive (M554, `web`):
//! the wasm send half, the browser sibling of the native
//! [`RemoteWsSink`](crate::remotewssink).
//!
//! `WsWireSink` accepts *any* caps and forwards the whole `PipelinePacket`
//! stream (the leading `CapsChanged`, `Segment`, every `DataFrame`, mid-stream
//! caps refinement, `Eos`) over a browser `WebSocket`, each packet serialized by
//! [`g2g_core::wire`] and sent as one binary message. It speaks the *identical*
//! wire protocol as the native pair, so a browser graph can cut an edge and ship
//! its downstream subgraph to a native [`RemoteWsSrc`](crate::remotewssrc)
//! server: `... -> WsWireSink` in the browser, `RemoteWsSrc -> ...` on the
//! server. This is the media-agnostic generalization of the bespoke M549
//! `WebRemoteDetect` shim (which hand-rolled a browser->server frame protocol);
//! the wire codec compiles on wasm32 with no changes, so the browser and a native
//! peer literally share the serializer.
//!
//! Unlike the raw-bytes [`WebSocketSink`](crate::websocketsink) (which ships an
//! H.264 elementary stream, one access unit per message, for a specific demo),
//! this ships wire-encoded `PipelinePacket`s: caps + timing + sequence +
//! per-frame metadata all cross the boundary, so the far side reconstructs the
//! exact stream. Only CPU-memory frames serialize; a device-resident frame
//! yields [`G2gError::UnsupportedDomain`](g2g_core::G2gError), as the wire codec
//! requires.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;

use g2g_core::wire::encode_packet;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, HardwareError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket,
};

use alloc::vec::Vec;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Event, WebSocket};

use crate::remotewire::map_wire;
use crate::webutil::Inbox;

pub struct WsWireSink {
    url: String,
    socket: Option<WebSocket>,
    open_inbox: Option<Inbox<()>>,
    _on_open: Option<Closure<dyn FnMut(Event)>>,
    opened: bool,
    configured: bool,
    /// The caps the runner handed `configure_pipeline`, sent (deduped against
    /// `last_sent`) as the leading wire `CapsChanged` before the first frame.
    configured_caps: Option<Caps>,
    last_sent: Option<Caps>,
    sent: u64,
}

impl core::fmt::Debug for WsWireSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WsWireSink")
            .field("url", &self.url)
            .field("configured", &self.configured)
            .field("opened", &self.opened)
            .field("sent", &self.sent)
            .finish_non_exhaustive()
    }
}

impl WsWireSink {
    /// Ship the packet stream to `url` (a native `RemoteWsSrc`, e.g.
    /// `ws://host:port`). The socket opens in `configure_pipeline`; the first send
    /// waits for it to connect.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            socket: None,
            open_inbox: None,
            _on_open: None,
            opened: false,
            configured: false,
            configured_caps: None,
            last_sent: None,
            sent: 0,
        }
    }

    /// Count of wire packets sent (caps + data + control). Useful in tests.
    pub fn sent(&self) -> u64 {
        self.sent
    }

    /// Wire-encode one packet and send it as a single binary WebSocket message,
    /// waiting for the socket to open on the first send.
    async fn send(&mut self, packet: &PipelinePacket) -> Result<(), G2gError> {
        let body: Vec<u8> = encode_packet(packet).map_err(map_wire)?;
        let err = || G2gError::Hardware(HardwareError::Other);
        if !self.opened {
            if let Some(inbox) = &self.open_inbox {
                inbox.next().await;
            }
            self.opened = true;
        }
        let socket = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
        socket.send_with_u8_array(&body).map_err(|_| err())?;
        self.sent += 1;
        Ok(())
    }
}

impl AsyncElement for WsWireSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // A generic transport: whatever upstream produces is what we ship.
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Record the current caps (sent as the leading wire CapsChanged in
        // `process`). Idempotent socket open: the runner re-calls configure on
        // every mid-stream CapsChanged, and recreating the socket would discard
        // the open connection (and its buffered `onopen`), hanging the first send.
        self.configured_caps = Some(absolute_caps.clone());
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
            // Emit the current caps as the leading ordered `CapsChanged` before any
            // data, deduped so a repeated configure (or the runner-forwarded
            // mid-stream `CapsChanged` packet) sends the caps exactly once.
            if let Some(caps) = self.configured_caps.clone() {
                if self.last_sent.as_ref() != Some(&caps) {
                    self.send(&PipelinePacket::CapsChanged(caps.clone())).await?;
                    self.last_sent = Some(caps);
                }
            }
            match packet {
                PipelinePacket::CapsChanged(caps) => {
                    if self.last_sent.as_ref() != Some(&caps) {
                        self.send(&PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_sent = Some(caps);
                    }
                }
                PipelinePacket::Eos => {
                    self.send(&PipelinePacket::Eos).await?;
                    if let Some(socket) = self.socket.as_ref() {
                        let _ = socket.close();
                    }
                }
                other => self.send(&other).await?,
            }
            Ok(())
        })
    }
}

impl PadTemplates for WsWireSink {
    /// Wildcard sink: accepts any caps (a media-agnostic transport), matching
    /// `caps_constraint_as_sink` of `AcceptsAny`.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}
