//! Browser remote transform for the distributed-graph primitive (M555, `web`):
//! the wasm twin of the native [`RemoteWsTransform`](crate::remotewstransform),
//! and the generic replacement for the bespoke `WebRemoteDetect` shim.
//!
//! `WsWireTransform` offloads a single middle graph stage to a remote peer: it
//! ships each input packet to `url` over one `WebSocket` (serialized by
//! [`g2g_core::wire`]) and emits the processed packet the peer returns, so the
//! stages *around* it (decode, overlay, canvas) stay in the browser while the
//! offloaded one (e.g. detection inference) runs on a native server. Where
//! `WebRemoteDetect` hand-rolled an RGBA-up / boxes-down protocol that knew about
//! detection, this knows nothing about the stage: it round-trips whole
//! `PipelinePacket`s (metadata included, e.g. `AnalyticsMeta` detections), so the
//! browser element is reusable for any remote stage and the server may run any
//! g2g subgraph behind a `RemoteWsSrc`-style intake.
//!
//! Caps are identity (the remote stage may attach metadata but does not change
//! the format). The protocol matches the native transform, kept strictly FIFO so
//! each per-frame read pairs with its frame: the leading `CapsChanged` (config,
//! no reply), then one `DataFrame` per frame (one processed reply each), then
//! `Eos`; `Segment` / `Flush` pass through locally. Requests are serialized (one
//! frame in flight; the linear browser chain processes one frame at a time).
//!
//! Bandwidth note: this round-trips the whole frame both ways, the honest cost of
//! a generic packet-in / packet-out transform (a `metadata`-only return is a
//! future optimization for the pixels-unchanged case).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::wire::{decode_packet, encode_packet};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, HardwareError,
    OutputSink, PipelinePacket,
};

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{BinaryType, CloseEvent, Event, MessageEvent, WebSocket};

use crate::remotewire::map_wire;
use crate::webutil::Inbox;

pub struct WsWireTransform {
    url: String,
    socket: Option<WebSocket>,
    open_inbox: Option<Inbox<()>>,
    /// Processed packets from the peer, one per frame sent (the callback -> async
    /// bridge for `onmessage`).
    msg_inbox: Option<Inbox<Vec<u8>>>,
    _on_open: Option<Closure<dyn FnMut(Event)>>,
    _on_message: Option<Closure<dyn FnMut(MessageEvent)>>,
    _on_close: Option<Closure<dyn FnMut(CloseEvent)>>,
    opened: bool,
    configured: bool,
    /// Caps recorded in `configure_pipeline`, sent to the peer (deduped) as the
    /// leading `CapsChanged` so its subgraph configures before the first frame.
    configured_caps: Option<Caps>,
    last_sent: Option<Caps>,
    emitted: u64,
}

impl core::fmt::Debug for WsWireTransform {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WsWireTransform")
            .field("url", &self.url)
            .field("configured", &self.configured)
            .field("opened", &self.opened)
            .field("emitted", &self.emitted)
            .finish_non_exhaustive()
    }
}

impl WsWireTransform {
    /// Offload the middle stage to `url` (a native `RemoteWsSrc`-style peer that
    /// replies one processed frame per frame). The socket opens in
    /// `configure_pipeline`; the first send waits for it to connect.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            socket: None,
            open_inbox: None,
            msg_inbox: None,
            _on_open: None,
            _on_message: None,
            _on_close: None,
            opened: false,
            configured: false,
            configured_caps: None,
            last_sent: None,
            emitted: 0,
        }
    }

    /// Count of processed frames emitted downstream. Useful in tests.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// Wire-encode and send one packet as a binary message, waiting for the
    /// socket to open on the first send.
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
        Ok(())
    }

    /// Send the current caps to the peer once, deduped, so its subgraph
    /// configures before the first frame.
    async fn send_caps_if_new(&mut self) -> Result<(), G2gError> {
        if let Some(caps) = self.configured_caps.clone() {
            if self.last_sent.as_ref() != Some(&caps) {
                self.send(&PipelinePacket::CapsChanged(caps.clone())).await?;
                self.last_sent = Some(caps);
            }
        }
        Ok(())
    }
}

impl AsyncElement for WsWireTransform {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Identity, media-agnostic: the remote stage may add metadata but does
        // not change the format.
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| CapsSet::one(input.clone())))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured_caps = Some(absolute_caps.clone());
        // Idempotent socket open: the runner re-calls configure on every
        // mid-stream CapsChanged; recreating the socket would drop the open
        // connection (and its buffered onopen), hanging the first send.
        if self.configured {
            return Ok(ConfigureOutcome::Accepted);
        }
        let err = || G2gError::Hardware(HardwareError::Other);
        let socket = WebSocket::new(&self.url).map_err(|_| err())?;
        socket.set_binary_type(BinaryType::Arraybuffer);

        let open_inbox: Inbox<()> = Inbox::new();
        let on_open = {
            let tx = open_inbox.sender();
            Closure::<dyn FnMut(Event)>::new(move |_e: Event| tx.push(()))
        };
        socket.set_onopen(Some(on_open.as_ref().unchecked_ref()));

        // Each processed frame comes back as a binary message.
        let msg_inbox: Inbox<Vec<u8>> = Inbox::new();
        let on_message = {
            let tx = msg_inbox.sender();
            Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
                if let Ok(buf) = e.data().dyn_into::<js_sys::ArrayBuffer>() {
                    let bytes = js_sys::Uint8Array::new(buf.as_ref()).to_vec();
                    if !bytes.is_empty() {
                        tx.push(bytes);
                    }
                }
            })
        };
        socket.set_onmessage(Some(on_message.as_ref().unchecked_ref()));

        // A close ends the response stream (unblocks a pending read as None).
        let on_close = {
            let tx = msg_inbox.sender();
            Closure::<dyn FnMut(CloseEvent)>::new(move |_e: CloseEvent| tx.close())
        };
        socket.set_onclose(Some(on_close.as_ref().unchecked_ref()));

        self.socket = Some(socket);
        self.open_inbox = Some(open_inbox);
        self.msg_inbox = Some(msg_inbox);
        self._on_open = Some(on_open);
        self._on_message = Some(on_message);
        self._on_close = Some(on_close);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            // The peer's subgraph must see the caps before the first frame.
            self.send_caps_if_new().await?;

            match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.send(&PipelinePacket::DataFrame(frame)).await?;
                    // Exactly one processed packet per frame (the peer never
                    // echoes control), so this read pairs with our frame.
                    let bytes = {
                        let inbox = self.msg_inbox.as_ref().ok_or(G2gError::NotConfigured)?;
                        inbox.next().await
                    };
                    let bytes = bytes.ok_or(G2gError::Hardware(HardwareError::Other))?;
                    let processed = decode_packet(&bytes).map_err(map_wire)?;
                    self.emitted += 1;
                    out.push(processed).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    if self.last_sent.as_ref() != Some(&caps) {
                        self.send(&PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_sent = Some(caps.clone());
                    }
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                PipelinePacket::Eos => {
                    let _ = self.send(&PipelinePacket::Eos).await;
                    if let Some(socket) = self.socket.as_ref() {
                        let _ = socket.close();
                    }
                }
                // Segment / Flush pass through locally (not sent to the peer).
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}
