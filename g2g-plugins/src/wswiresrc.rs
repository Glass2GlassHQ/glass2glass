//! Browser WebSocket source for the distributed-graph primitive (`web`): the
//! wasm receive half, the browser sibling of the native
//! [`RemoteWsSrc`](crate::remotewssrc) and the inverse of
//! [`WsWireSink`](crate::wswiresink).
//!
//! A browser can only dial out, so the native peer is the listening side
//! (`RemoteWsSink listen=true`) and pushes its stream down the accepted socket.
//! `WsWireSrc` opens the socket, reads the leading `CapsChanged` to discover
//! what the stream is, and emits every packet after it: frames with their
//! timing, sequence and metadata intact, ending on the sender's `Eos` or a
//! clean close.
//!
//! Unlike the raw-bytes [`WebSocketSrc`](crate::websocketsrc) (whose caps the
//! caller has to declare, since a byte stream carries none), this reconstructs
//! the exact `PipelinePacket` stream the sender serialized, so a native graph
//! can cut an edge and hand the rest to a browser.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::runtime::SourceLoop;
use g2g_core::wire::decode_packet;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, HardwareError, OutputSink,
    PipelinePacket,
};

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{BinaryType, CloseEvent, Event, MessageEvent, WebSocket};

use crate::remotewire::map_wire;
use crate::webutil::Inbox;

/// # Example
///
/// ```ignore
/// use g2g_plugins::wswiresrc::WsWireSrc;
///
/// let src = WsWireSrc::new("ws://localhost:9601");
/// ```
pub struct WsWireSrc {
    url: String,
    socket: Option<WebSocket>,
    /// Wire messages from the `onmessage` callback, the callback -> async
    /// bridge.
    inbox: Option<Inbox<Vec<u8>>>,
    _on_message: Option<Closure<dyn FnMut(MessageEvent)>>,
    _on_close: Option<Closure<dyn FnMut(CloseEvent)>>,
    _on_error: Option<Closure<dyn FnMut(Event)>>,
    /// The caps read off the wire in `intercept_caps`, re-emitted as the
    /// leading packet of `run`.
    discovered: Option<Caps>,
    configured: bool,
}

impl core::fmt::Debug for WsWireSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WsWireSrc")
            .field("url", &self.url)
            .field("discovered", &self.discovered)
            .field("configured", &self.configured)
            .finish_non_exhaustive()
    }
}

impl WsWireSrc {
    /// Read the packet stream a peer serves at `url` (a native
    /// `RemoteWsSink listen=true`, e.g. `ws://127.0.0.1:9601`).
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            socket: None,
            inbox: None,
            _on_message: None,
            _on_close: None,
            _on_error: None,
            discovered: None,
            configured: false,
        }
    }

    /// Open the socket and install the callbacks, once.
    fn open(&mut self) -> Result<(), G2gError> {
        if self.socket.is_some() {
            return Ok(());
        }
        let err = || G2gError::Hardware(HardwareError::Other);
        let socket = WebSocket::new(&self.url).map_err(|_| err())?;
        socket.set_binary_type(BinaryType::Arraybuffer);

        let inbox: Inbox<Vec<u8>> = Inbox::new();
        let on_message = {
            let tx = inbox.sender();
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

        // A close or an error ends the stream (an unblocked read returns None).
        let on_close = {
            let tx = inbox.sender();
            Closure::<dyn FnMut(CloseEvent)>::new(move |_e: CloseEvent| tx.close())
        };
        socket.set_onclose(Some(on_close.as_ref().unchecked_ref()));
        let on_error = {
            let tx = inbox.sender();
            Closure::<dyn FnMut(Event)>::new(move |_e: Event| tx.close())
        };
        socket.set_onerror(Some(on_error.as_ref().unchecked_ref()));

        self.socket = Some(socket);
        self.inbox = Some(inbox);
        self._on_message = Some(on_message);
        self._on_close = Some(on_close);
        self._on_error = Some(on_error);
        Ok(())
    }

    /// The next packet off the wire, `None` once the socket closes.
    async fn next_packet(&mut self) -> Result<Option<PipelinePacket>, G2gError> {
        let inbox = self.inbox.as_ref().ok_or(G2gError::NotConfigured)?;
        match inbox.next().await {
            Some(bytes) => Ok(Some(decode_packet(&bytes).map_err(map_wire)?)),
            None => Ok(None),
        }
    }

    /// Detach the callbacks before they drop, so the socket holds no reference
    /// into freed Rust state.
    fn close(&mut self) {
        if let Some(socket) = self.socket.take() {
            socket.set_onmessage(None);
            socket.set_onclose(None);
            socket.set_onerror(None);
            let _ = socket.close();
        }
        self._on_message = None;
        self._on_close = None;
        self._on_error = None;
    }
}

impl SourceLoop for WsWireSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    /// Discovers the media from the wire: the sender's leading `CapsChanged` is
    /// the first message, so negotiation waits for it (the async caps-discovery
    /// pattern `RemoteWsSrc` uses).
    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        Box::pin(async move {
            if let Some(caps) = self.discovered.clone() {
                return Ok(caps);
            }
            self.open()?;
            let caps = match self.next_packet().await? {
                Some(PipelinePacket::CapsChanged(caps)) => caps,
                // Anything else leading the stream breaks the protocol.
                _ => return Err(G2gError::Hardware(HardwareError::Other)),
            };
            self.discovered = Some(caps.clone());
            Ok(caps)
        })
    }

    async fn caps_constraint(&mut self) -> Result<CapsConstraint<'_>, G2gError> {
        let caps = self.intercept_caps().await?;
        Ok(CapsConstraint::Produces(CapsSet::one(caps)))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            // The caps discovered during negotiation lead the stream downstream,
            // the way the sender sent them.
            if let Some(caps) = self.discovered.clone() {
                out.push(PipelinePacket::CapsChanged(caps)).await?;
            }
            let mut frames = 0u64;
            // A failed push must not return early: the callbacks are still
            // installed on the socket, so they are detached below either way.
            let mut push_result = Ok(());
            loop {
                let packet = match self.next_packet().await {
                    Ok(Some(packet)) => packet,
                    // A clean close ends the stream, as the sender's Eos does.
                    Ok(None) => break,
                    Err(e) => {
                        push_result = Err(e);
                        break;
                    }
                };
                if matches!(packet, PipelinePacket::Eos) {
                    break;
                }
                let is_frame = matches!(packet, PipelinePacket::DataFrame(_));
                if let Err(e) = out.push(packet).await {
                    push_result = Err(e);
                    break;
                }
                if is_frame {
                    frames += 1;
                }
            }
            self.close();
            push_result?;
            out.push(PipelinePacket::Eos).await?;
            Ok(frames)
        })
    }
}
