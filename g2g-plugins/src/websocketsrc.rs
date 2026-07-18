//! WebSocket ingest source (browser/wasm). Opens a `WebSocket`, receives
//! binary messages (`ArrayBuffer`), and emits each as a `DataFrame` chunk in
//! the system-memory domain: the browser analog of `FileSrc`/`RtspSrc` for
//! pull-from-network ingest (DESIGN.md §6.3).
//!
//! Like `FileSrc`, a raw byte stream carries no caps, so the caller declares
//! them at construction; the source produces exactly that to the solver. Feed
//! the output through `H264Parse` (then a `WebCodecsDecode`, M40) to recover
//! access units. The JS `onmessage` callback is bridged to the async `run`
//! loop through a [`crate::webutil::Inbox`].

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError, HardwareError,
    MemoryDomain, OutputSink, PipelinePacket,
};

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{BinaryType, CloseEvent, Event, MessageEvent, WebSocket};

use crate::webutil::Inbox;

#[derive(Debug)]
pub struct WebSocketSrc {
    url: String,
    caps: Caps,
    configured: bool,
}

impl WebSocketSrc {
    /// `caps` is the stream's declared format (e.g. `Caps::CompressedVideo {
    /// codec: H264, .. }` for an Annex-B elementary stream). The socket is
    /// opened in `run`, so construction has no side effects.
    pub fn new(url: impl Into<String>, caps: Caps) -> Self {
        Self {
            url: url.into(),
            caps,
            configured: false,
        }
    }
}

impl SourceLoop for WebSocketSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps.clone()))
    }

    /// Produces exactly the caller-declared caps (no I/O during negotiation;
    /// the socket opens in `run`), mirroring `FileSrc`.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(
            self.caps.clone(),
        ))))
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

            let ws =
                WebSocket::new(&self.url).map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            ws.set_binary_type(BinaryType::Arraybuffer);

            let inbox: Inbox<Vec<u8>> = Inbox::new();

            // onmessage: copy each ArrayBuffer payload into the inbox.
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
            ws.set_onmessage(Some(on_message.as_ref().unchecked_ref()));

            // onclose / onerror both end the stream. Errors map to EOS for
            // M39; structured error propagation is a follow-up.
            let on_close = {
                let tx = inbox.sender();
                Closure::<dyn FnMut(CloseEvent)>::new(move |_e: CloseEvent| tx.close())
            };
            ws.set_onclose(Some(on_close.as_ref().unchecked_ref()));

            let on_error = {
                let tx = inbox.sender();
                Closure::<dyn FnMut(Event)>::new(move |_e: Event| tx.close())
            };
            ws.set_onerror(Some(on_error.as_ref().unchecked_ref()));

            let mut sequence = 0u64;
            while let Some(bytes) = inbox.next().await {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    // Raw ingest carries no timing; recovered downstream by the
                    // parser/decoder, as with FileSrc.
                    timing: FrameTiming::default(),
                    sequence,
                    meta: Default::default(),
                };
                sequence += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }

            // Detach the callbacks before dropping them so JS holds no
            // reference into freed Rust state, then drop (keeps them alive to
            // exactly here, for the socket's whole lifetime).
            ws.set_onmessage(None);
            ws.set_onclose(None);
            ws.set_onerror(None);
            drop(on_message);
            drop(on_close);
            drop(on_error);
            let _ = ws.close();

            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }
}
