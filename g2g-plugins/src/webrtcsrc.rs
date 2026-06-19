//! WebRTC data-channel ingest source (browser/wasm). Consumes binary messages
//! from a provided, already-open `RtcDataChannel` and emits each as a
//! system-memory `DataFrame` chunk: the second browser ingest path alongside
//! `WebSocketSrc` (M42). Signaling (offer/answer/ICE) is the application's job;
//! this element wraps the negotiated channel.
//!
//! Same callback-to-async bridge as `WebSocketSrc`: the channel's `onmessage`
//! handler feeds a [`crate::webutil::Inbox`] that the async `run` loop drains.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelinePacket,
};

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CloseEvent, MessageEvent, RtcDataChannel, RtcDataChannelType};

use crate::webutil::Inbox;

#[derive(Debug)]
pub struct WebRtcSrc {
    channel: RtcDataChannel,
    caps: Caps,
    configured: bool,
}

impl WebRtcSrc {
    /// `channel` is an already-open `RtcDataChannel` (the app performed the
    /// signaling handshake); `caps` is the stream's declared format, as with
    /// `FileSrc`/`WebSocketSrc`. The channel is wired in `run`.
    pub fn new(channel: RtcDataChannel, caps: Caps) -> Self {
        Self { channel, caps, configured: false }
    }
}

impl SourceLoop for WebRtcSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps.clone()))
    }

    /// Produces exactly the caller-declared caps, mirroring `WebSocketSrc`.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps.clone()))))
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
            self.channel.set_binary_type(RtcDataChannelType::Arraybuffer);

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
            self.channel
                .set_onmessage(Some(on_message.as_ref().unchecked_ref()));

            // onclose ends the stream.
            let on_close = {
                let tx = inbox.sender();
                Closure::<dyn FnMut(CloseEvent)>::new(move |_e: CloseEvent| tx.close())
            };
            self.channel
                .set_onclose(Some(on_close.as_ref().unchecked_ref()));

            let mut sequence = 0u64;
            while let Some(bytes) = inbox.next().await {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    // Raw ingest carries no timing; recovered downstream by the
                    // parser/decoder, as with FileSrc / WebSocketSrc.
                    timing: FrameTiming::default(),
                    sequence,
                    meta: Default::default(),
                };
                sequence += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }

            // Detach the callbacks before dropping them, then keep them alive to
            // exactly here (the channel held raw references for its lifetime).
            self.channel.set_onmessage(None);
            self.channel.set_onclose(None);
            drop(on_message);
            drop(on_close);
            self.channel.close();

            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }
}
