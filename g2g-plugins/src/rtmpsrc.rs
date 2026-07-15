//! RTMP ingest source (RtmpSrc, `rtmp` feature): accepts one RTMP publisher
//! (ffmpeg / OBS pushing `rtmp://host/app/key`) over TCP and streams the demuxed
//! FLV byte stream downstream as `Caps::ByteStream{Flv}` for `flvdemux`, then
//! `Eos`. The [`rtmp`](crate::rtmp) sans-IO session does the protocol work
//! (handshake, chunk stream, AMF0 publish flow, audio/video -> FLV); this element
//! is the tokio TCP I/O around it.
//!
//! Scope: one publisher / one connection, the simple handshake, H.264 + AAC. The
//! socket is bound in `configure_pipeline` (or supplied pre-bound via
//! `from_listener` so a test can pick an ephemeral port) and promoted to a tokio
//! listener in `run`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use std::net::{SocketAddr, TcpListener as StdTcpListener};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

use crate::filesink::io_err;
use crate::rtmp::RtmpSession;

/// TCP read buffer; one RTMP chunk fragment is at most the negotiated chunk size,
/// well under this.
const READ_BUF: usize = 65_536;

#[derive(Debug)]
pub struct RtmpSrc {
    bind: SocketAddr,
    /// Bound in `configure_pipeline`, or supplied pre-bound; promoted to tokio in
    /// `run`.
    listener: Option<StdTcpListener>,
    configured: bool,
}

impl RtmpSrc {
    /// Listen for a publisher on `bind` (e.g. `0.0.0.0:1935`, the RTMP port).
    pub fn new(bind: SocketAddr) -> Self {
        Self { bind, listener: None, configured: false }
    }

    /// Use an already-bound listener instead of binding `bind`, so a caller (a
    /// test) can pick an ephemeral port and learn it up front.
    pub fn from_listener(listener: StdTcpListener) -> Result<Self, G2gError> {
        let bind = listener.local_addr().map_err(io_err)?;
        // A pre-bound listener is already all `configure_pipeline` would set up,
        // so mark it configured (matching `RtspServerSrc::from_listener`); a
        // redundant `configure_pipeline` call stays harmless (idempotent).
        Ok(Self { bind, listener: Some(listener), configured: true })
    }

    fn output_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::Flv }
    }
}

fn flv_frame(bytes: alloc::vec::Vec<u8>, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            arrival_ns: g2g_core::metrics::monotonic_ns(),
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

impl SourceLoop for RtmpSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Self::output_caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(Self::output_caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.listener.is_none() {
            self.listener = Some(StdTcpListener::bind(self.bind).map_err(io_err)?);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "RTMP ingest source",
            "Source/Network",
            "Accepts an RTMP publisher and emits an FLV byte stream",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("address", PropKind::Str, "local bind address (IP to listen on)")
                .with_default("0.0.0.0"),
            PropertySpec::new("port", PropKind::Uint, "local TCP port to accept the publisher on")
                .with_range("0", "65535")
                .with_default("1935"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        crate::netprop::set_addr_prop(&mut self.bind, "address", name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        crate::netprop::get_addr_prop(&self.bind, "address", name)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let std_listener = self.listener.take().ok_or(G2gError::NotConfigured)?;
            std_listener.set_nonblocking(true).map_err(io_err)?;
            let listener = tokio::net::TcpListener::from_std(std_listener).map_err(io_err)?;
            let (mut stream, _addr) = listener.accept().await.map_err(io_err)?;

            let mut session = RtmpSession::new();
            let mut sequence = 0u64;
            let mut buf = [0u8; READ_BUF];
            loop {
                let n = stream.read(&mut buf).await.map_err(io_err)?;
                if n == 0 {
                    break; // publisher closed the connection
                }
                session.push(&buf[..n]);
                let response = session.take_outbound();
                if !response.is_empty() {
                    stream.write_all(&response).await.map_err(io_err)?;
                }
                let flv = session.take_flv();
                if !flv.is_empty() {
                    out.push(PipelinePacket::DataFrame(flv_frame(flv, sequence))).await?;
                    sequence += 1;
                }
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }
}
