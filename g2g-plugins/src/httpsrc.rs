//! HTTP(S) byte-stream source (HttpSrc, `http-src` feature): the souphttpsrc
//! analog and the network sibling of [`FileSrc`](crate::filesrc). It issues one
//! GET and streams the response body downstream as `DataFrame` chunks under the
//! declared caps, then `Eos`. This is the fetch layer under HLS/DASH (each media
//! segment is one GET), and feeds the byte-stream demuxers
//! (`tsdemux` / `matroskademux` / ...) the same way `FileSrc` does.
//!
//! Caps are declared at construction (`HttpSrc::new(url, caps)`) or via the
//! `bytestream-format` property, because the container cannot be known from the
//! URL alone. Header-sniff (`auto`) and a `uridecodebin` `http(s)://` handler are
//! follow-ups: both need a negotiation-time ranged fetch to detect the container.
//!
//! Runs on the caller's tokio runtime (reqwest is async); chunks carry no PTS
//! (timing is recovered by the downstream parser/decoder), matching `FileSrc`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError,
    HardwareError, MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

#[derive(Debug)]
pub struct HttpSrc {
    url: String,
    caps: Caps,
    configured: bool,
}

impl HttpSrc {
    /// `caps` is the stream's declared format, e.g.
    /// `Caps::ByteStream { encoding: MpegTs }` for an HLS `.ts` segment. No
    /// request is issued until `run`.
    pub fn new(url: impl Into<String>, caps: Caps) -> Self {
        Self { url: url.into(), caps, configured: false }
    }
}

/// reqwest transport / status failures map to a hardware-ish I/O error; the run
/// fails loud and the pipeline surfaces it.
fn http_err(_e: reqwest::Error) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

impl SourceLoop for HttpSrc {
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
            let response = reqwest::Client::new()
                .get(&self.url)
                .send()
                .await
                .map_err(http_err)?
                .error_for_status()
                .map_err(http_err)?;

            let mut response = response;
            let mut sequence = 0u64;
            while let Some(bytes) = response.chunk().await.map_err(http_err)? {
                if bytes.is_empty() {
                    continue;
                }
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        bytes.to_vec().into_boxed_slice(),
                    )),
                    timing: FrameTiming {
                        arrival_ns: g2g_core::metrics::monotonic_ns(),
                        ..FrameTiming::default()
                    },
                    sequence,
                    meta: Default::default(),
                };
                sequence += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        HTTPSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.url = String::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            "bytestream-format" => {
                let encoding = encoding_from_str(value.as_str().ok_or(PropError::Type)?)
                    .ok_or(PropError::Value)?;
                self.caps = Caps::ByteStream { encoding };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.url.clone())),
            "bytestream-format" => match &self.caps {
                Caps::ByteStream { encoding } => Some(PropValue::Str(encoding_to_str(*encoding).into())),
                _ => None,
            },
            _ => None,
        }
    }
}

static HTTPSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("location", PropKind::Str, "source URL (http:// or https://)"),
    PropertySpec::new(
        "bytestream-format",
        PropKind::Str,
        "container of the fetched byte stream: mpegts | matroska | ogg | flv",
    ),
];

fn encoding_from_str(s: &str) -> Option<ByteStreamEncoding> {
    match s {
        "mpegts" | "ts" => Some(ByteStreamEncoding::MpegTs),
        "matroska" | "mkv" | "webm" => Some(ByteStreamEncoding::Matroska),
        "ogg" | "opus" => Some(ByteStreamEncoding::Ogg),
        "flv" => Some(ByteStreamEncoding::Flv),
        _ => None,
    }
}

fn encoding_to_str(encoding: ByteStreamEncoding) -> &'static str {
    match encoding {
        ByteStreamEncoding::MpegTs => "mpegts",
        ByteStreamEncoding::Matroska => "matroska",
        ByteStreamEncoding::Ogg => "ogg",
        ByteStreamEncoding::Flv => "flv",
    }
}
