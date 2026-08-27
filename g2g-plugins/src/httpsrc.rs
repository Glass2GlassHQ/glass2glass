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
//!
//! Prebuffering (`prebuffer-bytes`, the queue2-buffering analog: g2g has no
//! queue element, so the network source owns its own window): when set, `run`
//! fills a byte window before pushing downstream, posting
//! [`BusMessage::Buffering`] percent on the attached bus as it fills; after
//! that it streams through, keeping the window topped up without waiting, and
//! a mid-stream underrun (window empty with the network not ready) re-enters
//! buffering, so an application can pause until it sees `100` and show a
//! "buffering..." indicator on a stall. `0` (the default) streams each chunk
//! straight through.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    BusHandle, BusMessage, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, HardwareError, MemoryDomain, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

/// # Example
///
/// ```no_run
/// use g2g_core::{ByteStreamEncoding, Caps};
/// use g2g_plugins::httpsrc::HttpSrc;
///
/// let src = HttpSrc::new(
///     "https://example.com/segment.ts",
///     Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs },
/// )
/// .with_prebuffer_bytes(64 * 1024);
/// ```
#[derive(Debug)]
pub struct HttpSrc {
    url: String,
    caps: Caps,
    prebuffer_bytes: usize,
    bus: Option<BusHandle>,
    configured: bool,
}

impl HttpSrc {
    /// `caps` is the stream's declared format, e.g.
    /// `Caps::ByteStream { encoding: MpegTs }` for an HLS `.ts` segment. No
    /// request is issued until `run`.
    pub fn new(url: impl Into<String>, caps: Caps) -> Self {
        Self {
            url: url.into(),
            caps,
            prebuffer_bytes: 0,
            bus: None,
            configured: false,
        }
    }

    /// Buffer this many bytes before pushing downstream (and again after an
    /// underrun). `0` disables prebuffering.
    pub fn with_prebuffer_bytes(mut self, bytes: usize) -> Self {
        self.prebuffer_bytes = bytes;
        self
    }

    /// Attach the pipeline bus so prebuffering posts
    /// [`BusMessage::Buffering`] level reports.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Push one body chunk downstream as a `DataFrame`.
    async fn push_chunk(
        out: &mut dyn OutputSink,
        bytes: Vec<u8>,
        sequence: &mut u64,
    ) -> Result<(), G2gError> {
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming {
                arrival_ns: g2g_core::metrics::monotonic_ns(),
                ..FrameTiming::default()
            },
            sequence: *sequence,
            meta: Default::default(),
        };
        *sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
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
            let mut response = reqwest::Client::new()
                .get(&self.url)
                .send()
                .await
                .map_err(http_err)?
                .error_for_status()
                .map_err(http_err)?;

            let mut sequence = 0u64;
            if self.prebuffer_bytes == 0 {
                // Plain streaming: each chunk straight through.
                while let Some(bytes) = response.chunk().await.map_err(http_err)? {
                    if bytes.is_empty() {
                        continue;
                    }
                    Self::push_chunk(out, bytes.to_vec(), &mut sequence).await?;
                }
                out.push(PipelinePacket::Eos).await?;
                return Ok(sequence);
            }

            // Prebuffered mode. The window is bounded: it never grows past the
            // target (plus one in-flight chunk), so a fast network cannot
            // balloon memory; a slow consumer backpressures via `out.push`.
            let target = self.prebuffer_bytes;
            let mut window: VecDeque<Vec<u8>> = VecDeque::new();
            let mut buffered = 0usize;
            let mut ended = false;
            let mut last_bucket: Option<u8> = None;
            // Post on quartile-band transitions only (like the runner's sink
            // report), so a fill is a handful of messages, not one per chunk.
            let post = |bus: &Option<BusHandle>, pct: u8, last: &mut Option<u8>| {
                if let Some(b) = bus {
                    let bucket = (pct / 25).min(4);
                    if *last != Some(bucket) {
                        *last = Some(bucket);
                        b.try_post(BusMessage::Buffering {
                            percent: pct,
                            element: None,
                        });
                    }
                }
            };
            let percent = |buffered: usize| ((buffered * 100 / target) as u8).min(100);

            loop {
                // Buffering phase: fill the window to the target, reporting the
                // level as it rises. Entered at start and after an underrun.
                post(&self.bus, percent(buffered), &mut last_bucket);
                while buffered < target && !ended {
                    match response.chunk().await.map_err(http_err)? {
                        Some(b) => {
                            if !b.is_empty() {
                                buffered += b.len();
                                window.push_back(b.to_vec());
                                post(&self.bus, percent(buffered), &mut last_bucket);
                            }
                        }
                        None => ended = true,
                    }
                }
                // The stream ending early also completes buffering: there is
                // nothing left to wait for, so the application resumes.
                post(&self.bus, 100, &mut last_bucket);

                // Drain phase: push the window down while topping it up with
                // whatever the network has ready now (never waiting, never past
                // the target).
                while let Some(chunk) = window.pop_front() {
                    buffered -= chunk.len();
                    Self::push_chunk(out, chunk, &mut sequence).await?;
                    while !ended && buffered < target {
                        match tokio::time::timeout(Duration::ZERO, response.chunk()).await {
                            Ok(r) => match r.map_err(http_err)? {
                                Some(b) => {
                                    if !b.is_empty() {
                                        buffered += b.len();
                                        window.push_back(b.to_vec());
                                    }
                                }
                                None => ended = true,
                            },
                            Err(_) => break, // nothing immediately ready
                        }
                    }
                }
                if ended {
                    break;
                }
                // Window drained with the stream still live: underrun. Loop
                // back into the buffering phase to refill before resuming.
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        HTTPSRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "HTTP source",
            "Source/Network",
            "Fetches a URL as a byte stream via reqwest",
            "g2g",
        )
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
            "prebuffer-bytes" => {
                let bytes = value.as_uint().ok_or(PropError::Type)?;
                if bytes > 1 << 30 {
                    return Err(PropError::Value);
                }
                self.prebuffer_bytes = bytes as usize;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.url.clone())),
            "bytestream-format" => match &self.caps {
                Caps::ByteStream { encoding } => {
                    Some(PropValue::Str(encoding_to_str(*encoding).into()))
                }
                _ => None,
            },
            "prebuffer-bytes" => Some(PropValue::Uint(self.prebuffer_bytes as u64)),
            _ => None,
        }
    }
}

static HTTPSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "source URL (http:// or https://)",
    ),
    PropertySpec::new(
        "bytestream-format",
        PropKind::Str,
        "container of the fetched byte stream: mpegts | matroska | ogg | flv | multipart",
    ),
    PropertySpec::new(
        "prebuffer-bytes",
        PropKind::Uint,
        "bytes to buffer before pushing downstream, reported as Buffering bus messages (0 = stream straight through)",
    )
    .with_default("0")
    .with_range("0", "1073741824"),
];

fn encoding_from_str(s: &str) -> Option<ByteStreamEncoding> {
    match s {
        "mpegts" | "ts" => Some(ByteStreamEncoding::MpegTs),
        "matroska" | "mkv" | "webm" => Some(ByteStreamEncoding::Matroska),
        "ogg" | "opus" => Some(ByteStreamEncoding::Ogg),
        "flv" => Some(ByteStreamEncoding::Flv),
        "mp4" | "isobmff" | "cmaf" | "fmp4" => Some(ByteStreamEncoding::IsoBmff),
        "multipart" | "mpjpeg" => Some(ByteStreamEncoding::Multipart),
        _ => None,
    }
}

fn encoding_to_str(encoding: ByteStreamEncoding) -> &'static str {
    match encoding {
        ByteStreamEncoding::MpegTs => "mpegts",
        ByteStreamEncoding::Matroska => "matroska",
        ByteStreamEncoding::Ogg => "ogg",
        ByteStreamEncoding::Flv => "flv",
        ByteStreamEncoding::IsoBmff => "mp4",
        ByteStreamEncoding::Multipart => "multipart",
        ByteStreamEncoding::Mp4 => "mp4",
        _ => unreachable!("httpsrc encoding is set only via encoding_from_str"),
    }
}
