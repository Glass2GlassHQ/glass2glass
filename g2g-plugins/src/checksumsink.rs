//! Checksum sink: digests every buffer it receives and reports the digest with
//! the buffer's pts, the gst `checksumsink` analog. It is how a codec change is
//! checked for bit-exactness: run the pipeline before and after and diff the
//! lines.
//!
//! One line per buffer, `<pts> <digest>`, with the pts in gst's
//! `h:mm:ss.nnnnnnnnn` form so the output diffs against a gst run's. gst prints
//! it on stdout; here it goes on the bus as [`BusMessage::Info`] and on the debug
//! log, so an application collects the lines instead of scraping a terminal.
//!
//! The whole buffer is hashed, exactly the bytes that arrived: no plane or
//! padding handling, so a frame carried as a strided view is not what this sink
//! takes (it needs one contiguous buffer).

use core::fmt::Write as _;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};

use md5::Md5;
use sha1::{Digest, Sha1};
use sha2::{Sha256, Sha512};

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_debug, AsyncElement, BusHandle, BusMessage, Caps, CapsConstraint, ConfigureOutcome,
    ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

/// The `hash` choices, gst's `GstChecksumSinkHash` nicks.
const HASH_VALUES: &str = "md5 | sha1 | sha256 | sha512";
/// gst `checksumsink`'s `hash` default.
const DEFAULT_HASH: &str = "sha1";

const NS_PER_SECOND: u64 = 1_000_000_000;
const SECONDS_PER_MINUTE: u64 = 60;
const MINUTES_PER_HOUR: u64 = 60;

/// # Example
///
/// ```no_run
/// use g2g_plugins::checksumsink::ChecksumSink;
///
/// // gst-launch equivalent: checksumsink hash=md5
/// let sink = ChecksumSink::new();
/// assert_eq!(sink.digested(), 0);
/// ```
#[derive(Debug)]
pub struct ChecksumSink {
    hash: &'static str,
    bus: Option<BusHandle>,
    last_line: String,
    digested: u64,
    configured: bool,
    log_name: LogName,
}

impl Default for ChecksumSink {
    fn default() -> Self {
        Self::new()
    }
}

impl ChecksumSink {
    pub fn new() -> Self {
        Self {
            hash: DEFAULT_HASH,
            bus: None,
            last_line: String::new(),
            digested: 0,
            configured: false,
            log_name: LogName::new(),
        }
    }

    /// Attach the pipeline bus the digest lines are posted on. Without one they
    /// only reach the debug log.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Buffers digested so far.
    pub fn digested(&self) -> u64 {
        self.digested
    }

    /// The most recent `<pts> <digest>` line, empty before the first buffer.
    pub fn last_line(&self) -> &str {
        &self.last_line
    }
}

impl AsyncElement for ChecksumSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ChecksumSink",
            "Sink",
            "Reports a digest of every buffer, for bit-exactness checks",
            "g2g",
        )
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Bytes are bytes: any media type can be digested.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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
            if let PipelinePacket::DataFrame(frame) = packet {
                let bytes = frame
                    .domain
                    .as_system_slice()
                    .ok_or(G2gError::UnsupportedDomain)?;
                let line = checksum_line(frame.timing.pts_ns, digest_hex(self.hash, bytes));
                g2g_debug!(self, "{}", line);
                if let Some(bus) = &self.bus {
                    bus.try_post(BusMessage::Info(line.clone()));
                }
                self.last_line = line;
                self.digested += 1;
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        CHECKSUMSINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if name != "hash" {
            return Err(PropError::Unknown);
        }
        let text = value.as_str().ok_or(PropError::Type)?;
        self.hash = CHECKSUMSINK_PROPS
            .iter()
            .find(|spec| spec.name == name)
            .and_then(|spec| spec.enum_nicks().find(|nick| *nick == text))
            .ok_or(PropError::Value)?;
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "hash" => Some(PropValue::Str(self.hash.to_string())),
            _ => None,
        }
    }
}

impl LogSource for ChecksumSink {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

impl PadTemplates for ChecksumSink {
    /// Wildcard sink, matching the `AcceptsAny` constraint.
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        alloc::vec::Vec::from([PadTemplate::sink_any()])
    }
}

/// One reported line: a pts in gst's `h:mm:ss.nnnnnnnnn` form, then the digest.
pub fn checksum_line(pts_ns: u64, digest: String) -> String {
    let seconds = pts_ns / NS_PER_SECOND;
    let mut line = String::new();
    let _ = write!(
        line,
        "{}:{:02}:{:02}.{:09} {digest}",
        seconds / (SECONDS_PER_MINUTE * MINUTES_PER_HOUR),
        (seconds / SECONDS_PER_MINUTE) % MINUTES_PER_HOUR,
        seconds % SECONDS_PER_MINUTE,
        pts_ns % NS_PER_SECOND,
    );
    line
}

/// The lowercase hex digest of `bytes` under the named hash. An unknown name
/// cannot arrive: `set_property` only accepts the declared nicks.
pub fn digest_hex(hash: &str, bytes: &[u8]) -> String {
    match hash {
        "md5" => hex(&Md5::digest(bytes)),
        "sha256" => hex(&Sha256::digest(bytes)),
        "sha512" => hex(&Sha512::digest(bytes)),
        _ => hex(&Sha1::digest(bytes)),
    }
}

/// `ChecksumSink`'s settable properties, named and defaulted as gst
/// `checksumsink`.
static CHECKSUMSINK_PROPS: &[PropertySpec] =
    &[
        PropertySpec::new("hash", PropKind::Str, "digest computed over each buffer")
            .with_enum_values(HASH_VALUES)
            .with_default(DEFAULT_HASH),
    ];

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}
