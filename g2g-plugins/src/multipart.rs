//! MIME multipart (`multipart/x-mixed-replace`): the transport an IP camera or
//! an `mjpg-streamer`-style server pushes MJPEG over HTTP with.
//! [`MultipartDemux`] splits the stream into its parts, [`MultipartMux`] writes
//! one, the `multipartdemux` / `multipartmux` analogs.
//!
//! ```text
//! httpsrc location=http://camera/stream bytestream-format=multipart ! multipartdemux ! mjpegdec ! fakesink
//! videotestsrc ! mjpegenc ! multipartmux ! filesink location=out.mjpg
//! ```
//!
//! The stream is a `--boundary` line, a header block ending in a blank line, and
//! a body: `Content-Length` bytes when the sender gave one, otherwise everything
//! up to the next boundary line. `--boundary--` closes the stream.
//!
//! Only `image/jpeg` parts are carried. A part with another `Content-Type`, or
//! with none at all, fails the parse rather than being typed by guesswork: there
//! is one output pad here, and mistyping a body is worse than refusing it.
//!
//! Everything the demuxer sizes anything from comes off the wire, so the
//! boundary is capped at `MAX_BOUNDARY_LEN`, the header block at
//! `MAX_HEADER_BYTES` and a body at `MAX_PART_BYTES`, each checked before a
//! byte is buffered against it.
//!
//! The transport carries no timestamps, so a part gets no pts; it inherits the
//! `arrival_ns` of the input chunk that completed it, which keeps the
//! glass-to-glass measurement a network source started. GStreamer's
//! `single-stream` property has no counterpart: it only decides when
//! `no-more-pads` fires, and this element has one static output pad.
//!
//! A preamble before the first boundary is not skipped: the stream must open on
//! its boundary line, which is what every MJPEG server sends.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_error, AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, VideoCodec,
};

/// What a delimiter line opens with, and what a closing one repeats at its end.
const DELIMITER_PREFIX: &[u8] = b"--";

/// Line ending every multipart header uses.
const CRLF: &[u8] = b"\r\n";

/// Longest boundary accepted, the RFC 2046 ceiling.
const MAX_BOUNDARY_LEN: usize = 70;

/// Longest delimiter line read: the leading `--`, a boundary at its ceiling, the
/// closing `--` and a CR.
const MAX_DELIMITER_LINE: usize = DELIMITER_PREFIX.len() * 2 + MAX_BOUNDARY_LEN + 1;

/// Byte ceiling on one part's header block, blank line included. A real part
/// carries two or three short headers, so a stream that has not closed its block
/// inside this is not one.
const MAX_HEADER_BYTES: usize = 4096;

/// Byte ceiling on one part's body. Far above any JPEG frame (a 4K still is a
/// couple of MB), so a bogus `Content-Length` or a boundary that never arrives
/// fails instead of growing the buffer.
const MAX_PART_BYTES: usize = 32 * 1024 * 1024;

/// Header naming a part's media type, matched case-insensitively (ffmpeg writes
/// `Content-type`, GStreamer `Content-Type`).
const CONTENT_TYPE_HEADER: &str = "content-type";

/// Header naming a part's body length in bytes.
const CONTENT_LENGTH_HEADER: &str = "content-length";

/// The media type each part carries, and the only one accepted on the way in.
const JPEG_MEDIA_TYPE: &str = "image/jpeg";

/// The `Content-Type` spellings read as JPEG. `image/jpg` is not registered but
/// is what several cameras send.
const JPEG_MEDIA_TYPES: [&str; 2] = [JPEG_MEDIA_TYPE, "image/jpg"];

/// The boundary [`MultipartMux`] writes when nothing sets one, GStreamer's own
/// default so a `multipartmux` line ported over produces the same bytes.
const DEFAULT_BOUNDARY: &str = "ThisRandomString";

/// The byte stream both elements sit against.
fn multipart_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Multipart,
    }
}

/// The JPEG access units the parts carry, at the fixable `Range` placeholder
/// geometry a coded video stream gets before anything has parsed a frame (never
/// `Any`, which cannot fixate); `mjpegdec` refines it per frame.
fn jpeg_caps() -> Caps {
    crate::typefind::elementary_video_caps(VideoCodec::Mjpeg)
}

/// True when a byte stream opens like a multipart one: a `--boundary` line, then
/// a part header. The leading `--` alone matches far too much, so the header line
/// behind it has to be there too.
pub(crate) fn looks_like_multipart(header: &[u8]) -> bool {
    let Ok(Some(end)) = line_end(header, MAX_DELIMITER_LINE) else {
        return false;
    };
    if learn_boundary(trim_cr(&header[..end])).is_none() {
        return false;
    }
    let rest = &header[end + 1..];
    let Ok(Some(field_end)) = line_end(rest, MAX_HEADER_BYTES) else {
        return false;
    };
    let field = trim_cr(&rest[..field_end]);
    matches!(header_name(field), Some(name)
        if name == CONTENT_TYPE_HEADER || name == CONTENT_LENGTH_HEADER)
}

/// Offset of the newline ending the line at the front of `buf`, `None` while more
/// bytes are needed. `Err` once `max` bytes have gone by without one.
fn line_end(buf: &[u8], max: usize) -> Result<Option<usize>, G2gError> {
    match buf.iter().take(max).position(|&b| b == b'\n') {
        Some(at) => Ok(Some(at)),
        None if buf.len() >= max => Err(G2gError::CapsMismatch),
        None => Ok(None),
    }
}

/// A line without the CR of its CRLF.
fn trim_cr(line: &[u8]) -> &[u8] {
    match line.split_last() {
        Some((b'\r', rest)) => rest,
        _ => line,
    }
}

/// RFC 2046 allows transport padding after a delimiter, so trailing blanks are
/// not part of the boundary.
fn trim_trailing_blanks(line: &[u8]) -> &[u8] {
    let end = line
        .iter()
        .rposition(|b| *b != b' ' && *b != b'\t')
        .map_or(0, |at| at + 1);
    &line[..end]
}

/// Whether a boundary can be written and read back: non-empty, inside the RFC
/// 2046 ceiling, and printable ASCII with no blanks. Every printable byte is
/// accepted where RFC 2046 names a narrower set, since senders do not stick to
/// it, but a binary or over-long line still cannot be read as a boundary.
fn usable_boundary(boundary: &[u8]) -> bool {
    !boundary.is_empty()
        && boundary.len() <= MAX_BOUNDARY_LEN
        && boundary.iter().all(|b| b.is_ascii_graphic())
}

/// The boundary a stream's first delimiter line declares, `None` when the line is
/// not a usable one.
fn learn_boundary(line: &[u8]) -> Option<&[u8]> {
    let rest = line.strip_prefix(DELIMITER_PREFIX)?;
    let boundary = trim_trailing_blanks(rest);
    usable_boundary(boundary).then_some(boundary)
}

/// The lowercased name of a `Name: value` header line, `None` when the line has
/// no colon.
fn header_name(line: &[u8]) -> Option<String> {
    let at = line.iter().position(|b| *b == b':')?;
    Some(
        core::str::from_utf8(&line[..at])
            .ok()?
            .trim()
            .to_ascii_lowercase(),
    )
}

/// The media type a `Content-Type` value names, its parameters (`;charset=...`)
/// dropped and lowercased.
fn media_type(value: &str) -> String {
    value
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

/// A header line this element acts on, `None` for the blank line that ends the
/// block and for headers it ignores.
#[derive(Debug, PartialEq)]
enum Field {
    ContentType(String),
    ContentLength(usize),
    /// A header the demuxer does not read.
    Ignored,
}

/// Read one header line. The length comes off the wire, so it is refused here,
/// before the buffer is asked to hold it, when it is unreadable or past
/// [`MAX_PART_BYTES`].
fn parse_header(line: &[u8]) -> Result<Option<Field>, G2gError> {
    if line.is_empty() {
        return Ok(None);
    }
    let Some(name) = header_name(line) else {
        return Ok(Some(Field::Ignored));
    };
    let value = core::str::from_utf8(line)
        .map_err(|_| G2gError::CapsMismatch)?
        .split_once(':')
        .map_or("", |(_, value)| value)
        .trim();
    if name == CONTENT_TYPE_HEADER {
        return Ok(Some(Field::ContentType(media_type(value))));
    }
    if name == CONTENT_LENGTH_HEADER {
        let length: usize = value.parse().map_err(|_| G2gError::CapsMismatch)?;
        if length > MAX_PART_BYTES {
            return Err(G2gError::CapsMismatch);
        }
        return Ok(Some(Field::ContentLength(length)));
    }
    Ok(Some(Field::Ignored))
}

/// Offset of `needle` in `haystack`.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Where the demuxer is in the stream.
#[derive(Debug, Default, PartialEq)]
enum State {
    /// Waiting for the `--boundary` line that opens a part.
    #[default]
    Delimiter,
    /// Reading a part's header block.
    Headers,
    /// Reading a part's body, `Some(n)` bytes still wanted when `Content-Length`
    /// gave a length.
    Body(Option<usize>),
    /// The closing `--boundary--` was read; the rest of the stream is epilogue.
    Finished,
}

/// The headers of the part being read.
#[derive(Debug, Default)]
struct PartHeaders {
    content_type: Option<String>,
    content_length: Option<usize>,
}

/// Splits a `multipart/x-mixed-replace` stream into its JPEG parts.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::multipart::MultipartDemux;
///
/// let element = MultipartDemux::new().with_boundary("ffmpeg");
/// ```
#[derive(Debug, Default)]
pub struct MultipartDemux {
    configured: bool,
    /// The `boundary` property: empty means the first delimiter line declares it.
    declared_boundary: String,
    /// The boundary in force, learned or declared.
    boundary: Option<Vec<u8>>,
    /// `\n--boundary`, the delimiter a body with no `Content-Length` ends at.
    body_terminator: Vec<u8>,
    /// Bytes accumulated across input chunks: a part that straddles two of them
    /// stays here until the rest arrives.
    buf: Vec<u8>,
    state: State,
    headers: PartHeaders,
    /// Header-block bytes read so far, against [`MAX_HEADER_BYTES`].
    header_bytes: usize,
    /// Arrival stamp of the input chunk being parsed, copied onto the parts it
    /// completes.
    arrival_ns: u64,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl MultipartDemux {
    pub fn new() -> Self {
        Self::default()
    }

    /// Expect this boundary instead of learning it from the stream's first line.
    pub fn with_boundary(mut self, boundary: impl Into<String>) -> Self {
        self.declared_boundary = boundary.into();
        self
    }

    /// Fix the boundary and the terminator derived from it.
    fn set_boundary(&mut self, boundary: &[u8]) {
        let mut terminator = Vec::with_capacity(1 + DELIMITER_PREFIX.len() + boundary.len());
        terminator.push(b'\n');
        terminator.extend_from_slice(DELIMITER_PREFIX);
        terminator.extend_from_slice(boundary);
        self.body_terminator = terminator;
        self.boundary = Some(Vec::from(boundary));
    }

    /// Read a delimiter line, returning whether it closes the stream. The first
    /// one declares the boundary unless the property already did.
    fn match_delimiter(&mut self, line: &[u8]) -> Result<bool, G2gError> {
        if self.boundary.is_none() {
            let declared = Vec::from(self.declared_boundary.as_bytes());
            match declared.is_empty() {
                true => {
                    let learned = Vec::from(learn_boundary(line).ok_or(G2gError::CapsMismatch)?);
                    self.set_boundary(&learned);
                }
                false => self.set_boundary(&declared),
            }
        }
        let boundary = self.boundary.as_deref().ok_or(G2gError::CapsMismatch)?;
        let rest = line
            .strip_prefix(DELIMITER_PREFIX)
            .ok_or(G2gError::CapsMismatch)?;
        let rest = trim_trailing_blanks(rest);
        if rest == boundary {
            return Ok(false);
        }
        let closing = rest
            .strip_suffix(DELIMITER_PREFIX)
            .is_some_and(|body| body == boundary);
        if !closing {
            return Err(G2gError::CapsMismatch);
        }
        Ok(true)
    }

    /// Push one part downstream, announcing its caps when they differ from the
    /// last part's.
    async fn emit(&mut self, body: Vec<u8>, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let Some(content_type) = self.headers.content_type.clone() else {
            g2g_error!(self, "a part with no Content-Type header");
            return Err(G2gError::CapsMismatch);
        };
        if !JPEG_MEDIA_TYPES.contains(&content_type.as_str()) {
            g2g_error!(self, "a part of an unsupported type: {}", content_type);
            return Err(G2gError::CapsMismatch);
        }
        let caps = jpeg_caps();
        if self.last_caps.as_ref() != Some(&caps) {
            out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
            self.last_caps = Some(caps);
        }
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(body.into_boxed_slice())),
            FrameTiming {
                arrival_ns: self.arrival_ns,
                // every JPEG access unit decodes on its own.
                keyframe: true,
                ..Default::default()
            },
            self.emitted,
        );
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    /// Whether the stream stopped inside a part. ffmpeg's mpjpeg muxer ends on a
    /// bare boundary rather than a closing one, so a stream that stopped between
    /// parts is complete.
    fn part_in_progress(&self) -> bool {
        match self.state {
            State::Finished => false,
            State::Delimiter => !self.buf.is_empty(),
            State::Headers => !self.buf.is_empty() || self.header_bytes > 0,
            State::Body(_) => true,
        }
    }

    /// Read every complete part the buffer holds.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        loop {
            match self.state {
                State::Finished => {
                    self.buf.clear();
                    return Ok(());
                }
                State::Delimiter => {
                    let Some(end) = line_end(&self.buf, MAX_DELIMITER_LINE)? else {
                        return Ok(());
                    };
                    let line = Vec::from(trim_cr(&self.buf[..end]));
                    // the CRLF a length-counted body ended on.
                    if line.is_empty() {
                        self.buf.drain(..end + 1);
                        continue;
                    }
                    let closing = self.match_delimiter(&line).inspect_err(|_| {
                        g2g_error!(self, "a line where a boundary was expected");
                    })?;
                    self.buf.drain(..end + 1);
                    self.headers = PartHeaders::default();
                    self.header_bytes = 0;
                    self.state = match closing {
                        true => State::Finished,
                        false => State::Headers,
                    };
                }
                State::Headers => {
                    let budget = MAX_HEADER_BYTES.saturating_sub(self.header_bytes);
                    let Some(end) = line_end(&self.buf, budget).inspect_err(|_| {
                        g2g_error!(self, "a header block past {} bytes", MAX_HEADER_BYTES);
                    })?
                    else {
                        return Ok(());
                    };
                    let field = parse_header(trim_cr(&self.buf[..end])).inspect_err(|_| {
                        g2g_error!(self, "an unreadable part header");
                    })?;
                    self.buf.drain(..end + 1);
                    self.header_bytes = self.header_bytes.saturating_add(end + 1);
                    match field {
                        None => self.state = State::Body(self.headers.content_length),
                        Some(Field::ContentType(value)) => self.headers.content_type = Some(value),
                        Some(Field::ContentLength(value)) => {
                            self.headers.content_length = Some(value)
                        }
                        Some(Field::Ignored) => {}
                    }
                }
                State::Body(Some(wanted)) => {
                    if self.buf.len() < wanted {
                        return Ok(());
                    }
                    let body: Vec<u8> = self.buf.drain(..wanted).collect();
                    self.state = State::Delimiter;
                    self.emit(body, out).await?;
                }
                State::Body(None) => {
                    let Some(at) = find(&self.buf, &self.body_terminator) else {
                        if self.buf.len() > MAX_PART_BYTES {
                            g2g_error!(self, "a part body past {} bytes", MAX_PART_BYTES);
                            return Err(G2gError::CapsMismatch);
                        }
                        return Ok(());
                    };
                    let body = Vec::from(trim_cr(&self.buf[..at]));
                    self.buf.drain(..at + 1);
                    self.state = State::Delimiter;
                    self.emit(body, out).await?;
                }
            }
        }
    }
}

impl AsyncElement for MultipartDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn is_format_boundary(&self) -> bool {
        true
    }

    /// Reads host memory, so it takes system frames only. The allocation cascade
    /// turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&multipart_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Multipart,
            } => CapsSet::one(jpeg_caps()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Multipart
            }
        ) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Multipart demuxer",
            "Codec/Demuxer",
            "Splits a multipart/x-mixed-replace stream into its JPEG parts",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DEMUX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "boundary" => {
                let boundary = value.as_str().ok_or(PropError::Type)?;
                // empty means the stream declares it, anything else has to be a
                // boundary a sender could have written.
                if !boundary.is_empty() && !usable_boundary(boundary.as_bytes()) {
                    return Err(PropError::Value);
                }
                self.declared_boundary = String::from(boundary);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "boundary" => Some(PropValue::Str(self.declared_boundary.clone())),
            _ => None,
        }
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    self.buf.extend_from_slice(slice);
                    self.arrival_ns = frame.timing.arrival_ns;
                    self.drain(out).await?;
                }
                PipelinePacket::Eos => {
                    self.drain(out).await?;
                    if self.part_in_progress() {
                        g2g_error!(
                            self,
                            "the stream ended mid-part with {} bytes buffered",
                            self.buf.len()
                        );
                        return Err(G2gError::CapsMismatch);
                    }
                }
                // The byte stream carries no geometry; the parts' caps come from
                // their own Content-Type instead.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

static DEMUX_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "boundary",
    PropKind::Str,
    "boundary separating the parts (empty = read it from the stream)",
)
.with_default("")];

impl LogSource for MultipartDemux {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
}

impl PadTemplates for MultipartDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(multipart_caps())),
            PadTemplate::source(CapsSet::one(jpeg_caps())),
        ])
    }
}

/// Wraps JPEG frames in a `multipart/x-mixed-replace` stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::multipart::MultipartMux;
///
/// let element = MultipartMux::new().with_boundary("ffmpeg");
/// ```
#[derive(Debug)]
pub struct MultipartMux {
    boundary: String,
    caps_announced: bool,
    configured: bool,
    emitted: u64,
}

impl Default for MultipartMux {
    fn default() -> Self {
        Self::new()
    }
}

impl MultipartMux {
    pub fn new() -> Self {
        Self {
            boundary: String::from(DEFAULT_BOUNDARY),
            caps_announced: false,
            configured: false,
            emitted: 0,
        }
    }

    /// Separate the parts with this boundary instead of `DEFAULT_BOUNDARY`.
    pub fn with_boundary(mut self, boundary: impl Into<String>) -> Self {
        self.boundary = boundary.into();
        self
    }

    /// One part: its delimiter, headers, the JPEG, and the CRLF closing the body.
    fn part(&self, jpeg: &[u8]) -> Vec<u8> {
        let header = format!(
            "--{}\r\nContent-Type: {JPEG_MEDIA_TYPE}\r\nContent-Length: {}\r\n\r\n",
            self.boundary,
            jpeg.len()
        );
        let mut bytes = Vec::with_capacity(header.len() + jpeg.len() + CRLF.len());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(jpeg);
        bytes.extend_from_slice(CRLF);
        bytes
    }

    /// The delimiter that closes the stream.
    fn terminator(&self) -> Vec<u8> {
        format!("--{}--\r\n", self.boundary).into_bytes()
    }

    fn frame(&mut self, bytes: Vec<u8>) -> PipelinePacket {
        let packet = PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming::default(),
            self.emitted,
        ));
        self.emitted += 1;
        packet
    }
}

impl AsyncElement for MultipartMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn is_format_boundary(&self) -> bool {
        true
    }

    /// Reads host memory, so it takes system frames only. The allocation cascade
    /// turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    /// JPEG only: every part is written with an `image/jpeg` Content-Type, so
    /// another codec is refused here rather than mislabelled on the wire.
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&jpeg_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                ..
            } => CapsSet::one(multipart_caps()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                ..
            }
        ) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Multipart muxer",
            "Codec/Muxer",
            "Wraps JPEG frames in a multipart/x-mixed-replace stream",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MUX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "boundary" => {
                let boundary = value.as_str().ok_or(PropError::Type)?;
                if !usable_boundary(boundary.as_bytes()) {
                    return Err(PropError::Value);
                }
                self.boundary = String::from(boundary);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "boundary" => Some(PropValue::Str(self.boundary.clone())),
            _ => None,
        }
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if !self.caps_announced {
                        out.push(PipelinePacket::CapsChanged(multipart_caps()))
                            .await?;
                        self.caps_announced = true;
                    }
                    let jpeg = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    let bytes = self.part(jpeg);
                    let packet = self.frame(bytes);
                    out.push(packet).await?;
                }
                PipelinePacket::Eos => {
                    if self.caps_announced {
                        let bytes = self.terminator();
                        let packet = self.frame(bytes);
                        out.push(packet).await?;
                    }
                }
                // The input caps describe the JPEG stream; this element's output
                // is the multipart stream it already announced.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

static MUX_PROPS: &[PropertySpec] =
    &[
        PropertySpec::new("boundary", PropKind::Str, "boundary separating the parts")
            .with_default(DEFAULT_BOUNDARY),
    ];

impl LogSource for MultipartMux {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
}

impl PadTemplates for MultipartMux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(jpeg_caps())),
            PadTemplate::source(CapsSet::one(multipart_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::runtime::block_on;
    use g2g_core::PushOutcome;

    /// The two-byte JPEG start-of-image marker, enough of a body for the parser.
    const SOI: &[u8] = &[0xFF, 0xD8];

    #[derive(Default)]
    struct Captured {
        caps: Vec<Caps>,
        frames: Vec<Frame>,
    }

    impl OutputSink for Captured {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            match packet_slot.take().expect("poll_push without a packet") {
                PipelinePacket::CapsChanged(caps) => self.caps.push(caps),
                PipelinePacket::DataFrame(frame) => self.frames.push(frame),
                _ => {}
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    /// Run the demuxer over `stream` and end it, returning what came out.
    fn demux(stream: &[u8]) -> Result<Captured, G2gError> {
        let mut element = MultipartDemux::new();
        element
            .configure_pipeline(&multipart_caps())
            .expect("a multipart byte stream");
        let mut sink = Captured::default();
        block_on(async {
            element
                .process(
                    PipelinePacket::DataFrame(Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(
                            stream.to_vec().into_boxed_slice(),
                        )),
                        FrameTiming::default(),
                        0,
                    )),
                    &mut sink,
                )
                .await?;
            element.process(PipelinePacket::Eos, &mut sink).await
        })?;
        Ok(sink)
    }

    fn demux_error(stream: &[u8]) -> Option<G2gError> {
        demux(stream).err()
    }

    /// A one-part stream, with the headers given verbatim.
    fn stream(headers: &str, body: &[u8], trailer: &str) -> Vec<u8> {
        let mut bytes = Vec::from(format!("--b\r\n{headers}\r\n\r\n").as_bytes());
        bytes.extend_from_slice(body);
        bytes.extend_from_slice(trailer.as_bytes());
        bytes
    }

    #[test]
    fn a_length_counted_part_is_cut_at_its_content_length() {
        let sink = stream(
            "Content-Type: image/jpeg\r\nContent-Length: 2",
            SOI,
            "\r\n--b--\r\n",
        );
        let sink = demux(&sink).expect("a well-formed part");
        assert_eq!(sink.caps, Vec::from([jpeg_caps()]));
        assert_eq!(sink.frames.len(), 1);
        assert_eq!(
            sink.frames[0]
                .domain
                .as_system_slice()
                .expect("system bytes"),
            SOI
        );
    }

    #[test]
    fn a_part_without_a_length_runs_to_the_next_boundary() {
        let sink = demux(&stream("Content-type: image/jpeg", SOI, "\r\n--b--\r\n"))
            .expect("a well-formed part");
        assert_eq!(sink.frames.len(), 1);
        assert_eq!(
            sink.frames[0]
                .domain
                .as_system_slice()
                .expect("system bytes"),
            SOI,
            "the CRLF before the boundary is not part of the body"
        );
    }

    #[test]
    fn the_boundary_is_learned_from_the_first_line_or_declared() {
        let bytes = stream("Content-Type: image/jpeg", SOI, "\r\n--b--\r\n");
        let mut declared = MultipartDemux::new().with_boundary("b");
        declared
            .configure_pipeline(&multipart_caps())
            .expect("a multipart byte stream");
        let mut sink = Captured::default();
        block_on(async {
            declared
                .process(
                    PipelinePacket::DataFrame(Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                        FrameTiming::default(),
                        0,
                    )),
                    &mut sink,
                )
                .await
        })
        .expect("the declared boundary matches the stream");
        assert_eq!(sink.frames.len(), 1);
    }

    #[test]
    fn a_declared_boundary_the_stream_does_not_use_is_refused() {
        let mut element = MultipartDemux::new().with_boundary("other");
        element
            .configure_pipeline(&multipart_caps())
            .expect("a multipart byte stream");
        let mut sink = Captured::default();
        let bytes = stream("Content-Type: image/jpeg", SOI, "\r\n--b--\r\n");
        let result = block_on(async {
            element
                .process(
                    PipelinePacket::DataFrame(Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                        FrameTiming::default(),
                        0,
                    )),
                    &mut sink,
                )
                .await
        });
        assert_eq!(result, Err(G2gError::CapsMismatch));
    }

    #[test]
    fn a_boundary_that_never_closes_a_part_ends_the_stream_with_an_error() {
        // headers begun, no blank line, no body, no closing boundary.
        assert_eq!(
            demux_error(b"--b\r\nContent-Type: image/jpeg\r\n"),
            Some(G2gError::CapsMismatch)
        );
        // a body with no length and no boundary behind it.
        assert_eq!(
            demux_error(&stream("Content-Type: image/jpeg", SOI, "")),
            Some(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn a_content_length_longer_than_the_body_ends_the_stream_with_an_error() {
        assert_eq!(
            demux_error(&stream(
                "Content-Type: image/jpeg\r\nContent-Length: 4096",
                SOI,
                "\r\n--b--\r\n",
            )),
            Some(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn a_content_length_past_the_ceiling_is_refused_before_anything_buffers_it() {
        for length in [
            MAX_PART_BYTES + 1,
            100 * 1024 * 1024,
            // wider than a usize can hold, so the parse itself fails.
            usize::MAX,
        ] {
            let headers = format!("Content-Type: image/jpeg\r\nContent-Length: {length}");
            assert_eq!(
                demux_error(&stream(&headers, SOI, "\r\n--b--\r\n")),
                Some(G2gError::CapsMismatch),
                "{length}"
            );
        }
        let headers = format!(
            "Content-Type: image/jpeg\r\nContent-Length: {}0",
            usize::MAX
        );
        assert_eq!(
            demux_error(&stream(&headers, SOI, "\r\n--b--\r\n")),
            Some(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn a_header_block_past_the_ceiling_is_refused() {
        let mut headers = String::from("Content-Type: image/jpeg\r\n");
        while headers.len() <= MAX_HEADER_BYTES {
            headers.push_str("X-Padding: 0123456789\r\n");
        }
        headers.push_str("Content-Length: 2");
        assert_eq!(
            demux_error(&stream(&headers, SOI, "\r\n--b--\r\n")),
            Some(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn a_part_of_another_media_type_or_none_at_all_is_refused() {
        for headers in [
            "Content-Type: text/plain\r\nContent-Length: 2",
            "Content-Length: 2",
        ] {
            assert_eq!(
                demux_error(&stream(headers, SOI, "\r\n--b--\r\n")),
                Some(G2gError::CapsMismatch),
                "{headers}"
            );
        }
    }

    #[test]
    fn a_boundary_line_that_is_not_one_is_refused() {
        for bytes in [
            b"GET /stream HTTP/1.0\r\n\r\n".as_slice(),
            // a boundary of nothing, and one past the RFC ceiling.
            b"--\r\nContent-Type: image/jpeg\r\n\r\n".as_slice(),
            b"\xff\xd8\xff\xe0\r\n".as_slice(),
        ] {
            assert_eq!(
                demux_error(bytes),
                Some(G2gError::CapsMismatch),
                "{bytes:?}"
            );
        }
        let long = format!("--{}\r\n", "b".repeat(MAX_BOUNDARY_LEN + 1));
        assert_eq!(demux_error(long.as_bytes()), Some(G2gError::CapsMismatch));
    }

    #[test]
    fn parts_split_across_input_chunks_are_reassembled() {
        let whole = {
            let mut bytes = stream(
                "Content-Type: image/jpeg\r\nContent-Length: 2",
                SOI,
                "\r\n--b\r\n",
            );
            bytes.extend_from_slice(b"Content-Type: image/jpeg\r\n\r\n");
            bytes.extend_from_slice(SOI);
            bytes.extend_from_slice(b"\r\n--b--\r\n");
            bytes
        };
        for chunk_len in [1, 3, 7, whole.len()] {
            let mut element = MultipartDemux::new();
            element
                .configure_pipeline(&multipart_caps())
                .expect("a multipart byte stream");
            let mut sink = Captured::default();
            block_on(async {
                for piece in whole.chunks(chunk_len) {
                    element
                        .process(
                            PipelinePacket::DataFrame(Frame::new(
                                MemoryDomain::System(SystemSlice::from_boxed(
                                    piece.to_vec().into_boxed_slice(),
                                )),
                                FrameTiming::default(),
                                0,
                            )),
                            &mut sink,
                        )
                        .await?;
                }
                element.process(PipelinePacket::Eos, &mut sink).await
            })
            .expect("the stream parses whatever the chunking");
            assert_eq!(sink.frames.len(), 2, "chunks of {chunk_len}");
            assert_eq!(
                sink.caps.len(),
                1,
                "chunks of {chunk_len}: the caps are announced once"
            );
        }
    }

    #[test]
    fn the_muxer_writes_parts_the_demuxer_reads_back() {
        let mut element = MultipartMux::new().with_boundary("b");
        element
            .configure_pipeline(&jpeg_caps())
            .expect("an mjpeg stream");
        let mut sink = Captured::default();
        block_on(async {
            for _ in 0..2 {
                element
                    .process(
                        PipelinePacket::DataFrame(Frame::new(
                            MemoryDomain::System(SystemSlice::from_boxed(
                                SOI.to_vec().into_boxed_slice(),
                            )),
                            FrameTiming::default(),
                            0,
                        )),
                        &mut sink,
                    )
                    .await?;
            }
            element.process(PipelinePacket::Eos, &mut sink).await
        })
        .expect("the muxer runs");
        assert_eq!(sink.caps, Vec::from([multipart_caps()]));
        let written: Vec<u8> = sink
            .frames
            .iter()
            .flat_map(|frame| {
                frame
                    .domain
                    .as_system_slice()
                    .expect("system bytes")
                    .to_vec()
            })
            .collect();
        assert!(
            written.ends_with(b"--b--\r\n"),
            "the closing boundary is written on Eos"
        );
        let read_back = demux(&written).expect("the muxer's own stream parses");
        assert_eq!(read_back.frames.len(), 2);
        for frame in &read_back.frames {
            assert_eq!(
                frame.domain.as_system_slice().expect("system bytes"),
                SOI,
                "every body survives the round trip"
            );
        }
    }

    #[test]
    fn a_boundary_property_that_could_not_be_written_is_refused() {
        let mut element = MultipartMux::new();
        for boundary in ["", " ", "with a space", &"b".repeat(MAX_BOUNDARY_LEN + 1)] {
            assert_eq!(
                element.set_property("boundary", PropValue::Str(boundary.into())),
                Err(PropError::Value),
                "{boundary}"
            );
        }
    }

    #[test]
    fn the_sniff_wants_a_delimiter_and_a_part_header_behind_it() {
        assert!(looks_like_multipart(
            b"--ffmpeg\r\nContent-type: image/jpeg\r\nContent-length: 2\r\n\r\n"
        ));
        assert!(!looks_like_multipart(b"--ffmpeg\r\n\xff\xd8\xff\xe0"));
        assert!(!looks_like_multipart(
            b"-- \r\nContent-Type: image/jpeg\r\n"
        ));
        assert!(!looks_like_multipart(b"YUV4MPEG2 W64 H48 F25:1 Ip\n"));
    }
}
