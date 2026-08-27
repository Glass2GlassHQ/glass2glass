//! YUV4MPEG2 (`.y4m`): the uncompressed planar-YUV stream format encoders,
//! conformance suites and quality tools exchange raw video in. [`Y4mDec`] reads
//! one (`Caps::ByteStream{Y4m}` in, `Caps::RawVideo` out), [`Y4mEnc`] writes one,
//! the `y4mdec` / `y4menc` analogs.
//!
//! ```text
//! filesrc location=in.y4m ! y4mdec ! fakesink
//! videotestsrc ! videoconvert ! y4menc ! filesink location=out.y4m
//! ```
//!
//! The format is a text stream header line (`YUV4MPEG2 W64 H48 F25:1 Ip C420jpeg`)
//! then, before each frame, a `FRAME` line and the frame's planes back to back
//! with no padding. Geometry, framerate and colourspace all come from that one
//! header, so a decoded stream's caps are known before the first frame and never
//! change mid-stream.
//!
//! The header is attacker-controlled: the dimensions are bounded by
//! [`MAX_DIMENSION`] and the frame size they imply by [`MAX_FRAME_BYTES`], both
//! checked before anything is sized from them, so a bogus `W99999999` fails the
//! parse instead of asking for terabytes.
//!
//! Not every y4m colourspace has a g2g format: `Cmono`, `C411`, `C444alpha` and
//! the 16-bit depths are rejected rather than guessed at. Neither is every scan
//! type: [`Interlace`] models progressive and top-field-first interleaved only,
//! so `Ib` (bottom field first), `Im` (mixed) and `I?` (unknown) are rejected.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_error, AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    Dim, ElementMetadata, FrameTiming, G2gError, Interlace, MemoryDomain, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, Rate, RawVideoFormat,
};

use crate::videorate::rate_fraction;

/// Signature every y4m stream opens with, the separating space included so a
/// file that merely starts with the word does not match.
const STREAM_MAGIC: &str = "YUV4MPEG2 ";
/// Signature of the line before each frame's planes.
const FRAME_MAGIC: &[u8] = b"FRAME";

/// Widest / tallest frame a header may declare. Above 8K in both axes, so no
/// real y4m file hits it, and low enough that one side cannot carry a bogus
/// allocation on its own.
const MAX_DIMENSION: u32 = 16384;

/// Byte ceiling on one frame, so a header inside [`MAX_DIMENSION`] on both axes
/// (16384 x 16384 4:4:4 at 12-bit is 1.5 GiB) still cannot ask for it.
const MAX_FRAME_BYTES: usize = 256 * 1024 * 1024;

/// Longest header line read, stream and frame alike. Both are a handful of short
/// parameters, so a stream with no newline inside this is not y4m.
const MAX_HEADER_LEN: usize = 1024;

/// The y4m `C` tag of each format g2g models, and the format each tag reads back
/// as. One table, so the reader and the writer cannot disagree about what a tag
/// means; the writer takes the first spelling of a format, which is the one
/// ffmpeg and gst write.
const COLOURSPACES: [(&str, RawVideoFormat); 12] = [
    ("420jpeg", RawVideoFormat::I420),
    ("420mpeg2", RawVideoFormat::I420),
    ("420paldv", RawVideoFormat::I420),
    ("420", RawVideoFormat::I420),
    ("422", RawVideoFormat::I422),
    ("444", RawVideoFormat::I444),
    ("420p10", RawVideoFormat::I420p10),
    ("422p10", RawVideoFormat::I422p10),
    ("444p10", RawVideoFormat::I444p10),
    ("420p12", RawVideoFormat::I420p12),
    ("422p12", RawVideoFormat::I422p12),
    ("444p12", RawVideoFormat::I444p12),
];

/// The colourspace a y4m file carries when its header names none.
const DEFAULT_COLOURSPACE: RawVideoFormat = RawVideoFormat::I420;

/// The format a `C` tag names, `None` for a colourspace with no g2g format.
fn colourspace_format(tag: &str) -> Option<RawVideoFormat> {
    COLOURSPACES
        .iter()
        .find(|(name, _)| *name == tag)
        .map(|(_, format)| *format)
}

/// The `C` tag written for a format, `None` for a format y4m cannot carry.
fn format_colourspace(format: RawVideoFormat) -> Option<&'static str> {
    COLOURSPACES
        .iter()
        .find(|(_, candidate)| *candidate == format)
        .map(|(name, _)| *name)
}

/// Every format a y4m file can carry, each once, in [`COLOURSPACES`] order.
fn y4m_formats() -> Vec<RawVideoFormat> {
    let mut formats = Vec::new();
    for (_, format) in COLOURSPACES {
        if !formats.contains(&format) {
            formats.push(format);
        }
    }
    formats
}

/// The byte stream both elements sit against.
fn y4m_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Y4m,
    }
}

/// What a y4m stream header describes, with every field already validated
/// against the bounds above.
#[derive(Debug, Clone, Copy, PartialEq)]
struct StreamHeader {
    format: RawVideoFormat,
    width: u32,
    height: u32,
    framerate_num: u32,
    framerate_den: u32,
    interlace: Interlace,
    /// Bytes of one frame's planes, [`MAX_FRAME_BYTES`] at most.
    frame_bytes: usize,
}

impl StreamHeader {
    fn caps(&self) -> Caps {
        Caps::RawVideo {
            format: self.format,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(rate_q16(self.framerate_num, self.framerate_den)),
            interlace: self.interlace,
        }
    }

    /// One frame's presentation span, taken from the header's own fraction
    /// rather than the Q16 rate it rounds to.
    fn frame_period_ns(&self) -> u64 {
        1_000_000_000u64 * self.framerate_den as u64 / self.framerate_num as u64
    }
}

/// A `num/den` framerate as the Q16 fixed-point fps [`Rate`] carries. The caller
/// has already bounded both, so the quotient fits.
fn rate_q16(num: u32, den: u32) -> u32 {
    (((num as u64) << 16) / den as u64) as u32
}

/// Read a `YUV4MPEG2 ...` header line (no trailing newline).
///
/// Every number here comes from the stream, so each is range-checked before it
/// reaches an allocation: the sides against [`MAX_DIMENSION`], the frame size
/// they imply against [`MAX_FRAME_BYTES`], and the framerate against a Q16 that
/// fits in a `u32`. Parameters this does not model (`A`, `X`, and anything a
/// writer invented) are skipped, as the format intends.
fn parse_stream_header(line: &[u8]) -> Result<StreamHeader, G2gError> {
    let text = core::str::from_utf8(line).map_err(|_| G2gError::CapsMismatch)?;
    let parameters = text
        .strip_prefix(STREAM_MAGIC)
        .ok_or(G2gError::CapsMismatch)?;

    let mut width = None;
    let mut height = None;
    let mut framerate = None;
    let mut interlace = Interlace::Progressive;
    let mut format = DEFAULT_COLOURSPACE;
    for token in parameters.split(' ').filter(|t| !t.is_empty()) {
        let mut characters = token.chars();
        let tag = characters.next().ok_or(G2gError::CapsMismatch)?;
        let value = characters.as_str();
        match tag {
            'W' => width = Some(parse_u32(value)?),
            'H' => height = Some(parse_u32(value)?),
            'F' => framerate = Some(parse_fraction(value)?),
            'I' => interlace = parse_interlace(value)?,
            'C' => format = colourspace_format(value).ok_or(G2gError::CapsMismatch)?,
            _ => {}
        }
    }

    let width = width.ok_or(G2gError::CapsMismatch)?;
    let height = height.ok_or(G2gError::CapsMismatch)?;
    let (framerate_num, framerate_den) = framerate.ok_or(G2gError::CapsMismatch)?;
    if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(G2gError::CapsMismatch);
    }
    if ((framerate_num as u64) << 16) / framerate_den as u64 > u32::MAX as u64 {
        return Err(G2gError::CapsMismatch);
    }
    let frame_bytes = format
        .unpadded_frame_bytes(width, height)
        .filter(|bytes| *bytes <= MAX_FRAME_BYTES as u64)
        .ok_or(G2gError::CapsMismatch)? as usize;
    Ok(StreamHeader {
        format,
        width,
        height,
        framerate_num,
        framerate_den,
        interlace,
        frame_bytes,
    })
}

fn parse_u32(value: &str) -> Result<u32, G2gError> {
    value.parse().map_err(|_| G2gError::CapsMismatch)
}

/// A `num:den` header fraction. Neither half may be zero: a zero denominator
/// divides and a zero numerator is a framerate no frame period follows from.
fn parse_fraction(value: &str) -> Result<(u32, u32), G2gError> {
    let (num, den) = value.split_once(':').ok_or(G2gError::CapsMismatch)?;
    let (num, den) = (parse_u32(num)?, parse_u32(den)?);
    if num == 0 || den == 0 {
        return Err(G2gError::CapsMismatch);
    }
    Ok((num, den))
}

/// The scan type an `I` parameter names. `Ib` / `Im` / `I?` have no [`Interlace`]
/// value, so a file declaring one is rejected rather than decoded as something
/// else.
fn parse_interlace(value: &str) -> Result<Interlace, G2gError> {
    match value {
        "p" => Ok(Interlace::Progressive),
        "t" => Ok(Interlace::Interleaved),
        _ => Err(G2gError::CapsMismatch),
    }
}

/// Offset of the newline ending the header line at the front of `buf`, `None`
/// while more bytes are needed. `Err` once [`MAX_HEADER_LEN`] bytes have gone by
/// without one.
fn header_line_end(buf: &[u8]) -> Result<Option<usize>, G2gError> {
    match buf.iter().take(MAX_HEADER_LEN).position(|&b| b == b'\n') {
        Some(at) => Ok(Some(at)),
        None if buf.len() >= MAX_HEADER_LEN => Err(G2gError::CapsMismatch),
        None => Ok(None),
    }
}

/// Reads the raw frames out of a YUV4MPEG2 byte stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::y4m::Y4mDec;
///
/// let element = Y4mDec::new();
/// ```
#[derive(Debug, Default)]
pub struct Y4mDec {
    configured: bool,
    /// Bytes accumulated across input chunks: a header line or a frame that
    /// straddles two of them stays here until the rest arrives.
    buf: Vec<u8>,
    header: Option<StreamHeader>,
    /// Bytes still wanted for the frame whose `FRAME` line was consumed.
    pending_frame: Option<usize>,
    emitted: u64,
}

impl Y4mDec {
    pub fn new() -> Self {
        Self::default()
    }

    /// The formats a y4m file can carry, at a fixable placeholder geometry: the
    /// real geometry, rate and format arrive with the `CapsChanged` the parsed
    /// header emits, so nothing here may be `Dim::Any` (which cannot fixate).
    fn output_alternatives() -> CapsSet {
        CapsSet::from_alternatives(
            y4m_formats()
                .into_iter()
                .map(|format| Caps::RawVideo {
                    format,
                    width: Dim::Range {
                        min: 1,
                        max: MAX_DIMENSION,
                    },
                    height: Dim::Range {
                        min: 1,
                        max: MAX_DIMENSION,
                    },
                    framerate: Rate::Range {
                        min_q16: 1 << 16,
                        max_q16: 240 << 16,
                    },
                    interlace: Interlace::Any,
                })
                .collect::<Vec<_>>(),
        )
    }

    /// Read the stream header, then every complete frame the buffer holds.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.header.is_none() {
            let Some(end) = header_line_end(&self.buf)? else {
                return Ok(());
            };
            let header = parse_stream_header(&self.buf[..end]).inspect_err(|_| {
                g2g_error!(
                    self,
                    "unreadable YUV4MPEG2 stream header: {}",
                    core::str::from_utf8(&self.buf[..end]).unwrap_or("<not utf-8>")
                );
            })?;
            self.buf.drain(..end + 1);
            out.push(PipelinePacket::CapsChanged(header.caps())).await?;
            self.header = Some(header);
        }
        let header = self.header.ok_or(G2gError::CapsMismatch)?;
        loop {
            if self.pending_frame.is_none() {
                let Some(end) = header_line_end(&self.buf)? else {
                    return Ok(());
                };
                let line = &self.buf[..end];
                // Parameters after `FRAME` are per-frame overrides nothing here
                // acts on, so only the signature is checked.
                let framed = line.starts_with(FRAME_MAGIC)
                    && (line.len() == FRAME_MAGIC.len() || line[FRAME_MAGIC.len()] == b' ');
                if !framed {
                    g2g_error!(self, "expected a FRAME header at byte {}", end);
                    return Err(G2gError::CapsMismatch);
                }
                self.buf.drain(..end + 1);
                self.pending_frame = Some(header.frame_bytes);
            }
            let wanted = self.pending_frame.unwrap_or(header.frame_bytes);
            if self.buf.len() < wanted {
                return Ok(());
            }
            let planes: Vec<u8> = self.buf.drain(..wanted).collect();
            self.pending_frame = None;
            let period_ns = header.frame_period_ns();
            let pts_ns = self.emitted.saturating_mul(period_ns);
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(planes.into_boxed_slice())),
                FrameTiming {
                    pts_ns,
                    dts_ns: pts_ns,
                    duration_ns: period_ns,
                    // raw video: every frame decodes on its own.
                    keyframe: true,
                    ..Default::default()
                },
                self.emitted,
            );
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
    }
}

impl AsyncElement for Y4mDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn is_format_boundary(&self) -> bool {
        true
    }

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&y4m_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Y4m,
            } => Self::output_alternatives(),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Y4m
            }
        ) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "YUV4MPEG2 decoder",
            "Codec/Demuxer/Video",
            "Reads the raw planar YUV frames out of a YUV4MPEG2 byte stream",
            "g2g",
        )
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
                    self.drain(out).await?;
                }
                PipelinePacket::Eos => {
                    self.drain(out).await?;
                    if self.header.is_none() || !self.buf.is_empty() {
                        g2g_error!(
                            self,
                            "the stream ended mid-frame: {} bytes short of a complete y4m stream",
                            self.pending_frame.map_or(self.buf.len(), |wanted| wanted
                                .saturating_sub(self.buf.len()))
                        );
                        return Err(G2gError::CapsMismatch);
                    }
                }
                // The byte stream carries no geometry; the concrete caps come
                // from the parsed stream header instead.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl LogSource for Y4mDec {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
}

impl PadTemplates for Y4mDec {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(y4m_caps())),
            PadTemplate::source(Self::output_alternatives()),
        ])
    }
}

/// The stream header for a raw video stream, `None` for a format y4m cannot
/// carry.
///
/// No `A` parameter: g2g caps carry no pixel aspect ratio, and writing `A1:1`
/// would claim square pixels the stream never declared.
fn stream_header(
    format: RawVideoFormat,
    width: u32,
    height: u32,
    framerate_q16: u32,
    interlace: Interlace,
) -> Option<Vec<u8>> {
    let colourspace = format_colourspace(format)?;
    let (num, den) = rate_fraction(framerate_q16);
    let scan = match interlace {
        Interlace::Interleaved => 't',
        Interlace::Any | Interlace::Progressive => 'p',
    };
    Some(
        format!("{STREAM_MAGIC}W{width} H{height} F{num}:{den} I{scan} C{colourspace}\n")
            .into_bytes(),
    )
}

/// Writes raw planar YUV frames as a YUV4MPEG2 byte stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::y4m::Y4mEnc;
///
/// let element = Y4mEnc::new();
/// ```
#[derive(Debug, Default)]
pub struct Y4mEnc {
    /// The negotiated stream, captured at configure time so the header can
    /// describe it.
    input: Option<StreamHeader>,
    header_written: bool,
    configured: bool,
    emitted: u64,
}

impl Y4mEnc {
    pub fn new() -> Self {
        Self::default()
    }

    /// The raw video a y4m file can carry, at the geometry wildcards a static
    /// pad template advertises.
    fn input_alternatives() -> CapsSet {
        CapsSet::from_alternatives(
            y4m_formats()
                .into_iter()
                .map(|format| Caps::RawVideo {
                    format,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                    interlace: Interlace::Any,
                })
                .collect::<Vec<_>>(),
        )
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

impl AsyncElement for Y4mEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn is_format_boundary(&self) -> bool {
        true
    }

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    /// Only the planar YUV a y4m file can hold. A packed or semi-planar input
    /// (`Rgba8`, `Nv12`, `Yuyv`) is refused here so the solver puts a
    /// `videoconvert` ahead instead of writing a file no reader can open.
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::RawVideo { format, .. } if format_colourspace(*format).is_some() => {
                Ok(upstream_caps.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo { format, .. } if format_colourspace(*format).is_some() => {
                CapsSet::one(y4m_caps())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate: Rate::Fixed(framerate_q16),
            interlace,
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if format_colourspace(*format).is_none() || *framerate_q16 == 0 {
            return Err(G2gError::CapsMismatch);
        }
        let frame_bytes = format
            .unpadded_frame_bytes(*width, *height)
            .filter(|bytes| *bytes > 0 && *bytes <= MAX_FRAME_BYTES as u64)
            .ok_or(G2gError::CapsMismatch)? as usize;
        let (framerate_num, framerate_den) = rate_fraction(*framerate_q16);
        self.input = Some(StreamHeader {
            format: *format,
            width: *width,
            height: *height,
            framerate_num,
            framerate_den,
            interlace: *interlace,
            frame_bytes,
        });
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "YUV4MPEG2 encoder",
            "Codec/Muxer/Video",
            "Wraps raw planar YUV frames in a YUV4MPEG2 byte stream",
            "g2g",
        )
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
                    let header = self.input.ok_or(G2gError::NotConfigured)?;
                    if !self.header_written {
                        out.push(PipelinePacket::CapsChanged(y4m_caps())).await?;
                        let bytes = stream_header(
                            header.format,
                            header.width,
                            header.height,
                            rate_q16(header.framerate_num, header.framerate_den),
                            header.interlace,
                        )
                        .ok_or(G2gError::CapsMismatch)?;
                        let packet = self.frame(bytes);
                        out.push(packet).await?;
                        self.header_written = true;
                    }
                    let planes = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    if planes.len() != header.frame_bytes {
                        return Err(G2gError::CapsMismatch);
                    }
                    let mut bytes = Vec::with_capacity(FRAME_MAGIC.len() + 1 + planes.len());
                    bytes.extend_from_slice(FRAME_MAGIC);
                    bytes.push(b'\n');
                    bytes.extend_from_slice(planes);
                    let packet = self.frame(bytes);
                    out.push(packet).await?;
                }
                // The input caps describe raw video; this element's output is the
                // y4m stream it already announced. A geometry change mid-stream
                // would need a second header, which the format has no room for.
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for Y4mEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(Self::input_alternatives()),
            PadTemplate::source(CapsSet::one(y4m_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::runtime::block_on;
    use g2g_core::PushOutcome;

    /// A well-formed 4:2:0 header, the shape ffmpeg writes.
    const I420_HEADER: &str = "YUV4MPEG2 W64 H48 F25:1 Ip A1:1 C420jpeg XYSCSS=420JPEG";

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

    /// Run the decoder over `stream` and end it, returning what came out.
    fn decode(stream: &[u8]) -> Result<Captured, G2gError> {
        let mut element = Y4mDec::new();
        element.configure_pipeline(&y4m_caps()).expect("y4m in");
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

    /// The error a malformed stream fails with, `None` when it decoded.
    fn decode_error(stream: &[u8]) -> Option<G2gError> {
        decode(stream).err()
    }

    /// A complete one-frame stream at `header`'s geometry.
    fn one_frame_stream(header: &str, frame_bytes: usize) -> Vec<u8> {
        let mut stream = Vec::from(header.as_bytes());
        stream.extend_from_slice(b"\nFRAME\n");
        stream.resize(stream.len() + frame_bytes, 0x80);
        stream
    }

    #[test]
    fn header_parses_every_field_and_skips_the_ones_not_modeled() {
        let header = parse_stream_header(I420_HEADER.as_bytes()).expect("a valid header");
        assert_eq!(
            header,
            StreamHeader {
                format: RawVideoFormat::I420,
                width: 64,
                height: 48,
                framerate_num: 25,
                framerate_den: 1,
                interlace: Interlace::Progressive,
                frame_bytes: 64 * 48 * 3 / 2,
            }
        );
        assert_eq!(header.frame_period_ns(), 40_000_000);
    }

    #[test]
    fn a_truncated_header_waits_rather_than_parsing_a_partial_line() {
        assert_eq!(
            decode_error(b"YUV4MPEG2 W64 H48 F25"),
            Some(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn a_header_without_a_newline_inside_the_budget_is_refused() {
        let mut line = Vec::from(b"YUV4MPEG2 W64 H48 F25:1 Ip".as_slice());
        line.resize(MAX_HEADER_LEN + 1, b' ');
        assert_eq!(decode_error(&line), Some(G2gError::CapsMismatch));
    }

    #[test]
    fn a_zero_or_oversized_geometry_is_refused_before_anything_is_sized_from_it() {
        for header in [
            "YUV4MPEG2 W0 H48 F25:1 Ip C420jpeg",
            "YUV4MPEG2 W64 H0 F25:1 Ip C420jpeg",
            "YUV4MPEG2 W99999999 H99999999 F25:1 Ip C420jpeg",
            // inside MAX_DIMENSION on both sides, past the byte budget
            "YUV4MPEG2 W16384 H16384 F25:1 Ip C444p12",
        ] {
            assert_eq!(
                parse_stream_header(header.as_bytes()),
                Err(G2gError::CapsMismatch),
                "{header}"
            );
        }
    }

    #[test]
    fn a_colourspace_or_scan_type_with_no_g2g_equivalent_is_refused() {
        for header in [
            "YUV4MPEG2 W64 H48 F25:1 Ip Cmono",
            "YUV4MPEG2 W64 H48 F25:1 Ip C411",
            "YUV4MPEG2 W64 H48 F25:1 Ip C420p16",
            "YUV4MPEG2 W64 H48 F25:1 Ib C420jpeg",
            "YUV4MPEG2 W64 H48 F25:1 Im C420jpeg",
            "YUV4MPEG2 W64 H48 F25:1 I? C420jpeg",
            "YUV4MPEG2 W64 H48 F0:1 Ip C420jpeg",
            "YUV4MPEG2 W64 H48 F25:0 Ip C420jpeg",
            "YUV4MPEG2 W64 H48 Ip C420jpeg",
            "MJPEGTOOLS W64 H48 F25:1 Ip C420jpeg",
        ] {
            assert_eq!(
                parse_stream_header(header.as_bytes()),
                Err(G2gError::CapsMismatch),
                "{header}"
            );
        }
    }

    #[test]
    fn a_frame_cut_short_ends_the_stream_with_an_error() {
        let full = one_frame_stream(I420_HEADER, 64 * 48 * 3 / 2);
        assert_eq!(decode(&full).expect("a whole frame").frames.len(), 1);
        assert_eq!(
            decode_error(&full[..full.len() - 1]),
            Some(G2gError::CapsMismatch),
            "one byte short of the frame"
        );
    }

    #[test]
    fn a_frame_not_introduced_by_its_own_header_is_refused() {
        let mut stream = Vec::from(I420_HEADER.as_bytes());
        stream.extend_from_slice(b"\nCHUNK\n");
        stream.resize(stream.len() + 64 * 48 * 3 / 2, 0);
        assert_eq!(decode_error(&stream), Some(G2gError::CapsMismatch));
    }

    #[test]
    fn the_writer_and_the_reader_agree_on_every_colourspace_tag() {
        for format in y4m_formats() {
            let tag = format_colourspace(format).expect("a y4m format has a tag");
            assert_eq!(colourspace_format(tag), Some(format));
            let header = stream_header(format, 64, 48, 25 << 16, Interlace::Progressive)
                .expect("a y4m format has a header");
            let line = &header[..header.len() - 1];
            assert_eq!(
                parse_stream_header(line).map(|h| (h.format, h.width, h.height)),
                Ok((format, 64, 48)),
                "{format:?}"
            );
        }
    }

    #[test]
    fn the_writer_refuses_a_format_the_file_cannot_hold() {
        for format in [
            RawVideoFormat::Rgba8,
            RawVideoFormat::Nv12,
            RawVideoFormat::Yuyv,
            RawVideoFormat::P010,
        ] {
            assert_eq!(format_colourspace(format), None, "{format:?}");
            assert_eq!(
                Y4mEnc::new().intercept_caps(&Caps::RawVideo {
                    format,
                    width: Dim::Fixed(64),
                    height: Dim::Fixed(48),
                    framerate: Rate::Fixed(25 << 16),
                    interlace: Interlace::Any,
                }),
                Err(G2gError::CapsMismatch),
                "{format:?}"
            );
        }
    }

    #[test]
    fn an_interlaced_stream_round_trips_its_scan_type() {
        // The caps carry a Q16 rate, so 30000/1001 is written as the Q16 value
        // over 65536: the same rate to 6 decimal places, not the same fraction.
        let ntsc_q16 = ((30_000u64 << 16) / 1001) as u32;
        let header = stream_header(
            RawVideoFormat::I422,
            64,
            48,
            ntsc_q16,
            Interlace::Interleaved,
        )
        .expect("a y4m format");
        assert_eq!(
            core::str::from_utf8(&header),
            Ok("YUV4MPEG2 W64 H48 F1964115:65536 It C422\n")
        );
        let parsed = parse_stream_header(&header[..header.len() - 1]).expect("its own header");
        assert_eq!(parsed.interlace, Interlace::Interleaved);
        assert_eq!(
            rate_q16(parsed.framerate_num, parsed.framerate_den),
            ntsc_q16
        );
    }
}
