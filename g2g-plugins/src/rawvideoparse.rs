//! Headerless raw-video framer (`rawvideoparse`): a `ByteStream{Raw}` dump in
//! (a `.yuv` file, arbitrary chunks from `filesrc`), one `RawVideo` frame out
//! per `width * height` frame's worth of bytes.
//!
//! Nothing in the file says what the pixels are, so the format, geometry and
//! framerate are properties, and the output caps state what they declare. The
//! frame size follows from format and geometry with no row padding
//! ([`RawVideoFormat::unpadded_frame_bytes`]); `frame-size` covers a dump whose
//! frames are spaced further apart than their pixels need, skipping the gap.
//!
//! A trailing partial frame at end of stream is dropped: half a frame has no
//! pixels for the rest of the picture.
//!
//! `plane-strides` and `plane-offsets` read a dump whose rows or planes are
//! padded (M1093), the layout a capture device or a GPU readback writes. Padding
//! is undone by packing the rows tight, unless a consumer asked for a
//! [`PlaneLayout`](g2g_core::meta::PlaneLayout), in which case the frame is
//! passed through as it lies with that layout declared and nothing is copied.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// Planes a raw format can have, the ceiling the property lists are checked
/// against (the same bound `PlaneLayout` carries).
const MAX_PLANES: usize = 4;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_warn, AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    Dim, ElementMetadata, FrameTiming, G2gError, Interlace, MemoryDomain, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate,
    RawVideoFormat,
};

use crate::compositor::frame_period_ns;
use crate::videoconvert::{raw_format_from_str, raw_format_to_str};

/// gst `rawvideoparse` defaults: I420 320x240 at 25 fps, frames tightly packed.
const DEFAULT_FORMAT: RawVideoFormat = RawVideoFormat::I420;
const DEFAULT_WIDTH: u32 = 320;
const DEFAULT_HEIGHT: u32 = 240;
const DEFAULT_FRAMERATE: (u32, u32) = (25, 1);

/// The same values as declared text, for `gst-inspect`.
const DEFAULT_WIDTH_TEXT: &str = "320";
const DEFAULT_HEIGHT_TEXT: &str = "240";
const DEFAULT_FRAMERATE_TEXT: &str = "25/1";

/// Widest / tallest frame accepted, so a bogus `width=4000000000` fails the
/// property rather than sizing a read from it. 8K in both axes.
const MAX_DIMENSION: u32 = 8192;

static RAWVIDEOPARSE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "format",
        PropKind::Str,
        "pixel format of the raw stream: I420 | NV12 | RGBA | BGRA | YUY2",
    )
    .with_default("I420"),
    PropertySpec::new("width", PropKind::Uint, "frame width in pixels")
        .with_default(DEFAULT_WIDTH_TEXT)
        .with_range("1", "8192"),
    PropertySpec::new("height", PropKind::Uint, "frame height in pixels")
        .with_default(DEFAULT_HEIGHT_TEXT)
        .with_range("1", "8192"),
    PropertySpec::new("framerate", PropKind::Fraction, "frames per second")
        .with_default(DEFAULT_FRAMERATE_TEXT),
    PropertySpec::new(
        "frame-size",
        PropKind::Uint,
        "bytes one frame occupies in the file (0 = the pixels' own size, frames packed back to back)",
    )
    .with_default("0"),
    PropertySpec::new(
        "plane-strides",
        PropKind::Str,
        "row stride of each plane in bytes, comma separated (empty = the format's own, unpadded)",
    ),
    PropertySpec::new(
        "plane-offsets",
        PropKind::Str,
        "byte offset of each plane within the frame, comma separated (empty = the planes back to back)",
    ),
];

/// One plane of a frame as it lies in the file: where its rows start, how far
/// apart they are, how wide each one is and how many there are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlaneSpan {
    offset: usize,
    stride: usize,
    row_bytes: usize,
    rows: usize,
}

/// Parse a comma-separated list of byte counts. `None` on a value that is not a
/// number, so a mistyped property fails rather than silently reading zeros.
fn parse_byte_list(text: &str) -> Option<Vec<usize>> {
    if text.trim().is_empty() {
        return Some(Vec::new());
    }
    text.split(',')
        .map(|part| part.trim().parse::<usize>().ok())
        .collect()
}

/// The same list as text, for `get_property`.
fn byte_list_text(values: &[usize]) -> String {
    let mut out = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&alloc::format!("{value}"));
    }
    out
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::rawvideoparse::RawVideoParse;
///
/// // gst-launch equivalent:
/// // filesrc location=frames.yuv ! rawvideoparse width=640 height=480 format=I420
/// let parser = RawVideoParse::new().with_geometry(640, 480);
/// ```
#[derive(Debug)]
pub struct RawVideoParse {
    format: RawVideoFormat,
    width: u32,
    height: u32,
    framerate: (u32, u32),
    /// gst `frame-size`: the file's per-frame stride. 0 means the pixels' own
    /// size, with nothing between frames.
    frame_size: u64,
    /// Per-plane row strides and offsets as the file lays them out; empty means
    /// the format's own tightly packed layout.
    plane_strides: Vec<usize>,
    plane_offsets: Vec<usize>,
    /// Set when a consumer asked for a `PlaneLayout`: the padded frame then goes
    /// downstream as it lies, with the layout declared, instead of being packed.
    keep_padding: bool,
    configured: bool,
    caps_sent: bool,
    /// Unconsumed input bytes.
    buf: Vec<u8>,
    emitted: u64,
    log_name: LogName,
}

impl Default for RawVideoParse {
    fn default() -> Self {
        Self::new()
    }
}

impl RawVideoParse {
    pub fn new() -> Self {
        Self {
            format: DEFAULT_FORMAT,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            framerate: DEFAULT_FRAMERATE,
            frame_size: 0,
            plane_strides: Vec::new(),
            plane_offsets: Vec::new(),
            keep_padding: false,
            configured: false,
            caps_sent: false,
            buf: Vec::new(),
            emitted: 0,
            log_name: LogName::default(),
        }
    }

    pub fn with_format(mut self, format: RawVideoFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_geometry(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_framerate(mut self, numerator: u32, denominator: u32) -> Self {
        if denominator > 0 {
            self.framerate = (numerator, denominator);
        }
        self
    }

    /// Frames emitted so far.
    pub fn frames_emitted(&self) -> u64 {
        self.emitted
    }

    /// The output framerate in the Q16 fixed-point fps `Rate` carries.
    fn rate_q16(&self) -> u32 {
        let (numerator, denominator) = self.framerate;
        if denominator == 0 {
            return 0;
        }
        u32::try_from((u64::from(numerator) << 16) / u64::from(denominator)).unwrap_or(u32::MAX)
    }

    /// Bytes of pixels in one frame, `None` for a format / geometry pair with no
    /// unpadded layout.
    fn pixel_bytes(&self) -> Option<usize> {
        let bytes = self.format.unpadded_frame_bytes(self.width, self.height)?;
        usize::try_from(bytes).ok().filter(|bytes| *bytes > 0)
    }

    /// Where each plane's rows lie in one frame of the file: the format's own
    /// unpadded layout, with `plane-strides` / `plane-offsets` overriding it
    /// plane by plane. `None` when a declared stride cannot hold its own row,
    /// when the lists do not describe every plane, or on overflow.
    fn plane_spans(&self) -> Option<Vec<PlaneSpan>> {
        let planes = self.format.plane_count();
        for declared in [&self.plane_strides, &self.plane_offsets] {
            if !declared.is_empty() && declared.len() != planes {
                return None;
            }
        }
        let mut spans = Vec::with_capacity(planes);
        for plane in 0..planes {
            let row_bytes = self.format.plane_stride(plane, self.width)? as usize;
            let rows = self.format.plane_rows(plane, self.height)? as usize;
            let stride = match self.plane_strides.get(plane) {
                Some(declared) if *declared < row_bytes => return None,
                Some(declared) => *declared,
                None => row_bytes,
            };
            let offset = match self.plane_offsets.get(plane) {
                Some(declared) => *declared,
                // Back to back, each plane after the one before it.
                None => spans
                    .last()
                    .map(|last: &PlaneSpan| {
                        last.offset.checked_add(last.stride.checked_mul(last.rows)?)
                    })
                    .unwrap_or(Some(0))?,
            };
            spans.push(PlaneSpan {
                offset,
                stride,
                row_bytes,
                rows,
            });
        }
        Some(spans)
    }

    /// Bytes one frame occupies in the file: past the last row of every plane,
    /// plus whatever `frame-size` puts after them.
    fn stride(&self) -> Option<usize> {
        let spans = self.plane_spans()?;
        let mut used = 0usize;
        for span in &spans {
            let end = span
                .offset
                .checked_add(span.stride.checked_mul(span.rows.saturating_sub(1))?)?
                .checked_add(span.row_bytes)?;
            used = used.max(end);
        }
        if used == 0 {
            return None;
        }
        if self.frame_size == 0 {
            return Some(used);
        }
        usize::try_from(self.frame_size)
            .ok()
            .filter(|declared| *declared >= used)
    }

    /// Whether the file's layout differs from the tightly packed one downstream
    /// assumes by default.
    fn padded(&self) -> bool {
        !self.plane_strides.is_empty() || !self.plane_offsets.is_empty()
    }

    /// Copy `frame`'s rows out of the padded layout into the tight one.
    fn pack_rows(&self, frame: &[u8]) -> Option<Vec<u8>> {
        let mut out = Vec::with_capacity(self.pixel_bytes()?);
        for span in self.plane_spans()? {
            for row in 0..span.rows {
                let start = span.offset.checked_add(span.stride.checked_mul(row)?)?;
                let end = start.checked_add(span.row_bytes)?;
                out.extend_from_slice(frame.get(start..end)?);
            }
        }
        Some(out)
    }

    /// Declare on `frame` where its padded rows sit, so a consumer that asked
    /// for the layout reads them in place.
    #[cfg(feature = "metadata")]
    fn declare_planes(&self, frame: &mut Frame) {
        let Some(spans) = self.plane_spans() else {
            return;
        };
        let planes: Vec<g2g_core::meta::Plane> = spans
            .iter()
            .map(|span| g2g_core::meta::Plane {
                offset: span.offset,
                stride: span.stride,
            })
            .collect();
        if let Some(layout) = g2g_core::meta::PlaneLayout::new(&planes) {
            frame.meta.attach(layout);
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::RawVideo {
            format: self.format,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(self.rate_q16()),
            interlace: Interlace::Progressive,
        }
    }

    fn input_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Raw,
        }
    }

    /// Emit every whole frame the buffer holds.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let pixels = self.pixel_bytes().ok_or(G2gError::CapsMismatch)?;
        let stride = self.stride().ok_or(G2gError::CapsMismatch)?;
        if !self.caps_sent {
            out.push(PipelinePacket::CapsChanged(self.output_caps()))
                .await?;
            self.caps_sent = true;
        }
        let period_ns = frame_period_ns(self.rate_q16());
        // A padded frame either goes downstream as it lies, with its layout
        // declared, or has its rows packed into the shape the caps promise.
        let declare = self.keep_padding && self.padded();
        while self.buf.len() >= stride {
            let data: Vec<u8> = self.buf.drain(..stride).collect();
            let payload = match (declare, self.padded()) {
                (true, _) => data,
                (false, true) => self.pack_rows(&data).ok_or(G2gError::CapsMismatch)?,
                (false, false) => data[..pixels].to_vec(),
            };
            let pts_ns = self.emitted.saturating_mul(period_ns);
            #[allow(unused_mut)]
            let mut frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
                FrameTiming {
                    pts_ns,
                    dts_ns: pts_ns,
                    duration_ns: period_ns,
                    // Raw pixels: every frame stands alone.
                    keyframe: true,
                    ..FrameTiming::default()
                },
                self.emitted,
            );
            #[cfg(feature = "metadata")]
            if declare {
                self.declare_planes(&mut frame);
            }
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for RawVideoParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Raw video framer",
            "Codec/Parser/Video",
            "Frames a headerless raw video dump using its declared format and geometry",
            "g2g",
        )
    }

    /// Reads host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let caps = self.output_caps();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Raw,
            } => CapsSet::one(caps.clone()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Raw
            }
        ) {
            return Err(G2gError::CapsMismatch);
        }
        // A zero rate or a format with no unpadded layout leaves the frame size
        // undefined, so there is nothing to cut the stream into.
        if self.rate_q16() == 0 || self.stride().is_none() {
            return Err(G2gError::CapsMismatch);
        }
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
                    if !self.buf.is_empty() {
                        g2g_warn!(self, "dropped {} trailing bytes", self.buf.len());
                        self.buf.clear();
                    }
                }
                PipelinePacket::Flush => {
                    self.buf.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                // The declared shape replaces the byte stream's caps, which
                // carry no geometry.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    /// A consumer that asked for a `PlaneLayout` reads padded rows in place, so
    /// the packing copy is skipped for it.
    fn configure_allocation(&mut self, params: &g2g_core::AllocationParams) {
        #[cfg(feature = "metadata")]
        {
            self.keep_padding = params.meta_requests.wants::<g2g_core::meta::PlaneLayout>();
        }
        #[cfg(not(feature = "metadata"))]
        let _ = params;
    }

    fn properties(&self) -> &'static [PropertySpec] {
        RAWVIDEOPARSE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "format" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                self.format = raw_format_from_str(text).ok_or(PropError::Value)?;
            }
            "width" | "height" => {
                let pixels = value.as_uint().ok_or(PropError::Type)?;
                if pixels == 0 || pixels > u64::from(MAX_DIMENSION) {
                    return Err(PropError::Value);
                }
                if name == "width" {
                    self.width = pixels as u32;
                } else {
                    self.height = pixels as u32;
                }
            }
            "framerate" => {
                let (numerator, denominator) = value.as_fraction().ok_or(PropError::Type)?;
                if numerator <= 0 || denominator <= 0 {
                    return Err(PropError::Value);
                }
                self.framerate = (numerator as u32, denominator as u32);
            }
            "frame-size" => {
                self.frame_size = value.as_uint().ok_or(PropError::Type)?;
            }
            "plane-strides" | "plane-offsets" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                let list = parse_byte_list(text).ok_or(PropError::Value)?;
                if list.len() > MAX_PLANES {
                    return Err(PropError::Value);
                }
                if name == "plane-strides" {
                    self.plane_strides = list;
                } else {
                    self.plane_offsets = list;
                }
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "format" => Some(PropValue::Str(raw_format_to_str(self.format).into())),
            "width" => Some(PropValue::Uint(u64::from(self.width))),
            "height" => Some(PropValue::Uint(u64::from(self.height))),
            "framerate" => Some(PropValue::Fraction(
                self.framerate.0 as i32,
                self.framerate.1 as i32,
            )),
            "frame-size" => Some(PropValue::Uint(self.frame_size)),
            "plane-strides" => Some(PropValue::Str(byte_list_text(&self.plane_strides))),
            "plane-offsets" => Some(PropValue::Str(byte_list_text(&self.plane_offsets))),
            _ => None,
        }
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }
}

impl LogSource for RawVideoParse {
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

impl PadTemplates for RawVideoParse {
    fn pad_templates() -> Vec<PadTemplate> {
        // Static superset: the declared geometry belongs to an instance, so the
        // template leaves it open.
        let raw = Caps::RawVideo {
            format: DEFAULT_FORMAT,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Any,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(CapsSet::one(raw)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use g2g_core::PushOutcome;

    #[derive(Default)]
    struct RecordingSink {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for RecordingSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            self.packets.push(packet);
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    impl RecordingSink {
        fn caps(&self) -> Vec<&Caps> {
            self.packets
                .iter()
                .filter_map(|p| match p {
                    PipelinePacket::CapsChanged(c) => Some(c),
                    _ => None,
                })
                .collect()
        }

        fn frames(&self) -> Vec<&Frame> {
            self.packets
                .iter()
                .filter_map(|p| match p {
                    PipelinePacket::DataFrame(f) => Some(f),
                    _ => None,
                })
                .collect()
        }
    }

    /// A small even geometry, so I420 chroma divides.
    const WIDTH: u32 = 16;
    const HEIGHT: u32 = 8;
    /// I420 at 16x8: luma 128 bytes, each chroma plane 32.
    const FRAME_BYTES: usize = 192;

    fn parser() -> RawVideoParse {
        let mut parser = RawVideoParse::new().with_geometry(WIDTH, HEIGHT);
        parser
            .configure_pipeline(&RawVideoParse::input_caps())
            .expect("a raw byte stream");
        parser
    }

    fn data_frame(bytes: Vec<u8>) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming::default(),
            0,
        ))
    }

    #[tokio::test]
    async fn cuts_the_stream_into_frames_of_the_declared_size() {
        let mut parser = parser();
        let mut sink = RecordingSink::default();
        // An odd chunk length, so frames straddle buffer boundaries.
        let bytes: Vec<u8> = (0..FRAME_BYTES * 3).map(|i| i as u8).collect();
        for piece in bytes.chunks(77) {
            parser
                .process(data_frame(piece.to_vec()), &mut sink)
                .await
                .expect("the chunk parses");
        }
        parser
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("the tail flushes");
        let frames = sink.frames();
        assert_eq!(frames.len(), 3);
        assert!(frames
            .iter()
            .all(|f| f.domain.as_system_slice().map(<[u8]>::len) == Some(FRAME_BYTES)));
        // The pixels arrive in file order, not reassembled out of the chunks.
        assert_eq!(
            frames[1].domain.as_system_slice().expect("system")[..4],
            bytes[FRAME_BYTES..FRAME_BYTES + 4]
        );
    }

    #[tokio::test]
    async fn drops_a_trailing_partial_frame() {
        let mut parser = parser();
        let mut sink = RecordingSink::default();
        parser
            .process(data_frame(vec![7u8; FRAME_BYTES + 5]), &mut sink)
            .await
            .expect("the buffer parses");
        parser
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("the tail flushes");
        assert_eq!(sink.frames().len(), 1);
    }

    #[tokio::test]
    async fn stamps_the_declared_framerate() {
        let mut parser = RawVideoParse::new()
            .with_geometry(WIDTH, HEIGHT)
            .with_framerate(50, 1);
        parser
            .configure_pipeline(&RawVideoParse::input_caps())
            .expect("a raw byte stream");
        let mut sink = RecordingSink::default();
        parser
            .process(data_frame(vec![0u8; FRAME_BYTES * 2]), &mut sink)
            .await
            .expect("the buffer parses");
        const PERIOD_NS: u64 = 20_000_000;
        let times: Vec<(u64, u64)> = sink
            .frames()
            .iter()
            .map(|f| (f.timing.pts_ns, f.timing.duration_ns))
            .collect();
        assert_eq!(times, vec![(0, PERIOD_NS), (PERIOD_NS, PERIOD_NS)]);
        assert_eq!(
            sink.caps(),
            vec![&Caps::RawVideo {
                format: RawVideoFormat::I420,
                width: Dim::Fixed(WIDTH),
                height: Dim::Fixed(HEIGHT),
                framerate: Rate::Fixed(50 << 16),
                interlace: Interlace::Progressive,
            }]
        );
    }

    #[tokio::test]
    async fn frame_size_skips_the_padding_after_each_frame() {
        const PADDING: usize = 8;
        let mut parser = parser();
        parser
            .set_property(
                "frame-size",
                PropValue::Uint((FRAME_BYTES + PADDING) as u64),
            )
            .expect("a size at least the pixels' own");
        let mut sink = RecordingSink::default();
        let mut bytes = vec![1u8; FRAME_BYTES];
        bytes.extend(vec![0xEE; PADDING]);
        bytes.extend(vec![2u8; FRAME_BYTES]);
        bytes.extend(vec![0xEE; PADDING]);
        parser
            .process(data_frame(bytes), &mut sink)
            .await
            .expect("the buffer parses");
        let frames = sink.frames();
        assert_eq!(frames.len(), 2);
        for (index, fill) in [1u8, 2u8].into_iter().enumerate() {
            let pixels = frames[index].domain.as_system_slice().expect("system");
            assert_eq!(pixels.len(), FRAME_BYTES);
            assert!(
                pixels.iter().all(|byte| *byte == fill),
                "padding is not read"
            );
        }
    }

    /// A padded dump: 8 bytes of slack on every luma row and 4 on each chroma
    /// row, the shape a capture device writes. The rows come out packed, so
    /// downstream sees exactly the pixels the caps promise.
    #[tokio::test]
    async fn packs_padded_rows_into_the_declared_frame() {
        const LUMA_STRIDE: usize = WIDTH as usize + 8;
        const CHROMA_STRIDE: usize = WIDTH as usize / 2 + 4;
        let mut parser = RawVideoParse::new().with_geometry(WIDTH, HEIGHT);
        parser
            .set_property(
                "plane-strides",
                PropValue::Str(alloc::format!(
                    "{LUMA_STRIDE},{CHROMA_STRIDE},{CHROMA_STRIDE}"
                )),
            )
            .expect("three strides for three planes");
        parser
            .configure_pipeline(&RawVideoParse::input_caps())
            .expect("a raw byte stream");

        // Build one padded frame whose payload bytes name their row, and whose
        // padding is a value the packed output must not contain.
        const PADDING: u8 = 0xEE;
        let mut file = Vec::new();
        let mut expected = Vec::new();
        for (rows, row_bytes, stride) in [
            (HEIGHT as usize, WIDTH as usize, LUMA_STRIDE),
            (HEIGHT as usize / 2, WIDTH as usize / 2, CHROMA_STRIDE),
            (HEIGHT as usize / 2, WIDTH as usize / 2, CHROMA_STRIDE),
        ] {
            for row in 0..rows {
                let payload = vec![row as u8; row_bytes];
                file.extend_from_slice(&payload);
                file.extend(vec![PADDING; stride - row_bytes]);
                expected.extend_from_slice(&payload);
            }
        }
        let mut sink = RecordingSink::default();
        parser
            .process(data_frame(file), &mut sink)
            .await
            .expect("the frame parses");
        let frames = sink.frames();
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0].domain.as_system_slice().expect("system"),
            &expected[..],
            "the padding is gone and every row is in place"
        );
    }

    /// The same dump, with a consumer that asked for the layout: nothing is
    /// copied and the strides are declared instead.
    #[cfg(feature = "metadata")]
    #[tokio::test]
    async fn declares_the_layout_when_a_consumer_wants_it() {
        const LUMA_STRIDE: usize = WIDTH as usize + 8;
        let mut parser = RawVideoParse::new().with_geometry(WIDTH, HEIGHT);
        parser
            .set_property(
                "plane-strides",
                PropValue::Str(alloc::format!("{LUMA_STRIDE},{0},{0}", WIDTH / 2)),
            )
            .expect("three strides");
        let params = g2g_core::AllocationParams::meta_demand(
            g2g_core::meta::MetaRequests::new().request::<g2g_core::meta::PlaneLayout>(),
        );
        parser.configure_allocation(&params);
        parser
            .configure_pipeline(&RawVideoParse::input_caps())
            .expect("a raw byte stream");
        let padded_bytes = parser.stride().expect("a padded frame size");
        let mut sink = RecordingSink::default();
        parser
            .process(data_frame(vec![1u8; padded_bytes]), &mut sink)
            .await
            .expect("the frame parses");
        let frames = sink.frames();
        assert_eq!(
            frames[0].domain.as_system_slice().map(<[u8]>::len),
            Some(padded_bytes),
            "the frame went through as it lies"
        );
        let layout = frames[0]
            .meta
            .get::<g2g_core::meta::PlaneLayout>()
            .expect("the layout is declared");
        assert_eq!(layout.plane(0).expect("plane 0").stride, LUMA_STRIDE);
    }

    #[test]
    fn refuses_a_stride_list_that_does_not_fit_the_format() {
        let mut parser = RawVideoParse::new().with_geometry(WIDTH, HEIGHT);
        // Two strides for a three-plane format.
        parser
            .set_property("plane-strides", PropValue::Str("64,32".into()))
            .expect("the property takes any list");
        assert_eq!(
            parser
                .configure_pipeline(&RawVideoParse::input_caps())
                .err(),
            Some(G2gError::CapsMismatch)
        );
        // A stride narrower than the row it has to hold.
        let mut parser = RawVideoParse::new().with_geometry(WIDTH, HEIGHT);
        parser
            .set_property("plane-strides", PropValue::Str("8,8,8".into()))
            .expect("the property takes any list");
        assert_eq!(
            parser
                .configure_pipeline(&RawVideoParse::input_caps())
                .err(),
            Some(G2gError::CapsMismatch)
        );
        // Not a number at all.
        assert_eq!(
            parser.set_property("plane-offsets", PropValue::Str("0,x".into())),
            Err(PropError::Value)
        );
    }

    #[test]
    fn refuses_a_frame_size_under_the_pixels() {
        let mut parser = RawVideoParse::new().with_geometry(WIDTH, HEIGHT);
        parser
            .set_property("frame-size", PropValue::Uint(FRAME_BYTES as u64 - 1))
            .expect("the property takes any number");
        assert_eq!(
            parser
                .configure_pipeline(&RawVideoParse::input_caps())
                .err(),
            Some(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn refuses_an_unnamed_format_and_an_absurd_geometry() {
        let mut parser = RawVideoParse::new();
        assert_eq!(
            parser.set_property("format", PropValue::Str("GRAY8".into())),
            Err(PropError::Value)
        );
        assert_eq!(
            parser.set_property("width", PropValue::Uint(100_000)),
            Err(PropError::Value)
        );
        assert_eq!(
            parser.set_property("height", PropValue::Uint(0)),
            Err(PropError::Value)
        );
    }
}
