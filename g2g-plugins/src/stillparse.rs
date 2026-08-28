//! Still-image parsers (`jpegparse`, `pngparse`): a JPEG / PNG byte stream in
//! (arbitrary chunks from `filesrc`, or one image per buffer from
//! `multifilesrc`), one whole image per buffer out with the geometry its own
//! header declares.
//!
//! This is what lets a still or an MJPEG dump reach a decoder at all: `MjpegDec`
//! and `PngDec` take one complete image per buffer, and a byte source hands over
//! whatever a read returned. The framing walks the format's own structure
//! ([`crate::stillframe`]), so a file split across reads and several files joined
//! in one buffer both come out as whole images.
//!
//! The geometry comes from the image, not from negotiation, so a `CapsChanged`
//! carries it before the first buffer and again on any mid-stream size change (an
//! image sequence can change size). Timestamps come from the negotiated
//! framerate, since a byte stream carries none; an upstream buffer that does
//! carry a presentation time re-bases the counter onto it.

use core::future::Future;
use core::marker::PhantomData;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_warn, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    Rate, VideoCodec,
};

use crate::compositor::frame_period_ns;
use crate::stillframe::{
    jpeg_frame_length, jpeg_geometry, png_frame_length, png_geometry, FrameLength, ImageAssembler,
};

/// One still-image format's framing and header geometry, the two things the
/// parser needs and the only thing that differs between `jpegparse` and
/// `pngparse`.
pub trait StillImageFormat: Send {
    /// The codec the caps name.
    const CODEC: VideoCodec;
    /// Long name for `gst-inspect`.
    const LONG_NAME: &'static str;
    /// Length of the complete image at the start of a buffer.
    const FRAME_LENGTH: FrameLength;
    /// `(width, height)` from one whole image's header.
    fn geometry(image: &[u8]) -> Option<(u32, u32)>;
}

/// JPEG framing: `SOI` to `EOI`, geometry from the frame header (`SOF`).
#[derive(Debug)]
pub struct Jpeg;

impl StillImageFormat for Jpeg {
    const CODEC: VideoCodec = VideoCodec::Mjpeg;
    const LONG_NAME: &'static str = "JPEG parser";
    const FRAME_LENGTH: FrameLength = jpeg_frame_length;

    fn geometry(image: &[u8]) -> Option<(u32, u32)> {
        jpeg_geometry(image)
    }
}

/// PNG framing: signature to `IEND`, geometry from `IHDR`.
#[derive(Debug)]
pub struct Png;

impl StillImageFormat for Png {
    const CODEC: VideoCodec = VideoCodec::Png;
    const LONG_NAME: &'static str = "PNG parser";
    const FRAME_LENGTH: FrameLength = png_frame_length;

    fn geometry(image: &[u8]) -> Option<(u32, u32)> {
        png_geometry(image)
    }
}

/// Frames a still-image byte stream into whole images and refines their caps.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::stillparse::JpegParse;
///
/// // gst-launch equivalent: filesrc location=frame.jpg ! jpegparse ! mjpegdec ! ...
/// let parser = JpegParse::new();
/// assert_eq!(parser.images_emitted(), 0);
/// ```
#[derive(Debug)]
pub struct StillImageParse<F: StillImageFormat> {
    assembler: ImageAssembler,
    /// The framerate negotiation fixed, which the timestamps step by.
    framerate: Rate,
    /// The geometry of the last image emitted, so caps are re-declared only on a
    /// real change.
    declared: Option<(u32, u32)>,
    /// A presentation time from upstream, applied to the next image emitted.
    pending_pts: Option<u64>,
    base_ns: u64,
    /// Images emitted since the last time base, the presentation-time counter.
    since_base: u64,
    sequence: u64,
    configured: bool,
    log_name: LogName,
    format: PhantomData<F>,
}

/// `jpegparse`: a JPEG / MJPEG byte stream framed into access units.
pub type JpegParse = StillImageParse<Jpeg>;
/// `pngparse`: a PNG byte stream framed into whole images.
pub type PngParse = StillImageParse<Png>;

impl<F: StillImageFormat> Default for StillImageParse<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: StillImageFormat> StillImageParse<F> {
    pub fn new() -> Self {
        Self {
            assembler: ImageAssembler::default(),
            framerate: Rate::Any,
            declared: None,
            pending_pts: None,
            base_ns: 0,
            since_base: 0,
            sequence: 0,
            configured: false,
            log_name: LogName::default(),
            format: PhantomData,
        }
    }

    /// Count of whole images emitted.
    pub fn images_emitted(&self) -> u64 {
        self.sequence
    }

    /// What the element accepts and produces: the format's codec at any
    /// geometry. The output is the same media type, framed, so the caps differ
    /// only once the header's geometry is known.
    fn stream_caps(width: Dim, height: Dim, framerate: Rate) -> Caps {
        Caps::CompressedVideo {
            codec: F::CODEC,
            width,
            height,
            framerate,
        }
    }

    /// Push one whole image, declaring its geometry first when it changed.
    async fn emit(&mut self, image: Vec<u8>, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let (width, height) = F::geometry(&image).ok_or(G2gError::CapsMismatch)?;
        if self.declared != Some((width, height)) {
            out.push(PipelinePacket::CapsChanged(Self::stream_caps(
                Dim::Fixed(width),
                Dim::Fixed(height),
                self.framerate.clone(),
            )))
            .await?;
            self.declared = Some((width, height));
        }
        if let Some(pts) = self.pending_pts.take() {
            self.base_ns = pts;
            self.since_base = 0;
        }
        let period_ns = match self.framerate {
            Rate::Fixed(q16) => frame_period_ns(q16),
            // No rate to step by: every image carries the base time, which is
            // what a single still wants.
            _ => 0,
        };
        let pts_ns = self.base_ns + self.since_base.saturating_mul(period_ns);
        self.since_base += 1;
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(image.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns: period_ns,
                // Every still decodes on its own.
                keyframe: true,
                ..FrameTiming::default()
            },
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

impl<F: StillImageFormat> AsyncElement for StillImageParse<F> {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            F::LONG_NAME,
            "Codec/Parser/Video",
            "Frames a still-image byte stream into whole images and refines their caps",
            "g2g",
        )
    }

    /// Reads host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::stream_caps(Dim::Any, Dim::Any, Rate::Any))
    }

    /// Pass-through identity: the media type is unchanged and the geometry is
    /// refined at runtime via `CapsChanged`, once the header is read.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::CompressedVideo {
            codec, framerate, ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if *codec != F::CODEC {
            return Err(G2gError::CapsMismatch);
        }
        self.framerate = framerate.clone();
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
                    // A byte source stamps every chunk 0, so only a real time
                    // (a demuxer's) re-bases the counter.
                    if let Some(pts) = frame.timing.pts().filter(|pts| *pts != 0) {
                        self.pending_pts = Some(pts);
                    }
                    let images = self.assembler.push(slice, F::FRAME_LENGTH)?;
                    for image in images {
                        self.emit(image, out).await?;
                    }
                }
                PipelinePacket::Eos => {
                    if let Err(e) = self.assembler.finish() {
                        g2g_warn!(self, "stream ends inside an image");
                        return Err(e);
                    }
                }
                PipelinePacket::Flush => {
                    self.assembler.reset();
                    self.pending_pts = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // The refined caps this element emits replace the upstream
                // declaration, whose geometry is a placeholder.
                PipelinePacket::CapsChanged(caps) => {
                    if let Caps::CompressedVideo { framerate, .. } = &caps {
                        self.framerate = framerate.clone();
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }
}

impl<F: StillImageFormat> LogSource for StillImageParse<F> {
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

impl<F: StillImageFormat> PadTemplates for StillImageParse<F> {
    fn pad_templates() -> Vec<PadTemplate> {
        // A still is a one-frame stream down to a single pixel; the geometry is
        // a fixable `Range` placeholder, refined from the header at runtime.
        let caps = CapsSet::one(crate::typefind::still_image_caps(F::CODEC));
        Vec::from([PadTemplate::sink(caps.clone()), PadTemplate::source(caps)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use g2g_core::PushOutcome;

    /// The geometry the test images declare, and the rate negotiation fixes.
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 48;
    const FRAMERATE_FPS: u32 = 25;

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
        fn payloads(&self) -> Vec<Vec<u8>> {
            self.packets
                .iter()
                .filter_map(|p| match p {
                    PipelinePacket::DataFrame(f) => {
                        Some(f.domain.as_system_slice().expect("system").to_vec())
                    }
                    _ => None,
                })
                .collect()
        }

        fn caps(&self) -> Vec<Caps> {
            self.packets
                .iter()
                .filter_map(|p| match p {
                    PipelinePacket::CapsChanged(c) => Some(c.clone()),
                    _ => None,
                })
                .collect()
        }

        fn pts(&self) -> Vec<u64> {
            self.packets
                .iter()
                .filter_map(|p| match p {
                    PipelinePacket::DataFrame(f) => Some(f.timing.pts_ns),
                    _ => None,
                })
                .collect()
        }
    }

    /// A PNG shaped file declaring `width` x `height`, with `filler` bytes of
    /// image data so two of them differ in length.
    fn png_image(width: u32, height: u32, filler: usize) -> Vec<u8> {
        const IHDR_PAYLOAD: u32 = 13;
        let mut file = Vec::from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        file.extend_from_slice(&IHDR_PAYLOAD.to_be_bytes());
        file.extend_from_slice(b"IHDR");
        file.extend_from_slice(&width.to_be_bytes());
        file.extend_from_slice(&height.to_be_bytes());
        file.extend_from_slice(&[8, 6, 0, 0, 0]);
        file.extend_from_slice(&[0u8; 4]);
        file.extend_from_slice(&(filler as u32).to_be_bytes());
        file.extend_from_slice(b"IDAT");
        file.extend_from_slice(&vec![0xAB; filler]);
        file.extend_from_slice(&[0u8; 4]);
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(b"IEND");
        file.extend_from_slice(&[0u8; 4]);
        file
    }

    fn png_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Png,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(FRAMERATE_FPS << 16),
        }
    }

    fn parser() -> PngParse {
        let mut parser = PngParse::new();
        parser
            .configure_pipeline(&png_caps())
            .expect("PNG at a fixed rate");
        parser
    }

    fn data_frame(bytes: &[u8], pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
            FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            0,
        ))
    }

    #[tokio::test]
    async fn rejoins_images_split_across_buffers() {
        let images = [png_image(WIDTH, HEIGHT, 30), png_image(WIDTH, HEIGHT, 41)];
        let stream: Vec<u8> = images.concat();
        let mut parser = parser();
        let mut sink = RecordingSink::default();
        // An odd chunk size, so both images straddle buffer boundaries.
        for piece in stream.chunks(17) {
            parser
                .process(data_frame(piece, 0), &mut sink)
                .await
                .expect("the chunk parses");
        }
        parser
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("the stream ends on an image boundary");
        assert_eq!(sink.payloads(), images.to_vec());
        assert_eq!(parser.images_emitted(), 2);
    }

    #[tokio::test]
    async fn declares_the_header_geometry_once_and_again_on_a_change() {
        let mut parser = parser();
        let mut sink = RecordingSink::default();
        for image in [
            png_image(WIDTH, HEIGHT, 8),
            png_image(WIDTH, HEIGHT, 8),
            png_image(WIDTH / 2, HEIGHT / 2, 8),
        ] {
            parser
                .process(data_frame(&image, 0), &mut sink)
                .await
                .expect("the image parses");
        }
        let rate = Rate::Fixed(FRAMERATE_FPS << 16);
        assert_eq!(
            sink.caps(),
            vec![
                StillImageParse::<Png>::stream_caps(
                    Dim::Fixed(WIDTH),
                    Dim::Fixed(HEIGHT),
                    rate.clone()
                ),
                StillImageParse::<Png>::stream_caps(
                    Dim::Fixed(WIDTH / 2),
                    Dim::Fixed(HEIGHT / 2),
                    rate
                ),
            ]
        );
    }

    #[tokio::test]
    async fn timestamps_step_by_the_negotiated_rate_and_rebase_on_upstream() {
        let image = png_image(WIDTH, HEIGHT, 8);
        let period_ns = 1_000_000_000 / u64::from(FRAMERATE_FPS);
        let mut parser = parser();
        let mut sink = RecordingSink::default();
        for _ in 0..3 {
            parser
                .process(data_frame(&image, 0), &mut sink)
                .await
                .expect("the image parses");
        }
        const DEMUXER_PTS_NS: u64 = 5_000_000_000;
        parser
            .process(data_frame(&image, DEMUXER_PTS_NS), &mut sink)
            .await
            .expect("the image parses");
        assert_eq!(
            sink.pts(),
            vec![0, period_ns, 2 * period_ns, DEMUXER_PTS_NS]
        );
    }

    #[tokio::test]
    async fn a_truncated_image_fails_the_stream() {
        let image = png_image(WIDTH, HEIGHT, 8);
        let mut parser = parser();
        let mut sink = RecordingSink::default();
        parser
            .process(data_frame(&image[..image.len() / 2], 0), &mut sink)
            .await
            .expect("a partial image is not an error yet");
        assert_eq!(
            parser.process(PipelinePacket::Eos, &mut sink).await.err(),
            Some(G2gError::CapsMismatch),
            "the half image is reported, not silently dropped"
        );
    }

    #[tokio::test]
    async fn a_jpeg_stream_frames_into_access_units() {
        // Two minimal baseline JPEGs back to back, the MJPEG dump case.
        let image = jpeg_image(WIDTH as u16, HEIGHT as u16);
        let mut parser = JpegParse::new();
        parser
            .configure_pipeline(&Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                width: Dim::Fixed(WIDTH),
                height: Dim::Fixed(HEIGHT),
                framerate: Rate::Fixed(FRAMERATE_FPS << 16),
            })
            .expect("JPEG at a fixed rate");
        let mut sink = RecordingSink::default();
        let mut stream = image.clone();
        stream.extend_from_slice(&image);
        parser
            .process(data_frame(&stream, 0), &mut sink)
            .await
            .expect("the buffer parses");
        assert_eq!(sink.payloads(), vec![image.clone(), image]);
        assert_eq!(
            sink.caps(),
            vec![StillImageParse::<Jpeg>::stream_caps(
                Dim::Fixed(WIDTH),
                Dim::Fixed(HEIGHT),
                Rate::Fixed(FRAMERATE_FPS << 16)
            )]
        );
    }

    /// A baseline-JPEG shaped file: `SOI`, `SOF0` at `width` x `height`, `SOS`
    /// with a few entropy bytes, `EOI`.
    fn jpeg_image(width: u16, height: u16) -> Vec<u8> {
        let mut file = vec![0xFF, 0xD8];
        file.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x0B, 0x08]);
        file.extend_from_slice(&height.to_be_bytes());
        file.extend_from_slice(&width.to_be_bytes());
        file.extend_from_slice(&[0x01, 0x01, 0x11, 0x00]);
        file.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]);
        file.extend_from_slice(&[0x37; 12]);
        file.extend_from_slice(&[0xFF, 0xD9]);
        file
    }
}
