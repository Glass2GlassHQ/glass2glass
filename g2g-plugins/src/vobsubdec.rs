//! VobSub (DVD subpicture) decoder element (M899): `Caps::SubPicture{VobSub}` in,
//! `Caps::RawVideo{Rgba8}` canvases out.
//!
//! Each cue becomes two frames on a fully transparent canvas the size of the
//! subpicture display area (720x576 unless the `.idx` says otherwise): the cue
//! painted at its display rectangle, stamped with the cue's PTS and duration,
//! then an all-transparent canvas at the cue's hide time. Downstream is an
//! ordinary compositor input, which holds the last frame it received on that
//! pad, so the clear frame is what makes the cue disappear on time. The stream
//! opens on one more empty canvas, so a compositor is not waiting on this input
//! for however long it is until the first cue.
//!
//! ```text
//! filesrc ! mkvdemux stream=vobsub ! vobsubdec ! compositor.
//! ```
//!
//! The palette and display geometry are not in the bitstream. They arrive as the
//! Matroska track's `CodecPrivate`, the `.idx` text, which the demuxer forwards
//! in band ahead of the first cue; the decoder tells it from a cue by parsing it
//! as `.idx` first (see [`crate::vobsub::parse_idx`]). Without one, the DVD
//! default palette and 720x576 stand.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, SubPictureFormat,
};

use crate::vobsub::{parse_idx, parse_spu};

/// PAL DVD subpicture geometry, the near-universal VobSub display size and what
/// stands until an `.idx` names another.
const DEFAULT_WIDTH: u32 = 720;
const DEFAULT_HEIGHT: u32 = 576;

/// Fallback palette for a stream whose `.idx` carries none: the greyscale ramp a
/// DVD player would show, so an unconfigured cue is legible rather than invisible.
const DEFAULT_PALETTE: [u32; 16] = [
    0x000000, 0xffffff, 0x808080, 0x000000, 0xbfbfbf, 0x404040, 0x202020, 0xe0e0e0, 0x606060,
    0xa0a0a0, 0x101010, 0xf0f0f0, 0x303030, 0xd0d0d0, 0x505050, 0x909090,
];

/// Decodes VobSub cues into full-frame RGBA canvases.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::vobsubdec::VobSubDec;
///
/// let decoder = VobSubDec::new().with_size(720, 576).with_framerate(25);
/// ```
#[derive(Debug)]
pub struct VobSubDec {
    width: u32,
    height: u32,
    framerate_q16: u32,
    palette: [u32; 16],
    configured: bool,
    emitted: u64,
    /// Whether the opening empty canvas has gone out (see [`prime`](Self::prime)).
    primed: bool,
    /// The output caps last announced, so a geometry refinement from the `.idx`
    /// emits one `CapsChanged` and no more.
    last_caps: Option<Caps>,
}

impl Default for VobSubDec {
    fn default() -> Self {
        Self::new()
    }
}

impl VobSubDec {
    pub fn new() -> Self {
        Self {
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            // Nominal: cues are sparse and carry their own PTS and duration, so
            // this only labels the output caps for downstream negotiation.
            framerate_q16: 25 << 16,
            palette: DEFAULT_PALETTE,
            configured: false,
            emitted: 0,
            primed: false,
            last_caps: None,
        }
    }

    /// Set the subpicture display geometry the canvases are produced at. An
    /// `.idx` `size:` line overrides this once it arrives.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the nominal output framerate in fps (labels the output caps only).
    pub fn with_framerate(mut self, fps: u32) -> Self {
        self.framerate_q16 = fps << 16;
        self
    }

    /// Canvases emitted so far (two per cue that carries a hide time).
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn input_caps() -> Caps {
        Caps::SubPicture {
            format: SubPictureFormat::VobSub,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(self.framerate_q16),
            interlace: g2g_core::Interlace::Any,
        }
    }

    fn canvas_frame(&mut self, pixels: Vec<u8>, timing: FrameTiming) -> Frame {
        let seq = self.emitted;
        self.emitted += 1;
        Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
            timing,
            seq,
        )
    }

    /// Adopt the `.idx` configuration a `CodecPrivate` blob carries. `None` when
    /// the bytes are not `.idx` text (so they are a cue), else whether the
    /// geometry changed, which the caller announces.
    fn apply_idx(&mut self, bytes: &[u8]) -> Option<bool> {
        let cfg = parse_idx(bytes)?;
        if let Some(palette) = cfg.palette {
            self.palette = palette;
        }
        Some(match cfg.size {
            Some((w, h)) if w > 0 && h > 0 && (w, h) != (self.width, self.height) => {
                self.width = w;
                self.height = h;
                true
            }
            _ => false,
        })
    }

    /// Emit the empty canvas the stream starts on, once, before the first cue.
    /// A zero-order-hold consumer needs a frame to represent "no subtitle yet",
    /// and a compositor waits for every overlay input before it releases output,
    /// so without this the video stalls until the first cue (which can be
    /// minutes into a film).
    async fn prime(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.primed {
            return Ok(());
        }
        self.primed = true;
        let bytes = (self.width as usize) * (self.height as usize) * 4;
        let frame = self.canvas_frame(vec![0u8; bytes], FrameTiming::default());
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    /// Decode one SPU packet into its canvas frames: the painted cue, then a
    /// clear canvas at its hide time when the control sequence gave one.
    async fn decode_cue(
        &mut self,
        data: &[u8],
        timing: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let Some(cue) = parse_spu(data) else {
            // A malformed cue is dropped: nothing downstream can render it, and
            // the stream continues at the next packet.
            return Ok(());
        };
        let bytes = (self.width as usize) * (self.height as usize) * 4;
        let mut canvas = vec![0u8; bytes];
        cue.paint(&self.palette, &mut canvas, self.width, self.height);

        let start = timing.pts_ns.saturating_add(cue.start_ns);
        let duration = cue
            .stop_ns
            .map(|stop| stop.saturating_sub(cue.start_ns))
            .unwrap_or(0);
        let shown = FrameTiming {
            pts_ns: start,
            dts_ns: start,
            duration_ns: duration,
            ..timing
        };
        let frame = self.canvas_frame(canvas, shown);
        out.push(PipelinePacket::DataFrame(frame)).await?;

        // The clear canvas is what a zero-order-hold consumer needs to stop
        // showing the cue; without a hide time the cue stands until the next one.
        if cue.stop_ns.is_some() {
            let end = start.saturating_add(duration);
            let cleared = FrameTiming {
                pts_ns: end,
                dts_ns: end,
                duration_ns: 0,
                ..timing
            };
            let frame = self.canvas_frame(vec![0u8; bytes], cleared);
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for VobSubDec {
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "VobSub subpicture decoder",
            "Codec/Decoder/Subtitle",
            "Decodes DVD subpicture cues into RGBA overlay canvases",
            "g2g",
        )
    }
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let output = self.output_caps();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::SubPicture {
                format: SubPictureFormat::VobSub,
            } => CapsSet::one(output.clone()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if absolute_caps != &Self::input_caps() {
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
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    // The `.idx` config and the cues share one pad; the config is
                    // the blob that parses as `.idx` text.
                    if let Some(resized) = self.apply_idx(slice) {
                        if resized {
                            let caps = self.output_caps();
                            self.last_caps = Some(caps.clone());
                            out.push(PipelinePacket::CapsChanged(caps)).await?;
                        }
                        return self.prime(out).await;
                    }
                    let timing = frame.timing;
                    let data = slice.to_vec();
                    self.prime(out).await?;
                    self.decode_cue(&data, timing, out).await?;
                }
                // The runner feeds back our own solved output caps; accept them
                // and re-announce only what we have not already sent.
                PipelinePacket::CapsChanged(caps) => {
                    if self.last_caps.as_ref() != Some(&caps) {
                        let ours = self.output_caps();
                        self.last_caps = Some(ours.clone());
                        out.push(PipelinePacket::CapsChanged(ours)).await?;
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        VOBSUBDEC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let v = value.as_uint().ok_or(PropError::Type)?;
        let v = u32::try_from(v).map_err(|_| PropError::Value)?;
        match name {
            "width" | "height" if v == 0 || v > crate::vobsub::MAX_CUE_DIM => Err(PropError::Value),
            "width" => {
                self.width = v;
                Ok(())
            }
            "height" => {
                self.height = v;
                Ok(())
            }
            "framerate" if v == 0 || v > 1000 => Err(PropError::Value),
            "framerate" => {
                self.framerate_q16 = v << 16;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        Some(match name {
            "width" => PropValue::Uint(self.width as u64),
            "height" => PropValue::Uint(self.height as u64),
            "framerate" => PropValue::Uint((self.framerate_q16 >> 16) as u64),
            _ => return None,
        })
    }
}

static VOBSUBDEC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "width",
        PropKind::Uint,
        "subpicture display width; an .idx `size:` line overrides it",
    )
    .with_default("720"),
    PropertySpec::new(
        "height",
        PropKind::Uint,
        "subpicture display height; an .idx `size:` line overrides it",
    )
    .with_default("576"),
    PropertySpec::new(
        "framerate",
        PropKind::Uint,
        "nominal output framerate in fps (labels the caps; cues stay sparse)",
    )
    .with_default("25"),
];

impl PadTemplates for VobSubDec {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(VobSubDec::input_caps())),
            PadTemplate::source(CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(DEFAULT_WIDTH),
                height: Dim::Fixed(DEFAULT_HEIGHT),
                framerate: Rate::Fixed(25 << 16),
                interlace: g2g_core::Interlace::Any,
            })),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::PushOutcome;

    /// The `.idx` text a Matroska `S_VOBSUB` `CodecPrivate` carries.
    const IDX: &[u8] = b"size: 32x16\npalette: 000000, ff0000, 00ff00, 0000ff, 000000, 000000, 000000, 000000, 000000, 000000, 000000, 000000, 000000, 000000, 000000, 000000\n";

    #[derive(Default)]
    struct CollectSink {
        packets: Vec<PipelinePacket>,
    }
    impl OutputSink for CollectSink {
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

    /// A one-cue SPU: a solid `w` x `h` block of sample value 1 at (`x`, `y`),
    /// shown immediately and hidden at `stop_date` control-sequence units.
    fn spu(x: u32, y: u32, w: u32, h: u32, stop_date: u16) -> Vec<u8> {
        let line = [0x00u8, 0x01];
        let mut top = Vec::new();
        for _ in 0..h.div_ceil(2) {
            top.extend_from_slice(&line);
        }
        let mut bottom = Vec::new();
        for _ in 0..h / 2 {
            bottom.extend_from_slice(&line);
        }
        let (top_off, bottom_off) = (4usize, 4 + top.len());
        let data_end = bottom_off + bottom.len();
        let (x2, y2) = (x + w - 1, y + h - 1);
        let mut show = vec![
            0x03,
            0x32,
            0x10,
            0x04,
            0xff,
            0xf0,
            0x05,
            (x >> 4) as u8,
            (((x & 0xf) << 4) | (x2 >> 8)) as u8,
            x2 as u8,
            (y >> 4) as u8,
            (((y & 0xf) << 4) | (y2 >> 8)) as u8,
            y2 as u8,
            0x06,
        ];
        show.extend_from_slice(&(top_off as u16).to_be_bytes());
        show.extend_from_slice(&(bottom_off as u16).to_be_bytes());
        show.extend_from_slice(&[0x01, 0xff]);
        let seq2 = data_end + 4 + show.len();
        let total = seq2 + 6;

        let mut out = Vec::new();
        out.extend_from_slice(&(total as u16).to_be_bytes());
        out.extend_from_slice(&(data_end as u16).to_be_bytes());
        out.extend_from_slice(&top);
        out.extend_from_slice(&bottom);
        out.extend_from_slice(&0u16.to_be_bytes());
        out.extend_from_slice(&(seq2 as u16).to_be_bytes());
        out.extend_from_slice(&show);
        out.extend_from_slice(&stop_date.to_be_bytes());
        out.extend_from_slice(&(seq2 as u16).to_be_bytes());
        out.extend_from_slice(&[0x02, 0xff]);
        out
    }

    fn frame(bytes: &[u8], pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                ..FrameTiming::default()
            },
            0,
        ))
    }

    async fn run(dec: &mut VobSubDec, packets: Vec<PipelinePacket>) -> Vec<PipelinePacket> {
        let mut sink = CollectSink::default();
        for p in packets {
            dec.process(p, &mut sink).await.unwrap();
        }
        sink.packets
    }

    #[tokio::test]
    async fn a_cue_emits_a_painted_canvas_then_a_clear_one() {
        let mut dec = VobSubDec::new();
        dec.configure_pipeline(&VobSubDec::input_caps()).unwrap();
        let out = run(
            &mut dec,
            vec![
                frame(IDX, 0),
                // 180 control units = 2.048 s
                frame(&spu(4, 2, 8, 4, 180), 1_000_000_000),
            ],
        )
        .await;

        let mut caps = Vec::new();
        let mut frames = Vec::new();
        for p in out {
            match p {
                PipelinePacket::CapsChanged(c) => caps.push(c),
                PipelinePacket::DataFrame(f) => frames.push(f),
                _ => {}
            }
        }
        assert_eq!(
            caps,
            vec![Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(32),
                height: Dim::Fixed(16),
                framerate: Rate::Fixed(25 << 16),
                interlace: g2g_core::Interlace::Any,
            }],
            "the .idx size: line refines the output geometry once"
        );
        assert_eq!(
            frames.len(),
            3,
            "the opening empty canvas, the painted cue, then the clear canvas"
        );
        assert!(
            frames[0]
                .domain
                .as_system_slice()
                .unwrap()
                .iter()
                .all(|&b| b == 0),
            "the stream opens on an empty canvas"
        );
        let frames = &frames[1..];

        let painted = frames[0].domain.as_system_slice().unwrap();
        assert_eq!(painted.len(), 32 * 16 * 4);
        assert_eq!(frames[0].timing.pts_ns, 1_000_000_000);
        assert_eq!(frames[0].timing.duration_ns, 2_048_000_000);
        // sample value 1 maps through colormap[1] = 1 to the palette's red
        let px = |x: usize, y: usize| &painted[(y * 32 + x) * 4..(y * 32 + x) * 4 + 4];
        assert_eq!(px(4, 2), [255, 0, 0, 255]);
        assert_eq!(px(11, 5), [255, 0, 0, 255]);
        assert_eq!(px(3, 2), [0, 0, 0, 0]);
        assert_eq!(px(12, 5), [0, 0, 0, 0]);

        let cleared = frames[1].domain.as_system_slice().unwrap();
        assert!(
            cleared.iter().all(|&b| b == 0),
            "the clear canvas is fully transparent"
        );
        assert_eq!(frames[1].timing.pts_ns, 3_048_000_000);
    }

    #[tokio::test]
    async fn a_malformed_cue_is_dropped_without_failing_the_stream() {
        let mut dec = VobSubDec::new();
        dec.configure_pipeline(&VobSubDec::input_caps()).unwrap();
        let mut bad = spu(0, 0, 8, 4, 90);
        bad[2..4].copy_from_slice(&1u16.to_be_bytes()); // control offset inside the header
        let out = run(&mut dec, vec![frame(IDX, 0), frame(&bad, 0)]).await;
        let frames = out
            .iter()
            .filter(|p| matches!(p, PipelinePacket::DataFrame(_)))
            .count();
        assert_eq!(frames, 1, "only the opening empty canvas, no cue");
    }

    #[test]
    fn rejects_input_that_is_not_a_vobsub_stream() {
        let mut dec = VobSubDec::new();
        assert!(dec.configure_pipeline(&Caps::Klv).is_err());
    }

    #[test]
    fn properties_round_trip() {
        let mut dec = VobSubDec::new();
        for (name, value) in [("width", 720u64), ("height", 480), ("framerate", 30)] {
            dec.set_property(name, PropValue::Uint(value)).unwrap();
            assert_eq!(dec.get_property(name), Some(PropValue::Uint(value)));
        }
        assert!(dec.set_property("width", PropValue::Uint(0)).is_err());
        assert!(dec
            .set_property("framerate", PropValue::Uint(5000))
            .is_err());
    }
}
