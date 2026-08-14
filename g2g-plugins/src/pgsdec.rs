//! Blu-ray PGS subtitle decoder element (M925): `Caps::SubPicture{Pgs}` in,
//! `Caps::RawVideo{Rgba8}` canvases out.
//!
//! Each display set becomes a full-frame transparent canvas at the video
//! geometry the presentation composition declares (1920x1080 until one arrives),
//! stamped with the display set's PTS. PGS has no end-of-display time: a cue
//! stands until a later display set replaces it, and a presentation composition
//! listing no object is how the stream says the cue is over. That empty display
//! set is what produces the all-transparent clear canvas, so the clear-frame
//! contract [`VobSubDec`](crate::vobsubdec::VobSubDec) and
//! [`DvbSubDec`](crate::dvbsubdec::DvbSubDec) offer holds here too, with the
//! stream supplying the clear rather than the decoder synthesizing one from a
//! hide time. Downstream is an ordinary compositor input, which holds the last
//! frame it received on that pad. The stream opens on one more empty canvas, so
//! a compositor is not waiting on this input for however long it is until the
//! first cue.
//!
//! ```text
//! filesrc ! mkvdemux stream=pgs ! pgsdec ! compositor.
//! ```
//!
//! Unlike the other two bitmap-subtitle codings there is no out-of-band
//! configuration: the palette is in band and the geometry is in the presentation
//! composition, so every input frame is a display set.

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

use crate::pgs::{PgsDecoder, DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH, MAX_VIDEO_DIM};

/// Decodes PGS display sets into full-frame RGBA canvases.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::pgsdec::PgsDec;
///
/// let dec = PgsDec::new()
///     .with_size(1920, 1080)
///     .with_forced_only(true);
/// ```
#[derive(Debug)]
pub struct PgsDec {
    dec: PgsDecoder,
    width: u32,
    height: u32,
    framerate_q16: u32,
    forced_only: bool,
    configured: bool,
    emitted: u64,
    /// Whether the opening empty canvas has gone out (see [`prime`](Self::prime)).
    primed: bool,
    /// Whether a cue is on screen, so a run of empty display sets does not emit
    /// a clear canvas per display set.
    showing: bool,
    /// The output caps last announced, so a geometry refinement from the
    /// presentation composition emits one `CapsChanged` and no more.
    last_caps: Option<Caps>,
}

impl Default for PgsDec {
    fn default() -> Self {
        Self::new()
    }
}

impl PgsDec {
    pub fn new() -> Self {
        Self {
            dec: PgsDecoder::new(),
            width: DEFAULT_VIDEO_WIDTH,
            height: DEFAULT_VIDEO_HEIGHT,
            // Nominal: display sets are sparse and carry their own PTS, so this
            // only labels the output caps for downstream negotiation.
            framerate_q16: 25 << 16,
            forced_only: false,
            configured: false,
            emitted: 0,
            primed: false,
            showing: false,
            last_caps: None,
        }
    }

    /// Set the video geometry the canvases are produced at. A presentation
    /// composition segment overrides this once one arrives.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self.dec.set_video_size(width, height);
        self
    }

    /// Set the nominal output framerate in fps (labels the output caps only).
    pub fn with_framerate(mut self, fps: u32) -> Self {
        self.framerate_q16 = fps << 16;
        self
    }

    /// Paint only the cues marked forced, dropping the ordinary subtitle track.
    pub fn with_forced_only(mut self, forced_only: bool) -> Self {
        self.forced_only = forced_only;
        self.dec.set_forced_only(forced_only);
        self
    }

    /// Canvases emitted so far.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn input_caps() -> Caps {
        Caps::SubPicture {
            format: SubPictureFormat::Pgs,
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

    /// Emit the empty canvas the stream starts on, once, at the first display
    /// set (which is what fixes the geometry, so it cannot go out before then).
    /// A zero-order-hold consumer needs a frame to represent "no subtitle yet",
    /// and a compositor waits for every overlay input before it releases output,
    /// so without this the video stalls until the first cue.
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

    /// Decode one buffer of segments into its canvas frames. A buffer normally
    /// holds one display set (a Matroska block); a `.sup` byte stream holds many,
    /// and the later ones are placed by their own PTS relative to the first.
    async fn decode_buffer(
        &mut self,
        data: &[u8],
        timing: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let sets = self.dec.feed(data);
        let base_pts_90k = sets.first().and_then(|s| s.pts_90k);
        for set in sets {
            if (set.width, set.height) != (self.width, self.height) {
                self.width = set.width;
                self.height = set.height;
                let caps = self.output_caps();
                self.last_caps = Some(caps.clone());
                out.push(PipelinePacket::CapsChanged(caps)).await?;
            }
            self.prime(out).await?;
            // An empty display set clears whatever is on screen; a run of them
            // (or one before any cue) owes nothing.
            if !set.visible && !self.showing {
                continue;
            }
            let offset_ns = match (base_pts_90k, set.pts_90k) {
                (Some(base), Some(pts)) => {
                    (pts.saturating_sub(base) as u64) * 1_000_000_000 / 90_000
                }
                _ => 0,
            };
            let pts_ns = timing.pts_ns.saturating_add(offset_ns);
            let shown = FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                // PGS states no end time: the cue stands until the display set
                // that replaces it.
                duration_ns: 0,
                ..timing
            };
            self.showing = set.visible;
            let frame = self.canvas_frame(set.canvas, shown);
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for PgsDec {
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "PGS subtitle decoder",
            "Codec/Decoder/Subtitle",
            "Decodes Blu-ray HDMV presentation graphics into RGBA overlay canvases",
            "g2g",
        )
    }
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let output = self.output_caps();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::SubPicture {
                format: SubPictureFormat::Pgs,
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
                    let timing = frame.timing;
                    let data = slice.to_vec();
                    self.decode_buffer(&data, timing, out).await?;
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
        PGSDEC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if name == "forced-subs-only" {
            self.forced_only = value.as_bool().ok_or(PropError::Type)?;
            self.dec.set_forced_only(self.forced_only);
            return Ok(());
        }
        let v = value.as_uint().ok_or(PropError::Type)?;
        let v = u32::try_from(v).map_err(|_| PropError::Value)?;
        match name {
            "width" | "height" if v == 0 || v > MAX_VIDEO_DIM => Err(PropError::Value),
            "width" => {
                self.width = v;
                self.dec.set_video_size(self.width, self.height);
                Ok(())
            }
            "height" => {
                self.height = v;
                self.dec.set_video_size(self.width, self.height);
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
            "forced-subs-only" => PropValue::Bool(self.forced_only),
            _ => return None,
        })
    }
}

static PGSDEC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "width",
        PropKind::Uint,
        "video width; a presentation composition segment overrides it",
    )
    .with_default("1920"),
    PropertySpec::new(
        "height",
        PropKind::Uint,
        "video height; a presentation composition segment overrides it",
    )
    .with_default("1080"),
    PropertySpec::new(
        "framerate",
        PropKind::Uint,
        "nominal output framerate in fps (labels the caps; cues stay sparse)",
    )
    .with_default("25"),
    PropertySpec::new(
        "forced-subs-only",
        PropKind::Bool,
        "paint only cues marked forced, dropping the ordinary subtitle track",
    )
    .with_default("false"),
];

impl PadTemplates for PgsDec {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(PgsDec::input_caps())),
            PadTemplate::source(CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(DEFAULT_VIDEO_WIDTH),
                height: Dim::Fixed(DEFAULT_VIDEO_HEIGHT),
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

    fn segment(kind: u8, body: &[u8]) -> Vec<u8> {
        let mut out = Vec::from([kind]);
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(body);
        out
    }

    /// A display set showing a 4x2 white block at (2, 1) on a 32x16 video.
    fn cue() -> Vec<u8> {
        let mut pcs = Vec::new();
        pcs.extend_from_slice(&32u16.to_be_bytes());
        pcs.extend_from_slice(&16u16.to_be_bytes());
        pcs.extend_from_slice(&[0x10, 0x00, 0x00, 0x80, 0x00, 0x00, 0x01]);
        pcs.extend_from_slice(&1u16.to_be_bytes());
        pcs.extend_from_slice(&[0x00, 0x00]);
        pcs.extend_from_slice(&2u16.to_be_bytes());
        pcs.extend_from_slice(&1u16.to_be_bytes());

        let mut rle = Vec::new();
        for _ in 0..2 {
            rle.extend_from_slice(&[0x00, 0x84, 0x01, 0x00, 0x00]);
        }
        let mut ods = Vec::new();
        ods.extend_from_slice(&1u16.to_be_bytes());
        ods.extend_from_slice(&[0x00, 0xC0]);
        ods.extend_from_slice(&((rle.len() + 4) as u32).to_be_bytes()[1..]);
        ods.extend_from_slice(&4u16.to_be_bytes());
        ods.extend_from_slice(&2u16.to_be_bytes());
        ods.extend_from_slice(&rle);

        let mut out = Vec::new();
        out.extend_from_slice(&segment(0x16, &pcs));
        out.extend_from_slice(&segment(0x14, &[0x00, 0x00, 0x01, 235, 128, 128, 255]));
        out.extend_from_slice(&segment(0x15, &ods));
        out.extend_from_slice(&segment(0x80, &[]));
        out
    }

    /// The empty display set that ends a cue.
    fn clear() -> Vec<u8> {
        let mut pcs = Vec::new();
        pcs.extend_from_slice(&32u16.to_be_bytes());
        pcs.extend_from_slice(&16u16.to_be_bytes());
        pcs.extend_from_slice(&[0x10, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00]);
        let mut out = segment(0x16, &pcs);
        out.extend_from_slice(&segment(0x80, &[]));
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

    async fn run(dec: &mut PgsDec, packets: Vec<PipelinePacket>) -> Vec<PipelinePacket> {
        let mut sink = CollectSink::default();
        for p in packets {
            dec.process(p, &mut sink).await.unwrap();
        }
        sink.packets
    }

    #[tokio::test]
    async fn a_cue_and_its_empty_display_set_bracket_the_canvas() {
        let mut dec = PgsDec::new();
        dec.configure_pipeline(&PgsDec::input_caps()).unwrap();
        let out = run(
            &mut dec,
            vec![frame(&cue(), 1_000_000_000), frame(&clear(), 3_000_000_000)],
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
            "the presentation composition refines the output geometry once"
        );
        assert_eq!(
            frames.len(),
            3,
            "the opening empty canvas, the painted cue, then the clear canvas"
        );
        assert!(frames[0]
            .domain
            .as_system_slice()
            .unwrap()
            .iter()
            .all(|&b| b == 0));

        let painted = frames[1].domain.as_system_slice().unwrap();
        assert_eq!(painted.len(), 32 * 16 * 4);
        assert_eq!(frames[1].timing.pts_ns, 1_000_000_000);
        let px = |x: usize, y: usize| &painted[(y * 32 + x) * 4..(y * 32 + x) * 4 + 4];
        assert_eq!(px(2, 1), [255, 255, 255, 255]);
        assert_eq!(px(5, 2), [255, 255, 255, 255]);
        assert_eq!(px(1, 1), [0, 0, 0, 0]);
        assert_eq!(px(6, 2), [0, 0, 0, 0]);

        assert!(frames[2]
            .domain
            .as_system_slice()
            .unwrap()
            .iter()
            .all(|&b| b == 0));
        assert_eq!(frames[2].timing.pts_ns, 3_000_000_000);
    }

    #[tokio::test]
    async fn an_empty_display_set_with_nothing_on_screen_emits_nothing() {
        let mut dec = PgsDec::new();
        dec.configure_pipeline(&PgsDec::input_caps()).unwrap();
        let out = run(&mut dec, vec![frame(&clear(), 0), frame(&clear(), 1)]).await;
        let frames = out
            .iter()
            .filter(|p| matches!(p, PipelinePacket::DataFrame(_)))
            .count();
        assert_eq!(frames, 1, "only the opening empty canvas");
    }

    #[test]
    fn rejects_input_that_is_not_a_pgs_stream() {
        let mut dec = PgsDec::new();
        assert!(dec.configure_pipeline(&Caps::Klv).is_err());
    }

    #[test]
    fn properties_round_trip() {
        let mut dec = PgsDec::new();
        for (name, value) in [("width", 1280u64), ("height", 720), ("framerate", 30)] {
            dec.set_property(name, PropValue::Uint(value)).unwrap();
            assert_eq!(dec.get_property(name), Some(PropValue::Uint(value)));
        }
        dec.set_property("forced-subs-only", PropValue::Bool(true))
            .unwrap();
        assert_eq!(
            dec.get_property("forced-subs-only"),
            Some(PropValue::Bool(true))
        );
        assert!(dec.set_property("width", PropValue::Uint(0)).is_err());
        assert!(dec.set_property("height", PropValue::Uint(99_999)).is_err());
    }
}
