//! DVB subtitle decoder element (M900): `Caps::SubPicture{DvbSub}` in,
//! `Caps::RawVideo{Rgba8}` canvases out.
//!
//! Each display set becomes a full-frame transparent canvas the size of the DVB
//! display (720x576 unless a display definition segment says otherwise), stamped
//! with the display set's PTS. A page composition listing no region, or a page
//! whose `page_time_out` runs out before the next display set, produces the
//! all-transparent clear canvas instead. Downstream is an ordinary compositor
//! input, which holds the last frame it received on that pad, so the clear frame
//! is what makes the cue disappear on time. The stream opens on one more empty
//! canvas, so a compositor is not waiting on this input for however long it is
//! until the first cue. This is the same contract
//! [`VobSubDec`](crate::vobsubdec::VobSubDec) offers.
//!
//! ```text
//! filesrc ! tsdemux stream=dvbsub ! dvbsubdec ! compositor.
//! filesrc ! mkvdemux stream=dvbsub ! dvbsubdec ! compositor.
//! ```
//!
//! The palette is in band (CLUT definition segments), but the composition and
//! ancillary page ids are not: they arrive as the five-byte blob the demuxer
//! forwards ahead of the first display set, built from the Matroska track's
//! `CodecPrivate` or the PMT `subtitling_descriptor` (see
//! [`crate::dvbsub::parse_page_ids`]). Without one, the first page the stream
//! composes is followed.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat, SubPictureFormat,
};

use crate::dvbsub::{
    parse_page_ids, DvbSubDecoder, PageIds, DEFAULT_DISPLAY_HEIGHT, DEFAULT_DISPLAY_WIDTH,
    MAX_DISPLAY_DIM,
};

/// Decodes DVB subtitle display sets into full-frame RGBA canvases.
#[derive(Debug)]
pub struct DvbSubDec {
    dec: DvbSubDecoder,
    width: u32,
    height: u32,
    framerate_q16: u32,
    /// The composition page pinned by the `page-id` property; `None` follows the
    /// first page the stream composes (and any page-id blob the demuxer sends).
    pinned_page: Option<u16>,
    configured: bool,
    emitted: u64,
    /// Whether the opening empty canvas has gone out (see [`prime`](Self::prime)).
    primed: bool,
    /// When the page on screen must be cleared if no further display set arrives:
    /// its PTS plus the page's `page_time_out`.
    deadline: Option<u64>,
    /// The output caps last announced, so a geometry refinement from a display
    /// definition segment emits one `CapsChanged` and no more.
    last_caps: Option<Caps>,
}

impl Default for DvbSubDec {
    fn default() -> Self {
        Self::new()
    }
}

impl DvbSubDec {
    pub fn new() -> Self {
        Self {
            dec: DvbSubDecoder::new(),
            width: DEFAULT_DISPLAY_WIDTH,
            height: DEFAULT_DISPLAY_HEIGHT,
            // Nominal: display sets are sparse and carry their own PTS, so this
            // only labels the output caps for downstream negotiation.
            framerate_q16: 25 << 16,
            pinned_page: None,
            configured: false,
            emitted: 0,
            primed: false,
            deadline: None,
            last_caps: None,
        }
    }

    /// Set the display geometry the canvases are produced at. A display
    /// definition segment overrides this once one arrives.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self.dec.set_display_size(width, height);
        self
    }

    /// Set the nominal output framerate in fps (labels the output caps only).
    pub fn with_framerate(mut self, fps: u32) -> Self {
        self.framerate_q16 = fps << 16;
        self
    }

    /// Compose this composition page rather than the first one the stream
    /// carries, for a stream that multiplexes several subtitle pages.
    pub fn with_page_id(mut self, page_id: u16) -> Self {
        self.pinned_page = Some(page_id);
        self.dec.select_pages(PageIds {
            composition: page_id,
            ancillary: page_id,
        });
        self
    }

    /// Canvases emitted so far.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn input_caps() -> Caps {
        Caps::SubPicture {
            format: SubPictureFormat::DvbSub,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(self.framerate_q16),
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

    /// Emit the empty canvas the stream starts on, once, before the first cue.
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

    /// Emit an all-transparent canvas at `pts_ns` and drop the pending timeout.
    async fn clear(&mut self, pts_ns: u64, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        self.deadline = None;
        let bytes = (self.width as usize) * (self.height as usize) * 4;
        let timing = FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        };
        let frame = self.canvas_frame(vec![0u8; bytes], timing);
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    /// Emit the clear canvas the expired page owes, if `now` is past its
    /// `page_time_out`. A stream that ends a cue with an empty page composition
    /// never reaches this; one that just stops sending does.
    async fn expire(&mut self, now: u64, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        match self.deadline {
            Some(deadline) if now > deadline => self.clear(deadline, out).await,
            _ => Ok(()),
        }
    }

    /// Decode one display set into its canvas frame.
    async fn decode_display_set(
        &mut self,
        data: &[u8],
        timing: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        self.expire(timing.pts_ns, out).await?;
        let Some(page) = self.dec.feed(data) else {
            // A data field that carries no page composition, or one whose
            // segments do not hold together: nothing to show, and the stream
            // continues at the next display set.
            return Ok(());
        };
        if (page.width, page.height) != (self.width, self.height) {
            self.width = page.width;
            self.height = page.height;
            let caps = self.output_caps();
            self.last_caps = Some(caps.clone());
            out.push(PipelinePacket::CapsChanged(caps)).await?;
        }
        if !page.visible {
            return self.clear(timing.pts_ns, out).await;
        }
        // The timeout is the stream's own statement of how long the page stands,
        // and it is the duration downstream sees unless a later display set
        // replaces the page sooner.
        let duration_ns = page.timeout_s as u64 * 1_000_000_000;
        self.deadline = Some(timing.pts_ns.saturating_add(duration_ns));
        let shown = FrameTiming {
            duration_ns,
            ..timing
        };
        let frame = self.canvas_frame(page.canvas, shown);
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    /// Adopt the composition / ancillary page ids a config blob carries. `None`
    /// when the bytes are not one (so they are a display set). A `page-id`
    /// property pins the page, and then the blob only names the ancillary page.
    fn apply_page_ids(&mut self, bytes: &[u8]) -> Option<()> {
        let mut ids = parse_page_ids(bytes)?;
        if let Some(pinned) = self.pinned_page {
            ids.composition = pinned;
        }
        self.dec.select_pages(ids);
        Some(())
    }
}

impl AsyncElement for DvbSubDec {
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
                format: SubPictureFormat::DvbSub,
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // The page-id config and the display sets share one pad.
                    if self.apply_page_ids(slice).is_some() {
                        return self.prime(out).await;
                    }
                    let timing = frame.timing;
                    let data = slice.to_vec();
                    self.prime(out).await?;
                    self.decode_display_set(&data, timing, out).await?;
                }
                // The last page still owes its clear canvas if the stream ended
                // without one; the runner forwards the EOS itself.
                PipelinePacket::Eos => {
                    if let Some(deadline) = self.deadline {
                        self.clear(deadline, out).await?;
                    }
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
        DVBSUBDEC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if name == "page-id" {
            let v = value.as_int().ok_or(PropError::Type)?;
            self.pinned_page = match v {
                -1 => None,
                0..=0xFFFF => Some(v as u16),
                _ => return Err(PropError::Value),
            };
            if let Some(page_id) = self.pinned_page {
                self.dec.select_pages(PageIds {
                    composition: page_id,
                    ancillary: page_id,
                });
            }
            return Ok(());
        }
        let v = value.as_uint().ok_or(PropError::Type)?;
        let v = u32::try_from(v).map_err(|_| PropError::Value)?;
        match name {
            "width" | "height" if v == 0 || v > MAX_DISPLAY_DIM => Err(PropError::Value),
            "width" => {
                self.width = v;
                self.dec.set_display_size(self.width, self.height);
                Ok(())
            }
            "height" => {
                self.height = v;
                self.dec.set_display_size(self.width, self.height);
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
            "page-id" => PropValue::Int(self.pinned_page.map_or(-1, i64::from)),
            _ => return None,
        })
    }
}

static DVBSUBDEC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "width",
        PropKind::Uint,
        "display width; a display definition segment overrides it",
    )
    .with_default("720"),
    PropertySpec::new(
        "height",
        PropKind::Uint,
        "display height; a display definition segment overrides it",
    )
    .with_default("576"),
    PropertySpec::new(
        "framerate",
        PropKind::Uint,
        "nominal output framerate in fps (labels the caps; cues stay sparse)",
    )
    .with_default("25"),
    PropertySpec::new(
        "page-id",
        PropKind::Int,
        "composition page to compose (-1 = the first page the stream carries)",
    )
    .with_default("-1"),
];

impl PadTemplates for DvbSubDec {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(DvbSubDec::input_caps())),
            PadTemplate::source(CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(DEFAULT_DISPLAY_WIDTH),
                height: Dim::Fixed(DEFAULT_DISPLAY_HEIGHT),
                framerate: Rate::Fixed(25 << 16),
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
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            self.packets.push(packet);
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }
    }

    /// Wrap a segment payload in its 6-byte header.
    fn seg(kind: u8, page_id: u16, body: &[u8]) -> Vec<u8> {
        let mut out = Vec::from([0x0Fu8, kind]);
        out.extend_from_slice(&page_id.to_be_bytes());
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(body);
        out
    }

    /// A display set placing a 2-bit region of solid pixel code 1 as
    /// `(x, y, w, h)`, or clearing the page when `at` is `None`. `w` must be in
    /// 29..=284, the run the 2-bit escape codes.
    fn display_set(page_id: u16, at: Option<(u16, u16, u16, u16)>, timeout_s: u8) -> Vec<u8> {
        let mut page = Vec::from([timeout_s, 0x08]); // version 0, state 2 (mode change)
        let mut rest = Vec::new();
        if let Some((x, y, w, h)) = at {
            page.extend_from_slice(&[0x00, 0xff]);
            page.extend_from_slice(&x.to_be_bytes());
            page.extend_from_slice(&y.to_be_bytes());
            // CLUT 0: 2-bit entry 1 opaque white, entry 0 transparent.
            rest.extend_from_slice(&seg(
                0x12,
                page_id,
                &[
                    0x00, 0x0f, //
                    0x00, 0xbf, 0x10, 0x80, 0x80, 0xff, // entry 0, transparent
                    0x01, 0xbf, 0xeb, 0x80, 0x80, 0x00, // entry 1, opaque white
                ],
            ));
            rest.extend_from_slice(&seg(
                0x11,
                page_id,
                &[
                    0x00,
                    0x07,
                    (w >> 8) as u8,
                    w as u8,
                    (h >> 8) as u8,
                    h as u8,
                    0x24, // level 1, depth 1 (2-bit)
                    0x00, // CLUT id
                    0x00, // 8-bit background
                    0x00, // 4- and 2-bit background
                    0x00,
                    0x00, // object id 0
                    0x00,
                    0x00, // type 0, provider 0, at x 0
                    0xf0,
                    0x00, // at y 0
                ],
            ));
            // One 2-bit line per field row, 22 bits over three bytes: the
            // escape, switch_1 = 0, switch_2 = 0, switch_3 = 3 (so an 8-bit
            // run_length_29-284), the pixel code 1, then the end-of-string
            // escape and two stuffing bits.
            let word: u32 = (0b000011 << 18) | (((w as u32) - 29) << 10) | (1 << 8);
            let line = [(word >> 16) as u8, (word >> 8) as u8, word as u8];
            let field = |rows: usize| {
                let mut out = Vec::new();
                for r in 0..rows {
                    out.push(0x10);
                    out.extend_from_slice(&line);
                    if r + 1 < rows {
                        out.push(0xf0);
                    }
                }
                out
            };
            let top = field(usize::from(h).div_ceil(2));
            let bottom = field(usize::from(h) / 2);
            let mut body = Vec::from([0x00u8, 0x00, 0x00]);
            body.extend_from_slice(&(top.len() as u16).to_be_bytes());
            body.extend_from_slice(&(bottom.len() as u16).to_be_bytes());
            body.extend_from_slice(&top);
            body.extend_from_slice(&bottom);
            rest.extend_from_slice(&seg(0x13, page_id, &body));
        }
        let mut out = Vec::from([0x20u8, 0x00]);
        out.extend_from_slice(&seg(0x10, page_id, &page));
        out.extend_from_slice(&rest);
        out.extend_from_slice(&seg(0x80, page_id, &[]));
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

    async fn run(dec: &mut DvbSubDec, packets: Vec<PipelinePacket>) -> Vec<PipelinePacket> {
        let mut sink = CollectSink::default();
        for p in packets {
            dec.process(p, &mut sink).await.unwrap();
        }
        sink.packets
    }

    fn frames(packets: Vec<PipelinePacket>) -> Vec<Frame> {
        packets
            .into_iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    fn opaque_count(f: &Frame) -> usize {
        f.domain
            .as_system_slice()
            .unwrap()
            .chunks_exact(4)
            .filter(|px| px[3] != 0)
            .count()
    }

    #[tokio::test]
    async fn a_display_set_emits_a_painted_canvas_and_an_empty_page_clears_it() {
        let mut dec = DvbSubDec::new();
        dec.configure_pipeline(&DvbSubDec::input_caps()).unwrap();
        let out = run(
            &mut dec,
            vec![
                frame(&[0x00, 0x01, 0x00, 0x01, 0x10], 0),
                frame(&display_set(1, Some((40, 20, 32, 8)), 30), 1_000_000_000),
                frame(&display_set(1, None, 30), 3_000_000_000),
            ],
        )
        .await;
        let frames = frames(out);
        assert_eq!(
            frames.len(),
            3,
            "the opening empty canvas, the painted page, then the clear canvas"
        );
        assert_eq!(opaque_count(&frames[0]), 0);

        assert_eq!(frames[1].timing.pts_ns, 1_000_000_000);
        assert_eq!(
            frames[1].timing.duration_ns, 30_000_000_000,
            "the page stands for its page_time_out"
        );
        assert_eq!(opaque_count(&frames[1]), 32 * 8);
        let painted = frames[1].domain.as_system_slice().unwrap();
        let px = |x: usize, y: usize| &painted[(y * 720 + x) * 4..(y * 720 + x) * 4 + 4];
        assert_eq!(px(40, 20), [255, 255, 255, 255]);
        assert_eq!(px(71, 27), [255, 255, 255, 255]);
        assert_eq!(px(39, 20), [0, 0, 0, 0]);
        assert_eq!(px(72, 27), [0, 0, 0, 0]);

        assert_eq!(frames[2].timing.pts_ns, 3_000_000_000);
        assert_eq!(opaque_count(&frames[2]), 0);
    }

    #[tokio::test]
    async fn a_page_that_outlives_its_timeout_is_cleared_at_the_deadline() {
        let mut dec = DvbSubDec::new();
        dec.configure_pipeline(&DvbSubDec::input_caps()).unwrap();
        let out = run(
            &mut dec,
            vec![
                // a 2 s timeout, then the next display set 5 s later
                frame(&display_set(1, Some((0, 0, 32, 4)), 2), 1_000_000_000),
                frame(&display_set(1, Some((0, 0, 32, 4)), 2), 6_000_000_000),
            ],
        )
        .await;
        let frames = frames(out);
        assert_eq!(frames.len(), 4, "prime, page, timeout clear, page");
        assert_eq!(frames[2].timing.pts_ns, 3_000_000_000);
        assert_eq!(opaque_count(&frames[2]), 0);
        assert_eq!(frames[3].timing.pts_ns, 6_000_000_000);
    }

    #[tokio::test]
    async fn eos_clears_a_page_the_stream_never_ended() {
        let mut dec = DvbSubDec::new();
        dec.configure_pipeline(&DvbSubDec::input_caps()).unwrap();
        let out = run(
            &mut dec,
            vec![
                frame(&display_set(1, Some((0, 0, 32, 4)), 4), 1_000_000_000),
                PipelinePacket::Eos,
            ],
        )
        .await;
        let frames = frames(out);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[2].timing.pts_ns, 5_000_000_000);
        assert_eq!(opaque_count(&frames[2]), 0);
    }

    #[tokio::test]
    async fn the_page_id_blob_selects_which_page_is_composed() {
        let mut dec = DvbSubDec::new();
        dec.configure_pipeline(&DvbSubDec::input_caps()).unwrap();
        let out = run(
            &mut dec,
            vec![
                // composition page 4, ancillary 4
                frame(&[0x00, 0x04, 0x00, 0x04, 0x10], 0),
                frame(&display_set(1, Some((0, 0, 32, 4)), 30), 1_000_000_000),
                frame(&display_set(4, Some((8, 8, 32, 4)), 30), 2_000_000_000),
            ],
        )
        .await;
        let frames = frames(out);
        assert_eq!(frames.len(), 2, "page 1's display set is not this page's");
        assert_eq!(frames[1].timing.pts_ns, 2_000_000_000);
        assert_eq!(opaque_count(&frames[1]), 32 * 4);
    }

    #[tokio::test]
    async fn a_display_definition_refines_the_output_geometry_once() {
        let mut dec = DvbSubDec::new();
        dec.configure_pipeline(&DvbSubDec::input_caps()).unwrap();
        let mut hd = Vec::from([0x20u8, 0x00]);
        // 1920x1080, no display window
        hd.extend_from_slice(&seg(0x14, 1, &[0x00, 0x07, 0x7f, 0x04, 0x37]));
        hd.extend_from_slice(&display_set(1, Some((0, 0, 32, 4)), 30)[2..]);
        let out = run(
            &mut dec,
            vec![
                frame(&hd, 1_000_000_000),
                frame(&display_set(1, Some((0, 0, 32, 4)), 30), 2_000_000_000),
            ],
        )
        .await;
        let caps: Vec<Caps> = out
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            caps,
            vec![Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(1920),
                height: Dim::Fixed(1080),
                framerate: Rate::Fixed(25 << 16),
            }],
            "the display definition segment refines the geometry once"
        );
        let frames = frames(out);
        assert_eq!(
            frames[2].domain.as_system_slice().unwrap().len(),
            1920 * 1080 * 4
        );
    }

    #[tokio::test]
    async fn a_malformed_display_set_is_dropped_without_failing_the_stream() {
        let mut dec = DvbSubDec::new();
        dec.configure_pipeline(&DvbSubDec::input_caps()).unwrap();
        let mut bad = display_set(1, Some((0, 0, 32, 4)), 30);
        bad[6..8].copy_from_slice(&0x7fffu16.to_be_bytes()); // segment length past the end
        let out = run(&mut dec, vec![frame(&bad, 0)]).await;
        assert_eq!(frames(out).len(), 1, "only the opening empty canvas");
    }

    #[test]
    fn rejects_input_that_is_not_a_dvb_subtitle_stream() {
        let mut dec = DvbSubDec::new();
        assert!(dec
            .configure_pipeline(&Caps::SubPicture {
                format: SubPictureFormat::VobSub
            })
            .is_err());
    }

    #[test]
    fn properties_round_trip() {
        let mut dec = DvbSubDec::new();
        for (name, value) in [("width", 1920u64), ("height", 1080), ("framerate", 30)] {
            dec.set_property(name, PropValue::Uint(value)).unwrap();
            assert_eq!(dec.get_property(name), Some(PropValue::Uint(value)));
        }
        assert!(dec.set_property("width", PropValue::Uint(0)).is_err());
        assert!(dec
            .set_property("framerate", PropValue::Uint(5000))
            .is_err());
        assert_eq!(dec.get_property("page-id"), Some(PropValue::Int(-1)));
        dec.set_property("page-id", PropValue::Int(7)).unwrap();
        assert_eq!(dec.get_property("page-id"), Some(PropValue::Int(7)));
        assert!(dec.set_property("page-id", PropValue::Int(-2)).is_err());
    }
}
