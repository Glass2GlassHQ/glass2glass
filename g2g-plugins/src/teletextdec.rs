//! EBU teletext subtitle decoder element (M924): `Caps::Text{Teletext}` in,
//! timed `Caps::Text{Utf8}` cues out.
//!
//! The bitmap-subtitle decoders turn their cues into RGBA canvases because their
//! payload is pixels; teletext is characters, so this one lands on the same
//! plain-text pad a `subparse` produces and everything downstream (a
//! [`TextOverlay`](crate::textoverlay), a caption encoder, a text sink) already
//! consumes:
//!
//! ```text
//! filesrc ! tsdemux stream=teletext ! teletextdec page=888 ! textoverlay name=o
//! ```
//!
//! A cue's duration is the span until the page is replaced or erased, which is
//! only known when the next page header arrives, so each cue goes out one page
//! late and the last one is released at `Eos`.
//!
//! Which page to follow is not in the bitstream: it arrives as the blob the
//! demuxer forwards ahead of the first data unit, built from the PMT
//! `teletext_descriptor` (see [`crate::teletext::parse_page_config`]). The `page`
//! property overrides it; with neither, the first subtitle page the stream
//! carries is adopted.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, TextFormat,
};

use crate::teletext::{parse_page_config, TeletextCue, TeletextDecoder};

/// Decodes teletext subtitle pages into plain-text cues.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::teletextdec::TeletextDec;
///
/// let decoder = TeletextDec::new().with_page(888);
/// ```
#[derive(Debug)]
pub struct TeletextDec {
    dec: TeletextDecoder,
    /// The page pinned by the `page` property, which outranks the demuxer's blob.
    pinned_page: Option<u16>,
    configured: bool,
    /// Whether the output `Caps::Text{Utf8}` has been announced downstream.
    caps_emitted: bool,
    /// The most recent payload's PTS, the end time a page still on screen at
    /// `Eos` is closed at.
    last_pts: u64,
    emitted: u64,
}

impl Default for TeletextDec {
    fn default() -> Self {
        Self::new()
    }
}

impl TeletextDec {
    pub fn new() -> Self {
        Self {
            dec: TeletextDecoder::new(),
            pinned_page: None,
            configured: false,
            caps_emitted: false,
            last_pts: 0,
            emitted: 0,
        }
    }

    /// Follow this teletext page rather than the one the PMT named (or the first
    /// subtitle page the stream carries).
    pub fn with_page(mut self, page: u16) -> Self {
        self.pinned_page = Some(page);
        self.dec.select_page(page);
        self
    }

    /// Cues emitted so far.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn input_caps() -> Caps {
        Caps::Text {
            format: TextFormat::Teletext,
        }
    }

    fn output_caps() -> Caps {
        Caps::Text {
            format: TextFormat::Utf8,
        }
    }

    /// Announce `Caps::Text{Utf8}` once, ahead of the first cue.
    async fn announce(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.caps_emitted {
            return Ok(());
        }
        self.caps_emitted = true;
        out.push(PipelinePacket::CapsChanged(Self::output_caps()))
            .await?;
        Ok(())
    }

    async fn emit(&mut self, cue: TeletextCue, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        self.announce(out).await?;
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(
                cue.text.into_bytes().into_boxed_slice(),
            )),
            FrameTiming {
                pts_ns: cue.pts_ns,
                dts_ns: cue.pts_ns,
                duration_ns: cue.duration_ns,
                ..FrameTiming::default()
            },
            self.emitted,
        );
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

impl AsyncElement for TeletextDec {
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
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Text {
                format: TextFormat::Teletext,
            } => CapsSet::one(Self::output_caps()),
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
                    // The page blob and the data units share one pad; the blob is
                    // the one that parses as a page selection.
                    if let Some(cfg) = parse_page_config(slice) {
                        if self.pinned_page.is_none() {
                            self.dec.select_page(cfg.page);
                        }
                        return Ok(());
                    }
                    let pts = frame.timing.pts_ns;
                    self.last_pts = pts;
                    let cues = self.dec.push(slice, pts);
                    for cue in cues {
                        self.emit(cue, out).await?;
                    }
                }
                // A page still on screen is ended at the last payload's PTS, the
                // way `ccextract` closes a caption; the runner arm forwards the
                // trailing Eos.
                PipelinePacket::Eos => {
                    if let Some(cue) = self.dec.flush(self.last_pts) {
                        self.emit(cue, out).await?;
                    }
                    out.push(PipelinePacket::Eos).await?;
                }
                // A seek leaves the rows behind the last header stranded; they
                // belong to the position being left.
                PipelinePacket::Flush => {
                    self.dec.reset();
                    out.push(PipelinePacket::Flush).await?;
                }
                // The runner feeds back our own solved output caps; accept them
                // and announce ours if the first cue has not already.
                PipelinePacket::CapsChanged(_) => self.announce(out).await?,
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        TELETEXTDEC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Teletext subtitle decoder",
            "Codec/Decoder/Subtitle",
            "Decodes an EBU teletext subtitle page to plain-text cues",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "page" => {
                let v = value.as_uint().ok_or(PropError::Type)?;
                // Teletext pages run 100..899 (magazine 1..8, two digits).
                if !(100..=899).contains(&v) {
                    return Err(PropError::Value);
                }
                self.pinned_page = Some(v as u16);
                self.dec.select_page(v as u16);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "page" => Some(PropValue::Uint(
                self.pinned_page.or_else(|| self.dec.page()).unwrap_or(0) as u64,
            )),
            _ => None,
        }
    }
}

static TELETEXTDEC_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "page",
    PropKind::Uint,
    "teletext page to decode (100..899); else the page the PMT named",
)];

impl PadTemplates for TeletextDec {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(TeletextDec::input_caps())),
            PadTemplate::source(CapsSet::one(TeletextDec::output_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::teletext::{encode_payload, page_config_blob, DataUnit};
    use alloc::string::String;
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

    fn frame(bytes: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                ..FrameTiming::default()
            },
            0,
        ))
    }

    fn cues(packets: Vec<PipelinePacket>) -> Vec<(u64, u64, String)> {
        packets
            .into_iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some((
                    f.timing.pts_ns,
                    f.timing.duration_ns,
                    String::from_utf8(f.domain.as_system_slice().unwrap().to_vec()).unwrap(),
                )),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn the_demuxer_blob_selects_the_page_and_cues_carry_their_display_span() {
        let mut dec = TeletextDec::new();
        dec.configure_pipeline(&TeletextDec::input_caps()).unwrap();
        let mut sink = CollectSink::default();

        dec.process(frame(page_config_blob(888, *b"eng").to_vec(), 0), &mut sink)
            .await
            .unwrap();
        dec.process(
            frame(
                encode_payload(&[
                    DataUnit::page_header(888, 0, true),
                    DataUnit::text_row(8, 20, "ON AIR"),
                ]),
                1_000_000_000,
            ),
            &mut sink,
        )
        .await
        .unwrap();
        // A page with no rows erases the subtitle and ends the cue.
        dec.process(
            frame(
                encode_payload(&[DataUnit::page_header(888, 0, true)]),
                4_000_000_000,
            ),
            &mut sink,
        )
        .await
        .unwrap();
        dec.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        assert_eq!(
            cues(sink.packets),
            Vec::from([(1_000_000_000, 3_000_000_000, String::from("ON AIR"))])
        );
    }

    #[tokio::test]
    async fn the_page_property_outranks_the_demuxer_blob() {
        let mut dec = TeletextDec::new();
        dec.set_property("page", PropValue::Uint(150)).unwrap();
        dec.configure_pipeline(&TeletextDec::input_caps()).unwrap();
        let mut sink = CollectSink::default();
        dec.process(frame(page_config_blob(888, *b"eng").to_vec(), 0), &mut sink)
            .await
            .unwrap();
        dec.process(
            frame(
                encode_payload(&[
                    DataUnit::page_header(888, 0, true),
                    DataUnit::text_row(8, 20, "WRONG PAGE"),
                    DataUnit::page_header(150, 0, true),
                    DataUnit::text_row(1, 20, "RIGHT PAGE"),
                ]),
                1_000_000_000,
            ),
            &mut sink,
        )
        .await
        .unwrap();
        dec.process(
            frame(
                encode_payload(&[DataUnit::page_header(150, 0, true)]),
                3_000_000_000,
            ),
            &mut sink,
        )
        .await
        .unwrap();
        dec.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        assert_eq!(
            cues(sink.packets),
            Vec::from([(1_000_000_000, 2_000_000_000, String::from("RIGHT PAGE"))])
        );
    }

    #[tokio::test]
    async fn the_output_caps_are_announced_once_ahead_of_the_first_cue() {
        let mut dec = TeletextDec::new();
        dec.set_property("page", PropValue::Uint(888)).unwrap();
        dec.configure_pipeline(&TeletextDec::input_caps()).unwrap();
        let mut sink = CollectSink::default();
        // The runner feeds our own solved output caps back in; that must not
        // announce them twice once a cue has.
        dec.process(
            PipelinePacket::CapsChanged(TeletextDec::output_caps()),
            &mut sink,
        )
        .await
        .unwrap();
        for (pts, rows) in [(1_000_000_000u64, "ON AIR"), (3_000_000_000, "")] {
            let mut units = Vec::from([DataUnit::page_header(888, 0, true)]);
            if !rows.is_empty() {
                units.push(DataUnit::text_row(8, 20, rows));
            }
            dec.process(frame(encode_payload(&units), pts), &mut sink)
                .await
                .unwrap();
        }
        let announced: Vec<Caps> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(announced, Vec::from([TeletextDec::output_caps()]));
        assert_eq!(cues(sink.packets).len(), 1, "the cue still went out");
    }

    #[test]
    fn rejects_input_that_is_not_a_teletext_stream() {
        let mut dec = TeletextDec::new();
        assert!(dec
            .configure_pipeline(&Caps::Text {
                format: TextFormat::Srt
            })
            .is_err());
    }

    #[test]
    fn the_page_property_round_trips_and_rejects_a_page_outside_the_magazines() {
        let mut dec = TeletextDec::new();
        dec.set_property("page", PropValue::Uint(801)).unwrap();
        assert_eq!(dec.get_property("page"), Some(PropValue::Uint(801)));
        assert!(dec.set_property("page", PropValue::Uint(99)).is_err());
        assert!(dec.set_property("page", PropValue::Uint(900)).is_err());
    }
}
