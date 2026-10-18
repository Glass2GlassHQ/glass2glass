//! Shared access-unit parser core for the H.264 and H.265 elementary-stream
//! parsers.
//!
//! `H264Parse` and `H265Parse` do the same job: scan each `DataFrame` for an SPS,
//! refine caps to the coded geometry (suppressing an unchanged re-emit), and
//! optionally re-frame the bitstream to one access unit per `DataFrame` while
//! re-inserting parameter sets on a `config-interval`. They differ only in
//! codec-specific pieces: the NAL classification, the access-unit boundary rules,
//! the SPS geometry parse, and which parameter sets to cache and re-insert.
//!
//! `NalParse<C>` holds the shared machinery; a [`NalCodec`] marker (`H264Codec` /
//! `H265Codec`) supplies the codec-specific hooks. `H264Parse` / `H265Parse` are
//! type aliases over it, so the public surface (constructors, `AsyncElement`,
//! `PadTemplates`) is unchanged.

use core::future::Future;
use core::marker::PhantomData;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, Colorimetry, ConfigureOutcome, Dim,
    ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropValue, PropertySpec, Rate, VideoCodec,
};

use crate::access_unit::{
    AccessUnitFramer, CapsRefinement, ConfigInterval, StartCodeRole, MAX_CONFIG_INTERVAL_SECONDS,
};
use crate::annexb::{nal_units, nal_units_any, ANNEX_B_START_CODE};

/// Coded-picture geometry recovered from an SPS (post conformance-window
/// cropping), plus the framerate when the codec recovers one.
#[derive(Debug)]
pub struct SpsGeometry {
    pub width: u32,
    pub height: u32,
    /// Framerate as Q16 fixed-point fps (e.g. from the H.264 VUI `timing_info`),
    /// `None` when the SPS carries none or the codec does not recover it.
    pub framerate: Option<u32>,
    /// Colour description from the VUI `video_signal_type` block,
    /// [`Colorimetry::UNKNOWN`] when the SPS carries none.
    pub colorimetry: Colorimetry,
    /// SPS VUI context an H.264 `pic_timing` SEI needs to be parseable. `None`
    /// for H.265, whose `time_code` SEI is self-contained.
    pub pic_timing: Option<crate::sei::PicTimingContext>,
    /// Whether the SPS proves the stream presents pictures in the order it
    /// codes them, so a consumer may treat coded order as display order. Set
    /// only from a normative field that forbids reordering outright (see the
    /// per-codec parses, which spell out which one); `false` means "may
    /// reorder", including when the parse could not tell.
    pub presents_in_decode_order: bool,
    /// The fields picture order count derives from, for a consumer that has to
    /// put a reordering stream's access units back in display order. `None` when
    /// the parse could not read them (the geometry is still valid).
    pub poc: Option<crate::poc::SpsPocParams>,
}

/// Codec-specific hooks for [`NalParse`]. Implemented by zero-sized markers.
pub trait NalCodec: Send + Sync + 'static {
    /// The caps codec tag this parser accepts and emits.
    const CODEC: VideoCodec;
    /// `ElementMetadata` long name.
    const NAME: &'static str;
    /// `ElementMetadata` description.
    const DESCRIPTION: &'static str;
    /// The element's runtime property specs (the `config-interval` knob; its
    /// help text names the codec's parameter sets, so it varies per codec).
    const PROPERTIES: &'static [PropertySpec];
    /// Parameter-set NAL types to cache and re-insert, in prepend order (H.264:
    /// SPS then PPS; H.265: VPS then SPS then PPS).
    const PARAM_SET_TYPES: &'static [u8];
    /// The NAL type whose presence in an access unit means it already carries
    /// config (the SPS): it resets the re-insertion clock and suppresses
    /// prefixing. Must appear in [`PARAM_SET_TYPES`](Self::PARAM_SET_TYPES).
    const SPS_TYPE: u8;

    /// NAL unit type of a NAL body (its header parsed), or `None` if too short.
    fn nal_type(nal: &[u8]) -> Option<u8>;
    /// What the NAL starting at `unit` (its header first) does to the access
    /// unit being assembled, `None` when more of its bytes are needed to tell.
    fn unit_role(unit: &[u8]) -> Option<StartCodeRole>;
    /// Recover SPS geometry (and optional framerate) from an access unit's NALs.
    fn sps_info(nals: &[&[u8]]) -> Option<SpsGeometry>;

    /// [`Self::sps_info`] over an access unit in either framing.
    fn extract_sps_info(au: &[u8]) -> Option<SpsGeometry> {
        Self::sps_info(&nal_units_any(au).collect::<Vec<_>>())
    }
}

/// Access-unit parser generic over a [`NalCodec`]. See the module docs.
pub struct NalParse<C: NalCodec> {
    configured: bool,
    caps: CapsRefinement,
    /// Re-framing mode: re-chunk the input into access-unit-aligned Annex-B
    /// buffers (one coded picture per `DataFrame`) rather than passing buffers
    /// through unchanged. A decoder fed un-aligned units (e.g. one MPEG-TS PES
    /// that is not one access unit) mis-parses slice boundaries; auto-plugged
    /// decode chains use this so the decoder always sees one access unit per
    /// packet, matching GStreamer's `h264parse` / `h265parse`. Off in the default
    /// (caps-refinement-only) construction.
    reframe: bool,
    /// Re-framing accumulator. Empty outside re-framing mode.
    framer: AccessUnitFramer,
    /// Monotonic sequence number for emitted re-framed access units.
    seq: u64,
    /// Input framing, latched from the first frame (`Some(true)` = Annex-B start
    /// codes, `Some(false)` = length prefixes). Fixed for a stream, so decided
    /// once: a per-frame guess misclassifies a mid-AU Annex-B continuation buffer
    /// (no leading start code) as length-prefixed and mangles it. Re-framing only.
    input_is_annexb: Option<bool>,
    /// Parameter-set re-insertion cadence, applied on the re-framing
    /// (access-unit-aligned Annex-B) output.
    config_interval: ConfigInterval,
    leading_rasl: crate::h265parse::LeadingRaslFilter,
    /// Cached parameter-set NAL bodies (start codes stripped), one bucket per
    /// [`NalCodec::PARAM_SET_TYPES`] entry in the same order, refreshed as they
    /// flow so a keyframe lacking inline parameter sets can be prefixed.
    cached: Vec<Vec<Vec<u8>>>,
    /// Last HDR10 static metadata seen in the stream's SEI. It describes the
    /// whole coded video sequence but is coded per IRAP, so it is latched and
    /// re-attached to every frame.
    #[cfg(feature = "metadata")]
    hdr: Option<g2g_core::meta::HdrStaticMeta>,
    /// SPS VUI context the H.264 `pic_timing` SEI parse needs, latched from the
    /// last SPS (the SEI may ride an access unit that carries no SPS).
    #[cfg(feature = "metadata")]
    pic_timing: crate::sei::PicTimingContext,
    /// Framerate the last SPS declared, stamped onto a recovered timecode so a
    /// consumer can convert the frame count to a duration.
    #[cfg(feature = "metadata")]
    sps_framerate: Option<u32>,
    _codec: PhantomData<C>,
}

impl<C: NalCodec> Default for NalParse<C> {
    fn default() -> Self {
        Self {
            configured: false,
            caps: CapsRefinement::default(),
            reframe: false,
            framer: AccessUnitFramer::new(true),
            seq: 0,
            input_is_annexb: None,
            config_interval: ConfigInterval::default(),
            leading_rasl: crate::h265parse::LeadingRaslFilter::default(),
            cached: C::PARAM_SET_TYPES.iter().map(|_| Vec::new()).collect(),
            #[cfg(feature = "metadata")]
            hdr: None,
            #[cfg(feature = "metadata")]
            pic_timing: crate::sei::PicTimingContext::default(),
            #[cfg(feature = "metadata")]
            sps_framerate: None,
            _codec: PhantomData,
        }
    }
}

impl<C: NalCodec> core::fmt::Debug for NalParse<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NalParse")
            .field("codec", &C::CODEC)
            .field("configured", &self.configured)
            .field("reframe", &self.reframe)
            .field("config_interval", &self.config_interval.seconds)
            .finish_non_exhaustive()
    }
}

impl<C: NalCodec> NalParse<C> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct the parser in access-unit re-framing mode: it accumulates the
    /// input bitstream and emits one access-unit-aligned Annex-B `DataFrame` per
    /// coded picture (see `reframe`).
    pub fn reframing() -> Self {
        Self {
            reframe: true,
            ..Self::default()
        }
    }

    /// Count of `CapsChanged` packets this element has pushed downstream. Useful
    /// for tests asserting re-emission is suppressed when the SPS geometry is
    /// unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.caps.changes_emitted
    }

    /// Set the parameter-set re-insertion interval in seconds (`0` off, `-1` every
    /// keyframe, `N` every N seconds); see `config_interval`.
    pub fn with_config_interval(mut self, seconds: i32) -> Self {
        self.config_interval.seconds = seconds;
        self
    }

    /// Refresh the cached parameter sets from `au` and, when re-insertion is due
    /// for this access unit, return the AU prefixed with the cached parameter sets
    /// (as Annex-B). `au` must be Annex-B (the re-framing output always is).
    pub(crate) fn apply_config_interval(
        &mut self,
        au: Vec<u8>,
        pts_ns: u64,
        keyframe: bool,
    ) -> Vec<u8> {
        if self.config_interval.is_off() {
            return au;
        }
        let mut refreshed_buckets = 0u32;
        for nal in nal_units(&au) {
            let bucket = C::nal_type(nal)
                .and_then(|nal_type| C::PARAM_SET_TYPES.iter().position(|&t| t == nal_type));
            let Some(bucket) = bucket else {
                continue;
            };
            if refreshed_buckets & (1 << bucket) == 0 {
                self.cached[bucket].clear();
                refreshed_buckets |= 1 << bucket;
            }
            self.cached[bucket].push(nal.to_vec());
        }
        let sps_bucket = C::PARAM_SET_TYPES
            .iter()
            .position(|&t| t == C::SPS_TYPE)
            .expect("SPS_TYPE must appear in PARAM_SET_TYPES");
        let carries_config = refreshed_buckets & (1 << sps_bucket) != 0;
        let have_cached_config = !self.cached[sps_bucket].is_empty();
        if !self
            .config_interval
            .insertion_due(pts_ns, keyframe, carries_config, have_cached_config)
        {
            return au;
        }
        let mut prefixed = Vec::new();
        for parameter_set in self.cached.iter().flatten() {
            prefixed.extend_from_slice(&ANNEX_B_START_CODE);
            prefixed.extend_from_slice(parameter_set);
        }
        prefixed.extend_from_slice(&au);
        prefixed
    }

    /// Refine caps from any SPS in `nals` (suppressing an unchanged re-emit).
    async fn refine_caps(
        &mut self,
        nals: &[&[u8]],
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let Some(info) = C::sps_info(nals) else {
            return Ok(());
        };
        #[cfg(feature = "metadata")]
        {
            // The SEI timecode parse and its frame-rate field both come from
            // the SPS, which may sit in an earlier access unit than the SEI.
            if let Some(ctx) = info.pic_timing {
                self.pic_timing = ctx;
            }
            self.sps_framerate = info.framerate;
        }
        let new_caps = Caps::CompressedVideo {
            codec: C::CODEC,
            width: Dim::Fixed(info.width),
            height: Dim::Fixed(info.height),
            framerate: self.caps.framerate(info.framerate),
            colorimetry: info.colorimetry,
        };
        self.caps.refine(new_caps, out).await
    }

    /// Re-framing path for one input `DataFrame`: normalize to Annex-B, then
    /// emit every access unit the framer completes. Non-`System` domains pass
    /// through unchanged (the byte re-framer only applies to host memory).
    async fn reframe_frame(
        &mut self,
        frame: Frame,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let Some(bytes) = frame.domain.as_system_slice() else {
            out.push(PipelinePacket::DataFrame(frame)).await?;
            return Ok(());
        };
        // Normalize to Annex-B: a length-prefixed stream is converted, an Annex-B
        // one is appended as-is. The framing is latched from the first frame, since
        // a mid-AU Annex-B continuation buffer (no leading start code) would
        // otherwise be misread as length-prefixed.
        let is_annexb = *self.input_is_annexb.get_or_insert_with(|| {
            // A raw `.h264` file may open with junk before its first start code.
            crate::annexb::is_annex_b(bytes) || !crate::annexb::is_avcc(bytes)
        });
        let access_units = if is_annexb {
            self.framer.push(bytes, frame.timing, C::unit_role)
        } else {
            self.framer
                .push_converted(
                    bytes,
                    frame.timing,
                    C::unit_role,
                    crate::annexb::avcc_to_annexb_in_place,
                )
                .ok_or(G2gError::CapsMismatch)?
        };
        for (au, timing) in access_units {
            self.emit_au(au, timing, out).await?;
        }
        Ok(())
    }

    /// Emit one access-unit-aligned Annex-B buffer as a `DataFrame`: refine caps
    /// from any SPS it carries (suppressing an unchanged re-emit) and stamp the
    /// keyframe flag, mirroring the pass-through path's per-frame work.
    async fn emit_au(
        &mut self,
        au: Vec<u8>,
        mut timing: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if au.is_empty() {
            return Ok(());
        }
        let nals: Vec<&[u8]> = nal_units(&au).collect();
        if C::CODEC == VideoCodec::H265 && !self.leading_rasl.keeps_access_unit(&nals) {
            return Ok(());
        }
        self.refine_caps(&nals, out).await?;
        timing.keyframe = crate::annexb::nal_units_contain_keyframe(C::CODEC, nals.iter().copied());
        #[cfg(feature = "metadata")]
        let sei = crate::sei::parse_nals(nals.iter().copied(), C::CODEC, self.pic_timing);
        drop(nals);
        // Re-insert cached parameter sets before this AU when config-interval calls
        // for it (no-op at interval 0 or when the AU already carries them).
        let au = self.apply_config_interval(au, timing.pts_ns, timing.keyframe);
        #[allow(unused_mut)]
        let mut frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            timing,
            self.seq,
        );
        #[cfg(feature = "metadata")]
        self.attach_stream_meta(&mut frame, sei);
        self.seq += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    /// Attach what the access unit's SEI carried, so it outlives the bitstream:
    /// a decoder throws the SEI away, but the meta rides the decoded frames (and
    /// a re-encode, see the metas' `propagate`) to a downstream caption inserter
    /// or HDR display sink.
    ///
    /// Captions are per-picture. HDR10 static metadata describes the whole coded
    /// video sequence and is typically coded once per IRAP, so the last one seen
    /// is latched and re-attached to every frame.
    #[cfg(feature = "metadata")]
    fn attach_stream_meta(&mut self, frame: &mut Frame, info: crate::sei::SeiInfo) {
        if let Some(hdr) = info.hdr {
            self.hdr = Some(hdr);
        }
        if !info.captions.is_empty() {
            let mut meta = g2g_core::meta::CaptionMeta::new();
            for t in info.captions {
                meta.push(t.into());
            }
            frame.meta.attach(meta);
        }
        if let Some(hdr) = self.hdr {
            frame.meta.attach(hdr);
        }
        if let Some(mut tc) = info.timecode {
            // The SEI codes the count, not the rate it runs at; that is the SPS's.
            tc.framerate_q16 = self.sps_framerate;
            frame.meta.attach(tc);
        }
    }

    /// The `CompressedVideo` caps at any geometry that this parser accepts and
    /// emits (it refines geometry mid-stream from the SPS but never changes media
    /// type).
    fn any_caps() -> Caps {
        Caps::CompressedVideo {
            codec: C::CODEC,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }
}

impl<C: NalCodec> AsyncElement for NalParse<C> {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // The parser consumes its codec at any geometry; intersecting against that
        // narrows the proposal and rejects other codecs.
        upstream_caps.intersect(&Self::any_caps())
    }

    /// Pass-through identity over the codec at any geometry. With a fully-native
    /// chain the solver couples input and output links and rejects a mismatched
    /// upstream at negotiation time instead of via the dynamic `intercept_caps`.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(CapsSet::one(Self::any_caps()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo { codec, .. } if *codec == C::CODEC => {
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(C::NAME, "Codec/Parser/Video", C::DESCRIPTION, "g2g")
    }

    fn properties(&self) -> &'static [PropertySpec] {
        C::PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "config-interval" => {
                let secs = value.as_int().ok_or(PropError::Type)?;
                if !(-1..=MAX_CONFIG_INTERVAL_SECONDS).contains(&secs) {
                    return Err(PropError::Value);
                }
                self.config_interval.seconds = secs as i32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "config-interval" => Some(PropValue::Int(self.config_interval.seconds as i64)),
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
                PipelinePacket::DataFrame(mut frame) => {
                    if self.reframe {
                        return self.reframe_frame(frame, out).await;
                    }
                    if let Some(slice) = frame.domain.as_system_slice() {
                        let nals: Vec<&[u8]> = nal_units_any(slice).collect();
                        // Surface the keyframe flag for trick-mode / keyframe seek
                        // (the parser is the producer that can detect it).
                        let is_keyframe = crate::annexb::nal_units_contain_keyframe(
                            C::CODEC,
                            nals.iter().copied(),
                        );
                        self.refine_caps(&nals, out).await?;
                        #[cfg(feature = "metadata")]
                        let sei =
                            crate::sei::parse_nals(nals.iter().copied(), C::CODEC, self.pic_timing);
                        drop(nals);
                        frame.timing.keyframe = is_keyframe;
                        #[cfg(feature = "metadata")]
                        self.attach_stream_meta(&mut frame, sei);
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    self.caps.observe_upstream(&c);
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    // A seek discontinuity: drop the partial AU rather than splice
                    // pre-seek bytes onto the post-seek stream. Reset SPS tracking
                    // so caps re-emit after the seek.
                    self.framer.clear();
                    self.caps.forget();
                    self.leading_rasl.restart();
                    out.push(PipelinePacket::Flush).await?;
                }
                // A new segment may splice in a bitstream the decoder has no references for.
                PipelinePacket::Segment(seg) => {
                    self.leading_rasl.restart();
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {
                    // Flush the final buffered access unit at end of stream, else
                    // the last coded picture would never reach the decoder.
                    if let Some((au, timing)) = self.framer.finish() {
                        self.emit_au(au, timing, out).await?;
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl<C: NalCodec> PadTemplates for NalParse<C> {
    fn pad_templates() -> Vec<PadTemplate> {
        let caps = Self::any_caps();
        Vec::from([
            PadTemplate::sink(CapsSet::one(caps.clone())),
            PadTemplate::source(CapsSet::one(caps)),
        ])
    }
}
