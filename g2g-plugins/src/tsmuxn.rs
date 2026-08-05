//! Multi-stream MPEG-TS multiplexer element (M207): N elementary streams in
//! (e.g. H.264 video + AAC audio), one MPEG-TS byte stream out. The A+V analog
//! of the single-input [`crate::tsmux::TsMux`], the everyday live-streaming
//! container case.
//!
//! A [`MultiInputElement`]: each input pad accepts one elementary stream
//! (`Caps::CompressedVideo{H264|H265}` or `Caps::Audio{Aac}`), and the access
//! units are interleaved into one multiplex by presentation timestamp before
//! being written to their per-stream PIDs. Time-ordering reuses the M204
//! [`InputAggregator::take_earliest_by`](g2g_core::InputAggregator::take_earliest_by)
//! merge (release the globally earliest AU once every contributing input has one
//! queued), so the muxed TS is monotonic across streams the way a decoder
//! expects. The PMT (built once all inputs are configured) names every stream of
//! its program.
//!
//! Scope: one program by default, several via `prog-map` (M783), which assigns a
//! program number per input pad; each program gets its own PMT and its own PCR on
//! its first stream's PID, on the `pcr-interval` cadence (see
//! [`crate::mpegts::TsMuxer`]). Reachable from
//! the `gst-launch` fan-in syntax (M208): registered as the `mpegtsmux` muxer in
//! [`default_registry`](crate::registry::default_registry), so
//! `v.! m.  a.! m.  mpegtsmux name=m ! sink` builds this element when more than
//! one input links (a single input still builds the single-stream
//! [`crate::tsmux::TsMux`]). Also runs programmatically through `run_muxer_sink`
//! / a `run_graph` muxer node.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    resolve_tags, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming,
    G2gError, InputAggregator, MemoryDomain, MultiInputElement, OutputSink, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, TagList,
};

use crate::mpegts::{TsMuxer, STREAM_TYPE_H264, STREAM_TYPE_H265};
use crate::tsmux::{language_from_tags, service_from_tags, stream_type_for};

/// Muxes N elementary streams into one MPEG-TS byte stream, PTS-ordered.
#[derive(Debug)]
pub struct TsMux {
    inputs: usize,
    /// PMT stream type per input pad, learned at configure; the muxer is built
    /// once all are known.
    stream_types: Vec<Option<u8>>,
    /// Built lazily once every input is configured (the PMT needs all streams).
    mux: Option<TsMuxer>,
    /// Per-input AU buffer; releases the globally earliest-PTS AU (M204).
    agg: InputAggregator<Frame>,
    emitted: u64,
    /// PAT/PMT re-emission cadence in milliseconds (`0` = once up front), applied
    /// to the inner `TsMuxer` when it is built. The PAT and PMT are emitted
    /// together, so `pat-interval` and `pmt-interval` share this cadence, matching
    /// the single-input [`crate::tsmux::TsMux`].
    table_interval_ms: u64,
    /// PCR insertion cadence in 90 kHz ticks (default 3600), applied to the inner
    /// `TsMuxer` when it is built.
    pcr_interval_90khz: u64,
    /// Program number per input pad, in pad order (default: every input in
    /// program 1). Zipped with the learned stream types to build the muxer.
    program_numbers: Vec<u16>,
    /// Carry a `Caps::Klv` input as synchronous KLV (metadata-in-PES 0x15)
    /// instead of the default asynchronous private PES (0x06 + 'KLVA').
    klv_sync: bool,
    /// Whole-service metadata (M872), written to the SDT: the service every
    /// program without its own is named with.
    tags: TagList,
    /// Per-program service metadata (M878), keyed by program number: that
    /// program's own SDT entry text.
    program_tags: Vec<(u16, TagList)>,
    /// Per-input metadata: its `Tag::Language` becomes that stream's
    /// `ISO_639_language_descriptor`. One (possibly empty) list per input pad.
    track_tags: Vec<TagList>,
}

impl TsMux {
    /// A muxer with `inputs` input pads. Each pad's stream type is determined
    /// from its negotiated caps; the order of inputs is the order of streams in
    /// the PMT (and their PID assignment).
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "TsMux needs at least one input");
        Self {
            inputs,
            stream_types: alloc::vec![None; inputs],
            mux: None,
            agg: InputAggregator::new(inputs),
            emitted: 0,
            table_interval_ms: 0,
            pcr_interval_90khz: 3600,
            program_numbers: alloc::vec![1; inputs],
            klv_sync: false,
            tags: TagList::new(),
            program_tags: Vec::new(),
            track_tags: alloc::vec![TagList::new(); inputs],
        }
    }

    /// Attach whole-service metadata (M872): the service name ([`g2g_core::Tag::Title`]
    /// or a `service_name` key) and provider (a `service_provider` key) written to
    /// the SDT, naming every program that has no service of its own
    /// ([`with_program_tags`](Self::with_program_tags)). TS defines no free-form tag
    /// carrier, so any other tag here is dropped.
    pub fn with_tags(mut self, tags: TagList) -> Self {
        self.tags = tags;
        self
    }

    /// Attach metadata scoped to one program of a multi-program mux (M878): its
    /// service name / provider become that program's own SDT entry, so each program
    /// carries distinct service text instead of repeating the
    /// [`with_tags`](Self::with_tags) default. The program is named by its number in
    /// the [`with_program_numbers`](Self::with_program_numbers) / `prog-map` layout.
    /// A [`g2g_core::Tag::Language`] here is the default for that program's streams,
    /// between the global list and a stream's own
    /// [`with_track_tags`](Self::with_track_tags); TS carries nothing else, so the
    /// rest of the list is dropped.
    ///
    /// Panics when no input is in `program`: the SDT would silently omit the service
    /// otherwise. Set the program map first (every input is in program 1 by
    /// default).
    pub fn with_program_tags(mut self, program: u16, tags: TagList) -> Self {
        assert!(
            self.program_numbers.contains(&program),
            "TsMux has no input in program {program}"
        );
        self.program_tags.retain(|(n, _)| *n != program);
        self.program_tags.push((program, tags));
        self
    }

    /// Attach metadata scoped to one input pad's elementary stream: its
    /// [`g2g_core::Tag::Language`] becomes an `ISO_639_language_descriptor` in that
    /// stream's PMT entry, so a reader reports the language on that stream. Nothing
    /// else rides a TS elementary stream, so the rest of the list is dropped.
    /// Out-of-range inputs are ignored.
    ///
    /// A language set on the stream's program ([`with_program_tags`](Self::with_program_tags))
    /// or globally by [`with_tags`](Self::with_tags) applies to every stream that
    /// does not name its own (`g2g_core::resolve_tags`).
    pub fn with_track_tags(mut self, input: usize, tags: TagList) -> Self {
        if input < self.inputs {
            self.track_tags[input] = tags;
        }
        self
    }

    /// Assign a program number to each input pad, in pad order (M783): inputs
    /// sharing a number land in one program, and each program gets its own PMT and
    /// PCR. `numbers` must have one entry per input.
    pub fn with_program_numbers(mut self, numbers: &[u16]) -> Self {
        assert!(
            numbers.len() == self.inputs,
            "TsMux needs one program number per input"
        );
        assert!(
            self.program_tags.iter().all(|(n, _)| numbers.contains(n)),
            "a program with per-program tags is missing from the new program map"
        );
        self.program_numbers = numbers.to_vec();
        self
    }

    /// Set the PAT/PMT re-emission interval in milliseconds (`0` = once up front).
    pub fn with_table_interval_ms(mut self, ms: u64) -> Self {
        self.table_interval_ms = ms;
        self
    }

    /// Set the PCR insertion interval in 90 kHz ticks (default 3600).
    pub fn with_pcr_interval(mut self, ticks: u64) -> Self {
        self.pcr_interval_90khz = ticks;
        self
    }

    /// Carry a `Caps::Klv` input as synchronous KLV: metadata-in-PES
    /// (`stream_type` 0x15) rather than the default asynchronous private PES.
    pub fn with_klv_sync(mut self, sync: bool) -> Self {
        self.klv_sync = sync;
        self
    }

    /// Count of TS byte frames emitted.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The tags attached to input `i`'s program, if that program has any.
    fn tags_of_program(&self, i: usize) -> Option<&TagList> {
        self.program_tags
            .iter()
            .find(|(n, _)| *n == self.program_numbers[i])
            .map(|(_, t)| t)
    }

    /// The canonical `prog-map` value: the per-input program numbers joined by
    /// commas in pad order.
    fn prog_map_string(&self) -> String {
        let mut s = String::new();
        for (i, n) in self.program_numbers.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&alloc::format!("{n}"));
        }
        s
    }

    fn output_caps_value() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        }
    }
}

/// Parse a `prog-map` value: comma-separated program numbers, one per input in
/// pad order. `None` on an empty list or an entry that is not a `u16`.
fn parse_prog_map(s: &str) -> Option<Vec<u16>> {
    let numbers: Option<Vec<u16>> = s.split(',').map(|n| n.trim().parse::<u16>().ok()).collect();
    numbers.filter(|n| !n.is_empty())
}

impl MultiInputElement for TsMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    /// Named request pads (M481): a container mux's inputs are caps-typed slots, so
    /// `video_%u` / `audio_%u` / `sink_%u` each claim the next positional slot (the
    /// track type is read from the input's caps, not its index), so a launch line
    /// can name the pads (`m.video_0` / `m.audio_0`) in any order.
    fn input_pad_index(
        &self,
        _req: &g2g_core::runtime::PadRequest,
        ordinal: usize,
    ) -> Option<usize> {
        (ordinal < self.inputs).then_some(ordinal)
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if stream_type_for(upstream_caps, self.klv_sync).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        // Each pad forwards its stream verbatim (frames carry their own caps);
        // the per-pad stream type is pinned at `configure_pipeline`, which rejects
        // an unsupported caps. `AcceptsAny` is the native muxer-input shape (as in
        // `InterleaveMux`); the legacy intercept-narrowing path is skipped.
        CapsConstraint::AcceptsAny
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(
            Self::output_caps_value(),
        )))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let stream_type =
            stream_type_for(absolute_caps, self.klv_sync).ok_or(G2gError::CapsMismatch)?;
        self.stream_types[input] = Some(stream_type);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(Self::output_caps_value())
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "pat-interval",
                PropKind::Uint,
                "PAT/PMT re-emission interval, milliseconds (0 = once)",
            )
            .with_default("0"),
            // The PAT and PMT are emitted as a pair, so this shares the cadence.
            PropertySpec::new(
                "pmt-interval",
                PropKind::Uint,
                "alias of pat-interval (the tables are emitted together)",
            )
            .with_default("0"),
            PropertySpec::new(
                "pcr-interval",
                PropKind::Uint,
                "PCR insertion interval, ticks of the 90kHz clock",
            )
            .with_default("3600"),
            // GStreamer mpegtsmux takes a pad-name -> program structure here; g2g
            // properties are scalar, so this is one number per input in pad order.
            PropertySpec::new(
                "prog-map",
                PropKind::Str,
                "program number per input, comma separated in pad order (e.g. 1,1,2; default all in program 1)",
            )
            .with_default("1"),
            PropertySpec::new(
                "klv-sync",
                PropKind::Bool,
                "carry KLV metadata as synchronous metadata-in-PES (stream_type 0x15) instead of asynchronous private PES",
            )
            .with_default("false"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "prog-map" => {
                // The program layout is baked into the PSI when the muxer is built,
                // so it can only be set before the first frame.
                if self.mux.is_some() {
                    return Err(PropError::ReadOnly);
                }
                self.program_numbers = parse_prog_map(value.as_str().ok_or(PropError::Type)?)
                    .filter(|n| n.len() == self.inputs)
                    // A map that drops a program whose service was set would write
                    // an SDT without that service; refuse it instead.
                    .filter(|n| self.program_tags.iter().all(|(p, _)| n.contains(p)))
                    .ok_or(PropError::Value)?;
                Ok(())
            }
            "pat-interval" | "pmt-interval" => {
                self.table_interval_ms = value.as_uint().ok_or(PropError::Type)?;
                // If the muxer is already built, update it in place; otherwise the
                // build path applies the stored cadence.
                if let Some(mux) = self.mux.as_mut() {
                    mux.set_table_interval_90khz(self.table_interval_ms.saturating_mul(90));
                }
                Ok(())
            }
            "pcr-interval" => {
                self.pcr_interval_90khz = value.as_uint().ok_or(PropError::Type)?;
                if let Some(mux) = self.mux.as_mut() {
                    mux.set_pcr_interval_90khz(self.pcr_interval_90khz);
                }
                Ok(())
            }
            "klv-sync" => {
                // Like prog-map: the stream types are baked into the PMT when the
                // muxer is built, so this only takes effect before the first frame.
                if self.mux.is_some() {
                    return Err(PropError::ReadOnly);
                }
                self.klv_sync = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "pat-interval" | "pmt-interval" => Some(PropValue::Uint(self.table_interval_ms)),
            "pcr-interval" => Some(PropValue::Uint(self.pcr_interval_90khz)),
            "prog-map" => Some(PropValue::Str(self.prog_map_string())),
            "klv-sync" => Some(PropValue::Bool(self.klv_sync)),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                // M204: buffer the AU; release the globally earliest below.
                PipelinePacket::DataFrame(frame) => self.agg.push(input, frame),
                // M22: a per-input Eos lets the merge release AUs held waiting on
                // this input, and flush its tail; the runner emits the merged Eos.
                PipelinePacket::Eos => self.agg.mark_ended(input),
                // CapsChanged is consumed by the runner's muxer arm; geometry /
                // params do not change the TS framing.
                PipelinePacket::CapsChanged(_) => return Ok(()),
                other => {
                    out.push(other).await?;
                    return Ok(());
                }
            }

            // The PMT needs every stream type, so build the muxer only once all
            // inputs are configured (always true by the first frame: the runner
            // negotiates every input pad before any process call).
            if self.mux.is_none() {
                if self.stream_types.iter().any(|s| s.is_none()) {
                    return Ok(());
                }
                let streams: Vec<(u16, u8)> = self
                    .program_numbers
                    .iter()
                    .zip(&self.stream_types)
                    .map(|(&program, s)| (program, s.expect("all set")))
                    .collect();
                let mut mux = TsMuxer::with_programs(&streams);
                // 90 kHz clock: 90 ticks per millisecond (matches the single-input path).
                mux.set_table_interval_90khz(self.table_interval_ms.saturating_mul(90));
                mux.set_pcr_interval_90khz(self.pcr_interval_90khz);
                if let Some((name, provider)) = service_from_tags(&self.tags) {
                    mux.set_service(&name, &provider);
                }
                for (program, tags) in &self.program_tags {
                    if let Some((name, provider)) = service_from_tags(tags) {
                        // The builder / property setter keep the map and these
                        // programs in step, so the program is always found.
                        let applied = mux.set_program_service(*program, &name, &provider);
                        debug_assert!(applied, "program {program} is in the program map");
                    }
                }
                for (i, track) in self.track_tags.iter().enumerate() {
                    let above = match self.tags_of_program(i) {
                        Some(program) => resolve_tags(&self.tags, program),
                        None => self.tags.clone(),
                    };
                    let effective = resolve_tags(&above, track);
                    if let Some(language) = language_from_tags(&effective) {
                        mux.set_stream_language(i, language);
                    }
                }
                self.mux = Some(mux);
            }

            // Drain every AU now safe to emit, in global PTS order, writing each
            // to its stream's PID.
            while let Some((stream, frame)) = self.agg.take_earliest_by(|f| f.timing.pts_ns) {
                let Some(slice) = frame.domain.as_system_slice() else {
                    return Err(G2gError::UnsupportedDomain);
                };
                let pts_90khz = (frame.timing.pts_ns as u128 * 90_000 / 1_000_000_000) as u64;
                // A DTS rides the PES only for reordered video (dts_ns set and
                // distinct from the PTS); 0 is the unset sentinel, equal adds nothing.
                let dts_90khz = (frame.timing.dts_ns != 0
                    && frame.timing.dts_ns != frame.timing.pts_ns)
                    .then(|| (frame.timing.dts_ns as u128 * 90_000 / 1_000_000_000) as u64);
                let ts = self.mux.as_mut().expect("built above").push_au_on(
                    stream,
                    slice,
                    Some(pts_90khz),
                    dts_90khz,
                );
                let out_frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(ts.into_boxed_slice())),
                    FrameTiming {
                        pts_ns: frame.timing.pts_ns,
                        // one output frame is one access unit's packets, so the
                        // AU's sync flag still describes it and a downstream
                        // segmenter cuts on it. Only the video pads count: every
                        // audio AU is a sync sample, which would cut everywhere.
                        keyframe: frame.timing.keyframe
                            && matches!(
                                self.stream_types[stream],
                                Some(STREAM_TYPE_H264) | Some(STREAM_TYPE_H265)
                            ),
                        ..FrameTiming::default()
                    },
                    self.emitted,
                );
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(out_frame)).await?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{AudioFormat, Dim, PropValue, PushOutcome, Rate, VideoCodec};

    #[derive(Default)]
    struct CaptureSink {
        bytes: Vec<u8>,
        /// Per output frame, its PTS and sync flag.
        timings: Vec<(u64, bool)>,
    }
    impl OutputSink for CaptureSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(f) = packet {
                    self.timings.push((f.timing.pts_ns, f.timing.keyframe));
                    if let Some(s) = f.domain.as_system_slice() {
                        self.bytes.extend_from_slice(s);
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn h264_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Any,
        }
    }

    fn aac_caps() -> Caps {
        Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        }
    }

    fn au_frame(au: Vec<u8>, pts_ns: u64, keyframe: bool) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                keyframe,
                ..FrameTiming::default()
            },
            0,
        ))
    }

    fn h264_frame(au: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        au_frame(au, pts_ns, false)
    }

    /// Count TS packets carrying PID 0 (the PAT) across all output bytes. A TS
    /// packet is 188 bytes, sync 0x47, PID = ((b1 & 0x1F) << 8) | b2.
    fn pat_packet_count(bytes: &[u8]) -> usize {
        bytes
            .chunks(188)
            .filter(|p| {
                p.len() == 188 && p[0] == 0x47 && (((p[1] & 0x1F) as u16) << 8 | p[2] as u16) == 0
            })
            .count()
    }

    /// The `pat-interval` knob is honored on the fan-in muxer (the `name=m` shape),
    /// the same as the single-input `TsMux`: the PAT/PMT are re-emitted at the
    /// cadence instead of only once. Set via `set_property`, the path
    /// `parse_launch` uses; `pmt-interval` shares the cadence.
    #[tokio::test]
    async fn pat_interval_property_re_emits_tables_on_the_fan_in_muxer() {
        let au = |b: u8| alloc::vec![0u8, 0, 0, 1, b, 0xAA, 0xBB];

        // 10 ms cadence, AUs at 0/20/40/60 ms: each AU past the first re-emits the
        // tables, so the PAT appears more than once.
        let mut mux = TsMux::new(1);
        mux.set_property("pat-interval", PropValue::Uint(10))
            .unwrap();
        assert_eq!(mux.get_property("pmt-interval"), Some(PropValue::Uint(10)));
        mux.configure_pipeline(0, &h264_caps()).unwrap();
        let mut sink = CaptureSink::default();
        for i in 0..4u64 {
            mux.process(0, h264_frame(au(0x65), i * 20_000_000), &mut sink)
                .await
                .unwrap();
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        assert!(
            pat_packet_count(&sink.bytes) > 1,
            "PAT re-emitted at the interval"
        );

        // Default (0): the PAT is emitted once up front.
        let mut once = TsMux::new(1);
        once.configure_pipeline(0, &h264_caps()).unwrap();
        let mut sink0 = CaptureSink::default();
        for i in 0..4u64 {
            once.process(0, h264_frame(au(0x65), i * 20_000_000), &mut sink0)
                .await
                .unwrap();
        }
        once.process(0, PipelinePacket::Eos, &mut sink0)
            .await
            .unwrap();
        assert_eq!(
            pat_packet_count(&sink0.bytes),
            1,
            "PAT emitted once by default"
        );
    }

    /// M908: an output frame carries the sync flag of the video AU it packetizes,
    /// so a downstream segmenter (`tsmuxn ! hlssink`) cuts on it. Audio never sets
    /// it: every AAC AU is a sync sample, which would cut everywhere.
    #[tokio::test]
    async fn output_keyframe_flag_comes_from_the_video_pad_only() {
        const FRAME_NS: u64 = 40_000_000;
        let idr = alloc::vec![0u8, 0, 0, 1, 0x65, 0xAA];
        let inter = alloc::vec![0u8, 0, 0, 1, 0x41, 0xBB];
        let adts = alloc::vec![0xFFu8, 0xF1, 0x4C, 0x80, 0x01, 0x00, 0xFC, 0x00];

        let mut mux = TsMux::new(2);
        mux.configure_pipeline(0, &h264_caps()).unwrap();
        mux.configure_pipeline(1, &aac_caps()).unwrap();
        let mut sink = CaptureSink::default();
        // Video AUs every 40 ms with an IDR every third; an audio AU 20 ms after
        // each, flagged a sync sample the way an AAC parser flags them.
        for i in 0..6u64 {
            let key = i % 3 == 0;
            let au = if key { idr.clone() } else { inter.clone() };
            mux.process(0, au_frame(au, i * FRAME_NS, key), &mut sink)
                .await
                .unwrap();
            mux.process(
                1,
                au_frame(adts.clone(), i * FRAME_NS + FRAME_NS / 2, true),
                &mut sink,
            )
            .await
            .unwrap();
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        mux.process(1, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();

        assert_eq!(sink.timings.len(), 12, "one output frame per AU");
        let flagged: Vec<u64> = sink
            .timings
            .iter()
            .filter(|(_, key)| *key)
            .map(|(pts, _)| *pts)
            .collect();
        assert_eq!(
            flagged,
            alloc::vec![0, 3 * FRAME_NS],
            "only the video IDRs are flagged, never an audio AU"
        );
    }
}
