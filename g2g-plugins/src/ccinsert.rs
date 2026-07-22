//! Closed-caption insertion element (M432, `no_std`): the encode-side counterpart
//! of [`CcExtract`](crate::ccextract). It takes a compressed H.264 / H.265 access
//! unit stream and a timed `Caps::Text{Utf8}` cue stream, encodes the cues to
//! CEA-608 `cc_data` with [`Cc608Enc`](crate::cea::Cc608Enc), and writes a `GA94`
//! caption SEI ([`build_cc_sei`](crate::cea::build_cc_sei)) into each access unit,
//! so a downstream decoder / TV recovers the captions. Compressed video in (plus a
//! cue pad), compressed video out, the inverse of `CcExtract`'s extract.
//!
//! The caption channel carries two `cc_data` bytes per video frame, and the video
//! provides that frame clock: each access unit drains the next pair from the
//! encoder (null padding when idle). A cue is queued as a pop-on sequence when it
//! arrives (the two pads are merged by PTS, so a cue lands just before the frame
//! it covers), and an erase is queued when the displayed caption's window ends.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::g2g_warn;
use g2g_core::log::{short_type_name, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, MemoryDomain,
    MultiInputElement, OutputSink, PipelinePacket, Rate, VideoCodec,
};

use crate::cea::{build_cc_sei, Cc608Enc, Cc708Enc, CcTriple};
use crate::subparse::Cue;

use core::future::Future;
use core::pin::Pin;

/// The caption encoder behind a [`CcInsert`]: CEA-608 (cc_type 0 byte pairs) or
/// CEA-708 DTVCC (cc_type 2/3 triples). One caption triple is drained per video
/// frame whichever it is.
#[derive(Debug)]
enum CcEncoder {
    Cea608(Cc608Enc),
    Cea708(Cc708Enc),
}

impl CcEncoder {
    fn pending(&self) -> bool {
        match self {
            CcEncoder::Cea608(e) => e.pending(),
            CcEncoder::Cea708(e) => e.pending(),
        }
    }
    fn push_cue(&mut self, cue: &Cue) {
        match self {
            CcEncoder::Cea608(e) => e.push_cue(cue),
            CcEncoder::Cea708(e) => e.push_cue(cue),
        }
    }
    fn erase(&mut self) {
        match self {
            CcEncoder::Cea608(e) => e.erase(),
            CcEncoder::Cea708(e) => e.erase(),
        }
    }
    /// The next caption triple to carry this frame (CEA-608 is always cc_type 0).
    fn next_triple(&mut self) -> CcTriple {
        match self {
            CcEncoder::Cea608(e) => {
                let (b0, b1) = e.next_pair();
                CcTriple { cc_type: 0, b0, b1 }
            }
            CcEncoder::Cea708(e) => {
                let (cc_type, b0, b1) = e.next_triple();
                CcTriple { cc_type, b0, b1 }
            }
        }
    }
}

/// Insert closed captions, encoded from a cue stream, into a compressed
/// H.264 / H.265 access-unit stream's SEI. A [`MultiInputElement`]: input 0 is the
/// video access units (and the merged output), input 1 the timed text cues.
/// Encodes CEA-608 by default ([`new`](CcInsert::new)) or CEA-708
/// ([`cea708`](CcInsert::cea708)).
#[derive(Debug)]
pub struct CcInsert {
    enc: CcEncoder,
    /// Input codec, fixed at `configure(VIDEO)`; selects the SEI NAL framing.
    codec: Option<VideoCodec>,
    /// The negotiated video caps, the merged output (it follows the video pad).
    video_caps: Option<Caps>,
    /// End PTS of the currently displayed caption; when a video access unit reaches
    /// it (and no newer cue has superseded it) an erase is queued. `None` when no
    /// caption is shown.
    erase_at: Option<u64>,
    /// Cues received on the cue pad, and whether any caption byte was actually
    /// written into a video access unit. If cues arrived but nothing was emitted
    /// (the video carried no usable PTS, so the PTS merge delivered every cue after
    /// the last frame), the captions are silently lost; a warning is logged once at
    /// end of stream. See [`Self::warn_if_dropped`].
    cues_received: u64,
    caption_emitted: bool,
    warned: bool,
    /// Runner-assigned instance name for logging.
    instance: Option<alloc::string::String>,
}

impl Default for CcInsert {
    fn default() -> Self {
        Self::with_encoder(CcEncoder::Cea608(Cc608Enc::new()))
    }
}

impl CcInsert {
    /// Input pad indices: compressed video on 0, the text-cue stream on 1.
    const VIDEO: usize = 0;
    const CUE: usize = 1;

    /// A CEA-608 caption inserter (channel CC1).
    pub fn new() -> Self {
        Self::default()
    }

    /// A CEA-708 (DTVCC) caption inserter writing `service` (1 = primary).
    pub fn cea708(service: u8) -> Self {
        Self::with_encoder(CcEncoder::Cea708(Cc708Enc::for_service(service)))
    }

    fn with_encoder(enc: CcEncoder) -> Self {
        Self {
            enc,
            codec: None,
            video_caps: None,
            erase_at: None,
            cues_received: 0,
            caption_emitted: false,
            warned: false,
            instance: None,
        }
    }

    /// Warn once at end of stream if cues were received but no caption byte ever
    /// reached a video access unit, the silent-drop symptom of an untimed video
    /// source (the PTS merge delivers every cue after the last frame, so there is
    /// nothing left to carry it). Called on each pad's `Eos`, so it fires whichever
    /// pad ends last.
    fn warn_if_dropped(&mut self) {
        if !self.warned && self.cues_received > 0 && !self.caption_emitted {
            self.warned = true;
            g2g_warn!(
                self,
                "{} cue(s) received but no caption was embedded: the video source carried no usable timestamps, so the captions could not be paced against it (author from a timed source, e.g. an MP4/MKV, not a raw elementary stream)",
                self.cues_received
            );
        }
    }

    /// The compressed-video caps the video pad accepts (its output follows it).
    fn video_alternatives() -> CapsSet {
        CapsSet::from_alternatives(Vec::from([
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            },
            Caps::CompressedVideo {
                codec: VideoCodec::H265,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            },
        ]))
    }

    /// Rewrite one access unit's bytes with the caption SEI inserted before the
    /// first VCL slice NAL (spec position: after any AUD / parameter sets), and
    /// emit it preserving the frame's timing / keyframe flag.
    async fn emit_au(&mut self, frame: Frame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let Some(codec) = self.codec else {
            return Err(G2gError::NotConfigured);
        };
        let Some(au) = frame.domain.as_system_slice() else {
            // A non-system buffer carries no walkable bitstream; pass it through.
            return out.push(PipelinePacket::DataFrame(frame)).await.map(|_| ());
        };
        // Drain this frame's caption triple (one per frame) and wrap it in a SEI.
        // Pending-before-drain marks a real caption byte (vs idle padding).
        let real = self.enc.pending();
        let triple = self.enc.next_triple();
        if real {
            self.caption_emitted = true;
        }
        let sei = build_cc_sei(&[triple], codec);

        let mut bytes = Vec::with_capacity(au.len() + sei.len());
        match vcl_start(au, codec) {
            Some(off) => {
                bytes.extend_from_slice(&au[..off]);
                bytes.extend_from_slice(&sei);
                bytes.extend_from_slice(&au[off..]);
            }
            // No VCL slice (e.g. a parameter-set-only AU): leave it unchanged.
            None => bytes.extend_from_slice(au),
        }
        let new = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            frame.timing,
            frame.sequence,
        );
        out.push(PipelinePacket::DataFrame(new)).await.map(|_| ())
    }
}

/// Byte offset of the start code of the first VCL slice NAL in an Annex-B access
/// unit, or `None` if there is none. H.264 VCL NAL types are 1..=5; H.265 VCL types
/// are 0..=31.
fn vcl_start(au: &[u8], codec: VideoCodec) -> Option<usize> {
    let mut i = 0usize;
    while i + 3 < au.len() {
        let sc = if au[i] == 0 && au[i + 1] == 0 && au[i + 2] == 1 {
            3
        } else if au[i] == 0 && au[i + 1] == 0 && au[i + 2] == 0 && au[i + 3] == 1 {
            4
        } else {
            i += 1;
            continue;
        };
        let hdr = *au.get(i + sc)?;
        let is_vcl = match codec {
            VideoCodec::H265 => ((hdr >> 1) & 0x3F) < 32,
            _ => (1..=5).contains(&(hdr & 0x1F)),
        };
        if is_vcl {
            return Some(i);
        }
        i += sc + 1;
    }
    None
}

impl MultiInputElement for CcInsert {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        2
    }

    /// Merge the video and cue pads by PTS, so a cue is queued just before the
    /// access units that carry its caption bytes.
    fn input_pts_ordered(&self) -> bool {
        true
    }

    /// The merged output is the video pad's compressed stream (with SEI inserted),
    /// so the solver derives the output caps from pad 0.
    fn output_follows_input(&self) -> Option<usize> {
        Some(Self::VIDEO)
    }

    fn intercept_caps(&self, input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match input {
            Self::VIDEO
                if matches!(
                    upstream_caps,
                    Caps::CompressedVideo {
                        codec: VideoCodec::H264 | VideoCodec::H265,
                        ..
                    }
                ) =>
            {
                Ok(upstream_caps.clone())
            }
            Self::CUE
                if matches!(
                    upstream_caps,
                    Caps::Text {
                        format: g2g_core::TextFormat::Utf8
                    }
                ) =>
            {
                Ok(upstream_caps.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_> {
        match input {
            Self::CUE => CapsConstraint::Accepts(CapsSet::one(Caps::Text {
                format: g2g_core::TextFormat::Utf8,
            })),
            // VIDEO (and any out-of-range pad, defensively): compressed H.264 / H.265.
            _ => CapsConstraint::Accepts(Self::video_alternatives()),
        }
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        match input {
            Self::VIDEO => match absolute_caps {
                Caps::CompressedVideo {
                    codec: codec @ (VideoCodec::H264 | VideoCodec::H265),
                    ..
                } => {
                    self.codec = Some(*codec);
                    self.video_caps = Some(absolute_caps.clone());
                    Ok(ConfigureOutcome::Accepted)
                }
                _ => Err(G2gError::CapsMismatch),
            },
            Self::CUE => match absolute_caps {
                Caps::Text {
                    format: g2g_core::TextFormat::Utf8,
                } => Ok(ConfigureOutcome::Accepted),
                _ => Err(G2gError::CapsMismatch),
            },
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.video_caps.clone().ok_or(G2gError::NotConfigured)
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match input {
                Self::VIDEO => match packet {
                    PipelinePacket::DataFrame(frame) => {
                        // Erase the displayed caption once its window has elapsed and
                        // no newer cue has replaced it.
                        if let Some(end) = self.erase_at {
                            if frame.timing.pts_ns >= end {
                                self.enc.erase();
                                self.erase_at = None;
                            }
                        }
                        self.emit_au(frame, out).await
                    }
                    // Forward stream control as-is; the runner aggregates the per-pad
                    // Eos, so the element does not forward it.
                    PipelinePacket::Eos => {
                        self.warn_if_dropped();
                        Ok(())
                    }
                    other => out.push(other).await.map(|_| ()),
                },
                // The cue pad (and any other pad, defensively, though there are two).
                _ => {
                    if let PipelinePacket::Eos = packet {
                        self.warn_if_dropped();
                    }
                    if let PipelinePacket::DataFrame(frame) = packet {
                        if let Some(slice) = frame.domain.as_system_slice() {
                            let text = String::from_utf8_lossy(slice).into_owned();
                            let start = frame.timing.pts_ns;
                            let end = start.saturating_add(frame.timing.duration_ns);
                            // Recover the cue placement from frame-meta (M406) when
                            // present; default otherwise (and on the ZST baseline).
                            #[cfg(feature = "metadata")]
                            let settings = frame
                                .meta
                                .get::<crate::subparse::TextCueMeta>()
                                .map(|m| m.settings)
                                .unwrap_or_default();
                            #[cfg(not(feature = "metadata"))]
                            let settings = crate::subparse::CueSettings::default();
                            let cue = crate::subparse::Cue {
                                start_ns: start,
                                end_ns: end,
                                text,
                                settings,
                            };
                            self.enc.push_cue(&cue);
                            self.cues_received += 1;
                            // The new pop-on caption supersedes any shown one; erase
                            // when this cue's window ends.
                            self.erase_at = Some(end);
                        }
                    }
                    // Cue pad packets do not drive the video output; nothing forwarded.
                    Ok(())
                }
            }
        })
    }
}

impl LogSource for CcInsert {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.instance.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cea::{extract_cc_data, Cea608, Cea708};
    use alloc::vec;
    use g2g_core::{FrameTiming, PushOutcome, TextFormat};

    #[derive(Default)]
    struct RecordingSink {
        aus: Vec<Frame>,
    }
    impl OutputSink for RecordingSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(f) = packet {
                    self.aus.push(f);
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn h264_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    /// A minimal Annex-B access unit: one VCL IDR slice NAL (type 5) with a little
    /// payload, no captions.
    fn plain_au() -> Vec<u8> {
        vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00]
    }

    fn video_frame(pts: u64) -> PipelinePacket {
        let f = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(plain_au().into_boxed_slice())),
            FrameTiming {
                pts_ns: pts,
                ..Default::default()
            },
            0,
        );
        PipelinePacket::DataFrame(f)
    }

    fn cue_frame(text: &str, start: u64, dur: u64) -> PipelinePacket {
        let payload = text.as_bytes().to_vec().into_boxed_slice();
        let f = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(payload)),
            FrameTiming {
                pts_ns: start,
                duration_ns: dur,
                ..Default::default()
            },
            0,
        );
        PipelinePacket::DataFrame(f)
    }

    #[tokio::test]
    async fn round_trips_a_caption_through_insert_then_extract() {
        // Encode a cue into the SEI of a run of access units, then extract + decode
        // the output and recover the text: CcInsert is the inverse of CcExtract.
        let mut el = CcInsert::new();
        el.configure_pipeline(CcInsert::VIDEO, &h264_caps())
            .unwrap();
        el.configure_pipeline(
            CcInsert::CUE,
            &Caps::Text {
                format: TextFormat::Utf8,
            },
        )
        .unwrap();
        let mut sink = RecordingSink::default();

        // Cue at t=0 for ~1s, then a run of frames at 33 ms to carry the cc_data.
        el.process(
            CcInsert::CUE,
            cue_frame("HELLO", 0, 1_000_000_000),
            &mut sink,
        )
        .await
        .unwrap();
        let frame_dur = 33_000_000u64;
        for n in 0..40u64 {
            el.process(CcInsert::VIDEO, video_frame(n * frame_dur), &mut sink)
                .await
                .unwrap();
        }
        assert_eq!(sink.aus.len(), 40, "every access unit is forwarded");

        // The SEI carries the captions: extract + decode the output stream.
        let mut dec = Cea608::new();
        for f in &sink.aus {
            let Some(s) = f.domain.as_system_slice() else {
                continue;
            };
            for t in extract_cc_data(s, VideoCodec::H264) {
                if t.cc_type == 0 {
                    dec.push_pair(t.b0, t.b1, f.timing.pts_ns);
                }
            }
        }
        dec.flush(100 * frame_dur);
        let cues = dec.take_cues();
        assert_eq!(cues.len(), 1, "the inserted caption is recovered");
        assert_eq!(cues[0].text, "HELLO");
    }

    #[test]
    fn inserts_sei_before_the_first_vcl_slice() {
        // The SEI NAL must precede the VCL slice: an AUD-led AU keeps the AUD first.
        let mut au = vec![0x00, 0x00, 0x00, 0x01, 0x09, 0x10]; // AUD (type 9)
        au.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x65, 0x88]); // IDR slice (type 5)
        let off = vcl_start(&au, VideoCodec::H264).unwrap();
        assert_eq!(off, 6, "first VCL slice starts after the 6-byte AUD NAL");
    }

    #[tokio::test]
    async fn requires_configure() {
        let mut el = CcInsert::new();
        let mut sink = RecordingSink::default();
        // The video pad before configure is NotConfigured (no codec for the SEI).
        assert!(el
            .process(CcInsert::VIDEO, video_frame(0), &mut sink)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn round_trips_a_708_caption_through_insert_then_extract() {
        // The CEA-708 mode authors DTVCC into the SEI; extract + decode the output
        // through the 708 decoder and recover the text.
        let mut el = CcInsert::cea708(1);
        el.configure_pipeline(CcInsert::VIDEO, &h264_caps())
            .unwrap();
        el.configure_pipeline(
            CcInsert::CUE,
            &Caps::Text {
                format: TextFormat::Utf8,
            },
        )
        .unwrap();
        let mut sink = RecordingSink::default();

        el.process(
            CcInsert::CUE,
            cue_frame("HI 708", 0, 1_000_000_000),
            &mut sink,
        )
        .await
        .unwrap();
        let frame_dur = 33_000_000u64;
        for n in 0..40u64 {
            el.process(CcInsert::VIDEO, video_frame(n * frame_dur), &mut sink)
                .await
                .unwrap();
        }

        let mut dec = Cea708::new();
        for f in &sink.aus {
            let Some(s) = f.domain.as_system_slice() else {
                continue;
            };
            for t in extract_cc_data(s, VideoCodec::H264) {
                dec.push_triple(t.cc_type, t.b0, t.b1, f.timing.pts_ns);
            }
        }
        dec.flush(100 * frame_dur);
        let cues = dec.take_cues();
        assert_eq!(cues.len(), 1, "the inserted 708 caption is recovered");
        assert_eq!(cues[0].text, "HI 708");
    }

    #[tokio::test]
    async fn untimed_video_drops_caption_and_flags_it() {
        // The untimed-video failure: the cue arrives only after every video frame
        // (as a pts=0 source makes the PTS merge do), so it is never carried. The
        // output has no real captions, and the element flags the drop rather than
        // emitting a caption-free stream silently.
        let mut el = CcInsert::new();
        el.configure_pipeline(CcInsert::VIDEO, &h264_caps())
            .unwrap();
        el.configure_pipeline(
            CcInsert::CUE,
            &Caps::Text {
                format: TextFormat::Utf8,
            },
        )
        .unwrap();
        let mut sink = RecordingSink::default();

        // All video frames first (no cue yet -> null padding), then the cue, then Eos.
        for n in 0..5u64 {
            el.process(CcInsert::VIDEO, video_frame(n * 33_000_000), &mut sink)
                .await
                .unwrap();
        }
        el.process(
            CcInsert::CUE,
            cue_frame("LATE", 0, 1_000_000_000),
            &mut sink,
        )
        .await
        .unwrap();
        el.process(CcInsert::VIDEO, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        el.process(CcInsert::CUE, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();

        // No caption byte reached any access unit (every SEI is a null pair).
        for f in &sink.aus {
            let Some(s) = f.domain.as_system_slice() else {
                continue;
            };
            assert!(extract_cc_data(s, VideoCodec::H264)
                .iter()
                .all(|t| (t.b0 & 0x7F) == 0));
        }
        assert!(el.warned, "the silent caption drop is flagged");
    }
}
