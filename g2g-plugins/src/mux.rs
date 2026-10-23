//! Interleaving muxer (M10). Combines N input streams into one output, ordering
//! every input's frames by presentation timestamp (M204). Negotiation is
//! per-input: each pad accepts and records its own caps; the merged output caps
//! are fixed at construction.
//!
//! Time-ordering (M204): frames are buffered per input in an [`InputAggregator`]
//! and released by smallest `pts_ns` only once every still-contributing input
//! has one queued, so the merged stream is globally PTS-ordered (the
//! `GstAggregator` collect-and-pick-earliest rule). A frame carries its own
//! caps, so reordering is format-safe. Earlier this forwarded frames in arrival
//! order; that interleaved two time-skewed inputs incorrectly. Ordering is by
//! PTS; container muxers that need decode-order (DTS) interleaving key on that
//! instead, a separate concern from this generic interleaver.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::log::Target;
use g2g_core::{
    g2g_error, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    Frame, G2gError, InputAggregator, MultiInputElement, OutputSink, PipelinePacket, VideoCodec,
    ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use crate::aacparse::audio_specific_config_info;
use crate::opusparse::{parse_opus_head, OPUS_RATE_HZ};

// the solver lets any caps onto a fan-in muxer pad and reports no refusal itself
pub(crate) fn refused_input(muxer: &'static str, caps: &Caps) -> G2gError {
    g2g_error!(
        Target::category(muxer),
        "cannot carry {}",
        caps.to_gst_string()
    );
    G2gError::CapsMismatch
}

// a muxer that carries the codec but not this stream beside the ones it already has
pub(crate) fn refused_input_because(muxer: &'static str, caps: &Caps, reason: &str) -> G2gError {
    g2g_error!(
        Target::category(muxer),
        "cannot carry {}: {reason}",
        caps.to_gst_string()
    );
    G2gError::CapsMismatch
}

pub(crate) const CODEC_CONFIG: &str = "codec config";

// what a fan-in muxer reads from a video stream's leading frames to write its track header
pub(crate) fn video_header_config(codec: VideoCodec) -> &'static str {
    match codec {
        VideoCodec::H264 => "SPS and PPS",
        VideoCodec::H265 => "VPS, SPS and PPS",
        VideoCodec::Av1 => "AV1 sequence header",
        _ => CODEC_CONFIG,
    }
}

// what a fan-in muxer reads from an audio stream's leading frames to write its track header
pub(crate) fn audio_header_config(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Aac => "ADTS header or leading AudioSpecificConfig",
        AudioFormat::Vorbis => "Vorbis identification and setup headers",
        AudioFormat::Flac => "fLaC stream header",
        _ => CODEC_CONFIG,
    }
}

pub(crate) fn missing_header_config(muxer: &'static str, input: usize, awaited: &str) -> G2gError {
    g2g_error!(
        Target::category(muxer),
        "input {input} has no {awaited} for its track header"
    );
    G2gError::CapsMismatch
}

// fails when `input` ends with frames still queued behind a track header that never got `awaited`
pub(crate) fn require_header_config_at_end<I, T>(
    muxer: &'static str,
    input: usize,
    inits: &[Option<I>],
    aggregator: &InputAggregator<T>,
    awaited: &str,
) -> Result<(), G2gError> {
    if inits[input].is_some() || aggregator.queued(input) == 0 {
        return Ok(());
    }
    Err(missing_header_config(muxer, input, awaited))
}

// an input that ended with no init sent no frames, so the file carries no track for it
pub(crate) fn every_track_header_ready<I, T>(
    inits: &[Option<I>],
    aggregator: &InputAggregator<T>,
) -> bool {
    inits
        .iter()
        .enumerate()
        .all(|(input, init)| init.is_some() || aggregator.is_ended(input))
}

// the channel count and sample rate an audio track header declares
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AudioTrackParams {
    pub(crate) channels: u8,
    pub(crate) sample_rate: u32,
}

// the caps' channels and rate where known, else the ones the AAC or Opus config states
pub(crate) fn audio_track_params(
    muxer: &'static str,
    format: AudioFormat,
    channels: u8,
    sample_rate: u32,
    config: &[u8],
) -> Result<AudioTrackParams, G2gError> {
    let (config_channels, config_rate) = match format {
        AudioFormat::Aac => match audio_specific_config_info(config) {
            Some(info) => (Some(info.channels), Some(info.sample_rate)),
            None => (None, None),
        },
        // Opus decodes at 48 kHz whatever rate the encoder took in
        AudioFormat::Opus => (
            parse_opus_head(config).map(|(channels, _)| channels),
            Some(OPUS_RATE_HZ),
        ),
        _ => (None, None),
    };
    let channels = (channels != ANY_CHANNELS)
        .then_some(channels)
        .or(config_channels);
    let sample_rate = (sample_rate != ANY_SAMPLE_RATE)
        .then_some(sample_rate)
        .or(config_rate);
    let (Some(channels), Some(sample_rate)) = (channels, sample_rate) else {
        g2g_error!(
            Target::category(muxer),
            "{format:?} stream states no channel count or sample rate"
        );
        return Err(G2gError::CapsMismatch);
    };
    Ok(AudioTrackParams {
        channels,
        sample_rate,
    })
}

/// # Example
///
/// ```no_run
/// use g2g_core::{AudioFormat, Caps, ChannelLayout};
/// use g2g_plugins::mux::InterleaveMux;
///
/// let format = AudioFormat::PcmS16Le;
/// let output = Caps::Audio {
///     format,
///     channels: 2,
///     sample_rate: 48_000,
///     channel_layout: ChannelLayout::UNSPECIFIED,
/// };
/// let element = InterleaveMux::new(2, output);
/// ```
#[derive(Debug)]
pub struct InterleaveMux {
    inputs: usize,
    output: Caps,
    configured: Vec<Option<Caps>>,
    /// Per-input frame buffer; releases the globally earliest-PTS frame once
    /// every contributor has one queued (M204).
    agg: InputAggregator<Frame>,
}

impl InterleaveMux {
    pub fn new(inputs: usize, output: Caps) -> Self {
        assert!(inputs > 0, "InterleaveMux needs at least one input");
        Self {
            inputs,
            output,
            configured: vec![None; inputs],
            agg: InputAggregator::new(inputs),
        }
    }

    /// The caps input pad `input` was configured with, i.e. the result of
    /// that input's independent negotiation. `None` before configuration.
    pub fn input_caps(&self, input: usize) -> Option<&Caps> {
        self.configured.get(input).and_then(|c| c.as_ref())
    }
}

impl MultiInputElement for InterleaveMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Each input negotiates independently; the interleave accepts any
        // per-input caps (frames carry their own caps downstream).
        Ok(upstream_caps.clone())
    }

    /// M18 step 1: per-input wildcard. The interleave forwards frames
    /// straight through tagged with their own per-frame caps; it has
    /// no per-input format constraint. `AcceptsAny` is the native
    /// shape (analogous to `FakeSink`'s migration in M16 5c). Skips
    /// the dynamic intercept callback on the per-input solver path.
    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    /// M18 step 1: the merged output caps are fixed at construction
    /// (`InterleaveMux::new(_, output)`). `Produces` is the native
    /// shape for a static muxer output.
    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(self.output.clone())))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configured[input] = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.output.clone())
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Funnel N-to-1 forwarder",
            "Generic/Muxer",
            "Interleaves N input streams into one output by presentation timestamp; the structural fan-in for text/data streams (the gst `funnel` analog).",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                // M204: buffer the frame; release any now globally-earliest
                // frames (every contributor has a head) in PTS order.
                PipelinePacket::DataFrame(frame) => self.agg.push(input, frame),
                // M22: a per-input Eos is informational (the runner aggregates
                // ends and emits the single merged Eos). M204: mark the input
                // ended so the merge can release frames held waiting on it, and
                // flush everything once the last input ends.
                PipelinePacket::Eos => self.agg.mark_ended(input),
                // Control packets (Segment, Flush, ...) carry no frame to order;
                // forward straight through.
                other => {
                    out.push(other).await?;
                    return Ok(());
                }
            }
            while let Some((_, frame)) = self.agg.take_earliest_by(|f| f.timing.pts_ns) {
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Dim, Rate, RawVideoFormat};

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    #[test]
    fn per_input_constraint_is_wildcard() {
        // M18 step 1: each input pad accepts any per-frame caps
        // because the interleave forwards frames straight through
        // tagged with their own caps. `AcceptsAny` is the native
        // shape.
        let mux = InterleaveMux::new(3, nv12(1920, 1080));
        for idx in 0..3 {
            let c = mux.caps_constraint_as_input(idx);
            assert!(
                matches!(c, CapsConstraint::AcceptsAny),
                "input {idx} should be AcceptsAny, got {c:?}"
            );
        }
    }

    #[test]
    fn output_constraint_is_produces_with_configured_output() {
        // M18 step 1: the merged output is fixed at construction.
        // `Produces(CapsSet::one(...))` is the native shape; the
        // solver fixates it identically to the pre-migration
        // `output_caps().fixate()` call.
        let out = nv12(1280, 720);
        let mux = InterleaveMux::new(2, out.clone());
        let c = mux.caps_constraint_for_output().unwrap();
        match c {
            CapsConstraint::Produces(set) => {
                assert_eq!(set.alternatives(), &[out]);
            }
            other => panic!("expected Produces, got {other:?}"),
        }
    }

    // the error `run` returns and the lines it logs on the category of `M`
    #[cfg(feature = "std")]
    fn logged_failure<M, T>(
        run: impl FnOnce() -> Result<T, G2gError>,
    ) -> (Option<G2gError>, Vec<alloc::string::String>) {
        use g2g_core::log::{self, RingSink};
        // every sink sees every test's records, so two failures logged at once mix
        static ONE_FAILURE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _serial = ONE_FAILURE_AT_A_TIME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let ring = RingSink::new(64);
        let sink = log::add_sink(alloc::boxed::Box::new(ring.clone()));
        let result = run();
        log::remove_sink(sink);
        let category = g2g_core::log::short_type_name::<M>();
        let lines = ring
            .drain()
            .into_iter()
            .filter(|record| record.category == category)
            .map(|record| record.message)
            .collect();
        (result.err(), lines)
    }

    // the refusal log lines for `caps`, configured onto a fresh two-pad `M`
    #[cfg(feature = "std")]
    fn refusal_lines<M: MultiInputElement>(mut mux: M, caps: &Caps) -> Vec<alloc::string::String> {
        let (error, lines) = logged_failure::<M, _>(|| mux.configure_pipeline(1, caps));
        assert_eq!(error, Some(G2gError::CapsMismatch));
        lines
    }

    #[cfg(feature = "std")]
    #[derive(Default)]
    struct CollectSink(Vec<u8>);

    #[cfg(feature = "std")]
    impl OutputSink for CollectSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
                if let Some(bytes) = frame.domain.as_system_slice() {
                    self.0.extend_from_slice(bytes);
                }
            }
            core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
        }
    }

    #[cfg(feature = "std")]
    fn system_frame(bytes: &[u8]) -> PipelinePacket {
        use g2g_core::memory::SystemSlice;
        use g2g_core::{FrameTiming, MemoryDomain};
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into())),
            FrameTiming::default(),
            0,
        ))
    }

    #[cfg(feature = "std")]
    fn compressed_video(codec: g2g_core::VideoCodec) -> Caps {
        Caps::CompressedVideo {
            codec,
            width: Dim::Fixed(16),
            height: Dim::Fixed(16),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        }
    }

    #[cfg(feature = "std")]
    fn stereo(format: AudioFormat) -> Caps {
        Caps::Audio {
            format,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    // the error and log lines of a one-pad `M` fed `access_unit` on caps `caps`, then Eos
    #[cfg(feature = "std")]
    fn end_of_stream_failure<M: MultiInputElement>(
        mut mux: M,
        caps: &Caps,
        access_unit: &[u8],
    ) -> (Option<G2gError>, Vec<alloc::string::String>) {
        mux.configure_pipeline(0, caps)
            .expect("the pad takes its caps");
        let mut sink = CollectSink::default();
        logged_failure::<M, _>(|| {
            g2g_core::runtime::block_on(async {
                mux.process(0, system_frame(access_unit), &mut sink).await?;
                mux.process(0, PipelinePacket::Eos, &mut sink).await
            })
        })
    }

    #[cfg(feature = "std")]
    #[test]
    fn every_fan_in_container_muxer_names_the_caps_it_refuses() {
        let raw = nv12(16, 16);
        let expected = alloc::vec![alloc::format!("cannot carry {}", raw.to_gst_string())];
        assert_eq!(
            refusal_lines(crate::mp4muxn::Mp4MuxN::new(2), &raw),
            expected
        );
        assert_eq!(
            refusal_lines(crate::mkvmuxn::MkvMuxN::new(2), &raw),
            expected
        );
        assert_eq!(refusal_lines(crate::tsmuxn::TsMux::new(2), &raw), expected);
        assert_eq!(
            refusal_lines(crate::flvmuxn::FlvMuxN::new(2), &raw),
            expected
        );
        assert_eq!(
            refusal_lines(crate::oggmuxn::OggMuxN::new(2), &raw),
            expected
        );
        assert_eq!(
            refusal_lines(crate::avimux::AviMuxN::new(2), &raw),
            expected
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_fan_in_muxer_fails_at_eos_on_a_stream_that_never_sent_its_config() {
        use crate::flvmuxn::FlvMuxN;
        use crate::mkvmuxn::MkvMuxN;
        use crate::mp4muxn::Mp4MuxN;
        let video = compressed_video;
        let aac = stereo(AudioFormat::Aac);
        // a non-IDR slice, with no parameter sets ahead of it
        let h264_slice: &[u8] = &[0, 0, 0, 1, 0x41, 0x9A, 0x02];
        // a TRAIL_R slice, with no parameter sets ahead of it
        let h265_slice: &[u8] = &[0, 0, 0, 1, 0x02, 0x01, 0xD0];
        // a temporal delimiter and a frame OBU, with no sequence header
        let av1_frame: &[u8] = &[0x12, 0x00, 0x32, 0x01, 0x00];
        // a raw channel pair element, neither ADTS framed nor a config
        let raw_aac: &[u8] = &[0x21, 0x10, 0x04, 0x60];
        let expected = |config: &str| {
            (
                Some(G2gError::CapsMismatch),
                alloc::vec![alloc::format!(
                    "input 0 has no {config} for its track header"
                )],
            )
        };
        assert_eq!(
            end_of_stream_failure(
                FlvMuxN::new(1),
                &video(g2g_core::VideoCodec::H264),
                h264_slice
            ),
            expected("SPS and PPS")
        );
        assert_eq!(
            end_of_stream_failure(FlvMuxN::new(1), &aac, raw_aac),
            expected("ADTS header or leading AudioSpecificConfig")
        );
        assert_eq!(
            end_of_stream_failure(
                Mp4MuxN::new(1),
                &video(g2g_core::VideoCodec::H265),
                h265_slice
            ),
            expected("VPS, SPS and PPS")
        );
        assert_eq!(
            end_of_stream_failure(Mp4MuxN::new(1), &aac, raw_aac),
            expected("ADTS header or leading AudioSpecificConfig")
        );
        assert_eq!(
            end_of_stream_failure(
                MkvMuxN::new(1),
                &video(g2g_core::VideoCodec::Av1),
                av1_frame
            ),
            expected("AV1 sequence header")
        );
        assert_eq!(
            end_of_stream_failure(MkvMuxN::new(1), &aac, raw_aac),
            expected("ADTS header or leading AudioSpecificConfig")
        );
        // an audio packet, its even packet type marking it as no Vorbis header
        let vorbis_audio: &[u8] = &[0x00, 0x12, 0x34];
        assert_eq!(
            end_of_stream_failure(
                crate::oggmuxn::OggMuxN::new(1),
                &stereo(AudioFormat::Vorbis),
                vorbis_audio
            ),
            expected("Vorbis identification and setup headers")
        );
    }

    // the bytes a two-pad `M` writes from one H.264 IDR while its AAC pad ends with no frames
    #[cfg(feature = "std")]
    fn bytes_beside_an_empty_audio_pad<M: MultiInputElement>(mut mux: M) -> Vec<u8> {
        let idr: &[u8] = &[
            0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1E, 0, 0, 0, 1, 0x68, 0xCE, 0x3C, 0x80, 0, 0, 0, 1,
            0x65, 0x88, 0x84,
        ];
        mux.configure_pipeline(0, &compressed_video(g2g_core::VideoCodec::H264))
            .expect("the video pad takes its caps");
        mux.configure_pipeline(1, &stereo(AudioFormat::Aac))
            .expect("the audio pad takes its caps");
        let mut sink = CollectSink::default();
        g2g_core::runtime::block_on(async {
            mux.process(1, PipelinePacket::Eos, &mut sink).await?;
            mux.process(0, system_frame(idr), &mut sink).await?;
            mux.process(0, PipelinePacket::Eos, &mut sink).await
        })
        .expect("the empty pad has no track to fail");
        sink.0
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_fan_in_muxer_writes_no_track_for_an_input_that_ended_with_no_frames() {
        // the byte after the `FLV` signature and version
        const FLV_HEADER_FLAGS_AT: usize = 4;
        let flv = bytes_beside_an_empty_audio_pad(crate::flvmuxn::FlvMuxN::new(2));
        assert_eq!(
            flv.get(FLV_HEADER_FLAGS_AT),
            Some(&crate::flv::FLV_FLAG_VIDEO)
        );
        let mkv = bytes_beside_an_empty_audio_pad(crate::mkvmuxn::MkvMuxN::new(2));
        let mut demuxer = crate::matroska::MatroskaDemuxer::new();
        demuxer.push_data(&mkv);
        assert_eq!(demuxer.tracks().len(), 1);
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_muxer_names_the_stream_it_has_no_room_for() {
        let h264 = Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Fixed(16),
            height: Dim::Fixed(16),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        let expected = |reason: &str| {
            alloc::vec![alloc::format!(
                "cannot carry {}: {reason}",
                h264.to_gst_string()
            )]
        };
        let mut flv = crate::flvmuxn::FlvMuxN::new(2);
        flv.configure_pipeline(0, &h264)
            .expect("the first video track");
        assert_eq!(
            refusal_lines(flv, &h264),
            expected("FLV holds one video and one audio track")
        );
        let mut avi = crate::avimux::AviMuxN::new(2);
        avi.configure_pipeline(0, &h264)
            .expect("the first video stream");
        assert_eq!(
            refusal_lines(avi, &h264),
            expected("AVI holds one video stream")
        );
        assert_eq!(
            refusal_lines(crate::mp4muxn::Mp4MuxN::new(2).with_cmaf(true), &h264),
            expected("a CMAF track file holds one stream, fragmented")
        );
    }

    #[test]
    fn an_unknown_opus_channel_count_and_rate_come_from_the_opus_head() {
        let channels = 6;
        let head = crate::opusparse::synth_opus_head(channels, OPUS_RATE_HZ, 0);
        assert_eq!(
            audio_track_params(
                "test",
                AudioFormat::Opus,
                ANY_CHANNELS,
                ANY_SAMPLE_RATE,
                &head
            ),
            Ok(AudioTrackParams {
                channels,
                sample_rate: OPUS_RATE_HZ
            })
        );
    }

    #[test]
    fn stated_caps_win_over_the_config_and_a_stream_stating_neither_fails() {
        let head = crate::opusparse::synth_opus_head(6, OPUS_RATE_HZ, 0);
        let stated = AudioTrackParams {
            channels: 2,
            sample_rate: 24_000,
        };
        assert_eq!(
            audio_track_params(
                "test",
                AudioFormat::Opus,
                stated.channels,
                stated.sample_rate,
                &head
            ),
            Ok(stated)
        );
        assert_eq!(
            audio_track_params("test", AudioFormat::Mp3, 2, ANY_SAMPLE_RATE, &[]),
            Err(G2gError::CapsMismatch)
        );
    }
}
