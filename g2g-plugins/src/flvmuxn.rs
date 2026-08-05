//! Multi-track FLV multiplexer element (M296): a video + audio elementary stream
//! in (`Caps::CompressedVideo{H264}` + `Caps::Audio{Aac}`), one FLV byte stream
//! out. The A/V analog of the single-track [`crate::flvmux::FlvMux`], so a muxed
//! recording carries video and audio together:
//!
//! ```text
//! videotestsrc ! x264enc ! flvmux name=m
//! audiotestsrc ! avenc_aac ! m.
//! m. ! filesink location=av.flv
//! ```
//!
//! A [`MultiInputElement`]: each pad takes one elementary stream (FLV implies the
//! track by tag type, so the muxer routes by pad codec, not pad index), and access
//! units interleave by presentation timestamp via the M204 [`InputAggregator`]
//! merge before being written as FLV tags. Unlike the single-track muxer, the
//! sequence-header tags a player needs are written up front, captured in-band from
//! the first access unit: the video track's `avcC` record from the parameter sets
//! in the first IDR, the audio track's `AudioSpecificConfig` from the first ADTS
//! header (the AAC bytes are written de-ADTS'd, video NALUs AVCC length-prefixed).
//!
//! Reachable from the `gst-launch` fan-in syntax: registered as the `flvmux` muxer
//! in `default_registry`, so >1 input link builds this element (a single input
//! builds the single-track [`crate::flvmux::FlvMux`]). Scope: FLV's one-video +
//! one-audio model over the codecs [`crate::flvmux::FlvMux`] accepts; a second pad
//! of either kind is rejected. The legacy Flash codecs have no sequence header, so
//! their tracks are ready to write from the first access unit.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError,
    InputAggregator, MemoryDomain, MultiInputElement, OutputSink, PipelinePacket, VideoCodec,
};

use crate::flv::{FlvCodec, FlvMuxer, FlvTrack};
use crate::fmp4mux::{avcc_record, avcc_sample, is_keyframe_nal, parameter_sets, split_annexb};
use crate::mp4muxn::{asc_from_adts, strip_adts};

/// A track's init, captured from its first access unit: the parameter sets
/// (video) or AudioSpecificConfig (audio) the FLV sequence header needs. The
/// legacy Flash codecs have no sequence header, so theirs is empty and ready
/// immediately.
#[derive(Debug, Clone)]
enum TrackInit {
    Video {
        param_sets: Vec<Vec<u8>>,
    },
    Audio {
        asc: Vec<u8>,
    },
    /// A codec with no sequence header to wait for.
    None,
}

/// Muxes a video + audio stream into one FLV byte stream, PTS-ordered.
#[derive(Debug)]
pub struct FlvMuxN {
    inputs: usize,
    /// Per-pad codec, learned at configure.
    kinds: Vec<Option<FlvCodec>>,
    /// The audio pad's negotiated layout, which MP3's tag flags declare.
    audio_params: Option<(u8, u32)>,
    /// Per-pad track init, captured from the first AU.
    inits: Vec<Option<TrackInit>>,
    agg: InputAggregator<Frame>,
    /// Built lazily once every track has its init (the sequence headers need it).
    mux: Option<FlvMuxer>,
    emitted: u64,
}

impl FlvMuxN {
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "FlvMuxN needs at least one input");
        Self {
            inputs,
            kinds: alloc::vec![None; inputs],
            audio_params: None,
            inits: alloc::vec![None; inputs],
            agg: InputAggregator::new(inputs),
            mux: None,
            emitted: 0,
        }
    }

    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output_caps_value() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Flv,
        }
    }

    /// The FLV codec an input pad's caps carries, or `None` if FLV cannot carry
    /// it (the same set the single-track muxer accepts).
    fn pad_kind_for(caps: &Caps) -> Option<FlvCodec> {
        crate::flvmux::FlvMux::codec_for(caps)
    }

    /// True once every pad has its init captured (the sequence headers need them).
    fn all_inits_ready(&self) -> bool {
        self.inits.iter().all(|i| i.is_some())
    }

    /// Capture a pad's track init from its first access unit, if not already set.
    fn capture_init(&mut self, input: usize, au: &[u8]) {
        if self.inits[input].is_some() {
            return;
        }
        match self.kinds[input] {
            Some(FlvCodec::H264) => {
                let nalus = split_annexb(au);
                // Parameter sets only ride the IDR; wait for the keyframe.
                if let Ok(param_sets) = parameter_sets(VideoCodec::H264, &nalus) {
                    let owned: Vec<Vec<u8>> = param_sets.iter().map(|s| s.to_vec()).collect();
                    self.inits[input] = Some(TrackInit::Video { param_sets: owned });
                }
            }
            Some(FlvCodec::Aac) => {
                if let Some(asc) = asc_from_adts(au) {
                    self.inits[input] = Some(TrackInit::Audio { asc: asc.to_vec() });
                }
            }
            Some(_) => self.inits[input] = Some(TrackInit::None),
            None => {}
        }
    }

    /// Build the A/V muxer from the captured inits (video `avcC` record + AAC ASC).
    fn build_mux(&self) -> FlvMuxer {
        let mut video_config = Vec::new();
        let mut audio_config = Vec::new();
        for init in self.inits.iter().flatten() {
            match init {
                TrackInit::Video { param_sets } => {
                    let refs: Vec<&[u8]> = param_sets.iter().map(|v| v.as_slice()).collect();
                    video_config = avcc_record(&refs);
                }
                TrackInit::Audio { asc } => audio_config = asc.clone(),
                TrackInit::None => {}
            }
        }
        let of_track = |t| {
            self.kinds
                .iter()
                .flatten()
                .copied()
                .find(|c| c.track() == t)
        };
        let mut mux = FlvMuxer::new_av(
            of_track(FlvTrack::Video).unwrap_or(FlvCodec::H264),
            video_config,
            of_track(FlvTrack::Audio).unwrap_or(FlvCodec::Aac),
            audio_config,
        );
        if let Some((channels, sample_rate)) = self.audio_params {
            mux.set_audio_params(channels, sample_rate);
        }
        mux
    }

    /// Emit one access unit as its track's FLV tag.
    async fn emit_au(
        &mut self,
        input: usize,
        frame: Frame,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let Some(slice) = frame.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        let pts_ms = (frame.timing.pts_ns / 1_000_000) as u32;
        let mux = self.mux.as_mut().ok_or(G2gError::NotConfigured)?;
        let bytes = match self.kinds[input] {
            Some(FlvCodec::H264) => {
                let nalus = split_annexb(slice);
                let keyframe = nalus.iter().any(|n| is_keyframe_nal(VideoCodec::H264, n));
                let (dts_ms, cts_ms) = FlvMuxer::video_tag_timing(&frame.timing);
                mux.push_video(&avcc_sample(&nalus), dts_ms, cts_ms, keyframe)
            }
            // The legacy Flash video codecs go into a tag as they arrive, their
            // keyframe flag coming from the producer rather than the bitstream.
            Some(c) if c.track() == FlvTrack::Video => {
                let (dts_ms, cts_ms) = FlvMuxer::video_tag_timing(&frame.timing);
                mux.push_video(slice, dts_ms, cts_ms, frame.timing.keyframe)
            }
            // AAC access units are raw frames once the ADTS header is stripped;
            // MP3 and Speex frames need no unwrapping.
            Some(FlvCodec::Aac) => mux.push_audio(strip_adts(slice), pts_ms),
            _ => mux.push_audio(slice, pts_ms),
        };

        let out_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming {
                pts_ns: frame.timing.pts_ns,
                ..FrameTiming::default()
            },
            self.emitted,
        );
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(out_frame)).await?;
        Ok(())
    }
}

impl MultiInputElement for FlvMuxN {
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
        if Self::pad_kind_for(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
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
        let kind = Self::pad_kind_for(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        // FLV carries at most one video and one audio track; reject a duplicate.
        if self
            .kinds
            .iter()
            .enumerate()
            .any(|(i, k)| i != input && k.is_some_and(|k| k.track() == kind.track()))
        {
            return Err(G2gError::CapsMismatch);
        }
        if let Caps::Audio {
            channels,
            sample_rate,
            ..
        } = absolute_caps
        {
            self.audio_params = Some((*channels, *sample_rate));
        }
        self.kinds[input] = Some(kind);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(Self::output_caps_value())
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if let Some(s) = frame.domain.as_system_slice() {
                        self.capture_init(input, s);
                    }
                    self.agg.push(input, frame);
                }
                PipelinePacket::Eos => self.agg.mark_ended(input),
                // CapsChanged is consumed by the runner's muxer arm; the sequence
                // headers are fixed from the first AU's in-band init.
                PipelinePacket::CapsChanged(_) => return Ok(()),
                // A per-input `Segment` maps that stream to running time; a muxed
                // container carries its own timestamps, so it is consumed rather
                // than forwarded into the byte stream.
                PipelinePacket::Segment(_) => return Ok(()),
                other => {
                    out.push(other).await?;
                    return Ok(());
                }
            }

            // Hold every AU until all tracks have their init (the sequence headers
            // are written before any media frame).
            if !self.all_inits_ready() {
                return Ok(());
            }
            if self.mux.is_none() {
                self.mux = Some(self.build_mux());
            }
            while let Some((track, frame)) = self.agg.take_earliest_by(|f| f.timing.pts_ns) {
                self.emit_au(track, frame, out).await?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flv::{FlvDemuxer, FlvTrack as DemuxTrack};
    use g2g_core::{AudioFormat, Dim};

    fn h264_idr() -> Vec<u8> {
        let mut au = Vec::new();
        for nal in [
            alloc::vec![0x67u8, 0x42, 0x00, 0x1E],
            alloc::vec![0x68u8, 0xCE, 0x3C, 0x80],
            alloc::vec![0x65u8, 0x88, 0x84],
        ] {
            au.extend_from_slice(&[0, 0, 0, 1]);
            au.extend_from_slice(&nal);
        }
        au
    }

    fn aac_adts() -> Vec<u8> {
        alloc::vec![0xFF, 0xF1, 0x4C, 0x80, 0x00, 0x1F, 0xFC, 0xAB, 0xCD]
    }

    fn video_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: g2g_core::Rate::Any,
        }
    }

    fn audio_caps() -> Caps {
        Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        }
    }

    fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            0,
        ))
    }

    #[derive(Default)]
    struct CaptureSink {
        bytes: Vec<u8>,
    }
    impl OutputSink for CaptureSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<g2g_core::PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(f) = packet {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.bytes.extend_from_slice(s);
                    }
                }
                Ok(g2g_core::PushOutcome::Accepted)
            })
        }
    }

    #[tokio::test]
    async fn av_streams_mux_into_one_flv_with_sequence_headers() {
        let mut mux = FlvMuxN::new(2);
        mux.configure_pipeline(0, &video_caps()).unwrap();
        mux.configure_pipeline(1, &audio_caps()).unwrap();

        let mut sink = CaptureSink::default();
        mux.process(0, frame(h264_idr(), 0), &mut sink)
            .await
            .unwrap();
        mux.process(1, frame(aac_adts(), 0), &mut sink)
            .await
            .unwrap();
        mux.process(0, frame(h264_idr(), 33_000_000), &mut sink)
            .await
            .unwrap();
        mux.process(1, frame(aac_adts(), 21_000_000), &mut sink)
            .await
            .unwrap();
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        mux.process(1, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();

        // The byte stream is FLV with both tracks present (flags bit0|bit2 = 0x05).
        assert_eq!(&sink.bytes[0..3], b"FLV");
        assert_eq!(sink.bytes[4], 0x05, "audio + video present flag");
        assert!(mux.emitted() >= 4, "all four media frames muxed");

        // The demuxer recovers the media access units (it skips the sequence
        // headers); two video + two audio, the audio de-ADTS'd to its 2 bytes.
        let mut d = FlvDemuxer::new();
        d.push_data(&sink.bytes);
        let units = d.take_units();
        let video: Vec<_> = units
            .iter()
            .filter(|u| u.track() == DemuxTrack::Video)
            .collect();
        let audio: Vec<_> = units
            .iter()
            .filter(|u| u.track() == DemuxTrack::Audio)
            .collect();
        assert_eq!(
            video.len(),
            2,
            "two video media frames (sequence header skipped)"
        );
        assert_eq!(audio.len(), 2, "two audio media frames");
        assert_eq!(
            audio[0].data,
            alloc::vec![0xAB, 0xCD],
            "AAC payload, ADTS stripped"
        );
        // The recovered video AU is AVCC (length-prefixed), so it carries no
        // Annex-B start code.
        assert!(
            !video[0].data.windows(4).any(|w| w == [0, 0, 0, 1]),
            "AVCC framing"
        );
    }

    #[test]
    fn rejects_duplicate_track_kind_and_unsupported_caps() {
        let mut mux = FlvMuxN::new(2);
        mux.configure_pipeline(0, &video_caps()).unwrap();
        // A second video pad is invalid for FLV's one-video model.
        assert!(mux.configure_pipeline(1, &video_caps()).is_err());
        // Audio is fine alongside the video.
        assert!(mux.configure_pipeline(1, &audio_caps()).is_ok());
        // Raw video is not a muxable FLV codec.
        let raw = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: g2g_core::Rate::Any,
        };
        assert!(mux.intercept_caps(0, &raw).is_err());
    }
}
