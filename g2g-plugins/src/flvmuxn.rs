//! FLV multiplexer element (M120, M296): a video and an audio elementary stream
//! in, either one optional (`Caps::CompressedVideo{H264}` Annex-B, `Caps::Audio{Aac}`
//! ADTS, or one of the legacy Flash codecs: Sorenson, VP6, VP6-alpha, MP3, Speex),
//! one FLV byte stream out:
//!
//! ```text
//! ... ! h264parse ! flvmux ! filesink location=out.flv
//! videotestsrc ! x264enc ! flvmux name=m
//! audiotestsrc ! avenc_aac ! m.
//! m. ! filesink location=av.flv
//! ```
//!
//! A [`MultiInputElement`]: each pad takes one elementary stream (FLV implies the
//! track by tag type, so the muxer routes by pad codec, not pad index), and access
//! units interleave by presentation timestamp via the M204 [`InputAggregator`]
//! merge before being written as FLV tags. The sequence-header tags a player needs
//! are written up front, captured in-band from the first access unit: the video
//! track's `avcC` record from the parameter sets in the first IDR, the audio
//! track's `AudioSpecificConfig` from the first ADTS header (the AAC bytes are
//! written de-ADTS'd, video NALUs AVCC length-prefixed). The header flags name only
//! the tracks the pads carry, and an `onMetaData` script tag is written when
//! metadata is attached via [`FlvMuxN::with_tags`].
//!
//! Reachable from the `gst-launch` fan-in syntax: registered only as the `flvmux`
//! muxer in `default_registry`, so any number of input links builds this element.
//! Scope: FLV's one-video + one-audio model; a second pad of either kind is
//! rejected. The legacy Flash codecs have no sequence header, so their tracks are
//! ready to write from the first access unit.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, InputAggregator, MemoryDomain, MultiInputElement,
    OutputSink, PipelinePacket, TagList, VideoCodec,
};

use crate::aacparse::{is_leading_aac_config, muxer_access_unit, muxer_audio_specific_config};
use crate::annexb::{au_is_keyframe, avcc_record, avcc_sample, parameter_sets, split_annexb};
use crate::flv::{FlvCodec, FlvMuxer, FlvTrack};
use crate::mux::{
    audio_header_config, audio_track_params, every_track_header_ready, refused_input,
    refused_input_because, require_header_config_at_end, video_header_config, CODEC_CONFIG,
};
use g2g_core::log::short_type_name;

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

impl TrackInit {
    fn asc(&self) -> Option<&[u8]> {
        match self {
            TrackInit::Audio { asc } => Some(asc),
            _ => None,
        }
    }
}

/// Muxes a video and an audio stream, either one optional, into one FLV byte
/// stream, PTS-ordered.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::flvmuxn::FlvMuxN;
///
/// let mux = FlvMuxN::new(2);
/// assert_eq!(mux.emitted(), 0);
/// ```
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
    tags: TagList,
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
            tags: TagList::new(),
            emitted: 0,
        }
    }

    /// Attach stream metadata, written as an `onMetaData` script tag in the header.
    pub fn with_tags(mut self, tags: TagList) -> Self {
        self.tags = tags;
        self
    }

    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output_caps_value() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Flv,
        }
    }

    /// The FLV codec an input pad's caps carries, or `None` if FLV cannot carry it.
    fn pad_kind_for(caps: &Caps) -> Option<FlvCodec> {
        match caps {
            Caps::CompressedVideo { codec, .. } => match codec {
                VideoCodec::H264 => Some(FlvCodec::H264),
                VideoCodec::SorensonH263 => Some(FlvCodec::SorensonH263),
                VideoCodec::Vp6 { alpha: false } => Some(FlvCodec::Vp6),
                VideoCodec::Vp6 { alpha: true } => Some(FlvCodec::Vp6Alpha),
                _ => None,
            },
            Caps::Audio { format, .. } => match format {
                AudioFormat::Aac => Some(FlvCodec::Aac),
                AudioFormat::Mp3 => Some(FlvCodec::Mp3),
                AudioFormat::Speex => Some(FlvCodec::Speex),
                _ => None,
            },
            _ => None,
        }
    }

    // what the pad's sequence header waits for from its stream
    fn header_config(&self, input: usize) -> &'static str {
        match self.kinds[input] {
            Some(FlvCodec::H264) => video_header_config(VideoCodec::H264),
            Some(FlvCodec::Aac) => audio_header_config(AudioFormat::Aac),
            _ => CODEC_CONFIG,
        }
    }

    /// Capture a pad's track init from its first access unit, if not already set.
    fn capture_init(&mut self, input: usize, au: &[u8]) -> Result<(), G2gError> {
        if self.inits[input].is_some() {
            return Ok(());
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
                if let Some(asc) = muxer_audio_specific_config(au, short_type_name::<Self>())? {
                    self.inits[input] = Some(TrackInit::Audio { asc });
                }
            }
            Some(_) => self.inits[input] = Some(TrackInit::None),
            None => {}
        }
        Ok(())
    }

    /// Build the A/V muxer from the captured inits (video `avcC` record + AAC ASC).
    fn build_mux(&self) -> Result<FlvMuxer, G2gError> {
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
        // a pad that ended with no frames has no init and writes no track
        let of_track = |t| {
            self.kinds
                .iter()
                .zip(&self.inits)
                .filter(|(_, init)| init.is_some())
                .filter_map(|(kind, _)| *kind)
                .find(|c| c.track() == t)
        };
        let audio_codec = of_track(FlvTrack::Audio);
        let mut mux = FlvMuxer::new_av(
            of_track(FlvTrack::Video),
            video_config,
            audio_codec,
            audio_config,
        )
        .with_tags(self.tags.clone());
        if let Some((channels, sample_rate)) = self.audio_params {
            // an MP3 tag's flags declare the rate and channel count
            if audio_codec == Some(FlvCodec::Mp3) {
                audio_track_params(
                    short_type_name::<Self>(),
                    AudioFormat::Mp3,
                    channels,
                    sample_rate,
                    &[],
                )?;
            }
            mux.set_audio_params(channels, sample_rate);
        }
        Ok(mux)
    }

    /// Emit one access unit as its track's FLV tag.
    async fn emit_au(
        &mut self,
        input: usize,
        frame: Frame,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let slice = frame
            .domain
            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
        let pts_ms = (frame.timing.pts_ns / 1_000_000) as u32;
        let mux = self.mux.as_mut().ok_or(G2gError::NotConfigured)?;
        let bytes = match self.kinds[input] {
            Some(FlvCodec::H264) => {
                let nalus = split_annexb(slice);
                let keyframe = au_is_keyframe(VideoCodec::H264, slice);
                let (dts_ms, cts_ms) = FlvMuxer::video_tag_timing(&frame.timing);
                mux.push_video(&avcc_sample(&nalus), dts_ms, cts_ms, keyframe)
            }
            // The legacy Flash video codecs go into a tag as they arrive, their
            // keyframe flag coming from the producer rather than the bitstream.
            Some(c) if c.track() == FlvTrack::Video => {
                let (dts_ms, cts_ms) = FlvMuxer::video_tag_timing(&frame.timing);
                mux.push_video(slice, dts_ms, cts_ms, frame.timing.keyframe)
            }
            // AAC access units are raw frames once any ADTS header is stripped;
            // MP3 and Speex frames need no unwrapping.
            _ => match self.inits[input].as_ref().and_then(TrackInit::asc) {
                Some(config) => mux.push_audio(muxer_access_unit(config, slice), pts_ms),
                None => mux.push_audio(slice, pts_ms),
            },
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

    /// Reads host memory, so every pad takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

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
        let kind = Self::pad_kind_for(absolute_caps)
            .ok_or_else(|| refused_input(short_type_name::<Self>(), absolute_caps))?;
        // FLV carries at most one video and one audio track; reject a duplicate.
        if self
            .kinds
            .iter()
            .enumerate()
            .any(|(i, k)| i != input && k.is_some_and(|k| k.track() == kind.track()))
        {
            return Err(refused_input_because(
                short_type_name::<Self>(),
                absolute_caps,
                "FLV holds one video and one audio track",
            ));
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

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "FLV muxer",
            "Codec/Muxer",
            "Muxes a video and an audio stream into a Flash Video stream",
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
                PipelinePacket::DataFrame(frame) => {
                    if let Some(s) = frame.domain.as_system_slice() {
                        self.capture_init(input, s)?;
                        // a raw AAC stream's config became the sequence header above
                        if self.inits[input]
                            .as_ref()
                            .and_then(TrackInit::asc)
                            .is_some_and(|config| is_leading_aac_config(config, s))
                        {
                            return Ok(());
                        }
                    }
                    self.agg.push(input, frame);
                }
                PipelinePacket::Eos => {
                    require_header_config_at_end(
                        short_type_name::<Self>(),
                        input,
                        &self.inits,
                        &self.agg,
                        self.header_config(input),
                    )?;
                    self.agg.mark_ended(input)
                }
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
            if !every_track_header_ready(&self.inits, &self.agg) {
                return Ok(());
            }
            if self.mux.is_none() {
                self.mux = Some(self.build_mux()?);
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
    use crate::flv::{FlvDemuxer, FlvTrack as DemuxTrack, FLV_FLAG_AUDIO, FLV_FLAG_VIDEO};
    use crate::flvdemux::{FlvDemux, FlvStream};
    use g2g_core::{AsyncElement, AudioFormat, Dim};

    const SPS: &[u8] = &[0x67, 0x42, 0x00, 0x1E];
    const PPS: &[u8] = &[0x68, 0xCE, 0x3C, 0x80];
    // the byte after the `FLV` signature and version
    const HEADER_FLAGS_AT: usize = 4;

    // an Annex-B access unit from raw NAL payloads, with 4-byte start codes
    fn annexb(nals: &[&[u8]]) -> Vec<u8> {
        let mut au = Vec::new();
        for nal in nals {
            au.extend_from_slice(&[0, 0, 0, 1]);
            au.extend_from_slice(nal);
        }
        au
    }

    fn h264_idr() -> Vec<u8> {
        annexb(&[SPS, PPS, &[0x65, 0x88, 0x84]])
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        }
    }

    fn audio_caps() -> Caps {
        Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
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
        frames: Vec<Vec<u8>>,
    }
    impl OutputSink for CaptureSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                if let PipelinePacket::DataFrame(f) = packet {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.bytes.extend_from_slice(s);
                        self.frames.push(s.to_vec());
                    }
                }
                Ok(g2g_core::PushOutcome::Accepted)
            })
        }
    }

    // mux `aus` through a one-pad muxer configured with `caps`
    async fn mux_one_pad(mux: FlvMuxN, caps: &Caps, aus: &[Vec<u8>]) -> (FlvMuxN, Vec<u8>) {
        let mut mux = mux;
        mux.configure_pipeline(0, caps).unwrap();
        let mut sink = CaptureSink::default();
        for (index, au) in aus.iter().enumerate() {
            mux.process(0, frame(au.clone(), index as u64 * 33_000_000), &mut sink)
                .await
                .unwrap();
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        (mux, sink.bytes)
    }

    // the access units `FlvDemux` recovers from `flv`, with what it posted on the bus
    async fn flvdemux_units(flv: Vec<u8>) -> (Vec<Vec<u8>>, Option<TagList>) {
        use g2g_core::{Bus, BusMessage};
        let (bus, handle) = Bus::new(8);
        let mut demux = FlvDemux::new()
            .with_stream(FlvStream::H264)
            .with_bus(handle);
        demux
            .configure_pipeline(&FlvMuxN::output_caps_value())
            .unwrap();
        let mut sink = CaptureSink::default();
        let flv_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(flv.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        demux
            .process(PipelinePacket::DataFrame(flv_frame), &mut sink)
            .await
            .unwrap();
        let mut posted = None;
        while let Some(message) = bus.try_recv() {
            if let BusMessage::Tag { tags, .. } = message {
                posted = Some(tags);
            }
        }
        (sink.frames, posted)
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
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert!(mux.intercept_caps(0, &raw).is_err());
    }

    #[test]
    fn produces_an_flv_byte_stream() {
        let mux = FlvMuxN::new(1);
        assert!(mux.intercept_caps(0, &video_caps()).is_ok());
        let Ok(CapsConstraint::Produces(set)) = mux.caps_constraint_for_output() else {
            panic!("expected Produces");
        };
        assert!(matches!(
            set.alternatives(),
            [Caps::ByteStream {
                encoding: ByteStreamEncoding::Flv
            }]
        ));
    }

    #[tokio::test]
    async fn one_pad_round_trips_through_flvdemux() {
        let au0 = annexb(&[SPS, PPS, &[0x65, 0xAA, 0xBB]]);
        let au1 = annexb(&[&[0x41, 0xCC]]);
        let (mux, flv) =
            mux_one_pad(FlvMuxN::new(1), &video_caps(), &[au0.clone(), au1.clone()]).await;
        assert_eq!(mux.emitted(), 2);
        assert_eq!(
            flv[HEADER_FLAGS_AT], FLV_FLAG_VIDEO,
            "a video-only file declares no audio track"
        );
        let (units, _) = flvdemux_units(flv).await;
        assert_eq!(
            units,
            alloc::vec![au0, au1],
            "AUs recovered through mux + demux"
        );
    }

    #[tokio::test]
    async fn one_pad_round_trips_tags_through_flvdemux() {
        use g2g_core::Tag;
        let tags: TagList = [Tag::Title("My Clip".into()), Tag::Encoder("g2g".into())]
            .into_iter()
            .collect();
        let au = annexb(&[SPS, PPS, &[0x65, 0xAA]]);
        let (_, flv) = mux_one_pad(
            FlvMuxN::new(1).with_tags(tags.clone()),
            &video_caps(),
            core::slice::from_ref(&au),
        )
        .await;
        let (units, posted) = flvdemux_units(flv).await;
        assert_eq!(posted.expect("a Tag message").tags(), tags.tags());
        assert_eq!(units, alloc::vec![au], "the AU still demuxes");
    }

    #[tokio::test]
    async fn one_pad_writes_the_sequence_header_from_the_first_idr() {
        let (_, flv) = mux_one_pad(FlvMuxN::new(1), &video_caps(), &[h264_idr()]).await;
        let mut d = FlvDemuxer::new();
        d.push_data(&flv);
        assert_eq!(
            d.video_config(),
            Some(&avcc_record(&[SPS, PPS])[..]),
            "the avcC sequence header rides ahead of the media tag"
        );
        let units = d.take_units();
        assert_eq!(units.len(), 1);
        assert!(units[0].keyframe, "the IDR NAL flags the FLV keyframe type");
    }

    #[tokio::test]
    async fn one_pad_writes_the_audio_sequence_header_from_adts() {
        // 48 kHz (index 3) stereo AAC-LC ADTS frame; the muxer derives the ASC.
        let (_, flv) = mux_one_pad(FlvMuxN::new(1), &audio_caps(), &[aac_adts()]).await;
        assert_eq!(
            flv[HEADER_FLAGS_AT], FLV_FLAG_AUDIO,
            "an audio-only file declares no video track"
        );
        let mut d = FlvDemuxer::new();
        d.push_data(&flv);
        assert_eq!(
            d.audio_config(),
            Some(&[0x11u8, 0x90][..]),
            "ASC from the ADTS header"
        );
        let units = d.take_units();
        assert_eq!(
            units[0].data,
            alloc::vec![0xAB, 0xCD],
            "media tag carries the raw AAC frame"
        );
    }
}
