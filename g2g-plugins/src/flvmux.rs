//! FLV multiplexer element (M120): one elementary stream in
//! (`Caps::CompressedVideo{H264}` Annex-B, or `Caps::Audio{Aac}` ADTS), an FLV
//! byte stream out (`Caps::ByteStream{Flv}`).
//!
//! Wraps the pure [`crate::flv::FlvMuxer`], the inverse of
//! [`crate::flvdemux::FlvDemux`]: each input access unit becomes an FLV tag, with
//! the "FLV" header emitted once up front. The track (video vs audio) is read from
//! the input caps at configure. Like [`crate::flvmuxn::FlvMuxN`], the decoder
//! config a player needs is captured in-band from the first access unit and
//! written as the track's sequence-header tag (M662): the `avcC` record from the
//! parameter sets in the first IDR, or the AAC `AudioSpecificConfig` from the
//! first ADTS header. Video NALUs are re-framed AVCC length-prefixed (keyframes
//! flagged from the IDR NAL), audio written de-ADTS'd, so the output is a real
//! playable FLV. CPU, `no_std` baseline.
//!
//! ```text
//! ... ! h264parse ! flvmux ! filesink location=out.flv
//! ```
//!
//! Scope: one track (a single input pad), mirroring the single-stream
//! `FlvDemux`. An `onMetaData` script tag is written when metadata is attached
//! via [`FlvMux::with_tags`]; A/V muxing is `FlvMuxN` (DESIGN.md §4.17).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, Rate, TagList, VideoCodec,
};

use crate::aacparse::{asc_from_adts, strip_adts};
use crate::annexb::{avcc_record, avcc_sample, is_keyframe_nal, parameter_sets, split_annexb};
use crate::flv::{FlvMuxer, FlvTrack};

/// Muxes one elementary stream into an FLV byte stream.
#[derive(Debug)]
pub struct FlvMux {
    /// Built at configure, once the input track is known.
    mux: Option<FlvMuxer>,
    /// The track the negotiated input carries, set at configure.
    track: Option<FlvTrack>,
    /// Whether the sequence-header config was captured from an access unit.
    init_captured: bool,
    tags: TagList,
    configured: bool,
    emitted: u64,
}

impl Default for FlvMux {
    fn default() -> Self {
        Self::new()
    }
}

impl FlvMux {
    pub fn new() -> Self {
        Self {
            mux: None,
            track: None,
            init_captured: false,
            tags: TagList::new(),
            configured: false,
            emitted: 0,
        }
    }

    /// Attach stream metadata, written as an `onMetaData` script tag in the header.
    pub fn with_tags(mut self, tags: TagList) -> Self {
        self.tags = tags;
        self
    }

    /// Count of FLV byte frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The output it produces: an FLV byte stream.
    fn output_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Flv,
        }
    }

    /// The FLV track for an input caps, or `None` if the codec is unsupported.
    fn track_for(caps: &Caps) -> Option<FlvTrack> {
        match caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            } => Some(FlvTrack::Video),
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            } => Some(FlvTrack::Audio),
            _ => None,
        }
    }

    /// The elementary streams this muxer accepts on its sink pad.
    fn input_alternatives() -> Vec<Caps> {
        Vec::from([
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            },
            Caps::Audio {
                format: AudioFormat::Aac,
                channels: 0,
                sample_rate: 0,
            },
        ])
    }
}

impl AsyncElement for FlvMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::track_for(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if Self::track_for(input).is_some() {
                CapsSet::one(Self::output_caps())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let track = Self::track_for(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.mux = Some(FlvMuxer::new(track).with_tags(self.tags.clone()));
        self.track = Some(track);
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let au = slice.as_slice();
                    let track = self.track.ok_or(G2gError::NotConfigured)?;
                    let mux = self.mux.as_mut().ok_or(G2gError::NotConfigured)?;
                    let pts_ms = (frame.timing.pts_ns / 1_000_000) as u32;
                    // Capture the decoder config from the first access unit that
                    // carries it, so the sequence header precedes the media tags
                    // (M662, the FlvMuxN pattern): video parameter sets only ride
                    // the IDR, audio config is in every ADTS header.
                    let flv = match track {
                        FlvTrack::Video => {
                            let nalus = split_annexb(au);
                            if !self.init_captured {
                                if let Ok(sets) = parameter_sets(VideoCodec::H264, &nalus) {
                                    mux.set_video_config(avcc_record(&sets));
                                    self.init_captured = true;
                                }
                            }
                            let keyframe =
                                nalus.iter().any(|n| is_keyframe_nal(VideoCodec::H264, n));
                            mux.push_video(&avcc_sample(&nalus), pts_ms, keyframe)
                        }
                        FlvTrack::Audio => {
                            if !self.init_captured {
                                if let Some(asc) = asc_from_adts(au) {
                                    mux.set_audio_config(asc.to_vec());
                                    self.init_captured = true;
                                }
                            }
                            mux.push_audio(strip_adts(au), pts_ms)
                        }
                    };
                    let out_frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(flv.into_boxed_slice())),
                        FrameTiming {
                            pts_ns: frame.timing.pts_ns,
                            ..FrameTiming::default()
                        },
                        self.emitted,
                    );
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                // The runner's transform arm forwards EOS, so don't push it here.
                PipelinePacket::Eos => {}
                // Input geometry / params don't change the FLV framing.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for FlvMux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(Self::input_alternatives())),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flvdemux::{FlvDemux, FlvStream};
    use g2g_core::{PushOutcome, RawVideoFormat};

    fn h264_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Any,
        }
    }

    #[derive(Default)]
    struct CaptureSink {
        frames: Vec<Vec<u8>>,
    }
    impl OutputSink for CaptureSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(f) = packet {
                    if let MemoryDomain::System(s) = &f.domain {
                        self.frames.push(s.as_slice().to_vec());
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn h264_frame(au: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            0,
        ))
    }

    #[test]
    fn caps_codec_in_byte_stream_out() {
        let m = FlvMux::new();
        assert!(m.intercept_caps(&h264_caps()).is_ok());
        let raw = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert!(m.intercept_caps(&raw).is_err());

        let CapsConstraint::DerivedOutput(f) = m.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        assert!(matches!(
            f(&h264_caps()).alternatives(),
            [Caps::ByteStream {
                encoding: ByteStreamEncoding::Flv
            }]
        ));
    }

    /// An Annex-B access unit from raw NAL payloads (4-byte start codes).
    fn annexb(nals: &[&[u8]]) -> Vec<u8> {
        let mut v = Vec::new();
        for n in nals {
            v.extend_from_slice(&[0, 0, 0, 1]);
            v.extend_from_slice(n);
        }
        v
    }

    #[tokio::test]
    async fn element_round_trips_tags_through_flvdemux() {
        use g2g_core::{Bus, BusMessage, Tag};

        let tags: TagList = [Tag::Title("My Clip".into()), Tag::Encoder("g2g".into())]
            .into_iter()
            .collect();
        let mut mux = FlvMux::new().with_tags(tags.clone());
        mux.configure_pipeline(&h264_caps()).unwrap();
        let mut flv_sink = CaptureSink::default();
        mux.process(h264_frame(annexb(&[&[0x65, 0xAA]]), 0), &mut flv_sink)
            .await
            .unwrap();

        let mut flv = Vec::new();
        for f in &flv_sink.frames {
            flv.extend_from_slice(f);
        }
        let (bus, handle) = Bus::new(8);
        let mut demux = FlvDemux::new()
            .with_stream(FlvStream::H264)
            .with_bus(handle);
        demux
            .configure_pipeline(&Caps::ByteStream {
                encoding: ByteStreamEncoding::Flv,
            })
            .unwrap();
        let mut au_sink = CaptureSink::default();
        let flv_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(flv.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        demux
            .process(PipelinePacket::DataFrame(flv_frame), &mut au_sink)
            .await
            .unwrap();

        let mut posted = None;
        while let Some(m) = bus.try_recv() {
            if let BusMessage::Tag(t) = m {
                posted = Some(t);
            }
        }
        assert_eq!(posted.expect("a Tag message").tags(), tags.tags());
        assert_eq!(
            au_sink.frames,
            alloc::vec![annexb(&[&[0x65, 0xAA]])],
            "the AU still demuxes"
        );
    }

    #[tokio::test]
    async fn element_round_trips_through_flvdemux() {
        let au0 = annexb(&[&[0x65u8, 0xAA, 0xBB]]);
        let au1 = annexb(&[&[0x41u8, 0xCC]]);

        let mut mux = FlvMux::new();
        mux.configure_pipeline(&h264_caps()).unwrap();
        let mut flv_sink = CaptureSink::default();
        mux.process(h264_frame(au0.clone(), 0), &mut flv_sink)
            .await
            .unwrap();
        mux.process(h264_frame(au1.clone(), 33_000_000), &mut flv_sink)
            .await
            .unwrap();
        assert_eq!(mux.emitted(), 2);

        // Feed the muxed FLV bytes back through the demuxer.
        let mut flv = Vec::new();
        for f in &flv_sink.frames {
            flv.extend_from_slice(f);
        }
        let mut demux = FlvDemux::new().with_stream(FlvStream::H264);
        demux
            .configure_pipeline(&Caps::ByteStream {
                encoding: ByteStreamEncoding::Flv,
            })
            .unwrap();
        let mut au_sink = CaptureSink::default();
        let flv_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(flv.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        demux
            .process(PipelinePacket::DataFrame(flv_frame), &mut au_sink)
            .await
            .unwrap();

        assert_eq!(
            au_sink.frames,
            alloc::vec![au0, au1],
            "AUs recovered through mux + demux"
        );
    }

    #[tokio::test]
    async fn element_writes_sequence_header_from_first_idr() {
        use crate::annexb::avcc_record;
        use crate::flv::FlvDemuxer;

        // A first AU carrying SPS + PPS + IDR: the muxer captures the parameter
        // sets and writes the avcC sequence-header tag ahead of the media tag.
        let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1E];
        let pps: &[u8] = &[0x68, 0xCE, 0x3C, 0x80];
        let idr: &[u8] = &[0x65, 0x88, 0x84];
        let mut mux = FlvMux::new();
        mux.configure_pipeline(&h264_caps()).unwrap();
        let mut flv_sink = CaptureSink::default();
        mux.process(h264_frame(annexb(&[sps, pps, idr]), 0), &mut flv_sink)
            .await
            .unwrap();

        let mut d = FlvDemuxer::new();
        d.push_data(&flv_sink.frames[0]);
        assert_eq!(
            d.video_config(),
            Some(&avcc_record(&[sps, pps])[..]),
            "the avcC sequence header rides ahead of the media tag"
        );
        let units = d.take_units();
        assert_eq!(units.len(), 1);
        assert!(units[0].keyframe, "the IDR NAL flags the FLV keyframe type");
    }

    #[tokio::test]
    async fn element_writes_audio_sequence_header_from_adts() {
        use crate::flv::FlvDemuxer;

        // 48 kHz (index 3) stereo AAC-LC ADTS frame; the muxer derives the ASC.
        let adts = alloc::vec![0xFFu8, 0xF1, 0x4C, 0x80, 0x00, 0x1F, 0xFC, 0xAB, 0xCD];
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        };
        let mut mux = FlvMux::new();
        mux.configure_pipeline(&aac).unwrap();
        let mut flv_sink = CaptureSink::default();
        mux.process(h264_frame(adts, 0), &mut flv_sink)
            .await
            .unwrap();

        let mut d = FlvDemuxer::new();
        d.push_data(&flv_sink.frames[0]);
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
