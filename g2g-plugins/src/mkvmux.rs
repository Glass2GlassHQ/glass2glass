//! Matroska / WebM multiplexer element (M115): one elementary stream in
//! (`Caps::CompressedVideo{H264|H265|VP8|VP9|AV1}` or `Caps::Audio{Aac|Opus}`),
//! a Matroska / WebM byte stream out (`Caps::ByteStream{Matroska}`).
//!
//! Wraps the pure [`crate::matroska::MatroskaMuxer`], the inverse of
//! [`crate::mkvmux::MkvMux`]'s sibling [`crate::mkvdemux::MkvDemux`]: the track is
//! built from the input caps (codec + geometry / audio params) and each frame
//! becomes a Cluster. WebM-subset codecs (VP8 / VP9 / AV1 / Opus) get the `webm`
//! DocType, the rest `matroska`. CPU, `no_std` baseline.
//!
//! ```text
//! ... ! mkvmux ! filesink location=out.webm
//! ```
//!
//! The muxer is built lazily on the first frame, so a `CapsChanged` that refines
//! the geometry (e.g. from a parser) is reflected in the written Tracks. Scope
//! (v1): one track, one frame per Cluster, every frame flagged a keyframe (no
//! upstream delta-frame signal yet). A `Cues` index is written at EOS (M375).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, TagList, VideoCodec,
};

use crate::matroska::{MatroskaMuxer, MkvCodec, MkvTrackSpec};

/// Muxes one elementary stream into a Matroska / WebM byte stream.
#[derive(Debug)]
pub struct MkvMux {
    /// Current input caps, set at configure and refined by `CapsChanged` until the
    /// first frame builds the muxer.
    caps: Option<Caps>,
    mux: Option<MatroskaMuxer>,
    tags: TagList,
    configured: bool,
    emitted: u64,
    /// Live / streamable mode (the gst `streamable` property): suppress the `Cues`
    /// seek index normally appended at EOS. The Cues let a read-to-end file seek,
    /// but a live consumer (a pipe, an HTTP push) cannot seek and the muxer would
    /// have to hold the cluster positions to the end; `streamable` drops it so the
    /// output is a pure forward stream. Off by default (a recording stays seekable).
    streamable: bool,
}

impl Default for MkvMux {
    fn default() -> Self {
        Self::new()
    }
}

impl MkvMux {
    pub fn new() -> Self {
        Self {
            caps: None,
            mux: None,
            tags: TagList::new(),
            configured: false,
            emitted: 0,
            streamable: false,
        }
    }

    /// Attach stream metadata, written as a `Tags` element in the header.
    pub fn with_tags(mut self, tags: TagList) -> Self {
        self.tags = tags;
        self
    }

    /// Live mode: suppress the EOS `Cues` index (see [`streamable`](Self::streamable)).
    pub fn with_streamable(mut self, streamable: bool) -> Self {
        self.streamable = streamable;
        self
    }

    /// Count of byte-stream frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        }
    }

    /// The Matroska track for an input caps, or `None` if the codec is unmappable.
    fn track_spec(caps: &Caps) -> Option<MkvTrackSpec> {
        match caps {
            Caps::CompressedVideo {
                codec,
                width,
                height,
                ..
            } => Some(MkvTrackSpec {
                codec: video_to_mkv(*codec)?,
                width: dim_u32(width),
                height: dim_u32(height),
                channels: 0,
                sample_rate: 0,
            }),
            Caps::Audio {
                format,
                channels,
                sample_rate,
            } => Some(MkvTrackSpec {
                codec: audio_to_mkv(*format)?,
                width: 0,
                height: 0,
                channels: *channels,
                sample_rate: *sample_rate,
            }),
            _ => None,
        }
    }

    /// The elementary streams this muxer accepts on its sink pad.
    fn input_alternatives() -> Vec<Caps> {
        let video = |codec| Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let audio = |format| Caps::Audio {
            format,
            channels: 0,
            sample_rate: 0,
        };
        Vec::from([
            video(VideoCodec::H264),
            video(VideoCodec::H265),
            video(VideoCodec::Vp8),
            video(VideoCodec::Vp9),
            video(VideoCodec::Av1),
            audio(AudioFormat::Aac),
            audio(AudioFormat::Opus),
        ])
    }
}

impl AsyncElement for MkvMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::track_spec(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if Self::track_spec(input).is_some() {
                CapsSet::one(Self::output_caps())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Self::track_spec(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.caps = Some(absolute_caps.clone());
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "streamable",
            PropKind::Bool,
            "live mode: omit the seekable Cues index written at EOS",
        )
        .with_default("false")];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "streamable" => {
                self.streamable = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "streamable" => Some(PropValue::Bool(self.streamable)),
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
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    if self.mux.is_none() {
                        let caps = self.caps.as_ref().ok_or(G2gError::NotConfigured)?;
                        let spec = Self::track_spec(caps).ok_or(G2gError::CapsMismatch)?;
                        self.mux = Some(MatroskaMuxer::new(spec).with_tags(self.tags.clone()));
                    }
                    let mux = self.mux.as_mut().ok_or(G2gError::NotConfigured)?;
                    // No upstream delta-frame signal yet: flag every frame a keyframe.
                    let bytes = mux.push_frame(slice.as_slice(), frame.timing.pts_ns, true);
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
                }
                PipelinePacket::CapsChanged(c) => {
                    // Refines the track until the first frame fixes the header.
                    if self.mux.is_none() && Self::track_spec(&c).is_some() {
                        self.caps = Some(c);
                    }
                }
                // At EOS, flush the Cues index after the last Cluster so the stream
                // is seekable on a read-to-end (M375); the runner then forwards EOS.
                PipelinePacket::Eos => {
                    if let Some(mux) = self.mux.as_ref().filter(|_| !self.streamable) {
                        let cues = mux.finish();
                        if !cues.is_empty() {
                            let out_frame = Frame::new(
                                MemoryDomain::System(SystemSlice::from_boxed(
                                    cues.into_boxed_slice(),
                                )),
                                FrameTiming::default(),
                                self.emitted,
                            );
                            self.emitted += 1;
                            out.push(PipelinePacket::DataFrame(out_frame)).await?;
                        }
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

impl PadTemplates for MkvMux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(Self::input_alternatives())),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
    }
}

fn video_to_mkv(codec: VideoCodec) -> Option<MkvCodec> {
    match codec {
        VideoCodec::H264 => Some(MkvCodec::H264),
        VideoCodec::H265 => Some(MkvCodec::H265),
        VideoCodec::Vp8 => Some(MkvCodec::Vp8),
        VideoCodec::Vp9 => Some(MkvCodec::Vp9),
        VideoCodec::Av1 => Some(MkvCodec::Av1),
        VideoCodec::Mjpeg => None,
        // A codec MKV cannot carry (or one added since): not muxable here.
        _ => None,
    }
}

fn audio_to_mkv(format: AudioFormat) -> Option<MkvCodec> {
    match format {
        AudioFormat::Aac => Some(MkvCodec::Aac),
        AudioFormat::Opus => Some(MkvCodec::Opus),
        AudioFormat::PcmS16Le | AudioFormat::PcmF32Le => None,
        // A format MKV cannot carry (or one added since): not muxable here.
        _ => None,
    }
}

fn dim_u32(d: &Dim) -> u32 {
    match d {
        Dim::Fixed(n) => *n,
        Dim::Range { min, .. } => *min,
        Dim::Any => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mkvdemux::{MkvDemux, MkvStream};
    use g2g_core::{PushOutcome, RawVideoFormat};

    fn vp9_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Any,
        }
    }

    #[derive(Default)]
    struct CaptureSink {
        frames: Vec<Vec<u8>>,
        eos: bool,
    }
    impl OutputSink for CaptureSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                match packet {
                    PipelinePacket::DataFrame(f) => {
                        if let MemoryDomain::System(s) = &f.domain {
                            self.frames.push(s.as_slice().to_vec());
                        }
                    }
                    PipelinePacket::Eos => self.eos = true,
                    _ => {}
                }
                Ok(PushOutcome::Accepted)
            })
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

    #[test]
    fn caps_codec_in_byte_stream_out() {
        let m = MkvMux::new();
        assert!(m.intercept_caps(&vp9_caps()).is_ok());
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
            f(&vp9_caps()).alternatives(),
            [Caps::ByteStream {
                encoding: ByteStreamEncoding::Matroska
            }]
        ));
    }

    #[tokio::test]
    async fn element_round_trips_tags_through_mkvdemux() {
        use g2g_core::{Bus, BusMessage, Tag, TagList};

        let tags: TagList = [Tag::Title("My Clip".into()), Tag::Encoder("g2g".into())]
            .into_iter()
            .collect();
        let mut mux = MkvMux::new().with_tags(tags.clone());
        mux.configure_pipeline(&vp9_caps()).unwrap();
        let mut mkv_sink = CaptureSink::default();
        mux.process(frame(alloc::vec![0x11, 0x22], 0), &mut mkv_sink)
            .await
            .unwrap();

        let mut mkv = Vec::new();
        for f in &mkv_sink.frames {
            mkv.extend_from_slice(f);
        }
        let (bus, handle) = Bus::new(8);
        let mut demux = MkvDemux::new().with_stream(MkvStream::Vp9).with_bus(handle);
        demux
            .configure_pipeline(&Caps::ByteStream {
                encoding: ByteStreamEncoding::Matroska,
            })
            .unwrap();
        let mut frame_sink = CaptureSink::default();
        let mkv_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(mkv.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        demux
            .process(PipelinePacket::DataFrame(mkv_frame), &mut frame_sink)
            .await
            .unwrap();

        let mut posted = None;
        while let Some(m) = bus.try_recv() {
            if let BusMessage::Tag(t) = m {
                posted = Some(t);
            }
        }
        assert_eq!(posted.expect("a Tag message").tags(), tags.tags());
        assert_eq!(frame_sink.frames, alloc::vec![alloc::vec![0x11, 0x22]]);
    }

    #[tokio::test]
    async fn element_round_trips_through_mkvdemux() {
        let f0 = alloc::vec![0x11u8, 0x22, 0x33];
        let f1 = alloc::vec![0x44u8, 0x55];

        let mut mux = MkvMux::new();
        mux.configure_pipeline(&vp9_caps()).unwrap();
        let mut mkv_sink = CaptureSink::default();
        mux.process(frame(f0.clone(), 0), &mut mkv_sink)
            .await
            .unwrap();
        mux.process(frame(f1.clone(), 40_000_000), &mut mkv_sink)
            .await
            .unwrap();
        mux.process(PipelinePacket::Eos, &mut mkv_sink)
            .await
            .unwrap();
        assert!(
            !mkv_sink.eos,
            "EOS is forwarded by the runner's arm, not the element"
        );

        let mut mkv = Vec::new();
        for f in &mkv_sink.frames {
            mkv.extend_from_slice(f);
        }
        let mut demux = MkvDemux::new().with_stream(MkvStream::Vp9);
        demux
            .configure_pipeline(&Caps::ByteStream {
                encoding: ByteStreamEncoding::Matroska,
            })
            .unwrap();
        let mut frame_sink = CaptureSink::default();
        let mkv_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(mkv.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        demux
            .process(PipelinePacket::DataFrame(mkv_frame), &mut frame_sink)
            .await
            .unwrap();
        demux
            .process(PipelinePacket::Eos, &mut frame_sink)
            .await
            .unwrap();

        assert_eq!(
            frame_sink.frames,
            alloc::vec![f0, f1],
            "frames recovered through mux + demux"
        );
        // Two frames plus the EOS Cues index (both frames share one Cluster, so the
        // dedup-per-Cluster index holds a single CuePoint, emitted as one frame).
        assert_eq!(mux.emitted(), 3);
    }

    #[tokio::test]
    async fn streamable_omits_cues_at_eos() {
        let mut mux = MkvMux::new().with_streamable(true);
        mux.configure_pipeline(&vp9_caps()).unwrap();
        assert_eq!(mux.get_property("streamable"), Some(PropValue::Bool(true)));
        let mut sink = CaptureSink::default();
        mux.process(frame(alloc::vec![1, 2, 3], 0), &mut sink)
            .await
            .unwrap();
        mux.process(frame(alloc::vec![4, 5, 6], 33_000_000), &mut sink)
            .await
            .unwrap();
        mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        // Two header+cluster frames, and no trailing Cues frame (the live mode).
        assert_eq!(
            mux.emitted(),
            2,
            "streamable mode emits no Cues index at EOS"
        );
        let all: Vec<u8> = sink.frames.concat();
        // Cues element id is 0x1C53BB6B; it must not appear in the output.
        assert!(
            !all.windows(4).any(|w| w == [0x1C, 0x53, 0xBB, 0x6B]),
            "no Cues element written in streamable mode"
        );
    }
}
