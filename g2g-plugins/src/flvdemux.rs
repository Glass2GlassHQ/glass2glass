//! FLV demuxer element (M119): `Caps::ByteStream{Flv}` in, one selected
//! elementary stream out. H.264 video leaves as `Caps::CompressedVideo` (AVCC);
//! AAC audio leaves as `Caps::Audio`.
//!
//! Wraps the pure [`crate::flv::FlvDemuxer`], the FLV sibling of
//! [`crate::tsdemux::TsDemux`]: incoming byte frames are fed to the parser, and
//! the access units of the selected stream ([`FlvStream`], default H.264) are
//! forwarded with their PTS, ready for the matching parser / decoder. CPU,
//! `no_std` baseline.
//!
//! ```text
//! filesrc(location=x.flv, caps=ByteStream{Flv}) ! flvdemux ! h264parse ! <decoder>
//! flvdemux stream=aac ! aacparse ! <audio>
//! ```
//!
//! One output pad carries one elementary stream; the [`FlvStream`] selection picks
//! which, so a second `flvdemux stream=aac` demuxes the audio. The choice is by
//! codec because the output caps are fixed at negotiation, before any tag is
//! parsed. Scope (v1): the H.264 video and AAC audio tracks (DESIGN.md §4.17).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, VideoCodec,
};

use crate::flv::{FlvDemuxer, FlvTrack, FlvUnit};

/// Which elementary stream an [`FlvDemux`] instance forwards. An FLV stream
/// interleaves one video and one audio track; this element has one output pad, so
/// it emits exactly one, chosen by codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlvStream {
    /// The H.264 (AVC) video track. The default.
    #[default]
    H264,
    /// The AAC audio track.
    Aac,
}

/// Demuxes an FLV byte stream into one selected elementary stream.
#[derive(Debug)]
pub struct FlvDemux {
    demux: FlvDemuxer,
    /// The elementary stream this instance forwards (the single output pad).
    stream: FlvStream,
    configured: bool,
    emitted: u64,
}

impl Default for FlvDemux {
    fn default() -> Self {
        Self::new()
    }
}

impl FlvDemux {
    pub fn new() -> Self {
        Self { demux: FlvDemuxer::new(), stream: FlvStream::H264, configured: false, emitted: 0 }
    }

    /// Select which elementary stream to forward (default [`FlvStream::H264`]).
    pub fn with_stream(mut self, stream: FlvStream) -> Self {
        self.stream = stream;
        self
    }

    /// The elementary stream this instance forwards.
    pub fn stream(&self) -> FlvStream {
        self.stream
    }

    /// Count of frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The input this element accepts: an FLV byte stream.
    fn input_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::Flv }
    }

    /// The output caps for the selected elementary stream. Video geometry is
    /// unknown until the bitstream parser reads the SPS, so H.264 advertises a
    /// fixatable placeholder `Range` refined downstream via `CapsChanged`. AAC has
    /// no open `Caps` field, so it advertises the sentinel channels/rate 0 that
    /// `aacparse` accepts pre-header.
    fn output_caps(stream: FlvStream) -> Caps {
        match stream {
            FlvStream::H264 => Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Range { min: 16, max: 65_535 },
                height: Dim::Range { min: 16, max: 65_535 },
                framerate: Rate::Range { min_q16: 1 << 16, max_q16: 240 << 16 },
            },
            FlvStream::Aac => Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 },
        }
    }

    /// The track this instance's selected stream corresponds to.
    fn selected_track(stream: FlvStream) -> FlvTrack {
        match stream {
            FlvStream::H264 => FlvTrack::Video,
            FlvStream::Aac => FlvTrack::Audio,
        }
    }

    /// Emit each access unit of the selected track as a frame, carrying its PTS
    /// (the FLV millisecond timestamp converted to nanoseconds).
    async fn emit_units(
        &mut self,
        units: Vec<FlvUnit>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let want = Self::selected_track(self.stream);
        for u in units {
            if u.track != want {
                continue;
            }
            let pts_ns = u.pts_ms as u64 * 1_000_000;
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(u.data.into_boxed_slice())),
                FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
                self.emitted,
            );
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for FlvDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let stream = self.stream;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream { encoding: ByteStreamEncoding::Flv } => {
                CapsSet::one(Self::output_caps(stream))
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(absolute_caps, Caps::ByteStream { encoding: ByteStreamEncoding::Flv }) {
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.demux.push_data(slice.as_slice());
                    let units = self.demux.take_units();
                    self.emit_units(units, out).await?;
                }
                PipelinePacket::Eos => {
                    // Emit any final access units. The runner's transform arm
                    // forwards the EOS itself, so pushing it here would double it
                    // (the second hits a closed sink under a full link).
                    let units = self.demux.take_units();
                    self.emit_units(units, out).await?;
                }
                // ByteStream caps don't carry geometry; nothing to forward.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        FLVDEMUX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "stream" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.stream = flv_stream_from_str(s).ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "stream" => Some(PropValue::Str(flv_stream_to_str(self.stream).into())),
            _ => None,
        }
    }
}

/// `FlvDemux`'s settable properties.
static FLVDEMUX_PROPS: &[PropertySpec] =
    &[PropertySpec::new("stream", PropKind::Str, "elementary stream to emit: h264 | aac")];

/// Parse a `stream` property string to an [`FlvStream`].
fn flv_stream_from_str(s: &str) -> Option<FlvStream> {
    match s {
        "h264" => Some(FlvStream::H264),
        "aac" => Some(FlvStream::Aac),
        _ => None,
    }
}

/// The `stream` property string for an [`FlvStream`].
fn flv_stream_to_str(stream: FlvStream) -> &'static str {
    match stream {
        FlvStream::H264 => "h264",
        FlvStream::Aac => "aac",
    }
}

impl PadTemplates for FlvDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        let source = CapsSet::from_alternatives(Vec::from([
            Self::output_caps(FlvStream::H264),
            Self::output_caps(FlvStream::Aac),
        ]));
        Vec::from([PadTemplate::sink(CapsSet::one(Self::input_caps())), PadTemplate::source(source)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Dim, PushOutcome, Rate, RawVideoFormat};

    fn push_u24(out: &mut Vec<u8>, v: u32) {
        out.push((v >> 16) as u8);
        out.push((v >> 8) as u8);
        out.push(v as u8);
    }

    fn tag(tag_type: u8, timestamp: u32, body: &[u8]) -> Vec<u8> {
        let mut t = alloc::vec![tag_type];
        push_u24(&mut t, body.len() as u32);
        push_u24(&mut t, timestamp & 0x00FF_FFFF);
        t.push((timestamp >> 24) as u8);
        push_u24(&mut t, 0);
        t.extend_from_slice(body);
        t
    }

    fn avc_nalu(au: &[u8]) -> Vec<u8> {
        let mut b = alloc::vec![0x17, 0x01, 0x00, 0x00, 0x00];
        b.extend_from_slice(au);
        b
    }

    fn aac_raw(frame: &[u8]) -> Vec<u8> {
        let mut b = alloc::vec![0xAF, 0x01];
        b.extend_from_slice(frame);
        b
    }

    fn flv_stream(tags: &[Vec<u8>]) -> Vec<u8> {
        let mut s = b"FLV".to_vec();
        s.push(1);
        s.push(0x05);
        s.extend_from_slice(&9u32.to_be_bytes());
        let mut prev = 0u32;
        for t in tags {
            s.extend_from_slice(&prev.to_be_bytes());
            s.extend_from_slice(t);
            prev = t.len() as u32;
        }
        s
    }

    #[derive(Default)]
    struct CaptureSink {
        frames: Vec<Vec<u8>>,
        pts: Vec<u64>,
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
                    self.pts.push(f.timing.pts_ns);
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    async fn run_demux(d: &mut FlvDemux, stream: &[u8], sink: &mut CaptureSink) {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        d.process(PipelinePacket::DataFrame(frame), sink).await.unwrap();
        d.process(PipelinePacket::Eos, sink).await.unwrap();
    }

    #[test]
    fn caps_byte_stream_in_h264_out() {
        let d = FlvDemux::new();
        assert!(d.intercept_caps(&FlvDemux::input_caps()).is_ok());
        let raw = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert!(d.intercept_caps(&raw).is_err());
        // The Matroska byte stream is the wrong container.
        let mkv = Caps::ByteStream { encoding: ByteStreamEncoding::Matroska };
        assert!(d.intercept_caps(&mkv).is_err());
    }

    #[tokio::test]
    async fn selects_video_or_audio_from_a_stream() {
        let v0 = [0u8, 0, 0, 5, 0x65, 0x11];
        let v1 = [0u8, 0, 0, 5, 0x41, 0x22];
        let a0 = [0x33u8, 0x44];
        let a1 = [0x55u8, 0x66];
        let stream = flv_stream(&[
            tag(9, 0, &avc_nalu(&v0)),
            tag(8, 0, &aac_raw(&a0)),
            tag(9, 40, &avc_nalu(&v1)),
            tag(8, 40, &aac_raw(&a1)),
        ]);

        // Default selects H.264: only the two video AUs come out, PTS in ns.
        let mut video = FlvDemux::new();
        video.configure_pipeline(&FlvDemux::input_caps()).unwrap();
        let mut vsink = CaptureSink::default();
        run_demux(&mut video, &stream, &mut vsink).await;
        assert_eq!(vsink.frames, alloc::vec![v0.to_vec(), v1.to_vec()], "video only");
        assert_eq!(vsink.pts, alloc::vec![0, 40_000_000], "ms timestamps to ns");
        assert_eq!(video.emitted(), 2);

        // stream=aac selects AAC: only the two audio frames come out.
        let mut audio = FlvDemux::new().with_stream(FlvStream::Aac);
        audio.configure_pipeline(&FlvDemux::input_caps()).unwrap();
        let mut asink = CaptureSink::default();
        run_demux(&mut audio, &stream, &mut asink).await;
        assert_eq!(asink.frames, alloc::vec![a0.to_vec(), a1.to_vec()], "audio only");
    }

    #[test]
    fn output_caps_track_the_selection() {
        assert!(matches!(
            FlvDemux::output_caps(FlvStream::H264),
            Caps::CompressedVideo { codec: VideoCodec::H264, .. }
        ));
        assert!(matches!(
            FlvDemux::output_caps(FlvStream::Aac),
            Caps::Audio { format: AudioFormat::Aac, .. }
        ));
    }

    #[test]
    fn stream_property_round_trips_and_drives_output() {
        let mut d = FlvDemux::new();
        assert_eq!(d.get_property("stream"), Some(PropValue::Str("h264".into())));

        d.set_property("stream", PropValue::Str("aac".into())).unwrap();
        assert_eq!(d.stream(), FlvStream::Aac);

        // An unsupported codec name is rejected, leaving the selection unchanged.
        assert_eq!(d.set_property("stream", PropValue::Str("vp9".into())), Err(PropError::Value));
        assert_eq!(d.stream(), FlvStream::Aac);

        let CapsConstraint::DerivedOutput(f) = d.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = f(&FlvDemux::input_caps());
        assert!(matches!(out.alternatives(), [Caps::Audio { format: AudioFormat::Aac, .. }]));
    }
}
