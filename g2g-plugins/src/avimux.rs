//! AVI muxer elements (M1071): one video stream plus at most one audio stream
//! in, a `ByteStream{Avi}` out. [`AviMux`] takes a single input
//! (`... ! mjpegenc ! avimux ! filesink location=out.avi`), [`AviMuxN`] takes
//! one per pad (`... ! m.video_0  ... ! m.audio_0  avimux name=m ! filesink`).
//!
//! AVI's headers carry each stream's sample count and its `idx1` carries every
//! chunk's offset, so nothing can be written before the last access unit
//! arrives: both elements queue their inputs, interleave them by pts, and emit
//! the whole file as one frame at `Eos` (the layout
//! [`Mp4MuxN`](crate::mp4muxn::Mp4MuxN) uses in its progressive mode).
//!
//! Access units are written as they arrive: H.264 stays Annex-B under the
//! `H264` FourCC, and audio keeps its container framing. There is no OpenDML
//! writing here, so a `movi` past the 1 GB an `idx1` offset can address fails
//! the mux rather than producing a file whose index is wrong.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    ElementMetadata, G2gError, InputAggregator, MultiInputElement, OutputSink, PadTemplate,
    PadTemplates, VideoCodec,
};

use crate::avi::{AviWriteStream, AviWriter};

fn output_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Avi,
    }
}

/// A negotiated side, or [`GEOMETRY_UNKNOWN`] when the solve left it open. A
/// decoded still-image or MJPEG stream carries its real size only at runtime (as
/// a `CapsChanged`), and the whole AVI is written at `Eos`, so an open geometry
/// at negotiation is fine as long as it is known before the first frame.
fn fixed_or_unknown(dim: &Dim) -> u32 {
    match dim {
        Dim::Fixed(value) => *value,
        _ => GEOMETRY_UNKNOWN,
    }
}

/// The width / height a video stream carries before its real caps arrive. AVI
/// has no way to write it, so the mux fails loud rather than emit a 0x0 header.
const GEOMETRY_UNKNOWN: u32 = 0;

/// Whether every stream the writer is about to describe has the geometry AVI's
/// headers need.
fn geometry_known(streams: &[AviWriteStream]) -> bool {
    streams.iter().all(|stream| match stream {
        AviWriteStream::Video { width, height, .. } => {
            *width != GEOMETRY_UNKNOWN && *height != GEOMETRY_UNKNOWN
        }
        AviWriteStream::Audio { .. } => true,
    })
}

/// The stream an input pad's caps describe, or `None` for a media type AVI
/// cannot carry (raw video, text, a codec with no `BITMAPINFOHEADER` FourCC or
/// `WAVEFORMATEX` tag).
fn write_stream_for(caps: &Caps) -> Option<AviWriteStream> {
    write_stream_of(caps).filter(AviWriteStream::is_writable)
}

fn write_stream_of(caps: &Caps) -> Option<AviWriteStream> {
    match caps {
        Caps::CompressedVideo {
            codec,
            width,
            height,
            ..
        } => Some(AviWriteStream::Video {
            codec: *codec,
            width: fixed_or_unknown(width),
            height: fixed_or_unknown(height),
        }),
        Caps::Audio {
            format,
            channels,
            sample_rate,
        } => Some(AviWriteStream::Audio {
            format: *format,
            channels: *channels,
            sample_rate: *sample_rate,
        }),
        _ => None,
    }
}

/// The caps a video pad negotiates against: the codecs AVI names, at a fixable
/// geometry span the encoder upstream pins.
fn video_input_alternatives() -> Vec<Caps> {
    /// A coded video stream is at least one macroblock.
    const MIN_CODED_DIM: u32 = 16;
    const MAX_CODED_DIM: u32 = 65_535;
    const MIN_RATE_Q16: u32 = 1 << 16;
    const MAX_RATE_Q16: u32 = 240 << 16;
    [VideoCodec::Mjpeg, VideoCodec::H264, VideoCodec::Mpeg4Part2]
        .into_iter()
        .map(|codec| Caps::CompressedVideo {
            codec,
            width: Dim::Range {
                min: MIN_CODED_DIM,
                max: MAX_CODED_DIM,
            },
            height: Dim::Range {
                min: MIN_CODED_DIM,
                max: MAX_CODED_DIM,
            },
            framerate: g2g_core::Rate::Range {
                min_q16: MIN_RATE_Q16,
                max_q16: MAX_RATE_Q16,
            },
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        })
        .collect()
}

/// Whether an access unit opens a decodable point. AVI records this per chunk
/// in `idx1`, and the parsers upstream already set it on the frame.
fn is_keyframe(stream: &AviWriteStream, timing: &FrameTiming) -> bool {
    match stream {
        // Every audio frame is a decodable point in the formats AVI carries.
        AviWriteStream::Audio { .. } => true,
        // An all-intra codec has nothing but keyframes, whatever a parser that
        // never looked at the bitstream left on the frame.
        AviWriteStream::Video {
            codec: VideoCodec::Mjpeg,
            ..
        } => true,
        AviWriteStream::Video { .. } => timing.keyframe,
    }
}

/// Wrap a finished file as the single outgoing frame.
fn file_frame(bytes: Vec<u8>) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    )
}

/// Writes one elementary stream into an AVI file.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::avimux::AviMux;
///
/// // gst-launch equivalent: ... ! mjpegenc ! avimux ! filesink location=out.avi
/// let mux = AviMux::new();
/// ```
#[derive(Debug, Default)]
pub struct AviMux {
    stream: Option<AviWriteStream>,
    writer: Option<AviWriter>,
    finished: bool,
}

impl AviMux {
    pub fn new() -> Self {
        Self::default()
    }

    /// The writer, built on first use from the stream descriptor as it stands by
    /// then (a runtime `CapsChanged` may have filled in the geometry). A video
    /// stream whose size is still unknown fails here: AVI's headers have no way
    /// to leave it out.
    fn writer(&mut self) -> Result<&mut AviWriter, G2gError> {
        if self.writer.is_none() {
            let stream = self.stream.clone().ok_or(G2gError::NotConfigured)?;
            let streams = Vec::from([stream]);
            if !geometry_known(&streams) {
                return Err(G2gError::CapsMismatch);
            }
            self.writer = Some(AviWriter::new(streams));
        }
        self.writer.as_mut().ok_or(G2gError::NotConfigured)
    }
}

impl AsyncElement for AviMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AVI muxer",
            "Codec/Muxer",
            "Writes one elementary stream into an AVI byte stream",
            "g2g",
        )
    }

    /// Writes host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if write_stream_for(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if write_stream_for(input).is_some() {
                CapsSet::one(output_caps())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.stream = Some(write_stream_for(absolute_caps).ok_or(G2gError::CapsMismatch)?);
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let stream = self.stream.clone().ok_or(G2gError::NotConfigured)?;
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let keyframe = is_keyframe(&stream, &frame.timing);
                    let pts_ns = frame.timing.pts_ns;
                    let data = Vec::from(slice);
                    self.writer()?.push(0, data, pts_ns, keyframe)?;
                }
                PipelinePacket::Eos => {
                    if !self.finished {
                        self.finished = true;
                        let bytes = self.writer()?.finish()?;
                        out.push(PipelinePacket::DataFrame(file_frame(bytes)))
                            .await?;
                    }
                }
                // A decoder refines the geometry (and compressed audio its
                // layout) at runtime, and the headers need the real one, so take
                // it while the file can still change.
                PipelinePacket::CapsChanged(caps) => {
                    if self.writer.is_none() {
                        if let Some(stream) = write_stream_for(&caps) {
                            self.stream = Some(stream);
                        }
                    }
                }
                // A muxed container carries its own timing.
                PipelinePacket::Segment(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for AviMux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(video_input_alternatives())),
            PadTemplate::source(CapsSet::one(output_caps())),
        ])
    }
}

/// Writes one video stream plus at most one audio stream into an AVI file, one
/// input pad per stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::avimux::AviMuxN;
///
/// // gst-launch equivalent: ... ! m.video_0  ... ! m.audio_0  avimux name=m ! filesink
/// let mux = AviMuxN::new(2);
/// ```
#[derive(Debug)]
pub struct AviMuxN {
    inputs: usize,
    /// Per pad, the stream its caps described; `None` until it configures.
    streams: Vec<Option<AviWriteStream>>,
    aggregator: InputAggregator<Frame>,
    writer: Option<AviWriter>,
    finished: bool,
}

impl AviMuxN {
    /// A muxer with `inputs` request pads.
    pub fn new(inputs: usize) -> Self {
        Self {
            inputs,
            streams: alloc::vec![None; inputs],
            aggregator: InputAggregator::new(inputs),
            writer: None,
            finished: false,
        }
    }

    /// Build the writer once every pad has declared its stream, so the `strl`
    /// list is in pad order.
    /// Built on first use, so a runtime `CapsChanged` on any pad has had its
    /// chance to fill in what negotiation left open. `None` while a pad has no
    /// stream yet, or while a video stream's size is still unknown.
    fn writer(&mut self) -> Option<&mut AviWriter> {
        if self.writer.is_none() {
            let streams: Option<Vec<AviWriteStream>> = self.streams.iter().cloned().collect();
            let streams = streams.filter(|streams| geometry_known(streams))?;
            self.writer = Some(AviWriter::new(streams));
        }
        self.writer.as_mut()
    }
}

impl MultiInputElement for AviMuxN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Writes host memory, so every pad takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AVI muxer",
            "Codec/Muxer",
            "Writes a video stream and an optional audio stream into an AVI byte stream",
            "g2g",
        )
    }

    fn input_count(&self) -> usize {
        self.inputs
    }

    /// Named request pads: a container mux's inputs are caps-typed slots, so
    /// `video_%u` / `audio_%u` / `sink_%u` each claim the next positional slot
    /// and a launch line can name them in any order.
    fn input_pad_index(
        &self,
        _req: &g2g_core::runtime::PadRequest,
        ordinal: usize,
    ) -> Option<usize> {
        (ordinal < self.inputs).then_some(ordinal)
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if write_stream_for(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(output_caps())))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let stream = write_stream_for(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        // AVI's main header describes one video stream, and nothing this writes
        // addresses a second one, so refuse rather than write a file whose
        // `avih` contradicts its `strl` list.
        let video = matches!(stream, AviWriteStream::Video { .. });
        let already_video = self
            .streams
            .iter()
            .enumerate()
            .any(|(i, s)| i != input && matches!(s, Some(AviWriteStream::Video { .. })));
        if video && already_video {
            return Err(G2gError::CapsMismatch);
        }
        self.streams[input] = Some(stream);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(output_caps())
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => self.aggregator.push(input, frame),
                PipelinePacket::Eos => self.aggregator.mark_ended(input),
                PipelinePacket::CapsChanged(caps) => {
                    // A demuxer refines compressed audio's layout at runtime,
                    // and the `strf` needs the real one, so take it while the
                    // header can still change.
                    if self.writer.is_none() {
                        if let Some(stream) = write_stream_for(&caps) {
                            self.streams[input] = Some(stream);
                        }
                    }
                    return Ok(());
                }
                // A muxed container carries its own timing.
                PipelinePacket::Segment(_) => return Ok(()),
                other => {
                    out.push(other).await?;
                    return Ok(());
                }
            }
            // Release access units in global pts order, so the chunks
            // interleave the way a player expects to read them.
            while let Some((pad, frame)) = self.aggregator.take_earliest_by(|f| f.timing.pts_ns) {
                let Some(stream) = self.streams[pad].clone() else {
                    return Err(G2gError::NotConfigured);
                };
                let slice = frame
                    .domain
                    .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                let data = Vec::from(slice);
                let keyframe = is_keyframe(&stream, &frame.timing);
                let pts_ns = frame.timing.pts_ns;
                self.writer()
                    .ok_or(G2gError::NotConfigured)?
                    .push(pad, data, pts_ns, keyframe)?;
            }
            if self.aggregator.is_drained() && !self.finished {
                self.finished = true;
                let bytes = self.writer().ok_or(G2gError::NotConfigured)?.finish()?;
                out.push(PipelinePacket::DataFrame(file_frame(bytes)))
                    .await?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avi::{parse, AviStreamKind};
    use alloc::vec;
    use g2g_core::runtime::block_on;
    use g2g_core::{AudioFormat, PushOutcome, Rate};

    /// Captures the muxer's finished file.
    #[derive(Debug, Default)]
    struct ByteCapture {
        bytes: Vec<u8>,
    }

    impl OutputSink for ByteCapture {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            if let PipelinePacket::DataFrame(f) =
                packet_slot.take().expect("poll_push without a packet")
            {
                if let Some(s) = f.domain.as_system_slice() {
                    self.bytes.extend_from_slice(s);
                }
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    const FRAME_PERIOD_NS: u64 = 40_000_000;

    fn frame(payload: Vec<u8>, pts_ns: u64, keyframe: bool) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                keyframe,
                ..FrameTiming::default()
            },
            sequence: 0,
            meta: Default::default(),
        })
    }

    #[test]
    fn writes_an_av_file_the_parser_reads_back() {
        let video = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Fixed(25 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let audio = Caps::Audio {
            format: AudioFormat::Mp3,
            channels: 2,
            sample_rate: 44_100,
        };
        let mut mux = AviMuxN::new(2);
        mux.configure_pipeline(0, &video).expect("video pad");
        mux.configure_pipeline(1, &audio).expect("audio pad");
        let mut sink = ByteCapture::default();
        block_on(async {
            for i in 0..3u64 {
                mux.process(
                    0,
                    frame(vec![0, 0, 0, 1, 0x65, i as u8], i * FRAME_PERIOD_NS, i == 0),
                    &mut sink,
                )
                .await
                .expect("video au");
                mux.process(
                    1,
                    frame(vec![0xff, i as u8], i * FRAME_PERIOD_NS, true),
                    &mut sink,
                )
                .await
                .expect("audio au");
            }
            mux.process(0, PipelinePacket::Eos, &mut sink)
                .await
                .expect("video eos");
            mux.process(1, PipelinePacket::Eos, &mut sink)
                .await
                .expect("audio eos");
        });

        let file = parse(&sink.bytes).expect("the written file parses");
        assert_eq!(
            file.streams[0].kind,
            AviStreamKind::Video {
                codec: VideoCodec::H264,
                width: 320,
                height: 240
            }
        );
        assert_eq!(
            file.streams[1].kind,
            AviStreamKind::Audio {
                format: AudioFormat::Mp3,
                channels: 2,
                sample_rate: 44_100
            }
        );
        let video_chunks: Vec<_> = file.chunks.iter().filter(|c| c.stream == 0).collect();
        assert_eq!(video_chunks.len(), 3);
        assert_eq!(
            video_chunks.iter().filter(|c| c.keyframe).count(),
            1,
            "only the first access unit was marked a keyframe"
        );
        for (i, chunk) in video_chunks.iter().enumerate() {
            assert_eq!(
                &sink.bytes[chunk.body.clone()],
                &[0, 0, 0, 1, 0x65, i as u8],
                "the access unit is written byte for byte"
            );
        }
        assert_eq!(file.chunks.iter().filter(|c| c.stream == 1).count(), 3);
    }

    #[test]
    fn refuses_a_second_video_pad() {
        let video = Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
            width: Dim::Fixed(64),
            height: Dim::Fixed(48),
            framerate: Rate::Fixed(25 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let mut mux = AviMuxN::new(2);
        mux.configure_pipeline(0, &video).expect("the first video");
        assert!(
            mux.configure_pipeline(1, &video).is_err(),
            "AVI's avih describes one video stream"
        );
    }

    /// A decoded still-image stream negotiates an open geometry and states its
    /// real size at runtime, so the mux accepts the caps and takes the size from
    /// the `CapsChanged` that precedes the first frame.
    #[test]
    fn takes_a_geometry_that_only_arrives_at_runtime() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 48;
        let open = Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let mut mux = AviMux::new();
        mux.configure_pipeline(&open)
            .expect("an open geometry negotiates");
        let mut sink = ByteCapture::default();
        block_on(async {
            mux.process(
                PipelinePacket::CapsChanged(Caps::CompressedVideo {
                    codec: VideoCodec::Mjpeg,
                    width: Dim::Fixed(WIDTH),
                    height: Dim::Fixed(HEIGHT),
                    framerate: Rate::Fixed(25 << 16),
                    colorimetry: g2g_core::Colorimetry::UNKNOWN,
                }),
                &mut sink,
            )
            .await
            .expect("the refinement is taken");
            mux.process(frame(vec![0xFF, 0xD8, 0xFF, 0xD9], 0, true), &mut sink)
                .await
                .expect("the frame is queued");
            mux.process(PipelinePacket::Eos, &mut sink)
                .await
                .expect("the file is written");
        });
        let file = parse(&sink.bytes).expect("the file parses");
        assert_eq!(
            file.streams[0].kind,
            AviStreamKind::Video {
                codec: VideoCodec::Mjpeg,
                width: WIDTH,
                height: HEIGHT
            },
            "the runtime geometry reached the header"
        );
    }

    /// Without that refinement there is no size to write, and AVI cannot leave
    /// it out, so the mux fails instead of writing a 0x0 header.
    #[test]
    fn refuses_a_geometry_that_never_arrives() {
        let open = Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let mut mux = AviMux::new();
        mux.configure_pipeline(&open).expect("negotiates");
        let mut sink = ByteCapture::default();
        let pushed = block_on(async {
            mux.process(frame(vec![0xFF, 0xD8, 0xFF, 0xD9], 0, true), &mut sink)
                .await
        });
        assert_eq!(pushed.err(), Some(G2gError::CapsMismatch));
    }

    #[test]
    fn refuses_a_codec_avi_cannot_name() {
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Fixed(64),
            height: Dim::Fixed(48),
            framerate: Rate::Fixed(25 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let mut mux = AviMuxN::new(1);
        assert!(mux.configure_pipeline(0, &vp9).is_err());
        assert!(AviMux::new().configure_pipeline(&vp9).is_err());
    }
}
