//! Multi-track Matroska / WebM multiplexer element (M294): N elementary streams
//! in (H.264 / H.265 video + AAC audio), one Matroska byte stream out. The A/V
//! analog of the single-track [`crate::mkvmux::MkvMux`], so a muxed recording
//! carries video and audio together:
//!
//! ```text
//! videotestsrc ! x264enc ! matroskamux name=m
//! audiotestsrc ! avenc_aac ! m.
//! m. ! filesink location=av.mkv
//! ```
//!
//! A [`MultiInputElement`] (input pad order = track order = Matroska TrackNumber):
//! each pad takes one elementary stream, and access units interleave by
//! presentation timestamp via the M204 [`InputAggregator`] merge before being
//! written to their track's SimpleBlocks. The Tracks element (one TrackEntry per
//! stream) is built once every track has its `CodecPrivate`, which arrives in-band:
//! a video track's avcC / hvcC record is synthesised from the parameter sets in
//! the first IDR, an audio track's AudioSpecificConfig from the first ADTS header
//! (the AAC bytes are written de-ADTS'd, and video NALUs AVCC length-prefixed, the
//! framing the Matroska codec mappings expect).
//!
//! Reachable from the `gst-launch` fan-in syntax: registered as the `matroskamux`
//! muxer in `default_registry`, so >1 input link builds this element (a single
//! input builds the single-track [`crate::mkvmux::MkvMux`]), the way gst's request
//! sink pads do. Scope (v1): H.264/H.265 + AAC; every input pad must carry a
//! stream (a pad that ends without an access unit stalls the build).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    FrameTiming, G2gError, InputAggregator, MemoryDomain, MultiInputElement, OutputSink,
    PipelinePacket, VideoCodec,
};

use crate::matroska::{MatroskaMuxer, MkvCodec, MkvTrackConfig, MkvTrackSpec};
use crate::mp4muxn::{asc_from_adts, strip_adts};
use crate::fmp4mux::{
    avcc_record, avcc_sample, hvcc_record, is_keyframe_nal, parameter_sets, split_annexb,
};

/// What an input pad carries, learned from its negotiated caps at configure.
#[derive(Debug, Clone, Copy)]
enum PadKind {
    Video(VideoCodec),
    Audio { channels: u8, rate: u32 },
}

/// A track's init data, captured from its first access unit: the parameter sets
/// (video) or AudioSpecificConfig (audio) the Tracks `CodecPrivate` needs.
#[derive(Debug, Clone)]
enum TrackInit {
    Video { codec: VideoCodec, width: u32, height: u32, param_sets: Vec<Vec<u8>> },
    Audio { channels: u8, rate: u32, asc: Vec<u8> },
}

/// Muxes N elementary streams into one Matroska byte stream, PTS-ordered.
#[derive(Debug)]
pub struct MkvMuxN {
    inputs: usize,
    /// Per-pad stream kind, learned at configure (the Tracks element needs all).
    kinds: Vec<Option<PadKind>>,
    /// Per-pad track init, captured from the first AU. Geometry comes from the
    /// caps; video parameter sets / audio ASC come in-band from the first AU.
    inits: Vec<Option<TrackInit>>,
    /// Per-pad caps geometry (video width/height), recorded at configure.
    dims: Vec<(u32, u32)>,
    agg: InputAggregator<Frame>,
    /// Built lazily once every track has its init (the Tracks element needs all).
    mux: Option<MatroskaMuxer>,
    emitted: u64,
}

impl MkvMuxN {
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "MkvMuxN needs at least one input");
        Self {
            inputs,
            kinds: alloc::vec![None; inputs],
            inits: alloc::vec![None; inputs],
            dims: alloc::vec![(0, 0); inputs],
            agg: InputAggregator::new(inputs),
            mux: None,
            emitted: 0,
        }
    }

    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output_caps_value() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::Matroska }
    }

    fn pad_kind_for(caps: &Caps) -> Option<PadKind> {
        match caps {
            Caps::CompressedVideo { codec: c @ (VideoCodec::H264 | VideoCodec::H265), .. } => {
                Some(PadKind::Video(*c))
            }
            Caps::Audio { format: AudioFormat::Aac, channels, sample_rate } => {
                Some(PadKind::Audio { channels: *channels, rate: *sample_rate })
            }
            _ => None,
        }
    }

    /// True once every pad has its init captured (the Tracks element, and so the
    /// track numbering, needs every track present).
    fn all_inits_ready(&self) -> bool {
        self.inits.iter().all(|i| i.is_some())
    }

    /// Capture a pad's track init from its first access unit, if not already set.
    fn capture_init(&mut self, input: usize, au: &[u8]) {
        if self.inits[input].is_some() {
            return;
        }
        match self.kinds[input] {
            Some(PadKind::Video(codec)) => {
                let nalus = split_annexb(au);
                // Parameter sets only ride the IDR; a leading P-frame has none, so
                // wait for the keyframe that carries them.
                if let Ok(param_sets) = parameter_sets(codec, &nalus) {
                    let owned: Vec<Vec<u8>> = param_sets.iter().map(|s| s.to_vec()).collect();
                    let (w, h) = self.dims[input];
                    self.inits[input] =
                        Some(TrackInit::Video { codec, width: w, height: h, param_sets: owned });
                }
            }
            Some(PadKind::Audio { channels, rate }) => {
                if let Some(asc) = asc_from_adts(au) {
                    self.inits[input] = Some(TrackInit::Audio { channels, rate, asc: asc.to_vec() });
                }
            }
            None => {}
        }
    }

    /// The SimpleBlock payload for a track: AVCC length-prefixed NALUs for video,
    /// the de-ADTS'd raw AAC for audio. Also returns whether it is a keyframe.
    fn sample_for(&self, input: usize, au: &[u8]) -> (Vec<u8>, bool) {
        match self.kinds[input] {
            Some(PadKind::Video(codec)) => {
                let nalus = split_annexb(au);
                let is_key = nalus.iter().any(|n| is_keyframe_nal(codec, n));
                (avcc_sample(&nalus), is_key)
            }
            // Audio access units are always keyframes; strip the ADTS header.
            _ => (strip_adts(au).to_vec(), true),
        }
    }

    /// Emit one access unit as its track's SimpleBlock (the muxer prepends the
    /// header + Tracks on the first call, and opens Clusters as time advances).
    async fn emit_au(&mut self, input: usize, frame: Frame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let MemoryDomain::System(slice) = &frame.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let pts_ns = frame.timing.pts_ns;
        let (sample, is_key) = self.sample_for(input, slice.as_slice());
        let mux = self.mux.as_mut().ok_or(G2gError::NotConfigured)?;
        let bytes = mux.push_frame_on(input, &sample, pts_ns, is_key);

        let out_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming { pts_ns, ..FrameTiming::default() },
            self.emitted,
        );
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(out_frame)).await?;
        Ok(())
    }
}

/// The muxer track config (spec + `CodecPrivate`) for a captured init: the avcC /
/// hvcC record for video, the AudioSpecificConfig for AAC.
fn track_config(init: &TrackInit) -> MkvTrackConfig {
    match init {
        TrackInit::Video { codec, width, height, param_sets } => {
            let refs: Vec<&[u8]> = param_sets.iter().map(|v| v.as_slice()).collect();
            let (mkv_codec, codec_private) = match codec {
                VideoCodec::H265 => (MkvCodec::H265, hvcc_record(&refs)),
                _ => (MkvCodec::H264, avcc_record(&refs)),
            };
            MkvTrackConfig {
                spec: MkvTrackSpec {
                    codec: mkv_codec,
                    width: *width,
                    height: *height,
                    channels: 0,
                    sample_rate: 0,
                },
                codec_private,
            }
        }
        TrackInit::Audio { channels, rate, asc } => MkvTrackConfig {
            spec: MkvTrackSpec {
                codec: MkvCodec::Aac,
                width: 0,
                height: 0,
                channels: *channels,
                sample_rate: *rate,
            },
            codec_private: asc.clone(),
        },
    }
}

impl MultiInputElement for MkvMuxN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
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
        Ok(CapsConstraint::Produces(CapsSet::one(Self::output_caps_value())))
    }

    fn configure_pipeline(&mut self, input: usize, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let kind = Self::pad_kind_for(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        if let Caps::CompressedVideo { width: Dim::Fixed(w), height: Dim::Fixed(h), .. } = absolute_caps {
            self.dims[input] = (*w, *h);
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
                    // Capture this track's init from its first AU before queueing.
                    if let MemoryDomain::System(s) = &frame.domain {
                        let au = s.as_slice().to_vec();
                        self.capture_init(input, &au);
                    }
                    self.agg.push(input, frame);
                }
                PipelinePacket::Eos => self.agg.mark_ended(input),
                // CapsChanged is consumed by the runner's muxer arm; the Tracks
                // element is fixed from the first AU's in-band init.
                PipelinePacket::CapsChanged(_) => return Ok(()),
                other => {
                    out.push(other).await?;
                    return Ok(());
                }
            }

            // Hold every AU until all tracks have their init (the Tracks element,
            // and the track numbering it pins, needs them all).
            if !self.all_inits_ready() {
                return Ok(());
            }
            if self.mux.is_none() {
                let configs: Vec<MkvTrackConfig> =
                    self.inits.iter().map(|i| track_config(i.as_ref().expect("ready"))).collect();
                self.mux = Some(MatroskaMuxer::new_multi(configs));
            }
            // Release AUs now safe to emit, in global PTS order.
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
    use crate::matroska::MatroskaDemuxer;

    /// A minimal Annex-B H.264 IDR access unit: SPS (type 7), PPS (type 8), IDR
    /// (type 5), each behind a 4-byte start code. The SPS carries 3 bytes after
    /// its header so avcC can copy profile/compat/level.
    fn h264_idr() -> Vec<u8> {
        let mut au = Vec::new();
        for nal in [
            alloc::vec![0x67, 0x42, 0x00, 0x1E],
            alloc::vec![0x68, 0xCE, 0x3C, 0x80],
            alloc::vec![0x65, 0x88, 0x84],
        ] {
            au.extend_from_slice(&[0, 0, 0, 1]);
            au.extend_from_slice(&nal);
        }
        au
    }

    /// A 7-byte ADTS AAC frame header (LC, 48 kHz, stereo) + 2 payload bytes.
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
        Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48_000 }
    }

    fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming { pts_ns, ..FrameTiming::default() },
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
                    if let MemoryDomain::System(s) = &f.domain {
                        self.bytes.extend_from_slice(s.as_slice());
                    }
                }
                Ok(g2g_core::PushOutcome::Accepted)
            })
        }
    }

    #[tokio::test]
    async fn av_streams_mux_into_two_matroska_tracks() {
        let mut mux = MkvMuxN::new(2);
        mux.configure_pipeline(0, &video_caps()).unwrap();
        mux.configure_pipeline(1, &audio_caps()).unwrap();

        let mut sink = CaptureSink::default();
        // Interleave a video IDR and an audio frame; the merge needs both inputs
        // to have queued before it releases, so push both then a second each.
        mux.process(0, frame(h264_idr(), 0), &mut sink).await.unwrap();
        mux.process(1, frame(aac_adts(), 0), &mut sink).await.unwrap();
        mux.process(0, frame(h264_idr(), 33_000_000), &mut sink).await.unwrap();
        mux.process(1, frame(aac_adts(), 21_000_000), &mut sink).await.unwrap();
        mux.process(0, PipelinePacket::Eos, &mut sink).await.unwrap();
        mux.process(1, PipelinePacket::Eos, &mut sink).await.unwrap();

        // Demux the produced Matroska back: two tracks (H.264 video + AAC audio).
        let mut d = MatroskaDemuxer::new();
        d.push_data(&sink.bytes);
        let tracks = d.tracks();
        assert_eq!(tracks.len(), 2, "video + audio tracks announced");
        assert_eq!(tracks[0].number, 1);
        assert_eq!(tracks[0].codec, MkvCodec::H264);
        assert_eq!(tracks[1].number, 2);
        assert_eq!(tracks[1].codec, MkvCodec::Aac);
        assert_eq!(tracks[1].channels, 2);
        assert_eq!(tracks[1].sample_rate, 48_000);

        // CodecPrivate is present for both tracks (avcC record, AAC ASC): the
        // bytes carry the avcC config-version byte and the A_AAC CodecID.
        assert!(sink.bytes.windows(5).any(|w| w == b"A_AAC"), "AAC CodecID written");
        assert!(
            sink.bytes.windows(4).any(|w| w == b"\x63\xA2\x00\x00" || w[0] == 0x63 && w[1] == 0xA2),
            "CodecPrivate element present"
        );
        assert!(mux.emitted() >= 4, "all four access units muxed");
    }

    #[test]
    fn rejects_unsupported_caps() {
        let mux = MkvMuxN::new(1);
        let raw = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: g2g_core::Rate::Any,
        };
        assert!(mux.intercept_caps(0, &raw).is_err());
        assert!(mux.intercept_caps(0, &video_caps()).is_ok());
        assert!(mux.intercept_caps(0, &audio_caps()).is_ok());
    }
}
