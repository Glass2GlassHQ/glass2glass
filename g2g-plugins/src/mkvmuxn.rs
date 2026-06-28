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
//! sink pads do. Video is H.264/H.265 (avcC/hvcC + AVCC samples) or VP8/VP9 (raw
//! frames, no CodecPrivate); audio is AAC (ASC) or Opus (synthesised `OpusHead`),
//! so VP9 + Opus muxes a WebM. Every input pad must carry a stream (a pad that
//! ends without an access unit stalls the build).

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
    vp8_keyframe, vp9_keyframe,
};

/// What an input pad carries, learned from its negotiated caps at configure.
#[derive(Debug, Clone, Copy)]
enum PadKind {
    Video(VideoCodec),
    Audio { format: AudioFormat, channels: u8, rate: u32 },
}

/// A track's init data, captured from its first access unit. `param_sets` is the
/// H.26x SPS/PPS the avcC/hvcC `CodecPrivate` needs (empty for VP8/VP9, which
/// carry none); `config` is the audio `CodecPrivate` (AAC AudioSpecificConfig or
/// Opus `OpusHead`).
#[derive(Debug, Clone)]
enum TrackInit {
    Video { codec: VideoCodec, width: u32, height: u32, param_sets: Vec<Vec<u8>> },
    Audio { format: AudioFormat, channels: u8, rate: u32, config: Vec<u8> },
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
    /// Set once the EOS `Cues` index has been flushed, so it is emitted only once.
    cues_emitted: bool,
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
            cues_emitted: false,
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
            Caps::CompressedVideo {
                codec: c @ (VideoCodec::H264 | VideoCodec::H265 | VideoCodec::Vp8 | VideoCodec::Vp9),
                ..
            } => Some(PadKind::Video(*c)),
            Caps::Audio {
                format: format @ (AudioFormat::Aac | AudioFormat::Opus),
                channels,
                sample_rate,
            } => Some(PadKind::Audio { format: *format, channels: *channels, rate: *sample_rate }),
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
                let (w, h) = self.dims[input];
                match codec {
                    VideoCodec::H264 | VideoCodec::H265 => {
                        let nalus = split_annexb(au);
                        // Parameter sets only ride the IDR; a leading P-frame has
                        // none, so wait for the keyframe that carries them.
                        if let Ok(param_sets) = parameter_sets(codec, &nalus) {
                            let owned: Vec<Vec<u8>> = param_sets.iter().map(|s| s.to_vec()).collect();
                            self.inits[input] =
                                Some(TrackInit::Video { codec, width: w, height: h, param_sets: owned });
                        }
                    }
                    // VP8/VP9 carry no out-of-band parameter sets; the track is
                    // ready at the first frame (its CodecPrivate stays empty).
                    _ => {
                        self.inits[input] =
                            Some(TrackInit::Video { codec, width: w, height: h, param_sets: Vec::new() });
                    }
                }
            }
            Some(PadKind::Audio { format, channels, rate }) => match format {
                // AAC's AudioSpecificConfig is synthesised from the first ADTS header.
                AudioFormat::Aac => {
                    if let Some(asc) = asc_from_adts(au) {
                        self.inits[input] =
                            Some(TrackInit::Audio { format, channels, rate, config: asc.to_vec() });
                    }
                }
                // Opus carries its config (OpusHead) out of band, built from the caps.
                _ => {
                    self.inits[input] = Some(TrackInit::Audio {
                        format,
                        channels,
                        rate,
                        config: opus_head(channels, rate),
                    });
                }
            },
            None => {}
        }
    }

    /// The SimpleBlock payload for a track and whether it is a keyframe. H.26x is
    /// AVCC length-prefixed (keyframe from the NAL types); VP8/VP9 frames are
    /// stored verbatim (keyframe from the frame header). AAC strips its ADTS
    /// header; Opus packets are stored raw. Audio frames are always sync samples.
    fn sample_for(&self, input: usize, au: &[u8]) -> (Vec<u8>, bool) {
        match self.kinds[input] {
            Some(PadKind::Video(codec)) => match codec {
                VideoCodec::H264 | VideoCodec::H265 => {
                    let nalus = split_annexb(au);
                    let is_key = nalus.iter().any(|n| is_keyframe_nal(codec, n));
                    (avcc_sample(&nalus), is_key)
                }
                VideoCodec::Vp8 => (au.to_vec(), vp8_keyframe(au)),
                _ => (au.to_vec(), vp9_keyframe(au)),
            },
            Some(PadKind::Audio { format: AudioFormat::Aac, .. }) => (strip_adts(au).to_vec(), true),
            _ => (au.to_vec(), true),
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
/// hvcC record for H.26x video (none for VP8/VP9), the AudioSpecificConfig for AAC
/// or the `OpusHead` for Opus.
fn track_config(init: &TrackInit) -> MkvTrackConfig {
    match init {
        TrackInit::Video { codec, width, height, param_sets } => {
            let refs: Vec<&[u8]> = param_sets.iter().map(|v| v.as_slice()).collect();
            let (mkv_codec, codec_private) = match codec {
                VideoCodec::H265 => (MkvCodec::H265, hvcc_record(&refs)),
                VideoCodec::Vp8 => (MkvCodec::Vp8, Vec::new()),
                VideoCodec::Vp9 => (MkvCodec::Vp9, Vec::new()),
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
        TrackInit::Audio { format, channels, rate, config } => {
            let mkv_codec = match format {
                AudioFormat::Opus => MkvCodec::Opus,
                _ => MkvCodec::Aac,
            };
            MkvTrackConfig {
                spec: MkvTrackSpec {
                    codec: mkv_codec,
                    width: 0,
                    height: 0,
                    channels: *channels,
                    sample_rate: *rate,
                },
                codec_private: config.clone(),
            }
        }
    }
}

/// The 19-byte Opus `OpusHead` identification header for an N-channel stream, the
/// Matroska `CodecPrivate` for `A_OPUS`. Channel mapping family 0 (mono/stereo); a
/// conventional 80 ms pre-skip (the exact encoder delay is not surfaced in caps).
fn opus_head(channels: u8, sample_rate: u32) -> Vec<u8> {
    let mut h = Vec::with_capacity(19);
    h.extend_from_slice(b"OpusHead");
    h.push(1); // version
    h.push(channels.max(1));
    h.extend_from_slice(&3840u16.to_le_bytes()); // pre-skip
    h.extend_from_slice(&sample_rate.to_le_bytes()); // input sample rate
    h.extend_from_slice(&0i16.to_le_bytes()); // output gain
    h.push(0); // channel mapping family 0
    h
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
                        self.capture_init(input, s.as_slice());
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
            // Once every track has ended and drained, flush the Cues index after
            // the last Cluster so the stream is seekable on a read-to-end (M375).
            if self.agg.is_drained() && !self.cues_emitted {
                if let Some(mux) = self.mux.as_ref() {
                    let cues = mux.finish();
                    if !cues.is_empty() {
                        let out_frame = Frame::new(
                            MemoryDomain::System(SystemSlice::from_boxed(cues.into_boxed_slice())),
                            FrameTiming::default(),
                            self.emitted,
                        );
                        self.emitted += 1;
                        out.push(PipelinePacket::DataFrame(out_frame)).await?;
                    }
                }
                self.cues_emitted = true;
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

        // A Cues index is written at EOS, indexing the video keyframes (track 1),
        // so the muxed A/V stream is seekable (M375).
        assert!(!d.cues().is_empty(), "Cues index written for the video keyframes");

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

    fn vp9_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: g2g_core::Rate::Any,
        }
    }

    fn opus_caps() -> Caps {
        Caps::Audio { format: AudioFormat::Opus, channels: 2, sample_rate: 48_000 }
    }

    /// A VP9 frame whose uncompressed header byte marks it a key frame (marker
    /// 0b10, profile 0, show_existing 0, frame_type 0), then arbitrary payload.
    fn vp9_key() -> Vec<u8> {
        alloc::vec![0x80, 0x49, 0x83, 0x42, 0x00, 0x11, 0x22]
    }

    #[test]
    fn vp9_and_vp8_keyframe_detection() {
        // VP9: 0x80 -> key (frame_type bit 0), 0x84 -> non-key (frame_type bit 1).
        assert!(vp9_keyframe(&[0x80]));
        assert!(!vp9_keyframe(&[0x84]));
        assert!(!vp9_keyframe(&[0x00]), "bad frame marker is not a keyframe");
        // VP8: frame tag bit 0 clear = key frame.
        assert!(vp8_keyframe(&[0x10]));
        assert!(!vp8_keyframe(&[0x11]));
    }

    #[tokio::test]
    async fn vp9_opus_streams_mux_into_a_webm() {
        let mut mux = MkvMuxN::new(2);
        mux.configure_pipeline(0, &vp9_caps()).unwrap();
        mux.configure_pipeline(1, &opus_caps()).unwrap();

        // Opus packets are stored raw; a recognizable payload to recover.
        let opus0: Vec<u8> = alloc::vec![0xFC, 0xDE, 0xAD];
        let opus1: Vec<u8> = alloc::vec![0xFC, 0xBE, 0xEF];

        let mut sink = CaptureSink::default();
        mux.process(0, frame(vp9_key(), 0), &mut sink).await.unwrap();
        mux.process(1, frame(opus0.clone(), 0), &mut sink).await.unwrap();
        mux.process(0, frame(vp9_key(), 20_000_000), &mut sink).await.unwrap();
        mux.process(1, frame(opus1.clone(), 20_000_000), &mut sink).await.unwrap();
        mux.process(0, PipelinePacket::Eos, &mut sink).await.unwrap();
        mux.process(1, PipelinePacket::Eos, &mut sink).await.unwrap();

        // VP9 + Opus are both WebM-subset codecs, so the DocType is `webm`, and the
        // Opus track carries an `OpusHead` CodecPrivate.
        assert!(sink.bytes.windows(4).any(|w| w == b"webm"), "WebM DocType for VP9 + Opus");
        assert!(sink.bytes.windows(8).any(|w| w == b"OpusHead"), "Opus CodecPrivate written");

        let mut d = MatroskaDemuxer::new();
        d.push_data(&sink.bytes);
        let tracks = d.tracks();
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].codec, MkvCodec::Vp9);
        assert_eq!((tracks[0].width, tracks[0].height), (320, 240));
        assert_eq!(tracks[1].codec, MkvCodec::Opus);
        assert_eq!(tracks[1].channels, 2);

        let frames = d.take_frames();
        let video: Vec<_> = frames.iter().filter(|f| f.track == 1).collect();
        let audio: Vec<_> = frames.iter().filter(|f| f.track == 2).collect();
        assert_eq!(video.len(), 2, "two VP9 frames");
        assert_eq!(video[0].data, vp9_key(), "VP9 frame stored verbatim (not reframed)");
        assert!(video[0].keyframe, "the key frame is flagged");
        assert_eq!(audio.len(), 2, "two Opus packets");
        assert_eq!(audio[0].data, opus0, "Opus packet stored raw");
    }
}
