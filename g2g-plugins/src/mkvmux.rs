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
//! A `Caps::Text{Utf8}` or `Caps::SubPicture` input is written as a subtitle
//! track instead (M928): one cue per frame, as a `BlockGroup` whose
//! `BlockDuration` is the cue's display window, in the `S_TEXT/*` syntax the
//! `subtitle-format` property picks or the `S_VOBSUB` / `S_DVBSUB` mapping of the
//! bitmap format. A bitmap pad's out-of-band configuration (the `.idx` text, the
//! DVB page ids) comes from the config blob the stream carries ahead of its first
//! cue, and becomes the track's `CodecPrivate`. So a sidecar subtitle file muxes
//! with one link rather than the fan-in shape:
//!
//! ```text
//! vobsubsrc location=movie.idx ! matroskamux ! filesink location=subs.mkv
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
    Dim, ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate,
    SubPictureFormat, TagList, TextFormat, VideoCodec,
};

use crate::dvbsub::DEFAULT_PAGE_ID;
use crate::matroska::{
    default_page_blob, finalize_seekable, subpicture_block, subpicture_config,
    subpicture_mkv_codec, subtitle_format_from_str, subtitle_format_str, text_codec_private,
    MatroskaMuxer, MkvCodec, MkvTrackConfig, MkvTrackSpec,
};
use crate::subparse::frame_subtitle_block;

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
    /// have to hold the cluster positions to the end; `streamable` drops it (and
    /// the positions it would have collected) so the output is a pure forward
    /// stream: an unknown-size Segment and unknown-size Clusters, nothing patched
    /// at EOS. Off by default (a recording stays seekable).
    streamable: bool,
    /// Two-pass / seekable-finalize mode (M770): buffer the whole file and emit
    /// it once at EOS with a front `SeekHead` indexing Info / Tracks / Cues, so
    /// the file seeks from byte 0 without reading past the Clusters, and with a
    /// real `Info` `Duration` (M794), which only a caller holding the whole file
    /// can fill in. Costs one file-sized buffer; mutually exclusive with
    /// `streamable`, whose live stream has no length to declare.
    seekable: bool,
    /// The buffered file bytes in `seekable` mode.
    pending: Vec<u8>,
    /// The on-disk syntax a `Caps::Text` pad's cues are written in (M928): the
    /// `S_TEXT/*` CodecID and the block framing that goes with it. The pad always
    /// carries plain cue text; this only picks how it is stored.
    subtitle_format: TextFormat,
    /// The ASS event counter: the `ReadOrder` field leading each text block.
    text_seq: u64,
    /// The `dvbsub-page-id` property: the composition and ancillary page an
    /// `S_DVBSUB` track's `CodecPrivate` names when the stream carries no page-id
    /// config blob of its own.
    dvbsub_page_id: u16,
    /// The `CodecPrivate` a bitmap subtitle pad's in-band config frame carried
    /// (the `.idx` text, the page ids), or `None` before one arrives.
    subpicture_private: Option<Vec<u8>>,
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
            seekable: false,
            pending: Vec::new(),
            subtitle_format: TextFormat::Utf8,
            text_seq: 0,
            dvbsub_page_id: DEFAULT_PAGE_ID,
            subpicture_private: None,
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

    /// Two-pass mode: buffer the file and finalize it at EOS with a front
    /// `SeekHead` (see the field note).
    pub fn with_seekable(mut self, seekable: bool) -> Self {
        self.seekable = seekable;
        self
    }

    /// The `S_TEXT/*` syntax a text pad's cues are written in: [`TextFormat::Utf8`]
    /// (the default, `S_TEXT/UTF8`, which ffmpeg reports as `subrip`) or
    /// [`TextFormat::Ssa`] (`S_TEXT/ASS`, cues framed as `Dialogue` fields behind
    /// an ASS script header `CodecPrivate`). Any other value leaves the default.
    pub fn with_subtitle_format(mut self, format: TextFormat) -> Self {
        if subtitle_format_str(format).is_some() {
            self.subtitle_format = format;
        }
        self
    }

    /// The composition and ancillary page an `S_DVBSUB` track's `CodecPrivate`
    /// names for a stream that carries no page-id config blob of its own. A blob
    /// wins over this.
    pub fn with_dvbsub_page_id(mut self, page_id: u16) -> Self {
        self.dvbsub_page_id = page_id;
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
    /// A text pad's storage syntax is the muxer's `subtitle-format`, which the
    /// caps does not name; a bitmap subtitle pad's codec is its format's mapping.
    fn track_spec(caps: &Caps, subtitle_format: TextFormat) -> Option<MkvTrackSpec> {
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
            // Only the elementary cue form: a `Text` pad of a document format
            // (`Srt` / `Ssa` / `Ttml`) carries whole-file bytes, not timed cues.
            Caps::Text {
                format: TextFormat::Utf8,
            } => Some(MkvTrackSpec::subtitle(MkvCodec::Subtitle(subtitle_format))),
            Caps::SubPicture { format } => {
                Some(MkvTrackSpec::subtitle(subpicture_mkv_codec(*format)?))
            }
            _ => None,
        }
    }

    /// The track's `CodecPrivate`: the ASS script header a text track needs to be
    /// read, or the out-of-band configuration a bitmap subtitle track declares
    /// (the config blob its stream carried, else what the muxer knows: nothing
    /// for VobSub, the `dvbsub-page-id` pages for DVB). The A/V streams this
    /// muxer writes carry their parameter sets in band, so they need none.
    fn codec_private(&self) -> Vec<u8> {
        match self.caps {
            Some(Caps::Text { .. }) => text_codec_private(self.subtitle_format),
            Some(Caps::SubPicture { format }) => {
                self.subpicture_private
                    .clone()
                    .unwrap_or_else(|| match format {
                        SubPictureFormat::DvbSub => default_page_blob(self.dvbsub_page_id),
                        _ => Vec::new(),
                    })
            }
            _ => Vec::new(),
        }
    }

    /// The block payload for one access unit: a text cue framed in the track's
    /// `S_TEXT/*` syntax, a bitmap cue in the framing its Matroska mapping takes,
    /// anything else verbatim.
    fn sample_for(&mut self, au: &[u8]) -> Vec<u8> {
        match self.caps {
            Some(Caps::Text { .. }) => {
                let seq = self.text_seq;
                self.text_seq = seq.saturating_add(1);
                let text = alloc::string::String::from_utf8_lossy(au);
                frame_subtitle_block(&text, self.subtitle_format, seq).into_bytes()
            }
            Some(Caps::SubPicture { format }) => subpicture_block(format, au),
            _ => au.to_vec(),
        }
    }

    /// Take an in-band config blob as the pad's `CodecPrivate`. `true` when the
    /// frame was one, so it is config rather than a cue and is never written as a
    /// block. Bytes arriving after the Tracks element is written are still
    /// dropped: the track already declared its configuration.
    fn adopt_subpicture_config(&mut self, data: &[u8]) -> bool {
        let Some(Caps::SubPicture { format }) = self.caps else {
            return false;
        };
        let Some(config) = subpicture_config(format, data) else {
            return false;
        };
        if self.mux.is_none() {
            self.subpicture_private = Some(config);
        }
        true
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
            Caps::Text {
                format: TextFormat::Utf8,
            },
            Caps::SubPicture {
                format: SubPictureFormat::VobSub,
            },
            Caps::SubPicture {
                format: SubPictureFormat::DvbSub,
            },
        ])
    }
}

impl AsyncElement for MkvMux {
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Matroska muxer",
            "Codec/Muxer",
            "Muxes one elementary stream into a Matroska / WebM file",
            "g2g",
        )
    }
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::track_spec(upstream_caps, self.subtitle_format).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let subtitle_format = self.subtitle_format;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            if Self::track_spec(input, subtitle_format).is_some() {
                CapsSet::one(Self::output_caps())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Self::track_spec(absolute_caps, self.subtitle_format).ok_or(G2gError::CapsMismatch)?;
        self.caps = Some(absolute_caps.clone());
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "streamable",
                PropKind::Bool,
                "live mode: omit the seekable Cues index written at EOS",
            )
            .with_default("false"),
            PropertySpec::new(
                "seekable",
                PropKind::Bool,
                "two-pass mode: buffer the file and finalize with a front SeekHead",
            )
            .with_default("false"),
            PropertySpec::new(
                "subtitle-format",
                PropKind::Str,
                "storage syntax for a text input: utf8 (S_TEXT/UTF8) | ass (S_TEXT/ASS)",
            )
            .with_default("utf8"),
            PropertySpec::new(
                "dvbsub-page-id",
                PropKind::Uint,
                "composition and ancillary page an S_DVBSUB track declares when the stream carries no page-id config",
            )
            .with_default("1"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            // streamable (pure forward stream) and seekable (whole-file buffer)
            // are opposites; setting both is a configuration error.
            "streamable" => {
                let v = value.as_bool().ok_or(PropError::Type)?;
                if v && self.seekable {
                    return Err(PropError::Type);
                }
                self.streamable = v;
                Ok(())
            }
            "seekable" => {
                let v = value.as_bool().ok_or(PropError::Type)?;
                if v && self.streamable {
                    return Err(PropError::Type);
                }
                self.seekable = v;
                Ok(())
            }
            "subtitle-format" => {
                let v = value.as_str().ok_or(PropError::Type)?;
                self.subtitle_format = subtitle_format_from_str(v).ok_or(PropError::Value)?;
                Ok(())
            }
            "dvbsub-page-id" => {
                self.dvbsub_page_id = u16::try_from(value.as_uint().ok_or(PropError::Type)?)
                    .map_err(|_| PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "streamable" => Some(PropValue::Bool(self.streamable)),
            "seekable" => Some(PropValue::Bool(self.seekable)),
            "subtitle-format" => Some(PropValue::Str(
                subtitle_format_str(self.subtitle_format)
                    .unwrap_or("utf8")
                    .into(),
            )),
            "dvbsub-page-id" => Some(PropValue::Uint(self.dvbsub_page_id as u64)),
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // A bitmap subtitle pad opens on its config blob (the `.idx`
                    // text, the page ids), which becomes the track's
                    // `CodecPrivate` and is never a cue to write.
                    if self.adopt_subpicture_config(slice) {
                        return Ok(());
                    }
                    let sample = self.sample_for(slice);
                    if self.mux.is_none() {
                        let caps = self.caps.as_ref().ok_or(G2gError::NotConfigured)?;
                        let spec = Self::track_spec(caps, self.subtitle_format)
                            .ok_or(G2gError::CapsMismatch)?;
                        let mut mux = MatroskaMuxer::new_multi(alloc::vec![MkvTrackConfig {
                            spec,
                            codec_private: self.codec_private(),
                        }])
                        .with_tags(self.tags.clone());
                        if self.seekable {
                            mux = mux.with_two_pass();
                        }
                        if self.streamable {
                            mux = mux.without_cues();
                        }
                        self.mux = Some(mux);
                    }
                    let mux = self.mux.as_mut().ok_or(G2gError::NotConfigured)?;
                    // No upstream delta-frame signal yet: flag every frame a keyframe.
                    let bytes = mux.push_frame_on(
                        0,
                        &sample,
                        frame.timing.pts_ns,
                        true,
                        frame.timing.duration_ns,
                    );
                    // Seekable (two-pass) mode: hold the whole file until EOS.
                    if self.seekable {
                        self.pending.extend_from_slice(&bytes);
                        return Ok(());
                    }
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
                    if self.mux.is_none() && Self::track_spec(&c, self.subtitle_format).is_some() {
                        self.caps = Some(c);
                    }
                }
                // At EOS, flush the Cues index after the last Cluster so the stream
                // is seekable on a read-to-end (M375); the runner then forwards EOS.
                // Seekable (two-pass) mode instead finalizes the buffered file: the
                // Cues are appended and the front SeekHead's placeholder patched to
                // their position (M770), then the whole file emits at once.
                PipelinePacket::Eos => {
                    if self.seekable {
                        if let Some(mux) = self.mux.as_ref() {
                            finalize_seekable(mux, &mut self.pending);
                        }
                        if !self.pending.is_empty() {
                            let file = core::mem::take(&mut self.pending);
                            let out_frame = Frame::new(
                                MemoryDomain::System(SystemSlice::from_boxed(
                                    file.into_boxed_slice(),
                                )),
                                FrameTiming::default(),
                                self.emitted,
                            );
                            self.emitted += 1;
                            out.push(PipelinePacket::DataFrame(out_frame)).await?;
                        }
                    } else if let Some(mux) = self.mux.as_ref().filter(|_| !self.streamable) {
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
                        if let Some(s) = f.domain.as_system_slice() {
                            self.frames.push(s.to_vec());
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
            if let BusMessage::Tag { tags: t, .. } = m {
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
