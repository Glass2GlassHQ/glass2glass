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
//! sink pads do. Video is H.264/H.265 (avcC/hvcC + AVCC samples), VP8/VP9 (raw
//! frames, no CodecPrivate) or AV1 (`V_AV1`, av1C `CodecPrivate` from the
//! sequence header, temporal delimiters stripped, M773); audio is AAC (ASC) or
//! Opus (an in-band `OpusHead` verbatim, else a synthesised one, plus the
//! `CodecDelay` / `SeekPreRoll` its mapping needs, M792), so VP9 + Opus muxes a
//! WebM. A `Caps::Text{Utf8}` pad adds a subtitle track (M898): one cue per frame,
//! written as a `BlockGroup` whose `BlockDuration` is the cue's display window, in
//! the `S_TEXT/*` syntax the `subtitle-format` property picks. A `Caps::SubPicture`
//! pad adds a bitmap subtitle track (M927): `S_VOBSUB` or `S_DVBSUB` by the pad's
//! format, with the out-of-band configuration each needs (the `.idx` text, the
//! page ids) taken from the config blob the stream carries ahead of its first cue
//! and written as the track's `CodecPrivate`. Every A/V input pad must carry a
//! stream (a pad that ends without an access unit stalls the build).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    split_tags, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    Dim, FrameTiming, G2gError, InputAggregator, MemoryDomain, MultiInputElement, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, SubPictureFormat, TagList,
    TextFormat, VideoCodec,
};

use crate::dvbsub::DEFAULT_PAGE_ID;
use crate::fmp4mux::{
    avcc_record, avcc_sample, hvcc_record, is_keyframe_nal, parameter_sets, split_annexb,
    vp8_keyframe, vp9_keyframe,
};
use crate::matroska::{
    default_page_blob, finalize_seekable, subpicture_block, subpicture_config,
    subpicture_mkv_codec, subtitle_format_from_str, subtitle_format_str, text_codec_private,
    MatroskaMuxer, MkvCodec, MkvTrackConfig, MkvTrackSpec,
};
use crate::mp4muxn::{asc_from_adts, strip_adts};
use crate::opusparse::{is_opus_config, parse_opus_head, synth_opus_head};
use crate::subparse::frame_subtitle_block;

/// What an input pad carries, learned from its negotiated caps at configure.
#[derive(Debug, Clone, Copy)]
enum PadKind {
    Video(VideoCodec),
    Audio {
        format: AudioFormat,
        channels: u8,
        rate: u32,
    },
    /// A timed-text subtitle pad (M898): the pad carries one plain-UTF-8 cue per
    /// frame, timed by the frame's PTS + duration. The on-disk `S_TEXT/*` syntax
    /// is the muxer's `subtitle-format`, not the pad's.
    Text,
    /// A bitmap subtitle pad (M927): each frame is one cue (a VobSub subpicture
    /// unit, a DVB display set), and the out-of-band configuration each format
    /// needs arrives in band ahead of the first cue, the way both demuxers and
    /// `vobsubsrc` send it.
    SubPicture(SubPictureFormat),
}

/// A track's init data, captured from its first access unit. `param_sets` is the
/// H.26x SPS/PPS the avcC/hvcC `CodecPrivate` needs (empty for VP8/VP9, which
/// carry none); `config` is the audio `CodecPrivate` (AAC AudioSpecificConfig or
/// Opus `OpusHead`).
#[derive(Debug, Clone)]
enum TrackInit {
    Video {
        codec: VideoCodec,
        width: u32,
        height: u32,
        param_sets: Vec<Vec<u8>>,
    },
    Audio {
        format: AudioFormat,
        channels: u8,
        rate: u32,
        config: Vec<u8>,
    },
    /// A subtitle track: it needs nothing from the stream, so it is ready at
    /// configure. The on-disk syntax comes from the muxer's `subtitle-format`.
    Text,
    /// A bitmap subtitle track: `codec_private` is the `.idx` text (`S_VOBSUB`)
    /// or the page-id blob (`S_DVBSUB`). Like a text track it is ready at
    /// configure, so a stream whose first cue is minutes in does not hold the
    /// Tracks element back; an in-band config blob replaces the bytes while the
    /// element is still unwritten.
    SubPicture {
        format: SubPictureFormat,
        codec_private: Vec<u8>,
    },
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
    /// Live / streamable mode (the gst `streamable` property): suppress the `Cues`
    /// seek index normally appended at EOS, matching the single-input
    /// [`crate::mkvmux::MkvMux`]. A live sink cannot hold cluster positions to the
    /// end, so `streamable` drops the index and the positions it would collect.
    streamable: bool,
    /// Two-pass / seekable-finalize mode (M770): buffer the whole file and emit
    /// it once at EOS with a front `SeekHead` and an `Info` `Duration` (M794,
    /// see [`crate::mkvmux::MkvMux`]). Mutually exclusive with `streamable`.
    seekable: bool,
    /// The buffered file bytes in `seekable` mode.
    pending: Vec<u8>,
    /// Whole-file metadata, written as an untargeted `Tag`.
    tags: TagList,
    /// Per-input metadata, written as `Targets`-scoped `Tag`s in the same `Tags`
    /// element (M787). One (possibly empty) list per input pad.
    track_tags: Vec<TagList>,
    /// The on-disk syntax a text pad's cues are written in (M898): the `S_TEXT/*`
    /// CodecID and the block framing that goes with it.
    subtitle_format: TextFormat,
    /// Per-pad ASS event counter: the `ReadOrder` field leading each block.
    text_seq: Vec<u64>,
    /// The `dvbsub-page-id` property: the composition and ancillary page an
    /// `S_DVBSUB` track's `CodecPrivate` names when the stream carries no page-id
    /// config blob of its own.
    dvbsub_page_id: u16,
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
            streamable: false,
            seekable: false,
            pending: Vec::new(),
            tags: TagList::new(),
            track_tags: alloc::vec![TagList::new(); inputs],
            subtitle_format: TextFormat::Utf8,
            text_seq: alloc::vec![0; inputs],
            dvbsub_page_id: DEFAULT_PAGE_ID,
        }
    }

    /// Attach whole-file metadata, written as a `Tags` element in the header (the
    /// [`crate::mkvmux::MkvMux`] builder for the multi-track muxer).
    pub fn with_tags(mut self, tags: TagList) -> Self {
        self.tags = tags;
        self
    }

    /// Attach metadata scoped to one input pad's track: written as a `Tag` whose
    /// `Targets` names that track's `TagTrackUID`, so a reader (ffmpeg, the g2g
    /// demuxer) reports it on that elementary stream rather than the file. Out-of
    /// range inputs are ignored.
    ///
    /// A tag every input carries identically moves up to the untargeted
    /// whole-file `Tag` instead (`g2g_core::split_tags`), and a tag also set by
    /// [`with_tags`](Self::with_tags) is not repeated per track unless the value
    /// differs, in which case the track's value stands for that track.
    pub fn with_track_tags(mut self, input: usize, tags: TagList) -> Self {
        if input < self.inputs {
            self.track_tags[input] = tags;
        }
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
    /// an ASS script header `CodecPrivate`). Any other value leaves the default:
    /// the pad always carries plain cue text, this only picks how it is stored.
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

    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output_caps_value() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        }
    }

    fn pad_kind_for(caps: &Caps) -> Option<PadKind> {
        match caps {
            Caps::CompressedVideo {
                codec:
                    c @ (VideoCodec::H264
                    | VideoCodec::H265
                    | VideoCodec::Vp8
                    | VideoCodec::Vp9
                    | VideoCodec::Av1),
                ..
            } => Some(PadKind::Video(*c)),
            Caps::Audio {
                format: format @ (AudioFormat::Aac | AudioFormat::Opus),
                channels,
                sample_rate,
            } => Some(PadKind::Audio {
                format: *format,
                channels: *channels,
                rate: *sample_rate,
            }),
            // Only the elementary cue form: a `Text` pad of a document format
            // (`Srt` / `Ssa` / `Ttml`) carries whole-file bytes, not timed cues.
            Caps::Text {
                format: TextFormat::Utf8,
            } => Some(PadKind::Text),
            // Only a format with a Matroska codec mapping: an `S_HDMV/PGS` track
            // is not one this muxer writes.
            Caps::SubPicture { format } if subpicture_mkv_codec(*format).is_some() => {
                Some(PadKind::SubPicture(*format))
            }
            _ => None,
        }
    }

    /// Whether the pad carries Opus, whose in-band headers are config rather
    /// than audio and so are dropped instead of muxed.
    fn is_opus_pad(&self, input: usize) -> bool {
        matches!(
            self.kinds[input],
            Some(PadKind::Audio {
                format: AudioFormat::Opus,
                ..
            })
        )
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
                            let owned: Vec<Vec<u8>> =
                                param_sets.iter().map(|s| s.to_vec()).collect();
                            self.inits[input] = Some(TrackInit::Video {
                                codec,
                                width: w,
                                height: h,
                                param_sets: owned,
                            });
                        }
                    }
                    // AV1's config record needs the sequence-header OBU; wait
                    // for the temporal unit that carries it (keyframes do).
                    VideoCodec::Av1 => {
                        if let Some((_, obu)) = crate::av1parse::seq_header_obu(au) {
                            self.inits[input] = Some(TrackInit::Video {
                                codec,
                                width: w,
                                height: h,
                                param_sets: alloc::vec![obu.to_vec()],
                            });
                        }
                    }
                    // VP8/VP9 carry no out-of-band parameter sets; the track is
                    // ready at the first frame (its CodecPrivate stays empty).
                    _ => {
                        self.inits[input] = Some(TrackInit::Video {
                            codec,
                            width: w,
                            height: h,
                            param_sets: Vec::new(),
                        });
                    }
                }
            }
            Some(PadKind::Audio {
                format,
                channels,
                rate,
            }) => match format {
                // AAC's AudioSpecificConfig is synthesised from the first ADTS header.
                AudioFormat::Aac => {
                    if let Some(asc) = asc_from_adts(au) {
                        self.inits[input] = Some(TrackInit::Audio {
                            format,
                            channels,
                            rate,
                            config: asc.to_vec(),
                        });
                    }
                }
                // An Opus stream out of a container leads with its `OpusHead`
                // (M791's in-band convention), which becomes the `CodecPrivate`
                // verbatim so the source's real pre-skip survives; a freshly
                // encoded one has none, so the header is synthesized with
                // libopus' lookahead (M792).
                _ => {
                    let config = match parse_opus_head(au) {
                        Some(_) => au.to_vec(),
                        // An `OpusTags` before the identification header is not
                        // the config: wait for it rather than fix a synthesized one.
                        None if is_opus_config(au) => return,
                        None => synth_opus_head(channels, rate),
                    };
                    self.inits[input] = Some(TrackInit::Audio {
                        format,
                        channels,
                        rate,
                        config,
                    });
                }
            },
            // A text or bitmap subtitle track's init is fixed at configure; the
            // bitmap one's `CodecPrivate` is refined by an in-band config blob
            // instead (see `adopt_subpicture_config`).
            Some(PadKind::Text) | Some(PadKind::SubPicture(_)) | None => {}
        }
    }

    /// The SimpleBlock payload for a track and whether it is a keyframe. H.26x is
    /// AVCC length-prefixed (keyframe from the NAL types); VP8/VP9 frames are
    /// stored verbatim (keyframe from the frame header). AAC strips its ADTS
    /// header; Opus packets are stored raw. Audio frames are always sync samples.
    /// A text cue is framed in the track's `S_TEXT/*` syntax (M898).
    fn sample_for(&mut self, input: usize, au: &[u8]) -> (Vec<u8>, bool) {
        match self.kinds[input] {
            Some(PadKind::Video(codec)) => match codec {
                VideoCodec::H264 | VideoCodec::H265 => {
                    let nalus = split_annexb(au);
                    let is_key = nalus.iter().any(|n| is_keyframe_nal(codec, n));
                    (avcc_sample(&nalus), is_key)
                }
                VideoCodec::Vp8 => (au.to_vec(), vp8_keyframe(au)),
                VideoCodec::Av1 => (
                    crate::av1parse::strip_temporal_delimiters(au),
                    crate::av1parse::av1_keyframe(au),
                ),
                _ => (au.to_vec(), vp9_keyframe(au)),
            },
            Some(PadKind::Audio {
                format: AudioFormat::Aac,
                ..
            }) => (strip_adts(au).to_vec(), true),
            Some(PadKind::Text) => {
                let seq = self.text_seq[input];
                self.text_seq[input] = seq.saturating_add(1);
                let text = alloc::string::String::from_utf8_lossy(au);
                let block = frame_subtitle_block(&text, self.subtitle_format, seq);
                (block.into_bytes(), true)
            }
            Some(PadKind::SubPicture(format)) => (subpicture_block(format, au), true),
            _ => (au.to_vec(), true),
        }
    }

    /// Take an in-band config blob as the pad's `CodecPrivate`. `true` when the
    /// frame was one, so it is config rather than a cue and is never written as a
    /// block. Bytes arriving after the Tracks element is written are still
    /// dropped: the track already declared its configuration.
    fn adopt_subpicture_config(&mut self, input: usize, data: &[u8]) -> bool {
        let Some(PadKind::SubPicture(format)) = self.kinds[input] else {
            return false;
        };
        let Some(config) = subpicture_config(format, data) else {
            return false;
        };
        if self.mux.is_none() {
            self.inits[input] = Some(TrackInit::SubPicture {
                format,
                codec_private: config,
            });
        }
        true
    }

    /// Emit one access unit as its track's SimpleBlock (the muxer prepends the
    /// header + Tracks on the first call, and opens Clusters as time advances).
    async fn emit_au(
        &mut self,
        input: usize,
        frame: Frame,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let Some(slice) = frame.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        let pts_ns = frame.timing.pts_ns;
        let (sample, is_key) = self.sample_for(input, slice);
        let mux = self.mux.as_mut().ok_or(G2gError::NotConfigured)?;
        let bytes = mux.push_frame_on(input, &sample, pts_ns, is_key, frame.timing.duration_ns);

        // Seekable (two-pass) mode: hold the whole file until every input drains.
        if self.seekable {
            self.pending.extend_from_slice(&bytes);
            return Ok(());
        }
        let out_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            self.emitted,
        );
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(out_frame)).await?;
        Ok(())
    }
}

/// The muxer track config (spec + `CodecPrivate`) for a captured init: the avcC /
/// hvcC record for H.26x video (none for VP8/VP9), the AudioSpecificConfig for AAC
/// or the `OpusHead` for Opus. A text track's `CodecPrivate` is the ASS script
/// header when it is stored as `S_TEXT/ASS`, and nothing otherwise.
fn track_config(init: &TrackInit, subtitle_format: TextFormat) -> MkvTrackConfig {
    match init {
        TrackInit::Video {
            codec,
            width,
            height,
            param_sets,
        } => {
            let refs: Vec<&[u8]> = param_sets.iter().map(|v| v.as_slice()).collect();
            let (mkv_codec, codec_private) = match codec {
                VideoCodec::H265 => (MkvCodec::H265, hvcc_record(&refs)),
                VideoCodec::Vp8 => (MkvCodec::Vp8, Vec::new()),
                VideoCodec::Vp9 => (MkvCodec::Vp9, Vec::new()),
                // The captured init is the sequence-header OBU; its parse
                // succeeded at capture, so the re-parse here cannot fail.
                VideoCodec::Av1 => {
                    let obu = refs.first().copied().unwrap_or(&[]);
                    let record = crate::av1parse::seq_header_obu(obu)
                        .map(|(seq, _)| crate::fmp4mux::av1c_record(&seq, obu))
                        .unwrap_or_default();
                    (MkvCodec::Av1, record)
                }
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
        TrackInit::Audio {
            format,
            channels,
            rate,
            config,
        } => {
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
        TrackInit::Text => MkvTrackConfig {
            spec: MkvTrackSpec::subtitle(MkvCodec::Subtitle(subtitle_format)),
            codec_private: text_codec_private(subtitle_format),
        },
        TrackInit::SubPicture {
            format,
            codec_private,
        } => MkvTrackConfig {
            // Gated at `pad_kind_for`, so the mapping is always there.
            spec: MkvTrackSpec::subtitle(subpicture_mkv_codec(*format).unwrap_or(MkvCodec::Other)),
            codec_private: codec_private.clone(),
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
        if let Caps::CompressedVideo {
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } = absolute_caps
        {
            self.dims[input] = (*w, *h);
        }
        // A text track's `TrackEntry` needs nothing from the stream, so it is
        // ready now: the first cue can be many seconds in, and the Tracks element
        // (which waits on every track) would hold the A/V until then.
        if matches!(kind, PadKind::Text) {
            self.inits[input] = Some(TrackInit::Text);
        }
        // Same for a bitmap subtitle track, whose configuration arrives in band:
        // it starts on what the muxer knows (nothing for VobSub, the
        // `dvbsub-page-id` pages for DVB) and the config blob replaces it.
        if let PadKind::SubPicture(format) = kind {
            self.inits[input] = Some(TrackInit::SubPicture {
                format,
                codec_private: match format {
                    SubPictureFormat::DvbSub => default_page_blob(self.dvbsub_page_id),
                    _ => Vec::new(),
                },
            });
        }
        self.kinds[input] = Some(kind);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(Self::output_caps_value())
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
                let v = u16::try_from(value.as_uint().ok_or(PropError::Type)?)
                    .map_err(|_| PropError::Value)?;
                let stale = default_page_blob(self.dvbsub_page_id);
                self.dvbsub_page_id = v;
                // Re-stamp a pad configured before this call, unless its stream
                // has already named its own pages.
                for init in self.inits.iter_mut() {
                    if let Some(TrackInit::SubPicture {
                        format: SubPictureFormat::DvbSub,
                        codec_private,
                    }) = init
                    {
                        if *codec_private == stale {
                            *codec_private = default_page_blob(v);
                        }
                    }
                }
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
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    // Capture this track's init from its first AU before queueing.
                    if let Some(s) = frame.domain.as_system_slice() {
                        self.capture_init(input, s);
                        // An in-band Opus header is codec config, not audio: it
                        // became the `CodecPrivate` above and must never be
                        // written as a Block (M792).
                        if self.is_opus_pad(input) && is_opus_config(s) {
                            return Ok(());
                        }
                        // A bitmap subtitle pad opens on its config blob (the
                        // `.idx` text, the page ids), which becomes the track's
                        // `CodecPrivate` and is never a cue to write.
                        if self.adopt_subpicture_config(input, s) {
                            return Ok(());
                        }
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
                let subtitle_format = self.subtitle_format;
                let configs: Vec<MkvTrackConfig> = self
                    .inits
                    .iter()
                    .map(|i| track_config(i.as_ref().expect("ready"), subtitle_format))
                    .collect();
                let (global, per_track) = split_tags(&self.tags, &self.track_tags);
                let mut mux = MatroskaMuxer::new_multi(configs).with_tags(global);
                for (input, tags) in per_track.into_iter().enumerate() {
                    if !tags.is_empty() {
                        mux = mux.with_track_tags(input, tags);
                    }
                }
                if self.seekable {
                    mux = mux.with_two_pass();
                }
                if self.streamable {
                    mux = mux.without_cues();
                }
                self.mux = Some(mux);
            }
            // Release AUs now safe to emit, in global PTS order.
            while let Some((track, frame)) = self.agg.take_earliest_by(|f| f.timing.pts_ns) {
                self.emit_au(track, frame, out).await?;
            }
            // Once every track has ended and drained, flush the Cues index after
            // the last Cluster so the stream is seekable on a read-to-end (M375).
            // In `streamable` (live) mode the index is suppressed. Seekable
            // (two-pass) mode instead finalizes the buffered file: the Cues are
            // appended and the front SeekHead's placeholder patched to their
            // position (M770), then the whole file emits at once.
            if self.agg.is_drained() && !self.cues_emitted {
                if self.seekable {
                    if let Some(mux) = self.mux.as_ref() {
                        finalize_seekable(mux, &mut self.pending);
                    }
                    if !self.pending.is_empty() {
                        let file = core::mem::take(&mut self.pending);
                        let out_frame = Frame::new(
                            MemoryDomain::System(SystemSlice::from_boxed(file.into_boxed_slice())),
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
    async fn av_streams_mux_into_two_matroska_tracks() {
        let mut mux = MkvMuxN::new(2);
        mux.configure_pipeline(0, &video_caps()).unwrap();
        mux.configure_pipeline(1, &audio_caps()).unwrap();

        let mut sink = CaptureSink::default();
        // Interleave a video IDR and an audio frame; the merge needs both inputs
        // to have queued before it releases, so push both then a second each.
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
        assert!(
            !d.cues().is_empty(),
            "Cues index written for the video keyframes"
        );

        // CodecPrivate is present for both tracks (avcC record, AAC ASC): the
        // bytes carry the avcC config-version byte and the A_AAC CodecID.
        assert!(
            sink.bytes.windows(5).any(|w| w == b"A_AAC"),
            "AAC CodecID written"
        );
        assert!(
            sink.bytes
                .windows(4)
                .any(|w| w == b"\x63\xA2\x00\x00" || w[0] == 0x63 && w[1] == 0xA2),
            "CodecPrivate element present"
        );
        assert!(mux.emitted() >= 4, "all four access units muxed");
    }

    /// The `streamable` knob is honored on the fan-in muxer (the `name=m` shape),
    /// the same as the single-track `MkvMux`: no `Cues` seek index is appended at
    /// EOS. Set via `set_property`, the path `parse_launch` uses.
    #[tokio::test]
    async fn streamable_property_omits_cues_on_the_fan_in_muxer() {
        let mut mux = MkvMuxN::new(2);
        mux.set_property("streamable", PropValue::Bool(true))
            .unwrap();
        assert_eq!(mux.get_property("streamable"), Some(PropValue::Bool(true)));
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

        // Cues element id is 0x1C53BB6B; it must not appear in streamable mode.
        assert!(
            !sink.bytes.windows(4).any(|w| w == [0x1C, 0x53, 0xBB, 0x6B]),
            "no Cues index written in streamable mode"
        );

        // A non-streamable mux of the same input does write the Cues index.
        let mut plain = MkvMuxN::new(2);
        plain.configure_pipeline(0, &video_caps()).unwrap();
        plain.configure_pipeline(1, &audio_caps()).unwrap();
        let mut sink2 = CaptureSink::default();
        plain
            .process(0, frame(h264_idr(), 0), &mut sink2)
            .await
            .unwrap();
        plain
            .process(1, frame(aac_adts(), 0), &mut sink2)
            .await
            .unwrap();
        plain
            .process(0, frame(h264_idr(), 33_000_000), &mut sink2)
            .await
            .unwrap();
        plain
            .process(1, frame(aac_adts(), 21_000_000), &mut sink2)
            .await
            .unwrap();
        plain
            .process(0, PipelinePacket::Eos, &mut sink2)
            .await
            .unwrap();
        plain
            .process(1, PipelinePacket::Eos, &mut sink2)
            .await
            .unwrap();
        assert!(
            sink2
                .bytes
                .windows(4)
                .any(|w| w == [0x1C, 0x53, 0xBB, 0x6B]),
            "default mode writes the Cues index"
        );
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
        Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
        }
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
        mux.process(0, frame(vp9_key(), 0), &mut sink)
            .await
            .unwrap();
        mux.process(1, frame(opus0.clone(), 0), &mut sink)
            .await
            .unwrap();
        mux.process(0, frame(vp9_key(), 20_000_000), &mut sink)
            .await
            .unwrap();
        mux.process(1, frame(opus1.clone(), 20_000_000), &mut sink)
            .await
            .unwrap();
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        mux.process(1, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();

        // VP9 + Opus are both WebM-subset codecs, so the DocType is `webm`, and the
        // Opus track carries an `OpusHead` CodecPrivate.
        assert!(
            sink.bytes.windows(4).any(|w| w == b"webm"),
            "WebM DocType for VP9 + Opus"
        );
        assert!(
            sink.bytes.windows(8).any(|w| w == b"OpusHead"),
            "Opus CodecPrivate written"
        );

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
        assert_eq!(
            video[0].data,
            vp9_key(),
            "VP9 frame stored verbatim (not reframed)"
        );
        assert!(video[0].keyframe, "the key frame is flagged");
        assert_eq!(audio.len(), 2, "two Opus packets");
        assert_eq!(audio[0].data, opus0, "Opus packet stored raw");
    }
}
