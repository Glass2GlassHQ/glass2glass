#[cfg(feature = "alloc")]
use alloc::format;
#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::channels::ChannelLayout;
use crate::error::G2gError;
use crate::tensor::MAX_TENSOR_RANK;

/// Sentinel sample rate meaning "any / unspecified" in [`Caps::Audio`] (M187).
/// `Caps::Audio.sample_rate` is a bare `u32` (not a ranged [`Dim`]); 0 Hz is
/// never a real rate, so it serves as the wildcard a caps-driven element
/// (`audioresample`) advertises so a downstream capsfilter can pin the rate.
/// `intersect` treats it as a wildcard and `fixate` rejects it (like
/// [`Dim::Any`]).
pub const ANY_SAMPLE_RATE: u32 = 0;

/// Sentinel channel count meaning "any / unknown" in [`Caps::Audio`]. Like
/// [`ANY_SAMPLE_RATE`], `0` is never a real channel count, so it serves as the
/// wildcard for two cases: a compressed stream whose layout is unknown until the
/// bitstream is parsed (a demuxer emits `Aac { channels: 0, .. }`), and a decoder
/// that defers its real channel count to a runtime `CapsChanged` (it advertises
/// `PcmS16Le { channels: 0, .. }` at negotiation). `intersect` treats it as a
/// wildcard in *both* the compressed and PCM cases (so a decoder's output channels
/// coupling back onto a `0` input is not an empty link); `fixate` collapses a PCM
/// `0` to a concrete stereo placeholder (the real layout arrives via `CapsChanged`,
/// mirroring video `Dim::Any` -> 16). A compressed `0` stays nominal (unfixed-but-
/// fixed, like a compressed `ANY_SAMPLE_RATE`), since nothing downstream of a
/// demuxer reads it before the decoder replaces it.
pub const ANY_CHANNELS: u8 = 0;

/// The placeholder channel count a PCM [`Caps::Audio`] with [`ANY_CHANNELS`]
/// fixates to (stereo): a concrete value the negotiation can pin while the stream's
/// real layout is still unknown, replaced by the decoder's first `CapsChanged`.
const FIXATE_CHANNELS_PLACEHOLDER: u8 = 2;

/// Caps describes one fixated (or partially-narrowed) link.
///
/// Video is split into [`Caps::CompressedVideo`] and [`Caps::RawVideo`]
/// because a codec bitstream and a raw pixel buffer are *different
/// kinds* of media, not different values of one "format" slot. A raw
/// sink (waylandsink, kmssink) intercepting a `CompressedVideo` caps is
/// a category error; the type system now expresses that as a variant
/// mismatch rather than a runtime enum compare. (Mirrors GStreamer's
/// `video/x-h264` vs `video/x-raw` distinction; M17 split.)
///
/// Both video variants carry geometry today. That's pragmatic, not
/// honest: GStreamer's `video/x-h264` doesn't have width/height because
/// they live in the SPS. Our solver, the RtspSrc placeholder Range, and
/// our `Range`-as-placeholder convention all hang off geometry on
/// compressed caps. Dropping it is a deeper rework that overlaps
/// workaround #1's redesign; out of scope here.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Caps {
    /// Compressed video bitstream (codec). Width/height/framerate are
    /// nominal until the bitstream parser confirms them via SPS/equivalent.
    CompressedVideo {
        codec: VideoCodec,
        width: Dim,
        height: Dim,
        framerate: Rate,
        /// Colour description from the bitstream (VUI / `color_config`),
        /// [`Colorimetry::UNKNOWN`] until a parser reads it.
        colorimetry: Colorimetry,
    },
    /// Raw pixel buffer in `format`. Geometry is authoritative.
    RawVideo {
        format: RawVideoFormat,
        width: Dim,
        height: Dim,
        framerate: Rate,
        interlace: Interlace,
        /// How the samples map to colour, [`Colorimetry::UNKNOWN`] when the
        /// producer does not know (a sink then applies its own default).
        colorimetry: Colorimetry,
    },
    Audio {
        format: AudioFormat,
        channels: u8,
        sample_rate: u32,
        /// Which speaker each interleaved channel feeds,
        /// [`ChannelLayout::UNSPECIFIED`] when the producer does not know. A
        /// consumer resolves an unspecified layout with
        /// [`ChannelLayout::or_default_for`] the channel count, so it behaves
        /// as a bare count.
        channel_layout: ChannelLayout,
    },
    /// A tensor stream (ML). Its `shape` ([`TensorShape`]) is a fixed-rank
    /// inline array (M636), so the variant is part of the no-alloc MCU subset
    /// like every other caps kind.
    Tensor {
        dtype: TensorDType,
        shape: TensorShape,
        layout: TensorLayout,
    },
    /// An opaque container / elementary byte stream, not yet demuxed or parsed
    /// into a typed media stream. The link type between a byte source (a file or
    /// network source carrying e.g. an MPEG-TS) and a demuxer that splits it into
    /// elementary streams. `encoding` names the wire format so a demuxer only
    /// accepts a stream it understands.
    ByteStream { encoding: ByteStreamEncoding },
    /// A text stream (subtitles, captions, transcription, OCR, overlay strings).
    /// `format` names the syntax ([`TextFormat`]); the payload is UTF-8 bytes in
    /// the frame's system buffer, and "subtitle" is just timed `Text` (cue PTS +
    /// duration on [`FrameTiming`](crate::frame::FrameTiming)). One kind, not a
    /// per-use-case variant, so an overlay, a caption sink, and a text analytics
    /// element all negotiate the same caps.
    Text { format: TextFormat },
    /// A KLV (SMPTE ST 336 key-length-value) metadata stream, each frame one KLV
    /// packet (for STANAG 4609 UAS streams, a MISB ST 0601 local set). The
    /// elementary metadata stream a transport demuxer splits out alongside video,
    /// timed by the frame's PTS (GStreamer `meta/x-klv`).
    Klv,
    /// A raw closed-caption stream: a container track carrying caption data as its
    /// own elementary stream rather than inside a video bitstream's SEI (the MP4
    /// `c608` / `c708` raw-caption tracks). Each frame is one sample's `cc_data`
    /// byte triples, `(marker | cc_valid | cc_type, cc_data_1, cc_data_2)`, the
    /// same ATSC A/53 layout an SEI caption block carries, so one caption decoder
    /// serves both paths. `format` names which carriage the track declared, and
    /// therefore which services its triples can hold: 608 line-21 fields, or 708
    /// DTVCC packets. Captions embedded in a coded video stream stay
    /// [`Caps::CompressedVideo`]; this is the separate-track case only.
    ClosedCaption { format: ClosedCaptionFormat },
    /// A coded bitmap-subtitle (subpicture) stream: each frame one cue's coded
    /// bitmap, not pixels. The bitmap subtitle counterpart of [`Caps::Text`],
    /// which stays the timed-text kind. `format` names the coding
    /// ([`SubPictureFormat`]); a subpicture decoder turns it into full-frame
    /// transparent [`Caps::RawVideo`] RGBA canvases a compositor can paint over
    /// video, so nothing downstream needs a bitmap-cue concept.
    SubPicture { format: SubPictureFormat },
}

impl Caps {
    /// Whether this caps carries a raw, uncompressed heavy media buffer: raw
    /// pixels, PCM audio samples, or a tensor. These are the payloads whose bytes
    /// a memory-domain crossing actually copies, so the copy/allocation plan
    /// counts a domain transfer as a real frame copy only between two raw caps
    /// (a codec boundary like `CompressedVideo` -> `RawVideo` is a decode, not a
    /// raw-frame copy). Compressed streams, opaque byte streams, and text are not
    /// raw media.
    pub fn is_raw_media(&self) -> bool {
        match self {
            Caps::RawVideo { .. } => true,
            Caps::Tensor { .. } => true,
            Caps::Audio { format, .. } => is_pcm(*format),
            Caps::CompressedVideo { .. }
            | Caps::ByteStream { .. }
            | Caps::Text { .. }
            | Caps::Klv
            | Caps::ClosedCaption { .. }
            | Caps::SubPicture { .. } => false,
        }
    }

    /// Phase 1 intersection (DESIGN.md §4.2). Narrow `self` against `other`,
    /// returning the overlap. Both must be the same variant; ranged fields
    /// (`Dim`/`Rate`) intersect field-wise, scalar fields (`codec` /
    /// `format`, `channels`, `sample_rate`, tensor dtype/shape/layout) must
    /// be equal. Any empty field overlap, variant mismatch, or scalar
    /// mismatch yields `CapsMismatch`.
    ///
    /// `CompressedVideo` and `RawVideo` are distinct variants — a raw
    /// sink offered compressed input gets `CapsMismatch` structurally,
    /// not a runtime format compare.
    pub fn intersect(&self, other: &Caps) -> Result<Caps, G2gError> {
        match (self, other) {
            (
                Caps::CompressedVideo {
                    codec: ca,
                    width: wa,
                    height: ha,
                    framerate: ra,
                    colorimetry: cla,
                },
                Caps::CompressedVideo {
                    codec: cb,
                    width: wb,
                    height: hb,
                    framerate: rb,
                    colorimetry: clb,
                },
            ) if ca == cb => Ok(Caps::CompressedVideo {
                codec: *ca,
                width: wa.intersect(wb).ok_or(G2gError::CapsMismatch)?,
                height: ha.intersect(hb).ok_or(G2gError::CapsMismatch)?,
                framerate: ra.intersect(rb).ok_or(G2gError::CapsMismatch)?,
                colorimetry: cla.intersect(clb).ok_or(G2gError::CapsMismatch)?,
            }),
            (
                Caps::RawVideo {
                    format: fa,
                    width: wa,
                    height: ha,
                    framerate: ra,
                    interlace: ia,
                    colorimetry: cla,
                },
                Caps::RawVideo {
                    format: fb,
                    width: wb,
                    height: hb,
                    framerate: rb,
                    interlace: ib,
                    colorimetry: clb,
                },
            ) if fa == fb => Ok(Caps::RawVideo {
                format: *fa,
                width: wa.intersect(wb).ok_or(G2gError::CapsMismatch)?,
                height: ha.intersect(hb).ok_or(G2gError::CapsMismatch)?,
                framerate: ra.intersect(rb).ok_or(G2gError::CapsMismatch)?,
                interlace: ia.intersect(ib).ok_or(G2gError::CapsMismatch)?,
                colorimetry: cla.intersect(clb).ok_or(G2gError::CapsMismatch)?,
            }),
            (
                Caps::Audio {
                    format: fa,
                    channels: ca,
                    sample_rate: sa,
                    channel_layout: la,
                },
                Caps::Audio {
                    format: fb,
                    channels: cb,
                    sample_rate: sb,
                    channel_layout: lb,
                },
            ) if fa == fb => {
                // Channels use the `ANY_CHANNELS` (0) wildcard in *both* the
                // compressed and PCM cases: a decoder's concrete output channels
                // coupling back onto a demuxer's unknown `0` input must not be an
                // empty link. The "any rate" wildcard (M187) is a raw-PCM concept
                // only: a caps-driven resampler leaves its output rate open, while
                // compressed audio (AAC/Opus) uses `sample_rate: 0` as "unknown
                // until parsed" and keeps strict equality, unchanged.
                let channels = intersect_channels(*ca, *cb);
                let rate = if is_pcm(*fa) {
                    intersect_sample_rate(*sa, *sb)
                } else {
                    (sa == sb).then_some(*sa)
                };
                // An unspecified layout is the wildcard, so a producer that knows
                // its speakers pins them onto a peer that does not, and two
                // different declared layouts refuse the link.
                let layout = la.intersect(*lb);
                match (channels, rate, layout) {
                    (Some(channels), Some(sample_rate), Some(channel_layout)) => Ok(Caps::Audio {
                        format: *fa,
                        channels,
                        sample_rate,
                        channel_layout,
                    }),
                    _ => Err(G2gError::CapsMismatch),
                }
            }
            (
                Caps::Tensor {
                    dtype: da,
                    shape: sha,
                    layout: la,
                },
                Caps::Tensor {
                    dtype: db,
                    shape: shb,
                    layout: lb,
                },
            ) if da == db && sha == shb && la == lb => Ok(self.clone()),
            (Caps::ByteStream { encoding: ea }, Caps::ByteStream { encoding: eb }) if ea == eb => {
                Ok(self.clone())
            }
            (Caps::Text { format: fa }, Caps::Text { format: fb }) if fa == fb => Ok(self.clone()),
            (Caps::Klv, Caps::Klv) => Ok(Caps::Klv),
            (Caps::ClosedCaption { format: a }, Caps::ClosedCaption { format: b }) if a == b => {
                Ok(Caps::ClosedCaption { format: *a })
            }
            (Caps::SubPicture { format: a }, Caps::SubPicture { format: b }) if a == b => {
                Ok(Caps::SubPicture { format: *a })
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// True when every ranged field is `Fixed`. Scalar-only variants are
    /// always fixed.
    pub fn is_fixed(&self) -> bool {
        if let Caps::Audio {
            format,
            channels,
            sample_rate,
            ..
        } = self
        {
            // Only raw PCM uses the "any rate" / "any channels" wildcards;
            // compressed audio keeps `0` as a fixed (if nominal) value, since the
            // decoder replaces it before anything reads it.
            return !(is_pcm(*format)
                && (*sample_rate == ANY_SAMPLE_RATE || *channels == ANY_CHANNELS));
        }
        match self.dims() {
            Some((width, height, framerate)) => {
                width.is_fixed() && height.is_fixed() && framerate.is_fixed()
            }
            None => true,
        }
    }

    /// Phase 2 fixation (DESIGN.md §4.2): collapse every ranged field to a
    /// single `Fixed` value. `Range` fixates to its **minimum**, reflecting
    /// the latency-first design (less data is lower latency); an element
    /// preferring a different value counter-proposes via
    /// `ConfigureOutcome::ReFixate`.
    /// `Any` carries no information to fixate against and yields
    /// `CapsMismatch`.
    pub fn fixate(&self) -> Result<Caps, G2gError> {
        match self {
            Caps::CompressedVideo {
                codec,
                width,
                height,
                framerate,
                colorimetry,
            } => Ok(Caps::CompressedVideo {
                codec: *codec,
                width: width.fixate().ok_or(G2gError::CapsMismatch)?,
                height: height.fixate().ok_or(G2gError::CapsMismatch)?,
                framerate: framerate.fixate().ok_or(G2gError::CapsMismatch)?,
                // Colorimetry `Unknown` survives fixation like `Interlace::Any`
                // below: the bitstream refines it later, and collapsing it to a
                // guess is exactly the wrong-matrix bug this field exists to fix.
                colorimetry: *colorimetry,
            }),
            Caps::RawVideo {
                format,
                width,
                height,
                framerate,
                interlace,
                colorimetry,
            } => Ok(Caps::RawVideo {
                format: *format,
                width: width.fixate().ok_or(G2gError::CapsMismatch)?,
                height: height.fixate().ok_or(G2gError::CapsMismatch)?,
                framerate: framerate.fixate().ok_or(G2gError::CapsMismatch)?,
                // `Interlace::Any` already means "progressive unless declared"
                // (GStreamer's absent field), so it is concrete enough to keep:
                // collapsing it would only churn solved-caps equalities.
                interlace: *interlace,
                colorimetry: *colorimetry,
            }),
            // A raw-PCM "any" sample rate carries no value to fixate against
            // (M187); compressed audio's nominal `0` fixates as-is.
            Caps::Audio {
                format,
                sample_rate,
                ..
            } if is_pcm(*format) && *sample_rate == ANY_SAMPLE_RATE => Err(G2gError::CapsMismatch),
            // A raw-PCM "any channels" collapses to a concrete stereo placeholder:
            // the negotiation needs a fixed count, the stream's real layout arrives
            // via the decoder's first `CapsChanged` (mirrors video `Dim::Any` -> 16).
            Caps::Audio {
                format,
                channels,
                sample_rate,
                channel_layout,
            } if is_pcm(*format) && *channels == ANY_CHANNELS => Ok(Caps::Audio {
                format: *format,
                channels: FIXATE_CHANNELS_PLACEHOLDER,
                sample_rate: *sample_rate,
                // An unspecified layout survives fixation like `Colorimetry::UNKNOWN`:
                // guessing speakers for a count that is itself a placeholder would
                // only be wrong twice.
                channel_layout: *channel_layout,
            }),
            Caps::Audio { .. }
            | Caps::ByteStream { .. }
            | Caps::Text { .. }
            | Caps::Klv
            | Caps::ClosedCaption { .. }
            | Caps::SubPicture { .. } => Ok(self.clone()),
            Caps::Tensor { .. } => Ok(self.clone()),
        }
    }

    /// Borrow the geometry triple if this caps carries one. Both video
    /// variants (compressed + raw) currently do; `Audio` and `Tensor`
    /// return `None`. Used by element code that needs width/height/fps
    /// without caring whether the link is pre- or post-decode.
    pub fn dims(&self) -> Option<(&Dim, &Dim, &Rate)> {
        match self {
            Caps::CompressedVideo {
                width,
                height,
                framerate,
                ..
            }
            | Caps::RawVideo {
                width,
                height,
                framerate,
                ..
            } => Some((width, height, framerate)),
            Caps::Audio { .. }
            | Caps::ByteStream { .. }
            | Caps::Text { .. }
            | Caps::Klv
            | Caps::ClosedCaption { .. }
            | Caps::SubPicture { .. } => None,
            Caps::Tensor { .. } => None,
        }
    }

    /// Render these caps as a GStreamer caps string, the inverse of
    /// [`CapsSet::from_gst_string`]. For `-v` pipeline dumps, logs, and porting
    /// diagnostics. The fixed media types round-trip through the parser;
    /// `Tensor` has no GStreamer media type and is rendered as a g2g-specific
    /// `tensor/x-raw` descriptor.
    #[cfg(feature = "alloc")]
    pub fn to_gst_string(&self) -> String {
        match self {
            Caps::RawVideo {
                format,
                width,
                height,
                framerate,
                interlace,
                colorimetry,
            } => {
                let mut s = format!("video/x-raw,format={}", raw_format_gst_name(*format));
                push_dim(&mut s, "width", width);
                push_dim(&mut s, "height", height);
                push_rate(&mut s, framerate);
                // Progressive is GStreamer's default for an absent field and
                // `Any` is the wildcard (absence), so only interleaved prints.
                if *interlace == Interlace::Interleaved {
                    s.push_str(",interlace-mode=interleaved");
                }
                push_colorimetry(&mut s, colorimetry);
                s
            }
            Caps::CompressedVideo {
                codec,
                width,
                height,
                framerate,
                colorimetry,
            } => {
                let mut s = String::from(codec_gst_media_type(*codec));
                push_dim(&mut s, "width", width);
                push_dim(&mut s, "height", height);
                push_rate(&mut s, framerate);
                push_colorimetry(&mut s, colorimetry);
                s
            }
            Caps::Audio {
                format,
                channels,
                sample_rate,
                channel_layout,
            } => {
                let (media_type, fmt) = audio_gst_media_type(*format);
                let mut s = String::from(media_type);
                if let Some(f) = fmt {
                    s.push_str(&format!(",format={f}"));
                }
                if *channels != 0 {
                    s.push_str(&format!(",channels={channels}"));
                }
                if *sample_rate != ANY_SAMPLE_RATE && *sample_rate != 0 {
                    s.push_str(&format!(",rate={sample_rate}"));
                }
                push_channel_mask(&mut s, *channel_layout);
                s
            }
            // No GStreamer media type for tensors; a g2g-specific descriptor.
            Caps::Tensor {
                dtype,
                shape,
                layout,
            } => {
                format!("tensor/x-raw,dtype={dtype:?},layout={layout:?},shape={shape:?}")
            }
            Caps::ByteStream { encoding } => String::from(bytestream_gst_media_type(*encoding)),
            Caps::Text { format } => String::from(text_format_gst_media_type(*format)),
            Caps::Klv => String::from("meta/x-klv"),
            Caps::ClosedCaption { format } => String::from(cc_format_gst_media_type(*format)),
            Caps::SubPicture { format } => String::from(subpicture_gst_media_type(*format)),
        }
    }
}

/// GStreamer media-type string for a [`TextFormat`]. Plain / markup text is
/// `text/x-raw` (with a `format=`); the structured subtitle formats carry their
/// own `application/x-subtitle-*` media types, mirroring GStreamer.
#[cfg(feature = "alloc")]
fn text_format_gst_media_type(f: TextFormat) -> &'static str {
    match f {
        TextFormat::Utf8 => "text/x-raw,format=utf8",
        TextFormat::PangoMarkup => "text/x-raw,format=pango-markup",
        TextFormat::Srt => "application/x-subtitle",
        TextFormat::WebVtt => "application/x-subtitle-vtt",
        TextFormat::Ssa => "application/x-ssa",
        TextFormat::Ttml => "application/ttml+xml",
        TextFormat::Teletext => "private/teletext",
    }
}

/// GStreamer media-type string for a [`ClosedCaptionFormat`]. The media types are
/// GStreamer's; `format=cc_data` (the ATSC triple layout) is GStreamer's own
/// spelling for 708 and g2g's for 608, which GStreamer instead carries as
/// `s334-1a` triplets (a different marker byte).
#[cfg(feature = "alloc")]
fn cc_format_gst_media_type(f: ClosedCaptionFormat) -> &'static str {
    match f {
        ClosedCaptionFormat::Cea608 => "closedcaption/x-cea-608,format=cc_data",
        ClosedCaptionFormat::Cea608Raw => "closedcaption/x-cea-608,format=raw",
        ClosedCaptionFormat::Cea608S334 => "closedcaption/x-cea-608,format=s334-1a",
        ClosedCaptionFormat::Cea708 => "closedcaption/x-cea-708,format=cc_data",
        ClosedCaptionFormat::Cea708Cdp => "closedcaption/x-cea-708,format=cdp",
    }
}

/// GStreamer media-type string for a [`SubPictureFormat`]. `subpicture/x-dvd` is
/// GStreamer's own type for the DVD SPU stream `dvdsubdec` consumes,
/// `subpicture/x-dvb` the one its `dvbsuboverlay` takes, and `subpicture/x-pgs`
/// the one its `matroskademux` puts on an `S_HDMV/PGS` track.
#[cfg(feature = "alloc")]
fn subpicture_gst_media_type(f: SubPictureFormat) -> &'static str {
    match f {
        SubPictureFormat::VobSub => "subpicture/x-dvd",
        SubPictureFormat::DvbSub => "subpicture/x-dvb",
        SubPictureFormat::Pgs => "subpicture/x-pgs",
    }
}

/// The GStreamer `format=` name for a raw video format (uppercase, the M182
/// vocabulary the parser also accepts).
#[cfg(feature = "alloc")]
fn raw_format_gst_name(f: RawVideoFormat) -> &'static str {
    match f {
        RawVideoFormat::Nv12 => "NV12",
        RawVideoFormat::I420 => "I420",
        RawVideoFormat::Rgba8 => "RGBA",
        RawVideoFormat::Bgra8 => "BGRA",
        RawVideoFormat::Rgb8 => "RGB",
        RawVideoFormat::Yuyv => "YUY2",
        RawVideoFormat::I420p10 => "I420_10LE",
        RawVideoFormat::I420p12 => "I420_12LE",
        RawVideoFormat::I422 => "Y42B",
        RawVideoFormat::I422p10 => "I422_10LE",
        RawVideoFormat::I422p12 => "I422_12LE",
        RawVideoFormat::I444 => "Y444",
        RawVideoFormat::I444p10 => "Y444_10LE",
        RawVideoFormat::I444p12 => "Y444_12LE",
        RawVideoFormat::P010 => "P010_10LE",
    }
}

/// The GStreamer media type for a compressed video codec.
#[cfg(feature = "alloc")]
fn codec_gst_media_type(c: VideoCodec) -> &'static str {
    match c {
        VideoCodec::H264 => "video/x-h264",
        VideoCodec::H265 => "video/x-h265",
        VideoCodec::Av1 => "video/x-av1",
        VideoCodec::Vp8 => "video/x-vp8",
        VideoCodec::Vp9 => "video/x-vp9",
        VideoCodec::Mjpeg => "image/jpeg",
        VideoCodec::Png => "image/png",
        VideoCodec::WebP => "image/webp",
        // GStreamer distinguishes MPEG versions with a `mpegversion` field on
        // `video/mpeg`; g2g's media-type string carries no fields, so MPEG-1/2
        // video and MPEG-4 Part 2 share it here the way mp2 and AAC share
        // `audio/mpeg` below (the codec split lives in the caps).
        VideoCodec::Mpeg4Part2 | VideoCodec::Mpeg2 => "video/mpeg",
        // JPEG XS codestream (GStreamer's `jpegxsdec` / `jpegxsenc` caps).
        VideoCodec::JpegXs => "image/x-jxsc",
        VideoCodec::SorensonH263 => "video/x-flash-video",
        VideoCodec::Vp6 { alpha: false } => "video/x-vp6-flash",
        VideoCodec::Vp6 { alpha: true } => "video/x-vp6-alpha",
        // GStreamer keeps VC-1 under the Windows Media Video type, told apart
        // by `wmvversion=3, format=WVC1`; the fields do not survive here, so the
        // codec split stays in the caps.
        VideoCodec::Vc1 => "video/x-wmv",
        // The four Netpbm media types collapse onto one codec; the PBM / PGM /
        // PPM split lives in the file magic, not in caps.
        VideoCodec::Pnm => "image/x-portable-anymap",
    }
}

/// The GStreamer media type (and raw `format=` name, if raw) for an audio format.
#[cfg(feature = "alloc")]
fn audio_gst_media_type(f: AudioFormat) -> (&'static str, Option<&'static str>) {
    match f {
        AudioFormat::Aac => ("audio/mpeg", None),
        AudioFormat::Opus => ("audio/x-opus", None),
        // gst distinguishes mp2 from AAC by mpegversion/layer on audio/mpeg; the
        // bare media type is the same. This helper carries no version field, so
        // mp2 shares the AAC media type here (the codec split lives in the caps).
        AudioFormat::Mp2 => ("audio/mpeg", None),
        // Same story as mp2: gst separates mp3 by mpegversion/layer fields.
        AudioFormat::Mp3 => ("audio/mpeg", None),
        AudioFormat::Speex => ("audio/x-speex", None),
        AudioFormat::Ac3 => ("audio/x-ac3", None),
        AudioFormat::Flac => ("audio/x-flac", None),
        AudioFormat::Vorbis => ("audio/x-vorbis", None),
        // Named in `PCM_FORMATS`, but still listed so a format added to the enum
        // has to be given a media type here rather than falling into a wildcard.
        AudioFormat::PcmS16Le
        | AudioFormat::PcmF32Le
        | AudioFormat::PcmS24Le
        | AudioFormat::PcmS32Le
        | AudioFormat::PcmU8 => ("audio/x-raw", pcm_gst_format(f)),
        AudioFormat::Mulaw => ("audio/x-mulaw", None),
        AudioFormat::Alaw => ("audio/x-alaw", None),
        AudioFormat::ImaAdpcm => ("audio/x-adpcm", None),
    }
}

/// The GStreamer media type for a container byte stream.
#[cfg(feature = "alloc")]
fn bytestream_gst_media_type(e: ByteStreamEncoding) -> &'static str {
    match e {
        ByteStreamEncoding::MpegTs => "video/mpegts",
        ByteStreamEncoding::Matroska => "video/x-matroska",
        ByteStreamEncoding::Ogg => "application/ogg",
        ByteStreamEncoding::Flv => "video/x-flv",
        ByteStreamEncoding::IsoBmff => "video/quicktime",
        ByteStreamEncoding::Mp4 => "video/quicktime",
        ByteStreamEncoding::Ivf => "video/x-ivf",
        ByteStreamEncoding::MpegPs => "video/mpeg",
        ByteStreamEncoding::Wav => "audio/x-wav",
        ByteStreamEncoding::Aiff => "audio/x-aiff",
        ByteStreamEncoding::Au => "audio/x-au",
        ByteStreamEncoding::Avi => "video/x-msvideo",
        ByteStreamEncoding::Y4m => "application/x-yuv4mpeg",
        ByteStreamEncoding::Multipart => "multipart/x-mixed-replace",
        ByteStreamEncoding::Raw => "application/octet-stream",
        ByteStreamEncoding::Rtp => "application/x-rtp",
        ByteStreamEncoding::Srtp => "application/x-srtp",
        ByteStreamEncoding::Rtcp => "application/x-rtcp",
        ByteStreamEncoding::Srtcp => "application/x-srtcp",
        ByteStreamEncoding::Dtls => "application/x-dtls",
    }
}

/// Append `,key=value` for a fixed dimension; omit `Any` / `Range` (a wildcard
/// is the absence of the field in GStreamer caps).
#[cfg(feature = "alloc")]
fn push_dim(s: &mut String, key: &str, d: &Dim) {
    if let Dim::Fixed(v) = d {
        s.push_str(&format!(",{key}={v}"));
    }
}

/// Append `,framerate=N/D` for a fixed rate (Q16 fps). A whole-number fps prints
/// as `fps/1`; otherwise the exact Q16 value prints as `q16/65536`, which the
/// parser reads back to the same Q16.
#[cfg(feature = "alloc")]
fn push_rate(s: &mut String, r: &Rate) {
    if let Rate::Fixed(q16) = r {
        if q16 % 65536 == 0 {
            s.push_str(&format!(",framerate={}/1", q16 >> 16));
        } else {
            s.push_str(&format!(",framerate={q16}/65536"));
        }
    }
}

/// Append `,colorimetry=value` for a non-`UNKNOWN` colorimetry (an unknown one
/// is the absent field, like a wildcard dimension).
#[cfg(feature = "alloc")]
fn push_colorimetry(s: &mut String, c: &Colorimetry) {
    if let Some(v) = c.to_gst_string() {
        s.push_str(&format!(",colorimetry={v}"));
    }
}

/// Append `,channel-mask=(bitmask)0x...` for a declared layout (an unspecified
/// one is the absent field). gst prints the mask as 16 hex digits.
#[cfg(feature = "alloc")]
fn push_channel_mask(s: &mut String, layout: ChannelLayout) {
    if !layout.is_unspecified() {
        s.push_str(&format!(
            ",channel-mask=(bitmask)0x{:016x}",
            layout.to_gst_mask()
        ));
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    Any,
    Range { min: u32, max: u32 },
    Fixed(u32),
}

impl Dim {
    /// Intersect two dimension constraints. `Any` is the identity; two
    /// `Range`s overlap to their tighter bounds (collapsing to `Fixed` when
    /// the bounds meet); disjoint constraints yield `None`.
    pub fn intersect(&self, other: &Dim) -> Option<Dim> {
        intersect_range(self.bounds(), other.bounds()).map(Dim::from_bounds)
    }

    pub fn is_fixed(&self) -> bool {
        matches!(self, Dim::Fixed(_))
    }

    /// Collapse to a single `Fixed` value: `Range` picks its minimum, `Any`
    /// has nothing to pick and yields `None`. An inverted range (`min > max`)
    /// is the empty set, as [`Dim::intersect`] treats it, so it also yields
    /// `None` rather than a value outside the set. See [`Caps::fixate`].
    pub fn fixate(&self) -> Option<Dim> {
        match self {
            Dim::Fixed(v) => Some(Dim::Fixed(*v)),
            Dim::Range { min, max } => (min <= max).then_some(Dim::Fixed(*min)),
            Dim::Any => None,
        }
    }

    fn bounds(&self) -> (u32, u32) {
        match self {
            Dim::Any => (u32::MIN, u32::MAX),
            Dim::Range { min, max } => (*min, *max),
            Dim::Fixed(v) => (*v, *v),
        }
    }

    fn from_bounds((min, max): (u32, u32)) -> Dim {
        match (min, max) {
            (lo, hi) if lo == hi => Dim::Fixed(lo),
            (u32::MIN, u32::MAX) => Dim::Any, // full span is unconstrained
            (lo, hi) => Dim::Range { min: lo, max: hi },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rate {
    Any,
    /// Min/max framerate in Q16 fixed-point fps.
    Range {
        min_q16: u32,
        max_q16: u32,
    },
    /// Framerate in Q16 fixed-point fps.
    Fixed(u32),
}

impl Rate {
    /// Intersect two framerate constraints over their Q16 values; same
    /// semantics as [`Dim::intersect`].
    pub fn intersect(&self, other: &Rate) -> Option<Rate> {
        intersect_range(self.bounds(), other.bounds()).map(Rate::from_bounds)
    }

    pub fn is_fixed(&self) -> bool {
        matches!(self, Rate::Fixed(_))
    }

    /// Collapse to a single `Fixed` value: `Range` picks its minimum, `Any`
    /// yields `None`. An inverted range (`min_q16 > max_q16`) is the empty set,
    /// as [`Rate::intersect`] treats it, so it also yields `None`. See
    /// [`Caps::fixate`].
    pub fn fixate(&self) -> Option<Rate> {
        match self {
            Rate::Fixed(v) => Some(Rate::Fixed(*v)),
            Rate::Range { min_q16, max_q16 } => {
                (min_q16 <= max_q16).then_some(Rate::Fixed(*min_q16))
            }
            Rate::Any => None,
        }
    }

    fn bounds(&self) -> (u32, u32) {
        match self {
            Rate::Any => (u32::MIN, u32::MAX),
            Rate::Range { min_q16, max_q16 } => (*min_q16, *max_q16),
            Rate::Fixed(v) => (*v, *v),
        }
    }

    fn from_bounds((min, max): (u32, u32)) -> Rate {
        match (min, max) {
            (lo, hi) if lo == hi => Rate::Fixed(lo),
            (u32::MIN, u32::MAX) => Rate::Any, // full span is unconstrained
            (lo, hi) => Rate::Range {
                min_q16: lo,
                max_q16: hi,
            },
        }
    }
}

/// Scan structure of raw video (M935). `Any` is the unconstrained default
/// nearly every caps site uses, and it reads as "progressive unless declared"
/// (GStreamer's meaning for an absent `interlace-mode`): a decoder that sees
/// interlaced pictures refines it to `Interleaved` via `CapsChanged`, which is
/// what `deinterlace mode=auto` acts on. Unlike `Dim::Any` / `Rate::Any` the
/// wildcard survives `fixate` and counts as fixed, so the field never blocks a
/// solve and pre-M935 negotiations are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interlace {
    Any,
    Progressive,
    /// Both fields woven into one frame, top field first (ffmpeg's default
    /// for a stream that declares no field order).
    Interleaved,
}

impl Interlace {
    /// Intersect two scan constraints: `Any` is the identity, equal values
    /// pass, `Progressive` vs `Interleaved` is an empty overlap.
    pub fn intersect(&self, other: &Interlace) -> Option<Interlace> {
        match (self, other) {
            (Interlace::Any, x) | (x, Interlace::Any) => Some(*x),
            (a, b) if a == b => Some(*a),
            _ => None,
        }
    }
}

/// YUV <-> RGB matrix coefficients of a video stream, the CICP
/// `matrix_coefficients` vocabulary (H.273, shared by the H.264 / H.265 VUI and
/// the AV1 `color_config`). `Unknown` is the wildcard every untagged stream
/// carries; a codepoint outside the modeled set also maps to `Unknown`, so a
/// malformed stream cannot smuggle a bogus value into caps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MatrixCoefficients {
    Unknown,
    /// Identity: the "YUV" planes are actually GBR (CICP 0, sRGB video).
    Identity,
    /// BT.601 (BT.470BG / SMPTE 170M, CICP 5 / 6): same coefficients, so the
    /// two codepoints share one variant (the primaries keep them apart).
    Bt601,
    /// BT.709 (CICP 1).
    Bt709,
    /// BT.2020 non-constant luminance (CICP 9), the HDR / wide-gamut matrix.
    Bt2020Ncl,
}

/// The codepoint every CICP colour field spells "unspecified", what `Unknown`
/// writes out and what `from_cicp` reads back as `Unknown`.
const CICP_UNSPECIFIED: u8 = 2;

impl MatrixCoefficients {
    /// The variant a CICP `matrix_coefficients` codepoint names; anything not
    /// modeled (unspecified 2 included) is `Unknown`.
    pub const fn from_cicp(codepoint: u8) -> Self {
        match codepoint {
            0 => MatrixCoefficients::Identity,
            1 => MatrixCoefficients::Bt709,
            5 | 6 => MatrixCoefficients::Bt601,
            9 => MatrixCoefficients::Bt2020Ncl,
            _ => MatrixCoefficients::Unknown,
        }
    }

    /// The CICP `matrix_coefficients` codepoint an encoder writes for this
    /// variant. `Bt601` writes 6 (SMPTE 170M) of the two codepoints that share
    /// its coefficients.
    pub const fn to_cicp(self) -> u8 {
        match self {
            MatrixCoefficients::Unknown => CICP_UNSPECIFIED,
            MatrixCoefficients::Identity => 0,
            MatrixCoefficients::Bt709 => 1,
            MatrixCoefficients::Bt601 => 6,
            MatrixCoefficients::Bt2020Ncl => 9,
        }
    }
}

/// The luma weights `(Kr, Kb)` that define a YUV matrix; the green weight is
/// [`kg`](Self::kg), whatever the other two leave. Every YUV <-> RGB coefficient
/// a converter, sink or shader uses is derived from this pair, and these are the
/// only place the numbers appear.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LumaCoefficients {
    pub kr: f32,
    pub kb: f32,
}

impl LumaCoefficients {
    /// BT.601 (BT.470BG / SMPTE 170M).
    pub const BT601: Self = Self {
        kr: 0.299,
        kb: 0.114,
    };
    /// BT.709.
    pub const BT709: Self = Self {
        kr: 0.2126,
        kb: 0.0722,
    };
    /// BT.2020 non-constant luminance.
    pub const BT2020_NCL: Self = Self {
        kr: 0.2627,
        kb: 0.0593,
    };

    /// The green weight, `1 - Kr - Kb`.
    pub const fn kg(self) -> f32 {
        1.0 - self.kr - self.kb
    }
}

/// The YUV <-> RGB conversion a converter or sink applies to one stream, as
/// [`Colorimetry::yuv_conversion`] resolves it: which matrix, its luma weights,
/// and whether the samples span 0..255 (full) or the studio range (limited).
/// Transfer and primaries are not part of it: nothing here tone-maps or
/// gamut-maps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YuvConversion {
    /// The matrix the weights came from, for a converter that has to declare
    /// what it produced.
    pub matrix: MatrixCoefficients,
    pub luma: LumaCoefficients,
    pub full_range: bool,
}

/// Opto-electronic transfer function of a video stream, the CICP
/// `transfer_characteristics` vocabulary. See [`MatrixCoefficients`] for the
/// `Unknown` / unmodeled-codepoint rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TransferCharacteristics {
    Unknown,
    /// sRGB (IEC 61966-2-1, CICP 13).
    Srgb,
    /// BT.601 (SMPTE 170M, CICP 6).
    Bt601,
    /// BT.709 (CICP 1).
    Bt709,
    /// BT.2020 10- / 12-bit (CICP 14 / 15): the same curve as BT.709, kept as
    /// its own variant because streams tag it distinctly.
    Bt2020,
    /// PQ (SMPTE ST 2084 / BT.2100, CICP 16), the HDR10 transfer.
    Pq,
    /// HLG (ARIB STD-B67 / BT.2100, CICP 18).
    Hlg,
}

impl TransferCharacteristics {
    /// The variant a CICP `transfer_characteristics` codepoint names; anything
    /// not modeled is `Unknown`.
    pub const fn from_cicp(codepoint: u8) -> Self {
        match codepoint {
            1 => TransferCharacteristics::Bt709,
            6 => TransferCharacteristics::Bt601,
            13 => TransferCharacteristics::Srgb,
            14 | 15 => TransferCharacteristics::Bt2020,
            16 => TransferCharacteristics::Pq,
            18 => TransferCharacteristics::Hlg,
            _ => TransferCharacteristics::Unknown,
        }
    }

    /// The CICP `transfer_characteristics` codepoint an encoder writes for this
    /// variant. `Bt2020` writes 14 (the 10-bit codepoint) of the two it covers.
    pub const fn to_cicp(self) -> u8 {
        match self {
            TransferCharacteristics::Unknown => CICP_UNSPECIFIED,
            TransferCharacteristics::Bt709 => 1,
            TransferCharacteristics::Bt601 => 6,
            TransferCharacteristics::Srgb => 13,
            TransferCharacteristics::Bt2020 => 14,
            TransferCharacteristics::Pq => 16,
            TransferCharacteristics::Hlg => 18,
        }
    }
}

/// Colour primaries of a video stream, the CICP `colour_primaries` vocabulary.
/// See [`MatrixCoefficients`] for the `Unknown` / unmodeled-codepoint rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorPrimaries {
    Unknown,
    /// BT.709 (CICP 1), also sRGB's primaries.
    Bt709,
    /// BT.470BG (CICP 5), 625-line BT.601.
    Bt470bg,
    /// SMPTE 170M (CICP 6), 525-line BT.601.
    Smpte170m,
    /// BT.2020 (CICP 9), the wide-gamut / HDR primaries.
    Bt2020,
}

impl ColorPrimaries {
    /// The variant a CICP `colour_primaries` codepoint names; anything not
    /// modeled is `Unknown`.
    pub const fn from_cicp(codepoint: u8) -> Self {
        match codepoint {
            1 => ColorPrimaries::Bt709,
            5 => ColorPrimaries::Bt470bg,
            6 => ColorPrimaries::Smpte170m,
            9 => ColorPrimaries::Bt2020,
            _ => ColorPrimaries::Unknown,
        }
    }

    /// The CICP `colour_primaries` codepoint an encoder writes for this variant.
    pub const fn to_cicp(self) -> u8 {
        match self {
            ColorPrimaries::Unknown => CICP_UNSPECIFIED,
            ColorPrimaries::Bt709 => 1,
            ColorPrimaries::Bt470bg => 5,
            ColorPrimaries::Smpte170m => 6,
            ColorPrimaries::Bt2020 => 9,
        }
    }
}

/// Quantization range of a video stream's samples: `Limited` is studio swing
/// (Y 16..235 at 8 bit), `Full` is 0..255.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorRange {
    Unknown,
    Limited,
    Full,
}

/// Colorimetry of a video caps (both [`Caps::RawVideo`] and
/// [`Caps::CompressedVideo`]): the four fields of GStreamer's `colorimetry`
/// caps value, each with an `Unknown` wildcard. Like [`Interlace`], `Unknown`
/// survives `fixate` and counts as fixed, so an untagged stream negotiates
/// exactly as before the field existed: a bitstream parser refines it from the
/// VUI / `color_config` via `CapsChanged`, and a converter or sink consumes the
/// concrete value through [`yuv_conversion`](Self::yuv_conversion), which
/// resolves the matrix and range it converts with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Colorimetry {
    pub range: ColorRange,
    pub matrix: MatrixCoefficients,
    pub transfer: TransferCharacteristics,
    pub primaries: ColorPrimaries,
}

impl Default for Colorimetry {
    /// [`Colorimetry::UNKNOWN`]: no colour information.
    fn default() -> Self {
        Colorimetry::UNKNOWN
    }
}

impl Colorimetry {
    /// Fully unknown: what every construction site that has no colour
    /// information declares, and the wildcard that intersects with anything.
    pub const UNKNOWN: Self = Self {
        range: ColorRange::Unknown,
        matrix: MatrixCoefficients::Unknown,
        transfer: TransferCharacteristics::Unknown,
        primaries: ColorPrimaries::Unknown,
    };

    /// GStreamer's `bt601` preset: limited range, BT.601 matrix + transfer,
    /// SMPTE 170M primaries. The SD default.
    pub const BT601: Self = Self {
        range: ColorRange::Limited,
        matrix: MatrixCoefficients::Bt601,
        transfer: TransferCharacteristics::Bt601,
        primaries: ColorPrimaries::Smpte170m,
    };

    /// GStreamer's `bt709` preset: limited range, BT.709 throughout. The HD
    /// default.
    pub const BT709: Self = Self {
        range: ColorRange::Limited,
        matrix: MatrixCoefficients::Bt709,
        transfer: TransferCharacteristics::Bt709,
        primaries: ColorPrimaries::Bt709,
    };

    /// GStreamer's `bt2020` preset: limited range, BT.2020 NCL matrix, BT.2020
    /// transfer and primaries. SDR wide-gamut UHD.
    pub const BT2020: Self = Self {
        range: ColorRange::Limited,
        matrix: MatrixCoefficients::Bt2020Ncl,
        transfer: TransferCharacteristics::Bt2020,
        primaries: ColorPrimaries::Bt2020,
    };

    /// GStreamer's `bt2100-pq` preset: BT.2020 with the PQ transfer (HDR10).
    pub const BT2100_PQ: Self = Self {
        range: ColorRange::Limited,
        matrix: MatrixCoefficients::Bt2020Ncl,
        transfer: TransferCharacteristics::Pq,
        primaries: ColorPrimaries::Bt2020,
    };

    /// GStreamer's `bt2100-hlg` preset: BT.2020 with the HLG transfer.
    pub const BT2100_HLG: Self = Self {
        range: ColorRange::Limited,
        matrix: MatrixCoefficients::Bt2020Ncl,
        transfer: TransferCharacteristics::Hlg,
        primaries: ColorPrimaries::Bt2020,
    };

    /// GStreamer's `sRGB` preset: full range, identity (GBR) matrix, sRGB
    /// transfer, BT.709 primaries. What an RGB source or converter output is.
    pub const SRGB: Self = Self {
        range: ColorRange::Full,
        matrix: MatrixCoefficients::Identity,
        transfer: TransferCharacteristics::Srgb,
        primaries: ColorPrimaries::Bt709,
    };

    /// What a JFIF file (ITU-T T.871) pins down: full-range BT.601 YCbCr over
    /// sRGB. A JPEG bitstream carries no colour signalling of its own, so this
    /// is the colorimetry of every baseline JPEG, and a JPEG encoder writes it
    /// on its output whatever the input was tagged.
    pub const JPEG: Self = Self {
        range: ColorRange::Full,
        matrix: MatrixCoefficients::Bt601,
        transfer: TransferCharacteristics::Srgb,
        primaries: ColorPrimaries::Bt709,
    };

    /// Resolve from raw CICP codepoints + the coded full-range flag, the shape
    /// an H.264 / H.265 VUI colour description or an AV1 `color_config`
    /// carries. Unmodeled codepoints (unspecified 2 included) become `Unknown`
    /// per field; the range is always concrete, since the flag is coded.
    pub const fn from_cicp(
        colour_primaries: u8,
        transfer_characteristics: u8,
        matrix_coefficients: u8,
        video_full_range_flag: bool,
    ) -> Self {
        Self {
            range: if video_full_range_flag {
                ColorRange::Full
            } else {
                ColorRange::Limited
            },
            matrix: MatrixCoefficients::from_cicp(matrix_coefficients),
            transfer: TransferCharacteristics::from_cicp(transfer_characteristics),
            primaries: ColorPrimaries::from_cicp(colour_primaries),
        }
    }

    /// Field-wise intersection: `Unknown` is the identity on each field, equal
    /// concrete values pass, two different concrete values are an empty
    /// overlap (`None`). So an untagged link never blocks a tagged peer, and a
    /// wrongly tagged link fails loud instead of silently converting with the
    /// wrong matrix.
    pub fn intersect(&self, other: &Colorimetry) -> Option<Colorimetry> {
        fn field<T: PartialEq + Copy>(a: T, b: T, unknown: T) -> Option<T> {
            if a == unknown {
                Some(b)
            } else if b == unknown || a == b {
                Some(a)
            } else {
                None
            }
        }
        Some(Colorimetry {
            range: field(self.range, other.range, ColorRange::Unknown)?,
            matrix: field(self.matrix, other.matrix, MatrixCoefficients::Unknown)?,
            transfer: field(
                self.transfer,
                other.transfer,
                TransferCharacteristics::Unknown,
            )?,
            primaries: field(self.primaries, other.primaries, ColorPrimaries::Unknown)?,
        })
    }

    /// The matrix and range a YUV <-> RGB converter or sink applies to a stream
    /// carrying this colorimetry. An `Unknown` matrix or range falls back to
    /// BT.601 limited, what every converter assumed before caps carried colour;
    /// `Identity` names GBR planes, which no converter here handles, and takes
    /// the same fallback. This is the only place that fallback lives, so a
    /// better guess (BT.709 above SD, say) is a change to this function alone.
    pub const fn yuv_conversion(self) -> YuvConversion {
        let (matrix, luma) = match self.matrix {
            MatrixCoefficients::Bt709 => (MatrixCoefficients::Bt709, LumaCoefficients::BT709),
            MatrixCoefficients::Bt2020Ncl => {
                (MatrixCoefficients::Bt2020Ncl, LumaCoefficients::BT2020_NCL)
            }
            _ => (MatrixCoefficients::Bt601, LumaCoefficients::BT601),
        };
        YuvConversion {
            matrix,
            luma,
            full_range: matches!(self.range, ColorRange::Full),
        }
    }

    /// The `(preset, colorimetry)` pairs of GStreamer's named colorimetries.
    /// One table so printing and parsing cannot drift.
    const GST_PRESETS: [(&'static str, Colorimetry); 6] = [
        ("bt709", Colorimetry::BT709),
        ("bt601", Colorimetry::BT601),
        ("bt2020", Colorimetry::BT2020),
        ("bt2100-pq", Colorimetry::BT2100_PQ),
        ("bt2100-hlg", Colorimetry::BT2100_HLG),
        ("sRGB", Colorimetry::SRGB),
    ];

    /// Render as a GStreamer `colorimetry=` value: a preset name when the four
    /// fields match one (`bt709`, `sRGB`, ...), the numeric
    /// `range:matrix:transfer:primaries` 4-part form otherwise, and `None` for
    /// fully [`UNKNOWN`](Self::UNKNOWN) (the field is omitted, GStreamer's
    /// spelling of an absent constraint).
    #[cfg(feature = "alloc")]
    pub fn to_gst_string(&self) -> Option<String> {
        if *self == Colorimetry::UNKNOWN {
            return None;
        }
        for (name, preset) in Self::GST_PRESETS {
            if *self == preset {
                return Some(String::from(name));
            }
        }
        Some(format!(
            "{}:{}:{}:{}",
            gst_range_num(self.range),
            gst_matrix_num(self.matrix),
            gst_transfer_num(self.transfer),
            gst_primaries_num(self.primaries)
        ))
    }

    /// Parse a GStreamer `colorimetry=` value: a preset name
    /// (case-insensitive, so `srgb` works) or the numeric
    /// `range:matrix:transfer:primaries` form. `None` on anything else,
    /// including a GStreamer enum number this model does not carry: a filter
    /// pinning e.g. SMPTE 240M cannot be honored, so it fails loud rather than
    /// widening to a wildcard.
    pub fn from_gst_string(s: &str) -> Option<Colorimetry> {
        let s = s.trim();
        for (name, preset) in Self::GST_PRESETS {
            if s.eq_ignore_ascii_case(name) {
                return Some(preset);
            }
        }
        let mut parts = s.split(':');
        let range = gst_range_from_num(parts.next()?.parse().ok()?)?;
        let matrix = gst_matrix_from_num(parts.next()?.parse().ok()?)?;
        let transfer = gst_transfer_from_num(parts.next()?.parse().ok()?)?;
        let primaries = gst_primaries_from_num(parts.next()?.parse().ok()?)?;
        if parts.next().is_some() {
            return None;
        }
        Some(Colorimetry {
            range,
            matrix,
            transfer,
            primaries,
        })
    }
}

// GStreamer numeric codes for the 4-part colorimetry string. These are
// GStreamer's own enum values (GstVideoColorRange etc.), NOT CICP codepoints.
#[cfg(feature = "alloc")]
fn gst_range_num(v: ColorRange) -> u8 {
    match v {
        ColorRange::Unknown => 0,
        ColorRange::Full => 1,
        ColorRange::Limited => 2,
    }
}
fn gst_range_from_num(v: u8) -> Option<ColorRange> {
    Some(match v {
        0 => ColorRange::Unknown,
        1 => ColorRange::Full,
        2 => ColorRange::Limited,
        _ => return None,
    })
}
#[cfg(feature = "alloc")]
fn gst_matrix_num(v: MatrixCoefficients) -> u8 {
    match v {
        MatrixCoefficients::Unknown => 0,
        MatrixCoefficients::Identity => 1,
        MatrixCoefficients::Bt709 => 3,
        MatrixCoefficients::Bt601 => 4,
        MatrixCoefficients::Bt2020Ncl => 6,
    }
}
fn gst_matrix_from_num(v: u8) -> Option<MatrixCoefficients> {
    Some(match v {
        0 => MatrixCoefficients::Unknown,
        1 => MatrixCoefficients::Identity,
        3 => MatrixCoefficients::Bt709,
        4 => MatrixCoefficients::Bt601,
        6 => MatrixCoefficients::Bt2020Ncl,
        _ => return None,
    })
}
#[cfg(feature = "alloc")]
fn gst_transfer_num(v: TransferCharacteristics) -> u8 {
    match v {
        TransferCharacteristics::Unknown => 0,
        TransferCharacteristics::Bt709 => 5,
        TransferCharacteristics::Srgb => 7,
        // BT2020_12; the 10-bit sibling (13) parses to the same variant.
        TransferCharacteristics::Bt2020 => 11,
        TransferCharacteristics::Pq => 14,
        TransferCharacteristics::Hlg => 15,
        TransferCharacteristics::Bt601 => 16,
    }
}
fn gst_transfer_from_num(v: u8) -> Option<TransferCharacteristics> {
    Some(match v {
        0 => TransferCharacteristics::Unknown,
        5 => TransferCharacteristics::Bt709,
        7 => TransferCharacteristics::Srgb,
        11 | 13 => TransferCharacteristics::Bt2020,
        14 => TransferCharacteristics::Pq,
        15 => TransferCharacteristics::Hlg,
        16 => TransferCharacteristics::Bt601,
        _ => return None,
    })
}
#[cfg(feature = "alloc")]
fn gst_primaries_num(v: ColorPrimaries) -> u8 {
    match v {
        ColorPrimaries::Unknown => 0,
        ColorPrimaries::Bt709 => 1,
        ColorPrimaries::Bt470bg => 3,
        ColorPrimaries::Smpte170m => 4,
        ColorPrimaries::Bt2020 => 7,
    }
}
fn gst_primaries_from_num(v: u8) -> Option<ColorPrimaries> {
    Some(match v {
        0 => ColorPrimaries::Unknown,
        1 => ColorPrimaries::Bt709,
        3 => ColorPrimaries::Bt470bg,
        4 => ColorPrimaries::Smpte170m,
        7 => ColorPrimaries::Bt2020,
        _ => return None,
    })
}

/// Every raw (uncompressed) PCM sample format, with the `format=` gst spells it.
///
/// One list, so a format g2g prints into a caps description is one it can parse
/// back: a `format=` missing from the parser makes the whole description
/// unreadable, which reaches the caller as a caps mismatch far from here.
pub const PCM_FORMATS: [(AudioFormat, &str); 5] = [
    (AudioFormat::PcmS16Le, "S16LE"),
    (AudioFormat::PcmF32Le, "F32LE"),
    (AudioFormat::PcmS24Le, "S24LE"),
    (AudioFormat::PcmS32Le, "S32LE"),
    (AudioFormat::PcmU8, "U8"),
];

/// Just the formats from [`PCM_FORMATS`], for a caps set covering all of them.
pub fn pcm_formats() -> [AudioFormat; 5] {
    PCM_FORMATS.map(|(format, _)| format)
}

/// The gst `format=` name of a raw PCM format, `None` for an encoded one.
pub fn pcm_gst_format(f: AudioFormat) -> Option<&'static str> {
    PCM_FORMATS
        .iter()
        .find_map(|(format, name)| (*format == f).then_some(*name))
}

/// The raw PCM format a gst `format=` names, case-insensitively.
pub fn pcm_from_gst_format(name: &str) -> Option<AudioFormat> {
    PCM_FORMATS
        .iter()
        .find_map(|(format, gst)| gst.eq_ignore_ascii_case(name).then_some(*format))
}

/// Raw (uncompressed) PCM formats, the only ones the "any rate" wildcard (M187)
/// and the resampler apply to.
fn is_pcm(f: AudioFormat) -> bool {
    pcm_gst_format(f).is_some()
}

/// Intersect two [`Caps::Audio`] sample rates, where [`ANY_SAMPLE_RATE`] (0) is
/// a wildcard (M187): `any ∩ x = x`, equal rates agree, distinct concrete rates
/// are disjoint (`None`).
pub(crate) fn intersect_sample_rate(a: u32, b: u32) -> Option<u32> {
    match (a, b) {
        (ANY_SAMPLE_RATE, x) | (x, ANY_SAMPLE_RATE) => Some(x),
        (x, y) if x == y => Some(x),
        _ => None,
    }
}

/// Intersect two [`Caps::Audio`] channel counts, where [`ANY_CHANNELS`] (0) is a
/// wildcard: `any ∩ x = x`, equal counts agree, distinct concrete counts are
/// disjoint (`None`). Unlike [`intersect_sample_rate`] this applies to compressed
/// audio too, so a decoder's concrete output channels coupling back onto a
/// demuxer's unknown `0` input intersects rather than emptying the link.
pub(crate) fn intersect_channels(a: u8, b: u8) -> Option<u8> {
    match (a, b) {
        (ANY_CHANNELS, x) | (x, ANY_CHANNELS) => Some(x),
        (x, y) if x == y => Some(x),
        _ => None,
    }
}

/// Which caps fields a transform passes through unchanged (output field ==
/// input field). Read off a
/// `CapsTransform` declaration (the
/// fields every output shape derives with `Identity`), or probed from a
/// `DerivedOutput` closure. The solver uses them to couple input and
/// output *field by field* in both directions, so a downstream pin on a
/// passthrough field narrows the corresponding input field (`Range ∩ Fixed =
/// Fixed`) instead of only dropping whole alternatives.
///
/// `format` covers the variant's scalar media identity:
/// [`Caps::RawVideo`]'s `format`, [`Caps::CompressedVideo`]'s `codec`, and
/// [`Caps::Audio`]'s `format`. The geometry / rate / channel flags apply to the
/// matching field where the variant has one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PassthroughFields {
    pub format: bool,
    pub width: bool,
    pub height: bool,
    pub framerate: bool,
    pub channels: bool,
    pub sample_rate: bool,
}

impl PassthroughFields {
    /// No field coupled (everything retargeted). Build with the `with_*`
    /// const setters: `PassthroughFields::NONE.with_format().with_framerate()`.
    pub const NONE: Self = Self {
        format: false,
        width: false,
        height: false,
        framerate: false,
        channels: false,
        sample_rate: false,
    };

    pub const fn with_format(mut self) -> Self {
        self.format = true;
        self
    }
    pub const fn with_width(mut self) -> Self {
        self.width = true;
        self
    }
    pub const fn with_height(mut self) -> Self {
        self.height = true;
        self
    }
    pub const fn with_framerate(mut self) -> Self {
        self.framerate = true;
        self
    }
    pub const fn with_channels(mut self) -> Self {
        self.channels = true;
        self
    }
    pub const fn with_sample_rate(mut self) -> Self {
        self.sample_rate = true;
        self
    }
}

/// Overlap two inclusive `[min, max]` bounds, returning `None` when disjoint.
/// Shared by [`Dim::intersect`] and [`Rate::intersect`].
fn intersect_range((amin, amax): (u32, u32), (bmin, bmax): (u32, u32)) -> Option<(u32, u32)> {
    let lo = amin.max(bmin);
    let hi = amax.min(bmax);
    (lo <= hi).then_some((lo, hi))
}

/// A preference-ordered set of acceptable `Caps` descriptions.
///
/// `Caps` itself remains the *fixed* description used at runtime
/// (`DataFrame.caps`, `configure_*`). `CapsSet` is the negotiation-time
/// vocabulary: it carries alternatives and preference, neither of which
/// fits in a single `Caps`. See DESIGN.md §4.13.1.
///
/// The first alternative is highest preference; later ones are
/// fallbacks the element will accept if no peer agrees on the first.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, PartialEq)]
pub struct CapsSet {
    alternatives: Vec<Caps>,
}

#[cfg(feature = "alloc")]
impl CapsSet {
    /// Build from a single concrete description (equivalent to today's
    /// `Caps` for static call sites that don't express alternatives).
    pub fn one(caps: Caps) -> Self {
        Self {
            alternatives: alloc::vec![caps],
        }
    }

    /// Build directly from an ordered list of alternatives. The first
    /// element is highest preference. Empty input is allowed and yields
    /// the empty set (no agreement possible with any peer).
    pub fn from_alternatives(alternatives: Vec<Caps>) -> Self {
        Self { alternatives }
    }

    /// Return the ordered alternatives.
    pub fn alternatives(&self) -> &[Caps] {
        &self.alternatives
    }

    /// True when no alternatives remain. An empty `CapsSet` on a link
    /// means the two peers' constraints do not intersect.
    pub fn is_empty(&self) -> bool {
        self.alternatives.is_empty()
    }

    /// Intersection: the caps both sets agree on, preserving `self`'s
    /// outer preference order, then `other`'s within each row.
    /// Empty result = no assignment exists for a link between elements
    /// with these two sets.
    pub fn intersect(&self, other: &Self) -> Self {
        let mut out = Vec::new();
        for a in &self.alternatives {
            for b in &other.alternatives {
                if let Ok(c) = a.intersect(b) {
                    if !out.contains(&c) {
                        out.push(c);
                    }
                }
            }
        }
        Self { alternatives: out }
    }

    /// Union: every alternative in `self` followed by every alternative
    /// in `other` not already present. Preserves `self`'s preference
    /// order and dedupes structurally-equal entries. Used by the
    /// `Mapping` solver path to combine the surviving (input, output)
    /// pair sides.
    pub fn union(&self, other: &Self) -> Self {
        let mut out = self.alternatives.clone();
        for c in &other.alternatives {
            if !out.contains(c) {
                out.push(c.clone());
            }
        }
        Self { alternatives: out }
    }

    /// Fixate the highest-preference alternative that can collapse to a
    /// single concrete `Caps`. Returns `None` if the set is empty or
    /// every alternative still has `Any` fields after attempting
    /// fixation.
    pub fn fixate(&self) -> Option<Caps> {
        self.alternatives.iter().find_map(|c| c.fixate().ok())
    }

    /// True if any alternative is compatible with `caps` (a non-empty
    /// intersection exists). The ACCEPT_CAPS predicate (DESIGN.md §4.13.1):
    /// "would a link carrying `caps` satisfy this set?" — a pure check,
    /// no negotiation.
    pub fn accepts(&self, caps: &Caps) -> bool {
        self.alternatives.iter().any(|a| a.intersect(caps).is_ok())
    }
}

/// Compressed video codec carried in a [`Caps::CompressedVideo`] link.
/// Split out of the old `VideoFormat` enum so a decoder's "I accept
/// codec, I emit raw" boundary is type-level rather than a runtime
/// format compare. M17 split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VideoCodec {
    H264,
    H265,
    Av1,
    Vp8,
    Vp9,
    /// Motion JPEG: each frame an independent baseline JPEG. The near-universal
    /// fallback output of cheap UVC webcams, decoded by `MjpegDec`.
    Mjpeg,
    /// PNG (ISO/IEC 15948): a lossless still image, one frame per buffer, decoded
    /// by `PngDec` and produced by `PngEnc`. A still-image codec sits in
    /// `CompressedVideo` for the same reason MJPEG does: the pipeline shape is
    /// one compressed access unit in, one raw frame out.
    Png,
    /// WebP (RIFF `WEBP`, VP8 lossy or VP8L lossless bitstream): a still image,
    /// one frame per buffer, decoded by `WebPDec`. Animated WebP is not handled;
    /// the decoder takes the first frame only.
    WebP,
    /// MPEG-4 Part 2 (Visual, ISO/IEC 14496-2): the DivX / Xvid family. A legacy
    /// codec with no hardware decode path on modern GPUs, decoded in software via
    /// `FfmpegVideoDec`. Carried in MP4 as an `mp4v` sample entry (esds
    /// objectTypeIndication `0x20`) and in MPEG-TS as stream_type `0x10`.
    Mpeg4Part2,
    /// MPEG-1 Video (ISO/IEC 11172-2) and MPEG-2 Video (ISO/IEC 13818-2, ITU-T
    /// H.262) as one codec: MPEG-2 is a strict superset of MPEG-1, and the one
    /// libavcodec `MPEG2VIDEO` decoder plays both, so a VCD `.mpg` and a DVD
    /// `.vob` negotiate the same link. The DVD / broadcast video codec, carried
    /// in an MPEG program stream under stream_id 0xE0..=0xEF and in MPEG-TS
    /// under stream_type 0x01 (MPEG-1) / 0x02 (MPEG-2). Decoded in software via
    /// `FfmpegVideoDec`.
    Mpeg2,
    /// Sorenson Spark (Sorenson H.263), the original Flash video codec, carried
    /// as FLV video codec id 2 (GStreamer `video/x-flash-video`, libavcodec
    /// `flv1`). An H.263 derivative with Flash's own picture header, so it is a
    /// distinct codec from ITU H.263. Decoded in software via `FfmpegVideoDec`.
    SorensonH263,
    /// On2 VP6 in its Flash variant: FLV video codec id 4, or id 5 when a second
    /// (alpha) plane rides in the same packet. `alpha` picks between them, since
    /// libavcodec decodes them with different decoders (`vp6f` / `vp6a`) and a
    /// consumer must know whether the stream carries transparency (GStreamer
    /// `video/x-vp6-flash` / `video/x-vp6-alpha`). The container's one-byte
    /// dimension adjustment travels as the codec-config side channel.
    Vp6 {
        alpha: bool,
    },
    /// JPEG XS (ISO/IEC 21122): a low-latency, visually lossless intra-frame
    /// mezzanine codec, each frame an independent codestream. The compressed
    /// essence of SMPTE ST 2110-22 (carried over RTP per RFC 9134), so a
    /// professional-AV plant can move near-uncompressed video at a fraction of
    /// -20's bandwidth while keeping sub-frame latency. Encoded / decoded via
    /// `FfmpegJpegXsEnc` / `FfmpegJpegXsDec` (libavcodec `Id::JPEGXS`).
    JpegXs,
    /// VC-1 (SMPTE 421M), the standardised Windows Media Video 9 bitstream. The
    /// Blu-ray and Silverlight codec, carried in ASF and in MPEG-TS under
    /// stream_type 0xEA. Advanced profile is a start-code byte stream that
    /// `Vc1Parse` frames; simple and main profile carry no start codes and take
    /// their sequence layer from the container's codec-configuration block.
    Vc1,
    /// Netpbm still image (PBM / PGM / PPM, magic `P1`..`P6`): one frame per
    /// buffer, decoded by `PnmDec` and produced by `PnmEnc`. Media type
    /// `image/x-portable-anymap` and the bitmap / graymap / pixmap variants.
    Pnm,
}

/// Wire format of a [`Caps::ByteStream`] link, so a demuxer accepts only the
/// container it parses (an MPEG-TS demuxer rejects an arbitrary byte stream
/// structurally, like the codec/raw split does for video).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ByteStreamEncoding {
    /// MPEG-2 Transport Stream (ISO/IEC 13818-1): 188-byte packets, PAT/PMT,
    /// PES. The broadcast / SRT / HLS-segment carrier.
    MpegTs,
    /// Matroska / WebM (EBML): nested variable-length elements; Tracks describe
    /// the elementary streams and Clusters carry the SimpleBlock frames. The
    /// common file container, WebM being the browser-delivery subset (VP8 / VP9 /
    /// AV1 video + Opus / Vorbis audio).
    Matroska,
    /// Ogg (RFC 3533): "OggS" pages with a segment-table lacing that frames the
    /// packets of a logical bitstream. The canonical Opus / Vorbis carrier.
    Ogg,
    /// FLV (Flash Video): an "FLV" header then `PreviousTagSize` / tag pairs, each
    /// tag a codec-tagged audio / video / script payload. The RTMP carrier.
    Flv,
    /// ISO Base Media File Format / fragmented MP4 (CMAF): `ftyp`/`moov` init then
    /// `moof`+`mdat` fragments. The modern HLS/DASH segment container, demuxed by
    /// `fmp4demux` incrementally (a live stream, no end).
    IsoBmff,
    /// Progressive / whole-file MP4 / QuickTime (M479): `ftyp` + `moov` (sample
    /// tables) + `mdat`, in either order. A seekable file rather than a live
    /// stream, so it is demuxed by `mp4demux` after the whole file is buffered (the
    /// `moov` may sit at the end, and `stco` chunk offsets are absolute). The local
    /// `.mp4` / `.mov` case, distinct from the streaming `IsoBmff` above so the
    /// auto-plugger picks the whole-file demuxer for files and the incremental one
    /// for HLS / DASH.
    Mp4,
    /// IVF: a 32-byte `DKIF` header (FourCC codec + geometry + timebase) then a
    /// 12-byte size+timestamp header before each frame. The simple raw container
    /// libvpx / libaom conformance vectors ship in (VP8 / VP9 / AV1).
    Ivf,
    /// MPEG-1 / MPEG-2 Program Stream (ISO/IEC 13818-1): packs, each a
    /// `00 00 01 BA` header plus the PES packets that follow it, streams
    /// identified by PES `stream_id` rather than a PID and a PMT. The `.mpg` /
    /// `.vob` file carrier (VCD, SVCD, DVD), demuxed by `mpegpsdemux`.
    MpegPs,
    /// RIFF/WAVE (`.wav`): a `RIFF` chunk holding a `fmt ` descriptor and a
    /// `data` chunk of interleaved PCM. The uncompressed file container, and the
    /// one an audio tool reads without a demuxer.
    Wav,
    /// AIFF / AIFC (`.aiff` / `.aif` / `.aifc`): an EA IFF 85 `FORM` of type
    /// `AIFF` or `AIFC` holding a `COMM` descriptor and an `SSND` sample chunk.
    /// The Mac / interchange sibling of WAVE, parsed by `aiffparse` and written
    /// by `aiffmux`. Multi-byte PCM is big-endian on the wire and swapped to the
    /// little-endian `AudioFormat` the rest of the graph uses.
    Aiff,
    /// Sun / NeXT AU (`.au` / `.snd`): a 24-byte big-endian `.snd` header then
    /// the samples. The Unix sibling of WAVE, parsed by `auparse` and written by
    /// `avmux_au`. Multi-byte PCM is big-endian on the wire, same swap as AIFF.
    Au,
    /// AVI (`.avi`): the RIFF sibling of WAVE, a `hdrl` list describing one
    /// stream per `strl` and a `movi` list of the interleaved data chunks, with
    /// an `idx1` at the end. Demuxed by `avidemux`, written by `avimux`.
    Avi,
    /// YUV4MPEG2 (`.y4m`): a `YUV4MPEG2 W.. H.. F..` text header then a `FRAME`
    /// line before each frame's planes. The uncompressed video counterpart of
    /// WAV, and what encoders and quality tools exchange raw frames in.
    Y4m,
    /// MIME multipart (`multipart/x-mixed-replace`, RFC 2046): a `--boundary`
    /// line, MIME headers, and a body, repeated. What an IP camera or an
    /// `mjpg-streamer`-style server pushes MJPEG over HTTP with, demuxed by
    /// `multipartdemux` and written by `multipartmux`.
    Multipart,
    /// No container and no framing: a headerless dump whose shape is declared
    /// out of band, the `.yuv` / `.pcm` file `rawvideoparse` / `rawaudioparse`
    /// frame from their properties. Content sniffing never answers `Raw`, since
    /// any byte sequence matches it.
    Raw,
    /// One complete RTP packet per frame.
    Rtp,
    /// One complete SRTP packet per frame.
    Srtp,
    /// One complete RTCP packet per frame.
    Rtcp,
    /// One complete SRTCP packet per frame.
    Srtcp,
    /// One datagram per frame carrying DTLS records multiplexed with SRTP
    /// and SRTCP, split on the first byte per RFC 7983.
    Dtls,
}

/// Format of a [`Caps::Text`] stream. Generalizes "subtitles": a `Text` link
/// carries any timed-or-untimed text payload (a subtitle cue, a caption, a
/// transcription, an OCR result, an overlay string), the format naming the
/// on-the-wire syntax. "Subtitle" is not a separate media kind here, just timed
/// `Text` frames (timing rides on [`FrameTiming`](crate::frame::FrameTiming)),
/// so one variant serves overlay, captioning, and analytics text alike. A parser
/// converts a structured format (`Srt` / `WebVtt` / `Ssa` / `Ttml`) to the plain
/// `Utf8` cues a renderer or consumer wants, like a codec decode for text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TextFormat {
    /// Plain UTF-8 text, no markup. The decoded/common-denominator form a
    /// subtitle parser emits and an overlay or sink consumes.
    Utf8,
    /// UTF-8 with Pango inline markup (`<b>`, `<i>`, `<span>`...), the styled
    /// text an overlay renderer draws directly (GStreamer `pango-markup`).
    PangoMarkup,
    /// SubRip (`.srt`): blank-line-separated cues, each an index, a
    /// `HH:MM:SS,mmm --> HH:MM:SS,mmm` time range, then the text lines.
    Srt,
    /// WebVTT (`.vtt`, RFC 8538): a `WEBVTT` header then `start --> end` cues with
    /// `.`-millisecond timestamps; the HTML5 / HLS subtitle format.
    WebVtt,
    /// SubStation Alpha / Advanced SSA (`.ssa` / `.ass`): a sectioned INI-like
    /// script with styled `Dialogue:` events. The fansub / Matroska text format.
    Ssa,
    /// Timed Text Markup Language (W3C TTML / SMPTE-TT / EBU-TT, also `DFXP`): an
    /// XML timed-text document. The broadcast / DASH caption format.
    Ttml,
    /// EBU teletext (ETSI EN 300 706) as a DVB private PES carries it (EN 300
    /// 472): a data_identifier byte then fixed-size data units, each one
    /// teletext line with a framing code, a hamming 8/4 magazine / packet
    /// address, and 40 odd-parity bytes. Not readable text yet: a teletext
    /// decoder assembles the addressed page's rows and emits [`Self::Utf8`]
    /// cues, the way a subtitle parser converts [`Self::Srt`].
    Teletext,
}

/// Which closed-caption carriage a [`Caps::ClosedCaption`] frame holds. It names
/// the service family (608 line-21 fields or 708 DTVCC packets) and the byte
/// layout the payload is in, since the same captions travel in four of them: a
/// converter reads one and writes another, and a muxer knows which sample entry to
/// write. Every layout carries the same underlying `(cc_type, cc_data_1,
/// cc_data_2)` triples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ClosedCaptionFormat {
    /// CEA-608 line-21 captions as packed ATSC `cc_data` triples, `cc_type` 0/1
    /// (the MP4 `c608` sample entry, whose samples hold `cdat` / `cdt2` byte-pair
    /// atoms).
    Cea608,
    /// CEA-608 line-21 captions as bare byte pairs, one line-21 field's worth per
    /// frame with no `cc_type` byte (GStreamer's `format=raw`). Which field they
    /// came from travels beside the stream, as the reader's `field` property.
    Cea608Raw,
    /// CEA-608 line-21 captions as SMPTE ST 334-1 Annex A triplets: a
    /// field / line-offset byte then the two data bytes (GStreamer's
    /// `format=s334-1a`), the form an ancillary-data packet carries.
    Cea608S334,
    /// CEA-708 DTVCC captions as packed ATSC `cc_data` triples, `cc_type` 2/3.
    Cea708,
    /// CEA-708 DTVCC captions in a SMPTE ST 334-2 caption distribution packet
    /// (GStreamer's `format=cdp`), the payload of a DID 0x61 ancillary packet and
    /// of the MP4 `c708` sample entry's `ccdp` atom.
    Cea708Cdp,
}

/// Which bitmap-subtitle coding a [`Caps::SubPicture`] stream carries. Each
/// coding has its own cue framing, palette carriage, and bitmap compression, so
/// a subpicture decoder declares the one it reads rather than sniffing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SubPictureFormat {
    /// DVD subpictures (VobSub): one MPEG-PS subpicture unit (SPU) per cue, a
    /// 2-bits-per-pixel run-length bitmap in two interlaced fields plus a control
    /// sequence carrying the display rectangle, the four palette indices and
    /// alphas, and the show / hide times. The 16-entry RGB palette and the
    /// display geometry ride out of band, in the `.idx` text a Matroska track
    /// carries as its `CodecPrivate`.
    VobSub,
    /// DVB subtitles (ETSI EN 300 743): a segment stream (page / region / CLUT /
    /// object / display definition) rather than one packet per cue, with 2-, 4-
    /// and 8-bit run-length coded objects placed into regions and regions placed
    /// on the display. The palette is in band (CLUT definition segments); the
    /// composition and ancillary page ids ride out of band, in the PMT
    /// `subtitling_descriptor` or a Matroska track's `CodecPrivate`.
    DvbSub,
    /// Blu-ray Presentation Graphic Stream subtitles (PGS / HDMV): like DVB a
    /// segment stream (presentation composition / window / palette / object)
    /// rather than one packet per cue, with 8-bit run-length coded objects
    /// placed straight on the video. Everything rides in band, the palette and
    /// the video geometry included, so there is no out-of-band configuration.
    Pgs,
}

/// Raw pixel layout carried in a [`Caps::RawVideo`] link. Split out of
/// the old `VideoFormat` enum so a raw sink (waylandsink/kmssink)
/// rejects compressed input structurally rather than via runtime check.
/// M17 split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RawVideoFormat {
    Nv12,
    I420,
    Rgba8,
    Bgra8,
    /// Packed 8-bit RGB, three bytes per pixel and no alpha (the GStreamer `RGB`
    /// format). The layout CPU vision and ML code reads, so a hosted inference
    /// element takes frames without an alpha channel it would only discard.
    Rgb8,
    /// Packed YUV 4:2:2, byte order Y0 U Y1 V (the V4L2 `YUYV` / `YUY2`
    /// fourcc). Two bytes per pixel; the near-universal UVC webcam output.
    /// Packed (not planar), so it needs unpacking before planar consumers.
    Yuyv,
    // Fully-planar YUV (three separate Y / U / V planes), the layout the AV1 /
    // HEVC / VP9 decoders produce. The `p10` / `p12` suffix is 10- / 12-bit
    // depth, each sample stored little-endian in the low bits of a 2-byte word
    // (the GStreamer `*_10LE` / `*_12LE` formats); the bare name is 8-bit. The
    // family covers the three chroma subsamplings: I420 = 4:2:0, I422 = 4:2:2,
    // I444 = 4:4:4. See [`RawVideoFormat::chroma_shift`] / [`bit_depth`].
    /// Planar 4:2:0, 10-bit (LE).
    I420p10,
    /// Planar 4:2:0, 12-bit (LE).
    I420p12,
    /// Planar 4:2:2 (full-height, half-width chroma), 8-bit.
    I422,
    /// Planar 4:2:2, 10-bit (LE).
    I422p10,
    /// Planar 4:2:2, 12-bit (LE).
    I422p12,
    /// Planar 4:4:4 (full-resolution chroma), 8-bit.
    I444,
    /// Planar 4:4:4, 10-bit (LE).
    I444p10,
    /// Planar 4:4:4, 12-bit (LE).
    I444p12,
    /// Semi-planar 4:2:0, 10-bit: NV12's layout (Y plane then interleaved UV) with
    /// 16-bit little-endian samples carrying the value in the *top* 10 bits (the
    /// GStreamer `P010_10LE` format, NVDEC's `P016` surface). The hardware 10-bit
    /// decode / encode surface format.
    P010,
}

impl RawVideoFormat {
    /// Bits per sample of a YUV format: 8, 10, or 12. The 10- and 12-bit formats
    /// store each sample little-endian in a 2-byte word (P010 in the word's top
    /// bits, the planar family in the low bits). The RGBA / packed formats
    /// report 8.
    pub const fn bit_depth(self) -> u8 {
        match self {
            RawVideoFormat::I420p10
            | RawVideoFormat::I422p10
            | RawVideoFormat::I444p10
            | RawVideoFormat::P010 => 10,
            RawVideoFormat::I420p12 | RawVideoFormat::I422p12 | RawVideoFormat::I444p12 => 12,
            _ => 8,
        }
    }

    /// Bytes per sample: 2 for the 10- / 12-bit planar formats (LE `u16`), else 1.
    pub const fn bytes_per_sample(self) -> usize {
        if self.bit_depth() > 8 {
            2
        } else {
            1
        }
    }

    /// Chroma subsampling of a fully-planar YUV format as the (horizontal,
    /// vertical) right-shift from luma to chroma dimensions: 4:2:0 = `(1, 1)`,
    /// 4:2:2 = `(1, 0)`, 4:4:4 = `(0, 0)`. `None` for the non-planar formats
    /// (NV12 is semi-planar; RGBA / YUYV are packed), which carry their own
    /// layout.
    pub const fn chroma_shift(self) -> Option<(u32, u32)> {
        match self {
            RawVideoFormat::I420 | RawVideoFormat::I420p10 | RawVideoFormat::I420p12 => {
                Some((1, 1))
            }
            RawVideoFormat::I422 | RawVideoFormat::I422p10 | RawVideoFormat::I422p12 => {
                Some((1, 0))
            }
            RawVideoFormat::I444 | RawVideoFormat::I444p10 | RawVideoFormat::I444p12 => {
                Some((0, 0))
            }
            _ => None,
        }
    }

    /// True for the fully-planar I420 / I422 / I444 family (three Y, U, V planes
    /// of [`Self::bytes_per_sample`]-byte samples). Excludes the semi-planar NV12
    /// and the packed formats.
    pub const fn is_planar_yuv(self) -> bool {
        self.chroma_shift().is_some()
    }

    /// Row stride in bytes of the luma / packed plane at `width`: 4 bytes per
    /// pixel for packed RGBA / BGRA, 3 for packed RGB, 2 for packed YUYV, 1 for
    /// 8-bit NV12 / I420 luma. `None` for a format with no single-stride byte
    /// layout.
    pub fn row_stride(self, width: u32) -> Option<u32> {
        match self {
            RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => width.checked_mul(4),
            RawVideoFormat::Rgb8 => width.checked_mul(3),
            RawVideoFormat::Yuyv => width.checked_mul(2),
            RawVideoFormat::Nv12 | RawVideoFormat::I420 => Some(width),
            _ => None,
        }
    }

    /// Bytes one frame occupies when every row is `stride` bytes, the layout a
    /// dma-buf or a V4L2 capture buffer uses. Packed RGBA / BGRA / RGB / YUYV
    /// are a single plane (`stride * height`); 8-bit NV12 / I420 add the half-height
    /// chroma region, which is the same total whether the chroma is interleaved
    /// (NV12) or split (I420) as long as the luma stride is used. `None` for a
    /// format with no single-stride byte layout.
    pub fn frame_bytes(self, stride: u64, height: u64) -> Option<u64> {
        let luma = stride.checked_mul(height)?;
        match self {
            RawVideoFormat::Rgba8
            | RawVideoFormat::Bgra8
            | RawVideoFormat::Rgb8
            | RawVideoFormat::Yuyv => Some(luma),
            RawVideoFormat::Nv12 | RawVideoFormat::I420 => {
                luma.checked_add(stride.checked_mul(height.div_ceil(2))?)
            }
            _ => None,
        }
    }

    /// How many separate planes the format stores: 1 packed (RGBA / BGRA / RGB /
    /// YUYV), 2 semi-planar (NV12 / P010, luma then interleaved chroma), 3 fully
    /// planar (the I420 / I422 / I444 family). Matched exhaustively so a new
    /// format has to state its own layout rather than inherit a wrong one.
    pub const fn plane_count(self) -> usize {
        match self {
            RawVideoFormat::Rgba8
            | RawVideoFormat::Bgra8
            | RawVideoFormat::Rgb8
            | RawVideoFormat::Yuyv => 1,
            RawVideoFormat::Nv12 | RawVideoFormat::P010 => 2,
            RawVideoFormat::I420
            | RawVideoFormat::I420p10
            | RawVideoFormat::I420p12
            | RawVideoFormat::I422
            | RawVideoFormat::I422p10
            | RawVideoFormat::I422p12
            | RawVideoFormat::I444
            | RawVideoFormat::I444p10
            | RawVideoFormat::I444p12 => 3,
        }
    }

    /// Bytes one pixel occupies in a packed format's single plane: 4 for RGBA /
    /// BGRA, 3 for RGB, 2 for YUYV. `None` for the multi-plane formats, where one
    /// pixel's samples are spread across planes.
    pub const fn pixel_stride(self) -> Option<usize> {
        match self {
            RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => Some(4),
            RawVideoFormat::Rgb8 => Some(3),
            RawVideoFormat::Yuyv => Some(2),
            _ => None,
        }
    }

    /// Row stride in bytes of `plane` at `width`, with no row padding. `None`
    /// for a plane this format does not have, or on overflow.
    pub fn plane_stride(self, plane: usize, width: u32) -> Option<u32> {
        if plane >= self.plane_count() {
            return None;
        }
        if let Some(pixel) = self.pixel_stride() {
            return width.checked_mul(pixel as u32);
        }
        let sample = self.bytes_per_sample() as u32;
        if plane == 0 {
            return width.checked_mul(sample);
        }
        match self {
            // One interleaved U+V pair per 2x2 block, so the chroma row is as
            // wide as the luma row (rounded up on an odd width).
            RawVideoFormat::Nv12 | RawVideoFormat::P010 => {
                width.div_ceil(2).checked_mul(2)?.checked_mul(sample)
            }
            _ => {
                let (horizontal, _) = self.chroma_shift()?;
                width.div_ceil(1 << horizontal).checked_mul(sample)
            }
        }
    }

    /// Rows in `plane` at `height`. `None` for a plane this format does not have.
    pub fn plane_rows(self, plane: usize, height: u32) -> Option<u32> {
        if plane >= self.plane_count() {
            return None;
        }
        if plane == 0 {
            return Some(height);
        }
        match self {
            RawVideoFormat::Nv12 | RawVideoFormat::P010 => Some(height.div_ceil(2)),
            _ => {
                let (_, vertical) = self.chroma_shift()?;
                Some(height.div_ceil(1 << vertical))
            }
        }
    }

    /// Bytes `plane` occupies with no row padding.
    pub fn plane_bytes(self, plane: usize, width: u32, height: u32) -> Option<u64> {
        let stride = self.plane_stride(plane, width)? as u64;
        let rows = self.plane_rows(plane, height)? as u64;
        stride.checked_mul(rows)
    }

    /// Byte offset of `plane` from the start of an unpadded frame, its planes
    /// laid out back to back in index order.
    pub fn plane_offset(self, plane: usize, width: u32, height: u32) -> Option<u64> {
        if plane >= self.plane_count() {
            return None;
        }
        let mut offset = 0u64;
        for earlier in 0..plane {
            offset = offset.checked_add(self.plane_bytes(earlier, width, height)?)?;
        }
        Some(offset)
    }

    /// Bytes a whole frame occupies with no row padding, covering every format.
    /// [`Self::frame_bytes`] answers the narrower dma-buf question instead: the
    /// size given an externally chosen row stride, for the formats that can be
    /// exported as one tightly strided buffer.
    pub fn unpadded_frame_bytes(self, width: u32, height: u32) -> Option<u64> {
        let mut total = 0u64;
        for plane in 0..self.plane_count() {
            total = total.checked_add(self.plane_bytes(plane, width, height)?)?;
        }
        Some(total)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AudioFormat {
    Aac,
    Opus,
    /// MPEG-1/2 Audio Layer II (`mp2`), the standard broadcast/DVB audio codec
    /// (GStreamer `audio/mpeg,mpegversion=1,layer=2`). Encoded like `Aac`, so it
    /// keeps a nominal channels/rate rather than the PCM wildcards; MPEG-TS carries
    /// it under stream_type 0x03 (MPEG-1) / 0x04 (MPEG-2). Decoded via libavcodec.
    Mp2,
    /// Dolby Digital / ATSC A/52 (`ac3`), the standard broadcast/DVD audio codec
    /// (GStreamer `audio/x-ac3`). Self-syncing frames (`0x0B77` sync + frame-size
    /// code); MPEG-TS carries it under stream_type 0x81 (ATSC) or a private PES
    /// (0x06) with an AC-3 descriptor (DVB), Matroska as `A_AC3`. Decoded via
    /// libavcodec.
    Ac3,
    /// MPEG-1/2/2.5 Audio Layer III (`mp3`), GStreamer
    /// `audio/mpeg,mpegversion=1,layer=3`. Self-syncing frames like `Mp2` (the
    /// same 4-byte header, a Layer III frame length), the legacy FLV / RTMP audio
    /// codec. Decoded via libavcodec.
    Mp3,
    /// Speex (RFC 5574), GStreamer `audio/x-speex`: the pre-Opus low-bitrate
    /// speech codec, carried by FLV at a fixed 16 kHz mono. Container-framed (one
    /// packet per FLV tag / Ogg packet). Carriage only, g2g registers no Speex
    /// decoder, so an autoplugged decode chain fails to build rather than
    /// pretending.
    Speex,
    /// Free Lossless Audio Codec (`flac`), GStreamer `audio/x-flac`. Frame headers
    /// are self-describing but not cheaply self-syncing, so g2g relies on the
    /// container framing (one frame per Matroska block / Ogg packet); the STREAMINFO
    /// header rides in-band as a leading `fLaC` frame the decoder takes as extradata.
    /// Decoded via libavcodec.
    Flac,
    /// Vorbis (GStreamer `audio/x-vorbis`), the Ogg / WebM audio codec that
    /// preceded Opus. Encoded like `Aac` / `Opus` (nominal channels/rate, not
    /// the PCM wildcards); container-framed (one packet per Ogg packet /
    /// Matroska block), with the identification + setup headers riding in-band
    /// ahead of the audio. Decoded in pure Rust (symphonia).
    Vorbis,
    PcmS16Le,
    PcmF32Le,
    /// 24-bit signed integer PCM, little-endian, 3 bytes packed (GStreamer `S24LE`).
    /// The integer sibling of `PcmF32Le` for the ST 2110-30 / AES67 L24 wire: a
    /// professional 24-bit source rides L24 without a detour through float.
    PcmS24Le,
    /// 32-bit signed integer PCM, little-endian (GStreamer `S32LE`). The native
    /// container width of most modern DACs, so a 24-bit source reaches the
    /// device without the 3-byte packing.
    PcmS32Le,
    /// 8-bit unsigned integer PCM, one byte per sample, silence at 0x80
    /// (GStreamer `U8`). The legacy WAV / telephony sample width.
    PcmU8,
    /// G.711 mu-law companded audio, one byte per sample (GStreamer
    /// `audio/x-mulaw`, RTP payload type 0 / PCMU). Encoded, not raw PCM: like
    /// `Aac` / `Opus` it keeps a nominal rate/channels rather than the PCM
    /// wildcards. Codec in `g2g-mcu::g711` (M638).
    Mulaw,
    /// G.711 A-law companded audio, one byte per sample (GStreamer
    /// `audio/x-alaw`, RTP payload type 8 / PCMA). See [`AudioFormat::Mulaw`].
    Alaw,
    /// IMA ADPCM, 4 bits per sample in the WAV / DVI4 block layout
    /// (GStreamer `audio/x-adpcm` layout=dvi). Encoded like the G.711 pair;
    /// codec in `g2g-mcu::adpcm` (M639).
    ImaAdpcm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TensorDType {
    F16,
    F32,
    I8,
    U8,
}

impl TensorDType {
    /// Size in bytes of one element of this dtype. Used by [`crate::tensor`]
    /// to turn element strides into byte strides and size a materialization.
    pub const fn size(self) -> usize {
        match self {
            TensorDType::F16 => 2,
            TensorDType::F32 => 4,
            TensorDType::I8 | TensorDType::U8 => 1,
        }
    }
}

/// Logical shape of a tensor stream: up to [`MAX_TENSOR_RANK`] dimensions
/// stored inline, so the type is `Copy`, heap-free, and part of the no-alloc
/// MCU subset (M636; it was a `Vec` before, which is why `Caps::Tensor` used
/// to be gated behind `alloc`). Unused trailing slots are always zero, so the
/// derived equality over the whole array agrees with slice equality on
/// [`dims`](TensorShape::dims).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    dims: [u32; MAX_TENSOR_RANK],
    rank: u8,
}

/// Post-monomorphization rank guard for [`TensorShape::new`]: evaluating
/// `VALID` with a rank of 0 or above [`MAX_TENSOR_RANK`] is a compile error,
/// so the constructor carries no runtime branch (the no-heap archive stays
/// panic-free).
struct RankCheck<const N: usize>;

impl<const N: usize> RankCheck<N> {
    const VALID: usize = {
        assert!(
            N >= 1 && N <= MAX_TENSOR_RANK,
            "tensor rank must be 1..=MAX_TENSOR_RANK"
        );
        N
    };
}

impl TensorShape {
    /// Shape from a fixed-size array, e.g. `TensorShape::new([1, 3, 224, 224])`.
    /// The rank is checked at compile time (out of range fails the build), so
    /// this cannot panic.
    pub const fn new<const N: usize>(dims: [u32; N]) -> Self {
        let _ = RankCheck::<N>::VALID;
        let mut d = [0u32; MAX_TENSOR_RANK];
        let mut i = 0;
        while i < N {
            d[i] = dims[i];
            i += 1;
        }
        Self {
            dims: d,
            rank: N as u8,
        }
    }

    /// Fallible shape from a runtime slice (a model's reported dims, a
    /// wire-decoded shape): `None` when empty or longer than
    /// [`MAX_TENSOR_RANK`], so untrusted input fails cleanly instead of
    /// panicking.
    pub fn from_slice(dims: &[u32]) -> Option<Self> {
        if dims.is_empty() || dims.len() > MAX_TENSOR_RANK {
            return None;
        }
        let mut d = [0u32; MAX_TENSOR_RANK];
        d[..dims.len()].copy_from_slice(dims);
        Some(Self {
            dims: d,
            rank: dims.len() as u8,
        })
    }

    /// The dimensions as a slice; its length is the rank.
    pub fn dims(&self) -> &[u32] {
        // rank <= MAX_TENSOR_RANK by construction; the clamp is what lets the
        // optimizer discharge the slice bounds check, keeping the no-heap
        // archive panic-free.
        &self.dims[..(self.rank as usize).min(MAX_TENSOR_RANK)]
    }

    /// Mutable view of the dimensions, for in-place edits that keep the rank
    /// (e.g. a batcher rewriting the batch dim). The rank itself is fixed at
    /// construction.
    pub fn dims_mut(&mut self) -> &mut [u32] {
        // Clamped like `dims` (see there).
        &mut self.dims[..(self.rank as usize).min(MAX_TENSOR_RANK)]
    }

    /// Element count (product of the dims), saturating on overflow so a
    /// bogus shape sizes to `usize::MAX` rather than wrapping or panicking.
    pub fn elements(&self) -> usize {
        self.dims()
            .iter()
            .fold(1usize, |acc, &d| acc.saturating_mul(d as usize))
    }
}

impl core::fmt::Debug for TensorShape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Print like the old `TensorShape(Vec)` tuple struct did.
        f.debug_tuple("TensorShape").field(&self.dims()).finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TensorLayout {
    Nchw,
    Nhwc,
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    fn video(width: Dim, height: Dim, framerate: Rate) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width,
            height,
            framerate,
            interlace: crate::Interlace::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
        }
    }

    #[test]
    fn closed_caption_caps_intersect_by_carriage() {
        let cea608 = Caps::ClosedCaption {
            format: ClosedCaptionFormat::Cea608,
        };
        let cea708 = Caps::ClosedCaption {
            format: ClosedCaptionFormat::Cea708,
        };
        assert_eq!(cea608.intersect(&cea608), Ok(cea608.clone()));
        assert!(cea608.intersect(&cea708).is_err());
        // A caption stream is not text, and carries no geometry to fixate.
        assert!(cea608
            .intersect(&Caps::Text {
                format: TextFormat::Utf8
            })
            .is_err());
        assert_eq!(cea608.fixate(), Ok(cea608.clone()));
        assert!(cea608.is_fixed());
        assert!(cea608.dims().is_none());
        assert!(!cea608.is_raw_media());
        assert_eq!(
            cea708.to_gst_string(),
            "closedcaption/x-cea-708,format=cc_data"
        );
    }

    #[test]
    fn dim_intersect_any_is_identity() {
        assert_eq!(Dim::Any.intersect(&Dim::Fixed(720)), Some(Dim::Fixed(720)));
        assert_eq!(Dim::Fixed(720).intersect(&Dim::Any), Some(Dim::Fixed(720)));
        assert_eq!(Dim::Any.intersect(&Dim::Any), Some(Dim::Any));
    }

    #[test]
    fn dim_intersect_fixed_pairs() {
        assert_eq!(
            Dim::Fixed(64).intersect(&Dim::Fixed(64)),
            Some(Dim::Fixed(64))
        );
        assert_eq!(Dim::Fixed(64).intersect(&Dim::Fixed(65)), None);
    }

    #[test]
    fn dim_intersect_fixed_against_range() {
        let range = Dim::Range { min: 100, max: 200 };
        assert_eq!(Dim::Fixed(150).intersect(&range), Some(Dim::Fixed(150)));
        assert_eq!(Dim::Fixed(100).intersect(&range), Some(Dim::Fixed(100))); // inclusive lo
        assert_eq!(Dim::Fixed(200).intersect(&range), Some(Dim::Fixed(200))); // inclusive hi
        assert_eq!(Dim::Fixed(99).intersect(&range), None);
        assert_eq!(Dim::Fixed(201).intersect(&range), None);
    }

    #[test]
    fn dim_intersect_overlapping_ranges_tighten() {
        let a = Dim::Range { min: 100, max: 300 };
        let b = Dim::Range { min: 200, max: 400 };
        assert_eq!(a.intersect(&b), Some(Dim::Range { min: 200, max: 300 }));
    }

    #[test]
    fn dim_intersect_ranges_meeting_at_a_point_collapse_to_fixed() {
        let a = Dim::Range { min: 100, max: 200 };
        let b = Dim::Range { min: 200, max: 300 };
        assert_eq!(a.intersect(&b), Some(Dim::Fixed(200)));
    }

    #[test]
    fn dim_intersect_disjoint_ranges_none() {
        let a = Dim::Range { min: 100, max: 199 };
        let b = Dim::Range { min: 200, max: 300 };
        assert_eq!(a.intersect(&b), None);
    }

    #[test]
    fn rate_intersect_mirrors_dim() {
        let a = Rate::Range {
            min_q16: 15 << 16,
            max_q16: 60 << 16,
        };
        let b = Rate::Fixed(30 << 16);
        assert_eq!(a.intersect(&b), Some(Rate::Fixed(30 << 16)));
        assert_eq!(Rate::Any.intersect(&b), Some(Rate::Fixed(30 << 16)));
        // 10 fps falls below the [15, 60] range → no overlap.
        assert_eq!(Rate::Fixed(10 << 16).intersect(&a), None);
    }

    #[test]
    fn dim_fixate_picks_range_minimum() {
        assert_eq!(
            Dim::Range {
                min: 480,
                max: 1080
            }
            .fixate(),
            Some(Dim::Fixed(480))
        );
        assert_eq!(Dim::Fixed(720).fixate(), Some(Dim::Fixed(720)));
        assert_eq!(Dim::Any.fixate(), None);
    }

    #[test]
    fn fixate_agrees_with_intersect_on_inverted_ranges() {
        // An inverted range is the empty set: `intersect` reports it empty, so
        // `fixate` must not hand back a value (the min) that is outside it.
        let bad_dim = Dim::Range { min: 200, max: 100 };
        assert_eq!(
            bad_dim.intersect(&Dim::Any),
            None,
            "inverted range is empty"
        );
        assert_eq!(bad_dim.fixate(), None, "and so cannot fixate to its min");

        let bad_rate = Rate::Range {
            min_q16: 60 << 16,
            max_q16: 30 << 16,
        };
        assert_eq!(bad_rate.intersect(&Rate::Any), None);
        assert_eq!(bad_rate.fixate(), None);
    }

    #[test]
    fn caps_intersect_video_fields() {
        let a = video(
            Dim::Range {
                min: 640,
                max: 1920,
            },
            Dim::Any,
            Rate::Any,
        );
        let b = video(Dim::Fixed(1280), Dim::Fixed(720), Rate::Fixed(30 << 16));
        assert_eq!(
            a.intersect(&b).unwrap(),
            video(Dim::Fixed(1280), Dim::Fixed(720), Rate::Fixed(30 << 16))
        );
    }

    #[test]
    fn caps_intersect_rejects_format_mismatch() {
        let a = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
        };
        let b = video(Dim::Any, Dim::Any, Rate::Any); // Rgba8
        assert_eq!(a.intersect(&b), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_intersect_rejects_empty_field_overlap() {
        let a = video(Dim::Fixed(640), Dim::Any, Rate::Any);
        let b = video(Dim::Fixed(1280), Dim::Any, Rate::Any);
        assert_eq!(a.intersect(&b), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_intersect_rejects_variant_mismatch() {
        let v = video(Dim::Any, Dim::Any, Rate::Any);
        let a = Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(v.intersect(&a), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_intersect_audio_and_tensor_require_scalar_equality() {
        let a = Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(a.intersect(&a), Ok(a.clone()));
        let b = Caps::Audio {
            format: AudioFormat::Opus,
            channels: 1,
            sample_rate: 48_000,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(a.intersect(&b), Err(G2gError::CapsMismatch));

        let t = Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, 3, 224, 224]),
            layout: TensorLayout::Nchw,
        };
        assert_eq!(t.intersect(&t), Ok(t.clone()));
    }

    #[test]
    fn caps_is_fixed() {
        assert!(video(Dim::Fixed(1), Dim::Fixed(1), Rate::Fixed(1)).is_fixed());
        assert!(!video(Dim::Any, Dim::Fixed(1), Rate::Fixed(1)).is_fixed());
        assert!(!video(Dim::Fixed(1), Dim::Range { min: 1, max: 2 }, Rate::Fixed(1)).is_fixed());
        assert!(Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 44_100,
            channel_layout: crate::ChannelLayout::UNSPECIFIED
        }
        .is_fixed());
    }

    #[test]
    fn audio_channels_wildcard_intersect() {
        let pcm = |ch, rate| Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: ch,
            sample_rate: rate,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        };
        let aac = |ch, rate| Caps::Audio {
            format: AudioFormat::Aac,
            channels: ch,
            sample_rate: rate,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        };
        // ANY_CHANNELS (0) is a wildcard for both PCM and compressed: the decoder's
        // concrete output channels coupling back onto a demuxer's unknown 0 input
        // must intersect, not empty the link (the M422 back-coupling fix).
        assert_eq!(
            aac(ANY_CHANNELS, 48_000).intersect(&aac(6, 48_000)),
            Ok(aac(6, 48_000))
        );
        assert_eq!(
            pcm(2, 48_000).intersect(&pcm(ANY_CHANNELS, 48_000)),
            Ok(pcm(2, 48_000))
        );
        assert_eq!(
            pcm(ANY_CHANNELS, 48_000).intersect(&pcm(ANY_CHANNELS, 48_000)),
            Ok(pcm(ANY_CHANNELS, 48_000))
        );
        // Two distinct concrete counts are still disjoint.
        assert_eq!(
            aac(2, 48_000).intersect(&aac(6, 48_000)),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn audio_channels_wildcard_is_fixed_and_fixate() {
        let pcm = |ch, rate| Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: ch,
            sample_rate: rate,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        };
        // A PCM "any channels" is not fixed; it fixates to the stereo placeholder
        // (the real layout arrives via the decoder's CapsChanged).
        assert!(!pcm(ANY_CHANNELS, 48_000).is_fixed());
        assert_eq!(pcm(ANY_CHANNELS, 48_000).fixate(), Ok(pcm(2, 48_000)));
        assert!(pcm(2, 48_000).is_fixed());
        // An unfixable rate still dominates: 0 channels + any-rate cannot fixate.
        assert_eq!(
            pcm(ANY_CHANNELS, ANY_SAMPLE_RATE).fixate(),
            Err(G2gError::CapsMismatch)
        );
        // A compressed "any channels" stays nominal/fixed (the decoder replaces it
        // before anything reads it), so it round-trips through fixate unchanged.
        let aac0 = Caps::Audio {
            format: AudioFormat::Aac,
            channels: ANY_CHANNELS,
            sample_rate: 0,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        };
        assert!(aac0.is_fixed());
        assert_eq!(aac0.fixate(), Ok(aac0.clone()));
    }

    #[test]
    fn audio_channel_layout_intersects_as_a_wildcard() {
        let pcm = |layout| Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 6,
            sample_rate: 48_000,
            channel_layout: layout,
        };
        let any = crate::ChannelLayout::UNSPECIFIED;
        let five_one = crate::ChannelLayout::SURROUND_5_1;
        let five_zero = crate::ChannelLayout::default_for(5).unwrap();
        // Unspecified is the wildcard: a declared layout pins it either way round.
        assert_eq!(pcm(any).intersect(&pcm(five_one)), Ok(pcm(five_one)));
        assert_eq!(pcm(five_one).intersect(&pcm(any)), Ok(pcm(five_one)));
        assert_eq!(pcm(any).intersect(&pcm(any)), Ok(pcm(any)));
        assert_eq!(pcm(five_one).intersect(&pcm(five_one)), Ok(pcm(five_one)));
        // Two different declared layouts do not overlap.
        assert_eq!(
            pcm(five_one).intersect(&pcm(five_zero)),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn audio_channel_layout_survives_fixation() {
        let pcm = |ch, layout| Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: ch,
            sample_rate: 48_000,
            channel_layout: layout,
        };
        let any = crate::ChannelLayout::UNSPECIFIED;
        let five_one = crate::ChannelLayout::SURROUND_5_1;
        // An unspecified layout carries no information to fixate against and is
        // already concrete enough (it means the count convention), so neither it
        // nor a declared one blocks or changes fixation.
        assert!(pcm(6, any).is_fixed());
        assert!(pcm(6, five_one).is_fixed());
        assert_eq!(pcm(6, five_one).fixate(), Ok(pcm(6, five_one)));
        assert_eq!(pcm(ANY_CHANNELS, five_one).fixate(), Ok(pcm(2, five_one)));
    }

    #[test]
    fn audio_channel_mask_round_trips_through_the_gst_string() {
        let caps = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 6,
            sample_rate: 48_000,
            channel_layout: crate::ChannelLayout::SURROUND_5_1,
        };
        let s = caps.to_gst_string();
        assert!(
            s.contains(",channel-mask=(bitmask)0x000000000000003f"),
            "printed {s}"
        );
        assert_eq!(CapsSet::from_gst_string(&s).unwrap().alternatives(), [caps]);
        // An unspecified layout prints no field, and an absent field parses back
        // to unspecified.
        let bare = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 6,
            sample_rate: 48_000,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        };
        let s = bare.to_gst_string();
        assert!(!s.contains("channel-mask"), "printed {s}");
        assert_eq!(CapsSet::from_gst_string(&s).unwrap().alternatives(), [bare]);
        // 7.1 exercises gst's side-left/side-right bits (10/11), which sit one
        // above the WAV positions the layout stores.
        let surround = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 8,
            sample_rate: 48_000,
            channel_layout: crate::ChannelLayout::SURROUND_7_1,
        };
        let s = surround.to_gst_string();
        assert!(
            s.contains(",channel-mask=(bitmask)0x0000000000000c3f"),
            "printed {s}"
        );
        assert_eq!(
            CapsSet::from_gst_string(&s).unwrap().alternatives(),
            [surround]
        );
    }

    #[test]
    fn a_channel_mask_naming_an_unmodeled_speaker_is_refused() {
        // gst bit 12 is TOP_FRONT_LEFT and bit 9 is LFE2; neither has a g2g
        // position, so the caps is rejected rather than losing that channel.
        assert!(CapsSet::from_gst_string(
            "audio/x-raw,format=S16LE,channels=8,rate=48000,channel-mask=(bitmask)0x103f"
        )
        .is_none());
        assert!(CapsSet::from_gst_string(
            "audio/x-raw,format=S16LE,channels=7,rate=48000,channel-mask=(bitmask)0x23f"
        )
        .is_none());
        // A malformed bitmask is refused too, not silently treated as absent.
        assert!(CapsSet::from_gst_string(
            "audio/x-raw,format=S16LE,channels=2,rate=48000,channel-mask=(bitmask)0xzz"
        )
        .is_none());
    }

    #[test]
    fn capsset_one_wraps_single_caps() {
        let c = video(Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(30 << 16));
        let set = CapsSet::one(c.clone());
        assert_eq!(set.alternatives(), &[c]);
        assert!(!set.is_empty());
    }

    #[test]
    fn capsset_intersect_single_pair() {
        let a = CapsSet::one(video(
            Dim::Range {
                min: 640,
                max: 1920,
            },
            Dim::Any,
            Rate::Any,
        ));
        let b = CapsSet::one(video(
            Dim::Fixed(1280),
            Dim::Fixed(720),
            Rate::Fixed(30 << 16),
        ));
        let i = a.intersect(&b);
        assert_eq!(
            i.alternatives(),
            &[video(
                Dim::Fixed(1280),
                Dim::Fixed(720),
                Rate::Fixed(30 << 16)
            )]
        );
    }

    #[test]
    fn capsset_intersect_empty_when_no_overlap() {
        let a = CapsSet::one(video(Dim::Fixed(640), Dim::Any, Rate::Any));
        let b = CapsSet::one(video(Dim::Fixed(1280), Dim::Any, Rate::Any));
        assert!(a.intersect(&b).is_empty());
    }

    #[test]
    fn capsset_intersect_preserves_self_preference_order() {
        // self: prefers Rgba8 then H264; other: accepts both with any dims.
        let rgba = |w| Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: w,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: crate::Interlace::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
        };
        let h264 = |w| Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: w,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
        };
        let a = CapsSet::from_alternatives(alloc::vec![rgba(Dim::Any), h264(Dim::Any)]);
        let b =
            CapsSet::from_alternatives(alloc::vec![h264(Dim::Fixed(1280)), rgba(Dim::Fixed(640))]);
        let i = a.intersect(&b);
        // self's outer order wins: Rgba8 first even though other lists H264 first.
        assert_eq!(
            i.alternatives(),
            &[rgba(Dim::Fixed(640)), h264(Dim::Fixed(1280))]
        );
    }

    #[test]
    fn capsset_intersect_dedupes_equal_results() {
        // Two self-alternatives that both intersect `other` to the same Caps.
        let any = video(Dim::Any, Dim::Any, Rate::Any);
        let fixed = video(Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(30 << 16));
        let a = CapsSet::from_alternatives(alloc::vec![any.clone(), any.clone()]);
        let b = CapsSet::one(fixed.clone());
        let i = a.intersect(&b);
        assert_eq!(i.alternatives(), &[fixed]);
    }

    #[test]
    fn capsset_union_preserves_self_order_and_dedupes() {
        let a = video(Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(30 << 16));
        let b = video(Dim::Fixed(1280), Dim::Fixed(720), Rate::Fixed(30 << 16));
        let c = video(Dim::Fixed(1920), Dim::Fixed(1080), Rate::Fixed(30 << 16));
        let lhs = CapsSet::from_alternatives(alloc::vec![a.clone(), b.clone()]);
        let rhs = CapsSet::from_alternatives(alloc::vec![b.clone(), c.clone()]);
        let u = lhs.union(&rhs);
        assert_eq!(u.alternatives(), &[a, b, c]);
    }

    #[test]
    fn capsset_fixate_picks_first_fixable_alternative() {
        // First alt has framerate Any (not fixable); second is fully fixable.
        let unfixable = video(Dim::Fixed(640), Dim::Fixed(480), Rate::Any);
        let fixable = video(
            Dim::Range {
                min: 800,
                max: 1920,
            },
            Dim::Fixed(720),
            Rate::Fixed(30 << 16),
        );
        let set = CapsSet::from_alternatives(alloc::vec![unfixable, fixable]);
        assert_eq!(
            set.fixate(),
            Some(video(
                Dim::Fixed(800),
                Dim::Fixed(720),
                Rate::Fixed(30 << 16)
            ))
        );
    }

    #[test]
    fn capsset_fixate_empty_or_all_unfixable_yields_none() {
        assert!(CapsSet::from_alternatives(Vec::new()).fixate().is_none());
        let only_any = video(Dim::Any, Dim::Any, Rate::Any);
        assert!(CapsSet::one(only_any).fixate().is_none());
    }

    #[test]
    fn caps_fixate_collapses_ranges_and_rejects_any() {
        let ranged = video(
            Dim::Range {
                min: 640,
                max: 1920,
            },
            Dim::Fixed(480),
            Rate::Any,
        );
        assert_eq!(ranged.fixate(), Err(G2gError::CapsMismatch)); // framerate Any

        let fixable = video(
            Dim::Range {
                min: 640,
                max: 1920,
            },
            Dim::Fixed(480),
            Rate::Fixed(30 << 16),
        );
        let fixed = fixable.fixate().unwrap();
        assert!(fixed.is_fixed());
        assert_eq!(
            fixed,
            video(Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(30 << 16))
        );
    }

    #[test]
    fn planar_format_layout_helpers() {
        use RawVideoFormat::*;
        // Bit depth and the 2-byte sample size for the high-bit-depth variants.
        for f in [I420, I422, I444] {
            assert_eq!(f.bit_depth(), 8);
            assert_eq!(f.bytes_per_sample(), 1);
        }
        for f in [I420p10, I422p10, I444p10] {
            assert_eq!(f.bit_depth(), 10);
            assert_eq!(f.bytes_per_sample(), 2);
        }
        for f in [I420p12, I422p12, I444p12] {
            assert_eq!(f.bit_depth(), 12);
            assert_eq!(f.bytes_per_sample(), 2);
        }
        // Chroma subsampling shift: 4:2:0 = (1,1), 4:2:2 = (1,0), 4:4:4 = (0,0).
        assert_eq!(I420p10.chroma_shift(), Some((1, 1)));
        assert_eq!(I422.chroma_shift(), Some((1, 0)));
        assert_eq!(I444p12.chroma_shift(), Some((0, 0)));
        // The non-planar formats are not in the fully-planar family.
        for f in [Nv12, Rgba8, Bgra8, Yuyv] {
            assert!(!f.is_planar_yuv());
            assert_eq!(f.chroma_shift(), None);
        }
        assert!(I444p10.is_planar_yuv());
        // Semi-planar 10-bit: 2-byte samples, outside the fully-planar family.
        assert_eq!(P010.bit_depth(), 10);
        assert_eq!(P010.bytes_per_sample(), 2);
        assert!(!P010.is_planar_yuv());
    }

    #[test]
    fn single_stride_frame_size_covers_packed_and_semi_planar() {
        use RawVideoFormat::*;
        // Packed: one plane. YUYV is two bytes per pixel, RGBA four.
        assert_eq!(Yuyv.row_stride(640), Some(1280));
        assert_eq!(Yuyv.frame_bytes(1280, 480), Some(1280 * 480));
        assert_eq!(Rgba8.row_stride(640), Some(2560));
        assert_eq!(Rgba8.frame_bytes(2560, 480), Some(2560 * 480));
        // NV12 luma is one byte per pixel plus the half-height chroma region, and
        // a padded stride pads every row of both.
        assert_eq!(Nv12.row_stride(640), Some(640));
        assert_eq!(Nv12.frame_bytes(640, 480), Some(640 * 480 * 3 / 2));
        assert_eq!(Nv12.frame_bytes(704, 480), Some(704 * 480 * 3 / 2));
        // Odd height still rounds the chroma rows up.
        assert_eq!(Nv12.frame_bytes(4, 3), Some(4 * 3 + 4 * 2));
        // A bogus stride/height cannot overflow into a small allocation.
        assert_eq!(Nv12.frame_bytes(u64::MAX, 4), None);
        // Formats with no single-stride layout report nothing rather than a guess.
        assert_eq!(I420p10.frame_bytes(640, 480), None);
        assert_eq!(P010.row_stride(640), None);
    }

    #[test]
    fn plane_layout_covers_every_format_family() {
        use RawVideoFormat::*;
        // Packed: one plane, the pixel stride straight off the format.
        assert_eq!((Rgba8.plane_count(), Rgba8.pixel_stride()), (1, Some(4)));
        assert_eq!(Rgba8.plane_stride(0, 640), Some(2560));
        assert_eq!(Rgba8.unpadded_frame_bytes(640, 480), Some(640 * 480 * 4));
        assert_eq!(Yuyv.plane_stride(0, 640), Some(1280));
        assert_eq!(Rgba8.plane_stride(1, 640), None, "no second plane");

        // Semi-planar: luma then one interleaved chroma plane at half height.
        assert_eq!((Nv12.plane_count(), Nv12.pixel_stride()), (2, None));
        assert_eq!(Nv12.plane_stride(1, 640), Some(640));
        assert_eq!(Nv12.plane_rows(1, 480), Some(240));
        assert_eq!(Nv12.unpadded_frame_bytes(640, 480), Some(640 * 480 * 3 / 2));
        assert_eq!(Nv12.plane_offset(1, 640, 480), Some(640 * 480));
        // P010 is NV12's layout with 2-byte samples, which `frame_bytes` refuses.
        assert_eq!(P010.plane_stride(0, 640), Some(1280));
        assert_eq!(P010.unpadded_frame_bytes(640, 480), Some(640 * 480 * 3));

        // Fully planar: chroma dimensions follow the subsampling shift.
        assert_eq!(I420.plane_count(), 3);
        assert_eq!(I420.plane_stride(1, 640), Some(320));
        assert_eq!(I420.plane_rows(1, 480), Some(240));
        assert_eq!(I420.unpadded_frame_bytes(640, 480), Some(640 * 480 * 3 / 2));
        // 4:2:2 halves width only, 4:4:4 neither.
        assert_eq!(I422.plane_rows(1, 480), Some(480));
        assert_eq!(I422.unpadded_frame_bytes(640, 480), Some(640 * 480 * 2));
        assert_eq!(I444.plane_stride(1, 640), Some(640));
        assert_eq!(I444.unpadded_frame_bytes(640, 480), Some(640 * 480 * 3));
        // 10-bit doubles every plane.
        assert_eq!(I420p10.unpadded_frame_bytes(640, 480), Some(640 * 480 * 3));

        // Odd geometry rounds chroma up rather than truncating a row away.
        assert_eq!(I420.unpadded_frame_bytes(3, 3), Some(9 + 2 * 2 * 2));
        assert_eq!(Nv12.unpadded_frame_bytes(3, 3), Some(9 + 4 * 2));

        // A width that overflows its stride yields nothing, never a short buffer.
        assert_eq!(Rgba8.unpadded_frame_bytes(u32::MAX, 4), None);
    }

    fn video_with_colorimetry(colorimetry: Colorimetry) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
            interlace: crate::Interlace::Any,
            colorimetry,
        }
    }

    #[test]
    fn cicp_codepoints_round_trip_through_from_cicp() {
        // What an encoder writes must read back as the value it was handed, so
        // a re-encode cannot rename a stream's colour description.
        for matrix in [
            MatrixCoefficients::Identity,
            MatrixCoefficients::Bt601,
            MatrixCoefficients::Bt709,
            MatrixCoefficients::Bt2020Ncl,
        ] {
            assert_eq!(MatrixCoefficients::from_cicp(matrix.to_cicp()), matrix);
        }
        for transfer in [
            TransferCharacteristics::Srgb,
            TransferCharacteristics::Bt601,
            TransferCharacteristics::Bt709,
            TransferCharacteristics::Bt2020,
            TransferCharacteristics::Pq,
            TransferCharacteristics::Hlg,
        ] {
            assert_eq!(
                TransferCharacteristics::from_cicp(transfer.to_cicp()),
                transfer
            );
        }
        for primaries in [
            ColorPrimaries::Bt709,
            ColorPrimaries::Bt470bg,
            ColorPrimaries::Smpte170m,
            ColorPrimaries::Bt2020,
        ] {
            assert_eq!(ColorPrimaries::from_cicp(primaries.to_cicp()), primaries);
        }
    }

    #[test]
    fn unknown_writes_the_unspecified_codepoint() {
        // CICP 2 on every field: an untagged stream stays untagged through an
        // encoder rather than picking up a guessed description.
        assert_eq!(MatrixCoefficients::Unknown.to_cicp(), 2);
        assert_eq!(TransferCharacteristics::Unknown.to_cicp(), 2);
        assert_eq!(ColorPrimaries::Unknown.to_cicp(), 2);
    }

    /// The bt709 preset writes the 1/1/1 description an H.264 VUI carries for
    /// HD, the codepoints `h264parse` reads back as `Colorimetry::BT709`.
    #[test]
    fn bt709_preset_writes_the_hd_codepoints() {
        assert_eq!(Colorimetry::BT709.primaries.to_cicp(), 1);
        assert_eq!(Colorimetry::BT709.transfer.to_cicp(), 1);
        assert_eq!(Colorimetry::BT709.matrix.to_cicp(), 1);
        assert_eq!(
            Colorimetry::from_cicp(
                Colorimetry::BT709.primaries.to_cicp(),
                Colorimetry::BT709.transfer.to_cicp(),
                Colorimetry::BT709.matrix.to_cicp(),
                false,
            ),
            Colorimetry::BT709
        );
    }

    #[test]
    fn colorimetry_unknown_is_a_wildcard_in_intersect() {
        let unknown = video_with_colorimetry(Colorimetry::UNKNOWN);
        let bt709 = video_with_colorimetry(Colorimetry::BT709);
        // Unknown yields to the concrete side, in both directions.
        assert_eq!(unknown.intersect(&bt709), Ok(bt709.clone()));
        assert_eq!(bt709.intersect(&unknown), Ok(bt709.clone()));
        assert_eq!(unknown.intersect(&unknown), Ok(unknown.clone()));
        assert_eq!(bt709.intersect(&bt709), Ok(bt709.clone()));
        // Two different concrete colorimetries are disjoint: converting with
        // the wrong matrix must fail the link, not silently pick one.
        let bt601 = video_with_colorimetry(Colorimetry::BT601);
        assert_eq!(bt709.intersect(&bt601), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn colorimetry_intersects_field_wise() {
        // A matrix-only constraint meets a range-only constraint: the result
        // carries both, since Unknown is per-field, not all-or-nothing.
        let matrix_only = Colorimetry {
            matrix: MatrixCoefficients::Bt709,
            ..Colorimetry::UNKNOWN
        };
        let range_only = Colorimetry {
            range: ColorRange::Full,
            ..Colorimetry::UNKNOWN
        };
        assert_eq!(
            matrix_only.intersect(&range_only),
            Some(Colorimetry {
                matrix: MatrixCoefficients::Bt709,
                range: ColorRange::Full,
                ..Colorimetry::UNKNOWN
            })
        );
        assert_eq!(
            matrix_only.intersect(&Colorimetry {
                matrix: MatrixCoefficients::Bt601,
                ..Colorimetry::UNKNOWN
            }),
            None
        );
    }

    #[test]
    fn colorimetry_survives_fixate() {
        let bt709 = video_with_colorimetry(Colorimetry::BT709);
        assert!(bt709.is_fixed());
        assert_eq!(bt709.fixate(), Ok(bt709.clone()));
        // Unknown stays Unknown: fixation never invents a colour description.
        let unknown = video_with_colorimetry(Colorimetry::UNKNOWN);
        assert_eq!(unknown.fixate(), Ok(unknown.clone()));
    }

    #[test]
    fn colorimetry_gst_string_presets_and_4_part_form() {
        assert_eq!(Colorimetry::BT709.to_gst_string().as_deref(), Some("bt709"));
        assert_eq!(Colorimetry::BT601.to_gst_string().as_deref(), Some("bt601"));
        assert_eq!(
            Colorimetry::BT2020.to_gst_string().as_deref(),
            Some("bt2020")
        );
        assert_eq!(Colorimetry::SRGB.to_gst_string().as_deref(), Some("sRGB"));
        assert_eq!(Colorimetry::UNKNOWN.to_gst_string(), None);
        // A combo with no preset name prints GStreamer's numeric 4-part form
        // (range:matrix:transfer:primaries) and parses back to itself.
        let mixed = Colorimetry {
            range: ColorRange::Limited,
            matrix: MatrixCoefficients::Bt709,
            transfer: TransferCharacteristics::Bt601,
            primaries: ColorPrimaries::Bt709,
        };
        let printed = mixed.to_gst_string().unwrap();
        assert_eq!(printed, "2:3:16:1");
        assert_eq!(Colorimetry::from_gst_string(&printed), Some(mixed));
        // Preset names parse case-insensitively; garbage and unmodeled
        // GStreamer numbers are rejected, not widened to a wildcard.
        assert_eq!(
            Colorimetry::from_gst_string("srgb"),
            Some(Colorimetry::SRGB)
        );
        assert_eq!(Colorimetry::from_gst_string("banana"), None);
        assert_eq!(Colorimetry::from_gst_string("2:5:16:1"), None); // smpte240m matrix
    }

    #[test]
    fn caps_to_gst_string_carries_colorimetry() {
        let caps = video_with_colorimetry(Colorimetry::BT709);
        assert!(caps.to_gst_string().contains(",colorimetry=bt709"));
        // Unknown is the absent field, like a wildcard dimension.
        let caps = video_with_colorimetry(Colorimetry::UNKNOWN);
        assert!(!caps.to_gst_string().contains("colorimetry"));
        let compressed = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: Colorimetry::BT709,
        };
        assert!(compressed.to_gst_string().contains(",colorimetry=bt709"));
    }

    #[test]
    fn every_raw_format_has_a_distinct_gst_name() {
        use RawVideoFormat::*;
        let all = [
            Nv12, I420, Rgba8, Bgra8, Yuyv, I420p10, I420p12, I422, I422p10, I422p12, I444,
            I444p10, I444p12, P010,
        ];
        let mut names: Vec<&str> = all.iter().map(|f| raw_format_gst_name(*f)).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "gst format names must be unique");
    }
}
