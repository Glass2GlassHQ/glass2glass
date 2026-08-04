//! MPEG-TS multiplexer element (M114): one elementary stream in
//! (`Caps::CompressedVideo{H264|H265}` Annex-B, or `Caps::Audio{Aac}` ADTS), an
//! MPEG-TS byte stream out (`Caps::ByteStream{MpegTs}`).
//!
//! Wraps the pure [`crate::mpegts::TsMuxer`], the inverse of
//! [`crate::tsdemux::TsDemux`]: each input access unit becomes a PES packet split
//! across 188-byte TS packets, with PAT + PMT emitted once up front. The PMT
//! stream type is read from the input caps at configure. CPU, `no_std` baseline.
//!
//! ```text
//! ... ! h264parse ! mpegtsmux ! filesink location=out.ts
//! ```
//!
//! Scope (v1): one program / one stream (a single input pad), mirroring the
//! single-stream `TsDemux`. A PCR rides the stream's PID on the `pcr-interval`
//! cadence; multi-stream (A+V) muxing is the sibling `tsmuxn::TsMux`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, Tag, TagList, VideoCodec,
};

use crate::mpegts::{
    TsMuxer, STREAM_TYPE_AAC, STREAM_TYPE_H264, STREAM_TYPE_H265, STREAM_TYPE_METADATA_PES,
    STREAM_TYPE_PRIVATE_PES, TAG_KEY_SERVICE_NAME, TAG_KEY_SERVICE_PROVIDER,
};

/// The PMT `stream_type` for an input caps, or `None` if unsupported. Shared by
/// the single-input [`TsMux`] and the multi-input `tsmuxn::TsMux`. KLV metadata
/// rides a private PES (0x06) with the 'KLVA' registration descriptor
/// (asynchronous KLV, MISB ST 1402 / STANAG 4609), or, with `klv_sync`,
/// metadata-in-PES (0x15) with a metadata descriptor (synchronous KLV).
pub(crate) fn stream_type_for(caps: &Caps, klv_sync: bool) -> Option<u8> {
    match caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            ..
        } => Some(STREAM_TYPE_H264),
        Caps::CompressedVideo {
            codec: VideoCodec::H265,
            ..
        } => Some(STREAM_TYPE_H265),
        Caps::Audio {
            format: AudioFormat::Aac,
            ..
        } => Some(STREAM_TYPE_AAC),
        Caps::Klv => Some(if klv_sync {
            STREAM_TYPE_METADATA_PES
        } else {
            STREAM_TYPE_PRIVATE_PES
        }),
        _ => None,
    }
}

/// The SDT service text a tag list names (M872): the service name from
/// [`Tag::Title`] or a `service_name` key, the provider from a `service_provider`
/// key (ffprobe's names for the two `service_descriptor` fields). `None` when the
/// list names neither, in which case no SDT is written. Shared by the single-input
/// [`TsMux`] and `tsmuxn::TsMux`.
///
/// TS standardizes only these two whole-service strings and a per-stream language,
/// with no free-form tag carrier, so every other tag is dropped rather than
/// smuggled into a private descriptor.
pub(crate) fn service_from_tags(tags: &TagList) -> Option<(String, String)> {
    let mut name: Option<&str> = None;
    let mut provider: Option<&str> = None;
    for tag in tags.tags() {
        match tag {
            Tag::Title(v) => name = Some(v),
            Tag::Other { key, value } if key.eq_ignore_ascii_case(TAG_KEY_SERVICE_NAME) => {
                name = Some(value)
            }
            Tag::Other { key, value } if key.eq_ignore_ascii_case(TAG_KEY_SERVICE_PROVIDER) => {
                provider = Some(value)
            }
            _ => {}
        }
    }
    (name.is_some() || provider.is_some()).then(|| {
        (
            String::from(name.unwrap_or_default()),
            String::from(provider.unwrap_or_default()),
        )
    })
}

/// The language a tag list declares, for a stream's
/// `ISO_639_language_descriptor`. The only per-stream tag TS carries.
pub(crate) fn language_from_tags(tags: &TagList) -> Option<&str> {
    tags.tags().iter().find_map(|tag| match tag {
        Tag::Language(v) => Some(v.as_str()),
        _ => None,
    })
}

/// Muxes one elementary stream into an MPEG-TS byte stream.
#[derive(Debug)]
pub struct TsMux {
    /// Built at configure, once the input codec (and so the PMT stream type) is
    /// known.
    mux: Option<TsMuxer>,
    configured: bool,
    emitted: u64,
    /// PAT/PMT re-emission cadence in milliseconds (`0` = once up front). Applied
    /// to the inner `TsMuxer` when it is built at configure. The PAT and PMT are
    /// emitted together, so `pat-interval` and `pmt-interval` share this cadence.
    table_interval_ms: u64,
    /// PCR insertion cadence in 90 kHz ticks (default 3600). Applied to the inner
    /// `TsMuxer` when it is built at configure.
    pcr_interval_90khz: u64,
    /// Carry a `Caps::Klv` input as synchronous KLV (metadata-in-PES 0x15)
    /// instead of the default asynchronous private PES (0x06 + 'KLVA').
    klv_sync: bool,
    /// Metadata for the one service this muxer writes (M872): the service name /
    /// provider go to the SDT, a `Tag::Language` to the single stream's PMT entry.
    tags: TagList,
}

impl Default for TsMux {
    fn default() -> Self {
        Self::new()
    }
}

impl TsMux {
    pub fn new() -> Self {
        Self {
            mux: None,
            configured: false,
            emitted: 0,
            table_interval_ms: 0,
            pcr_interval_90khz: 3600,
            klv_sync: false,
            tags: TagList::new(),
        }
    }

    /// Attach metadata (M872). A transport stream carries two kinds: the service
    /// name / provider, written to the SDT ([`Tag::Title`] or a `service_name` key,
    /// and a `service_provider` key), and this stream's language
    /// ([`Tag::Language`]), written as an `ISO_639_language_descriptor` in its PMT
    /// entry. TS defines no free-form tag carrier, so any other tag is dropped.
    /// The multi-input sibling splits the two scopes across
    /// `with_tags` / `with_track_tags`; a single stream needs only one list.
    pub fn with_tags(mut self, tags: TagList) -> Self {
        self.tags = tags;
        self
    }

    /// Set the PAT/PMT re-emission interval in milliseconds (`0` = once up front).
    pub fn with_table_interval_ms(mut self, ms: u64) -> Self {
        self.table_interval_ms = ms;
        self
    }

    /// Set the PCR insertion interval in 90 kHz ticks (default 3600).
    pub fn with_pcr_interval(mut self, ticks: u64) -> Self {
        self.pcr_interval_90khz = ticks;
        self
    }

    /// Carry a `Caps::Klv` input as synchronous KLV: metadata-in-PES
    /// (`stream_type` 0x15) rather than the default asynchronous private PES.
    pub fn with_klv_sync(mut self, sync: bool) -> Self {
        self.klv_sync = sync;
        self
    }

    /// Count of TS byte frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The output it produces: an MPEG-TS byte stream.
    fn output_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
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
        Vec::from([
            video(VideoCodec::H264),
            video(VideoCodec::H265),
            Caps::Audio {
                format: AudioFormat::Aac,
                channels: 0,
                sample_rate: 0,
            },
            Caps::Klv,
        ])
    }
}

impl AsyncElement for TsMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if stream_type_for(upstream_caps, self.klv_sync).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // Any supported elementary stream maps to one MPEG-TS byte stream.
        let klv_sync = self.klv_sync;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            if stream_type_for(input, klv_sync).is_some() {
                CapsSet::one(Self::output_caps())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let stream_type =
            stream_type_for(absolute_caps, self.klv_sync).ok_or(G2gError::CapsMismatch)?;
        let mut mux = TsMuxer::new(stream_type);
        // 90 kHz clock: 90 ticks per millisecond.
        mux.set_table_interval_90khz(self.table_interval_ms.saturating_mul(90));
        mux.set_pcr_interval_90khz(self.pcr_interval_90khz);
        if let Some((name, provider)) = service_from_tags(&self.tags) {
            mux.set_service(&name, &provider);
        }
        if let Some(language) = language_from_tags(&self.tags) {
            mux.set_stream_language(0, language);
        }
        self.mux = Some(mux);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "pat-interval",
                PropKind::Uint,
                "PAT/PMT re-emission interval, milliseconds (0 = once)",
            )
            .with_default("0"),
            // The PAT and PMT are emitted as a pair, so this shares the cadence.
            PropertySpec::new(
                "pmt-interval",
                PropKind::Uint,
                "alias of pat-interval (the tables are emitted together)",
            )
            .with_default("0"),
            PropertySpec::new(
                "pcr-interval",
                PropKind::Uint,
                "PCR insertion interval, ticks of the 90kHz clock",
            )
            .with_default("3600"),
            PropertySpec::new(
                "klv-sync",
                PropKind::Bool,
                "carry KLV metadata as synchronous metadata-in-PES (stream_type 0x15) instead of asynchronous private PES",
            )
            .with_default("false"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "pat-interval" | "pmt-interval" => {
                self.table_interval_ms = value.as_uint().ok_or(PropError::Type)?;
                if let Some(mux) = self.mux.as_mut() {
                    mux.set_table_interval_90khz(self.table_interval_ms.saturating_mul(90));
                }
                Ok(())
            }
            "pcr-interval" => {
                self.pcr_interval_90khz = value.as_uint().ok_or(PropError::Type)?;
                if let Some(mux) = self.mux.as_mut() {
                    mux.set_pcr_interval_90khz(self.pcr_interval_90khz);
                }
                Ok(())
            }
            "klv-sync" => {
                // The stream type is baked into the PMT when the muxer is built at
                // configure, so this only takes effect before that.
                if self.mux.is_some() {
                    return Err(PropError::ReadOnly);
                }
                self.klv_sync = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "pat-interval" | "pmt-interval" => Some(PropValue::Uint(self.table_interval_ms)),
            "pcr-interval" => Some(PropValue::Uint(self.pcr_interval_90khz)),
            "klv-sync" => Some(PropValue::Bool(self.klv_sync)),
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
                    let mux = self.mux.as_mut().ok_or(G2gError::NotConfigured)?;
                    let pts_90khz = (frame.timing.pts_ns as u128 * 90_000 / 1_000_000_000) as u64;
                    // A DTS rides the PES only for reordered video (dts_ns set and
                    // distinct from the PTS); 0 is the unset sentinel, equal adds nothing.
                    let dts_90khz = (frame.timing.dts_ns != 0
                        && frame.timing.dts_ns != frame.timing.pts_ns)
                        .then(|| (frame.timing.dts_ns as u128 * 90_000 / 1_000_000_000) as u64);
                    let ts = mux.push_au(slice, Some(pts_90khz), dts_90khz);
                    let out_frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(ts.into_boxed_slice())),
                        FrameTiming {
                            pts_ns: frame.timing.pts_ns,
                            ..FrameTiming::default()
                        },
                        self.emitted,
                    );
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                // The runner's transform arm forwards EOS; nothing to flush here.
                PipelinePacket::Eos => {}
                // Input geometry / params don't change the TS framing.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for TsMux {
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
    use crate::tsdemux::TsDemux;
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
        let m = TsMux::new();
        assert!(m.intercept_caps(&h264_caps()).is_ok());
        // Raw video / an existing byte stream have nothing to mux.
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
                encoding: ByteStreamEncoding::MpegTs
            }]
        ));
    }

    #[tokio::test]
    async fn element_round_trips_through_tsdemux() {
        let au0 = alloc::vec![0u8, 0, 0, 1, 0x65, 0xAA];
        let au1 = alloc::vec![0u8, 0, 0, 1, 0x41, 0xBB, 0xCC];

        let mut mux = TsMux::new();
        mux.configure_pipeline(&h264_caps()).unwrap();
        let mut ts_sink = CaptureSink::default();
        mux.process(h264_frame(au0.clone(), 10_000_000), &mut ts_sink)
            .await
            .unwrap();
        mux.process(h264_frame(au1.clone(), 20_000_000), &mut ts_sink)
            .await
            .unwrap();
        mux.process(PipelinePacket::Eos, &mut ts_sink)
            .await
            .unwrap();
        assert!(
            !ts_sink.eos,
            "EOS is forwarded by the runner's arm, not the element"
        );

        // Feed the muxed TS bytes back through the demuxer.
        let mut ts = Vec::new();
        for f in &ts_sink.frames {
            ts.extend_from_slice(f);
        }
        let mut demux = TsDemux::new();
        demux
            .configure_pipeline(&Caps::ByteStream {
                encoding: ByteStreamEncoding::MpegTs,
            })
            .unwrap();
        let mut au_sink = CaptureSink::default();
        let ts_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(ts.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        demux
            .process(PipelinePacket::DataFrame(ts_frame), &mut au_sink)
            .await
            .unwrap();
        demux
            .process(PipelinePacket::Eos, &mut au_sink)
            .await
            .unwrap();

        assert_eq!(
            au_sink.frames,
            alloc::vec![au0, au1],
            "AUs recovered through mux + demux"
        );
        assert_eq!(mux.emitted(), 2);
    }
}
