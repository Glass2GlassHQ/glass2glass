// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::task::{Context, Poll};

use crate::decoderecovery::DecodeErrorRecovery;
use crate::displayorder::{PictureMeta, ReorderBuffer};
use crate::h264dpb::H264_DAMAGE_REFUSALS;
use crate::h264parse::{
    h264_parameter_sets_in, sps_reorder_frames, H264Sps, H264StreamRefused, MALFORMED_SPS,
};
use crate::inputstamp::{InputStamp, PendingInputStamps};
use g2g_core::frame::Frame;
use g2g_core::log::{ClockTime, Target};
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    AllocationParams, AsyncElement, BitstreamAlignment, Caps, CapsConstraint, CapsSet, Colorimetry,
    ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError, Interlace, MemoryDomain,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PixelAspect, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat, Segment, VideoCodec,
};

pub trait H264HardwareDecoder: Sized + Send + core::fmt::Debug {
    type Picture: Send;
    type Error: core::fmt::Debug + From<H264StreamRefused>;
    const LOG_CATEGORY: &'static str;
    const OUTPUT_DOMAINS: DomainSet;

    fn metadata() -> ElementMetadata;
    fn decodes_h264() -> bool;
    fn open() -> Result<Self, Self::Error>;
    // later calls, for the SPS a keyframe repeats, leave the running session alone
    fn start_session(&mut self, sps: &H264Sps, reorder_depth: usize) -> Result<(), Self::Error>;
    fn has_session(&self) -> bool;
    fn poll_free_picture(&self, cx: &mut Context<'_>) -> Poll<()>;
    fn decode_access_unit(
        &mut self,
        access_unit: &[u8],
    ) -> Result<Vec<(PictureMeta, Self::Picture)>, Self::Error>;
    fn reset(&mut self);
    fn picture_extent(&self) -> (u32, u32);
    fn output(
        &mut self,
        picture: Self::Picture,
        domain: MemoryDomainKind,
    ) -> Result<MemoryDomain, Self::Error>;
    fn stream_refusal(error: &Self::Error) -> Option<&'static str>;
    fn refused(error: Self::Error) -> G2gError;
}

pub struct H264HardwareDec<B: H264HardwareDecoder> {
    decoder: Option<B>,
    out_domain: MemoryDomainKind,
    framerate: Rate,
    input_pixel_aspect: PixelAspect,
    stream_colorimetry: Colorimetry,
    reorder: ReorderBuffer<(FrameTiming, B::Picture)>,
    stream_reorder_depth: usize,
    low_latency: bool,
    input_stamps: PendingInputStamps,
    recovery: DecodeErrorRecovery,
    segment: Segment,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl<B: H264HardwareDecoder> core::fmt::Debug for H264HardwareDec<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(B::LOG_CATEGORY)
            .field("decoder", &self.decoder)
            .field("out_domain", &self.out_domain)
            .finish_non_exhaustive()
    }
}

impl<B: H264HardwareDecoder> Default for H264HardwareDec<B> {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn nv12_caps(
    width: Dim,
    height: Dim,
    framerate: Rate,
    colorimetry: Colorimetry,
    pixel_aspect: PixelAspect,
) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width,
        height,
        framerate,
        interlace: Interlace::Any,
        colorimetry,
        pixel_aspect,
    }
}

fn h264_caps(width: Dim, height: Dim) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width,
        height,
        framerate: Rate::Any,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: PixelAspect::Any,
        alignment: BitstreamAlignment::Unspecified,
    }
}

impl<B: H264HardwareDecoder> H264HardwareDec<B> {
    pub fn new() -> Self {
        Self {
            decoder: None,
            out_domain: MemoryDomainKind::System,
            framerate: Rate::Any,
            input_pixel_aspect: PixelAspect::Any,
            stream_colorimetry: Colorimetry::UNKNOWN,
            reorder: ReorderBuffer::new(),
            stream_reorder_depth: 0,
            low_latency: false,
            input_stamps: PendingInputStamps::default(),
            recovery: DecodeErrorRecovery::default(),
            segment: Segment::new(),
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn accepts_input(input: &Caps) -> bool {
        let is_h264 = matches!(
            input,
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }
        );
        is_h264 && B::decodes_h264()
    }

    #[cfg(all(target_os = "windows", feature = "d3d12-decode"))]
    pub(crate) fn decoder(&self) -> Option<&B> {
        self.decoder.as_ref()
    }

    fn apply_reorder_depth(&mut self) {
        self.reorder.max_hold = self.stream_reorder_depth;
        self.reorder.low_latency = self.low_latency;
    }

    // false while no SPS has arrived to start the session from
    fn ensure_session(&mut self, access_unit: &[u8]) -> Result<bool, G2gError> {
        let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
        let Some(sets) = h264_parameter_sets_in(access_unit)
            .map_err(B::Error::from)
            .map_err(B::refused)?
        else {
            return Ok(decoder.has_session());
        };
        self.stream_colorimetry = Colorimetry::from_cicp(
            sets.sps.color_primaries,
            sets.sps.transfer_characteristics,
            sets.sps.matrix_coefficients,
            sets.sps.video_full_range_flag,
        );
        self.stream_reorder_depth = sps_reorder_frames(access_unit)
            .map(usize::from)
            .ok_or(B::Error::from(MALFORMED_SPS))
            .map_err(B::refused)?;
        decoder
            .start_session(&sets.sps, self.stream_reorder_depth)
            .map_err(B::refused)?;
        self.apply_reorder_depth();
        Ok(true)
    }

    async fn decode(&mut self, frame: Frame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let access_unit = frame.domain.require_system_slice(B::LOG_CATEGORY)?;
        if !self.ensure_session(access_unit)? {
            return Ok(());
        }
        if !self
            .recovery
            .admits(|| crate::annexb::au_is_keyframe(VideoCodec::H264, access_unit))
        {
            return Ok(());
        }
        let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
        core::future::poll_fn(|cx| decoder.poll_free_picture(cx)).await;
        let pictures = match decoder.decode_access_unit(access_unit) {
            Ok(pictures) => pictures,
            Err(error) => {
                let damaged = B::stream_refusal(&error).is_some_and(|reason| {
                    H264_DAMAGE_REFUSALS.contains(&H264StreamRefused(reason))
                });
                if !damaged {
                    return Err(B::refused(error));
                }
                self.recovery.failed().map_err(|_| B::refused(error))?;
                decoder.reset();
                return Ok(());
            }
        };
        self.recovery.decoded();
        let timing = frame.timing;
        self.input_stamps
            .insert(timing.pts_ns, InputStamp::of(&frame));
        let mut ready = Vec::new();
        for (meta, picture) in pictures {
            let release = self.reorder.push(&meta, (timing, picture));
            for (dropped, _) in release.dropped {
                self.input_stamps.take(dropped.pts_ns);
            }
            ready.extend(release.output);
        }
        self.emit(ready, out).await
    }

    async fn emit(
        &mut self,
        pictures: Vec<(FrameTiming, B::Picture)>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
        let (width, height) = decoder.picture_extent();
        for (timing, picture) in pictures {
            let input = self.input_stamps.take(timing.pts_ns);
            if !self.segment.presents(timing.pts_ns) {
                g2g_core::g2g_log!(
                    Target::category(B::LOG_CATEGORY),
                    "dropping frame pts {} outside the segment",
                    ClockTime(timing.pts_ns)
                );
                continue;
            }
            let caps = nv12_caps(
                Dim::Fixed(width),
                Dim::Fixed(height),
                self.framerate.clone(),
                self.stream_colorimetry,
                self.input_pixel_aspect,
            );
            if self.last_caps.as_ref() != Some(&caps) {
                out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                self.last_caps = Some(caps);
            }
            let domain = decoder
                .output(picture, self.out_domain)
                .map_err(B::refused)?;
            let frame = Frame {
                domain,
                timing: FrameTiming {
                    dts_ns: timing.pts_ns,
                    duration_ns: 0,
                    ..timing
                },
                sequence: self.emitted,
                meta: input.meta,
            };
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl<B: H264HardwareDecoder> PadTemplates for H264HardwareDec<B> {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(Vec::from([h264_caps(
                Dim::Any,
                Dim::Any,
            )]))),
            PadTemplate::source(CapsSet::from_alternatives(Vec::from([nv12_caps(
                Dim::Any,
                Dim::Any,
                Rate::Any,
                Colorimetry::UNKNOWN,
                PixelAspect::Any,
            )]))),
        ])
    }
}

impl<B: H264HardwareDecoder> AsyncElement for H264HardwareDec<B> {
    type ProcessFuture<'a>
        = core::pin::Pin<Box<dyn core::future::Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps
            .intersect(&h264_caps(Dim::Any, Dim::Any))
            .map_err(|_| G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width,
                height,
                framerate,
                pixel_aspect,
                ..
            } => CapsSet::from_alternatives(Vec::from([nv12_caps(
                width.clone(),
                height.clone(),
                framerate.clone(),
                Colorimetry::UNKNOWN,
                *pixel_aspect,
            )])),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.out_domain
    }

    fn output_domains(&self) -> DomainSet {
        B::OUTPUT_DOMAINS
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        if let Ok(resolved) = params.clone().resolve_for_producer(B::OUTPUT_DOMAINS) {
            self.out_domain = resolved.domain;
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::CompressedVideo {
            codec: VideoCodec::H264,
            framerate,
            pixel_aspect,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        self.framerate = framerate.clone();
        self.input_pixel_aspect = *pixel_aspect;
        // configure runs again on every caps change, and pictures already sent belong to the open device
        if self.decoder.is_some() {
            return Ok(ConfigureOutcome::Accepted);
        }
        self.decoder = Some(B::open().map_err(B::refused)?);
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        B::metadata()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        H264_HARDWARE_DEC_PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "low-latency" => {
                self.low_latency = value.as_bool().ok_or(PropError::Type)?;
                self.apply_reorder_depth();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "low-latency" => Some(PropValue::Bool(self.low_latency)),
            _ => None,
        }
    }

    fn log_category(&self) -> &'static str {
        B::LOG_CATEGORY
    }

    fn take_reconfigure(&mut self) -> Option<g2g_core::Reconfigure> {
        self.recovery.take_keyframe_request()
    }

    fn latency(&self) -> g2g_core::query::LatencyReport {
        self.reorder.latency(0, self.framerate.clone())
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => self.decode(frame, out).await,
                PipelinePacket::CapsChanged(caps) => match &caps {
                    Caps::CompressedVideo {
                        codec: VideoCodec::H264,
                        pixel_aspect,
                        ..
                    } => {
                        self.input_pixel_aspect = *pixel_aspect;
                        Ok(())
                    }
                    // the runner hands the element its own solved output caps before the first frame
                    Caps::RawVideo {
                        format: RawVideoFormat::Nv12,
                        ..
                    } => {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
                        Ok(())
                    }
                    _ => Err(G2gError::CapsMismatch),
                },
                PipelinePacket::Eos => {
                    let held = self.reorder.flush();
                    self.reorder.reset();
                    self.emit(held, out).await?;
                    out.push(PipelinePacket::Eos).await?;
                    Ok(())
                }
                PipelinePacket::Flush => {
                    if let Some(decoder) = self.decoder.as_mut() {
                        decoder.reset();
                    }
                    self.reorder.reset();
                    self.input_stamps = PendingInputStamps::default();
                    self.recovery = DecodeErrorRecovery::default();
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                    Ok(())
                }
                PipelinePacket::Segment(segment) => {
                    self.segment = segment;
                    out.push(PipelinePacket::Segment(segment)).await?;
                    Ok(())
                }
                other => {
                    out.push(other).await?;
                    Ok(())
                }
            }
        })
    }
}

static H264_HARDWARE_DEC_PROPERTIES: &[PropertySpec] = &[PropertySpec::new(
    "low-latency",
    PropKind::Bool,
    "emit each picture as soon as its display position is certain, as gst's low-latency bump mode does",
)];
