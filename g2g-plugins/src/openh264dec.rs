use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_warn, AsyncElement, BitstreamAlignment, Caps, CapsConstraint, CapsSet, Colorimetry,
    ConfigureOutcome, Dim, ElementMetadata, G2gError, HardwareError, Interlace, MemoryDomain,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PixelAspect, Rate, RawVideoFormat,
    Segment, VideoCodec,
};

use openh264::decoder::{DecodedYUV, Decoder, DecoderConfig, Flush};
use openh264::formats::YUVSource;
use openh264::OpenH264API;

use crate::inputstamp::{InputStamp, PendingInputStamps};

struct DecodedPicture {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

fn pack_i420(yuv: &DecodedYUV<'_>) -> Result<DecodedPicture, G2gError> {
    let (width, height) = yuv.dimensions();
    let width = u32::try_from(width).map_err(|_| G2gError::CapsMismatch)?;
    let height = u32::try_from(height).map_err(|_| G2gError::CapsMismatch)?;
    let (y_stride, u_stride, v_stride) = yuv.strides();
    let format = RawVideoFormat::I420;
    let mut pixels = Vec::new();
    for (plane, source, stride) in [
        (0, yuv.y(), y_stride),
        (1, yuv.u(), u_stride),
        (2, yuv.v(), v_stride),
    ] {
        let row_bytes = format
            .plane_stride(plane, width)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(G2gError::CapsMismatch)?;
        let rows = format
            .plane_rows(plane, height)
            .and_then(|rows| usize::try_from(rows).ok())
            .ok_or(G2gError::CapsMismatch)?;
        for row in 0..rows {
            let start = row.checked_mul(stride).ok_or(G2gError::CapsMismatch)?;
            let end = start.checked_add(row_bytes).ok_or(G2gError::CapsMismatch)?;
            pixels.extend_from_slice(source.get(start..end).ok_or(G2gError::CapsMismatch)?);
        }
    }
    Ok(DecodedPicture {
        width,
        height,
        pixels,
    })
}

fn open_decoder() -> Result<Decoder, G2gError> {
    let api = OpenH264API::from_source();
    // openh264's default flushes after every access unit, which hands a B-frame stream out of order
    let config = DecoderConfig::new().flush_after_decode(Flush::NoFlush);
    Decoder::with_api_config(api, config).map_err(|_| G2gError::Hardware(HardwareError::Other))
}

fn sequence_parameter_set(access_unit: &[u8]) -> Option<&[u8]> {
    const NAL_TYPE_SPS: u8 = 7;
    crate::annexb::nal_units_any(access_unit).find(|nal| {
        nal.first()
            .is_some_and(|header| header & 0x1F == NAL_TYPE_SPS)
    })
}

fn output_caps(
    width: Dim,
    height: Dim,
    framerate: Rate,
    colorimetry: Colorimetry,
    pixel_aspect: PixelAspect,
) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::I420,
        width,
        height,
        framerate,
        interlace: Interlace::Progressive,
        colorimetry,
        pixel_aspect,
    }
}

pub struct OpenH264Dec {
    decoder: Option<Decoder>,
    framerate: Rate,
    colorimetry: Colorimetry,
    pixel_aspect: PixelAspect,
    emitted_caps: Option<Caps>,
    last_sps: Option<Vec<u8>>,
    // openh264 hands pictures out in presentation order, so each takes the earliest pts still held
    pending_pts: BTreeSet<u64>,
    input_stamps: PendingInputStamps,
    last_timing: FrameTiming,
    segment: Segment,
    sequence: u64,
    configured: bool,
}

impl core::fmt::Debug for OpenH264Dec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OpenH264Dec")
            .field("emitted_caps", &self.emitted_caps)
            .field("pending", &self.pending_pts.len())
            .field("sequence", &self.sequence)
            .field("configured", &self.configured)
            .finish_non_exhaustive()
    }
}

impl Default for OpenH264Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenH264Dec {
    pub fn new() -> Self {
        Self {
            decoder: None,
            framerate: Rate::Any,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: PixelAspect::Any,
            emitted_caps: None,
            last_sps: None,
            pending_pts: BTreeSet::new(),
            input_stamps: PendingInputStamps::default(),
            last_timing: FrameTiming::default(),
            segment: Segment::new(),
            sequence: 0,
            configured: false,
        }
    }

    fn input_template() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: BitstreamAlignment::AccessUnit,
        }
    }

    fn decode(&mut self, access_unit: &[u8], pts_ns: u64) -> Result<Vec<DecodedPicture>, G2gError> {
        let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
        match decoder.decode(access_unit) {
            Ok(Some(yuv)) => Ok(Vec::from([pack_i420(&yuv)?])),
            Ok(None) => Ok(Vec::new()),
            // the ffmpeg CLI logs an access unit its decoder rejects and decodes on
            Err(error) => {
                g2g_warn!(
                    Target::category(short_type_name::<Self>()),
                    "skipped a {}-byte access unit openh264 rejects: {error}",
                    access_unit.len()
                );
                self.pending_pts.remove(&pts_ns);
                Ok(Vec::new())
            }
        }
    }

    fn drain(&mut self) -> Result<Vec<DecodedPicture>, G2gError> {
        let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
        let remaining = decoder
            .flush_remaining()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        remaining.iter().map(pack_i420).collect()
    }

    // openh264 fails the B-frames that follow a flush, so the next pictures come from a new decoder
    fn drain_and_reopen(&mut self) -> Result<Vec<DecodedPicture>, G2gError> {
        let pictures = self.drain()?;
        self.decoder = Some(open_decoder()?);
        Ok(pictures)
    }

    async fn emit(
        &mut self,
        pictures: Vec<DecodedPicture>,
        timing: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        for picture in pictures {
            let pts_ns = self
                .pending_pts
                .pop_first()
                .unwrap_or(FrameTiming::PTS_NONE);
            let input = self.input_stamps.take(pts_ns);
            if pts_ns != FrameTiming::PTS_NONE && !self.segment.presents(pts_ns) {
                continue;
            }
            let caps = output_caps(
                Dim::Fixed(picture.width),
                Dim::Fixed(picture.height),
                self.framerate.clone(),
                self.colorimetry,
                self.pixel_aspect,
            );
            if self.emitted_caps.as_ref() != Some(&caps) {
                out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                self.emitted_caps = Some(caps);
            }
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(
                    picture.pixels.into_boxed_slice(),
                )),
                timing: FrameTiming {
                    pts_ns,
                    dts_ns: pts_ns,
                    arrival_ns: input.arrival_ns,
                    ..timing
                },
                sequence: self.sequence,
                meta: input.meta,
            };
            self.sequence += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for OpenH264Dec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_template())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width,
                height,
                framerate,
                colorimetry,
                pixel_aspect,
                ..
            } => CapsSet::one(output_caps(
                width.clone(),
                height.clone(),
                framerate.clone(),
                *colorimetry,
                *pixel_aspect,
            )),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::CompressedVideo {
            codec: VideoCodec::H264,
            framerate,
            colorimetry,
            pixel_aspect,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        self.framerate = framerate.clone();
        self.colorimetry = *colorimetry;
        self.pixel_aspect = *pixel_aspect;
        if self.decoder.is_none() {
            self.decoder = Some(open_decoder()?);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "OpenH264 video decoder",
            "Codec/Decoder/Video",
            "Decodes H.264 to I420 with OpenH264",
            "g2g",
        )
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
                    let access_unit = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    let pts_ns = frame.timing.pts_ns;
                    if pts_ns != FrameTiming::PTS_NONE {
                        self.pending_pts.insert(pts_ns);
                        self.input_stamps.insert(pts_ns, InputStamp::of(&frame));
                    }
                    let sps = sequence_parameter_set(access_unit);
                    // openh264 drops the picture it holds back when a different SPS starts a new sequence
                    let new_sequence = sps
                        .zip(self.last_sps.as_deref())
                        .is_some_and(|(sps, last)| sps != last);
                    let mut pictures = if new_sequence {
                        self.drain_and_reopen()?
                    } else {
                        Vec::new()
                    };
                    if let Some(sps) = sps {
                        self.last_sps = Some(sps.to_vec());
                    }
                    pictures.extend(self.decode(access_unit, pts_ns)?);
                    self.last_timing = frame.timing;
                    self.emit(pictures, frame.timing, out).await?;
                }
                PipelinePacket::CapsChanged(Caps::CompressedVideo {
                    colorimetry,
                    pixel_aspect,
                    ..
                }) => {
                    self.colorimetry = colorimetry;
                    self.pixel_aspect = pixel_aspect;
                }
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Segment(segment) => {
                    self.segment = segment;
                    out.push(PipelinePacket::Segment(segment)).await?;
                }
                // a B-frame stream holds its last pictures back until the end
                PipelinePacket::Eos => {
                    let pictures = self.drain()?;
                    let timing = self.last_timing;
                    self.emit(pictures, timing, out).await?;
                }
                PipelinePacket::Flush => {
                    self.drain_and_reopen()?;
                    self.last_sps = None;
                    self.pending_pts.clear();
                    self.input_stamps = PendingInputStamps::default();
                    out.push(PipelinePacket::Flush).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for OpenH264Dec {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_template())),
            PadTemplate::source(CapsSet::one(output_caps(
                Dim::Any,
                Dim::Any,
                Rate::Any,
                Colorimetry::UNKNOWN,
                PixelAspect::Any,
            ))),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h264parse::H264Parse;
    use crate::testutil::{data_bytes, run, run_packets};

    fn h264(alignment: BitstreamAlignment) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment,
        }
    }

    fn decode(annexb: &[u8]) -> Vec<PipelinePacket> {
        let parsed = run(
            &mut H264Parse::reframing(),
            &h264(BitstreamAlignment::Unspecified),
            &[annexb],
        );
        let access_units = parsed
            .packets
            .into_iter()
            .filter(|packet| !matches!(packet, PipelinePacket::Eos));
        let decoded = run_packets(
            &mut OpenH264Dec::new(),
            &h264(BitstreamAlignment::AccessUnit),
            access_units,
        );
        decoded.packets
    }

    #[test]
    fn a_b_frame_stream_matches_ffmpeg() {
        let packets = decode(include_bytes!("../tests/fixtures/h264_64x48_bframes.h264"));
        assert_eq!(
            data_bytes(&packets),
            include_bytes!("../tests/fixtures/h264_64x48_bframes.h264.i420"),
        );
    }

    #[test]
    fn a_resolution_change_keeps_every_picture() {
        let packets = decode(include_bytes!(
            "../tests/fixtures/h264_reconfig_640x480_to_320x240.h264"
        ));
        let mut pictures_per_geometry: Vec<((Dim, Dim), usize)> = Vec::new();
        for packet in &packets {
            match packet {
                PipelinePacket::CapsChanged(Caps::RawVideo { width, height, .. }) => {
                    pictures_per_geometry.push(((width.clone(), height.clone()), 0));
                }
                PipelinePacket::DataFrame(_) => {
                    pictures_per_geometry
                        .last_mut()
                        .expect("caps before the first picture")
                        .1 += 1;
                }
                _ => {}
            }
        }
        assert_eq!(
            pictures_per_geometry,
            [
                ((Dim::Fixed(640), Dim::Fixed(480)), 6),
                ((Dim::Fixed(320), Dim::Fixed(240)), 6),
            ]
        );
    }
}
