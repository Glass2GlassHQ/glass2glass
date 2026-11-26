// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::segment::Segment;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming,
    G2gError, OutputSink, PadCaps, PadDirection, PadTemplate, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

use crate::mpegcrc::crc16_update;

const HEADER_LENGTH: usize = 62;
// the header checksum covers everything before the two checksum fields
const HEADER_CHECKSUM_OFFSET: usize = 58;
const VERSION: [u8; 2] = [1, 0];
const HEADER_CRC_FLAG: u8 = 1 << 0;
const PAYLOAD_CRC_FLAG: u8 = 1 << 1;
const CRC16_POLYNOMIAL: u16 = 0x1021;

const BUFFER_PAYLOAD: u16 = 1;
const CAPS_PAYLOAD: u16 = 2;
// gst event types are `number << 8 | flags`, and GDP adds 64 to them
const EVENT_PAYLOAD_BASE: u16 = 64;
const STREAM_START_EVENT: u16 = EVENT_PAYLOAD_BASE + (40 << 8 | 0b1110);
const SEGMENT_EVENT: u16 = EVENT_PAYLOAD_BASE + (70 << 8 | 0b1110);

const CLOCK_TIME_NONE: u64 = u64::MAX;
const BUFFER_FLAG_DELTA_UNIT: u16 = 1 << 13;

// crafted headers announce payloads of up to 4 GiB
const MAX_PAYLOAD_LENGTH: usize = 256 << 20;

const STREAM_ID: &str = "g2g-gdppay";
const TRICKMODE_KEY_UNITS_FLAG: &str = "GST_SEGMENT_FLAG_TRICKMODE_KEY_UNITS";

fn crc16(bytes: &[u8]) -> u16 {
    if bytes.is_empty() {
        return 0;
    }
    !bytes.iter().fold(0xFFFF, |crc, &byte| {
        crc16_update(CRC16_POLYNOMIAL, crc, byte)
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Checksums {
    header: bool,
    payload: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BufferFields {
    pts: u64,
    duration: u64,
    offset: u64,
    offset_end: u64,
    flags: u16,
    dts: u64,
}

const EVENT_FIELDS: BufferFields = BufferFields {
    pts: CLOCK_TIME_NONE,
    duration: 0,
    offset: 0,
    offset_end: 0,
    flags: 0,
    dts: 0,
};

fn packet(
    payload_type: u16,
    fields: BufferFields,
    payload: &[u8],
    checksums: Checksums,
) -> Result<Vec<u8>, G2gError> {
    let length = u32::try_from(payload.len()).map_err(|_| G2gError::CapsMismatch)?;
    let mut bytes = Vec::with_capacity(HEADER_LENGTH + payload.len());
    bytes.extend_from_slice(&VERSION);
    let header_flag = if checksums.header { HEADER_CRC_FLAG } else { 0 };
    let payload_flag = if checksums.payload {
        PAYLOAD_CRC_FLAG
    } else {
        0
    };
    let flags = header_flag | payload_flag;
    bytes.extend_from_slice(&[flags, 0]);
    bytes.extend_from_slice(&payload_type.to_be_bytes());
    bytes.extend_from_slice(&length.to_be_bytes());
    for value in [
        fields.pts,
        fields.duration,
        fields.offset,
        fields.offset_end,
    ] {
        bytes.extend_from_slice(&value.to_be_bytes());
    }
    bytes.extend_from_slice(&fields.flags.to_be_bytes());
    bytes.extend_from_slice(&fields.dts.to_be_bytes());
    bytes.resize(HEADER_CHECKSUM_OFFSET, 0);
    let header_crc = if checksums.header { crc16(&bytes) } else { 0 };
    let payload_crc = if checksums.payload { crc16(payload) } else { 0 };
    bytes.extend_from_slice(&header_crc.to_be_bytes());
    bytes.extend_from_slice(&payload_crc.to_be_bytes());
    bytes.extend_from_slice(payload);
    Ok(bytes)
}

fn string_payload(text: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(text.len() + 1);
    bytes.extend_from_slice(text.as_bytes());
    // gst reads caps and event payloads as nul-terminated strings
    bytes.push(0);
    bytes
}

fn segment_event_payload(segment: &Segment) -> Result<String, G2gError> {
    // gst segment times are unsigned
    let time = u64::try_from(segment.time).map_err(|_| G2gError::CapsMismatch)?;
    let flags = if segment.key_units_only {
        TRICKMODE_KEY_UNITS_FLAG
    } else {
        "GST_SEGMENT_FLAG_NONE"
    };
    Ok(format!(
        "GstEventSegment, segment=(GstSegment)\"segment, flags=(GstSegmentFlags){flags}, rate=(double){}, applied-rate=(double){}, format=(GstFormat)time, base=(guint64){}, offset=(guint64)0, start=(guint64){}, stop=(guint64){}, time=(guint64){time}, position=(guint64){}, duration=(guint64){CLOCK_TIME_NONE};\";",
        segment.rate,
        segment.applied_rate,
        segment.base,
        segment.start,
        segment.stop.unwrap_or(CLOCK_TIME_NONE),
        segment.position,
    ))
}

fn segment_from_event_payload(text: &str) -> Result<Option<Segment>, G2gError> {
    let quoted = text
        .split_once("segment=(GstSegment)\"")
        .and_then(|(_, rest)| rest.rsplit_once('"'))
        .map(|(inner, _)| inner)
        .ok_or(G2gError::CapsMismatch)?;
    let mut inner = String::with_capacity(quoted.len());
    let mut chars = quoted.chars();
    while let Some(c) = chars.next() {
        inner.push(if c == '\\' {
            chars.next().ok_or(G2gError::CapsMismatch)?
        } else {
            c
        });
    }
    let mut fields = Vec::new();
    for field in inner.trim_end().trim_end_matches(';').split(',').skip(1) {
        let (key, value) = field.split_once('=').ok_or(G2gError::CapsMismatch)?;
        let value = value
            .trim()
            .strip_prefix('(')
            .and_then(|typed| typed.split_once(')'))
            .map_or(value.trim(), |(_, value)| value);
        fields.push((key.trim(), value));
    }
    let field = |key: &str| {
        fields
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| *value)
            .ok_or(G2gError::CapsMismatch)
    };
    let number = |key: &str| -> Result<u64, G2gError> {
        field(key)?.parse().map_err(|_| G2gError::CapsMismatch)
    };
    let rate = |key: &str| -> Result<f64, G2gError> {
        field(key)?
            .parse::<f64>()
            .ok()
            .filter(|rate| rate.is_finite() && *rate != 0.0)
            .ok_or(G2gError::CapsMismatch)
    };
    // g2g segments are in time only
    if field("format")? != "time" {
        return Ok(None);
    }
    let stop = number("stop")?;
    Ok(Some(Segment {
        rate: rate("rate")?,
        applied_rate: rate("applied-rate")?,
        base: number("base")?,
        start: number("start")?,
        stop: (stop != CLOCK_TIME_NONE).then_some(stop),
        time: i64::try_from(number("time")?).map_err(|_| G2gError::CapsMismatch)?,
        position: number("position")?,
        key_units_only: field("flags")?
            .split('+')
            .any(|flag| flag == TRICKMODE_KEY_UNITS_FLAG),
    }))
}

const UNTIMED: FrameTiming = FrameTiming {
    pts_ns: FrameTiming::PTS_NONE,
    dts_ns: FrameTiming::DTS_NONE,
    duration_ns: 0,
    capture_ns: 0,
    arrival_ns: 0,
    keyframe: false,
    start_discard_ns: 0,
    end_discard_ns: 0,
};

fn gdp_caps(original: Option<&Caps>) -> Caps {
    Caps::Gdp {
        original: original.map(|original| Box::new(original.clone())),
    }
}

static PAY_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "crc-header",
        PropKind::Bool,
        "calculate and store a CRC checksum on the header",
    ),
    PropertySpec::new(
        "crc-payload",
        PropKind::Bool,
        "calculate and store a CRC checksum on the payload",
    ),
];

#[derive(Debug)]
pub struct GdpPay {
    checksums: Checksums,
    input: Option<Caps>,
    stream_started: bool,
    caps_in_stream: Option<Caps>,
    segment_in_stream: bool,
    last_output_caps: Option<Caps>,
    emitted: u64,
}

impl Default for GdpPay {
    fn default() -> Self {
        Self::new()
    }
}

impl GdpPay {
    pub fn new() -> Self {
        Self {
            checksums: Checksums {
                header: true,
                payload: false,
            },
            input: None,
            stream_started: false,
            caps_in_stream: None,
            segment_in_stream: false,
            last_output_caps: None,
            emitted: 0,
        }
    }

    async fn emit(
        &mut self,
        bytes: Vec<u8>,
        timing: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let caps = gdp_caps(self.input.as_ref());
        if self.last_output_caps.as_ref() != Some(&caps) {
            out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
            self.last_output_caps = Some(caps);
        }
        let frame = Frame::from_system_bytes(bytes, timing, self.emitted);
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    async fn emit_event(
        &mut self,
        event_type: u16,
        text: &str,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let bytes = packet(
            event_type,
            EVENT_FIELDS,
            &string_payload(text),
            self.checksums,
        )?;
        self.emit(bytes, UNTIMED, out).await
    }

    async fn emit_stream_header(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        // gst streams open with stream-start, then caps, then segment
        if !self.stream_started {
            let text = format!(
                "GstEventStreamStart, stream-id=(string){STREAM_ID}, flags=(GstStreamFlags)GST_STREAM_FLAG_NONE;"
            );
            self.emit_event(STREAM_START_EVENT, &text, out).await?;
            self.stream_started = true;
        }
        let input = self.input.clone().ok_or(G2gError::NotConfigured)?;
        if self.caps_in_stream.as_ref() != Some(&input) {
            let payload = string_payload(&input.to_gst_string());
            let bytes = packet(
                CAPS_PAYLOAD,
                BufferFields::default(),
                &payload,
                self.checksums,
            )?;
            self.emit(bytes, UNTIMED, out).await?;
            self.caps_in_stream = Some(input);
        }
        Ok(())
    }

    async fn emit_segment(
        &mut self,
        segment: &Segment,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        self.emit_stream_header(out).await?;
        let text = segment_event_payload(segment)?;
        self.emit_event(SEGMENT_EVENT, &text, out).await?;
        self.segment_in_stream = true;
        Ok(())
    }
}

impl AsyncElement for GdpPay {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "GDP Payloader",
            "GDP/Payloader",
            "Payloads GStreamer Data Protocol buffers",
            crate::AARON_BOXER,
        )
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::SystemView)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| CapsSet::one(gdp_caps(Some(input)))))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PAY_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let enabled = || value.as_bool().ok_or(PropError::Type);
        match name {
            "crc-header" => self.checksums.header = enabled()?,
            "crc-payload" => self.checksums.payload = enabled()?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "crc-header" => Some(PropValue::Bool(self.checksums.header)),
            "crc-payload" => Some(PropValue::Bool(self.checksums.payload)),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet_in: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet_in {
                PipelinePacket::DataFrame(frame) => {
                    self.emit_stream_header(out).await?;
                    if !self.segment_in_stream {
                        self.emit_segment(&Segment::new(), out).await?;
                    }
                    let timing = frame.timing;
                    let fields = BufferFields {
                        pts: timing.pts_ns,
                        duration: match timing.duration_ns {
                            0 => CLOCK_TIME_NONE,
                            duration => duration,
                        },
                        offset: CLOCK_TIME_NONE,
                        offset_end: CLOCK_TIME_NONE,
                        flags: if timing.keyframe {
                            0
                        } else {
                            BUFFER_FLAG_DELTA_UNIT
                        },
                        dts: timing.dts_ns,
                    };
                    let bytes = frame.domain.require_system_bytes("gdppay")?;
                    let bytes = packet(BUFFER_PAYLOAD, fields, &bytes, self.checksums)?;
                    self.emit(bytes, timing, out).await?;
                }
                // the runner also sends an element its own solved output caps
                PipelinePacket::CapsChanged(caps @ Caps::Gdp { .. }) => {
                    out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                    self.last_output_caps = Some(caps);
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.input = Some(caps);
                    self.emit_stream_header(out).await?;
                }
                PipelinePacket::Segment(segment) => self.emit_segment(&segment, out).await?,
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

pub fn pay_pad_templates() -> Vec<PadTemplate> {
    Vec::from([
        PadTemplate::sink_any(),
        PadTemplate::source(CapsSet::one(gdp_caps(None))),
    ])
}

struct PacketHeader {
    payload_type: u16,
    payload_length: usize,
    payload_crc: Option<u16>,
    fields: BufferFields,
}

fn parse_header(header: &[u8; HEADER_LENGTH]) -> Result<PacketHeader, G2gError> {
    let u16_at = |at: usize| u16::from_be_bytes([header[at], header[at + 1]]);
    let u64_at = |at: usize| {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(&header[at..at + 8]);
        u64::from_be_bytes(bytes)
    };
    if header[..2] != VERSION[..] {
        return Err(G2gError::CapsMismatch);
    }
    let flags = header[2];
    if flags & HEADER_CRC_FLAG != 0
        && crc16(&header[..HEADER_CHECKSUM_OFFSET]) != u16_at(HEADER_CHECKSUM_OFFSET)
    {
        return Err(G2gError::CapsMismatch);
    }
    let payload_length = u32::from_be_bytes([header[6], header[7], header[8], header[9]]) as usize;
    if payload_length > MAX_PAYLOAD_LENGTH {
        return Err(G2gError::CapsMismatch);
    }
    Ok(PacketHeader {
        payload_type: u16_at(4),
        payload_length,
        payload_crc: (flags & PAYLOAD_CRC_FLAG != 0).then(|| u16_at(HEADER_CHECKSUM_OFFSET + 2)),
        fields: BufferFields {
            pts: u64_at(10),
            duration: u64_at(18),
            offset: u64_at(26),
            offset_end: u64_at(34),
            flags: u16_at(42),
            dts: u64_at(44),
        },
    })
}

fn payload_text(payload: &[u8]) -> Result<&str, G2gError> {
    let text = payload.strip_suffix(&[0]).unwrap_or(payload);
    core::str::from_utf8(text).map_err(|_| G2gError::CapsMismatch)
}

fn caps_from_payload(payload: &[u8]) -> Result<Caps, G2gError> {
    let set = CapsSet::from_gst_string(payload_text(payload)?).ok_or(G2gError::CapsMismatch)?;
    match set.alternatives() {
        [caps] if caps.is_fixed() => Ok(caps.clone()),
        _ => Err(G2gError::CapsMismatch),
    }
}

pub fn sniff_original_caps(mut bytes: &[u8]) -> Option<Caps> {
    while let Some((header, rest)) = bytes.split_first_chunk::<HEADER_LENGTH>() {
        let header = parse_header(header).ok()?;
        let payload = rest.get(..header.payload_length)?;
        match header.payload_type {
            CAPS_PAYLOAD => return caps_from_payload(payload).ok(),
            payload_type if payload_type >= EVENT_PAYLOAD_BASE => {}
            _ => return None,
        }
        bytes = &rest[header.payload_length..];
    }
    None
}

static DEPAY_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "ts-offset",
    PropKind::Int,
    "nanoseconds added to each buffer's timestamps, which stop at 0",
)];

#[derive(Debug, Default)]
pub struct GdpDepay {
    ts_offset: i64,
    pending: Vec<u8>,
    stream_caps: Option<Caps>,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl GdpDepay {
    pub fn new() -> Self {
        Self::default()
    }

    fn offset_time(&self, time: u64) -> Result<u64, G2gError> {
        if time == CLOCK_TIME_NONE {
            return Ok(time);
        }
        // gst moves times the offset would make negative to zero
        let shifted = i128::from(time) + i128::from(self.ts_offset);
        u64::try_from(shifted.max(0))
            .ok()
            .filter(|shifted| *shifted != CLOCK_TIME_NONE)
            .ok_or(G2gError::CapsMismatch)
    }

    async fn handle_packet(
        &mut self,
        header: &PacketHeader,
        payload: &[u8],
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if header.payload_crc.is_some_and(|crc| crc != crc16(payload)) {
            return Err(G2gError::CapsMismatch);
        }
        match header.payload_type {
            BUFFER_PAYLOAD => {
                let caps = self.stream_caps.clone().ok_or(G2gError::CapsMismatch)?;
                if self.last_caps.as_ref() != Some(&caps) {
                    out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                    self.last_caps = Some(caps);
                }
                let fields = header.fields;
                let timing = FrameTiming {
                    pts_ns: self.offset_time(fields.pts)?,
                    dts_ns: self.offset_time(fields.dts)?,
                    duration_ns: match fields.duration {
                        CLOCK_TIME_NONE => 0,
                        duration => duration,
                    },
                    keyframe: fields.flags & BUFFER_FLAG_DELTA_UNIT == 0,
                    ..FrameTiming::default()
                };
                let frame = Frame::from_system_bytes(payload.to_vec(), timing, self.emitted);
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            CAPS_PAYLOAD => self.stream_caps = Some(caps_from_payload(payload)?),
            SEGMENT_EVENT => {
                if let Some(segment) = segment_from_event_payload(payload_text(payload)?)? {
                    out.push(PipelinePacket::Segment(segment)).await?;
                }
            }
            // g2g streams have no counterpart for the other gst events
            payload_type if payload_type >= EVENT_PAYLOAD_BASE => {}
            _ => return Err(G2gError::CapsMismatch),
        }
        Ok(())
    }
}

impl AsyncElement for GdpDepay {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "GDP Depayloader",
            "GDP/Depayloader",
            "Depayloads GStreamer Data Protocol buffers",
            crate::AARON_BOXER,
        )
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::SystemView)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&gdp_caps(None))
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(depayloaded_caps))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Gdp { original: Some(_) } => Ok(ConfigureOutcome::Accepted),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DEPAY_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "ts-offset" => {
                self.ts_offset = value.as_int().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "ts-offset" => Some(PropValue::Int(self.ts_offset)),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet_in: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet_in {
                PipelinePacket::DataFrame(frame) => {
                    self.pending
                        .extend_from_slice(&frame.domain.require_system_bytes("gdpdepay")?);
                    let pending = core::mem::take(&mut self.pending);
                    let mut rest = pending.as_slice();
                    while let Some((header, after_header)) =
                        rest.split_first_chunk::<HEADER_LENGTH>()
                    {
                        let header = parse_header(header)?;
                        let Some(payload) = after_header.get(..header.payload_length) else {
                            break;
                        };
                        self.handle_packet(&header, payload, out).await?;
                        rest = &after_header[header.payload_length..];
                    }
                    self.pending = rest.to_vec();
                }
                // the runner also sends an element its own solved output caps
                PipelinePacket::CapsChanged(Caps::Gdp { .. }) => {}
                PipelinePacket::CapsChanged(caps) => {
                    out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                    self.last_caps = Some(caps);
                }
                PipelinePacket::Flush => {
                    self.pending.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos if !self.pending.is_empty() => {
                    return Err(G2gError::CapsMismatch);
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

pub fn depayloaded_caps(input: &Caps) -> CapsSet {
    match input {
        Caps::Gdp {
            original: Some(original),
        } => CapsSet::one((**original).clone()),
        _ => CapsSet::from_alternatives(Vec::new()),
    }
}

pub fn depay_pad_templates() -> Vec<PadTemplate> {
    Vec::from([
        PadTemplate::sink(CapsSet::one(gdp_caps(None))),
        PadTemplate {
            direction: PadDirection::Source,
            caps: PadCaps::Any,
        },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_matches_the_check_value() {
        assert_eq!(crc16(b"123456789"), 0xD64E);
        assert_eq!(crc16(&[]), 0);
    }

    #[test]
    fn segments_round_trip_through_the_gst_event_string() {
        let segment = Segment {
            rate: -0.5,
            applied_rate: 1.0,
            base: 7,
            start: 1_000,
            stop: Some(9_000),
            time: 500,
            position: 2_000,
            key_units_only: true,
        };
        let text = segment_event_payload(&segment).expect("time is positive");
        assert_eq!(segment_from_event_payload(&text), Ok(Some(segment)));
    }

    #[test]
    fn gst_segment_events_parse() {
        let text = "GstEventSegment, segment=(GstSegment)\"segment, flags=(GstSegmentFlags)GST_SEGMENT_FLAG_NONE, rate=(double)1, applied-rate=(double)1, format=(GstFormat)time, base=(guint64)0, offset=(guint64)0, start=(guint64)0, stop=(guint64)18446744073709551615, time=(guint64)0, position=(guint64)0, duration=(guint64)18446744073709551615;\";";
        assert_eq!(segment_from_event_payload(text), Ok(Some(Segment::new())));
        let bytes = text.replace("(GstFormat)time", "(GstFormat)bytes");
        assert_eq!(segment_from_event_payload(&bytes), Ok(None));
    }
}
