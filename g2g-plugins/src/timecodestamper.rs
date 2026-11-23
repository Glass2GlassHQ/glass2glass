// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Stamp SMPTE timecode on raw video without touching the frame's pixels.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::caps::{framerate_q16_from_fraction, framerate_q16_to_fraction};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, Frame, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Rate, Segment, TimecodeMeta,
};

const NANOS_PER_SECOND: u128 = core::time::Duration::from_secs(1).as_nanos();
const SECONDS_PER_MINUTE: u64 = 60;
const SECONDS_PER_HOUR: u64 = 3600;
const HOURS_PER_DAY: u64 = 24;
const MINUTES_PER_CYCLE: u64 = 10;

#[derive(Debug, Clone, Copy)]
struct TimecodeRate {
    q16: u32,
    numerator: u32,
    denominator: u32,
    nominal: u64,
    dropped: u64,
}

impl TimecodeRate {
    fn new(q16: u32, drop_frame: bool) -> Self {
        // Caps store Q16. Recover the exact standard ratios for frame indexing.
        let (numerator, denominator) = [24_000, 30_000, 48_000, 60_000, 120_000]
            .into_iter()
            .find(|&num| framerate_q16_from_fraction(num, 1001) == Some(q16))
            .map_or_else(|| framerate_q16_to_fraction(q16), |num| (num, 1001));
        let nominal = u64::from(numerator.div_ceil(denominator));
        let dropped = match (drop_frame, numerator, denominator) {
            (true, 30_000, 1001) => 2,
            (true, 60_000, 1001) => 4,
            _ => 0,
        };
        Self {
            q16,
            numerator,
            denominator,
            nominal,
            dropped,
        }
    }

    fn ten_minutes(self) -> u64 {
        self.nominal * SECONDS_PER_MINUTE * MINUTES_PER_CYCLE
            - self.dropped * (MINUTES_PER_CYCLE - 1)
    }

    fn frames_at(self, ns: u64) -> u64 {
        (u128::from(ns) * u128::from(self.numerator)
            / (u128::from(self.denominator) * NANOS_PER_SECOND)) as u64
    }

    fn index(self, tc: TimecodeMeta) -> Option<u64> {
        let (hours, minutes, seconds, frames) = (
            u64::from(tc.hours),
            u64::from(tc.minutes),
            u64::from(tc.seconds),
            u64::from(tc.frames),
        );
        if hours >= HOURS_PER_DAY
            || minutes >= SECONDS_PER_MINUTE
            || seconds >= SECONDS_PER_MINUTE
            || frames >= self.nominal
            || (seconds == 0 && !minutes.is_multiple_of(MINUTES_PER_CYCLE) && frames < self.dropped)
        {
            return None;
        }
        let minutes = hours * SECONDS_PER_MINUTE + minutes;
        let skipped = self.dropped * (minutes - minutes / MINUTES_PER_CYCLE);
        Some((minutes * SECONDS_PER_MINUTE + seconds) * self.nominal + frames - skipped)
    }

    fn label(self, index: u64, offset: i32) -> TimecodeMeta {
        let cycle = self.ten_minutes();
        let day = cycle * HOURS_PER_DAY * (SECONDS_PER_MINUTE / MINUTES_PER_CYCLE);
        let index = (i128::from(index) + i128::from(offset)).rem_euclid(i128::from(day)) as u64;
        let short_minute = self.nominal * SECONDS_PER_MINUTE - self.dropped;
        let minutes_skipped = index / cycle * (MINUTES_PER_CYCLE - 1)
            + (index % cycle).saturating_sub(self.dropped) / short_minute;
        let label = index + minutes_skipped * self.dropped;
        let seconds = label / self.nominal;
        TimecodeMeta {
            hours: (seconds / SECONDS_PER_HOUR) as u8,
            minutes: (seconds / SECONDS_PER_MINUTE % SECONDS_PER_MINUTE) as u8,
            seconds: (seconds % SECONDS_PER_MINUTE) as u8,
            frames: (label % self.nominal) as u16,
            drop_frame: self.dropped != 0,
            framerate_q16: Some(self.q16),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    Internal,
    Zero,
    LastKnown,
    LastKnownOrZero,
    RunningTime,
}

impl Source {
    const VALUES: [(Self, &'static str); 5] = [
        (Self::Internal, "internal"),
        (Self::Zero, "zero"),
        (Self::LastKnown, "last-known"),
        (Self::LastKnownOrZero, "last-known-or-zero"),
        (Self::RunningTime, "running-time"),
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Set {
    Never,
    Keep,
    Always,
}

impl Set {
    const VALUES: [(Self, &'static str); 3] = [
        (Self::Never, "never"),
        (Self::Keep, "keep"),
        (Self::Always, "always"),
    ];
}

#[derive(Debug)]
struct LastKnown {
    rate: TimecodeRate,
    index: u64,
    running_time: Option<u64>,
}

/// The `timecodestamper` raw-video transform.
#[derive(Debug)]
pub struct TimecodeStamper {
    rate: Option<TimecodeRate>,
    segment: Segment,
    next_internal: Option<u64>,
    resume_from_seek: bool,
    last_known: Option<LastKnown>,
    source: Source,
    set: Set,
    start: Option<TimecodeMeta>,
    offset: i32,
    drop_frame: bool,
    auto_resync: bool,
    timeout: u64,
}

impl Default for TimecodeStamper {
    fn default() -> Self {
        Self {
            rate: None,
            segment: Segment::new(),
            next_internal: None,
            resume_from_seek: false,
            last_known: None,
            source: Source::Internal,
            set: Set::Keep,
            start: None,
            offset: 0,
            drop_frame: true,
            auto_resync: true,
            timeout: u64::MAX,
        }
    }
}

impl TimecodeStamper {
    pub fn new() -> Self {
        Self::default()
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let Caps::RawVideo {
            framerate: Rate::Fixed(q16),
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if *q16 == 0 {
            return Err(G2gError::CapsMismatch);
        }
        let rate = TimecodeRate::new(*q16, self.drop_frame);
        if self.start.is_some_and(|start| rate.index(start).is_none()) {
            return Err(G2gError::CapsMismatch);
        }
        if let (Some(old), Some(index)) = (self.rate, &mut self.next_internal) {
            let scaled =
                u128::from(*index) * u128::from(old.denominator) * u128::from(rate.numerator)
                    / (u128::from(old.numerator) * u128::from(rate.denominator));
            *index = scaled.min(u128::from(u64::MAX)) as u64;
        }
        self.rate = Some(rate);
        Ok(())
    }

    fn stamp(&mut self, frame: &mut Frame, rate: TimecodeRate) {
        let running_time = frame
            .timing
            .pts()
            .and_then(|pts| self.segment.to_running_time(pts));
        let index = self.next_internal.unwrap_or_else(|| {
            let resume_from_seek = core::mem::take(&mut self.resume_from_seek);
            self.start
                .and_then(|start| rate.index(start))
                .unwrap_or_else(|| {
                    if resume_from_seek {
                        rate.frames_at(self.segment.position.max(self.segment.start))
                    } else {
                        0
                    }
                })
        });
        self.next_internal = Some(index.saturating_add(1));

        let upstream = frame.meta.get::<TimecodeMeta>().copied();
        let reference = upstream.and_then(|tc| {
            let q16 = tc.framerate_q16.filter(|&q16| q16 != 0).unwrap_or(rate.q16);
            let rate = TimecodeRate::new(q16, tc.drop_frame);
            Some(LastKnown {
                rate,
                index: rate.index(tc)?,
                running_time,
            })
        });
        match (&mut self.last_known, reference) {
            (last, Some(reference)) if self.auto_resync || last.is_none() => {
                *last = Some(reference)
            }
            (Some(last), _) => {
                if self.auto_resync
                    && self.timeout != u64::MAX
                    && running_time
                        .zip(last.running_time)
                        .is_some_and(|(now, then)| {
                            now.saturating_sub(then) >= self.timeout
                                || then.saturating_sub(now) > self.timeout
                        })
                {
                    self.last_known = None;
                } else {
                    last.index = last.index.saturating_add(1);
                }
            }
            _ => {}
        }

        if self.set == Set::Never || (self.set == Set::Keep && upstream.is_some()) {
            return;
        }
        let known = self.last_known.as_ref().map(|last| (last.rate, last.index));
        let selected = match self.source {
            Source::Internal => Some((rate, index)),
            Source::Zero => Some((rate, 0)),
            Source::LastKnown => known.or(Some((rate, index))),
            Source::LastKnownOrZero => known.or(Some((rate, 0))),
            Source::RunningTime => running_time.map(|ns| (rate, rate.frames_at(ns))),
        };
        if let Some((rate, index)) = selected {
            frame.meta.attach(rate.label(index, self.offset));
        } else if self.set == Set::Always {
            frame.meta.remove::<TimecodeMeta>();
        }
    }
}

impl AsyncElement for TimecodeStamper {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Timecode stamper",
            "Filter/Video",
            "Attaches SMPTE timecode to raw video frames",
            crate::AARON_BOXER,
        )
    }

    fn output_domain_follows_input(&self) -> bool {
        true
    }

    fn intercept_caps(&self, caps: &Caps) -> Result<Caps, G2gError> {
        if matches!(caps, Caps::RawVideo { framerate, .. } if *framerate != Rate::Fixed(0)) {
            Ok(caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(crate::fakemediasink::raw_video_caps())
    }

    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configure(caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        mut packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let rate = self.rate.ok_or(G2gError::NotConfigured)?;
            match &mut packet {
                PipelinePacket::DataFrame(frame) => self.stamp(frame, rate),
                PipelinePacket::CapsChanged(caps) => self.configure(caps)?,
                PipelinePacket::Segment(segment) => self.segment = *segment,
                PipelinePacket::Flush => {
                    self.segment = Segment::new();
                    self.next_internal = None;
                    self.last_known = None;
                    self.resume_from_seek = true;
                }
                PipelinePacket::Eos => return Ok(()),
                _ => {}
            }
            out.push(packet).await?;
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "source" => self.source = from_nick(&Source::VALUES, &value)?,
            "set" => self.set = from_nick(&Set::VALUES, &value)?,
            "auto-resync" => self.auto_resync = value.as_bool().ok_or(PropError::Type)?,
            "timeout" => self.timeout = value.as_uint().ok_or(PropError::Type)?,
            "drop-frame" => {
                let drop_frame = value.as_bool().ok_or(PropError::Type)?;
                if let Some(rate) = self.rate {
                    let rate = TimecodeRate::new(rate.q16, drop_frame);
                    if self.start.is_some_and(|tc| rate.index(tc).is_none()) {
                        return Err(PropError::Value);
                    }
                    self.rate = Some(rate);
                }
                self.drop_frame = drop_frame;
            }
            "timecode-offset" => {
                self.offset = i32::try_from(value.as_int().ok_or(PropError::Type)?)
                    .map_err(|_| PropError::Value)?;
            }
            "set-internal-timecode" => {
                let value = value.as_str().ok_or(PropError::Type)?;
                let start = if value.is_empty() {
                    None
                } else {
                    Some(parse_label(value).ok_or(PropError::Value)?)
                };
                if self
                    .rate
                    .zip(start)
                    .is_some_and(|(rate, tc)| rate.index(tc).is_none())
                {
                    return Err(PropError::Value);
                }
                self.start = start;
                self.next_internal = None;
                self.resume_from_seek = false;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        Some(match name {
            "source" => nick(&Source::VALUES, self.source),
            "set" => nick(&Set::VALUES, self.set),
            "auto-resync" => PropValue::Bool(self.auto_resync),
            "timeout" => PropValue::Uint(self.timeout),
            "drop-frame" => PropValue::Bool(self.drop_frame),
            "timecode-offset" => PropValue::Int(self.offset.into()),
            "set-internal-timecode" => PropValue::Str(
                self.start
                    .as_ref()
                    .map(crate::timeoverlay::format_timecode)
                    .unwrap_or_default(),
            ),
            _ => return None,
        })
    }
}

impl PadTemplates for TimecodeStamper {
    fn pad_templates() -> Vec<PadTemplate> {
        crate::pcm_passthrough::pad_templates(crate::fakemediasink::raw_video_caps())
    }
}

fn from_nick<T: Copy>(values: &[(T, &str)], value: &PropValue) -> Result<T, PropError> {
    let name = value.as_str().ok_or(PropError::Type)?;
    values
        .iter()
        .find_map(|&(value, nick)| (nick == name).then_some(value))
        .ok_or(PropError::Value)
}

fn nick<T: PartialEq>(values: &[(T, &str)], value: T) -> PropValue {
    let (_, name) = values
        .iter()
        .find(|(candidate, _)| *candidate == value)
        .expect("every variant has a nick");
    PropValue::Str((*name).into())
}

fn parse_label(value: &str) -> Option<TimecodeMeta> {
    let (hours, rest) = value.split_once(':')?;
    let (minutes, rest) = rest.split_once(':')?;
    let (seconds, frames) = rest.split_once([':', ';'])?;
    let (hours, minutes, seconds) = (
        hours.parse().ok()?,
        minutes.parse().ok()?,
        seconds.parse().ok()?,
    );
    if hours >= 24 || minutes >= 60 || seconds >= 60 {
        return None;
    }
    Some(TimecodeMeta {
        hours,
        minutes,
        seconds,
        frames: frames.parse().ok()?,
        drop_frame: rest.contains(';'),
        framerate_q16: None,
    })
}

static PROPERTIES: &[PropertySpec] = &[
    PropertySpec::new("source", PropKind::Str, "timecode source")
        .with_enum_values("internal | zero | last-known | last-known-or-zero | running-time")
        .with_default("internal"),
    PropertySpec::new("set", PropKind::Str, "how to handle upstream timecode")
        .with_enum_values("never | keep | always")
        .with_default("keep"),
    PropertySpec::new(
        "drop-frame",
        PropKind::Bool,
        "use drop-frame numbering at 29.97 or 59.94 fps",
    )
    .with_default("true"),
    PropertySpec::new(
        "auto-resync",
        PropKind::Bool,
        "re-anchor the last-known counter to new upstream timecode",
    )
    .with_default("true"),
    PropertySpec::new(
        "timeout",
        PropKind::Uint,
        "upstream timecode timeout in nanoseconds",
    )
    .with_default("18446744073709551615"),
    PropertySpec::new(
        "set-internal-timecode",
        PropKind::Str,
        "initial counter label as HH:MM:SS:FF or HH:MM:SS;FF",
    )
    .with_default(""),
    PropertySpec::new(
        "timecode-offset",
        PropKind::Int,
        "signed frame offset applied to generated timecode",
    )
    .with_default("0"),
];
