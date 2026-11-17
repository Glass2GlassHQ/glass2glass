// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::audiofx::{passthrough_constraint, PcmScratch};

use g2g_core::log::{short_type_name, LogName};
use g2g_core::time::ns_to_ticks;
use g2g_core::{
    AsyncElement, AudioFormat, BusHandle, BusMessage, Caps, CapsConstraint, ConfigureOutcome,
    ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

/// # Example
///
/// ```no_run
/// use g2g_plugins::level::Level;
///
/// let level = Level::new();
/// ```
#[derive(Debug)]
pub struct Level {
    post_messages: bool,
    interval_ns: u64,
    format: AudioFormat,
    channels: usize,
    sample_rate: u32,
    scratch: PcmScratch,
    interval: IntervalSums,
    bus: Option<BusHandle>,
    log_name: LogName,
}

// per channel, over the samples of the interval so far
#[derive(Debug, Default)]
struct IntervalSums {
    peak: Vec<f64>,
    sum_of_squares: Vec<f64>,
    samples: u64,
    first_pts_ns: Option<u64>,
}

impl IntervalSums {
    fn new(channels: usize) -> Self {
        IntervalSums {
            peak: vec![0.0; channels],
            sum_of_squares: vec![0.0; channels],
            samples: 0,
            first_pts_ns: None,
        }
    }
}

impl Default for Level {
    fn default() -> Self {
        Self::new()
    }
}

// gst level's default interval
const DEFAULT_INTERVAL_NS: u64 = 100_000_000;

fn decibels(power: f64) -> f64 {
    if power <= 0.0 {
        return f64::NEG_INFINITY;
    }
    // 10 * log10(x) = 10 * log10(2) * log2(x)
    10.0 * core::f64::consts::LOG10_2 * crate::mathf::log2(power)
}

impl Level {
    pub fn new() -> Self {
        Self {
            post_messages: true,
            interval_ns: DEFAULT_INTERVAL_NS,
            format: AudioFormat::PcmS16Le,
            channels: 0,
            sample_rate: 0,
            scratch: PcmScratch::default(),
            interval: IntervalSums::default(),
            bus: None,
            log_name: LogName::new(),
        }
    }

    fn accept_input(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let Caps::Audio {
            format,
            channels,
            sample_rate,
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if *channels == 0 || *sample_rate == 0 {
            return Err(G2gError::CapsMismatch);
        }
        self.format = *format;
        self.channels = usize::from(*channels);
        self.sample_rate = *sample_rate;
        self.interval = IntervalSums::new(self.channels);
        Ok(())
    }

    fn interval_length(&self) -> u64 {
        ns_to_ticks(self.interval_ns, self.sample_rate).max(1)
    }

    fn completed_intervals(&mut self, bytes: &[u8], pts_ns: u64) -> Vec<(u64, Vec<f64>, Vec<f64>)> {
        let (channels, sample_rate) = (self.channels, self.sample_rate);
        let interval_length = self.interval_length();
        let samples = self.scratch.decode(bytes, self.format);
        let interval = &mut self.interval;
        let mut readings = Vec::new();
        for (index, frame) in samples.chunks_exact(channels).enumerate() {
            let at_ns =
                pts_ns.saturating_add(g2g_core::time::ticks_to_ns(index as u64, sample_rate));
            interval.first_pts_ns.get_or_insert(at_ns);
            for (channel, &sample) in frame.iter().enumerate() {
                let value = f64::from(sample);
                interval.peak[channel] = interval.peak[channel].max(value.abs());
                interval.sum_of_squares[channel] += value * value;
            }
            interval.samples += 1;
            if interval.samples == interval_length {
                let count = interval.samples as f64;
                let peak_db = interval
                    .peak
                    .iter()
                    .map(|peak| decibels(peak * peak))
                    .collect();
                let rms_db = interval
                    .sum_of_squares
                    .iter()
                    .map(|sum| decibels(sum / count))
                    .collect();
                let first_pts_ns = interval.first_pts_ns.expect("set by its first sample");
                readings.push((first_pts_ns, peak_db, rms_db));
                *interval = IntervalSums::new(channels);
            }
        }
        readings
    }

    fn element_name(&self) -> String {
        self.log_name
            .instance()
            .unwrap_or_else(|| short_type_name::<Self>())
            .to_string()
    }
}

impl AsyncElement for Level {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        passthrough_constraint(None)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.accept_input(absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn set_bus(&mut self, bus: BusHandle) {
        self.bus = Some(bus);
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if self.post_messages && self.bus.is_some() {
                        let bytes = frame
                            .domain
                            .require_system_slice(short_type_name::<Self>())?;
                        let readings = self.completed_intervals(bytes, frame.timing.pts_ns);
                        let element = self.element_name();
                        let bus = self.bus.as_ref().expect("checked above");
                        for (pts_ns, peak_db, rms_db) in readings {
                            bus.try_post(BusMessage::Level {
                                element: element.clone(),
                                pts_ns,
                                peak_db,
                                rms_db,
                            });
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.accept_input(&caps)?;
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                PipelinePacket::Flush => {
                    self.interval = IntervalSums::new(self.channels);
                    out.push(PipelinePacket::Flush).await?;
                }
                // The runner emits the final Eos after process(Eos) returns
                // (the transform contract), so do not forward it here.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        LEVEL_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Level",
            "Filter/Analyzer/Audio",
            "Posts each channel's peak and RMS level in dB on the bus once per interval",
            crate::AARON_BOXER,
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "post-messages" => self.post_messages = value.as_bool().ok_or(PropError::Type)?,
            "interval" => {
                let interval_ns = value.as_uint().ok_or(PropError::Type)?;
                if interval_ns == 0 {
                    return Err(PropError::Value);
                }
                self.interval_ns = interval_ns;
                self.interval = IntervalSums::new(self.channels);
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "post-messages" => Some(PropValue::Bool(self.post_messages)),
            "interval" => Some(PropValue::Uint(self.interval_ns)),
            _ => None,
        }
    }
}

static LEVEL_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "post-messages",
        PropKind::Bool,
        "post a level message on the bus for each interval",
    )
    .with_default("true"),
    PropertySpec::new(
        "interval",
        PropKind::Uint,
        "nanoseconds of audio each level message measures",
    )
    .with_default("100000000"),
];

impl PadTemplates for Level {
    fn pad_templates() -> Vec<PadTemplate> {
        crate::audiofx::pad_templates(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Bus, ChannelLayout};

    const RATE: u32 = 8_000;
    // a tenth of a second at RATE
    const INTERVAL_SAMPLES: usize = 800;

    fn stereo_s16(left: i16, right: i16, frames: usize) -> Vec<u8> {
        (0..frames)
            .flat_map(|_| left.to_le_bytes().into_iter().chain(right.to_le_bytes()))
            .collect()
    }

    fn configured_level() -> (Level, Bus) {
        let (bus, handle) = Bus::new(16);
        let mut level = Level::new();
        level.set_bus(handle);
        level
            .configure_pipeline(&Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: RATE,
                channel_layout: ChannelLayout::UNSPECIFIED,
            })
            .unwrap();
        (level, bus)
    }

    #[test]
    fn one_interval_reads_half_scale_as_minus_six_db_and_silence_as_minus_infinity() {
        let (mut level, _bus) = configured_level();
        let half_scale = i16::MAX / 2 + 1;
        let readings = level.completed_intervals(&stereo_s16(half_scale, 0, INTERVAL_SAMPLES), 0);
        let [(pts_ns, peak_db, rms_db)] = readings.as_slice() else {
            panic!("one interval gives one reading, got {}", readings.len());
        };
        assert_eq!(*pts_ns, 0);
        let half_scale_db = decibels(0.25);
        assert!((half_scale_db + 6.02).abs() < 0.01, "{half_scale_db}");
        assert!((peak_db[0] - half_scale_db).abs() < 1e-9);
        assert!((rms_db[0] - half_scale_db).abs() < 1e-9);
        assert_eq!(peak_db[1], f64::NEG_INFINITY);
        assert_eq!(rms_db[1], f64::NEG_INFINITY);
    }

    #[test]
    fn an_interval_spans_buffers_and_starts_at_its_first_sample() {
        let (mut level, _bus) = configured_level();
        let half = INTERVAL_SAMPLES / 2;
        let half_ns = g2g_core::time::ticks_to_ns(half as u64, RATE);
        assert!(level
            .completed_intervals(&stereo_s16(1000, 1000, half), 0)
            .is_empty());
        let readings =
            level.completed_intervals(&stereo_s16(1000, 1000, INTERVAL_SAMPLES), half_ns);
        let pts: Vec<u64> = readings.iter().map(|(pts_ns, _, _)| *pts_ns).collect();
        assert_eq!(pts, [0]);
        let readings =
            level.completed_intervals(&stereo_s16(1000, 1000, half), 2 * half_ns + half_ns);
        assert_eq!(readings[0].0, 2 * half_ns);
    }
}
