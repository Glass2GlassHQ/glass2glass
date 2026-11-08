use alloc::sync::Arc;
use alloc::vec::Vec;

use g2g_core::{
    Caps, DriftClock, G2gError, PipelineClock, PropError, PropKind, PropValue, PropertySpec,
};

use crate::audioconvert::{frame_bytes, ns_to_samples, silence_byte};

// gst audiobasesink drift-tolerance default, in microseconds
const DEFAULT_DRIFT_TOLERANCE_US: u64 = 40_000;
const NANOS_PER_MICROSECOND: u64 = 1_000;
// gst audiobasesink's moving average weighs the newest skew 1/32
const SKEW_AVERAGE_WEIGHT: i128 = 32;

pub(crate) const SLAVE_METHOD_PROPERTY: PropertySpec = PropertySpec::new(
    "slave-method",
    PropKind::Str,
    "how the sink follows a pipeline clock it does not provide: skew drops or inserts samples",
)
.with_default("skew")
.with_enum_values("skew | none");

pub(crate) const DRIFT_TOLERANCE_PROPERTY: PropertySpec = PropertySpec::new(
    "drift-tolerance",
    PropKind::Int,
    "drift from the pipeline clock allowed before resynchronising, microseconds",
)
.with_default("40000")
.with_range("1", "9223372036854775807");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlaveMethod {
    Skew,
    None,
}

impl SlaveMethod {
    fn name(self) -> &'static str {
        match self {
            SlaveMethod::Skew => "skew",
            SlaveMethod::None => "none",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PcmShape {
    frame_len: usize,
    rate: u32,
    silence: u8,
}

pub(crate) struct SkewSlaver {
    method: SlaveMethod,
    drift_tolerance_us: u64,
    master: Option<Arc<dyn PipelineClock + Send + Sync>>,
    shape: Option<PcmShape>,
    // device minus master clock time at the first measurement, moved by each correction
    offset_ns: Option<i128>,
    average_skew_ns: i128,
    // positive inserts silence frames, negative drops frames
    pending_frames: i64,
}

impl core::fmt::Debug for SkewSlaver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SkewSlaver")
            .field("method", &self.method)
            .field("drift_tolerance_us", &self.drift_tolerance_us)
            .field("offset_ns", &self.offset_ns)
            .field("average_skew_ns", &self.average_skew_ns)
            .field("pending_frames", &self.pending_frames)
            .finish()
    }
}

impl Default for SkewSlaver {
    fn default() -> Self {
        Self {
            method: SlaveMethod::Skew,
            drift_tolerance_us: DEFAULT_DRIFT_TOLERANCE_US,
            master: None,
            shape: None,
            offset_ns: None,
            average_skew_ns: 0,
            pending_frames: 0,
        }
    }
}

impl SkewSlaver {
    pub(crate) fn set_master(&mut self, master: Arc<dyn PipelineClock + Send + Sync>) {
        self.master = Some(master);
    }

    pub(crate) fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let Caps::Audio {
            format,
            channels,
            sample_rate,
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        let frame_len = frame_bytes(*format, *channels).ok_or(G2gError::CapsMismatch)?;
        self.shape = Some(PcmShape {
            frame_len,
            rate: *sample_rate,
            silence: silence_byte(*format),
        });
        Ok(())
    }

    // gst audiobasesink skew slaving: device clock minus master, averaged, and past half the
    // tolerance the playout position moves by the whole average
    pub(crate) fn measure(&mut self, device: &DriftClock) {
        let (SlaveMethod::Skew, Some(master), Some(shape)) =
            (self.method, &self.master, self.shape)
        else {
            return;
        };
        // before this the device clock is still wall time or a fit too short to trust
        if device.observations() < DriftClock::MIN_SAMPLES_TO_GATE {
            return;
        }
        let difference = i128::from(device.now_ns()) - i128::from(master.now_ns());
        let offset = *self.offset_ns.get_or_insert(difference);
        let skew = difference - offset;
        let average =
            ((SKEW_AVERAGE_WEIGHT - 1) * self.average_skew_ns + skew) / SKEW_AVERAGE_WEIGHT;
        let half_tolerance = i128::from(self.drift_tolerance_us * NANOS_PER_MICROSECOND / 2);
        if average.abs() <= half_tolerance {
            self.average_skew_ns = average;
            return;
        }
        self.offset_ns = Some(offset + average);
        self.average_skew_ns = 0;
        let frames = ns_to_samples(average.unsigned_abs() as u64, shape.rate) as i64;
        // a device ahead of the master plays silence to fall back, one behind skips samples
        self.pending_frames += frames * average.signum() as i64;
    }

    // `None` when nothing is pending and the buffer plays as it is
    pub(crate) fn correct(&mut self, bytes: &[u8]) -> Option<Vec<u8>> {
        let shape = self.shape.filter(|_| self.pending_frames != 0)?;
        if self.pending_frames > 0 {
            let silence_len = self.pending_frames as usize * shape.frame_len;
            let mut corrected = alloc::vec![shape.silence; silence_len];
            corrected.extend_from_slice(bytes);
            self.pending_frames = 0;
            return Some(corrected);
        }
        let available = (bytes.len() / shape.frame_len) as i64;
        let dropped = self.pending_frames.unsigned_abs().min(available as u64) as i64;
        self.pending_frames += dropped;
        Some(bytes[dropped as usize * shape.frame_len..].to_vec())
    }

    #[cfg(any(feature = "alsa-sink", feature = "pulse-sink"))]
    pub(crate) fn correct_frame(
        &mut self,
        frame: g2g_core::frame::Frame,
        element: &'static str,
    ) -> Result<g2g_core::frame::Frame, G2gError> {
        use g2g_core::frame::Frame;
        use g2g_core::memory::SystemSlice;
        use g2g_core::MemoryDomain;
        if self.pending_frames == 0 {
            return Ok(frame);
        }
        let corrected = {
            let mapped = crate::dmabufmap::frame_bytes(&frame, element)?;
            self.correct(mapped.as_slice())
        };
        let Some(corrected) = corrected else {
            return Ok(frame);
        };
        Ok(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(corrected.into_boxed_slice())),
            ..frame
        })
    }

    pub(crate) fn set_property(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        Some(match name {
            "slave-method" => match value.as_str() {
                Some("skew") => {
                    self.method = SlaveMethod::Skew;
                    Ok(())
                }
                Some("none") => {
                    self.method = SlaveMethod::None;
                    Ok(())
                }
                Some(_) => Err(PropError::Value),
                None => Err(PropError::Type),
            },
            "drift-tolerance" => match value.as_int() {
                Some(us) if us >= 1 => {
                    self.drift_tolerance_us = us as u64;
                    Ok(())
                }
                Some(_) => Err(PropError::Value),
                None => Err(PropError::Type),
            },
            _ => return None,
        })
    }

    pub(crate) fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "slave-method" => Some(PropValue::Str(self.method.name().into())),
            "drift-tolerance" => Some(PropValue::Int(self.drift_tolerance_us as i64)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::ManualClock;
    use g2g_core::{AudioFormat, ChannelLayout};

    const RATE: u32 = 48_000;
    const CHANNELS: u8 = 2;
    const FRAME_LEN: usize = 4;
    const BUFFER_NS: u64 = 10_000_000;
    const BUFFERS: u64 = 2_000;

    fn stereo_s16() -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: CHANNELS,
            sample_rate: RATE,
            channel_layout: ChannelLayout::UNSPECIFIED,
        }
    }

    // a device whose playout runs `device_rate` times as fast as the master, measured once a buffer
    fn pending_after_drift(device_rate: f64, configure: impl FnOnce(&mut SkewSlaver)) -> i64 {
        let local = ManualClock::default();
        let master = ManualClock::default();
        let device = DriftClock::new(Arc::new(local.clone()));
        let mut slaver = SkewSlaver::default();
        slaver.configure(&stereo_s16()).unwrap();
        slaver.set_master(Arc::new(master.clone()));
        configure(&mut slaver);
        let mut total = 0;
        for buffer in 0..BUFFERS {
            let now_ns = buffer * BUFFER_NS;
            local.set(now_ns);
            master.set(now_ns);
            device.observe(now_ns, (now_ns as f64 * device_rate) as u64);
            slaver.measure(&device);
            total += core::mem::take(&mut slaver.pending_frames);
        }
        total
    }

    #[test]
    fn a_fast_device_plays_silence_and_a_slow_one_skips() {
        const DRIFT: f64 = 0.002;
        // 20 s at 2000 ppm is 40 ms of drift, twice half the 40 ms tolerance
        let drifted_ns = ((BUFFERS * BUFFER_NS) as f64 * DRIFT) as u64;
        // each correction recentres the window, so the total never exceeds the drift
        let corrected = 0..=ns_to_samples(drifted_ns, RATE) as i64;
        let inserted = pending_after_drift(1.0 + DRIFT, |_| {});
        let skipped = pending_after_drift(1.0 - DRIFT, |_| {});
        assert!(inserted > 0 && corrected.contains(&inserted), "{inserted}");
        assert!(skipped < 0 && corrected.contains(&-skipped), "{skipped}");
    }

    #[test]
    fn a_device_on_time_never_corrects() {
        assert_eq!(pending_after_drift(1.0, |_| {}), 0);
    }

    #[test]
    fn slave_method_none_never_corrects() {
        let pending = pending_after_drift(1.002, |slaver| {
            slaver
                .set_property("slave-method", &PropValue::Str("none".into()))
                .unwrap()
                .unwrap();
        });
        assert_eq!(pending, 0);
    }

    #[test]
    fn the_sinks_own_clock_as_master_never_corrects() {
        let local = ManualClock::default();
        let device = Arc::new(DriftClock::new(Arc::new(local.clone())));
        let mut slaver = SkewSlaver::default();
        slaver.configure(&stereo_s16()).unwrap();
        slaver.set_master(device.clone());
        for buffer in 0..BUFFERS {
            let now_ns = buffer * BUFFER_NS;
            local.set(now_ns);
            device.observe(now_ns, (now_ns as f64 * 1.002) as u64);
            slaver.measure(&device);
        }
        assert_eq!(slaver.pending_frames, 0);
    }

    #[test]
    fn corrections_insert_silence_or_skip_whole_frames() {
        let mut slaver = SkewSlaver::default();
        slaver.configure(&stereo_s16()).unwrap();
        let buffer: Vec<u8> = (1..=4 * FRAME_LEN as u8).collect();

        slaver.pending_frames = 2;
        let inserted = slaver.correct(&buffer).unwrap();
        assert_eq!(&inserted[..2 * FRAME_LEN], &[0; 2 * FRAME_LEN]);
        assert_eq!(&inserted[2 * FRAME_LEN..], &buffer[..]);

        slaver.pending_frames = -6;
        let skipped = slaver.correct(&buffer).unwrap();
        assert!(skipped.is_empty(), "the whole buffer is skipped");
        assert_eq!(
            slaver.pending_frames, -2,
            "the rest carries to the next buffer"
        );
        assert_eq!(slaver.correct(&buffer).unwrap(), &buffer[2 * FRAME_LEN..]);
        assert_eq!(slaver.correct(&buffer), None);
    }
}
