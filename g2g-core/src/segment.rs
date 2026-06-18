//! Seek requests and the playback segment (M79).
//!
//! The pure, `no_std` foundation of the seek track: a [`Seek`] request
//! (`seek(rate, start, stop, flags)`, the GStreamer seek-event analog) and a
//! [`Segment`] (the `GstSegment` analog, TIME format, nanoseconds) carrying the
//! `rate` / `start` / `stop` / `base` / `time` decomposition plus the
//! **running-time** and **stream-time** conversions that AV sync and trick-play
//! depend on. This milestone is data + math only; wiring a `Segment` into the
//! packet stream and a `Seekable` source into the runner (flush-and-resume) is
//! the next milestone.
//!
//! Two timelines, mirroring GStreamer:
//! - **running time** is pipeline-clock time. It is direction- and rate-aware:
//!   playing twice as fast (`rate == 2.0`) advances running time half as much
//!   per buffer, and reverse playback (`rate < 0`) measures from `stop` down.
//!   This is the timeline a sink compares against the clock to schedule.
//! - **stream time** is the position within the media, scaled by
//!   `applied_rate` (the rate already baked into the buffers). Direction-
//!   agnostic: it answers "how far into the asset is this?" for seeking and UI.

/// Absolute value of an `f64` without pulling in `std` (`f64::abs` is a `std`
/// method; the `no_std` core has no libm). Used for the rate magnitude.
fn fabs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// What a seek's `start` / `stop` value means, mirroring `GstSeekType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SeekType {
    /// Leave this edge of the segment unchanged.
    None,
    /// Set this edge to the absolute value carried by the [`Seek`].
    Set,
    /// Set this edge relative to the end of the stream (the value is an offset
    /// back from the duration; `0` means the very end).
    End,
}

/// Seek modifier flags, a subset of `GstSeekFlags`. A bitset over a `u32` so
/// flags compose with `|` and are queried with [`SeekFlags::contains`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeekFlags(u32);

impl SeekFlags {
    /// No flags.
    pub const NONE: SeekFlags = SeekFlags(0);
    /// Flush the pipeline (discard in-flight data) before repositioning. The
    /// common interactive-seek case; without it the seek accumulates after the
    /// current data drains.
    pub const FLUSH: SeekFlags = SeekFlags(1 << 0);
    /// Seek to the exact position rather than the nearest cheap point.
    pub const ACCURATE: SeekFlags = SeekFlags(1 << 1);
    /// Snap to a key unit (keyframe) at or around the target.
    pub const KEY_UNIT: SeekFlags = SeekFlags(1 << 2);
    /// Emit a `SEGMENT`-done at `stop` instead of running to `Eos` (segment
    /// playback / looping).
    pub const SEGMENT: SeekFlags = SeekFlags(1 << 3);
    /// Trick mode: allow dropping non-key frames for fast scrub.
    pub const TRICKMODE: SeekFlags = SeekFlags(1 << 4);
    /// With `KEY_UNIT`, snap to the key unit at or before the target.
    pub const SNAP_BEFORE: SeekFlags = SeekFlags(1 << 5);
    /// With `KEY_UNIT`, snap to the key unit at or after the target.
    pub const SNAP_AFTER: SeekFlags = SeekFlags(1 << 6);

    /// Whether every flag in `other` is set.
    pub fn contains(self, other: SeekFlags) -> bool {
        (self.0 & other.0) == other.0
    }

    /// The raw bits.
    pub fn bits(self) -> u32 {
        self.0
    }
}

impl core::ops::BitOr for SeekFlags {
    type Output = SeekFlags;
    fn bitor(self, rhs: SeekFlags) -> SeekFlags {
        SeekFlags(self.0 | rhs.0)
    }
}

/// A seek request: change the playback `rate` and/or reposition the
/// `[start, stop]` window. The GStreamer seek-event analog. Times are
/// nanoseconds on the stream timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Seek {
    /// Playback rate. `1.0` is normal; `> 1.0` faster, `0 < r < 1` slower,
    /// `< 0` reverse. Must be non-zero.
    pub rate: f64,
    /// Modifier flags.
    pub flags: SeekFlags,
    /// How to interpret `start`.
    pub start_type: SeekType,
    /// New segment start (ns), interpreted per `start_type`.
    pub start: u64,
    /// How to interpret `stop`.
    pub stop_type: SeekType,
    /// New segment stop (ns), interpreted per `stop_type`.
    pub stop: u64,
}

impl Seek {
    /// A flushing seek to an absolute `position` (ns) at normal rate, leaving
    /// `stop` unchanged. The everyday "scrub to here" request.
    pub fn flush_to(position: u64) -> Seek {
        Seek {
            rate: 1.0,
            flags: SeekFlags::FLUSH,
            start_type: SeekType::Set,
            start: position,
            stop_type: SeekType::None,
            stop: 0,
        }
    }

    /// Whether this is a flushing seek.
    pub fn is_flush(self) -> bool {
        self.flags.contains(SeekFlags::FLUSH)
    }

    /// Whether playback is reverse (`rate < 0`).
    pub fn is_reverse(self) -> bool {
        self.rate < 0.0
    }
}

/// A playback segment (the `GstSegment` analog, TIME format). Describes the
/// portion of the stream currently being played, the rate, and the mapping
/// onto the pipeline running-time clock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    /// Playback rate (sign carries direction).
    pub rate: f64,
    /// Rate already applied to the buffers (scales stream time). Usually `1.0`.
    pub applied_rate: f64,
    /// Running time of the segment's playback start. Accumulates across
    /// non-flushing seeks; resets on a flushing seek.
    pub base: u64,
    /// Earliest valid stream timestamp in the segment (ns).
    pub start: u64,
    /// Latest valid stream timestamp (ns), or `None` for an open-ended segment.
    pub stop: Option<u64>,
    /// Stream time of the segment start (ns).
    pub time: u64,
    /// Current playback position within the segment (ns).
    pub position: u64,
}

impl Segment {
    /// An open-ended, normal-rate segment starting at `0`.
    pub fn new() -> Segment {
        Segment {
            rate: 1.0,
            applied_rate: 1.0,
            base: 0,
            start: 0,
            stop: None,
            time: 0,
            position: 0,
        }
    }

    /// Whether `ts` (ns, stream timeline) lies within `[start, stop]`.
    pub fn contains(&self, ts: u64) -> bool {
        ts >= self.start && self.stop.map_or(true, |stop| ts <= stop)
    }

    /// Map a stream timestamp to **running time** (pipeline-clock ns), or
    /// `None` if `ts` is outside the segment (or the segment is reverse with no
    /// `stop` to measure from). Rate- and direction-aware:
    /// - forward (`rate > 0`): `base + (ts - start) / |rate|`
    /// - reverse (`rate < 0`): `base + (stop - ts) / |rate|`
    pub fn to_running_time(&self, ts: u64) -> Option<u64> {
        if !self.contains(ts) {
            return None;
        }
        let abs_rate = fabs(self.rate);
        if abs_rate == 0.0 {
            return None;
        }
        let span = if self.rate < 0.0 {
            // Reverse needs a finite stop to measure down from.
            self.stop?.checked_sub(ts)?
        } else {
            ts.checked_sub(self.start)?
        };
        let scaled = (span as f64 / abs_rate) as u64;
        Some(self.base.saturating_add(scaled))
    }

    /// Map a stream timestamp to **stream time** (ns), or `None` if `ts` is
    /// outside the segment. Direction-agnostic, scaled by `applied_rate`:
    /// `time + (ts - start) * |applied_rate|`.
    pub fn to_stream_time(&self, ts: u64) -> Option<u64> {
        if !self.contains(ts) {
            return None;
        }
        let span = ts.checked_sub(self.start)?;
        let scaled = (span as f64 * fabs(self.applied_rate)) as u64;
        Some(self.time.saturating_add(scaled))
    }

    /// Clip a buffer's `[b_start, b_stop]` (ns, stream timeline) to the
    /// segment, returning the visible sub-range, or `None` if the buffer falls
    /// entirely outside. `b_stop` is exclusive-ish (a `None` open buffer end is
    /// clipped to the segment `stop`). The GStreamer `gst_segment_clip` analog.
    pub fn clip(&self, b_start: u64, b_stop: Option<u64>) -> Option<(u64, Option<u64>)> {
        // Fully after the segment stop?
        if let Some(seg_stop) = self.stop {
            if b_start >= seg_stop {
                return None;
            }
        }
        // Fully before the segment start?
        if let Some(bs) = b_stop {
            if bs <= self.start {
                return None;
            }
        }
        let out_start = b_start.max(self.start);
        let out_stop = match (b_stop, self.stop) {
            (Some(bs), Some(ss)) => Some(bs.min(ss)),
            (Some(bs), None) => Some(bs),
            (None, Some(ss)) => Some(ss),
            (None, None) => None,
        };
        Some((out_start, out_stop))
    }

    /// Build the fresh segment produced by applying a **flushing** `seek` from
    /// a stream of total `duration` ns (used to resolve `SeekType::End`). The
    /// flushing case resets `base` to `0` (running time restarts after a
    /// flush). `SeekType::None` leaves that edge at its default (`start`
    /// unchanged from `0`, `stop` open). The non-flushing (accumulating) seek,
    /// which advances `base` by the elapsed running time, is a follow-up.
    pub fn for_flush_seek(seek: &Seek, duration: Option<u64>) -> Segment {
        let start = match seek.start_type {
            SeekType::None => 0,
            SeekType::Set => seek.start,
            SeekType::End => duration.unwrap_or(0).saturating_sub(seek.start),
        };
        let stop = match seek.stop_type {
            SeekType::None => None,
            SeekType::Set => Some(seek.stop),
            SeekType::End => Some(duration.unwrap_or(0).saturating_sub(seek.stop)),
        };
        Segment {
            rate: seek.rate,
            applied_rate: 1.0,
            base: 0,
            start,
            stop,
            time: start,
            position: start,
        }
    }
}

impl Default for Segment {
    fn default() -> Self {
        Segment::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_compose_and_query() {
        let f = SeekFlags::FLUSH | SeekFlags::KEY_UNIT;
        assert!(f.contains(SeekFlags::FLUSH));
        assert!(f.contains(SeekFlags::KEY_UNIT));
        assert!(!f.contains(SeekFlags::ACCURATE));
        // contains is subset, so it holds for NONE and for the exact set.
        assert!(f.contains(SeekFlags::NONE));
        assert!(f.contains(SeekFlags::FLUSH | SeekFlags::KEY_UNIT));
    }

    #[test]
    fn flush_to_builds_a_flushing_forward_seek() {
        let s = Seek::flush_to(5_000);
        assert!(s.is_flush());
        assert!(!s.is_reverse());
        assert_eq!(s.start_type, SeekType::Set);
        assert_eq!(s.start, 5_000);
        assert_eq!(s.stop_type, SeekType::None);
    }

    #[test]
    fn running_time_forward_normal_rate() {
        let seg = Segment {
            start: 1_000,
            base: 100,
            ..Segment::new()
        };
        // base + (ts - start) / 1.0
        assert_eq!(seg.to_running_time(1_000), Some(100));
        assert_eq!(seg.to_running_time(3_000), Some(2_100));
        // Before the segment start: outside.
        assert_eq!(seg.to_running_time(500), None);
    }

    #[test]
    fn running_time_scales_with_rate() {
        // 2x: running time advances half as fast.
        let fast = Segment {
            rate: 2.0,
            ..Segment::new()
        };
        assert_eq!(fast.to_running_time(2_000), Some(1_000));
        // 0.5x: running time advances twice as fast.
        let slow = Segment {
            rate: 0.5,
            ..Segment::new()
        };
        assert_eq!(slow.to_running_time(2_000), Some(4_000));
    }

    #[test]
    fn running_time_reverse_measures_from_stop() {
        let seg = Segment {
            rate: -1.0,
            start: 0,
            stop: Some(10_000),
            base: 0,
            ..Segment::new()
        };
        // At stop, running time is base (0); earlier positions are later in
        // running time.
        assert_eq!(seg.to_running_time(10_000), Some(0));
        assert_eq!(seg.to_running_time(6_000), Some(4_000));
        // Outside the segment.
        assert_eq!(seg.to_running_time(11_000), None);
        // Reverse with no stop cannot be measured.
        let open = Segment {
            rate: -1.0,
            stop: None,
            ..Segment::new()
        };
        assert_eq!(open.to_running_time(1_000), None);
    }

    #[test]
    fn stream_time_uses_applied_rate_and_is_direction_agnostic() {
        let seg = Segment {
            start: 1_000,
            time: 50_000,
            applied_rate: 2.0,
            ..Segment::new()
        };
        // time + (ts - start) * |applied_rate|
        assert_eq!(seg.to_stream_time(1_000), Some(50_000));
        assert_eq!(seg.to_stream_time(2_000), Some(52_000));
        assert_eq!(seg.to_stream_time(500), None);
    }

    #[test]
    fn clip_trims_to_segment_bounds() {
        let seg = Segment {
            start: 1_000,
            stop: Some(5_000),
            ..Segment::new()
        };
        // Fully inside: unchanged.
        assert_eq!(seg.clip(2_000, Some(3_000)), Some((2_000, Some(3_000))));
        // Straddles the start: trimmed up to start.
        assert_eq!(seg.clip(500, Some(2_000)), Some((1_000, Some(2_000))));
        // Straddles the stop: trimmed down to stop.
        assert_eq!(seg.clip(4_000, Some(9_000)), Some((4_000, Some(5_000))));
        // Open buffer end clips to segment stop.
        assert_eq!(seg.clip(2_000, None), Some((2_000, Some(5_000))));
        // Fully before / fully after: dropped.
        assert_eq!(seg.clip(0, Some(1_000)), None);
        assert_eq!(seg.clip(5_000, Some(6_000)), None);
    }

    #[test]
    fn flush_seek_builds_reset_segment() {
        // Set start, open stop, 2x rate.
        let seek = Seek {
            rate: 2.0,
            flags: SeekFlags::FLUSH,
            start_type: SeekType::Set,
            start: 3_000,
            stop_type: SeekType::None,
            stop: 0,
        };
        let seg = Segment::for_flush_seek(&seek, Some(100_000));
        assert_eq!(seg.rate, 2.0);
        assert_eq!(seg.start, 3_000);
        assert_eq!(seg.stop, None);
        assert_eq!(seg.base, 0, "flushing seek restarts running time");
        assert_eq!(seg.time, 3_000);
        assert_eq!(seg.position, 3_000);

        // SeekType::End resolves against the duration.
        let to_end = Seek {
            rate: 1.0,
            flags: SeekFlags::FLUSH,
            start_type: SeekType::End,
            start: 10_000, // 10us back from the end
            stop_type: SeekType::None,
            stop: 0,
        };
        let seg = Segment::for_flush_seek(&to_end, Some(100_000));
        assert_eq!(seg.start, 90_000);
    }
}
