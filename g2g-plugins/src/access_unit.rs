use alloc::vec::Vec;

use g2g_core::frame::FrameTiming;
use g2g_core::{Caps, G2gError, OutputSink, PipelinePacket, Rate};

use crate::annexb::{find_start_code_prefix, START_CODE_PREFIX};

/// What one start-code unit does to the access unit being assembled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartCodeRole {
    /// A header that leads a picture (sequence header, GOP / entry point). It
    /// opens a new access unit once the current one already holds a picture.
    Leads,
    /// The coded picture itself, or its first slice.
    Picture,
    /// Part of the picture in progress (a slice, an extension, user data).
    Continues,
    /// A later slice of the picture in progress: it opens no access unit, but
    /// marks the one in progress as holding a picture.
    PictureContinuation,
}

/// Upper bound on bytes buffered while waiting for an access-unit boundary. A
/// real stream emits start codes frequently, so this only guards against an
/// unbounded buffer on non-conforming input: past it, the pending bytes go out
/// as one access unit. A single intra access unit at 4K stays well under it.
const MAX_ACCESS_UNIT_BYTES: usize = 16 * 1024 * 1024;

/// Consecutive buffers that must arrive on an access-unit boundary before the
/// framer stops holding the trailing access unit back. Two, so a byte-stream
/// chunk boundary that lands on an access-unit start by chance does not switch
/// the mode on its own.
const ACCESS_UNIT_ALIGNED_CONFIRMATIONS: u8 = 2;

pub(crate) type FramedAccessUnit = (Vec<u8>, FrameTiming);

#[derive(Debug)]
pub(crate) struct AccessUnitFramer {
    /// H.264 / H.265 Annex-B lets a zero byte ahead of `00 00 01` belong to the
    /// start code. The MPEG video streams do not: that zero ends the unit before.
    four_byte_start_codes: bool,
    /// Bytes received but not yet emitted as a complete access unit.
    pending: Vec<u8>,
    /// Offsets in `pending` where an access unit begins, ascending.
    access_unit_starts: Vec<usize>,
    /// The lowest offset the next start code may begin at: just past the code
    /// byte of the last unit classified.
    unit_floor: usize,
    // resumes past bytes already scanned, so an access unit fed in small chunks is scanned once
    search_from: usize,
    /// Whether the access unit in progress already holds a picture.
    seen_picture: bool,
    /// Timing captured when the first byte of the head access unit arrived.
    head_timing: FrameTiming,
    /// Consecutive buffers that began exactly where an access unit begins. At
    /// [`ACCESS_UNIT_ALIGNED_CONFIRMATIONS`] the trailing access unit is emitted
    /// with the buffer it arrived in instead of waiting for the next start
    /// code, which is a frame period of latency on a live stream. Every
    /// access-unit-per-buffer upstream (RTP depayloaders, the mp4 / TS demuxers)
    /// earns it within two buffers, a raw byte stream never does.
    aligned_runs: u8,
}

impl AccessUnitFramer {
    pub(crate) fn new(four_byte_start_codes: bool) -> Self {
        Self {
            four_byte_start_codes,
            pending: Vec::new(),
            access_unit_starts: Vec::new(),
            unit_floor: 0,
            search_from: 0,
            seen_picture: false,
            head_timing: FrameTiming::default(),
            aligned_runs: 0,
        }
    }

    /// Append one input buffer and return every access unit whose end is now
    /// known, in stream order. `classify` gets the bytes after a start code
    /// prefix and returns `None` when it needs more of them than have arrived.
    pub(crate) fn push(
        &mut self,
        bytes: &[u8],
        timing: FrameTiming,
        classify: impl Fn(&[u8]) -> Option<StartCodeRole>,
    ) -> Vec<FramedAccessUnit> {
        let boundary = self.append(bytes, timing);
        self.frame_appended(boundary, timing, classify)
    }

    // a refused conversion leaves nothing of `bytes` buffered
    pub(crate) fn push_converted(
        &mut self,
        bytes: &[u8],
        timing: FrameTiming,
        classify: impl Fn(&[u8]) -> Option<StartCodeRole>,
        convert: impl FnOnce(&mut [u8]) -> Option<()>,
    ) -> Option<Vec<FramedAccessUnit>> {
        let boundary = self.append(bytes, timing);
        if convert(&mut self.pending[boundary..]).is_none() {
            self.pending.truncate(boundary);
            return None;
        }
        Some(self.frame_appended(boundary, timing, classify))
    }

    // the offset in `pending` where `bytes` begin
    fn append(&mut self, bytes: &[u8], timing: FrameTiming) -> usize {
        if self.pending.is_empty() {
            self.head_timing = timing;
        }
        let boundary = self.pending.len();
        self.pending.extend_from_slice(bytes);
        boundary
    }

    fn frame_appended(
        &mut self,
        boundary: usize,
        timing: FrameTiming,
        classify: impl Fn(&[u8]) -> Option<StartCodeRole>,
    ) -> Vec<FramedAccessUnit> {
        if self.pending.len() > MAX_ACCESS_UNIT_BYTES {
            return self.finish().into_iter().collect();
        }

        self.scan(classify);
        if self.access_unit_starts.is_empty() {
            return Vec::new();
        }
        // The first start code of a buffer would always open an access unit on
        // its own, so this needs the bytes before it.
        if boundary > 0 {
            if self.access_unit_starts.binary_search(&boundary).is_ok() {
                self.aligned_runs = self.aligned_runs.saturating_add(1);
            } else {
                self.aligned_runs = 0;
            }
        }

        let complete_through = if self.aligned_runs >= ACCESS_UNIT_ALIGNED_CONFIRMATIONS {
            self.pending.len()
        } else if let [_, .., last] = self.access_unit_starts[..] {
            last
        } else {
            // at most one access unit buffered, still open
            return Vec::new();
        };
        let access_units = self.split_off_complete(complete_through, timing);
        self.head_timing = timing;
        access_units
    }

    /// Everything still buffered, as one access unit (end of stream).
    pub(crate) fn finish(&mut self) -> Option<FramedAccessUnit> {
        let pending = core::mem::take(&mut self.pending);
        self.reset_scan();
        (!pending.is_empty()).then_some((pending, self.head_timing))
    }

    /// Drop everything buffered (a seek discontinuity).
    pub(crate) fn clear(&mut self) {
        self.pending.clear();
        self.reset_scan();
    }

    fn reset_scan(&mut self) {
        self.access_unit_starts.clear();
        self.unit_floor = 0;
        self.search_from = 0;
        self.seen_picture = false;
    }

    fn scan(&mut self, classify: impl Fn(&[u8]) -> Option<StartCodeRole>) {
        loop {
            let Some(prefix) = find_start_code_prefix(&self.pending, self.search_from) else {
                let last_possible_prefix = self
                    .pending
                    .len()
                    .saturating_sub(START_CODE_PREFIX.len() - 1);
                self.search_from = self.search_from.max(last_possible_prefix);
                return;
            };
            let four_byte = self.four_byte_start_codes
                && prefix > self.unit_floor
                && self.pending[prefix - 1] == 0;
            let start = if four_byte { prefix - 1 } else { prefix };
            let code_offset = prefix + START_CODE_PREFIX.len();
            let Some(role) = classify(&self.pending[code_offset..]) else {
                self.search_from = prefix;
                return;
            };
            let opens_after_picture = matches!(role, StartCodeRole::Leads | StartCodeRole::Picture);
            if self.access_unit_starts.is_empty() || (self.seen_picture && opens_after_picture) {
                self.access_unit_starts.push(start);
                self.seen_picture = false;
            }
            if matches!(
                role,
                StartCodeRole::Picture | StartCodeRole::PictureContinuation
            ) {
                self.seen_picture = true;
            }
            // a picture start code is 0x00, a search resuming on it could re-match its own prefix
            self.unit_floor = code_offset + 1;
            self.search_from = self.unit_floor;
        }
    }

    fn split_off_complete(
        &mut self,
        complete_through: usize,
        timing: FrameTiming,
    ) -> Vec<FramedAccessUnit> {
        let retained_from = self
            .access_unit_starts
            .partition_point(|&start| start < complete_through);
        let complete_starts: Vec<usize> = self.access_unit_starts.drain(..retained_from).collect();
        let retained = self.pending.split_off(complete_through);
        let mut complete = core::mem::replace(&mut self.pending, retained);

        if self.pending.is_empty() {
            self.reset_scan();
        } else {
            for start in &mut self.access_unit_starts {
                *start -= complete_through;
            }
            self.unit_floor -= complete_through;
            self.search_from -= complete_through;
        }

        let Some((&first_start, later_starts)) = complete_starts.split_first() else {
            return Vec::new();
        };
        let mut access_units = Vec::with_capacity(complete_starts.len());
        for &start in later_starts.iter().rev() {
            access_units.push((complete.split_off(start), timing));
        }
        // bytes ahead of the first start code belong to no access unit
        complete.drain(..first_start);
        let head_timing = if first_start == 0 {
            self.head_timing
        } else {
            timing
        };
        access_units.push((complete, head_timing));
        access_units.reverse();
        access_units
    }
}

/// The access-unit start offsets `classify` finds in `data` taken whole.
pub(crate) fn access_unit_starts(
    data: &[u8],
    four_byte_start_codes: bool,
    classify: impl Fn(&[u8]) -> Option<StartCodeRole>,
) -> Vec<usize> {
    let mut framer = AccessUnitFramer::new(four_byte_start_codes);
    framer.pending.extend_from_slice(data);
    framer.scan(classify);
    framer.access_unit_starts
}

/// The gst `config-interval` cadence for re-inserting cached parameter sets or
/// configuration headers before a keyframe: `0` never, `-1` every keyframe,
/// `N > 0` at the first keyframe once `N` seconds elapsed.
#[derive(Debug, Default)]
pub(crate) struct ConfigInterval {
    pub(crate) seconds: i32,
    last_inserted_pts_ns: Option<u64>,
}

/// Widest `config-interval` accepted, in seconds.
pub(crate) const MAX_CONFIG_INTERVAL_SECONDS: i64 = 3600;

const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;

impl ConfigInterval {
    pub(crate) fn is_off(&self) -> bool {
        self.seconds == 0
    }

    /// Whether to prepend the cached configuration before this access unit.
    /// An access unit that carries its own configuration restarts the clock.
    pub(crate) fn insertion_due(
        &mut self,
        pts_ns: u64,
        keyframe: bool,
        carries_config: bool,
        have_cached_config: bool,
    ) -> bool {
        if self.is_off() || !keyframe {
            return false;
        }
        if carries_config {
            self.last_inserted_pts_ns = Some(pts_ns);
            return false;
        }
        let interval_ns =
            u64::from(self.seconds.unsigned_abs()).saturating_mul(NANOSECONDS_PER_SECOND);
        let every_keyframe = self.seconds < 0;
        let elapsed = self
            .last_inserted_pts_ns
            .is_none_or(|last| pts_ns.saturating_sub(last) >= interval_ns);
        if !(every_keyframe || elapsed) || !have_cached_config {
            return false;
        }
        self.last_inserted_pts_ns = Some(pts_ns);
        true
    }
}

/// Pushes `CapsChanged` for refined caps, skipping a repeat of the last ones.
#[derive(Debug, Default)]
pub(crate) struct CapsRefinement {
    last_emitted: Option<Caps>,
    pub(crate) changes_emitted: u64,
    // read from upstream CapsChanged packets only, configure sees the fixated placeholder
    upstream_framerate_q16: Option<u32>,
}

impl CapsRefinement {
    pub(crate) fn observe_upstream(&mut self, caps: &Caps) {
        self.upstream_framerate_q16 = match caps {
            Caps::CompressedVideo {
                framerate: Rate::Fixed(q16),
                ..
            } => Some(*q16),
            _ => None,
        };
    }

    // the bitstream's own rate first, then the one upstream declared
    pub(crate) fn framerate(&self, bitstream_q16: Option<u32>) -> Rate {
        bitstream_q16
            .or(self.upstream_framerate_q16)
            .map_or(Rate::Any, Rate::Fixed)
    }

    pub(crate) async fn refine(
        &mut self,
        caps: Caps,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if self.last_emitted.as_ref() == Some(&caps) {
            return Ok(());
        }
        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
        self.last_emitted = Some(caps);
        self.changes_emitted += 1;
        Ok(())
    }

    /// Forget the last caps so the next refinement re-emits (after a seek).
    pub(crate) fn forget(&mut self) {
        self.last_emitted = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn mpeg_classify(unit: &[u8]) -> Option<StartCodeRole> {
        Some(match *unit.first()? {
            0xB3 => StartCodeRole::Leads,
            0x00 => StartCodeRole::Picture,
            _ => StartCodeRole::Continues,
        })
    }

    fn timing(pts_ns: u64) -> FrameTiming {
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        }
    }

    fn picture(payload: u8) -> Vec<u8> {
        vec![
            0, 0, 1, 0xB3, payload, 0, 0, 1, 0x00, payload, 0, 0, 1, 0x01, payload,
        ]
    }

    #[test]
    fn byte_stream_chunks_hold_the_trailing_access_unit() {
        let stream: Vec<u8> = [picture(1), picture(2), picture(3)].concat();
        let mut framer = AccessUnitFramer::new(false);
        let mut emitted = Vec::new();
        // chunk sizes that never land on an access-unit start
        for chunk in stream.chunks(7) {
            emitted.extend(framer.push(chunk, timing(0), mpeg_classify));
        }
        let emitted: Vec<Vec<u8>> = emitted.into_iter().map(|(au, _)| au).collect();
        assert_eq!(emitted, vec![picture(1), picture(2)]);
        assert_eq!(framer.finish().map(|(au, _)| au), Some(picture(3)));
    }

    #[test]
    fn aligned_buffers_release_the_trailing_access_unit() {
        let mut framer = AccessUnitFramer::new(false);
        let pushed: Vec<usize> = (1..=4)
            .map(|n| {
                framer
                    .push(&picture(n), timing(u64::from(n)), mpeg_classify)
                    .len()
            })
            .collect();
        // held until two buffers in a row start an access unit, then released with its buffer
        assert_eq!(pushed, vec![0, 1, 2, 1]);
        assert!(framer.finish().is_none());
    }

    #[test]
    fn a_start_code_split_across_buffers_is_found() {
        let stream: Vec<u8> = [picture(1), picture(2)].concat();
        let split = picture(1).len() + 1;
        let mut framer = AccessUnitFramer::new(false);
        let mut emitted = framer.push(&stream[..split], timing(1), mpeg_classify);
        emitted.extend(framer.push(&stream[split..], timing(2), mpeg_classify));
        assert_eq!(emitted, vec![(picture(1), timing(1))]);
    }

    #[test]
    fn four_byte_start_codes_take_the_leading_zero() {
        let first = [0u8, 0, 0, 1, 0x00, 0xAA];
        let second = [0u8, 0, 0, 1, 0x00, 0xBB];
        let stream = [first, second].concat();
        assert_eq!(
            access_unit_starts(&stream, true, mpeg_classify),
            vec![0, first.len()]
        );
        // the MPEG framing leaves that zero with the unit before
        assert_eq!(
            access_unit_starts(&stream, false, mpeg_classify),
            vec![1, first.len() + 1]
        );
    }

    #[test]
    fn config_interval_inserts_by_cadence() {
        let second = NANOSECONDS_PER_SECOND;
        let mut every_two_seconds = ConfigInterval {
            seconds: 2,
            ..ConfigInterval::default()
        };
        assert!(every_two_seconds.insertion_due(0, true, false, true));
        assert!(!every_two_seconds.insertion_due(second, true, false, true));
        assert!(!every_two_seconds.insertion_due(2 * second, false, false, true));
        assert!(every_two_seconds.insertion_due(2 * second, true, false, true));
        assert!(!every_two_seconds.insertion_due(4 * second, true, true, true));
        assert!(!every_two_seconds.insertion_due(5 * second, true, false, true));

        let mut off = ConfigInterval::default();
        assert!(!off.insertion_due(0, true, false, true));
    }
}
