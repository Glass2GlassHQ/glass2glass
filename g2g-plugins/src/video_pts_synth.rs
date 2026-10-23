use alloc::vec::Vec;

use g2g_core::frame::FrameTiming;
use g2g_core::time::{ns_to_ticks, ticks_to_ns};
use g2g_core::VideoCodec;

use crate::mpeg2video::Mpeg2TimestampSynth;
use crate::mpegts::PES_CLOCK_HZ;
use crate::poc::AccessUnitPoc;

#[derive(Debug, Default)]
pub(crate) struct ElementaryStreamPts {
    synth: VideoPtsSynth,
    started: bool,
    latest_pts_90khz: Option<u64>,
    packed_vops: crate::mpeg4videoparse::PackedVops,
}

// a byte source stamps every chunk 0
fn source_pts(pts_ns: u64) -> Option<u64> {
    Some(pts_ns).filter(|pts| *pts != 0 && *pts != FrameTiming::PTS_NONE)
}

impl ElementaryStreamPts {
    pub(crate) fn split_packed_input(
        &mut self,
        codec: VideoCodec,
        input: &[u8],
        pts_ns: u64,
    ) -> Option<crate::mpeg4videoparse::PackedInputSplit> {
        if codec != VideoCodec::Mpeg4Part2 {
            return None;
        }
        self.packed_vops.observe_layer_header(input);
        self.packed_vops.split(input, source_pts(pts_ns)?)
    }

    // An unstamped first unit opens the timeline at 0, as gst's baseparse does.
    pub(crate) fn stamp(&mut self, codec: VideoCodec, au: &[u8], pts_ns: u64) -> u64 {
        let source_pts_ns = source_pts(pts_ns);
        let mut pts_90khz = source_pts_ns.map(|ns| ns_to_ticks(ns, PES_CLOCK_HZ));
        if !self.started {
            self.started = true;
            if pts_90khz.is_none() {
                pts_90khz = Some(first_display_offset_90khz(codec, au));
            }
        } else if pts_90khz.is_none() && resets_picture_order_count(codec, au) {
            // every picture after an IDR in coded order is shown after it
            pts_90khz = self
                .latest_pts_90khz
                .zip(self.synth.period_90)
                .map(|(latest, period)| latest.saturating_add(period));
        }
        let mut dts_90khz = None;
        self.synth.stamp(codec, au, &mut pts_90khz, &mut dts_90khz);
        if let Some(pts) = pts_90khz {
            self.latest_pts_90khz =
                Some(self.latest_pts_90khz.map_or(pts, |latest| latest.max(pts)));
        }
        source_pts_ns
            .or(pts_90khz.map(|ticks| ticks_to_ns(ticks, PES_CLOCK_HZ)))
            .unwrap_or(FrameTiming::PTS_NONE)
    }
}

// A muxer stamps video at least every 700 ms (13818-1 2.7.4), far fewer units than this at any frame rate.
const MAX_UNITS_BEFORE_FIRST_STAMP: usize = 64;

// The units ahead of a stream's first stamp wait for it, so they are back-filled from it as ffmpeg does instead of opening the timeline at 0.
#[derive(Debug)]
pub(crate) struct LeadingUnits<U> {
    held: Vec<U>,
    stamp_seen: bool,
}

impl<U> Default for LeadingUnits<U> {
    fn default() -> Self {
        Self {
            held: Vec::new(),
            stamp_seen: false,
        }
    }
}

impl<U> LeadingUnits<U> {
    // `backfill` gets the held units and the first stamped one after them
    pub(crate) fn pass(
        &mut self,
        unit: U,
        stamped: bool,
        backfill: impl FnOnce(&mut [U], &U),
    ) -> Vec<U> {
        if self.stamp_seen {
            return Vec::from([unit]);
        }
        if !stamped {
            self.held.push(unit);
            if self.held.len() < MAX_UNITS_BEFORE_FIRST_STAMP {
                return Vec::new();
            }
            self.stamp_seen = true;
            return core::mem::take(&mut self.held);
        }
        self.stamp_seen = true;
        backfill(&mut self.held, &unit);
        let mut released = core::mem::take(&mut self.held);
        released.push(unit);
        released
    }

    // end of stream: a stream that never showed a stamp keeps its units unstamped
    pub(crate) fn release(&mut self) -> Vec<U> {
        core::mem::take(&mut self.held)
    }
}

// Far from both ends of the range, so a provisional run neither saturates nor overflows.
const PROVISIONAL_ANCHOR_90KHZ: u64 = 1 << 62;

// The pts of the `leading` units ahead of a stream's first stamped unit. A fresh
// synthesis anchored provisionally on the first leading unit also places the
// stamped one, and its distance from the real stamp moves the others onto the
// stream's timeline. `stamp` runs that fresh synthesis on one unit.
pub(crate) fn backfill_leading_pts_90khz(
    leading: &[&[u8]],
    first_stamped: &[u8],
    first_stamp_90khz: u64,
    mut stamp: impl FnMut(&[u8], &mut Option<u64>),
) -> Vec<Option<u64>> {
    let provisional: Vec<Option<u64>> = leading
        .iter()
        .enumerate()
        .map(|(index, au)| {
            let mut pts = (index == 0).then_some(PROVISIONAL_ANCHOR_90KHZ);
            stamp(au, &mut pts);
            pts
        })
        .collect();
    let mut placed = None;
    stamp(first_stamped, &mut placed);
    let Some(placed) = placed else {
        return alloc::vec![None; leading.len()];
    };
    provisional
        .into_iter()
        .map(|pts| pts?.checked_add(first_stamp_90khz)?.checked_sub(placed))
        .collect()
}

fn resets_picture_order_count(codec: VideoCodec, au: &[u8]) -> bool {
    const H264_NAL_IDR: u8 = 5;
    const H265_NAL_IDR_N_LP: u8 = 20;
    crate::annexb::nal_units_any(au).any(|nal| match codec {
        VideoCodec::H264 => nal
            .first()
            .is_some_and(|header| header & 0x1F == H264_NAL_IDR),
        VideoCodec::H265 => nal.first().is_some_and(|header| {
            (crate::poc::H265_NAL_BLA_W_LP..=H265_NAL_IDR_N_LP).contains(&((header >> 1) & 0x3F))
        }),
        _ => false,
    })
}

// An MPEG-2 picture coded ahead of the ones it follows on screen sits its temporal reference in frames after 0.
fn first_display_offset_90khz(codec: VideoCodec, au: &[u8]) -> u64 {
    if codec != VideoCodec::Mpeg2 {
        return 0;
    }
    let period = crate::mpeg2video::parse_sequence_header(au)
        .and_then(|sequence| frame_period_90khz(sequence.framerate_q16));
    let temporal_reference = crate::mpeg2video::picture_temporal_reference(au);
    match (period, temporal_reference) {
        (Some(period), Some((_, temporal_reference))) => {
            period.saturating_mul(u64::from(temporal_reference))
        }
        _ => 0,
    }
}

/// Per-frame video PTS synthesis for one elementary stream (M948). A transport
/// stream need only carry a PES timestamp every 700 ms, so a conforming mux may
/// leave most access units unstamped, and forwarding those unstamped lands them
/// all at 0: a pacing sink plays that as a burst then a freeze.
///
/// An unstamped unit gets `last + frame_period` when the stream presents
/// pictures in the order it codes them, which an SPS proves (H.264 POC type 2,
/// H.265 `sps_max_num_reorder_pics` 0). Every real stamp re-anchors, so a period
/// that is slightly off never drifts past one stamping interval.
///
/// A stream that may reorder takes the display order from the picture order
/// count instead (M952), and MPEG-2 from the picture header's
/// `temporal_reference`. Both anchor on the real stamps the same way.
#[derive(Debug, Default)]
pub(crate) struct VideoPtsSynth {
    /// Whether the SPS proved coded order is display order. `None` until one
    /// parses; the per-unit period path runs only on `Some(true)`.
    presents_in_decode_order: Option<bool>,
    /// Frame period in 90 kHz ticks.
    period_90: Option<u64>,
    /// Whether `period_90` came from the SPS VUI timing info. A measured period
    /// keeps being refined; a declared one is not overridden.
    period_from_sps: bool,
    /// The last PTS this stream carried, real or synthesized.
    last_pts_90: Option<u64>,
    /// Units seen since `last_pts_90` was set, so a unit that could not be
    /// stamped still costs its display slot.
    units_since_last_pts: u64,
    /// The last real PES stamp, and the units seen since, for measuring the
    /// period from the two nearest real stamps when the SPS declares none.
    last_real_pts_90: Option<u64>,
    units_since_last_real: u64,
    /// Picture order count per access unit, for a stream that may reorder.
    poc: AccessUnitPoc,
    /// Where those counts sit on the presentation timeline.
    reorder: ReorderPts,
    /// MPEG-2 display-order synthesis, shared with the program-stream demuxer.
    mpeg2: Mpeg2TimestampSynth,
}

impl VideoPtsSynth {
    /// Stamp one access unit, filling `pts_90khz` (and, for MPEG-2, `dts_90khz`)
    /// in place when it is unstamped and this stream qualifies. On the
    /// coded-order path the decode timestamp equals the presentation one and
    /// needs no synthesis (the element falls back to the PTS for a missing DTS).
    pub(crate) fn stamp(
        &mut self,
        codec: VideoCodec,
        au: &[u8],
        pts_90khz: &mut Option<u64>,
        dts_90khz: &mut Option<u64>,
    ) {
        if codec == VideoCodec::Mpeg2 {
            self.stamp_mpeg2(au, pts_90khz, dts_90khz);
            return;
        }
        self.units_since_last_pts = self.units_since_last_pts.saturating_add(1);
        self.units_since_last_real = self.units_since_last_real.saturating_add(1);
        match codec {
            VideoCodec::H264 | VideoCodec::H265 => self.read_sps(codec, au),
            _ => self.read_sequence_header(codec, au),
        }
        if self.presents_in_decode_order == Some(false) {
            self.stamp_by_poc(codec, au, pts_90khz);
            return;
        }

        if let Some(real) = *pts_90khz {
            self.measure_period(real);
            self.last_pts_90 = Some(real);
            self.units_since_last_pts = 0;
            return;
        }
        if self.presents_in_decode_order != Some(true) {
            return;
        }
        let (Some(last), Some(period)) = (self.last_pts_90, self.period_90) else {
            return;
        };
        let synthesized = last.saturating_add(period.saturating_mul(self.units_since_last_pts));
        *pts_90khz = Some(synthesized);
        self.last_pts_90 = Some(synthesized);
        self.units_since_last_pts = 0;
    }

    /// Take the frame period from the span between this real stamp and the
    /// previous one, divided by the units in between. Only for a stream whose
    /// SPS declared no timing info; a backwards or repeated stamp yields nothing.
    fn measure_period(&mut self, real: u64) {
        if !self.period_from_sps {
            if let Some(previous) = self.last_real_pts_90 {
                let span = real.saturating_sub(previous);
                if let Some(measured) = span
                    .checked_div(self.units_since_last_real)
                    .filter(|period| *period > 0)
                {
                    self.period_90 = Some(measured);
                }
            }
        }
        self.last_real_pts_90 = Some(real);
        self.units_since_last_real = 0;
    }

    /// Re-read the SPS this access unit carries, if any. Every unit is scanned,
    /// since parameter sets travel in band and may change mid-stream.
    fn read_sps(&mut self, codec: VideoCodec, au: &[u8]) {
        use crate::nalparse::NalCodec;
        let info = match codec {
            VideoCodec::H265 => crate::h265parse::H265Codec::extract_sps_info(au),
            _ => crate::h264parse::H264Codec::extract_sps_info(au),
        };
        let Some(info) = info else {
            return;
        };
        self.presents_in_decode_order = Some(info.presents_in_decode_order);
        if let Some(poc) = info.poc {
            self.poc.set_sps(poc);
        }
        let Some(period) = info.framerate.and_then(frame_period_90khz) else {
            return;
        };
        self.period_90 = Some(period);
        self.period_from_sps = true;
    }

    fn read_sequence_header(&mut self, codec: VideoCodec, au: &[u8]) {
        use crate::startcodeparse::StartCodeCodec;
        let geometry = match codec {
            VideoCodec::Mpeg4Part2 => crate::mpeg4videoparse::Mpeg4VideoCodec::geometry(au),
            VideoCodec::Vc1 => crate::vc1parse::Vc1Codec::geometry(au),
            _ => return,
        };
        // these headers do not say whether the stream codes B pictures
        self.presents_in_decode_order = Some(true);
        let Some(period) = geometry
            .and_then(|geometry| geometry.framerate)
            .and_then(frame_period_90khz)
        else {
            return;
        };
        self.period_90 = Some(period);
        self.period_from_sps = true;
    }

    /// Stamp an access unit of a stream that may reorder, by its picture order
    /// count. A real stamp re-anchors the timeline (and, with a second one,
    /// measures how many ticks a count unit spans); an unstamped unit lands at
    /// its own distance from that anchor, which is behind it for a picture the
    /// stream coded ahead of its display slot.
    ///
    /// The first stamp alone carries the timeline when the SPS declares the
    /// frame period (M1156): the count step per frame follows from the codec,
    /// so the slope is known before a second stamp measures it.
    fn stamp_by_poc(&mut self, codec: VideoCodec, au: &[u8], pts_90khz: &mut Option<u64>) {
        let Some(poc) = self.poc.push_access_unit(codec, au).map(i64::from) else {
            return;
        };
        let declared_period = self.period_90.filter(|_| self.period_from_sps);
        if let Some(real) = *pts_90khz {
            self.reorder.anchor(poc, real, declared_period, codec);
        }
        let frame_coded_h264 = codec != VideoCodec::H265 && self.poc.frame_mbs_only() == Some(true);
        if frame_coded_h264 {
            self.reorder.refine_h264_step_from_parity(poc);
        }
        if pts_90khz.is_none() {
            *pts_90khz = self.reorder.synthesize(poc);
        }
    }

    /// Stamp an MPEG-2 access unit from its picture header, on the frame period
    /// the sequence header in effect declares.
    fn stamp_mpeg2(&mut self, au: &[u8], pts_90khz: &mut Option<u64>, dts_90khz: &mut Option<u64>) {
        if let Some(period) = crate::mpeg2video::parse_sequence_header(au)
            .and_then(|seq| frame_period_90khz(seq.framerate_q16))
        {
            self.period_90 = Some(period);
        }
        let Some(period) = self.period_90 else {
            return;
        };
        self.mpeg2.stamp(au, pts_90khz, dts_90khz, period);
    }
}

/// Largest picture-order-count step per coded frame [`ReorderPts`] will snap a
/// declared frame period onto: H.265 counts pictures (1), H.264 counts fields
/// (2, or 4 across a field pair). A measurement implying more than that is a
/// count this synthesis cannot read as a frame grid, so the measured ratio
/// stands instead.
const MAX_POC_STEP_PER_FRAME: u64 = 4;

/// Picture order counts one coded frame advances in H.265, which counts pictures.
const H265_POC_STEP_PER_FRAME: u64 = 1;
/// The same for H.264, which counts fields: a frame-coded picture advances both,
/// and a field-coded stream codes one access unit per field, half a period apart.
const H264_POC_STEP_PER_FRAME: u64 = 2;
/// The H.264 step when the encoder counts pictures instead of fields, which an
/// odd count in a frame-coded stream proves.
const H264_PICTURE_COUNTING_POC_STEP: u64 = 1;

/// Where a reordering stream's picture order counts sit on the presentation
/// timeline (M952). Presentation time is linear in the count, so two real PES
/// stamps fix the line exactly, whatever reorder depth the stream codes at: the
/// later of the two anchors it and the pair measures its slope.
///
/// One stamp is enough when the SPS declares the frame period (M1156): the
/// codec's count step per frame turns that period into a provisional slope, and
/// the second stamp replaces it with the measured one.
#[derive(Debug, Default)]
struct ReorderPts {
    /// The last real stamp and the count it named.
    anchor: Option<(i64, u64)>,
    /// Ticks per count, as the exact ratio `(ticks, counts)`.
    scale: Option<(u64, u64)>,
    /// Whether `scale` is the declared period over an assumed count step rather
    /// than a span measured between two stamps.
    scale_is_provisional: bool,
}

impl ReorderPts {
    /// Take a real stamp: re-anchor on it, and measure the slope against the
    /// previous anchor. A declared frame period replaces the measured slope
    /// when it divides into it a whole number of counts per frame, so the grid
    /// follows the rate the stream declares rather than a rounded span. The
    /// first stamp has nothing to measure against, so a declared period stands
    /// in over the codec's count step until the second one arrives.
    fn anchor(
        &mut self,
        poc: i64,
        pts_90khz: u64,
        declared_period_90: Option<u64>,
        codec: VideoCodec,
    ) {
        match self.anchor {
            Some((previous_poc, previous_pts)) => {
                let counts = poc.abs_diff(previous_poc);
                let ticks = pts_90khz.abs_diff(previous_pts);
                if counts > 0 && ticks > 0 {
                    self.scale = Some(snap_to_declared_period(ticks, counts, declared_period_90));
                    self.scale_is_provisional = false;
                }
            }
            None => {
                if let Some(period) = declared_period_90 {
                    self.scale = Some((period, poc_step_per_frame(codec)));
                    self.scale_is_provisional = true;
                }
            }
        }
        self.anchor = Some((poc, pts_90khz));
    }

    /// An H.264 encoder may step the order count once per frame rather than once
    /// per field. Every count of a frame-coded stream is then even under the
    /// field step, so the first odd count proves the step is one. A measured
    /// slope already spans whatever the stream does, so only a provisional one
    /// is rewritten.
    fn refine_h264_step_from_parity(&mut self, poc: i64) {
        if !self.scale_is_provisional || poc % 2 == 0 {
            return;
        }
        if let Some((ticks, _)) = self.scale {
            self.scale = Some((ticks, H264_PICTURE_COUNTING_POC_STEP));
        }
    }

    /// The presentation time of an unstamped unit whose count is `poc`, once an
    /// anchor and a slope are both known.
    fn synthesize(&self, poc: i64) -> Option<u64> {
        let (anchor_poc, anchor_pts) = self.anchor?;
        let (ticks, counts) = self.scale?;
        let distance = poc - anchor_poc;
        let offset = distance.unsigned_abs().checked_mul(ticks)? / counts;
        Some(if distance >= 0 {
            anchor_pts.saturating_add(offset)
        } else {
            anchor_pts.saturating_sub(offset)
        })
    }
}

/// How many picture order counts one coded frame of `codec` advances.
fn poc_step_per_frame(codec: VideoCodec) -> u64 {
    match codec {
        VideoCodec::H265 => H265_POC_STEP_PER_FRAME,
        _ => H264_POC_STEP_PER_FRAME,
    }
}

/// Express a measured `ticks` per `counts` slope as the declared frame period
/// over a whole count step per frame, when the measurement implies a plausible
/// one. Otherwise the measurement stands.
fn snap_to_declared_period(ticks: u64, counts: u64, declared_period_90: Option<u64>) -> (u64, u64) {
    let Some(period) = declared_period_90 else {
        return (ticks, counts);
    };
    let step = (period.saturating_mul(counts).saturating_add(ticks / 2)) / ticks;
    if (1..=MAX_POC_STEP_PER_FRAME).contains(&step) {
        (period, step)
    } else {
        (ticks, counts)
    }
}

/// The frame period in 90 kHz units for a Q16 fixed-point frame rate, rounded
/// to the nearest tick. `None` for a rate of zero, or one so high the period
/// rounds away.
pub(crate) fn frame_period_90khz(framerate_q16: u32) -> Option<u64> {
    let q16 = u64::from(framerate_q16);
    if q16 == 0 {
        return None;
    }
    Some(((90_000u64 << 16) + q16 / 2) / q16).filter(|period| *period > 0)
}
